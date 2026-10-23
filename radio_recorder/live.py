"""Capture perangkat lokal; satu reader audio dan satu worker VAD serial."""

from dataclasses import asdict
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
from pathlib import Path
from queue import Queue
import sqlite3
from threading import Event, Lock, Thread
from uuid import uuid4

import sounddevice as sd

from .segmenter import Config, SoundTriggeredRecorder
from .vad import SileroVAD, VADConfig, VADResult


class RecordingStore:
    """Metadata persisten; hanya file terdaftar yang boleh diakses dari web."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "recordings.sqlite3"
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS recordings (
                id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL,
                started_at TEXT NOT NULL, duration REAL NOT NULL,
                device TEXT NOT NULL, samplerate INTEGER NOT NULL,
                channels INTEGER NOT NULL, status TEXT NOT NULL,
                vad_error TEXT NOT NULL DEFAULT '', metrics TEXT NOT NULL DEFAULT '{}'
            )""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def register(self, path, segment, started_at, device, channels):
        identifier = uuid4().hex
        relative = path.resolve().relative_to(self.root)
        with self.connect() as db:
            db.execute("INSERT INTO recordings "
                       "(id,path,started_at,duration,device,samplerate,channels,status) "
                       "VALUES (?,?,?,?,?,?,?,?)",
                       (identifier, relative.as_posix(), started_at.isoformat(),
                        segment.duration_seconds, device, segment.samplerate, channels,
                        "pending"))
        return identifier

    def complete(self, identifier, result):
        with self.connect() as db:
            db.execute("UPDATE recordings SET status=?,vad_error=?,metrics=? WHERE id=?",
                       (result.vad_status, result.vad_error,
                        json.dumps(result.csv_fields()), identifier))

    def get(self, identifier):
        with self.connect() as db:
            row = db.execute("SELECT * FROM recordings WHERE id=?", (identifier,)).fetchone()
        return dict(row) if row else None

    def audio_path(self, identifier):
        row = self.get(identifier)
        if row is None:
            raise FileNotFoundError("Rekaman tidak ditemukan")
        path = (self.root / row["path"]).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise FileNotFoundError("File rekaman tidak tersedia")
        return path

    def pending(self):
        with self.connect() as db:
            return [row[0] for row in db.execute(
                "SELECT id FROM recordings WHERE status='pending' ORDER BY started_at")]

    def listing(self, status="speech", limit=100):
        with self.connect() as db:
            counts = {row[0]: row[1] for row in db.execute(
                "SELECT status,COUNT(*) FROM recordings GROUP BY status")}
            if status == "all":
                rows = db.execute("SELECT * FROM recordings ORDER BY started_at DESC LIMIT ?",
                                  (limit,)).fetchall()
            else:
                rows = db.execute("SELECT * FROM recordings WHERE status=? "
                                  "ORDER BY started_at DESC LIMIT ?", (status, limit)).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item.pop("path")
            item["metrics"] = json.loads(item["metrics"])
            item["audio_url"] = f"/api/recordings/{item['id']}/audio"
            items.append(item)
        return {"items": items, "counts": counts}


class LiveRecorder:
    def __init__(self, root: Path, audio_backend=None, analyzer=None):
        self.audio = audio_backend if audio_backend is not None else sd
        self.config = Config()
        self.vad_config = VADConfig()
        self.store = RecordingStore(root)
        self.analyzer = analyzer if analyzer is not None else SileroVAD(self.vad_config)
        self._control = Lock()
        self._lock = Lock()
        self._stop = Event()
        self._capture = None
        self._closed = False
        self._jobs = Queue()
        self._status = {
            "state": "stopped", "level_db": -120.0, "duration": 0.0,
            "silence_elapsed": 0.0, "device": None, "error": "",
            "overflows": 0, "vad_busy": False,
        }
        for identifier in self.store.pending():
            self._jobs.put(identifier)
        self._worker = Thread(target=self._analyze, name="ht-vad", daemon=True)
        self._worker.start()

    def devices(self):
        apis = self.audio.query_hostapis()
        return [{"id": index, "name": device["name"],
                 "hostapi": apis[device["hostapi"]]["name"],
                 "max_channels": int(device["max_input_channels"]),
                 "sample_rate": int(device["default_samplerate"])}
                for index, device in enumerate(self.audio.query_devices())
                if device["max_input_channels"] > 0]

    def _update(self, **values):
        with self._lock:
            self._status.update(values)

    def status(self):
        with self._lock:
            status = dict(self._status)
        status.update(config=asdict(self.config), vad_config=asdict(self.vad_config),
                      pending=len(self.store.pending()))
        return status

    def start(self, device_id: int, channels: int = 1):
        with self._control:
            if self._closed:
                raise ValueError("Recorder sudah ditutup")
            if self._capture and self._capture.is_alive():
                raise ValueError("Monitoring masih berjalan")
            device = next((d for d in self.devices() if d["id"] == device_id), None)
            if device is None:
                raise ValueError("Pilih perangkat input yang tersedia")
            if not 1 <= channels <= min(device["max_channels"], 2):
                raise ValueError("Jumlah kanal tidak didukung (pilih mono/stereo)")
            rate = device["sample_rate"]
            frames = round(rate * .03)
            self.audio.check_input_settings(device=device_id, channels=channels,
                                            samplerate=rate, dtype="float32")
            stream = self.audio.InputStream(device=device_id, channels=channels,
                                            samplerate=rate, dtype="float32",
                                            blocksize=frames)
            try:
                stream.start()
                started = datetime.now().astimezone()
                directory = (self.store.root / started.strftime("%Y-%m-%d") /
                             f"{started:%H-%M-%S}_{uuid4().hex[:8]}")
                engine = SoundTriggeredRecorder(rate, channels, directory, self.config)
                (directory / "session.json").write_text(json.dumps({
                    "started_at": started.isoformat(), "device": device,
                    "channels": channels, "sample_rate": rate, "frame_ms": 30,
                    "sad": asdict(self.config), "vad": asdict(self.vad_config),
                }, indent=2), encoding="utf-8")
            except Exception:
                stream.close()
                raise
            self._stop.clear()
            self._update(state="listening", error="", device=device["name"],
                         level_db=-120.0, duration=0.0, silence_elapsed=0.0, overflows=0)
            self._capture = Thread(target=self._record,
                                   args=(stream, engine, frames, started, device["name"]),
                                   name="ht-capture", daemon=True)
            self._capture.start()
        return self.status()

    def _record(self, stream, engine, frames, started, device):
        published = 0

        def publish():
            nonlocal published
            for segment in engine.segments[published:]:
                identifier = self.store.register(
                    engine.output_dir / segment.filename, segment,
                    started + timedelta(seconds=segment.start_seconds),
                    device, engine.channels)
                self._jobs.put(identifier)
                published += 1

        error = ""
        try:
            while not self._stop.is_set():
                block, overflowed = stream.read(frames)
                if overflowed:
                    with self._lock:
                        self._status["overflows"] += 1
                engine.process(block)
                level = engine.levels[-1][1]
                engine.levels.clear()  # Monitoring berjam-jam tidak menumpuk plot RMS.
                recording = engine.writer is not None
                self._update(
                    state=("waiting" if engine.silence_frames else "recording")
                    if recording else "listening", level_db=level,
                    duration=(engine.cursor - engine.current_start) / engine.samplerate
                    if recording else 0.0,
                    silence_elapsed=engine.silence_frames / engine.samplerate)
                publish()
        except Exception as exc:
            error = f"Input audio gagal: {exc}"
        finally:
            try:
                stream.close()
            except Exception as exc:
                error = error or f"Gagal menutup perangkat: {exc}"
            try:
                engine.finish()
                publish()
            except Exception as exc:
                error = error or f"Gagal menyimpan rekaman: {exc}"
            self._update(state="error" if error else "stopped", error=error,
                         level_db=-120.0, duration=0.0, silence_elapsed=0.0)

    def stop(self):
        with self._control:
            self._stop.set()
            if self._capture:
                self._capture.join(timeout=3)
                if self._capture.is_alive():
                    self._update(error="Perangkat belum berhenti; coba Stop kembali")
                    raise ValueError("Menunggu perangkat audio berhenti")
        return self.status()

    def _analyze(self):
        while True:
            identifier = self._jobs.get()
            try:
                if identifier is None:
                    return
                self._update(vad_busy=True)
                result = self.analyzer.analyze_file(self.store.audio_path(identifier))
                self.store.complete(identifier, result)
            except Exception as exc:
                self.store.complete(identifier, VADResult(vad_error=str(exc)))
            finally:
                self._update(vad_busy=False)
                self._jobs.task_done()

    def close(self):
        self.stop()
        self._closed = True
        self._jobs.put(None)
        self._worker.join(timeout=10)
