"""Capture perangkat lokal; satu reader audio dan satu worker VAD serial."""

from collections import deque
from dataclasses import asdict
from contextlib import contextmanager
from datetime import date, datetime, timedelta
import json
from pathlib import Path
from queue import Queue
import sqlite3
from threading import Event, Lock, Thread
from uuid import uuid4

import numpy as np
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
            db.execute("CREATE INDEX IF NOT EXISTS recordings_latest "
                       "ON recordings(started_at DESC,id DESC)")
            db.execute("CREATE INDEX IF NOT EXISTS recordings_by_date "
                       "ON recordings(substr(started_at,1,10),started_at DESC,id DESC)")

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

    def listing(self, status="speech", limit=20, page=1, recorded_date: date | None = None):
        if not 1 <= limit <= 100 or page < 1:
            raise ValueError("Ukuran halaman 1–100 dan nomor halaman minimal 1")
        where = "WHERE substr(started_at,1,10)=?" if recorded_date else ""
        parameters = [recorded_date.isoformat()] if recorded_date else []
        with self.connect() as db:
            db.execute("BEGIN")  # Hitungan dan isi halaman memakai snapshot yang sama.
            counts = {row[0]: row[1] for row in db.execute(
                f"SELECT status,COUNT(*) FROM recordings {where} GROUP BY status", parameters)}
            total = sum(counts.values()) if status == "all" else counts.get(status, 0)
            total_pages = (total + limit - 1) // limit
            page = min(page, max(1, total_pages))
            if status != "all":
                where += " AND status=?" if where else "WHERE status=?"
                parameters.append(status)
            rows = db.execute(
                f"SELECT * FROM recordings {where} "
                "ORDER BY started_at DESC,id DESC LIMIT ? OFFSET ?",
                [*parameters, limit, (page-1)*limit]).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item.pop("path")
            item["metrics"] = json.loads(item["metrics"])
            item["audio_url"] = f"/api/recordings/{item['id']}/audio"
            items.append(item)
        return {"items": items, "counts": counts, "total": total,
                "page": page, "page_size": limit, "total_pages": total_pages,
                "date": recorded_date.isoformat() if recorded_date else None}


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
        self._waveform = deque(maxlen=400)
        self._elapsed = 0.0
        self._status = {
            "state": "stopped", "level_db": -120.0, "duration": 0.0,
            "silence_elapsed": 0.0, "device": None, "error": "",
            "overflows": 0, "vad_busy": False,
            "device_id": None, "channels": None, "channel_mode": None,
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

    def _update(self, waveform_point=None, **values):
        with self._lock:
            if waveform_point is not None:
                self._waveform.append(waveform_point)
                self._elapsed = waveform_point["time"]
            self._status.update(values)

    def status(self):
        with self._lock:
            status = dict(self._status)
            status["waveform"] = {
                "window_seconds": 10,
                "elapsed": self._elapsed,
                "points": [point for point in self._waveform
                           if point["time"] >= self._elapsed - 10],
            }
        status.update(config=asdict(self.config), vad_config=asdict(self.vad_config),
                      pending=len(self.store.pending()))
        return status

    def start(self, device_id: int, channels: int | None = None):
        with self._control:
            if self._closed:
                raise ValueError("Recorder sudah ditutup")
            if self._capture and self._capture.is_alive():
                raise ValueError("Monitoring masih berjalan")
            device = next((d for d in self.devices() if d["id"] == device_id), None)
            if device is None:
                raise ValueError("Pilih perangkat input yang tersedia")
            channel_mode = "auto" if channels is None else "manual"
            if channels is None:
                channels = min(device["max_channels"], 2)
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
                    "channels": channels, "channel_mode": channel_mode,
                    "sample_rate": rate, "frame_ms": 30,
                    "sad": asdict(self.config), "vad": asdict(self.vad_config),
                }, indent=2), encoding="utf-8")
            except Exception:
                stream.close()
                raise
            self._stop.clear()
            with self._lock:
                self._waveform.clear()
                self._elapsed = 0.0
            self._update(state="listening", error="", device=device["name"],
                         device_id=device_id, channels=channels, channel_mode=channel_mode,
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
                    waveform_point={
                        "time": engine.cursor / engine.samplerate,
                        "min": float(np.min(block)), "max": float(np.max(block)),
                        "rms_db": level,
                    },
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
