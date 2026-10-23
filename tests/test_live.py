from datetime import date, datetime, timedelta, timezone
import json
from queue import Empty, Queue
from threading import Event
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from radio_recorder.live import LiveRecorder, RecordingStore
from radio_recorder.segmenter import Segment
from radio_recorder.vad import VADResult
from web_app import create_app


class FakeStream:
    def __init__(self, channels):
        self.channels = channels
        self.blocks = Queue()
        self.closed = False

    def start(self):
        pass

    def read(self, frames):
        try:
            block = self.blocks.get(timeout=.02)
        except Empty:
            block = np.zeros((frames, self.channels), dtype=np.float32)
        if isinstance(block, Exception):
            raise block
        return block, False

    def close(self):
        self.closed = True


class FakeAudio:
    def __init__(self, max_channels=2, reject_at=None):
        self.max_channels = max_channels
        self.reject_at = reject_at
        self.checked_channels = []
        self.opened_channels = []

    def query_hostapis(self):
        return [{"name": "Fake API"}]

    def query_devices(self):
        return [{"name": "USB HT", "hostapi": 0, "max_input_channels": self.max_channels,
                 "default_samplerate": 16000},
                {"name": "Speaker", "hostapi": 0, "max_input_channels": 0}]

    def check_input_settings(self, **kwargs):
        self.checked_channels.append(kwargs["channels"])
        if self.reject_at == "check":
            raise ValueError("Driver menolak konfigurasi stereo")

    def InputStream(self, **kwargs):
        self.opened_channels.append(kwargs["channels"])
        if self.reject_at == "open":
            raise OSError("Driver menolak konfigurasi stereo")
        self.stream = FakeStream(kwargs["channels"])
        return self.stream


class FakeVAD:
    def __init__(self, gate=None):
        self.entered = Event()
        self.gate = gate

    def analyze_file(self, path):
        # Pembacaan WAV membuktikan writer sudah ditutup sebelum VAD.
        assert sf.info(path).frames > 0
        self.entered.set()
        if self.gate:
            assert self.gate.wait(5)
        return VADResult(vad_status="speech", speech_duration_seconds=.5)


def wait_for(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Timeout menunggu worker recorder")


def active_block(channels=1):
    return np.full((480, channels), .2, dtype=np.float32)


@pytest.mark.parametrize("supported,expected", [(1, 1), (2, 2), (8, 2)])
def test_automatic_channels_follow_device_and_are_saved(tmp_path, supported, expected):
    backend = FakeAudio(max_channels=supported)
    recorder = LiveRecorder(tmp_path, backend, FakeVAD())
    try:
        status = recorder.start(0)
        assert status["channels"] == expected
        assert status["channel_mode"] == "auto"
        assert status["device_id"] == 0
        assert backend.stream.channels == expected
        backend.stream.blocks.put(active_block(expected))
        wait_for(lambda: recorder.status()["state"] in ("recording", "waiting"))
        recorder.stop()
        recorder._jobs.join()
        item = recorder.store.listing("all")["items"][0]
        path = recorder.store.audio_path(item["id"])
        assert sf.info(path).channels == expected
        assert item["channels"] == expected
        session = json.loads((path.parent / "session.json").read_text())
        assert session["channels"] == expected
        assert session["channel_mode"] == "auto"
    finally:
        recorder.close()


@pytest.mark.parametrize("reject_at", ["check", "open"])
def test_automatic_stereo_failure_never_falls_back_to_mono(tmp_path, reject_at):
    backend = FakeAudio(reject_at=reject_at)
    recorder = LiveRecorder(tmp_path, backend, FakeVAD())
    try:
        with pytest.raises((ValueError, OSError), match="menolak konfigurasi stereo"):
            recorder.start(0)
        assert backend.checked_channels == [2]
        assert backend.opened_channels == ([] if reject_at == "check" else [2])
        assert recorder.status()["state"] == "stopped"
        assert not list(tmp_path.rglob("session.json"))
    finally:
        recorder.close()


def test_manual_mono_override_and_unsupported_stereo(tmp_path):
    backend = FakeAudio()
    recorder = LiveRecorder(tmp_path, backend, FakeVAD())
    try:
        status = recorder.start(0, 1)
        assert status["channels"] == 1
        assert status["channel_mode"] == "manual"
        recorder.stop()
        backend.max_channels = 1
        with pytest.raises(ValueError, match="Jumlah kanal tidak didukung"):
            recorder.start(0, 2)
    finally:
        recorder.close()


def test_capture_keeps_running_while_vad_busy_and_stop_saves_final_wav(tmp_path):
    backend, gate = FakeAudio(), Event()
    vad = FakeVAD(gate)
    recorder = LiveRecorder(tmp_path, backend, vad)
    try:
        assert len(recorder.devices()) == 1
        recorder.start(0, 2)
        with pytest.raises(ValueError, match="masih berjalan"):
            recorder.start(0)
        backend.stream.blocks.put(active_block(2))
        for _ in range(334):
            backend.stream.blocks.put(np.zeros((480, 2), dtype=np.float32))
        assert vad.entered.wait(3)
        assert recorder.status()["vad_busy"]
        backend.stream.blocks.put(active_block(2))
        wait_for(lambda: recorder.status()["state"] in ("recording", "waiting"))
        recorder.stop()
        assert backend.stream.closed
        rows = recorder.store.listing("all")["items"]
        assert len(rows) == 2
        assert recorder.status()["state"] == "stopped"
        assert recorder.status()["config"] == {
            "threshold_db": -50, "prebuffer_seconds": 1, "silence_seconds": 10}
        assert recorder.status()["vad_config"]["min_speech_duration_ms"] == 300
        for item in rows:
            info = sf.info(recorder.store.audio_path(item["id"]))
            assert info.samplerate == 16000
            assert info.channels == 2
        gate.set()
        recorder._jobs.join()
        assert len(recorder.store.listing()["items"]) == 2
        recorder.start(0)
        recorder.stop()
    finally:
        gate.set()
        recorder.close()


def test_device_error_finalizes_wav_and_allows_restart(tmp_path):
    backend = FakeAudio()
    recorder = LiveRecorder(tmp_path, backend, FakeVAD())
    try:
        with pytest.raises(ValueError):
            recorder.start(3)
        recorder.start(0)
        backend.stream.blocks.put(active_block(2))
        backend.stream.blocks.put(OSError("USB disconnected"))
        wait_for(lambda: recorder.status()["state"] == "error")
        recorder.stop()
        recorder._jobs.join()
        assert "USB disconnected" in recorder.status()["error"]
        assert len(recorder.store.listing("all")["items"]) == 1
        recorder.start(0)
        assert recorder.status()["error"] == ""
    finally:
        recorder.close()


def test_waveform_preserves_stereo_spike_between_polls_and_bounds_history(tmp_path):
    backend = FakeAudio()
    recorder = LiveRecorder(tmp_path, backend, FakeVAD())
    try:
        initial = recorder.status()["waveform"]
        assert initial == {"window_seconds": 10, "elapsed": 0, "points": []}
        recorder.start(0)
        spike = np.empty((480, 2), dtype=np.float32)
        spike[:, 0] = .2
        spike[:, 1] = -.2  # Antar-kanal tidak di-downmix menjadi hening.
        backend.stream.blocks.put(spike)
        for _ in range(20):
            backend.stream.blocks.put(np.zeros((480, 2), dtype=np.float32))
        wait_for(lambda: recorder.status()["waveform"]["elapsed"] >= .63)
        data = recorder.status()["waveform"]
        spikes = [point for point in data["points"] if point["max"] > .1]
        assert len(spikes) == 1
        assert spikes[0]["min"] == pytest.approx(-.2)
        assert spikes[0]["max"] == pytest.approx(.2)
        assert spikes[0]["rms_db"] == pytest.approx(20*np.log10(.2))
        assert data["points"][-1]["rms_db"] == -120
        for _ in range(420):
            backend.stream.blocks.put(np.zeros((480, 2), dtype=np.float32))
        wait_for(lambda: recorder.status()["waveform"]["elapsed"] >= 13.23)
        recorder.stop()
        data = recorder.status()["waveform"]
        assert len(data["points"]) <= 335
        assert all(point["time"] >= data["elapsed"]-10 for point in data["points"])
        assert not any(point["max"] > .1 for point in data["points"])
        assert recorder.status()["waveform"] == data  # Stop mempertahankan grafik terakhir.
        recorder.start(0)
        assert recorder.status()["waveform"]["elapsed"] < 1
        assert not any(point["max"] > .1 for point in recorder.status()["waveform"]["points"])
    finally:
        recorder.close()


def register_clip(store, status="speech"):
    path = store.root / "clip.wav"
    sf.write(path, active_block(), 16000)
    segment = Segment(1, path.name, 0, 480, 16000)
    identifier = store.register(path, segment, datetime.now().astimezone(), "USB HT", 1)
    if status != "pending":
        store.complete(identifier, VADResult(vad_status=status))
    return identifier


def test_pending_jobs_resume_and_metadata_survives_restart(tmp_path):
    store = RecordingStore(tmp_path)
    identifier = register_clip(store, "pending")
    recorder = LiveRecorder(tmp_path, FakeAudio(), FakeVAD())
    try:
        recorder._jobs.join()
        assert recorder.store.get(identifier)["status"] == "speech"
    finally:
        recorder.close()
    assert RecordingStore(tmp_path).get(identifier)["status"] == "speech"


def test_web_filters_playback_control_and_unknown_audio(tmp_path):
    recorder = LiveRecorder(tmp_path, FakeAudio(), FakeVAD())
    identifier = register_clip(recorder.store, "uncertain")
    with TestClient(create_app(recorder)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/waveform.mjs").status_code == 200
        assert client.get("/api/status").json()["waveform"]["points"] == []
        assert len(client.get("/api/devices").json()["items"]) == 1
        assert client.get("/api/recordings").json()["items"] == []
        assert len(client.get("/api/recordings?status=uncertain").json()["items"]) == 1
        assert client.get("/api/recordings?status=invalid").status_code == 422
        response = client.get(f"/api/recordings/{identifier}/audio")
        assert response.status_code == 200
        assert response.content.startswith(b"RIFF")
        assert client.get("/api/recordings/not-found/audio").status_code == 404
        assert client.post("/api/start", json={"device_id": 0}).status_code == 403
        headers = {"X-Recorder-Control": "1"}
        assert client.post("/api/start", json={"device_id": 0},
                           headers={**headers, "Origin": "https://other-site.test"}).status_code == 403
        response = client.post("/api/start", json={"device_id": 0}, headers=headers)
        assert response.status_code == 200
        assert response.json()["channels"] == 2
        assert response.json()["channel_mode"] == "auto"
        assert client.post("/api/start", json={"device_id": 0}, headers=headers).status_code == 409
        assert client.post("/api/stop", headers=headers).status_code == 200
        response = client.post("/api/start", json={"device_id": 0, "channels": None}, headers=headers)
        assert response.status_code == 200
        assert response.json()["channels"] == 2
        assert response.json()["channel_mode"] == "auto"
        assert client.post("/api/stop", headers=headers).status_code == 200
        response = client.post("/api/start", json={"device_id": 0, "channels": 1}, headers=headers)
        assert response.status_code == 200
        assert response.json()["channels"] == 1
        assert response.json()["channel_mode"] == "manual"
        assert client.post("/api/stop", headers=headers).status_code == 200
        assert client.post("/api/start", json={"device_id": 0, "channels": 3}, headers=headers).status_code == 422


def test_recording_path_cannot_escape_archive(tmp_path):
    store = RecordingStore(tmp_path / "archive")
    identifier = register_clip(store)
    with store.connect() as db:
        db.execute("UPDATE recordings SET path='../outside.wav' WHERE id=?", (identifier,))
    with pytest.raises(FileNotFoundError):
        store.audio_path(identifier)


def seed_paginated_archive(store):
    start = datetime(2026, 10, 23, 9, tzinfo=timezone(timedelta(hours=7)))
    statuses = ["speech", "uncertain", "non-speech", "pending"]
    rows = [(f"r{i:04d}", f"archive/clip_{i}.wav",
             (start+timedelta(seconds=i)).isoformat(), 1.0, "USB", 48000, 2,
             statuses[i % 4]) for i in range(125)]
    rows.append(("previous-day", "archive/previous.wav",
                 "2026-10-22T23:59:59+07:00", 1.0, "USB", 48000, 2, "speech"))
    with store.connect() as db:
        db.executemany("INSERT INTO recordings "
                       "(id,path,started_at,duration,device,samplerate,channels,status) "
                       "VALUES (?,?,?,?,?,?,?,?)", rows)


def test_archive_pagination_reaches_all_records_and_filters_local_date(tmp_path):
    store = RecordingStore(tmp_path)
    seed_paginated_archive(store)
    sentinel = tmp_path / "untouched.wav"
    sf.write(sentinel, active_block(), 16000)
    original = sentinel.read_bytes()
    first = store.listing("all")
    assert first["total"] == 126
    assert first["page_size"] == 20
    assert first["total_pages"] == 7
    assert [item["id"] for item in first["items"]] == [f"r{i:04d}" for i in range(124, 104, -1)]
    identifiers = [item["id"] for page in range(1, 8)
                   for item in store.listing("all", page=page)["items"]]
    assert len(identifiers) == len(set(identifiers)) == 126
    assert identifiers[-1] == "previous-day"
    last = store.listing("all", page=999)
    assert last["page"] == 7
    assert len(last["items"]) == 6
    selected = store.listing("all", page=7, recorded_date=date(2026, 10, 23))
    assert selected["total"] == 125
    assert len(selected["items"]) == 5
    speech = store.listing("speech", recorded_date=date(2026, 10, 23))
    assert speech["total"] == 32
    assert speech["counts"] == {"speech": 32, "uncertain": 31, "non-speech": 31, "pending": 31}
    assert all(item["status"] == "speech" for item in speech["items"])
    previous = store.listing("all", recorded_date=date(2026, 10, 22))
    assert [item["id"] for item in previous["items"]] == ["previous-day"]
    empty = store.listing("all", page=7, recorded_date=date(2026, 10, 24))
    assert empty["page"] == 1
    assert empty["total"] == empty["total_pages"] == 0
    assert empty["items"] == []
    assert empty["counts"] == {}
    assert store.listing("all")["total"] == 126
    assert sentinel.read_bytes() == original


def test_pagination_stays_consistent_for_identical_start_times(tmp_path):
    store = RecordingStore(tmp_path)
    seed_paginated_archive(store)
    with store.connect() as db:
        db.execute("UPDATE recordings SET started_at='2026-10-23T09:00:00+07:00' "
                   "WHERE id IN ('r0000','r0001')")
    last = store.listing("all", page=7, recorded_date=date(2026, 10, 23))
    assert [item["id"] for item in last["items"]][-2:] == ["r0001", "r0000"]


def test_web_pagination_and_date_validation(tmp_path):
    recorder = LiveRecorder(tmp_path, FakeAudio(), FakeVAD())
    seed_paginated_archive(recorder.store)
    with TestClient(create_app(recorder)) as client:
        default = client.get("/api/recordings").json()
        assert default["total"] == 33
        assert len(default["items"]) == 20
        page = client.get("/api/recordings?status=all&page=7&date=2026-10-23").json()
        assert page["date"] == "2026-10-23"
        assert page["total"] == 125
        assert len(page["items"]) == 5
        assert page["page_size"] == 20
        pending = client.get("/api/recordings?status=pending&date=2026-10-23").json()
        assert pending["total"] == 31
        assert all(item["status"] == "pending" for item in pending["items"])
        for query in ["page=0", "page=-1", "page_size=0", "page_size=101",
                      "date=not-a-date", "date=2026-02-30"]:
            assert client.get(f"/api/recordings?{query}").status_code == 422
