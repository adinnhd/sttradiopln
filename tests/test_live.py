from datetime import datetime
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
    def query_hostapis(self):
        return [{"name": "Fake API"}]

    def query_devices(self):
        return [{"name": "USB HT", "hostapi": 0, "max_input_channels": 2,
                 "default_samplerate": 16000},
                {"name": "Speaker", "hostapi": 0, "max_input_channels": 0}]

    def check_input_settings(self, **kwargs):
        pass

    def InputStream(self, **kwargs):
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
        assert recorder.status()["vad_config"]["min_speech_duration_ms"] == 200
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
        backend.stream.blocks.put(active_block())
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
        assert client.post("/api/start", json={"device_id": 0}, headers=headers).status_code == 200
        assert client.post("/api/start", json={"device_id": 0}, headers=headers).status_code == 409
        assert client.post("/api/stop", headers=headers).status_code == 200


def test_recording_path_cannot_escape_archive(tmp_path):
    store = RecordingStore(tmp_path / "archive")
    identifier = register_clip(store)
    with store.connect() as db:
        db.execute("UPDATE recordings SET path='../outside.wav' WHERE id=?", (identifier,))
    with pytest.raises(FileNotFoundError):
        store.audio_path(identifier)
