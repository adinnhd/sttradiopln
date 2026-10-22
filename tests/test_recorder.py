from pathlib import Path

import numpy as np
import soundfile as sf

from radio_recorder.segmenter import Config, SoundTriggeredRecorder
from main import process_wav


def synthetic_radio(sr=16000):
    # 12 detik: transmis 2-3 dan 3.6-4.4 menjadi satu segmen;
    # transmisi 7-8 segmen berikutnya.
    x = np.zeros((12 * sr, 1), dtype=np.float32)
    for start, end in [(2.0, 3.0), (3.6, 4.4), (7.0, 8.0)]:
        begin, stop = round(start * sr), round(end * sr)
        t = np.arange(stop - begin) / sr
        x[begin:stop, 0] = 0.10 * np.sin(2 * np.pi * 440 * t)
    return x


def test_segments_and_prebuffer(tmp_path):
    sr = 16000
    engine = SoundTriggeredRecorder(sr, 1, tmp_path, Config(silence_seconds=2.0))
    data = synthetic_radio(sr)
    blocksize = 480
    for index in range(0, len(data), blocksize):
        engine.process(data[index:index+blocksize])
    segments = engine.finish()
    assert len(segments) == 2
    # Prebuffer mengambil kurang-lebih detik 1-2 sebelum trigger pada detik 2.
    assert 0.98 <= segments[0].start_seconds <= 1.02
    assert 6.30 < segments[0].end_seconds < 6.50
    assert 6.30 < segments[1].start_seconds < 7.02
    assert 9.95 < segments[1].end_seconds < 10.1
    for seg in segments:
        y, actual_sr = sf.read(tmp_path / seg.filename)
        assert len(y) == seg.end_sample - seg.start_sample
        assert actual_sr == sr


def test_silence_does_not_create_file(tmp_path):
    engine = SoundTriggeredRecorder(16000, 1, tmp_path, Config())
    silent = np.zeros((480, 1), dtype=np.float32)
    for _ in range(500):
        engine.process(silent)
    assert engine.finish() == []
    assert not list(tmp_path.glob('*.wav'))


def test_mono_and_stereo_preserved(tmp_path):
    sr = 16000
    stereo = np.zeros((sr * 4, 2), dtype=np.float32)
    stereo[sr:2*sr, 1] = 0.2
    engine = SoundTriggeredRecorder(sr, 2, tmp_path, Config())
    for begin in range(0, len(stereo), 480):
        engine.process(stereo[begin:begin+480])
    assert len(engine.finish()) == 1
    recorded, _ = sf.read(tmp_path / 'rekaman_0001.wav', always_2d=True)
    assert recorded.shape[1] == 2
    assert recorded[:, 1].max() > 0.1
    assert recorded[:, 0].max() == 0


def test_end_of_input_finalizes_active_recording(tmp_path):
    sr = 16000
    engine = SoundTriggeredRecorder(sr, 1, tmp_path, Config())
    engine.process(np.full((480, 1), .3, dtype=np.float32))
    segments = engine.finish()
    assert len(segments) == 1
    assert segments[0].end_seconds == 0.03


def test_default_rms_timeout_stays_ten_seconds(tmp_path):
    engine = SoundTriggeredRecorder(16000, 1, tmp_path, Config())
    engine.process(np.full((480, 1), .3, dtype=np.float32))
    for _ in range(333):
        engine.process(np.zeros((480, 1), dtype=np.float32))
    assert engine.writer is not None  # 9.99 detik belum menutup WAV.
    engine.process(np.zeros((480, 1), dtype=np.float32))
    assert engine.writer is None
    segments = engine.finish()
    assert len(segments) == 1
    assert segments[0].duration_seconds == 10.05
    assert engine.config.threshold_db == -50
    assert engine.config.prebuffer_seconds == 1


def test_end_to_end_creates_csv_and_plot(tmp_path):
    source = tmp_path / 'baseline.wav'
    sf.write(source, synthetic_radio(), 16000, subtype='PCM_16')
    output = process_wav(source, tmp_path / 'hasil', -50.0, 1.0, 2.0, 30)
    assert (output / 'segments.csv').is_file()
    assert (output / 'aktivitas_suara.png').is_file()
    assert len(list(output.glob('rekaman_*.wav'))) == 2
