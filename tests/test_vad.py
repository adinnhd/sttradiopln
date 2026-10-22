import csv
from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import silero_vad
import torch

from main import process_segments, process_wav
from radio_recorder.report import save_csv
from radio_recorder.segmenter import Config, SoundTriggeredRecorder
from radio_recorder.vad import (
    SAMPLE_RATE, WINDOW_SAMPLES, SileroVAD, VADConfig, VADResult,
    analyze_files, classify, print_summary,
)


class ProbabilityModel:
    """Probabilitas terkontrol, post-processing Silero tetap yang asli."""
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.reset_count = 0
        self.chunks = []

    def reset_states(self):
        self.index = 0
        self.reset_count += 1

    def __call__(self, chunk, rate):
        assert rate == SAMPLE_RATE
        assert chunk.shape == (WINDOW_SAMPLES,)
        assert chunk.dtype == torch.float32
        assert not torch.is_grad_enabled()
        self.chunks.append(chunk.clone())
        value = self.probabilities[self.index % len(self.probabilities)]
        self.index += 1
        if isinstance(value, Exception):
            raise value
        return torch.tensor(value, dtype=torch.float64)


def use_model(monkeypatch, probabilities):
    model = ProbabilityModel(probabilities)
    monkeypatch.setattr(silero_vad, "load_silero_vad", lambda **kwargs: model)
    return model


def write_wav(tmp_path, name="clip.wav", samples=16000, channels=1, sr=16000):
    path = tmp_path / name
    sf.write(path, np.full((samples, channels), 0.1, dtype=np.float32), sr)
    return path


@pytest.mark.parametrize("probabilities,expected", [
    ([0.8] * 32, "speech"),
    ([0.01] * 32, "non-speech"),
    ([0.2] * 32, "uncertain"),
    ([0.9] + [0.01] * 31, "uncertain"),  # Spike tunggal tidak menjadi non-speech.
    ([0.1] * 32, "uncertain"),  # Tepat di ambang non-speech tetap ditinjau.
])
def test_classification_uses_official_timestamps(tmp_path, monkeypatch, probabilities, expected):
    use_model(monkeypatch, probabilities)
    result = SileroVAD().analyze_file(write_wav(tmp_path, samples=32 * 512))
    assert result.vad_status == expected
    assert result.speech_probability_max == max(probabilities)
    assert result.speech_probability_mean == pytest.approx(np.mean(probabilities))
    assert result.vad_error == ""
    if expected == "speech":
        assert result.speech_duration_seconds == 1.024
    else:
        assert result.speech_duration_seconds == 0


def test_brief_noisy_speech_survives_ten_second_tail(tmp_path, monkeypatch):
    # 224 ms dengan probabilitas moderat, lalu 10 s hening: mean sangat rendah.
    use_model(monkeypatch, [0.4] * 7 + [0.001] * 313)
    result = SileroVAD().analyze_file(write_wav(tmp_path, samples=320 * 512))
    assert result.speech_probability_mean < 0.01
    assert result.speech_duration_seconds == pytest.approx(.224)
    assert result.vad_status == "speech"


def test_default_200ms_rejects_128ms_spike_but_100ms_is_configurable(tmp_path, monkeypatch):
    path = write_wav(tmp_path, samples=317 * 512)
    use_model(monkeypatch, [0.4] * 4 + [0.001] * 313)
    assert VADConfig().min_speech_duration_ms == 200
    assert SileroVAD().analyze_file(path).vad_status == "uncertain"
    use_model(monkeypatch, [0.4] * 4 + [0.001] * 313)
    assert SileroVAD(VADConfig(min_speech_duration_ms=100)).analyze_file(path).vad_status == "speech"


def test_weak_noisy_speech_is_uncertain_not_non_speech(tmp_path, monkeypatch):
    use_model(monkeypatch, [0.15] * 4 + [0.001] * 313)
    result = SileroVAD().analyze_file(write_wav(tmp_path, samples=317 * 512))
    assert result.speech_probability_mean < 0.01
    assert result.vad_status == "uncertain"


def test_partial_window_padding_weighting_and_duration(tmp_path, monkeypatch):
    model = use_model(monkeypatch, [0.01] * 9 + [0.8])
    path = write_wav(tmp_path, samples=9 * 512 + 100)
    result = SileroVAD().analyze_file(path)
    assert len(model.chunks) == 10
    assert torch.count_nonzero(model.chunks[-1][100:]) == 0
    assert result.speech_probability_mean == pytest.approx((9 * 512 * .01 + 100 * .8) / (9 * 512 + 100))
    assert result.speech_duration_seconds == 0
    assert result.vad_status == "uncertain"
    use_model(monkeypatch, [0.9])
    result = SileroVAD().analyze_file(path)
    assert result.speech_duration_seconds == pytest.approx((9 * 512 + 100) / 16000)


def test_short_clip_is_uncertain(tmp_path, monkeypatch):
    use_model(monkeypatch, [0.001])
    result = SileroVAD().analyze_file(write_wav(tmp_path, samples=200))
    assert result.vad_status == "uncertain"


@pytest.mark.parametrize("sr", [8000, 16000, 44100, 48000])
def test_resampling_only_changes_memory(tmp_path, monkeypatch, sr):
    model = use_model(monkeypatch, [0.7])
    path = write_wav(tmp_path, samples=sr, sr=sr)
    original = path.read_bytes()
    result = SileroVAD().analyze_file(path)
    assert result.vad_status == "speech"
    assert result.speech_duration_seconds == 1
    assert len(model.chunks) == 32
    assert path.read_bytes() == original
    assert sf.info(path).samplerate == sr


def test_stereo_is_evaluated_separately_without_phase_cancellation(tmp_path, monkeypatch):
    class SignalModel(ProbabilityModel):
        def __call__(self, chunk, rate):
            return torch.tensor(.9 if chunk.abs().max() > .05 else .001)

    model = SignalModel([])
    monkeypatch.setattr(silero_vad, "load_silero_vad", lambda **kwargs: model)
    path = tmp_path / "stereo.wav"
    # Kanal berlawanan fase akan hilang jika dirata-rata menjadi mono.
    sf.write(path, np.tile([.1, -.1], (16000, 1)), 16000)
    analyzer = SileroVAD()
    assert analyzer.analyze_file(path).vad_status == "speech"
    assert analyzer.analyze_file(path).vad_status == "speech"
    assert model.reset_count == 4
    sf.write(path, np.tile([0., .1], (16000, 1)), 16000)
    assert analyzer.analyze_file(path).vad_status == "speech"


@pytest.mark.parametrize("failure", [ImportError("silero tidak tersedia"), OSError("model rusak")])
def test_model_load_failure_keeps_wav_and_reports_error(tmp_path, monkeypatch, caplog, failure):
    def fail(**kwargs):
        raise failure
    monkeypatch.setattr(silero_vad, "load_silero_vad", fail)
    paths = [write_wav(tmp_path, name=f"{i}.wav") for i in range(2)]
    originals = [path.read_bytes() for path in paths]
    results = analyze_files(paths)
    assert len(results) == 2
    for path, original in zip(paths, originals):
        result = results[path.name]
        assert result.vad_status == "uncertain"
        assert result.speech_probability_max is None
        assert result.speech_duration_seconds is None
        assert str(failure) in result.vad_error
        assert path.read_bytes() == original
        assert path.name in caplog.text


def test_inference_failure_does_not_stop_next_file(tmp_path, monkeypatch):
    model = use_model(monkeypatch, [RuntimeError("inferensi gagal")])
    analyzer = SileroVAD()
    path = write_wav(tmp_path)
    assert analyzer.analyze_file(path).vad_status == "uncertain"
    model.probabilities = [.8]
    assert analyzer.analyze_file(path).vad_status == "speech"
    assert model.reset_count == 2


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probability_is_error(tmp_path, monkeypatch, probability):
    use_model(monkeypatch, [probability])
    result = SileroVAD().analyze_file(write_wav(tmp_path))
    assert result.vad_status == "uncertain"
    assert result.vad_error
    assert result.speech_probability_mean is None


def test_empty_and_missing_wav_are_uncertain(tmp_path):
    path = write_wav(tmp_path, samples=0)
    for file in [path, tmp_path / "missing.wav"]:
        result = SileroVAD().analyze_file(file)
        assert result.vad_status == "uncertain"
        assert result.vad_error


@pytest.mark.parametrize("kwargs", [
    {"speech_threshold": float("nan")}, {"non_speech_threshold": .5},
    {"speech_threshold": 1.1}, {"non_speech_threshold": -.1},
    {"min_speech_duration_ms": -1}, {"min_silence_duration_ms": -1},
    {"min_analysis_seconds": 0}, {"min_analysis_seconds": float("inf")},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        VADConfig(**kwargs)


def test_threshold_configuration_changes_decision():
    config = VADConfig()
    assert classify(.08, 0, 2, config) == "non-speech"
    assert classify(.08, 0, 2, replace(config, non_speech_threshold=.05)) == "uncertain"


def test_main_runs_vad_only_after_all_wavs_close(tmp_path, monkeypatch):
    source = write_wav(tmp_path)
    events = []
    original_finish = SoundTriggeredRecorder.finish
    original_analyze = SileroVAD.analyze_file
    def finish(self):
        result = original_finish(self)
        assert self.writer is None
        events.append("closed")
        return result
    def analyze(self, path):
        assert events == ["closed"]
        with sf.SoundFile(path) as wav:
            assert wav.frames == 16000
        events.append("vad")
        return original_analyze(self, path)
    monkeypatch.setattr(SoundTriggeredRecorder, "finish", finish)
    monkeypatch.setattr(SileroVAD, "analyze_file", analyze)
    use_model(monkeypatch, [.8])
    output = process_wav(source, tmp_path / "output", -50, 1, 10, 30)
    assert events == ["closed", "vad"]
    with (output / "segments.csv").open() as file:
        row = next(csv.DictReader(file))
    assert row["vad_status"] == "speech"
    assert row["speech_duration_seconds"] == "1.000000"
    assert row["vad_error"] == ""
    assert (output / "vad_config.json").is_file()


def test_pipeline_error_writes_csv_plot_and_summary(tmp_path, monkeypatch, capsys):
    use_model(monkeypatch, [RuntimeError("model error")])
    source = write_wav(tmp_path)
    original = source.read_bytes()
    output = process_wav(source, tmp_path / "output", -50, 1, 10, 30)
    with (output / "segments.csv").open() as file:
        row = next(csv.DictReader(file))
    assert row["vad_status"] == "uncertain"
    assert "model error" in row["vad_error"]
    assert row["speech_probability_max"] == ""
    assert row["speech_duration_seconds"] == ""
    assert (output / "rekaman_0001.wav").is_file()
    assert (output / "aktivitas_suara.png").is_file()
    assert "speech=0, non-speech=0, uncertain=1 (error=1)" in capsys.readouterr().out
    assert source.read_bytes() == original


def test_reclassify_legacy_csv_keeps_metadata_and_all_wavs(tmp_path, monkeypatch):
    path = write_wav(tmp_path)
    csv_path = tmp_path / "segments.csv"
    csv_path.write_text("nomor,file,mulai_detik,selesai_detik,durasi_detik,catatan\n"
                        "1,clip.wav,12.345,13.345,1.000,arsip\n"
                        "2,missing.wav,15.000,16.000,1.000,hilang\n")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    use_model(monkeypatch, [.001])
    for _ in range(2):  # Mengulang tidak menduplikasi kolom metadata.
        process_segments(tmp_path)
        with csv_path.open() as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            assert len(reader.fieldnames) == len(set(reader.fieldnames))
        assert rows[0]["vad_status"] == "non-speech"
        assert rows[0]["mulai_detik"] == "12.345"
        assert rows[0]["catatan"] == "arsip"
        assert rows[1]["vad_status"] == "uncertain"
        assert rows[1]["vad_error"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_legacy_report_call_and_summary(tmp_path, capsys):
    save_csv([], tmp_path / "empty.csv")
    assert "vad_status" in (tmp_path / "empty.csv").read_text()
    print_summary({})
    assert "speech=0, non-speech=0, uncertain=0" in capsys.readouterr().out
    print_summary({"a": VADResult(vad_status="speech"),
                   "b": VADResult(vad_status="non-speech"), "c": VADResult()})
    assert "speech=1, non-speech=1, uncertain=1" in capsys.readouterr().out


def test_existing_cli_and_new_directory_mode(tmp_path):
    source = write_wav(tmp_path)
    project = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "runs"
    old_command = subprocess.run([
        sys.executable, str(project / "main.py"), str(source), "--output", str(output_root),
        "--threshold-db", "-50", "--prebuffer", "1", "--silence", "10", "--frame-ms", "30",
    ], capture_output=True, text=True)
    assert old_command.returncode == 0, old_command.stderr
    directory = next(output_root.iterdir())
    new_command = subprocess.run([
        sys.executable, str(project / "main.py"), "--segments-dir", str(directory),
        "--vad-speech-threshold", ".3", "--vad-non-speech-threshold", ".05",
    ], capture_output=True, text=True)
    assert new_command.returncode == 0, new_command.stderr
    assert "VAD:" in new_command.stdout


@pytest.mark.parametrize("kind,expected", [
    ("beep", "non-speech"), ("buzz", "non-speech"), ("noise", "non-speech"),
    ("ambiguous_beep", "uncertain"),
])
def test_real_silero_on_rms_segmented_non_speech(tmp_path, kind, expected):
    # Sinyal sungguhan melewati RMS -> WAV -> model Silero yang dibundel paket.
    sr = 16000
    t = np.arange(sr * 2) / sr
    if kind == "beep":
        signal = .15 * np.sin(2 * np.pi * 2000 * t)
    elif kind == "ambiguous_beep":
        signal = .15 * np.sin(2 * np.pi * 1000 * t) * ((t % .5) < .15)
    elif kind == "buzz":
        signal = .15 * np.sign(np.sin(2 * np.pi * 100 * t))
    else:
        signal = np.random.default_rng(42).normal(0, .05, len(t))
    engine = SoundTriggeredRecorder(sr, 1, tmp_path, Config())
    audio = np.concatenate([np.zeros(sr), signal, np.zeros(sr * 11)]).astype(np.float32)
    for start in range(0, len(audio), 480):
        engine.process(audio[start:start + 480, None])
    segments = engine.finish()
    assert len(segments) == 1
    path = tmp_path / segments[0].filename
    before = path.read_bytes()
    result = SileroVAD().analyze_file(path)
    assert not result.vad_error
    assert result.vad_status == expected, result
    assert result.speech_duration_seconds == 0
    assert path.read_bytes() == before


@pytest.mark.parametrize("probability", [.8, .001, .2, RuntimeError("VAD gagal")])
def test_all_vad_outcomes_preserve_rms_wavs_and_timestamps(tmp_path, monkeypatch, probability):
    use_model(monkeypatch, [probability])
    sr = 16000
    audio = np.zeros((30 * sr, 2), dtype=np.float32)
    audio[2 * sr:3 * sr, 1] = .1
    audio[20 * sr:21 * sr, 1] = -.1
    source = tmp_path / "source.wav"
    sf.write(source, audio, sr, subtype="PCM_16")
    reference = SoundTriggeredRecorder(sr, 2, tmp_path / "rms_only", Config())
    with sf.SoundFile(source) as input_file:
        while len(block := input_file.read(480, dtype="float32", always_2d=True)):
            reference.process(block)
    segments = reference.finish()
    output = process_wav(source, tmp_path / "with_vad", -50, 1, 10, 30)
    with (output / "segments.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(segments) == 2
    for segment, row in zip(segments, rows):
        assert (output / segment.filename).read_bytes() == (reference.output_dir / segment.filename).read_bytes()
        assert row["mulai_detik"] == f"{segment.start_seconds:.3f}"
        assert row["selesai_detik"] == f"{segment.end_seconds:.3f}"


def test_real_model_uses_local_package_without_hub_or_network(tmp_path, monkeypatch):
    import socket
    import urllib.request

    def reject_network(*args, **kwargs):
        raise AssertionError("VAD lokal tidak boleh mengakses jaringan/torch.hub")
    monkeypatch.setattr(torch.hub, "load", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(urllib.request, "urlopen", reject_network)
    result = SileroVAD().analyze_file(write_wav(tmp_path))
    assert result.vad_error == ""


@pytest.mark.parametrize("snr_db", [0, -5])
def test_real_noisy_radio_sample_when_available(tmp_path, snr_db):
    source = (Path(__file__).resolve().parents[1] / "data/processed"
              / "ht_20261022_092400/rekaman_0003.wav")
    if not source.is_file():
        pytest.skip("Sampel radio lokal opsional tidak tersedia")
    original = source.read_bytes()
    audio, sr = sf.read(source, dtype="float32", always_2d=True)
    signal_rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    noise = np.random.default_rng(42).normal(
        0, signal_rms / (10 ** (snr_db / 20)), audio.shape)
    noisy = (audio + noise).astype(np.float32)
    noisy /= max(1., float(np.abs(noisy).max()))
    engine = SoundTriggeredRecorder(sr, audio.shape[1], tmp_path, Config())
    for start in range(0, len(noisy), 480):
        engine.process(noisy[start:start + 480])
    segments = engine.finish()
    assert segments
    analyzer = SileroVAD()
    for segment in segments:
        result = analyzer.analyze_file(tmp_path / segment.filename)
        assert not result.vad_error
        assert result.vad_status in ("speech", "uncertain"), result
    assert source.read_bytes() == original
