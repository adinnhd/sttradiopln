import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

import main


def run_audio(path, output_root):
    return main.process_audio(path, output_root, -50, 1, 10, 30)


def test_wav_does_not_require_ffmpeg(tmp_path, monkeypatch):
    source = tmp_path / "original.wav"
    source.touch()
    monkeypatch.setattr(main.shutil, "which", lambda _: None)
    monkeypatch.setattr(main, "process_wav", lambda path, *args: path)
    assert run_audio(source, tmp_path / "runs") == source


def test_m4a_requires_ffmpeg(tmp_path, monkeypatch):
    source = tmp_path / "original.m4a"
    source.touch()
    monkeypatch.setattr(main.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="winget install Gyan.FFmpeg"):
        run_audio(source, tmp_path / "runs")


def test_missing_or_unsupported_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="File tidak ditemukan"):
        run_audio(tmp_path / "missing.m4a", tmp_path / "runs")
    source = tmp_path / "recording.txt"
    source.touch()
    with pytest.raises(ValueError, match="Format input"):
        run_audio(source, tmp_path / "runs")


@pytest.mark.parametrize("failure", ["conversion", "pipeline", None])
def test_temporary_wav_cleanup_and_original_preserved(tmp_path, monkeypatch, failure):
    source = tmp_path / "radio recording.M4A"
    source.write_bytes(b"original recording")
    temporary_paths = []
    monkeypatch.setattr(main.shutil, "which", lambda _: "ffmpeg")

    def convert(command, **kwargs):
        converted = Path(command[-1])
        temporary_paths.append(converted)
        converted.write_bytes(b"temporary WAV")
        return subprocess.CompletedProcess(command, 1 if failure == "conversion" else 0,
                                           stderr="invalid audio")

    def process(path, *args):
        assert path.is_file()
        assert path.stem == source.stem
        if failure == "pipeline":
            raise ValueError("pipeline failed")
        return tmp_path / "runs"

    monkeypatch.setattr(main.subprocess, "run", convert)
    monkeypatch.setattr(main, "process_wav", process)
    if failure:
        with pytest.raises(ValueError):
            run_audio(source, tmp_path / "runs")
    else:
        assert run_audio(source, tmp_path / "runs") == tmp_path / "runs"
    assert not temporary_paths[0].parent.exists()
    assert source.read_bytes() == b"original recording"


def test_real_m4a_cli_preserves_channels_rate_and_source(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg tidak tersedia")
    wav = tmp_path / "source.wav"
    source = tmp_path / "radio recording.m4a"
    sf.write(wav, np.full((44100, 2), .2, dtype=np.float32), 44100)
    subprocess.run([ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(wav),
                    "-c:a", "aac", str(source)], check=True, capture_output=True)
    digest = hashlib.sha256(source.read_bytes()).digest()
    result = subprocess.run([
        sys.executable, str(Path(main.__file__).resolve()), str(source),
        "--output", str(tmp_path / "runs"),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Mengonversi M4A" in result.stdout
    assert "VAD:" in result.stdout
    output = next((tmp_path / "runs").iterdir())
    assert output.name.startswith("radio_recording_")
    segment = sf.info(next(output.glob("rekaman_*.wav")))
    assert segment.samplerate == 44100
    assert segment.channels == 2
    assert (output / "segments.csv").is_file()
    assert (output / "aktivitas_suara.png").is_file()
    assert hashlib.sha256(source.read_bytes()).digest() == digest
