"""Klasifikasi WAV yang SUDAH ditutup; tidak pernah menulis/menghapus audio.

Model JIT dimuat dari paket silero-vad lokal (tanpa torch.hub/API jaringan).
Satu instance dapat dipakai ulang, termasuk sebelum transkripsi di masa depan.
Instance bersifat serial karena model Silero menyimpan state antar-window.
"""

from collections import Counter
from dataclasses import dataclass
import logging
from math import gcd, isfinite
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf


logger = logging.getLogger(__name__)
VADStatus = Literal["speech", "non-speech", "uncertain"]
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 512  # 32 ms; wajib untuk model Silero pada 16 kHz.


@dataclass(frozen=True)
class VADConfig:
    # Default konservatif untuk recall; kalibrasikan dengan rekaman berlabel.
    speech_threshold: float = 0.35
    non_speech_threshold: float = 0.10
    min_speech_duration_ms: int = 100
    min_silence_duration_ms: int = 100
    min_analysis_seconds: float = 0.25

    def __post_init__(self) -> None:
        if not (isfinite(self.speech_threshold) and isfinite(self.non_speech_threshold)
                and 0 <= self.non_speech_threshold < self.speech_threshold <= 1):
            raise ValueError("VAD: 0 <= non_speech_threshold < speech_threshold <= 1")
        if self.min_speech_duration_ms < 0 or self.min_silence_duration_ms < 0:
            raise ValueError("Durasi minimum VAD harus >= 0 ms")
        if not isfinite(self.min_analysis_seconds) or self.min_analysis_seconds <= 0:
            raise ValueError("min_analysis_seconds VAD harus positif dan finite")


@dataclass(frozen=True)
class VADResult:
    speech_probability_max: float | None = None
    speech_probability_mean: float | None = None
    speech_duration_seconds: float | None = None
    vad_status: VADStatus = "uncertain"
    vad_error: str = ""

    def csv_fields(self) -> dict[str, str]:
        return {
            "speech_probability_max": _number(self.speech_probability_max),
            "speech_probability_mean": _number(self.speech_probability_mean),
            "speech_duration_seconds": _number(self.speech_duration_seconds),
            "vad_status": self.vad_status,
            "vad_error": self.vad_error,
        }


def _number(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


VAD_COLUMNS = list(VADResult().csv_fields())


def classify(max_probability: float, speech_duration_seconds: float,
             analyzed_seconds: float, config: VADConfig) -> VADStatus:
    """Mean/rasio speech tidak menjadi veto: ekor hening RMS bisa 10 detik.

    Bahkan satu window di atas ambang non-speech mencegah keputusan non-speech.
    Spike singkat dan cuplikan sangat pendek tetap uncertain.
    """
    if speech_duration_seconds > 0:
        return "speech"
    if (analyzed_seconds >= config.min_analysis_seconds
            and max_probability < config.non_speech_threshold):
        return "non-speech"
    return "uncertain"


class SileroVAD:
    def __init__(self, config: VADConfig | None = None):
        self.config = config if config is not None else VADConfig()
        self._model = None
        self._load_error: str | None = None

    def _load(self) -> None:
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        if self._model is not None:
            return
        try:
            import torch
            from scipy.signal import resample_poly
            from silero_vad import load_silero_vad, get_speech_timestamps_from_probs

            self._torch = torch
            self._resample = resample_poly
            self._timestamps = get_speech_timestamps_from_probs
            self._model = load_silero_vad(onnx=False)
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise

    def analyze_file(self, path: Path) -> VADResult:
        """Kegagalan file/model menjadi uncertain dengan metrik kosong, bukan nol."""
        try:
            self._load()
            return self._analyze(Path(path))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error("VAD gagal untuk %s: %s", path, error)
            return VADResult(vad_error=error)

    def _analyze(self, path: Path) -> VADResult:
        with sf.SoundFile(path, mode="r") as source:
            if source.format != "WAV":
                raise ValueError("Input VAD harus WAV hasil segmentasi")
            audio = source.read(dtype="float32", always_2d=True)
            source_rate = source.samplerate
        if len(audio) == 0 or not np.isfinite(audio).all():
            raise ValueError("WAV kosong atau mengandung sampel non-finite")

        # Resampling hanya salinan RAM; WAV asli tetap utuh. Evaluasi tiap kanal
        # agar speech satu sisi/fase berlawanan tidak hilang karena downmix.
        divisor = gcd(source_rate, SAMPLE_RATE)
        if source_rate != SAMPLE_RATE:
            audio = self._resample(audio, SAMPLE_RATE // divisor,
                                   source_rate // divisor, axis=0).astype(np.float32)
        sample_count = len(audio)
        window_count = (sample_count + WINDOW_SAMPLES - 1) // WINDOW_SAMPLES
        probabilities = np.zeros(window_count, dtype=np.float64)
        with self._torch.inference_mode():
            for channel in range(audio.shape[1]):
                self._model.reset_states()  # Isolasi file DAN kanal.
                for index, start in enumerate(range(0, sample_count, WINDOW_SAMPLES)):
                    chunk = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
                    valid = audio[start:start + WINDOW_SAMPLES, channel]
                    chunk[:len(valid)] = valid
                    probability = float(self._model(
                        self._torch.from_numpy(chunk), SAMPLE_RATE).item())
                    if not isfinite(probability) or not 0 <= probability <= 1:
                        raise ValueError("Probabilitas Silero tidak valid")
                    probabilities[index] = max(probabilities[index], probability)

        # Gunakan post-processing resmi Silero pada probabilitas maksimum kanal.
        # Tanpa padding speech: durasi tidak membengkak oleh context tambahan.
        timestamps = self._timestamps(
            probabilities.tolist(), sampling_rate=SAMPLE_RATE,
            threshold=self.config.speech_threshold,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_ms=0, audio_length_samples=sample_count,
        )
        speech_duration = sum(t["end"] - t["start"] for t in timestamps) / SAMPLE_RATE
        # Window terakhir di-zero-pad untuk inferensi, tetapi bobot/durasi
        # statistik hanya menghitung sampel asli, bukan padding.
        weights = np.minimum(WINDOW_SAMPLES,
                             sample_count - np.arange(window_count) * WINDOW_SAMPLES)
        maximum = float(probabilities.max())
        mean = float(np.average(probabilities, weights=weights))
        return VADResult(maximum, mean, speech_duration,
                         classify(maximum, speech_duration,
                                  sample_count / SAMPLE_RATE, self.config))


def analyze_files(paths: list[Path], config: VADConfig | None = None) -> dict[str, VADResult]:
    """Satu model untuk satu batch WAV; error per file tidak menghentikan batch."""
    analyzer = SileroVAD(config)
    return {path.name: analyzer.analyze_file(path) for path in paths}


def print_summary(results: dict[str, VADResult]) -> None:
    counts = Counter(result.vad_status for result in results.values())
    errors = sum(bool(result.vad_error) for result in results.values())
    print(f"VAD: speech={counts['speech']}, non-speech={counts['non-speech']}, "
          f"uncertain={counts['uncertain']} (error={errors})")
