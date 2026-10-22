"""Deteksi *semua aktivitas audio* lewat RMS, bukan deteksi ucapan/AI."""

import numpy as np


def rms_dbfs(audio: np.ndarray) -> float:
    """Hitung RMS dBFS dari satu frame, pakai kanal dengan RMS tertinggi.

    Audio berbentuk (jumlah_sampel, jumlah_kanal), float32 di rentang [-1, 1].
    Hening digital menghasilkan -120 dBFS agar mudah diplot.
    """
    if audio.size == 0:
        return -120.0
    squared = np.square(audio.astype(np.float64))
    mean_square = np.mean(squared, axis=0)
    rms = float(np.sqrt(np.max(mean_square)))
    return max(-120.0, 20.0 * float(np.log10(max(rms, 1e-6))))


def is_active(audio: np.ndarray, threshold_db: float) -> tuple[bool, float]:
    """Kembalikan (aktif?, RMS dalam dBFS)."""
    level_db = rms_dbfs(audio)
    return level_db >= threshold_db, level_db
