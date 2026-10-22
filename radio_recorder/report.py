"""Membuat CSV dan grafik diagnostik level suara vs threshold."""

import csv
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")  # Tetap bisa dijalankan tanpa desktop/GUI.
import matplotlib.pyplot as plt

from .segmenter import Segment
from .vad import VAD_COLUMNS, VADResult


def save_csv(segments: list[Segment], filepath: Path,
             vad_results: dict[str, VADResult] | None = None) -> None:
    with filepath.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["nomor", "file", "mulai_detik", "selesai_detik", "durasi_detik",
                         *VAD_COLUMNS])
        for seg in segments:
            result = (vad_results or {}).get(seg.filename, VADResult())
            writer.writerow([
                seg.number, seg.filename,
                f"{seg.start_seconds:.3f}",
                f"{seg.end_seconds:.3f}",
                f"{seg.duration_seconds:.3f}",
                *result.csv_fields().values(),
            ])


def read_segment_csv(filepath: Path) -> tuple[list[str], list[dict[str, str]]]:
    with filepath.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        if "file" not in fields:
            raise ValueError("segments.csv harus memiliki kolom file")
        rows = list(reader)
    names = [row["file"] for row in rows]
    if any(not name or Path(name).name != name or Path(name).suffix.lower() != ".wav"
           for name in names):
        raise ValueError("Kolom file harus nama WAV lokal dalam folder segmen")
    if len(set(names)) != len(names):
        raise ValueError("Nama file duplikat dalam segments.csv")
    return fields, rows


def update_csv_vad(filepath: Path, results: dict[str, VADResult]) -> None:
    """Tambahkan/perbarui VAD pada CSV V1, pertahankan semua metadata lain."""
    fields, rows = read_segment_csv(filepath)
    fields += [name for name in VAD_COLUMNS if name not in fields]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=filepath.parent, delete=False) as output:
            temporary = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                row.update(results[row["file"]].csv_fields())
                writer.writerow(row)
        os.replace(temporary, filepath)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_plot(
    levels: list[tuple[float, float]],
    segments: list[Segment],
    threshold_db: float,
    filepath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    if levels:
        times, db = zip(*levels)
        ax.plot(times, db, linewidth=0.8, label="Level RMS (dBFS)")
    ax.axhline(threshold_db, linestyle="--", linewidth=1.5,
               color="tab:red", label=f"Threshold {threshold_db:g} dBFS")
    for idx, segment in enumerate(segments):
        ax.axvspan(segment.start_seconds, segment.end_seconds,
                   color="tab:green", alpha=0.15,
                   label="Disimpan ke WAV" if idx == 0 else None)
    ax.set_title("Pemeriksaan deteksi aktivitas suara HT")
    ax.set_xlabel("Waktu dari awal rekaman asli (detik)")
    ax.set_ylabel("Level audio (dBFS)")
    ax.set_ylim(bottom=-125, top=5)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(filepath, dpi=140)
    plt.close(fig)
