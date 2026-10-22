"""CLI offline untuk mendeteksi dan menyimpan bagian bersuara dari WAV.

Contoh:
    python main.py data/baseline/ht.wav --threshold-db -50
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import soundfile as sf

from radio_recorder.report import read_segment_csv, save_csv, save_plot, update_csv_vad
from radio_recorder.segmenter import Config, SoundTriggeredRecorder
from radio_recorder.vad import VADConfig, analyze_files, print_summary


def create_run_directory(base_dir: Path, source_stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_stem)
    root = base_dir / f"{safe_stem}_{timestamp}"
    candidate = root
    idx = 2
    while candidate.exists():
        candidate = Path(f"{root}_{idx}")
        idx += 1
    candidate.mkdir(parents=True)
    return candidate


def process_wav(path: Path, output_root: Path, threshold_db: float,
                prebuffer: float, silence: float, frame_ms: int,
                vad_config: VADConfig | None = None) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    with sf.SoundFile(path, mode="r") as infile:
        if infile.format != "WAV":
            raise ValueError("Untuk MVP, masukkan file .wav (ekspor sebagai WAV dari Audacity).")
        if infile.frames == 0:
            raise ValueError("Rekaman kosong.")
        frame_size = round(infile.samplerate * frame_ms / 1000)
        if frame_size < 1:
            raise ValueError("Ukuran frame terlalu kecil")
        output_dir = create_run_directory(output_root, path.stem)
        engine = SoundTriggeredRecorder(
            infile.samplerate, infile.channels, output_dir,
            Config(threshold_db=threshold_db,
                   prebuffer_seconds=prebuffer,
                   silence_seconds=silence),
        )
        print(f"Memproses: {path}")
        print(f"Format: {infile.samplerate} Hz, {infile.channels} kanal, "
              f"durasi {infile.frames / infile.samplerate:.1f} detik")
        try:
            while True:
                block = infile.read(frame_size, dtype="float32", always_2d=True)
                if len(block) == 0:
                    break
                engine.process(block)
        finally:
            segments = engine.finish()

    # Semua writer WAV sudah ditutup oleh finish(); VAD tidak masuk loop RMS.
    config = vad_config if vad_config is not None else VADConfig()
    print(f"Menjalankan Silero VAD pada {len(segments)} WAV selesai...")
    results = analyze_files([output_dir / seg.filename for seg in segments], config)
    save_csv(segments, output_dir / "segments.csv", results)
    save_vad_config(output_dir, config)
    save_plot(engine.levels, segments, threshold_db, output_dir / "aktivitas_suara.png")
    recording_duration = sum(s.duration_seconds for s in segments)
    print(f"\nJumlah segmen: {len(segments)}")
    print(f"Total audio disimpan: {recording_duration:.1f} detik")
    print(f"Durasi sumber: {engine.cursor / engine.samplerate:.1f} detik")
    print_summary(results)
    print(f"Output: {output_dir.resolve()}")
    if not segments:
        print("Tidak ada suara terdeteksi. Coba turunkan threshold, misalnya -60 dBFS.")
    return output_dir


def save_vad_config(directory: Path, config: VADConfig) -> None:
    (directory / "vad_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")


def process_segments(directory: Path, vad_config: VADConfig | None = None) -> Path:
    """Evaluasi ulang segmen RMS lama tanpa menyentuh WAV atau mengulang RMS."""
    _, rows = read_segment_csv(directory / "segments.csv")
    config = vad_config if vad_config is not None else VADConfig()
    print(f"Menjalankan Silero VAD pada {len(rows)} WAV di {directory}...")
    results = analyze_files([directory / row["file"] for row in rows], config)
    update_csv_vad(directory / "segments.csv", results)
    save_vad_config(directory, config)
    print_summary(results)
    print(f"Metadata: {(directory / 'segments.csv').resolve()}")
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segmentasi RMS radio HT, lalu klasifikasi WAV dengan Silero VAD lokal."
    )
    parser.add_argument("input", type=Path, nargs="?", help="File WAV asli hasil rekaman HT")
    parser.add_argument("--segments-dir", type=Path,
                        help="Klasifikasi ulang folder segmen RMS dengan segments.csv")
    parser.add_argument("--output", type=Path, default=Path("data/processed"),
                        help="Folder induk hasil (default: data/processed)")
    parser.add_argument("--threshold-db", type=float, default=-50.0,
                        help="Ambang RMS dBFS; lebih negatif = lebih sensitif (default -50)")
    parser.add_argument("--prebuffer", type=float, default=1.0,
                        help="Audio sebelum trigger yang ikut disimpan, detik (default 1)")
    parser.add_argument("--silence", type=float, default=10.0,
                        help="Lama hening sebelum stop, detik (default 10)")
    parser.add_argument("--frame-ms", type=int, default=30,
                        help="Ukuran blok deteksi dalam milidetik (default 30)")
    parser.add_argument("--vad-speech-threshold", type=float, default=0.35,
                        help="Ambang awal speech Silero (default 0.35)")
    parser.add_argument("--vad-non-speech-threshold", type=float, default=0.10,
                        help="Non-speech hanya jika probabilitas max < nilai ini (default 0.10)")
    parser.add_argument("--vad-min-speech-ms", type=int, default=100,
                        help="Durasi minimum rentang speech Silero, ms (default 100)")
    parser.add_argument("--vad-min-silence-ms", type=int, default=100,
                        help="Jeda pemisah rentang speech VAD, ms (default 100); bukan timeout RMS")
    parser.add_argument("--vad-min-analysis-seconds", type=float, default=0.25,
                        help="WAV lebih pendek tanpa speech menjadi uncertain (default 0.25)")
    args = parser.parse_args()
    if (args.input is None) == (args.segments_dir is None):
        parser.error("Pilih satu: input WAV atau --segments-dir")
    if not 10 <= args.frame_ms <= 100:
        parser.error("--frame-ms harus 10 sampai 100")
    if not -120 < args.threshold_db <= 0:
        parser.error("--threshold-db harus > -120 dan <= 0")
    if args.prebuffer < 0 or args.silence <= 0:
        parser.error("--prebuffer >= 0 dan --silence > 0")
    try:
        vad_config = VADConfig(
            speech_threshold=args.vad_speech_threshold,
            non_speech_threshold=args.vad_non_speech_threshold,
            min_speech_duration_ms=args.vad_min_speech_ms,
            min_silence_duration_ms=args.vad_min_silence_ms,
            min_analysis_seconds=args.vad_min_analysis_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        if args.segments_dir is not None:
            process_segments(args.segments_dir, vad_config)
        else:
            process_wav(args.input, args.output, args.threshold_db,
                        args.prebuffer, args.silence, args.frame_ms, vad_config)
    except (OSError, ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
