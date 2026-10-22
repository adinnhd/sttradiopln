# Evaluasi Silero VAD — 22 Oktober 2026

Evaluasi memakai dua folder WAV hasil segmentasi RMS yang sudah ada. VAD hanya membaca WAV dan memperbarui metadata. Algoritma `detector.py` dan `segmenter.py` identik dengan berkas sebelum perubahan.

| Folder RMS | Jumlah WAV | speech | non-speech | uncertain | Error |
|---|---:|---:|---:|---:|---:|
| `data/processed/ht_20261022_090920` | 43 | 11 | 22 | 10 | 0 |
| `data/processed/ht_20261022_092400` | 12 | 4 | 2 | 6 | 0 |
| Total | 55 | 15 | 24 | 16 | 0 |

Konfigurasi: speech threshold 0.35, non-speech threshold 0.10, minimum speech 100 ms, minimum silence VAD 100 ms, minimum panjang untuk non-speech 0.25 detik. Model JIT lokal Silero 6.2.3, Python 3.12.3, PyTorch 2.14.1+cpu. Versi dependensi lengkap, konfigurasi, dan ringkasan tersedia di [summary.json](data/vad_evaluation/summary.json).

Kolom VAD sudah ditambahkan pada kedua `segments.csv`; metadata awal diarsipkan di [metadata_before](data/vad_evaluation/metadata_before). Semua nilai lima kolom V1 tetap sama. Perhitungan SHA-256 sebelum/sesudah menunjukkan **56 WAV tetap identik**, yaitu 55 segmen dan satu WAV sumber. Tidak ada WAV proyek yang dihapus, diedit, atau ditambahkan saat evaluasi metadata. Hash pembanding tersedia di [wav_sha256.json](data/vad_evaluation/wav_sha256.json).

Verifikasi hash dari root proyek:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib
import json
hashes = json.loads(Path('data/vad_evaluation/wav_sha256.json').read_text())
for name, expected in hashes.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print(f'{len(hashes)} WAV identik dengan sebelum integrasi VAD')
PY
```

Pipeline penuh juga dijalankan pada `data/baseline/ht.wav` berdurasi 1011.4 detik ke direktori sementara, dengan threshold RMS -50 dBFS, pre-buffer 1 detik, silence timeout 10 detik, frame 30 ms. Hasilnya 12 segmen, total 338.8 detik. Byte seluruh WAV dan seluruh kolom CSV identik dengan hasil evaluasi ulang folder `ht_20261022_092400`. Output sementara ini tidak disimpan sebagai duplikasi WAV di proyek.

`python -m pytest -q`: **53 passed**, termasuk:

- Tiga status klasifikasi dengan probabilitas terkontrol dan post-processing Silero asli.
- Speech 128 ms dengan probabilitas 0.4 tetap `speech` walaupun mean turun di bawah 0.01 karena ekor hening 10 detik; indikasi lemah 0.15 menjadi `uncertain`.
- Beep sinus 2 kHz, buzz gelombang kotak 100 Hz, dan noise Gaussian melewati RMS lalu model Silero asli dan menjadi `non-speech`.
- Beep berulang 1 kHz menghasilkan spike max sekitar 0.382, tanpa rentang speech yang memenuhi durasi. Hasilnya `uncertain`, sesuai prioritas recall. Tidak semua beep harus menjadi non-speech.
- Sampel radio lokal `ht_20261022_092400/rekaman_0003.wav` diberi noise Gaussian dengan SNR 0 dan -5 dB (terhadap RMS seluruh file), disegmentasi RMS, lalu diperiksa dengan Silero asli; hasil tidak menjadi `non-speech`. Kedua tes ini otomatis dilewati jika sampel lokal tidak tersedia.
- Kegagalan pemuatan model/inferensi, file kosong/hilang, probabilitas invalid, reset state, window parsial, resampling, dan stereo berlawanan fase.
- Kompatibilitas CLI V1, CSV lama, ringkasan terminal, serta byte WAV/waktu segmen RMS yang identik untuk speech, non-speech, uncertain, dan kegagalan VAD.
- Model tersedia secara lokal dan tidak membutuhkan torch.hub atau koneksi jaringan saat inferensi.

Ada peringatan deprecation dari paket upstream Silero/PyTorch tentang API resource dan TorchScript; tidak ada kegagalan tes.

Jumlah di atas adalah output model, **bukan pengukuran akurasi atau recall lapangan**. Belum tersedia label manusia untuk 55 segmen ini. Uji penambahan noise pada satu sampel juga tidak mewakili seluruh kondisi radio. Sebelum memakai status untuk keputusan lanjutan, dengarkan semua hasil `non-speech` dan contoh noisy/pendek, beri label manual, lalu bandingkan konfigurasi seperti panduan di [README.md](README.md). Seluruh WAV tetap tersedia untuk evaluasi ulang.
