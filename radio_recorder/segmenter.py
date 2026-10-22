"""Segmentasi streaming dengan pre-buffer di RAM dan silence timeout.

Sumber audio bisa file atau (ke depan) live USB sound card.
Tidak memuat seluruh rekaman ke RAM.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .detector import is_active


@dataclass(frozen=True)
class Config:
    threshold_db: float = -50.0
    prebuffer_seconds: float = 1.0
    silence_seconds: float = 10.0


@dataclass(frozen=True)
class Segment:
    number: int
    filename: str
    start_sample: int
    end_sample: int
    samplerate: int

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.samplerate

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.samplerate

    @property
    def duration_seconds(self) -> float:
        return (self.end_sample - self.start_sample) / self.samplerate


class SoundTriggeredRecorder:
    """Menerima audio per frame; hanya menyimpan segmen yang terpicu suara.

    Implementasi ini berfungsi untuk sumber file maupun live streaming.
    Klien wajib memanggil finish() setelah semua frame selesai.
    """

    def __init__(self, samplerate: int, channels: int, output_dir: Path, config: Config):
        if samplerate <= 0 or channels <= 0:
            raise ValueError("samplerate dan channels harus positif")
        if not -120 < config.threshold_db <= 0:
            raise ValueError("threshold_db harus di antara -120 (eksklusif) dan 0")
        if config.prebuffer_seconds < 0 or config.silence_seconds <= 0:
            raise ValueError("prebuffer_seconds >= 0 dan silence_seconds > 0")
        self.samplerate = samplerate
        self.channels = channels
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.prebuffer_size = round(config.prebuffer_seconds * samplerate)
        self.silence_limit = round(config.silence_seconds * samplerate)
        self.buffer: deque[tuple[int, np.ndarray]] = deque()
        self.buffer_frames = 0
        self.cursor = 0
        self.silence_frames = 0
        self.writer: sf.SoundFile | None = None
        self.current_start: int | None = None
        self.current_filename: str | None = None
        self.segments: list[Segment] = []
        self.levels: list[tuple[float, float]] = []

    def _push_buffer(self, start_sample: int, audio: np.ndarray) -> None:
        if self.prebuffer_size == 0:
            return
        self.buffer.append((start_sample, audio.copy()))
        self.buffer_frames += len(audio)
        while self.buffer_frames > self.prebuffer_size:
            old_start, old_audio = self.buffer.popleft()
            excess = self.buffer_frames - self.prebuffer_size
            if len(old_audio) > excess:
                self.buffer.appendleft((old_start + excess, old_audio[excess:]))
                self.buffer_frames -= excess
            else:
                self.buffer_frames -= len(old_audio)

    def _start(self, now_sample: int, audio: np.ndarray) -> None:
        index = len(self.segments) + 1
        # Saat aktif, tidak ada segment sebelumnya yang belum ditutup.
        filename = f"rekaman_{index:04d}.wav"
        self.current_filename = filename
        self.current_start = self.buffer[0][0] if self.buffer else now_sample
        self.writer = sf.SoundFile(
            self.output_dir / filename,
            mode="w",
            samplerate=self.samplerate,
            channels=self.channels,
            format="WAV",
            subtype="PCM_16",
        )
        for _, buffered_audio in self.buffer:
            self.writer.write(buffered_audio)
        self.buffer.clear()
        self.buffer_frames = 0
        self.writer.write(audio)
        self.silence_frames = 0

    def _close(self) -> None:
        if self.writer is None:
            return
        self.writer.close()
        assert self.current_start is not None and self.current_filename is not None
        self.segments.append(
            Segment(
                number=len(self.segments) + 1,
                filename=self.current_filename,
                start_sample=self.current_start,
                end_sample=self.cursor,
                samplerate=self.samplerate,
            )
        )
        self.writer = None
        self.current_start = None
        self.current_filename = None
        self.silence_frames = 0

    def process(self, audio: np.ndarray) -> None:
        if audio.ndim != 2 or audio.shape[1] != self.channels:
            raise ValueError("Audio harus berbentuk (frame, channels) sesuai perangkat")
        if len(audio) == 0:
            return
        start = self.cursor
        active, level_db = is_active(audio, self.config.threshold_db)
        self.levels.append(((start + len(audio) / 2) / self.samplerate, level_db))
        self.cursor += len(audio)

        if self.writer is None:
            if active:
                self._start(start, audio)
            else:
                self._push_buffer(start, audio)
            return

        self.writer.write(audio)
        if active:
            self.silence_frames = 0
        else:
            self.silence_frames += len(audio)
            if self.silence_frames >= self.silence_limit:
                self._close()

    def finish(self) -> list[Segment]:
        """Tutup rekaman terakhir, termasuk bila file berakhir saat transmisi."""
        self._close()
        return self.segments
