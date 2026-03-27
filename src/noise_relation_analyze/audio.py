from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import wave
from typing import Iterable


@dataclass(frozen=True)
class Waveform:
    sample_rate: int
    samples: list[float]


def load_wav_samples(path: Path) -> Waveform:
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is supported in V1")
        if handle.getnchannels() != 1:
            raise ValueError("only mono WAV is supported in V1")

        frames = handle.readframes(handle.getnframes())
        samples = [
            int.from_bytes(frames[index : index + 2], byteorder="little", signed=True) / 32768.0
            for index in range(0, len(frames), 2)
        ]
        return Waveform(sample_rate=handle.getframerate(), samples=samples)


def compute_windowed_rms(
    samples: Iterable[float], window_size: int, hop_size: int
) -> list[float]:
    values = list(samples)
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("window_size and hop_size must be positive")
    if len(values) < window_size:
        return [_rms(values)] if values else []

    energies: list[float] = []
    for start in range(0, len(values) - window_size + 1, hop_size):
        window = values[start : start + window_size]
        energies.append(_rms(window))
    return energies


def segment_cycles_from_energy(
    energy: Iterable[float], threshold: float, min_length: int
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    values = list(energy)

    for index, value in enumerate(values):
        if value >= threshold and start is None:
            start = index
        elif value < threshold and start is not None:
            if index - start >= min_length:
                segments.append((start, index))
            start = None

    if start is not None and len(values) - start >= min_length:
        segments.append((start, len(values)))

    return segments


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return sqrt(sum(value * value for value in values) / len(values))
