from __future__ import annotations

import math
from pathlib import Path
import wave

from noise_relation_analyze.audio import (
    compute_windowed_rms,
    load_wav_samples,
    segment_cycles_from_energy,
)


def test_load_wav_samples_reads_pcm16_and_normalizes(tmp_path: Path) -> None:
    wav_path = tmp_path / "demo.wav"
    _write_pcm16_wave(wav_path, [0, 16384, -16384, 0], sample_rate=8000)

    waveform = load_wav_samples(wav_path)

    assert waveform.sample_rate == 8000
    assert len(waveform.samples) == 4
    assert max(waveform.samples) > 0.49
    assert min(waveform.samples) < -0.49


def test_compute_windowed_rms_supports_cycle_segmentation(tmp_path: Path) -> None:
    wav_path = tmp_path / "cycles.wav"
    samples = (
        [0] * 400
        + [22000] * 400
        + [0] * 400
        + [22000] * 400
        + [0] * 400
    )
    _write_pcm16_wave(wav_path, samples, sample_rate=8000)

    waveform = load_wav_samples(wav_path)
    energy = compute_windowed_rms(waveform.samples, window_size=200, hop_size=100)
    segments = segment_cycles_from_energy(energy, threshold=0.3, min_length=2)

    assert len(segments) == 2
    assert all(end > start for start, end in segments)


def _write_pcm16_wave(path: Path, samples: list[int], sample_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = b"".join(int(sample).to_bytes(2, signed=True, byteorder="little") for sample in samples)
        handle.writeframes(frames)
