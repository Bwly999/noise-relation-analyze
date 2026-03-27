from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
import numpy as np

from noise_relation_analyze.audio import compute_windowed_rms, segment_cycles_from_energy, Waveform


@dataclass(frozen=True)
class CycleFeature:
    phone_id: str
    direction_id: str
    cycle_id: str
    rms: float
    impulse_score: float
    crest_factor: float = 0.0
    zero_crossing_rate: float = 0.0
    spectral_centroid: float = 0.0
    high_band_ratio: float = 0.0


@dataclass(frozen=True)
class PhoneFeatureSummary:
    phone_id: str
    primary_direction: str
    direction_count: int
    asymmetry_score: float
    severity_index: float
    rms_mean: float
    impulse_score_mean: float
    crest_factor_mean: float
    zero_crossing_rate_mean: float
    spectral_centroid_mean: float
    high_band_ratio_mean: float
    dir_1_score: float
    dir_2_score: float
    dir_3_score: float
    dir_4_score: float
    dir_1_share: float
    dir_2_share: float
    dir_3_share: float
    dir_4_share: float


def extract_cycle_features(
    phone_id: str,
    direction_id: str,
    waveform: Waveform,
    window_size: int = 200,
    hop_size: int = 100,
    energy_threshold: float = 0.1,
    min_segment_length: int = 2,
) -> list[CycleFeature]:
    energy = compute_windowed_rms(
        waveform.samples,
        window_size=window_size,
        hop_size=hop_size,
    )
    segments = segment_cycles_from_energy(
        energy,
        threshold=energy_threshold,
        min_length=min_segment_length,
    )
    if not segments and energy:
        segments = [(0, len(energy))]

    cycle_features: list[CycleFeature] = []
    for cycle_index, (start, end) in enumerate(segments, start=1):
        segment_energy = energy[start:end]
        sample_start = start * hop_size
        sample_end = min(len(waveform.samples), max(sample_start + 1, ((end - 1) * hop_size) + window_size))
        segment_samples = waveform.samples[sample_start:sample_end]
        rms = max(segment_energy) if segment_energy else 0.0
        impulse = (max(segment_energy) - fmean(segment_energy)) if segment_energy else 0.0
        crest = _crest_factor(segment_samples)
        zcr = _zero_crossing_rate(segment_samples)
        centroid = _spectral_centroid(segment_samples, waveform.sample_rate)
        high_band_ratio = _high_band_ratio(segment_samples)
        cycle_features.append(
            CycleFeature(
                phone_id=phone_id,
                direction_id=direction_id,
                cycle_id=f"{direction_id}_c{cycle_index}",
                rms=rms,
                impulse_score=max(impulse, 0.0),
                crest_factor=crest,
                zero_crossing_rate=zcr,
                spectral_centroid=centroid,
                high_band_ratio=high_band_ratio,
            )
        )

    return cycle_features


def aggregate_phone_features(
    phone_id: str, cycle_features: list[CycleFeature]
) -> PhoneFeatureSummary:
    grouped: dict[str, list[CycleFeature]] = {}
    for feature in cycle_features:
        if feature.phone_id != phone_id:
            continue
        grouped.setdefault(feature.direction_id, []).append(feature)

    if not grouped:
        raise ValueError(f"no cycle features for phone_id={phone_id}")

    direction_scores = {
        direction: fmean(
            ((feature.rms * 0.7) + (feature.impulse_score * 0.3))
            for feature in direction_features
        )
        for direction, direction_features in grouped.items()
    }
    primary_direction = max(direction_scores, key=direction_scores.get)
    max_score = max(direction_scores.values())
    min_score = min(direction_scores.values())
    asymmetry = max_score - min_score if len(direction_scores) > 1 else max_score
    severity = fmean(direction_scores.values())
    total_score = sum(direction_scores.values()) or 1.0
    phone_cycles = [feature for feature in cycle_features if feature.phone_id == phone_id]

    return PhoneFeatureSummary(
        phone_id=phone_id,
        primary_direction=primary_direction,
        direction_count=len(direction_scores),
        asymmetry_score=asymmetry,
        severity_index=severity,
        rms_mean=fmean(feature.rms for feature in phone_cycles),
        impulse_score_mean=fmean(feature.impulse_score for feature in phone_cycles),
        crest_factor_mean=fmean(feature.crest_factor for feature in phone_cycles),
        zero_crossing_rate_mean=fmean(feature.zero_crossing_rate for feature in phone_cycles),
        spectral_centroid_mean=fmean(feature.spectral_centroid for feature in phone_cycles),
        high_band_ratio_mean=fmean(feature.high_band_ratio for feature in phone_cycles),
        dir_1_score=direction_scores.get("dir_1", 0.0),
        dir_2_score=direction_scores.get("dir_2", 0.0),
        dir_3_score=direction_scores.get("dir_3", 0.0),
        dir_4_score=direction_scores.get("dir_4", 0.0),
        dir_1_share=direction_scores.get("dir_1", 0.0) / total_score,
        dir_2_share=direction_scores.get("dir_2", 0.0) / total_score,
        dir_3_share=direction_scores.get("dir_3", 0.0) / total_score,
        dir_4_share=direction_scores.get("dir_4", 0.0) / total_score,
    )


def _crest_factor(samples: list[float]) -> float:
    if not samples:
        return 0.0
    values = np.asarray(samples, dtype=float)
    rms = float(np.sqrt(np.mean(values**2)))
    if rms <= 1e-12:
        return 0.0
    return float(np.max(np.abs(values)) / rms)


def _zero_crossing_rate(samples: list[float]) -> float:
    if len(samples) < 2:
        return 0.0
    values = np.asarray(samples, dtype=float)
    signs = np.signbit(values)
    crossings = np.count_nonzero(signs[1:] != signs[:-1])
    return float(crossings / (len(values) - 1))


def _spectral_centroid(samples: list[float], sample_rate: int) -> float:
    if len(samples) < 4:
        return 0.0
    values = np.asarray(samples, dtype=float)
    windowed = values * np.hanning(len(values))
    magnitude = np.abs(np.fft.rfft(windowed))
    if not np.any(magnitude):
        return 0.0
    frequencies = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate)
    return float(np.sum(frequencies * magnitude) / np.sum(magnitude))


def _high_band_ratio(samples: list[float]) -> float:
    if len(samples) < 4:
        return 0.0
    values = np.asarray(samples, dtype=float)
    magnitude = np.abs(np.fft.rfft(values * np.hanning(len(values))))
    if magnitude.size == 0 or not np.any(magnitude):
        return 0.0
    split_index = max(1, magnitude.size // 2)
    return float(np.sum(magnitude[split_index:]) / np.sum(magnitude))
