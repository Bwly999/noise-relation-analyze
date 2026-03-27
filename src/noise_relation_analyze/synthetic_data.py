from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
import wave


@dataclass(frozen=True)
class SyntheticDatasetManifest:
    root_dir: Path
    phone_master_csv: Path
    audio_asset_csv: Path
    labels_csv: Path
    dimension_csv: Path
    audio_dir: Path


def generate_synthetic_dataset(
    output_dir: Path,
    phone_count: int = 24,
    labeled_fraction: float = 0.5,
    seed: int = 42,
    noise_types: list[str] | None = None,
) -> SyntheticDatasetManifest:
    if phone_count <= 0:
        raise ValueError("phone_count must be positive")
    if not 0.0 <= labeled_fraction <= 1.0:
        raise ValueError("labeled_fraction must be between 0 and 1")

    rng = random.Random(seed)
    chosen_noise_types = noise_types or ["type_1", "type_2", "type_3", "normal"]
    if not chosen_noise_types:
        raise ValueError("noise_types must not be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    phone_master_csv = output_dir / "phone_master.csv"
    audio_asset_csv = output_dir / "audio_asset.csv"
    labels_csv = output_dir / "labels.csv"
    dimension_csv = output_dir / "dimensions.csv"

    phone_rows: list[dict[str, str]] = []
    audio_rows: list[dict[str, str]] = []
    label_rows: list[dict[str, str]] = []
    dimension_rows: list[dict[str, str]] = []

    label_count = int(phone_count * labeled_fraction)
    labeled_ids = {f"P{index + 1:04d}" for index in rng.sample(range(phone_count), label_count)}

    for index in range(phone_count):
        phone_id = f"P{index + 1:04d}"
        noise_type = _sample_noise_type(index, chosen_noise_types)
        metadata = _build_phone_metadata(index)
        dimensions = _build_dimensions(rng, noise_type)
        direction_profiles = _build_direction_profiles(noise_type, dimensions)

        phone_rows.append({"phone_id": phone_id, **metadata})
        dimension_rows.append({"phone_id": phone_id, **_stringify_numeric_map(dimensions)})

        if phone_id in labeled_ids:
            label_rows.append(
                {
                    "phone_id": phone_id,
                    "noise_type_label": noise_type,
                    "label_source": "synthetic_rule",
                }
            )

        for direction_id, profile in direction_profiles.items():
            wav_path = audio_dir / f"{phone_id}_{direction_id}.wav"
            _write_synthetic_wave(
                wav_path=wav_path,
                sample_rate=8000,
                base_amplitude=profile["base_amplitude"],
                burst_amplitude=profile["burst_amplitude"],
                pulse_interval=profile["pulse_interval"],
                pulse_width=profile["pulse_width"],
                cycles_per_file=4,
                rng=rng,
            )
            audio_rows.append(
                {
                    "phone_id": phone_id,
                    "direction_id": direction_id,
                    "wav_path": str(wav_path),
                }
            )

    _write_csv(
        phone_master_csv,
        [
            "phone_id",
            "model_id",
            "batch_id",
            "vendor_id",
            "structure_version",
            "material_version",
        ],
        phone_rows,
    )
    _write_csv(audio_asset_csv, ["phone_id", "direction_id", "wav_path"], audio_rows)
    _write_csv(labels_csv, ["phone_id", "noise_type_label", "label_source"], label_rows)
    _write_csv(
        dimension_csv,
        [
            "phone_id",
            "hinge_gap",
            "left_support_gap",
            "right_support_gap",
            "torsion_delta",
            "panel_flushness",
            "adhesive_thickness",
        ],
        dimension_rows,
    )

    return SyntheticDatasetManifest(
        root_dir=output_dir,
        phone_master_csv=phone_master_csv,
        audio_asset_csv=audio_asset_csv,
        labels_csv=labels_csv,
        dimension_csv=dimension_csv,
        audio_dir=audio_dir,
    )


def _sample_noise_type(index: int, chosen_noise_types: list[str]) -> str:
    return chosen_noise_types[index % len(chosen_noise_types)]


def _build_phone_metadata(index: int) -> dict[str, str]:
    return {
        "model_id": f"M{(index % 2) + 1}",
        "batch_id": f"B{(index % 3) + 1}",
        "vendor_id": f"V{(index % 2) + 1}",
        "structure_version": f"S{(index % 2) + 1}",
        "material_version": f"MAT{(index % 3) + 1}",
    }


def _build_dimensions(rng: random.Random, noise_type: str) -> dict[str, float]:
    dimensions = {
        "hinge_gap": 0.10 + rng.uniform(-0.01, 0.01),
        "left_support_gap": 0.08 + rng.uniform(-0.01, 0.01),
        "right_support_gap": 0.08 + rng.uniform(-0.01, 0.01),
        "torsion_delta": 0.03 + rng.uniform(-0.005, 0.005),
        "panel_flushness": 0.02 + rng.uniform(-0.004, 0.004),
        "adhesive_thickness": 0.15 + rng.uniform(-0.01, 0.01),
    }

    if noise_type == "type_1":
        dimensions["hinge_gap"] += rng.uniform(0.08, 0.12)
        dimensions["left_support_gap"] += rng.uniform(0.05, 0.08)
    elif noise_type == "type_2":
        dimensions["torsion_delta"] += rng.uniform(0.07, 0.10)
        dimensions["right_support_gap"] += rng.uniform(0.05, 0.08)
    elif noise_type == "type_3":
        dimensions["panel_flushness"] += rng.uniform(0.06, 0.09)
        dimensions["adhesive_thickness"] -= rng.uniform(0.03, 0.05)
    return dimensions


def _build_direction_profiles(
    noise_type: str, dimensions: dict[str, float]
) -> dict[str, dict[str, float]]:
    type_1_score = max(
        0.0,
        (dimensions["hinge_gap"] - 0.10) * 2.6 + (dimensions["left_support_gap"] - 0.08) * 1.8,
    )
    type_2_score = max(
        0.0,
        (dimensions["torsion_delta"] - 0.03) * 3.5
        + (dimensions["right_support_gap"] - 0.08) * 1.9,
    )
    type_3_score = max(
        0.0,
        (dimensions["panel_flushness"] - 0.02) * 3.2
        + (0.15 - dimensions["adhesive_thickness"]) * 1.8,
    )

    base = {
        "dir_1": {"base_amplitude": 0.025, "burst_amplitude": 0.08, "pulse_interval": 44, "pulse_width": 5},
        "dir_2": {"base_amplitude": 0.025, "burst_amplitude": 0.08, "pulse_interval": 44, "pulse_width": 5},
        "dir_3": {"base_amplitude": 0.025, "burst_amplitude": 0.08, "pulse_interval": 44, "pulse_width": 5},
        "dir_4": {"base_amplitude": 0.025, "burst_amplitude": 0.08, "pulse_interval": 44, "pulse_width": 5},
    }

    if noise_type == "type_1":
        base["dir_1"]["burst_amplitude"] += type_1_score
        base["dir_2"]["burst_amplitude"] += type_1_score * 0.35
        base["dir_1"]["pulse_interval"] = 32
        base["dir_1"]["pulse_width"] = 3
    elif noise_type == "type_2":
        base["dir_3"]["burst_amplitude"] += type_2_score
        base["dir_4"]["burst_amplitude"] += type_2_score * 0.30
        base["dir_3"]["pulse_interval"] = 18
        base["dir_3"]["pulse_width"] = 2
    elif noise_type == "type_3":
        base["dir_2"]["burst_amplitude"] += type_3_score
        base["dir_1"]["burst_amplitude"] += type_3_score * 0.25
        base["dir_2"]["pulse_interval"] = 60
        base["dir_2"]["pulse_width"] = 8

    return base


def _write_synthetic_wave(
    wav_path: Path,
    sample_rate: int,
    base_amplitude: float,
    burst_amplitude: float,
    pulse_interval: int,
    pulse_width: int,
    cycles_per_file: int,
    rng: random.Random,
) -> None:
    samples: list[int] = []
    cycle_length = 1800
    active_start = 500
    active_end = 1300

    for cycle_index in range(cycles_per_file):
        phase_shift = rng.uniform(0.0, math.pi / 6)
        for sample_index in range(cycle_length):
            if active_start <= sample_index < active_end:
                within_active = sample_index - active_start
                carrier = math.sin((within_active / 24.0) + phase_shift)
                pulse = 1.0 if (within_active % pulse_interval) < pulse_width else 0.0
                amplitude = base_amplitude + (burst_amplitude * 0.55) + (burst_amplitude * 0.45 * pulse)
                noise = rng.uniform(-0.015, 0.015)
                value = amplitude * carrier + noise
            else:
                value = rng.uniform(-0.008, 0.008)

            samples.append(max(-32767, min(32767, int(value * 32767))))

    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = b"".join(sample.to_bytes(2, signed=True, byteorder="little") for sample in samples)
        handle.writeframes(frames)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stringify_numeric_map(values: dict[str, float]) -> dict[str, str]:
    return {key: f"{value:.6f}" for key, value in values.items()}
