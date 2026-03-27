from __future__ import annotations

import csv
from pathlib import Path
import wave

from noise_relation_analyze.pipeline import run_extract_features, run_train_noise_scorer
from noise_relation_analyze.synthetic_data import generate_synthetic_dataset


def test_extract_features_outputs_richer_signal_descriptors(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    wav_a = audio_dir / "dir_1.wav"
    _write_pcm16_wave(
        wav_a,
        ([0] * 200 + [20000, -18000] * 160 + [0] * 200) * 2,
        sample_rate=8000,
    )
    asset_csv = tmp_path / "audio_asset.csv"
    output_csv = tmp_path / "phone_features.csv"
    asset_csv.write_text(
        "\n".join(
            [
                "phone_id,direction_id,wav_path",
                f"P001,dir_1,{wav_a}",
            ]
        ),
        encoding="utf-8",
    )

    run_extract_features(audio_asset_csv=asset_csv, output_csv=output_csv)

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert "rms_mean" in row
    assert "crest_factor_mean" in row
    assert "zero_crossing_rate_mean" in row
    assert "spectral_centroid_mean" in row
    assert float(row["crest_factor_mean"]) > 1.0


def test_train_noise_scorer_returns_cross_validated_metrics(tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(
        output_dir=tmp_path / "binary_synth",
        phone_count=24,
        labeled_fraction=1.0,
        seed=43,
        noise_types=["type_1", "normal"],
    )
    phone_features_csv = tmp_path / "phone_features.csv"
    model_path = tmp_path / "noise_model.bin"

    run_extract_features(audio_asset_csv=manifest.audio_asset_csv, output_csv=phone_features_csv)
    training_summary = run_train_noise_scorer(
        phone_features_csv=phone_features_csv,
        labels_csv=manifest.labels_csv,
        output_json=model_path,
    )

    assert training_summary["model_type"] == "random_forest"
    assert training_summary["cv_accuracy"] >= 0.8
    assert training_summary["cv_macro_f1"] >= 0.8


def _write_pcm16_wave(path: Path, samples: list[int], sample_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = b"".join(int(sample).to_bytes(2, signed=True, byteorder="little") for sample in samples)
        handle.writeframes(frames)
