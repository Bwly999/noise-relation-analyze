from __future__ import annotations

import csv
from pathlib import Path

from noise_relation_analyze.pipeline import run_extract_features, run_prepare_factor_data
from noise_relation_analyze.synthetic_data import generate_synthetic_dataset


def test_prepare_factor_data_merges_dimensions_labels_and_features(tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(
        output_dir=tmp_path / "synthetic",
        phone_count=8,
        labeled_fraction=1.0,
        seed=11,
    )
    phone_features_csv = tmp_path / "phone_features.csv"
    factor_input_csv = tmp_path / "factor_input.csv"

    run_extract_features(audio_asset_csv=manifest.audio_asset_csv, output_csv=phone_features_csv)
    run_prepare_factor_data(
        phone_features_csv=phone_features_csv,
        dimension_csv=manifest.dimension_csv,
        labels_csv=manifest.labels_csv,
        output_csv=factor_input_csv,
        noise_type="type_1",
    )

    with factor_input_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 8
    assert "risk_score" in rows[0]
    assert "is_type_1" in rows[0]
    assert "hinge_gap" in rows[0]
    assert "left_support_gap" in rows[0]
