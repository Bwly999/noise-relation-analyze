from __future__ import annotations

import csv
from pathlib import Path

from noise_relation_analyze.pipeline import (
    run_analyze_factors,
    run_extract_features,
    run_prepare_factor_data,
    run_validate_joins,
)
from noise_relation_analyze.synthetic_data import generate_synthetic_dataset


def test_synthetic_dataset_runs_through_current_pipeline(tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(
        output_dir=tmp_path / "synthetic",
        phone_count=18,
        labeled_fraction=1.0,
        seed=19,
    )
    join_report = tmp_path / "join_report.json"
    phone_features = tmp_path / "phone_features.csv"
    factor_input = tmp_path / "factor_input.csv"
    factor_output = tmp_path / "factor_output.csv"

    validation = run_validate_joins(
        phone_master_csv=manifest.phone_master_csv,
        audio_asset_csv=manifest.audio_asset_csv,
        labels_csv=manifest.labels_csv,
        output_json=join_report,
    )
    run_extract_features(audio_asset_csv=manifest.audio_asset_csv, output_csv=phone_features)
    run_prepare_factor_data(
        phone_features_csv=phone_features,
        dimension_csv=manifest.dimension_csv,
        labels_csv=manifest.labels_csv,
        output_csv=factor_input,
        noise_type="type_1",
    )
    run_analyze_factors(
        input_csv=factor_input,
        output_csv=factor_output,
        target_key="risk_score",
        factor_keys=["hinge_gap", "left_support_gap", "right_support_gap", "torsion_delta"],
    )

    with factor_output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert validation.valid_phone_count == 18
    assert rows
    assert rows[0]["factor_key"] in {"hinge_gap", "left_support_gap"}
