from __future__ import annotations

import json
from pathlib import Path

from noise_relation_analyze.pipeline import (
    run_build_noise_report,
    run_extract_features,
    run_score_noise_types,
    run_train_noise_scorer,
)
from noise_relation_analyze.synthetic_data import generate_synthetic_dataset


def test_build_noise_report_highlights_expected_type1_factors(tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(
        output_dir=tmp_path / "synthetic",
        phone_count=32,
        labeled_fraction=1.0,
        seed=29,
    )
    phone_features_csv = tmp_path / "phone_features.csv"
    model_json = tmp_path / "noise_model.json"
    scores_csv = tmp_path / "scores.csv"
    report_json = tmp_path / "type1_report.json"

    run_extract_features(audio_asset_csv=manifest.audio_asset_csv, output_csv=phone_features_csv)
    run_train_noise_scorer(
        phone_features_csv=phone_features_csv,
        labels_csv=manifest.labels_csv,
        output_json=model_json,
    )
    run_score_noise_types(
        phone_features_csv=phone_features_csv,
        model_json=model_json,
        output_csv=scores_csv,
    )
    report = run_build_noise_report(
        scores_csv=scores_csv,
        dimension_csv=manifest.dimension_csv,
        labels_csv=manifest.labels_csv,
        output_json=report_json,
        noise_type="type_1",
        factor_keys=["hinge_gap", "left_support_gap", "right_support_gap", "torsion_delta"],
    )

    saved_report = json.loads(report_json.read_text(encoding="utf-8"))
    top_factors = [item["factor_key"] for item in report["top_factors"][:2]]

    assert report_json.exists()
    assert saved_report["noise_type"] == "type_1"
    assert top_factors == ["hinge_gap", "left_support_gap"] or top_factors == [
        "left_support_gap",
        "hinge_gap",
    ]
