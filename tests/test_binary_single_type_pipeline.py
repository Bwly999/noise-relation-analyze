from __future__ import annotations

import csv
import json
from pathlib import Path

from noise_relation_analyze.pipeline import (
    run_build_noise_report,
    run_evaluate_scores,
    run_extract_features,
    run_score_noise_types,
    run_train_noise_scorer,
)
from noise_relation_analyze.synthetic_data import generate_synthetic_dataset


def test_single_type_binary_pipeline_reaches_high_accuracy(tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(
        output_dir=tmp_path / "binary_synth",
        phone_count=24,
        labeled_fraction=1.0,
        seed=37,
        noise_types=["type_1", "normal"],
    )
    phone_features_csv = tmp_path / "phone_features.csv"
    model_json = tmp_path / "binary_model.json"
    scores_csv = tmp_path / "scores.csv"
    metrics_json = tmp_path / "metrics.json"

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
    metrics = run_evaluate_scores(
        scores_csv=scores_csv,
        labels_csv=manifest.labels_csv,
        output_json=metrics_json,
    )

    with scores_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert "score_type_1" in rows[0]
    assert "score_normal" in rows[0]
    assert metrics["overall_accuracy"] >= 0.9


def test_single_type_report_contains_shap_style_outputs(tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(
        output_dir=tmp_path / "binary_synth",
        phone_count=28,
        labeled_fraction=1.0,
        seed=41,
        noise_types=["type_1", "normal"],
    )
    phone_features_csv = tmp_path / "phone_features.csv"
    model_json = tmp_path / "binary_model.json"
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
    assert saved_report["noise_type"] == "type_1"
    assert "analysis_method" in report
    assert "top_factors" in report
    assert "shap_summary" in report
    assert "sample_explanations" in report
    assert report["top_factors"][0]["factor_key"] in {"hinge_gap", "left_support_gap"}
    assert report["shap_summary"][0]["factor_key"] in {"hinge_gap", "left_support_gap"}
