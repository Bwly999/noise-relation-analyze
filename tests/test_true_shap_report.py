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


def test_build_noise_report_uses_tree_shap_outputs(tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(
        output_dir=tmp_path / "binary_synth",
        phone_count=28,
        labeled_fraction=1.0,
        seed=47,
        noise_types=["type_1", "normal"],
    )
    phone_features_csv = tmp_path / "phone_features.csv"
    model_path = tmp_path / "noise_model.bin"
    scores_csv = tmp_path / "scores.csv"
    report_json = tmp_path / "type1_report.json"

    run_extract_features(audio_asset_csv=manifest.audio_asset_csv, output_csv=phone_features_csv)
    run_train_noise_scorer(
        phone_features_csv=phone_features_csv,
        labels_csv=manifest.labels_csv,
        output_json=model_path,
    )
    run_score_noise_types(
        phone_features_csv=phone_features_csv,
        model_json=model_path,
        output_csv=scores_csv,
    )
    report = run_build_noise_report(
        scores_csv=scores_csv,
        dimension_csv=manifest.dimension_csv,
        labels_csv=manifest.labels_csv,
        output_json=report_json,
        noise_type="type_1",
        factor_keys=["hinge_gap", "left_support_gap", "right_support_gap", "torsion_delta"],
        model_path=model_path,
        phone_features_csv=phone_features_csv,
    )

    saved_report = json.loads(report_json.read_text(encoding="utf-8"))
    assert saved_report["analysis_method"].startswith("tree_shap")
    assert saved_report["model_summary"]["model_type"] == "random_forest"
    assert saved_report["model_summary"]["cv_accuracy"] >= 0.8
    assert saved_report["feature_shap_summary"][0]["feature_key"] in {"dir_1_share", "dir_1_score", "severity_index"}
