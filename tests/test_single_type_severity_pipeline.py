from __future__ import annotations

import csv
import json
from pathlib import Path

from noise_relation_analyze.pipeline import (
    run_build_single_type_dataset,
    run_build_single_type_report,
    run_extract_features,
    run_render_single_type_report_html,
    run_score_single_type_models,
    run_train_single_type_models,
)
from noise_relation_analyze.synthetic_data import generate_single_type_severity_dataset


FACTOR_KEYS = [
    "hinge_gap",
    "left_support_gap",
    "right_support_gap",
    "torsion_delta",
    "panel_flushness",
    "adhesive_thickness",
]


def test_generate_single_type_severity_dataset_has_continuous_severity_and_ng_labels(
    tmp_path: Path,
) -> None:
    manifest = generate_single_type_severity_dataset(
        output_dir=tmp_path / "single_type",
        phone_count=40,
        labeled_fraction=1.0,
        seed=19,
        noise_type="type_1",
    )

    with manifest.labels_csv.open("r", encoding="utf-8", newline="") as handle:
        label_rows = list(csv.DictReader(handle))
    with manifest.dimension_csv.open("r", encoding="utf-8", newline="") as handle:
        dimension_rows = list(csv.DictReader(handle))

    ng_rows = [row for row in label_rows if row["is_ng"] == "1"]
    ok_rows = [row for row in label_rows if row["is_ng"] == "0"]
    ng_severity = sum(float(row["true_severity"]) for row in ng_rows) / len(ng_rows)
    ok_severity = sum(float(row["true_severity"]) for row in ok_rows) / len(ok_rows)
    ng_ids = {row["phone_id"] for row in ng_rows}
    ng_hinge_gap = sum(
        float(row["hinge_gap"]) for row in dimension_rows if row["phone_id"] in ng_ids
    ) / len(ng_rows)
    ok_hinge_gap = sum(
        float(row["hinge_gap"]) for row in dimension_rows if row["phone_id"] not in ng_ids
    ) / len(ok_rows)

    assert label_rows
    assert all(row["noise_type_label"] == "type_1" for row in label_rows)
    assert ng_rows
    assert ok_rows
    assert ng_severity > ok_severity
    assert ng_hinge_gap > ok_hinge_gap


def test_single_type_pipeline_learns_severity_and_ng_from_dimensions(tmp_path: Path) -> None:
    manifest = generate_single_type_severity_dataset(
        output_dir=tmp_path / "single_type",
        phone_count=56,
        labeled_fraction=1.0,
        seed=23,
        noise_type="type_1",
    )
    phone_features_csv = tmp_path / "phone_features.csv"
    dataset_csv = tmp_path / "single_type_dataset.csv"
    model_path = tmp_path / "single_type_models.bin"
    scores_csv = tmp_path / "single_type_scores.csv"
    report_json = tmp_path / "single_type_report.json"

    run_extract_features(audio_asset_csv=manifest.audio_asset_csv, output_csv=phone_features_csv)
    run_build_single_type_dataset(
        phone_features_csv=phone_features_csv,
        dimension_csv=manifest.dimension_csv,
        labels_csv=manifest.labels_csv,
        output_csv=dataset_csv,
        noise_type="type_1",
    )
    metrics = run_train_single_type_models(
        input_csv=dataset_csv,
        output_model=model_path,
        noise_type="type_1",
        factor_keys=FACTOR_KEYS,
    )
    run_score_single_type_models(
        input_csv=dataset_csv,
        model_path=model_path,
        output_csv=scores_csv,
    )
    report = run_build_single_type_report(
        input_csv=dataset_csv,
        scores_csv=scores_csv,
        model_path=model_path,
        output_json=report_json,
    )

    with scores_csv.open("r", encoding="utf-8", newline="") as handle:
        score_rows = list(csv.DictReader(handle))
    saved_report = json.loads(report_json.read_text(encoding="utf-8"))

    assert score_rows
    assert "severity_score_pred" in score_rows[0]
    assert "ng_risk" in score_rows[0]
    assert metrics["severity_cv_r2"] >= 0.75
    assert metrics["severity_cv_spearman"] >= 0.85
    assert metrics["ng_cv_auc"] >= 0.90
    assert report["severity_top_factors"][0]["factor_key"] in {"hinge_gap", "left_support_gap"}
    assert report["ng_top_factors"][0]["factor_key"] in {"hinge_gap", "left_support_gap"}
    assert saved_report["analysis_method"] == "single_type_severity_plus_xgboost_tree_shap"
    assert saved_report["acoustic_quantification"]["severity_ng_auc"] >= 0.85


def test_render_single_type_report_html_creates_shap_assets(tmp_path: Path) -> None:
    manifest = generate_single_type_severity_dataset(
        output_dir=tmp_path / "single_type",
        phone_count=48,
        labeled_fraction=1.0,
        seed=29,
        noise_type="type_1",
    )
    phone_features_csv = tmp_path / "phone_features.csv"
    dataset_csv = tmp_path / "single_type_dataset.csv"
    model_path = tmp_path / "single_type_models.bin"
    scores_csv = tmp_path / "single_type_scores.csv"
    report_json = tmp_path / "single_type_report.json"
    report_dir = tmp_path / "html_report"

    run_extract_features(audio_asset_csv=manifest.audio_asset_csv, output_csv=phone_features_csv)
    run_build_single_type_dataset(
        phone_features_csv=phone_features_csv,
        dimension_csv=manifest.dimension_csv,
        labels_csv=manifest.labels_csv,
        output_csv=dataset_csv,
        noise_type="type_1",
    )
    run_train_single_type_models(
        input_csv=dataset_csv,
        output_model=model_path,
        noise_type="type_1",
        factor_keys=FACTOR_KEYS,
    )
    run_score_single_type_models(
        input_csv=dataset_csv,
        model_path=model_path,
        output_csv=scores_csv,
    )
    run_build_single_type_report(
        input_csv=dataset_csv,
        scores_csv=scores_csv,
        model_path=model_path,
        output_json=report_json,
    )

    html_path = run_render_single_type_report_html(
        report_json=report_json,
        input_csv=dataset_csv,
        model_path=model_path,
        output_dir=report_dir,
        highlight_factor="hinge_gap",
    )

    html = html_path.read_text(encoding="utf-8")
    assert html_path.exists()
    assert (report_dir / "assets" / "severity_shap_bar.png").exists()
    assert (report_dir / "assets" / "severity_shap_beeswarm.png").exists()
    assert (report_dir / "assets" / "severity_shap_dependence_hinge_gap.png").exists()
    assert (report_dir / "assets" / "ng_shap_bar.png").exists()
    assert (report_dir / "assets" / "ng_shap_beeswarm.png").exists()
    assert (report_dir / "assets" / "ng_shap_dependence_hinge_gap.png").exists()
    assert "Single-Type Severity Report" in html
    assert "severity_shap_bar.png" in html
