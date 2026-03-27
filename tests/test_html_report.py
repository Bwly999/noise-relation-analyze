from __future__ import annotations

from pathlib import Path

from noise_relation_analyze.pipeline import (
    run_build_noise_report,
    run_extract_features,
    run_render_noise_report_html,
    run_score_noise_types,
    run_train_noise_scorer,
)
from noise_relation_analyze.synthetic_data import generate_synthetic_dataset


def test_render_noise_report_html_creates_visual_artifacts(tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(
        output_dir=tmp_path / "binary_synth",
        phone_count=28,
        labeled_fraction=1.0,
        seed=53,
        noise_types=["type_1", "normal"],
    )
    phone_features_csv = tmp_path / "phone_features.csv"
    model_path = tmp_path / "noise_model.bin"
    scores_csv = tmp_path / "scores.csv"
    report_json = tmp_path / "type1_report.json"
    report_dir = tmp_path / "html_report"

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
    run_build_noise_report(
        scores_csv=scores_csv,
        dimension_csv=manifest.dimension_csv,
        labels_csv=manifest.labels_csv,
        output_json=report_json,
        noise_type="type_1",
        factor_keys=["hinge_gap", "left_support_gap", "right_support_gap", "torsion_delta"],
        model_path=model_path,
        phone_features_csv=phone_features_csv,
    )

    html_path = run_render_noise_report_html(
        report_json=report_json,
        scores_csv=scores_csv,
        phone_features_csv=phone_features_csv,
        model_path=model_path,
        output_dir=report_dir,
        highlight_feature="dir_1_share",
        highlight_factor="hinge_gap",
    )

    html = html_path.read_text(encoding="utf-8")
    assert html_path.exists()
    assert (report_dir / "assets" / "shap_bar.png").exists()
    assert (report_dir / "assets" / "shap_beeswarm.png").exists()
    assert (report_dir / "assets" / "shap_dependence_dir_1_share.png").exists()
    assert (report_dir / "assets" / "factor_ranking.png").exists()
    assert (report_dir / "assets" / "factor_trend_hinge_gap.png").exists()
    assert "Noise Analysis Report" in html
    assert "shap_bar.png" in html
