from __future__ import annotations

import json
from pathlib import Path

from noise_relation_analyze.pipeline import run_demo_pipeline


def test_run_demo_pipeline_produces_full_artifact_set(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo_run"

    run_demo_pipeline(output_dir=output_dir, phone_count=20, labeled_fraction=1.0, seed=31)

    metrics_json = output_dir / "artifacts" / "metrics.json"
    scores_csv = output_dir / "artifacts" / "scores.csv"
    model_json = output_dir / "artifacts" / "noise_model.bin"
    report_dir = output_dir / "artifacts" / "reports"

    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))

    assert model_json.exists()
    assert scores_csv.exists()
    assert metrics_json.exists()
    assert (report_dir / "type_1_report.json").exists()
    assert (report_dir / "type_2_report.json").exists()
    assert (report_dir / "type_3_report.json").exists()
    assert metrics["overall_accuracy"] >= 0.80
