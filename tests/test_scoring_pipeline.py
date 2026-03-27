from __future__ import annotations

import csv
import json
from pathlib import Path

from noise_relation_analyze.pipeline import (
    run_evaluate_scores,
    run_extract_features,
    run_score_noise_types,
    run_train_noise_scorer,
)
from noise_relation_analyze.synthetic_data import generate_synthetic_dataset


def test_train_and_score_noise_types_on_synthetic_data(tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(
        output_dir=tmp_path / "synthetic",
        phone_count=28,
        labeled_fraction=1.0,
        seed=23,
    )
    phone_features_csv = tmp_path / "phone_features.csv"
    model_json = tmp_path / "noise_model.json"
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

    assert model_json.exists()
    assert scores_csv.exists()
    assert metrics_json.exists()
    assert rows
    assert "predicted_label" in rows[0]
    assert "score_type_1" in rows[0]
    assert metrics["overall_accuracy"] >= 0.80
    assert metrics["macro_f1"] >= 0.80
