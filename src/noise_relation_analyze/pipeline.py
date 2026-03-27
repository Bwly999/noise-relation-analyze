from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import shap

from noise_relation_analyze.audio import load_wav_samples
from noise_relation_analyze.factor_analysis import analyze_factor_impacts, build_shap_style_summary
from noise_relation_analyze.features import aggregate_phone_features, extract_cycle_features
from noise_relation_analyze.registry import (
    AudioAsset,
    JoinValidationReport,
    LabelRecord,
    PhoneMasterRecord,
    validate_joined_records,
)
from noise_relation_analyze.scoring import (
    evaluate_scored_rows,
    load_model,
    save_model,
    score_rows,
    train_noise_scorer,
)
from noise_relation_analyze.synthetic_data import generate_synthetic_dataset


def run_validate_joins(
    phone_master_csv: Path,
    audio_asset_csv: Path,
    labels_csv: Path,
    output_json: Path,
) -> JoinValidationReport:
    phones = [
        PhoneMasterRecord(**row)
        for row in _read_csv_rows(phone_master_csv)
    ]
    assets = [
        AudioAsset(**row)
        for row in _read_csv_rows(audio_asset_csv)
    ]
    labels = [
        LabelRecord(**row)
        for row in _read_csv_rows(labels_csv)
    ]
    report = validate_joined_records(phones, assets, labels)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "valid_phone_count": report.valid_phone_count,
                "issues": [
                    {"phone_id": issue.phone_id, "code": issue.code, "detail": issue.detail}
                    for issue in report.issues
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def run_analyze_factors(
    input_csv: Path, output_csv: Path, target_key: str, factor_keys: list[str]
) -> None:
    rows = []
    for row in _read_csv_rows(input_csv):
        numeric_row = {target_key: float(row[target_key])}
        for factor_key in factor_keys:
            numeric_row[factor_key] = float(row[factor_key])
        rows.append(numeric_row)
    results = analyze_factor_impacts(rows, target_key=target_key, factor_keys=factor_keys)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["factor_key", "strength", "abs_strength", "direction"]
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "factor_key": result.factor_key,
                    "strength": f"{result.strength:.6f}",
                    "abs_strength": f"{result.abs_strength:.6f}",
                    "direction": result.direction,
                }
            )


def run_extract_features(audio_asset_csv: Path, output_csv: Path) -> None:
    assets = [AudioAsset(**row) for row in _read_csv_rows(audio_asset_csv)]
    cycle_features = []
    phone_ids: set[str] = set()
    for asset in assets:
        waveform = load_wav_samples(Path(asset.wav_path))
        cycle_features.extend(
            extract_cycle_features(
                phone_id=asset.phone_id,
                direction_id=asset.direction_id,
                waveform=waveform,
            )
        )
        phone_ids.add(asset.phone_id)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "phone_id",
                "primary_direction",
                "direction_count",
                "asymmetry_score",
                "severity_index",
                "rms_mean",
                "impulse_score_mean",
                "crest_factor_mean",
                "zero_crossing_rate_mean",
                "spectral_centroid_mean",
                "high_band_ratio_mean",
                "dir_1_score",
                "dir_2_score",
                "dir_3_score",
                "dir_4_score",
                "dir_1_share",
                "dir_2_share",
                "dir_3_share",
                "dir_4_share",
            ],
        )
        writer.writeheader()
        for phone_id in sorted(phone_ids):
            summary = aggregate_phone_features(phone_id, cycle_features)
            writer.writerow(
                {
                    "phone_id": summary.phone_id,
                    "primary_direction": summary.primary_direction,
                    "direction_count": str(summary.direction_count),
                    "asymmetry_score": f"{summary.asymmetry_score:.6f}",
                    "severity_index": f"{summary.severity_index:.6f}",
                    "rms_mean": f"{summary.rms_mean:.6f}",
                    "impulse_score_mean": f"{summary.impulse_score_mean:.6f}",
                    "crest_factor_mean": f"{summary.crest_factor_mean:.6f}",
                    "zero_crossing_rate_mean": f"{summary.zero_crossing_rate_mean:.6f}",
                    "spectral_centroid_mean": f"{summary.spectral_centroid_mean:.6f}",
                    "high_band_ratio_mean": f"{summary.high_band_ratio_mean:.6f}",
                    "dir_1_score": f"{summary.dir_1_score:.6f}",
                    "dir_2_score": f"{summary.dir_2_score:.6f}",
                    "dir_3_score": f"{summary.dir_3_score:.6f}",
                    "dir_4_score": f"{summary.dir_4_score:.6f}",
                    "dir_1_share": f"{summary.dir_1_share:.6f}",
                    "dir_2_share": f"{summary.dir_2_share:.6f}",
                    "dir_3_share": f"{summary.dir_3_share:.6f}",
                    "dir_4_share": f"{summary.dir_4_share:.6f}",
                }
            )


def run_prepare_factor_data(
    phone_features_csv: Path,
    dimension_csv: Path,
    labels_csv: Path,
    output_csv: Path,
    noise_type: str,
    scores_csv: Path | None = None,
) -> None:
    features_by_phone = {
        row["phone_id"]: row
        for row in _read_csv_rows(phone_features_csv)
    }
    score_rows_by_phone = (
        {row["phone_id"]: row for row in _read_csv_rows(scores_csv)}
        if scores_csv is not None
        else {}
    )
    labels_by_phone = {
        row["phone_id"]: row["noise_type_label"]
        for row in _read_csv_rows(labels_csv)
    }
    dimension_rows = _read_csv_rows(dimension_csv)
    fieldnames = [
        "phone_id",
        "risk_score",
        "severity_index",
        "asymmetry_score",
        f"is_{noise_type}",
        "hinge_gap",
        "left_support_gap",
        "right_support_gap",
        "torsion_delta",
        "panel_flushness",
        "adhesive_thickness",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dimension_row in dimension_rows:
            phone_id = dimension_row["phone_id"]
            feature_row = features_by_phone.get(phone_id)
            if feature_row is None:
                continue
            label = labels_by_phone.get(phone_id, "")
            score_row = score_rows_by_phone.get(phone_id)
            if score_row is not None and f"score_{noise_type}" in score_row:
                risk_score = float(score_row[f"score_{noise_type}"])
            else:
                severity = float(feature_row["severity_index"])
                if label == noise_type:
                    risk_score = severity
                elif label:
                    risk_score = severity * 0.2
                else:
                    risk_score = severity
            writer.writerow(
                {
                    "phone_id": phone_id,
                    "risk_score": f"{risk_score:.6f}",
                    "severity_index": feature_row["severity_index"],
                    "asymmetry_score": feature_row["asymmetry_score"],
                    f"is_{noise_type}": "1" if label == noise_type else "0" if label else "",
                    "hinge_gap": dimension_row["hinge_gap"],
                    "left_support_gap": dimension_row["left_support_gap"],
                    "right_support_gap": dimension_row["right_support_gap"],
                    "torsion_delta": dimension_row["torsion_delta"],
                    "panel_flushness": dimension_row["panel_flushness"],
                    "adhesive_thickness": dimension_row["adhesive_thickness"],
                }
            )


def run_train_noise_scorer(phone_features_csv: Path, labels_csv: Path, output_json: Path) -> dict:
    feature_rows = _read_csv_rows(phone_features_csv)
    labels = {
        row["phone_id"]: row["noise_type_label"]
        for row in _read_csv_rows(labels_csv)
    }
    model = train_noise_scorer(feature_rows, labels)
    save_model(model, output_json)
    return model.summary()


def run_score_noise_types(phone_features_csv: Path, model_json: Path, output_csv: Path) -> None:
    model = load_model(model_json)
    feature_rows = _read_csv_rows(phone_features_csv)
    scored_rows = score_rows(model, feature_rows)
    fieldnames = list(scored_rows[0].keys()) if scored_rows else []
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_rows)


def run_evaluate_scores(scores_csv: Path, labels_csv: Path, output_json: Path) -> dict:
    scored_rows = _read_csv_rows(scores_csv)
    labels = {
        row["phone_id"]: row["noise_type_label"]
        for row in _read_csv_rows(labels_csv)
    }
    metrics = evaluate_scored_rows(scored_rows, labels)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_build_noise_report(
    scores_csv: Path,
    dimension_csv: Path,
    labels_csv: Path,
    output_json: Path,
    noise_type: str,
    factor_keys: list[str],
    model_path: Path | None = None,
    phone_features_csv: Path | None = None,
) -> dict:
    score_rows_by_phone = {
        row["phone_id"]: row
        for row in _read_csv_rows(scores_csv)
    }
    labels = {
        row["phone_id"]: row["noise_type_label"]
        for row in _read_csv_rows(labels_csv)
    }
    analysis_rows: list[dict[str, float]] = []
    positive_count = 0
    for dimension_row in _read_csv_rows(dimension_csv):
        phone_id = dimension_row["phone_id"]
        score_row = score_rows_by_phone.get(phone_id)
        if score_row is None or f"score_{noise_type}" not in score_row:
            continue
        if labels.get(phone_id) == noise_type:
            positive_count += 1
        analysis_row = {
            "risk_score": float(score_row[f"score_{noise_type}"]),
        }
        for factor_key in factor_keys:
            analysis_row[factor_key] = float(dimension_row[factor_key])
        analysis_rows.append(analysis_row)

    factor_impacts = analyze_factor_impacts(
        analysis_rows,
        target_key="risk_score",
        factor_keys=factor_keys,
    )
    shap_summary, sample_explanations = build_shap_style_summary(
        analysis_rows,
        target_key="risk_score",
        factor_keys=factor_keys,
    )
    analysis_method = "spearman_bootstrap_bins_with_shap_style_linear_attribution"
    report = {
        "noise_type": noise_type,
        "analysis_method": analysis_method,
        "sample_count": len(analysis_rows),
        "positive_label_count": positive_count,
        "top_factors": [
            {
                "factor_key": impact.factor_key,
                "strength": round(impact.strength, 6),
                "abs_strength": round(impact.abs_strength, 6),
                "direction": impact.direction,
                "spearman_strength": round(impact.spearman_strength, 6),
                "bootstrap_stability": round(impact.bootstrap_stability, 6),
                "bin_means": [round(value, 6) for value in impact.bin_means],
                "mean_abs_shap": round(impact.mean_abs_shap, 6),
            }
            for impact in factor_impacts
        ],
        "shap_summary": shap_summary,
        "sample_explanations": sample_explanations,
    }
    if model_path is not None and phone_features_csv is not None:
        feature_rows = _read_csv_rows(phone_features_csv)
        model = load_model(model_path)
        feature_shap_summary, feature_sample_explanations = _build_tree_shap_summary(
            model=model,
            feature_rows=feature_rows,
            target_label=noise_type,
        )
        report["analysis_method"] = "tree_shap_plus_spearman_bootstrap_factor_analysis"
        report["model_summary"] = model.summary()
        report["feature_shap_summary"] = feature_shap_summary
        report["feature_sample_explanations"] = feature_sample_explanations
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_demo_pipeline(
    output_dir: Path,
    phone_count: int = 24,
    labeled_fraction: float = 1.0,
    seed: int = 42,
    noise_types: list[str] | None = None,
) -> None:
    manifest = generate_synthetic_dataset(
        output_dir=output_dir,
        phone_count=phone_count,
        labeled_fraction=labeled_fraction,
        seed=seed,
        noise_types=noise_types,
    )
    artifacts_dir = output_dir / "artifacts"
    reports_dir = artifacts_dir / "reports"

    run_validate_joins(
        phone_master_csv=manifest.phone_master_csv,
        audio_asset_csv=manifest.audio_asset_csv,
        labels_csv=manifest.labels_csv,
        output_json=artifacts_dir / "join_report.json",
    )
    run_extract_features(
        audio_asset_csv=manifest.audio_asset_csv,
        output_csv=artifacts_dir / "phone_features.csv",
    )
    run_train_noise_scorer(
        phone_features_csv=artifacts_dir / "phone_features.csv",
        labels_csv=manifest.labels_csv,
        output_json=artifacts_dir / "noise_model.bin",
    )
    run_score_noise_types(
        phone_features_csv=artifacts_dir / "phone_features.csv",
        model_json=artifacts_dir / "noise_model.bin",
        output_csv=artifacts_dir / "scores.csv",
    )
    run_evaluate_scores(
        scores_csv=artifacts_dir / "scores.csv",
        labels_csv=manifest.labels_csv,
        output_json=artifacts_dir / "metrics.json",
    )
    configured_reports = {
        "type_1": ["hinge_gap", "left_support_gap", "right_support_gap", "torsion_delta"],
        "type_2": ["torsion_delta", "right_support_gap", "hinge_gap", "panel_flushness"],
        "type_3": ["panel_flushness", "adhesive_thickness", "hinge_gap", "left_support_gap"],
    }
    active_noise_types = noise_types or ["type_1", "type_2", "type_3", "normal"]
    for noise_type in active_noise_types:
        if noise_type == "normal" or noise_type not in configured_reports:
            continue
        run_build_noise_report(
            scores_csv=artifacts_dir / "scores.csv",
            dimension_csv=manifest.dimension_csv,
            labels_csv=manifest.labels_csv,
            output_json=reports_dir / f"{noise_type}_report.json",
            noise_type=noise_type,
            factor_keys=configured_reports[noise_type],
            model_path=artifacts_dir / "noise_model.bin",
            phone_features_csv=artifacts_dir / "phone_features.csv",
        )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_tree_shap_summary(model, feature_rows: list[dict[str, str]], target_label: str) -> tuple[list[dict], list[dict]]:
    x = np.asarray(
        [[float(row[key]) for key in model.feature_keys] for row in feature_rows],
        dtype=float,
    )
    explainer = shap.TreeExplainer(model.estimator)
    shap_values = explainer.shap_values(x)
    class_order = list(model.estimator.classes_)
    target_index = class_order.index(target_label)
    target_shap = _extract_target_shap_values(shap_values, target_index)
    base_values = _extract_base_values(explainer, x, target_index)

    mean_abs = np.mean(np.abs(target_shap), axis=0)
    preferred_feature_keys = [
        "severity_index",
        "asymmetry_score",
        "dir_1_score",
        "dir_2_score",
        "dir_3_score",
        "dir_4_score",
        "dir_1_share",
        "dir_2_share",
        "dir_3_share",
        "dir_4_share",
    ]
    ranking = sorted(
        [
            {
                "feature_key": feature_key,
                "mean_abs_shap": round(float(mean_abs[index]), 6),
            }
            for index, feature_key in enumerate(model.feature_keys)
            if feature_key in preferred_feature_keys
        ],
        key=lambda item: item["mean_abs_shap"],
        reverse=True,
    )

    top_indices = np.argsort(np.abs(target_shap).sum(axis=1))[::-1][:5]
    sample_explanations = []
    for row_index in top_indices:
        ranked_features = sorted(
            [
                (model.feature_keys[col_index], float(target_shap[row_index, col_index]))
                for col_index in range(target_shap.shape[1])
                if model.feature_keys[col_index] in preferred_feature_keys
            ],
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:5]
        sample_explanations.append(
            {
                "phone_id": feature_rows[row_index]["phone_id"],
                "target_label": target_label,
                "base_value": round(float(base_values[row_index]), 6),
                "prediction": round(float(model.estimator.predict_proba(x)[row_index, target_index]), 6),
                "top_contributions": {
                    feature_key: round(value, 6)
                    for feature_key, value in ranked_features
                },
            }
        )

    return ranking, sample_explanations


def _extract_target_shap_values(shap_values, target_index: int) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[target_index], dtype=float)
    array = np.asarray(shap_values, dtype=float)
    if array.ndim == 3:
        return array[:, :, target_index]
    return array


def _extract_base_values(explainer, x: np.ndarray, target_index: int) -> np.ndarray:
    explanation = explainer(x)
    base_values = np.asarray(explanation.base_values, dtype=float)
    if base_values.ndim == 2:
        return base_values[:, target_index]
    return base_values
