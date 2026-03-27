from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier, XGBRegressor

from noise_relation_analyze.factor_analysis import analyze_factor_impacts


ACOUSTIC_SEVERITY_FEATURE_KEYS = [
    "severity_index",
    "impulse_score_mean",
    "asymmetry_score",
]

ACOUSTIC_DIAGNOSTIC_FEATURE_KEYS = [
    "crest_factor_mean",
    "high_band_ratio_mean",
    "spectral_centroid_mean",
]

DEFAULT_DIMENSION_FACTOR_KEYS = [
    "hinge_gap",
    "left_support_gap",
    "right_support_gap",
    "torsion_delta",
    "panel_flushness",
    "adhesive_thickness",
]


@dataclass(frozen=True)
class SingleTypeModelBundle:
    noise_type: str
    factor_keys: list[str]
    severity_estimator: Any
    ng_estimator: Any
    severity_cv_r2: float
    severity_cv_mae: float
    severity_cv_spearman: float
    ng_cv_auc: float
    ng_cv_f1: float
    ng_positive_rate: float

    def summary(self) -> dict[str, float | str | list[str]]:
        return {
            "noise_type": self.noise_type,
            "factor_keys": list(self.factor_keys),
            "severity_model_type": "xgboost_regressor",
            "ng_model_type": "xgboost_classifier",
            "severity_cv_r2": self.severity_cv_r2,
            "severity_cv_mae": self.severity_cv_mae,
            "severity_cv_spearman": self.severity_cv_spearman,
            "ng_cv_auc": self.ng_cv_auc,
            "ng_cv_f1": self.ng_cv_f1,
            "ng_positive_rate": self.ng_positive_rate,
        }


def quantify_acoustic_severity(feature_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not feature_rows:
        return []

    feature_matrix = np.asarray(
        [
            [float(row[key]) for key in ACOUSTIC_SEVERITY_FEATURE_KEYS]
            for row in feature_rows
        ],
        dtype=float,
    )
    robust_scaled = _robust_scale(feature_matrix)
    composite = (
        (0.65 * robust_scaled[:, 0])
        + (0.20 * robust_scaled[:, 1])
        + (0.15 * robust_scaled[:, 2])
    )
    scaled_score = _minmax_scale(composite, lower=0.0, upper=100.0)
    rank_score = _rank_percentile(composite)

    quantified_rows: list[dict[str, str]] = []
    for index, row in enumerate(feature_rows):
        quantified = dict(row)
        quantified["severity_raw"] = f"{float(composite[index]):.6f}"
        quantified["severity_score"] = f"{float(scaled_score[index]):.6f}"
        quantified["severity_rank"] = f"{float(rank_score[index]):.6f}"
        quantified_rows.append(quantified)
    return quantified_rows


def build_single_type_analysis_rows(
    phone_feature_rows: list[dict[str, str]],
    dimension_rows: list[dict[str, str]],
    label_rows: list[dict[str, str]],
    noise_type: str,
) -> list[dict[str, str]]:
    quantified_features = {
        row["phone_id"]: row
        for row in quantify_acoustic_severity(phone_feature_rows)
    }
    labels_by_phone = {row["phone_id"]: row for row in label_rows}

    analysis_rows: list[dict[str, str]] = []
    for dimension_row in dimension_rows:
        phone_id = dimension_row["phone_id"]
        feature_row = quantified_features.get(phone_id)
        label_row = labels_by_phone.get(phone_id)
        if feature_row is None:
            continue
        if label_row is None:
            continue
        if label_row.get("noise_type_label") not in {"", noise_type}:
            continue

        merged = {
            "phone_id": phone_id,
            "noise_type": noise_type,
            "primary_direction": feature_row["primary_direction"],
            "severity_raw": feature_row["severity_raw"],
            "severity_score": feature_row["severity_score"],
            "severity_rank": feature_row["severity_rank"],
            "severity_index": feature_row["severity_index"],
            "impulse_score_mean": feature_row["impulse_score_mean"],
            "crest_factor_mean": feature_row["crest_factor_mean"],
            "high_band_ratio_mean": feature_row["high_band_ratio_mean"],
            "spectral_centroid_mean": feature_row["spectral_centroid_mean"],
            "asymmetry_score": feature_row["asymmetry_score"],
            "is_ng": _normalize_binary_label(label_row.get("is_ng", "")),
            "label_source": label_row.get("label_source", ""),
        }
        if "true_severity" in label_row:
            merged["true_severity"] = label_row["true_severity"]
        for factor_key in DEFAULT_DIMENSION_FACTOR_KEYS:
            if factor_key in dimension_row:
                merged[factor_key] = dimension_row[factor_key]
        analysis_rows.append(merged)
    return analysis_rows


def train_single_type_models(
    rows: list[dict[str, str]],
    noise_type: str,
    factor_keys: list[str] | None = None,
) -> SingleTypeModelBundle:
    if not rows:
        raise ValueError("no rows available for training")

    active_factor_keys = factor_keys or list(DEFAULT_DIMENSION_FACTOR_KEYS)
    x = np.asarray(
        [[float(row[key]) for key in active_factor_keys] for row in rows],
        dtype=float,
    )
    severity_target = np.asarray([float(row["severity_score"]) for row in rows], dtype=float)
    ng_target = np.asarray([int(row["is_ng"]) for row in rows], dtype=int)

    severity_cv = KFold(n_splits=min(5, len(rows)), shuffle=True, random_state=42)
    severity_estimator = _build_severity_regressor()
    severity_cv_predictions = cross_val_predict(
        severity_estimator,
        x,
        severity_target,
        cv=severity_cv,
        method="predict",
    )

    positive_count = int(np.sum(ng_target))
    negative_count = int(len(ng_target) - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("is_ng must contain both positive and negative samples")
    ng_splits = max(2, min(5, positive_count, negative_count))
    ng_cv = StratifiedKFold(n_splits=ng_splits, shuffle=True, random_state=42)
    ng_estimator = _build_ng_classifier(positive_count=positive_count, negative_count=negative_count)
    ng_cv_probabilities = cross_val_predict(
        ng_estimator,
        x,
        ng_target,
        cv=ng_cv,
        method="predict_proba",
    )[:, 1]
    ng_cv_predictions = (ng_cv_probabilities >= 0.5).astype(int)

    severity_estimator.fit(x, severity_target)
    ng_estimator.fit(x, ng_target)

    return SingleTypeModelBundle(
        noise_type=noise_type,
        factor_keys=list(active_factor_keys),
        severity_estimator=severity_estimator,
        ng_estimator=ng_estimator,
        severity_cv_r2=float(r2_score(severity_target, severity_cv_predictions)),
        severity_cv_mae=float(mean_absolute_error(severity_target, severity_cv_predictions)),
        severity_cv_spearman=float(_spearman(severity_target, severity_cv_predictions)),
        ng_cv_auc=float(roc_auc_score(ng_target, ng_cv_probabilities)),
        ng_cv_f1=float(f1_score(ng_target, ng_cv_predictions)),
        ng_positive_rate=float(np.mean(ng_target)),
    )


def score_single_type_rows(
    model: SingleTypeModelBundle,
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not rows:
        return []

    x = np.asarray(
        [[float(row[key]) for key in model.factor_keys] for row in rows],
        dtype=float,
    )
    severity_predictions = model.severity_estimator.predict(x)
    ng_probabilities = model.ng_estimator.predict_proba(x)[:, 1]
    ng_predictions = (ng_probabilities >= 0.5).astype(int)

    scored_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        scored_row = dict(row)
        scored_row["severity_score_pred"] = f"{float(severity_predictions[index]):.6f}"
        scored_row["ng_risk"] = f"{float(ng_probabilities[index]):.6f}"
        scored_row["predicted_is_ng"] = str(int(ng_predictions[index]))
        scored_rows.append(scored_row)
    return scored_rows


def summarize_single_type_report(
    rows: list[dict[str, str]],
    scored_rows: list[dict[str, str]],
    model: SingleTypeModelBundle,
) -> dict:
    severity_impacts = analyze_factor_impacts(
        [
            {"severity_score": float(row["severity_score"]), **_select_factor_values(row, model.factor_keys)}
            for row in rows
        ],
        target_key="severity_score",
        factor_keys=model.factor_keys,
    )
    ng_impacts = analyze_factor_impacts(
        [
            {"is_ng": float(row["is_ng"]), **_select_factor_values(row, model.factor_keys)}
            for row in rows
        ],
        target_key="is_ng",
        factor_keys=model.factor_keys,
    )
    severity_shap_summary, severity_samples = _build_tree_shap_summary(
        estimator=model.severity_estimator,
        rows=rows,
        factor_keys=model.factor_keys,
        prediction_key="severity_score_pred",
        phone_id_key="phone_id",
    )
    ng_shap_summary, ng_samples = _build_tree_shap_summary(
        estimator=model.ng_estimator,
        rows=rows,
        factor_keys=model.factor_keys,
        prediction_key="ng_risk",
        phone_id_key="phone_id",
        probability_output=True,
    )

    scored_by_phone = {row["phone_id"]: row for row in scored_rows}
    truth_alignment: dict[str, float] = {}
    if rows and all("true_severity" in row for row in rows):
        truth = [float(row["true_severity"]) for row in rows]
        severity = [float(row["severity_score"]) for row in rows]
        truth_alignment["severity_truth_spearman"] = float(_spearman(truth, severity))
    ng_labels = [int(row["is_ng"]) for row in rows]
    severity_scores = [float(row["severity_score"]) for row in rows]
    truth_alignment["severity_ng_auc"] = float(roc_auc_score(ng_labels, severity_scores))
    truth_alignment["severity_ng_gap"] = float(
        np.mean([severity_scores[index] for index, value in enumerate(ng_labels) if value == 1])
        - np.mean([severity_scores[index] for index, value in enumerate(ng_labels) if value == 0])
    )

    return {
        "noise_type": model.noise_type,
        "analysis_method": "single_type_severity_plus_xgboost_tree_shap",
        "sample_count": len(rows),
        "ng_count": int(sum(int(row["is_ng"]) for row in rows)),
        "acoustic_quantification": {
            "method": "robust_weighted_intensity_impulse_asymmetry",
            "feature_keys": list(ACOUSTIC_SEVERITY_FEATURE_KEYS),
            "diagnostic_feature_keys": list(ACOUSTIC_DIAGNOSTIC_FEATURE_KEYS),
            **truth_alignment,
        },
        "severity_model": {
            "model_type": "xgboost_regressor",
            "cv_r2": round(model.severity_cv_r2, 6),
            "cv_mae": round(model.severity_cv_mae, 6),
            "cv_spearman": round(model.severity_cv_spearman, 6),
        },
        "ng_model": {
            "model_type": "xgboost_classifier",
            "cv_auc": round(model.ng_cv_auc, 6),
            "cv_f1": round(model.ng_cv_f1, 6),
            "positive_rate": round(model.ng_positive_rate, 6),
        },
        "severity_top_factors": _merge_impacts_with_shap(severity_impacts, severity_shap_summary),
        "ng_top_factors": _merge_impacts_with_shap(ng_impacts, ng_shap_summary),
        "severity_shap_summary": severity_shap_summary,
        "severity_sample_explanations": severity_samples,
        "ng_shap_summary": ng_shap_summary,
        "ng_sample_explanations": ng_samples,
        "prediction_summary": _build_prediction_summary(scored_by_phone, rows),
    }


def save_single_type_model(model: SingleTypeModelBundle, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(model, handle)


def load_single_type_model(model_path: Path) -> SingleTypeModelBundle:
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def _build_severity_regressor() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=260,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=1.2,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )


def _build_ng_classifier(positive_count: int, negative_count: int) -> XGBClassifier:
    scale_pos_weight = negative_count / max(1, positive_count)
    return XGBClassifier(
        n_estimators=220,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.02,
        reg_lambda=1.2,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
    )


def _robust_scale(values: np.ndarray) -> np.ndarray:
    medians = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - medians), axis=0, keepdims=True)
    scales = np.where(mad > 1e-9, mad * 1.4826, np.std(values, axis=0, keepdims=True))
    scales = np.where(scales > 1e-9, scales, 1.0)
    return np.clip((values - medians) / scales, -6.0, 6.0)


def _minmax_scale(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum <= 1e-9:
        midpoint = (lower + upper) / 2.0
        return np.full_like(values, fill_value=midpoint, dtype=float)
    scaled = (values - minimum) / (maximum - minimum)
    return lower + (scaled * (upper - lower))


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    denominator = max(1, len(values) - 1)
    return (ranks / denominator) * 100.0


def _normalize_binary_label(value: str) -> str:
    normalized = str(value).strip().lower()
    return "1" if normalized in {"1", "true", "yes", "ng"} else "0"


def _pearson(left: np.ndarray | list[float], right: np.ndarray | list[float]) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.size == 0 or right_values.size == 0:
        return 0.0
    left_centered = left_values - np.mean(left_values)
    right_centered = right_values - np.mean(right_values)
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _spearman(left: np.ndarray | list[float], right: np.ndarray | list[float]) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    return _pearson(_rank_array(left_values), _rank_array(right_values))


def _rank_array(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    return ranks


def _select_factor_values(row: dict[str, str], factor_keys: list[str]) -> dict[str, float]:
    return {factor_key: float(row[factor_key]) for factor_key in factor_keys}


def _merge_impacts_with_shap(impacts, shap_summary: list[dict[str, float | str]]) -> list[dict]:
    shap_by_factor = {
        str(item["factor_key"]): float(item["mean_abs_shap"])
        for item in shap_summary
    }
    merged = [
        {
            "factor_key": impact.factor_key,
            "direction": impact.direction,
            "strength": round(impact.strength, 6),
            "abs_strength": round(impact.abs_strength, 6),
            "spearman_strength": round(impact.spearman_strength, 6),
            "bootstrap_stability": round(impact.bootstrap_stability, 6),
            "bin_means": [round(value, 6) for value in impact.bin_means],
            "mean_abs_shap": round(shap_by_factor.get(impact.factor_key, 0.0), 6),
        }
        for impact in impacts
    ]
    return sorted(
        merged,
        key=lambda item: (float(item["mean_abs_shap"]), float(item["abs_strength"])),
        reverse=True,
    )


def _build_tree_shap_summary(
    estimator,
    rows: list[dict[str, str]],
    factor_keys: list[str],
    prediction_key: str,
    phone_id_key: str,
    probability_output: bool = False,
) -> tuple[list[dict], list[dict]]:
    import shap

    x = np.asarray(
        [[float(row[key]) for key in factor_keys] for row in rows],
        dtype=float,
    )
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(x)
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, 0]

    mean_abs = np.mean(np.abs(values), axis=0)
    ranking = sorted(
        [
            {
                "factor_key": factor_key,
                "mean_abs_shap": round(float(mean_abs[index]), 6),
            }
            for index, factor_key in enumerate(factor_keys)
        ],
        key=lambda item: item["mean_abs_shap"],
        reverse=True,
    )

    if probability_output:
        predictions = estimator.predict_proba(x)[:, 1]
    else:
        predictions = estimator.predict(x)

    top_indices = np.argsort(np.abs(values).sum(axis=1))[::-1][:5]
    samples: list[dict] = []
    for row_index in top_indices:
        contributions = sorted(
            [
                (factor_keys[col_index], float(values[row_index, col_index]))
                for col_index in range(len(factor_keys))
            ],
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:5]
        samples.append(
            {
                "phone_id": rows[row_index][phone_id_key],
                "prediction": round(float(predictions[row_index]), 6),
                "top_contributions": {
                    factor_key: round(value, 6)
                    for factor_key, value in contributions
                },
            }
        )
    return ranking, samples


def _build_prediction_summary(
    scored_by_phone: dict[str, dict[str, str]],
    rows: list[dict[str, str]],
) -> dict[str, float]:
    severity_true = [float(row["severity_score"]) for row in rows]
    severity_pred = [float(scored_by_phone[row["phone_id"]]["severity_score_pred"]) for row in rows]
    ng_true = [int(row["is_ng"]) for row in rows]
    ng_risk = [float(scored_by_phone[row["phone_id"]]["ng_risk"]) for row in rows]
    return {
        "severity_r2_in_sample": round(float(r2_score(severity_true, severity_pred)), 6),
        "severity_spearman_in_sample": round(float(_spearman(severity_true, severity_pred)), 6),
        "ng_auc_in_sample": round(float(roc_auc_score(ng_true, ng_risk)), 6),
    }
