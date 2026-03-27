from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


FEATURE_KEYS = [
    "severity_index",
    "asymmetry_score",
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
]


@dataclass(frozen=True)
class NoiseScorerModel:
    model_type: str
    feature_keys: list[str]
    labels: list[str]
    estimator: Any
    cv_accuracy: float
    cv_macro_f1: float

    def summary(self) -> dict[str, float | str]:
        return {
            "model_type": self.model_type,
            "cv_accuracy": self.cv_accuracy,
            "cv_macro_f1": self.cv_macro_f1,
        }


def train_noise_scorer(rows: list[dict[str, str]], labels: dict[str, str]) -> NoiseScorerModel:
    training_rows = [row for row in rows if row["phone_id"] in labels]
    if not training_rows:
        raise ValueError("no labeled rows available for training")

    label_names = sorted(set(labels.values()))
    x = np.asarray(
        [[float(row[key]) for key in FEATURE_KEYS] for row in training_rows],
        dtype=float,
    )
    y = np.asarray([labels[row["phone_id"]] for row in training_rows], dtype=object)

    min_class_count = min(sum(1 for label in y if label == label_name) for label_name in label_names)
    n_splits = max(2, min(5, min_class_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    estimator = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    cv_predictions = cross_val_predict(estimator, x, y, cv=cv, method="predict")
    cv_accuracy = float(accuracy_score(y, cv_predictions))
    cv_macro_f1 = float(f1_score(y, cv_predictions, average="macro"))
    estimator.fit(x, y)

    return NoiseScorerModel(
        model_type="random_forest",
        feature_keys=list(FEATURE_KEYS),
        labels=label_names,
        estimator=estimator,
        cv_accuracy=cv_accuracy,
        cv_macro_f1=cv_macro_f1,
    )


def score_rows(model: NoiseScorerModel, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    x = np.asarray(
        [[float(row[key]) for key in model.feature_keys] for row in rows],
        dtype=float,
    )
    probabilities = model.estimator.predict_proba(x)
    predicted_labels = model.estimator.predict(x)
    class_order = list(model.estimator.classes_)
    scored_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        scored_row = dict(row)
        for class_index, label_name in enumerate(class_order):
            scored_row[f"score_{label_name}"] = f"{float(probabilities[index, class_index]):.6f}"
        scored_row["predicted_label"] = str(predicted_labels[index])
        scored_rows.append(scored_row)
    return scored_rows


def evaluate_scored_rows(
    scored_rows: list[dict[str, str]], labels: dict[str, str]
) -> dict[str, float | dict[str, dict[str, float]]]:
    labeled_rows = [row for row in scored_rows if row["phone_id"] in labels]
    if not labeled_rows:
        raise ValueError("no overlapping labeled rows for evaluation")

    label_names = sorted(set(labels.values()))
    correct = sum(1 for row in labeled_rows if row["predicted_label"] == labels[row["phone_id"]])
    overall_accuracy = correct / len(labeled_rows)

    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label_name in label_names:
        tp = sum(
            1
            for row in labeled_rows
            if row["predicted_label"] == label_name and labels[row["phone_id"]] == label_name
        )
        fp = sum(
            1
            for row in labeled_rows
            if row["predicted_label"] == label_name and labels[row["phone_id"]] != label_name
        )
        fn = sum(
            1
            for row in labeled_rows
            if row["predicted_label"] != label_name and labels[row["phone_id"]] == label_name
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "overall_accuracy": overall_accuracy,
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_label": per_label,
    }


def save_model(model: NoiseScorerModel, output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("wb") as handle:
        pickle.dump(model, handle)


def load_model(model_json: Path) -> NoiseScorerModel:
    with model_json.open("rb") as handle:
        return pickle.load(handle)
