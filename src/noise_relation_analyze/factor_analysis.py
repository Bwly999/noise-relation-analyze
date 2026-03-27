from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import random


@dataclass(frozen=True)
class FactorImpact:
    factor_key: str
    strength: float
    abs_strength: float
    direction: str
    spearman_strength: float
    bootstrap_stability: float
    bin_means: list[float]
    mean_abs_shap: float


def analyze_factor_impacts(
    rows: list[dict[str, float]],
    target_key: str,
    factor_keys: list[str],
    bootstrap_rounds: int = 32,
    bins: int = 4,
    seed: int = 42,
) -> list[FactorImpact]:
    if len(rows) < 2:
        raise ValueError("at least two rows are required")

    target_values = [float(row[target_key]) for row in rows]
    impacts = [
        _build_factor_impact(
            factor_key=factor_key,
            factor_values=[float(row[factor_key]) for row in rows],
            target_values=target_values,
            bootstrap_rounds=bootstrap_rounds,
            bins=bins,
            seed=seed,
        )
        for factor_key in factor_keys
    ]
    return sorted(
        impacts,
        key=lambda impact: (
            1 if impact.direction == "positive" else 0,
            impact.abs_strength,
            abs(impact.spearman_strength),
            impact.bootstrap_stability,
            impact.mean_abs_shap,
        ),
        reverse=True,
    )


def _build_factor_impact(
    factor_key: str,
    factor_values: list[float],
    target_values: list[float],
    bootstrap_rounds: int,
    bins: int,
    seed: int,
) -> FactorImpact:
    pearson_strength = _pearson(factor_values, target_values)
    spearman_strength = _spearman(factor_values, target_values)
    direction = "positive" if spearman_strength >= 0 else "negative"
    bin_means = _bin_means(factor_values, target_values, bins=bins)
    bootstrap_stability = _bootstrap_stability(
        factor_values,
        target_values,
        expected_sign=1 if spearman_strength >= 0 else -1,
        rounds=bootstrap_rounds,
        seed=seed + sum(ord(character) for character in factor_key),
    )
    mean_abs_shap = _mean_abs_shap(factor_values, target_values)

    return FactorImpact(
        factor_key=factor_key,
        strength=pearson_strength,
        abs_strength=abs(pearson_strength),
        direction=direction,
        spearman_strength=spearman_strength,
        bootstrap_stability=bootstrap_stability,
        bin_means=bin_means,
        mean_abs_shap=mean_abs_shap,
    )


def build_shap_style_summary(
    rows: list[dict[str, float]],
    target_key: str,
    factor_keys: list[str],
    top_n: int = 5,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str | dict[str, float]]]]:
    impacts = analyze_factor_impacts(rows, target_key=target_key, factor_keys=factor_keys)
    shap_summary = [
        {
            "factor_key": impact.factor_key,
            "mean_abs_shap": round(impact.mean_abs_shap, 6),
            "direction": impact.direction,
        }
        for impact in impacts
    ]

    target_mean = _mean(float(row[target_key]) for row in rows)
    sample_explanations: list[dict[str, float | str | dict[str, float]]] = []
    top_factors = [impact.factor_key for impact in impacts[: min(top_n, len(impacts))]]
    factor_means = {
        factor_key: _mean(float(row[factor_key]) for row in rows)
        for factor_key in factor_keys
    }
    factor_scales = {
        factor_key: max(_std(float(row[factor_key]) for row in rows), 1e-6)
        for factor_key in factor_keys
    }
    factor_coefficients = {
        factor_key: _standardized_linear_weight(
            [float(row[factor_key]) for row in rows],
            [float(row[target_key]) for row in rows],
        )
        for factor_key in factor_keys
    }

    ranked_rows = sorted(rows, key=lambda row: float(row[target_key]), reverse=True)[:top_n]
    for row in ranked_rows:
        contributions = {
            factor_key: round(
                factor_coefficients[factor_key]
                * ((float(row[factor_key]) - factor_means[factor_key]) / factor_scales[factor_key]),
                6,
            )
            for factor_key in top_factors
        }
        sample_explanations.append(
            {
                "target_value": round(float(row[target_key]), 6),
                "baseline": round(target_mean, 6),
                "contributions": contributions,
            }
        )

    return shap_summary, sample_explanations


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered, strict=True))
    left_norm = sqrt(sum(value * value for value in left_centered))
    right_norm = sqrt(sum(value * value for value in right_centered))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_rank_values(left), _rank_values(right))


def _rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        next_position = position
        while next_position < len(indexed) and indexed[next_position][1] == indexed[position][1]:
            next_position += 1
        average_rank = (position + next_position - 1) / 2 + 1
        for index, _ in indexed[position:next_position]:
            ranks[index] = average_rank
        position = next_position
    return ranks


def _bin_means(factor_values: list[float], target_values: list[float], bins: int) -> list[float]:
    paired = sorted(zip(factor_values, target_values, strict=True), key=lambda item: item[0])
    chunk_size = max(1, len(paired) // bins)
    results: list[float] = []
    for start in range(0, len(paired), chunk_size):
        chunk = paired[start : start + chunk_size]
        if not chunk:
            continue
        results.append(_mean(target for _, target in chunk))
    while len(results) > bins:
        results[-2] = (results[-2] + results[-1]) / 2
        results.pop()
    return results


def _bootstrap_stability(
    factor_values: list[float],
    target_values: list[float],
    expected_sign: int,
    rounds: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    if rounds <= 0:
        return 0.0
    stable = 0
    for _ in range(rounds):
        sampled_indices = [rng.randrange(len(factor_values)) for _ in range(len(factor_values))]
        sampled_factor = [factor_values[index] for index in sampled_indices]
        sampled_target = [target_values[index] for index in sampled_indices]
        sign = 1 if _spearman(sampled_factor, sampled_target) >= 0 else -1
        if sign == expected_sign:
            stable += 1
    return stable / rounds


def _mean_abs_shap(factor_values: list[float], target_values: list[float]) -> float:
    return abs(_standardized_linear_weight(factor_values, target_values))


def _linear_coefficient(factor_values: list[float], target_values: list[float]) -> float:
    factor_mean = _mean(factor_values)
    target_mean = _mean(target_values)
    numerator = sum(
        (factor - factor_mean) * (target - target_mean)
        for factor, target in zip(factor_values, target_values, strict=True)
    )
    denominator = sum((factor - factor_mean) ** 2 for factor in factor_values)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _standardized_linear_weight(factor_values: list[float], target_values: list[float]) -> float:
    target_scale = max(_std(target_values), 1e-6)
    return _pearson(factor_values, target_values) * target_scale


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _std(values) -> float:
    values = list(values)
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / len(values))
