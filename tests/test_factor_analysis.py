from noise_relation_analyze.factor_analysis import analyze_factor_impacts


def test_analyze_factor_impacts_ranks_strongest_positive_factor_first() -> None:
    rows = [
        {"gap_a": 0.10, "gap_b": 1.00, "risk_score": 0.10},
        {"gap_a": 0.20, "gap_b": 0.90, "risk_score": 0.20},
        {"gap_a": 0.30, "gap_b": 0.70, "risk_score": 0.35},
        {"gap_a": 0.40, "gap_b": 0.60, "risk_score": 0.50},
        {"gap_a": 0.50, "gap_b": 0.50, "risk_score": 0.65},
    ]

    results = analyze_factor_impacts(rows, target_key="risk_score", factor_keys=["gap_a", "gap_b"])

    assert results[0].factor_key == "gap_a"
    assert results[0].direction == "positive"
    assert results[0].abs_strength >= results[1].abs_strength
