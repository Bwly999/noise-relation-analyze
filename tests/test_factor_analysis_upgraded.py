from noise_relation_analyze.factor_analysis import analyze_factor_impacts


def test_analyze_factor_impacts_returns_spearman_bootstrap_and_shap_summary() -> None:
    rows = [
        {"hinge_gap": 0.10, "left_support_gap": 0.08, "risk_score": 0.05},
        {"hinge_gap": 0.14, "left_support_gap": 0.10, "risk_score": 0.20},
        {"hinge_gap": 0.18, "left_support_gap": 0.13, "risk_score": 0.50},
        {"hinge_gap": 0.20, "left_support_gap": 0.14, "risk_score": 0.75},
        {"hinge_gap": 0.22, "left_support_gap": 0.15, "risk_score": 0.90},
        {"hinge_gap": 0.12, "left_support_gap": 0.09, "risk_score": 0.12},
    ]

    results = analyze_factor_impacts(
        rows,
        target_key="risk_score",
        factor_keys=["hinge_gap", "left_support_gap"],
        bootstrap_rounds=12,
        bins=3,
    )

    assert results[0].factor_key == "hinge_gap"
    assert results[0].spearman_strength > 0
    assert results[0].bootstrap_stability >= 0.5
    assert len(results[0].bin_means) == 3
    assert results[0].mean_abs_shap >= results[1].mean_abs_shap
