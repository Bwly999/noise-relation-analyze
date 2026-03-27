from noise_relation_analyze.features import CycleFeature, aggregate_phone_features


def test_aggregate_phone_features_preserves_direction_asymmetry() -> None:
    cycle_features = [
        CycleFeature(phone_id="P001", direction_id="dir_1", cycle_id="c1", rms=0.8, impulse_score=0.6),
        CycleFeature(phone_id="P001", direction_id="dir_1", cycle_id="c2", rms=0.7, impulse_score=0.5),
        CycleFeature(phone_id="P001", direction_id="dir_2", cycle_id="c1", rms=0.2, impulse_score=0.1),
        CycleFeature(phone_id="P001", direction_id="dir_2", cycle_id="c2", rms=0.3, impulse_score=0.2),
    ]

    summary = aggregate_phone_features("P001", cycle_features)

    assert summary.primary_direction == "dir_1"
    assert summary.direction_count == 2
    assert summary.asymmetry_score > 0.4
    assert summary.severity_index > 0.0
