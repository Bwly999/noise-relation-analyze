from noise_relation_analyze.audio import segment_cycles_from_energy


def test_segment_cycles_from_energy_detects_three_regions() -> None:
    energy = [
        0.0,
        0.2,
        0.8,
        0.9,
        0.1,
        0.0,
        0.0,
        0.7,
        0.8,
        0.1,
        0.0,
        0.0,
        0.75,
        0.85,
        0.05,
        0.0,
    ]

    segments = segment_cycles_from_energy(energy, threshold=0.5, min_length=2)

    assert segments == [(2, 4), (7, 9), (12, 14)]
