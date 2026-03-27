from __future__ import annotations

import csv
from pathlib import Path

from noise_relation_analyze.synthetic_data import generate_synthetic_dataset


def test_generate_synthetic_dataset_writes_csvs_and_wavs(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic"

    manifest = generate_synthetic_dataset(output_dir=output_dir, phone_count=6, labeled_fraction=0.5, seed=7)

    assert manifest.phone_master_csv.exists()
    assert manifest.audio_asset_csv.exists()
    assert manifest.labels_csv.exists()
    assert manifest.dimension_csv.exists()

    with manifest.phone_master_csv.open("r", encoding="utf-8", newline="") as handle:
        phone_rows = list(csv.DictReader(handle))
    with manifest.audio_asset_csv.open("r", encoding="utf-8", newline="") as handle:
        audio_rows = list(csv.DictReader(handle))
    with manifest.labels_csv.open("r", encoding="utf-8", newline="") as handle:
        label_rows = list(csv.DictReader(handle))

    assert len(phone_rows) == 6
    assert len(audio_rows) == 24
    assert len(label_rows) == 3
    assert all(Path(row["wav_path"]).exists() for row in audio_rows)
