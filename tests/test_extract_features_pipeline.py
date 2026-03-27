from __future__ import annotations

import csv
from pathlib import Path
import wave

from noise_relation_analyze.pipeline import run_extract_features


def test_run_extract_features_writes_phone_level_summary(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    wav_a = audio_dir / "dir_1.wav"
    wav_b = audio_dir / "dir_2.wav"
    _write_pcm16_wave(wav_a, ([0] * 400 + [22000] * 500 + [0] * 400) * 2, sample_rate=8000)
    _write_pcm16_wave(wav_b, ([0] * 400 + [12000] * 500 + [0] * 400) * 2, sample_rate=8000)

    audio_asset_csv = tmp_path / "audio_asset.csv"
    output_csv = tmp_path / "phone_features.csv"
    audio_asset_csv.write_text(
        "\n".join(
            [
                "phone_id,direction_id,wav_path",
                f"P001,dir_1,{wav_a}",
                f"P001,dir_2,{wav_b}",
            ]
        ),
        encoding="utf-8",
    )

    run_extract_features(audio_asset_csv=audio_asset_csv, output_csv=output_csv)

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["phone_id"] == "P001"
    assert rows[0]["primary_direction"] == "dir_1"
    assert rows[0]["direction_count"] == "2"
    assert float(rows[0]["severity_index"]) > 0.0


def _write_pcm16_wave(path: Path, samples: list[int], sample_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = b"".join(int(sample).to_bytes(2, signed=True, byteorder="little") for sample in samples)
        handle.writeframes(frames)
