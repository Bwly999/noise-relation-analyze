from pathlib import Path
import csv
import json

from noise_relation_analyze.pipeline import run_analyze_factors, run_validate_joins


def test_run_validate_joins_writes_report_json(tmp_path: Path) -> None:
    phone_master = tmp_path / "phone_master.csv"
    audio_assets = tmp_path / "audio_asset.csv"
    labels = tmp_path / "labels.csv"
    output = tmp_path / "join_report.json"

    phone_master.write_text(
        "\n".join(
            [
                "phone_id,model_id,batch_id,vendor_id,structure_version,material_version",
                "P001,M1,B1,V1,S1,MAT1",
            ]
        ),
        encoding="utf-8",
    )
    audio_assets.write_text(
        "\n".join(
            [
                "phone_id,direction_id,wav_path",
                "P001,dir_1,a.wav",
                "P001,dir_2,b.wav",
                "P001,dir_3,c.wav",
                "P001,dir_4,d.wav",
            ]
        ),
        encoding="utf-8",
    )
    labels.write_text(
        "\n".join(
            [
                "phone_id,noise_type_label,label_source",
                "P001,type_1,manual",
            ]
        ),
        encoding="utf-8",
    )

    run_validate_joins(phone_master, audio_assets, labels, output)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["valid_phone_count"] == 1
    assert report["issues"] == []


def test_run_analyze_factors_writes_ranked_csv(tmp_path: Path) -> None:
    input_csv = tmp_path / "factor_input.csv"
    output_csv = tmp_path / "factor_output.csv"
    input_csv.write_text(
        "\n".join(
            [
                "gap_a,gap_b,risk_score",
                "0.10,1.00,0.10",
                "0.20,0.90,0.20",
                "0.30,0.70,0.35",
                "0.40,0.60,0.50",
                "0.50,0.50,0.65",
            ]
        ),
        encoding="utf-8",
    )

    run_analyze_factors(
        input_csv=input_csv,
        output_csv=output_csv,
        target_key="risk_score",
        factor_keys=["gap_a", "gap_b"],
    )

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["factor_key"] == "gap_a"
    assert rows[0]["direction"] == "positive"
