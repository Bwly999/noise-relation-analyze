from pathlib import Path

from noise_relation_analyze.cli import build_parser, load_pipeline_config


def test_load_pipeline_config_reads_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        "\n".join(
            [
                "[project]",
                'name = "demo"',
                "",
                "[paths]",
                'raw_audio = "data/raw/audio"',
                'dimensions = "data/raw/dimensions.csv"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)

    assert config["project"]["name"] == "demo"
    assert config["paths"]["raw_audio"] == "data/raw/audio"


def test_build_parser_exposes_expected_commands() -> None:
    parser = build_parser()

    actions = {
        action.dest
        for action in parser._subparsers._group_actions  # type: ignore[attr-defined]
        if hasattr(action, "choices")
    }

    assert "command" in actions
