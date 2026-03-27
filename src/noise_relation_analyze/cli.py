from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

from noise_relation_analyze.pipeline import (
    run_analyze_factors,
    run_build_noise_report,
    run_demo_pipeline,
    run_evaluate_scores,
    run_extract_features,
    run_prepare_factor_data,
    run_render_noise_report_html,
    run_score_noise_types,
    run_train_noise_scorer,
    run_validate_joins,
)
from noise_relation_analyze.synthetic_data import generate_synthetic_dataset


def load_pipeline_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="noise-relation-analyze",
        description="Foldable-device noise correlation analysis toolkit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate-joins", help="Validate phone/audio/label joins.")
    validate.add_argument("--phone-master", type=Path)
    validate.add_argument("--audio-assets", type=Path)
    validate.add_argument("--labels", type=Path)
    validate.add_argument("--output", type=Path)

    extract = subparsers.add_parser("extract-features", help="Extract phone-level features.")
    extract.add_argument("--audio-assets", type=Path)
    extract.add_argument("--output", type=Path)

    synth = subparsers.add_parser("generate-synthetic-data", help="Generate synthetic WAV and CSV samples.")
    synth.add_argument("--output-dir", type=Path)
    synth.add_argument("--phone-count", type=int, default=24)
    synth.add_argument("--labeled-fraction", type=float, default=0.5)
    synth.add_argument("--seed", type=int, default=42)

    demo = subparsers.add_parser("run-demo-pipeline", help="Run the full synthetic MVP pipeline.")
    demo.add_argument("--output-dir", type=Path)
    demo.add_argument("--phone-count", type=int, default=24)
    demo.add_argument("--labeled-fraction", type=float, default=1.0)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--noise-types", nargs="+")

    prepare = subparsers.add_parser("prepare-factor-data", help="Merge features, dimensions, and labels.")
    prepare.add_argument("--phone-features", type=Path)
    prepare.add_argument("--dimensions", type=Path)
    prepare.add_argument("--labels", type=Path)
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--noise-type", type=str)

    train = subparsers.add_parser("train-noise-scorer", help="Train one-vs-rest noise scorer.")
    train.add_argument("--phone-features", type=Path)
    train.add_argument("--labels", type=Path)
    train.add_argument("--output", type=Path)

    score = subparsers.add_parser("score-noise-types", help="Score phones with a trained model.")
    score.add_argument("--phone-features", type=Path)
    score.add_argument("--model", type=Path)
    score.add_argument("--output", type=Path)

    evaluate = subparsers.add_parser("evaluate-scores", help="Evaluate scored phones against labels.")
    evaluate.add_argument("--scores", type=Path)
    evaluate.add_argument("--labels", type=Path)
    evaluate.add_argument("--output", type=Path)

    report = subparsers.add_parser("build-noise-report", help="Build a per-noise-type factor report.")
    report.add_argument("--scores", type=Path)
    report.add_argument("--dimensions", type=Path)
    report.add_argument("--labels", type=Path)
    report.add_argument("--output", type=Path)
    report.add_argument("--noise-type", type=str)
    report.add_argument("--factors", nargs="+")
    report.add_argument("--model", type=Path)
    report.add_argument("--phone-features", type=Path)

    html = subparsers.add_parser("render-noise-report-html", help="Render a local HTML report with charts.")
    html.add_argument("--report-json", type=Path)
    html.add_argument("--scores", type=Path)
    html.add_argument("--phone-features", type=Path)
    html.add_argument("--model", type=Path)
    html.add_argument("--output-dir", type=Path)
    html.add_argument("--highlight-feature", type=str)
    html.add_argument("--highlight-factor", type=str)

    analyze = subparsers.add_parser("analyze-factors", help="Rank dimensional factors.")
    analyze.add_argument("--input", type=Path)
    analyze.add_argument("--output", type=Path)
    analyze.add_argument("--target", type=str)
    analyze.add_argument("--factors", nargs="+")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "validate-joins":
        run_validate_joins(args.phone_master, args.audio_assets, args.labels, args.output)
        return 0
    if args.command == "extract-features":
        run_extract_features(args.audio_assets, args.output)
        return 0
    if args.command == "generate-synthetic-data":
        generate_synthetic_dataset(
            output_dir=args.output_dir,
            phone_count=args.phone_count,
            labeled_fraction=args.labeled_fraction,
            seed=args.seed,
        )
        return 0
    if args.command == "run-demo-pipeline":
        run_demo_pipeline(
            output_dir=args.output_dir,
            phone_count=args.phone_count,
            labeled_fraction=args.labeled_fraction,
            seed=args.seed,
            noise_types=args.noise_types,
        )
        return 0
    if args.command == "prepare-factor-data":
        run_prepare_factor_data(
            phone_features_csv=args.phone_features,
            dimension_csv=args.dimensions,
            labels_csv=args.labels,
            output_csv=args.output,
            noise_type=args.noise_type,
        )
        return 0
    if args.command == "train-noise-scorer":
        run_train_noise_scorer(args.phone_features, args.labels, args.output)
        return 0
    if args.command == "score-noise-types":
        run_score_noise_types(args.phone_features, args.model, args.output)
        return 0
    if args.command == "evaluate-scores":
        run_evaluate_scores(args.scores, args.labels, args.output)
        return 0
    if args.command == "build-noise-report":
        run_build_noise_report(
            scores_csv=args.scores,
            dimension_csv=args.dimensions,
            labels_csv=args.labels,
            output_json=args.output,
            noise_type=args.noise_type,
            factor_keys=args.factors,
            model_path=args.model,
            phone_features_csv=args.phone_features,
        )
        return 0
    if args.command == "render-noise-report-html":
        run_render_noise_report_html(
            report_json=args.report_json,
            scores_csv=args.scores,
            phone_features_csv=args.phone_features,
            model_path=args.model,
            output_dir=args.output_dir,
            highlight_feature=args.highlight_feature,
            highlight_factor=args.highlight_factor,
        )
        return 0
    if args.command == "analyze-factors":
        run_analyze_factors(args.input, args.output, args.target, args.factors)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
