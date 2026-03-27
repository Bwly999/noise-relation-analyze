# noise-relation-analyze

Offline pipeline for foldable-device abnormal-noise analysis. The repository is structured to join phone-level metadata, process four-direction audio, derive phone-level acoustic summaries, and rank dimensional factors associated with each noise type.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
pytest -q
python -m noise_relation_analyze.cli --help
```

## Layout

- `src/noise_relation_analyze/`: package source
- `tests/`: regression and contract tests
- `docs/superpowers/specs/`: approved design documents
- `configs/pipeline.template.toml`: starter config template

## Planned Pipeline Stages

1. Join validation for phone, audio, and label data.
2. Audio QC and bend-cycle segmentation.
3. Cycle-level and phone-level acoustic feature generation.
4. Noise-type scoring.
5. Dimensional factor impact analysis and reporting.

## Available Commands

Validate joined source tables:

```powershell
python -m noise_relation_analyze.cli validate-joins `
  --phone-master data/raw/phone_master.csv `
  --audio-assets data/raw/audio_asset.csv `
  --labels data/raw/labels.csv `
  --output artifacts/reports/join_report.json
```

Extract phone-level acoustic summaries from WAV assets:

```powershell
python -m noise_relation_analyze.cli extract-features `
  --audio-assets data/raw/audio_asset.csv `
  --output data/processed/phone_features.csv
```

Generate a synthetic dataset when real production data is unavailable:

```powershell
python -m noise_relation_analyze.cli generate-synthetic-data `
  --output-dir examples/synthetic_demo `
  --phone-count 24 `
  --labeled-fraction 0.75 `
  --seed 42
```

Run the full synthetic MVP pipeline in one command:

```powershell
python -m noise_relation_analyze.cli run-demo-pipeline `
  --output-dir examples/demo_pipeline `
  --phone-count 24 `
  --labeled-fraction 1.0 `
  --seed 42
```

Run the focused single-type version first:

```powershell
python -m noise_relation_analyze.cli run-demo-pipeline `
  --output-dir examples/type1_pipeline `
  --phone-count 24 `
  --labeled-fraction 1.0 `
  --seed 42 `
  --noise-types type_1 normal
```

Prepare a factor-analysis table by joining dimensions, features, and labels:

```powershell
python -m noise_relation_analyze.cli prepare-factor-data `
  --phone-features data/processed/phone_features.csv `
  --dimensions examples/synthetic_demo/dimensions.csv `
  --labels examples/synthetic_demo/labels.csv `
  --output data/processed/factor_input.csv `
  --noise-type type_1
```

Rank dimensional factors from an engineered factor table:

```powershell
python -m noise_relation_analyze.cli analyze-factors `
  --input data/processed/factor_input.csv `
  --output artifacts/reports/factor_ranking.csv `
  --target risk_score `
  --factors gap_a gap_b gap_c
```

Train and evaluate the one-vs-rest scorer directly:

```powershell
python -m noise_relation_analyze.cli train-noise-scorer `
  --phone-features examples/demo_pipeline/artifacts/phone_features.csv `
  --labels examples/demo_pipeline/labels.csv `
  --output examples/demo_pipeline/artifacts/noise_model.bin

python -m noise_relation_analyze.cli score-noise-types `
  --phone-features examples/demo_pipeline/artifacts/phone_features.csv `
  --model examples/demo_pipeline/artifacts/noise_model.bin `
  --output examples/demo_pipeline/artifacts/scores.csv

python -m noise_relation_analyze.cli evaluate-scores `
  --scores examples/demo_pipeline/artifacts/scores.csv `
  --labels examples/demo_pipeline/labels.csv `
  --output examples/demo_pipeline/artifacts/metrics.json
```

Build a per-noise-type report from scores and dimensions:

```powershell
python -m noise_relation_analyze.cli build-noise-report `
  --scores examples/demo_pipeline/artifacts/scores.csv `
  --dimensions examples/demo_pipeline/dimensions.csv `
  --labels examples/demo_pipeline/labels.csv `
  --output examples/demo_pipeline/artifacts/reports/type_1_report.json `
  --noise-type type_1 `
  --factors hinge_gap left_support_gap right_support_gap torsion_delta `
  --model examples/demo_pipeline/artifacts/noise_model.bin `
  --phone-features examples/demo_pipeline/artifacts/phone_features.csv
```

Render a local HTML report with SHAP plots and factor curves:

```powershell
python -m noise_relation_analyze.cli render-noise-report-html `
  --report-json examples/type1_pipeline/artifacts/reports/type_1_report.json `
  --scores examples/type1_pipeline/artifacts/scores.csv `
  --phone-features examples/type1_pipeline/artifacts/phone_features.csv `
  --model examples/type1_pipeline/artifacts/noise_model.bin `
  --output-dir examples/type1_pipeline/artifacts/html/type_1 `
  --highlight-feature dir_1_share `
  --highlight-factor hinge_gap
```

## Current Status

The repository currently provides:

- join validation against phone, audio, and label tables
- WAV loading for mono 16-bit PCM files
- RMS-energy based bend-cycle segmentation
- phone-level acoustic summary aggregation from directional WAV assets
- synthetic dataset generation for `type_1`, `type_2`, `type_3`, and `normal`
- factor-input preparation for one-vs-rest analysis using synthetic or labeled data
- richer phone-level acoustic features including crest factor, ZCR, spectral centroid, and high-band ratio
- one-vs-rest noise scoring with `RandomForestClassifier`
- cross-validated training metrics for model reliability
- score evaluation with overall accuracy and macro F1
- per-noise-type factor report generation with `spearman`, bin trends, bootstrap stability, SHAP-style factor summaries, and true Tree SHAP acoustic-feature summaries
- basic factor ranking by signed correlation strength

The current synthetic generator intentionally encodes dimension-to-noise relationships:

- `type_1`: larger `hinge_gap` and `left_support_gap`, dominant energy on `dir_1`
- `type_2`: larger `torsion_delta` and `right_support_gap`, dominant energy on `dir_3`
- `type_3`: larger `panel_flushness` and lower `adhesive_thickness`, dominant energy on `dir_2`

The current recommended entrypoint for focused development is:

```powershell
python -m noise_relation_analyze.cli run-demo-pipeline `
  --output-dir examples/type1_pipeline `
  --phone-count 24 `
  --labeled-fraction 1.0 `
  --seed 42 `
  --noise-types type_1 normal
```

This produces:

- `join_report.json`
- `phone_features.csv`
- `noise_model.bin` as a serialized trained model artifact
- `scores.csv`
- `metrics.json`
- `reports/type_1_report.json`
- `html/type_1/report.html`

The next upgrade path is:

- add transient-count and wavelet-style descriptors beyond the current spectral features
- replace the current random forest with a more robust tuned ensemble and holdout evaluation
- add confidence filtering and pseudo-label expansion for unlabeled phones
- upgrade report generation from signed-correlation ranking to multi-model evidence fusion
