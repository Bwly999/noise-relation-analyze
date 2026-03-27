# Noise Relation Analyze Status And Plan

## Current Status

The project has reached a usable MVP for `type_1 vs normal`, while preserving the ability to expand back to multi-class analysis later.

### 1. Project Foundation

- Git repository initialized locally.
- Python package structure, CLI entrypoints, tests, examples, and docs are in place.
- Main package root: `src/noise_relation_analyze/`

### 2. Design And Scope

- The original end-to-end design has been written and retained in:
  - `docs/superpowers/specs/2026-03-28-foldable-noise-analysis-design.md`
- The current engineering focus has been intentionally narrowed to:
  - one noise type first
  - `type_1` versus `normal`
  - reliable risk scoring and reliable factor analysis before re-expanding to multi-type

### 3. Synthetic Data Capability

- A synthetic dataset generator has been built.
- It can generate:
  - `phone_master.csv`
  - `audio_asset.csv`
  - `labels.csv`
  - `dimensions.csv`
  - 4-direction WAV files per phone
- It supports both:
  - focused binary mode such as `type_1 + normal`
  - broader multi-type mode
- Main implementation:
  - `src/noise_relation_analyze/synthetic_data.py`

### 4. Audio Processing

- WAV loading is implemented for mono 16-bit PCM files.
- RMS energy tracing is implemented.
- Bend-cycle segmentation is implemented with an energy-threshold method.
- This is enough to support repeatable local development and synthetic-data validation.
- Main implementation:
  - `src/noise_relation_analyze/audio.py`

### 5. Acoustic Feature Extraction

- The feature layer now outputs richer phone-level descriptors, not only severity and direction.
- Current phone-level outputs include:
  - `severity_index`
  - `asymmetry_score`
  - directional scores and directional shares
  - `rms_mean`
  - `impulse_score_mean`
  - `crest_factor_mean`
  - `zero_crossing_rate_mean`
  - `spectral_centroid_mean`
  - `high_band_ratio_mean`
- Main implementation:
  - `src/noise_relation_analyze/features.py`

### 6. Noise Scoring Model

- The original lightweight centroid scorer has been replaced.
- The current scorer uses `RandomForestClassifier`.
- Training now returns cross-validated reliability metrics:
  - `cv_accuracy`
  - `cv_macro_f1`
- Scoring outputs calibrated class probabilities from the trained estimator.
- Serialized model artifact is stored as `.bin`.
- Main implementation:
  - `src/noise_relation_analyze/scoring.py`

### 7. Factor Analysis

- The factor-analysis layer is no longer plain Pearson-only ranking.
- Current factor analysis includes:
  - Pearson correlation strength
  - Spearman correlation strength
  - binned trend means
  - bootstrap sign stability
  - SHAP-style global factor contribution summary
  - sample-level linear contribution explanations
- Main implementation:
  - `src/noise_relation_analyze/factor_analysis.py`

### 8. True Model Explainability

- The reporting layer now supports real `Tree SHAP` for the trained random-forest model.
- The report includes:
  - factor-level dimension analysis
  - model summary with CV metrics
  - acoustic feature SHAP summary
  - per-sample acoustic feature contribution explanations
- Main implementation:
  - `src/noise_relation_analyze/pipeline.py`

### 9. End-To-End Pipeline

- The pipeline currently supports:
  - join validation
  - synthetic data generation
  - feature extraction
  - factor-input preparation
  - model training
  - model scoring
  - score evaluation
  - noise report generation
  - one-command demo pipeline execution
- Main implementation:
  - `src/noise_relation_analyze/pipeline.py`
  - `src/noise_relation_analyze/cli.py`

### 10. Current Example Outputs

- Focused binary demo output:
  - `examples/type1_pipeline/`
- Current key artifacts:
  - `examples/type1_pipeline/artifacts/metrics.json`
  - `examples/type1_pipeline/artifacts/noise_model.bin`
  - `examples/type1_pipeline/artifacts/reports/type_1_report.json`

### 11. Verification Status

- Full test suite currently passes.
- Latest verified result:
  - `23 passed`

## What The Current MVP Can Reliably Do

The current repository can already support the following development loop without real factory data:

1. Generate a controlled binary dataset for `type_1` and `normal`.
2. Generate 4-direction synthetic WAV files aligned to dimensional data.
3. Extract phone-level acoustic descriptors from WAV input.
4. Train a classifier and quantify cross-validated performance.
5. Score all phones with predicted probabilities.
6. Rank structural dimensions against the target noise risk.
7. Explain both:
   - which dimensions are associated with the noise type
   - which acoustic features drive the trained classifier

## Limits Of The Current Version

The current version is reliable enough for internal development, but it is still not the final production-grade analysis system.

Current limits:

- Audio segmentation is still threshold-based and not angle-signal-aware.
- Acoustic features are stronger than before, but still not exhaustive.
- The model is stronger than the original baseline, but not yet tuned for domain robustness.
- The factor-analysis evidence stack is improved, but still not a full multi-model causal workflow.
- No real production data ingestion or schema adapter has been implemented yet.
- No batch-level drift validation or leave-batch-out validation has been implemented yet.

## Battle Plan

The next phase should be executed in this order.

### Phase 1. Real Data Adapter

Goal:

- Connect the actual production-line files to the current pipeline without changing the analysis core.

Work:

- define the exact schema mapping from real factory tables to:
  - phone master
  - audio asset registry
  - labels
  - dimensions
- build import adapters and validation checks
- support missing direction files and partial labels safely

Success condition:

- real input files can be converted into the current canonical tables and pass join validation

### Phase 2. Stronger Audio Features

Goal:

- Improve feature fidelity so the classifier is less dependent on simple direction-energy patterns.

Work:

- add transient count and burst density features
- add band-energy features
- add MFCC or log-mel summary features
- add cycle-to-cycle consistency features
- add optional denoising and normalization controls

Success condition:

- cross-validated performance remains strong under more realistic data variation

### Phase 3. Stronger Modeling

Goal:

- Improve reliability and reduce overconfidence before using real engineering conclusions.

Work:

- benchmark random forest versus boosted tree models
- add proper holdout evaluation
- add batch-aware or group-aware validation when real metadata is available
- add probability calibration if needed
- define acceptance thresholds for deployment-quality model behavior

Success condition:

- the chosen model performs well under grouped validation, not only random CV

### Phase 4. Stronger Factor Analysis

Goal:

- Make dimension conclusions more defensible and harder to break under resampling or scenario shifts.

Work:

- move from single-model ranking to multi-evidence fusion
- add univariate screening and multivariate modeling
- add threshold candidate extraction
- add factor-cluster handling for collinearity
- add confidence tiers for factor recommendations

Success condition:

- each reported factor has stable evidence across resampling and modeling choices

### Phase 5. Semi-Supervised Extension

Goal:

- start exploiting the unlabeled majority safely.

Work:

- add confidence-gated pseudo labels
- separate primary evidence from pseudo-labeled secondary evidence
- compare labeled-only versus labeled-plus-pseudo performance

Success condition:

- unlabeled data improves recall or ranking stability without degrading trustworthiness

### Phase 6. Multi-Type Expansion

Goal:

- expand from `type_1 vs normal` back to multiple abnormal-noise types.

Work:

- add one-vs-rest training for each major noise type
- generate per-type reports independently
- ensure factor conclusions do not cross-contaminate across noise types

Success condition:

- each type has its own reliable scoring and factor report, not one overloaded shared model

## Recommended Immediate Next Step

The immediate next battle should be:

1. connect real data into the canonical tables
2. keep `type_1 vs normal` as the first production-grade target
3. validate the current model and report behavior on real samples before any multi-type expansion

That path gives the fastest route from current synthetic MVP to a trustworthy engineering analysis tool.
