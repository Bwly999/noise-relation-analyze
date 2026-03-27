# Foldable Device Noise-Cause Analysis Design

## Summary

This design defines a repeatable offline analysis pipeline for foldable-device abnormal noise. The goal is not only to classify noise types, but to identify which dimensional factors are most associated with each noise type, with usable directionality, threshold ranges, and confidence for design optimization and spec control.

The pipeline assumes:

- Each phone can be joined by a stable `phone_id`.
- Each phone has four directional audio files.
- A small subset of phones has one phone-level noise type label.
- Most phones are unlabeled.
- Dimensional and product metadata can be joined to the same `phone_id`.

The recommended V1 is a semi-supervised, multimodal, explainable workflow:

1. Clean and segment raw audio into bend cycles.
2. Build phone-level acoustic features and embeddings from four directions.
3. Train phone-level noise-type scorers with limited labeled data.
4. Use hard labels plus risk scores to analyze dimensional-factor impact.
5. Output engineering-facing factor cards per noise type.

## Target Outputs

For each noise type `type_k`, the system must produce:

- Top dimensional factors associated with the noise type.
- Effect direction for each factor.
- Sensitive value range or change-point candidates.
- Stability rating across model, batch, and sampling variations.
- Recommended action category: design optimization, tighter spec control, or validation experiment.

Secondary outputs:

- Phone-level noise type probabilities.
- Phone-level severity index.
- Factor-to-noise-type matrix.
- Noise-type-to-structure-module matrix.

## Canonical Data Model

The analysis unit is `phone_id`. All downstream aggregation must roll up to phone level even if features are extracted at cycle level or direction level.

### Table 1: `phone_master`

- `phone_id`
- `model_id`
- `batch_id`
- `vendor_id`
- `structure_version`
- `material_version`
- `test_date`
- `test_station_id`
- `has_label`
- `noise_type_label`
- `label_source`
- `label_quality`

### Table 2: `audio_asset`

- `phone_id`
- `direction_id` with fixed values such as `dir_1` to `dir_4`
- `wav_path`
- `sample_rate`
- `duration_sec`
- `bend_count_expected`
- `acquisition_status`

### Table 3: `audio_cycle`

- `phone_id`
- `direction_id`
- `cycle_id`
- `start_sec`
- `end_sec`
- `cycle_quality_score`
- `segmentation_method`

### Table 4: `audio_feature_cycle`

- `phone_id`
- `direction_id`
- `cycle_id`
- Handcrafted acoustic features such as RMS, crest factor, kurtosis, spectral centroid, spectral bandwidth, rolloff, zero crossing rate, impulse count, transient energy ratio, band energy ratios.
- Embedding vector columns or a reference to a serialized embedding artifact.

### Table 5: `audio_feature_phone`

- `phone_id`
- Per-direction aggregated features.
- Cross-direction asymmetry features.
- Cross-direction dispersion features.
- Phone-level embedding.
- Severity components and final severity index.

### Table 6: `dimension_feature`

- `phone_id`
- Raw dimensional features.
- Feature metadata: unit, module, nominal direction, upstream source.
- Preprocessed variants where needed, such as standardized value, winsorized value, and missing flag.

### Table 7: `analysis_target`

- `phone_id`
- `noise_type`
- `y_hard`
- `y_score`
- `y_severity`
- `confidence_tier`

## Pipeline Design

### Stage 1: Data Ingestion and Join Validation

Requirements:

- Validate one-to-one phone join between audio, dimension, and label sources.
- Reject duplicate `phone_id` rows unless explicit resolution rules exist.
- Produce a join report with missing-rate and conflict-rate statistics.

Rules:

- A phone with missing one or more directional audio files is retained but flagged.
- A phone with conflicting labels is excluded from supervised training and retained for unsupervised exploration only.
- Dimensional columns with severe missingness remain in raw storage but are downgraded during modeling.

Deliverables:

- Joined phone registry.
- Data quality report.

### Stage 2: Audio Quality Control and Segmentation

Requirements:

- Normalize sampling rate and amplitude representation.
- Run audio QC checks: silence, clipping, truncated file, abnormal duration, low signal-to-noise proxy.
- Segment each direction into bend cycles.

Preferred segmentation order:

1. Use synchronized bend-angle or device event signals if available.
2. Else detect periodic structure from envelope, short-time energy, or spectral flux.

Outputs:

- Cycle boundaries with quality scores.
- Phone-level exclusion flags for severe segmentation failure.

Acceptance rule:

- A phone may remain usable if at least two directional channels have acceptable cycle segmentation.

### Stage 3: Acoustic Representation

Features per cycle:

- Time-domain statistics.
- Frequency-domain statistics.
- Time-frequency representations such as log-mel or MFCC summaries.
- Impulse and transient descriptors that better capture abnormal click or crack signatures.

Aggregation:

- Aggregate cycle features to direction level with robust statistics such as median, quantiles, and worst-cycle score.
- Aggregate direction level to phone level while preserving directional asymmetry rather than simple averaging.

Required phone-level constructs:

- `severity_index`: weighted composite of intensity, impulsiveness, repeatability, and directional consistency.
- `embedding_phone`: learned or pretrained audio embedding aggregated across directions.
- Direction dominance features, asymmetry features, and dispersion features.

### Stage 4: Semi-Supervised Noise-Type Scoring

Modeling objective:

- For each noise type `type_k`, build a one-vs-rest phone-level scorer.

Inputs:

- `audio_feature_phone`
- Optional metadata that does not leak target definitions

Model strategy:

- V1: pretrained or self-supervised audio embedding plus a tabular classifier such as LightGBM or logistic/elastic-net baseline.
- Train on labeled phones only for the initial model.
- Score all phones.
- Promote only high-confidence unlabeled phones to pseudo-labeled training augmentation.

Rules:

- Pseudo-labeled phones must not dominate the supervised core set.
- Final factor conclusions use labeled data as primary evidence and scored data as secondary evidence.

Outputs:

- `P(type_k)` for each phone and each noise type.
- Confidence tier per phone.

### Stage 5: Dimensional Factor Impact Analysis

This stage is the main business output. Each `type_k` is analyzed separately.

Targets:

- Binary target: `y_hard(type_k)`
- Continuous targets: `y_score(type_k)` and `y_severity(type_k)`

#### 5.1 Feature Conditioning

- Preserve raw dimensional features.
- Create standardized variants for model comparability.
- Add missingness indicators where missingness may be informative.
- Cluster highly collinear dimensional factors into correlated groups.
- Tag each factor with a structure module for engineering interpretation.

#### 5.2 Evidence Layer A: Single-Factor Screening

Methods:

- Spearman rank association for continuous targets.
- Distribution shift tests and effect size for binary targets.
- Quantile or bin trend plots for threshold discovery.
- Bootstrap resampling for direction stability.

Purpose:

- Fast candidate generation.
- Early rejection of noise variables with no stable trend.

#### 5.3 Evidence Layer B: Multi-Factor Models

Run two model families in parallel:

- Sparse linear model: elastic net logistic or elastic net regression.
- Nonlinear tree model: LightGBM or XGBoost.

When product hierarchy effects are strong, add:

- Mixed-effect or group-aware variant with batch or model controls.

Control variables:

- `model_id`
- `batch_id`
- `vendor_id`
- `structure_version`
- `material_version`
- Any test-condition variables that materially shift measurements

Selection rule:

- A factor is core only if it remains important after control variables are included.

#### 5.4 Evidence Layer C: Explainability and Threshold Extraction

Methods:

- SHAP for global and local importance.
- ALE preferred over PDP when feature correlation is strong.
- Pairwise interaction scan for top candidate factor pairs.
- Bootstrap SHAP ranking stability.

Required conclusions per factor:

- Importance level.
- Effect direction.
- Sensitive range or turning point candidate.
- Whether the factor acts alone or via interaction.

### Stage 6: Stability and Anti-Spurious-Correlation Checks

Mandatory checks before a factor becomes an engineering recommendation:

- Batch-stratified re-evaluation.
- Model-stratified re-evaluation.
- Train-sample bootstrap stability.
- Labeled-only versus labeled-plus-high-confidence comparison.
- Correlated-group substitution test to detect proxy-factor artifacts.

A factor is recommendation-ready only if:

- Direction is consistent across checks.
- Importance remains within the top candidate tier across checks.
- The result is not explained away by one obvious confounder.

### Stage 7: Reporting

Primary artifact per noise type: engineering decision card.

Each card must include:

- Noise type name.
- Sample coverage.
- Top factors.
- Effect direction.
- Sensitive range.
- Stability rating.
- Controlled confounders.
- Recommended action category.
- Follow-up validation suggestion if evidence is medium or low confidence.

Portfolio-level artifacts:

- Factor-to-noise-type matrix.
- Noise-type-to-module matrix.
- Risk-distribution dashboard table for filtering by model, batch, vendor, and version.

## Validation and Metrics

The system is not judged by classification accuracy alone.

### Audio/Scoring Metrics

- Cross-validated AUC or macro F1 on labeled phones for each one-vs-rest scorer.
- Calibration quality of `P(type_k)`.
- Stability of risk-score distribution across batches.

### Factor Analysis Metrics

- Top-factor overlap across bootstrap runs.
- Direction consistency rate.
- Sensitivity-range reproducibility.
- Agreement between linear and nonlinear evidence layers.

### Business Acceptance

The V1 is considered successful if:

- Main noise types can be separated at a usable level on labeled data.
- At least one stable factor set per major noise type is identified.
- Engineering can read the output and know what to tighten, what to redesign, and what to validate next.

## Recommended V1 Stack

- Audio features: log-mel summaries, transient descriptors, robust cycle statistics.
- Audio embeddings: pretrained audio embedding model or lightweight self-supervised encoder.
- Noise scoring: LightGBM plus elastic-net baseline.
- Factor analysis: Spearman plus elastic net plus LightGBM plus SHAP plus ALE.
- Stability: bootstrap and group-stratified validation.

This V1 intentionally favors explainability and stability over maximum end-to-end deep-model complexity.

## V1 Non-Goals

- Full root-cause causality proof.
- Online real-time inference on the production line.
- Automated redesign recommendation generation.
- Fine-grained per-cycle ground-truth labeling.

## Implementation Sequence

1. Build canonical joined tables and QC reports.
2. Implement audio QC and cycle segmentation.
3. Generate direction-level and phone-level acoustic features.
4. Train one-vs-rest noise scorers and score unlabeled phones.
5. Run per-noise-type factor analysis with confounder controls.
6. Export engineering decision cards and summary matrices.

## Explicit Defaults

- Analysis unit defaults to `phone_id`.
- Primary scenario split defaults to noise type.
- Primary business output defaults to factor ranking plus direction plus sensitive range.
- Pseudo labels are secondary evidence, not primary truth.
- Phone-level analysis has priority over direction-level classification.
