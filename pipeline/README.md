# Fair ML Pipeline

A modular, end-to-end system for auditing and mitigating bias in machine learning models — from raw dataset inspection to CI/CD-enforced fairness gates.

---

## Overview

Fair ML Pipeline combines two complementary layers:

- **FairnessAnalyzer** — a unified auditing interface wrapping Fairlearn, AIF360, and Aequitas into a single consistent API, with structured outputs, bootstrap confidence intervals, and MLflow integration.
- **FairPipelineBuilder** — a config-driven pipeline orchestrator that assembles sklearn-compatible preprocessing, bias mitigation transformers, and a classifier from a single YAML file.

Together they cover the full lifecycle: detect bias in the raw data, mitigate it through preprocessing, enforce thresholds in CI/CD, and log results for reproducibility.

---

## Repository Structure

├── analyzer.py                  # FairnessAnalyzer — unified audit interface
├── detection_engine.py          # BiasDetectionEngine — representation, disparity, proxy detection
├── fair_transformers.py         # sklearn-compatible bias mitigation transformers
├── pipeline_framework.py        # FairPipelineBuilder — config-driven orchestrator
├── config.yml                   # Central configuration file
├── requirements.txt
└── tests/
├── test_pipeline_fairness.py    # pytest suite (~40 tests, 7 classes)
└── .github/workflows/
└── fairness_check.yml           # GitHub Actions CI/CD workflow

---

## Components

### `analyzer.py` — FairnessAnalyzer

A unified interface for auditing fairness, wrapping **Fairlearn**, **AIF360**, and **Aequitas**.

Every metric method returns a `FairnessResult` dataclass with:

| Field | Description |
|---|---|
| `metric_name` | Human-readable metric name |
| `value` | Point estimate |
| `confidence_interval` | `(lower, upper)` 95% bootstrap CI |
| `effect_size` | Risk ratio or similar effect measure |
| `sample_sizes` | Per-group sample counts |
| `metadata` | Library, excluded groups, per-group rates |

Key methods:

| Method | Description |
|---|---|
| `set_config()` | Set target column, sensitive column, positive label |
| `bin_column()` | Discretize continuous variables |
| `calculate_classification_metrics()` | Demographic parity & equalized odds (Fairlearn / AIF360) |
| `calculate_regression_metrics()` | MAE difference across groups |
| `get_fairness_audit()` | Selection rate disparity with bootstrap CI |
| `intersectional_audit()` | Multi-attribute analysis with Benjamini-Hochberg FDR correction |
| `get_aequitas_metrics()` | Aequitas disparity table |
| `assert_fairness()` | Threshold-based assertion for CI/CD pipelines |
| `generate_report_visualizations()` | Plot metrics with confidence intervals |

---

### `detection_engine.py` — BiasDetectionEngine

Audits the raw dataset on three fronts before any model is trained:

- **Representation bias** — compares observed demographic distributions against configurable benchmarks (e.g. 50/50 for Gender).
- **Statistical disparity** — tests whether each non-sensitive feature distributes differently across groups (ANOVA/Kruskal for continuous, chi-square for categorical).
- **Proxy variable detection** — computes correlation between non-sensitive features and protected attributes (Cramér's V, point-biserial, eta).

Output is a `BiasReport` dataclass serializable to JSON, including a `summary` field with an overall `LOW / MEDIUM / HIGH` risk rating.

---

### `fair_transformers.py` — Bias Mitigation Transformers

Four sklearn-compatible transformers (`BaseEstimator` + `TransformerMixin`):

| Transformer | What it does |
|---|---|
| `InstanceReweighting` | Computes per-sample weights (Kamiran & Calders). Exposes `sample_weight_` for `model.fit()` — does not alter rows or columns. |
| `SMOTEResampler` | Group-aware oversampling: runs SMOTE separately within each sensitive group to avoid cross-group synthetic samples. Use with `fit_transform` outside the pipeline, as it changes the row count. |
| `DisparateImpactRemover` | Quantile-based repair (Feldman 2015): shifts each numeric feature's distribution within each group toward the overall marginal distribution. `repair_level` controls aggressiveness. |
| `CorrelationSuppressor` | Drops features whose correlation with the sensitive attribute exceeds a threshold. Automatically selects the correct correlation method based on data types. |

---

### `config.yml` — Central Configuration

Controls the entire pipeline without touching code:

```yaml
bias_detection:
  enabled: true
  thresholds: ...
  demographic_benchmarks: ...

preprocessing:
  age_binning: ...
  drop_columns: [...]

transformers:
  - name: InstanceReweighting
    enabled: true
    params: ...
  - name: CorrelationSuppressor
    enabled: false
    params: ...

classifier:
  name: LogisticRegression
  params: ...

fairness_gates:
  max_underrepresentation: 0.10
  max_proxy_correlation: 0.60
```

---

### `pipeline_framework.py` — FairPipelineBuilder

Reads `config.yml` and assembles a `sklearn.Pipeline` dynamically.

```python
from pipeline_framework import FairPipelineBuilder

# Load config
builder = FairPipelineBuilder.from_config("config.yml")
print(builder.summary())

# Run full pipeline: preprocess → bias detection → build → fit
result = builder.run(df)

# Use results
result.pipeline.predict(X_test)
result.bias_report.summary    # {'overall_risk': 'MEDIUM', ...}
result.sample_weight          # Per-sample weights if InstanceReweighting is active
```

| Method | Description |
|---|---|
| `from_config(path)` | Load YAML and return the builder |
| `run_bias_detection(df)` | Run the detection engine and save JSON report |
| `build(df)` | Assemble the sklearn Pipeline from config |
| `run(df)` | Full sequence: preprocess → detection → build → fit |

---

### `tests/test_pipeline_fairness.py` — Test Suite

~40 tests across 7 classes:

| Class | What it tests |
|---|---|
| `TestBiasDetectionEngine` | Report fields, flags, representation, proxy, disparity |
| `TestInstanceReweighting` | Shape, weight positivity, normalization, mean = 1 |
| `TestSMOTEResampler` | Output shape, column preservation, group preservation |
| `TestDisparateImpactRemover` | Gap reduced, `repair_level=0` is identity, feature subsets |
| `TestCorrelationSuppressor` | Proxy drop, threshold behavior, method selection |
| `TestSklearnPipelineIntegration` | Full pipeline fit / predict / predict_proba |
| `TestCICDThresholds` | **Build gates** — fail if fairness requirements are not met |

`TestCICDThresholds` is what matters for CI/CD: if the dataset has underrepresented groups beyond 10%, proxy correlations above 0.6, or reweighting that fails to balance sufficiently, the build fails with an explicit message.

---

### `.github/workflows/fairness_check.yml` — GitHub Actions

Triggers on every PR to `main` / `develop` and on push to `main`:

1. Installs dependencies from `requirements.txt`
2. Runs pytest with JUnit XML output
3. Uploads the report as an artifact (30-day retention)
4. Automatically comments on the PR if tests fail

---

## Quick Start

```python
from pipeline_framework import FairPipelineBuilder

builder = FairPipelineBuilder.from_config("config.yml")
result  = builder.run(df)

result.pipeline.predict(X_test)
print(result.bias_report.summary)
```

For standalone auditing:

```python
from analyzer import FairnessAnalyzer

analyzer = FairnessAnalyzer(df="loan_approval_dataset.csv")
analyzer.set_config(target_col="Loan_Approval_Status", sensitive_col="Gender", positive_label=1)

audit = analyzer.get_fairness_audit(n_bootstrap=200)
analyzer.assert_fairness(audit, threshold=0.8, metric="effect_size")
```

---

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib statsmodels
pip install fairlearn aif360 aequitas mlflow imbalanced-learn
```

---

## Notes

- `y_pred=None` in `FairnessAnalyzer` falls back to ground-truth labels — useful for dataset auditing, but metrics like Equalized Odds become trivially zero. A warning is always raised.
- Bootstrap CIs use stratified resampling by group for stability.
- Groups below `min_group_size` are excluded and reported in `metadata`.
- `SMOTEResampler` must be used with `fit_transform` outside the main sklearn pipeline, since it changes the number of rows.