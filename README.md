# FairnessAnalyzer

A unified interface for auditing fairness in machine learning models and datasets, wrapping **Fairlearn**, **AIF360**, and **Aequitas** into a single consistent API.

---

## Features

- **Unified interface** — one class, three libraries
- **Structured outputs** — every metric returns a `FairnessResult` with point estimate, 95% bootstrap CI, and effect size
- **Data utilities** — built-in binning for continuous variables and intersectional feature creation
- **Statistical rigor** — stratified bootstrap CIs, Benjamini-Hochberg FDR correction for intersectional audits
- **MLOps ready** — MLflow integration and `assert_fairness` helper for CI/CD pipelines
- **Visualization** — automated fairness reports with error bars

---

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib statsmodels
pip install fairlearn aif360 aequitas mlflow
```

---

## Quick Start

```python
from analyzer import FairnessAnalyzer

# 1. Load and configure
analyzer = FairnessAnalyzer(df="loan_approval_dataset.csv")
analyzer.set_config(
    target_col="Loan_Approval_Status",
    sensitive_col="Gender",
    positive_label=1
)

# 2. Bin continuous features
analyzer.bin_column(
    column_name="Age",
    bins=[0, 25, 35, 45, 55, 65, float('inf')],
    labels=['0-25', '26-35', '36-45', '46-55', '56-65', '65+'],
    new_column_name="Age_Group"
)

# 3. Classification metrics
results = analyzer.calculate_classification_metrics(engine="fairlearn")
print(results['demographic_parity_difference'])

# 4. Statistical audit
audit = analyzer.get_fairness_audit(n_bootstrap=200)
print(audit)

# 5. Intersectional analysis
inter_df = analyzer.intersectional_audit(
    column_names=["Gender", "Age_Group"],
    apply_fdr_correction=True
)

# 6. Aequitas disparity metrics
aequitas_df = analyzer.get_aequitas_metrics()

# 7. Visualize
analyzer.generate_report_visualizations(results, output_path="report.png")
```

---

## Core Components

### `FairnessResult`
Dataclass returned by every metric method.

| Field | Description |
|---|---|
| `metric_name` | Human-readable metric name |
| `value` | Point estimate |
| `confidence_interval` | `(lower, upper)` 95% bootstrap CI |
| `effect_size` | Risk ratio or similar effect measure |
| `sample_sizes` | Per-group sample counts |
| `metadata` | Library, excluded groups, per-group rates |

### `FairnessAnalyzer`
Main entry point.

| Method | Description |
|---|---|
| `set_config()` | Set target, sensitive column, positive label |
| `bin_column()` | Discretize continuous variables |
| `calculate_classification_metrics()` | Demographic parity & equalized odds (Fairlearn / AIF360) |
| `calculate_regression_metrics()` | MAE difference across groups |
| `get_fairness_audit()` | Selection rate disparity with bootstrap CI |
| `intersectional_audit()` | Multi-attribute group analysis with FDR correction |
| `get_aequitas_metrics()` | Aequitas disparity table |
| `assert_fairness()` | Threshold-based assertion for pipelines |
| `generate_report_visualizations()` | Plot metrics with confidence intervals |

---

## CI/CD Integration

```python
# Raises AssertionError if effect_size falls below threshold
analyzer.assert_fairness(result, threshold=0.8, metric="effect_size")
```

Compatible with **Pytest** and any standard CI pipeline.

---

## MLflow Tracking

Metrics and parameters are logged automatically when MLflow is configured in your environment.

---

## Dataset

Tested on the [Loan Approval dataset](https://www.kaggle.com/) from Kaggle, using Gender, Age, Income, and Credit Score as sensitive and target attributes.

---

## Notes

- `y_pred=None` falls back to ground-truth labels — useful for auditing dataset bias, but metrics like Equalized Odds become trivially zero. A warning is always raised.
- Bootstrap CIs use **stratified resampling** by group for stability.
- Groups below `min_group_size` are excluded and reported in `metadata`.
