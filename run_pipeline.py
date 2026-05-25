"""
run_pipeline.py
===============
Fairness Pipeline Orchestrator — FairML Development Toolkit (Part 5).

Executes a three-step fair ML workflow defined in config.yml:

  Step 1 — Baseline fairness audit on raw data         (MeasurementModule)
  Step 2 — Data transformation + fair model training   (PipelineModule + TrainingModule)
  Step 3 — Final validation and report card            (MeasurementModule)

All key metrics and artifacts are logged to MLflow.

Usage
-----
    python run_pipeline.py
    python run_pipeline.py --config path/to/config.yml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from mlflow import sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Module imports — adjust paths to match your repository structure
# ---------------------------------------------------------------------------
from measurements.analyzer import FairnessAnalyzer
from pipeline.fair_transformers import (
    CorrelationSuppressor,
    DisparateImpactRemover,
    InstanceReweighting,
)
from training.reductions_wrapper import ReductionsWrapper
from fairlearn.reductions import BoundedGroupLoss, DemographicParity, EqualizedOdds

# ---------------------------------------------------------------------------
# Registries
# Extend these dicts to expose additional classes without changing the logic.
# ---------------------------------------------------------------------------

TRANSFORMER_REGISTRY = {
    "DisparateImpactRemover": DisparateImpactRemover,
    "CorrelationSuppressor":  CorrelationSuppressor,
    "InstanceReweighting":    InstanceReweighting,
}

ESTIMATOR_REGISTRY = {
    "LogisticRegression": LogisticRegression,
}

CONSTRAINT_REGISTRY = {
    "DemographicParity":  DemographicParity,
    "EqualizedOdds":      EqualizedOdds,
    "BoundedGroupLoss":   BoundedGroupLoss,
}


# ===========================================================================
# Helpers
# ===========================================================================

def load_config(path: str = "config.yml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _divider(title: str = "") -> None:
    line = "=" * 60
    print(f"\n{line}")
    if title:
        print(title)
        print(line)


def load_data(cfg: dict) -> pd.DataFrame:
    """Load CSV and apply raw preprocessing defined in config."""
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["path"])
    df.columns = df.columns.str.strip()

    # Drop unwanted columns
    drop_cols = [c for c in data_cfg.get("drop_cols", []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # One-hot encoding
    for enc in cfg.get("preprocessing", {}).get("encode", []):
        col = enc["column"]
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    # Binning
    for b in cfg.get("preprocessing", {}).get("binning", []):
        col = b["column"]
        if col in df.columns:
            bins = [float("inf") if str(x) == ".inf" else float(x) for x in b["bins"]]
            df[b["new_column"]] = pd.cut(df[col], bins=bins, labels=b["labels"])

    print(f"✓ Data loaded: {len(df):,} rows × {len(df.columns)} columns.")
    return df


# ===========================================================================
# Step 1 — Baseline measurement
# ===========================================================================

def step1_baseline(df: pd.DataFrame, cfg: dict) -> dict:
    _divider("STEP 1 — BASELINE FAIRNESS AUDIT")

    data_cfg = cfg["data"]
    val_cfg  = cfg["validation"]
    primary  = val_cfg["primary_metric"]

    analyzer = FairnessAnalyzer(
        df,
        target_col     = data_cfg["target_col"],
        sensitive_col  = data_cfg["sensitive_col"],
        positive_label = data_cfg.get("positive_label", 1),
    )

    # y_pred=None → FairnessAnalyzer uses the target column itself.
    # This measures label-level disparity in the raw data (the starting point).
    results = analyzer.calculate_classification_metrics(
        y_pred       = None,
        n_bootstrap  = val_cfg.get("n_bootstrap", 200),
        random_state = 42,
    )

    if primary in results:
        r = results[primary]
        print(f"\n  {r.metric_name}")
        print(f"  Value       : {r.value:.4f}")
        print(f"  95% CI      : ({r.confidence_interval[0]:.4f}, {r.confidence_interval[1]:.4f})")
        print(f"  Effect size : {r.effect_size}")
        print(f"  Groups (n)  : {r.sample_sizes}")

    return results


# ===========================================================================
# Step 2 — Transform data and train fair model
# ===========================================================================

def step2_transform_and_train(df: pd.DataFrame, cfg: dict):
    """
    Returns
    -------
    model          : fitted ReductionsWrapper
    X_test_tr      : transformed test features (sensitive col removed)
    y_test         : test labels
    s_test         : test sensitive attribute values
    """
    _divider("STEP 2 — DATA TRANSFORMATION & FAIR TRAINING")

    data_cfg     = cfg["data"]
    target_col   = data_cfg["target_col"]
    sensitive_col = data_cfg["sensitive_col"]
    val_cfg      = cfg["validation"]

    # Feature matrix: all numeric columns except the target
    feature_cols = [
        c for c in df.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    X         = df[feature_cols].fillna(0)
    y         = df[target_col]
    sensitive = df[sensitive_col]

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive,
        test_size    = val_cfg.get("test_size", 0.2),
        random_state = 42,
        stratify     = y,
    )

    # ── 2a: Apply transformer ──────────────────────────────────────────────
    t_cfg  = cfg["transformer"]
    t_cls  = TRANSFORMER_REGISTRY[t_cfg["class"]]
    transformer = t_cls(**t_cfg.get("params", {}))

    print(f"\n  Applying {t_cfg['class']} (repair_level="
          f"{t_cfg.get('params', {}).get('repair_level', 'N/A')})...")

    # DisparateImpactRemover needs the sensitive column present in X
    train_with_s = X_train.copy()
    train_with_s[sensitive_col] = s_train.values

    test_with_s = X_test.copy()
    test_with_s[sensitive_col] = s_test.values

    X_train_tr = transformer.fit_transform(train_with_s)
    X_test_tr  = transformer.transform(test_with_s)

    # Remove sensitive col from feature matrices before training
    for frame in [X_train_tr, X_test_tr]:
        if sensitive_col in frame.columns:
            frame.drop(columns=[sensitive_col], inplace=True)

    print(f"  ✓ Transformation complete — feature shape: {X_train_tr.shape}")

    # ── 2b: Build and train fair model ────────────────────────────────────
    tr_cfg = cfg["training"]

    base_est = ESTIMATOR_REGISTRY[tr_cfg["base_estimator"]](
        **tr_cfg.get("base_estimator_params", {})
    )
    constraint = CONSTRAINT_REGISTRY[tr_cfg["constraint"]](
        **tr_cfg.get("constraint_params", {})
    )

    model = ReductionsWrapper(
        estimator = base_est,
        constraint = constraint,
        eps      = tr_cfg.get("eps", 0.01),
        max_iter = tr_cfg.get("max_iter", 50),
    )

    print(f"\n  Training {tr_cfg['method']} with {tr_cfg['constraint']} constraint...")
    model.fit(X_train_tr, y_train, sensitive_features=s_train)
    print(f"\n{model.summary()}")

    return model, X_test_tr, y_test.reset_index(drop=True), s_test.reset_index(drop=True)


# ===========================================================================
# Step 3 — Final validation and report card
# ===========================================================================

def step3_validate(
    model,
    X_test_tr: pd.DataFrame,
    y_test: pd.Series,
    s_test: pd.Series,
    cfg: dict,
    baseline_results: dict,
) -> tuple[dict, float]:
    _divider("STEP 3 — FINAL VALIDATION & REPORT CARD")

    data_cfg  = cfg["data"]
    val_cfg   = cfg["validation"]
    primary   = val_cfg["primary_metric"]
    threshold = val_cfg["threshold"]

    preds    = model.predict(X_test_tr)
    accuracy = float(accuracy_score(y_test, preds))
    print(f"\n  Accuracy on test set: {accuracy:.4f}")

    # Build evaluation DataFrame for FairnessAnalyzer
    eval_df = X_test_tr.copy().reset_index(drop=True)
    eval_df[data_cfg["target_col"]]   = y_test.values
    eval_df[data_cfg["sensitive_col"]] = s_test.values

    analyzer = FairnessAnalyzer(
        eval_df,
        target_col     = data_cfg["target_col"],
        sensitive_col  = data_cfg["sensitive_col"],
        positive_label = data_cfg.get("positive_label", 1),
    )

    final_results = analyzer.calculate_classification_metrics(
        y_pred       = pd.Series(preds),
        n_bootstrap  = val_cfg.get("n_bootstrap", 200),
        random_state = 42,
    )

    # ── Report card ────────────────────────────────────────────────────────
    print("\n  ┌──────────────────────────────────────────────────────────┐")
    print("  │                   FAIRNESS REPORT CARD                   │")
    print("  └──────────────────────────────────────────────────────────┘")

    if primary in baseline_results and primary in final_results:
        b = baseline_results[primary]
        f = final_results[primary]
        delta  = f.value - b.value
        passed = abs(f.value) <= threshold

        print(f"\n  Metric    : {primary}")
        print(f"  Baseline  : {b.value:.4f}  "
              f"CI ({b.confidence_interval[0]:.4f}, {b.confidence_interval[1]:.4f})")
        print(f"  Final     : {f.value:.4f}  "
              f"CI ({f.confidence_interval[0]:.4f}, {f.confidence_interval[1]:.4f})")
        print(f"  Change    : {delta:+.4f}")
        print(f"  Threshold : ≤ {threshold}")
        print(f"\n  Status    : {'✅ PASS' if passed else '❌ FAIL — metric exceeds threshold'}")
    else:
        print(f"  [!] Primary metric '{primary}' not found in results.")

    return final_results, accuracy


# ===========================================================================
# MLflow logging
# ===========================================================================

def log_to_mlflow(
    model,
    baseline_results: dict,
    final_results: dict,
    accuracy: float,
    cfg: dict,
    config_path: str,
) -> None:
    _divider("LOGGING TO MLFLOW")

    mlflow_cfg = cfg.get("mlflow", {})
    primary    = cfg["validation"]["primary_metric"]

    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "fairness_pipeline"))

    with mlflow.start_run(run_name=mlflow_cfg.get("run_name", "pipeline_run")):

        # ── Performance ────────────────────────────────────────────────────
        mlflow.log_metric("accuracy", accuracy)

        # ── Fairness metrics ───────────────────────────────────────────────
        if primary in baseline_results:
            mlflow.log_metric(f"baseline_{primary}", baseline_results[primary].value)

        if primary in final_results:
            r = final_results[primary]
            mlflow.log_metric(f"final_{primary}",          r.value)
            mlflow.log_metric(f"final_{primary}_ci_lower", r.confidence_interval[0])
            mlflow.log_metric(f"final_{primary}_ci_upper", r.confidence_interval[1])
            if r.effect_size is not None:
                mlflow.log_metric(f"final_{primary}_effect_size", r.effect_size)

        # ── Pipeline params ────────────────────────────────────────────────
        mlflow.log_param("transformer",        cfg["transformer"]["class"])
        mlflow.log_param("training_method",    cfg["training"]["method"])
        mlflow.log_param("constraint",         cfg["training"]["constraint"])
        mlflow.log_param("fairness_threshold", cfg["validation"]["threshold"])
        mlflow.log_param("sensitive_col",      cfg["data"]["sensitive_col"])

        # ── Artifacts ──────────────────────────────────────────────────────
        mlflow_sklearn.log_model(model, artifact_path="fair_model")
        mlflow.log_artifact(config_path, artifact_path="config")

    exp_name = mlflow_cfg.get("experiment_name", "fairness_pipeline")
    print(f"  ✓ Run logged to experiment '{exp_name}'.")
    print("  Run  mlflow ui  to explore metrics and artifacts.")


# ===========================================================================
# Entry point
# ===========================================================================

def main(config_path: str = "config.yml") -> None:
    cfg = load_config(config_path)

    df               = load_data(cfg)
    baseline         = step1_baseline(df, cfg)
    model, X_test_tr, y_test, s_test = step2_transform_and_train(df, cfg)
    final, accuracy  = step3_validate(model, X_test_tr, y_test, s_test, cfg, baseline)
    log_to_mlflow(model, baseline, final, accuracy, cfg, config_path)

    _divider()
    print("✓ Pipeline complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FairML Pipeline Orchestrator")
    parser.add_argument(
        "--config", default="config.yml",
        help="Path to the pipeline configuration file (default: config.yml)"
    )
    args = parser.parse_args()
    main(args.config)