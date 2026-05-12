"""
pipeline_framework.py
=====================
Configurable Fair Pipeline Framework.

Loads config.yml and assembles a scikit-learn Pipeline dynamically,
wiring together the BiasDetectionEngine and the fairness transformers.

Classes
-------
FairPipelineBuilder
    Main entry point.  Load from a YAML config file, then call build()
    to get a ready-to-use sklearn Pipeline.

FairPipelineResult
    Dataclass returned by FairPipelineBuilder.run(), bundling the fitted
    pipeline, bias report, and sample weights (if InstanceReweighting
    is enabled).

Usage
-----
    from pipeline_framework import FairPipelineBuilder

    builder = FairPipelineBuilder.from_config("config.yml")

    # Optional: run bias detection first
    report = builder.run_bias_detection(df)

    # Build the sklearn Pipeline
    pipeline = builder.build()

    # Fit (pass sample_weight automatically if reweighting is enabled)
    result = builder.run(df)
    result.pipeline.predict(X_test)
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from detection_engine import BiasDetectionEngine, BiasReport
from fair_transformers import (
    CorrelationSuppressor,
    DisparateImpactRemover,
    InstanceReweighting,
    SMOTEResampler,
)

# ---------------------------------------------------------------------------
# Transformer registry  – maps config class names → Python classes
# ---------------------------------------------------------------------------

_TRANSFORMER_REGISTRY: Dict[str, type] = {
    "DisparateImpactRemover": DisparateImpactRemover,
    "CorrelationSuppressor": CorrelationSuppressor,
    "InstanceReweighting": InstanceReweighting,
    "SMOTEResampler": SMOTEResampler,
    "StandardScaler": StandardScaler,
}

# Classifier registry – extend as needed
_CLASSIFIER_REGISTRY: Dict[str, str] = {
    "LogisticRegression": "sklearn.linear_model",
    "RandomForestClassifier": "sklearn.ensemble",
    "GradientBoostingClassifier": "sklearn.ensemble",
    "SVC": "sklearn.svm",
    "XGBClassifier": "xgboost",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FairPipelineResult:
    pipeline: Pipeline
    bias_report: Optional[BiasReport]
    sample_weight: Optional[np.ndarray]
    feature_cols: List[str]
    config: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class FairPipelineBuilder:
    """
    Builds a scikit-learn Pipeline from a YAML configuration file.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary (usually from a YAML file).

    Example
    -------
    >>> builder = FairPipelineBuilder.from_config("config.yml")
    >>> result  = builder.run(df)
    >>> result.pipeline.predict(X_test)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._bias_report: Optional[BiasReport] = None
        self._sample_weight: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, path: str | Path) -> "FairPipelineBuilder":
        """Load config from a YAML file and return a FairPipelineBuilder."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls(cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_bias_detection(
        self,
        df: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> BiasReport:
        """
        Run the BiasDetectionEngine using parameters from config.

        Parameters
        ----------
        df : pd.DataFrame
        output_path : str, optional
            Override the output path from config.

        Returns
        -------
        BiasReport
        """
        bd_cfg = self.config.get("bias_detection", {})
        if not bd_cfg.get("enabled", True):
            warnings.warn("Bias detection is disabled in config — skipping.")
            return None

        engine = BiasDetectionEngine(
            df=df,
            sensitive_cols=self.config["dataset"]["sensitive_cols"],
            benchmarks=bd_cfg.get("benchmarks"),
            p_threshold=bd_cfg.get("p_threshold", 0.05),
            proxy_threshold=bd_cfg.get("proxy_threshold", 0.30),
            representation_gap=bd_cfg.get("representation_gap", 0.05),
            output_path=output_path or bd_cfg.get("output_path", "bias_report.json"),
        )
        self._bias_report = engine.run()
        return self._bias_report

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply any preprocessing steps defined in config (e.g. binning).

        Returns a new DataFrame with the additional columns added.
        """
        cfg = self.config.get("preprocessing", {})
        out = df.copy()

        for bin_cfg in cfg.get("binning", []):
            col = bin_cfg["column"]
            bins = [float(b) for b in bin_cfg["bins"]]
            labels = bin_cfg["labels"]
            new_col = bin_cfg.get("new_column", f"{col}_bin")

            if col not in out.columns:
                warnings.warn(f"Binning: column '{col}' not found — skipped.")
                continue

            out[new_col] = pd.cut(
                out[col],
                bins=bins,
                labels=labels,
                right=True,
                include_lowest=True,
            )

        drop_cols = self.config.get("dataset", {}).get("drop_cols", [])
        out = out.drop(columns=[c for c in drop_cols if c in out.columns])

        return out

    def build(
        self,
        df: Optional[pd.DataFrame] = None,
        extra_steps: Optional[List[tuple]] = None,
    ) -> Pipeline:
        """
        Assemble and return the sklearn Pipeline from config.

        Parameters
        ----------
        df : pd.DataFrame, optional
            If provided, InstanceReweighting is fitted here so that
            sample_weight_ is available before pipeline.fit() is called.
        extra_steps : list of (name, transformer) tuples, optional
            Additional steps appended before the classifier.

        Returns
        -------
        sklearn.pipeline.Pipeline
        """
        steps: List[tuple] = []

        for t_cfg in self.config.get("transformers", []):
            if not t_cfg.get("enabled", True):
                continue

            cls_name = t_cfg["class"]
            params = t_cfg.get("params", {})

            if cls_name not in _TRANSFORMER_REGISTRY:
                raise ValueError(
                    f"Unknown transformer '{cls_name}'. "
                    f"Available: {list(_TRANSFORMER_REGISTRY.keys())}"
                )

            transformer_cls = _TRANSFORMER_REGISTRY[cls_name]

            # InstanceReweighting: fit now if df is available, so weights
            # are ready for pipeline.fit(sample_weight=builder.sample_weight_)
            if cls_name == "InstanceReweighting" and df is not None:
                t = transformer_cls(**params)
                t.fit(df)
                self._sample_weight = t.sample_weight_
                steps.append((t_cfg["name"], t))
            else:
                steps.append((t_cfg["name"], transformer_cls(**params)))

        # Extra custom steps
        if extra_steps:
            steps.extend(extra_steps)

        # Classifier
        clf = self._build_classifier()
        steps.append(("classifier", clf))

        return Pipeline(steps)

    def run(
        self,
        df: pd.DataFrame,
        run_detection: bool = True,
        extra_steps: Optional[List[tuple]] = None,
    ) -> FairPipelineResult:
        """
        Full end-to-end run:
          1. Preprocess (binning, drops)
          2. Bias detection (optional)
          3. Build pipeline
          4. Fit pipeline

        Parameters
        ----------
        df : pd.DataFrame
            Raw input DataFrame.
        run_detection : bool
            Whether to run the BiasDetectionEngine (default True).
        extra_steps : list, optional
            Additional sklearn steps to inject before the classifier.

        Returns
        -------
        FairPipelineResult
        """
        ds_cfg = self.config["dataset"]
        target_col = ds_cfg["target_col"]
        sensitive_cols = ds_cfg["sensitive_cols"]

        # 1. Preprocess
        df_processed = self.preprocess(df)

        # 2. Bias detection
        if run_detection:
            self.run_bias_detection(df_processed)

        # 3. Build pipeline (also fits InstanceReweighting if present)
        pipeline = self.build(df=df_processed, extra_steps=extra_steps)

        # 4. Determine feature columns
        exclude = set(sensitive_cols) | {target_col}
        feature_cols = [c for c in df_processed.columns if c not in exclude]
        X = df_processed[feature_cols + sensitive_cols]   # keep sensitive for transformers
        y = df_processed[target_col]

        # 5. Fit — pass sample_weight if InstanceReweighting was used
        fit_params = {}
        if self._sample_weight is not None:
            # sklearn Pipeline passes kwargs as step__param
            fit_params["classifier__sample_weight"] = self._sample_weight

        pipeline.fit(X, y, **fit_params)

        return FairPipelineResult(
            pipeline=pipeline,
            bias_report=self._bias_report,
            sample_weight=self._sample_weight,
            feature_cols=feature_cols,
            config=self.config,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sample_weight_(self) -> Optional[np.ndarray]:
        """Sample weights computed by InstanceReweighting (after build/run)."""
        return self._sample_weight

    @property
    def bias_report_(self) -> Optional[BiasReport]:
        """BiasReport from the last run_bias_detection call."""
        return self._bias_report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_classifier(self):
        clf_cfg = self.config.get("classifier", {})
        cls_name = clf_cfg.get("class", "LogisticRegression")
        params = clf_cfg.get("params", {})

        if cls_name in _CLASSIFIER_REGISTRY:
            module = importlib.import_module(_CLASSIFIER_REGISTRY[cls_name])
            cls = getattr(module, cls_name)
        else:
            raise ValueError(
                f"Unknown classifier '{cls_name}'. "
                f"Available: {list(_CLASSIFIER_REGISTRY.keys())}"
            )

        return cls(**params)

    def summary(self) -> str:
        """Return a human-readable summary of the configured pipeline."""
        lines = ["=" * 60, "FAIR PIPELINE CONFIGURATION SUMMARY", "=" * 60]

        ds = self.config.get("dataset", {})
        lines += [
            f"Dataset       : {ds.get('path', 'N/A')}",
            f"Target        : {ds.get('target_col')}",
            f"Sensitive cols: {ds.get('sensitive_cols')}",
            "",
            "Transformers (in order):",
        ]

        for t in self.config.get("transformers", []):
            status = "✓" if t.get("enabled", True) else "✗"
            lines.append(f"  {status} {t['class']} — {t.get('params', {})}")

        clf = self.config.get("classifier", {})
        lines += [
            "",
            f"Classifier    : {clf.get('class')} {clf.get('params', {})}",
            "=" * 60,
        ]

        bd = self.config.get("bias_detection", {})
        lines += [
            f"Bias detection: {'enabled' if bd.get('enabled', True) else 'disabled'}",
            f"  proxy_threshold     = {bd.get('proxy_threshold', 0.30)}",
            f"  representation_gap  = {bd.get('representation_gap', 0.05)}",
            f"  p_threshold         = {bd.get('p_threshold', 0.05)}",
            "=" * 60,
        ]

        return "\n".join(lines)