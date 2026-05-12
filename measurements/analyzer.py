"""
fairness_analyzer.py
====================
Unified Fairness Measurement Module.

Wraps Fairlearn, AIF360, and Aequitas behind a single FairnessAnalyzer class.
Every public method returns a FairnessResult named-tuple so callers always get
a consistent, structured object with: value, confidence_interval, effect_size,
sample_sizes, and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Optional heavy dependencies – imported lazily so the module can be imported
# even when a library is not installed (only raises at call time).
# ---------------------------------------------------------------------------

def _import_fairlearn():
    try:
        from fairlearn.metrics import (
            MetricFrame,
            demographic_parity_difference,
            equalized_odds_difference,
        )
        return MetricFrame, demographic_parity_difference, equalized_odds_difference
    except ImportError as exc:
        raise ImportError("Install fairlearn: pip install fairlearn") from exc


def _import_aif360():
    try:
        from aif360.datasets import BinaryLabelDataset
        from aif360.metrics import ClassificationMetric
        return BinaryLabelDataset, ClassificationMetric
    except ImportError as exc:
        raise ImportError("Install aif360: pip install aif360") from exc


def _import_aequitas():
    try:
        from aequitas.group import Group
        from aequitas.bias import Bias
        return Group, Bias
    except ImportError as exc:
        raise ImportError("Install aequitas: pip install aequitas") from exc


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

@dataclass
class FairnessResult:
    """
    Attributes
    ----------
    metric_name:
        Human-readable name of the metric.
    value:
        Point estimate of the metric.
    confidence_interval:
        Tuple (lower, upper) representing the 95% bootstrap CI.
    effect_size:
        Standardised effect size (e.g. Risk Ratio).  ``None`` when not applicable.
    sample_sizes:
        Dict mapping group name → sample count.
    metadata:
        Any extra information the specific engine wants to surface.
    """
    metric_name: str
    value: float
    confidence_interval: Tuple[float, float]
    effect_size: Optional[float]
    sample_sizes: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        ci_lo, ci_hi = self.confidence_interval
        return (
            f"FairnessResult({self.metric_name})\n"
            f"  value              = {self.value:.4f}\n"
            f"  95% CI             = ({ci_lo:.4f}, {ci_hi:.4f})\n"
            f"  effect_size        = {self.effect_size}\n"
            f"  sample_sizes       = {self.sample_sizes}\n"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FairnessAnalyzer:
    """
    Unified Library Integration Layer for Fairness Analysis.

    Acts as a single entry-point that delegates to Fairlearn, AIF360, and
    Aequitas.  Users interact only with this class.

    Parameters
    ----------
    df:
        Input data as a DataFrame *or* a path to a CSV file.
    target_col:
        Name of the ground-truth label column.
    sensitive_col:
        Name of the sensitive-attribute column.
    positive_label:
        Value that represents the positive / favourable outcome.
    """

    def __init__(
        self,
        df: pd.DataFrame | str,
        target_col: Optional[str] = None,
        sensitive_col: Optional[str] = None,
        positive_label: Any = 1,
    ) -> None:
        if isinstance(df, str):
            self.df = pd.read_csv(df)
            self.df.columns = self.df.columns.str.strip()
        elif isinstance(df, pd.DataFrame):
            self.df = df.copy()
        else:
            raise TypeError("'df' must be a DataFrame or a CSV file path.")

        self.target_col = target_col
        self.sensitive_col = sensitive_col
        self.positive_label = positive_label

        print(f"FairnessAnalyzer ready — {len(self.df):,} rows loaded.")

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_config(
        self,
        target_col: str,
        sensitive_col: str,
        positive_label: Any = 1,
    ) -> "FairnessAnalyzer":
        """Fluent setter for core configuration."""
        self.target_col = target_col
        self.sensitive_col = sensitive_col
        self.positive_label = positive_label
        return self

    def bin_column(
        self,
        column_name: str,
        bins: Sequence,
        labels: Sequence[str],
        new_column_name: str,
        *,
        set_as_sensitive: bool = False,
    ) -> "FairnessAnalyzer":
        """
        Bin a continuous column into labelled categories.

        Parameters
        ----------
        set_as_sensitive:
            When *True*, automatically update ``sensitive_col`` to the new
            binned column.  Defaults to *False* to avoid silent side-effects.
        """
        self.df[new_column_name] = pd.cut(
            self.df[column_name], bins=bins, labels=labels
        )
        if set_as_sensitive:
            self.sensitive_col = new_column_name
        print(f"'{column_name}' → '{new_column_name}' (set_as_sensitive={set_as_sensitive}).")
        return self

    def create_intersectional_feature(
        self,
        column_names: List[str],
        new_column_name: str,
        *,
        set_as_sensitive: bool = True,
    ) -> "FairnessAnalyzer":
        """
        Requirement 3 – Intersectionality.

        Combines multiple sensitive columns into a single compound feature.
        Example: ['Gender', 'Age_Group'] → 'Male_Young'.
        """
        self.df[new_column_name] = (
            self.df[column_names].astype(str).agg("_".join, axis=1)
        )
        if set_as_sensitive:
            self.sensitive_col = new_column_name
        print(
            f"Intersectional feature '{new_column_name}' created from {column_names}."
        )
        return self

    # ------------------------------------------------------------------
    # Internal adapters (abstraction layer)
    # ------------------------------------------------------------------

    def _require_config(self) -> None:
        if not self.target_col or not self.sensitive_col:
            raise ValueError(
                "Both 'target_col' and 'sensitive_col' must be set. "
                "Call set_config() first."
            )

    def _filter_by_group_size(
        self, df: pd.DataFrame, sensitive_col: str, min_group_size: int
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Requirement 3 – min_group_size filter.

        Returns (filtered_df, excluded_group_names).
        """
        counts = df[sensitive_col].value_counts()
        valid = counts[counts >= min_group_size].index
        excluded = counts[counts < min_group_size].index.tolist()
        if excluded:
            warnings.warn(
                f"Excluding {len(excluded)} group(s) with < {min_group_size} samples: "
                f"{excluded}",
                UserWarning,
                stacklevel=3,
            )
        return df[df[sensitive_col].isin(valid)].copy(), excluded

    def _prepare_aif360_dataset(self, df: pd.DataFrame):
        """
        Internal adapter – converts a DataFrame to an AIF360 BinaryLabelDataset.
        Requirement 2 – abstracts AIF360 complexity.
        """
        BinaryLabelDataset, _ = _import_aif360()
        favorable = self.positive_label
        # Derive unfavorable label robustly from the unique values present
        unique_labels = df[self.target_col].unique().tolist()
        unfavorable_candidates = [v for v in unique_labels if v != favorable]
        if not unfavorable_candidates:
            raise ValueError("Only one unique label found – cannot build BinaryLabelDataset.")
        unfavorable = unfavorable_candidates[0]

        return BinaryLabelDataset(
            df=df[[self.target_col, self.sensitive_col]].reset_index(drop=True),
            label_names=[self.target_col],
            protected_attribute_names=[self.sensitive_col],
            favorable_label=favorable,
            unfavorable_label=unfavorable,
        )

    def _prepare_aequitas_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal adapter for Aequitas.
        Aequitas requires columns: 'score', 'label_value', plus the original
        sensitive attribute column(s) kept under their original names.
        """
        ae_df = df[[self.target_col, self.sensitive_col]].copy()
        ae_df = ae_df.rename(
            columns={self.target_col: "label_value"}
        )
        # score = predicted probability / hard label; here we expose the target
        # so callers can override it externally before calling get_aequitas_metrics.
        ae_df["score"] = ae_df["label_value"].astype(float)
        ae_df["label_value"] = ae_df["label_value"].astype(int)
        return ae_df

    # ------------------------------------------------------------------
    # Bootstrap CI engine
    # ------------------------------------------------------------------

    def _compute_bootstrap_ci(
        self,
        metric_func: Callable[[pd.DataFrame], float],
        df: pd.DataFrame,
        n_iterations: int = 1000,
        random_state: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Requirement 4 – 95% Bootstrap Confidence Interval.
        Optimized: stratified sampling + pre-reset index for speed.
        """
        rng = np.random.default_rng(random_state)
        stats: List[float] = []

        # Pre-reset index once instead of inside the loop
        df_reset = df.reset_index(drop=True)

        # Pre-compute group indices once (stratified bootstrap)
        groups = df.groupby(self.sensitive_col, sort=False)
        group_indices = {name: grp.index.to_numpy() for name, grp in groups}

        for _ in range(n_iterations):
            # Stratified resample: sample within each group independently
            sampled_idx = np.concatenate([
                rng.choice(idxs, size=len(idxs), replace=True)
                for idxs in group_indices.values()
            ])
            sample = df_reset.iloc[sampled_idx].reset_index(drop=True)
            try:
                val = metric_func(sample)
                if np.isfinite(val):
                    stats.append(val)
            except Exception:
                pass

        if not stats:
            raise RuntimeError("Bootstrap produced no valid samples.")

        lower = float(np.percentile(stats, 2.5))
        upper = float(np.percentile(stats, 97.5))
        return lower, upper
    # ------------------------------------------------------------------
    # Classification metrics (Requirement 3)
    # ------------------------------------------------------------------

    def calculate_classification_metrics(
        self,
        y_pred: Optional[pd.Series] = None,
        engine: str = "fairlearn",
        min_group_size: int = 5,
        n_bootstrap: int = 1000,
        random_state: Optional[int] = None,
    ) -> Dict[str, FairnessResult]:
        """
        Requirement 3 – Robust metrics engine for Classification.

        Computes demographic_parity_difference and equalized_odds_difference.

        Parameters
        ----------
        y_pred:
            Model predictions.  If *None*, falls back to ``target_col`` values
            (useful for auditing ground-truth disparities, but a warning is
            raised so the caller is never silently misled).
        engine:
            ``'fairlearn'`` (default) or ``'aif360'``.
        min_group_size:
            Groups smaller than this are excluded (Requirement 3).
        n_bootstrap:
            Number of bootstrap replications for CIs.
        random_state:
            Seed for reproducibility.

        Returns
        -------
        Dict mapping metric name → FairnessResult.
        """
        self._require_config()

        if y_pred is None:
            warnings.warn(
                "y_pred not provided – using target column as proxy. "
                "Metrics will reflect label disparities, not model bias.",
                UserWarning,
                stacklevel=2,
            )
            y_pred = self.df[self.target_col]

        # Align y_pred with df index
        work_df = self.df.copy()
        work_df["_y_pred"] = y_pred.values if hasattr(y_pred, "values") else y_pred

        filtered_df, excluded = self._filter_by_group_size(
            work_df, self.sensitive_col, min_group_size
        )
        if filtered_df.empty:
            raise ValueError("No groups meet the minimum size requirement.")

        results: Dict[str, FairnessResult] = {}
        sample_sizes = (
            filtered_df[self.sensitive_col].value_counts().to_dict()
        )

        if engine == "fairlearn":
            MetricFrame, dp_diff_fn, eo_diff_fn = _import_fairlearn()

            y_true = filtered_df[self.target_col]
            y_hat = filtered_df["_y_pred"]
            sf = filtered_df[self.sensitive_col]

            # --- Demographic Parity Difference ---
            dp_point = dp_diff_fn(y_true, y_hat, sensitive_features=sf)

            def _dp(df: pd.DataFrame) -> float:
                return dp_diff_fn(
                    df[self.target_col],
                    df["_y_pred"],
                    sensitive_features=df[self.sensitive_col],
                )

            dp_ci = self._compute_bootstrap_ci(
                _dp, filtered_df, n_bootstrap, random_state
            )

            # Effect size: ratio of max to min positive rates across groups
            mf_dp = MetricFrame(
                metrics={"pos_rate": lambda yt, yp: (yp == self.positive_label).mean()},
                y_true=y_true,
                y_pred=y_hat,
                sensitive_features=sf,
            )
            rates = mf_dp.by_group["pos_rate"]
            dp_rr = (rates.min() / rates.max()) if rates.max() > 0 else None

            results["demographic_parity_difference"] = FairnessResult(
                metric_name="Demographic Parity Difference",
                value=float(dp_point),
                confidence_interval=dp_ci,
                effect_size=float(dp_rr) if dp_rr is not None else None,
                sample_sizes=sample_sizes,
                metadata={
                    "library": "Fairlearn",
                    "excluded_groups": excluded,
                    "positive_rate_by_group": rates.to_dict(),
                },
            )

            # --- Equalized Odds Difference ---
            eo_point = eo_diff_fn(y_true, y_hat, sensitive_features=sf)

            def _eo(df: pd.DataFrame) -> float:
                return eo_diff_fn(
                    df[self.target_col],
                    df["_y_pred"],
                    sensitive_features=df[self.sensitive_col],
                )

            eo_ci = self._compute_bootstrap_ci(
                _eo, filtered_df, n_bootstrap, random_state
            )

            results["equalized_odds_difference"] = FairnessResult(
                metric_name="Equalized Odds Difference",
                value=float(eo_point),
                confidence_interval=eo_ci,
                effect_size=None,  # EO diff has no canonical single effect size
                sample_sizes=sample_sizes,
                metadata={"library": "Fairlearn", "excluded_groups": excluded,"positive_rate_by_group": rates.to_dict(),}
            )

        elif engine == "aif360":
            _, ClassificationMetric = _import_aif360()
            dataset = self._prepare_aif360_dataset(filtered_df)

            # Split into privileged / unprivileged based on most-frequent group
            mode_group = filtered_df[self.sensitive_col].mode()[0]
            attr_idx = dataset.protected_attribute_names.index(self.sensitive_col)

            priv_val = dataset.unprivileged_protected_attributes = [
                {self.sensitive_col: v}
                for v in filtered_df[self.sensitive_col].unique()
                if v != mode_group
            ]
            unpriv = [{"attr": mode_group}]

            # For AIF360 we need a predicted dataset too – use the same object
            # when y_pred ≈ y_true (audit mode).
            pred_dataset = dataset.copy()
            pred_dataset.labels = filtered_df["_y_pred"].values.reshape(-1, 1)

            cm = ClassificationMetric(
                dataset,
                pred_dataset,
                unprivileged_groups=[{self.sensitive_col: v}
                                     for v in filtered_df[self.sensitive_col].unique()
                                     if v != mode_group],
                privileged_groups=[{self.sensitive_col: mode_group}],
            )
            dp_point = cm.statistical_parity_difference()

            results["statistical_parity_difference"] = FairnessResult(
                metric_name="Statistical Parity Difference (AIF360)",
                value=float(dp_point),
                confidence_interval=(float("nan"), float("nan")),  # no bootstrap for AIF360 path
                effect_size=None,
                sample_sizes=sample_sizes,
                metadata={"library": "AIF360", "excluded_groups": excluded},
            )

        else:
            raise ValueError(f"Unknown engine '{engine}'. Choose 'fairlearn' or 'aif360'.")

        return results

    # ------------------------------------------------------------------
    # Regression metrics (Requirement 3)
    # ------------------------------------------------------------------

    def calculate_regression_metrics(
        self,
        y_pred: Optional[pd.Series] = None,
        target_col: Optional[str] = None,
        min_group_size: int = 5,
        n_bootstrap: int = 1000,
        random_state: Optional[int] = None,
    ) -> FairnessResult:
        """
        Requirement 3 – Robust metrics engine for Regression.

        Computes the difference in Mean Absolute Error across groups.

        Parameters
        ----------
        y_pred:
            Model predictions.  Required for a meaningful audit.
        target_col:
            Override the instance-level target column (e.g. 'Interest_Rate').
        """
        if target_col:
            self.target_col = target_col
        self._require_config()

        if y_pred is None:
            warnings.warn(
                "y_pred not provided – regression metrics will be trivially zero "
                "because y_true is used as prediction.",
                UserWarning,
                stacklevel=2,
            )
            y_pred = self.df[self.target_col]

        MetricFrame, _, _ = _import_fairlearn()

        work_df = self.df.copy()
        work_df["_y_pred"] = y_pred.values if hasattr(y_pred, "values") else y_pred

        filtered_df, excluded = self._filter_by_group_size(
            work_df, self.sensitive_col, min_group_size
        )
        if filtered_df.empty:
            raise ValueError("No groups meet the minimum size requirement.")

        y_true = filtered_df[self.target_col]
        y_hat = filtered_df["_y_pred"]
        sf = filtered_df[self.sensitive_col]

        mf = MetricFrame(
            metrics={"mae": mean_absolute_error},
            y_true=y_true,
            y_pred=y_hat,
            sensitive_features=sf,
        )

        mae_diff_point = float(mf.difference(method="between_groups")["mae"])
        mae_by_group = mf.by_group["mae"]

        # Effect size: ratio worst/best MAE
        effect_size = (
            float(mae_by_group.max() / mae_by_group.min())
            if mae_by_group.min() > 0
            else None
        )

        def _mae_diff(df: pd.DataFrame) -> float:
            mf_ = MetricFrame(
                metrics={"mae": mean_absolute_error},
                y_true=df[self.target_col],
                y_pred=df["_y_pred"],
                sensitive_features=df[self.sensitive_col],
            )
            return float(mf_.difference(method="between_groups")["mae"])

        ci = self._compute_bootstrap_ci(_mae_diff, filtered_df, n_bootstrap, random_state)

        return FairnessResult(
            metric_name="MAE Difference Across Groups",
            value=mae_diff_point,
            confidence_interval=ci,
            effect_size=effect_size,
            sample_sizes=filtered_df[self.sensitive_col].value_counts().to_dict(),
            metadata={
                "library": "Fairlearn",
                "overall_mae": float(mf.overall["mae"]),
                "mae_by_group": mae_by_group.to_dict(),
                "excluded_groups": excluded,
            },
        )

    # ------------------------------------------------------------------
    # Full audit (selection-rate analysis)  – Requirement 4
    # ------------------------------------------------------------------

    def _calculate_selection_rates(
        self, df: pd.DataFrame
    ) -> pd.Series:
        """Returns positive-label rate per group."""
        return (
            df.groupby(self.sensitive_col)[self.target_col]
            .apply(lambda s: (s == self.positive_label).mean())
        )

    def get_fairness_audit(
        self,
        n_bootstrap: int = 200,
        random_state: Optional[int] = None,
        min_group_size: int = 5,
    ) -> FairnessResult:
        self._require_config()

        filtered_df, excluded = self._filter_by_group_size(
            self.df, self.sensitive_col, min_group_size
        )
        if filtered_df.empty:
            raise ValueError("No groups meet the minimum size requirement.")

        rates = self._calculate_selection_rates(filtered_df)
        priv_group = rates.idxmax()
        unpriv_group = rates.idxmin()
        base_rate = float(rates[priv_group])
        target_rate = float(rates[unpriv_group])

        point_estimate = base_rate - target_rate
        risk_ratio = target_rate / base_rate if base_rate > 0 else None

        # ✅ CI sulla differenza — coerente con il point estimate
        def _rate_diff(df: pd.DataFrame) -> float:
            col = df[self.sensitive_col]
            target_col = df[self.target_col]
            pos = target_col == self.positive_label

            priv_mask = col == priv_group
            unpriv_mask = col == unpriv_group

            n_priv = priv_mask.sum()
            n_unpriv = unpriv_mask.sum()

            if n_priv == 0 or n_unpriv == 0:
                return float("nan")

            return float(pos[priv_mask].mean()) - float(pos[unpriv_mask].mean())

        ci = self._compute_bootstrap_ci(
            _rate_diff, filtered_df, n_bootstrap, random_state
        )

        return FairnessResult(
            metric_name="Selection Rate Disparity",
            value=point_estimate,
            confidence_interval=ci,
            effect_size=risk_ratio,
            sample_sizes=filtered_df[self.sensitive_col].value_counts().to_dict(),
            metadata={
                "privileged_group": priv_group,
                "privileged_rate": base_rate,
                "unprivileged_group": unpriv_group,
                "unprivileged_rate": target_rate,
                "all_rates": rates.to_dict(),
                "excluded_groups": excluded,
            },
        )
    # ------------------------------------------------------------------
    # Aequitas integration  – Requirement 2 (third library)
    # ------------------------------------------------------------------

    def get_aequitas_metrics(
        self,
        y_pred: Optional[pd.Series] = None,
        min_group_size: int = 5,
    ) -> pd.DataFrame:
        """
        Requirement 2 – Unified layer (Aequitas integration).

        Uses Aequitas to compute group-level and disparity metrics.
        The reference group defaults to the most-frequent group.

        Parameters
        ----------
        y_pred:
            Hard-label predictions.  Falls back to ``target_col`` with a warning.
        """
        self._require_config()
        Group, Bias = _import_aequitas()

        if y_pred is None:
            warnings.warn(
                "y_pred not provided – using target_col as score (audit mode).",
                UserWarning,
                stacklevel=2,
            )
            y_pred = self.df[self.target_col]

        filtered_df, _ = self._filter_by_group_size(
            self.df, self.sensitive_col, min_group_size
        )

        # Build Aequitas-compliant DataFrame
        ae_df = filtered_df[[self.target_col, self.sensitive_col]].copy()
        ae_df = ae_df.rename(columns={self.target_col: "label_value"})
        ae_df["score"] = (
            y_pred.loc[filtered_df.index].values
            if hasattr(y_pred, "loc")
            else np.array(y_pred)[filtered_df.index]
        )
        ae_df["label_value"] = ae_df["label_value"].astype(int)
        # Aequitas expects the sensitive attribute under its original column name
        ae_df = ae_df.rename(columns={self.sensitive_col: self.sensitive_col})

        g = Group()
        xtab, _ = g.get_crosstabs(ae_df, attr_cols=[self.sensitive_col])

        ref_group = self.df[self.sensitive_col].mode()[0]
        b = Bias()
        bias_df = b.get_disparity_predefined_groups(
            xtab,
            ae_df,
            ref_groups_dict={self.sensitive_col: ref_group},
        )
        return bias_df

    # ------------------------------------------------------------------
    # Intersectional analysis with multiple-comparison correction
    # Requirement 3 + Stretch Goal
    # ------------------------------------------------------------------

    def intersectional_audit(
        self,
        column_names: List[str],
        intersectional_col: str = "_intersectional",
        min_group_size: int = 10,
        n_bootstrap: int = 500,
        random_state: Optional[int] = None,
        apply_fdr_correction: bool = True,
    ) -> pd.DataFrame:
        """
        Requirement 3 – Intersectional analysis with min_group_size.
        Stretch Goal – Benjamini-Hochberg FDR correction.

        Computes per-intersectional-group selection rates, bootstrapped CIs,
        and p-values (approximated via CI overlap with the global rate), then
        optionally applies B-H correction.

        Returns a DataFrame with one row per intersectional group.
        """
        self._require_config()

        # Build intersectional feature on a working copy so self.df is untouched
        work_df = self.df.copy()
        work_df[intersectional_col] = (
            work_df[column_names].astype(str).agg("_".join, axis=1)
        )

        counts = work_df[intersectional_col].value_counts()
        valid_groups = counts[counts >= min_group_size].index
        excluded = counts[counts < min_group_size].index.tolist()
        if excluded:
            warnings.warn(
                f"Excluding {len(excluded)} intersectional group(s) with < "
                f"{min_group_size} samples.",
                UserWarning,
                stacklevel=2,
            )

        global_rate = float(
            (work_df[self.target_col] == self.positive_label).mean()
        )

        records = []
        for group in valid_groups:
            sub = work_df[work_df[intersectional_col] == group].copy()
            group_rate = float(
                (sub[self.target_col] == self.positive_label).mean()
            )

            def _rate(df: pd.DataFrame, g=group) -> float:  # noqa: B023
                s = df[df[intersectional_col] == g]
                if len(s) == 0:
                    return float("nan")
                return float((s[self.target_col] == self.positive_label).mean())

            ci_lo, ci_hi = self._compute_bootstrap_ci(
                _rate, work_df, n_bootstrap, random_state
            )

            # Approximate p-value: fraction of bootstrap samples that exclude
            # the global rate (non-parametric significance proxy).
            rr = group_rate / global_rate if global_rate > 0 else None

            records.append(
                {
                    "group": group,
                    "n": len(sub),
                    "selection_rate": group_rate,
                    "global_rate": global_rate,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "risk_ratio": rr,
                    # p-value proxy: CI excludes global rate?
                    "p_value_proxy": float(
                        not (ci_lo <= global_rate <= ci_hi)
                    ),
                }
            )

        result_df = pd.DataFrame(records).reset_index(drop=True)

        # Stretch Goal: Benjamini-Hochberg FDR correction
        if apply_fdr_correction and len(result_df) > 1:
            _, pvals_corrected, _, _ = multipletests(
                result_df["p_value_proxy"], method="fdr_bh"
            )
            result_df["p_value_bh_corrected"] = pvals_corrected
            result_df["significant_after_correction"] = pvals_corrected < 0.05

        result_df.attrs["excluded_groups"] = excluded
        return result_df

    # ------------------------------------------------------------------
    # MLOps integration – Requirement 5
    # ------------------------------------------------------------------

    def log_to_mlflow(
        self,
        results: Dict[str, Any],
        run_id: Optional[str] = None,
    ) -> None:
        """
        Requirement 5 – Log all fairness results to an active MLflow run.

        ``results`` may be a single FairnessResult or a dict of FairnessResults
        (as returned by calculate_classification_metrics).
        """
        try:
            import mlflow  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Install mlflow: pip install mlflow") from exc

        if not mlflow.active_run() and run_id is None:
            warnings.warn(
                "No active MLflow run found.  Start one with mlflow.start_run() "
                "or pass run_id.",
                UserWarning,
                stacklevel=2,
            )
            return

        def _log_single(result: FairnessResult, prefix: str = "") -> None:
            key = (prefix + result.metric_name).replace(" ", "_").lower()
            mlflow.log_metric(f"{key}.value", result.value)
            mlflow.log_metric(f"{key}.ci_lower", result.confidence_interval[0])
            mlflow.log_metric(f"{key}.ci_upper", result.confidence_interval[1])
            if result.effect_size is not None:
                mlflow.log_metric(f"{key}.effect_size", result.effect_size)
            for group, n in result.sample_sizes.items():
                mlflow.log_metric(f"{key}.n_{group}", n)

        mlflow.log_param("fairness.sensitive_column", self.sensitive_col)
        mlflow.log_param("fairness.target_column", self.target_col)

        if isinstance(results, FairnessResult):
            _log_single(results)
        elif isinstance(results, dict):
            for name, res in results.items():
                if isinstance(res, FairnessResult):
                    _log_single(res, prefix=f"{name}.")
        else:
            raise TypeError("results must be a FairnessResult or a dict of FairnessResult.")

        print("✓ Fairness metrics logged to MLflow.")

    # ------------------------------------------------------------------
    # CI/CD testing helper – Requirement 5
    # ------------------------------------------------------------------

    @staticmethod
    def assert_fairness(
        result: FairnessResult,
        threshold: float = 0.8,
        metric: str = "effect_size",
    ) -> bool:
        """
        Requirement 5 – Custom pytest assertion for CI/CD pipelines.

        Parameters
        ----------
        result:
            A FairnessResult object.
        threshold:
            Minimum acceptable value for the chosen metric.
        metric:
            ``'effect_size'`` (default, checks Risk Ratio ≥ threshold) or
            ``'value'`` (checks |value| ≤ threshold).

        Raises
        ------
        AssertionError
            When the fairness check fails.
        """
        if metric == "effect_size":
            val = result.effect_size
            if val is None:
                raise AssertionError(
                    f"Fairness check failed: effect_size is None for "
                    f"'{result.metric_name}'."
                )
            if val < threshold:
                raise AssertionError(
                    f"Fairness check FAILED — '{result.metric_name}': "
                    f"Risk Ratio {val:.4f} < threshold {threshold:.4f}."
                )
        elif metric == "value":
            val = abs(result.value)
            if val > threshold:
                raise AssertionError(
                    f"Fairness check FAILED — '{result.metric_name}': "
                    f"|value| {val:.4f} > threshold {threshold:.4f}."
                )
        else:
            raise ValueError(f"Unknown metric '{metric}'. Choose 'effect_size' or 'value'.")
        return True

    # ------------------------------------------------------------------
    # Stretch Goal – Visualisation
    # ------------------------------------------------------------------

    def generate_report_visualizations(
        self,
        results: Dict[str, FairnessResult],
        output_path: str = "fairness_report.png",
    ) -> str:
        """
        Stretch Goal – Generate and save a fairness audit visualisation.

        Produces a bar chart of metric values with 95% CI error bars.
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Install matplotlib: pip install matplotlib") from exc

        if not results:
            raise ValueError("No results to visualise.")

        labels = list(results.keys())
        values = [r.value for r in results.values()]
        ci_lowers = [r.confidence_interval[0] for r in results.values()]
        ci_uppers = [r.confidence_interval[1] for r in results.values()]

        yerr_lower = [v - lo for v, lo in zip(values, ci_lowers)]
        yerr_upper = [hi - v for v, hi in zip(values, ci_uppers)]

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color="#4C72B0", alpha=0.85, zorder=2)
        ax.errorbar(
            x, values,
            yerr=[yerr_lower, yerr_upper],
            fmt="none", color="black", capsize=5, linewidth=1.5, zorder=3,
        )
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Metric Value")
        ax.set_title(
            f"Fairness Audit — {self.sensitive_col}", fontsize=12, fontweight="bold"
        )
        ax.grid(axis="y", alpha=0.3, zorder=0)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"✓ Visualisation saved to '{output_path}'.")
        return output_path

    # ------------------------------------------------------------------
    # Stretch Goal – Multiple comparisons correction (standalone helper)
    # ------------------------------------------------------------------

    @staticmethod
    def apply_bias_correction(
        p_values: Sequence[float],
        method: str = "fdr_bh",
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stretch Goal – Multiple-comparison correction.

        Parameters
        ----------
        p_values:
            Raw p-values (one per test).
        method:
            Any method accepted by ``statsmodels.stats.multitest.multipletests``
            (default: ``'fdr_bh'`` = Benjamini-Hochberg).
        alpha:
            Family-wise significance level.

        Returns
        -------
        (rejected, pvals_corrected)
            Boolean array of rejected hypotheses and corrected p-values.
        """
        rejected, pvals_corrected, _, _ = multipletests(
            p_values, alpha=alpha, method=method
        )
        return rejected, pvals_corrected