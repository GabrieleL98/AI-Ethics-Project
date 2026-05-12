"""
detection_engine.py
====================
Stage 1 of the Fair Pipeline — Bias Detection Engine.

Responsibilities
----------------
1. Representation bias   – compare demographic distributions against benchmarks.
2. Statistical disparity – identify features distributed differently across groups
                           (chi-square for categorical, ANOVA / Kruskal-Wallis for numeric).
3. Proxy variable ID     – correlations between non-sensitive features and protected
                           attributes (Cramér's V for cat-cat, point-biserial for num-cat).

Output
------
A structured JSON report written to disk (path configurable).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal, pointbiserialr


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RepresentationFinding:
    group: str
    observed_rate: float
    benchmark_rate: float
    absolute_gap: float
    flagged: bool


@dataclass
class DisparityFinding:
    feature: str
    feature_type: str          # 'categorical' | 'numeric'
    test: str                  # 'chi2' | 'anova' | 'kruskal'
    statistic: float
    p_value: float
    flagged: bool


@dataclass
class ProxyFinding:
    feature: str
    sensitive_attribute: str
    method: str                # 'cramers_v' | 'point_biserial'
    correlation: float
    flagged: bool


@dataclass
class BiasReport:
    dataset_shape: Tuple[int, int]
    sensitive_columns: List[str]
    representation: Dict[str, List[RepresentationFinding]] = field(default_factory=dict)
    disparity: List[DisparityFinding] = field(default_factory=list)
    proxy: List[ProxyFinding] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V between two categorical series."""
    ct = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def _point_biserial(numeric: pd.Series, binary: pd.Series) -> float:
    """
    Point-biserial correlation between a numeric and a binary series.
    Encodes categorical strings to 0/1 before calculation.
    """
    from scipy.stats import pointbiserialr
    
    if not pd.api.types.is_numeric_dtype(binary):
        binary_values = pd.factorize(binary)[0]
    else:
        binary_values = binary.astype(int)
        
    try:
        corr, _ = pointbiserialr(binary_values, numeric)
        return float(abs(corr))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class BiasDetectionEngine:
    """
    Audits a DataFrame for representation bias, statistical disparity,
    and proxy variable leakage.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to audit.
    sensitive_cols : list[str]
        Protected attribute column names.
    benchmarks : dict, optional
        ``{col: {group_label: expected_rate}}``  e.g.
        ``{"Gender": {"Male": 0.50, "Female": 0.50}}``.
    p_threshold : float
        Significance level for disparity tests (default 0.05).
    proxy_threshold : float
        Correlation threshold above which a feature is flagged as proxy
        (default 0.3).
    representation_gap : float
        Absolute gap above which representation is flagged (default 0.05).
    output_path : str or Path, optional
        Where to write the JSON report.  ``None`` skips writing.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sensitive_cols: List[str],
        benchmarks: Optional[Dict[str, Dict[str, float]]] = None,
        p_threshold: float = 0.05,
        proxy_threshold: float = 0.3,
        representation_gap: float = 0.05,
        output_path: Optional[str | Path] = "bias_report.json",
    ) -> None:
        self.df = df.copy()
        self.sensitive_cols = sensitive_cols
        self.benchmarks = benchmarks or {}
        self.p_threshold = p_threshold
        self.proxy_threshold = proxy_threshold
        self.representation_gap = representation_gap
        self.output_path = Path(output_path) if output_path else None

        self._non_sensitive = [c for c in df.columns if c not in sensitive_cols]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BiasReport:
        """Execute all checks and return (and optionally save) a BiasReport."""
        report = BiasReport(
            dataset_shape=self.df.shape,
            sensitive_columns=self.sensitive_cols,
        )

        report.representation = self._check_representation()
        report.disparity = self._check_disparity()
        report.proxy = self._check_proxies()
        report.summary = self._build_summary(report)

        if self.output_path is not None:
            self._save(report)

        return report

    # ------------------------------------------------------------------
    # 1. Representation bias
    # ------------------------------------------------------------------

    def _check_representation(self) -> Dict[str, List[RepresentationFinding]]:
        results: Dict[str, List[RepresentationFinding]] = {}

        for col in self.sensitive_cols:
            if col not in self.df.columns:
                warnings.warn(f"Sensitive column '{col}' not found — skipped.")
                continue

            observed = self.df[col].value_counts(normalize=True)
            bench = self.benchmarks.get(col, {})
            findings: List[RepresentationFinding] = []

            for group, obs_rate in observed.items():
                bench_rate = bench.get(group, 1.0 / len(observed))  # uniform if no benchmark
                gap = abs(float(obs_rate) - bench_rate)
                findings.append(RepresentationFinding(
                    group=str(group),
                    observed_rate=round(float(obs_rate), 4),
                    benchmark_rate=round(bench_rate, 4),
                    absolute_gap=round(gap, 4),
                    flagged=gap > self.representation_gap,
                ))

            results[col] = findings

        return results

    # ------------------------------------------------------------------
    # 2. Statistical disparity
    # ------------------------------------------------------------------

    def _check_disparity(self) -> List[DisparityFinding]:
        findings: List[DisparityFinding] = []

        for sensitive_col in self.sensitive_cols:
            if sensitive_col not in self.df.columns:
                continue
            groups_series = self.df[sensitive_col]

            for feat in self._non_sensitive:
                series = self.df[feat].dropna()
                aligned = groups_series.loc[series.index]

                if pd.api.types.is_numeric_dtype(series):
                    finding = self._disparity_numeric(feat, series, aligned)
                else:
                    finding = self._disparity_categorical(feat, series, aligned)

                if finding:
                    findings.append(finding)

        return findings

    def _disparity_numeric(
        self, feat: str, series: pd.Series, groups: pd.Series
    ) -> Optional[DisparityFinding]:
        group_arrays = [
            series[groups == g].values
            for g in groups.unique()
            if (groups == g).sum() > 1
        ]
        if len(group_arrays) < 2:
            return None

        # Try ANOVA; fall back to Kruskal-Wallis if normality unlikely
        try:
            stat, p = f_oneway(*group_arrays)
            test = "anova"
        except Exception:
            stat, p = kruskal(*group_arrays)
            test = "kruskal"

        if not np.isfinite(p):
            return None

        return DisparityFinding(
            feature=feat,
            feature_type="numeric",
            test=test,
            statistic=round(float(stat), 4),
            p_value=round(float(p), 6),
            flagged=p < self.p_threshold,
        )

    def _disparity_categorical(
        self, feat: str, series: pd.Series, groups: pd.Series
    ) -> Optional[DisparityFinding]:
        ct = pd.crosstab(series, groups)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return None
        try:
            chi2, p, _, _ = chi2_contingency(ct)
        except Exception:
            return None

        return DisparityFinding(
            feature=feat,
            feature_type="categorical",
            test="chi2",
            statistic=round(float(chi2), 4),
            p_value=round(float(p), 6),
            flagged=p < self.p_threshold,
        )

    # ------------------------------------------------------------------
    # 3. Proxy variable identification
    # ------------------------------------------------------------------

    def _check_proxies(self) -> List[ProxyFinding]:
        findings: List[ProxyFinding] = []

        for sensitive_col in self.sensitive_cols:
            if sensitive_col not in self.df.columns:
                continue

            sensitive = self.df[sensitive_col]
            sensitive_is_binary = sensitive.nunique() == 2

            for feat in self._non_sensitive:
                if feat == sensitive_col:
                    continue

                feat_series = self.df[feat].dropna()
                aligned_sensitive = sensitive.loc[feat_series.index]

                feat_is_numeric = pd.api.types.is_numeric_dtype(feat_series)

                if feat_is_numeric and sensitive_is_binary:
                    corr = _point_biserial(feat_series, aligned_sensitive)
                    method = "point_biserial"
                elif not feat_is_numeric:
                    corr = _cramers_v(feat_series, aligned_sensitive)
                    method = "cramers_v"
                else:
                    # numeric feature, non-binary sensitive → eta-squared approximation
                    groups = [
                        feat_series[aligned_sensitive == g].values
                        for g in aligned_sensitive.unique()
                        if (aligned_sensitive == g).sum() > 1
                    ]
                    if len(groups) < 2:
                        continue
                    try:
                        stat, _ = f_oneway(*groups)
                        # η² ≈ SS_between / SS_total — approximated via F
                        k = len(groups)
                        n = feat_series.shape[0]
                        corr = float(np.clip((stat * (k - 1)) / (stat * (k - 1) + (n - k)), 0, 1))
                        corr = float(np.sqrt(corr))   # eta
                    except Exception:
                        continue
                    method = "eta"

                findings.append(ProxyFinding(
                    feature=feat,
                    sensitive_attribute=sensitive_col,
                    method=method,
                    correlation=round(corr, 4),
                    flagged=corr > self.proxy_threshold,
                ))

        return findings

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(self, report: BiasReport) -> Dict[str, Any]:
        rep_flagged = sum(
            1
            for findings in report.representation.values()
            for f in findings
            if f.flagged
        )
        disp_flagged = sum(1 for f in report.disparity if f.flagged)
        proxy_flagged = sum(1 for f in report.proxy if f.flagged)

        return {
            "representation_flags": rep_flagged,
            "disparity_flags": disp_flagged,
            "proxy_flags": proxy_flagged,
            "total_flags": rep_flagged + disp_flagged + proxy_flagged,
            "overall_risk": (
                "HIGH" if (rep_flagged + disp_flagged + proxy_flagged) > 5
                else "MEDIUM" if (rep_flagged + disp_flagged + proxy_flagged) > 1
                else "LOW"
            ),
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _save(self, report: BiasReport) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(report), fh, indent=2, default=str)
        print(f"[BiasDetectionEngine] Report saved → {self.output_path}")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_bias_detection(
    df: pd.DataFrame,
    sensitive_cols: List[str],
    benchmarks: Optional[Dict[str, Dict[str, float]]] = None,
    p_threshold: float = 0.05,
    proxy_threshold: float = 0.3,
    representation_gap: float = 0.05,
    output_path: Optional[str] = "bias_report.json",
) -> BiasReport:
    """
    One-shot helper — instantiates BiasDetectionEngine and calls run().

    Example
    -------
    >>> report = run_bias_detection(
    ...     df=df,
    ...     sensitive_cols=["Gender", "Age_Group"],
    ...     benchmarks={"Gender": {"Male": 0.5, "Female": 0.5}},
    ...     output_path="reports/bias_report.json",
    ... )
    >>> print(report.summary)
    """
    engine = BiasDetectionEngine(
        df=df,
        sensitive_cols=sensitive_cols,
        benchmarks=benchmarks,
        p_threshold=p_threshold,
        proxy_threshold=proxy_threshold,
        representation_gap=representation_gap,
        output_path=output_path,
    )
    return engine.run()