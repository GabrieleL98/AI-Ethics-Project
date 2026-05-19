"""
monitoring/ab_testing.py
========================
Statistical tools for evaluating fairness interventions via A/B tests.

FairnessABTestAnalyzer
----------------------
- calculate_power()          : statistical power per subgroup
- heterogeneous_effects()    : HTE across intersectional groups with CIs
- mediation_analysis()       : optional causal mediation decomposition
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _proportion_ci(
    successes: int, n: int, alpha: float = 0.05
) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = float(stats.norm.ppf(1 - alpha / 2))
    p_hat = float(successes) / float(n)
    denom = 1 + z**2 / float(n)
    centre = (p_hat + z**2 / (2 * float(n))) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / float(n) + z**2 / (4 * float(n)**2)) / denom
    return (float(centre - margin), float(centre + margin))


def _mean_ci(
    values: np.ndarray, alpha: float = 0.05
) -> Tuple[float, float]:
    """95% CI for a mean using t-distribution."""
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    se = float(stats.sem(values))
    h = se * float(stats.t.ppf(1 - alpha / 2, df=n - 1))    
    m = float(values.mean())
    return (m - h, m + h)


def _dp_gap(preds: np.ndarray, sensitive: np.ndarray) -> float:
    groups = np.unique(sensitive)
    if len(groups) < 2:
        return 0.0
    means = [preds[sensitive == g].mean() for g in groups]
    return float(max(means) - min(means))


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class FairnessABTestAnalyzer:
    """
    Rigorous evaluation of fairness interventions via A/B testing.

    Parameters
    ----------
    control : pd.DataFrame
        Data from the control group. Must contain columns for predictions,
        labels, and sensitive attributes.
    treatment : pd.DataFrame
        Data from the treatment group (same schema as control).
    pred_col : str, default='prediction'
        Column of binary predictions {0, 1}.
    label_col : str, default='label'
        Column of ground-truth labels {0, 1}.
    sensitive_cols : list of str
        Columns identifying sensitive attributes. Multiple columns enable
        intersectional analysis.
    alpha : float, default=0.05
        Significance level for confidence intervals and tests.
    """

    def __init__(
        self,
        control: pd.DataFrame,
        treatment: pd.DataFrame,
        pred_col: str = "prediction",
        label_col: str = "label",
        sensitive_cols: Optional[List[str]] = None,
        alpha: float = 0.05,
    ) -> None:
        self.control = control.copy()
        self.treatment = treatment.copy()
        self.pred_col = pred_col
        self.label_col = label_col
        self.sensitive_cols = sensitive_cols or []
        self.alpha = alpha

    # ------------------------------------------------------------------
    # 1. Statistical power
    # ------------------------------------------------------------------

    def calculate_power(
        self,
        effect_size: float = 0.05,
        metric: str = "accuracy",
    ) -> pd.DataFrame:
        """
        Estimate statistical power for detecting ``effect_size`` in ``metric``
        within each demographic subgroup.

        Parameters
        ----------
        effect_size : float
            Minimum detectable effect (absolute difference in proportions).
        metric : str
            'accuracy' or 'positive_rate'.

        Returns
        -------
        pd.DataFrame with columns: subgroup, n_control, n_treatment, power,
            baseline_rate, detectable_effect.
        """
        rows = []
        subgroups = self._subgroups()

        for label, ctrl_mask, trt_mask in subgroups:
            ctrl = self.control[ctrl_mask]
            trt = self.treatment[trt_mask]

            n_c, n_t = len(ctrl), len(trt)

            if n_c < 5 or n_t < 5:
                rows.append({
                    "subgroup": label, "n_control": n_c,
                    "n_treatment": n_t, "power": float("nan"),
                    "baseline_rate": float("nan"),
                    "detectable_effect": effect_size,
                })
                continue

            # Baseline rate
            if metric == "accuracy":
                baseline = float((ctrl[self.pred_col] == ctrl[self.label_col]).mean())
            else:
                baseline = float(ctrl[self.pred_col].mean())

            # Power via normal approximation for two proportions
            p1 = baseline
            p2 = min(baseline + effect_size, 1.0)
            p_bar = (p1 * n_c + p2 * n_t) / (n_c + n_t)

            se_null = np.sqrt(p_bar * (1 - p_bar) * (1 / n_c + 1 / n_t))
            se_alt = np.sqrt(p1 * (1 - p1) / n_c + p2 * (1 - p2) / n_t)

            if se_null == 0:
                power = float("nan")
            else:
                z_alpha = stats.norm.ppf(1 - self.alpha / 2)
                z_beta = (abs(p2 - p1) - z_alpha * se_null) / (se_alt + 1e-12)
                power = float(stats.norm.cdf(z_beta))

            rows.append({
                "subgroup": label,
                "n_control": n_c,
                "n_treatment": n_t,
                "power": round(power, 4),
                "baseline_rate": round(baseline, 4),
                "detectable_effect": effect_size,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 2. Heterogeneous treatment effects
    # ------------------------------------------------------------------

    def heterogeneous_effects(
        self,
        business_metric: str = "accuracy",
        fairness_metric: str = "positive_rate",
    ) -> pd.DataFrame:
        """
        Compute the change in business and fairness metrics per intersectional
        subgroup, with confidence intervals and significance tests.

        Parameters
        ----------
        business_metric : str
            'accuracy' or 'positive_rate'.
        fairness_metric : str
            'positive_rate', 'tpr', or 'fpr'.

        Returns
        -------
        pd.DataFrame with one row per subgroup.
        """
        rows = []
        subgroups = self._subgroups()

        for label, ctrl_mask, trt_mask in subgroups:
            ctrl = self.control[ctrl_mask]
            trt = self.treatment[trt_mask]

            if len(ctrl) < 5 or len(trt) < 5:
                continue

            # Business metric
            bm_ctrl = self._compute_metric(ctrl, business_metric)
            bm_trt = self._compute_metric(trt, business_metric)
            bm_delta = bm_trt - bm_ctrl
            bm_ci = self._delta_ci(ctrl, trt, business_metric)

            # Fairness metric
            fm_ctrl = self._compute_metric(ctrl, fairness_metric)
            fm_trt = self._compute_metric(trt, fairness_metric)
            fm_delta = fm_trt - fm_ctrl
            fm_ci = self._delta_ci(ctrl, trt, fairness_metric)

            # Two-proportion z-test for business metric significance
            n_c, n_t = len(ctrl), len(trt)
            p_c = bm_ctrl
            p_t = bm_trt
            p_bar = (p_c * n_c + p_t * n_t) / (n_c + n_t) if (n_c + n_t) > 0 else 0.5
            se = np.sqrt(p_bar * (1 - p_bar) * (1 / n_c + 1 / n_t) + 1e-12)
            z = (p_t - p_c) / se
            p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

            rows.append({
                "subgroup": label,
                "n_control": n_c,
                "n_treatment": n_t,
                f"{business_metric}_control": round(bm_ctrl, 4),
                f"{business_metric}_treatment": round(bm_trt, 4),
                f"{business_metric}_delta": round(bm_delta, 4),
                f"{business_metric}_ci_low": round(bm_ci[0], 4),
                f"{business_metric}_ci_high": round(bm_ci[1], 4),
                f"{fairness_metric}_control": round(fm_ctrl, 4),
                f"{fairness_metric}_treatment": round(fm_trt, 4),
                f"{fairness_metric}_delta": round(fm_delta, 4),
                f"{fairness_metric}_ci_low": round(fm_ci[0], 4),
                f"{fairness_metric}_ci_high": round(fm_ci[1], 4),
                "p_value": round(p_value, 4),
                "significant": p_value < self.alpha,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 3. Mediation analysis (stretch goal)
    # ------------------------------------------------------------------

    def mediation_analysis(
        self,
        outcome_col: str,
        mediator_col: str,
        treatment_indicator_col: str,
    ) -> Dict[str, float]:
        """
        Simple Baron-Kenny causal mediation analysis.

        Decomposes the total treatment effect into:
            - Direct effect   (treatment → outcome, controlling for mediator)
            - Indirect effect (treatment → mediator → outcome)
            - Total effect

        Parameters
        ----------
        outcome_col : str
            Column in both DataFrames representing the outcome (e.g. accuracy).
        mediator_col : str
            Column representing the mediator variable (e.g. positive_rate).
        treatment_indicator_col : str
            Column name to use for the treatment dummy (created internally).

        Returns
        -------
        dict with keys: total_effect, direct_effect, indirect_effect,
            proportion_mediated, p_total, p_direct.
        """
        ctrl = self.control.copy()
        trt = self.treatment.copy()
        ctrl[treatment_indicator_col] = 0
        trt[treatment_indicator_col] = 1
        combined = pd.concat([ctrl, trt], ignore_index=True)

        T = combined[treatment_indicator_col].to_numpy(dtype=float)
        Y = combined[outcome_col].to_numpy(dtype=float) 
        M = combined[mediator_col].to_numpy(dtype=float)

        # Step 1: Total effect  Y ~ T
        coeffs_total = np.polyfit(T, Y, 1)
        slope_total = float(coeffs_total[0])

        # p-value for total effect via manual t-test
        y_pred_total = np.polyval(coeffs_total, T)
        residuals_total = Y - y_pred_total
        n = len(T)
        se_total = float(np.sqrt(np.sum(residuals_total**2) / (n - 2) / np.sum((T - T.mean())**2)))
        t_total = slope_total / (se_total + 1e-12)
        p_total = float(2 * (1 - stats.t.cdf(float(abs(t_total)), df=n - 2)))

        # Step 2: T → M
        coeffs_tm = np.polyfit(T, M, 1)
        slope_tm = float(coeffs_tm[0])

        # Step 3: Y ~ T + M  (OLS via normal equations)
        X = np.column_stack([np.ones(len(T)), T.astype(float), M.astype(float)])
        beta = np.zeros(3)
        try:
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
            direct_effect = float(beta[1])
            beta_m = float(beta[2])
        except np.linalg.LinAlgError:
            direct_effect = float("nan")
            beta_m = float("nan")

        indirect_effect = float(slope_tm * beta_m)
        total_effect = float(slope_total)
        proportion_mediated = (
            indirect_effect / total_effect if abs(total_effect) > 1e-9 else float("nan")
        )

        # p-value for direct effect via t-test approximation
        n = len(T)
        residuals = Y - (X @ beta)
        sigma2 = residuals.var()
        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
            se_direct = float(np.sqrt(cov[1, 1]))
            t_direct = direct_effect / (se_direct + 1e-12)
            p_direct = float(2 * (1 - stats.t.cdf(abs(t_direct), df=n - 3)))
        except np.linalg.LinAlgError:
            p_direct = float("nan")

        return {
            "total_effect": round(total_effect, 4),
            "direct_effect": round(direct_effect, 4),
            "indirect_effect": round(indirect_effect, 4),
            "proportion_mediated": round(proportion_mediated, 4) if not np.isnan(proportion_mediated) else float("nan"),
            "p_total": round(p_total, 4),
            "p_direct": round(p_direct, 4) if not np.isnan(p_direct) else float("nan"),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _subgroups(self):
        """
        Yield (label, ctrl_mask, trt_mask) for each intersectional subgroup.
        Falls back to the full dataset if no sensitive columns are specified.
        """
        if not self.sensitive_cols:
            yield ("all", pd.Series([True] * len(self.control)), pd.Series([True] * len(self.treatment)))
            return

        # Combine all sensitive columns into one intersectional label
        def _label(df: pd.DataFrame, cols: List[str]) -> pd.Series:
            return df[cols].astype(str).agg("×".join, axis=1)

        ctrl_labels = _label(self.control, self.sensitive_cols)
        trt_labels = _label(self.treatment, self.sensitive_cols)
        all_labels = sorted(set(ctrl_labels.unique()) | set(trt_labels.unique()))

        for label in all_labels:
            yield (
                label,
                ctrl_labels == label,
                trt_labels == label,
            )

    def _compute_metric(self, df: pd.DataFrame, metric: str) -> float:
        if metric == "accuracy":
            return float((df[self.pred_col] == df[self.label_col]).mean())
        if metric == "positive_rate":
            return float(df[self.pred_col].mean())
        if metric == "tpr":
            pos = df[self.label_col] == 1
            return float(df.loc[pos, self.pred_col].mean()) if pos.sum() > 0 else float("nan")
        if metric == "fpr":
            neg = df[self.label_col] == 0
            return float(df.loc[neg, self.pred_col].mean()) if neg.sum() > 0 else float("nan")
        raise ValueError(f"Unknown metric: {metric}")

    def _delta_ci(
        self, ctrl: pd.DataFrame, trt: pd.DataFrame, metric: str
    ) -> Tuple[float, float]:
        """Bootstrap 95% CI for the difference in metric between groups."""
        rng = np.random.default_rng(42)
        deltas = []
        for _ in range(500):
            c = ctrl.sample(len(ctrl), replace=True, random_state=int(rng.integers(int(1e6))))
            t = trt.sample(len(trt), replace=True, random_state=int(rng.integers(int(1e6))))
            try:
                deltas.append(self._compute_metric(t, metric) - self._compute_metric(c, metric))
            except Exception:
                continue
        if not deltas:
            return (float("nan"), float("nan"))
        low = float(np.percentile(deltas, 2.5))
        high = float(np.percentile(deltas, 97.5))
        return (low, high)