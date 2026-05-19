"""
monitoring/drift.py
===================
Drift detection and alert engine for fairness metrics.

Components
----------
AdaptiveThresholdManager
    Tracks false-positive rates and adjusts alert thresholds dynamically.

FairnessDriftAndAlertEngine
    - KS-test drift detection vs. a reference window
    - Multi-scale temporal analysis via wavelet decomposition (PyWavelets)
    - Alert prioritization with severity scores (CRITICAL / HIGH / LOW)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Optional wavelet import (PyWavelets)
# ---------------------------------------------------------------------------
try:
    import pywt  # type: ignore
    _WAVELET_AVAILABLE = True
except ImportError:
    _WAVELET_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """A single fairness alert."""
    timestamp: pd.Timestamp
    metric: str
    severity: str          # 'CRITICAL' | 'HIGH' | 'LOW'
    severity_score: float  # 0–1
    drift_statistic: float
    p_value: float
    message: str
    group: Optional[str] = None
    wavelet_trend: Optional[str] = None   # 'spike' | 'trend' | None


@dataclass
class ThresholdRecord:
    """Historical record for adaptive threshold tuning."""
    alerts_fired: int = 0
    false_positives: int = 0

    @property
    def fp_rate(self) -> float:
        return self.false_positives / max(self.alerts_fired, 1)


# ---------------------------------------------------------------------------
# Adaptive Threshold Manager (stretch goal)
# ---------------------------------------------------------------------------

class AdaptiveThresholdManager:
    """
    Adjusts alert thresholds based on historical false-positive rates.

    Parameters
    ----------
    base_thresholds : dict
        Initial thresholds per metric, e.g. {'demographic_parity': 0.05}.
    target_fp_rate : float, default=0.1
        Desired false-positive rate. If actual FP rate exceeds this, the
        threshold is tightened (raised); if below, it is relaxed.
    adjustment_step : float, default=0.005
        Amount by which to shift thresholds each adjustment cycle.
    """

    def __init__(
        self,
        base_thresholds: Dict[str, float],
        target_fp_rate: float = 0.10,
        adjustment_step: float = 0.005,
    ) -> None:
        self.thresholds = dict(base_thresholds)
        self.target_fp_rate = target_fp_rate
        self.adjustment_step = adjustment_step
        self._records: Dict[str, ThresholdRecord] = {
            m: ThresholdRecord() for m in base_thresholds
        }

    def record_alert(self, metric: str, false_positive: bool) -> None:
        """Log an alert outcome to inform future threshold adjustments."""
        rec = self._records.setdefault(metric, ThresholdRecord())
        rec.alerts_fired += 1
        if false_positive:
            rec.false_positives += 1

    def adjust(self) -> Dict[str, float]:
        """
        Recompute thresholds based on observed FP rates.
        Returns the updated threshold dict.
        """
        for metric, rec in self._records.items():
            if rec.alerts_fired < 10:
                continue  # not enough data yet
            if rec.fp_rate > self.target_fp_rate:
                # Too many false positives → raise threshold (less sensitive)
                self.thresholds[metric] = min(
                    self.thresholds[metric] + self.adjustment_step, 0.5
                )
            elif rec.fp_rate < self.target_fp_rate * 0.5:
                # Very few false positives → lower threshold (more sensitive)
                self.thresholds[metric] = max(
                    self.thresholds[metric] - self.adjustment_step, 0.01
                )
        return self.thresholds

    def get(self, metric: str, default: float = 0.05) -> float:
        return self.thresholds.get(metric, default)

    def summary(self) -> pd.DataFrame:
        rows = []
        for metric, rec in self._records.items():
            rows.append({
                "metric": metric,
                "threshold": self.thresholds.get(metric),
                "alerts_fired": rec.alerts_fired,
                "false_positives": rec.false_positives,
                "fp_rate": round(rec.fp_rate, 3),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Drift & Alert Engine
# ---------------------------------------------------------------------------

class FairnessDriftAndAlertEngine:
    """
    Analyzes tracker output for drift and generates prioritized alerts.

    Parameters
    ----------
    reference_window : int, default=20
        Number of historical rows used as the reference distribution for
        KS-test drift detection.
    ks_alpha : float, default=0.05
        Significance level for the KS test.
    thresholds : dict or None
        Per-metric drift thresholds for alert severity. Defaults provided.
    adaptive : bool, default=False
        If True, wraps thresholds in an AdaptiveThresholdManager.
    group_size_weight : bool, default=True
        If True, boosts severity score for alerts affecting large groups.
    wavelet : str, default='db4'
        Wavelet family used for multi-scale decomposition (requires pywt).
    """

    DEFAULT_THRESHOLDS = {
        "demographic_parity": 0.05,
        "equalized_odds": 0.07,
        "predictive_parity": 0.07,
    }

    SEVERITY_RULES = [
        (0.70, "CRITICAL"),
        (0.40, "HIGH"),
        (0.00, "LOW"),
    ]

    def __init__(
        self,
        reference_window: int = 20,
        ks_alpha: float = 0.05,
        thresholds: Optional[Dict[str, float]] = None,
        adaptive: bool = False,
        group_size_weight: bool = True,
        wavelet: str = "db4",
    ) -> None:
        self.reference_window = reference_window
        self.ks_alpha = ks_alpha
        self.group_size_weight = group_size_weight
        self.wavelet = wavelet

        base = thresholds or self.DEFAULT_THRESHOLDS
        if adaptive:
            self.threshold_manager: Optional[AdaptiveThresholdManager] = (
                AdaptiveThresholdManager(base)
            )
        else:
            self.threshold_manager = None
            self._thresholds = base

        self.alerts: List[Alert] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        history: pd.DataFrame,
        metrics: Optional[List[str]] = None,
    ) -> List[Alert]:
        """
        Run drift detection on the tracker's history DataFrame.

        Parameters
        ----------
        history : pd.DataFrame
            Output of ``RealTimeFairnessTracker.history``.
        metrics : list of str or None
            Metrics to analyze. Defaults to all numeric fairness columns.

        Returns
        -------
        list of Alert
        """
        if history.empty or len(history) < self.reference_window + 1:
            return []

        if metrics is None:
            metrics = [
                c for c in ["demographic_parity", "equalized_odds", "predictive_parity"]
                if c in history.columns
            ]

        new_alerts: List[Alert] = []
        for metric in metrics:
            series = history[metric].dropna()
            if len(series) < self.reference_window + 1:
                continue

            alerts = self._detect_drift(history, series, metric)
            new_alerts.extend(alerts)

        self.alerts.extend(new_alerts)
        return new_alerts

    def alert_summary(self) -> pd.DataFrame:
        if not self.alerts:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "timestamp": a.timestamp,
                "metric": a.metric,
                "group": a.group,
                "severity": a.severity,
                "severity_score": round(a.severity_score, 3),
                "drift_statistic": round(a.drift_statistic, 4),
                "p_value": round(a.p_value, 4),
                "wavelet_trend": a.wavelet_trend,
                "message": a.message,
            }
            for a in self.alerts
        ])

    def get_threshold(self, metric: str) -> float:
        if self.threshold_manager is not None:
            return self.threshold_manager.get(metric)
        return self._thresholds.get(metric, 0.05)

    # ------------------------------------------------------------------
    # Internal: KS drift detection
    # ------------------------------------------------------------------

    def _detect_drift(
        self,
        history: pd.DataFrame,
        series: pd.Series,
        metric: str,
    ) -> List[Alert]:
        reference = series.iloc[:self.reference_window].values
        recent = series.iloc[self.reference_window:].values

        ks_stat, p_value = stats.ks_2samp(reference, recent)
        threshold = self.get_threshold(metric)

        if p_value >= self.ks_alpha and ks_stat < threshold:
            return []  # no significant drift

        # Compute severity score
        severity_score = self._severity_score(
            ks_stat, threshold, p_value, history, metric
        )
        severity = self._classify_severity(severity_score)

        # Wavelet multi-scale analysis
        wavelet_trend = self._wavelet_analysis(series.values)

        # Identify most-affected group if group column present
        group = self._most_affected_group(history, metric)

        ts = series.index[-1] if hasattr(series.index, "max") else pd.Timestamp.utcnow()
        if isinstance(ts, pd.DatetimeIndex):
            ts = ts.max()

        msg = (
            f"[{severity}] Drift detected in '{metric}' "
            f"(KS={ks_stat:.3f}, p={p_value:.4f}, threshold={threshold}). "
            f"Wavelet: {wavelet_trend or 'N/A'}."
        )

        return [Alert(
            timestamp=ts,
            metric=metric,
            severity=severity,
            severity_score=severity_score,
            drift_statistic=ks_stat,
            p_value=p_value,
            message=msg,
            group=group,
            wavelet_trend=wavelet_trend,
        )]

    def _severity_score(
        self,
        ks_stat: float,
        threshold: float,
        p_value: float,
        history: pd.DataFrame,
        metric: str,
    ) -> float:
        """Combine KS magnitude, p-value, and group size into 0–1 score."""
        # KS magnitude relative to threshold (capped at 1)
        magnitude = min(ks_stat / max(threshold, 1e-9), 1.0)
        # p-value component: lower p → higher score
        p_component = 1.0 - min(p_value / self.ks_alpha, 1.0)

        score = 0.6 * magnitude + 0.4 * p_component

        if self.group_size_weight and "group_size" in history.columns:
            max_size = history["group_size"].max()
            if max_size > 0:
                size_factor = history["group_size"].mean() / max_size
                score = min(score * (1 + 0.2 * size_factor), 1.0)

        return float(np.clip(score, 0.0, 1.0))

    def _classify_severity(self, score: float) -> str:
        for cutoff, label in self.SEVERITY_RULES:
            if score >= cutoff:
                return label
        return "LOW"

    # ------------------------------------------------------------------
    # Internal: Wavelet multi-scale analysis
    # ------------------------------------------------------------------

    def _wavelet_analysis(self, signal: np.ndarray) -> Optional[str]:
        """
        Decompose the metric time series with a discrete wavelet transform.
        Returns 'spike' (short-term), 'trend' (long-term), or None.
        """
        if not _WAVELET_AVAILABLE or len(signal) < 8:
            return None

        try:
            coeffs = pywt.wavedec(signal, self.wavelet, level=min(3, pywt.dwt_max_level(len(signal), self.wavelet)))
        except Exception:
            return None

        # Detail coefficients at finest scale → short-term spikes
        fine_energy = float(np.sum(coeffs[1] ** 2)) if len(coeffs) > 1 else 0.0
        # Approximation coefficients → long-term trend
        coarse_energy = float(np.sum(coeffs[0] ** 2))

        if fine_energy == 0 and coarse_energy == 0:
            return None
        ratio = fine_energy / (fine_energy + coarse_energy + 1e-9)

        if ratio > 0.6:
            return "spike"
        if coarse_energy > fine_energy * 2:
            return "trend"
        return None

    # ------------------------------------------------------------------
    # Internal: most-affected group
    # ------------------------------------------------------------------

    def _most_affected_group(
        self, history: pd.DataFrame, metric: str
    ) -> Optional[str]:
        """Return the group with the highest recent metric value."""
        group_cols = [c for c in history.columns if c not in [
            "group_size", "small_group_warning", "positive_rate",
            "tpr", "fpr", "ppv", "window_batches",
            "demographic_parity", "equalized_odds", "predictive_parity",
        ]]
        if not group_cols or metric not in history.columns:
            return None
        group_col = group_cols[0]
        recent = history.tail(self.reference_window)
        group_means = recent.groupby(group_col)[metric].mean()
        if group_means.empty:
            return None
        return str(group_means.idxmax())