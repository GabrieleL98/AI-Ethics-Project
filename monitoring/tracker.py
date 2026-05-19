"""
monitoring/tracker.py
=====================
Real-time fairness tracker that ingests production prediction batches and
computes fairness metrics over configurable sliding windows.

Metrics computed
----------------
- demographic_parity : |P(ŷ=1|A=a) - P(ŷ=1|A=b)|  for every group pair
- equalized_odds     : max of |TPR_a - TPR_b| and |FPR_a - FPR_b|
- predictive_parity  : |PPV_a - PPV_b|

Output
------
A pandas DataFrame with DatetimeIndex, one row per batch, columns:
    timestamp, group, demographic_parity, equalized_odds, predictive_parity,
    group_size, positive_rate
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _positive_rate(preds: np.ndarray) -> float:
    return float(preds.mean()) if len(preds) > 0 else float("nan")


def _tpr(preds: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    return float(preds[pos].mean()) if pos.sum() > 0 else float("nan")


def _fpr(preds: np.ndarray, labels: np.ndarray) -> float:
    neg = labels == 0
    return float(preds[neg].mean()) if neg.sum() > 0 else float("nan")


def _ppv(preds: np.ndarray, labels: np.ndarray) -> float:
    pred_pos = preds == 1
    return float(labels[pred_pos].mean()) if pred_pos.sum() > 0 else float("nan")


def _max_pairwise(values: Dict[str, float]) -> float:
    """Max absolute pairwise difference among group values."""
    vals = [v for v in values.values() if not np.isnan(v)]
    if len(vals) < 2:
        return float("nan")
    return float(max(abs(a - b) for i, a in enumerate(vals) for b in vals[i + 1:]))


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class RealTimeFairnessTracker:
    """
    Ingests production prediction batches and computes fairness metrics
    over a configurable sliding window.

    Parameters
    ----------
    window_size : int, default=10
        Number of batches retained in the sliding window.
    metrics : list of str or None
        Subset of ['demographic_parity', 'equalized_odds', 'predictive_parity'].
        Defaults to all three.
    min_group_size : int, default=10
        Batches where any group has fewer than this many samples are flagged
        but still stored (metrics will be NaN for that group).

    Attributes
    ----------
    history : pd.DataFrame
        Full time-series of all ingested batches and computed metrics.
    """

    SUPPORTED_METRICS = ["demographic_parity", "equalized_odds", "predictive_parity"]

    def __init__(
        self,
        window_size: int = 10,
        metrics: Optional[List[str]] = None,
        min_group_size: int = 10,
    ) -> None:
        self.window_size = window_size
        self.metrics = metrics or self.SUPPORTED_METRICS
        self.min_group_size = min_group_size

        # Sliding window: each entry is a dict of raw arrays for one batch
        self._window: Deque[Dict] = deque(maxlen=window_size)
        self._records: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        sensitive: np.ndarray,
        timestamp: Optional[datetime] = None,
        sensitive_name: str = "group",
    ) -> pd.DataFrame:
        """
        Ingest one batch of production data and compute metrics.

        Parameters
        ----------
        predictions : np.ndarray, shape (n,)
            Binary predictions {0, 1}.
        labels : np.ndarray, shape (n,)
            Ground-truth labels {0, 1}.
        sensitive : np.ndarray, shape (n,)
            Sensitive attribute values (any hashable type).
        timestamp : datetime or None
            Defaults to now.
        sensitive_name : str
            Name of the sensitive attribute (used in output columns).

        Returns
        -------
        pd.DataFrame
            Metrics for this batch, one row per group.
        """
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        sensitive = np.asarray(sensitive)
        ts = timestamp or datetime.utcnow()

        self._window.append({
            "predictions": predictions,
            "labels": labels,
            "sensitive": sensitive,
            "timestamp": ts,
        })

        rows = self._compute_window_metrics(ts, sensitive_name)
        self._records.extend(rows)
        return pd.DataFrame(rows).set_index("timestamp")

    @property
    def history(self) -> pd.DataFrame:
        """Full time-series DataFrame of all ingested batches."""
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame(self._records).set_index("timestamp")

    def latest(self) -> pd.DataFrame:
        """Metrics from the most recent window only."""
        if not self._records:
            return pd.DataFrame()
        df = self.history
        latest_ts = df.index.max()
        return df.loc[df.index == latest_ts]

    def reset(self) -> None:
        """Clear all stored data."""
        self._window.clear()
        self._records.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_window_metrics(
        self, timestamp: datetime, sensitive_name: str
    ) -> List[Dict]:
        """Aggregate the sliding window and compute per-group metrics."""
        # Concatenate all batches in the window
        all_preds = np.concatenate([b["predictions"] for b in self._window])
        all_labels = np.concatenate([b["labels"] for b in self._window])
        all_sensitive = np.concatenate([b["sensitive"] for b in self._window])

        groups = np.unique(all_sensitive)

        # Per-group positive rates, TPR, FPR, PPV
        pr: Dict[str, float] = {}
        tpr: Dict[str, float] = {}
        fpr: Dict[str, float] = {}
        ppv: Dict[str, float] = {}
        sizes: Dict[str, int] = {}

        for g in groups:
            mask = all_sensitive == g
            p = all_preds[mask]
            l = all_labels[mask]
            sizes[str(g)] = int(mask.sum())
            pr[str(g)] = _positive_rate(p)
            tpr[str(g)] = _tpr(p, l)
            fpr[str(g)] = _fpr(p, l)
            ppv[str(g)] = _ppv(p, l)

        # Aggregate gap metrics
        dp = _max_pairwise(pr) if "demographic_parity" in self.metrics else float("nan")
        eo = (
            max(_max_pairwise(tpr), _max_pairwise(fpr))
            if "equalized_odds" in self.metrics
            else float("nan")
        )
        pp = _max_pairwise(ppv) if "predictive_parity" in self.metrics else float("nan")

        rows = []
        for g in groups:
            g_str = str(g)
            small = sizes[g_str] < self.min_group_size
            rows.append({
                "timestamp": timestamp,
                sensitive_name: g_str,
                "group_size": sizes[g_str],
                "small_group_warning": small,
                "positive_rate": pr[g_str],
                "tpr": tpr[g_str],
                "fpr": fpr[g_str],
                "ppv": ppv[g_str],
                "demographic_parity": dp,
                "equalized_odds": eo,
                "predictive_parity": pp,
                "window_batches": len(self._window),
            })
        return rows