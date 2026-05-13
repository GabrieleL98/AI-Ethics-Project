"""
calibrator.py
=============
Post-training group-specific calibration to correct prediction inconsistencies
across demographic groups.

Supported calibration methods
------------------------------
- ``'platt'``       — Platt Scaling (logistic regression on raw scores)
- ``'isotonic'``    — Isotonic Regression (non-parametric monotone mapping)
- ``'temperature'`` — Temperature Scaling for neural-network logits (stretch goal)

Usage
-----
    from training.calibrator import GroupFairnessCalibrator

    cal = GroupFairnessCalibrator(method='platt')
    cal.fit(val_scores, val_labels, val_groups)
    proba = cal.predict_proba(test_scores, test_groups)
    preds = cal.predict(test_scores, test_groups, threshold=0.5)

    # Different method per group
    cal = GroupFairnessCalibrator(
        method='platt',
        group_methods={0: 'isotonic', 1: 'temperature'},
    )
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Individual calibrators
# ---------------------------------------------------------------------------

class PlattScaler:
    """
    Platt Scaling: fit a logistic regression on top of raw scores to
    produce calibrated probabilities.
    """

    def __init__(self, C: float = 1.0) -> None:
        self.C = C
        self._lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattScaler":
        self._lr.fit(scores.reshape(-1, 1), labels)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(scores.reshape(-1, 1))[:, 1]


class IsotonicScaler:
    """
    Isotonic Regression calibration: non-parametric monotone mapping from
    raw scores to probabilities.
    """

    def __init__(self) -> None:
        self._ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicScaler":
        self._ir.fit(scores, labels)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        return self._ir.predict(scores).astype(float)


class TemperatureScaler(nn.Module):
    """
    Temperature Scaling for neural-network logits.  (Stretch goal)

    Learns a single scalar temperature T such that
    ``sigmoid(logits / T)`` is calibrated.

    Fit using LBFGS on a validation set.

    Parameters
    ----------
    init_temperature : float, default=1.0
        Starting temperature value.
    lr : float, default=0.01
        LBFGS learning rate.
    max_iter : int, default=100
        Maximum LBFGS iterations.
    """

    def __init__(
        self,
        init_temperature: float = 1.0,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.max_iter = max_iter
        self.temperature = nn.Parameter(
            torch.tensor([init_temperature], dtype=torch.float32)
        )
        self._bce = nn.BCEWithLogitsLoss()

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        """
        Parameters
        ----------
        scores : np.ndarray
            Raw logits from the neural network (pre-sigmoid).
        labels : np.ndarray
            Binary ground-truth labels.
        """
        logits = torch.tensor(scores, dtype=torch.float32)
        targets = torch.tensor(labels, dtype=torch.float32)

        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=self.lr, max_iter=self.max_iter
        )

        def closure():
            optimizer.zero_grad()
            scaled = logits / self.temperature.clamp(min=1e-6)
            loss = self._bce(scaled, targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = torch.tensor(scores, dtype=torch.float32)
            scaled = logits / self.temperature.clamp(min=1e-6)
            return torch.sigmoid(scaled).numpy()

    @property
    def temperature_value(self) -> float:
        return self.temperature.item()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_METHOD_MAP: Dict[str, type] = {
    "platt":       PlattScaler,
    "isotonic":    IsotonicScaler,
    "temperature": TemperatureScaler,
}


# ---------------------------------------------------------------------------
# GroupFairnessCalibrator
# ---------------------------------------------------------------------------

class GroupFairnessCalibrator:
    """
    Post-training calibrator that fits a separate calibrator for each
    demographic group, correcting group-specific prediction inconsistencies.

    Parameters
    ----------
    method : str, default='platt'
        Default calibration method applied to all groups unless overridden.
        One of ``'platt'``, ``'isotonic'``, ``'temperature'``.
    group_methods : dict or None
        Per-group method overrides: ``{group_label: method_name}``.
        Groups not listed here use the default ``method``.

    Examples
    --------
    Same method for all groups:

        cal = GroupFairnessCalibrator(method='isotonic')

    Different method per group:

        cal = GroupFairnessCalibrator(
            method='platt',
            group_methods={0: 'temperature', 2: 'isotonic'},
        )
    """

    def __init__(
        self,
        method: str = "platt",
        group_methods: Optional[Dict] = None,
    ) -> None:
        if method not in _METHOD_MAP:
            raise ValueError(
                f"Unknown method '{method}'. Choose from {list(_METHOD_MAP.keys())}"
            )
        self.method = method
        self.group_methods = group_methods or {}
        self._calibrators: Dict = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_calibrator(self, group):
        method = self.group_methods.get(group, self.method)
        if method not in _METHOD_MAP:
            raise ValueError(
                f"Unknown method '{method}' for group {group}. "
                f"Choose from {list(_METHOD_MAP.keys())}"
            )
        return _METHOD_MAP[method]()

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
    ) -> "GroupFairnessCalibrator":
        """
        Fit a calibrator for each group present in ``groups``.

        Parameters
        ----------
        scores : np.ndarray of shape (N,)
            Raw model scores / logits.
        labels : np.ndarray of shape (N,)
            Binary ground-truth labels.
        groups : np.ndarray of shape (N,)
            Group membership for each sample.

        Returns
        -------
        self
        """
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels)
        groups = np.asarray(groups)

        for g in np.unique(groups):
            mask = groups == g
            n_pos = labels[mask].sum()
            n_neg = mask.sum() - n_pos
            if n_pos == 0 or n_neg == 0:
                import warnings
                warnings.warn(
                    f"Group {g} has no positive or negative samples — "
                    "skipping calibration for this group.",
                    stacklevel=2,
                )
                continue
            calibrator = self._make_calibrator(g)
            calibrator.fit(scores[mask], labels[mask])
            self._calibrators[g] = calibrator

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        scores: np.ndarray,
        groups: np.ndarray,
    ) -> np.ndarray:
        """
        Apply group-specific calibration and return calibrated probabilities.

        Parameters
        ----------
        scores : np.ndarray of shape (N,)
        groups : np.ndarray of shape (N,)

        Returns
        -------
        np.ndarray of shape (N,) — calibrated probabilities in [0, 1].
        """
        scores = np.asarray(scores, dtype=float)
        groups = np.asarray(groups)
        output = np.full(len(scores), np.nan)

        for g, cal in self._calibrators.items():
            mask = groups == g
            if mask.sum() == 0:
                continue
            output[mask] = cal.predict_proba(scores[mask])

        # For groups not seen during fit, fall back to sigmoid-like clipping
        missing = np.isnan(output)
        if missing.any():
            import warnings
            warnings.warn(
                "Some groups were not seen during fit — using raw scores as fallback.",
                stacklevel=2,
            )
            output[missing] = np.clip(scores[missing], 0.0, 1.0)

        return output

    def predict(
        self,
        scores: np.ndarray,
        groups: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Return binary predictions using calibrated probabilities."""
        return (self.predict_proba(scores, groups) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def calibrators_(self) -> Dict:
        """Fitted calibrator instances keyed by group label."""
        return self._calibrators

    def summary(self) -> str:
        lines = ["=== GroupFairnessCalibrator ==="]
        for g, cal in self._calibrators.items():
            method = type(cal).__name__
            extra = ""
            if isinstance(cal, TemperatureScaler):
                extra = f" (T={cal.temperature_value:.4f})"
            lines.append(f"  Group {g}: {method}{extra}")
        return "\n".join(lines)