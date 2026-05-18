"""
reductions_wrapper.py
=====================
Scikit-learn compatible wrapper that applies the reductions approach to
conventional ML models using fairlearn's ExponentiatedGradient algorithm.

Usage
-----
    from training.reductions_wrapper import ReductionsWrapper
    from fairlearn.reductions import DemographicParity
    from xgboost import XGBClassifier

    wrapper = ReductionsWrapper(
        estimator=XGBClassifier(),
        constraint=DemographicParity(),
    )
    wrapper.fit(X_train, y_train, sensitive_features=s_train)
    preds = wrapper.predict(X_test)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted
from fairlearn.reductions import ExponentiatedGradient


class ReductionsWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for fairness-constrained training via
    the reductions approach (ExponentiatedGradient).

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Base classifier to be made fair (e.g. XGBClassifier,
        LogisticRegression). Must implement fit/predict.
    constraint : fairlearn.reductions.Moment
        Fairness constraint object (e.g. DemographicParity(),
        EqualizedOdds(), BoundedGroupLoss()).
    eps : float, default=0.01
        Allowed fairness constraint violation tolerance.
    max_iter : int, default=50
        Maximum number of ExponentiatedGradient iterations.
    nu : float or None
        Convergence criterion. If None, fairlearn uses its default.
    eta0 : float, default=2.0
        Initial step size for ExponentiatedGradient.
    sample_weight_name : str, default='sample_weight'
        Name of the sample_weight parameter in the base estimator's fit().
        XGBoost uses 'sample_weight'; sklearn estimators also use this.
    """

    def __init__(
        self,
        estimator,
        constraint,
        eps: float = 0.01,
        max_iter: int = 50,
        nu: float | None = None,
        eta0: float = 2.0,
        sample_weight_name: str = "sample_weight",
    ) -> None:
        self.estimator = estimator
        self.constraint = constraint
        self.eps = eps
        self.max_iter = max_iter
        self.nu = nu
        self.eta0 = eta0
        self.sample_weight_name = sample_weight_name

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        sensitive_features,
    ) -> "ReductionsWrapper":
        """
        Fit the fairness-constrained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sensitive_features : array-like of shape (n_samples,)
            Group labels for the fairness constraint.

        Returns
        -------
        self
        """
        kwargs = {}
        if self.nu is not None:
            kwargs["nu"] = self.nu

        self._reduction = ExponentiatedGradient(
            estimator=clone(self.estimator),
            constraints=self.constraint,
            eps=self.eps,
            max_iter=self.max_iter,
            eta0=self.eta0,
            **kwargs,
        )
        self._reduction.fit(
            X,
            y,
            sensitive_features=sensitive_features,
        )
        self.classes_ = np.unique(y)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X) -> np.ndarray:
        """Return predicted class labels, consistent with predict_proba."""
        check_is_fitted(self)
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities (if base estimator supports it)."""
        check_is_fitted(self)

        # ExponentiatedGradient does not directly support predict_proba, so we implement it here.
        # Calculated as weighted average of the base predictors' probabilities.
        predictions = np.array([
            est.predict_proba(X)
            for est in self._reduction.predictors_
        ])
        weights = np.array(self._reduction.weights_)

        # predictions: shape (n_estimators, n_samples, n_classes)
        # weights:     shape (n_estimators,)
        weighted = np.tensordot(weights, predictions, axes=([0], [0]))
        # weighted: shape (n_samples, n_classes)
        return weighted
   # ------------------------------------------------------------------
# Inspection helpers
# ------------------------------------------------------------------

    @property
    def predictors_(self):
        """List of base estimators trained during the reductions sweep."""
        check_is_fitted(self)
        return self._reduction.predictors_

    @property
    def weights_(self) -> np.ndarray:
        """Mixture weights over the predictor ensemble."""
        check_is_fitted(self)
        return np.array(self._reduction.weights_)

    @property
    def best_estimator_(self):
        """The single estimator with the highest mixture weight."""
        check_is_fitted(self)
        best_idx = np.argmax(self.weights_)
        return self.predictors_[best_idx]

    @property
    def best_weight_(self) -> float:
        """The highest mixture weight in the ensemble."""
        check_is_fitted(self)
        return float(np.max(self.weights_))

    def summary(self) -> str:
        """Human-readable training summary."""
        check_is_fitted(self)
        n_pred = len(self.predictors_)
        top_w = sorted(
            zip(self.weights_, range(n_pred)), reverse=True
        )[:3]
        lines = [
            "=== ReductionsWrapper Summary ===",
            f"Constraint    : {type(self.constraint).__name__}",
            f"eps           : {self.eps}",
            f"Predictors    : {n_pred}",
            f"Top weights   : {[(round(w, 4), i) for w, i in top_w]}",
            f"Best estimator: predictor[{np.argmax(self.weights_)}] "
            f"(weight={self.best_weight_:.4f})",
        ]
        return "\n".join(lines)