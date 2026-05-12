"""
fair_transformers.py
====================
Library of scikit-learn-compatible fairness transformers.

Each class inherits from TransformerMixin + BaseEstimator and can be
dropped into a standard sklearn.pipeline.Pipeline.

Classes
-------
InstanceReweighting
    Reweighting transformer. Computes per-sample weights so that each
    demographic group contributes equally to downstream training.
    Stores weights in ``sample_weight_`` for retrieval after fit_transform.

SMOTEResampler
    Thin wrapper around imbalanced-learn's SMOTE that respects a sensitive
    column, oversampling the minority *within* each sensitive group rather
    than globally.

DisparateImpactRemover
    Feature-level bias mitigation. Repairs numeric feature distributions
    across groups toward a common reference (the overall distribution) using
    a quantile-based approach.  ``repair_level`` ∈ [0, 1] controls how
    aggressively features are repaired (0 = no repair, 1 = full repair).

CorrelationSuppressor
    Drops or attenuates features whose correlation with a protected attribute
    exceeds a configurable threshold, reducing proxy variable leakage.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X)


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    denom = n * (min(ct.shape) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# 1. InstanceReweighting
# ---------------------------------------------------------------------------

class InstanceReweighting(BaseEstimator, TransformerMixin):
    """
    Reweighting transformer (Kamiran & Calders, 2012).

    Computes per-sample weights so that every (group, label) cell has the
    weight it *would* have under independence between the sensitive attribute
    and the target.  The transformer passes X through unchanged but stores
    ``sample_weight_`` on the fitted instance for use in downstream steps.

    Parameters
    ----------
    sensitive_col : str
        Column name of the protected attribute in X.
    target_col : str
        Column name of the target variable in X.
    normalise : bool
        If True (default), weights are normalised so they average to 1.

    Attributes
    ----------
    sample_weight_ : np.ndarray, shape (n_samples,)
        Per-sample weights available after fit / fit_transform.

    Example
    -------
    >>> rw = InstanceReweighting(sensitive_col="Gender", target_col="Loan_Status")
    >>> X_out = rw.fit_transform(X_train)
    >>> model.fit(X_out.drop(columns=["Gender","Loan_Status"]),
    ...           X_out["Loan_Status"],
    ...           sample_weight=rw.sample_weight_)
    """

    def __init__(
        self,
        sensitive_col: str,
        target_col: str,
        normalise: bool = True,
    ) -> None:
        self.sensitive_col = sensitive_col
        self.target_col = target_col
        self.normalise = normalise

    def fit(self, X, y=None):
        df = _to_frame(X)
        self._validate_cols(df)

        n = len(df)
        p_s = df[self.sensitive_col].value_counts(normalize=True)   # P(S=s)
        p_y = df[self.target_col].value_counts(normalize=True)       # P(Y=y)

        # Joint P(S=s, Y=y) — observed
        joint = (
            df.groupby([self.sensitive_col, self.target_col])
            .size()
            .div(n)
        )

        # Weight = P(S) * P(Y) / P(S, Y)   [independence assumption]
        weights_map = {}
        for (s, y_val), p_joint in joint.items():
            expected = float(p_s[s]) * float(p_y[y_val])
            weights_map[(s, y_val)] = expected / p_joint if p_joint > 0 else 1.0

        raw_weights = df.apply(
            lambda row: weights_map.get(
                (row[self.sensitive_col], row[self.target_col]), 1.0
            ),
            axis=1,
        ).values.astype(float)

        if self.normalise:
            raw_weights = raw_weights / raw_weights.mean()

        self.sample_weight_ = raw_weights
        self.feature_names_in_ = list(df.columns)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "sample_weight_")
        # Pass-through: weights are accessed via .sample_weight_
        return _to_frame(X)

    def _validate_cols(self, df: pd.DataFrame) -> None:
        for col in [self.sensitive_col, self.target_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in X.")


# ---------------------------------------------------------------------------
# 2. SMOTEResampler
# ---------------------------------------------------------------------------

class SMOTEResampler(BaseEstimator, TransformerMixin):
    """
    Group-aware SMOTE resampler.

    Wraps imbalanced-learn's SMOTE to oversample *within* each sensitive
    group independently, preventing synthetic samples from bridging group
    boundaries and introducing spurious inter-group patterns.

    The sensitive column and target column are temporarily excluded during
    SMOTE and re-attached afterward.

    Parameters
    ----------
    sensitive_col : str
        Protected attribute column (excluded from SMOTE feature space).
    target_col : str
        Target column used as the SMOTE label.
    k_neighbors : int
        Number of nearest neighbours for SMOTE (default 5).
    random_state : int, optional

    Notes
    -----
    Requires ``imbalanced-learn``:  pip install imbalanced-learn
    fit_transform returns a *new* DataFrame that may be larger than the
    input (oversampled).  Because SMOTE changes the number of rows,
    transform() alone (without fit) raises NotImplementedError.

    Example
    -------
    >>> rs = SMOTEResampler(sensitive_col="Gender", target_col="Loan_Status")
    >>> X_resampled = rs.fit_transform(X_train)
    """

    def __init__(
        self,
        sensitive_col: str,
        target_col: str,
        k_neighbors: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        self.sensitive_col = sensitive_col
        self.target_col = target_col
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit(self, X, y=None):
        # Nothing to learn at fit time; actual work happens in fit_transform.
        self.feature_names_in_ = list(_to_frame(X).columns)
        self.is_fitted_ = True
        return self

    def fit_transform(self, X, y=None, **fit_params):
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise ImportError(
                "SMOTEResampler requires imbalanced-learn: pip install imbalanced-learn"
            ) from exc

        df = _to_frame(X)
        for col in [self.sensitive_col, self.target_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in X.")

        feature_cols = [
            c for c in df.columns
            if c not in (self.sensitive_col, self.target_col)
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        resampled_chunks: List[pd.DataFrame] = []

        for group_val, group_df in df.groupby(self.sensitive_col):
            X_g = group_df[feature_cols].values
            y_g = group_df[self.target_col].values

            label_counts = pd.Series(y_g).value_counts()
            if len(label_counts) < 2 or label_counts.min() < self.k_neighbors + 1:
                warnings.warn(
                    f"Group '{group_val}' skipped (too few minority samples for SMOTE).",
                    UserWarning,
                )
                resampled_chunks.append(group_df.reset_index(drop=True))
                continue

            smote = SMOTE(k_neighbors=self.k_neighbors, random_state=self.random_state)
            X_res, y_res = smote.fit_resample(X_g, y_g)

            chunk = pd.DataFrame(X_res, columns=feature_cols)
            chunk[self.target_col] = y_res
            chunk[self.sensitive_col] = group_val

            # Restore non-numeric / non-feature cols as mode (best effort)
            for col in df.columns:
                if col not in chunk.columns:
                    chunk[col] = group_df[col].mode()[0]

            resampled_chunks.append(chunk)

        result = pd.concat(resampled_chunks, ignore_index=True)
        self.feature_names_in_ = list(df.columns)
        self.is_fitted_ = True
        return result

    def transform(self, X, y=None):
        raise NotImplementedError(
            "SMOTEResampler modifies row count and cannot be used in transform-only mode. "
            "Use fit_transform during training."
        )


# ---------------------------------------------------------------------------
# 3. DisparateImpactRemover
# ---------------------------------------------------------------------------

class DisparateImpactRemover(BaseEstimator, TransformerMixin):
    """
    Quantile-based Disparate Impact Remover.

    For each numeric feature (except the sensitive column), the distribution
    within every group is shifted toward the overall (marginal) distribution
    via quantile interpolation.

    ``repair_level`` ∈ [0.0, 1.0] controls the strength:
    - 0.0 → no change
    - 1.0 → full repair (group distributions match the overall distribution)
    - 0.8 → partial repair (recommended default to preserve some predictive signal)

    Parameters
    ----------
    sensitive_col : str
        Protected attribute column.  Left unchanged in the output.
    repair_level : float
        Strength of repair, in [0, 1].  Default 0.8.
    features_to_repair : list[str], optional
        Explicit list of columns to repair.  If None, all numeric columns
        except ``sensitive_col`` are repaired.

    Example
    -------
    >>> dir_ = DisparateImpactRemover(sensitive_col="Gender", repair_level=0.8)
    >>> X_repaired = dir_.fit_transform(X_train)
    """

    def __init__(
        self,
        sensitive_col: str,
        repair_level: float = 0.8,
        features_to_repair: Optional[List[str]] = None,
    ) -> None:
        if not 0.0 <= repair_level <= 1.0:
            raise ValueError("repair_level must be in [0, 1].")
        self.sensitive_col = sensitive_col
        self.repair_level = repair_level
        self.features_to_repair = features_to_repair

    def fit(self, X, y=None):
        df = _to_frame(X)
        if self.sensitive_col not in df.columns:
            raise ValueError(f"sensitive_col '{self.sensitive_col}' not found.")

        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c != self.sensitive_col
        ]
        self.repair_cols_ = (
            [c for c in self.features_to_repair if c in numeric_cols]
            if self.features_to_repair
            else numeric_cols
        )

        # Store the overall (marginal) sorted values per feature
        self.marginal_sorted_: dict = {
            col: np.sort(df[col].dropna().values)
            for col in self.repair_cols_
        }
        self.feature_names_in_ = list(df.columns)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "repair_cols_")
        df = _to_frame(X)
        out = df.copy()

        for col in self.repair_cols_:
            if col not in df.columns:
                continue
            marginal = self.marginal_sorted_[col]

            for group_val in df[self.sensitive_col].unique():
                mask = df[self.sensitive_col] == group_val
                group_vals = df.loc[mask, col].values

                # Quantile-map group values → marginal distribution
                group_sorted_idx = np.argsort(group_vals)
                quantiles = (np.arange(len(group_vals)) + 0.5) / len(group_vals)
                marginal_at_q = np.quantile(marginal, quantiles)

                repaired = np.empty_like(group_vals, dtype=float)
                repaired[group_sorted_idx] = marginal_at_q

                # Blend: repair_level controls how far we move toward marginal
                blended = (
                    self.repair_level * repaired
                    + (1.0 - self.repair_level) * group_vals.astype(float)
                )
                out.loc[mask, col] = blended

        return out


# ---------------------------------------------------------------------------
# 4. CorrelationSuppressor
# ---------------------------------------------------------------------------

class CorrelationSuppressor(BaseEstimator, TransformerMixin):
    """
    Drops features whose correlation with a protected attribute exceeds
    a configurable threshold, reducing proxy variable leakage.

    For numeric features vs a binary sensitive attribute, Pearson |r| is used.
    For categorical features vs any sensitive attribute, Cramér's V is used.

    Parameters
    ----------
    sensitive_col : str
        Protected attribute column.  Always dropped from the output.
    threshold : float
        Correlation threshold above which a feature is removed (default 0.3).
    method : str
        ``'auto'`` (default) selects pearson or cramers_v based on dtype.
        ``'pearson'`` forces Pearson for all numeric features.
        ``'cramers_v'`` forces Cramér's V for all features (encodes numerics).

    Attributes
    ----------
    dropped_features_ : list[str]
        Features removed due to high correlation with the sensitive attribute.
    kept_features_ : list[str]
        Features retained in the output.

    Example
    -------
    >>> cs = CorrelationSuppressor(sensitive_col="Gender", threshold=0.3)
    >>> X_clean = cs.fit_transform(X_train)
    >>> print(cs.dropped_features_)
    """

    def __init__(
        self,
        sensitive_col: str,
        threshold: float = 0.3,
        method: str = "auto",
    ) -> None:
        if method not in ("auto", "pearson", "cramers_v"):
            raise ValueError("method must be 'auto', 'pearson', or 'cramers_v'.")
        self.sensitive_col = sensitive_col
        self.threshold = threshold
        self.method = method

    def fit(self, X, y=None):
        df = _to_frame(X)
        if self.sensitive_col not in df.columns:
            raise ValueError(f"sensitive_col '{self.sensitive_col}' not found.")

        sensitive = df[self.sensitive_col]
        self.correlations_: dict = {}
        self.dropped_features_: List[str] = []
        self.kept_features_: List[str] = []

        for col in df.columns:
            if col == self.sensitive_col:
                continue

            feat = df[col]
            corr = self._compute_corr(feat, sensitive)
            self.correlations_[col] = round(corr, 4)

            if corr > self.threshold:
                self.dropped_features_.append(col)
            else:
                self.kept_features_.append(col)

        self.feature_names_in_ = list(df.columns)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "kept_features_")
        df = _to_frame(X)
        available = [c for c in self.kept_features_ if c in df.columns]
        return df[available]

    def _compute_corr(self, feat: pd.Series, sensitive: pd.Series) -> float:
        feat_is_numeric = pd.api.types.is_numeric_dtype(feat)
        sens_is_binary = sensitive.nunique() == 2

        if self.method == "cramers_v":
            return _cramers_v(feat.astype(str), sensitive.astype(str))

        if self.method == "pearson" or (self.method == "auto" and feat_is_numeric):
            if sens_is_binary:
                from scipy.stats import pointbiserialr
                
                if not pd.api.types.is_numeric_dtype(sensitive):
                    # pd.factorize mappa automaticamente le stringhe in 0 e 1
                    sens_values = pd.factorize(sensitive)[0]
                else:
                    sens_values = sensitive.astype(int)
                
                try:
                    corr, _ = pointbiserialr(sens_values, feat)
                    return float(abs(corr))
                except Exception:
                    return 0.0
            else:
        
                groups = [feat[sensitive == g].dropna().values for g in sensitive.unique()]
                if len(groups) < 2:
                    return 0.0
                from scipy.stats import f_oneway
                try:
                    f, _ = f_oneway(*groups)
                    k, n = len(groups), feat.shape[0]
                    # η² ≈ (F * df_num) / (F * df_num + df_den)
                    eta2 = (f * (k - 1)) / (f * (k - 1) + (n - k))
                    return float(np.sqrt(np.clip(eta2, 0, 1)))
                except Exception:
                    return 0.0

        return _cramers_v(feat.astype(str), sensitive.astype(str))