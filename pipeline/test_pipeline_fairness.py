"""
tests/test_pipeline_fairness.py
================================
Pytest fairness suite for the Fair Pipeline.

Covers
------
1. BiasDetectionEngine  – representation, disparity, proxy checks
2. InstanceReweighting  – weight properties and pass-through
3. SMOTEResampler       – group-aware oversampling
4. DisparateImpactRemover – repair reduces distributional gap
5. CorrelationSuppressor  – drops high-proxy features
6. End-to-end sklearn Pipeline integration
7. CI/CD threshold assertions (the ones meant to fail the build)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from detection_engine import BiasDetectionEngine, run_bias_detection
from fair_transformers import (
    CorrelationSuppressor,
    DisparateImpactRemover,
    InstanceReweighting,
    SMOTEResampler,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_df(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic loan-like dataset.

    - Gender       : binary sensitive attribute (Male/Female, ~50/50)
    - Age          : numeric, correlated with Gender (proxy)
    - Income       : numeric, weakly correlated
    - Credit_Score : numeric, uncorrelated
    - Loan_Status  : binary target (0/1), higher approval for Male
    """
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], size=n, p=[0.55, 0.45])

    age = np.where(
        gender == "Male",
        rng.normal(38, 8, n),
        rng.normal(34, 8, n),
    ).clip(18, 70)

    income = rng.normal(50_000, 15_000, n).clip(10_000, 120_000)
    credit = rng.normal(650, 80, n).clip(300, 850)

    # Approval: Male 70%, Female 50%
    p_approval = np.where(gender == "Male", 0.70, 0.50)
    loan_status = rng.binomial(1, p_approval)

    return pd.DataFrame({
        "Gender": gender,
        "Age": age.round(1),
        "Income": income.round(0),
        "Credit_Score": credit.round(0),
        "Loan_Status": loan_status,
    })


@pytest.fixture(scope="module")
def df():
    return _make_df()


@pytest.fixture(scope="module")
def engine(df):
    return BiasDetectionEngine(
        df=df,
        sensitive_cols=["Gender"],
        benchmarks={"Gender": {"Male": 0.50, "Female": 0.50}},
        p_threshold=0.05,
        proxy_threshold=0.3,
        representation_gap=0.05,
        output_path=None,   # don't write to disk during tests
    )


@pytest.fixture(scope="module")
def report(engine):
    return engine.run()


# ===========================================================================
# 1. BiasDetectionEngine
# ===========================================================================

class TestBiasDetectionEngine:

    def test_report_has_correct_shape(self, report, df):
        assert report.dataset_shape == df.shape

    def test_representation_keys_match_sensitive_cols(self, report):
        assert "Gender" in report.representation

    def test_representation_findings_cover_all_groups(self, report, df):
        groups_in_data = set(df["Gender"].unique())
        groups_in_report = {f.group for f in report.representation["Gender"]}
        assert groups_in_data == groups_in_report

    def test_representation_rates_sum_to_one(self, report):
        total = sum(f.observed_rate for f in report.representation["Gender"])
        assert abs(total - 1.0) < 1e-3

    def test_representation_flags_imbalanced_group(self, df):
        """A dataset skewed 80/20 should flag both groups."""
        skewed = df.copy()
        skewed["Gender"] = np.where(np.arange(len(df)) < len(df) * 0.8, "Male", "Female")
        eng = BiasDetectionEngine(
            df=skewed,
            sensitive_cols=["Gender"],
            benchmarks={"Gender": {"Male": 0.50, "Female": 0.50}},
            representation_gap=0.05,
            output_path=None,
        )
        r = eng.run()
        flagged = [f for f in r.representation["Gender"] if f.flagged]
        assert len(flagged) >= 1

    def test_disparity_detects_age_difference(self, report):
        """Age is designed to differ across Gender groups → should be flagged."""
        age_findings = [f for f in report.disparity if f.feature == "Age"]
        assert len(age_findings) >= 1
        assert any(f.flagged for f in age_findings)

    def test_disparity_finding_fields(self, report):
        for f in report.disparity:
            assert f.feature_type in ("numeric", "categorical")
            assert f.test in ("anova", "kruskal", "chi2")
            assert 0.0 <= f.p_value <= 1.0

    def test_proxy_detects_age_as_proxy(self, report):
        """Age is correlated with Gender by construction → proxy flag expected."""
        age_proxies = [f for f in report.proxy if f.feature == "Age"]
        assert len(age_proxies) >= 1
        assert any(f.flagged for f in age_proxies)

    def test_proxy_correlation_non_negative(self, report):
        for f in report.proxy:
            assert f.correlation >= 0.0

    def test_summary_keys_present(self, report):
        for key in ("representation_flags", "disparity_flags", "proxy_flags",
                    "total_flags", "overall_risk"):
            assert key in report.summary

    def test_summary_risk_valid(self, report):
        assert report.summary["overall_risk"] in ("LOW", "MEDIUM", "HIGH")

    def test_run_bias_detection_convenience(self, df):
        r = run_bias_detection(df, sensitive_cols=["Gender"], output_path=None)
        assert r.dataset_shape == df.shape

    def test_missing_sensitive_col_warns(self, df):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eng = BiasDetectionEngine(df=df, sensitive_cols=["NonExistent"], output_path=None)
            eng.run()
        assert any("NonExistent" in str(warning.message) for warning in w)


# ===========================================================================
# 2. InstanceReweighting
# ===========================================================================

class TestInstanceReweighting:

    def test_returns_same_shape(self, df):
        rw = InstanceReweighting(sensitive_col="Gender", target_col="Loan_Status")
        out = rw.fit_transform(df)
        assert out.shape == df.shape

    def test_sample_weight_shape(self, df):
        rw = InstanceReweighting(sensitive_col="Gender", target_col="Loan_Status")
        rw.fit(df)
        assert rw.sample_weight_.shape == (len(df),)

    def test_weights_positive(self, df):
        rw = InstanceReweighting(sensitive_col="Gender", target_col="Loan_Status")
        rw.fit(df)
        assert np.all(rw.sample_weight_ > 0)

    def test_weights_average_to_one_when_normalised(self, df):
        rw = InstanceReweighting(sensitive_col="Gender", target_col="Loan_Status", normalise=True)
        rw.fit(df)
        assert abs(rw.sample_weight_.mean() - 1.0) < 1e-6

    def test_weights_differ_across_groups(self, df):
        """Weights should not be uniform when groups have different approval rates."""
        rw = InstanceReweighting(sensitive_col="Gender", target_col="Loan_Status")
        rw.fit(df)
        assert rw.sample_weight_.std() > 0.01

    def test_missing_col_raises(self, df):
        rw = InstanceReweighting(sensitive_col="Missing", target_col="Loan_Status")
        with pytest.raises(ValueError, match="Missing"):
            rw.fit(df)

    def test_not_fitted_raises(self, df):
        from sklearn.exceptions import NotFittedError
        rw = InstanceReweighting(sensitive_col="Gender", target_col="Loan_Status")
        with pytest.raises(NotFittedError):
            rw.transform(df)


# ===========================================================================
# 3. SMOTEResampler
# ===========================================================================

class TestSMOTEResampler:

    @pytest.fixture(scope="class")
    def resampled(self, df):
        try:
            rs = SMOTEResampler(
                sensitive_col="Gender",
                target_col="Loan_Status",
                k_neighbors=3,
                random_state=0,
            )
            return rs.fit_transform(df)
        except ImportError:
            pytest.skip("imbalanced-learn not installed")

    def test_output_is_dataframe(self, resampled):
        assert isinstance(resampled, pd.DataFrame)

    def test_output_has_same_columns(self, df, resampled):
        assert set(df.columns) == set(resampled.columns)

    def test_output_not_smaller_than_input(self, df, resampled):
        assert len(resampled) >= len(df)

    def test_both_groups_still_present(self, resampled):
        assert set(resampled["Gender"].unique()) == {"Male", "Female"}

    def test_transform_raises_not_implemented(self, df):
        try:
            rs = SMOTEResampler(sensitive_col="Gender", target_col="Loan_Status")
            rs.fit(df)
            with pytest.raises(NotImplementedError):
                rs.transform(df)
        except ImportError:
            pytest.skip("imbalanced-learn not installed")


# ===========================================================================
# 4. DisparateImpactRemover
# ===========================================================================

class TestDisparateImpactRemover:

    def test_output_shape_unchanged(self, df):
        dir_ = DisparateImpactRemover(sensitive_col="Gender", repair_level=0.8)
        out = dir_.fit_transform(df)
        assert out.shape == df.shape

    def test_sensitive_col_unchanged(self, df):
        dir_ = DisparateImpactRemover(sensitive_col="Gender", repair_level=0.8)
        out = dir_.fit_transform(df)
        pd.testing.assert_series_equal(out["Gender"], df["Gender"])

    def test_repair_reduces_group_mean_gap(self, df):
        """After full repair, group means for Age should converge."""
        dir_ = DisparateImpactRemover(sensitive_col="Gender", repair_level=1.0)
        out = dir_.fit_transform(df)

        before_gap = abs(
            df[df["Gender"] == "Male"]["Age"].mean()
            - df[df["Gender"] == "Female"]["Age"].mean()
        )
        after_gap = abs(
            out[out["Gender"] == "Male"]["Age"].mean()
            - out[out["Gender"] == "Female"]["Age"].mean()
        )
        assert after_gap < before_gap

    def test_repair_level_zero_is_identity(self, df):
        dir_ = DisparateImpactRemover(sensitive_col="Gender", repair_level=0.0)
        out = dir_.fit_transform(df)
        np.testing.assert_allclose(out["Age"].values, df["Age"].values, rtol=1e-5)

    def test_invalid_repair_level_raises(self):
        with pytest.raises(ValueError, match="repair_level"):
            DisparateImpactRemover(sensitive_col="Gender", repair_level=1.5)

    def test_features_to_repair_subset(self, df):
        dir_ = DisparateImpactRemover(
            sensitive_col="Gender",
            repair_level=1.0,
            features_to_repair=["Age"],
        )
        out = dir_.fit_transform(df)
        # Income should be untouched
        np.testing.assert_allclose(out["Income"].values, df["Income"].values, rtol=1e-5)

    def test_not_fitted_raises(self, df):
        from sklearn.exceptions import NotFittedError
        dir_ = DisparateImpactRemover(sensitive_col="Gender")
        with pytest.raises(NotFittedError):
            dir_.transform(df)


# ===========================================================================
# 5. CorrelationSuppressor
# ===========================================================================

class TestCorrelationSuppressor:

    def test_drops_high_proxy_feature(self, df):
        """Age is correlated with Gender → should be dropped at threshold 0.1."""
        cs = CorrelationSuppressor(sensitive_col="Gender", threshold=0.1)
        out = cs.fit_transform(df)
        assert "Age" not in out.columns

    def test_sensitive_col_not_in_output(self, df):
        cs = CorrelationSuppressor(sensitive_col="Gender", threshold=0.3)
        out = cs.fit_transform(df)
        assert "Gender" not in out.columns

    def test_low_threshold_drops_more(self, df):
        cs_strict = CorrelationSuppressor(sensitive_col="Gender", threshold=0.05)
        cs_loose = CorrelationSuppressor(sensitive_col="Gender", threshold=0.5)
        out_strict = cs_strict.fit_transform(df)
        out_loose = cs_loose.fit_transform(df)
        assert len(out_strict.columns) <= len(out_loose.columns)

    def test_correlations_dict_populated(self, df):
        cs = CorrelationSuppressor(sensitive_col="Gender", threshold=0.3)
        cs.fit(df)
        assert "Age" in cs.correlations_
        assert all(v >= 0 for v in cs.correlations_.values())

    def test_kept_and_dropped_are_disjoint(self, df):
        cs = CorrelationSuppressor(sensitive_col="Gender", threshold=0.3)
        cs.fit(df)
        assert set(cs.kept_features_).isdisjoint(set(cs.dropped_features_))

    def test_transform_only_keeps_fitted_features(self, df):
        cs = CorrelationSuppressor(sensitive_col="Gender", threshold=0.3)
        cs.fit(df)
        out = cs.transform(df)
        assert list(out.columns) == cs.kept_features_

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            CorrelationSuppressor(sensitive_col="Gender", method="invalid")


# ===========================================================================
# 6. End-to-end sklearn Pipeline
# ===========================================================================

class TestSklearnPipelineIntegration:

    def test_dir_suppressor_pipeline_runs(self, df):
        feature_cols = ["Age", "Income", "Credit_Score", "Gender"]
        X = df[feature_cols]
        y = df["Loan_Status"]

        pipe = Pipeline([
            ("dir", DisparateImpactRemover(sensitive_col="Gender", repair_level=0.8)),
            ("cs", CorrelationSuppressor(sensitive_col="Gender", threshold=0.3)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=0, max_iter=200)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_pipeline_predict_proba(self, df):
        X = df[["Age", "Income", "Credit_Score", "Gender"]]
        y = df["Loan_Status"]

        pipe = Pipeline([
            ("dir", DisparateImpactRemover(sensitive_col="Gender", repair_level=0.5)),
            ("cs", CorrelationSuppressor(sensitive_col="Gender", threshold=0.5)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=0, max_iter=200)),
        ])
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ===========================================================================
# 7. CI/CD threshold assertions
#    These tests encode hard fairness requirements for the build.
#    A failure here means the dataset/model does NOT meet fairness standards.
# ===========================================================================

class TestCICDThresholds:

    # --- Representation ---
    def test_no_group_underrepresented_beyond_10pct(self, report):
        """
        CICD GATE: No demographic group may deviate more than 10% from its
        benchmark representation.
        """
        for finding in report.representation.get("Gender", []):
            assert finding.absolute_gap <= 0.10, (
                f"Group '{finding.group}' is underrepresented: "
                f"observed={finding.observed_rate:.2%}, "
                f"benchmark={finding.benchmark_rate:.2%}, "
                f"gap={finding.absolute_gap:.2%}"
            )

    # --- Proxy correlation ---
    def test_no_feature_highly_correlated_with_sensitive(self, report):
        """
        CICD GATE: No non-sensitive feature may have correlation > 0.6
        with a protected attribute.
        """
        MAX_ALLOWED_CORRELATION = 0.6
        violations = [
            f for f in report.proxy
            if f.correlation > MAX_ALLOWED_CORRELATION
        ]
        assert len(violations) == 0, (
            f"Features with correlation > {MAX_ALLOWED_CORRELATION} found:\n"
            + "\n".join(
                f"  {v.feature} ↔ {v.sensitive_attribute}: {v.correlation:.3f}"
                for v in violations
            )
        )

    # --- Disparity: not all features should be flagged ---
    def test_majority_of_features_not_disparate(self, report):
        """
        CICD GATE: At most 70% of features may show statistically significant
        disparity across groups.
        """
        total = len(report.disparity)
        if total == 0:
            return
        flagged_rate = sum(1 for f in report.disparity if f.flagged) / total
        assert flagged_rate <= 0.70, (
            f"{flagged_rate:.0%} of features are disparate (threshold: 70%)"
        )

    # --- DisparateImpactRemover actually reduces disparity ---
    def test_dir_reduces_age_gap(self):
        df = _make_df(n=600)
        dir_ = DisparateImpactRemover(sensitive_col="Gender", repair_level=0.8)
        out = dir_.fit_transform(df)

        gap_before = abs(
            df[df["Gender"] == "Male"]["Age"].mean()
            - df[df["Gender"] == "Female"]["Age"].mean()
        )
        gap_after = abs(
            out[out["Gender"] == "Male"]["Age"].mean()
            - out[out["Gender"] == "Female"]["Age"].mean()
        )
        assert gap_after < gap_before, (
            f"DisparateImpactRemover did not reduce Age gap: "
            f"before={gap_before:.3f}, after={gap_after:.3f}"
        )

    # --- InstanceReweighting produces balanced effective counts ---
    def test_reweighting_balances_group_weights(self):
        df = _make_df(n=600)
        rw = InstanceReweighting(sensitive_col="Gender", target_col="Loan_Status")
        rw.fit(df)

        male_w = rw.sample_weight_[df["Gender"] == "Male"].sum()
        female_w = rw.sample_weight_[df["Gender"] == "Female"].sum()
        ratio = min(male_w, female_w) / max(male_w, female_w)

        assert ratio >= 0.85, (
            f"Reweighting did not balance groups sufficiently: "
            f"Male_sum={male_w:.1f}, Female_sum={female_w:.1f}, ratio={ratio:.3f}"
        )