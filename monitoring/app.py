"""
app.py
======
Interactive Streamlit dashboard for the Fairness Monitoring Module.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from tracker    import RealTimeFairnessTracker
from drift      import FairnessDriftAndAlertEngine
from reporting  import FairnessReportingDashboard
from ab_testing import FairnessABTestAnalyzer
# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fairness Monitoring Dashboard",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _simulate_batch(
    n: int = 200,
    drift: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate a synthetic production batch with optional bias drift."""
    rng = rng or np.random.default_rng()
    sensitive = rng.choice([0, 1], size=n, p=[0.5, 0.5])
    labels = rng.binomial(1, 0.5, size=n)

    # Group 1 gets a gradually increasing positive-rate gap
    base_rate = np.where(sensitive == 0, 0.5, 0.5 + drift)
    predictions = rng.binomial(1, np.clip(base_rate, 0, 1))

    return {
        "predictions": predictions,
        "labels": labels,
        "sensitive": sensitive,
    }


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "tracker" not in st.session_state:
    st.session_state.tracker = RealTimeFairnessTracker(window_size=15, min_group_size=20)
if "engine" not in st.session_state:
    st.session_state.engine = FairnessDriftAndAlertEngine(
        reference_window=10, ks_alpha=0.05, adaptive=True
    )
if "dashboard" not in st.session_state:
    st.session_state.dashboard = FairnessReportingDashboard("FairML Live Monitor")
if "batch_count" not in st.session_state:
    st.session_state.batch_count = 0
if "all_alerts" not in st.session_state:
    st.session_state.all_alerts = []
if "rng" not in st.session_state:
    st.session_state.rng = np.random.default_rng(42)

tracker = st.session_state.tracker
engine = st.session_state.engine
dashboard = st.session_state.dashboard

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
st.sidebar.title("⚖️ Fairness Monitor")
st.sidebar.markdown("---")

drift_level = st.sidebar.slider(
    "Bias Drift Level", min_value=0.0, max_value=0.4, value=0.0, step=0.02,
    help="Simulates increasing disparity in group positive rates."
)
batch_size = st.sidebar.slider("Batch Size", 100, 500, 200, step=50)
auto_run = st.sidebar.checkbox("Auto-ingest batches (1/sec)", value=False)
n_prefill = st.sidebar.number_input("Pre-fill batches", min_value=0, max_value=50, value=15)

if st.sidebar.button("🔄 Pre-fill history"):
    for i in range(int(n_prefill)):
        drift = drift_level * (i / max(n_prefill - 1, 1))
        batch = _simulate_batch(batch_size, drift=drift, rng=st.session_state.rng)
        ts = datetime.utcnow() - timedelta(minutes=int(n_prefill) - i)
        tracker.ingest(**batch, timestamp=ts)
        st.session_state.batch_count += 1
    st.sidebar.success(f"Ingested {int(n_prefill)} batches.")

if st.sidebar.button("➕ Ingest one batch"):
    batch = _simulate_batch(batch_size, drift=drift_level, rng=st.session_state.rng)
    tracker.ingest(**batch)
    st.session_state.batch_count += 1

if st.sidebar.button("🗑️ Reset"):
    tracker.reset()
    st.session_state.all_alerts = []
    st.session_state.batch_count = 0

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Batches ingested:** {st.session_state.batch_count}")

# ---------------------------------------------------------------------------
# Auto-ingest loop
# ---------------------------------------------------------------------------
if auto_run:
    batch = _simulate_batch(batch_size, drift=drift_level, rng=st.session_state.rng)
    tracker.ingest(**batch)
    st.session_state.batch_count += 1
    time.sleep(1)
    st.rerun()

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("⚖️ Real-Time Fairness Monitoring Dashboard")

history = tracker.history
if history.empty:
    st.info("No data yet. Use the sidebar to ingest batches.")
    st.stop()

# Run drift analysis
new_alerts = engine.analyze(history)
st.session_state.all_alerts.extend(new_alerts)
all_alerts = st.session_state.all_alerts

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
latest = history.tail(1)

dp = latest["demographic_parity"].values[0] if "demographic_parity" in latest.columns else float("nan")
eo = latest["equalized_odds"].values[0] if "equalized_odds" in latest.columns else float("nan")
pp = latest["predictive_parity"].values[0] if "predictive_parity" in latest.columns else float("nan")
n_alerts = len(all_alerts)
critical = sum(1 for a in all_alerts if a.severity == "CRITICAL")

col1.metric("Demographic Parity Gap", f"{dp:.4f}", delta=None)
col2.metric("Equalized Odds Gap", f"{eo:.4f}")
col3.metric("Predictive Parity Gap", f"{pp:.4f}")
col4.metric("Active Alerts", n_alerts, delta=f"{critical} CRITICAL" if critical else None,
            delta_color="inverse")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🗺️ Intersectional", "🚨 Alerts", "🧪 A/B Test"])

# ── Tab 1: Trends ─────────────────────────────────────────────────────────
with tab1:
    metric_choice = st.selectbox(
        "Metric",
        ["demographic_parity", "equalized_odds", "predictive_parity"],
        key="trend_metric",
    )
    threshold_val = engine.get_threshold(metric_choice)
    alert_ts = [a.timestamp for a in all_alerts if a.metric == metric_choice]

    group_col = "group" if "group" in history.columns else history.columns[0]
    fig_trend = dashboard.trend_plot(
        history,
        metric=metric_choice,
        group_col=group_col,
        alert_timestamps=alert_ts,
        threshold=threshold_val,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Group Positive Rates")
    if "positive_rate" in history.columns:
        fig_pr = dashboard.trend_plot(
            history, metric="positive_rate", group_col=group_col,
            title="Positive Rate by Group Over Time"
        )
        st.plotly_chart(fig_pr, use_container_width=True)

# ── Tab 2: Intersectional ─────────────────────────────────────────────────
with tab2:
    metric_int = st.selectbox(
        "Metric",
        ["demographic_parity", "equalized_odds", "predictive_parity", "positive_rate"],
        key="int_metric",
    )
    fig_int = dashboard.intersectional_plot(
        history, metric=metric_int, attr1_col=group_col, recent_n=10
    )
    st.plotly_chart(fig_int, use_container_width=True)

    st.subheader("Recent History Table")
    st.dataframe(history.tail(20).reset_index(), use_container_width=True)

# ── Tab 3: Alerts ─────────────────────────────────────────────────────────
with tab3:
    if not all_alerts:
        st.success("✅ No alerts detected.")
    else:
        alert_df = engine.alert_summary()
        st.dataframe(alert_df, use_container_width=True)

    st.subheader("Adaptive Threshold Status")
    if engine.threshold_manager is not None:
        st.dataframe(engine.threshold_manager.summary(), use_container_width=True)

    st.subheader("Generate Markdown Report")
    if st.button("📄 Generate Report"):
        path = dashboard.generate_report(
            history, all_alerts, output_path="fairness_report.md"
        )
        with open(path) as f:
            content = f.read()
        st.download_button("⬇️ Download Report", content, file_name="fairness_report.md")
        with st.expander("Preview"):
            st.markdown(content)

# ── Tab 4: A/B Test ───────────────────────────────────────────────────────
with tab4:
    st.markdown("### Simulate a fairness intervention A/B test")

    col_a, col_b = st.columns(2)
    with col_a:
        n_ctrl = st.slider("Control group size", 200, 2000, 500, step=100)
        ctrl_drift = st.slider("Control drift", 0.0, 0.3, 0.1, step=0.01)
    with col_b:
        n_trt = st.slider("Treatment group size", 200, 2000, 500, step=100)
        trt_drift = st.slider("Treatment drift", 0.0, 0.3, 0.02, step=0.01,
                               help="Lower = fairer intervention")

    if st.button("▶️ Run A/B Analysis"):
        rng = np.random.default_rng(0)
        ctrl_batch = _simulate_batch(n_ctrl, drift=ctrl_drift, rng=rng)
        trt_batch = _simulate_batch(n_trt, drift=trt_drift, rng=rng)

        ctrl_df = pd.DataFrame({
            "prediction": ctrl_batch["predictions"],
            "label": ctrl_batch["labels"],
            "group": ctrl_batch["sensitive"].astype(str),
        })
        trt_df = pd.DataFrame({
            "prediction": trt_batch["predictions"],
            "label": trt_batch["labels"],
            "group": trt_batch["sensitive"].astype(str),
        })

        analyzer = FairnessABTestAnalyzer(
            ctrl_df, trt_df,
            pred_col="prediction", label_col="label",
            sensitive_cols=["group"],
        )

        st.subheader("Statistical Power per Subgroup")
        power_df = analyzer.calculate_power(effect_size=0.05)
        st.dataframe(power_df, use_container_width=True)

        st.subheader("Heterogeneous Treatment Effects")
        hte_df = analyzer.heterogeneous_effects()
        st.dataframe(hte_df, use_container_width=True)