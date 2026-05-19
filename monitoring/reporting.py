"""
monitoring/reporting.py
=======================
Fairness reporting dashboard: Plotly visualisations and Markdown reports.

Visualisations
--------------
1. trend_plot        – fairness metric over time, one line per group
2. intersectional_plot – heatmap of a metric across intersectional subgroups

Reports
-------
generate_report() → writes a human-readable Markdown file summarising the
current fairness status and any active alerts.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class FairnessReportingDashboard:
    """
    Generates Plotly visualisations and Markdown reports from tracker history
    and alert engine output.

    Parameters
    ----------
    title : str
        Organisation / project name shown in report headers.
    """

    def __init__(self, title: str = "Fairness Monitoring Dashboard") -> None:
        self.title = title

    # ------------------------------------------------------------------
    # 1. Trend plot
    # ------------------------------------------------------------------

    def trend_plot(
        self,
        history: pd.DataFrame,
        metric: str = "demographic_parity",
        group_col: str = "group",
        title: Optional[str] = None,
        alert_timestamps: Optional[List[pd.Timestamp]] = None,
        threshold: Optional[float] = None,
    ) -> go.Figure:
        """
        Line chart of ``metric`` over time, one line per demographic group.

        Parameters
        ----------
        history : pd.DataFrame
            Output of ``RealTimeFairnessTracker.history``.
        metric : str
            Column name of the fairness metric to plot.
        group_col : str
            Column identifying the demographic group.
        alert_timestamps : list of Timestamps or None
            If provided, vertical lines are drawn at alert times.
        threshold : float or None
            If provided, a horizontal dashed line marks the alert threshold.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if metric not in history.columns:
            raise ValueError(f"Metric '{metric}' not found in history columns.")

        df = history.reset_index()
        if "timestamp" not in df.columns:
            df = df.rename(columns={df.columns[0]: "timestamp"})

        fig = px.line(
            df,
            x="timestamp",
            y=metric,
            color=group_col if group_col in df.columns else None,
            markers=True,
            title=title or f"{metric.replace('_', ' ').title()} Over Time",
            labels={metric: metric.replace("_", " ").title(), "timestamp": "Time"},
            template="plotly_white",
        )

        # Threshold line
        if threshold is not None:
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold ({threshold})",
                annotation_position="top left",
            )

        # Alert markers
        if alert_timestamps:
            for ts in alert_timestamps:
                fig.add_vline(
                    x=ts,
                    line_dash="dot",
                    line_color="orange",
                    opacity=0.6,
                )

        fig.update_layout(
            legend_title_text="Group",
            hovermode="x unified",
            margin=dict(l=60, r=30, t=60, b=60),
        )
        return fig

    # ------------------------------------------------------------------
    # 2. Intersectional disparity plot
    # ------------------------------------------------------------------

    def intersectional_plot(
        self,
        history: pd.DataFrame,
        metric: str = "demographic_parity",
        attr1_col: str = "group",
        attr2_col: Optional[str] = None,
        title: Optional[str] = None,
        recent_n: int = 5,
    ) -> go.Figure:
        """
        Heatmap or bar chart of ``metric`` across intersectional subgroups.

        If ``attr2_col`` is provided → heatmap (attr1 × attr2).
        Otherwise → bar chart per group (most recent ``recent_n`` batches).

        Parameters
        ----------
        history : pd.DataFrame
        metric : str
        attr1_col : str
            Primary sensitive attribute column.
        attr2_col : str or None
            Secondary sensitive attribute for intersectional analysis.
        recent_n : int
            Number of most recent rows used to compute the subgroup means.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        df = history.reset_index()
        if "timestamp" not in df.columns:
            df = df.rename(columns={df.columns[0]: "timestamp"})

        recent = df.tail(recent_n)

        if attr2_col and attr2_col in recent.columns:
            # ── Heatmap ──────────────────────────────────────────────
            pivot = (
                recent
                .groupby([attr1_col, attr2_col])[metric]
                .mean()
                .unstack(attr2_col)
            )
            fig = go.Figure(
                go.Heatmap(
                    z=pivot.values,
                    x=[str(c) for c in pivot.columns],
                    y=[str(r) for r in pivot.index],
                    colorscale="RdYlGn_r",
                    colorbar=dict(title=metric.replace("_", " ").title()),
                    text=np.round(pivot.values, 3),
                    texttemplate="%{text}",
                )
            )
            fig.update_layout(
                title=title or f"Intersectional {metric.replace('_', ' ').title()}",
                xaxis_title=attr2_col,
                yaxis_title=attr1_col,
                template="plotly_white",
            )
        else:
            # ── Bar chart ─────────────────────────────────────────────
            group_means = (
                recent.groupby(attr1_col)[metric].mean().reset_index()
            )
            fig = px.bar(
                group_means,
                x=attr1_col,
                y=metric,
                color=metric,
                color_continuous_scale="RdYlGn_r",
                title=title or f"{metric.replace('_', ' ').title()} by Group",
                labels={metric: metric.replace("_", " ").title()},
                template="plotly_white",
            )
            fig.update_layout(coloraxis_showscale=False)

        return fig

    # ------------------------------------------------------------------
    # 3. Multi-metric summary plot
    # ------------------------------------------------------------------

    def summary_plot(
        self,
        history: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        group_col: str = "group",
    ) -> go.Figure:
        """
        Subplot grid: one trend chart per metric, all groups overlaid.
        """
        metrics = metrics or [
            c for c in ["demographic_parity", "equalized_odds", "predictive_parity"]
            if c in history.columns
        ]
        df = history.reset_index()

        fig = make_subplots(
            rows=len(metrics), cols=1,
            shared_xaxes=True,
            subplot_titles=[m.replace("_", " ").title() for m in metrics],
            vertical_spacing=0.08,
        )
        colors = px.colors.qualitative.Plotly
        groups = df[group_col].unique() if group_col in df.columns else ["all"]

        for row_idx, metric in enumerate(metrics, start=1):
            if metric not in df.columns:
                continue
            for g_idx, g in enumerate(groups):
                g_df = df[df[group_col] == g] if group_col in df.columns else df
                fig.add_trace(
                    go.Scatter(
                        x=g_df.get("timestamp", g_df.index),
                        y=g_df[metric],
                        mode="lines+markers",
                        name=str(g),
                        line=dict(color=colors[g_idx % len(colors)]),
                        showlegend=(row_idx == 1),
                    ),
                    row=row_idx, col=1,
                )

        fig.update_layout(
            title=self.title,
            template="plotly_white",
            height=300 * len(metrics),
            hovermode="x unified",
        )
        return fig

    # ------------------------------------------------------------------
    # 4. Automated Markdown report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        history: pd.DataFrame,
        alerts,
        output_path: str = "fairness_report.md",
        metrics: Optional[List[str]] = None,
        group_col: str = "group",
    ) -> str:
        """
        Write a human-readable Markdown report summarising fairness status.

        Parameters
        ----------
        history : pd.DataFrame
            Tracker history.
        alerts : list of Alert
            Active alerts from the drift engine.
        output_path : str
            Where to save the .md file.
        metrics : list of str or None

        Returns
        -------
        str : path to the written file.
        """
        metrics = metrics or [
            c for c in ["demographic_parity", "equalized_odds", "predictive_parity"]
            if c in history.columns
        ]
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        n_batches = len(history)
        n_alerts = len(alerts)
        critical = sum(1 for a in alerts if a.severity == "CRITICAL")
        high = sum(1 for a in alerts if a.severity == "HIGH")

        lines = [
            f"# {self.title}",
            f"*Generated: {now}*",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"| | |",
            f"|---|---|",
            f"| **Batches processed** | {n_batches} |",
            f"| **Active alerts** | {n_alerts} |",
            f"| **Critical alerts** | {critical} |",
            f"| **High alerts** | {high} |",
            "",
        ]

        # Overall status
        if critical > 0:
            lines += ["### ⛔ Status: CRITICAL — Immediate action required.", ""]
        elif high > 0:
            lines += ["### ⚠️ Status: HIGH — Review recommended.", ""]
        else:
            lines += ["### ✅ Status: NOMINAL — No significant drift detected.", ""]

        # Current metric values
        lines += ["## Current Fairness Metrics", ""]
        if not history.empty:
            latest = history.tail(1)
            for metric in metrics:
                if metric in latest.columns:
                    val = latest[metric].values[0]
                    lines.append(f"- **{metric.replace('_', ' ').title()}**: `{val:.4f}`")
            lines.append("")

        # Per-group positive rates
        if group_col in history.columns and "positive_rate" in history.columns:
            lines += ["## Group Positive Rates (latest window)", ""]
            lines += ["| Group | Positive Rate | Group Size |", "|---|---|---|"]
            recent = history.tail(10)
            for g, gdf in recent.groupby(group_col):
                pr = gdf["positive_rate"].mean()
                sz = int(gdf["group_size"].mean())
                lines.append(f"| {g} | {pr:.4f} | {sz} |")
            lines.append("")

        # Alerts
        lines += ["## Active Alerts", ""]
        if not alerts:
            lines.append("_No alerts at this time._")
        else:
            lines += ["| Time | Metric | Severity | KS Stat | p-value | Message |",
                      "|---|---|---|---|---|---|"]
            for a in sorted(alerts, key=lambda x: x.severity_score, reverse=True):
                ts = str(a.timestamp)[:19]
                lines.append(
                    f"| {ts} | {a.metric} | **{a.severity}** | "
                    f"{a.drift_statistic:.4f} | {a.p_value:.4f} | {a.message} |"
                )
        lines.append("")

        # Recommendations
        lines += ["## Recommendations", ""]
        if critical > 0:
            lines.append("- 🔴 **Investigate immediately**: critical drift in fairness metrics detected.")
            lines.append("- Consider rolling back the model or applying a post-processing correction.")
        if high > 0:
            lines.append("- 🟠 **Schedule review**: fairness metrics are degrading.")
            lines.append("- Run a fresh audit and consider retraining with updated data.")
        if n_alerts == 0:
            lines.append("- 🟢 Continue monitoring. No action required at this time.")
        lines.append("")
        lines.append("---")
        lines.append("*Report generated automatically by FairnessReportingDashboard.*")

        report = "\n".join(lines)
        Path(output_path).write_text(report, encoding="utf-8")
        return output_path