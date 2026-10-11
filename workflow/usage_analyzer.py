"""
workflow/usage_analyzer.py
===========================
Phase 7 - Usage analysis layer.

Wraps WorkflowAnalyzer to expose the six metrics as a clean,
Streamlit-ready API.  Also provides:

  - record_event()          single-call event logging
  - get_usage_summary()     per-book metric table (DataFrame)
  - get_frequency_chart()   Plotly bar chart of usage frequency
  - get_timeline_chart()    Plotly timeline of interactions
  - get_duration_chart()    Plotly bar chart of total contact time

This module does NOT hard-code results — every output is derived
from the actual recorded events stored in the WorkflowAnalyzer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from workflow.workflow_analyzer import WorkflowAnalyzer, BookMetrics
from utils.logger import get_logger

log = get_logger(__name__)


class UsageAnalyzer:
    """
    High-level usage analysis facade over WorkflowAnalyzer.

    Parameters
    ----------
    workflow_analyzer : shared WorkflowAnalyzer instance
    """

    def __init__(self, workflow_analyzer: WorkflowAnalyzer) -> None:
        self._wa = workflow_analyzer

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_event(
        self,
        book_id:     int,
        book_title:  str,
        interaction: str,
        position_x:  float = 0.0,
        position_y:  float = 0.0,
        context:     str   = "",
        duration_s:  float = 0.0,
        timestamp:   Optional[datetime] = None,
    ) -> None:
        """Delegate to WorkflowAnalyzer.record_event()."""
        self._wa.record_event(
            book_id     = book_id,
            book_title  = book_title,
            interaction = interaction,
            position_x  = position_x,
            position_y  = position_y,
            context     = context,
            duration_s  = duration_s,
            timestamp   = timestamp,
        )

    # ------------------------------------------------------------------
    # Metric summary
    # ------------------------------------------------------------------

    def get_usage_summary(
        self,
        current_book_id: Optional[int] = None,
        context:         Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of all six metrics for every known book,
        sorted by priority_score descending.

        Columns
        -------
        book_title, usage_frequency, recent_usage, usage_duration_s,
        last_used_s_ago, sequential_score, context_score, priority_score
        """
        metrics = self._wa.compute_book_metrics(
            current_book_id = current_book_id,
            context         = context,
        )
        if not metrics:
            return pd.DataFrame()

        rows = [m.to_dict() for m in metrics]
        df   = pd.DataFrame(rows)
        # Friendly column order
        cols = [
            "book_title", "usage_frequency", "recent_usage",
            "usage_duration_s", "last_used_s_ago",
            "sequential_score", "context_score", "priority_score",
        ]
        return df[[c for c in cols if c in df.columns]]

    # ------------------------------------------------------------------
    # Plotly charts — all data-driven, never hard-coded
    # ------------------------------------------------------------------

    def get_frequency_chart(self) -> go.Figure:
        """Horizontal bar chart: usage frequency per book."""
        df = self.get_usage_summary()
        if df.empty:
            return _empty_fig("No interactions recorded yet.")

        df_sorted = df.sort_values("usage_frequency", ascending=True)
        fig = px.bar(
            df_sorted,
            x="usage_frequency",
            y="book_title",
            orientation="h",
            color="priority_score",
            color_continuous_scale=["#1a3a52", "#22d3ee"],
            labels={"usage_frequency": "Interaction Count", "book_title": ""},
            title="Usage Frequency",
        )
        _style_fig(fig)
        fig.update_coloraxes(colorbar_title="Priority")
        return fig

    def get_recent_usage_chart(self) -> go.Figure:
        """Bar chart: recency-weighted usage score per book."""
        df = self.get_usage_summary()
        if df.empty:
            return _empty_fig("No interactions recorded yet.")

        df_sorted = df.sort_values("recent_usage", ascending=True)
        fig = px.bar(
            df_sorted,
            x="recent_usage",
            y="book_title",
            orientation="h",
            color="recent_usage",
            color_continuous_scale=["#1a3a52", "#48bb78"],
            labels={"recent_usage": "Recency Score", "book_title": ""},
            title="Recent Usage (decay-weighted)",
        )
        _style_fig(fig)
        return fig

    def get_duration_chart(self) -> go.Figure:
        """Bar chart: total contact time per book (seconds)."""
        df = self.get_usage_summary()
        if df.empty:
            return _empty_fig("No interactions recorded yet.")

        df_sorted = df.sort_values("usage_duration_s", ascending=True)
        fig = px.bar(
            df_sorted,
            x="usage_duration_s",
            y="book_title",
            orientation="h",
            color="usage_duration_s",
            color_continuous_scale=["#1a3a52", "#9f7aea"],
            labels={"usage_duration_s": "Total Duration (s)", "book_title": ""},
            title="Total Contact Time",
        )
        _style_fig(fig)
        return fig

    def get_timeline_chart(self) -> go.Figure:
        """
        Scatter/timeline chart: book interactions over time.
        X = timestamp, Y = book_title, colour = interaction type.
        """
        df = self._wa.get_events_df()
        if df.empty:
            return _empty_fig("No interactions recorded yet.")

        color_map = {
            "pickup":    "#22d3ee",
            "put_down":  "#f6ad55",
            "gaze":      "#9f7aea",
            "reference": "#48bb78",
            "open":      "#63b3ed",
            "close":     "#fc8181",
        }

        fig = px.scatter(
            df,
            x="timestamp",
            y="book_title",
            color="interaction",
            color_discrete_map=color_map,
            size_max=10,
            labels={"timestamp": "Time", "book_title": ""},
            title="Interaction Timeline",
        )
        fig.update_traces(marker_size=10)
        _style_fig(fig)
        return fig

    def get_priority_radar(self) -> go.Figure:
        """
        Radar chart overlaying all six metrics for the top-5 books.
        """
        df = self.get_usage_summary()
        if df.empty:
            return _empty_fig("No interactions recorded yet.")

        top5 = df.head(5)
        dims = [
            "usage_frequency", "recent_usage", "usage_duration_s",
            "sequential_score", "context_score",
        ]

        # Normalise each dim to [0,1] for radar display
        norm = top5.copy()
        for d in dims:
            col_max = norm[d].max()
            if col_max > 0:
                norm[d] = norm[d] / col_max

        cats = ["Frequency", "Recency", "Duration", "Sequential", "Context"]
        colors = ["#22d3ee", "#48bb78", "#f6ad55", "#9f7aea", "#fc8181"]

        fig = go.Figure()
        for i, row in norm.iterrows():
            vals = [row[d] for d in dims] + [row[dims[0]]]
            fig.add_trace(go.Scatterpolar(
                r    = vals,
                theta= cats + [cats[0]],
                fill = "toself",
                name = row["book_title"],
                line_color = colors[list(norm.index).index(i) % len(colors)],
                opacity    = 0.7,
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="rgba(255,255,255,0.08)",
                                color="#4a5568"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                                 color="#a0aec0"),
                bgcolor="rgba(0,0,0,0)",
            ),
            title="Book Metric Radar (top-5)",
        )
        _style_fig(fig)
        return fig

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def event_count(self) -> int:
        return len(self._wa._events)

    def known_books(self) -> Dict[int, str]:
        return self._wa.books


# ---------------------------------------------------------------------------
#  Figure styling helpers
# ---------------------------------------------------------------------------

def _style_fig(fig: go.Figure) -> None:
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(color="#a0aec0", family="Inter, sans-serif", size=12),
        title_font    = dict(color="#e2e8f0", size=14),
        xaxis         = dict(gridcolor="rgba(255,255,255,0.05)", color="#718096"),
        yaxis         = dict(gridcolor="rgba(255,255,255,0.05)", color="#a0aec0"),
        margin        = dict(l=10, r=10, t=40, b=20),
        height        = 320,
        legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#a0aec0")),
    )


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color="#4a5568", size=14),
    )
    _style_fig(fig)
    return fig
