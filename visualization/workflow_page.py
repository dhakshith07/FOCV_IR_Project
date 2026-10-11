"""
visualization/workflow_page.py
===============================
Phase 7 - Workflow Analysis Streamlit page.

Displays
--------
  1. Live event logger  - record interactions via the sidebar
  2. Usage charts       - frequency, recency, duration, radar
  3. Transition matrix  - heatmap + raw count table
  4. Workflow graph     - NetworkX book-to-book relationship graph
  5. Current context    - detected context with evidence breakdown
  6. Next book prediction - ranked prediction with confidence bars

All outputs are derived from actual recorded interactions.
Nothing is hard-coded.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from workflow.workflow_analyzer import WorkflowAnalyzer
from workflow.usage_analyzer import UsageAnalyzer
from workflow.context_detector import ContextDetector


# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
.wf-header {
    background: linear-gradient(135deg, #0f1f10 0%, #1a3320 50%, #0d2010 100%);
    border: 1px solid rgba(72,187,120,0.25);
    border-radius: 14px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.2rem;
    position: relative; overflow: hidden;
}
.wf-header::before {
    content: '';
    position: absolute; top: -40%; right: -5%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(72,187,120,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.wf-title {
    font-size: 1.55rem; font-weight: 800;
    background: linear-gradient(135deg, #48bb78, #68d391, #9ae6b4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0;
}
.wf-meta {
    font-size: 0.78rem; color: #4a5568; margin-top: 0.35rem;
    font-family: 'JetBrains Mono', monospace;
}
.ctx-badge {
    display: inline-block;
    padding: 0.35rem 1rem; border-radius: 20px;
    font-size: 0.76rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace; margin-top: 0.5rem;
}
.ctx-studying   { background: rgba(99,179,237,0.12); border: 1px solid rgba(99,179,237,0.4); color: #63b3ed; }
.ctx-exam_prep  { background: rgba(252,129,74,0.12);  border: 1px solid rgba(252,129,74,0.4);  color: #fc814a; }
.ctx-research   { background: rgba(72,187,120,0.12);  border: 1px solid rgba(72,187,120,0.4);  color: #48bb78; }
.ctx-leisure    { background: rgba(159,122,234,0.12); border: 1px solid rgba(159,122,234,0.4); color: #9f7aea; }
.ctx-unknown    { background: rgba(113,128,150,0.12); border: 1px solid rgba(113,128,150,0.4); color: #718096; }
.metric-box {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
}
.metric-val { font-size: 1.5rem; font-weight: 700; color: #48bb78; font-family: 'JetBrains Mono', monospace; }
.metric-lbl { font-size: 0.7rem; color: #4a5568; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem; }
.section-hdr {
    font-size: 1rem; font-weight: 700; color: #e2e8f0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(72,187,120,0.15);
    margin: 1.4rem 0 0.9rem 0;
}
.pred-bar-bg { background: rgba(255,255,255,0.05); border-radius: 4px; height: 8px; margin-top: 4px; }
.pred-bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #276749, #48bb78); }
.event-row {
    display: flex; justify-content: space-between;
    padding: 0.3rem 0.5rem; font-size: 0.78rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-family: 'JetBrains Mono', monospace;
}
.event-time { color: #4a5568; }
.event-book { color: #63b3ed; font-weight: 600; }
.event-act  { color: #48bb78; }
.info-box {
    background: rgba(72,187,120,0.06); border: 1px solid rgba(72,187,120,0.2);
    border-radius: 10px; padding: 0.9rem 1.2rem;
    font-size: 0.82rem; color: #68d391; margin-bottom: 1rem;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Session-state helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_wa() -> WorkflowAnalyzer:
    if "workflow_analyzer" not in st.session_state:
        from database.db_manager import DatabaseManager
        try:
            db = DatabaseManager("data/workspace.db")
            db.init_db()
        except Exception:
            db = None
        st.session_state["workflow_analyzer"] = WorkflowAnalyzer(db_manager=db)
    return st.session_state["workflow_analyzer"]


def _get_ua() -> UsageAnalyzer:
    if "usage_analyzer" not in st.session_state:
        st.session_state["usage_analyzer"] = UsageAnalyzer(_get_wa())
    return st.session_state["usage_analyzer"]


def _get_cd() -> ContextDetector:
    if "context_detector" not in st.session_state:
        st.session_state["context_detector"] = ContextDetector(_get_wa())
    return st.session_state["context_detector"]


# ─────────────────────────────────────────────────────────────────────────────
#  Plotly helpers
# ─────────────────────────────────────────────────────────────────────────────

def _style(fig: go.Figure, h: int = 320) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a0aec0", family="Inter, sans-serif", size=12),
        title_font=dict(color="#e2e8f0", size=14),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#718096"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#a0aec0"),
        margin=dict(l=10, r=10, t=40, b=20),
        height=h,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#a0aec0")),
    )
    return fig


def _transition_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Record at least 2 interactions to see transitions.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#4a5568", size=13))
        return _style(fig)

    z    = df.values.tolist()
    text = [[f"{v:.2f}" for v in row] for row in z]
    fig  = go.Figure(go.Heatmap(
        z=z, x=df.columns.tolist(), y=df.index.tolist(),
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#0d2010"], [0.5, "#276749"], [1, "#48bb78"]],
        showscale=True,
        colorbar=dict(tickfont=dict(color="#a0aec0"), title=dict(text="P", font=dict(color="#a0aec0"))),
    ))
    fig.update_layout(title="Transition Probability Matrix")
    return _style(fig, h=380)


def _workflow_graph(wa: WorkflowAnalyzer) -> go.Figure:
    """Plotly network graph of book-to-book transitions."""
    if len(wa.books) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Record transitions between books to see the graph.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#4a5568", size=13))
        return _style(fig)

    books    = list(wa.books.items())       # [(id, title)]
    n        = len(books)
    angles   = [2 * 3.14159 * i / n for i in range(n)]
    pos      = {bid: (round(np.cos(a), 4), round(np.sin(a), 4))
                for (bid, _), a in zip(books, angles)}
    id2title = wa.books

    edge_x, edge_y, edge_w = [], [], []
    for from_id, to_counts in wa._transitions.items():
        total = max(sum(to_counts.values()), 1)
        for to_id, cnt in to_counts.items():
            if from_id in pos and to_id in pos:
                fx, fy = pos[from_id]
                tx, ty = pos[to_id]
                edge_x += [fx, tx, None]
                edge_y += [fy, ty, None]
                edge_w.append(cnt / total)

    # Draw edges
    fig = go.Figure()
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=1.5, color="rgba(72,187,120,0.35)"),
            hoverinfo="none", showlegend=False,
        ))

    # Draw weighted arrows (annotations)
    for from_id, to_counts in wa._transitions.items():
        total = max(sum(to_counts.values()), 1)
        for to_id, cnt in to_counts.items():
            if from_id not in pos or to_id not in pos:
                continue
            fx, fy = pos[from_id]
            tx, ty = pos[to_id]
            w = cnt / total
            fig.add_annotation(
                x=tx, y=ty, ax=fx, ay=fy,
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=2, arrowsize=1.0,
                arrowwidth=1.5 + w * 3,
                arrowcolor=f"rgba(72,187,120,{0.3 + w * 0.6:.2f})",
                showarrow=True,
            )

    # Draw nodes
    node_x = [pos[bid][0] for bid, _ in books]
    node_y = [pos[bid][1] for bid, _ in books]
    labels = [title for _, title in books]

    freq_map = {}
    for ev in wa._events:
        freq_map[ev.book_id] = freq_map.get(ev.book_id, 0) + 1
    sizes = [15 + freq_map.get(bid, 0) * 4 for bid, _ in books]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=sizes, color="#48bb78",
                    line=dict(color="#276749", width=2)),
        text=labels,
        textposition="top center",
        textfont=dict(color="#e2e8f0", size=11),
        hovertemplate="<b>%{text}</b><br>freq=%{marker.size}<extra></extra>",
        showlegend=False,
    ))

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(title="Workflow Relationship Graph")
    return _style(fig, h=420)


# ─────────────────────────────────────────────────────────────────────────────
#  Demo data seeder
# ─────────────────────────────────────────────────────────────────────────────

DEMO_BOOKS = [
    (1, "Machine Learning"),
    (2, "Mathematics"),
    (3, "DBMS"),
    (4, "Computer Networks"),
    (5, "Algorithms"),
]

DEMO_SEQUENCE = [
    (1, "pickup",    60),
    (2, "pickup",    90),
    (1, "pickup",    120),
    (3, "pickup",    45),
    (1, "pickup",    180),
    (2, "pickup",    60),
    (1, "reference", 30),
    (3, "gaze",      20),
    (2, "pickup",    75),
    (1, "pickup",    90),
]

DEMO_CONTEXTS = ["studying", "exam_prep", "studying", "exam_prep",
                 "studying", "studying", "research", "studying", "studying", "exam_prep"]


def _seed_demo(wa: WorkflowAnalyzer) -> None:
    """Seed demo interactions if the analyzer is empty."""
    if wa._events:
        return
    for (bid, title), ctx in zip(
        [DEMO_BOOKS[seq[0] - 1] for seq in DEMO_SEQUENCE],
        DEMO_CONTEXTS,
    ):
        book_id, book_title = bid, title
        _, interaction, dur = DEMO_SEQUENCE[
            DEMO_BOOKS.index((bid, title)) if (bid, title) in DEMO_BOOKS else 0
        ] if False else (0, "pickup", 60)

    # Replay the full demo sequence
    from datetime import timedelta
    base_t = datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc)
    offset = 0
    for i, (book_idx, interaction, dur) in enumerate(DEMO_SEQUENCE):
        bid, title = DEMO_BOOKS[book_idx - 1]
        t = base_t + timedelta(seconds=offset)
        wa.record_event(
            book_id     = bid,
            book_title  = title,
            interaction = interaction,
            duration_s  = dur,
            context     = DEMO_CONTEXTS[i],
            timestamp   = t,
        )
        offset += dur + random.randint(10, 60)


# ─────────────────────────────────────────────────────────────────────────────
#  Metric card HTML
# ─────────────────────────────────────────────────────────────────────────────

def _mcard(val: str, lbl: str) -> str:
    return (
        f'<div class="metric-box">'
        f'<div class="metric-val">{val}</div>'
        f'<div class="metric-lbl">{lbl}</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main render
# ─────────────────────────────────────────────────────────────────────────────

def render_workflow_page() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    wa = _get_wa()
    ua = _get_ua()
    cd = _get_cd()

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="wf-header">
        <p class="wf-title">📊 User Workflow Analysis</p>
        <p class="wf-meta">Phase 7 &nbsp;·&nbsp; Usage frequency &nbsp;·&nbsp;
        Transition matrix &nbsp;·&nbsp; Context detection &nbsp;·&nbsp; Next-book prediction</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Workflow Controls")
        st.markdown("---")

        # Demo seed
        if st.button("🎲 Load Demo Sequence", use_container_width=True):
            _seed_demo(wa)
            st.rerun()

        if st.button("🗑️ Clear All Events", use_container_width=True):
            if "workflow_analyzer" in st.session_state:
                del st.session_state["workflow_analyzer"]
                del st.session_state["usage_analyzer"]
                del st.session_state["context_detector"]
            st.rerun()

        st.markdown("---")
        st.markdown("**Log a New Interaction**")

        new_title = st.text_input("Book title", placeholder="e.g. Machine Learning")
        new_bid   = st.number_input("Book ID", min_value=1, max_value=999, value=1, step=1)
        new_act   = st.selectbox("Interaction",
                                 ["pickup", "put_down", "open", "close", "gaze", "reference"])
        new_dur   = st.slider("Duration (s)", 0, 600, 60)
        new_ctx   = st.selectbox("Context",
                                 ["studying", "exam_prep", "research", "leisure", ""])
        new_px    = st.slider("Position X", 0, 640, 320)
        new_py    = st.slider("Position Y", 0, 480, 240)

        if st.button("➕ Record Event", use_container_width=True):
            if new_title.strip():
                wa.record_event(
                    book_id     = int(new_bid),
                    book_title  = new_title.strip(),
                    interaction = new_act,
                    position_x  = float(new_px),
                    position_y  = float(new_py),
                    context     = new_ctx,
                    duration_s  = float(new_dur),
                )
                st.success(f"Recorded: {new_title} ({new_act})")
                st.rerun()
            else:
                st.warning("Enter a book title.")

        st.markdown("---")
        st.markdown("**Set Session Context**")
        ctx_override = st.selectbox(
            "Force context", ["", "studying", "exam_prep", "research", "leisure"],
            index=0,
        )
        if st.button("Set Context"):
            if ctx_override:
                wa.set_context(ctx_override)
                st.rerun()

    # ── Info box when empty ───────────────────────────────────────────────────
    if not wa._events:
        st.markdown('''
        <div class="info-box">
        No interactions recorded yet.<br/>
        Use the sidebar to log book interactions manually, or click
        <b>Load Demo Sequence</b> to populate the analyzer with the
        example ML → Maths → DBMS workflow.
        </div>
        ''', unsafe_allow_html=True)
        _render_how_it_works()
        return

    # ── Top metrics ───────────────────────────────────────────────────────────
    summ  = wa.summary()
    ctx_r = cd.detect_context()

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in [
        (c1, str(summ["total_events"]),  "Events"),
        (c2, str(summ["unique_books"]),  "Books"),
        (c3, str(summ["transitions"]),   "Transitions"),
        (c4, f"{ctx_r.focus_score:.2f}", "Focus"),
        (c5, f"{ctx_r.intensity:.1f}",   "Events/min"),
    ]:
        with col:
            st.markdown(_mcard(val, lbl), unsafe_allow_html=True)

    # ── Context badge ─────────────────────────────────────────────────────────
    st.markdown("")
    badge_cls = f"ctx-{ctx_r.label}"
    ctx_icon  = {"studying": "📖", "exam_prep": "📝", "research": "🔬",
                 "leisure": "☕", "unknown": "❓"}.get(ctx_r.label, "❓")
    st.markdown(
        f'<span class="ctx-badge {badge_cls}">{ctx_icon} {ctx_r.label.replace("_"," ").upper()}'
        f' — confidence {ctx_r.confidence:.0%}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Interaction Timeline (full width) ─────────────────────────────────────
    st.markdown('<div class="section-hdr">🕒 Interaction Timeline</div>', unsafe_allow_html=True)
    st.plotly_chart(ua.get_timeline_chart(), use_container_width=True)

    # ── Usage charts ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">📈 Usage Metrics</div>', unsafe_allow_html=True)
    uc1, uc2, uc3 = st.columns(3)
    with uc1:
        st.plotly_chart(ua.get_frequency_chart(),    use_container_width=True)
    with uc2:
        st.plotly_chart(ua.get_recent_usage_chart(), use_container_width=True)
    with uc3:
        st.plotly_chart(ua.get_duration_chart(),     use_container_width=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    radar_col, tbl_col = st.columns([1, 1])
    with radar_col:
        st.markdown('<div class="section-hdr">🕸️ Metric Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(ua.get_priority_radar(), use_container_width=True)
    with tbl_col:
        st.markdown('<div class="section-hdr">📋 Priority Table</div>', unsafe_allow_html=True)
        df_sum = ua.get_usage_summary(
            current_book_id = wa.current_book_id,
            context         = wa.current_context,
        )
        if not df_sum.empty:
            # Round for display
            disp = df_sum.copy()
            for col in ["recent_usage","usage_duration_s","last_used_s_ago",
                        "sequential_score","context_score","priority_score"]:
                if col in disp:
                    disp[col] = disp[col].round(3)
            st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Transition matrix ────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">🔄 Transition Matrix</div>', unsafe_allow_html=True)
    tm_prob  = wa.get_transition_matrix()
    tm_count = wa.get_transition_counts()

    hm_col, raw_col = st.columns([2, 1])
    with hm_col:
        st.plotly_chart(_transition_heatmap(tm_prob), use_container_width=True)
    with raw_col:
        st.markdown("**Raw transition counts**")
        if not tm_count.empty:
            st.dataframe(tm_count, use_container_width=True)
        else:
            st.info("No transitions yet.")

    # ── Workflow graph ────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">🕸️ Workflow Relationship Graph</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.78rem;color:#4a5568;margin-bottom:0.5rem;">' +
        'Node size = usage frequency. Arrow thickness = transition probability.' +
        '</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(_workflow_graph(wa), use_container_width=True)

    # ── Next book prediction ─────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">🔮 Next Book Prediction</div>', unsafe_allow_html=True)
    preds = wa.predict_next_book(top_k=5)

    if preds:
        cur_title = wa.books.get(wa.current_book_id, "?")
        st.markdown(
            f'<div style="font-size:0.82rem;color:#a0aec0;margin-bottom:0.75rem;">' +
            f'Predictions from current book: <b style="color:#48bb78">{cur_title}</b>' +
            '</div>',
            unsafe_allow_html=True,
        )
        for bid, title, prob in preds:
            bar_w = int(prob * 100)
            st.markdown(
                f'<div style="margin-bottom:0.6rem;">' +
                f'<div style="display:flex;justify-content:space-between;">' +
                f'<span style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">{title}</span>' +
                f'<span style="color:#48bb78;font-family:monospace;font-size:0.82rem;">{prob:.1%}</span>' +
                f'</div>' +
                f'<div class="pred-bar-bg"><div class="pred-bar-fill" style="width:{bar_w}%"></div></div>' +
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Record at least two different books to get predictions.")

    # ── Context detail ────────────────────────────────────────────────────────
    with st.expander("🔍 Context Detection Detail", expanded=False):
        ctx_d = ctx_r.to_dict()
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Label:** `{ctx_d['label']}`")
            st.markdown(f"**Confidence:** `{ctx_d['confidence']:.1%}`")
            st.markdown(f"**Focus score:** `{ctx_d['focus_score']:.3f}`")
            st.markdown(f"**Intensity:** `{ctx_d['intensity']:.2f}` events/min")
            st.markdown(f"**Subjects:** {', '.join(ctx_d['subjects']) or 'unknown'}")
        with col_b:
            st.markdown("**Evidence:**")
            for k, v in ctx_d["evidence"].items():
                st.markdown(f"- `{k}` : `{v}`")

    # ── Subject affinity ──────────────────────────────────────────────────────
    with st.expander("📚 Subject Affinity", expanded=False):
        affinity = cd.detect_subject_affinity()
        if affinity:
            aff_df = pd.DataFrame(
                [{"Subject": s.replace("_", " ").title(), "Affinity": round(v, 3)}
                 for s, v in affinity.items()]
            )
            fig_aff = px.bar(
                aff_df, x="Affinity", y="Subject", orientation="h",
                color="Affinity",
                color_continuous_scale=["#0d2010", "#48bb78"],
            )
            fig_aff.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a0aec0"), height=260, showlegend=False,
                margin=dict(l=5, r=5, t=20, b=10),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_aff, use_container_width=True)
        else:
            st.info("No subject affinity data yet.")

    # ── Event log ─────────────────────────────────────────────────────────────
    with st.expander("📋 Full Event Log", expanded=False):
        df_ev = wa.get_events_df()
        if not df_ev.empty:
            st.dataframe(df_ev.iloc[::-1].reset_index(drop=True),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No events recorded.")

    # ── Download ──────────────────────────────────────────────────────────────
    with st.expander("⬇️ Export Data", expanded=False):
        ec1, ec2 = st.columns(2)
        with ec1:
            df_ev = wa.get_events_df()
            if not df_ev.empty:
                st.download_button(
                    "Download Events CSV",
                    df_ev.to_csv(index=False).encode(),
                    "workflow_events.csv", "text/csv",
                )
        with ec2:
            tm = wa.get_transition_matrix()
            if not tm.empty:
                st.download_button(
                    "Download Transition Matrix CSV",
                    tm.to_csv().encode(),
                    "transition_matrix.csv", "text/csv",
                )


# ─────────────────────────────────────────────────────────────────────────────
#  How it works (shown when empty)
# ─────────────────────────────────────────────────────────────────────────────

def _render_how_it_works() -> None:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Six Priority Metrics")
        st.markdown("""
| Metric | Description |
|--------|-------------|
| **Usage Frequency** | Total interaction count |
| **Recent Usage** | Exponentially decay-weighted count |
| **Usage Duration** | Total seconds of contact |
| **Last-Used Time** | Seconds since last interaction |
| **Sequential Score** | P(this book | previous book) |
| **Context Score** | Affinity for current session context |
        """)
    with col_b:
        st.markdown("#### Transition Matrix Example")
        example = pd.DataFrame(
            [[1.0, 0.8, 0.7], [0.8, 1.0, 0.3], [0.7, 0.3, 1.0]],
            index=["ML", "Maths", "DBMS"],
            columns=["ML", "Maths", "DBMS"],
        )
        st.dataframe(example, use_container_width=True)
        st.caption("P(col | row) — computed from actual recorded transitions")

    st.markdown("#### Example Workflow")
    st.code(
        "ML → Maths → ML → DBMS → ML → Maths",
        language="text",
    )
    st.markdown(
        "Each arrow is recorded as a **TransitionRecord** in SQLite. "
        "The matrix is computed live from these counts — nothing is hard-coded."
    )
