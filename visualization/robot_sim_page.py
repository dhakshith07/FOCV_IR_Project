"""
visualization/robot_sim_page.py
================================
Phase 12 — Streamlit page for the robotic arm simulation.

Renders:
  • 3-D Plotly scene  (arm skeleton, gripper, table, books, obstacles)
  • Top-down 2-D view (X-Y plan)
  • Joint angle bar chart
  • IK diagnostics panel
  • Task step progress list
  • Control buttons: Start | Pause | Reset | Step | Run
  • Real-time simulation log
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Lazy simulator import ─────────────────────────────────────────────────────

def _get_simulator():
    from simulation.robot_simulator import RobotSimulator
    return RobotSimulator


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotly helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _rgba_str(c) -> str:
    r, g, b = int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
    a = c[3] if len(c) > 3 else 1.0
    return f"rgba({r},{g},{b},{a:.2f})"


def _make_box_mesh(cx, cy, cz, hw, hd, hh, color):
    """Axis-aligned box as Plotly Mesh3d."""
    x = [cx-hw, cx+hw, cx+hw, cx-hw, cx-hw, cx+hw, cx+hw, cx-hw]
    y = [cy-hd, cy-hd, cy+hd, cy+hd, cy-hd, cy-hd, cy+hd, cy+hd]
    z = [cz-hh, cz-hh, cz-hh, cz-hh, cz+hh, cz+hh, cz+hh, cz+hh]
    i = [0, 0, 1, 1, 2, 2, 4, 4, 5, 5, 6, 6]
    j = [1, 2, 2, 3, 3, 0, 5, 6, 6, 7, 7, 4]
    k = [2, 3, 3, 0, 0, 1, 6, 7, 7, 4, 4, 5]
    c_str = _rgba_str(color)
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=c_str, opacity=color[3] if len(color) > 3 else 1.0,
        flatshading=True, showscale=False, hoverinfo="skip",
    )


def _make_cylinder_scatter(cx, cy, cz, r, h, color, n=18):
    """Approximate cylinder as vertical line strips."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs, ys, zs = [], [], []
    for t in theta:
        xs += [cx + r * np.cos(t), cx + r * np.cos(t), None]
        ys += [cy + r * np.sin(t), cy + r * np.sin(t), None]
        zs += [cz, cz + h, None]
    # Top ring
    top_x = (cx + r * np.cos(theta)).tolist() + [cx + r * np.cos(theta[0])]
    top_y = (cy + r * np.sin(theta)).tolist() + [cy + r * np.sin(theta[0])]
    top_z = [cz + h] * (n + 1)
    xs += top_x + [None]
    ys += top_y + [None]
    zs += top_z + [None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=_rgba_str(color), width=5),
        hoverinfo="skip", showlegend=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_3d_figure(snap: dict) -> go.Figure:
    """Build the main 3-D Plotly scene."""
    traces = []

    # ── Table ──────────────────────────────────────────────────────────────────
    t = snap["table"]
    traces.append(_make_box_mesh(
        t["position"][0], t["position"][1], t["position"][2],
        t["width"] / 2, t["depth"] / 2, t["height"] / 2,
        t["color"],
    ))

    # ── Obstacles ──────────────────────────────────────────────────────────────
    for obs in snap["obstacles"]:
        traces.append(_make_cylinder_scatter(
            obs["position"][0], obs["position"][1], obs["position"][2],
            obs["radius"], obs["height"], obs["color"],
        ))

    # ── Books ──────────────────────────────────────────────────────────────────
    active_id = snap.get("active_book")
    for book in snap["books"]:
        c = book["color"]
        s = book["size"]
        p = book["position"]
        is_active = (book["id"] == active_id)
        opacity   = 0.55 if book["is_held"] else (1.0 if is_active else 0.85)
        traces.append(_make_box_mesh(
            p[0], p[1], p[2],
            s[0] / 2, s[1] / 2, s[2] / 2,
            (c[0], c[1], c[2], opacity),
        ))
        # Book label
        traces.append(go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p[2] + s[2] / 2 + 0.025],
            mode="text",
            text=[book["title"][:14]],
            textfont=dict(color="white", size=9),
            hoverinfo="skip", showlegend=False,
        ))
        # Target marker
        if book["target"] and not book["is_held"]:
            tp = book["target"]
            traces.append(go.Scatter3d(
                x=[tp[0]], y=[tp[1]], z=[tp[2]],
                mode="markers",
                marker=dict(size=6, color=_rgba_str(c), symbol="cross"),
                hoverinfo="skip", showlegend=False,
            ))

    # ── Arm skeleton ───────────────────────────────────────────────────────────
    joints = snap["joint_positions"]
    jx = [p[0] for p in joints]
    jy = [p[1] for p in joints]
    jz = [p[2] for p in joints]

    traces.append(go.Scatter3d(
        x=jx, y=jy, z=jz,
        mode="lines+markers",
        line=dict(color="#63b3ed", width=9),
        marker=dict(size=6, color="#90cdf4",
                    line=dict(color="#4299e1", width=1)),
        name="Arm", showlegend=True,
        hovertemplate="x=%{x:.3f}m<br>y=%{y:.3f}m<br>z=%{z:.3f}m<extra></extra>",
    ))

    # Base sphere
    traces.append(go.Scatter3d(
        x=[jx[0]], y=[jy[0]], z=[jz[0]],
        mode="markers",
        marker=dict(size=14, color="#2b6cb0",
                    line=dict(color="#63b3ed", width=2)),
        name="Base", showlegend=False, hoverinfo="skip",
    ))

    # End-effector
    ee = snap["ee_pos"]
    traces.append(go.Scatter3d(
        x=[ee[0]], y=[ee[1]], z=[ee[2]],
        mode="markers",
        marker=dict(size=11, color="#f6ad55", symbol="diamond",
                    line=dict(color="#ecc94b", width=2)),
        name="End-Effector", showlegend=True,
        hovertemplate="EE: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
    ))

    # ── Gripper fingers ────────────────────────────────────────────────────────
    fp = snap["finger_positions"]
    gripper_state = snap["gripper"]["state"]
    g_color = "#48bb78" if gripper_state == "closed" else (
        "#f6ad55" if gripper_state == "moving" else "#a0aec0"
    )
    for base_key, tip_key in [("left_base", "left_tip"), ("right_base", "right_tip")]:
        b_pos  = fp[base_key]
        t_pos  = fp[tip_key]
        traces.append(go.Scatter3d(
            x=[b_pos[0], t_pos[0]],
            y=[b_pos[1], t_pos[1]],
            z=[b_pos[2], t_pos[2]],
            mode="lines",
            line=dict(color=g_color, width=7),
            hoverinfo="skip", showlegend=False,
        ))

    # ── EE trajectory ──────────────────────────────────────────────────────────
    hist = snap.get("ee_history", [])
    if len(hist) > 1:
        traces.append(go.Scatter3d(
            x=[p[0] for p in hist],
            y=[p[1] for p in hist],
            z=[p[2] for p in hist],
            mode="lines",
            line=dict(color="rgba(159,122,234,0.45)", width=2, dash="dot"),
            name="EE trajectory", showlegend=True, hoverinfo="skip",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(10,14,26,1)",
            xaxis=dict(title="X (m)", range=[-0.75, 0.75],
                       gridcolor="rgba(255,255,255,0.05)", color="#718096"),
            yaxis=dict(title="Y (m)", range=[-0.15, 0.95],
                       gridcolor="rgba(255,255,255,0.05)", color="#718096"),
            zaxis=dict(title="Z (m)", range=[-0.05, 0.70],
                       gridcolor="rgba(255,255,255,0.05)", color="#718096"),
            camera=dict(eye=dict(x=1.6, y=-1.1, z=1.0)),
            aspectmode="manual",
            aspectratio=dict(x=1.4, y=1.0, z=0.8),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(color="#a0aec0", family="Inter"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#a0aec0", size=10),
        ),
        height=540,
    )
    return fig


def build_topdown_figure(snap: dict) -> go.Figure:
    """Top-down 2-D plan view."""
    traces = []

    # Table outline
    t = snap["table"]
    hw, hd = t["width"] / 2, t["depth"] / 2
    cy = t["position"][1]
    traces.append(go.Scatter(
        x=[-hw, hw, hw, -hw, -hw],
        y=[cy - hd, cy - hd, cy + hd, cy + hd, cy - hd],
        mode="lines", fill="toself",
        fillcolor="rgba(139,100,64,0.25)",
        line=dict(color="rgba(139,100,64,0.6)", width=1.5),
        name="Table", showlegend=False,
    ))

    # Books
    active_id = snap.get("active_book")
    for book in snap["books"]:
        p, s, c = book["position"], book["size"], book["color"]
        is_active = (book["id"] == active_id)
        fill_opacity = 0.9 if (book["is_held"] or is_active) else 0.55
        traces.append(go.Scatter(
            x=[p[0] - s[0]/2, p[0] + s[0]/2, p[0] + s[0]/2, p[0] - s[0]/2, p[0] - s[0]/2],
            y=[p[1] - s[1]/2, p[1] - s[1]/2, p[1] + s[1]/2, p[1] + s[1]/2, p[1] - s[1]/2],
            mode="lines", fill="toself",
            fillcolor=f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{fill_opacity})",
            line=dict(color=_rgba_str(c), width=2),
            name=book["title"], showlegend=True,
        ))
        if book["target"]:
            tp = book["target"]
            traces.append(go.Scatter(
                x=[tp[0]], y=[tp[1]], mode="markers",
                marker=dict(size=10, color=_rgba_str(c), symbol="x-thin", line=dict(width=2)),
                showlegend=False, hoverinfo="skip",
            ))

    # Obstacles
    for obs in snap["obstacles"]:
        theta = np.linspace(0, 2 * np.pi, 24)
        ox, oy = obs["position"][0], obs["position"][1]
        traces.append(go.Scatter(
            x=(ox + obs["radius"] * np.cos(theta)).tolist(),
            y=(oy + obs["radius"] * np.sin(theta)).tolist(),
            mode="lines", fill="toself",
            fillcolor="rgba(220,60,60,0.35)",
            line=dict(color="rgba(220,60,60,0.8)", width=2),
            showlegend=False,
        ))

    # EE trajectory
    hist = snap.get("ee_history", [])
    if hist:
        traces.append(go.Scatter(
            x=[p[0] for p in hist],
            y=[p[1] for p in hist],
            mode="lines",
            line=dict(color="rgba(159,122,234,0.55)", width=2, dash="dot"),
            name="EE path", showlegend=True,
        ))

    # EE current position
    ee = snap["ee_pos"]
    traces.append(go.Scatter(
        x=[ee[0]], y=[ee[1]], mode="markers",
        marker=dict(size=13, color="#f6ad55", symbol="star",
                    line=dict(color="#ecc94b", width=2)),
        name="EE", showlegend=True,
    ))

    # Base position
    traces.append(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(size=10, color="#4299e1", symbol="circle",
                    line=dict(color="#63b3ed", width=2)),
        name="Base", showlegend=True,
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,1)",
        xaxis=dict(title="X (m)", range=[-0.75, 0.75],
                   gridcolor="rgba(255,255,255,0.05)", color="#718096",
                   zeroline=True, zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Y (m)", range=[-0.15, 0.95], scaleanchor="x",
                   gridcolor="rgba(255,255,255,0.05)", color="#718096",
                   zeroline=True, zerolinecolor="rgba(255,255,255,0.1)"),
        margin=dict(l=40, r=10, t=10, b=40),
        font=dict(color="#a0aec0", family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)",
                    font=dict(color="#a0aec0", size=9)),
        height=320,
    )
    return fig


def build_joint_figure(snap: dict) -> go.Figure:
    """Bar chart of current joint angles."""
    angles = snap["joint_angles_deg"]
    names  = ["J1\nBase", "J2\nShoulder", "J3\nElbow",
              "J4\nForearm", "J5\nWrist", "J6\nRoll"]
    colors = ["#63b3ed", "#90cdf4", "#48bb78", "#f6ad55", "#9f7aea", "#fc8181"]

    fig = go.Figure(go.Bar(
        x=names, y=angles,
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        text=[f"{a:.1f}°" for a in angles],
        textposition="outside",
        textfont=dict(color="#a0aec0", size=10),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,1)",
        yaxis=dict(
            title="Angle (°)", range=[-200, 200],
            gridcolor="rgba(255,255,255,0.05)", color="#718096",
            zeroline=True, zerolinecolor="rgba(255,255,255,0.15)",
        ),
        xaxis=dict(color="#a0aec0", tickfont=dict(size=9)),
        margin=dict(l=40, r=10, t=20, b=40),
        font=dict(color="#a0aec0", family="Inter"),
        height=250,
    )
    return fig


def build_ee_trajectory_figure(snap: dict) -> go.Figure:
    """3-D EE trajectory over time."""
    hist = snap.get("ee_history", [])
    if len(hist) < 2:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,14,26,1)",
            height=200,
            annotations=[dict(text="No trajectory yet", showarrow=False,
                              font=dict(color="#4a5568", size=13))],
        )
        return fig

    xs = [p[0] for p in hist]
    ys = [p[1] for p in hist]
    zs = [p[2] for p in hist]
    n  = len(xs)

    fig = go.Figure(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines+markers",
        marker=dict(size=3,
                    color=list(range(n)),
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Step", len=0.7,
                                  tickfont=dict(color="#a0aec0", size=9))),
        line=dict(color="rgba(159,122,234,0.7)", width=4),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(10,14,26,1)",
            xaxis=dict(title="X", gridcolor="rgba(255,255,255,0.05)", color="#718096"),
            yaxis=dict(title="Y", gridcolor="rgba(255,255,255,0.05)", color="#718096"),
            zaxis=dict(title="Z", gridcolor="rgba(255,255,255,0.05)", color="#718096"),
        ),
        font=dict(color="#a0aec0", family="Inter"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=260,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Page CSS
# ═══════════════════════════════════════════════════════════════════════════════

_PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Control button row ── */
.robot-btn-row { display:flex; gap:0.5rem; margin-bottom:1rem; }

/* ── State badge ── */
.sim-state-badge {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}

/* ── Step list ── */
.step-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.3rem 0.6rem;
    border-radius: 6px;
    margin: 0.15rem 0;
    font-size: 0.86rem;
    font-family: 'Inter', sans-serif;
}
.step-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── IK card ── */
.ik-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
.ik-row {
    display: flex;
    justify-content: space-between;
    padding: 0.18rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.8rem;
}
.ik-key { color: #718096; }
.ik-val { color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }

/* ── Metric mini card ── */
.mini-metric {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    text-align: center;
}
.mini-metric-value {
    font-size: 1.3rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.mini-metric-label {
    font-size: 0.68rem;
    color: #718096;
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Main render function
# ═══════════════════════════════════════════════════════════════════════════════

def render_robot_sim_page() -> None:
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a365d,#2d3748,#1a202c);
                border:1px solid rgba(99,179,237,0.2);border-radius:16px;
                padding:1.5rem 2rem;margin-bottom:1.2rem;">
        <p style="font-size:1.65rem;font-weight:800;margin:0;
                  background:linear-gradient(135deg,#63b3ed,#90cdf4,#bee3f8);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;">
            🦾 Robotic Arm Simulation
        </p>
        <p style="color:#718096;margin-top:0.4rem;font-size:0.9rem;">
            Phase 12 — 6-DOF arm · Parallel-jaw gripper · Jacobian IK ·
            Pick &amp; Place · PyBullet physics (optional)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state bootstrap ───────────────────────────────────────────────
    SimCls = _get_simulator()
    if "p12_sim" not in st.session_state:
        st.session_state["p12_sim"]      = SimCls(headless=True)
        st.session_state["p12_running"]  = False
        st.session_state["p12_book_sel"] = "B1"

    sim: object = st.session_state["p12_sim"]

    # ── Sidebar: status + book selection ─────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🦾 Robot Controls")

        book_options = {b.book_id: b.title for b in sim.scene.books}
        sel_book = st.selectbox(
            "Book to pick",
            options=list(book_options.keys()),
            format_func=lambda k: f"{k} — {book_options[k]}",
            key="p12_book_sel",
        )

        st.markdown("---")
        snap0 = sim.snapshot()
        state_val = snap0["state"].upper()
        state_colors_map = {
            "IDLE": "#718096", "MOVING": "#63b3ed", "GRASPING": "#f6ad55",
            "LIFTING": "#48bb78", "CARRYING": "#9f7aea", "PLACING": "#fc8181",
            "RELEASING": "#ed64a6", "DONE": "#48bb78", "PAUSED": "#718096",
            "ERROR": "#fc8181",
        }
        sc = state_colors_map.get(state_val, "#718096")
        st.markdown(
            f'<div style="padding:0.5rem 0.75rem;background:rgba(0,0,0,0.3);'
            f'border:1px solid {sc}44;border-radius:8px;margin-bottom:0.5rem;">'
            f'<span style="color:{sc};font-weight:700;font-family:monospace;">'
            f'● {state_val}</span><br>'
            f'<span style="color:#4a5568;font-size:0.75rem;">Step: {snap0["step_count"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        gi = snap0["gripper"]
        st.markdown(
            f'<div style="font-size:0.82rem;padding:0.4rem 0;">'
            f'<b>Gripper:</b> <code>{gi["state"].upper()}</code> '
            f'({gi["gap_m"]*100:.1f} cm)</div>',
            unsafe_allow_html=True,
        )
        if gi.get("holding"):
            st.markdown(
                f'<div style="color:#48bb78;font-size:0.82rem;">'
                f'🤲 Holding: <b>{gi["holding"]}</b></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div style="font-size:0.75rem;color:#4a5568;margin-top:0.5rem;">'
            f'Backend: <code>{"PyBullet" if snap0["use_pybullet"] else "NumPy"}</code></div>',
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    #  Control Buttons:  Start | Pause | Reset | Step | Run
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("### ⚙️ Controls")
    bc1, bc2, bc3, bc4, bc5 = st.columns(5)

    with bc1:
        if st.button("▶ Start", use_container_width=True, key="p12_btn_start",
                     help="Start picking the selected book"):
            sim.start(book_id=st.session_state["p12_book_sel"])
            st.session_state["p12_running"] = False
            st.rerun()

    with bc2:
        pause_label = "▶ Resume" if snap0["paused"] else "⏸ Pause"
        if st.button(pause_label, use_container_width=True, key="p12_btn_pause",
                     help="Pause / resume the simulation"):
            if sim._paused:
                sim.resume()
            else:
                sim.pause()
            st.rerun()

    with bc3:
        if st.button("🔄 Reset", use_container_width=True, key="p12_btn_reset",
                     help="Reset to initial state"):
            sim.reset()
            st.session_state["p12_running"] = False
            st.rerun()

    with bc4:
        if st.button("⏭ Step", use_container_width=True, key="p12_btn_step",
                     help="Advance one task step"):
            sim.step()
            st.rerun()

    with bc5:
        run_label = "⏹ Stop" if st.session_state["p12_running"] else "🚀 Run"
        if st.button(run_label, use_container_width=True, key="p12_btn_run",
                     help="Run all steps automatically"):
            st.session_state["p12_running"] = not st.session_state["p12_running"]
            st.rerun()

    # Auto-advance when running
    if st.session_state["p12_running"]:
        executed = sim.run_n_steps(6)
        if executed == 0:
            st.session_state["p12_running"] = False

    # ── Fresh snapshot ────────────────────────────────────────────────────────
    snap = sim.snapshot()

    # ── State banner ──────────────────────────────────────────────────────────
    state_icons = {
        "idle": "⏹", "moving": "🔵", "grasping": "🟡", "lifting": "🟢",
        "carrying": "🟣", "placing": "🔴", "releasing": "🩷",
        "done": "✅", "paused": "⏸", "error": "❌",
    }
    s_color = state_colors_map.get(snap["state"].upper(), "#718096")
    s_icon  = state_icons.get(snap["state"], "❓")
    st.markdown(
        f'<div style="background:rgba(0,0,0,0.3);border:1px solid {s_color}44;'
        f'border-radius:10px;padding:0.65rem 1.1rem;margin-bottom:0.9rem;'
        f'display:flex;align-items:center;gap:0.8rem;">'
        f'<span style="font-size:1.5rem">{s_icon}</span>'
        f'<div>'
        f'<span style="color:{s_color};font-weight:700;font-size:1.05rem;">'
        f'{snap["state"].upper()}</span>'
        f'<span style="color:#718096;font-size:0.85rem;margin-left:1rem;">'
        f'Step {snap["step_count"]} — {snap["current_step"]}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Running banner
    if st.session_state["p12_running"]:
        st.info("🔄 Running automatically — click **⏹ Stop** to pause.")

    # ── Quick metrics row ─────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)

    def _mini_metric(col, value: str, label: str, color: str = "#63b3ed") -> None:
        col.markdown(
            f'<div class="mini-metric">'
            f'<div class="mini-metric-value" style="color:{color}">{value}</div>'
            f'<div class="mini-metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    g = snap["gripper"]
    ik_ok  = snap.get("ik_latest", {}).get("converged", True) if snap.get("ik_latest") else True
    ik_err = snap.get("ik_latest", {}).get("err_mm", 0) if snap.get("ik_latest") else 0

    _mini_metric(m1, str(snap["step_count"]),   "Steps run",       "#63b3ed")
    _mini_metric(m2, g["state"].upper(),         "Gripper",
                 "#48bb78" if g["state"] == "closed" else "#a0aec0")
    _mini_metric(m3, g.get("holding") or "—",   "Holding",
                 "#f6ad55" if g.get("holding") else "#4a5568")
    _mini_metric(m4, f"{ik_err:.1f} mm",         "Last IK error",
                 "#48bb78" if ik_err < 15 else "#fc8181")
    _mini_metric(m5, str(len(snap["ik_history"])), "IK solves", "#9f7aea")

    st.markdown("")

    # ── 3-D scene ─────────────────────────────────────────────────────────────
    st.markdown("### 🎬 3-D Scene")
    st.plotly_chart(build_3d_figure(snap), use_container_width=True, key="p12_3d")

    # ── Two-column: top-down + joint angles ───────────────────────────────────
    col_td, col_j = st.columns([1, 1])
    with col_td:
        st.markdown("##### 🗺 Top-Down View")
        st.plotly_chart(build_topdown_figure(snap), use_container_width=True, key="p12_top")
    with col_j:
        st.markdown("##### 📐 Joint Angles")
        st.plotly_chart(build_joint_figure(snap), use_container_width=True, key="p12_joints")

    # ── EE trajectory ─────────────────────────────────────────────────────────
    with st.expander("📈 End-Effector Trajectory", expanded=False):
        st.plotly_chart(
            build_ee_trajectory_figure(snap),
            use_container_width=True, key="p12_traj",
        )

    # ── Task progress ─────────────────────────────────────────────────────────
    st.markdown("### 📋 Task Progress")
    steps = snap.get("task_steps", [])
    if steps:
        total   = len(steps)
        done_n  = sum(1 for s in steps if s["completed"])
        pct     = int(done_n / total * 100)
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.04);border-radius:6px;'
            f'height:6px;margin-bottom:0.8rem;overflow:hidden;">'
            f'<div style="height:100%;width:{pct}%;'
            f'background:linear-gradient(90deg,#2c5282,#63b3ed);'
            f'border-radius:6px;transition:width 0.5s;"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for step_info in steps:
            done    = step_info["completed"]
            active  = step_info["active"]
            dot_col = "#48bb78" if done else ("#f6ad55" if active else "#4a5568")
            bg_col  = "rgba(72,187,120,0.08)" if done else (
                       "rgba(246,173,85,0.06)" if active else "transparent")
            icon    = "✅" if done else ("▶️" if active else "⬜")
            text_col = "#e2e8f0" if active else ("#a0aec0" if done else "#4a5568")
            st.markdown(
                f'<div class="step-row" style="background:{bg_col};">'
                f'<div class="step-dot" style="background:{dot_col};"></div>'
                f'<span>{icon}</span>'
                f'<span style="color:{text_col};">{step_info["description"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<p style="color:#4a5568;font-size:0.85rem;">'
            '⬅️ Click <b>▶ Start</b> to begin the pick-and-place task.</p>',
            unsafe_allow_html=True,
        )

    # ── IK Diagnostics ────────────────────────────────────────────────────────
    st.markdown("### 🧠 Jacobian IK Diagnostics")
    ik_latest = snap.get("ik_latest")
    if ik_latest is None:
        st.info("💡 No IK results yet — click **▶ Start** then **⏭ Step** or **🚀 Run**.")
    else:
        ik_c1, ik_c2 = st.columns(2)

        def _ik_row(key: str, val: str) -> str:
            return (
                f'<div class="ik-row">'
                f'<span class="ik-key">{key}</span>'
                f'<span class="ik-val">{val}</span>'
                f'</div>'
            )

        with ik_c1:
            st.markdown("**Latest IK solve**")
            tgt = ik_latest.get("target", [0, 0, 0])
            ee  = ik_latest.get("ee_pos", [0, 0, 0])
            html = '<div class="ik-card">'
            html += _ik_row("Step",      ik_latest.get("step", "—"))
            html += _ik_row("Target",    f"({tgt[0]:+.3f}, {tgt[1]:+.3f}, {tgt[2]:+.3f}) m")
            html += _ik_row("Achieved",  f"({ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f}) m")
            html += _ik_row("Error",     f"{ik_latest.get('err_mm', 0):.2f} mm")
            html += _ik_row("Converged", "✅ Yes" if ik_latest.get("converged") else "⚠️ Approx")
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

        with ik_c2:
            st.markdown("**Joint angles after IK**")
            ang = ik_latest.get("angles_deg", [])
            names_j = ["J1 Base", "J2 Shoulder", "J3 Elbow",
                       "J4 Forearm", "J5 Wrist", "J6 Roll"]
            html2 = '<div class="ik-card">'
            for nm, ang_val in zip(names_j, ang):
                html2 += _ik_row(nm, f"{ang_val:.2f}°")
            html2 += "</div>"
            st.markdown(html2, unsafe_allow_html=True)

    # ── IK History table ──────────────────────────────────────────────────────
    ik_hist = snap.get("ik_history", [])
    if ik_hist:
        with st.expander(f"📊 IK History ({len(ik_hist)} motions)", expanded=False):
            rows = []
            for rec in ik_hist:
                rows.append({
                    "Step": rec.get("step", ""),
                    "Err (mm)": round(rec.get("err_mm", 0), 2),
                    "Converged": "✅" if rec.get("converged") else "⚠️",
                    "EE X": round(rec["ee_pos"][0], 3) if rec.get("ee_pos") else "",
                    "EE Y": round(rec["ee_pos"][1], 3) if rec.get("ee_pos") else "",
                    "EE Z": round(rec["ee_pos"][2], 3) if rec.get("ee_pos") else "",
                })
            df_ik = pd.DataFrame(rows)
            st.dataframe(df_ik, use_container_width=True, hide_index=True)

    # ── Joint info table ──────────────────────────────────────────────────────
    with st.expander("🔩 Joint Limits & Current Angles", expanded=False):
        ji = snap.get("joint_info", [])
        if ji:
            df_ji = pd.DataFrame(ji)
            df_ji.columns = ["#", "Joint", "Angle (°)", "Min (°)", "Max (°)", "Max vel (rad/s)"]
            st.dataframe(df_ji, use_container_width=True, hide_index=True)

    # ── Simulation log ────────────────────────────────────────────────────────
    with st.expander("📜 Simulation Log", expanded=False):
        log_msgs = snap.get("log_messages", [])
        if log_msgs:
            log_html = "<br>".join(
                f'<span style="font-family:monospace;font-size:0.78rem;'
                f'color:#a0aec0;">{m}</span>'
                for m in reversed(log_msgs)
            )
            st.markdown(log_html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="color:#4a5568;">No log entries yet.</p>',
                unsafe_allow_html=True,
            )

    # Auto-refresh while running
    if st.session_state["p12_running"]:
        import time
        time.sleep(0.04)
        st.rerun()
