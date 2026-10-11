"""
visualization/segmentation_page.py
Phase 6 - SegFormer Semantic Segmentation Streamlit page.

Panels displayed
----------------
  1. Original Image   - with workspace region contours overlaid
  2. Segmentation Mask - colour-coded class overlay
  3. Workspace Mask   - green = valid placement surface
  4. Obstacle Map     - red = blocked / invalid placement
  5. Robot Nav Map    - violet-to-cyan heat-map (distance to obstacle)

Modes
-----
  Demo (Synthetic) - Always works. Procedurally generated desk scene.
  Image Upload     - Run segmentation on a user-provided image.
  Camera           - Run on a webcam snapshot (st.camera_input).

Model honesty note
------------------
  This page clearly labels the operating mode in every panel.
  PRETRAINED mode uses a generic ADE20K model - NOT a custom
  book/workspace segmenter. The UI prompts users to supply a
  fine-tuned checkpoint for accurate production results.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── Project imports ───────────────────────────────────────────────────────────
from cv.segformer.workspace_segmenter import (
    WorkspaceSegmenter,
    SegmentationResult,
    SegmenterMode,
    WORKSPACE_CLASS_IDS,
    OBSTACLE_CLASS_IDS,
    ADE20K_LABELS,
    DEFAULT_MODEL,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════════════════

_CSS = """
<style>
.seg-header {
    background: linear-gradient(135deg, #0d2137 0%, #1a3a52 50%, #0f2a40 100%);
    border: 1px solid rgba(34,211,238,0.25);
    border-radius: 14px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.seg-header::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -5%;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(34,211,238,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.seg-title {
    font-size: 1.55rem;
    font-weight: 800;
    background: linear-gradient(135deg, #22d3ee, #67e8f9, #a5f3fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.seg-meta {
    font-size: 0.78rem;
    color: #4a5568;
    margin-top: 0.35rem;
    font-family: 'JetBrains Mono', monospace;
}
.mode-badge-pretrained {
    display: inline-block;
    background: rgba(246,173,85,0.12);
    border: 1px solid rgba(246,173,85,0.4);
    color: #f6ad55;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.6rem;
}
.mode-badge-custom {
    display: inline-block;
    background: rgba(72,187,120,0.12);
    border: 1px solid rgba(72,187,120,0.4);
    color: #48bb78;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.6rem;
}
.mode-badge-synthetic {
    display: inline-block;
    background: rgba(128,90,213,0.12);
    border: 1px solid rgba(128,90,213,0.4);
    color: #9f7aea;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.6rem;
}
.stat-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #22d3ee;
    font-family: 'JetBrains Mono', monospace;
}
.stat-lbl {
    font-size: 0.7rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
}
.panel-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #a0aec0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.4rem;
}
.legend-dot {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.warning-box {
    background: rgba(246,173,85,0.08);
    border: 1px solid rgba(246,173,85,0.3);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    color: #f6ad55;
}
.info-box {
    background: rgba(34,211,238,0.06);
    border: 1px solid rgba(34,211,238,0.2);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    color: #67e8f9;
}
.section-hdr {
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(34,211,238,0.15);
    margin: 1.4rem 0 0.9rem 0;
}
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Session state helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_segmenter(custom_cp: Optional[str] = None) -> WorkspaceSegmenter:
    """Return a loaded WorkspaceSegmenter, cached in session state."""
    key = f"segmenter_{custom_cp or 'pretrained'}"
    if key not in st.session_state:
        seg = WorkspaceSegmenter(
            custom_checkpoint=custom_cp if custom_cp else None,
        )
        seg.load()
        st.session_state[key] = seg
    return st.session_state[key]


def _reset_segmenter() -> None:
    """Clear all cached segmenters from session state."""
    for k in list(st.session_state.keys()):
        if k.startswith("segmenter_"):
            del st.session_state[k]


# ═══════════════════════════════════════════════════════════════════════════════
#  Image helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return rgb[:, :, ::-1].astype(np.uint8)


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return bgr[:, :, ::-1]


def _make_synthetic_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a realistic synthetic desk scene BGR image."""
    import cv2
    frame = np.full((h, w, 3), (35, 28, 22), dtype=np.uint8)   # dark bg

    # Desk surface - warm wood colour
    dy1, dy2 = int(h * 0.35), h
    dx1, dx2 = int(w * 0.05), int(w * 0.95)
    frame[dy1:dy2, dx1:dx2] = (70, 110, 155)   # warm brown in BGR

    # Slight gradient on desk
    for row in range(dy1, dy2):
        blend = (row - dy1) / (dy2 - dy1)
        val   = int(155 - blend * 30)
        frame[row, dx1:dx2, 0] = 70
        frame[row, dx1:dx2, 1] = 110
        frame[row, dx1:dx2, 2] = val

    # Book 1 - blue
    b1_y1, b1_y2 = int(h*0.42), int(h*0.74)
    b1_x1, b1_x2 = int(w*0.12), int(w*0.26)
    frame[b1_y1:b1_y2, b1_x1:b1_x2] = (170, 80, 50)
    cv2.rectangle(frame, (b1_x1, b1_y1), (b1_x2, b1_y2), (200, 100, 60), 2)
    cv2.putText(frame, "BOOK 1", (b1_x1+8, b1_y1+30), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (230, 180, 150), 1, cv2.LINE_AA)

    # Book 2 - red
    b2_y1, b2_y2 = int(h*0.38), int(h*0.74)
    b2_x1, b2_x2 = int(w*0.35), int(w*0.53)
    frame[b2_y1:b2_y2, b2_x1:b2_x2] = (50, 60, 180)
    cv2.rectangle(frame, (b2_x1, b2_y1), (b2_x2, b2_y2), (70, 80, 210), 2)
    cv2.putText(frame, "BOOK 2", (b2_x1+8, b2_y1+30), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (180, 180, 230), 1, cv2.LINE_AA)

    # Book 3 - green
    b3_y1, b3_y2 = int(h*0.45), int(h*0.75)
    b3_x1, b3_x2 = int(w*0.62), int(w*0.76)
    frame[b3_y1:b3_y2, b3_x1:b3_x2] = (50, 150, 60)
    cv2.rectangle(frame, (b3_x1, b3_y1), (b3_x2, b3_y2), (60, 180, 70), 2)
    cv2.putText(frame, "BOOK 3", (b3_x1+5, b3_y1+30), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (180, 230, 180), 1, cv2.LINE_AA)

    # Laptop in centre
    lp_y1, lp_y2 = int(h*0.36), int(h*0.54)
    lp_x1, lp_x2 = int(w*0.40), int(w*0.60)
    frame[lp_y1:lp_y2, lp_x1:lp_x2] = (55, 55, 60)
    cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (80, 80, 90), 2)
    # Screen glow
    frame[lp_y1+5:lp_y2-5, lp_x1+5:lp_x2-5] = (90, 100, 130)
    cv2.putText(frame, "LAPTOP", (lp_x1+12, lp_y1+35), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (160, 180, 200), 1, cv2.LINE_AA)

    # Hand/user region bottom-right
    hand_frame = frame[int(h*0.74):h, int(w*0.70):w]
    hand_frame[:] = (100, 130, 190)   # skin tone approximation

    cv2.putText(frame, "HAND/USER", (int(w*0.72), int(h*0.80)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (50, 50, 80), 1, cv2.LINE_AA)

    # Desk edge line
    cv2.line(frame, (dx1, dy1), (dx2, dy1), (120, 140, 160), 2)

    return frame


# ═══════════════════════════════════════════════════════════════════════════════
#  Mode badge helper
# ═══════════════════════════════════════════════════════════════════════════════

def _mode_badge_html(mode: str) -> str:
    configs = {
        "custom":     ("mode-badge-custom",     "✅  CUSTOM CHECKPOINT"),
        "pretrained": ("mode-badge-pretrained",  "⚠️  PRETRAINED / DEMO  —  generic ADE20K"),
        "synthetic":  ("mode-badge-synthetic",   "🎭  SYNTHETIC DEMO  —  no model"),
    }
    css, label = configs.get(mode, ("mode-badge-synthetic", mode.upper()))
    return f'<span class="{css}">{label}</span>'


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics row
# ═══════════════════════════════════════════════════════════════════════════════

def _render_metrics(result: SegmentationResult) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    stats = [
        (c1, f"{result.workspace_pct:.1%}", "Workspace"),
        (c2, f"{result.obstacle_pct:.1%}",  "Obstacle"),
        (c3, f"{100*(1-result.workspace_pct-result.obstacle_pct):.1%}", "Other"),
        (c4, f"{len(result.workspace_regions)}", "Regions"),
        (c5, f"{result.elapsed_ms:.0f}ms",   "Elapsed"),
    ]
    for col, val, lbl in stats:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div class="stat-val">{val}</div>'
                f'<div class="stat-lbl">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  Four-panel display
# ═══════════════════════════════════════════════════════════════════════════════

def _render_four_panels(frame_bgr: np.ndarray, result: SegmentationResult) -> None:
    """Display the four segmentation panels in a 2x2 grid."""

    def _show(col, bgr: np.ndarray, title: str) -> None:
        with col:
            st.markdown(f'<div class="panel-label">{title}</div>', unsafe_allow_html=True)
            st.image(_bgr_to_rgb(bgr), use_container_width=True)

    # Row 1: Original | Segmentation Mask
    r1c1, r1c2 = st.columns(2)

    # Annotate original with workspace contours
    try:
        import cv2
        orig_ann = frame_bgr.copy()
        for reg in result.workspace_regions:
            cv2.drawContours(orig_ann, [reg.contour], -1, (72, 187, 120), 2)
            cx, cy = reg.centroid
            cv2.circle(orig_ann, (cx, cy), 5, (72, 187, 120), -1)
    except Exception:
        orig_ann = frame_bgr

    _show(r1c1, orig_ann, "📷  Original Image  (workspace contours shown)")
    _show(r1c2, result.segmentation_bgr, "🗺️  Segmentation Mask")

    # Row 2: Workspace Mask | Obstacle Map
    r2c1, r2c2 = st.columns(2)
    _show(r2c1, result.workspace_mask_bgr, "✅  Workspace Mask  (Valid Placement)")
    _show(r2c2, result.obstacle_map_bgr,   "🚫  Obstacle Map  (Invalid Placement)")

    # Row 3: Navigation Map (full width)
    st.markdown('<div class="section-hdr">🤖  Robot Navigation Map</div>', unsafe_allow_html=True)
    nav_col, info_col = st.columns([3, 1])
    with nav_col:
        st.image(_bgr_to_rgb(result.nav_map_bgr), use_container_width=True)
    with info_col:
        st.markdown(
            '<div class="info-box">'
            '<b>Navigation colour key</b><br/><br/>'
            '<span style="color:#9f7aea">■</span> <b>Violet</b> — near obstacle<br/>'
            '<span style="color:#22d3ee">■</span> <b>Cyan</b> — safe passage<br/>'
            '<span style="color:#4a5568">■</span> <b>Dark</b> — obstacle / wall<br/><br/>'
            'Heat-map encodes distance-to-nearest-obstacle (cv2.distanceTransform).'
            '</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Region table
# ═══════════════════════════════════════════════════════════════════════════════

def _render_region_table(result: SegmentationResult) -> None:
    if not result.workspace_regions:
        st.info("No workspace regions detected.")
        return

    rows = []
    for i, reg in enumerate(result.workspace_regions, 1):
        x, y, w, h = reg.bbox
        rows.append({
            "Region": i,
            "Area (px)":  f"{reg.area_px:,}",
            "Centroid":   f"({reg.centroid[0]}, {reg.centroid[1]})",
            "BBox":       f"x={x} y={y} w={w} h={h}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Class breakdown chart
# ═══════════════════════════════════════════════════════════════════════════════

def _render_class_chart(result: SegmentationResult) -> None:
    if not result.class_areas:
        return

    total = sum(result.class_areas.values())
    rows = [
        {
            "Class": ADE20K_LABELS.get(cid, f"class_{cid}"),
            "Pixels": cnt,
            "Pct": round(cnt / total * 100, 2),
        }
        for cid, cnt in sorted(result.class_areas.items(), key=lambda x: -x[1])
        if cnt > total * 0.005  # hide < 0.5%
    ]
    if not rows:
        return

    df = pd.DataFrame(rows)
    color_map = {
        "desk":   "#22d3ee",
        "table":  "#22d3ee",
        "floor":  "#67e8f9",
        "book":   "#f87171",
        "person": "#fbbf24",
        "computer": "#a78bfa",
    }

    fig = px.bar(
        df, x="Class", y="Pct",
        color="Class",
        color_discrete_map={r["Class"]: color_map.get(r["Class"], "#6b7280") for r in rows},
        labels={"Pct": "Coverage (%)"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a0aec0", family="Inter"),
        showlegend=False,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#718096"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#a0aec0"),
        margin=dict(l=10, r=10, t=20, b=10),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main render function
# ═══════════════════════════════════════════════════════════════════════════════

def render_segmentation_page() -> None:
    """Entry point called from app.py."""
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="seg-header">
        <p class="seg-title">🗺️ SegFormer Workspace Segmentation</p>
        <p class="seg-meta">Phase 6 &nbsp;·&nbsp; Semantic understanding at pixel level &nbsp;·&nbsp; ADE20K indoor classes</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Segmentation Controls")
        st.markdown("---")

        # Model mode selector
        model_mode = st.radio(
            "Model Mode",
            options=["🎭 Synthetic Demo", "🌐 Pretrained ADE20K", "📁 Custom Checkpoint"],
            index=0,
            help=(
                "Synthetic: no model needed, procedural demo.\n"
                "Pretrained: generic ADE20K — NOT workspace-specific.\n"
                "Custom: your fine-tuned checkpoint."
            ),
        )

        custom_cp = None
        if model_mode == "📁 Custom Checkpoint":
            custom_cp = st.text_input(
                "Custom checkpoint path or HF repo",
                placeholder="e.g. models/my_segformer  or  myorg/ws-seg-v1",
            )
            if st.button("🔄 Reload Model", use_container_width=True):
                _reset_segmenter()
                st.rerun()

        st.markdown("---")

        # Input source
        input_src = st.radio(
            "Input Source",
            options=["🎭 Demo Scene", "🖼️ Upload Image", "📷 Camera"],
            index=0,
        )

        st.markdown("---")

        # Legend
        st.markdown("**Colour Legend**")
        legend_items = [
            ("#48bb78", "Workspace (valid)"),
            ("#dc143c", "Obstacle (blocked)"),
            ("#fbbf24", "User / hand"),
            ("#22d3ee", "Free space"),
            ("#9f7aea", "Near obstacle (nav)"),
            ("#646464", "Unknown / wall"),
        ]
        for color, label in legend_items:
            st.markdown(
                f'<span class="legend-dot" style="background:{color}"></span>'
                f'<span style="font-size:0.8rem;color:#a0aec0">{label}</span><br/>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ADE20K class IDs info
        with st.expander("ADE20K Class Mapping"):
            st.markdown("**Workspace classes:**")
            from cv.segformer.workspace_segmenter import ADE20K_LABELS
            ws_labels = [ADE20K_LABELS.get(i, str(i)) for i in WORKSPACE_CLASS_IDS]
            st.code(", ".join(ws_labels))
            st.markdown("**Obstacle classes:**")
            ob_labels = [ADE20K_LABELS.get(i, str(i)) for i in OBSTACLE_CLASS_IDS]
            st.code(", ".join(ob_labels))

    # ── Model honesty warning ─────────────────────────────────────────────────
    if model_mode == "🌐 Pretrained ADE20K":
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ PRETRAINED / DEMO MODE</b><br/>
        This uses <code>nvidia/segformer-b0-finetuned-ade-512-512</code> — a generic
        indoor ADE20K segmenter trained on 150 common indoor classes. It was
        <b>NOT</b> trained on book-workspace or robotics lab data. Results are
        approximate and should not be treated as ground truth for production use.<br/><br/>
        To get accurate workspace segmentation, train or fine-tune a SegFormer
        checkpoint on your specific environment and load it via
        <b>Custom Checkpoint</b> mode.
        </div>
        """, unsafe_allow_html=True)

    # ── Load model ────────────────────────────────────────────────────────────
    use_synthetic = (model_mode == "🎭 Synthetic Demo")
    use_custom    = (model_mode == "📁 Custom Checkpoint" and bool(custom_cp))

    if use_synthetic:
        seg = WorkspaceSegmenter()
        # Do NOT call load() — synthetic mode works without a model
        seg.mode = SegmenterMode.SYNTHETIC
    elif use_custom:
        with st.spinner("Loading custom checkpoint..."):
            seg = _get_segmenter(custom_cp)
    else:
        with st.spinner("Loading pretrained ADE20K SegFormer..."):
            seg = _get_segmenter(None)

    # Show current mode badge
    mode_str = seg.mode.value
    st.markdown(_mode_badge_html(mode_str), unsafe_allow_html=True)
    st.markdown("")

    # ── Acquire input frame ───────────────────────────────────────────────────
    frame_bgr: Optional[np.ndarray] = None

    if input_src == "🎭 Demo Scene":
        frame_bgr = _make_synthetic_frame(h=480, w=640)
        st.markdown(
            '<div class="info-box">Using procedurally generated synthetic desk scene. '
            'Upload an image or enable camera for real workspace analysis.</div>',
            unsafe_allow_html=True,
        )

    elif input_src == "🖼️ Upload Image":
        uploaded = st.file_uploader(
            "Upload workspace image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            pil = Image.open(uploaded)
            frame_bgr = _pil_to_bgr(pil)
        else:
            st.info("Upload a workspace / desk image to run segmentation.")

    elif input_src == "📷 Camera":
        snap = st.camera_input("Take workspace snapshot")
        if snap:
            pil = Image.open(snap)
            frame_bgr = _pil_to_bgr(pil)
        else:
            st.info("Allow camera access and click the snapshot button.")

    if frame_bgr is None:
        st.markdown("---")
        st.markdown('<div class="section-hdr">📖 How It Works</div>', unsafe_allow_html=True)
        _render_how_it_works()
        return

    # ── Run segmentation ──────────────────────────────────────────────────────
    with st.spinner("Running SegFormer inference..."):
        result = seg.segment(frame_bgr)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">📊 Segmentation Statistics</div>', unsafe_allow_html=True)
    _render_metrics(result)

    # ── Four panels ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">🖼️ Segmentation Panels</div>', unsafe_allow_html=True)
    _render_four_panels(frame_bgr, result)

    # ── Region table ──────────────────────────────────────────────────────────
    with st.expander("📐 Workspace Regions Detail", expanded=False):
        _render_region_table(result)

    # ── Class breakdown ───────────────────────────────────────────────────────
    with st.expander("📈 Class Area Breakdown", expanded=False):
        _render_class_chart(result)

    # ── Segmentation result JSON ──────────────────────────────────────────────
    with st.expander("🔎 Raw Result (JSON)", expanded=False):
        st.json(result.to_dict())

    # ── Download 4-panel composite ────────────────────────────────────────────
    try:
        composite = seg.draw_all_panels(frame_bgr, result)
        composite_rgb = composite[:, :, ::-1]
        pil_composite = Image.fromarray(composite_rgb)
        buf = BytesIO()
        pil_composite.save(buf, format="PNG")
        st.download_button(
            label="⬇️  Download 4-Panel Composite (PNG)",
            data=buf.getvalue(),
            file_name="segmentation_panels.png",
            mime="image/png",
        )
    except Exception:
        pass

    # ── Custom checkpoint upgrade notice ─────────────────────────────────────
    if result.mode == "pretrained":
        st.markdown("---")
        st.markdown("""
        <div class="warning-box">
        <b>Want better accuracy?</b><br/>
        Fine-tune a SegFormer checkpoint on your actual workspace data and load it
        via the <b>Custom Checkpoint</b> mode in the sidebar. The interface is
        identical — only the weights change. See
        <code>WorkspaceSegmenter.set_custom_checkpoint()</code> for the runtime API.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  How it works section
# ═══════════════════════════════════════════════════════════════════════════════

def _render_how_it_works() -> None:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### What SegFormer does")
        st.markdown("""
- Performs **pixel-level semantic segmentation** on the input frame
- Maps every pixel to an ADE20K indoor class (desk, book, person, …)
- Runs efficiently on CPU with the B0 backbone
- Produces a `(H × W)` class-ID mask in a single forward pass
        """)

        st.markdown("#### Four derived maps")
        st.markdown("""
| Map | Description |
|-----|-------------|
| **Workspace Mask** | Valid desk surface — robot may place objects here |
| **Obstacle Map** | Blocked pixels — books, laptop, person |
| **Invalid Mask** | Union of obstacle + user + non-workspace |
| **Nav Map** | Free navigable space with distance-to-obstacle heat-map |
        """)

    with col_b:
        st.markdown("#### Pipeline position")
        st.code(
            "YOLO11 → ByteTrack → TrOCR\n"
            "       ↓\n"
            "  SegFormer  ←── you are here\n"
            "       ↓\n"
            "Workflow Analyzer → DQN → GA\n"
            "       ↓\n"
            "  RRT* → Jacobian IK → Robot",
            language="text"
        )

        st.markdown("#### Model modes")
        st.markdown("""
| Mode | Description |
|------|-------------|
| **Synthetic** | Procedural demo — no model needed |
| **Pretrained** | Generic ADE20K — approximate results |
| **Custom** | Fine-tuned on your workspace — accurate |
        """)

        st.markdown("#### Custom fine-tuning")
        st.code(
            "# Switch to your fine-tuned model at runtime:\n"
            "seg = WorkspaceSegmenter()\n"
            "seg.set_custom_checkpoint('models/my_segformer')\n"
            "seg.load()\n"
            "result = seg.segment(frame)",
            language="python"
        )
