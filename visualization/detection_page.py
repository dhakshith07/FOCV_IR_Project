"""
visualization/detection_page.py
Streamlit page — YOLO11 Book Detection (Phase 3).

Modes
-----
  Demo    — Synthetic desk image + simulated detections (always works)
  Image   — Upload a real image and run YOLO11
  Video   — Upload a video clip and sample frames
  Camera  — Use Streamlit's st.camera_input (browser webcam)
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from cv.yolo.book_detector import BookDetector, ModelStatus
from cv.yolo.detection import Detection
from cv.yolo.synthetic_generator import SyntheticDeskGenerator
from cv.yolo.video_processor import VideoProcessor


# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
<style>
.det-header {
    background: linear-gradient(135deg, #1a2744 0%, #2d1b69 100%);
    border: 1px solid rgba(128,90,213,0.35);
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1rem;
}
.det-title {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #b794f4, #d6bcfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.det-meta {
    font-size: 0.8rem;
    color: #4a5568;
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}
.model-card-ok {
    background: rgba(72,187,120,0.08);
    border: 1px solid rgba(72,187,120,0.3);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.8rem;
}
.model-card-warn {
    background: rgba(237,137,54,0.08);
    border: 1px solid rgba(237,137,54,0.3);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.8rem;
}
.model-card-err {
    background: rgba(245,101,101,0.08);
    border: 1px solid rgba(245,101,101,0.3);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.8rem;
}
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
}
.stat-val { font-size: 1.5rem; font-weight: 700; color: #b794f4; font-family: 'JetBrains Mono', monospace; }
.stat-lbl { font-size: 0.68rem; color: #718096; text-transform: uppercase; letter-spacing: 0.06em; }
.det-row {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    margin-bottom: 0.3rem;
    font-size: 0.82rem;
}
.tag-yolo { color: #68d391; font-weight: 700; }
.tag-sim  { color: #f6ad55; font-weight: 700; }
.tag-custom { color: #b794f4; font-weight: 700; }
</style>
"""


# ── Session-state init ────────────────────────────────────────────────────────
def _init_state() -> None:
    if "detector" not in st.session_state:
        st.session_state.detector = BookDetector()
    if "det_results" not in st.session_state:
        st.session_state.det_results = []   # List[Detection]
    if "det_frame" not in st.session_state:
        st.session_state.det_frame = None   # annotated np.ndarray
    if "det_raw_frame" not in st.session_state:
        st.session_state.det_raw_frame = None
    if "det_mode" not in st.session_state:
        st.session_state.det_mode = "Demo"


def _det() -> BookDetector:
    return st.session_state.detector


# ── Status card HTML ──────────────────────────────────────────────────────────
def _model_status_card(detector: BookDetector) -> None:
    s = detector.status
    if s in (ModelStatus.LOADED_CUSTOM, ModelStatus.LOADED_PRETRAINED):
        css  = "model-card-ok"
        icon = "✅"
        tag  = "CUSTOM" if s == ModelStatus.LOADED_CUSTOM else "PRETRAINED YOLO11n"
    elif s == ModelStatus.SIMULATION_MODE:
        css, icon, tag = "model-card-warn", "🔬", "SIMULATION MODE"
    elif s == ModelStatus.NOT_LOADED:
        css, icon, tag = "model-card-warn", "⏳", "NOT LOADED"
    else:
        css, icon, tag = "model-card-err", "❌", s.value.upper()

    msg_html = detector.status_message.replace("\n", "<br>")
    st.markdown(f"""
    <div class="{css}">
        <span style="font-weight:700;color:#e2e8f0">{icon} {tag}</span>&nbsp;&nbsp;
        <span style="font-size:0.72rem;font-family:JetBrains Mono,monospace;color:#718096">
            {detector.loaded_model_name}
        </span><br>
        <small style="color:#718096;font-family:JetBrains Mono,monospace;white-space:pre-line">{msg_html}</small>
    </div>
    """, unsafe_allow_html=True)


# ── Stat card ─────────────────────────────────────────────────────────────────
def _stat(val: str, lbl: str) -> str:
    return f"""<div class="stat-card">
        <div class="stat-val">{val}</div>
        <div class="stat-lbl">{lbl}</div>
    </div>"""


# ── BGR numpy → PIL → streamlit-displayable ───────────────────────────────────
def _bgr_to_pil(frame: np.ndarray) -> Image.Image:
    rgb = frame[:, :, ::-1]
    return Image.fromarray(rgb.astype(np.uint8))


# ── Detection table ───────────────────────────────────────────────────────────
def _show_detection_table(detections: List[Detection]) -> None:
    if not detections:
        st.caption("No detections to show.")
        return
    rows = [
        {
            "#":         i + 1,
            "Class":     d.class_name,
            "Confidence":round(d.confidence, 3),
            "Cx (px)":   round(d.center_x),
            "Cy (px)":   round(d.center_y),
            "W (px)":    round(d.width),
            "H (px)":    round(d.height),
            "Source":    d.source,
        }
        for i, d in enumerate(detections)
    ]
    df = pd.DataFrame(rows)
    st.dataframe(
        df, hide_index=True, use_container_width=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=1, format="%.3f"
            )
        },
    )


# ── Confidence distribution chart ─────────────────────────────────────────────
def _confidence_chart(detections: List[Detection]) -> go.Figure:
    if not detections:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e", height=200,
            annotations=[dict(text="No detections", x=0.5, y=0.5,
                              showarrow=False, font=dict(color="#4a5568", size=14))]
        )
        return fig

    confs  = [d.confidence for d in detections]
    colors = ["#68d391" if d.source != "simulation" else "#f6ad55" for d in detections]
    labels = [f"{d.class_name} #{i+1}" for i, d in enumerate(detections)]

    fig = go.Figure(go.Bar(
        x=labels, y=confs,
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Confidence: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.2)",
                  annotation_text="Threshold 0.5")
    fig.update_layout(
        paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
        height=220, margin=dict(l=10, r=10, t=10, b=40),
        yaxis=dict(range=[0, 1], color="#4a5568"),
        xaxis=dict(color="#4a5568"),
        font=dict(family="Inter, sans-serif", color="#a0aec0", size=9),
    )
    return fig


# ── Show annotated frame + detections ─────────────────────────────────────────
def _show_results(
    raw_frame: Optional[np.ndarray],
    annotated: Optional[np.ndarray],
    detections: List[Detection],
) -> None:
    if annotated is None:
        st.info("Run a detection first.")
        return

    img_col, info_col = st.columns([3, 2], gap="medium")

    with img_col:
        st.markdown("**📷 Annotated Frame**")
        st.image(_bgr_to_pil(annotated), use_container_width=True)

        # Download annotated image
        pil_img = _bgr_to_pil(annotated)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download Annotated Image",
            data=buf.getvalue(), file_name="detection_result.png", mime="image/png",
        )

    with info_col:
        st.markdown("**📊 Detection Results**")
        # Quick stats
        total = len(detections)
        avg_conf = sum(d.confidence for d in detections) / max(total, 1)
        sources  = set(d.source for d in detections)

        mc1, mc2, mc3 = st.columns(3)
        mc1.markdown(_stat(str(total),   "Books"), unsafe_allow_html=True)
        mc2.markdown(_stat(f"{avg_conf:.2f}", "Avg Conf"), unsafe_allow_html=True)
        mc3.markdown(_stat(",".join(sources) if sources else "—", "Source"),
                     unsafe_allow_html=True)

        st.markdown("")
        st.markdown("**📈 Confidence Distribution**")
        st.plotly_chart(_confidence_chart(detections),
                        use_container_width=True, config={"displayModeBar": False})

        st.markdown("**📋 Detections**")
        _show_detection_table(detections)

        # Save results button
        if st.button("💾 Save JSON Results", use_container_width=True):
            det = _det()
            save_path = Path("data") / "detections" / f"result_{datetime.now().strftime('%H%M%S')}.json"
            det.save_results(
                detections, save_path, frame=raw_frame,
                metadata={"mode": st.session_state.det_mode},
            )
            st.success(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main page render
# ═══════════════════════════════════════════════════════════════════════════════

def render_detection_page() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state()
    det = _det()

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="det-header">
        <p class="det-title">🔍 YOLO11 Book Detection</p>
        <p class="det-meta">Phase 3 — Object Detection Pipeline (120 × 80 cm virtual desk)</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model control row ─────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([3, 1, 1])

    with ctrl1:
        _model_status_card(det)

    with ctrl2:
        conf_val = st.slider(
            "Confidence", 0.10, 0.95,
            det.confidence_threshold, step=0.05, key="conf_slider",
        )
        det.confidence_threshold = conf_val

    with ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        if not det.is_loaded:
            if st.button("🚀 Load YOLO11", use_container_width=True, type="primary"):
                with st.spinner("Loading YOLO11n — first run downloads ~6 MB..."):
                    success = det.load()
                if success:
                    st.success("Model loaded!")
                else:
                    st.error(det.status_message)
                st.rerun()
        else:
            st.markdown(f"""
            <div style="background:rgba(72,187,120,0.08);border:1px solid rgba(72,187,120,0.2);
                        border-radius:8px;padding:0.6rem;text-align:center;font-size:0.8rem;">
                <span style="color:#48bb78">● Online</span><br>
                <small style="color:#718096">
                    {det.load_time_seconds:.1f}s load
                </small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Mode tabs ─────────────────────────────────────────────────────────
    tabs = st.tabs(["🎯 Demo", "🖼️ Image Upload", "🎬 Video Upload", "📷 Camera"])

    # ─────────────────────────────────────────────────────────────────────
    # TAB 1 — Demo (synthetic desk)
    # ─────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("""
        > **Demo mode** — generates a synthetic top-down desk image from the Phase 2
        > workspace state and runs YOLO11 detection (or simulation if model not loaded).
        """)

        col_l, col_r = st.columns([1, 3])
        with col_l:
            st.markdown("**⚙️ Options**")
            use_workspace = st.checkbox("Use live workspace state", value=True)
            force_sim     = st.checkbox("Force simulation detections", value=not det.is_loaded)
            run_demo      = st.button("▶️ Run Detection", use_container_width=True, type="primary")

        with col_r:
            if run_demo:
                st.session_state.det_mode = "Demo"
                gen = SyntheticDeskGenerator()

                # Get workspace if available
                workspace = st.session_state.get("workspace", None)
                if not use_workspace:
                    workspace = None

                with st.spinner("Generating desk image..."):
                    raw_frame = gen.generate_desk_image(workspace)

                with st.spinner("Running detection..."):
                    if not force_sim and det.is_loaded:
                        dets = det.detect(raw_frame)
                        if not dets:
                            st.info(
                                "ℹ️ YOLO found 0 books — synthetic top-down images may "
                                "not trigger COCO 'book' detections. Showing simulation overlay."
                            )
                            dets = gen.generate_detections(workspace, raw_frame.shape) if workspace else []
                    else:
                        dets = (
                            gen.generate_detections(workspace, raw_frame.shape)
                            if workspace else []
                        )

                    annotated = det.draw_detections(raw_frame, dets)

                st.session_state.det_frame     = annotated
                st.session_state.det_raw_frame = raw_frame
                st.session_state.det_results   = dets
                st.rerun()

        _show_results(
            st.session_state.det_raw_frame,
            st.session_state.det_frame,
            st.session_state.det_results,
        )

    # ─────────────────────────────────────────────────────────────────────
    # TAB 2 — Image Upload
    # ─────────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("""
        > Upload a real desk photo. YOLO11 will detect books using the pretrained COCO
        > 'book' class. For best results, use a top-down or slight-angle desk photo.
        """)

        uploaded = st.file_uploader(
            "Upload image", type=["jpg", "jpeg", "png", "bmp"],
            key="img_upload",
        )

        if uploaded:
            import cv2
            pil_img  = Image.open(uploaded).convert("RGB")
            raw_np   = np.array(pil_img)[:, :, ::-1]   # RGB → BGR

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(pil_img, caption="Uploaded image", use_container_width=True)
            with col2:
                st.markdown(f"**Size:** {pil_img.size[0]} × {pil_img.size[1]} px")
                run_img = st.button("▶️ Detect Books", use_container_width=True, type="primary")

            if run_img:
                st.session_state.det_mode = "Image"
                if not det.is_loaded:
                    st.warning("⚠️ Model not loaded. Load YOLO11 above, or use Demo mode.")
                else:
                    with st.spinner("Running YOLO11..."):
                        dets      = det.detect(raw_np)
                        annotated = det.draw_detections(raw_np, dets)

                    st.session_state.det_frame     = annotated
                    st.session_state.det_raw_frame = raw_np
                    st.session_state.det_results   = dets
                    st.rerun()

        _show_results(
            st.session_state.det_raw_frame,
            st.session_state.det_frame,
            st.session_state.det_results,
        )

    # ─────────────────────────────────────────────────────────────────────
    # TAB 3 — Video Upload
    # ─────────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("""
        > Upload a short video clip. YOLO11 samples frames at a configurable interval
        > and runs book detection on each sample.
        """)

        vid_upload = st.file_uploader(
            "Upload video", type=["mp4", "avi", "mov", "mkv"], key="vid_upload"
        )
        v1, v2 = st.columns(2)
        with v1:
            sample_every = st.number_input("Sample every N frames", 1, 30, 10)
        with v2:
            max_frames_v = st.number_input("Max frames to process", 1, 20, 8)

        if vid_upload:
            # Save to temp file
            tmp_path = Path("data") / "tmp_video_upload.mp4"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(vid_upload.read())
            st.video(str(tmp_path))

            run_vid = st.button("▶️ Process Video", use_container_width=True, type="primary")
            if run_vid:
                st.session_state.det_mode = "Video"
                if not det.is_loaded:
                    st.warning("⚠️ Model not loaded.")
                else:
                    proc = VideoProcessor(det, sample_every=int(sample_every))
                    prog = st.progress(0, text="Processing frames...")

                    def _cb(done, total):
                        pct = min(done / max(total, 1), 1.0)
                        prog.progress(pct, text=f"Frame {done}/{total}")

                    with st.spinner("Running detection..."):
                        results, stats = proc.process_video(tmp_path, progress_cb=_cb)

                    prog.empty()
                    st.success(
                        f"Done — {stats.processed_frames} frames, "
                        f"{stats.total_detections} detections, "
                        f"{stats.avg_fps:.1f} FPS"
                    )

                    # Show grid of sampled frames
                    cols = st.columns(min(3, len(results)))
                    for ci, (frame, dets, fidx) in enumerate(results[: len(cols) * 2]):
                        with cols[ci % len(cols)]:
                            st.image(
                                _bgr_to_pil(frame),
                                caption=f"Frame {fidx} — {len(dets)} book(s)",
                                use_container_width=True,
                            )

                    # Show aggregate detections
                    st.markdown("---")
                    st.markdown("**Aggregate detections across all frames:**")
                    _show_detection_table(stats.all_detections[:20])

    # ─────────────────────────────────────────────────────────────────────
    # TAB 4 — Camera (browser webcam via st.camera_input)
    # ─────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("""
        > Uses your browser webcam to capture a snapshot.
        > YOLO11 runs book detection on the snapshot.
        > *No webcam required — if unavailable, use Demo mode.*
        """)

        # Hardware webcam availability check
        hw_available = BookDetector.detect_webcam_available(0)
        if hw_available:
            st.success("✅ Hardware webcam detected (index 0).")
        else:
            st.info("ℹ️ No hardware webcam detected. Browser webcam (st.camera_input) may still work.")

        cam_snapshot = st.camera_input("📷 Take a snapshot")

        if cam_snapshot:
            pil_cam  = Image.open(cam_snapshot).convert("RGB")
            raw_cam  = np.array(pil_cam)[:, :, ::-1]   # RGB → BGR

            if not det.is_loaded:
                st.warning("⚠️ Model not loaded. Load YOLO11 above.")
            else:
                with st.spinner("Running YOLO11 on snapshot..."):
                    dets_cam  = det.detect(raw_cam)
                    ann_cam   = det.draw_detections(raw_cam, dets_cam)

                st.session_state.det_frame     = ann_cam
                st.session_state.det_raw_frame = raw_cam
                st.session_state.det_results   = dets_cam
                st.session_state.det_mode      = "Camera"

                _show_results(raw_cam, ann_cam, dets_cam)

        if not cam_snapshot:
            _show_results(
                st.session_state.det_raw_frame,
                st.session_state.det_frame,
                st.session_state.det_results,
            )

    # ── Running totals ────────────────────────────────────────────────────
    if det.is_loaded:
        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:0.75rem;color:#4a5568;font-family:JetBrains Mono,monospace;">
            Session totals — model: {det.loaded_model_name} &nbsp;|&nbsp;
            cumulative detections: {det.total_detections} &nbsp;|&nbsp;
            confidence threshold: {det.confidence_threshold:.2f}
        </div>
        """, unsafe_allow_html=True)
