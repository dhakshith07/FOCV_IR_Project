"""
app.py — Autonomous Context-Aware Workspace Organizer
Streamlit Master Application — Phase 15 Final Dashboard.

Run:
    streamlit run app.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prevent faulty Anaconda sentencepiece C-extension from causing native Windows SEH crash (0xC0000005)
if "sentencepiece" not in sys.modules:
    sys.modules["sentencepiece"] = None

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.settings import get_config
from utils.logger import setup_logger, get_logger
from utils.system_info import get_system_info, get_torch_info

# Initialize logger
setup_logger()
log = get_logger(__name__)
cfg = get_config()

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Autonomous Context-Aware Workspace Organizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Premium Dark Glassmorphism Theme
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #0d1527 50%, #0f1c2e 100%);
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a101d 0%, #0f172a 100%);
        border-right: 1px solid rgba(99, 179, 237, 0.15);
    }

    /* Modern scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 179, 237, 0.25);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 179, 237, 0.45);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2b6cb0, #3182ce);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3182ce, #4299e1);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(99, 179, 237, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 0.8rem 0;">
        <div style="font-size:1.15rem;font-weight:800;background:linear-gradient(135deg, #63b3ed, #90cdf4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🤖 IR Workspace Organizer
        </div>
        <div style="font-size:0.75rem;color:#718096;margin-top:0.15rem;">
            Autonomous Robotics Project — Phase 15
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # The 10 Canonical Pages requested
    page_options = [
        "1. Dashboard",
        "2. Computer Vision",
        "3. Workspace",
        "4. Workflow",
        "5. Optimization",
        "6. DQN Decision",
        "7. Robot Simulation",
        "8. Analytics",
        "9. Experiments",
        "10. Settings",
    ]

    selected_page = st.radio(
        label="Select Navigation Page",
        options=page_options,
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Quick System Status Box
    st.markdown("""
    <div style="padding:0.75rem;background:rgba(72,187,120,0.08);border:1px solid rgba(72,187,120,0.25);border-radius:10px;">
        <div style="display:flex;align-items:center;gap:0.4rem;">
            <span style="color:#48bb78;font-size:0.8rem;">●</span>
            <span style="color:#48bb78;font-size:0.85rem;font-weight:700;">System Operational</span>
        </div>
        <div style="color:#a0aec0;font-size:0.75rem;margin-top:0.25rem;">
            Phase 15 Master Dashboard<br>
            All 15 Subsystems Online ✓
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    now_str = datetime.now().strftime("%d %b %Y | %H:%M:%S")
    st.markdown(f"""
    <div style="font-size:0.72rem;color:#4a5568;font-family:'JetBrains Mono',monospace;text-align:center;">
        🕐 {now_str}
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTING (10 PAGES)
# ═══════════════════════════════════════════════════════════════════════════════

# Page 1: Dashboard
if selected_page == "1. Dashboard":
    from visualization.dashboard_page import render_dashboard_page
    render_dashboard_page()

# Page 2: Computer Vision
elif selected_page == "2. Computer Vision":
    from visualization.cv_page import render_cv_page
    render_cv_page()

# Page 3: Workspace
elif selected_page == "3. Workspace":
    from visualization.workspace_page import render_workspace_page
    render_workspace_page()

# Page 4: Workflow
elif selected_page == "4. Workflow":
    from visualization.workflow_page import render_workflow_page
    render_workflow_page()

# Page 5: Optimization
elif selected_page == "5. Optimization":
    from visualization.optimization_page import render_optimization_page
    render_optimization_page()

# Page 6: DQN Decision
elif selected_page == "6. DQN Decision":
    from visualization.dqn_page import render_dqn_page
    render_dqn_page()

# Page 7: Robot Simulation
elif selected_page == "7. Robot Simulation":
    from visualization.robot_sim_page import render_robot_sim_page
    render_robot_sim_page()

# Page 8: Analytics
elif selected_page == "8. Analytics":
    from visualization.analytics_page import render_analytics_page
    render_analytics_page()

# Page 9: Experiments
elif selected_page == "9. Experiments":
    from visualization.experiments_page import render_experiments_page
    render_experiments_page()

# Page 10: Settings
elif selected_page == "10. Settings":
    from visualization.settings_page import render_settings_page
    render_settings_page()
