"""
visualization/dqn_page.py
=========================
Phase 15 — Deep Q-Network Reinforcement Learning Page.

Presents:
  • DQN Architecture & Policy Network
  • Multi-Modal State Observation Vector
  • Action Space & Q-Value Distribution
  • Step Reward Dynamics
  • Experience Replay Buffer & Training Metrics
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotics.dqn.dqn_agent import DQNAgent, DQNConfig
from robotics.dqn.environment import WorkspaceEnv
from orchestrator.system_state import get_system_state


def render_dqn_page() -> None:
    st.markdown("""
    <div style="margin-bottom:1rem;">
        <h1 style="margin:0;font-size:1.85rem;font-weight:800;background:linear-gradient(135deg, #9f7aea, #d6bcfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🧠 Deep Q-Network (DQN) Decision Agent
        </h1>
        <p style="margin:0.2rem 0 0 0;color:#718096;font-size:0.9rem;">
            Sequential Decision Making & High-Level Task Execution Policy
        </p>
    </div>
    """, unsafe_allow_html=True)

    state = get_system_state(force_refresh=False)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Observation Dim", f"{len(state.dqn_state_vector) if state.dqn_state_vector is not None else 16} features")
    with col2:
        st.metric("Action Space", "11 Discrete Actions")
    with col3:
        st.metric("Selected Action", state.dqn_selected_action_name)
    with col4:
        st.metric("Step Reward", f"+{state.dqn_step_reward:.2f}")

    # Network architecture
    with st.expander("🏗️ Q-Network Deep Learning Architecture", expanded=True):
        st.markdown("""
        ```
        Input Layer       : State Observation Vector ∈ ℝ¹⁶
        Hidden Layer 1    : Linear(16 → 128) + ReLU Activation + LayerNorm
        Hidden Layer 2    : Linear(128 → 128) + ReLU Activation
        Output Layer      : Linear(128 → 11) → Q(s, a) for all candidate actions a ∈ A
        Target Network    : Polyak soft-update / periodic weight sync (τ = 0.005)
        Optimizer         : Adam (lr = 1e-4) with Huber Loss (Smooth L1)
        ```
        """)

    q_col1, q_col2 = st.columns([1.2, 1.0])

    with q_col1:
        st.markdown("### 📊 Q-Value Distribution Across Action Space")
        if state.dqn_q_values is not None:
            n_acts = len(state.dqn_q_values)
            n_books = max(1, n_acts - 3)
            books_list = list(state.demo_result.books.values()) if state.demo_result and state.demo_result.books else []
            act_names = []
            for i in range(n_books):
                title = books_list[i].title[:12] if i < len(books_list) else f"Book_{i}"
                act_names.append(f"MOVE_TO_{title}")
            act_names.extend(["WAIT", "VERIFY", "FINISH"])
            act_names = act_names[:n_acts]

            q_df = pd.DataFrame({
                "Action": act_names,
                "Q-Value": state.dqn_q_values,
                "Selected": ["★ OPTIMAL" if i == state.dqn_selected_action_id else "" for i in range(len(act_names))]
            })

            fig_q = px.bar(
                q_df, x="Action", y="Q-Value",
                color="Q-Value", color_continuous_scale="Purples",
                text="Selected"
            )
            fig_q.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a0aec0", family="Inter"),
                height=340, margin=dict(l=10, r=10, t=30, b=50),
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickangle=-30),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_q, use_container_width=True)
        else:
            st.info("Q-values loaded from active agent.")

    with q_col2:
        st.markdown("### 🔍 State Observation Vector Breakdown")
        if state.dqn_state_vector is not None:
            obs = state.dqn_state_vector
            features = [
                ("Robot End-Effector X", obs[0]),
                ("Robot End-Effector Y", obs[1]),
                ("Robot End-Effector Z", obs[2]),
                ("Gripper Holding Flag", obs[3]),
                ("Active Book Target X", obs[4] if len(obs) > 4 else 0.0),
                ("Active Book Target Y", obs[5] if len(obs) > 5 else 0.0),
                ("Goal Target X", obs[6] if len(obs) > 6 else 0.0),
                ("Goal Target Y", obs[7] if len(obs) > 7 else 0.0),
            ]
            df_feat = pd.DataFrame(features, columns=["State Feature", "Normalized Value"])
            st.dataframe(df_feat, hide_index=True, use_container_width=True, height=340)

    # Interactive Step Simulation
    st.markdown("### 🎮 Interactive DQN Environment Stepper")
    if st.button("Step DQN Agent in Virtual Environment", use_container_width=True):
        b_pos = np.array([[-0.15, 0.25], [0.05, 0.45]], dtype=np.float32)
        t_pos = np.array([[0.08, 0.22], [0.07, 0.42]], dtype=np.float32)
        u_freq = np.array([2.0, 1.0], dtype=np.float32)
        wf_m = np.eye(2, dtype=np.float32)
        env = WorkspaceEnv(
            n_books=2,
            book_positions=b_pos,
            target_positions=t_pos,
            usage_freq=u_freq,
            workflow_matrix=wf_m,
            user_position=(0.0, 0.0),
            workspace=(-0.60, -0.05, 0.60, 0.75),
            seed=42,
        )
        obs = env.reset()
        agent = DQNAgent(config=DQNConfig(obs_size=len(obs), n_actions=env.n_actions, force_cpu=True))
        act_id = agent.select_action(obs, evaluate=True)
        q_vals = agent.get_q_values(obs)
        next_obs, r, done, info = env.step(act_id)
        act_name = env.action_names[act_id]
        st.success(f"Agent selected action #{act_id} ({act_name}) with Q={q_vals[act_id]:.3f} yielding Reward = {r:+.2f}")
