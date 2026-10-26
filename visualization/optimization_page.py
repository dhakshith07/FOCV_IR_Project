"""
visualization/optimization_page.py
==================================
Phase 15 — Genetic Algorithm Optimization Page.

Presents:
  • Layout Optimization Objective Function (Reach + Flow + Safety)
  • Hyperparameters Configuration
  • Generation Convergence History
  • Spatial Layout Evolution: Initial vs. Evolved
  • Quantitative Component Cost Breakdown
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimization.placement_optimizer import PlacementOptimizer, OptimizationResult, BookInfo
from optimization.fitness_function import FitnessConfig
from optimization.genetic_algorithm import GAConfig
from orchestrator.system_state import get_system_state


def render_optimization_page() -> None:
    st.markdown("""
    <div style="margin-bottom:1rem;">
        <h1 style="margin:0;font-size:1.85rem;font-weight:800;background:linear-gradient(135deg, #48bb78, #68d391);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🧬 Genetic Algorithm Layout Optimization
        </h1>
        <p style="margin:0.2rem 0 0 0;color:#718096;font-size:0.9rem;">
            Global Multi-Objective Optimization of Physical Book Positions on Study Desk
        </p>
    </div>
    """, unsafe_allow_html=True)

    state = get_system_state(force_refresh=False)
    opt_res = state.demo_result.optimization_result if state.demo_result else None

    # Sidebar parameters or top columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Initial Cost", f"{state.analytics.optimization_cost_before:.3f}")
    with col2:
        st.metric("Optimized Cost", f"{state.analytics.optimization_cost_after:.3f}")
    with col3:
        st.metric("Cost Reduction", f"-{state.analytics.optimization_cost_improvement_pct:.1f}%")
    with col4:
        st.metric("Generations", f"{len(opt_res.history) if opt_res and opt_res.history else 30}")

    # Objective formulation explanation
    with st.expander("📐 Multi-Objective Fitness Function Mathematical Formulation", expanded=False):
        st.latex(r"F(\mathbf{X}) = w_1 \sum_{i} f_i \|\mathbf{x}_i - \mathbf{x}_{\text{user}}\| + w_2 \sum_{i,j} T_{ij} \|\mathbf{x}_i - \mathbf{x}_j\| + P_{\text{obs}}(\mathbf{X}) + P_{\text{bounds}}(\mathbf{X})")
        st.markdown("""
        - **Term 1 (User Reach Cost)**: Frequently accessed books are pulled closer to the student ($x_{\text{user}} = 0, y_{\text{user}} = 0$).
        - **Term 2 (Workflow Transition Cost)**: Books accessed sequentially in study sessions are placed within close cluster proximity.
        - **Term 3 & 4 (Safety Penalties)**: Heavy penalties for placing objects within obstacle buffer zones or outside robot reach boundaries.
        """)

    # Plots: Convergence curve & Layout comparison
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.markdown("### 📉 Fitness Convergence History")
        if opt_res and opt_res.history:
            history = opt_res.history
            gens = list(range(len(history)))
            best_fits = [getattr(h, "best_cost", getattr(h, "best_fitness", 0.0)) for h in history]
            avg_fits = [getattr(h, "mean_cost", getattr(h, "avg_fitness", 0.0)) for h in history]

            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(x=gens, y=best_fits, mode="lines", name="Best Fitness", line=dict(color="#48bb78", width=2.5)))
            fig_conv.add_trace(go.Scatter(x=gens, y=avg_fits, mode="lines", name="Population Avg", line=dict(color="#63b3ed", width=1.5, dash="dash")))
            fig_conv.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a0aec0", family="Inter"),
                height=340, margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(title="Generation", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Fitness Cost", gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_conv, use_container_width=True)
        else:
            st.info("Convergence history available after running optimization.")

    with plot_col2:
        st.markdown("### 📍 Coordinate Displacement (Initial vs. Optimal)")
        comp_data = []
        if state.demo_result and state.demo_result.books:
            for bid, b in state.demo_result.books.items():
                comp_data.append({
                    "Book ID": bid,
                    "Title": b.title[:16],
                    "Init X": round(float(b.initial_position[0]), 3),
                    "Init Y": round(float(b.initial_position[1]), 3),
                    "Opt X": round(float(b.target_position[0]), 3),
                    "Opt Y": round(float(b.target_position[1]), 3),
                    "Displacement (m)": round(float(np.linalg.norm(b.target_position[:2] - b.initial_position[:2])), 3),
                })
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp, hide_index=True, use_container_width=True, height=340)

    # Re-run GA Optimizer widget
    st.markdown("### ⚙️ Live Genetic Algorithm Experimenter")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pop_size = st.slider("Population Size", 20, 150, 50, 10)
    with c2:
        generations = st.slider("Generations", 10, 100, 30, 5)
    with c3:
        cx_prob = st.slider("Crossover Prob (Cx)", 0.5, 0.95, 0.8, 0.05)
    with c4:
        mut_prob = st.slider("Mutation Prob (Mut)", 0.05, 0.5, 0.2, 0.05)

    if st.button("🚀 Run Live GA Optimization with Custom Parameters", use_container_width=True):
        with st.spinner("Evolving population individuals with DEAP..."):
            ga_cfg = GAConfig(
                population_size=pop_size,
                generations=generations,
                crossover_prob=cx_prob,
                mutation_prob=mut_prob,
            )
            fit_cfg = FitnessConfig()
            books_sample = [
                BookInfo(1, "Machine Learning", (-0.15, 0.25), 2.0),
                BookInfo(2, "Mathematics", (0.05, 0.45), 1.0),
                BookInfo(3, "Database Systems", (0.25, 0.35), 1.0),
                BookInfo(4, "Networks", (-0.30, 0.50), 1.0),
                BookInfo(5, "Operating Systems", (0.35, 0.55), 1.0),
            ]
            wf_mat = np.array([
                [1.0, 0.8, 0.1, 0.0, 0.0],
                [0.8, 1.0, 0.2, 0.0, 0.0],
                [0.1, 0.2, 1.0, 0.5, 0.1],
                [0.0, 0.0, 0.5, 1.0, 0.3],
                [0.0, 0.0, 0.1, 0.3, 1.0],
            ])
            opt = PlacementOptimizer(
                books=books_sample,
                workflow_matrix=wf_mat,
                user_position=(0.0, 0.0),
                workspace=(-0.28, 0.22, 0.28, 0.42),
                ga_config=ga_cfg,
                fitness_config=fit_cfg,
            )
            res = opt.optimize()
            st.success(f"GA Evolved! Cost improved from {res.cost_before:.3f} to {res.cost_after:.3f} ({res.improvement_pct:.1f}% improvement) in {generations} generations.")
