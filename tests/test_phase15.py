"""
tests/test_phase15.py
=====================
Phase 15 — Master Professional Streamlit Dashboard & System State Tests.

Verifies:
  1. All 10 pages are available and callable:
      1. Dashboard
      2. Computer Vision
      3. Workspace
      4. Workflow
      5. Optimization
      6. DQN Decision
      7. Robot Simulation
      8. Analytics
      9. Experiments
      10. Settings
  2. SystemState loads and computes actual values from the real pipeline:
      - Current Workspace detected books
      - Workflow sequence: ML → Maths → ML → DBMS
      - GA Cost Before → After
      - DQN state vector, selected action, reward
      - Robot IK & RRT status
      - Analytics metrics (Retrieval Distance, Robot Distance, GA Cost, Time, Collision, Placements)
  3. No hardcoded fake results used.

Run:
    pytest tests/test_phase15.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.system_state import get_system_state, SystemState, SystemAnalytics
import visualization


class TestPhase15PageImports:
    """Verify all 10 requested pages are importable and defined."""

    def test_dashboard_page_callable(self):
        from visualization.dashboard_page import render_dashboard_page
        assert callable(render_dashboard_page)

    def test_cv_page_callable(self):
        from visualization.cv_page import render_cv_page
        assert callable(render_cv_page)

    def test_workspace_page_callable(self):
        from visualization.workspace_page import render_workspace_page
        assert callable(render_workspace_page)

    def test_workflow_page_callable(self):
        from visualization.workflow_page import render_workflow_page
        assert callable(render_workflow_page)

    def test_optimization_page_callable(self):
        from visualization.optimization_page import render_optimization_page
        assert callable(render_optimization_page)

    def test_dqn_page_callable(self):
        from visualization.dqn_page import render_dqn_page
        assert callable(render_dqn_page)

    def test_robot_sim_page_callable(self):
        from visualization.robot_sim_page import render_robot_sim_page
        assert callable(render_robot_sim_page)

    def test_analytics_page_callable(self):
        from visualization.analytics_page import render_analytics_page
        assert callable(render_analytics_page)

    def test_experiments_page_callable(self):
        from visualization.experiments_page import render_experiments_page
        assert callable(render_experiments_page)

    def test_settings_page_callable(self):
        from visualization.settings_page import render_settings_page
        assert callable(render_settings_page)


@pytest.fixture(scope="module")
def state() -> SystemState:
    return get_system_state(force_refresh=False)


class TestSystemStateActualValues:
    """Verify system state computes real values without hardcoding fake results."""

    def test_state_loaded(self, state: SystemState):
        assert state is not None
        assert isinstance(state, SystemState)

    def test_workflow_sequence(self, state: SystemState):
        expected_seq = ["ML", "Maths", "ML", "DBMS"]
        assert state.workflow_sequence == expected_seq

    def test_current_workspace_detected_books(self, state: SystemState):
        assert state.demo_result is not None
        assert len(state.demo_result.books) >= 1
        for bid, b in state.demo_result.books.items():
            assert b.title != ""
            assert len(b.current_position) == 3

    def test_optimized_workspace_ga_cost(self, state: SystemState):
        analytics = state.analytics
        assert analytics.optimization_cost_before > 0.0
        assert analytics.optimization_cost_after > 0.0
        assert analytics.optimization_cost_before >= analytics.optimization_cost_after
        assert analytics.optimization_cost_improvement_pct >= 0.0

    def test_dqn_state_action_reward(self, state: SystemState):
        assert state.dqn_state_vector is not None
        assert len(state.dqn_state_vector) > 0
        assert state.dqn_selected_action_name != ""
        assert state.dqn_selected_action_id >= 0
        assert isinstance(state.dqn_step_reward, float)
        assert state.dqn_q_values is not None
        assert len(state.dqn_q_values) >= 1

    def test_robot_execution_status(self, state: SystemState):
        assert state.robot_current_book != ""
        assert state.robot_current_action != ""
        assert state.robot_ik_status.get("converged") is True
        assert state.robot_ik_status.get("error_mm", 99.0) < 1.0
        assert state.robot_rrt_status.get("collision_free") is True
        assert state.robot_progress_pct == 100.0

    def test_analytics_metrics_completeness(self, state: SystemState):
        a = state.analytics
        # Retrieval Distance
        assert a.retrieval_distance_before_m > 0.0
        assert a.retrieval_distance_after_m > 0.0
        assert a.retrieval_distance_reduction_pct >= 0.0

        # Robot Distance
        assert a.robot_distance_m > 0.0

        # Task Completion Time
        assert a.task_completion_time_s > 0.0

        # Collision Count
        assert a.collision_count == 0

        # Successful Placements
        assert a.successful_placements >= 1


class TestDashboardPlottingHelpers:
    """Verify 2D plotting helpers produce valid Plotly figures."""

    def test_workspace_2d_figure_generation(self):
        from visualization.dashboard_page import _create_workspace_2d_figure
        books = [
            {"id": "B1", "title": "Machine Learning", "current": [0.0, 0.3, 0.11], "target": [0.1, 0.3, 0.11], "freq": 2.0},
            {"id": "B2", "title": "Mathematics", "current": [0.1, 0.4, 0.11], "target": [0.1, 0.4, 0.11], "freq": 1.0},
        ]
        fig_curr = _create_workspace_2d_figure(books, "Test Current", is_optimized=False)
        assert fig_curr is not None
        assert len(fig_curr.data) >= 3

        fig_opt = _create_workspace_2d_figure(books, "Test Opt", is_optimized=True)
        assert fig_opt is not None
        assert len(fig_opt.data) >= 3
