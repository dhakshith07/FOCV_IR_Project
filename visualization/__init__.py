"""
visualization/ — Charts, heatmaps, and real-time dashboard pages.
Phase 15 Master Dashboard.
"""

from visualization.dashboard_page import render_dashboard_page
from visualization.cv_page import render_cv_page
from visualization.workspace_page import render_workspace_page
from visualization.workflow_page import render_workflow_page
from visualization.optimization_page import render_optimization_page
from visualization.dqn_page import render_dqn_page
from visualization.robot_sim_page import render_robot_sim_page
from visualization.analytics_page import render_analytics_page
from visualization.experiments_page import render_experiments_page
from visualization.settings_page import render_settings_page

__all__ = [
    "render_dashboard_page",
    "render_cv_page",
    "render_workspace_page",
    "render_workflow_page",
    "render_optimization_page",
    "render_dqn_page",
    "render_robot_sim_page",
    "render_analytics_page",
    "render_experiments_page",
    "render_settings_page",
]
