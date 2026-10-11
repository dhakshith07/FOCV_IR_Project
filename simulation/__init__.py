"""
simulation/ — Virtual study desk simulation environment.

Phase 2 Public API
------------------
  Book              — desk object data model
  WorkspaceSimulator— full desk state engine
  Zone / Obstacle   — spatial zone and obstacle data models
  create_workspace_figure — Plotly 2D renderer
"""

from simulation.book import Book, BOOK_COLORS
from simulation.zones import Zone, Obstacle, create_default_zones, create_default_obstacles
from simulation.workspace import WorkspaceSimulator, CONTEXT_BOOK_MAP, USER_PROFILES
from simulation.renderer_2d import create_workspace_figure

__all__ = [
    "Book",
    "BOOK_COLORS",
    "Zone",
    "Obstacle",
    "create_default_zones",
    "create_default_obstacles",
    "WorkspaceSimulator",
    "CONTEXT_BOOK_MAP",
    "USER_PROFILES",
    "create_workspace_figure",
]
