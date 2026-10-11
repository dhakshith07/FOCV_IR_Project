"""
robotics/rrt/rrt_planner.py
=============================
Phase 10 — Rapidly-exploring Random Tree (RRT) path planner.

Generates collision-free paths for the virtual robot arm moving across
the desk workspace (2-D, metres).

Algorithm implemented
---------------------
  Standard RRT with:
    - Uniform random sampling with goal bias
    - Nearest-node search (brute-force KD-style)
    - Fixed-step steering
    - Segment-level collision checking
    - Tree expansion until goal is reached or iterations exhausted
    - Path reconstruction via parent-pointer trace
    - Optional path smoothing (greedy shortcutting)

Data structures
---------------
  RRTNode    : Tree node holding (x, y) config and parent reference.
  PlanResult : Rich result object containing path, stats, and tree.

Display / visualisation
-----------------------
  RRTVisualizer.render()  → matplotlib Figure (Start, Goal, Obstacles,
                            Tree, Final Path all clearly marked).
  RRTVisualizer.to_ascii()→ compact ASCII art (no matplotlib needed).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from robotics.rrt.configuration_space import (
    Bounds2D,
    Config,
    ConfigSpace,
    Obstacle,
)
from robotics.rrt.collision_checker import CollisionChecker, SafetyValidator
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# RRTConfig
# ---------------------------------------------------------------------------

@dataclass
class RRTConfig:
    """Tunable parameters for the RRT planner."""
    max_iterations:  int   = 5000
    step_size:       float = 0.05      # metres per expansion step
    goal_bias:       float = 0.10      # probability of sampling goal
    goal_threshold:  float = 0.05      # metres — goal considered reached
    n_collision_pts: int   = 20        # interpolation steps for collision
    smooth_path:     bool  = True      # apply shortcut smoothing
    smooth_iters:    int   = 50        # shortcut attempts
    seed:            Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "RRTConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


# ---------------------------------------------------------------------------
# RRTNode
# ---------------------------------------------------------------------------

class RRTNode:
    """
    A single node in the RRT tree.

    Attributes
    ----------
    config : (x, y) configuration.
    parent : Parent RRTNode in the tree (None for root).
    """
    __slots__ = ("config", "parent")

    def __init__(self, config: Config, parent: Optional["RRTNode"] = None) -> None:
        self.config = config
        self.parent = parent

    def __repr__(self) -> str:
        return f"RRTNode(({self.config[0]:.3f}, {self.config[1]:.3f}))"


# ---------------------------------------------------------------------------
# PlanResult
# ---------------------------------------------------------------------------

@dataclass
class PlanResult:
    """
    Complete result of one RRT planning call.

    Attributes
    ----------
    success          : True if a path was found.
    path             : Ordered list of configs from start to goal (empty on failure).
    path_length      : Total Euclidean length of the path (metres).
    n_nodes          : Total nodes in the RRT tree.
    n_iterations     : Iterations consumed.
    planning_time_s  : Wall-clock time for the planning call.
    collision_status : "clear" or a description of the first obstacle encountered.
    start            : Start configuration.
    goal             : Goal configuration.
    tree_nodes       : All nodes in the tree (for visualisation).
    smoothed         : True if path smoothing was applied.
    raw_path_length  : Path length BEFORE smoothing (0 if not smoothed).
    """
    success:          bool
    path:             List[Config]
    path_length:      float
    n_nodes:          int
    n_iterations:     int
    planning_time_s:  float
    collision_status: str
    start:            Config
    goal:             Config
    tree_nodes:       List[Config] = field(default_factory=list)
    smoothed:         bool         = False
    raw_path_length:  float        = 0.0

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  RRT PLANNING RESULT",
            "=" * 55,
            f"  Status            : {'SUCCESS' if self.success else 'FAILED'}",
            f"  Start             : ({self.start[0]:.3f}, {self.start[1]:.3f})",
            f"  Goal              : ({self.goal[0]:.3f}, {self.goal[1]:.3f})",
            f"  Path Length       : {self.path_length:.4f} m",
        ]
        if self.smoothed and self.raw_path_length > 0:
            reduction = 100.0 * (1 - self.path_length / self.raw_path_length)
            lines.append(
                f"  Raw Path Length   : {self.raw_path_length:.4f} m  "
                f"({reduction:.1f}% smoothed)"
            )
        lines += [
            f"  Waypoints         : {len(self.path)}",
            f"  Tree Nodes        : {self.n_nodes}",
            f"  Iterations        : {self.n_iterations}",
            f"  Planning Time     : {self.planning_time_s * 1000:.1f} ms",
            f"  Collision Status  : {self.collision_status}",
            "=" * 55,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "success":          self.success,
            "path_length":      self.path_length,
            "n_waypoints":      len(self.path),
            "n_nodes":          self.n_nodes,
            "n_iterations":     self.n_iterations,
            "planning_time_ms": round(self.planning_time_s * 1000, 2),
            "collision_status": self.collision_status,
            "smoothed":         self.smoothed,
            "raw_path_length":  self.raw_path_length,
        }


# ---------------------------------------------------------------------------
# RRTPlanner (full implementation)
# ---------------------------------------------------------------------------

class RRTPlanner:
    """
    Rapidly-exploring Random Tree (RRT) planner for 2-D desk workspace.

    Parameters
    ----------
    cspace : ConfigSpace describing workspace bounds and obstacles.
    config : RRTConfig with algorithm hyper-parameters.
    """

    STATUS = "PHASE_10"

    def __init__(
        self,
        cspace: ConfigSpace,
        config: Optional[RRTConfig] = None,
    ) -> None:
        self.cspace  = cspace
        self.cfg     = config or RRTConfig()
        self.checker = CollisionChecker(cspace, n_checks=self.cfg.n_collision_pts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        start: Config,
        goal:  Config,
    ) -> PlanResult:
        """
        Plan a collision-free path from `start` to `goal`.

        Steps
        -----
        1. Validate start / goal with SafetyValidator.
        2. Initialise tree with the start node.
        3. Iterate:
             a. Sample a random free configuration (with goal bias).
             b. Find the nearest tree node.
             c. Steer toward the sample by step_size.
             d. Check the new edge for collisions.
             e. If clear, add the new node to the tree.
             f. Check if the new node is within goal_threshold of the goal.
        4. Reconstruct the path via parent pointers.
        5. Optionally smooth the path.
        6. Return PlanResult.

        Parameters
        ----------
        start : Start (x, y) configuration.
        goal  : Goal  (x, y) configuration.

        Returns
        -------
        PlanResult
        """
        t_start = time.perf_counter()
        self.checker.reset_stats()

        # ------ 1. Safety validation ------
        validator = SafetyValidator(self.checker)
        ok, msg   = validator.validate(start, goal)
        if not ok:
            elapsed = time.perf_counter() - t_start
            log.warning(f"RRT: invalid start/goal — {msg}")
            return PlanResult(
                success          = False,
                path             = [],
                path_length      = 0.0,
                n_nodes          = 0,
                n_iterations     = 0,
                planning_time_s  = elapsed,
                collision_status = msg,
                start            = start,
                goal             = goal,
            )

        # ------ 2. Initialise tree ------
        root            = RRTNode(start, parent=None)
        nodes: List[RRTNode] = [root]
        # Map config -> node (for fast parent lookup)
        config_list: List[Config] = [start]

        found_node: Optional[RRTNode] = None

        # ------ 3. Main loop ------
        for iteration in range(1, self.cfg.max_iterations + 1):
            # a. Random sample
            q_rand = self.cspace.sample_free(goal, goal_bias=self.cfg.goal_bias)

            # b. Nearest node
            nearest_config = self.cspace.nearest(config_list, q_rand)
            nearest_idx    = config_list.index(nearest_config)
            nearest_node   = nodes[nearest_idx]

            # c. Steer
            q_new = self.cspace.steer(nearest_config, q_rand, self.cfg.step_size)

            # d. Collision check (edge)
            if not self.checker.is_segment_free(nearest_config, q_new):
                continue

            # e. Add to tree
            new_node = RRTNode(q_new, parent=nearest_node)
            nodes.append(new_node)
            config_list.append(q_new)

            # f. Goal check
            if math.hypot(q_new[0] - goal[0], q_new[1] - goal[1]) <= self.cfg.goal_threshold:
                # Connect directly to goal if possible
                if self.checker.is_segment_free(q_new, goal):
                    goal_node = RRTNode(goal, parent=new_node)
                    nodes.append(goal_node)
                    config_list.append(goal)
                    found_node = goal_node
                else:
                    found_node = new_node
                log.info(f"RRT: goal reached after {iteration} iterations.")
                break

        elapsed = time.perf_counter() - t_start
        n_nodes = len(nodes)
        all_configs = [n.config for n in nodes]

        # ------ 4. Path reconstruction ------
        if found_node is None:
            log.warning(f"RRT: failed to reach goal after {self.cfg.max_iterations} iterations.")
            return PlanResult(
                success          = False,
                path             = [],
                path_length      = 0.0,
                n_nodes          = n_nodes,
                n_iterations     = self.cfg.max_iterations,
                planning_time_s  = elapsed,
                collision_status = "Path not found within iteration limit.",
                start            = start,
                goal             = goal,
                tree_nodes       = all_configs,
            )

        raw_path = self._reconstruct_path(found_node)
        raw_len  = self.cspace.path_length(raw_path)

        # ------ 5. Smoothing ------
        final_path = raw_path
        smoothed   = False
        if self.cfg.smooth_path and len(raw_path) > 2:
            final_path = self._smooth_path(raw_path)
            smoothed   = True

        final_len = self.cspace.path_length(final_path)

        # ------ 6. Final collision check ------
        path_report = self.checker.check_path(final_path)
        collision_status = "clear" if not path_report else path_report.description

        log.info(
            f"RRT: path found | nodes={n_nodes} | "
            f"length={final_len:.4f}m | time={elapsed*1000:.1f}ms"
        )

        return PlanResult(
            success          = True,
            path             = final_path,
            path_length      = final_len,
            n_nodes          = n_nodes,
            n_iterations     = iteration,
            planning_time_s  = elapsed,
            collision_status = collision_status,
            start            = start,
            goal             = goal,
            tree_nodes       = all_configs,
            smoothed         = smoothed,
            raw_path_length  = raw_len,
        )

    # ------------------------------------------------------------------
    # Path reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_path(self, node: RRTNode) -> List[Config]:
        """
        Trace parent pointers from `node` back to the root.

        Returns the path in start-to-goal order.
        """
        path: List[Config] = []
        current = node
        while current is not None:
            path.append(current.config)
            current = current.parent
        return list(reversed(path))

    # ------------------------------------------------------------------
    # Path smoothing (greedy shortcutting)
    # ------------------------------------------------------------------

    def _smooth_path(self, path: List[Config]) -> List[Config]:
        """
        Greedy shortcut smoothing.

        Repeatedly tries to connect non-adjacent waypoints directly.
        If the shortcut is collision-free, the intermediate nodes are removed.

        Parameters
        ----------
        path : Raw path from RRT.

        Returns
        -------
        Smoothed path (same or fewer waypoints).
        """
        smoothed = list(path)
        for _ in range(self.cfg.smooth_iters):
            if len(smoothed) <= 2:
                break
            # Pick two random indices i < j, skip adjacent
            import random as _rnd
            i = _rnd.randint(0, len(smoothed) - 3)
            j = _rnd.randint(i + 2, len(smoothed) - 1)
            if self.checker.is_segment_free(smoothed[i], smoothed[j]):
                # Remove nodes between i and j
                smoothed = smoothed[: i + 1] + smoothed[j:]
        return smoothed


# ---------------------------------------------------------------------------
# RRTVisualizer
# ---------------------------------------------------------------------------

class RRTVisualizer:
    """
    Visualise the RRT tree, obstacles, start, goal and final path.

    Two rendering backends are provided:
      render()    — matplotlib Figure (rich, colour).
      to_ascii()  — plain ASCII art (no dependencies).

    Parameters
    ----------
    result : PlanResult from RRTPlanner.plan().
    cspace : ConfigSpace (for obstacle / bounds info).
    """

    def __init__(self, result: PlanResult, cspace: ConfigSpace) -> None:
        self.result = result
        self.cspace = cspace

    # ------------------------------------------------------------------
    # Matplotlib renderer
    # ------------------------------------------------------------------

    def render(
        self,
        figsize: Tuple[float, float] = (10, 7),
        show:    bool = False,
    ):
        """
        Produce a matplotlib Figure.

        Layers (bottom to top)
        ----------------------
        1. Grey workspace background
        2. Red obstacle rectangles
        3. Light-blue RRT tree edges
        4. Thick green path
        5. Orange start marker
        6. Red star goal marker
        7. Legend

        Parameters
        ----------
        figsize : Figure size (width, height) in inches.
        show    : If True, call plt.show() at the end.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.patches import FancyArrowPatch
        except ImportError:
            raise ImportError(
                "matplotlib is required for RRTVisualizer.render(). "
                "Install with: pip install matplotlib"
            )

        fig, ax = plt.subplots(figsize=figsize)
        bounds  = self.cspace.bounds

        # ── Background ────────────────────────────────────────────────
        ax.set_xlim(bounds.x_min - 0.02, bounds.x_max + 0.02)
        ax.set_ylim(bounds.y_min - 0.02, bounds.y_max + 0.02)
        bg = mpatches.FancyBboxPatch(
            (bounds.x_min, bounds.y_min), bounds.width, bounds.height,
            boxstyle="square,pad=0", linewidth=2,
            edgecolor="#333333", facecolor="#F5F5F0",
        )
        ax.add_patch(bg)

        # ── Obstacles ─────────────────────────────────────────────────
        for obs in self.cspace.obstacles:
            rect = mpatches.Rectangle(
                (obs.x_min, obs.y_min), obs.x_max - obs.x_min, obs.y_max - obs.y_min,
                linewidth=1.5, edgecolor="#8B0000", facecolor="#FF6B6B", alpha=0.75,
                label="_nolegend_",
            )
            ax.add_patch(rect)
            # Label
            cx = (obs.x_min + obs.x_max) / 2
            cy = (obs.y_min + obs.y_max) / 2
            ax.text(
                cx, cy, obs.label or "obs",
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold",
            )

        # ── RRT Tree edges ────────────────────────────────────────────
        tree_nodes_map: Dict[Config, Optional[Config]] = {}
        # Rebuild parent map from tree_nodes in result
        # (We stored only configs, so we draw edges from the path only
        #  and the full node set for scatter)
        # Draw tree edge lines using a thin plot (we cannot reconstruct full
        # parent pointers from config list alone, so we draw the raw nodes
        # as scattered dots and the final path as edges)

        if self.result.tree_nodes:
            xs = [n[0] for n in self.result.tree_nodes]
            ys = [n[1] for n in self.result.tree_nodes]
            ax.scatter(
                xs, ys,
                c="#AED6F1", s=8, zorder=3, alpha=0.6,
                label=f"RRT nodes ({len(xs)})",
            )

        # ── Final Path ────────────────────────────────────────────────
        if self.result.path and len(self.result.path) >= 2:
            px = [p[0] for p in self.result.path]
            py = [p[1] for p in self.result.path]
            ax.plot(
                px, py,
                color="#27AE60", linewidth=2.5, zorder=5,
                label=f"Path ({self.result.path_length:.3f} m)",
            )

        # ── Start marker ──────────────────────────────────────────────
        sx, sy = self.result.start
        ax.scatter(
            sx, sy, s=200, c="#E67E22", marker="o", zorder=7,
            label=f"Start ({sx:.2f},{sy:.2f})",
        )
        ax.annotate(
            "START", (sx, sy), textcoords="offset points",
            xytext=(6, 6), fontsize=8, color="#E67E22", fontweight="bold",
        )

        # ── Goal marker ───────────────────────────────────────────────
        gx, gy = self.result.goal
        ax.scatter(
            gx, gy, s=250, c="#C0392B", marker="*", zorder=7,
            label=f"Goal ({gx:.2f},{gy:.2f})",
        )
        ax.annotate(
            "GOAL", (gx, gy), textcoords="offset points",
            xytext=(6, 6), fontsize=8, color="#C0392B", fontweight="bold",
        )

        # ── Obstacle legend patch ─────────────────────────────────────
        obs_patch = mpatches.Patch(facecolor="#FF6B6B", edgecolor="#8B0000",
                                   label=f"Obstacles ({len(self.cspace.obstacles)})")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(obs_patch)
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

        # ── Titles and annotations ────────────────────────────────────
        status = "SUCCESS" if self.result.success else "FAILED"
        ax.set_title(
            f"RRT Path Planning — {status}  |  "
            f"Nodes: {self.result.n_nodes}  |  "
            f"Length: {self.result.path_length:.3f} m  |  "
            f"Time: {self.result.planning_time_s * 1000:.0f} ms",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("X (metres)", fontsize=9)
        ax.set_ylabel("Y (metres)", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # ASCII renderer
    # ------------------------------------------------------------------

    def to_ascii(
        self,
        cols: int = 60,
        rows: int = 25,
    ) -> str:
        """
        Render a compact ASCII map of the planning result.

        Symbols
        -------
          S   Start
          G   Goal
          #   Obstacle
          *   Path waypoint
          .   RRT tree node
          (space) Free space
        """
        bounds = self.cspace.bounds
        grid   = [["  " for _ in range(cols)] for _ in range(rows)]

        def _to_grid(x: float, y: float) -> Tuple[int, int]:
            col = int((x - bounds.x_min) / bounds.width  * (cols - 1))
            row = int((1.0 - (y - bounds.y_min) / bounds.height) * (rows - 1))
            col = max(0, min(cols - 1, col))
            row = max(0, min(rows - 1, row))
            return row, col

        # Obstacles
        for obs in self.cspace.obstacles:
            for r_frac in range(rows):
                for c_frac in range(cols):
                    x = bounds.x_min + c_frac / (cols - 1) * bounds.width
                    y = bounds.y_max - r_frac / (rows - 1) * bounds.height
                    if obs.contains_point((x, y)):
                        grid[r_frac][c_frac] = "##"

        # Tree nodes
        for cfg in self.result.tree_nodes:
            r, c = _to_grid(*cfg)
            if grid[r][c] == "  ":
                grid[r][c] = " ."

        # Path
        for cfg in self.result.path:
            r, c = _to_grid(*cfg)
            grid[r][c] = " *"

        # Start / Goal
        sr, sc = _to_grid(*self.result.start)
        gr, gc = _to_grid(*self.result.goal)
        grid[sr][sc] = " S"
        grid[gr][gc] = " G"

        border = "+" + "-" * (cols * 2) + "+"
        lines  = [border]
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append(border)
        lines.append(
            f"  S=Start  G=Goal  #=Obstacle  *=Path  .=Tree  "
            f"nodes={self.result.n_nodes}  len={self.result.path_length:.3f}m"
        )
        return "\n".join(lines)
