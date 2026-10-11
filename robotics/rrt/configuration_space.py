"""
robotics/rrt/configuration_space.py
=====================================
Phase 10 — 2-D Configuration Space for RRT path planning.

Models the robot's workspace as a 2-D (x, y) C-space defined by
axis-aligned bounds and a set of rectangular obstacle regions.

Classes
-------
Bounds2D       : Axis-aligned rectangular workspace boundary.
Obstacle       : Axis-aligned rectangular obstacle with an optional margin.
ConfigSpace    : Full C-space: bounds + obstacles + sampling utilities.

All coordinates are in metres.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Config = Tuple[float, float]          # (x, y) configuration


# ---------------------------------------------------------------------------
# Bounds2D
# ---------------------------------------------------------------------------

@dataclass
class Bounds2D:
    """
    Axis-aligned rectangular workspace boundary.

    Attributes
    ----------
    x_min, x_max : Horizontal extent (metres).
    y_min, y_max : Vertical / depth extent (metres).
    """
    x_min: float = 0.0
    x_max: float = 1.20
    y_min: float = 0.0
    y_max: float = 0.80

    # ------------------------------------------------------------------
    def contains(self, p: Config) -> bool:
        """Return True if point p = (x, y) lies inside the bounds."""
        return self.x_min <= p[0] <= self.x_max and self.y_min <= p[1] <= self.y_max

    def clamp(self, p: Config) -> Config:
        """Clamp p to the bounds and return the clamped coordinate."""
        x = max(self.x_min, min(self.x_max, p[0]))
        y = max(self.y_min, min(self.y_max, p[1]))
        return (x, y)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @classmethod
    def from_dict(cls, d: dict) -> "Bounds2D":
        return cls(
            x_min = float(d.get("x_min", 0.0)),
            x_max = float(d.get("x_max", 1.20)),
            y_min = float(d.get("y_min", 0.0)),
            y_max = float(d.get("y_max", 0.80)),
        )


# ---------------------------------------------------------------------------
# Obstacle
# ---------------------------------------------------------------------------

@dataclass
class Obstacle:
    """
    Axis-aligned rectangular obstacle.

    Attributes
    ----------
    x, y      : Lower-left corner (metres).
    width     : Extent along x-axis (metres).
    height    : Extent along y-axis (metres).
    margin    : Extra clearance buffer added around the obstacle (metres).
    label     : Optional human-readable name for display / debug.
    """
    x:      float
    y:      float
    width:  float
    height: float
    margin: float = 0.02
    label:  str   = ""

    # ------------------------------------------------------------------

    @property
    def x_min(self) -> float:
        return self.x - self.margin

    @property
    def x_max(self) -> float:
        return self.x + self.width + self.margin

    @property
    def y_min(self) -> float:
        return self.y - self.margin

    @property
    def y_max(self) -> float:
        return self.y + self.height + self.margin

    def contains_point(self, p: Config) -> bool:
        """Return True if point p lies inside the obstacle (+ margin)."""
        return self.x_min <= p[0] <= self.x_max and self.y_min <= p[1] <= self.y_max

    def intersects_segment(
        self,
        a: Config,
        b: Config,
        n_checks: int = 20,
    ) -> bool:
        """
        Approximate segment-obstacle collision test using linear interpolation.

        Checks `n_checks` evenly spaced points along segment a->b.
        Returns True if any point falls inside the obstacle.

        Parameters
        ----------
        a, b     : Segment endpoints.
        n_checks : Number of interpolation steps (higher = more accurate).
        """
        ax, ay = a
        bx, by = b
        for i in range(n_checks + 1):
            t = i / n_checks
            px = ax + t * (bx - ax)
            py = ay + t * (by - ay)
            if self.contains_point((px, py)):
                return True
        return False

    def corners(self) -> List[Config]:
        """Return the four corners of the obstacle bounding box (with margin)."""
        return [
            (self.x_min, self.y_min),
            (self.x_max, self.y_min),
            (self.x_max, self.y_max),
            (self.x_min, self.y_max),
        ]

    @classmethod
    def from_aabb(
        cls,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        margin: float = 0.02,
        label: str = "",
    ) -> "Obstacle":
        """Construct from min/max corners."""
        return cls(
            x      = x_min,
            y      = y_min,
            width  = x_max - x_min,
            height = y_max - y_min,
            margin = margin,
            label  = label,
        )

    def __repr__(self) -> str:
        return (
            f"Obstacle(x={self.x:.3f}, y={self.y:.3f}, "
            f"w={self.width:.3f}, h={self.height:.3f}, "
            f"label={self.label!r})"
        )


# ---------------------------------------------------------------------------
# ConfigSpace
# ---------------------------------------------------------------------------

class ConfigSpace:
    """
    2-D Configuration Space: workspace bounds + list of obstacles.

    Provides:
      - Random free-space sampling (biased toward goal)
      - Collision checking for points and segments
      - Nearest-neighbour search (brute-force, no external library)

    Parameters
    ----------
    bounds    : Workspace boundary (Bounds2D).
    obstacles : List of Obstacle objects.
    seed      : Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        bounds:    Bounds2D,
        obstacles: Optional[List[Obstacle]] = None,
        seed:      Optional[int] = None,
    ) -> None:
        self.bounds    = bounds
        self.obstacles = obstacles or []
        self._rng      = random.Random(seed)
        self._np_rng   = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_free(
        self,
        goal:           Config,
        goal_bias:      float = 0.10,
        max_attempts:   int   = 1000,
    ) -> Config:
        """
        Sample a uniformly random collision-free configuration.

        With probability `goal_bias` the goal itself is returned (goal
        sampling — accelerates convergence).

        Parameters
        ----------
        goal         : Goal configuration used for biased sampling.
        goal_bias    : Probability of returning the goal directly.
        max_attempts : Maximum tries before returning a random point
                       (fallback for very cluttered spaces).

        Returns
        -------
        A free-space (x, y) configuration.
        """
        if self._rng.random() < goal_bias:
            if self.is_free(goal):
                return goal

        for _ in range(max_attempts):
            x = self._rng.uniform(self.bounds.x_min, self.bounds.x_max)
            y = self._rng.uniform(self.bounds.y_min, self.bounds.y_max)
            p = (x, y)
            if self.is_free(p):
                return p

        # Fallback — return bounds centre (always valid for sensible maps)
        cx = (self.bounds.x_min + self.bounds.x_max) / 2.0
        cy = (self.bounds.y_min + self.bounds.y_max) / 2.0
        return (cx, cy)

    # ------------------------------------------------------------------
    # Collision queries
    # ------------------------------------------------------------------

    def is_free(self, p: Config) -> bool:
        """Return True if point p is inside the bounds and not in any obstacle."""
        if not self.bounds.contains(p):
            return False
        return not any(obs.contains_point(p) for obs in self.obstacles)

    def is_segment_free(
        self,
        a: Config,
        b: Config,
        n_checks: int = 20,
    ) -> bool:
        """
        Return True if the straight line segment a->b is collision-free.

        Both endpoints are also checked.
        """
        if not self.is_free(a) or not self.is_free(b):
            return False
        return not any(
            obs.intersects_segment(a, b, n_checks) for obs in self.obstacles
        )

    # ------------------------------------------------------------------
    # Nearest-neighbour
    # ------------------------------------------------------------------

    def nearest(self, nodes: List[Config], query: Config) -> Config:
        """
        Brute-force O(n) nearest-node search.

        Parameters
        ----------
        nodes : List of existing tree nodes.
        query : Query configuration.

        Returns
        -------
        The node in `nodes` closest to `query` (Euclidean distance).
        """
        best      = nodes[0]
        best_dist = self._dist(nodes[0], query)
        for node in nodes[1:]:
            d = self._dist(node, query)
            if d < best_dist:
                best_dist = d
                best      = node
        return best

    # ------------------------------------------------------------------
    # Steering
    # ------------------------------------------------------------------

    def steer(
        self,
        from_cfg: Config,
        toward:   Config,
        step_size: float,
    ) -> Config:
        """
        Move from `from_cfg` toward `toward` by at most `step_size` metres.

        If `toward` is closer than `step_size` the exact target is returned.

        Returns
        -------
        New configuration (x, y).
        """
        dx = toward[0] - from_cfg[0]
        dy = toward[1] - from_cfg[1]
        dist = math.hypot(dx, dy)
        if dist <= step_size:
            return toward
        scale = step_size / dist
        return (from_cfg[0] + dx * scale, from_cfg[1] + dy * scale)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dist(a: Config, b: Config) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def path_length(self, path: List[Config]) -> float:
        """Return the total Euclidean length of a path (list of configs)."""
        if len(path) < 2:
            return 0.0
        return sum(
            self._dist(path[i], path[i + 1]) for i in range(len(path) - 1)
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def make_desk_space(
        cls,
        obstacles: Optional[List[Obstacle]] = None,
        seed: Optional[int] = None,
    ) -> "ConfigSpace":
        """
        Create a standard 120 cm x 80 cm desk C-space.

        Default obstacles match a typical desk setup:
          - Laptop (centre)
          - Pen holder (right side)
          - Coffee mug (left side)
        """
        bounds = Bounds2D(x_min=0.0, x_max=1.20, y_min=0.0, y_max=0.80)
        default_obstacles = [
            Obstacle(0.40, 0.25, 0.25, 0.30, margin=0.02, label="laptop"),
            Obstacle(0.95, 0.60, 0.08, 0.08, margin=0.02, label="pen_holder"),
            Obstacle(0.05, 0.55, 0.08, 0.08, margin=0.02, label="mug"),
        ]
        return cls(
            bounds    = bounds,
            obstacles = obstacles if obstacles is not None else default_obstacles,
            seed      = seed,
        )

    def __repr__(self) -> str:
        return (
            f"ConfigSpace(bounds=({self.bounds.x_min:.2f},{self.bounds.y_min:.2f})-"
            f"({self.bounds.x_max:.2f},{self.bounds.y_max:.2f}), "
            f"obstacles={len(self.obstacles)})"
        )
