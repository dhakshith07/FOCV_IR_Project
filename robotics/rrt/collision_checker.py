"""
robotics/rrt/collision_checker.py
===================================
Phase 10 — Collision Checker for RRT path planning.

Provides a dedicated, high-level collision-checking interface layered on
top of the ConfigSpace geometric primitives.

CollisionChecker wraps:
  - Point collision (is configuration inside any obstacle?)
  - Segment collision (does the straight line cross any obstacle?)
  - Path collision   (are ALL segments in a path collision-free?)
  - Statistics       (total checks, collision count, time)

Also provides:
  SafetyValidator    : Verifies that start and goal are reachable.
  CollisionReport    : Structured result describing a collision event.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from robotics.rrt.configuration_space import Config, ConfigSpace, Obstacle


# ---------------------------------------------------------------------------
# CollisionReport
# ---------------------------------------------------------------------------

@dataclass
class CollisionReport:
    """
    Describes a detected collision event.

    Attributes
    ----------
    in_collision    : True if a collision was found.
    point           : The configuration at which collision was detected
                      (for segment checks this is the first offending point).
    obstacle        : The obstacle involved (None if point is out-of-bounds).
    segment_index   : Index of the offending segment in a path (-1 if N/A).
    description     : Human-readable description.
    """
    in_collision:  bool
    point:         Optional[Config]    = None
    obstacle:      Optional[Obstacle]  = None
    segment_index: int                 = -1
    description:   str                 = ""

    @classmethod
    def clear(cls) -> "CollisionReport":
        return cls(in_collision=False, description="No collision detected.")

    def __bool__(self) -> bool:
        return self.in_collision

    def __repr__(self) -> str:
        if not self.in_collision:
            return "CollisionReport(clear)"
        obs_label = self.obstacle.label if self.obstacle else "out-of-bounds"
        return (
            f"CollisionReport(COLLISION at {self.point}, "
            f"obs={obs_label!r}, seg={self.segment_index})"
        )


# ---------------------------------------------------------------------------
# CollisionChecker
# ---------------------------------------------------------------------------

class CollisionChecker:
    """
    High-level collision checker for the RRT planner.

    Parameters
    ----------
    cspace     : ConfigSpace defining the workspace and obstacles.
    n_checks   : Number of interpolation steps for segment checks.
                 Higher = more accurate, slower. Default 30.
    """

    def __init__(self, cspace: ConfigSpace, n_checks: int = 30) -> None:
        self.cspace   = cspace
        self.n_checks = n_checks

        # Running statistics
        self._total_checks:     int   = 0
        self._collision_count:  int   = 0
        self._total_time_s:     float = 0.0

    # ------------------------------------------------------------------
    # Point check
    # ------------------------------------------------------------------

    def check_point(self, p: Config) -> CollisionReport:
        """
        Check whether a single configuration is collision-free.

        Returns
        -------
        CollisionReport
        """
        t0 = time.perf_counter()
        self._total_checks += 1

        # Bounds check
        if not self.cspace.bounds.contains(p):
            self._collision_count += 1
            self._total_time_s += time.perf_counter() - t0
            return CollisionReport(
                in_collision  = True,
                point         = p,
                description   = f"Point {p} is outside workspace bounds.",
            )

        # Obstacle check
        for obs in self.cspace.obstacles:
            if obs.contains_point(p):
                self._collision_count += 1
                self._total_time_s += time.perf_counter() - t0
                return CollisionReport(
                    in_collision = True,
                    point        = p,
                    obstacle     = obs,
                    description  = f"Point {p} inside obstacle {obs.label!r}.",
                )

        self._total_time_s += time.perf_counter() - t0
        return CollisionReport.clear()

    # ------------------------------------------------------------------
    # Segment check
    # ------------------------------------------------------------------

    def check_segment(
        self,
        a: Config,
        b: Config,
        segment_index: int = -1,
    ) -> CollisionReport:
        """
        Check whether the straight-line segment from a to b is collision-free.

        Uses linear interpolation with `self.n_checks` intermediate points.

        Returns
        -------
        CollisionReport — collision at the first offending interpolated point.
        """
        t0 = time.perf_counter()
        self._total_checks += 1

        ax, ay = a
        bx, by = b

        for i in range(self.n_checks + 1):
            t  = i / self.n_checks
            px = ax + t * (bx - ax)
            py = ay + t * (by - ay)
            p  = (px, py)

            # Bounds
            if not self.cspace.bounds.contains(p):
                self._collision_count += 1
                self._total_time_s += time.perf_counter() - t0
                return CollisionReport(
                    in_collision  = True,
                    point         = p,
                    segment_index = segment_index,
                    description   = f"Segment leaves bounds at {p}.",
                )

            # Obstacles
            for obs in self.cspace.obstacles:
                if obs.contains_point(p):
                    self._collision_count += 1
                    self._total_time_s += time.perf_counter() - t0
                    return CollisionReport(
                        in_collision  = True,
                        point         = p,
                        obstacle      = obs,
                        segment_index = segment_index,
                        description   = (
                            f"Segment enters obstacle {obs.label!r} at {p}."
                        ),
                    )

        self._total_time_s += time.perf_counter() - t0
        return CollisionReport.clear()

    # ------------------------------------------------------------------
    # Path check
    # ------------------------------------------------------------------

    def check_path(self, path: List[Config]) -> CollisionReport:
        """
        Check every segment in a path.

        Returns
        -------
        First CollisionReport that is in collision, or a clear report
        if the entire path is collision-free.
        """
        if not path:
            return CollisionReport.clear()

        # Check all individual points first (fast)
        for i, p in enumerate(path):
            report = self.check_point(p)
            if report.in_collision:
                report.segment_index = i
                return report

        # Check all segments
        for i in range(len(path) - 1):
            report = self.check_segment(path[i], path[i + 1], segment_index=i)
            if report.in_collision:
                return report

        return CollisionReport.clear()

    # ------------------------------------------------------------------
    # Fast boolean helpers (no report overhead)
    # ------------------------------------------------------------------

    def is_free(self, p: Config) -> bool:
        """Return True if point p is collision-free (fast, no report)."""
        return self.cspace.is_free(p)

    def is_point_free(self, p: Config) -> bool:
        """Return True if point p is collision-free (alias for is_free)."""
        return self.cspace.is_free(p)

    def is_segment_free(self, a: Config, b: Config) -> bool:
        """Return True if segment a->b is collision-free (fast, no report)."""
        return self.cspace.is_segment_free(a, b, self.n_checks)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def total_checks(self) -> int:
        return self._total_checks

    @property
    def collision_count(self) -> int:
        return self._collision_count

    @property
    def collision_rate(self) -> float:
        if self._total_checks == 0:
            return 0.0
        return self._collision_count / self._total_checks

    @property
    def total_time_s(self) -> float:
        return self._total_time_s

    def reset_stats(self) -> None:
        self._total_checks    = 0
        self._collision_count = 0
        self._total_time_s    = 0.0

    def stats_dict(self) -> dict:
        return {
            "total_checks":    self._total_checks,
            "collision_count": self._collision_count,
            "collision_rate":  round(self.collision_rate, 4),
            "total_time_s":    round(self._total_time_s, 6),
        }


# ---------------------------------------------------------------------------
# SafetyValidator
# ---------------------------------------------------------------------------

class SafetyValidator:
    """
    Checks that start and goal configurations are reachable.

    A configuration is reachable if:
      1. It lies inside workspace bounds.
      2. It is not inside any obstacle (including margins).

    Parameters
    ----------
    checker : CollisionChecker instance.
    """

    def __init__(self, checker: CollisionChecker) -> None:
        self.checker = checker

    def validate(
        self,
        start: Config,
        goal:  Config,
    ) -> Tuple[bool, str]:
        """
        Validate start and goal configurations.

        Returns
        -------
        (ok: bool, message: str)
          ok=True if both are valid, False otherwise with an explanation.
        """
        start_report = self.checker.check_point(start)
        if start_report.in_collision:
            return False, f"START is invalid — {start_report.description}"

        goal_report = self.checker.check_point(goal)
        if goal_report.in_collision:
            return False, f"GOAL is invalid — {goal_report.description}"

        return True, "Start and goal are both valid (collision-free)."
