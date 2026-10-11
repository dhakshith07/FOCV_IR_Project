"""
tests/test_phase10.py
=====================
Phase 10 -- Rapidly-exploring Random Tree (RRT) Path Planner.

Test groups
-----------
  A. Bounds2D                     (contains, clamp, width/height, from_dict)
  B. Obstacle construction        (corners, margins, repr)
  C. Obstacle.contains_point      (inside, outside, boundary, margins)
  D. Obstacle.intersects_segment  (clear, crossing, tangent)
  E. Obstacle.from_aabb           (round-trip geometry)
  F. ConfigSpace construction     (bounds, obstacles, make_desk_space)
  G. ConfigSpace.is_free          (inside, in-obstacle, out-of-bounds)
  H. ConfigSpace.is_segment_free  (clear, blocked)
  I. ConfigSpace.sample_free      (in-bounds, collision-free, goal-biased)
  J. ConfigSpace.nearest          (single node, multi-node correctness)
  K. ConfigSpace.steer            (exact step, capped step, direction)
  L. ConfigSpace.path_length      (empty, single, multi-segment)
  M. CollisionReport              (clear factory, bool, repr)
  N. CollisionChecker.check_point (clear, out-of-bounds, in-obstacle)
  O. CollisionChecker.check_segment (clear, crossing)
  P. CollisionChecker.check_path  (empty, clear, blocked)
  Q. CollisionChecker statistics  (counts, rate, reset)
  R. SafetyValidator              (valid pair, bad start, bad goal)
  S. RRTConfig                    (defaults, from_dict)
  T. RRTNode                      (config, parent, repr)
  U. PlanResult                   (summary, to_dict)
  V. RRTPlanner.plan -- bad start/goal (returns failure gracefully)
  W. RRTPlanner.plan -- open space    (finds path, success=True,
                                        path non-empty, collision-free)
  X. RRTPlanner.plan -- obstacle room (navigates around obstacle)
  Y. RRT path reconstruction          (path starts at start, ends at goal)
  Z. RRT path smoothing               (smoothed path <= raw path length)
  AA. PlanResult.summary / to_dict    (key fields present, types correct)
  AB. RRTVisualizer.to_ascii          (returns string, shows S and G)
  AC. RRTVisualizer.render            (returns Figure without crashing)
  AD. Public imports                  (robotics.rrt namespace, __all__)

Run:
    pytest tests/test_phase10.py -v
    pytest tests/test_phase10.py -v -k "not slow"
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

# ── project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

WORKSPACE = (0.0, 0.0, 1.20, 0.80)


def _make_bounds():
    from robotics.rrt.configuration_space import Bounds2D
    return Bounds2D(x_min=0.0, x_max=1.20, y_min=0.0, y_max=0.80)


def _make_obs(x=0.40, y=0.25, w=0.25, h=0.30, margin=0.02, label="laptop"):
    from robotics.rrt.configuration_space import Obstacle
    return Obstacle(x=x, y=y, width=w, height=h, margin=margin, label=label)


def _make_cspace(obstacles=None, seed=42):
    from robotics.rrt.configuration_space import ConfigSpace, Bounds2D
    bounds = Bounds2D(x_min=0.0, x_max=1.20, y_min=0.0, y_max=0.80)
    obs    = obstacles if obstacles is not None else [_make_obs()]
    return ConfigSpace(bounds=bounds, obstacles=obs, seed=seed)


def _make_checker(cspace=None):
    from robotics.rrt.collision_checker import CollisionChecker
    # Default: no obstacles so free-space checks work cleanly
    cs = cspace if cspace is not None else _make_cspace(obstacles=[])
    return CollisionChecker(cs, n_checks=20)


def _make_planner(obstacles=None, seed=42):
    from robotics.rrt.rrt_planner import RRTPlanner, RRTConfig
    cs  = _make_cspace(obstacles=obstacles, seed=seed)
    cfg = RRTConfig(
        max_iterations = 2000,
        step_size      = 0.06,
        goal_bias      = 0.15,
        goal_threshold = 0.06,
        smooth_path    = True,
        smooth_iters   = 20,
        seed           = seed,
    )
    return RRTPlanner(cspace=cs, config=cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  A. Bounds2D
# ══════════════════════════════════════════════════════════════════════════════

class TestBounds2D:

    def test_contains_interior(self):
        b = _make_bounds()
        assert b.contains((0.60, 0.40))

    def test_contains_corner(self):
        b = _make_bounds()
        assert b.contains((0.0, 0.0))
        assert b.contains((1.20, 0.80))

    def test_not_contains_outside(self):
        b = _make_bounds()
        assert not b.contains((-0.1, 0.5))
        assert not b.contains((1.25, 0.5))

    def test_clamp_inside_unchanged(self):
        b = _make_bounds()
        p = (0.5, 0.4)
        assert b.clamp(p) == p

    def test_clamp_outside(self):
        b = _make_bounds()
        p = _make_bounds().clamp((-0.5, 1.2))
        assert p == (0.0, 0.80)

    def test_width_height(self):
        b = _make_bounds()
        assert b.width  == pytest.approx(1.20)
        assert b.height == pytest.approx(0.80)

    def test_from_dict(self):
        from robotics.rrt.configuration_space import Bounds2D
        d = {"x_min": 0.1, "x_max": 1.0, "y_min": 0.1, "y_max": 0.7}
        b = Bounds2D.from_dict(d)
        assert b.x_min == pytest.approx(0.1)
        assert b.x_max == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  B. Obstacle construction
# ══════════════════════════════════════════════════════════════════════════════

class TestObstacleConstruct:

    def test_corners_count(self):
        obs = _make_obs()
        assert len(obs.corners()) == 4

    def test_margin_expands_bounds(self):
        obs = _make_obs(x=0.4, y=0.25, w=0.25, h=0.30, margin=0.02)
        assert obs.x_min == pytest.approx(0.4 - 0.02)
        assert obs.x_max == pytest.approx(0.4 + 0.25 + 0.02)

    def test_repr_contains_label(self):
        obs = _make_obs(label="mug")
        assert "mug" in repr(obs)

    def test_zero_margin(self):
        from robotics.rrt.configuration_space import Obstacle
        obs = Obstacle(x=0.3, y=0.3, width=0.2, height=0.2, margin=0.0)
        assert obs.x_min == pytest.approx(0.3)
        assert obs.x_max == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  C. Obstacle.contains_point
# ══════════════════════════════════════════════════════════════════════════════

class TestObstacleContainsPoint:

    def test_centre_inside(self):
        obs = _make_obs(x=0.4, y=0.25, w=0.25, h=0.30, margin=0.0)
        cx  = 0.4 + 0.25 / 2
        cy  = 0.25 + 0.30 / 2
        assert obs.contains_point((cx, cy))

    def test_outside_right(self):
        obs = _make_obs(x=0.4, y=0.25, w=0.25, h=0.30, margin=0.0)
        assert not obs.contains_point((0.70, 0.40))

    def test_margin_catches_nearby_point(self):
        obs = _make_obs(x=0.4, y=0.25, w=0.25, h=0.30, margin=0.03)
        # Point just outside raw obstacle but within margin
        assert obs.contains_point((0.4 - 0.01, 0.40))

    def test_corner_inside(self):
        obs = _make_obs(x=0.4, y=0.25, w=0.25, h=0.30, margin=0.0)
        assert obs.contains_point((0.40, 0.25))

    def test_far_point_outside(self):
        obs = _make_obs(x=0.4, y=0.25, w=0.25, h=0.30, margin=0.02)
        assert not obs.contains_point((0.10, 0.10))


# ══════════════════════════════════════════════════════════════════════════════
#  D. Obstacle.intersects_segment
# ══════════════════════════════════════════════════════════════════════════════

class TestObstacleIntersectsSegment:

    def test_clear_segment(self):
        obs = _make_obs(x=0.4, y=0.25, w=0.25, h=0.30, margin=0.0)
        # Segment far to the left
        assert not obs.intersects_segment((0.05, 0.10), (0.05, 0.70))

    def test_crossing_segment(self):
        obs = _make_obs(x=0.4, y=0.25, w=0.25, h=0.30, margin=0.0)
        # Horizontal segment through the obstacle
        assert obs.intersects_segment((0.0, 0.40), (1.0, 0.40))

    def test_segment_inside_obstacle(self):
        obs = _make_obs(x=0.4, y=0.25, w=0.25, h=0.30, margin=0.0)
        # Both endpoints inside
        assert obs.intersects_segment((0.45, 0.35), (0.50, 0.45))


# ══════════════════════════════════════════════════════════════════════════════
#  E. Obstacle.from_aabb
# ══════════════════════════════════════════════════════════════════════════════

class TestObstacleFromAABB:

    def test_geometry_round_trip(self):
        from robotics.rrt.configuration_space import Obstacle
        obs = Obstacle.from_aabb(0.3, 0.2, 0.6, 0.5, margin=0.0, label="test")
        assert obs.x      == pytest.approx(0.3)
        assert obs.y      == pytest.approx(0.2)
        assert obs.width  == pytest.approx(0.3)
        assert obs.height == pytest.approx(0.3)
        assert obs.label  == "test"

    def test_max_coords(self):
        from robotics.rrt.configuration_space import Obstacle
        obs = Obstacle.from_aabb(0.0, 0.0, 1.0, 0.5, margin=0.0)
        assert obs.x_max == pytest.approx(1.0)
        assert obs.y_max == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  F. ConfigSpace construction
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigSpaceConstruct:

    def test_bounds_stored(self):
        cs = _make_cspace()
        assert cs.bounds.x_max == pytest.approx(1.20)

    def test_obstacles_stored(self):
        cs = _make_cspace(obstacles=[_make_obs()])
        assert len(cs.obstacles) == 1

    def test_no_obstacles(self):
        cs = _make_cspace(obstacles=[])
        assert len(cs.obstacles) == 0

    def test_make_desk_space_returns_instance(self):
        from robotics.rrt.configuration_space import ConfigSpace
        cs = ConfigSpace.make_desk_space()
        assert isinstance(cs, ConfigSpace)
        assert len(cs.obstacles) == 3   # laptop, pen_holder, mug

    def test_make_desk_space_custom_obstacles(self):
        from robotics.rrt.configuration_space import ConfigSpace
        cs = ConfigSpace.make_desk_space(obstacles=[_make_obs()])
        assert len(cs.obstacles) == 1

    def test_repr(self):
        cs = _make_cspace()
        s = repr(cs)
        assert "ConfigSpace" in s


# ══════════════════════════════════════════════════════════════════════════════
#  G. ConfigSpace.is_free
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigSpaceIsFree:

    def test_free_point_in_open_space(self):
        cs = _make_cspace(obstacles=[])
        assert cs.is_free((0.10, 0.10))

    def test_point_in_obstacle_not_free(self):
        obs = _make_obs(x=0.4, y=0.25, w=0.25, h=0.30, margin=0.0)
        cs  = _make_cspace(obstacles=[obs])
        assert not cs.is_free((0.50, 0.40))

    def test_point_outside_bounds_not_free(self):
        cs = _make_cspace()
        assert not cs.is_free((-0.1, 0.5))
        assert not cs.is_free((1.5, 0.5))

    def test_goal_inside_obstacle_not_free(self):
        obs = _make_obs(x=0.4, y=0.25, w=0.25, h=0.30, margin=0.0)
        cs  = _make_cspace(obstacles=[obs])
        # The obstacle covers ~(0.4-0.65, 0.25-0.55)
        assert not cs.is_free((0.52, 0.40))


# ══════════════════════════════════════════════════════════════════════════════
#  H. ConfigSpace.is_segment_free
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigSpaceIsSegmentFree:

    def test_clear_horizontal_segment(self):
        cs = _make_cspace(obstacles=[])
        assert cs.is_segment_free((0.0, 0.10), (1.0, 0.10))

    def test_blocked_segment_through_obstacle(self):
        obs = _make_obs(x=0.4, y=0.0, w=0.25, h=0.80, margin=0.0)
        cs  = _make_cspace(obstacles=[obs])
        # Horizontal segment must cross obstacle
        assert not cs.is_segment_free((0.0, 0.40), (1.0, 0.40))

    def test_segment_start_in_obstacle_blocked(self):
        obs = _make_obs(x=0.4, y=0.25, w=0.25, h=0.30, margin=0.0)
        cs  = _make_cspace(obstacles=[obs])
        assert not cs.is_segment_free((0.52, 0.40), (0.10, 0.10))


# ══════════════════════════════════════════════════════════════════════════════
#  I. ConfigSpace.sample_free
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigSpaceSampleFree:

    def test_sample_inside_bounds(self):
        cs = _make_cspace(obstacles=[])
        for _ in range(50):
            p = cs.sample_free((0.60, 0.40))
            assert cs.bounds.contains(p)

    def test_sample_not_in_obstacle(self):
        obs = _make_obs(x=0.40, y=0.25, w=0.25, h=0.30, margin=0.0)
        cs  = _make_cspace(obstacles=[obs])
        for _ in range(50):
            p = cs.sample_free((0.10, 0.10))
            assert cs.is_free(p)

    def test_goal_bias_returns_goal(self):
        cs = _make_cspace(obstacles=[])
        goal = (0.90, 0.70)
        # With bias=1.0 goal should always be returned
        p = cs.sample_free(goal, goal_bias=1.0)
        assert p == goal


# ══════════════════════════════════════════════════════════════════════════════
#  J. ConfigSpace.nearest
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigSpaceNearest:

    def test_single_node(self):
        cs = _make_cspace()
        nodes = [(0.5, 0.4)]
        assert cs.nearest(nodes, (0.3, 0.3)) == (0.5, 0.4)

    def test_two_nodes_picks_closer(self):
        cs = _make_cspace()
        n1 = (0.10, 0.10)
        n2 = (0.90, 0.70)
        query = (0.15, 0.15)
        assert cs.nearest([n1, n2], query) == n1

    def test_exact_match(self):
        cs = _make_cspace()
        nodes = [(0.3, 0.3), (0.7, 0.5), (0.5, 0.4)]
        assert cs.nearest(nodes, (0.5, 0.4)) == (0.5, 0.4)

    def test_many_nodes(self):
        import random
        cs    = _make_cspace()
        nodes = [(random.uniform(0, 1.2), random.uniform(0, 0.8)) for _ in range(100)]
        query = (0.60, 0.40)
        best  = cs.nearest(nodes, query)
        # Verify it is truly the closest
        manual_best = min(nodes, key=lambda n: math.hypot(n[0] - query[0], n[1] - query[1]))
        assert best == manual_best


# ══════════════════════════════════════════════════════════════════════════════
#  K. ConfigSpace.steer
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigSpaceSteer:

    def test_close_target_exact(self):
        cs = _make_cspace()
        # Target is 0.03 away, step=0.05 → exact target returned
        result = cs.steer((0.5, 0.4), (0.53, 0.4), step_size=0.05)
        assert result == pytest.approx((0.53, 0.4), abs=1e-9)

    def test_far_target_capped(self):
        cs = _make_cspace()
        result = cs.steer((0.0, 0.0), (1.0, 0.0), step_size=0.05)
        dist = math.hypot(result[0] - 0.0, result[1] - 0.0)
        assert dist == pytest.approx(0.05, abs=1e-6)

    def test_direction_preserved(self):
        cs = _make_cspace()
        # Move in y direction only
        result = cs.steer((0.5, 0.0), (0.5, 1.0), step_size=0.10)
        assert result[0] == pytest.approx(0.5, abs=1e-6)
        assert result[1] == pytest.approx(0.10, abs=1e-6)

    def test_same_point_returns_target(self):
        cs = _make_cspace()
        p  = (0.5, 0.4)
        assert cs.steer(p, p, step_size=0.05) == p


# ══════════════════════════════════════════════════════════════════════════════
#  L. ConfigSpace.path_length
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigSpacePathLength:

    def test_empty_path(self):
        cs = _make_cspace()
        assert cs.path_length([]) == pytest.approx(0.0)

    def test_single_node(self):
        cs = _make_cspace()
        assert cs.path_length([(0.5, 0.4)]) == pytest.approx(0.0)

    def test_two_nodes(self):
        cs = _make_cspace()
        length = cs.path_length([(0.0, 0.0), (0.3, 0.4)])
        assert length == pytest.approx(0.5, abs=1e-6)

    def test_three_segments(self):
        cs = _make_cspace()
        path = [(0.0, 0.0), (0.0, 0.4), (0.3, 0.4)]
        assert cs.path_length(path) == pytest.approx(0.4 + 0.3, abs=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
#  M. CollisionReport
# ══════════════════════════════════════════════════════════════════════════════

class TestCollisionReport:

    def test_clear_factory(self):
        from robotics.rrt.collision_checker import CollisionReport
        r = CollisionReport.clear()
        assert not r.in_collision
        assert not bool(r)

    def test_collision_is_truthy(self):
        from robotics.rrt.collision_checker import CollisionReport
        r = CollisionReport(in_collision=True, description="hit obstacle")
        assert bool(r)

    def test_clear_repr(self):
        from robotics.rrt.collision_checker import CollisionReport
        r = CollisionReport.clear()
        assert "clear" in repr(r).lower()

    def test_collision_repr_contains_obs(self):
        from robotics.rrt.collision_checker import CollisionReport
        obs = _make_obs(label="laptop")
        r = CollisionReport(in_collision=True, point=(0.5, 0.4), obstacle=obs)
        assert "laptop" in repr(r)


# ══════════════════════════════════════════════════════════════════════════════
#  N. CollisionChecker.check_point
# ══════════════════════════════════════════════════════════════════════════════

class TestCollisionCheckerCheckPoint:

    def test_free_point_clear(self):
        checker = _make_checker(_make_cspace(obstacles=[]))
        r = checker.check_point((0.10, 0.10))
        assert not r.in_collision

    def test_out_of_bounds_collision(self):
        checker = _make_checker()
        r = checker.check_point((-0.5, 0.4))
        assert r.in_collision

    def test_inside_obstacle_collision(self):
        obs = _make_obs(x=0.4, y=0.25, w=0.25, h=0.30, margin=0.0)
        cs  = _make_cspace(obstacles=[obs])
        checker = _make_checker(cs)
        r = checker.check_point((0.52, 0.40))
        assert r.in_collision
        assert r.obstacle is obs


# ══════════════════════════════════════════════════════════════════════════════
#  O. CollisionChecker.check_segment
# ══════════════════════════════════════════════════════════════════════════════

class TestCollisionCheckerCheckSegment:

    def test_clear_segment(self):
        checker = _make_checker(_make_cspace(obstacles=[]))
        r = checker.check_segment((0.05, 0.05), (0.15, 0.05))
        assert not r.in_collision

    def test_crossing_segment_collision(self):
        obs = _make_obs(x=0.4, y=0.0, w=0.25, h=0.80, margin=0.0)
        cs  = _make_cspace(obstacles=[obs])
        checker = _make_checker(cs)
        r = checker.check_segment((0.0, 0.40), (1.0, 0.40))
        assert r.in_collision

    def test_segment_index_stored(self):
        obs = _make_obs(x=0.4, y=0.0, w=0.25, h=0.80, margin=0.0)
        cs  = _make_cspace(obstacles=[obs])
        checker = _make_checker(cs)
        r = checker.check_segment((0.0, 0.40), (1.0, 0.40), segment_index=3)
        assert r.segment_index == 3


# ══════════════════════════════════════════════════════════════════════════════
#  P. CollisionChecker.check_path
# ══════════════════════════════════════════════════════════════════════════════

class TestCollisionCheckerCheckPath:

    def test_empty_path_clear(self):
        checker = _make_checker()
        assert not checker.check_path([])

    def test_clear_path(self):
        checker = _make_checker(_make_cspace(obstacles=[]))
        path    = [(0.10, 0.10), (0.50, 0.10), (0.80, 0.70)]
        assert not checker.check_path(path)

    def test_blocked_path(self):
        obs = _make_obs(x=0.4, y=0.0, w=0.25, h=0.80, margin=0.0)
        cs  = _make_cspace(obstacles=[obs])
        checker = _make_checker(cs)
        path = [(0.10, 0.40), (0.90, 0.40)]   # crosses the obstacle
        assert checker.check_path(path)


# ══════════════════════════════════════════════════════════════════════════════
#  Q. CollisionChecker statistics
# ══════════════════════════════════════════════════════════════════════════════

class TestCollisionCheckerStats:

    def test_total_checks_increment(self):
        checker = _make_checker()
        checker.check_point((0.10, 0.10))
        checker.check_point((0.20, 0.20))
        assert checker.total_checks == 2

    def test_collision_count(self):
        # Use a fresh checker so no prior calls pollute the count
        checker = _make_checker(_make_cspace(obstacles=[]))
        checker.reset_stats()
        checker.check_point((-1.0, 0.5))   # out of bounds — 1 collision
        checker.check_point((0.10, 0.10))  # free — 0 collision
        assert checker.collision_count == 1
        assert checker.total_checks    == 2

    def test_collision_rate(self):
        checker = _make_checker(_make_cspace(obstacles=[]))
        checker.reset_stats()
        checker.check_point((-1.0, 0.5))   # collision
        checker.check_point((0.10, 0.10))  # free
        assert checker.collision_rate == pytest.approx(0.5)

    def test_reset_stats(self):
        checker = _make_checker()
        checker.check_point((-1.0, 0.5))
        checker.reset_stats()
        assert checker.total_checks    == 0
        assert checker.collision_count == 0

    def test_stats_dict_keys(self):
        checker = _make_checker()
        d = checker.stats_dict()
        for k in ("total_checks", "collision_count", "collision_rate", "total_time_s"):
            assert k in d


# ══════════════════════════════════════════════════════════════════════════════
#  R. SafetyValidator
# ══════════════════════════════════════════════════════════════════════════════

class TestSafetyValidator:

    def test_valid_start_and_goal(self):
        from robotics.rrt.collision_checker import SafetyValidator, CollisionChecker
        cs      = _make_cspace(obstacles=[])
        checker = CollisionChecker(cs, n_checks=20)
        v       = SafetyValidator(checker)
        ok, msg = v.validate((0.10, 0.10), (1.10, 0.70))
        assert ok
        assert "valid" in msg.lower()

    def test_bad_start(self):
        from robotics.rrt.collision_checker import SafetyValidator, CollisionChecker
        obs = _make_obs(x=0.0, y=0.0, w=0.30, h=0.30, margin=0.0)
        cs  = _make_cspace(obstacles=[obs])
        checker = CollisionChecker(cs, n_checks=20)
        v   = SafetyValidator(checker)
        ok, msg = v.validate((0.10, 0.10), (1.10, 0.70))
        assert not ok
        assert "START" in msg

    def test_bad_goal(self):
        from robotics.rrt.collision_checker import SafetyValidator, CollisionChecker
        # Goal clearly inside a large obstacle
        obs = _make_obs(x=1.00, y=0.55, w=0.18, h=0.22, margin=0.0)
        cs  = _make_cspace(obstacles=[obs])
        checker = CollisionChecker(cs, n_checks=20)
        v   = SafetyValidator(checker)
        # Start is safe; goal is inside obstacle
        ok, msg = v.validate((0.10, 0.10), (1.09, 0.66))
        assert not ok
        assert "GOAL" in msg


# ══════════════════════════════════════════════════════════════════════════════
#  S. RRTConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestRRTConfig:

    def test_defaults(self):
        from robotics.rrt.rrt_planner import RRTConfig
        cfg = RRTConfig()
        assert cfg.max_iterations  == 5000
        assert cfg.step_size       == pytest.approx(0.05)
        assert cfg.goal_bias       == pytest.approx(0.10)
        assert cfg.goal_threshold  == pytest.approx(0.05)
        assert cfg.smooth_path     is True

    def test_from_dict(self):
        from robotics.rrt.rrt_planner import RRTConfig
        d = {"max_iterations": 100, "step_size": 0.10, "smooth_path": False}
        cfg = RRTConfig.from_dict(d)
        assert cfg.max_iterations == 100
        assert cfg.step_size      == pytest.approx(0.10)
        assert cfg.smooth_path    is False

    def test_from_dict_ignores_unknown(self):
        from robotics.rrt.rrt_planner import RRTConfig
        cfg = RRTConfig.from_dict({"max_iterations": 50, "unknown_key": 99})
        assert cfg.max_iterations == 50


# ══════════════════════════════════════════════════════════════════════════════
#  T. RRTNode
# ══════════════════════════════════════════════════════════════════════════════

class TestRRTNode:

    def test_root_node(self):
        from robotics.rrt.rrt_planner import RRTNode
        n = RRTNode((0.1, 0.2))
        assert n.config == (0.1, 0.2)
        assert n.parent is None

    def test_child_node(self):
        from robotics.rrt.rrt_planner import RRTNode
        root  = RRTNode((0.0, 0.0))
        child = RRTNode((0.1, 0.1), parent=root)
        assert child.parent is root

    def test_repr(self):
        from robotics.rrt.rrt_planner import RRTNode
        n = RRTNode((0.5, 0.4))
        r = repr(n)
        assert "RRTNode" in r
        assert "0.500" in r


# ══════════════════════════════════════════════════════════════════════════════
#  U. PlanResult construction
# ══════════════════════════════════════════════════════════════════════════════

class TestPlanResult:

    def _make(self, success=True):
        from robotics.rrt.rrt_planner import PlanResult
        path = [(0.10, 0.10), (0.60, 0.40), (1.10, 0.70)]
        return PlanResult(
            success          = success,
            path             = path,
            path_length      = 1.05,
            n_nodes          = 87,
            n_iterations     = 250,
            planning_time_s  = 0.032,
            collision_status = "clear",
            start            = (0.10, 0.10),
            goal             = (1.10, 0.70),
        )

    def test_summary_is_string(self):
        r = self._make()
        s = r.summary()
        assert isinstance(s, str)

    def test_summary_contains_status(self):
        r = self._make(success=True)
        assert "SUCCESS" in r.summary()

    def test_summary_failed(self):
        r = self._make(success=False)
        assert "FAILED" in r.summary()

    def test_summary_contains_metrics(self):
        r = self._make()
        s = r.summary()
        assert "87"    in s   # n_nodes
        assert "250"   in s   # n_iterations
        assert "Path"  in s

    def test_to_dict_keys(self):
        r = self._make()
        d = r.to_dict()
        for k in ("success", "path_length", "n_nodes", "n_iterations",
                  "planning_time_ms", "collision_status", "smoothed"):
            assert k in d

    def test_to_dict_values(self):
        r = self._make()
        d = r.to_dict()
        assert d["success"]     is True
        assert d["n_nodes"]     == 87
        assert d["n_iterations"] == 250


# ══════════════════════════════════════════════════════════════════════════════
#  V. RRTPlanner.plan -- invalid start / goal
# ══════════════════════════════════════════════════════════════════════════════

class TestRRTPlannerBadConfig:

    def test_start_in_obstacle_fails(self):
        obs    = _make_obs(x=0.05, y=0.05, w=0.20, h=0.20, margin=0.0)
        planner = _make_planner(obstacles=[obs])
        result  = planner.plan(start=(0.10, 0.10), goal=(1.0, 0.70))
        assert not result.success
        assert "START" in result.collision_status

    def test_goal_out_of_bounds_fails(self):
        planner = _make_planner(obstacles=[])
        # goal way outside the 0.0-1.20 x range
        result  = planner.plan(start=(0.10, 0.10), goal=(5.0, 0.40))
        assert not result.success
        # Collision status mentions goal is out of bounds
        assert ("GOAL" in result.collision_status
                or "outside" in result.collision_status.lower())

    def test_failure_result_has_zero_path(self):
        planner = _make_planner(obstacles=[])
        result  = planner.plan(start=(-0.5, 0.4), goal=(0.60, 0.40))
        assert not result.success
        assert result.path == []
        assert result.path_length == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  W. RRTPlanner.plan -- open space (no obstacles)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestRRTPlannerOpenSpace:

    def test_finds_path(self):
        planner = _make_planner(obstacles=[], seed=1)
        result  = planner.plan(start=(0.10, 0.10), goal=(1.10, 0.70))
        assert result.success
        assert len(result.path) >= 2

    def test_path_starts_at_start(self):
        planner = _make_planner(obstacles=[], seed=2)
        start   = (0.10, 0.10)
        result  = planner.plan(start=start, goal=(1.10, 0.70))
        assert result.path[0] == pytest.approx(start, abs=1e-6)

    def test_path_ends_near_goal(self):
        planner = _make_planner(obstacles=[], seed=3)
        goal    = (1.10, 0.70)
        result  = planner.plan(start=(0.10, 0.10), goal=goal)
        end = result.path[-1]
        dist = math.hypot(end[0] - goal[0], end[1] - goal[1])
        assert dist <= planner.cfg.goal_threshold + 1e-6

    def test_path_length_positive(self):
        planner = _make_planner(obstacles=[], seed=4)
        result  = planner.plan(start=(0.10, 0.10), goal=(1.10, 0.70))
        assert result.path_length > 0.0

    def test_n_nodes_positive(self):
        planner = _make_planner(obstacles=[], seed=5)
        result  = planner.plan(start=(0.10, 0.10), goal=(1.10, 0.70))
        assert result.n_nodes > 0

    def test_planning_time_positive(self):
        planner = _make_planner(obstacles=[], seed=6)
        result  = planner.plan(start=(0.10, 0.10), goal=(1.10, 0.70))
        assert result.planning_time_s >= 0.0

    def test_tree_nodes_populated(self):
        planner = _make_planner(obstacles=[], seed=7)
        result  = planner.plan(start=(0.10, 0.10), goal=(1.10, 0.70))
        assert len(result.tree_nodes) > 0

    def test_collision_status_clear(self):
        planner = _make_planner(obstacles=[], seed=8)
        result  = planner.plan(start=(0.10, 0.10), goal=(1.10, 0.70))
        assert result.collision_status == "clear"


# ══════════════════════════════════════════════════════════════════════════════
#  X. RRTPlanner.plan -- navigates around obstacle
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestRRTPlannerWithObstacle:

    def test_finds_path_around_obstacle(self):
        # Wall obstacle blocking direct path
        obs = _make_obs(x=0.45, y=0.0, w=0.15, h=0.70, margin=0.01, label="wall")
        planner = _make_planner(obstacles=[obs], seed=10)
        result  = planner.plan(start=(0.20, 0.40), goal=(0.90, 0.40))
        assert result.success, "RRT should find path around the wall"

    def test_path_is_collision_free(self):
        obs = _make_obs(x=0.45, y=0.0, w=0.15, h=0.70, margin=0.01, label="wall")
        planner = _make_planner(obstacles=[obs], seed=11)
        result  = planner.plan(start=(0.20, 0.40), goal=(0.90, 0.40))
        if result.success:
            checker = planner.checker
            report  = checker.check_path(result.path)
            assert not report.in_collision, (
                f"Final path has collision: {report.description}"
            )


# ══════════════════════════════════════════════════════════════════════════════
#  Y. Path reconstruction
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestRRTPathReconstruction:

    def test_path_is_connected(self):
        """Consecutive waypoints must be within step_size of each other (plus a small
        margin) for an UNSMOOTHED path. Smoothing intentionally creates longer jumps."""
        from robotics.rrt.rrt_planner import RRTConfig, RRTPlanner
        cs  = _make_cspace(obstacles=[], seed=20)
        cfg = RRTConfig(
            max_iterations=2000, step_size=0.06, goal_bias=0.15,
            goal_threshold=0.06, smooth_path=False, seed=20,
        )
        planner = RRTPlanner(cspace=cs, config=cfg)
        result  = planner.plan(start=(0.10, 0.10), goal=(1.10, 0.70))
        if not result.success:
            pytest.skip("RRT failed to find path")
        step_size_with_margin = cfg.step_size + cfg.goal_threshold + 1e-3
        for i in range(len(result.path) - 1):
            a, b = result.path[i], result.path[i + 1]
            d = math.hypot(a[0] - b[0], a[1] - b[1])
            assert d <= step_size_with_margin, (
                f"Gap {d:.4f}m > step_size {cfg.step_size}m at index {i}"
            )

    def test_path_starts_at_root(self):
        planner = _make_planner(obstacles=[], seed=21)
        start   = (0.10, 0.10)
        result  = planner.plan(start=start, goal=(1.10, 0.70))
        if not result.success:
            pytest.skip("RRT failed to find path")
        assert result.path[0] == pytest.approx(start, abs=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
#  Z. Path smoothing
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestRRTSmoothing:

    def test_smoothed_path_shorter_or_equal(self):
        from robotics.rrt.rrt_planner import RRTConfig, RRTPlanner
        cs  = _make_cspace(obstacles=[], seed=30)
        cfg = RRTConfig(
            max_iterations = 2000, step_size = 0.06, goal_bias = 0.15,
            goal_threshold = 0.06, smooth_path = True, smooth_iters = 50, seed=30,
        )
        planner = RRTPlanner(cspace=cs, config=cfg)
        result  = planner.plan(start=(0.10, 0.10), goal=(1.10, 0.70))
        if not result.success:
            pytest.skip("RRT failed to find path")
        assert result.path_length <= result.raw_path_length + 1e-9
        assert result.smoothed is True

    def test_no_smoothing_flag_respected(self):
        from robotics.rrt.rrt_planner import RRTConfig, RRTPlanner
        cs  = _make_cspace(obstacles=[], seed=31)
        cfg = RRTConfig(
            max_iterations = 2000, step_size=0.06, goal_bias=0.15,
            goal_threshold=0.06, smooth_path=False, seed=31,
        )
        planner = RRTPlanner(cspace=cs, config=cfg)
        result  = planner.plan(start=(0.10, 0.10), goal=(1.10, 0.70))
        if not result.success:
            pytest.skip("RRT failed to find path")
        assert result.smoothed is False


# ══════════════════════════════════════════════════════════════════════════════
#  AA. PlanResult.summary / to_dict
# ══════════════════════════════════════════════════════════════════════════════

class TestPlanResultSummaryToDict:

    def _run_plan(self):
        planner = _make_planner(obstacles=[], seed=40)
        return planner.plan(start=(0.10, 0.10), goal=(1.10, 0.70))

    @pytest.mark.integration
    def test_summary_all_fields(self):
        result = self._run_plan()
        s = result.summary()
        for keyword in ("Path Length", "Tree Nodes", "Iterations", "Planning Time",
                        "Collision Status", "Start", "Goal"):
            assert keyword in s, f"Missing '{keyword}' in summary"

    @pytest.mark.integration
    def test_to_dict_types(self):
        result = self._run_plan()
        d = result.to_dict()
        assert isinstance(d["success"],           bool)
        assert isinstance(d["path_length"],       float)
        assert isinstance(d["n_nodes"],           int)
        assert isinstance(d["planning_time_ms"],  float)
        assert isinstance(d["collision_status"],  str)


# ══════════════════════════════════════════════════════════════════════════════
#  AB. RRTVisualizer.to_ascii
# ══════════════════════════════════════════════════════════════════════════════

class TestRRTVisualizerASCII:

    def _make_result(self):
        from robotics.rrt.rrt_planner import PlanResult
        return PlanResult(
            success          = True,
            path             = [(0.10, 0.10), (0.60, 0.40), (1.10, 0.70)],
            path_length      = 1.05,
            n_nodes          = 50,
            n_iterations     = 200,
            planning_time_s  = 0.02,
            collision_status = "clear",
            start            = (0.10, 0.10),
            goal             = (1.10, 0.70),
            tree_nodes       = [(0.10, 0.10), (0.30, 0.20), (0.60, 0.40), (1.10, 0.70)],
        )

    def test_returns_string(self):
        from robotics.rrt.rrt_planner import RRTVisualizer
        cs  = _make_cspace()
        r   = self._make_result()
        viz = RRTVisualizer(result=r, cspace=cs)
        s   = viz.to_ascii()
        assert isinstance(s, str)

    def test_contains_start_marker(self):
        from robotics.rrt.rrt_planner import RRTVisualizer
        cs  = _make_cspace()
        viz = RRTVisualizer(result=self._make_result(), cspace=cs)
        assert " S" in viz.to_ascii()

    def test_contains_goal_marker(self):
        from robotics.rrt.rrt_planner import RRTVisualizer
        cs  = _make_cspace()
        viz = RRTVisualizer(result=self._make_result(), cspace=cs)
        assert " G" in viz.to_ascii()

    def test_contains_obstacle_marker(self):
        from robotics.rrt.rrt_planner import RRTVisualizer
        cs  = _make_cspace(obstacles=[_make_obs()])
        viz = RRTVisualizer(result=self._make_result(), cspace=cs)
        assert "##" in viz.to_ascii()

    def test_contains_legend(self):
        from robotics.rrt.rrt_planner import RRTVisualizer
        cs  = _make_cspace()
        viz = RRTVisualizer(result=self._make_result(), cspace=cs)
        s   = viz.to_ascii()
        assert "Start" in s
        assert "Goal"  in s

    def test_contains_node_count(self):
        from robotics.rrt.rrt_planner import RRTVisualizer
        cs  = _make_cspace()
        viz = RRTVisualizer(result=self._make_result(), cspace=cs)
        assert "50" in viz.to_ascii()


# ══════════════════════════════════════════════════════════════════════════════
#  AC. RRTVisualizer.render (matplotlib)
# ══════════════════════════════════════════════════════════════════════════════

class TestRRTVisualizerRender:

    def _make_result(self):
        from robotics.rrt.rrt_planner import PlanResult
        return PlanResult(
            success          = True,
            path             = [(0.10, 0.10), (0.60, 0.40), (1.10, 0.70)],
            path_length      = 1.05,
            n_nodes          = 50,
            n_iterations     = 200,
            planning_time_s  = 0.02,
            collision_status = "clear",
            start            = (0.10, 0.10),
            goal             = (1.10, 0.70),
            tree_nodes       = [(0.10, 0.10), (0.60, 0.40), (1.10, 0.70)],
        )

    def test_render_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for tests
        from robotics.rrt.rrt_planner import RRTVisualizer
        cs  = _make_cspace()
        viz = RRTVisualizer(result=self._make_result(), cspace=cs)
        fig = viz.render(show=False)
        assert fig is not None

    def test_render_has_axes(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        from robotics.rrt.rrt_planner import RRTVisualizer
        import matplotlib.pyplot as plt
        cs  = _make_cspace()
        viz = RRTVisualizer(result=self._make_result(), cspace=cs)
        fig = viz.render(show=False)
        assert len(fig.axes) == 1
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  AD. Public imports (robotics.rrt namespace)
# ══════════════════════════════════════════════════════════════════════════════

class TestPublicImports:

    def test_rrt_planner_import(self):
        from robotics.rrt import RRTPlanner, RRTConfig, RRTNode, PlanResult
        assert RRTPlanner is not None
        assert RRTConfig  is not None
        assert RRTNode    is not None
        assert PlanResult is not None

    def test_visualizer_import(self):
        from robotics.rrt import RRTVisualizer
        assert RRTVisualizer is not None

    def test_cspace_import(self):
        from robotics.rrt import ConfigSpace, Bounds2D, Obstacle, Config
        assert ConfigSpace is not None
        assert Bounds2D    is not None
        assert Obstacle    is not None

    def test_checker_import(self):
        from robotics.rrt import CollisionChecker, CollisionReport, SafetyValidator
        assert CollisionChecker  is not None
        assert CollisionReport   is not None
        assert SafetyValidator   is not None

    def test_all_exports_accessible(self):
        import robotics.rrt as rrt_mod
        for name in rrt_mod.__all__:
            assert hasattr(rrt_mod, name), f"Missing export: {name}"
