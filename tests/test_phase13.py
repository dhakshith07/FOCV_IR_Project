"""
tests/test_phase13.py
=====================
Phase 13 — Integrated Planning Pipeline Tests

Verifies the strict pipeline:

    Target Book
      ↓  Target Cartesian Position
    Jacobian IK          (Phase 11 — never bypassed)
      ↓  Target Joint Configuration
    Joint-Space RRT      (Phase 10 logic — never bypassed)
      ↓  Collision-Free Joint-Space Path
    Virtual Robot        (Phase 12 — execute + verify)
      ↓  Final EE pose

Run:
    pytest tests/test_phase13.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ── Ensure project root is on path ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_scene():
    from robotics.robot_model import SceneConfig
    return SceneConfig.default_scene()


def _make_pipeline(scene):
    from robotics.pipeline import RobotPipeline
    return RobotPipeline(arm=scene.arm, scene=scene)


def _reachable_target(scene):
    """
    Return a Cartesian target that is reliably within the arm's reachable
    workspace.  Home angles give EE at (~0, 0, 0.51) – we pick a point
    slightly in front of the base that the DLS IK can solve.
    """
    # Workspace centre-ish point comfortably above the table
    return np.array([0.0, 0.30, 0.35])


def _run_full_pipeline(scene, target):
    """Fresh pipeline + fresh arm for each call, returns PipelineResult."""
    scene.arm.set_home()
    pipe = _make_pipeline(scene)
    return pipe.plan_and_execute(target_pos=target), pipe


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def scene():
    """Fresh scene per test (function scope)."""
    return _make_scene()


@pytest.fixture
def pipeline(scene):
    """Fresh pipeline per test."""
    return _make_pipeline(scene)


@pytest.fixture
def target(scene):
    """A reachable Cartesian target."""
    return _reachable_target(scene)


@pytest.fixture
def result(pipeline, scene, target):
    """Run the full pipeline once and return the PipelineResult."""
    scene.arm.set_home()
    return pipeline.plan_and_execute(target_pos=target)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Import and Class Availability
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase13Imports:
    def test_pipeline_importable(self):
        from robotics.pipeline import RobotPipeline  # noqa

    def test_pipeline_result_importable(self):
        from robotics.pipeline import PipelineResult  # noqa

    def test_pipeline_step_importable(self):
        from robotics.pipeline import PipelineStep  # noqa

    def test_joint_space_rrt_importable(self):
        from robotics.pipeline import JointSpaceRRT  # noqa

    def test_jacobian_ik_importable(self):
        from robotics.ik.jacobian_ik import JacobianIK  # noqa

    def test_scene_config_importable(self):
        from robotics.robot_model import SceneConfig  # noqa


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Pipeline Object Construction
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineConstruction:
    def test_pipeline_has_ik_solver(self, pipeline):
        from robotics.ik.jacobian_ik import JacobianIK
        assert hasattr(pipeline, "_ik")
        assert isinstance(pipeline._ik, JacobianIK)

    def test_pipeline_has_rrt(self, pipeline):
        from robotics.pipeline import JointSpaceRRT
        assert hasattr(pipeline, "_rrt")
        assert isinstance(pipeline._rrt, JointSpaceRRT)

    def test_pipeline_has_cspace(self, pipeline):
        from robotics.rrt.configuration_space import ConfigSpace
        assert hasattr(pipeline, "_cspace_2d")
        assert isinstance(pipeline._cspace_2d, ConfigSpace)

    def test_pipeline_has_arm(self, pipeline, scene):
        assert pipeline.arm is scene.arm

    def test_pipeline_history_exists(self, pipeline):
        assert hasattr(pipeline, "pipeline_history")
        assert isinstance(pipeline.pipeline_history, list)
        assert len(pipeline.pipeline_history) == 0

    def test_pipeline_status_tag(self, pipeline):
        assert pipeline.STATUS == "PHASE_13"


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Scene Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class TestSceneConfiguration:
    def test_books_exist(self, scene):
        assert len(scene.books) >= 1

    def test_obstacles_exist(self, scene):
        assert len(scene.obstacles) >= 1

    def test_book_b1_has_grasp_pos(self, scene):
        bk = scene.get_book("B1")
        g = bk.grasp_position()
        assert g.shape == (3,)
        assert not np.any(np.isnan(g))

    def test_book_b1_has_place_pos(self, scene):
        bk = scene.get_book("B1")
        pp = bk.place_position()
        assert pp is not None
        assert pp.shape == (3,)

    def test_arm_starts_at_home(self, scene):
        scene.arm.set_home()
        assert np.allclose(scene.arm.angles, 0.0, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Jacobian IK Stage (Phase 11) — must not be bypassed
# ═══════════════════════════════════════════════════════════════════════════════

class TestJacobianIKStage:
    """IK is ALWAYS called — ik_result must never be None."""

    def test_ik_result_not_none(self, result):
        assert result.ik_result is not None, \
            "IK must always run — ik_result must not be None"

    def test_ik_result_has_joint_angles(self, result):
        assert result.ik_result.joint_angles_rad.shape[0] == 6

    def test_ik_records_iterations(self, result):
        assert result.ik_result.iterations >= 1

    def test_ik_records_error(self, result):
        err = result.ik_result.position_error_m
        assert 0.0 <= err < 10.0

    def test_ik_stage_in_stages_list(self, result):
        stage_names = [s.stage for s in result.stages]
        assert "Jacobian IK" in stage_names, \
            "IK stage must always be recorded"

    def test_ik_per_iter_error_available(self):
        """IKResult.per_iter_error is populated for convergence plotting."""
        from robotics.ik.jacobian_ik import JacobianIK, IKConfig, ArmConfig
        arm_cfg = ArmConfig.desk_arm_6dof()
        ik = JacobianIK(arm=arm_cfg, config=IKConfig(max_iterations=50))
        s = _make_scene()
        s.arm.set_home()
        target = _reachable_target(s)
        res = ik.solve(target_position=target, initial_angles=s.arm.angles)
        assert hasattr(res, "per_iter_error"), \
            "IKResult must expose per_iter_error"
        assert len(res.per_iter_error) >= 1

    def test_ik_converged_for_reachable_target(self, result):
        """A target within the arm's workspace must converge."""
        assert result.ik_result.converged, \
            f"IK should converge for reachable target; " \
            f"error={result.ik_result.position_error_m*1000:.2f}mm"


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Joint-Space RRT Stage (Phase 10 logic) — must not be bypassed
# ═══════════════════════════════════════════════════════════════════════════════

class TestJointSpaceRRTStage:
    """RRT is called after IK, path is returned."""

    def test_rrt_produces_path(self, result):
        # Even on RRT failure, fallback gives waypoints
        assert result.rrt_n_waypoints > 0, \
            "RRT must always produce waypoints (or fallback interpolation)"

    def test_rrt_path_is_joint_space(self, result):
        for q in result.rrt_path_joints:
            assert q.shape[0] == 6

    def test_rrt_stage_recorded(self, result):
        stage_names = [s.stage for s in result.stages]
        assert "Joint-Space RRT" in stage_names, \
            "RRT stage must always be recorded when IK succeeds"

    def test_rrt_planning_time_recorded(self, result):
        assert result.rrt_planning_ms >= 0.0

    def test_rrt_n_nodes_recorded(self, result):
        assert result.rrt_n_nodes >= 0

    def test_rrt_cartesian_projection(self, result):
        assert len(result.rrt_path_cartesian) == result.rrt_n_waypoints

    def test_rrt_collision_status_reported(self, result):
        assert isinstance(result.collision_status, str)
        assert isinstance(result.collision_free, bool)

    def test_rrt_uses_phase10_collision_logic(self, pipeline):
        from robotics.rrt.collision_checker import CollisionChecker
        assert hasattr(pipeline._rrt, "checker_2d")
        assert isinstance(pipeline._rrt.checker_2d, CollisionChecker)

    def test_rrt_direct_plan_small_move(self, pipeline):
        """Plan a trivial joint-space move; RRT must return success."""
        q_start = np.zeros(6)
        q_goal  = np.array([0.1, 0.05, 0.05, 0.0, 0.0, 0.0])
        ok, path, n_nodes, _, length, status = pipeline._rrt.plan(q_start, q_goal)
        assert ok, f"RRT must find path for small move; status={status}"
        assert len(path) >= 2
        assert path[0].shape[0] == 6


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Virtual Robot Execution Stage (Phase 12) — must not be bypassed
# ═══════════════════════════════════════════════════════════════════════════════

class TestVirtualRobotExecution:
    """Robot executes every RRT waypoint and updates arm state."""

    def test_execution_stage_recorded(self, result):
        stage_names = [s.stage for s in result.stages]
        assert "Execute + Verify" in stage_names

    def test_robot_moves_arm(self, pipeline, scene, target):
        scene.arm.set_home()
        initial_angles = scene.arm.angles.copy()
        pipeline.plan_and_execute(target_pos=target)
        assert not np.allclose(initial_angles, scene.arm.angles, atol=1e-4), \
            "Arm must move during execution"

    def test_executed_steps_positive(self, result):
        assert result.executed_steps > 0

    def test_final_ee_recorded(self, result):
        assert result.final_ee_pos is not None
        assert result.final_ee_pos.shape == (3,)
        assert not np.any(np.isnan(result.final_ee_pos))

    def test_final_error_computed(self, result):
        assert result.final_error_m >= 0.0

    def test_steps_proportional_to_waypoints(self, result):
        min_steps = max(0, result.rrt_n_waypoints - 1)
        assert result.executed_steps >= min_steps


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Pipeline Order Verification  (IK → RRT → Execute)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineOrder:
    """All three stages are run in strict order."""

    def test_three_stages_recorded(self, result):
        assert len(result.stages) >= 3

    def test_stage_1_is_ik(self, result):
        assert result.stages[0].stage == "Jacobian IK"

    def test_stage_2_is_rrt(self, result):
        assert result.stages[1].stage == "Joint-Space RRT"

    def test_stage_3_is_exec(self, result):
        assert result.stages[2].stage == "Execute + Verify"

    def test_all_stages_have_nonneg_duration(self, result):
        for s in result.stages:
            assert s.duration_ms >= 0.0, f"Stage {s.stage!r} must record timing"

    def test_pipeline_history_grows_per_call(self, pipeline, target, scene):
        scene.arm.set_home()
        before = len(pipeline.pipeline_history)
        pipeline.plan_and_execute(target_pos=target)
        assert len(pipeline.pipeline_history) == before + 1


# ═══════════════════════════════════════════════════════════════════════════════
#  8. Serialisation — latest_result_dict
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineSerialisation:
    def test_latest_result_dict_not_none(self, pipeline, scene, target):
        scene.arm.set_home()
        pipeline.plan_and_execute(target_pos=target)
        assert pipeline.latest_result_dict() is not None

    def test_dict_ik_keys(self, pipeline, scene, target):
        scene.arm.set_home()
        pipeline.plan_and_execute(target_pos=target)
        d = pipeline.latest_result_dict()
        for k in ("converged", "iterations", "error_mm", "joint_angles_deg", "final_pos"):
            assert k in d["ik"], f"Missing IK key: {k}"

    def test_dict_rrt_keys(self, pipeline, scene, target):
        scene.arm.set_home()
        pipeline.plan_and_execute(target_pos=target)
        d = pipeline.latest_result_dict()
        for k in ("n_waypoints", "n_nodes", "planning_ms", "collision_status", "path_cartesian"):
            assert k in d["rrt"], f"Missing RRT key: {k}"

    def test_dict_execution_keys(self, pipeline, scene, target):
        scene.arm.set_home()
        pipeline.plan_and_execute(target_pos=target)
        d = pipeline.latest_result_dict()
        for k in ("steps", "final_ee", "final_err_mm"):
            assert k in d["execution"], f"Missing execution key: {k}"

    def test_dict_stages_list(self, pipeline, scene, target):
        scene.arm.set_home()
        pipeline.plan_and_execute(target_pos=target)
        d = pipeline.latest_result_dict()
        assert isinstance(d["stages"], list)
        assert len(d["stages"]) >= 3

    def test_dict_success_is_bool(self, pipeline, scene, target):
        scene.arm.set_home()
        pipeline.plan_and_execute(target_pos=target)
        d = pipeline.latest_result_dict()
        assert isinstance(d["success"], bool)

    def test_rrt_path_cartesian_pairs(self, pipeline, scene, target):
        scene.arm.set_home()
        pipeline.plan_and_execute(target_pos=target)
        d = pipeline.latest_result_dict()
        for pt in d["rrt"]["path_cartesian"]:
            assert len(pt) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
#  9. Full Pick-and-Place (6 motions)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPickAndPlace:
    """Run all 6 motions; every motion must invoke IK + RRT + Robot."""

    @staticmethod
    def _build_motions(scene):
        bk  = scene.get_book("B1")
        g   = bk.grasp_position()
        ab  = g.copy();   ab[2]  += 0.12
        pp  = bk.place_position()
        if pp is None:
            pp = g.copy(); pp[0] += 0.30
        ap  = pp.copy();  ap[2]  += 0.12
        ret = ap.copy();  ret[2] += 0.08
        return [
            ("Approach",    ab,  None,    None),
            ("Lower-grasp", g,   "close", bk),
            ("Lift",        ab,  None,    bk),
            ("Carry",       ap,  None,    bk),
            ("Place",       pp,  "open",  None),
            ("Retreat",     ret, None,    None),
        ]

    def test_six_motions_all_have_ik(self, scene):
        from robotics.pipeline import RobotPipeline
        scene.arm.set_home()
        pipe = RobotPipeline(arm=scene.arm, scene=scene)
        for name, tgt, gcmd, abk in self._build_motions(scene):
            r = pipe.plan_and_execute(target_pos=tgt, gripper_action=gcmd, active_book=abk)
            assert r.ik_result is not None, f"{name}: IK must always run"

    def test_six_motions_all_have_rrt(self, scene):
        from robotics.pipeline import RobotPipeline
        scene.arm.set_home()
        pipe = RobotPipeline(arm=scene.arm, scene=scene)
        for name, tgt, gcmd, abk in self._build_motions(scene):
            r = pipe.plan_and_execute(target_pos=tgt, gripper_action=gcmd, active_book=abk)
            if r.ik_result and r.ik_result.converged:
                assert r.rrt_n_waypoints > 0, f"{name}: RRT must produce waypoints"

    def test_six_motions_stages_count(self, scene):
        from robotics.pipeline import RobotPipeline
        scene.arm.set_home()
        pipe = RobotPipeline(arm=scene.arm, scene=scene)
        for name, tgt, gcmd, abk in self._build_motions(scene):
            r = pipe.plan_and_execute(target_pos=tgt, gripper_action=gcmd, active_book=abk)
            # When IK succeeds: 3 stages; when it fails: 1 stage
            assert len(r.stages) >= 1, f"{name}: at least 1 stage must be recorded"
            if r.ik_result and r.ik_result.converged:
                assert len(r.stages) >= 3, f"{name}: 3 stages expected when IK converges"

    def test_history_accumulates_correctly(self, scene):
        from robotics.pipeline import RobotPipeline
        scene.arm.set_home()
        pipe = RobotPipeline(arm=scene.arm, scene=scene)
        motions = self._build_motions(scene)
        for _, tgt, gcmd, abk in motions:
            pipe.plan_and_execute(target_pos=tgt, gripper_action=gcmd, active_book=abk)
        assert len(pipe.pipeline_history) == len(motions)

    def test_all_history_dict(self, scene):
        from robotics.pipeline import RobotPipeline
        scene.arm.set_home()
        pipe = RobotPipeline(arm=scene.arm, scene=scene)
        tgt = _reachable_target(scene)
        pipe.plan_and_execute(target_pos=tgt)
        pipe.plan_and_execute(target_pos=tgt)
        hist = pipe.all_history_dict()
        assert len(hist) == 2
        for entry in hist:
            assert "ik_ok" in entry
            assert "rrt_ok" in entry
            assert "collision_free" in entry


# ═══════════════════════════════════════════════════════════════════════════════
#  10. JointSpaceRRT Unit Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestJointSpaceRRTUnit:
    def test_rrt_has_arm_config(self, pipeline):
        assert hasattr(pipeline._rrt, "arm_config")

    def test_rrt_has_cspace_2d(self, pipeline):
        assert hasattr(pipeline._rrt, "cspace_2d")

    def test_rrt_has_phase10_checker(self, pipeline):
        from robotics.rrt.collision_checker import CollisionChecker
        assert isinstance(pipeline._rrt.checker_2d, CollisionChecker)

    def test_rrt_has_fk(self, pipeline):
        from robotics.ik.jacobian_ik import ForwardKinematics
        assert hasattr(pipeline._rrt, "fk")
        assert isinstance(pipeline._rrt.fk, ForwardKinematics)

    def test_ee_xy_returns_floats(self, pipeline):
        xy = pipeline._rrt._ee_xy(np.zeros(6))
        assert len(xy) == 2
        assert all(isinstance(v, float) for v in xy)

    def test_home_config_is_free(self, pipeline):
        assert pipeline._rrt._is_config_free(np.zeros(6))

    def test_sample_in_bounds(self, pipeline):
        rrt = pipeline._rrt
        q = rrt._sample(np.zeros(6))
        assert q.shape[0] == 6
        assert np.all(q >= rrt._lo - 1e-6)
        assert np.all(q <= rrt._hi + 1e-6)

    def test_steer_bounded_step(self, pipeline):
        rrt = pipeline._rrt
        q_new = rrt._steer(np.zeros(6), np.ones(6) * 1.5)
        assert float(np.linalg.norm(q_new)) <= rrt.step_size + 1e-9

    def test_nearest_correct_index(self, pipeline):
        rrt = pipeline._rrt
        nodes = [np.zeros(6), np.ones(6), np.ones(6) * 2]
        assert rrt._nearest(nodes, np.array([0.01]*6)) == 0

    def test_plan_trivial_goal(self, pipeline):
        q = np.zeros(6)
        ok, path, _, _, _, status = pipeline._rrt.plan(q, q.copy())
        assert ok
        assert len(path) >= 2
        assert status == "clear"

    def test_plan_small_move(self, pipeline):
        q_start = np.zeros(6)
        q_goal  = np.array([0.1, 0.05, 0.05, 0.0, 0.0, 0.0])
        ok, path, _, _, _, _ = pipeline._rrt.plan(q_start, q_goal)
        if ok:
            for q in path:
                assert q.shape[0] == 6


# ═══════════════════════════════════════════════════════════════════════════════
#  11. IKResult Fields Required by Phase 13 UI
# ═══════════════════════════════════════════════════════════════════════════════

class TestIKResultContract:
    def test_has_converged(self, result):
        assert hasattr(result.ik_result, "converged")

    def test_has_iterations(self, result):
        assert result.ik_result.iterations >= 1

    def test_has_position_error_m(self, result):
        assert hasattr(result.ik_result, "position_error_m")

    def test_has_joint_angles_deg(self, result):
        assert result.ik_result.joint_angles_deg.shape[0] == 6

    def test_has_final_position(self, result):
        assert result.ik_result.final_position.shape == (3,)

    def test_has_to_dict(self, result):
        d = result.ik_result.to_dict()
        assert isinstance(d, dict)
        assert "converged" in d

    def test_has_per_iter_error(self, result):
        assert hasattr(result.ik_result, "per_iter_error")
        assert isinstance(result.ik_result.per_iter_error, list)


# ═══════════════════════════════════════════════════════════════════════════════
#  12. Collision Reporting
# ═══════════════════════════════════════════════════════════════════════════════

class TestCollisionReporting:
    def test_collision_free_is_bool(self, result):
        assert isinstance(result.collision_free, bool)

    def test_collision_status_is_str(self, result):
        assert isinstance(result.collision_status, str)

    def test_collision_in_dict(self, pipeline, scene, target):
        scene.arm.set_home()
        pipeline.plan_and_execute(target_pos=target)
        d = pipeline.latest_result_dict()
        assert "collision_status" in d["rrt"]
        assert "collision_free"   in d["rrt"]


# ═══════════════════════════════════════════════════════════════════════════════
#  13. Phase 13 Identity
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase13Identity:
    def test_robot_pipeline_status(self):
        from robotics.pipeline import RobotPipeline
        assert RobotPipeline.STATUS == "PHASE_13"

    def test_pipeline_page_exists(self):
        page_path = PROJECT_ROOT / "visualization" / "pipeline_page.py"
        assert page_path.exists(), "pipeline_page.py must exist"

    def test_pipeline_page_has_render_function(self):
        page_path = PROJECT_ROOT / "visualization" / "pipeline_page.py"
        src = page_path.read_text(encoding="utf-8")
        assert "def render_pipeline_page" in src

    def test_pipeline_page_references_all_three_subsystems(self):
        page_path = PROJECT_ROOT / "visualization" / "pipeline_page.py"
        src = page_path.read_text(encoding="utf-8")
        assert "Jacobian IK" in src or "jacobian" in src.lower()
        assert "RRT" in src
        assert "Robot" in src or "execute" in src.lower()

    def test_app_py_has_phase13_route(self):
        app_path = PROJECT_ROOT / "app.py"
        src = app_path.read_text(encoding="utf-8")
        assert "Phase 13" in src or "pipeline_page" in src

    def test_app_py_calls_render_pipeline_page(self):
        app_path = PROJECT_ROOT / "app.py"
        src = app_path.read_text(encoding="utf-8")
        assert "render_pipeline_page" in src


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
