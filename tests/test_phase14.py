"""
tests/test_phase14.py
=====================
Phase 14 — Intelligent Robotics End-to-End Pipeline Integration Tests.

Verifies the strict pipeline:

    YOLO11
      ↓  Book detection
    ByteTrack
      ↓  Book tracking
    TrOCR
      ↓  Book title/text recognition
    SegFormer
      ↓  Workspace/obstacle segmentation
    Workspace State
      ↓  Consolidated multi-modal state
    Workflow Analysis
      ↓  Usage frequency & transition matrix
    Genetic Algorithm
      ↓  Global placement optimization
    Optimal Book Arrangement
      ↓  Target Cartesian coordinates
    DQN
      ↓  Next robot action selection
    Jacobian IK
      ↓  End-effector target → joint configuration
    Target Joint Configuration
      ↓  Convergence & joint limit check
    RRT
      ↓  Collision-free robot path planning
    Collision-Free Robot Path
      ↓  Smoothed joint-space waypoints
    Virtual Robot
      ↓  Movement execution & FK
    Book Placement
      ↓  Pick, transport, place
    Verification
      ↓  Final pose, clearance & containment

Verifies that the complete demo executes the 15-step sequence:
    1. Detect books
    2. Identify books
    3. Track books
    4. Analyze workflow
    5. Optimize layout
    6. Select robot action
    7. Calculate IK
    8. Generate RRT path
    9. Move robot
    10. Pick book
    11. Move to target
    12. Place book
    13. Verify
    14. Continue
    15. Finish

Run:
    pytest tests/test_phase14.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

# ── Ensure project root is on sys.path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.pipeline_controller import (
    ExecutionStage,
    PipelineStepResult,
    PipelineBookItem,
    PipelineDemoResult,
    PipelineController,
)
from cv.yolo.book_detector import BookDetector
from cv.tracking.bytetrack_tracker import ByteTrackTracker
from cv.trocr.book_text_reader import BookTextReader
from cv.segformer.workspace_segmenter import WorkspaceSegmenter
from workflow.workflow_analyzer import WorkflowAnalyzer
from optimization.placement_optimizer import PlacementOptimizer, OptimizationResult
from robotics.dqn.dqn_agent import DQNAgent
from robotics.ik.jacobian_ik import JacobianIK, IKResult
from robotics.pipeline import JointSpaceRRT, RobotPipeline
from robotics.robot_model import RobotArm, SceneConfig


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def controller() -> PipelineController:
    """Fresh PipelineController instance."""
    scene = SceneConfig.default_scene()
    scene.arm.set_home()
    return PipelineController(scene=scene, seed=42)


@pytest.fixture
def demo_result(controller: PipelineController) -> PipelineDemoResult:
    """Run full demo (single-book pass) to test all 15 stages."""
    return controller.run_demo(single_book=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Imports and Class Availability
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase14Imports:
    def test_controller_importable(self):
        from orchestrator import PipelineController
        assert PipelineController is not None

    def test_demo_result_importable(self):
        from orchestrator import PipelineDemoResult
        assert PipelineDemoResult is not None

    def test_step_result_importable(self):
        from orchestrator import PipelineStepResult
        assert PipelineStepResult is not None

    def test_book_item_importable(self):
        from orchestrator import PipelineBookItem
        assert PipelineBookItem is not None

    def test_execution_stage_enum_complete(self):
        expected_stages = [
            "1. Detect books",
            "2. Identify books",
            "3. Track books",
            "4. Analyze workflow",
            "5. Optimize layout",
            "6. Select robot action",
            "7. Calculate IK",
            "8. Generate RRT path",
            "9. Move robot",
            "10. Pick book",
            "11. Move to target",
            "12. Place book",
            "13. Verify",
            "14. Continue",
            "15. Finish",
        ]
        values = [s.value for s in ExecutionStage]
        for stage in expected_stages:
            assert stage in values, f"Stage '{stage}' missing from ExecutionStage enum"


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Component Verification (No Subsystem Replaced or Bypassed)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSubsystemIntegrity:
    def test_controller_status(self, controller: PipelineController):
        assert controller.STATUS == "PHASE_14"

    def test_yolo11_detector_attached(self, controller: PipelineController):
        assert hasattr(controller, "detector")
        assert isinstance(controller.detector, BookDetector)

    def test_bytetrack_tracker_attached(self, controller: PipelineController):
        assert hasattr(controller, "tracker")
        assert isinstance(controller.tracker, ByteTrackTracker)

    def test_trocr_reader_attached(self, controller: PipelineController):
        assert hasattr(controller, "ocr_reader")
        assert isinstance(controller.ocr_reader, BookTextReader)

    def test_segformer_segmenter_attached(self, controller: PipelineController):
        assert hasattr(controller, "segmenter")
        assert isinstance(controller.segmenter, WorkspaceSegmenter)

    def test_workflow_analyzer_attached(self, controller: PipelineController):
        assert hasattr(controller, "workflow_analyzer")
        assert isinstance(controller.workflow_analyzer, WorkflowAnalyzer)

    def test_dqn_agent_attached(self, controller: PipelineController):
        assert hasattr(controller, "dqn_agent")
        assert isinstance(controller.dqn_agent, DQNAgent)

    def test_jacobian_ik_attached(self, controller: PipelineController):
        assert hasattr(controller, "ik_solver")
        assert isinstance(controller.ik_solver, JacobianIK)

    def test_joint_space_rrt_attached(self, controller: PipelineController):
        assert hasattr(controller, "rrt_planner")
        assert isinstance(controller.rrt_planner, JointSpaceRRT)

    def test_robot_arm_attached(self, controller: PipelineController):
        assert hasattr(controller, "arm")
        assert isinstance(controller.arm, RobotArm)

    def test_scene_attached(self, controller: PipelineController):
        assert hasattr(controller, "scene")
        assert isinstance(controller.scene, SceneConfig)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Step-by-Step Execution Verification (Steps 1 to 15)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCanonicalSteps:
    def test_step_1_detect_books(self, controller: PipelineController):
        dets, frame, rec = controller.step_1_detect_books()
        assert rec.step_number == 1
        assert "YOLO11" in rec.algorithm_role
        assert rec.success is True
        assert len(dets) >= 1
        assert frame is not None and frame.shape[0] > 0

    def test_step_2_identify_books(self, controller: PipelineController):
        dets, frame, _ = controller.step_1_detect_books()
        identifications, rec = controller.step_2_identify_books(frame, dets)
        assert rec.step_number == 2
        assert "TrOCR" in rec.algorithm_role
        assert rec.success is True
        assert len(identifications) >= 1
        for idx, item in identifications.items():
            assert item.title != ""

    def test_step_3_track_books(self, controller: PipelineController):
        dets, frame, _ = controller.step_1_detect_books()
        identifications, _ = controller.step_2_identify_books(frame, dets)
        tracks, rec = controller.step_3_track_books(frame, dets, identifications)
        assert rec.step_number == 3
        assert "ByteTrack" in rec.algorithm_role
        assert rec.success is True
        assert len(tracks) >= 1
        assert len(controller.tracked_items) >= 1

    def test_segformer_segmentation(self, controller: PipelineController):
        _, frame, _ = controller.step_1_detect_books()
        seg_res, rec = controller.execute_segformer_segmentation(frame)
        assert "SegFormer" in rec.algorithm_role
        assert seg_res.workspace_mask is not None
        assert seg_res.obstacle_map is not None

    def test_step_4_analyze_workflow(self, controller: PipelineController):
        dets, frame, _ = controller.step_1_detect_books()
        idents, _ = controller.step_2_identify_books(frame, dets)
        controller.step_3_track_books(frame, dets, idents)

        usage_freqs, wf_mat, metrics, rec = controller.step_4_analyze_workflow()
        assert rec.step_number == 4
        assert rec.success is True
        assert len(usage_freqs) >= 1
        assert wf_mat.ndim == 2
        assert wf_mat.shape[0] == wf_mat.shape[1]

    def test_step_5_optimize_layout(self, controller: PipelineController):
        dets, frame, _ = controller.step_1_detect_books()
        idents, _ = controller.step_2_identify_books(frame, dets)
        controller.step_3_track_books(frame, dets, idents)
        freqs, wf_mat, _, _ = controller.step_4_analyze_workflow()

        opt_res, rec = controller.step_5_optimize_layout(controller.tracked_items, freqs, wf_mat)
        assert rec.step_number == 5
        assert "Genetic Algorithm" in rec.algorithm_role
        assert rec.success is True
        assert isinstance(opt_res, OptimizationResult)
        assert len(opt_res.optimized_arrangement) == len(controller.tracked_items)

    def test_step_6_select_robot_action(self, controller: PipelineController):
        items = list(controller.tracked_items.values())
        if not items:
            dets, frame, _ = controller.step_1_detect_books()
            idents, _ = controller.step_2_identify_books(frame, dets)
            controller.step_3_track_books(frame, dets, idents)
            items = list(controller.tracked_items.values())

        action_id, action_desc, rec = controller.step_6_select_robot_action(items, items)
        assert rec.step_number == 6
        assert "DQN" in rec.algorithm_role
        assert rec.success is True
        assert action_id >= 0
        assert action_desc != ""

    def test_step_7_calculate_ik(self, controller: PipelineController):
        target = np.array([0.0, 0.30, 0.35], dtype=np.float64)
        ik_res, rec = controller.step_7_calculate_ik(target)
        assert rec.step_number == 7
        assert "Jacobian IK" in rec.algorithm_role
        assert isinstance(ik_res, IKResult)
        assert ik_res.converged is True
        assert ik_res.position_error_m < 0.01

    def test_step_8_generate_rrt_path(self, controller: PipelineController):
        q_start = np.zeros(6)
        q_goal = np.array([0.1, 0.05, 0.05, 0.0, 0.0, 0.0])
        ok, path, rec = controller.step_8_generate_rrt_path(q_start, q_goal)
        assert rec.step_number == 8
        assert "RRT" in rec.algorithm_role
        assert rec.success is True
        assert len(path) >= 2

    def test_step_9_move_robot(self, controller: PipelineController):
        q0 = np.zeros(6)
        q1 = np.array([0.05, 0.02, 0.02, 0.0, 0.0, 0.0])
        path = [q0, q1]
        steps, rec = controller.step_9_move_robot(path)
        assert rec.step_number == 9
        assert "Virtual Robot" in rec.algorithm_role
        assert steps > 0
        assert rec.success is True

    def test_step_10_pick_book(self, controller: PipelineController):
        book = controller.scene.books[0]
        item = PipelineBookItem(
            book_id=book.book_id,
            title=book.title,
            current_position=np.array(book.position, dtype=np.float64),
        )
        ok, rec = controller.step_10_pick_book(item)
        assert rec.step_number == 10
        assert ok is True
        assert item.is_held is True
        assert rec.success is True

    def test_step_11_move_to_target(self, controller: PipelineController):
        book = controller.scene.books[0]
        item = PipelineBookItem(
            book_id=book.book_id,
            title=book.title,
            current_position=np.array(book.position, dtype=np.float64),
            is_held=True,
        )
        target = np.array([0.20, 0.40, 0.11], dtype=np.float64)
        ok, rec = controller.step_11_move_to_target(item, target)
        assert rec.step_number == 11
        assert ok is True
        assert rec.success is True

    def test_step_12_place_book(self, controller: PipelineController):
        book = controller.scene.books[0]
        item = PipelineBookItem(
            book_id=book.book_id,
            title=book.title,
            current_position=np.array(book.position, dtype=np.float64),
            is_held=True,
        )
        target = np.array([0.20, 0.40, 0.11], dtype=np.float64)
        ok, rec = controller.step_12_place_book(item, target)
        assert rec.step_number == 12
        assert ok is True
        assert item.is_held is False
        assert rec.success is True

    def test_step_13_verify(self, controller: PipelineController):
        target = np.array([0.20, 0.40, 0.11], dtype=np.float64)
        item = PipelineBookItem(
            book_id="B1",
            title="Machine Learning",
            current_position=target.copy(),
        )
        verified, err_mm, rec = controller.step_13_verify(item, target)
        assert rec.step_number == 13
        assert verified
        assert err_mm < 1.0
        assert item.verified
        assert rec.success

    def test_step_14_continue(self, controller: PipelineController):
        item = PipelineBookItem(book_id="B2", title="Test")
        has_next, rec = controller.step_14_continue([item])
        assert rec.step_number == 14
        assert has_next is True
        assert rec.success is True

        has_next_empty, _ = controller.step_14_continue([])
        assert has_next_empty is False

    def test_step_15_finish(self, controller: PipelineController):
        rec = controller.step_15_finish()
        assert rec.step_number == 15
        assert rec.success is True
        assert np.allclose(controller.arm.angles, 0.0, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Full End-to-End Demo Execution Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipelineDemo:
    def test_demo_returns_result_object(self, demo_result: PipelineDemoResult):
        assert isinstance(demo_result, PipelineDemoResult)

    def test_demo_contains_all_15_steps(self, demo_result: PipelineDemoResult):
        executed_numbers = {s.step_number for s in demo_result.steps}
        for n in range(1, 16):
            assert n in executed_numbers, f"Step {n} was not executed in the demo"

    def test_demo_overall_success(self, demo_result: PipelineDemoResult):
        assert demo_result.success is True

    def test_demo_books_detected_and_identified(self, demo_result: PipelineDemoResult):
        assert len(demo_result.books) >= 1
        for bid, b in demo_result.books.items():
            assert b.title != ""
            assert b.track_id > 0

    def test_demo_ga_layout_optimized(self, demo_result: PipelineDemoResult):
        assert demo_result.optimization_result is not None
        assert demo_result.optimization_result.cost_before >= demo_result.optimization_result.cost_after

    def test_demo_dqn_action_selected(self, demo_result: PipelineDemoResult):
        assert len(demo_result.dqn_actions) >= 1

    def test_demo_verification_completed(self, demo_result: PipelineDemoResult):
        placed_books = [b for b in demo_result.books.values() if b.is_placed]
        assert len(placed_books) >= 1
        for b in placed_books:
            assert b.verified
            assert b.final_error_mm <= 45.0

    def test_demo_summary_output(self, demo_result: PipelineDemoResult):
        summary = demo_result.summary()
        assert "PHASE 14" in summary
        assert "YOLO11" in summary or "Detect books" in summary
        assert "TrOCR" in summary or "Identify books" in summary
        assert "ByteTrack" in summary or "Track books" in summary
        assert "Genetic Algorithm" in summary or "Optimize layout" in summary
        assert "Jacobian IK" in summary or "Calculate IK" in summary
        assert "RRT" in summary or "Generate RRT path" in summary
        assert "SUCCESS" in summary

    def test_demo_to_dict_serialisation(self, demo_result: PipelineDemoResult):
        d = demo_result.to_dict()
        assert isinstance(d, dict)
        assert d["success"] is True
        assert len(d["steps"]) >= 15
        assert "books" in d
        assert "dqn_actions" in d

    def test_demo_multi_book_execution(self, controller: PipelineController):
        """Execute complete pipeline placing multiple books."""
        res = controller.run_demo(max_books=2, single_book=False)
        assert res.success is True
        placed = [b for b in res.books.values() if b.is_placed]
        assert len(placed) >= 2
        for b in placed:
            assert b.verified
