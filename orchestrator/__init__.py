"""
orchestrator
============
Phase 14 — Intelligent Robotics End-to-End Orchestrator.

Integrates the complete perception-to-action robotics pipeline:
    YOLO11 → ByteTrack → TrOCR → SegFormer → Workspace State →
    Workflow Analysis → Genetic Algorithm → Optimal Arrangement →
    DQN Action Selection → Jacobian IK → Joint-Space RRT →
    Virtual Robot Execution → Book Placement → Verification.
"""

from orchestrator.pipeline_controller import (
    ExecutionStage,
    PipelineStepResult,
    PipelineBookItem,
    PipelineDemoResult,
    PipelineController,
)

__all__ = [
    "ExecutionStage",
    "PipelineStepResult",
    "PipelineBookItem",
    "PipelineDemoResult",
    "PipelineController",
]
