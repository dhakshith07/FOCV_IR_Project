"""
workflow/ - User Workflow Analysis module.
Phase 7 implementation.

Exports
-------
WorkflowAnalyzer  : Core engine - records events, builds transition matrix,
                    computes all six metrics, predicts next book.
UsageAnalyzer     : High-level facade - charts, summary DataFrames.
ContextDetector   : Session context classifier (exam_prep, research, ...).
InteractionEvent  : In-memory event container.
BookMetrics       : Per-book metric container.
ContextResult     : Context detection result.
"""

from workflow.workflow_analyzer import (
    WorkflowAnalyzer,
    InteractionEvent,
    BookMetrics,
)
from workflow.usage_analyzer import UsageAnalyzer
from workflow.context_detector import ContextDetector, ContextResult

__all__ = [
    "WorkflowAnalyzer",
    "UsageAnalyzer",
    "ContextDetector",
    "InteractionEvent",
    "BookMetrics",
    "ContextResult",
]
