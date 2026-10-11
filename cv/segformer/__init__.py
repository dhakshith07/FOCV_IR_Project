"""
cv/segformer/__init__.py
Phase 6 - SegFormer semantic workspace segmentation.

Exports
-------
WorkspaceSegmenter  : Main segmenter - load(), segment(), draw_all_panels().
SegmentationResult  : Full result from one inference pass.
WorkspaceRegion     : A contiguous workspace sub-region.
SegmenterMode       : NOT_LOADED | PRETRAINED | CUSTOM | SYNTHETIC
WorkspaceZone       : Enum of pixel-level semantic labels.
WORKSPACE_CLASS_IDS : ADE20K IDs treated as valid surface.
OBSTACLE_CLASS_IDS  : ADE20K IDs treated as obstacles.
"""

from cv.segformer.workspace_segmenter import (
    WorkspaceSegmenter,
    SegmentationResult,
    WorkspaceRegion,
    SegmenterMode,
    WorkspaceZone,
    WORKSPACE_CLASS_IDS,
    OBSTACLE_CLASS_IDS,
    ADE20K_LABELS,
    DEFAULT_MODEL,
)

__all__ = [
    "WorkspaceSegmenter",
    "SegmentationResult",
    "WorkspaceRegion",
    "SegmenterMode",
    "WorkspaceZone",
    "WORKSPACE_CLASS_IDS",
    "OBSTACLE_CLASS_IDS",
    "ADE20K_LABELS",
    "DEFAULT_MODEL",
]
