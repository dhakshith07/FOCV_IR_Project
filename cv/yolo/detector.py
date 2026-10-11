"""
cv/yolo/detector.py
YOLO11 detector — Phase 3 public re-export.

This module is the legacy entry-point kept for backward compatibility.
The full implementation lives in cv/yolo/book_detector.py.

Quick usage
-----------
    from cv.yolo.detector import BookDetector, Detection, ModelStatus

    detector = BookDetector()
    detector.load()
    frame = cv2.imread("desk.jpg")
    detections = detector.detect(frame)
    annotated  = detector.draw_detections(frame, detections)

Custom model
------------
    Place a fine-tuned weight file at:
        models/yolo11_book_detector.pt

    BookDetector will automatically prefer it over the COCO pretrained model.
    If the file is absent a clear info message is logged and YOLO11n (COCO) is used.

Simulation fallback
-------------------
    When neither ultralytics nor a model can be loaded the detector enters
    SIMULATION_MODE.  Call detector.simulate_detections(workspace, shape) to
    obtain realistic synthetic Detection objects from the Phase 2 WorkspaceSimulator.
"""

from __future__ import annotations

# ── Re-export the full implementation ────────────────────────────────────────
from cv.yolo.book_detector import (  # noqa: F401  (re-export)
    BookDetector,
    ModelStatus,
    COCO_BOOK_CLASS_ID,
    CUSTOM_MODEL_PATH,
    FALLBACK_MODEL,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
)
from cv.yolo.detection import Detection  # noqa: F401
from cv.yolo.video_processor import VideoProcessor, ProcessingStats  # noqa: F401
from cv.yolo.synthetic_generator import SyntheticDeskGenerator  # noqa: F401

__all__ = [
    "BookDetector",
    "Detection",
    "ModelStatus",
    "COCO_BOOK_CLASS_ID",
    "CUSTOM_MODEL_PATH",
    "FALLBACK_MODEL",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_IOU",
    "VideoProcessor",
    "ProcessingStats",
    "SyntheticDeskGenerator",
]
