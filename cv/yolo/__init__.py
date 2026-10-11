"""
cv/yolo/ — YOLO11 object detection package.

Phase 3 Public API
------------------
  Detection       — single detection result dataclass
  ModelStatus     — enum of model loading states
  BookDetector    — YOLO11 book detector (custom / pretrained / simulation)
  VideoProcessor  — batch video/webcam processing with statistics
  SyntheticDeskGenerator — simulation fallback image + detection generator
"""

from cv.yolo.detection          import Detection
from cv.yolo.book_detector      import BookDetector, ModelStatus, COCO_BOOK_CLASS_ID
from cv.yolo.video_processor    import VideoProcessor, ProcessingStats
from cv.yolo.synthetic_generator import SyntheticDeskGenerator

__all__ = [
    "Detection",
    "BookDetector",
    "ModelStatus",
    "COCO_BOOK_CLASS_ID",
    "VideoProcessor",
    "ProcessingStats",
    "SyntheticDeskGenerator",
]
