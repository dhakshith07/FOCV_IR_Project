"""
cv/yolo/detection.py
Detection dataclass — one detected object per instance.

Produced by BookDetector and consumed by the tracker (Phase 4),
workflow analyzer (Phase 4), and the Streamlit dashboard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple


# ── Colour palette for visualisation (BGR order for OpenCV) ─────────────────
_BOOK_COLOR_BGR = (72, 187, 120)     # green — real YOLO detection
_SIM_COLOR_BGR  = (237, 137, 54)     # orange — simulated detection


@dataclass
class Detection:
    """
    A single object detection result from YOLO11.

    Attributes
    ----------
    x1, y1         : Top-left corner of bounding box (pixels)
    x2, y2         : Bottom-right corner of bounding box (pixels)
    confidence      : Detection confidence [0, 1]
    class_id        : YOLO class index
    class_name      : Human-readable class label (e.g. 'book')
    source          : 'yolo' | 'simulation' | 'custom'
    frame_index     : Frame number within video/webcam stream
    timestamp       : Wall-clock time of detection
    color_bgr       : BGR colour used for drawing
    """

    x1:          float
    y1:          float
    x2:          float
    y2:          float
    confidence:  float
    class_id:    int
    class_name:  str
    source:      str              = "yolo"
    frame_index: int              = 0
    timestamp:   datetime         = field(default_factory=datetime.now)
    color_bgr:   Tuple[int, int, int] = field(default=None)

    def __post_init__(self) -> None:
        # Ensure x1 < x2 and y1 < y2
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1
        # Default colour based on source
        if self.color_bgr is None:
            self.color_bgr = (
                _SIM_COLOR_BGR if self.source == "simulation" else _BOOK_COLOR_BGR
            )

    # ── Geometry ──────────────────────────────────────────────────────────
    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_x, self.center_y)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """(x1, y1, x2, y2) bounding box."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def bbox_int(self) -> Tuple[int, int, int, int]:
        """Integer bounding box for OpenCV drawing."""
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    @property
    def center_int(self) -> Tuple[int, int]:
        return (int(self.center_x), int(self.center_y))

    def iou(self, other: "Detection") -> float:
        """Intersection-over-Union with another detection."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    # ── Label ─────────────────────────────────────────────────────────────
    @property
    def label(self) -> str:
        return f"{self.class_name} {self.confidence:.2f}"

    # ── Serialisation ─────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "x1":          round(self.x1, 2),
            "y1":          round(self.y1, 2),
            "x2":          round(self.x2, 2),
            "y2":          round(self.y2, 2),
            "center_x":    round(self.center_x, 2),
            "center_y":    round(self.center_y, 2),
            "width":       round(self.width, 2),
            "height":      round(self.height, 2),
            "area":        round(self.area, 2),
            "confidence":  round(self.confidence, 4),
            "class_id":    self.class_id,
            "class_name":  self.class_name,
            "source":      self.source,
            "frame_index": self.frame_index,
            "timestamp":   self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"Detection(class={self.class_name!r}, "
            f"conf={self.confidence:.2f}, "
            f"center=({self.center_x:.0f},{self.center_y:.0f}), "
            f"source={self.source!r})"
        )
