"""
cv/tracking/tracker.py
ByteTrack multi-object tracker wrapper.

Status: STUB — Implementation in Phase 2.
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TrackedObject:
    """A single object being tracked across frames."""
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    confidence: float
    state: str = "active"          # active | lost | removed

    @property
    def bbox(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class ByteTracker:
    """
    Wraps ByteTrack for persistent object tracking.
    Phase 2 implementation will use supervision.ByteTrack.
    """

    STATUS = "NOT_STARTED"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._tracker = None

    def initialize(self) -> None:
        """Initialize the ByteTrack tracker."""
        # TODO Phase 2: self._tracker = sv.ByteTrack(...)
        raise NotImplementedError("ByteTracker: Phase 2 not yet implemented.")

    def update(self, detections) -> List[TrackedObject]:
        """Update tracker with new detections. Returns tracked objects."""
        raise NotImplementedError("ByteTracker: Phase 2 not yet implemented.")
