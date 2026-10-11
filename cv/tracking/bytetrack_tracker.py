"""
cv/tracking/bytetrack_tracker.py
=================================
Phase 4 — ByteTrack integration with YOLO11.

Pipeline
--------
  Video Frame
      ↓
  YOLO11 Detection  (BookDetector → List[Detection])
      ↓
  ByteTrackTracker.update()
      ↓
  List[BookTrack]  — persistent track_id across frames
      ↓
  Book Position (center_x, center_y) + Timestamp
      ↓
  SQLite  (BookTrackRecord, via TrackingRepository)

Visual output
-------------
  draw_tracks()  — renders coloured boxes + "Track ID N" labels on a frame copy.

Usage
-----
    detector = BookDetector(); detector.load()
    tracker  = ByteTrackTracker()
    tracker.initialize()

    for frame in video_frames:
        detections = detector.detect(frame)
        tracks     = tracker.update(detections, frame)
        annotated  = tracker.draw_tracks(frame, tracks)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  BookTrack  — a single persisted book track
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BookTrack:
    """
    One tracked book across video frames.

    Attributes
    ----------
    track_id   : Stable integer assigned by ByteTrack (persists across frames).
    bbox       : (x1, y1, x2, y2) bounding box in pixels.
    center_x   : Horizontal centre of the bounding box.
    center_y   : Vertical centre of the bounding box.
    confidence : Detection confidence from YOLO11 [0, 1].
    timestamp  : Wall-clock time when this track was last updated.
    book_title : TrOCR-extracted title, populated in a later phase (default "Unknown").
    frame_index: Frame number within the video/stream.
    """

    track_id:   int
    bbox:       Tuple[float, float, float, float]   # (x1, y1, x2, y2)
    center_x:   float
    center_y:   float
    confidence: float
    timestamp:  datetime
    book_title: str  = "Unknown"
    frame_index: int = 0

    # ── Convenience ───────────────────────────────────────────────────────
    @property
    def bbox_int(self) -> Tuple[int, int, int, int]:
        return tuple(int(v) for v in self.bbox)  # type: ignore[return-value]

    @property
    def center_int(self) -> Tuple[int, int]:
        return (int(self.center_x), int(self.center_y))

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for JSON / DB storage)."""
        return {
            "track_id":   self.track_id,
            "book_title": self.book_title,
            "bbox":       list(self.bbox),
            "center_x":   round(self.center_x, 2),
            "center_y":   round(self.center_y, 2),
            "confidence": round(self.confidence, 4),
            "timestamp":  self.timestamp.isoformat(),
            "frame_index": self.frame_index,
        }

    def __repr__(self) -> str:
        return (
            f"BookTrack(id={self.track_id}, "
            f"book={self.book_title!r}, "
            f"pos=({self.center_x:.0f},{self.center_y:.0f}), "
            f"conf={self.confidence:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Colour palette — one stable BGR colour per track_id
# ═══════════════════════════════════════════════════════════════════════════════

_PALETTE_BGR: List[Tuple[int, int, int]] = [
    (72,  187, 120),  # green
    (59,  130, 246),  # blue
    (245, 101,  101), # red
    (251, 191,  36),  # yellow
    (167,  139, 250), # purple
    (34,  211, 238),  # cyan
    (249, 115,  22),  # orange
    (236,  72, 153),  # pink
    (16,  185, 129),  # teal
    (239, 246, 255),  # white-ish
]


def _track_color(track_id: int) -> Tuple[int, int, int]:
    """Return a deterministic BGR colour for a given track_id."""
    return _PALETTE_BGR[track_id % len(_PALETTE_BGR)]


# ═══════════════════════════════════════════════════════════════════════════════
#  ByteTrackTracker
# ═══════════════════════════════════════════════════════════════════════════════

class ByteTrackTracker:
    """
    Wraps ``supervision.ByteTrack`` to provide persistent book track IDs.

    Parameters
    ----------
    track_activation_threshold : Min confidence to activate a new track (default 0.25).
    lost_track_buffer          : Frames to keep a lost track alive (default 30).
    minimum_matching_threshold : IoU threshold for track-detection matching (default 0.8).
    frame_rate                 : Expected frame rate, used to calibrate lost-track buffer.

    Example
    -------
        tracker = ByteTrackTracker()
        tracker.initialize()

        detections = detector.detect(frame)        # List[Detection]
        tracks     = tracker.update(detections)    # List[BookTrack]
        annotated  = tracker.draw_tracks(frame, tracks)
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer:          int   = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate:                 int   = 30,
    ) -> None:
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer          = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.frame_rate                 = frame_rate

        self._tracker = None          # supervision.ByteTrack instance
        self._frame_index: int = 0    # internal frame counter

        # ── Track memory: track_id → last BookTrack seen ──────────────
        self._active_tracks: Dict[int, BookTrack] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def initialize(self) -> None:
        """
        Instantiate the supervision ByteTrack object.
        Safe to call multiple times (re-initialises state).
        """
        try:
            import supervision as sv  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "supervision is not installed. "
                "Run:  pip install supervision>=0.22.0"
            ) from exc

        import supervision as sv

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = sv.ByteTrack(
                track_activation_threshold=self.track_activation_threshold,
                lost_track_buffer=self.lost_track_buffer,
                minimum_matching_threshold=self.minimum_matching_threshold,
                frame_rate=self.frame_rate,
            )

        self._frame_index   = 0
        self._active_tracks = {}
        log.info(
            "ByteTrackTracker initialised "
            f"(thresh={self.track_activation_threshold}, "
            f"buffer={self.lost_track_buffer}, "
            f"match={self.minimum_matching_threshold})"
        )

    def reset(self) -> None:
        """Reset tracker state (clear all tracks)."""
        if self._tracker is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                self._tracker.reset()
        self._frame_index   = 0
        self._active_tracks = {}
        log.debug("ByteTrackTracker reset.")

    # ── Core update ───────────────────────────────────────────────────────
    def update(
        self,
        detections: list,            # List[cv.yolo.detection.Detection]
        frame_index: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> List[BookTrack]:
        """
        Feed YOLO11 detections into ByteTrack and return live book tracks.

        Parameters
        ----------
        detections  : List[Detection] from BookDetector.detect().
        frame_index : Override the internal frame counter if supplied.
        timestamp   : Override the wall-clock timestamp if supplied.

        Returns
        -------
        List[BookTrack] — one entry per tracked book with a stable track_id.
        """
        if self._tracker is None:
            raise RuntimeError(
                "ByteTrackTracker not initialised. Call initialize() first."
            )

        if frame_index is None:
            frame_index = self._frame_index
        self._frame_index += 1

        ts = timestamp or datetime.now()

        # ── Build supervision.Detections ──────────────────────────────
        import supervision as sv

        if not detections:
            sv_dets = sv.Detections.empty()
        else:
            xyxy       = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=float)
            confidence = np.array([d.confidence                for d in detections], dtype=float)
            class_id   = np.array([d.class_id                  for d in detections], dtype=int)
            sv_dets    = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

        # ── Run ByteTrack ─────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            tracked = self._tracker.update_with_detections(sv_dets)

        # ── Convert to BookTrack list ─────────────────────────────────
        book_tracks: List[BookTrack] = []

        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            return book_tracks

        for i, tid in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = tracked.xyxy[i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

            # Recover book_title from previously seen track (persisted across frames)
            prev = self._active_tracks.get(int(tid))
            title = prev.book_title if prev else "Unknown"

            bt = BookTrack(
                track_id=int(tid),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                center_x=float(cx),
                center_y=float(cy),
                confidence=conf,
                timestamp=ts,
                book_title=title,
                frame_index=frame_index,
            )
            book_tracks.append(bt)
            self._active_tracks[int(tid)] = bt

        log.debug(
            f"Frame {frame_index}: {len(detections)} det(s) → "
            f"{len(book_tracks)} track(s) {[t.track_id for t in book_tracks]}"
        )
        return book_tracks

    # ── Convenience: set title for a given track ──────────────────────────
    def set_book_title(self, track_id: int, title: str) -> None:
        """Assign a TrOCR-extracted title to a known track_id."""
        if track_id in self._active_tracks:
            self._active_tracks[track_id].book_title = title

    # ── Visualisation ─────────────────────────────────────────────────────
    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[BookTrack],
        show_confidence: bool = True,
        show_center:     bool = True,
    ) -> np.ndarray:
        """
        Render track IDs, bounding boxes, and book titles on a frame copy.

        Returns an annotated BGR numpy array (original frame is unchanged).

        Example overlay
        ---------------
          ┌─────────────────────────────┐
          │ Track ID 1 — Unknown  0.90  │
          └─────────────────────────────┘
          Position: (250, 180)
        """
        import cv2

        output = frame.copy()
        font        = cv2.FONT_HERSHEY_SIMPLEX
        font_scale  = 0.55
        thickness   = 1

        for bt in tracks:
            color = _track_color(bt.track_id)
            x1i, y1i, x2i, y2i = bt.bbox_int
            cxi,  cyi           = bt.center_int

            # ── Semi-transparent fill ──────────────────────────────────
            overlay = output.copy()
            cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), color, -1)
            output  = cv2.addWeighted(overlay, 0.15, output, 0.85, 0)

            # ── Bounding box ───────────────────────────────────────────
            cv2.rectangle(output, (x1i, y1i), (x2i, y2i), color, 2)

            # ── Label ─────────────────────────────────────────────────
            parts = [f"Track ID {bt.track_id}", f"— {bt.book_title}"]
            if show_confidence:
                parts.append(f"{bt.confidence:.2f}")
            label = " ".join(parts)

            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            lx = x1i
            ly = max(y1i - 6, th + 6)

            cv2.rectangle(
                output,
                (lx, ly - th - baseline - 4),
                (lx + tw + 6, ly + baseline),
                color, -1,
            )
            cv2.putText(
                output, label,
                (lx + 3, ly - baseline),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )

            # ── Centre dot ────────────────────────────────────────────
            if show_center:
                cv2.circle(output, (cxi, cyi), 5, (255, 255, 255), -1)
                cv2.circle(output, (cxi, cyi), 6, color,           2)

            # ── Position label (bottom-right of box) ───────────────────
            pos_label = f"({cxi}, {cyi})"
            cv2.putText(
                output, pos_label,
                (x1i + 4, y2i - 6),
                font, 0.38, (220, 220, 220), 1, cv2.LINE_AA,
            )

        # ── Frame info ─────────────────────────────────────────────────
        info = f"Tracks: {len(tracks)}  |  ByteTrack"
        cv2.putText(
            output, info, (8, frame.shape[0] - 8),
            font, 0.45, (160, 160, 160), 1, cv2.LINE_AA,
        )

        return output

    # ── Properties ────────────────────────────────────────────────────────
    @property
    def is_initialized(self) -> bool:
        return self._tracker is not None

    @property
    def active_tracks(self) -> Dict[int, BookTrack]:
        """Read-only view of the current track_id → BookTrack map."""
        return dict(self._active_tracks)

    @property
    def frame_count(self) -> int:
        return self._frame_index

    def __repr__(self) -> str:
        status = "ready" if self.is_initialized else "not initialised"
        return (
            f"ByteTrackTracker("
            f"status={status!r}, "
            f"active={len(self._active_tracks)}, "
            f"frames={self._frame_index})"
        )
