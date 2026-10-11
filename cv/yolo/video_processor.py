"""
cv/yolo/video_processor.py
VideoProcessor — batch frame processing wrapper around BookDetector.

Provides:
- Synchronous batch processing (for Streamlit)
- Frame-level statistics aggregation
- Progress tracking via callbacks
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from cv.yolo.detection import Detection
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ProcessingStats:
    """Aggregate statistics from a processing run."""
    total_frames:     int         = 0
    processed_frames: int         = 0
    total_detections: int         = 0
    avg_detections:   float       = 0.0
    avg_confidence:   float       = 0.0
    avg_fps:          float       = 0.0
    duration_s:       float       = 0.0
    frames_with_books:int         = 0
    all_detections:   List[Detection] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        """Fraction of processed frames that contained at least one book."""
        return (
            self.frames_with_books / self.processed_frames
            if self.processed_frames > 0 else 0.0
        )

    def to_dict(self) -> dict:
        return {
            "total_frames":      self.total_frames,
            "processed_frames":  self.processed_frames,
            "total_detections":  self.total_detections,
            "avg_detections":    round(self.avg_detections, 2),
            "avg_confidence":    round(self.avg_confidence, 4),
            "avg_fps":           round(self.avg_fps, 1),
            "duration_s":        round(self.duration_s, 2),
            "frames_with_books": self.frames_with_books,
            "detection_rate":    round(self.detection_rate, 3),
        }


class VideoProcessor:
    """
    Wraps BookDetector to provide batch video/webcam processing.

    Parameters
    ----------
    detector      : A BookDetector instance (must be loaded before processing).
    sample_every  : Process every Nth frame.
    max_output    : Max number of result frames to keep in memory.

    Example
    -------
        processor = VideoProcessor(detector, sample_every=10)
        results, stats = processor.process_video("desk_video.mp4")
        for frame, dets, idx in results:
            ...
    """

    def __init__(
        self,
        detector,
        sample_every:  int = 5,
        max_output:    int = 50,
    ) -> None:
        self.detector    = detector
        self.sample_every = sample_every
        self.max_output  = max_output

    # ── Video file ────────────────────────────────────────────────────────
    def process_video(
        self,
        video_path: str | Path,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[Tuple[np.ndarray, List[Detection], int]], ProcessingStats]:
        """
        Process a video file and return sampled results.

        Parameters
        ----------
        video_path   : Path to video file.
        progress_cb  : Optional callback(processed_count, estimated_total).

        Returns
        -------
        (results, stats) where results is a list of (frame, detections, frame_idx).
        """
        import cv2

        path = Path(video_path)
        cap  = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_input    = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()

        stats   = ProcessingStats(total_frames=total_frames)
        results = []
        t0      = time.time()

        for frame, dets, idx in self.detector.detect_video(
            path,
            sample_interval=self.sample_every,
            max_frames=self.max_output,
        ):
            stats.processed_frames += 1
            stats.total_detections += len(dets)
            stats.all_detections.extend(dets)
            if dets:
                stats.frames_with_books += 1

            results.append((frame, dets, idx))
            if progress_cb:
                progress_cb(stats.processed_frames, total_frames // self.sample_every)

        stats.duration_s    = time.time() - t0
        stats.avg_detections = stats.total_detections / max(stats.processed_frames, 1)
        stats.avg_fps        = stats.processed_frames / max(stats.duration_s, 0.001)

        if stats.all_detections:
            stats.avg_confidence = sum(d.confidence for d in stats.all_detections) / len(stats.all_detections)

        log.info(f"Video processed: {stats.processed_frames} frames, "
                 f"{stats.total_detections} detections, {stats.avg_fps:.1f} fps.")
        return results, stats

    # ── Synthetic batch processing ─────────────────────────────────────────
    def process_synthetic_frames(
        self, workspace, n_frames: int = 6
    ) -> Tuple[List[Tuple[np.ndarray, List[Detection], int]], ProcessingStats]:
        """
        Generate and process N synthetic desk frames.
        Used in demo/simulation mode when no real video is available.
        """
        from cv.yolo.synthetic_generator import SyntheticDeskGenerator
        gen   = SyntheticDeskGenerator()
        stats = ProcessingStats(total_frames=n_frames)
        results = []
        t0 = time.time()

        for i in range(n_frames):
            frame = gen.generate_desk_image(workspace)
            if self.detector.is_loaded:
                dets = self.detector.detect(frame, i)
            else:
                dets = gen.generate_detections(workspace, frame.shape)

            annotated = self.detector.draw_detections(frame, dets)
            results.append((annotated, dets, i))

            stats.processed_frames += 1
            stats.total_detections += len(dets)
            stats.all_detections.extend(dets)
            if dets:
                stats.frames_with_books += 1

        stats.duration_s    = time.time() - t0
        stats.avg_detections = stats.total_detections / max(stats.processed_frames, 1)
        stats.avg_fps        = stats.processed_frames / max(stats.duration_s, 0.001)
        if stats.all_detections:
            stats.avg_confidence = sum(d.confidence for d in stats.all_detections) / len(stats.all_detections)

        return results, stats
