"""
cv/yolo/book_detector.py
BookDetector — YOLO11-powered book detection engine.

Loading strategy (in order)
----------------------------
1. Custom model:   models/yolo11_book_detector.pt  (user-trained on books)
2. Pretrained:     yolo11n.pt  (COCO-pretrained, detects 'book' class 73)
3. Simulation:     Synthetic detections from WorkspaceSimulator (no YOLO needed)

All three modes produce identical List[Detection] output, allowing
downstream pipeline stages (ByteTrack, Workflow Analyzer) to remain
model-agnostic.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np

from cv.yolo.detection import Detection
from utils.logger import get_logger

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

COCO_BOOK_CLASS_ID = 73          # 'book' in COCO 80-class dataset
CUSTOM_MODEL_PATH  = Path("models/yolo11_book_detector.pt")
FALLBACK_MODEL     = "yolo11n.pt"  # downloaded automatically by ultralytics
DEFAULT_CONFIDENCE = 0.40
DEFAULT_IOU        = 0.50
DETECTION_COLOR_BGR = (72, 187, 120)   # green
LABEL_COLOR_BGR     = (255, 255, 255)  # white text


# ═══════════════════════════════════════════════════════════════════════════════
#  Model status enum
# ═══════════════════════════════════════════════════════════════════════════════

class ModelStatus(Enum):
    NOT_LOADED          = "not_loaded"
    LOADED_CUSTOM       = "loaded_custom"
    LOADED_PRETRAINED   = "loaded_pretrained"
    LOAD_FAILED         = "load_failed"
    YOLO_NOT_INSTALLED  = "yolo_not_installed"
    SIMULATION_MODE     = "simulation_mode"


# ═══════════════════════════════════════════════════════════════════════════════
#  BookDetector
# ═══════════════════════════════════════════════════════════════════════════════

class BookDetector:
    """
    YOLO11 book detector with graceful fallback to simulation mode.

    Parameters
    ----------
    custom_model_path : Path to a fine-tuned book detector.
                        Falls back to pretrained yolo11n.pt if not found.
    confidence        : Minimum confidence threshold [0, 1].
    iou_threshold     : NMS IoU threshold.
    device            : Inference device ('cpu' | 'cuda' | '0').
    filter_books_only : If True (pretrained mode), only returns COCO class-73 detections.

    Example
    -------
        detector = BookDetector()
        success = detector.load()
        frame = cv2.imread("desk.jpg")
        detections = detector.detect(frame)
        annotated = detector.draw_detections(frame, detections)
    """

    def __init__(
        self,
        custom_model_path: str | Path = CUSTOM_MODEL_PATH,
        confidence: float            = DEFAULT_CONFIDENCE,
        iou_threshold: float         = DEFAULT_IOU,
        device: str                  = "cpu",
        filter_books_only: bool      = True,
    ) -> None:
        self.custom_model_path   = Path(custom_model_path)
        self.confidence_threshold = confidence
        self.iou_threshold       = iou_threshold
        self.device              = device
        self.filter_books_only   = filter_books_only

        self._model              = None
        self._is_custom          = False
        self._class_names: dict  = {}
        self.status              = ModelStatus.NOT_LOADED
        self.status_message      = "Model not loaded. Call load() to initialise."
        self._loaded_model_name  = ""
        self._total_detections   = 0
        self._load_time_s        = 0.0

    # ── Loading ───────────────────────────────────────────────────────────
    def load(self) -> bool:
        """
        Attempt to load a YOLO11 model.

        Returns True on success (custom or pretrained),
        False if ultralytics is not installed or both models fail.
        """
        # ── Check ultralytics availability ────────────────────────────
        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError:
            self.status = ModelStatus.YOLO_NOT_INSTALLED
            self.status_message = (
                "❌ ultralytics is not installed.\n"
                "Run:  pip install ultralytics\n"
                "Falling back to simulation mode."
            )
            log.warning(self.status_message)
            return False

        t0 = time.time()

        # ── 1. Try custom model ────────────────────────────────────────
        if self.custom_model_path.exists():
            try:
                from ultralytics import YOLO
                self._model       = YOLO(str(self.custom_model_path))
                self._is_custom   = True
                self._class_names = self._model.names or {}
                self.status       = ModelStatus.LOADED_CUSTOM
                self._loaded_model_name = self.custom_model_path.name
                self._load_time_s = time.time() - t0
                self.status_message = (
                    f"✅ Custom model loaded: {self.custom_model_path.name}\n"
                    f"   Classes: {list(self._class_names.values())}"
                )
                log.info(self.status_message)
                return True
            except Exception as exc:
                log.warning(f"Custom model load failed: {exc} — falling back to pretrained.")

        # ── 2. Custom not found — log clearly and load pretrained ─────
        if not self.custom_model_path.exists():
            msg = (
                f"ℹ️  Custom model not found at:  {self.custom_model_path}\n"
                f"   To use your own model, place it at that path.\n"
                f"   Loading pretrained YOLO11n (COCO) instead..."
            )
            log.info(msg)

        try:
            from ultralytics import YOLO
            self._model       = YOLO(FALLBACK_MODEL)   # auto-downloads on first run
            self._is_custom   = False
            self._class_names = self._model.names or {}
            self.status       = ModelStatus.LOADED_PRETRAINED
            self._loaded_model_name = FALLBACK_MODEL
            self._load_time_s = time.time() - t0
            self.status_message = (
                f"✅ Pretrained YOLO11n loaded (COCO).\n"
                f"   Book class: ID={COCO_BOOK_CLASS_ID} 'book'\n"
                f"   Custom model path:  {self.custom_model_path}  (not found)"
            )
            log.info(self.status_message)
            return True

        except Exception as exc:
            self.status = ModelStatus.LOAD_FAILED
            self.status_message = (
                f"❌ Model load failed: {exc}\n"
                f"   Ensure ultralytics is installed and you have internet "
                f"access for the first-time model download."
            )
            log.error(self.status_message)
            return False

    # ── Single-frame detection ────────────────────────────────────────────
    def detect(self, frame: np.ndarray, frame_index: int = 0) -> List[Detection]:
        """
        Run YOLO11 on a single BGR frame.

        Parameters
        ----------
        frame        : BGR numpy array (H × W × 3).
        frame_index  : Frame number in a video/webcam stream.

        Returns
        -------
        List[Detection] — empty list if model not loaded or no books found.
        """
        if not self.is_loaded:
            log.debug("detect() called but model not loaded — returning empty list.")
            return []

        try:
            results = self._model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            log.error(f"YOLO inference failed: {exc}")
            return []

        detections: List[Detection] = []
        ts = datetime.now()

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())

                # For pretrained model: only keep 'book' class (ID 73)
                if (not self._is_custom
                        and self.filter_books_only
                        and class_id != COCO_BOOK_CLASS_ID):
                    continue

                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                conf            = float(box.conf.item())
                class_name      = self._class_names.get(class_id, f"class_{class_id}")

                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    class_id=class_id,
                    class_name="book" if (not self._is_custom and self.filter_books_only) else class_name,
                    source="custom" if self._is_custom else "yolo",
                    frame_index=frame_index,
                    timestamp=ts,
                ))

        self._total_detections += len(detections)
        log.debug(f"Frame {frame_index}: {len(detections)} book(s) detected.")
        return detections

    # ── Image mode ───────────────────────────────────────────────────────
    def detect_image(
        self, image_path: str | Path
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Load an image file and run detection.

        Returns
        -------
        (frame_bgr, detections) — the raw image and detection results.
        Raises FileNotFoundError if the image path does not exist.
        """
        import cv2
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not read image: {path}")
        dets = self.detect(frame)
        log.info(f"Image detection: {path.name} → {len(dets)} book(s).")
        return frame, dets

    # ── Video mode ────────────────────────────────────────────────────────
    def detect_video(
        self,
        video_path: str | Path,
        sample_interval: int = 5,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, List[Detection], int], None, None]:
        """
        Process a video file frame-by-frame.

        Yields
        ------
        (annotated_frame, detections, frame_index) for every sampled frame.

        Parameters
        ----------
        sample_interval : Process every Nth frame (default every 5th).
        max_frames      : Stop after this many processed frames (None = all).
        """
        import cv2
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        frame_idx   = 0
        proc_count  = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_interval == 0:
                    dets      = self.detect(frame, frame_idx)
                    annotated = self.draw_detections(frame, dets)
                    yield annotated, dets, frame_idx
                    proc_count += 1
                    if max_frames is not None and proc_count >= max_frames:
                        break

                frame_idx += 1
        finally:
            cap.release()

        log.info(f"Video processed: {proc_count} frames from {path.name}.")

    # ── Webcam mode ───────────────────────────────────────────────────────
    def detect_webcam(
        self,
        camera_index: int = 0,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, List[Detection], int], None, None]:
        """
        Capture from webcam and yield detected frames.

        Falls back to a helpful error if the camera is not available.
        Use detect_webcam_available() first to check.

        Yields
        ------
        (annotated_frame, detections, frame_index)
        """
        import cv2
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Camera index {camera_index} not available. "
                "Use demo/simulation mode or upload an image/video instead."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                dets      = self.detect(frame, frame_idx)
                annotated = self.draw_detections(frame, dets)
                yield annotated, dets, frame_idx
                frame_idx += 1
                if max_frames is not None and frame_idx >= max_frames:
                    break
        finally:
            cap.release()

    @staticmethod
    def detect_webcam_available(camera_index: int = 0) -> bool:
        """Return True if the specified camera can be opened."""
        try:
            import cv2
            cap = cv2.VideoCapture(camera_index)
            ok  = cap.isOpened()
            cap.release()
            return ok
        except Exception:
            return False

    # ── Draw detections ───────────────────────────────────────────────────
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_confidence: bool = True,
        show_center: bool = True,
        show_id: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes, labels, and centre points on a frame copy.

        Parameters
        ----------
        frame           : BGR numpy array.
        detections      : List of Detection objects.
        show_confidence : Include confidence score in label.
        show_center     : Draw a small dot at the detection centre.
        show_id         : Show detection index number.

        Returns
        -------
        Annotated BGR numpy array (copy of input — original unchanged).
        """
        import cv2
        output = frame.copy()

        for i, det in enumerate(detections):
            color  = det.color_bgr
            x1i, y1i, x2i, y2i = det.bbox_int
            cxi, cyi            = det.center_int

            # ── Semi-transparent fill ──────────────────────────────────
            overlay = output.copy()
            cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), color, -1)
            output = cv2.addWeighted(overlay, 0.18, output, 0.82, 0)

            # ── Bounding box border ────────────────────────────────────
            cv2.rectangle(output, (x1i, y1i), (x2i, y2i), color, 2)

            # ── Label background + text ────────────────────────────────
            label_parts = [det.class_name]
            if show_id:
                label_parts.insert(0, f"#{i+1}")
            if show_confidence:
                label_parts.append(f"{det.confidence:.2f}")
            label = " ".join(label_parts)

            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            thickness  = 1

            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            lx = x1i
            ly = max(y1i - 6, th + 4)

            # Label background
            cv2.rectangle(
                output,
                (lx, ly - th - baseline - 4),
                (lx + tw + 4, ly + baseline - 2),
                color, -1,
            )
            # Label text
            cv2.putText(
                output, label,
                (lx + 2, ly - baseline),
                font, font_scale, LABEL_COLOR_BGR, thickness, cv2.LINE_AA,
            )

            # ── Centre point ───────────────────────────────────────────
            if show_center:
                cv2.circle(output, (cxi, cyi), 4,  (255, 255, 255), -1)
                cv2.circle(output, (cxi, cyi), 5,  color, 2)

            # ── Source tag ─────────────────────────────────────────────
            if det.source == "simulation":
                cv2.putText(
                    output, "SIM",
                    (x2i - 32, y2i - 4),
                    font, 0.35, (255, 200, 0), 1, cv2.LINE_AA,
                )

        # ── Frame info overlay ─────────────────────────────────────────
        info = f"Detections: {len(detections)}  |  {self._loaded_model_name}"
        cv2.putText(
            output, info, (8, frame.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
        )

        return output

    # ── Save results ──────────────────────────────────────────────────────
    def save_results(
        self,
        detections:  List[Detection],
        output_path: str | Path,
        frame:       Optional[np.ndarray] = None,
        metadata:    Optional[dict]       = None,
    ) -> Path:
        """
        Save detection results to a JSON file (and optionally the annotated image).

        Parameters
        ----------
        detections  : List of Detection objects to save.
        output_path : Path to the output JSON file.
        frame       : If provided, saves annotated image alongside the JSON.
        metadata    : Extra key-value pairs to include in the JSON.

        Returns
        -------
        Path of the saved JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "timestamp":           datetime.now().isoformat(),
            "model_type":          self.status.value,
            "model_name":          self._loaded_model_name,
            "confidence_threshold":self.confidence_threshold,
            "total_detections":    len(detections),
            "detections":          [d.to_dict() for d in detections],
        }
        if metadata:
            payload["metadata"] = metadata

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        log.info(f"Detection results saved: {output_path}")

        # Optionally save annotated image
        if frame is not None:
            import cv2
            img_path = output_path.with_suffix(".jpg")
            annotated = self.draw_detections(frame, detections)
            cv2.imwrite(str(img_path), annotated)
            log.info(f"Annotated image saved:  {img_path}")

        return output_path

    # ── Simulation mode detection ─────────────────────────────────────────
    def simulate_detections(
        self, workspace, image_shape: Tuple[int, int, int]
    ) -> List[Detection]:
        """
        Generate simulated detections from WorkspaceSimulator state.
        Used when YOLO is not loaded or as a demo mode.
        """
        from cv.yolo.synthetic_generator import SyntheticDeskGenerator
        gen = SyntheticDeskGenerator(image_shape[1], image_shape[0])
        return gen.generate_detections(workspace, image_shape)

    # ── Properties ───────────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_type(self) -> str:
        return self.status.value

    @property
    def loaded_model_name(self) -> str:
        return self._loaded_model_name

    @property
    def total_detections(self) -> int:
        return self._total_detections

    @property
    def load_time_seconds(self) -> float:
        return self._load_time_s

    def __repr__(self) -> str:
        return (
            f"BookDetector("
            f"status={self.status.value!r}, "
            f"model={self._loaded_model_name!r}, "
            f"conf={self.confidence_threshold})"
        )
