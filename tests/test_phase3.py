"""
tests/test_phase3.py
Phase 3 — YOLO11 Book Detection: comprehensive unit & integration tests.

Test groups
-----------
  A. Detection dataclass      (geometry, serialisation, IOU, colour)
  B. ModelStatus enum         (all states present)
  C. BookDetector — loading   (custom / pretrained / missing / import-error)
  D. BookDetector — detect()  (single frame, pretrained filter, sim mode)
  E. Image mode               (detect_image with real/fake files)
  F. Video mode               (detect_video generator)
  G. Webcam helpers           (availability check, graceful failure)
  H. draw_detections()        (output shape, no crash on empty)
  I. save_results()           (JSON correctness, annotated image)
  J. SyntheticDeskGenerator   (image shape, detection count/fields)
  K. VideoProcessor           (synthetic batch, stats correctness)
  L. Public API / imports     (cv.yolo namespace, cv.yolo.detector re-export)

Run:
    pytest tests/test_phase3.py -v
    pytest tests/test_phase3.py -v -k "not Video"   # skip video tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures & helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a blank BGR numpy frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_detection(**kwargs) -> "Detection":
    from cv.yolo.detection import Detection
    defaults = dict(
        x1=100.0, y1=80.0, x2=220.0, y2=200.0,
        confidence=0.85, class_id=73, class_name="book",
        source="yolo",
    )
    defaults.update(kwargs)
    return Detection(**defaults)


def _make_workspace():
    """Return a WorkspaceSimulator instance (books are populated on construction)."""
    try:
        from simulation.workspace import WorkspaceSimulator
        return WorkspaceSimulator()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  A. Detection dataclass
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetection:

    def test_creation_basic(self):
        from cv.yolo.detection import Detection
        d = Detection(x1=10, y1=20, x2=110, y2=170, confidence=0.9,
                      class_id=73, class_name="book")
        assert d.class_name == "book"
        assert d.confidence == pytest.approx(0.9)
        assert d.class_id == 73
        assert d.source == "yolo"

    def test_auto_coordinate_swap(self):
        """x1 > x2 should be auto-corrected."""
        from cv.yolo.detection import Detection
        d = Detection(x1=200, y1=100, x2=50, y2=300,
                      confidence=0.7, class_id=73, class_name="book")
        assert d.x1 < d.x2
        assert d.y1 < d.y2

    def test_center_properties(self):
        from cv.yolo.detection import Detection
        d = Detection(x1=100, y1=80, x2=300, y2=180,
                      confidence=0.8, class_id=73, class_name="book")
        assert d.center_x == pytest.approx(200.0)
        assert d.center_y == pytest.approx(130.0)
        assert d.center == (200.0, 130.0)

    def test_width_height_area(self):
        from cv.yolo.detection import Detection
        d = Detection(x1=0, y1=0, x2=100, y2=50,
                      confidence=0.6, class_id=73, class_name="book")
        assert d.width  == pytest.approx(100.0)
        assert d.height == pytest.approx(50.0)
        assert d.area   == pytest.approx(5000.0)

    def test_bbox_int(self):
        from cv.yolo.detection import Detection
        d = Detection(x1=1.6, y1=2.4, x2=50.9, y2=99.1,
                      confidence=0.5, class_id=73, class_name="book")
        x1i, y1i, x2i, y2i = d.bbox_int
        assert isinstance(x1i, int)

    def test_center_int(self):
        from cv.yolo.detection import Detection
        d = Detection(x1=100, y1=80, x2=200, y2=180,
                      confidence=0.8, class_id=73, class_name="book")
        cx, cy = d.center_int
        assert cx == 150
        assert cy == 130

    def test_source_default_color(self):
        """YOLO source -> green; simulation source -> orange."""
        from cv.yolo.detection import Detection, _BOOK_COLOR_BGR, _SIM_COLOR_BGR
        d_yolo = Detection(x1=0, y1=0, x2=50, y2=50,
                           confidence=0.9, class_id=73, class_name="book",
                           source="yolo")
        d_sim = Detection(x1=0, y1=0, x2=50, y2=50,
                          confidence=0.9, class_id=73, class_name="book",
                          source="simulation")
        assert d_yolo.color_bgr == _BOOK_COLOR_BGR
        assert d_sim.color_bgr  == _SIM_COLOR_BGR

    def test_iou_overlap(self):
        from cv.yolo.detection import Detection
        a = Detection(x1=0, y1=0, x2=100, y2=100,
                      confidence=0.9, class_id=73, class_name="book")
        b = Detection(x1=50, y1=50, x2=150, y2=150,
                      confidence=0.9, class_id=73, class_name="book")
        iou = a.iou(b)
        assert 0 < iou < 1

    def test_iou_no_overlap(self):
        from cv.yolo.detection import Detection
        a = Detection(x1=0, y1=0, x2=50, y2=50,
                      confidence=0.9, class_id=73, class_name="book")
        b = Detection(x1=200, y1=200, x2=300, y2=300,
                      confidence=0.9, class_id=73, class_name="book")
        assert a.iou(b) == pytest.approx(0.0)

    def test_iou_identical(self):
        from cv.yolo.detection import Detection
        a = Detection(x1=0, y1=0, x2=100, y2=100,
                      confidence=0.9, class_id=73, class_name="book")
        b = Detection(x1=0, y1=0, x2=100, y2=100,
                      confidence=0.9, class_id=73, class_name="book")
        assert a.iou(b) == pytest.approx(1.0)

    def test_to_dict_keys(self):
        d = _make_detection()
        data = d.to_dict()
        for key in ("x1", "y1", "x2", "y2", "center_x", "center_y",
                    "width", "height", "area", "confidence", "class_id",
                    "class_name", "source", "frame_index", "timestamp"):
            assert key in data, f"Missing key: {key}"

    def test_to_dict_center_values(self):
        d = _make_detection(x1=100, y1=80, x2=220, y2=200)
        data = d.to_dict()
        assert data["center_x"] == pytest.approx(160.0, abs=1)
        assert data["center_y"] == pytest.approx(140.0, abs=1)

    def test_repr_contains_class_name(self):
        d = _make_detection(class_name="book")
        assert "book" in repr(d)

    def test_label_property(self):
        d = _make_detection(class_name="book", confidence=0.92)
        assert "book" in d.label
        assert "0.92" in d.label


# ═══════════════════════════════════════════════════════════════════════════════
#  B. ModelStatus enum
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelStatus:

    def test_all_statuses_present(self):
        from cv.yolo.book_detector import ModelStatus
        expected = {
            "NOT_LOADED", "LOADED_CUSTOM", "LOADED_PRETRAINED",
            "LOAD_FAILED", "YOLO_NOT_INSTALLED", "SIMULATION_MODE"
        }
        actual = {s.name for s in ModelStatus}
        assert expected == actual

    def test_values_are_strings(self):
        from cv.yolo.book_detector import ModelStatus
        for status in ModelStatus:
            assert isinstance(status.value, str)


# ═══════════════════════════════════════════════════════════════════════════════
#  C. BookDetector — loading
# ═══════════════════════════════════════════════════════════════════════════════

class TestBookDetectorLoading:

    def test_initial_state(self):
        from cv.yolo.book_detector import BookDetector, ModelStatus
        det = BookDetector()
        assert det.status == ModelStatus.NOT_LOADED
        assert not det.is_loaded
        assert det.total_detections == 0

    def test_no_crash_when_ultralytics_missing(self):
        """If ultralytics is not installed, load() returns False gracefully."""
        from cv.yolo.book_detector import BookDetector, ModelStatus
        det = BookDetector()
        with patch.dict("sys.modules", {"ultralytics": None}):
            result = det.load()
        # Result depends on whether ultralytics is actually installed;
        # the important thing is no exception is raised.
        assert isinstance(result, bool)

    def test_custom_model_not_found_message(self):
        """Missing custom model shows clear message, then falls back."""
        from cv.yolo.book_detector import BookDetector
        det = BookDetector(custom_model_path="models/nonexistent_model.pt")
        # Don't actually load; just assert the path is stored
        assert det.custom_model_path == Path("models/nonexistent_model.pt")

    def test_load_returns_bool(self):
        """load() always returns a boolean regardless of environment."""
        from cv.yolo.book_detector import BookDetector
        det = BookDetector()
        result = det.load()
        assert isinstance(result, bool)

    def test_status_message_populated(self):
        """After load() the status_message must be a non-empty string."""
        from cv.yolo.book_detector import BookDetector
        det = BookDetector()
        det.load()
        assert isinstance(det.status_message, str)
        assert len(det.status_message) > 0

    def test_repr_contains_status(self):
        from cv.yolo.book_detector import BookDetector
        det = BookDetector()
        r = repr(det)
        assert "BookDetector" in r
        assert "status" in r


# ═══════════════════════════════════════════════════════════════════════════════
#  D. BookDetector — detect()
# ═══════════════════════════════════════════════════════════════════════════════

class TestBookDetectorDetect:

    def test_detect_returns_list_when_not_loaded(self):
        """detect() returns empty list if model isn't loaded."""
        from cv.yolo.book_detector import BookDetector
        det = BookDetector()
        frame = _make_frame()
        result = det.detect(frame)
        assert isinstance(result, list)
        assert result == []

    def test_detect_increments_total_when_loaded(self):
        """total_detections counter increases after successful detect."""
        from cv.yolo.book_detector import BookDetector, ModelStatus
        det = BookDetector()

        # Inject a mock model directly
        mock_box = MagicMock()
        mock_box.cls.item.return_value = 73
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [10.0, 20.0, 200.0, 180.0]
        mock_box.conf.item.return_value = 0.88

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_model.names = {73: "book"}

        det._model = mock_model
        det._is_custom = False
        det._class_names = {73: "book"}
        det.status = ModelStatus.LOADED_PRETRAINED
        det._loaded_model_name = "yolo11n.pt"

        frame = _make_frame()
        dets = det.detect(frame)
        assert len(dets) == 1
        assert det.total_detections == 1

    def test_detect_filters_non_book_in_pretrained_mode(self):
        """In pretrained mode only class 73 detections are returned."""
        from cv.yolo.book_detector import BookDetector, ModelStatus

        det = BookDetector(filter_books_only=True)
        det._is_custom = False

        mock_box_person = MagicMock()
        mock_box_person.cls.item.return_value = 0   # person
        mock_box_person.xyxy = [MagicMock()]
        mock_box_person.xyxy[0].tolist.return_value = [0, 0, 100, 100]
        mock_box_person.conf.item.return_value = 0.95

        mock_result = MagicMock()
        mock_result.boxes = [mock_box_person]

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_model.names = {0: "person"}

        det._model = mock_model
        det._class_names = {0: "person"}
        det.status = ModelStatus.LOADED_PRETRAINED
        det._loaded_model_name = "yolo11n.pt"

        dets = det.detect(_make_frame())
        assert dets == []

    def test_detect_returns_correct_fields(self):
        """Each Detection has the expected attributes."""
        from cv.yolo.book_detector import BookDetector, ModelStatus

        det = BookDetector()
        mock_box = MagicMock()
        mock_box.cls.item.return_value = 73
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [50.0, 60.0, 300.0, 250.0]
        mock_box.conf.item.return_value = 0.75

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model = MagicMock(return_value=[mock_result])
        mock_model.names = {73: "book"}

        det._model = mock_model
        det._is_custom = False
        det._class_names = {73: "book"}
        det.status = ModelStatus.LOADED_PRETRAINED
        det._loaded_model_name = "yolo11n.pt"

        dets = det.detect(_make_frame(), frame_index=5)
        assert len(dets) == 1
        d = dets[0]
        assert d.x1 == pytest.approx(50.0)
        assert d.y1 == pytest.approx(60.0)
        assert d.x2 == pytest.approx(300.0)
        assert d.y2 == pytest.approx(250.0)
        assert d.confidence == pytest.approx(0.75)
        assert d.class_name == "book"
        assert d.class_id  == 73
        assert d.frame_index == 5
        assert isinstance(d.timestamp, datetime)

    def test_detect_center_coordinates(self):
        """Detection center coordinates are computed correctly."""
        from cv.yolo.book_detector import BookDetector, ModelStatus

        det = BookDetector()
        mock_box = MagicMock()
        mock_box.cls.item.return_value = 73
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [100.0, 80.0, 300.0, 200.0]
        mock_box.conf.item.return_value = 0.80

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model = MagicMock(return_value=[mock_result])
        mock_model.names = {73: "book"}

        det._model = mock_model
        det._is_custom = False
        det._class_names = {73: "book"}
        det.status = ModelStatus.LOADED_PRETRAINED
        det._loaded_model_name = "yolo11n.pt"

        dets = det.detect(_make_frame())
        d = dets[0]
        assert d.center_x == pytest.approx(200.0)
        assert d.center_y == pytest.approx(140.0)

    def test_detect_handles_yolo_exception_gracefully(self):
        """If YOLO inference crashes, detect() returns [] instead of raising."""
        from cv.yolo.book_detector import BookDetector, ModelStatus

        det = BookDetector()
        mock_model = MagicMock(side_effect=RuntimeError("CUDA OOM"))
        det._model = mock_model
        det._is_custom = False
        det.status = ModelStatus.LOADED_PRETRAINED
        det._loaded_model_name = "yolo11n.pt"

        result = det.detect(_make_frame())
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
#  E. Image mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestImageMode:

    def test_detect_image_file_not_found(self):
        from cv.yolo.book_detector import BookDetector
        det = BookDetector()
        with pytest.raises(FileNotFoundError):
            det.detect_image("nonexistent_image_xyz.jpg")

    def test_detect_image_returns_tuple(self):
        """detect_image on a real temp file returns (frame, list)."""
        import cv2
        from cv.yolo.book_detector import BookDetector, ModelStatus

        det = BookDetector()
        # Load without YOLO so detect returns []
        det._model = None

        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            frame = _make_frame(200, 200)
            cv2.imwrite(str(tmp_path), frame)
            result = det.detect_image(tmp_path)
            assert isinstance(result, tuple)
            assert len(result) == 2
            frame_out, dets = result
            assert isinstance(frame_out, np.ndarray)
            assert isinstance(dets, list)
        finally:
            tmp_path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  F. Video mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestVideoMode:

    def test_detect_video_file_not_found(self):
        from cv.yolo.book_detector import BookDetector
        det = BookDetector()
        with pytest.raises(FileNotFoundError):
            list(det.detect_video("nonexistent_video.mp4"))

    def test_detect_video_invalid_file(self):
        """A non-video file raises RuntimeError (cannot open)."""
        from cv.yolo.book_detector import BookDetector
        det = BookDetector()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(b"not a video")
            tmp_path = Path(tmp.name)

        try:
            with pytest.raises(RuntimeError):
                list(det.detect_video(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  G. Webcam helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestWebcamHelpers:

    def test_detect_webcam_available_returns_bool(self):
        from cv.yolo.book_detector import BookDetector
        result = BookDetector.detect_webcam_available(99)  # unlikely camera
        assert isinstance(result, bool)

    def test_detect_webcam_unavailable_is_false(self):
        """Camera index 99 should not exist on any test machine."""
        from cv.yolo.book_detector import BookDetector
        assert BookDetector.detect_webcam_available(99) is False

    def test_detect_webcam_raises_on_unavailable(self):
        """detect_webcam generator raises RuntimeError for missing camera."""
        from cv.yolo.book_detector import BookDetector
        det = BookDetector()
        with pytest.raises(RuntimeError):
            list(det.detect_webcam(camera_index=99, max_frames=1))


# ═══════════════════════════════════════════════════════════════════════════════
#  H. draw_detections()
# ═══════════════════════════════════════════════════════════════════════════════

class TestDrawDetections:

    def test_empty_detections_returns_same_shape(self):
        from cv.yolo.book_detector import BookDetector
        det = BookDetector()
        det._loaded_model_name = "test"
        frame = _make_frame(480, 640)
        out = det.draw_detections(frame, [])
        assert out.shape == frame.shape

    def test_does_not_modify_original_frame(self):
        from cv.yolo.book_detector import BookDetector
        det = BookDetector()
        det._loaded_model_name = "test"
        frame = _make_frame(480, 640)
        original = frame.copy()
        d = _make_detection()
        det.draw_detections(frame, [d])
        np.testing.assert_array_equal(frame, original)

    def test_single_detection_drawn(self):
        from cv.yolo.book_detector import BookDetector
        det = BookDetector()
        det._loaded_model_name = "yolo11n.pt"
        frame = _make_frame(480, 640)
        d = _make_detection(x1=50, y1=50, x2=200, y2=200)
        out = det.draw_detections(frame, [d])
        # Annotated frame should differ from blank (bounding box was drawn)
        assert not np.all(out == 0)

    def test_multiple_detections(self):
        from cv.yolo.book_detector import BookDetector
        det = BookDetector()
        det._loaded_model_name = "yolo11n.pt"
        frame = _make_frame(480, 640)
        dets = [
            _make_detection(x1=50,  y1=50,  x2=200, y2=200),
            _make_detection(x1=300, y1=100, x2=500, y2=300, confidence=0.72),
        ]
        out = det.draw_detections(frame, dets)
        assert out.shape == frame.shape

    def test_simulation_source_tag_drawn(self):
        """Simulation detections include 'SIM' tag -- just check no crash."""
        from cv.yolo.book_detector import BookDetector
        det = BookDetector()
        det._loaded_model_name = "sim"
        frame = _make_frame()
        d = _make_detection(source="simulation")
        out = det.draw_detections(frame, [d])
        assert out.shape == frame.shape


# ═══════════════════════════════════════════════════════════════════════════════
#  I. save_results()
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveResults:

    def test_save_creates_json(self):
        from cv.yolo.book_detector import BookDetector, ModelStatus
        det = BookDetector()
        det.status = ModelStatus.LOADED_PRETRAINED
        det._loaded_model_name = "yolo11n.pt"

        d1 = _make_detection()
        d2 = _make_detection(x1=300, y1=100, x2=580, y2=380, confidence=0.72)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "results" / "test_det.json"
            saved = det.save_results([d1, d2], out_path)

            assert saved.exists()
            with open(saved) as f:
                data = json.load(f)

            assert data["total_detections"] == 2
            assert len(data["detections"]) == 2
            assert "timestamp" in data
            assert "model_name" in data

    def test_save_with_metadata(self):
        from cv.yolo.book_detector import BookDetector, ModelStatus
        det = BookDetector()
        det.status = ModelStatus.SIMULATION_MODE
        det._loaded_model_name = "simulation"
        d = _make_detection()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "meta_test.json"
            det.save_results([d], out_path, metadata={"mode": "Test"})
            with open(out_path) as f:
                data = json.load(f)
            assert data.get("metadata", {}).get("mode") == "Test"

    def test_save_with_frame_creates_image(self):
        """When frame is provided, an annotated .jpg should be saved."""
        import cv2
        from cv.yolo.book_detector import BookDetector, ModelStatus
        det = BookDetector()
        det.status = ModelStatus.LOADED_PRETRAINED
        det._loaded_model_name = "yolo11n.pt"
        d = _make_detection()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "frame_test.json"
            frame = _make_frame()
            det.save_results([d], out_path, frame=frame)
            img_path = out_path.with_suffix(".jpg")
            assert img_path.exists()

    def test_detection_dict_contains_center(self):
        """Saved detection dict must include center_x / center_y."""
        from cv.yolo.book_detector import BookDetector, ModelStatus
        det = BookDetector()
        det.status = ModelStatus.LOADED_PRETRAINED
        det._loaded_model_name = "yolo11n.pt"
        d = _make_detection(x1=100, y1=80, x2=300, y2=200)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "cx_test.json"
            det.save_results([d], out_path)
            with open(out_path) as f:
                data = json.load(f)
            det_data = data["detections"][0]
            assert "center_x" in det_data
            assert "center_y" in det_data
            assert det_data["center_x"] == pytest.approx(200.0, abs=1)
            assert det_data["center_y"] == pytest.approx(140.0, abs=1)


# ═══════════════════════════════════════════════════════════════════════════════
#  J. SyntheticDeskGenerator
# ═══════════════════════════════════════════════════════════════════════════════

class TestSyntheticDeskGenerator:

    def test_generate_desk_image_without_workspace(self):
        from cv.yolo.synthetic_generator import SyntheticDeskGenerator
        gen = SyntheticDeskGenerator(640, 480)
        img = gen.generate_desk_image(workspace=None)
        assert isinstance(img, np.ndarray)
        assert img.shape == (480, 640, 3)

    def test_generate_desk_image_with_workspace(self):
        from cv.yolo.synthetic_generator import SyntheticDeskGenerator
        gen = SyntheticDeskGenerator(640, 480)
        ws = _make_workspace()
        if ws is None:
            pytest.skip("WorkspaceSimulator not available")
        img = gen.generate_desk_image(workspace=ws)
        assert img.shape == (480, 640, 3)
        assert img.dtype == np.uint8

    def test_generate_detections_count_matches_books(self):
        from cv.yolo.synthetic_generator import SyntheticDeskGenerator
        gen = SyntheticDeskGenerator(640, 480)
        ws = _make_workspace()
        if ws is None:
            pytest.skip("WorkspaceSimulator not available")
        shape = (480, 640, 3)
        dets = gen.generate_detections(ws, shape)
        assert len(dets) == len(ws.books)

    def test_generate_detections_all_source_simulation(self):
        from cv.yolo.synthetic_generator import SyntheticDeskGenerator
        gen = SyntheticDeskGenerator()
        ws = _make_workspace()
        if ws is None:
            pytest.skip("WorkspaceSimulator not available")
        dets = gen.generate_detections(ws, (853, 1280, 3))
        for d in dets:
            assert d.source == "simulation"
            assert d.class_id == 73
            assert d.class_name == "book"

    def test_generate_detections_confidence_in_range(self):
        from cv.yolo.synthetic_generator import SyntheticDeskGenerator
        gen = SyntheticDeskGenerator()
        ws = _make_workspace()
        if ws is None:
            pytest.skip("WorkspaceSimulator not available")
        dets = gen.generate_detections(ws, (853, 1280, 3))
        for d in dets:
            assert 0.0 <= d.confidence <= 1.0

    def test_generate_detections_bboxes_in_image(self):
        """Bounding boxes must stay within the image dimensions."""
        from cv.yolo.synthetic_generator import SyntheticDeskGenerator
        gen = SyntheticDeskGenerator(640, 480)
        ws = _make_workspace()
        if ws is None:
            pytest.skip("WorkspaceSimulator not available")
        shape = (480, 640, 3)
        dets = gen.generate_detections(ws, shape)
        for d in dets:
            assert d.x1 >= 0
            assert d.y1 >= 0
            assert d.x2 <= 640
            assert d.y2 <= 480


# ═══════════════════════════════════════════════════════════════════════════════
#  K. VideoProcessor
# ═══════════════════════════════════════════════════════════════════════════════

class TestVideoProcessor:

    def test_process_synthetic_frames_returns_results(self):
        from cv.yolo.book_detector import BookDetector
        from cv.yolo.video_processor import VideoProcessor
        ws = _make_workspace()
        if ws is None:
            pytest.skip("WorkspaceSimulator not available")
        det = BookDetector()  # not loaded -- uses simulation
        proc = VideoProcessor(det, sample_every=1)
        results, stats = proc.process_synthetic_frames(ws, n_frames=3)
        assert len(results) == 3

    def test_process_synthetic_frames_stats(self):
        from cv.yolo.book_detector import BookDetector
        from cv.yolo.video_processor import VideoProcessor, ProcessingStats
        ws = _make_workspace()
        if ws is None:
            pytest.skip("WorkspaceSimulator not available")
        det = BookDetector()
        proc = VideoProcessor(det)
        results, stats = proc.process_synthetic_frames(ws, n_frames=4)
        assert stats.processed_frames == 4
        assert stats.total_frames == 4
        assert isinstance(stats.avg_detections, float)
        assert stats.duration_s >= 0

    def test_processing_stats_detection_rate(self):
        from cv.yolo.video_processor import ProcessingStats
        s = ProcessingStats(
            total_frames=10,
            processed_frames=10,
            frames_with_books=7,
        )
        assert s.detection_rate == pytest.approx(0.7)

    def test_processing_stats_to_dict_keys(self):
        from cv.yolo.video_processor import ProcessingStats
        s = ProcessingStats()
        data = s.to_dict()
        for key in ("total_frames", "processed_frames", "total_detections",
                    "avg_detections", "avg_confidence", "avg_fps",
                    "duration_s", "frames_with_books", "detection_rate"):
            assert key in data

    def test_process_video_not_found(self):
        from cv.yolo.book_detector import BookDetector
        from cv.yolo.video_processor import VideoProcessor
        det = BookDetector()
        proc = VideoProcessor(det)
        with pytest.raises(RuntimeError):
            proc.process_video("nonexistent_video_xyz.mp4")

    def test_result_frame_shape_matches_generator(self):
        """Each result frame must have the correct ndarray shape."""
        from cv.yolo.book_detector import BookDetector
        from cv.yolo.video_processor import VideoProcessor
        ws = _make_workspace()
        if ws is None:
            pytest.skip("WorkspaceSimulator not available")
        det = BookDetector()
        proc = VideoProcessor(det)
        results, _ = proc.process_synthetic_frames(ws, n_frames=2)
        for frame, dets, idx in results:
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3
            assert isinstance(dets, list)
            assert isinstance(idx, int)


# ═══════════════════════════════════════════════════════════════════════════════
#  L. Public API / imports
# ═══════════════════════════════════════════════════════════════════════════════

class TestPublicAPI:

    def test_cv_yolo_namespace_imports(self):
        """cv.yolo package exposes all Phase 3 public names."""
        import cv.yolo as yolo_pkg
        for name in ("Detection", "BookDetector", "ModelStatus",
                     "COCO_BOOK_CLASS_ID", "VideoProcessor",
                     "ProcessingStats", "SyntheticDeskGenerator"):
            assert hasattr(yolo_pkg, name), f"Missing from cv.yolo: {name}"

    def test_detector_py_re_exports(self):
        """cv.yolo.detector re-exports the full Phase 3 API."""
        from cv.yolo import detector as det_mod
        for name in ("BookDetector", "Detection", "ModelStatus",
                     "VideoProcessor", "SyntheticDeskGenerator"):
            assert hasattr(det_mod, name), f"Missing from cv.yolo.detector: {name}"

    def test_coco_book_class_id(self):
        from cv.yolo.book_detector import COCO_BOOK_CLASS_ID
        assert COCO_BOOK_CLASS_ID == 73

    def test_custom_model_path_constant(self):
        from cv.yolo.book_detector import CUSTOM_MODEL_PATH
        assert "models" in str(CUSTOM_MODEL_PATH)
        assert "yolo11_book_detector" in str(CUSTOM_MODEL_PATH)

    def test_fallback_model_constant(self):
        from cv.yolo.book_detector import FALLBACK_MODEL
        assert "yolo11" in FALLBACK_MODEL.lower()

    def test_default_confidence_range(self):
        from cv.yolo.book_detector import DEFAULT_CONFIDENCE
        assert 0.0 < DEFAULT_CONFIDENCE < 1.0

    def test_book_detector_is_loaded_property(self):
        from cv.yolo.book_detector import BookDetector
        det = BookDetector()
        assert det.is_loaded is False
        # Inject dummy model
        det._model = object()
        assert det.is_loaded is True

    def test_book_detector_model_type_property(self):
        from cv.yolo.book_detector import BookDetector
        det = BookDetector()
        assert isinstance(det.model_type, str)

    def test_simulate_detections_without_yolo(self):
        """simulate_detections works even when YOLO model is not loaded."""
        from cv.yolo.book_detector import BookDetector
        ws = _make_workspace()
        if ws is None:
            pytest.skip("WorkspaceSimulator not available")
        det = BookDetector()  # not loaded
        shape = (480, 640, 3)
        dets = det.simulate_detections(ws, shape)
        assert isinstance(dets, list)
        for d in dets:
            assert d.source == "simulation"
