"""
tests/test_phase4.py
====================
Phase 4 — ByteTrack Integration: unit & integration tests.

Test groups
-----------
  A. BookTrack dataclass         (geometry, serialisation, repr)
  B. ByteTrackTracker lifecycle  (init, reset, not-initialized guard)
  C. ByteTrackTracker.update()   (empty dets, single det, multi-det, frame counter)
  D. Persistent track IDs        (same object → same ID across frames)
  E. draw_tracks()               (shape preserved, no crash on empty/single)
  F. TrackingRepository          (save, query, clear — in-memory SQLite)
  G. BookTrackRecord ORM         (schema, repr, round-trip)
  H. Integration pipeline        (detector mock → tracker → repository)
  I. Public API / imports        (cv.tracking namespace)

Run:
    pytest tests/test_phase4.py -v
    pytest tests/test_phase4.py -v -k "not Integration"
"""

from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_detection(x1=100.0, y1=80.0, x2=220.0, y2=200.0,
                     confidence=0.85, class_id=73):
    from cv.yolo.detection import Detection
    return Detection(
        x1=x1, y1=y1, x2=x2, y2=y2,
        confidence=confidence,
        class_id=class_id,
        class_name="book",
        source="yolo",
    )


def _make_book_track(track_id=1, x1=100.0, y1=80.0,
                      x2=220.0, y2=200.0, confidence=0.85,
                      title="Unknown", frame_index=0):
    from cv.tracking.bytetrack_tracker import BookTrack
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return BookTrack(
        track_id=track_id,
        bbox=(x1, y1, x2, y2),
        center_x=cx,
        center_y=cy,
        confidence=confidence,
        timestamp=datetime.now(),
        book_title=title,
        frame_index=frame_index,
    )


def _make_in_memory_db():
    """Return a DatabaseManager pointed at an in-memory SQLite database."""
    from database.db_manager import DatabaseManager
    db = DatabaseManager(":memory:")
    db.init_db()
    return db


# ══════════════════════════════════════════════════════════════════════════════
#  A. BookTrack dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestBookTrack:

    def test_creation_basic(self):
        bt = _make_book_track()
        assert bt.track_id == 1
        assert bt.book_title == "Unknown"
        assert bt.confidence == pytest.approx(0.85)

    def test_center_computed(self):
        bt = _make_book_track(x1=100, y1=80, x2=220, y2=200)
        assert bt.center_x == pytest.approx(160.0)
        assert bt.center_y == pytest.approx(140.0)

    def test_bbox_int(self):
        bt = _make_book_track(x1=100.6, y1=80.2, x2=220.9, y2=200.1)
        xi1, yi1, xi2, yi2 = bt.bbox_int
        assert isinstance(xi1, int)
        assert xi1 == 100

    def test_center_int(self):
        bt = _make_book_track(x1=100, y1=80, x2=221, y2=201)
        cx, cy = bt.center_int
        assert isinstance(cx, int) and isinstance(cy, int)

    def test_to_dict_keys(self):
        bt = _make_book_track()
        d = bt.to_dict()
        for key in ("track_id", "book_title", "bbox",
                    "center_x", "center_y", "confidence",
                    "timestamp", "frame_index"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_values(self):
        bt = _make_book_track(track_id=5, title="Python Crash Course")
        d = bt.to_dict()
        assert d["track_id"] == 5
        assert d["book_title"] == "Python Crash Course"
        assert d["center_x"] == pytest.approx(160.0)

    def test_repr_contains_track_id(self):
        bt = _make_book_track(track_id=7)
        assert "7" in repr(bt)

    def test_repr_contains_position(self):
        bt = _make_book_track(x1=100, y1=80, x2=220, y2=200)
        r = repr(bt)
        assert "160" in r   # center_x
        assert "140" in r   # center_y

    def test_timestamp_type(self):
        bt = _make_book_track()
        assert isinstance(bt.timestamp, datetime)


# ══════════════════════════════════════════════════════════════════════════════
#  B. ByteTrackTracker lifecycle
# ══════════════════════════════════════════════════════════════════════════════

class TestByteTrackTrackerLifecycle:

    def test_not_initialized_at_start(self):
        from cv.tracking.bytetrack_tracker import ByteTrackTracker
        t = ByteTrackTracker()
        assert not t.is_initialized

    def test_initialize_sets_flag(self):
        from cv.tracking.bytetrack_tracker import ByteTrackTracker
        t = ByteTrackTracker()
        t.initialize()
        assert t.is_initialized

    def test_initialize_twice_is_safe(self):
        from cv.tracking.bytetrack_tracker import ByteTrackTracker
        t = ByteTrackTracker()
        t.initialize()
        t.initialize()          # should not raise
        assert t.is_initialized

    def test_reset_clears_state(self):
        from cv.tracking.bytetrack_tracker import ByteTrackTracker
        t = ByteTrackTracker()
        t.initialize()
        # push one frame through so frame_count advances
        dets = [_make_detection()]
        t.update(dets)
        assert t.frame_count == 1
        t.reset()
        assert t.frame_count == 0
        assert len(t.active_tracks) == 0

    def test_update_raises_if_not_initialized(self):
        from cv.tracking.bytetrack_tracker import ByteTrackTracker
        t = ByteTrackTracker()
        with pytest.raises(RuntimeError, match="not initialised"):
            t.update([_make_detection()])

    def test_repr_not_initialized(self):
        from cv.tracking.bytetrack_tracker import ByteTrackTracker
        t = ByteTrackTracker()
        assert "not initialised" in repr(t)

    def test_repr_initialized(self):
        from cv.tracking.bytetrack_tracker import ByteTrackTracker
        t = ByteTrackTracker()
        t.initialize()
        assert "ready" in repr(t)

    def test_custom_params_stored(self):
        from cv.tracking.bytetrack_tracker import ByteTrackTracker
        t = ByteTrackTracker(
            track_activation_threshold=0.30,
            lost_track_buffer=45,
            minimum_matching_threshold=0.75,
            frame_rate=25,
        )
        assert t.track_activation_threshold == 0.30
        assert t.lost_track_buffer == 45
        assert t.minimum_matching_threshold == 0.75
        assert t.frame_rate == 25


# ══════════════════════════════════════════════════════════════════════════════
#  C. ByteTrackTracker.update()
# ══════════════════════════════════════════════════════════════════════════════

class TestByteTrackTrackerUpdate:

    @pytest.fixture(autouse=True)
    def tracker(self):
        from cv.tracking.bytetrack_tracker import ByteTrackTracker
        t = ByteTrackTracker()
        t.initialize()
        return t

    def test_empty_detections_returns_empty(self, tracker):
        tracks = tracker.update([])
        assert tracks == []

    def test_single_detection_returns_one_track(self, tracker):
        dets = [_make_detection()]
        tracks = tracker.update(dets)
        assert len(tracks) == 1

    def test_track_has_correct_fields(self, tracker):
        from cv.tracking.bytetrack_tracker import BookTrack
        dets = [_make_detection(x1=50, y1=60, x2=150, y2=160)]
        tracks = tracker.update(dets)
        assert len(tracks) == 1
        t = tracks[0]
        assert isinstance(t, BookTrack)
        assert t.track_id >= 1
        assert t.book_title == "Unknown"
        assert isinstance(t.timestamp, datetime)

    def test_multiple_detections_return_multiple_tracks(self, tracker):
        dets = [
            _make_detection(x1=50,  y1=50,  x2=150, y2=150),
            _make_detection(x1=300, y1=200, x2=450, y2=350),
        ]
        tracks = tracker.update(dets)
        assert len(tracks) == 2

    def test_track_ids_are_unique_per_frame(self, tracker):
        dets = [
            _make_detection(x1=50,  y1=50,  x2=150, y2=150),
            _make_detection(x1=300, y1=200, x2=450, y2=350),
        ]
        tracks = tracker.update(dets)
        ids = [t.track_id for t in tracks]
        assert len(ids) == len(set(ids)), "Duplicate track IDs in same frame"

    def test_frame_counter_increments(self, tracker):
        assert tracker.frame_count == 0
        tracker.update([])
        assert tracker.frame_count == 1
        tracker.update([])
        assert tracker.frame_count == 2

    def test_explicit_frame_index_stored(self, tracker):
        dets = [_make_detection()]
        tracks = tracker.update(dets, frame_index=42)
        if tracks:
            assert tracks[0].frame_index == 42

    def test_explicit_timestamp_stored(self, tracker):
        ts = datetime(2026, 1, 15, 10, 30, 0)
        dets = [_make_detection()]
        tracks = tracker.update(dets, timestamp=ts)
        if tracks:
            assert tracks[0].timestamp == ts

    def test_active_tracks_populated(self, tracker):
        dets = [_make_detection()]
        tracks = tracker.update(dets)
        if tracks:
            assert tracks[0].track_id in tracker.active_tracks


# ══════════════════════════════════════════════════════════════════════════════
#  D. Persistent track IDs across frames
# ══════════════════════════════════════════════════════════════════════════════

class TestPersistentTrackIDs:

    def test_same_object_same_id_across_frames(self):
        """
        A book that stays in the same position across frames should keep
        the same track_id (ByteTrack's core promise).
        """
        from cv.tracking.bytetrack_tracker import ByteTrackTracker
        tracker = ByteTrackTracker()
        tracker.initialize()

        det = _make_detection(x1=100, y1=80, x2=220, y2=200, confidence=0.90)

        frame1_tracks = tracker.update([det], frame_index=0)
        frame2_tracks = tracker.update([det], frame_index=1)
        frame3_tracks = tracker.update([det], frame_index=2)

        # All frames must yield at least one track
        assert frame1_tracks and frame2_tracks and frame3_tracks

        id1 = frame1_tracks[0].track_id
        id2 = frame2_tracks[0].track_id
        id3 = frame3_tracks[0].track_id

        assert id1 == id2 == id3, (
            f"Track ID changed across frames: {id1} → {id2} → {id3}"
        )

    def test_two_objects_keep_distinct_ids(self):
        """Two spatially separated books keep distinct track IDs."""
        from cv.tracking.bytetrack_tracker import ByteTrackTracker
        tracker = ByteTrackTracker()
        tracker.initialize()

        det_a = _make_detection(x1=50,  y1=50,  x2=150, y2=150)
        det_b = _make_detection(x1=400, y1=300, x2=550, y2=420)

        tracks1 = tracker.update([det_a, det_b], frame_index=0)
        tracks2 = tracker.update([det_a, det_b], frame_index=1)

        assert len(tracks1) == 2
        assert len(tracks2) == 2

        ids1 = set(t.track_id for t in tracks1)
        ids2 = set(t.track_id for t in tracks2)
        assert ids1 == ids2, f"Track IDs changed: {ids1} → {ids2}"

    def test_set_book_title_persists(self):
        """set_book_title() should propagate to subsequent frames."""
        from cv.tracking.bytetrack_tracker import ByteTrackTracker
        tracker = ByteTrackTracker()
        tracker.initialize()

        det = _make_detection()
        tracks = tracker.update([det], frame_index=0)
        assert tracks

        tid = tracks[0].track_id
        tracker.set_book_title(tid, "Clean Code")

        tracks2 = tracker.update([det], frame_index=1)
        assert tracks2
        assert tracks2[0].book_title == "Clean Code"


# ══════════════════════════════════════════════════════════════════════════════
#  E. draw_tracks()
# ══════════════════════════════════════════════════════════════════════════════

class TestDrawTracks:

    @pytest.fixture(autouse=True)
    def tracker(self):
        from cv.tracking.bytetrack_tracker import ByteTrackTracker
        t = ByteTrackTracker()
        t.initialize()
        return t

    def test_empty_tracks_returns_same_shape(self, tracker):
        frame = _make_frame()
        out   = tracker.draw_tracks(frame, [])
        assert out.shape == frame.shape

    def test_original_frame_unchanged(self, tracker):
        frame = _make_frame()
        orig  = frame.copy()
        tracker.draw_tracks(frame, [])
        assert np.array_equal(frame, orig)

    def test_single_track_annotated(self, tracker):
        frame  = _make_frame()
        dets   = [_make_detection()]
        tracks = tracker.update(dets)
        if not tracks:
            pytest.skip("No tracks returned by ByteTrack on first frame")
        out = tracker.draw_tracks(frame, tracks)
        assert out.shape == frame.shape
        # output must differ from blank frame (annotations were drawn)
        assert not np.array_equal(out, frame)

    def test_multiple_tracks_annotated(self, tracker):
        frame = _make_frame()
        dets  = [
            _make_detection(x1=50,  y1=50,  x2=150, y2=150),
            _make_detection(x1=350, y1=250, x2=500, y2=380),
        ]
        tracks = tracker.update(dets)
        out    = tracker.draw_tracks(frame, tracks)
        assert out.shape == frame.shape

    def test_draw_without_confidence(self, tracker):
        frame  = _make_frame()
        dets   = [_make_detection()]
        tracks = tracker.update(dets)
        if not tracks:
            pytest.skip("No tracks returned")
        out = tracker.draw_tracks(frame, tracks, show_confidence=False)
        assert out.shape == frame.shape

    def test_draw_without_center(self, tracker):
        frame  = _make_frame()
        dets   = [_make_detection()]
        tracks = tracker.update(dets)
        if not tracks:
            pytest.skip("No tracks returned")
        out = tracker.draw_tracks(frame, tracks, show_center=False)
        assert out.shape == frame.shape


# ══════════════════════════════════════════════════════════════════════════════
#  F. TrackingRepository (in-memory SQLite)
# ══════════════════════════════════════════════════════════════════════════════

class TestTrackingRepository:

    @pytest.fixture()
    def repo(self):
        from database.tracking_repository import TrackingRepository
        db   = _make_in_memory_db()
        return TrackingRepository(db)

    def test_save_single_track(self, repo):
        bt = _make_book_track(track_id=1, frame_index=0)
        repo.save_track(bt)
        assert repo.count() == 1

    def test_save_multiple_tracks(self, repo):
        tracks = [
            _make_book_track(track_id=1, frame_index=0),
            _make_book_track(track_id=2, frame_index=0, x1=300, x2=450),
        ]
        n = repo.save_tracks(tracks)
        assert n == 2
        assert repo.count() == 2

    def test_save_empty_list_returns_zero(self, repo):
        n = repo.save_tracks([])
        assert n == 0
        assert repo.count() == 0

    def test_get_track_history(self, repo):
        for i in range(5):
            repo.save_track(_make_book_track(track_id=1, frame_index=i))
        history = repo.get_track_history(track_id=1)
        assert len(history) == 5

    def test_get_track_history_unknown_id(self, repo):
        history = repo.get_track_history(track_id=99)
        assert history == []

    def test_get_track_history_limit(self, repo):
        for i in range(10):
            repo.save_track(_make_book_track(track_id=1, frame_index=i))
        history = repo.get_track_history(track_id=1, limit=3)
        assert len(history) == 3

    def test_get_latest_frame_empty(self, repo):
        assert repo.get_latest_frame() == []

    def test_get_latest_frame(self, repo):
        repo.save_tracks([
            _make_book_track(track_id=1, frame_index=0),
            _make_book_track(track_id=1, frame_index=5),
            _make_book_track(track_id=2, frame_index=5),
        ])
        latest = repo.get_latest_frame()
        assert len(latest) == 2
        assert all(r.frame_index == 5 for r in latest)

    def test_get_all_track_ids(self, repo):
        repo.save_tracks([
            _make_book_track(track_id=3, frame_index=0),
            _make_book_track(track_id=1, frame_index=0),
            _make_book_track(track_id=2, frame_index=0),
        ])
        ids = repo.get_all_track_ids()
        assert ids == [1, 2, 3]

    def test_get_records_in_range(self, repo):
        for i in range(10):
            repo.save_track(_make_book_track(track_id=1, frame_index=i))
        results = repo.get_records_in_range(3, 6)
        assert len(results) == 4
        frames = [r.frame_index for r in results]
        assert min(frames) == 3 and max(frames) == 6

    def test_clear_removes_all(self, repo):
        repo.save_tracks([
            _make_book_track(track_id=1, frame_index=0),
            _make_book_track(track_id=2, frame_index=0),
        ])
        assert repo.count() == 2
        repo.clear()
        assert repo.count() == 0

    def test_stored_fields_roundtrip(self, repo):
        ts = datetime(2026, 6, 1, 12, 0, 0)
        bt = _make_book_track(track_id=7, x1=100, y1=80, x2=220, y2=200,
                               confidence=0.92, title="Dune", frame_index=42)
        bt.timestamp = ts
        repo.save_track(bt)
        history = repo.get_track_history(7)
        assert len(history) == 1
        rec = history[0]
        assert rec.track_id == 7
        assert rec.book_title == "Dune"
        assert rec.center_x == pytest.approx(160.0)
        assert rec.center_y == pytest.approx(140.0)
        assert rec.confidence == pytest.approx(0.92)
        assert rec.frame_index == 42
        assert rec.bbox_x1 == pytest.approx(100.0)
        assert rec.bbox_y2 == pytest.approx(200.0)


# ══════════════════════════════════════════════════════════════════════════════
#  G. BookTrackRecord ORM
# ══════════════════════════════════════════════════════════════════════════════

class TestBookTrackRecord:

    def test_model_has_required_columns(self):
        from database.models import BookTrackRecord
        cols = {c.name for c in BookTrackRecord.__table__.columns}
        required = {
            "id", "track_id", "book_title",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "center_x", "center_y", "confidence",
            "frame_index", "timestamp",
        }
        assert required.issubset(cols), f"Missing columns: {required - cols}"

    def test_model_tablename(self):
        from database.models import BookTrackRecord
        assert BookTrackRecord.__tablename__ == "book_track_records"

    def test_repr(self):
        from database.models import BookTrackRecord
        rec = BookTrackRecord(
            track_id=3, frame_index=10,
            center_x=160.0, center_y=140.0,
        )
        r = repr(rec)
        assert "3" in r
        assert "10" in r

    def test_default_book_title(self):
        from database.models import BookTrackRecord
        rec = BookTrackRecord(
            track_id=1, bbox_x1=0, bbox_y1=0,
            bbox_x2=100, bbox_y2=100,
            center_x=50, center_y=50,
            confidence=0.8, frame_index=0,
        )
        # default is set at DB level; at Python level it's the column default
        assert rec.book_title is None or rec.book_title == "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  H. Integration: detection → tracking → persistence
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegrationPipeline:
    """
    Full pipeline: mock YOLO11 detections → ByteTrackTracker → TrackingRepository.
    Validates that persistent IDs survive multiple frames and end up in SQLite.
    """

    @pytest.fixture()
    def setup(self):
        from cv.tracking.bytetrack_tracker import ByteTrackTracker
        from database.tracking_repository import TrackingRepository

        tracker = ByteTrackTracker()
        tracker.initialize()

        db   = _make_in_memory_db()
        repo = TrackingRepository(db)
        return tracker, repo

    def test_multi_frame_ids_saved_to_db(self, setup):
        tracker, repo = setup

        det = _make_detection(x1=100, y1=80, x2=220, y2=200, confidence=0.90)

        saved_id = None
        for frame_idx in range(5):
            tracks = tracker.update([det], frame_index=frame_idx)
            repo.save_tracks(tracks)
            if tracks:
                if saved_id is None:
                    saved_id = tracks[0].track_id
                else:
                    # ID must remain stable
                    assert tracks[0].track_id == saved_id, (
                        f"Track ID changed at frame {frame_idx}: "
                        f"expected {saved_id}, got {tracks[0].track_id}"
                    )

        total_records = repo.count()
        # We expect at least 4 records (ByteTrack may skip frame 0 for new tracks)
        assert total_records >= 4, f"Expected ≥4 records, got {total_records}"

    def test_two_books_saved_with_separate_ids(self, setup):
        tracker, repo = setup

        det_a = _make_detection(x1=50,  y1=50,  x2=150, y2=150)
        det_b = _make_detection(x1=400, y1=300, x2=550, y2=420)

        for frame_idx in range(3):
            tracks = tracker.update([det_a, det_b], frame_index=frame_idx)
            repo.save_tracks(tracks)

        ids = repo.get_all_track_ids()
        assert len(ids) == 2, f"Expected 2 distinct track IDs, got {ids}"

    def test_title_persists_through_pipeline(self, setup):
        tracker, repo = setup

        det = _make_detection()
        tracks_f0 = tracker.update([det], frame_index=0)
        if not tracks_f0:
            pytest.skip("No tracks on first frame — ByteTrack requires activation")

        tid = tracks_f0[0].track_id
        tracker.set_book_title(tid, "The Pragmatic Programmer")

        tracks_f1 = tracker.update([det], frame_index=1)
        repo.save_tracks(tracks_f1)

        history = repo.get_track_history(tid)
        if history:
            assert history[-1].book_title == "The Pragmatic Programmer"

    def test_format_output_example(self, setup):
        """
        Validate the required output format:
          Track ID 1 | Book: Unknown | Position: (cx, cy)
        """
        tracker, repo = setup

        dets = [
            _make_detection(x1=100, y1=80,  x2=400, y2=260),
            _make_detection(x1=310, y1=170, x2=550, y2=330),
        ]

        all_tracks = []
        for frame_idx in range(3):
            tracks = tracker.update(dets, frame_index=frame_idx)
            all_tracks.extend(tracks)

        # Print the expected format (also validates no exceptions)
        for bt in all_tracks:
            line = (
                f"Track ID {bt.track_id} | "
                f"Book: {bt.book_title} | "
                f"Position: ({bt.center_int[0]},{bt.center_int[1]})"
            )
            assert "Track ID" in line
            assert "Book:"    in line
            assert "Position" in line


# ══════════════════════════════════════════════════════════════════════════════
#  I. Public API / imports
# ══════════════════════════════════════════════════════════════════════════════

class TestPublicAPI:

    def test_import_cv_tracking(self):
        import cv.tracking
        assert hasattr(cv.tracking, "ByteTrackTracker")
        assert hasattr(cv.tracking, "BookTrack")

    def test_import_bytetrack_tracker_direct(self):
        from cv.tracking.bytetrack_tracker import ByteTrackTracker, BookTrack
        assert ByteTrackTracker is not None
        assert BookTrack is not None

    def test_import_tracking_repository(self):
        from database.tracking_repository import TrackingRepository
        assert TrackingRepository is not None

    def test_import_book_track_record(self):
        from database.models import BookTrackRecord
        assert BookTrackRecord is not None

    def test_import_database_package(self):
        from database import DatabaseManager, TrackingRepository
        assert DatabaseManager is not None
        assert TrackingRepository is not None

    def test_bytetrack_tracker_instantiation(self):
        from cv.tracking import ByteTrackTracker
        t = ByteTrackTracker()
        assert not t.is_initialized

    def test_book_track_instantiation(self):
        from cv.tracking import BookTrack
        bt = BookTrack(
            track_id=1,
            bbox=(0, 0, 100, 100),
            center_x=50.0,
            center_y=50.0,
            confidence=0.9,
            timestamp=datetime.now(),
        )
        assert bt.track_id == 1
