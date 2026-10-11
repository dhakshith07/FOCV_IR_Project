"""
database/tracking_repository.py
================================
Phase 4 — SQLite persistence layer for ByteTrack results.

Provides a thin repository around ``BookTrackRecord`` so the tracker can
store one row per (track_id, frame) observation without coupling business
logic to SQLAlchemy session management.

Usage
-----
    from database.db_manager import DatabaseManager
    from database.tracking_repository import TrackingRepository

    db   = DatabaseManager("data/workspace.db")
    db.init_db()
    repo = TrackingRepository(db)

    # After each tracker.update() call:
    repo.save_tracks(book_tracks)

    # Query:
    history = repo.get_track_history(track_id=1)
    latest  = repo.get_latest_frame()
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from database.models import BookTrackRecord
from utils.logger import get_logger

log = get_logger(__name__)


class TrackingRepository:
    """
    CRUD interface for ``BookTrackRecord`` rows.

    Parameters
    ----------
    db_manager : A ``DatabaseManager`` instance (engine + session factory).
    """

    def __init__(self, db_manager) -> None:
        self._db = db_manager

    # ── Write ─────────────────────────────────────────────────────────────
    def save_tracks(self, tracks: list) -> int:
        """
        Persist a list of ``BookTrack`` objects produced by ``ByteTrackTracker``.

        Parameters
        ----------
        tracks : List[BookTrack]

        Returns
        -------
        Number of rows inserted.
        """
        if not tracks:
            return 0

        records = [self._to_record(t) for t in tracks]

        with self._session() as session:
            session.add_all(records)
            session.commit()

        log.debug(f"Saved {len(records)} track record(s) to SQLite.")
        return len(records)

    def save_track(self, track) -> None:
        """Persist a single ``BookTrack`` observation."""
        record = self._to_record(track)
        with self._session() as session:
            session.add(record)
            session.commit()
        log.debug(f"Saved track record: {record!r}")

    # ── Read ──────────────────────────────────────────────────────────────
    def get_track_history(
        self,
        track_id: int,
        limit: Optional[int] = None,
    ) -> List[BookTrackRecord]:
        """
        Return all observations for a given track_id, ordered by frame.

        Parameters
        ----------
        track_id : ByteTrack persistent ID.
        limit    : If set, return only the most recent *limit* rows.
        """
        with self._session() as session:
            q = (
                session.query(BookTrackRecord)
                .filter(BookTrackRecord.track_id == track_id)
                .order_by(BookTrackRecord.frame_index.asc())
            )
            if limit:
                q = q.limit(limit)
            results = q.all()
            session.expunge_all()   # detach so they survive outside context
        return results

    def get_latest_frame(self) -> List[BookTrackRecord]:
        """Return all track records for the most recently saved frame."""
        with self._session() as session:
            row = (
                session.query(BookTrackRecord.frame_index)
                .order_by(BookTrackRecord.frame_index.desc())
                .first()
            )
            if row is None:
                return []
            max_frame = row[0]
            results = (
                session.query(BookTrackRecord)
                .filter(BookTrackRecord.frame_index == max_frame)
                .all()
            )
            session.expunge_all()
        return results

    def get_all_track_ids(self) -> List[int]:
        """Return a sorted list of all distinct track IDs stored so far."""
        with self._session() as session:
            rows = session.query(BookTrackRecord.track_id).distinct().all()
        return sorted(r[0] for r in rows)

    def get_records_in_range(
        self,
        frame_start: int,
        frame_end: int,
    ) -> List[BookTrackRecord]:
        """Return all observations between frame_start and frame_end (inclusive)."""
        with self._session() as session:
            results = (
                session.query(BookTrackRecord)
                .filter(
                    BookTrackRecord.frame_index >= frame_start,
                    BookTrackRecord.frame_index <= frame_end,
                )
                .order_by(BookTrackRecord.frame_index, BookTrackRecord.track_id)
                .all()
            )
            session.expunge_all()
        return results

    def count(self) -> int:
        """Return the total number of stored track records."""
        with self._session() as session:
            return session.query(BookTrackRecord).count()

    def clear(self) -> None:
        """Delete ALL track records (useful for testing / fresh sessions)."""
        with self._session() as session:
            n = session.query(BookTrackRecord).delete()
            session.commit()
        log.info(f"Cleared {n} track record(s) from SQLite.")

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _to_record(track) -> BookTrackRecord:
        """Convert a ``BookTrack`` dataclass to a ``BookTrackRecord`` ORM object."""
        return BookTrackRecord(
            track_id    = track.track_id,
            book_title  = track.book_title,
            bbox_x1     = track.bbox[0],
            bbox_y1     = track.bbox[1],
            bbox_x2     = track.bbox[2],
            bbox_y2     = track.bbox[3],
            center_x    = track.center_x,
            center_y    = track.center_y,
            confidence  = track.confidence,
            frame_index = track.frame_index,
            timestamp   = track.timestamp,
        )

    @contextmanager
    def _session(self):
        """Yield a managed SQLAlchemy session."""
        yield from self._db.get_session()
