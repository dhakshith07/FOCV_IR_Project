"""
database/ocr_repository.py
===========================
Phase 5 — SQLite persistence layer for TrOCR results.

Provides a repository around ``OCRRecord`` so the BookTextReader pipeline
can store every OCR read result without coupling business logic to SQLAlchemy.

Usage
-----
    from database.db_manager import DatabaseManager
    from database.ocr_repository import OCRRepository

    db   = DatabaseManager("data/workspace.db")
    db.init_db()
    repo = OCRRepository(db)

    # After each reader.read_from_frame() call:
    repo.save_result(ocr_result, frame_index=42)

    # Query:
    history = repo.get_history_for_track(track_id=1)
    best    = repo.get_best_title(track_id=1)
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional

from database.models import OCRRecord
from utils.logger import get_logger

log = get_logger(__name__)


class OCRRepository:
    """
    CRUD interface for ``OCRRecord`` rows.

    Parameters
    ----------
    db_manager : A ``DatabaseManager`` instance.
    """

    def __init__(self, db_manager) -> None:
        self._db = db_manager

    # ── Write ─────────────────────────────────────────────────────────────
    def save_result(self, result, frame_index: int = 0) -> None:
        """
        Persist a ``BookReadResult`` produced by ``BookTextReader``.

        Parameters
        ----------
        result      : BookReadResult dataclass.
        frame_index : Frame number within the video/stream.
        """
        record = OCRRecord(
            track_id    = result.track_id,
            frame_index = frame_index,
            raw_text    = result.raw_text,
            clean_text  = result.clean_text,
            book_id     = result.book_id,
            book_title  = result.title,
            confidence  = result.confidence,
            match_score = result.match_score,
            uncertain   = result.uncertain,
            elapsed_ms  = result.elapsed_ms,
            timestamp   = datetime.now(timezone.utc),
        )
        with self._session() as session:
            session.add(record)
            session.commit()
        log.debug(f"OCR record saved: track={result.track_id} title={result.title!r}")

    def save_results(self, results: list, frame_index: int = 0) -> int:
        """
        Bulk-persist a list of BookReadResult objects.

        Returns
        -------
        Number of rows inserted.
        """
        if not results:
            return 0
        records = [
            OCRRecord(
                track_id    = r.track_id,
                frame_index = frame_index,
                raw_text    = r.raw_text,
                clean_text  = r.clean_text,
                book_id     = r.book_id,
                book_title  = r.title,
                confidence  = r.confidence,
                match_score = r.match_score,
                uncertain   = r.uncertain,
                elapsed_ms  = r.elapsed_ms,
                timestamp   = datetime.now(timezone.utc),
            )
            for r in results
        ]
        with self._session() as session:
            session.add_all(records)
            session.commit()
        log.debug(f"Saved {len(records)} OCR record(s).")
        return len(records)

    # ── Read ──────────────────────────────────────────────────────────────
    def get_history_for_track(
        self,
        track_id: int,
        limit: Optional[int] = None,
    ) -> List[OCRRecord]:
        """Return all OCR results for *track_id*, ordered by frame index."""
        with self._session() as session:
            q = (
                session.query(OCRRecord)
                .filter(OCRRecord.track_id == track_id)
                .order_by(OCRRecord.frame_index.asc())
            )
            if limit:
                q = q.limit(limit)
            results = q.all()
            session.expunge_all()
        return results

    def get_best_title(self, track_id: int) -> Optional[str]:
        """
        Return the highest-confidence non-uncertain title for *track_id*.
        Returns None if no confident result exists.
        """
        with self._session() as session:
            row = (
                session.query(OCRRecord)
                .filter(
                    OCRRecord.track_id == track_id,
                    OCRRecord.uncertain == False,  # noqa: E712
                )
                .order_by(OCRRecord.confidence.desc())
                .first()
            )
            if row is None:
                return None
            title = row.book_title
        return title

    def get_all_identified_books(self) -> List[OCRRecord]:
        """Return the most recent confident result per track_id."""
        with self._session() as session:
            # SQLite subquery: latest frame per track with uncertain=False
            results = (
                session.query(OCRRecord)
                .filter(OCRRecord.uncertain == False)  # noqa: E712
                .order_by(
                    OCRRecord.track_id,
                    OCRRecord.confidence.desc(),
                )
                .all()
            )
            session.expunge_all()
        # De-duplicate: keep highest-confidence per track_id
        seen: dict = {}
        for rec in results:
            if rec.track_id not in seen:
                seen[rec.track_id] = rec
        return list(seen.values())

    def count(self) -> int:
        """Return total OCR records stored."""
        with self._session() as session:
            return session.query(OCRRecord).count()

    def clear(self) -> None:
        """Delete ALL OCR records (useful for testing / fresh sessions)."""
        with self._session() as session:
            n = session.query(OCRRecord).delete()
            session.commit()
        log.info(f"Cleared {n} OCR record(s) from SQLite.")

    # ── Helpers ───────────────────────────────────────────────────────────
    @contextmanager
    def _session(self):
        with self._db.get_session() as s:
            yield s
