"""
workflow/workflow_analyzer.py
==============================
Phase 7 - Core Workflow Analysis engine.

Responsibilities
----------------
- Record book interaction events to SQLite (via UsageAnalyzer)
- Build and maintain an in-memory transition count matrix
- Compute all six priority metrics from raw event history
- Predict the next book based on current transition probabilities
- Expose the transition matrix as a normalised pandas DataFrame

Six priority metrics
--------------------
  1. usage_frequency   - total interaction count (all time)
  2. recent_usage      - exponentially-decayed count (recency-weighted)
  3. usage_duration    - total seconds of contact
  4. last_used_time    - seconds since last interaction (lower = more recent)
  5. sequential_score  - transition probability from current book to this one
  6. context_score     - affinity for the current session context

Transition matrix
-----------------
Matrix[A][B] = P(user picks B | just finished with A)
              = count(A->B) / sum_over_all_k(count(A->k))

The diagonal is always 1.0 (self-reference anchor).

Usage
-----
    wa = WorkflowAnalyzer(db_manager)
    wa.record_event(book_id=1, book_title="ML", interaction="pickup")
    wa.record_event(book_id=2, book_title="Maths", interaction="pickup")

    matrix_df = wa.get_transition_matrix()
    metrics   = wa.compute_book_metrics()
    next_book = wa.predict_next_book(current_book_id=1)
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
#  Data containers
# ---------------------------------------------------------------------------

class InteractionEvent:
    """In-memory representation of one book interaction."""
    __slots__ = ("book_id", "book_title", "interaction", "position_x",
                 "position_y", "context", "session_tag", "duration_s", "timestamp")

    def __init__(
        self,
        book_id:    int,
        book_title: str,
        interaction: str,
        position_x:  float = 0.0,
        position_y:  float = 0.0,
        context:     str   = "",
        session_tag: str   = "",
        duration_s:  float = 0.0,
        timestamp:   Optional[datetime] = None,
    ) -> None:
        self.book_id     = book_id
        self.book_title  = book_title
        self.interaction = interaction
        self.position_x  = position_x
        self.position_y  = position_y
        self.context     = context
        self.session_tag = session_tag
        self.duration_s  = duration_s
        self.timestamp   = timestamp or datetime.now(timezone.utc)

    def __repr__(self) -> str:
        return (
            f"InteractionEvent(book={self.book_title!r}, "
            f"action={self.interaction}, t={self.timestamp.strftime('%H:%M:%S')})"
        )


class BookMetrics:
    """All computed metrics for a single book."""
    __slots__ = (
        "book_id", "book_title",
        "usage_frequency", "recent_usage", "usage_duration",
        "last_used_time", "sequential_score", "context_score",
        "priority_score",
    )

    def __init__(
        self,
        book_id:         int,
        book_title:      str,
        usage_frequency: int   = 0,
        recent_usage:    float = 0.0,
        usage_duration:  float = 0.0,
        last_used_time:  float = float("inf"),
        sequential_score: float = 0.0,
        context_score:   float = 0.0,
    ) -> None:
        self.book_id          = book_id
        self.book_title       = book_title
        self.usage_frequency  = usage_frequency
        self.recent_usage     = recent_usage
        self.usage_duration   = usage_duration
        self.last_used_time   = last_used_time
        self.sequential_score = sequential_score
        self.context_score    = context_score
        self.priority_score   = 0.0

    def to_dict(self) -> dict:
        return {
            "book_id":          self.book_id,
            "book_title":       self.book_title,
            "usage_frequency":  self.usage_frequency,
            "recent_usage":     round(self.recent_usage, 4),
            "usage_duration_s": round(self.usage_duration, 2),
            "last_used_s_ago":  round(self.last_used_time, 1),
            "sequential_score": round(self.sequential_score, 4),
            "context_score":    round(self.context_score, 4),
            "priority_score":   round(self.priority_score, 4),
        }

    def __repr__(self) -> str:
        return (
            f"BookMetrics(id={self.book_id}, title={self.book_title!r}, "
            f"freq={self.usage_frequency}, priority={self.priority_score:.3f})"
        )


# ---------------------------------------------------------------------------
#  WorkflowAnalyzer
# ---------------------------------------------------------------------------

class WorkflowAnalyzer:
    """
    Core workflow analysis engine.

    Parameters
    ----------
    db_manager       : DatabaseManager instance. If None, runs in-memory only.
    decay_factor     : Exponential decay per hour for recent_usage (0-1).
    metric_weights   : Dict weighting the six metrics into priority_score.
    """

    # Default metric weights  (must sum to 1.0)
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "usage_frequency":  0.20,
        "recent_usage":     0.25,
        "usage_duration":   0.15,
        "last_used_time":   0.20,   # inverted: lower time → higher score
        "sequential_score": 0.15,
        "context_score":    0.05,
    }

    def __init__(
        self,
        db_manager=None,
        decay_factor:   float = 0.85,
        metric_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._db            = db_manager
        self._decay_factor  = decay_factor
        self._weights       = metric_weights or dict(self.DEFAULT_WEIGHTS)

        # In-memory event log
        self._events: List[InteractionEvent] = []

        # Transition count matrix:  {from_id: {to_id: count}}
        self._transitions: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # Book registry:  {book_id: book_title}
        self._books: Dict[int, str] = {}

        # State tracking
        self._current_book_id:    Optional[int] = None
        self._current_context:    str           = ""
        self._session_start:      datetime      = datetime.now(timezone.utc)

        # Load historical data from DB if available
        self._load_from_db()

        log.info("WorkflowAnalyzer initialised.")

    # ------------------------------------------------------------------
    # Public API — recording & control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all in-memory events, transitions, and book registry."""
        self._events.clear()
        self._transitions.clear()
        self._books.clear()
        self._current_book_id = None
        self._current_context = ""

    def record_event(
        self,
        book_id:    int,
        book_title: str,
        interaction: str,
        position_x:  float = 0.0,
        position_y:  float = 0.0,
        context:     str   = "",
        session_tag: str   = "",
        duration_s:  float = 0.0,
        timestamp:   Optional[datetime] = None,
    ) -> InteractionEvent:
        """
        Record one book interaction event.

        Updates the transition matrix when the user switches to a different
        book (pickup/open/reference interactions).

        Parameters
        ----------
        book_id     : unique integer ID for this book
        book_title  : human-readable title
        interaction : one of pickup | put_down | gaze | reference | open | close
        position_x  : x-coordinate on desk (pixels or normalised)
        position_y  : y-coordinate on desk
        context     : current session context string
        session_tag : session grouping tag
        duration_s  : duration of the interaction in seconds
        timestamp   : UTC timestamp (defaults to now)
        """
        event = InteractionEvent(
            book_id     = book_id,
            book_title  = book_title,
            interaction = interaction,
            position_x  = position_x,
            position_y  = position_y,
            context     = context or self._current_context,
            session_tag = session_tag,
            duration_s  = duration_s,
            timestamp   = timestamp,
        )

        self._events.append(event)
        self._books[book_id] = book_title

        if context:
            self._current_context = context

        # Update transition matrix on active interactions
        if interaction in ("pickup", "open", "reference", "gaze"):
            if (
                self._current_book_id is not None
                and self._current_book_id != book_id
            ):
                self._transitions[self._current_book_id][book_id] += 1
                log.debug(
                    f"Transition: {self._books.get(self._current_book_id, '?')} "
                    f"-> {book_title}"
                )
            self._current_book_id = book_id

        # Persist to DB
        self._persist_event(event)

        log.debug(f"Event recorded: {event}")
        return event

    def set_context(self, context: str) -> None:
        """Update the current session context."""
        self._current_context = context
        log.info(f"Context set to: {context!r}")

    # ------------------------------------------------------------------
    # Public API — metrics
    # ------------------------------------------------------------------

    def compute_book_metrics(
        self,
        current_book_id: Optional[int] = None,
        context:         Optional[str] = None,
    ) -> List[BookMetrics]:
        """
        Compute all six metrics for every known book.

        Parameters
        ----------
        current_book_id : book currently in use (for sequential_score)
        context         : current session context (for context_score)

        Returns
        -------
        List[BookMetrics] sorted by priority_score descending.
        """
        if not self._books:
            return []

        ctx     = context or self._current_context
        cur_id  = current_book_id or self._current_book_id
        now     = datetime.now(timezone.utc)

        # ── Per-book aggregation ──────────────────────────────────────
        freq:      Dict[int, int]   = defaultdict(int)
        total_dur: Dict[int, float] = defaultdict(float)
        last_seen: Dict[int, float] = {}   # seconds ago

        for ev in self._events:
            freq[ev.book_id]      += 1
            total_dur[ev.book_id] += ev.duration_s
            age = (now - ev.timestamp).total_seconds()
            if ev.book_id not in last_seen or age < last_seen[ev.book_id]:
                last_seen[ev.book_id] = age

        # ── Recent usage (exponential decay by hour) ──────────────────
        recent: Dict[int, float] = defaultdict(float)
        for ev in self._events:
            age_h = (now - ev.timestamp).total_seconds() / 3600.0
            recent[ev.book_id] += self._decay_factor ** age_h

        # ── Sequential scores from current book ───────────────────────
        seq_raw: Dict[int, int] = {}
        if cur_id is not None and cur_id in self._transitions:
            seq_raw = dict(self._transitions[cur_id])
        seq_total = max(sum(seq_raw.values()), 1)

        # ── Context scores ────────────────────────────────────────────
        ctx_scores = self._compute_context_scores(ctx)

        # ── Assemble BookMetrics ──────────────────────────────────────
        max_freq    = max(freq.values(),     default=1)
        max_recent  = max(recent.values(),   default=1.0)
        max_dur     = max(total_dur.values(), default=1.0)
        max_age     = max(last_seen.values(), default=1.0)

        metrics_list = []
        for bid, title in self._books.items():
            ls  = last_seen.get(bid, float("inf"))

            m = BookMetrics(
                book_id          = bid,
                book_title       = title,
                usage_frequency  = freq.get(bid, 0),
                recent_usage     = recent.get(bid, 0.0),
                usage_duration   = total_dur.get(bid, 0.0),
                last_used_time   = ls,
                sequential_score = seq_raw.get(bid, 0) / seq_total,
                context_score    = ctx_scores.get(bid, 0.0),
            )

            # Normalise to [0,1] for weighted sum
            norm_freq  = freq.get(bid, 0)       / max_freq
            norm_rec   = recent.get(bid, 0.0)   / max_recent
            norm_dur   = total_dur.get(bid, 0.0)/ max_dur
            norm_age   = 1.0 - min(ls / max(max_age, 1), 1.0)  # inverted

            m.priority_score = (
                self._weights["usage_frequency"]  * norm_freq
                + self._weights["recent_usage"]   * norm_rec
                + self._weights["usage_duration"] * norm_dur
                + self._weights["last_used_time"] * norm_age
                + self._weights["sequential_score"] * m.sequential_score
                + self._weights["context_score"]  * m.context_score
            )
            metrics_list.append(m)

        return sorted(metrics_list, key=lambda m: m.priority_score, reverse=True)

    # ------------------------------------------------------------------
    # Public API — transition matrix
    # ------------------------------------------------------------------

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Return the normalised transition probability matrix as a DataFrame.

        Matrix[A][B] = P(go to B | currently at A).
        Diagonal = 1.0 (self-reference).
        Returns an empty DataFrame if fewer than 2 books are known.
        """
        if len(self._books) < 2:
            return pd.DataFrame()

        book_ids = sorted(self._books.keys())
        titles   = [self._books[bid] for bid in book_ids]
        n        = len(book_ids)
        idx_map  = {bid: i for i, bid in enumerate(book_ids)}

        mat = np.zeros((n, n), dtype=float)
        np.fill_diagonal(mat, 1.0)   # self-reference anchor

        for from_id, to_counts in self._transitions.items():
            if from_id not in idx_map:
                continue
            fi        = idx_map[from_id]
            row_total = sum(to_counts.values())
            if row_total == 0:
                continue
            for to_id, cnt in to_counts.items():
                if to_id in idx_map:
                    mat[fi][idx_map[to_id]] = cnt / row_total

        df = pd.DataFrame(mat, index=titles, columns=titles)
        return df

    def get_transition_counts(self) -> pd.DataFrame:
        """Return the raw (unnormalised) transition count matrix."""
        if len(self._books) < 2:
            return pd.DataFrame()

        book_ids = sorted(self._books.keys())
        titles   = [self._books[bid] for bid in book_ids]
        n        = len(book_ids)
        idx_map  = {bid: i for i, bid in enumerate(book_ids)}

        mat = np.zeros((n, n), dtype=int)
        for from_id, to_counts in self._transitions.items():
            if from_id not in idx_map:
                continue
            fi = idx_map[from_id]
            for to_id, cnt in to_counts.items():
                if to_id in idx_map:
                    mat[fi][idx_map[to_id]] = cnt

        return pd.DataFrame(mat, index=titles, columns=titles)

    # ------------------------------------------------------------------
    # Public API — prediction
    # ------------------------------------------------------------------

    def predict_next_book(
        self,
        current_book_id: Optional[int] = None,
        top_k: int = 3,
    ) -> List[Tuple[int, str, float]]:
        """
        Predict the most likely next book(s) given the current book.

        Returns
        -------
        List of (book_id, book_title, probability) sorted descending.
        Returns an empty list if no transitions have been recorded.
        """
        cur = current_book_id or self._current_book_id
        if cur is None or cur not in self._transitions:
            # Fall back to most frequently used books
            freq: Dict[int, int] = defaultdict(int)
            for ev in self._events:
                freq[ev.book_id] += 1
            total = max(sum(freq.values()), 1)
            ranked = sorted(freq.items(), key=lambda x: -x[1])[:top_k]
            return [
                (bid, self._books.get(bid, f"Book {bid}"), cnt / total)
                for bid, cnt in ranked
                if bid != cur
            ]

        to_counts = self._transitions[cur]
        total     = max(sum(to_counts.values()), 1)
        ranked    = sorted(to_counts.items(), key=lambda x: -x[1])[:top_k]
        return [
            (to_id, self._books.get(to_id, f"Book {to_id}"), cnt / total)
            for to_id, cnt in ranked
        ]

    # ------------------------------------------------------------------
    # Public API — introspection
    # ------------------------------------------------------------------

    def get_events_df(self) -> pd.DataFrame:
        """Return all recorded events as a DataFrame."""
        if not self._events:
            return pd.DataFrame(columns=[
                "timestamp", "book_id", "book_title",
                "interaction", "duration_s", "context",
            ])
        rows = [
            {
                "timestamp":   ev.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "book_id":     ev.book_id,
                "book_title":  ev.book_title,
                "interaction": ev.interaction,
                "duration_s":  round(ev.duration_s, 2),
                "context":     ev.context,
                "position_x":  round(ev.position_x, 1),
                "position_y":  round(ev.position_y, 1),
            }
            for ev in self._events
        ]
        return pd.DataFrame(rows)

    def get_timeline(self) -> List[str]:
        """Return the chronological book-title access sequence."""
        seen = set()
        result = []
        for ev in self._events:
            if ev.interaction in ("pickup", "open", "reference"):
                result.append(ev.book_title)
        return result

    def summary(self) -> dict:
        return {
            "total_events":    len(self._events),
            "unique_books":    len(self._books),
            "transitions":     sum(
                sum(v.values()) for v in self._transitions.values()
            ),
            "current_book":    self._books.get(self._current_book_id, "None"),
            "current_context": self._current_context,
            "session_start":   self._session_start.isoformat(),
        }

    @property
    def books(self) -> Dict[int, str]:
        return dict(self._books)

    @property
    def current_book_id(self) -> Optional[int]:
        return self._current_book_id

    @property
    def current_context(self) -> str:
        return self._current_context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_context_scores(self, context: str) -> Dict[int, float]:
        """
        Compute context affinity scores per book.
        Uses co-occurrence: books used more often in *context* get higher score.
        """
        if not context:
            return {}

        ctx_freq: Dict[int, int] = defaultdict(int)
        for ev in self._events:
            if ev.context == context:
                ctx_freq[ev.book_id] += 1

        total = max(sum(ctx_freq.values()), 1)
        return {bid: cnt / total for bid, cnt in ctx_freq.items()}

    def _persist_event(self, event: InteractionEvent) -> None:
        """Write event to SQLite (silently skips if no DB)."""
        if self._db is None:
            return
        try:
            from database.models import WorkflowEvent
            with self._db.get_session() as session:
                row = WorkflowEvent(
                    book_id     = event.book_id,
                    book_title  = event.book_title,
                    interaction = event.interaction,
                    position_x  = event.position_x,
                    position_y  = event.position_y,
                    context     = event.context,
                    session_tag = event.session_tag,
                    duration_s  = event.duration_s,
                    timestamp   = event.timestamp,
                )
                session.add(row)
                session.commit()
        except Exception as exc:
            log.warning(f"Failed to persist event: {exc}")

    def _load_from_db(self) -> None:
        """Replay historical events from SQLite on startup."""
        if self._db is None:
            return
        try:
            from database.models import WorkflowEvent
            with self._db.get_session() as session:
                rows = session.query(WorkflowEvent).order_by(
                    WorkflowEvent.timestamp
                ).all()
                for row in rows:
                    ev = InteractionEvent(
                        book_id     = row.book_id,
                        book_title  = row.book_title,
                        interaction = row.interaction,
                        position_x  = row.position_x or 0.0,
                        position_y  = row.position_y or 0.0,
                        context     = row.context or "",
                        session_tag = row.session_tag or "",
                        duration_s  = row.duration_s or 0.0,
                        timestamp   = row.timestamp,
                    )
                    self._events.append(ev)
                    self._books[row.book_id] = row.book_title

                    if ev.interaction in ("pickup", "open", "reference", "gaze"):
                        if (
                            self._current_book_id is not None
                            and self._current_book_id != ev.book_id
                        ):
                            self._transitions[self._current_book_id][ev.book_id] += 1
                        self._current_book_id = ev.book_id

                if rows:
                    log.info(f"Loaded {len(rows)} historical events from DB.")
        except Exception as exc:
            log.warning(f"Could not load history from DB: {exc}")
