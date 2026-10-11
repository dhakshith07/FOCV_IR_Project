"""
workflow/context_detector.py
=============================
Phase 7 - Session context detection.

Detects the current study context from the sequence of books the user
has been interacting with.  Context labels are data-driven — the system
builds a keyword → context mapping from a configurable vocabulary and
matches it against observed book titles.

Contexts
--------
  studying        - general learning session
  exam_prep       - multiple subjects in rapid succession
  research        - deep focus on one or two books for long durations
  leisure         - relaxed, long gaps, low frequency
  unknown         - insufficient data to classify

The detector also tracks:
  - subject affinity:  which subjects dominate the current session
  - focus score:       single-book concentration vs. multi-book scatter
  - intensity:         events per minute (activity rate)

Usage
-----
    cd = ContextDetector(workflow_analyzer)
    ctx = cd.detect_context()
    print(ctx.label)          # "exam_prep"
    print(ctx.focus_score)    # 0.72
    print(ctx.subjects)       # ["machine_learning", "mathematics"]
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from workflow.workflow_analyzer import WorkflowAnalyzer
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
#  Subject keyword vocabulary  (extend as needed)
# ---------------------------------------------------------------------------

SUBJECT_KEYWORDS: Dict[str, List[str]] = {
    "machine_learning": [
        "machine learning", "ml", "deep learning", "neural", "ai",
        "artificial intelligence", "sklearn", "tensorflow", "pytorch",
        "pattern recognition", "data science",
    ],
    "mathematics": [
        "mathematics", "math", "calculus", "algebra", "statistics",
        "probability", "linear algebra", "discrete", "analysis", "maths",
    ],
    "databases": [
        "dbms", "database", "sql", "nosql", "mongodb", "relational",
        "data management", "query",
    ],
    "computer_science": [
        "computer science", "algorithms", "data structures", "os",
        "operating system", "networking", "computer networks", "compiler",
    ],
    "physics": [
        "physics", "mechanics", "thermodynamics", "electromagnetism",
        "quantum", "optics",
    ],
    "chemistry": [
        "chemistry", "organic", "inorganic", "biochemistry", "molecules",
    ],
    "programming": [
        "programming", "python", "java", "c++", "software", "coding",
        "development", "debugging",
    ],
}

# Context classification rules  (all thresholds are data-driven)
_EXAM_PREP_MIN_BOOKS        = 3     # at least 3 different books accessed
_EXAM_PREP_MAX_AVG_DUR      = 120   # average <2 min per book
_RESEARCH_MIN_AVG_DUR       = 300   # average >5 min per book
_RESEARCH_MAX_BOOKS         = 2     # concentrated on ≤2 books
_LEISURE_MIN_GAP_S          = 600   # >10 min gaps between interactions


# ---------------------------------------------------------------------------
#  Context result
# ---------------------------------------------------------------------------

@dataclass
class ContextResult:
    """Detected session context with supporting evidence."""
    label:       str                      # context label
    confidence:  float                    # 0–1 confidence
    subjects:    List[str]                # detected subject areas
    focus_score: float                    # 1.0 = single-book focus
    intensity:   float                    # events/minute
    top_books:   List[Tuple[str, int]]    # [(title, count)]
    session_duration_s: float
    evidence:    Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label":              self.label,
            "confidence":         round(self.confidence, 3),
            "subjects":           self.subjects,
            "focus_score":        round(self.focus_score, 3),
            "intensity":          round(self.intensity, 3),
            "top_books":          self.top_books,
            "session_duration_s": round(self.session_duration_s, 1),
            "evidence":           {k: round(v, 3) for k, v in self.evidence.items()},
        }

    def __repr__(self) -> str:
        return (
            f"ContextResult(label={self.label!r}, "
            f"confidence={self.confidence:.2f}, "
            f"subjects={self.subjects})"
        )


# ---------------------------------------------------------------------------
#  ContextDetector
# ---------------------------------------------------------------------------

class ContextDetector:
    """
    Data-driven session context classifier.

    Parameters
    ----------
    workflow_analyzer : shared WorkflowAnalyzer instance
    subject_keywords  : vocabulary override (defaults to SUBJECT_KEYWORDS)
    window_events     : number of most-recent events to consider (0 = all)
    """

    def __init__(
        self,
        workflow_analyzer: WorkflowAnalyzer,
        subject_keywords:  Optional[Dict[str, List[str]]] = None,
        window_events:     int = 0,
    ) -> None:
        self._wa      = workflow_analyzer
        self._vocab   = subject_keywords or SUBJECT_KEYWORDS
        self._window  = window_events

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_context(self) -> ContextResult:
        """
        Detect the current session context from recorded events.

        Returns a ContextResult with label, confidence, and evidence.
        Returns label="unknown" when fewer than 2 events are available.
        """
        events = self._wa._events
        if self._window > 0:
            events = events[-self._window:]

        if len(events) < 2:
            return ContextResult(
                label="unknown", confidence=0.0, subjects=[],
                focus_score=0.0, intensity=0.0, top_books=[],
                session_duration_s=0.0,
                evidence={"reason": 0.0},
            )

        # ── Feature extraction ────────────────────────────────────────
        book_counter   = Counter(ev.book_title for ev in events)
        unique_books   = len(book_counter)
        total_events   = len(events)
        top_books      = book_counter.most_common(5)

        # Focus score: entropy-based (1.0 = all events on one book)
        focus_score = self._focus_score(book_counter)

        # Duration features
        avg_dur  = sum(ev.duration_s for ev in events) / total_events
        total_s  = (events[-1].timestamp - events[0].timestamp).total_seconds()
        total_s  = max(total_s, 1.0)

        # Intensity: events per minute
        intensity = (total_events / total_s) * 60

        # Gap analysis
        gaps = [
            (events[i+1].timestamp - events[i].timestamp).total_seconds()
            for i in range(len(events) - 1)
        ]
        avg_gap = sum(gaps) / max(len(gaps), 1)

        # Subject detection
        subjects = self._detect_subjects(list(self._wa.books.values()))

        # ── Context classification ────────────────────────────────────
        label, confidence, evidence = self._classify(
            unique_books = unique_books,
            avg_dur      = avg_dur,
            avg_gap      = avg_gap,
            focus_score  = focus_score,
            intensity    = intensity,
        )

        return ContextResult(
            label              = label,
            confidence         = confidence,
            subjects           = subjects,
            focus_score        = focus_score,
            intensity          = intensity,
            top_books          = top_books,
            session_duration_s = total_s,
            evidence           = evidence,
        )

    def detect_subject_affinity(self) -> Dict[str, float]:
        """
        Return a normalised affinity score per subject area based on
        the titles of books that have been accessed.
        """
        if not self._wa.books:
            return {}

        raw: Dict[str, int] = defaultdict(int)
        for title in self._wa.books.values():
            for subj, keywords in self._vocab.items():
                title_lower = title.lower()
                for kw in keywords:
                    if kw in title_lower:
                        raw[subj] += 1
                        break

        total = max(sum(raw.values()), 1)
        return {subj: cnt / total for subj, cnt in sorted(raw.items(), key=lambda x: -x[1])}

    def get_book_relationship_score(self, book_id_a: int, book_id_b: int) -> float:
        """
        Return a [0, 1] relationship score between two books.

        Score is based on:
          - Transition probability A→B and B→A (normalised)
          - Subject overlap
        """
        trans = self._wa._transitions
        a_to_b = trans.get(book_id_a, {}).get(book_id_b, 0)
        b_to_a = trans.get(book_id_b, {}).get(book_id_a, 0)

        a_total = max(sum(trans.get(book_id_a, {}).values()), 1)
        b_total = max(sum(trans.get(book_id_b, {}).values()), 1)

        p_ab = a_to_b / a_total
        p_ba = b_to_a / b_total

        title_a = self._wa.books.get(book_id_a, "")
        title_b = self._wa.books.get(book_id_b, "")
        subj_overlap = self._subject_overlap(title_a, title_b)

        score = 0.4 * p_ab + 0.4 * p_ba + 0.2 * subj_overlap
        return round(min(score, 1.0), 4)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _focus_score(counter: Counter) -> float:
        """
        Normalised inverse entropy: 1.0 = perfect focus on one book,
        0.0 = perfectly uniform distribution across all books.
        """
        total = sum(counter.values())
        if total == 0 or len(counter) == 1:
            return 1.0
        entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in counter.values() if c > 0
        )
        max_entropy = math.log2(len(counter))
        return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0

    def _detect_subjects(self, titles: List[str]) -> List[str]:
        """Return list of detected subject areas from book titles."""
        found = []
        for subj, keywords in self._vocab.items():
            for title in titles:
                tl = title.lower()
                if any(kw in tl for kw in keywords):
                    if subj not in found:
                        found.append(subj)
                    break
        return found

    def _subject_overlap(self, title_a: str, title_b: str) -> float:
        """Binary overlap: 1.0 if both books share at least one subject."""
        subj_a = set(self._detect_subjects([title_a]))
        subj_b = set(self._detect_subjects([title_b]))
        if not subj_a or not subj_b:
            return 0.0
        return 1.0 if subj_a & subj_b else 0.0

    def _classify(
        self,
        unique_books: int,
        avg_dur:      float,
        avg_gap:      float,
        focus_score:  float,
        intensity:    float,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Rule-based context classifier.
        Returns (label, confidence, evidence_dict).
        """
        evidence = {
            "unique_books":  float(unique_books),
            "avg_dur_s":     avg_dur,
            "avg_gap_s":     avg_gap,
            "focus_score":   focus_score,
            "intensity_epm": intensity,
        }

        # Exam prep: many books, short durations, high activity
        if (unique_books >= _EXAM_PREP_MIN_BOOKS
                and avg_dur < _EXAM_PREP_MAX_AVG_DUR):
            conf = min(0.5 + (unique_books - _EXAM_PREP_MIN_BOOKS) * 0.1, 0.95)
            return "exam_prep", conf, evidence

        # Research: concentrated on 1-2 books, long durations
        if (unique_books <= _RESEARCH_MAX_BOOKS
                and avg_dur >= _RESEARCH_MIN_AVG_DUR):
            conf = min(0.5 + (avg_dur - _RESEARCH_MIN_AVG_DUR) / 300 * 0.1, 0.95)
            return "research", conf, evidence

        # Leisure: long gaps, low intensity
        if avg_gap >= _LEISURE_MIN_GAP_S:
            conf = min(0.5 + (avg_gap - _LEISURE_MIN_GAP_S) / 600 * 0.1, 0.9)
            return "leisure", conf, evidence

        # Default: studying
        conf = 0.5 + focus_score * 0.3
        return "studying", round(min(conf, 0.9), 3), evidence
