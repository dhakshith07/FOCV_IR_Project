"""
database/models.py
SQLAlchemy ORM table definitions for the Workspace Organizer.
"""

from datetime import datetime
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class TrackedObject(Base):
    """Every unique physical object detected on the desk."""
    __tablename__ = "objects"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    track_id    = Column(Integer, nullable=False, index=True)
    label       = Column(String(256))          # TrOCR-extracted title
    class_name  = Column(String(64))           # YOLO11 class
    first_seen  = Column(DateTime, default=datetime.utcnow)
    last_seen   = Column(DateTime, onupdate=datetime.utcnow)
    is_active   = Column(Boolean, default=True)

    usage_events = relationship("UsageEvent", back_populates="obj")
    priority_scores = relationship("PriorityScore", back_populates="obj")


class Session(Base):
    """A single study session."""
    __tablename__ = "sessions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    start_time      = Column(DateTime, default=datetime.utcnow)
    end_time        = Column(DateTime, nullable=True)
    session_type    = Column(String(64), default="unknown")
    total_objects   = Column(Integer, default=0)

    usage_events = relationship("UsageEvent", back_populates="session")
    snapshots    = relationship("WorkspaceSnapshot", back_populates="session")
    robot_actions = relationship("RobotAction", back_populates="session")


class UsageEvent(Base):
    """Every interaction with a tracked object."""
    __tablename__ = "usage_events"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    object_id   = Column(Integer, ForeignKey("objects.id"), nullable=False)
    session_id  = Column(Integer, ForeignKey("sessions.id"), nullable=True)
    event_type  = Column(String(32))           # pickup | put_down | gaze | reference
    timestamp   = Column(DateTime, default=datetime.utcnow)
    duration_s  = Column(Float, nullable=True)

    obj     = relationship("TrackedObject", back_populates="usage_events")
    session = relationship("Session", back_populates="usage_events")


class WorkspaceSnapshot(Base):
    """Point-in-time snapshot of the desk layout."""
    __tablename__ = "workspace_snapshots"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    session_id  = Column(Integer, ForeignKey("sessions.id"), nullable=True)
    timestamp   = Column(DateTime, default=datetime.utcnow)
    layout_json = Column(Text)                 # JSON: {obj_id: {x,y,zone}}
    source      = Column(String(32))           # detected | ga_proposed | applied

    session = relationship("Session", back_populates="snapshots")


class RobotAction(Base):
    """Log of every virtual robot arm action."""
    __tablename__ = "robot_actions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    session_id      = Column(Integer, ForeignKey("sessions.id"), nullable=True)
    object_id       = Column(Integer, ForeignKey("objects.id"), nullable=True)
    action_type     = Column(String(32))       # move | pick | place
    start_pos_json  = Column(Text)             # {x, y, z}
    end_pos_json    = Column(Text)
    joint_path_json = Column(Text)             # Full IK joint sequence
    duration_ms     = Column(Integer)
    timestamp       = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="robot_actions")


class DQNCheckpoint(Base):
    """Saved DQN model checkpoints."""
    __tablename__ = "dqn_checkpoints"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    episode     = Column(Integer)
    reward      = Column(Float)
    epsilon     = Column(Float)
    saved_at    = Column(DateTime, default=datetime.utcnow)
    model_path  = Column(String(512))


class PriorityScore(Base):
    """Computed priority scores per object per session."""
    __tablename__ = "priority_scores"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    object_id       = Column(Integer, ForeignKey("objects.id"), nullable=False)
    session_id      = Column(Integer, ForeignKey("sessions.id"), nullable=True)
    score           = Column(Float)
    zone_affinity   = Column(String(32))       # active | secondary | storage
    computed_at     = Column(DateTime, default=datetime.utcnow)

    obj = relationship("TrackedObject", back_populates="priority_scores")


# ── Phase 4 ────────────────────────────────────────────────────────────────────

class BookTrackRecord(Base):
    """
    One ByteTrack observation per (track_id, frame_index).

    Stores the full snapshot produced by ByteTrackTracker.update() so that
    the complete positional history of every tracked book is available in
    SQLite for later analysis (workflow analysis, priority scoring, etc.).
    """
    __tablename__ = "book_track_records"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    track_id    = Column(Integer,  nullable=False,  index=True)
    book_title  = Column(String(512), default="Unknown")
    bbox_x1     = Column(Float,   nullable=False)
    bbox_y1     = Column(Float,   nullable=False)
    bbox_x2     = Column(Float,   nullable=False)
    bbox_y2     = Column(Float,   nullable=False)
    center_x    = Column(Float,   nullable=False)
    center_y    = Column(Float,   nullable=False)
    confidence  = Column(Float,   nullable=False)
    frame_index = Column(Integer, nullable=False, index=True)
    timestamp   = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        # Use __dict__ to avoid DetachedInstanceError when accessed outside a session
        d = self.__dict__
        return (
            f"<BookTrackRecord id={d.get('id', '?')} "
            f"track={d.get('track_id', '?')} "
            f"frame={d.get('frame_index', '?')} "
            f"pos=({d.get('center_x', 0):.0f},{d.get('center_y', 0):.0f})>"
        )


# ── Phase 5 ────────────────────────────────────────────────────────────────────

class OCRRecord(Base):
    """
    One TrOCR read result per (track_id, frame_index).

    Stores raw OCR text, cleaned text, matched catalog entry, confidence
    score, and whether the result was flagged as uncertain.
    """
    __tablename__ = "ocr_records"

    id          = Column(Integer,  primary_key=True, autoincrement=True)
    track_id    = Column(Integer,  nullable=False, index=True)
    frame_index = Column(Integer,  nullable=False, index=True)
    raw_text    = Column(Text,     nullable=False, default="")
    clean_text  = Column(Text,     nullable=False, default="")
    book_id     = Column(Integer,  nullable=True)        # None if uncertain
    book_title  = Column(String(512), nullable=False, default="Book identification uncertain")
    confidence  = Column(Float,    nullable=False, default=0.0)
    match_score = Column(Float,    nullable=False, default=0.0)
    uncertain   = Column(Boolean,  nullable=False, default=True)
    elapsed_ms  = Column(Float,    nullable=False, default=0.0)
    timestamp   = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:
        d = self.__dict__
        return (
            f"<OCRRecord id={d.get('id', '?')} "
            f"track={d.get('track_id', '?')} "
            f"title={d.get('book_title', '?')!r} "
            f"conf={d.get('confidence', 0):.2f}>"
        )



# -- Phase 6 -------------------------------------------------------------------

class SegmentationRecord(Base):
    """
    One SegFormer inference result per video frame.

    Captures workspace coverage, obstacle coverage, segmenter mode,
    and elapsed time for analysis and dashboard display.
    """
    __tablename__ = "segmentation_records"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    frame_index   = Column(Integer,  nullable=False, index=True)
    mode          = Column(String(32),  nullable=False)
    workspace_pct = Column(Float,    nullable=False, default=0.0)
    obstacle_pct  = Column(Float,    nullable=False, default=0.0)
    workspace_px  = Column(Integer,  nullable=False, default=0)
    obstacle_px   = Column(Integer,  nullable=False, default=0)
    frame_width   = Column(Integer,  nullable=False, default=0)
    frame_height  = Column(Integer,  nullable=False, default=0)
    elapsed_ms    = Column(Float,    nullable=False, default=0.0)
    timestamp     = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:
        d = self.__dict__
        m = d.get('mode', '?')
        wi = d.get('frame_index', '?')
        wp = d.get('workspace_pct', 0)
        op = d.get('obstacle_pct', 0)
        return f"<SegmentationRecord frame={wi} mode={m} workspace={wp:.1%} obstacle={op:.1%}>"


# -- Phase 7 -------------------------------------------------------------------

class WorkflowEvent(Base):
    """
    One user-interaction event with a specific book.

    Recorded by UsageAnalyzer whenever the user picks up, puts down,
    gazes at, or references a book.  The sequence of these events drives
    the transition matrix and context detection.

    Fields
    ------
    book_id         : application-level book identifier (not track_id)
    book_title      : human-readable title (may be OCR-derived)
    interaction     : pickup | put_down | gaze | reference | open | close
    position_x/y   : pixel or normalised desk position at event time
    context         : session context tag (studying | exam_prep | research …)
    session_tag     : free-form session grouping string
    duration_s      : seconds the interaction lasted (0 if instantaneous)
    timestamp       : UTC time of event
    """
    __tablename__ = "workflow_events"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    book_id      = Column(Integer,     nullable=False, index=True)
    book_title   = Column(String(512), nullable=False, default="Unknown")
    interaction  = Column(String(32),  nullable=False)           # pickup | put_down …
    position_x   = Column(Float,       nullable=True)
    position_y   = Column(Float,       nullable=True)
    context      = Column(String(128), nullable=True)
    session_tag  = Column(String(128), nullable=True)
    duration_s   = Column(Float,       nullable=False, default=0.0)
    timestamp    = Column(DateTime,    nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<WorkflowEvent id={self.id} book={self.book_title!r} "
            f"action={self.interaction} t={self.timestamp}>"
        )


class TransitionRecord(Base):
    """
    One observed book-to-book sequential transition.

    Written by WorkflowAnalyzer every time the user switches from one
    book to another.  The full transition matrix is recomputed from the
    aggregated count of these records.

    Fields
    ------
    from_book_id / from_book_title : source book
    to_book_id   / to_book_title   : destination book
    count                          : cumulative transition count
    last_seen                      : last time this transition occurred
    """
    __tablename__ = "transition_records"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    from_book_id    = Column(Integer,     nullable=False, index=True)
    from_book_title = Column(String(512), nullable=False, default="Unknown")
    to_book_id      = Column(Integer,     nullable=False, index=True)
    to_book_title   = Column(String(512), nullable=False, default="Unknown")
    count           = Column(Integer,     nullable=False, default=1)
    last_seen       = Column(DateTime,    nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<TransitionRecord {self.from_book_title!r} -> "
            f"{self.to_book_title!r} x{self.count}>"
        )
