"""
workflow/usage_logger.py
Records object interaction events to the database.
Status: STUB — Implementation in Phase 4.
"""


class UsageLogger:
    """Logs interaction events for tracked desk objects."""

    STATUS = "NOT_STARTED"

    def __init__(self, db_manager=None):
        self.db = db_manager

    def log_event(self, track_id: int, event_type: str, duration_s: float = 0.0):
        """Record a usage event for a tracked object."""
        raise NotImplementedError("UsageLogger: Phase 4 not yet implemented.")
