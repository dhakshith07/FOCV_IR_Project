"""
database/ — SQLite data access layer.

Submodules:
    db_manager           Connection pool + CRUD helpers
    models               SQLAlchemy ORM table definitions
    tracking_repository  Phase 4 — BookTrackRecord persistence
    ocr_repository       Phase 5 — OCRRecord persistence
    migrations/          SQL schema migration scripts
"""

from database.db_manager import DatabaseManager
from database.tracking_repository import TrackingRepository
from database.ocr_repository import OCRRepository

__all__ = ["DatabaseManager", "TrackingRepository", "OCRRepository"]

