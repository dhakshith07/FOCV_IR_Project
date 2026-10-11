"""
database/db_manager.py
SQLite connection, engine creation, and session factory.
"""

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base


class DatabaseManager:
    """
    Manages the SQLAlchemy engine and session factory.

    Usage:
        db = DatabaseManager("data/workspace.db")
        db.init_db()
        with db.get_session() as session:
            session.add(SomeModel(...))
            session.commit()
    """

    def __init__(self, db_path: str = "data/workspace.db") -> None:
        # Special case: ":memory:" → in-memory SQLite (used in tests)
        if str(db_path) == ":memory:":
            url = "sqlite:///:memory:"
            self._engine: Engine = create_engine(
                url, echo=False,
                connect_args={"check_same_thread": False},
            )
        else:
            db_path = Path(db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{db_path.as_posix()}"
            self._engine = create_engine(
                url, echo=False,
                connect_args={"check_same_thread": False},
            )
        self._Session = sessionmaker(bind=self._engine, autoflush=True, autocommit=False)

    def init_db(self) -> None:
        """Create all tables if they do not exist."""
        Base.metadata.create_all(self._engine)

    def get_session(self) -> Generator[Session, None, None]:
        """Context manager that yields a database session."""
        session = self._Session()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @property
    def engine(self) -> Engine:
        return self._engine
