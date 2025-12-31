"""
Database connection management for the RDT Trading System.
"""

import os
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .models import Base


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, db_url: str = None):
        if db_url is None:
            # Default to SQLite in the data directory
            data_dir = Path(__file__).parent.parent
            db_path = data_dir / "rdt_trading.db"
            db_url = f"sqlite:///{db_path}"

        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(self.engine)

    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# Global database manager instance
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_database():
    """Initialize the database (create tables)."""
    manager = get_db_manager()
    manager.create_tables()
