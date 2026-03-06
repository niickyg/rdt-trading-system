"""
Database connection management for the RDT Trading System.

Supports both PostgreSQL (production) and SQLite (development/testing).
Includes connection pooling, health checks, async support, and TimescaleDB detection.

TimescaleDB Support:
    The system automatically detects if TimescaleDB is available and enables
    optimized time-series queries when it is. Use the `is_timescale` property
    to check if TimescaleDB features are available.
"""

import os
import asyncio
from pathlib import Path
from contextlib import contextmanager, asynccontextmanager
from typing import Generator, Optional, Tuple, AsyncGenerator, Dict, Any

from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from sqlalchemy.engine import Engine

# Async support - optional dependency
try:
    from sqlalchemy.ext.asyncio import (
        create_async_engine,
        AsyncSession,
        async_sessionmaker,
        AsyncEngine,
    )
    ASYNC_SUPPORT = True
except ImportError:
    ASYNC_SUPPORT = False
    AsyncSession = None
    AsyncEngine = None

from .models import Base

from loguru import logger


# =============================================================================
# Retry Logic and Error Handling
# =============================================================================

import threading
import time
import random
from functools import wraps
from sqlalchemy.exc import OperationalError, InterfaceError


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential: bool = True,
    jitter: float = 0.1,
    retryable_exceptions: tuple = (OperationalError, InterfaceError),
):
    """
    Retry decorator with exponential backoff and jitter.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential: Use exponential backoff if True, linear if False
        jitter: Random jitter factor (0.0 to 1.0) to prevent thundering herd
        retryable_exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        if exponential:
                            delay = min(base_delay * (2 ** attempt), max_delay)
                        else:
                            delay = min(base_delay * (attempt + 1), max_delay)
                        # Add jitter to prevent thundering herd
                        delay += random.uniform(0, delay * jitter)
                        logger.warning(
                            f"Database operation failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Database operation failed after {max_attempts} attempts: {e}"
                        )
            raise last_exception
        return wrapper
    return decorator


async def with_retry_async(
    func,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (OperationalError, InterfaceError),
):
    """
    Async retry logic with exponential backoff.

    Args:
        func: Async function to call
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        retryable_exceptions: Tuple of exception types to retry on

    Returns:
        Result of the function call
    """
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                delay += random.uniform(0, delay * 0.1)
                logger.warning(
                    f"Async database operation failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"Async database operation failed after {max_attempts} attempts: {e}"
                )
    raise last_exception


def is_deadlock(exception: Exception) -> bool:
    """
    Check if an exception is a database deadlock.

    Args:
        exception: Exception to check

    Returns:
        True if the exception is a deadlock
    """
    error_msg = str(exception).lower()
    deadlock_patterns = [
        'deadlock',
        'lock wait timeout',
        'could not serialize access',
        'database is locked',  # SQLite
        'concurrent update',
    ]
    return any(pattern in error_msg for pattern in deadlock_patterns)


def is_connection_error(exception: Exception) -> bool:
    """
    Check if an exception is a connection error.

    Args:
        exception: Exception to check

    Returns:
        True if the exception is a connection error
    """
    error_msg = str(exception).lower()
    connection_patterns = [
        'connection refused',
        'connection reset',
        'connection timed out',
        'cannot connect',
        'lost connection',
        'server closed the connection',
        'connection unexpectedly closed',
    ]
    return any(pattern in error_msg for pattern in connection_patterns)


# =============================================================================
# TimescaleDB Detection Cache
# =============================================================================
_timescale_cache: Dict[str, bool] = {}


def get_environment() -> str:
    """
    Determine the current environment.

    Returns:
        str: 'production', 'development', or 'testing'
    """
    env = os.environ.get("RDT_ENV") or os.environ.get("FLASK_ENV") or os.environ.get("ENV")
    if env:
        return env.lower()

    # Check for test runners
    if "pytest" in os.environ.get("_", "") or "test" in os.environ.get("_", ""):
        return "testing"

    # Check if DATABASE_URL is set (indicates production/staging)
    if os.environ.get("DATABASE_URL"):
        return "production"

    return "development"


def get_database_url(db_url: Optional[str] = None) -> str:
    """
    Get the database URL from parameter, environment, or default.

    Priority:
    1. Provided db_url parameter
    2. DATABASE_URL environment variable
    3. RDT_DATABASE_URL environment variable
    4. Default SQLite database in data directory (development/testing only)

    Args:
        db_url: Optional database URL to use

    Returns:
        str: The database connection URL
    """
    if db_url:
        url = db_url
    else:
        # Check environment variables
        url = os.environ.get("DATABASE_URL")
        if not url:
            url = os.environ.get("RDT_DATABASE_URL")

    if url:
        # Handle Heroku-style postgres:// URLs
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url

    # Default to SQLite in development/testing only
    env = get_environment()
    if env in ("development", "testing"):
        data_dir = Path(__file__).parent.parent
        db_path = data_dir / "rdt_trading.db"
        logger.info(f"Using SQLite database at {db_path} ({env} environment)")
        return f"sqlite:///{db_path}"

    # In production, DATABASE_URL must be set
    raise ValueError(
        "DATABASE_URL environment variable must be set in production. "
        "Set DATABASE_URL to your PostgreSQL connection string."
    )


def get_async_database_url(db_url: Optional[str] = None) -> str:
    """
    Get the async database URL (converts to async driver).

    Args:
        db_url: Optional database URL to use

    Returns:
        str: The async database connection URL
    """
    url = get_database_url(db_url)

    # Convert to async drivers
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql+psycopg2://"):
        return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    elif url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)

    return url


def is_postgres(db_url: str) -> bool:
    """Check if the database URL is for PostgreSQL."""
    return db_url.startswith(("postgresql://", "postgresql+", "postgres://"))


def is_sqlite(db_url: str) -> bool:
    """Check if the database URL is for SQLite."""
    return db_url.startswith("sqlite")


def check_timescale_extension(engine: Engine) -> bool:
    """
    Check if TimescaleDB extension is available and enabled.

    This function caches results based on the database URL to avoid
    repeated queries during the application lifecycle.

    Args:
        engine: SQLAlchemy engine to check.

    Returns:
        bool: True if TimescaleDB is available and enabled.
    """
    # Use URL as cache key (without password)
    cache_key = str(engine.url).split('@')[-1] if '@' in str(engine.url) else str(engine.url)

    if cache_key in _timescale_cache:
        return _timescale_cache[cache_key]

    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
            ))
            is_available = result.scalar() or False
            _timescale_cache[cache_key] = is_available

            if is_available:
                # Get version for logging
                version_result = conn.execute(text(
                    "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
                ))
                version = version_result.scalar()
                logger.info(f"TimescaleDB {version} detected and available")
            else:
                logger.debug("TimescaleDB extension not found (using standard PostgreSQL)")

            return is_available
    except Exception as e:
        logger.debug(f"TimescaleDB check failed (likely not PostgreSQL): {e}")
        _timescale_cache[cache_key] = False
        return False


def get_timescale_version(engine: Engine) -> Optional[str]:
    """
    Get the TimescaleDB version if available.

    Args:
        engine: SQLAlchemy engine.

    Returns:
        str or None: TimescaleDB version string or None if not available.
    """
    if not check_timescale_extension(engine):
        return None

    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
            ))
            return result.scalar()
    except Exception:
        return None


def get_pool_config(db_url: str) -> dict:
    """
    Get connection pool configuration based on database type.

    Args:
        db_url: Database URL

    Returns:
        dict: Pool configuration for create_engine
    """
    if is_sqlite(db_url):
        # SQLite doesn't support connection pooling the same way
        # Use StaticPool for testing, NullPool for development
        env = get_environment()
        if env == "testing":
            return {
                "poolclass": StaticPool,
                "connect_args": {"check_same_thread": False},
            }
        return {
            "poolclass": NullPool,
            "connect_args": {"check_same_thread": False},
        }

    # PostgreSQL pool configuration
    pool_size = int(os.environ.get("DB_POOL_SIZE", "5"))
    max_overflow = int(os.environ.get("DB_POOL_MAX_OVERFLOW", "10"))
    pool_timeout = int(os.environ.get("DB_POOL_TIMEOUT", "30"))
    pool_recycle = int(os.environ.get("DB_POOL_RECYCLE", "1800"))  # 30 minutes
    pool_pre_ping = os.environ.get("DB_POOL_PRE_PING", "true").lower() == "true"

    return {
        "poolclass": QueuePool,
        "pool_size": pool_size,
        "max_overflow": max_overflow,
        "pool_timeout": pool_timeout,
        "pool_recycle": pool_recycle,
        "pool_pre_ping": pool_pre_ping,  # Health check on checkout
    }


class DatabaseManager:
    """
    Manages database connections and sessions with pooling and health checks.

    This class provides:
    - Connection pooling for PostgreSQL
    - Async session support
    - TimescaleDB detection and integration
    - Health checks and migration management

    TimescaleDB:
        When TimescaleDB is available, the `is_timescale` property returns True
        and you can use TimescaleDB-specific features through the timescale module.

    Example:
        manager = DatabaseManager()
        if manager.is_timescale:
            # Use TimescaleDB optimized queries
            from data.timescale import get_signals_time_bucket
            signals = get_signals_time_bucket('1 hour', engine=manager.engine)
    """

    def __init__(self, db_url: str = None):
        self.db_url = get_database_url(db_url)
        self._is_postgres = is_postgres(self.db_url)
        self._is_sqlite = is_sqlite(self.db_url)
        self._is_timescale: Optional[bool] = None
        self._timescale_version: Optional[str] = None

        # Get pool configuration
        pool_config = get_pool_config(self.db_url)

        # Create sync engine
        self.engine: Engine = create_engine(
            self.db_url,
            echo=os.environ.get("DB_ECHO", "false").lower() == "true",
            **pool_config
        )

        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )

        # Setup async engine if supported and using PostgreSQL
        self._async_engine: Optional[AsyncEngine] = None
        self._async_session_maker = None
        if ASYNC_SUPPORT:
            self._setup_async_engine()

        # Log connection info (without sensitive data)
        self._log_connection_info()

        # Check for TimescaleDB (lazy loaded on first access)
        if self._is_postgres:
            self._check_timescale()

    def _setup_async_engine(self):
        """Setup async engine and session maker."""
        try:
            async_url = get_async_database_url(self.db_url)
            pool_config = get_pool_config(self.db_url)

            # Remove sync-specific options for async
            async_pool_config = {
                k: v for k, v in pool_config.items()
                if k not in ("connect_args",)
            }

            # Handle async-specific connect_args
            if self._is_sqlite:
                async_pool_config["connect_args"] = {}

            self._async_engine = create_async_engine(
                async_url,
                echo=os.environ.get("DB_ECHO", "false").lower() == "true",
                **async_pool_config
            )

            self._async_session_maker = async_sessionmaker(
                bind=self._async_engine,
                class_=AsyncSession,
                autoflush=False,
                autocommit=False,
                expire_on_commit=False,
            )
            logger.debug("Async database engine initialized")
        except Exception as e:
            logger.warning(f"Could not initialize async engine: {e}")
            self._async_engine = None
            self._async_session_maker = None

    def _log_connection_info(self):
        """Log connection info without sensitive data."""
        if self._is_postgres:
            # Parse and sanitize URL for logging
            from urllib.parse import urlparse
            parsed = urlparse(self.db_url)
            safe_url = f"{parsed.scheme}://{parsed.username}:***@{parsed.hostname}:{parsed.port or 5432}/{parsed.path.lstrip('/')}"
            logger.info(f"Connected to PostgreSQL: {safe_url}")
        elif self._is_sqlite:
            logger.info(f"Connected to SQLite: {self.db_url}")
        else:
            logger.info(f"Connected to database: {self.db_url.split('@')[-1] if '@' in self.db_url else self.db_url}")

    def _check_timescale(self):
        """Check for TimescaleDB availability and cache the result."""
        if not self._is_postgres:
            self._is_timescale = False
            return

        try:
            self._is_timescale = check_timescale_extension(self.engine)
            if self._is_timescale:
                self._timescale_version = get_timescale_version(self.engine)
        except Exception as e:
            logger.debug(f"Error checking TimescaleDB: {e}")
            self._is_timescale = False

    @property
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL."""
        return self._is_postgres

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite."""
        return self._is_sqlite

    @property
    def is_timescale(self) -> bool:
        """
        Check if TimescaleDB is available.

        Returns True if:
        - Using PostgreSQL
        - TimescaleDB extension is installed and enabled

        Use this to conditionally enable TimescaleDB-optimized queries.

        Example:
            if manager.is_timescale:
                from data.timescale import get_signals_time_bucket
                signals = get_signals_time_bucket('1 hour')
            else:
                # Use standard SQL queries
                signals = session.query(Signal).all()
        """
        if self._is_timescale is None:
            self._check_timescale()
        return self._is_timescale or False

    @property
    def timescale_version(self) -> Optional[str]:
        """Get the TimescaleDB version if available."""
        if self._is_timescale is None:
            self._check_timescale()
        return self._timescale_version

    @property
    def async_engine(self) -> Optional[AsyncEngine]:
        """Get the async engine if available."""
        return self._async_engine

    def get_timescale_manager(self):
        """
        Get a TimescaleManager instance for this database.

        Returns None if TimescaleDB is not available.

        Returns:
            TimescaleManager or None
        """
        if not self.is_timescale:
            return None

        try:
            from data.timescale import TimescaleManager
            return TimescaleManager(self.engine)
        except ImportError:
            logger.warning("TimescaleDB module not available")
            return None

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

    @contextmanager
    def get_session_with_retry(
        self, max_retries: int = 3, retry_on_deadlock: bool = True
    ) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup.

        Note: A @contextmanager must yield exactly once, so retry logic
        cannot be implemented here. Callers needing deadlock retry should
        wrap their ``with`` block in their own retry loop.

        Args:
            max_retries: Unused, kept for API compatibility
            retry_on_deadlock: Unused, kept for API compatibility

        Yields:
            Session: Database session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session with automatic cleanup."""
        if not self._async_session_maker:
            raise RuntimeError(
                "Async sessions not available. Install asyncpg and aiosqlite: "
                "pip install asyncpg aiosqlite"
            )

        session = self._async_session_maker()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    @asynccontextmanager
    async def get_async_session_with_retry(
        self, max_retries: int = 3, retry_on_deadlock: bool = True
    ) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session with automatic cleanup.

        Note: An @asynccontextmanager must yield exactly once, so retry logic
        cannot be implemented here. Callers needing deadlock retry should
        wrap their ``async with`` block in their own retry loop.

        Args:
            max_retries: Unused, kept for API compatibility
            retry_on_deadlock: Unused, kept for API compatibility

        Yields:
            AsyncSession: Async database session
        """
        if not self._async_session_maker:
            raise RuntimeError(
                "Async sessions not available. Install asyncpg and aiosqlite: "
                "pip install asyncpg aiosqlite"
            )

        session = self._async_session_maker()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    def check_connection(
        self, timeout: float = 5.0, max_retries: int = 3
    ) -> Tuple[bool, Optional[str]]:
        """
        Check database connection health with retry logic.

        Args:
            timeout: Connection timeout in seconds
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (is_healthy, error_message)
        """
        @with_retry(max_attempts=max_retries, base_delay=0.5, max_delay=5.0)
        def _check():
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True

        try:
            _check()
            return True, None
        except Exception as e:
            return False, str(e)

    async def check_connection_async(
        self, timeout: float = 5.0, max_retries: int = 3
    ) -> Tuple[bool, Optional[str]]:
        """
        Check database connection health asynchronously with retry logic.

        Args:
            timeout: Connection timeout in seconds
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (is_healthy, error_message)
        """
        if not self._async_engine:
            return False, "Async engine not available"

        async def _check():
            async with self._async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True

        try:
            await with_retry_async(_check, max_attempts=max_retries, base_delay=0.5)
            return True, None
        except Exception as e:
            return False, str(e)

    def get_pool_status(self) -> dict:
        """
        Get connection pool status.

        Returns:
            dict: Pool statistics
        """
        pool = self.engine.pool

        if hasattr(pool, "size"):
            return {
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalidatedcount() if hasattr(pool, "invalidatedcount") else 0,
            }

        return {"pool_type": type(pool).__name__, "status": "no_pooling"}

    def check_migrations_status(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if database migrations are up to date.

        Returns:
            Tuple of (is_up_to_date, current_revision, head_revision)
        """
        try:
            from alembic.config import Config
            from alembic.script import ScriptDirectory
            from alembic.runtime.migration import MigrationContext

            # Find alembic.ini
            project_root = Path(__file__).parent.parent.parent
            alembic_ini = project_root / "alembic.ini"

            if not alembic_ini.exists():
                logger.warning("alembic.ini not found, cannot check migration status")
                return True, None, None

            config = Config(str(alembic_ini))
            config.set_main_option("script_location", str(project_root / "migrations"))

            script = ScriptDirectory.from_config(config)
            head_rev = script.get_current_head()

            with self.engine.connect() as conn:
                context = MigrationContext.configure(conn)
                current_rev = context.get_current_revision()

            is_up_to_date = current_rev == head_rev
            return is_up_to_date, current_rev, head_rev

        except ImportError:
            logger.debug("Alembic not installed, skipping migration check")
            return True, None, None
        except Exception as e:
            logger.warning(f"Could not check migration status: {e}")
            return True, None, None

    def run_migrations(self, target: str = "head") -> bool:
        """
        Run database migrations programmatically.

        Args:
            target: Target revision (default: "head" for latest)

        Returns:
            bool: True if migrations ran successfully
        """
        try:
            from alembic.config import Config
            from alembic import command

            project_root = Path(__file__).parent.parent.parent
            alembic_ini = project_root / "alembic.ini"

            if not alembic_ini.exists():
                logger.error("alembic.ini not found, cannot run migrations")
                return False

            config = Config(str(alembic_ini))
            config.set_main_option("script_location", str(project_root / "migrations"))

            # Set the database URL
            config.set_main_option("sqlalchemy.url", self.db_url)

            logger.info(f"Running migrations to {target}...")
            command.upgrade(config, target)
            logger.info("Migrations completed successfully")
            return True

        except ImportError:
            logger.error("Alembic not installed, cannot run migrations")
            return False
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

    def stamp_revision(self, revision: str = "head") -> bool:
        """
        Stamp the database with a specific revision without running migrations.

        This is useful for marking an existing database as being at a specific
        migration version without actually running the migrations.

        Args:
            revision: The revision to stamp (default: "head")

        Returns:
            bool: True if stamp was successful
        """
        try:
            from alembic.config import Config
            from alembic import command

            project_root = Path(__file__).parent.parent.parent
            alembic_ini = project_root / "alembic.ini"

            if not alembic_ini.exists():
                logger.error("alembic.ini not found, cannot stamp revision")
                return False

            config = Config(str(alembic_ini))
            config.set_main_option("script_location", str(project_root / "migrations"))
            config.set_main_option("sqlalchemy.url", self.db_url)

            command.stamp(config, revision)
            logger.info(f"Database stamped with revision: {revision}")
            return True

        except ImportError:
            logger.error("Alembic not installed, cannot stamp revision")
            return False
        except Exception as e:
            logger.error(f"Stamp failed: {e}")
            return False

    def close(self):
        """Close all connections and dispose of the engine."""
        self.engine.dispose()
        if self._async_engine:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._async_engine.dispose())
            except RuntimeError:
                pass  # No running event loop, async engine will be cleaned up on GC
        logger.debug("Database connections closed")

    async def close_async(self):
        """Close async connections."""
        if self._async_engine:
            await self._async_engine.dispose()
            logger.debug("Async database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None
_db_manager_lock = threading.Lock()


def get_db_manager(db_url: Optional[str] = None) -> DatabaseManager:
    """Get or create the global database manager (thread-safe)."""
    global _db_manager
    if _db_manager is None:
        with _db_manager_lock:
            if _db_manager is None:
                _db_manager = DatabaseManager(db_url)
    return _db_manager


def reset_db_manager():
    """Reset the global database manager (useful for testing)."""
    global _db_manager
    if _db_manager is not None:
        _db_manager.close()
        _db_manager = None


def init_database(check_migrations: bool = True, auto_migrate: bool = False):
    """
    Initialize the database.

    Args:
        check_migrations: If True, check if migrations are up to date and log a warning if not
        auto_migrate: If True, automatically run pending migrations

    Returns:
        DatabaseManager: The initialized database manager
    """
    manager = get_db_manager()

    if check_migrations:
        is_up_to_date, current_rev, head_rev = manager.check_migrations_status()

        if not is_up_to_date:
            if auto_migrate:
                logger.info("Running pending migrations...")
                manager.run_migrations()
            else:
                logger.warning(
                    f"Database migrations are not up to date. "
                    f"Current: {current_rev or 'None'}, Latest: {head_rev}. "
                    f"Run 'python scripts/db_migrate.py upgrade' to apply pending migrations."
                )
    else:
        # Legacy behavior: create tables directly (bypasses migrations)
        manager.create_tables()

    return manager


def run_migrations(target: str = "head") -> bool:
    """
    Convenience function to run migrations.

    Args:
        target: Target revision (default: "head" for latest)

    Returns:
        bool: True if migrations ran successfully
    """
    manager = get_db_manager()
    return manager.run_migrations(target)


def check_migrations() -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Convenience function to check migration status.

    Returns:
        Tuple of (is_up_to_date, current_revision, head_revision)
    """
    manager = get_db_manager()
    return manager.check_migrations_status()


def health_check() -> dict:
    """
    Perform a comprehensive health check on the database.

    Returns:
        dict: Health check results including TimescaleDB status
    """
    manager = get_db_manager()

    is_healthy, error = manager.check_connection()
    pool_status = manager.get_pool_status()
    is_up_to_date, current_rev, head_rev = manager.check_migrations_status()

    # Determine database type with TimescaleDB info
    if manager.is_timescale:
        db_type = f"timescaledb ({manager.timescale_version})"
    elif manager.is_postgres:
        db_type = "postgresql"
    else:
        db_type = "sqlite"

    result = {
        "healthy": is_healthy,
        "error": error,
        "database_type": db_type,
        "pool_status": pool_status,
        "migrations": {
            "up_to_date": is_up_to_date,
            "current_revision": current_rev,
            "head_revision": head_rev,
        },
        "timescale": {
            "available": manager.is_timescale,
            "version": manager.timescale_version,
        },
    }

    # Add hypertable info if TimescaleDB is available
    if manager.is_timescale:
        try:
            from data.timescale import get_timescale_info
            ts_info = get_timescale_info(manager.engine)
            result["timescale"]["hypertables"] = ts_info.get("hypertables", [])
            result["timescale"]["compression_enabled"] = ts_info.get("compression_enabled", False)
        except ImportError:
            pass

    return result
