"""
Alembic environment configuration for RDT Trading System.

This module configures Alembic to work with SQLAlchemy models
and supports both SQLite and PostgreSQL databases.
"""

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool, create_engine
from alembic import context

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all models to ensure they are registered with Base.metadata
from data.database.models import (
    Base,
    Trade,
    Position,
    Signal,
    DailyStats,
    WatchlistItem,
    APIUser,
    User,
    TradeDirection,
    TradeStatus,
    ExitReason,
    SignalStatus,
    SubscriptionTierEnum,
)

# This is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object for 'autogenerate' support
target_metadata = Base.metadata


def get_database_url() -> str:
    """
    Get the database URL from environment or config.

    Priority:
    1. DATABASE_URL environment variable
    2. RDT_DATABASE_URL environment variable
    3. alembic.ini sqlalchemy.url setting

    Returns:
        str: The database connection URL
    """
    # Check environment variables first
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        # Handle Heroku-style postgres:// URLs
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return db_url

    db_url = os.environ.get("RDT_DATABASE_URL")
    if db_url:
        return db_url

    # Fall back to config file
    return config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_as_batch=True,  # Required for SQLite ALTER TABLE support
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    url = get_database_url()

    # Determine if we're using SQLite for batch mode configuration
    is_sqlite = url.startswith("sqlite")

    # Create engine with appropriate settings
    if is_sqlite:
        connectable = create_engine(
            url,
            poolclass=pool.NullPool,
        )
    else:
        # For PostgreSQL and other databases
        configuration = config.get_section(config.config_ini_section, {})
        configuration["sqlalchemy.url"] = url
        connectable = engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=is_sqlite,  # Enable batch mode for SQLite
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
