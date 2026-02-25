#!/usr/bin/env python3
"""
Database Initialization Script for RDT Trading System.

This script initializes the database schema, runs Alembic migrations,
creates default data (admin user, etc.), and verifies the database connection.

Usage:
    python scripts/init_db.py [options]

Options:
    --skip-migrations   Skip running Alembic migrations
    --skip-defaults     Skip creating default data
    --reset             Drop all tables and reinitialize (DANGEROUS!)
    --verify-only       Only verify connection, don't make changes
    --verbose           Enable verbose output
"""

import argparse
import hashlib
import logging
import os
import secrets
import sys
from datetime import datetime, date
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def verify_connection(manager) -> bool:
    """
    Verify database connection is working.

    Args:
        manager: DatabaseManager instance

    Returns:
        bool: True if connection is healthy
    """
    logger.info("Verifying database connection...")

    is_healthy, error = manager.check_connection()

    if is_healthy:
        logger.info("Database connection: OK")
        db_type = "PostgreSQL" if manager.is_postgres else "SQLite"
        logger.info(f"Database type: {db_type}")

        # Log pool status for PostgreSQL
        if manager.is_postgres:
            pool_status = manager.get_pool_status()
            logger.info(f"Connection pool: {pool_status}")

        return True
    else:
        logger.error(f"Database connection FAILED: {error}")
        return False


def run_migrations(manager, verbose: bool = False) -> bool:
    """
    Run Alembic database migrations.

    Args:
        manager: DatabaseManager instance
        verbose: Enable verbose output

    Returns:
        bool: True if migrations ran successfully
    """
    logger.info("Running database migrations...")

    try:
        from alembic.config import Config
        from alembic import command

        alembic_ini = project_root / "alembic.ini"

        if not alembic_ini.exists():
            logger.warning("alembic.ini not found. Creating tables directly...")
            manager.create_tables()
            logger.info("Tables created successfully.")
            return True

        config = Config(str(alembic_ini))
        config.set_main_option("script_location", str(project_root / "migrations"))
        config.set_main_option("sqlalchemy.url", manager.db_url)

        # Run upgrade to head
        command.upgrade(config, "head")
        logger.info("Migrations completed successfully.")

        # Check current revision
        is_up_to_date, current_rev, head_rev = manager.check_migrations_status()
        logger.info(f"Database revision: {current_rev}")

        return True

    except ImportError:
        logger.warning("Alembic not installed. Creating tables directly...")
        manager.create_tables()
        logger.info("Tables created successfully.")
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def create_default_admin(session, verbose: bool = False) -> bool:
    """
    Create default admin user if none exists.

    Args:
        session: Database session
        verbose: Enable verbose output

    Returns:
        bool: True if admin was created or already exists
    """
    from data.database.models import User

    logger.info("Checking for admin user...")

    # Check if any admin user exists
    existing_admin = session.query(User).filter(User.is_admin == True).first()

    if existing_admin:
        logger.info(f"Admin user already exists: {existing_admin.username}")
        return True

    # Create default admin user
    admin_username = os.environ.get("ADMIN_USERNAME", "admin")
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@localhost")
    admin_password = os.environ.get("ADMIN_PASSWORD")

    if not admin_password:
        # Generate a secure random password if not provided
        admin_password = secrets.token_urlsafe(16)
        logger.warning(f"Generated admin password: {admin_password}")
        logger.warning("Please save this password and change it after first login!")

    # Hash the password (using simple SHA-256 for compatibility)
    # In production, use proper password hashing like bcrypt
    password_hash = hashlib.sha256(admin_password.encode()).hexdigest()

    admin_user = User(
        username=admin_username,
        email=admin_email,
        password_hash=password_hash,
        is_active=True,
        is_admin=True,
        created_at=datetime.utcnow(),
    )

    session.add(admin_user)
    logger.info(f"Created admin user: {admin_username}")

    return True


def create_default_watchlist(session, verbose: bool = False) -> bool:
    """
    Create default watchlist items.

    Args:
        session: Database session
        verbose: Enable verbose output

    Returns:
        bool: True if watchlist was created or already exists
    """
    from data.database.models import WatchlistItem

    logger.info("Checking watchlist...")

    # Check if any watchlist items exist
    existing_count = session.query(WatchlistItem).count()

    if existing_count > 0:
        logger.info(f"Watchlist already has {existing_count} items.")
        return True

    # Default symbols for RDT Trading (relative strength trading)
    default_symbols = [
        ("SPY", "S&P 500 ETF - Market benchmark"),
        ("QQQ", "Nasdaq 100 ETF - Tech benchmark"),
        ("IWM", "Russell 2000 ETF - Small cap benchmark"),
        ("AAPL", "Apple Inc."),
        ("MSFT", "Microsoft Corporation"),
        ("GOOGL", "Alphabet Inc."),
        ("AMZN", "Amazon.com Inc."),
        ("NVDA", "NVIDIA Corporation"),
        ("META", "Meta Platforms Inc."),
        ("TSLA", "Tesla Inc."),
    ]

    for symbol, notes in default_symbols:
        item = WatchlistItem(
            symbol=symbol,
            added_date=date.today(),
            notes=notes,
            active=True,
        )
        session.add(item)

    logger.info(f"Created {len(default_symbols)} default watchlist items.")
    return True


def create_initial_daily_stats(session, verbose: bool = False) -> bool:
    """
    Create initial daily stats entry for today.

    Args:
        session: Database session
        verbose: Enable verbose output

    Returns:
        bool: True if stats were created or already exist
    """
    from data.database.models import DailyStats

    logger.info("Checking daily stats...")

    today = date.today()
    existing_stats = session.query(DailyStats).filter(DailyStats.date == today).first()

    if existing_stats:
        logger.info(f"Daily stats for {today} already exist.")
        return True

    # Get initial account size from environment or use default
    account_size = float(os.environ.get("ACCOUNT_SIZE", "25000"))

    stats = DailyStats(
        date=today,
        starting_balance=account_size,
        ending_balance=account_size,
        pnl=0.0,
        num_trades=0,
        winners=0,
        losers=0,
    )

    session.add(stats)
    logger.info(f"Created daily stats for {today} with starting balance ${account_size:,.2f}")

    return True


def create_default_data(manager, verbose: bool = False) -> bool:
    """
    Create default data (admin user, watchlist, etc.).

    Args:
        manager: DatabaseManager instance
        verbose: Enable verbose output

    Returns:
        bool: True if all default data was created successfully
    """
    logger.info("Creating default data...")

    try:
        with manager.get_session() as session:
            success = True

            # Create admin user
            if not create_default_admin(session, verbose):
                success = False

            # Create default watchlist
            if not create_default_watchlist(session, verbose):
                success = False

            # Create initial daily stats
            if not create_initial_daily_stats(session, verbose):
                success = False

            if success:
                logger.info("Default data created successfully.")
            else:
                logger.warning("Some default data creation failed.")

            return success

    except Exception as e:
        logger.error(f"Failed to create default data: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def reset_database(manager) -> bool:
    """
    Drop all tables and reinitialize the database.

    WARNING: This will delete all data!

    Args:
        manager: DatabaseManager instance

    Returns:
        bool: True if reset was successful
    """
    logger.warning("=" * 60)
    logger.warning("WARNING: This will DELETE ALL DATA in the database!")
    logger.warning("=" * 60)

    confirm = os.environ.get("CONFIRM_RESET", "").lower()
    if confirm != "yes":
        response = input("Type 'yes' to confirm database reset: ")
        if response.lower() != "yes":
            logger.info("Reset cancelled.")
            return False

    logger.info("Dropping all tables...")
    manager.drop_tables()

    logger.info("Database reset complete.")
    return True


def print_summary(manager):
    """Print database summary information."""
    from data.database.models import User, Trade, Signal, WatchlistItem, DailyStats

    logger.info("=" * 60)
    logger.info("Database Summary")
    logger.info("=" * 60)

    try:
        with manager.get_session() as session:
            user_count = session.query(User).count()
            trade_count = session.query(Trade).count()
            signal_count = session.query(Signal).count()
            watchlist_count = session.query(WatchlistItem).count()
            stats_count = session.query(DailyStats).count()

            logger.info(f"Users:          {user_count}")
            logger.info(f"Trades:         {trade_count}")
            logger.info(f"Signals:        {signal_count}")
            logger.info(f"Watchlist:      {watchlist_count}")
            logger.info(f"Daily Stats:    {stats_count}")

    except Exception as e:
        logger.warning(f"Could not get summary: {e}")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Initialize RDT Trading System Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Initialize database with defaults
    python scripts/init_db.py

    # Verify connection only
    python scripts/init_db.py --verify-only

    # Skip creating default data
    python scripts/init_db.py --skip-defaults

    # Reset database (DANGEROUS!)
    CONFIRM_RESET=yes python scripts/init_db.py --reset

Environment variables:
    DATABASE_URL        PostgreSQL connection string
    ADMIN_USERNAME      Admin username (default: admin)
    ADMIN_EMAIL         Admin email (default: admin@localhost)
    ADMIN_PASSWORD      Admin password (generated if not set)
    ACCOUNT_SIZE        Initial account size (default: 25000)
    CONFIRM_RESET       Set to 'yes' to confirm database reset
        """,
    )

    parser.add_argument(
        "--skip-migrations",
        action="store_true",
        help="Skip running Alembic migrations",
    )
    parser.add_argument(
        "--skip-defaults",
        action="store_true",
        help="Skip creating default data",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all tables and reinitialize (DANGEROUS!)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify connection, don't make changes",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Import database manager
    try:
        from data.database.connection import DatabaseManager
    except ImportError as e:
        logger.error(f"Failed to import database module: {e}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("RDT Trading System - Database Initialization")
    logger.info("=" * 60)

    # Create database manager
    try:
        manager = DatabaseManager()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Verify connection
    if not verify_connection(manager):
        sys.exit(1)

    if args.verify_only:
        logger.info("Verification complete. Exiting.")
        sys.exit(0)

    # Reset if requested
    if args.reset:
        if not reset_database(manager):
            sys.exit(1)

    # Run migrations
    if not args.skip_migrations:
        if not run_migrations(manager, args.verbose):
            sys.exit(1)

    # Create default data
    if not args.skip_defaults:
        if not create_default_data(manager, args.verbose):
            logger.warning("Some default data creation failed, but continuing...")

    # Print summary
    print_summary(manager)

    logger.info("Database initialization complete!")
    sys.exit(0)


if __name__ == "__main__":
    main()
