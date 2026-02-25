#!/usr/bin/env python3
"""
Database Migration CLI Tool for RDT Trading System.

This script provides a command-line interface for managing database migrations
using Alembic. It supports common operations like upgrade, downgrade, and
generating new migrations.

Usage:
    python scripts/db_migrate.py <command> [options]

Commands:
    init        - Initialize the database with latest migrations
    migrate     - Generate a new migration from model changes
    upgrade     - Upgrade database to a revision (default: head)
    downgrade   - Downgrade database by one revision or to specific revision
    history     - Show migration history
    current     - Show current database revision
    heads       - Show current available heads
    stamp       - Stamp the database with a specific revision without running migrations
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_alembic_config():
    """Get Alembic configuration object."""
    from alembic.config import Config

    alembic_ini = project_root / "alembic.ini"
    if not alembic_ini.exists():
        print(f"Error: alembic.ini not found at {alembic_ini}")
        sys.exit(1)

    config = Config(str(alembic_ini))

    # Set the script location relative to project root
    config.set_main_option("script_location", str(project_root / "migrations"))

    return config


def cmd_init(args):
    """Initialize database with all migrations (upgrade to head)."""
    from alembic import command

    config = get_alembic_config()
    print("Initializing database (upgrading to head)...")
    command.upgrade(config, "head")
    print("Database initialization complete.")


def cmd_migrate(args):
    """Generate a new migration from model changes."""
    from alembic import command

    config = get_alembic_config()
    message = args.message or "auto-generated migration"

    print(f"Generating migration: {message}")
    command.revision(config, message=message, autogenerate=True)
    print("Migration generated. Review the generated file before applying.")


def cmd_upgrade(args):
    """Upgrade database to a specific revision."""
    from alembic import command

    config = get_alembic_config()
    revision = args.revision or "head"

    print(f"Upgrading database to: {revision}")
    command.upgrade(config, revision)
    print("Upgrade complete.")


def cmd_downgrade(args):
    """Downgrade database to a specific revision."""
    from alembic import command

    config = get_alembic_config()
    revision = args.revision or "-1"

    print(f"Downgrading database to: {revision}")
    command.downgrade(config, revision)
    print("Downgrade complete.")


def cmd_history(args):
    """Show migration history."""
    from alembic import command

    config = get_alembic_config()

    if args.verbose:
        command.history(config, verbose=True)
    else:
        command.history(config)


def cmd_current(args):
    """Show current database revision."""
    from alembic import command

    config = get_alembic_config()
    command.current(config, verbose=args.verbose)


def cmd_heads(args):
    """Show current available head revisions."""
    from alembic import command

    config = get_alembic_config()
    command.heads(config, verbose=args.verbose)


def cmd_stamp(args):
    """Stamp the database with a specific revision without running migrations."""
    from alembic import command

    config = get_alembic_config()
    revision = args.revision or "head"

    print(f"Stamping database with revision: {revision}")
    command.stamp(config, revision)
    print("Stamp complete.")


def cmd_check(args):
    """Check if there are pending migrations."""
    from alembic import command
    from alembic.script import ScriptDirectory
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import create_engine

    config = get_alembic_config()
    script = ScriptDirectory.from_config(config)

    # Get database URL
    db_url = os.environ.get("DATABASE_URL") or os.environ.get("RDT_DATABASE_URL")
    if not db_url:
        db_url = config.get_main_option("sqlalchemy.url")

    engine = create_engine(db_url)

    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        current_rev = context.get_current_revision()
        head_rev = script.get_current_head()

        if current_rev == head_rev:
            print("Database is up to date.")
            return True
        else:
            print(f"Database is NOT up to date.")
            print(f"  Current revision: {current_rev or 'None (empty database)'}")
            print(f"  Latest revision:  {head_rev}")
            return False


def cmd_show(args):
    """Show the SQL that would be generated for a revision."""
    from alembic import command

    config = get_alembic_config()
    revision = args.revision or "head"

    print(f"SQL for upgrading to {revision}:")
    print("-" * 60)
    command.upgrade(config, revision, sql=True)


def main():
    parser = argparse.ArgumentParser(
        description="RDT Trading System Database Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Initialize/upgrade to latest
    python scripts/db_migrate.py init

    # Create a new migration
    python scripts/db_migrate.py migrate -m "Add user preferences table"

    # Upgrade to latest
    python scripts/db_migrate.py upgrade

    # Upgrade to specific revision
    python scripts/db_migrate.py upgrade -r abc123

    # Downgrade one revision
    python scripts/db_migrate.py downgrade

    # Downgrade to specific revision
    python scripts/db_migrate.py downgrade -r abc123

    # Check if migrations are up to date
    python scripts/db_migrate.py check

    # Show migration history
    python scripts/db_migrate.py history

    # Show current revision
    python scripts/db_migrate.py current
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize database with all migrations")
    init_parser.set_defaults(func=cmd_init)

    # migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Generate a new migration")
    migrate_parser.add_argument("-m", "--message", help="Migration message", required=False)
    migrate_parser.set_defaults(func=cmd_migrate)

    # upgrade command
    upgrade_parser = subparsers.add_parser("upgrade", help="Upgrade database")
    upgrade_parser.add_argument("-r", "--revision", help="Target revision (default: head)")
    upgrade_parser.set_defaults(func=cmd_upgrade)

    # downgrade command
    downgrade_parser = subparsers.add_parser("downgrade", help="Downgrade database")
    downgrade_parser.add_argument("-r", "--revision", help="Target revision (default: -1)")
    downgrade_parser.set_defaults(func=cmd_downgrade)

    # history command
    history_parser = subparsers.add_parser("history", help="Show migration history")
    history_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    history_parser.set_defaults(func=cmd_history)

    # current command
    current_parser = subparsers.add_parser("current", help="Show current revision")
    current_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    current_parser.set_defaults(func=cmd_current)

    # heads command
    heads_parser = subparsers.add_parser("heads", help="Show head revisions")
    heads_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    heads_parser.set_defaults(func=cmd_heads)

    # stamp command
    stamp_parser = subparsers.add_parser("stamp", help="Stamp database with revision")
    stamp_parser.add_argument("-r", "--revision", help="Revision to stamp (default: head)")
    stamp_parser.set_defaults(func=cmd_stamp)

    # check command
    check_parser = subparsers.add_parser("check", help="Check if migrations are up to date")
    check_parser.set_defaults(func=cmd_check)

    # show command
    show_parser = subparsers.add_parser("show", help="Show SQL for a migration")
    show_parser.add_argument("-r", "--revision", help="Target revision (default: head)")
    show_parser.set_defaults(func=cmd_show)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Change to project root directory
    os.chdir(project_root)

    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        if os.environ.get("DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
