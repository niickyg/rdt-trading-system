#!/usr/bin/env python3
"""
Database Backup Script for RDT Trading System.

This script creates backups of the PostgreSQL database with support for:
- Scheduled backups (via cron or task scheduler)
- Retention policy (keep last N backups)
- Compression
- Backup verification
- Restore functionality

Usage:
    python scripts/backup_db.py backup [options]
    python scripts/backup_db.py restore <backup_file>
    python scripts/backup_db.py list
    python scripts/backup_db.py cleanup

Options:
    --output-dir DIR    Directory to store backups (default: ./backups)
    --retention N       Number of backups to retain (default: 7)
    --compress          Compress backup with gzip
    --verify            Verify backup after creation
    --verbose           Enable verbose output
"""

import argparse
import gzip
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

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


class DatabaseBackup:
    """Handles PostgreSQL database backup and restore operations."""

    def __init__(
        self,
        output_dir: str = "./backups",
        retention: int = 7,
        compress: bool = True,
    ):
        """
        Initialize backup manager.

        Args:
            output_dir: Directory to store backups
            retention: Number of backups to retain
            compress: Whether to compress backups
        """
        self.output_dir = Path(output_dir)
        self.retention = retention
        self.compress = compress

        # Parse database URL
        self.db_url = os.environ.get("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable is required")

        self._parse_db_url()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _parse_db_url(self):
        """Parse database URL into components."""
        # Handle postgres:// and postgresql:// prefixes
        url = self.db_url
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)

        parsed = urlparse(url)

        self.db_host = parsed.hostname or "localhost"
        self.db_port = parsed.port or 5432
        self.db_name = parsed.path.lstrip("/")
        self.db_user = parsed.username or "rdt"
        self.db_password = parsed.password or ""

    def _get_backup_filename(self, timestamp: Optional[datetime] = None) -> str:
        """Generate backup filename with timestamp."""
        if timestamp is None:
            timestamp = datetime.now()

        filename = f"rdt_trading_{timestamp.strftime('%Y%m%d_%H%M%S')}.dump"
        if self.compress:
            filename += ".gz"

        return filename

    def _run_pg_command(
        self,
        command: List[str],
        capture_output: bool = True,
        input_data: Optional[bytes] = None,
    ) -> Tuple[int, str, str]:
        """
        Run a PostgreSQL command with proper environment.

        Args:
            command: Command and arguments
            capture_output: Whether to capture stdout/stderr
            input_data: Optional input data for stdin

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        env = os.environ.copy()
        env["PGPASSWORD"] = self.db_password

        try:
            result = subprocess.run(
                command,
                env=env,
                capture_output=capture_output,
                input=input_data,
                timeout=3600,  # 1 hour timeout
            )

            stdout = result.stdout.decode("utf-8") if result.stdout else ""
            stderr = result.stderr.decode("utf-8") if result.stderr else ""

            return result.returncode, stdout, stderr

        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)

    def backup(self, verify: bool = False) -> Optional[Path]:
        """
        Create a database backup.

        Args:
            verify: Whether to verify the backup after creation

        Returns:
            Path to the backup file, or None if backup failed
        """
        timestamp = datetime.now()
        filename = self._get_backup_filename(timestamp)
        backup_path = self.output_dir / filename
        temp_path = backup_path.with_suffix(".tmp")

        logger.info(f"Starting backup to {backup_path}...")

        # Build pg_dump command
        command = [
            "pg_dump",
            "-h", self.db_host,
            "-p", str(self.db_port),
            "-U", self.db_user,
            "-d", self.db_name,
            "-F", "c",  # Custom format (most flexible)
            "-Z", "0" if self.compress else "0",  # No compression in pg_dump if we compress separately
            "-f", str(temp_path.with_suffix(".dump") if self.compress else temp_path),
        ]

        returncode, stdout, stderr = self._run_pg_command(command)

        if returncode != 0:
            logger.error(f"Backup failed: {stderr}")
            return None

        # Compress if requested
        if self.compress:
            dump_path = temp_path.with_suffix(".dump")
            logger.info("Compressing backup...")

            try:
                with open(dump_path, "rb") as f_in:
                    with gzip.open(temp_path, "wb", compresslevel=6) as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Remove uncompressed file
                dump_path.unlink()

            except Exception as e:
                logger.error(f"Compression failed: {e}")
                return None

        # Rename temp file to final name
        temp_path.rename(backup_path)

        # Get backup size
        size_mb = backup_path.stat().st_size / (1024 * 1024)
        logger.info(f"Backup completed: {backup_path} ({size_mb:.2f} MB)")

        # Verify if requested
        if verify:
            if not self.verify_backup(backup_path):
                logger.warning("Backup verification failed!")
            else:
                logger.info("Backup verified successfully.")

        return backup_path

    def verify_backup(self, backup_path: Path) -> bool:
        """
        Verify a backup file is valid.

        Args:
            backup_path: Path to the backup file

        Returns:
            bool: True if backup is valid
        """
        logger.info(f"Verifying backup: {backup_path}...")

        if not backup_path.exists():
            logger.error("Backup file does not exist")
            return False

        # Decompress if needed
        if str(backup_path).endswith(".gz"):
            try:
                with gzip.open(backup_path, "rb") as f:
                    # Read first few bytes to verify it's valid gzip
                    f.read(1024)
            except Exception as e:
                logger.error(f"Invalid gzip file: {e}")
                return False

        # Use pg_restore to verify
        command = [
            "pg_restore",
            "--list",
            str(backup_path),
        ]

        # Handle compressed files
        if str(backup_path).endswith(".gz"):
            # Decompress to temp file for verification
            temp_path = backup_path.with_suffix("")
            try:
                with gzip.open(backup_path, "rb") as f_in:
                    with open(temp_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                command[-1] = str(temp_path)
                returncode, stdout, stderr = self._run_pg_command(command)
                temp_path.unlink()

            except Exception as e:
                logger.error(f"Verification failed: {e}")
                return False
        else:
            returncode, stdout, stderr = self._run_pg_command(command)

        if returncode != 0:
            logger.error(f"Backup verification failed: {stderr}")
            return False

        return True

    def restore(self, backup_path: Path, clean: bool = False) -> bool:
        """
        Restore a database from backup.

        WARNING: This will overwrite existing data!

        Args:
            backup_path: Path to the backup file
            clean: Whether to drop existing objects before restore

        Returns:
            bool: True if restore was successful
        """
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False

        logger.warning("=" * 60)
        logger.warning("WARNING: This will OVERWRITE data in the database!")
        logger.warning("=" * 60)

        confirm = os.environ.get("CONFIRM_RESTORE", "").lower()
        if confirm != "yes":
            response = input("Type 'yes' to confirm restore: ")
            if response.lower() != "yes":
                logger.info("Restore cancelled.")
                return False

        logger.info(f"Restoring from {backup_path}...")

        # Handle compressed files
        restore_path = backup_path
        temp_path = None

        if str(backup_path).endswith(".gz"):
            temp_path = backup_path.with_suffix("")
            logger.info("Decompressing backup...")

            try:
                with gzip.open(backup_path, "rb") as f_in:
                    with open(temp_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                restore_path = temp_path

            except Exception as e:
                logger.error(f"Decompression failed: {e}")
                return False

        # Build pg_restore command
        command = [
            "pg_restore",
            "-h", self.db_host,
            "-p", str(self.db_port),
            "-U", self.db_user,
            "-d", self.db_name,
            "--no-owner",
            "--no-privileges",
        ]

        if clean:
            command.append("--clean")

        command.append(str(restore_path))

        returncode, stdout, stderr = self._run_pg_command(command)

        # Clean up temp file
        if temp_path and temp_path.exists():
            temp_path.unlink()

        if returncode != 0:
            # pg_restore may return non-zero even on partial success
            if "error" in stderr.lower():
                logger.error(f"Restore failed: {stderr}")
                return False
            else:
                logger.warning(f"Restore completed with warnings: {stderr}")

        logger.info("Restore completed successfully.")
        return True

    def list_backups(self) -> List[Tuple[Path, datetime, float]]:
        """
        List all backup files.

        Returns:
            List of tuples: (path, timestamp, size_mb)
        """
        backups = []

        for path in self.output_dir.glob("rdt_trading_*.dump*"):
            try:
                # Parse timestamp from filename
                name = path.stem
                if name.endswith(".dump"):
                    name = name[:-5]

                timestamp_str = name.replace("rdt_trading_", "")
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                size_mb = path.stat().st_size / (1024 * 1024)

                backups.append((path, timestamp, size_mb))

            except Exception:
                continue

        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x[1], reverse=True)

        return backups

    def cleanup(self, dry_run: bool = False) -> int:
        """
        Remove old backups according to retention policy.

        Args:
            dry_run: If True, only report what would be deleted

        Returns:
            Number of backups deleted
        """
        backups = self.list_backups()

        if len(backups) <= self.retention:
            logger.info(f"No cleanup needed. Have {len(backups)} backups, retention is {self.retention}.")
            return 0

        to_delete = backups[self.retention:]
        deleted = 0

        for path, timestamp, size_mb in to_delete:
            if dry_run:
                logger.info(f"Would delete: {path.name} ({size_mb:.2f} MB)")
            else:
                try:
                    path.unlink()
                    logger.info(f"Deleted: {path.name} ({size_mb:.2f} MB)")
                    deleted += 1
                except Exception as e:
                    logger.error(f"Failed to delete {path.name}: {e}")

        logger.info(f"Cleanup complete. Deleted {deleted} backup(s).")
        return deleted


def cmd_backup(args):
    """Run backup command."""
    backup = DatabaseBackup(
        output_dir=args.output_dir,
        retention=args.retention,
        compress=args.compress,
    )

    backup_path = backup.backup(verify=args.verify)

    if backup_path:
        logger.info(f"Backup successful: {backup_path}")

        if args.cleanup:
            backup.cleanup()
    else:
        logger.error("Backup failed!")
        sys.exit(1)


def cmd_restore(args):
    """Run restore command."""
    backup = DatabaseBackup(output_dir=args.output_dir)
    backup_path = Path(args.backup_file)

    if not backup.restore(backup_path, clean=args.clean):
        sys.exit(1)


def cmd_list(args):
    """List available backups."""
    backup = DatabaseBackup(output_dir=args.output_dir)
    backups = backup.list_backups()

    if not backups:
        logger.info("No backups found.")
        return

    print("\nAvailable backups:")
    print("-" * 70)
    print(f"{'Filename':<45} {'Date':<20} {'Size':<10}")
    print("-" * 70)

    for path, timestamp, size_mb in backups:
        print(f"{path.name:<45} {timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20} {size_mb:.2f} MB")

    print("-" * 70)
    print(f"Total: {len(backups)} backup(s)")


def cmd_cleanup(args):
    """Clean up old backups."""
    backup = DatabaseBackup(
        output_dir=args.output_dir,
        retention=args.retention,
    )

    backup.cleanup(dry_run=args.dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="RDT Trading System Database Backup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create a backup
    python scripts/backup_db.py backup

    # Create a verified, compressed backup
    python scripts/backup_db.py backup --compress --verify

    # List available backups
    python scripts/backup_db.py list

    # Restore from a backup
    python scripts/backup_db.py restore backups/rdt_trading_20240101_120000.dump.gz

    # Clean up old backups (keep last 7)
    python scripts/backup_db.py cleanup --retention 7

Scheduling backups (cron example):
    # Daily backup at 2:00 AM
    0 2 * * * cd /path/to/rdt-trading-system && python scripts/backup_db.py backup --cleanup

Environment variables:
    DATABASE_URL        PostgreSQL connection string (required)
    CONFIRM_RESTORE     Set to 'yes' to skip restore confirmation
        """,
    )

    parser.add_argument(
        "--output-dir",
        default="./backups",
        help="Directory to store backups (default: ./backups)",
    )
    parser.add_argument(
        "--retention",
        type=int,
        default=7,
        help="Number of backups to retain (default: 7)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # backup command
    backup_parser = subparsers.add_parser("backup", help="Create a database backup")
    backup_parser.add_argument(
        "--compress",
        action="store_true",
        default=True,
        help="Compress backup with gzip (default: True)",
    )
    backup_parser.add_argument(
        "--no-compress",
        action="store_false",
        dest="compress",
        help="Don't compress backup",
    )
    backup_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify backup after creation",
    )
    backup_parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up old backups after creating new one",
    )
    backup_parser.set_defaults(func=cmd_backup)

    # restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from a backup")
    restore_parser.add_argument(
        "backup_file",
        help="Path to the backup file",
    )
    restore_parser.add_argument(
        "--clean",
        action="store_true",
        help="Drop existing objects before restore",
    )
    restore_parser.set_defaults(func=cmd_restore)

    # list command
    list_parser = subparsers.add_parser("list", help="List available backups")
    list_parser.set_defaults(func=cmd_list)

    # cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove old backups")
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    cleanup_parser.set_defaults(func=cmd_cleanup)

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
