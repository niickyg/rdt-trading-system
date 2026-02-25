#!/usr/bin/env python3
"""
Data Retention CLI for RDT Trading System.

Run retention policies to archive and clean up old data.

Usage:
    python scripts/run_retention.py archive [options]
    python scripts/run_retention.py clean [options]
    python scripts/run_retention.py restore <archive_id>
    python scripts/run_retention.py status
    python scripts/run_retention.py list-archives

Examples:
    # Preview what would be archived (dry run)
    python scripts/run_retention.py archive --dry-run

    # Archive old signals
    python scripts/run_retention.py archive --data-type signals

    # Clean up expired data (dry run)
    python scripts/run_retention.py clean --dry-run

    # Actually clean up data
    python scripts/run_retention.py clean --no-dry-run --force

    # List all archives
    python scripts/run_retention.py list-archives

    # Restore from archive
    python scripts/run_retention.py restore signals_20240101_120000

    # Check retention status
    python scripts/run_retention.py status

Scheduling with cron:
    # Run cleanup daily at 3 AM
    0 3 * * * cd /path/to/rdt-trading-system && python scripts/run_retention.py clean --no-dry-run --force >> /var/log/rdt-retention.log 2>&1
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retention")


def progress_callback(progress):
    """Print progress updates."""
    print(
        f"\r  [{progress.data_type}] "
        f"Batch {progress.current_batch}/{progress.total_batches} | "
        f"{progress.processed_records}/{progress.total_records} records "
        f"({progress.percent_complete:.1f}%) | "
        f"{progress.records_per_second:.1f} rec/s",
        end="",
        flush=True,
    )


def cmd_archive(args):
    """Archive old data based on retention policies."""
    from data.retention import DataArchiver, DataType, get_retention_config, get_default_policies

    config = get_retention_config()
    archiver = DataArchiver(config)

    # Determine which data types to archive
    if args.data_type:
        try:
            data_types = [DataType(args.data_type)]
        except ValueError:
            print(f"Error: Invalid data type '{args.data_type}'")
            print(f"Valid types: {', '.join(dt.value for dt in DataType)}")
            sys.exit(1)
    else:
        policies = get_default_policies(config)
        data_types = [
            dt for dt, policy in policies.items()
            if policy.archive_before_delete
        ]

    print(f"Archive Operation {'(DRY RUN)' if args.dry_run else ''}")
    print("=" * 60)
    print(f"Archive path: {config.archive_path}")
    print(f"Archive format: {config.archive_format}")
    print(f"S3 enabled: {config.archive_to_s3}")
    print(f"Data types: {', '.join(dt.value for dt in data_types)}")
    print("=" * 60)
    print()

    results = []
    for data_type in data_types:
        print(f"Processing {data_type.value}...")
        result = archiver.archive_old_data(data_type, dry_run=args.dry_run)

        status = "SUCCESS" if result.success else "FAILED"
        print(f"  {status}: {result.record_count} records")
        if result.file_path:
            print(f"  File: {result.file_path}")
        if result.s3_key:
            print(f"  S3: s3://{config.s3_bucket}/{result.s3_key}")
        if result.error:
            print(f"  Error: {result.error}")
        print()

        results.append({
            "data_type": data_type.value,
            "success": result.success,
            "record_count": result.record_count,
            "duration_seconds": result.duration_seconds,
            "error": result.error,
        })

    # Summary
    print("=" * 60)
    total_records = sum(r["record_count"] for r in results)
    successful = sum(1 for r in results if r["success"])
    print(f"Total: {total_records} records archived from {successful}/{len(results)} data types")

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump({"results": results, "dry_run": args.dry_run}, f, indent=2)
        print(f"Results written to: {args.output_json}")


def cmd_clean(args):
    """Clean up expired data based on retention policies."""
    from data.retention import DataCleaner, DataType, get_retention_config, get_default_policies

    config = get_retention_config()
    cleaner = DataCleaner(config)

    # Add progress callback
    if args.verbose:
        cleaner.add_progress_callback(progress_callback)

    # Determine which data types to clean
    if args.data_type:
        try:
            data_types = [DataType(args.data_type)]
        except ValueError:
            print(f"Error: Invalid data type '{args.data_type}'")
            print(f"Valid types: {', '.join(dt.value for dt in DataType)}")
            sys.exit(1)
    else:
        data_types = None  # All types

    print(f"Cleanup Operation {'(DRY RUN)' if args.dry_run else ''}")
    print("=" * 60)

    # Show preview first
    print("Preview of records to process:")
    preview = cleaner.get_cleanup_preview(data_types)
    for dt, info in preview.items():
        count = info.get("records_to_process", 0)
        action = info.get("action", "unknown")
        reason = info.get("reason", "")
        if count > 0 or reason:
            print(f"  {dt}: {count} records ({action}){' - ' + reason if reason else ''}")
    print()

    if not args.dry_run and not args.force:
        confirm = input("Proceed with cleanup? (yes/no): ")
        if confirm.lower() != "yes":
            print("Cleanup cancelled.")
            sys.exit(0)

    print("=" * 60)
    print()

    # Run cleanup
    results = cleaner.clean_all_expired_data(
        dry_run=args.dry_run,
        force=args.force,
        data_types=data_types,
    )

    # Print results
    for data_type, result in results.items():
        status = "SUCCESS" if result.success else "FAILED"
        print(f"\n{data_type}: {status}")
        print(f"  Deleted: {result.total_deleted}")
        print(f"  Archived: {result.total_archived}")
        print(f"  Soft deleted: {result.total_soft_deleted}")
        print(f"  Errors: {result.total_errors}")
        print(f"  Duration: {result.duration_seconds:.2f}s")
        if result.error_messages:
            for msg in result.error_messages[:5]:
                print(f"  Error: {msg}")

    # Summary
    print()
    print("=" * 60)
    total_deleted = sum(r.total_deleted for r in results.values())
    total_archived = sum(r.total_archived for r in results.values())
    total_errors = sum(r.total_errors for r in results.values())
    print(f"Total: {total_deleted} deleted, {total_archived} archived, {total_errors} errors")

    if args.output_json:
        output = {
            "results": {k: v.to_dict() for k, v in results.items()},
            "dry_run": args.dry_run,
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results written to: {args.output_json}")


def cmd_restore(args):
    """Restore data from an archive."""
    from data.retention import DataArchiver, get_retention_config

    config = get_retention_config()
    archiver = DataArchiver(config)

    print(f"Restore Operation {'(DRY RUN)' if args.dry_run else ''}")
    print("=" * 60)
    print(f"Archive ID: {args.archive_id}")
    print()

    if not args.dry_run and not args.force:
        confirm = input("Proceed with restore? (yes/no): ")
        if confirm.lower() != "yes":
            print("Restore cancelled.")
            sys.exit(0)

    result = archiver.restore_from_archive(
        args.archive_id,
        dry_run=args.dry_run,
        skip_duplicates=not args.no_skip_duplicates,
    )

    if result.get("success"):
        print("Restore successful!")
        if args.dry_run:
            print(f"  Would restore: {result.get('would_restore', 0)} records")
        else:
            print(f"  Restored: {result.get('restored', 0)} records")
            print(f"  Skipped: {result.get('skipped', 0)} records")
            print(f"  Errors: {result.get('errors', 0)} records")
    else:
        print(f"Restore failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


def cmd_status(args):
    """Show retention status."""
    from data.retention import DataCleaner, get_retention_config, get_policies_summary

    config = get_retention_config()
    cleaner = DataCleaner(config)

    print("Retention Status")
    print("=" * 60)
    print()

    # Show configuration
    print("Configuration:")
    print(f"  Archive path: {config.archive_path}")
    print(f"  Archive format: {config.archive_format}")
    print(f"  S3 enabled: {config.archive_to_s3}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Soft delete: {config.soft_delete_enabled}")
    print(f"  Cleanup schedule: {config.cleanup_schedule}")
    print()

    # Show policies
    print("Retention Policies:")
    print("-" * 60)
    policies = get_policies_summary()
    for policy in policies:
        print(f"  {policy['data_type']}:")
        print(f"    Retention: {policy['retention_human']}")
        print(f"    Action: {policy['action']}")
        if policy['regulatory_requirement']:
            print(f"    ** Regulatory Requirement **")
        if policy['cutoff_date']:
            print(f"    Cutoff date: {policy['cutoff_date']}")
    print()

    # Show current status
    status = cleaner.get_retention_status()
    print("Current Status:")
    print("-" * 60)
    for data_type, info in status["data_types"].items():
        total = info.get("total_records", 0)
        expired = info.get("expired_records", 0)
        print(f"  {data_type}:")
        print(f"    Total records: {total}")
        print(f"    Expired records: {expired}")
        if "error" in info:
            print(f"    Error: {info['error']}")
    print()

    # Show preview
    print("Cleanup Preview (records eligible for processing):")
    print("-" * 60)
    preview = cleaner.get_cleanup_preview()
    for data_type, info in preview.items():
        count = info.get("records_to_process", 0)
        action = info.get("action", "unknown")
        if count > 0:
            print(f"  {data_type}: {count} records ({action})")
            if info.get("oldest_record"):
                print(f"    Date range: {info['oldest_record']} to {info['newest_record']}")

    if args.output_json:
        output = {
            "config": config.to_dict(),
            "policies": policies,
            "status": status,
            "preview": preview,
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nFull status written to: {args.output_json}")


def cmd_list_archives(args):
    """List available archives."""
    from data.retention import DataArchiver, DataType, get_retention_config

    config = get_retention_config()
    archiver = DataArchiver(config)

    # Filter by data type if specified
    data_type_filter = None
    if args.data_type:
        try:
            data_type_filter = DataType(args.data_type)
        except ValueError:
            print(f"Error: Invalid data type '{args.data_type}'")
            sys.exit(1)

    archives = archiver.list_archives(data_type=data_type_filter, limit=args.limit)

    if not archives:
        print("No archives found.")
        return

    print("Available Archives")
    print("=" * 80)
    print(f"{'Archive ID':<35} {'Type':<15} {'Records':<10} {'Size':<10} {'Date':<20}")
    print("-" * 80)

    for archive in archives:
        print(
            f"{archive['archive_id']:<35} "
            f"{archive['data_type']:<15} "
            f"{archive['record_count']:<10} "
            f"{archive['size_mb']:.2f} MB    "
            f"{archive['created_at'][:19]}"
        )

    print("-" * 80)
    print(f"Total: {len(archives)} archive(s)")

    # Show stats
    stats = archiver.get_archive_stats()
    print()
    print("Archive Statistics:")
    print(f"  Total archives: {stats['total_archives']}")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump({"archives": archives, "stats": stats}, f, indent=2)
        print(f"\nArchive list written to: {args.output_json}")


def cmd_verify(args):
    """Verify archive integrity."""
    from data.retention import DataArchiver, get_retention_config

    config = get_retention_config()
    archiver = DataArchiver(config)

    print(f"Verifying archive: {args.archive_id}")
    result = archiver.verify_archive(args.archive_id)

    if result.get("success"):
        print("Archive verification: PASSED")
    else:
        print("Archive verification: FAILED")
        for issue in result.get("issues", []):
            print(f"  - {issue}")
        sys.exit(1)


def cmd_delete_archive(args):
    """Delete an archive."""
    from data.retention import DataArchiver, get_retention_config

    config = get_retention_config()
    archiver = DataArchiver(config)

    if not args.force:
        confirm = input(f"Delete archive '{args.archive_id}'? (yes/no): ")
        if confirm.lower() != "yes":
            print("Delete cancelled.")
            sys.exit(0)

    success = archiver.delete_archive(
        args.archive_id,
        delete_from_s3=not args.keep_s3,
    )

    if success:
        print(f"Archive deleted: {args.archive_id}")
    else:
        print(f"Failed to delete archive: {args.archive_id}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="RDT Trading System Data Retention CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # archive command
    archive_parser = subparsers.add_parser("archive", help="Archive old data")
    archive_parser.add_argument(
        "--data-type", "-t",
        help="Specific data type to archive (default: all archivable types)",
    )
    archive_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview what would be archived (default)",
    )
    archive_parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Actually perform the archive",
    )
    archive_parser.add_argument(
        "--output-json", "-o",
        help="Write results to JSON file",
    )
    archive_parser.set_defaults(func=cmd_archive)

    # clean command
    clean_parser = subparsers.add_parser("clean", help="Clean up expired data")
    clean_parser.add_argument(
        "--data-type", "-t",
        help="Specific data type to clean (default: all)",
    )
    clean_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview what would be cleaned (default)",
    )
    clean_parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Actually perform the cleanup",
    )
    clean_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompts and safety checks",
    )
    clean_parser.add_argument(
        "--output-json", "-o",
        help="Write results to JSON file",
    )
    clean_parser.set_defaults(func=cmd_clean)

    # restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from archive")
    restore_parser.add_argument(
        "archive_id",
        help="ID of the archive to restore",
    )
    restore_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be restored",
    )
    restore_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompt",
    )
    restore_parser.add_argument(
        "--no-skip-duplicates",
        action="store_true",
        help="Don't skip records that already exist",
    )
    restore_parser.set_defaults(func=cmd_restore)

    # status command
    status_parser = subparsers.add_parser("status", help="Show retention status")
    status_parser.add_argument(
        "--output-json", "-o",
        help="Write status to JSON file",
    )
    status_parser.set_defaults(func=cmd_status)

    # list-archives command
    list_parser = subparsers.add_parser("list-archives", help="List available archives")
    list_parser.add_argument(
        "--data-type", "-t",
        help="Filter by data type",
    )
    list_parser.add_argument(
        "--limit", "-n",
        type=int,
        default=100,
        help="Maximum number of archives to list (default: 100)",
    )
    list_parser.add_argument(
        "--output-json", "-o",
        help="Write archive list to JSON file",
    )
    list_parser.set_defaults(func=cmd_list_archives)

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify archive integrity")
    verify_parser.add_argument(
        "archive_id",
        help="ID of the archive to verify",
    )
    verify_parser.set_defaults(func=cmd_verify)

    # delete-archive command
    delete_parser = subparsers.add_parser("delete-archive", help="Delete an archive")
    delete_parser.add_argument(
        "archive_id",
        help="ID of the archive to delete",
    )
    delete_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompt",
    )
    delete_parser.add_argument(
        "--keep-s3",
        action="store_true",
        help="Keep the S3 copy (only delete local)",
    )
    delete_parser.set_defaults(func=cmd_delete_archive)

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
