"""
Data Archiver for RDT Trading System.

Archives old data to:
- Compressed JSON files
- AWS S3
- Separate archive database

Maintains a manifest of all archives for restoration.
"""

import gzip
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
from contextlib import contextmanager

from sqlalchemy import and_, text
from sqlalchemy.orm import Session

from .config import get_retention_config, RetentionConfig
from .policies import DataType, RetentionPolicy, get_policy_for_data_type

from loguru import logger


class ArchiveFormat(str, Enum):
    """Supported archive formats."""
    JSON = "json"
    JSON_GZ = "json_gz"
    PARQUET = "parquet"


@dataclass
class ArchiveManifestEntry:
    """Entry in the archive manifest."""
    archive_id: str
    data_type: str
    file_path: str
    s3_key: Optional[str]
    record_count: int
    date_range_start: str
    date_range_end: str
    created_at: str
    checksum: str
    size_bytes: int
    format: str
    compression: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchiveManifest:
    """Manifest tracking all archives."""
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    entries: List[ArchiveManifestEntry] = field(default_factory=list)

    def add_entry(self, entry: ArchiveManifestEntry):
        """Add an entry to the manifest."""
        self.entries.append(entry)
        self.updated_at = datetime.utcnow().isoformat()

    def find_entries(
        self,
        data_type: Optional[str] = None,
        date_start: Optional[datetime] = None,
        date_end: Optional[datetime] = None,
    ) -> List[ArchiveManifestEntry]:
        """Find entries matching criteria."""
        results = self.entries

        if data_type:
            results = [e for e in results if e.data_type == data_type]

        if date_start:
            results = [e for e in results if datetime.fromisoformat(e.date_range_end) >= date_start]

        if date_end:
            results = [e for e in results if datetime.fromisoformat(e.date_range_start) <= date_end]

        return results

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "entries": [
                {
                    "archive_id": e.archive_id,
                    "data_type": e.data_type,
                    "file_path": e.file_path,
                    "s3_key": e.s3_key,
                    "record_count": e.record_count,
                    "date_range_start": e.date_range_start,
                    "date_range_end": e.date_range_end,
                    "created_at": e.created_at,
                    "checksum": e.checksum,
                    "size_bytes": e.size_bytes,
                    "format": e.format,
                    "compression": e.compression,
                    "metadata": e.metadata,
                }
                for e in self.entries
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchiveManifest":
        """Create manifest from dictionary."""
        manifest = cls(
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
        )
        for entry_data in data.get("entries", []):
            entry = ArchiveManifestEntry(
                archive_id=entry_data["archive_id"],
                data_type=entry_data["data_type"],
                file_path=entry_data["file_path"],
                s3_key=entry_data.get("s3_key"),
                record_count=entry_data["record_count"],
                date_range_start=entry_data["date_range_start"],
                date_range_end=entry_data["date_range_end"],
                created_at=entry_data["created_at"],
                checksum=entry_data["checksum"],
                size_bytes=entry_data["size_bytes"],
                format=entry_data["format"],
                compression=entry_data.get("compression", False),
                metadata=entry_data.get("metadata", {}),
            )
            manifest.entries.append(entry)
        return manifest


@dataclass
class ArchiveResult:
    """Result of an archive operation."""
    success: bool
    archive_id: Optional[str] = None
    file_path: Optional[str] = None
    s3_key: Optional[str] = None
    record_count: int = 0
    error: Optional[str] = None
    duration_seconds: float = 0.0


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for database objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return super().default(obj)


class DataArchiver:
    """
    Archives old data based on retention policies.

    Supports archiving to:
    - Local compressed JSON files
    - AWS S3
    - Separate archive database
    """

    def __init__(
        self,
        config: Optional[RetentionConfig] = None,
        db_manager=None,
    ):
        """
        Initialize the archiver.

        Args:
            config: Optional retention configuration
            db_manager: Optional database manager instance
        """
        self.config = config or get_retention_config()
        self._db_manager = db_manager
        self._manifest: Optional[ArchiveManifest] = None
        self._s3_client = None

        # Setup paths
        self.archive_path = Path(self.config.archive_path)
        self.archive_path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.archive_path / "manifest.json"

        # Load existing manifest
        self._load_manifest()

    @property
    def db_manager(self):
        """Lazy-load database manager."""
        if self._db_manager is None:
            from data.database import get_db_manager
            self._db_manager = get_db_manager()
        return self._db_manager

    @property
    def s3_client(self):
        """Lazy-load S3 client."""
        if self._s3_client is None and self.config.archive_to_s3:
            try:
                import boto3
                self._s3_client = boto3.client(
                    's3',
                    region_name=self.config.s3_region,
                )
            except ImportError:
                logger.warning("boto3 not installed, S3 archiving disabled")
            except Exception as e:
                logger.error(f"Failed to create S3 client: {e}")
        return self._s3_client

    def _load_manifest(self):
        """Load the archive manifest from disk."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    data = json.load(f)
                self._manifest = ArchiveManifest.from_dict(data)
                logger.debug(f"Loaded manifest with {len(self._manifest.entries)} entries")
            except Exception as e:
                logger.error(f"Failed to load manifest: {e}")
                self._manifest = ArchiveManifest()
        else:
            self._manifest = ArchiveManifest()

    def _save_manifest(self):
        """Save the archive manifest to disk."""
        try:
            with open(self.manifest_path, 'w') as f:
                json.dump(self._manifest.to_dict(), f, indent=2)
            logger.debug("Saved archive manifest")
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")

    def _generate_archive_id(self, data_type: DataType) -> str:
        """Generate a unique archive ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{data_type.value}_{timestamp}"

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        import hashlib
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_model_class(self, data_type: DataType):
        """Get the SQLAlchemy model class for a data type."""
        from data.database.models import (
            Signal, Trade, Position, DailyStats, AuditLog,
            APIUser, MarketDataCache, OrderExecution, PaymentHistory, User
        )

        model_map = {
            DataType.SIGNALS: Signal,
            DataType.TRADES: Trade,
            DataType.POSITIONS: Position,
            DataType.DAILY_STATS: DailyStats,
            DataType.AUDIT_LOGS: AuditLog,
            DataType.API_USERS: APIUser,
            DataType.MARKET_DATA_CACHE: MarketDataCache,
            DataType.ORDER_EXECUTIONS: OrderExecution,
            DataType.PAYMENT_HISTORY: PaymentHistory,
            DataType.SESSIONS: User,  # Using User model for sessions
        }
        return model_map.get(data_type)

    def _model_to_dict(self, obj) -> Dict[str, Any]:
        """Convert a SQLAlchemy model instance to a dictionary."""
        result = {}
        for column in obj.__table__.columns:
            value = getattr(obj, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, date):
                value = value.isoformat()
            elif isinstance(value, Decimal):
                value = float(value)
            elif isinstance(value, Enum):
                value = value.value
            result[column.name] = value
        return result

    def archive_old_data(
        self,
        data_type: DataType,
        policy: Optional[RetentionPolicy] = None,
        dry_run: bool = False,
    ) -> ArchiveResult:
        """
        Archive old data based on retention policy.

        Args:
            data_type: Type of data to archive
            policy: Optional custom retention policy
            dry_run: If True, don't actually archive, just report what would happen

        Returns:
            ArchiveResult with details of the operation
        """
        import time
        start_time = time.time()

        if policy is None:
            policy = get_policy_for_data_type(data_type)

        if policy is None:
            return ArchiveResult(
                success=False,
                error=f"No policy found for data type: {data_type.value}",
            )

        cutoff_date = policy.get_cutoff_date()
        if cutoff_date is None:
            return ArchiveResult(
                success=True,
                record_count=0,
                error="Policy specifies to keep data forever",
            )

        model_class = self._get_model_class(data_type)
        if model_class is None:
            return ArchiveResult(
                success=False,
                error=f"No model class for data type: {data_type.value}",
            )

        archive_id = self._generate_archive_id(data_type)
        logger.info(f"Starting archive {archive_id} for {data_type.value} (cutoff: {cutoff_date})")

        try:
            records = []
            date_field = getattr(model_class, policy.date_field, None)

            if date_field is None:
                return ArchiveResult(
                    success=False,
                    error=f"Date field '{policy.date_field}' not found on model",
                )

            with self.db_manager.get_session() as session:
                # Query records older than cutoff
                query = session.query(model_class).filter(date_field < cutoff_date)

                # For positions, also check if closed
                if data_type == DataType.POSITIONS and policy.closed_date_field:
                    closed_field = getattr(model_class, policy.closed_date_field, None)
                    if closed_field:
                        query = query.filter(closed_field.isnot(None))

                records_to_archive = query.all()

                if not records_to_archive:
                    return ArchiveResult(
                        success=True,
                        record_count=0,
                        duration_seconds=time.time() - start_time,
                    )

                # Convert to dictionaries
                for record in records_to_archive:
                    records.append(self._model_to_dict(record))

                record_count = len(records)
                logger.info(f"Found {record_count} records to archive")

                if dry_run:
                    return ArchiveResult(
                        success=True,
                        archive_id=archive_id,
                        record_count=record_count,
                        duration_seconds=time.time() - start_time,
                    )

                # Determine date range
                date_values = [r.get(policy.date_field) for r in records if r.get(policy.date_field)]
                if date_values:
                    date_range_start = min(date_values)
                    date_range_end = max(date_values)
                else:
                    date_range_start = date_range_end = cutoff_date.isoformat()

                # Write archive file
                file_path = self._write_archive_file(archive_id, data_type, records)

                # Upload to S3 if configured
                s3_key = None
                if self.config.archive_to_s3:
                    s3_key = self._upload_to_s3(file_path, archive_id)

                # Calculate checksum
                checksum = self._calculate_checksum(file_path)
                size_bytes = file_path.stat().st_size

                # Create manifest entry
                entry = ArchiveManifestEntry(
                    archive_id=archive_id,
                    data_type=data_type.value,
                    file_path=str(file_path),
                    s3_key=s3_key,
                    record_count=record_count,
                    date_range_start=str(date_range_start),
                    date_range_end=str(date_range_end),
                    created_at=datetime.utcnow().isoformat(),
                    checksum=checksum,
                    size_bytes=size_bytes,
                    format=self.config.archive_format,
                    compression=self.config.archive_format == "json_gz",
                    metadata={
                        "cutoff_date": cutoff_date.isoformat(),
                        "policy_retention_days": policy.retention_days,
                    },
                )

                self._manifest.add_entry(entry)
                self._save_manifest()

                return ArchiveResult(
                    success=True,
                    archive_id=archive_id,
                    file_path=str(file_path),
                    s3_key=s3_key,
                    record_count=record_count,
                    duration_seconds=time.time() - start_time,
                )

        except Exception as e:
            logger.error(f"Archive failed: {e}")
            return ArchiveResult(
                success=False,
                archive_id=archive_id,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    def _write_archive_file(
        self,
        archive_id: str,
        data_type: DataType,
        records: List[Dict[str, Any]],
    ) -> Path:
        """Write records to an archive file."""
        # Determine file extension
        if self.config.archive_format == "json_gz":
            file_name = f"{archive_id}.json.gz"
        elif self.config.archive_format == "parquet":
            file_name = f"{archive_id}.parquet"
        else:
            file_name = f"{archive_id}.json"

        # Create subdirectory for data type
        type_dir = self.archive_path / data_type.value
        type_dir.mkdir(parents=True, exist_ok=True)
        file_path = type_dir / file_name

        # Write the file
        archive_data = {
            "archive_id": archive_id,
            "data_type": data_type.value,
            "created_at": datetime.utcnow().isoformat(),
            "record_count": len(records),
            "records": records,
        }

        if self.config.archive_format == "json_gz":
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                json.dump(archive_data, f, cls=JSONEncoder, indent=None)
        elif self.config.archive_format == "parquet":
            self._write_parquet(file_path, records)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(archive_data, f, cls=JSONEncoder, indent=2)

        logger.info(f"Wrote archive file: {file_path} ({file_path.stat().st_size} bytes)")
        return file_path

    def _write_parquet(self, file_path: Path, records: List[Dict[str, Any]]):
        """Write records to a Parquet file."""
        try:
            import pandas as pd
            df = pd.DataFrame(records)
            df.to_parquet(file_path, index=False, compression='gzip')
        except ImportError:
            raise ImportError("pandas and pyarrow required for Parquet format")

    def _upload_to_s3(self, file_path: Path, archive_id: str) -> Optional[str]:
        """Upload an archive file to S3."""
        if not self.s3_client or not self.config.s3_bucket:
            return None

        try:
            s3_key = f"{self.config.s3_prefix}/{file_path.parent.name}/{file_path.name}"

            self.s3_client.upload_file(
                str(file_path),
                self.config.s3_bucket,
                s3_key,
                ExtraArgs={'ServerSideEncryption': 'AES256'},
            )

            logger.info(f"Uploaded archive to S3: s3://{self.config.s3_bucket}/{s3_key}")
            return s3_key

        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return None

    def restore_from_archive(
        self,
        archive_id: str,
        dry_run: bool = False,
        skip_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """
        Restore data from an archive.

        Args:
            archive_id: ID of the archive to restore
            dry_run: If True, don't actually restore, just report what would happen
            skip_duplicates: Skip records that already exist in the database

        Returns:
            Dict with restore results
        """
        # Find archive in manifest
        entry = None
        for e in self._manifest.entries:
            if e.archive_id == archive_id:
                entry = e
                break

        if not entry:
            return {
                "success": False,
                "error": f"Archive not found: {archive_id}",
            }

        logger.info(f"Restoring archive {archive_id} ({entry.record_count} records)")

        try:
            # Load archive data
            file_path = Path(entry.file_path)
            records = self._load_archive_file(file_path, entry)

            if dry_run:
                return {
                    "success": True,
                    "archive_id": archive_id,
                    "would_restore": len(records),
                    "dry_run": True,
                }

            # Get model class
            data_type = DataType(entry.data_type)
            model_class = self._get_model_class(data_type)

            if model_class is None:
                return {
                    "success": False,
                    "error": f"No model class for data type: {entry.data_type}",
                }

            # Restore records
            restored = 0
            skipped = 0
            errors = 0

            with self.db_manager.get_session() as session:
                for record in records:
                    try:
                        # Check for existing record
                        if skip_duplicates and 'id' in record:
                            existing = session.query(model_class).filter(
                                model_class.id == record['id']
                            ).first()
                            if existing:
                                skipped += 1
                                continue

                        # Remove id to let database auto-generate if needed
                        record_data = {k: v for k, v in record.items() if k != 'id'}

                        # Create new record
                        new_record = model_class(**record_data)
                        session.add(new_record)
                        restored += 1

                    except Exception as e:
                        logger.warning(f"Failed to restore record: {e}")
                        errors += 1

                session.commit()

            logger.info(f"Restore complete: {restored} restored, {skipped} skipped, {errors} errors")

            return {
                "success": True,
                "archive_id": archive_id,
                "restored": restored,
                "skipped": skipped,
                "errors": errors,
            }

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return {
                "success": False,
                "archive_id": archive_id,
                "error": str(e),
            }

    def _load_archive_file(
        self,
        file_path: Path,
        entry: ArchiveManifestEntry
    ) -> List[Dict[str, Any]]:
        """Load records from an archive file."""
        # Try local file first
        if file_path.exists():
            return self._read_local_archive(file_path, entry)

        # Try S3 if configured
        if entry.s3_key and self.s3_client:
            return self._read_s3_archive(entry)

        raise FileNotFoundError(f"Archive file not found: {file_path}")

    def _read_local_archive(
        self,
        file_path: Path,
        entry: ArchiveManifestEntry
    ) -> List[Dict[str, Any]]:
        """Read archive from local file."""
        if entry.format == "json_gz" or str(file_path).endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        elif entry.format == "parquet":
            import pandas as pd
            df = pd.read_parquet(file_path)
            return df.to_dict('records')
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        return data.get("records", [])

    def _read_s3_archive(self, entry: ArchiveManifestEntry) -> List[Dict[str, Any]]:
        """Read archive from S3."""
        import io

        response = self.s3_client.get_object(
            Bucket=self.config.s3_bucket,
            Key=entry.s3_key,
        )

        body = response['Body'].read()

        if entry.format == "json_gz":
            data = json.loads(gzip.decompress(body).decode('utf-8'))
        elif entry.format == "parquet":
            import pandas as pd
            df = pd.read_parquet(io.BytesIO(body))
            return df.to_dict('records')
        else:
            data = json.loads(body.decode('utf-8'))

        return data.get("records", [])

    def list_archives(
        self,
        data_type: Optional[DataType] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List available archives.

        Args:
            data_type: Optional filter by data type
            limit: Maximum number of archives to return

        Returns:
            List of archive metadata dictionaries
        """
        entries = self._manifest.entries

        if data_type:
            entries = [e for e in entries if e.data_type == data_type.value]

        # Sort by created_at descending
        entries = sorted(entries, key=lambda e: e.created_at, reverse=True)

        return [
            {
                "archive_id": e.archive_id,
                "data_type": e.data_type,
                "record_count": e.record_count,
                "date_range": f"{e.date_range_start} to {e.date_range_end}",
                "created_at": e.created_at,
                "size_mb": round(e.size_bytes / (1024 * 1024), 2),
                "format": e.format,
                "file_path": e.file_path,
                "s3_key": e.s3_key,
            }
            for e in entries[:limit]
        ]

    def get_archive_stats(self) -> Dict[str, Any]:
        """
        Get statistics about archives.

        Returns:
            Dictionary with archive statistics
        """
        total_size = sum(e.size_bytes for e in self._manifest.entries)
        total_records = sum(e.record_count for e in self._manifest.entries)

        by_type = {}
        for entry in self._manifest.entries:
            if entry.data_type not in by_type:
                by_type[entry.data_type] = {
                    "count": 0,
                    "records": 0,
                    "size_bytes": 0,
                }
            by_type[entry.data_type]["count"] += 1
            by_type[entry.data_type]["records"] += entry.record_count
            by_type[entry.data_type]["size_bytes"] += entry.size_bytes

        return {
            "total_archives": len(self._manifest.entries),
            "total_records": total_records,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "by_data_type": by_type,
            "oldest_archive": min((e.created_at for e in self._manifest.entries), default=None),
            "newest_archive": max((e.created_at for e in self._manifest.entries), default=None),
        }

    def delete_archive(self, archive_id: str, delete_from_s3: bool = True) -> bool:
        """
        Delete an archive.

        Args:
            archive_id: ID of the archive to delete
            delete_from_s3: Whether to also delete from S3

        Returns:
            True if deleted successfully
        """
        entry = None
        entry_index = None
        for i, e in enumerate(self._manifest.entries):
            if e.archive_id == archive_id:
                entry = e
                entry_index = i
                break

        if not entry:
            logger.warning(f"Archive not found: {archive_id}")
            return False

        # Delete local file
        file_path = Path(entry.file_path)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted local archive file: {file_path}")

        # Delete from S3
        if delete_from_s3 and entry.s3_key and self.s3_client:
            try:
                self.s3_client.delete_object(
                    Bucket=self.config.s3_bucket,
                    Key=entry.s3_key,
                )
                logger.info(f"Deleted S3 archive: {entry.s3_key}")
            except Exception as e:
                logger.error(f"Failed to delete from S3: {e}")

        # Remove from manifest
        del self._manifest.entries[entry_index]
        self._save_manifest()

        return True

    def verify_archive(self, archive_id: str) -> Dict[str, Any]:
        """
        Verify an archive's integrity.

        Args:
            archive_id: ID of the archive to verify

        Returns:
            Dictionary with verification results
        """
        entry = None
        for e in self._manifest.entries:
            if e.archive_id == archive_id:
                entry = e
                break

        if not entry:
            return {
                "success": False,
                "error": f"Archive not found: {archive_id}",
            }

        file_path = Path(entry.file_path)
        issues = []

        # Check file exists
        if not file_path.exists():
            issues.append("Local file not found")
        else:
            # Verify checksum
            current_checksum = self._calculate_checksum(file_path)
            if current_checksum != entry.checksum:
                issues.append(f"Checksum mismatch: expected {entry.checksum}, got {current_checksum}")

            # Verify file size
            current_size = file_path.stat().st_size
            if current_size != entry.size_bytes:
                issues.append(f"Size mismatch: expected {entry.size_bytes}, got {current_size}")

            # Try to read the file
            try:
                records = self._read_local_archive(file_path, entry)
                if len(records) != entry.record_count:
                    issues.append(f"Record count mismatch: expected {entry.record_count}, got {len(records)}")
            except Exception as e:
                issues.append(f"Failed to read archive: {e}")

        return {
            "success": len(issues) == 0,
            "archive_id": archive_id,
            "issues": issues,
        }
