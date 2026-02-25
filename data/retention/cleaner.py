"""
Data Cleaner for RDT Trading System.

Removes expired data based on retention policies with support for:
- Soft delete vs hard delete
- Batch processing for large datasets
- Progress reporting
- Dry-run mode
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from sqlalchemy import and_, delete, update, func

from .config import get_retention_config, RetentionConfig
from .policies import (
    DataType,
    RetentionAction,
    RetentionPolicy,
    get_policy_for_data_type,
    get_default_policies,
)
from .archiver import DataArchiver

from loguru import logger


@dataclass
class CleanupProgress:
    """Progress information for cleanup operations."""
    data_type: str
    total_records: int
    processed_records: int
    deleted_records: int
    archived_records: int
    skipped_records: int
    error_count: int
    current_batch: int
    total_batches: int
    start_time: datetime
    elapsed_seconds: float = 0.0

    @property
    def percent_complete(self) -> float:
        """Calculate completion percentage."""
        if self.total_records == 0:
            return 100.0
        return (self.processed_records / self.total_records) * 100

    @property
    def records_per_second(self) -> float:
        """Calculate processing rate."""
        if self.elapsed_seconds == 0:
            return 0.0
        return self.processed_records / self.elapsed_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data_type": self.data_type,
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "deleted_records": self.deleted_records,
            "archived_records": self.archived_records,
            "skipped_records": self.skipped_records,
            "error_count": self.error_count,
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
            "percent_complete": round(self.percent_complete, 2),
            "records_per_second": round(self.records_per_second, 2),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""
    success: bool
    data_type: str
    total_deleted: int = 0
    total_archived: int = 0
    total_soft_deleted: int = 0
    total_skipped: int = 0
    total_errors: int = 0
    error_messages: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data_type": self.data_type,
            "total_deleted": self.total_deleted,
            "total_archived": self.total_archived,
            "total_soft_deleted": self.total_soft_deleted,
            "total_skipped": self.total_skipped,
            "total_errors": self.total_errors,
            "error_messages": self.error_messages[:10],  # Limit error messages
            "duration_seconds": round(self.duration_seconds, 2),
            "dry_run": self.dry_run,
        }


class DataCleaner:
    """
    Removes expired data based on retention policies.

    Supports:
    - Soft delete (mark as deleted but retain)
    - Hard delete (permanent removal)
    - Archive before delete
    - Batch processing for performance
    - Progress callbacks for monitoring
    """

    def __init__(
        self,
        config: Optional[RetentionConfig] = None,
        db_manager=None,
        archiver: Optional[DataArchiver] = None,
    ):
        """
        Initialize the cleaner.

        Args:
            config: Optional retention configuration
            db_manager: Optional database manager instance
            archiver: Optional data archiver instance
        """
        self.config = config or get_retention_config()
        self._db_manager = db_manager
        self._archiver = archiver
        self._progress_callbacks: List[Callable[[CleanupProgress], None]] = []

    @property
    def db_manager(self):
        """Lazy-load database manager."""
        if self._db_manager is None:
            from data.database import get_db_manager
            self._db_manager = get_db_manager()
        return self._db_manager

    @property
    def archiver(self) -> DataArchiver:
        """Lazy-load archiver."""
        if self._archiver is None:
            self._archiver = DataArchiver(self.config, self.db_manager)
        return self._archiver

    def add_progress_callback(self, callback: Callable[[CleanupProgress], None]):
        """
        Add a callback to receive progress updates.

        Args:
            callback: Function to call with progress updates
        """
        self._progress_callbacks.append(callback)

    def remove_progress_callback(self, callback: Callable[[CleanupProgress], None]):
        """
        Remove a progress callback.

        Args:
            callback: The callback to remove
        """
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)

    def _notify_progress(self, progress: CleanupProgress):
        """Send progress update to all callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

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
            DataType.SESSIONS: User,
        }
        return model_map.get(data_type)

    def clean_expired_data(
        self,
        data_type: DataType,
        policy: Optional[RetentionPolicy] = None,
        dry_run: bool = True,
        force: bool = False,
    ) -> CleanupResult:
        """
        Clean expired data for a specific data type.

        Args:
            data_type: Type of data to clean
            policy: Optional custom retention policy
            dry_run: If True, don't actually delete, just report what would happen
            force: If True, skip confirmation prompts and minimum record checks

        Returns:
            CleanupResult with details of the operation
        """
        start_time = time.time()

        if policy is None:
            policy = get_policy_for_data_type(data_type)

        if policy is None:
            return CleanupResult(
                success=False,
                data_type=data_type.value,
                error_messages=[f"No policy found for data type: {data_type.value}"],
            )

        # Check if policy allows deletion
        if policy.action == RetentionAction.KEEP:
            return CleanupResult(
                success=True,
                data_type=data_type.value,
                total_skipped=0,
                dry_run=dry_run,
            )

        cutoff_date = policy.get_cutoff_date()
        if cutoff_date is None:
            return CleanupResult(
                success=True,
                data_type=data_type.value,
                error_messages=["Policy specifies to keep data forever"],
                dry_run=dry_run,
            )

        model_class = self._get_model_class(data_type)
        if model_class is None:
            return CleanupResult(
                success=False,
                data_type=data_type.value,
                error_messages=[f"No model class for data type: {data_type.value}"],
            )

        logger.info(f"Starting cleanup for {data_type.value} (cutoff: {cutoff_date}, dry_run: {dry_run})")

        try:
            date_field = getattr(model_class, policy.date_field, None)
            if date_field is None:
                return CleanupResult(
                    success=False,
                    data_type=data_type.value,
                    error_messages=[f"Date field '{policy.date_field}' not found on model"],
                )

            with self.db_manager.get_session() as session:
                # Count total records to process
                base_filter = date_field < cutoff_date

                # For positions, also check if closed
                if data_type == DataType.POSITIONS and policy.closed_date_field:
                    closed_field = getattr(model_class, policy.closed_date_field, None)
                    if closed_field:
                        base_filter = and_(base_filter, closed_field.isnot(None))

                total_count = session.query(func.count(model_class.id)).filter(base_filter).scalar()

                if total_count == 0:
                    return CleanupResult(
                        success=True,
                        data_type=data_type.value,
                        total_deleted=0,
                        duration_seconds=time.time() - start_time,
                        dry_run=dry_run,
                    )

                # Safety check
                if not force and total_count < self.config.min_records_threshold:
                    logger.warning(
                        f"Only {total_count} records found for {data_type.value}, "
                        f"minimum threshold is {self.config.min_records_threshold}. "
                        f"Use force=True to override."
                    )

                logger.info(f"Found {total_count} records to process for {data_type.value}")

                if dry_run:
                    return CleanupResult(
                        success=True,
                        data_type=data_type.value,
                        total_deleted=total_count,
                        duration_seconds=time.time() - start_time,
                        dry_run=True,
                    )

                # Archive first if needed
                archived_count = 0
                if policy.archive_before_delete and policy.action in (
                    RetentionAction.ARCHIVE, RetentionAction.DELETE
                ):
                    archive_result = self.archiver.archive_old_data(
                        data_type, policy, dry_run=False
                    )
                    if archive_result.success:
                        archived_count = archive_result.record_count
                        logger.info(f"Archived {archived_count} records for {data_type.value}")
                    else:
                        logger.error(f"Archive failed: {archive_result.error}")
                        return CleanupResult(
                            success=False,
                            data_type=data_type.value,
                            error_messages=[f"Archive failed: {archive_result.error}"],
                        )

                # Perform cleanup based on action type
                if policy.action == RetentionAction.SOFT_DELETE:
                    result = self._soft_delete_records(
                        session, model_class, base_filter, total_count, data_type
                    )
                else:
                    result = self._hard_delete_records(
                        session, model_class, base_filter, total_count, data_type
                    )

                result.total_archived = archived_count
                result.duration_seconds = time.time() - start_time
                return result

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return CleanupResult(
                success=False,
                data_type=data_type.value,
                error_messages=[str(e)],
                duration_seconds=time.time() - start_time,
            )

    def _soft_delete_records(
        self,
        session,
        model_class,
        filter_condition,
        total_count: int,
        data_type: DataType,
    ) -> CleanupResult:
        """Soft delete records by setting a deleted_at timestamp."""
        deleted_count = 0
        error_count = 0
        batch_size = self.config.batch_size
        total_batches = (total_count + batch_size - 1) // batch_size

        # Check if model has soft delete field
        soft_delete_field = getattr(model_class, self.config.soft_delete_field, None)

        if soft_delete_field is None:
            # Fall back to is_active if available
            is_active_field = getattr(model_class, 'is_active', None)
            if is_active_field is not None:
                # Use is_active = False as soft delete
                try:
                    stmt = (
                        update(model_class)
                        .where(filter_condition)
                        .values(is_active=False)
                    )
                    result = session.execute(stmt)
                    deleted_count = result.rowcount
                    session.commit()

                    return CleanupResult(
                        success=True,
                        data_type=data_type.value,
                        total_soft_deleted=deleted_count,
                    )
                except Exception as e:
                    return CleanupResult(
                        success=False,
                        data_type=data_type.value,
                        error_messages=[f"Soft delete failed: {e}"],
                    )
            else:
                return CleanupResult(
                    success=False,
                    data_type=data_type.value,
                    error_messages=[
                        f"Model {model_class.__name__} does not have soft delete field "
                        f"'{self.config.soft_delete_field}' or 'is_active'"
                    ],
                )

        # Batch update with soft delete field
        progress = CleanupProgress(
            data_type=data_type.value,
            total_records=total_count,
            processed_records=0,
            deleted_records=0,
            archived_records=0,
            skipped_records=0,
            error_count=0,
            current_batch=0,
            total_batches=total_batches,
            start_time=datetime.utcnow(),
        )

        current_batch = 0
        while deleted_count < total_count:
            try:
                # Get batch of IDs
                ids_query = (
                    session.query(model_class.id)
                    .filter(filter_condition)
                    .filter(soft_delete_field.is_(None))
                    .limit(batch_size)
                )
                ids = [row[0] for row in ids_query.all()]

                if not ids:
                    break

                # Update batch
                stmt = (
                    update(model_class)
                    .where(model_class.id.in_(ids))
                    .values(**{self.config.soft_delete_field: datetime.utcnow()})
                )
                result = session.execute(stmt)
                session.commit()

                batch_deleted = result.rowcount
                deleted_count += batch_deleted
                current_batch += 1

                # Update progress
                progress.processed_records = deleted_count
                progress.deleted_records = deleted_count
                progress.current_batch = current_batch
                progress.elapsed_seconds = (datetime.utcnow() - progress.start_time).total_seconds()
                self._notify_progress(progress)

                logger.debug(f"Soft deleted batch {current_batch}/{total_batches}: {batch_deleted} records")

            except Exception as e:
                error_count += 1
                logger.error(f"Batch soft delete error: {e}")
                if error_count > 10:
                    break

        return CleanupResult(
            success=error_count == 0,
            data_type=data_type.value,
            total_soft_deleted=deleted_count,
            total_errors=error_count,
        )

    def _hard_delete_records(
        self,
        session,
        model_class,
        filter_condition,
        total_count: int,
        data_type: DataType,
    ) -> CleanupResult:
        """Hard delete records in batches."""
        deleted_count = 0
        error_count = 0
        batch_size = self.config.batch_size
        total_batches = (total_count + batch_size - 1) // batch_size

        progress = CleanupProgress(
            data_type=data_type.value,
            total_records=total_count,
            processed_records=0,
            deleted_records=0,
            archived_records=0,
            skipped_records=0,
            error_count=0,
            current_batch=0,
            total_batches=total_batches,
            start_time=datetime.utcnow(),
        )

        current_batch = 0
        while deleted_count < total_count:
            try:
                # Get batch of IDs to delete
                ids_query = (
                    session.query(model_class.id)
                    .filter(filter_condition)
                    .limit(batch_size)
                )
                ids = [row[0] for row in ids_query.all()]

                if not ids:
                    break

                # Delete batch
                stmt = delete(model_class).where(model_class.id.in_(ids))
                result = session.execute(stmt)
                session.commit()

                batch_deleted = result.rowcount
                deleted_count += batch_deleted
                current_batch += 1

                # Update progress
                progress.processed_records = deleted_count
                progress.deleted_records = deleted_count
                progress.current_batch = current_batch
                progress.elapsed_seconds = (datetime.utcnow() - progress.start_time).total_seconds()
                self._notify_progress(progress)

                logger.debug(f"Deleted batch {current_batch}/{total_batches}: {batch_deleted} records")

            except Exception as e:
                error_count += 1
                logger.error(f"Batch delete error: {e}")
                session.rollback()
                if error_count > 10:
                    break

        return CleanupResult(
            success=error_count == 0,
            data_type=data_type.value,
            total_deleted=deleted_count,
            total_errors=error_count,
        )

    def clean_all_expired_data(
        self,
        dry_run: bool = True,
        force: bool = False,
        data_types: Optional[List[DataType]] = None,
    ) -> Dict[str, CleanupResult]:
        """
        Clean expired data for all (or specified) data types.

        Args:
            dry_run: If True, don't actually delete
            force: If True, skip safety checks
            data_types: Optional list of specific data types to clean

        Returns:
            Dictionary mapping data type to cleanup result
        """
        policies = get_default_policies(self.config)
        results = {}

        types_to_clean = data_types or list(policies.keys())

        for data_type in types_to_clean:
            if data_type not in policies:
                logger.warning(f"No policy for data type: {data_type.value}")
                continue

            policy = policies[data_type]

            # Skip types that should be kept
            if policy.action == RetentionAction.KEEP:
                logger.info(f"Skipping {data_type.value}: policy is KEEP")
                results[data_type.value] = CleanupResult(
                    success=True,
                    data_type=data_type.value,
                    total_skipped=0,
                    dry_run=dry_run,
                )
                continue

            result = self.clean_expired_data(
                data_type=data_type,
                policy=policy,
                dry_run=dry_run,
                force=force,
            )
            results[data_type.value] = result

        return results

    def get_cleanup_preview(
        self,
        data_types: Optional[List[DataType]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get a preview of what would be cleaned up.

        Args:
            data_types: Optional list of specific data types to preview

        Returns:
            Dictionary with preview information for each data type
        """
        policies = get_default_policies(self.config)
        preview = {}

        types_to_preview = data_types or list(policies.keys())

        for data_type in types_to_preview:
            policy = policies.get(data_type)
            if not policy:
                continue

            cutoff_date = policy.get_cutoff_date()
            model_class = self._get_model_class(data_type)

            if not cutoff_date or not model_class:
                preview[data_type.value] = {
                    "action": policy.action.value,
                    "retention_days": policy.retention_days,
                    "records_to_process": 0,
                    "reason": "Keep forever" if not cutoff_date else "No model",
                }
                continue

            try:
                date_field = getattr(model_class, policy.date_field, None)
                if not date_field:
                    preview[data_type.value] = {
                        "action": policy.action.value,
                        "retention_days": policy.retention_days,
                        "records_to_process": 0,
                        "reason": f"Date field not found: {policy.date_field}",
                    }
                    continue

                with self.db_manager.get_session() as session:
                    filter_condition = date_field < cutoff_date

                    if data_type == DataType.POSITIONS and policy.closed_date_field:
                        closed_field = getattr(model_class, policy.closed_date_field, None)
                        if closed_field:
                            filter_condition = and_(filter_condition, closed_field.isnot(None))

                    count = session.query(func.count(model_class.id)).filter(
                        filter_condition
                    ).scalar()

                    # Get date range of records to be processed
                    oldest = session.query(func.min(date_field)).filter(
                        filter_condition
                    ).scalar()
                    newest = session.query(func.max(date_field)).filter(
                        filter_condition
                    ).scalar()

                    preview[data_type.value] = {
                        "action": policy.action.value,
                        "retention_days": policy.retention_days,
                        "cutoff_date": cutoff_date.isoformat(),
                        "records_to_process": count,
                        "oldest_record": oldest.isoformat() if oldest else None,
                        "newest_record": newest.isoformat() if newest else None,
                        "archive_before_delete": policy.archive_before_delete,
                        "regulatory_requirement": policy.regulatory_requirement,
                    }

            except Exception as e:
                preview[data_type.value] = {
                    "action": policy.action.value,
                    "retention_days": policy.retention_days,
                    "records_to_process": 0,
                    "error": str(e),
                }

        return preview

    def get_retention_status(self) -> Dict[str, Any]:
        """
        Get current retention status for all data types.

        Returns:
            Dictionary with retention status information
        """
        policies = get_default_policies(self.config)
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "batch_size": self.config.batch_size,
                "soft_delete_enabled": self.config.soft_delete_enabled,
                "dry_run_default": self.config.dry_run_default,
            },
            "data_types": {},
        }

        for data_type, policy in policies.items():
            model_class = self._get_model_class(data_type)

            type_status = {
                "policy": policy.to_dict(),
                "total_records": 0,
                "expired_records": 0,
            }

            if model_class:
                try:
                    with self.db_manager.get_session() as session:
                        total = session.query(func.count(model_class.id)).scalar()
                        type_status["total_records"] = total

                        cutoff = policy.get_cutoff_date()
                        if cutoff:
                            date_field = getattr(model_class, policy.date_field, None)
                            if date_field:
                                expired = session.query(func.count(model_class.id)).filter(
                                    date_field < cutoff
                                ).scalar()
                                type_status["expired_records"] = expired

                except Exception as e:
                    type_status["error"] = str(e)

            status["data_types"][data_type.value] = type_status

        return status
