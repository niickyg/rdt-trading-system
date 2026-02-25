"""
Alert Delivery Tracker
Tracks alert delivery status and provides methods for retry management.
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
from loguru import logger


class DeliveryStatus(str, Enum):
    """Alert delivery status."""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class DeliveryAttempt:
    """Record of a single delivery attempt."""
    attempt_number: int
    timestamp: str
    success: bool
    error_message: Optional[str] = None
    retry_delay: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeliveryAttempt':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AlertRecord:
    """Record of an alert for delivery tracking."""
    alert_id: str
    channel: str
    title: str
    message: str
    priority: str
    status: DeliveryStatus
    created_at: str
    updated_at: str
    attempts: List[DeliveryAttempt] = field(default_factory=list)
    extra_data: Dict[str, Any] = field(default_factory=dict)
    max_attempts: int = 3
    next_retry_at: Optional[str] = None

    @property
    def attempt_count(self) -> int:
        """Get number of delivery attempts."""
        return len(self.attempts)

    @property
    def last_error(self) -> Optional[str]:
        """Get last error message if any."""
        if self.attempts:
            return self.attempts[-1].error_message
        return None

    @property
    def can_retry(self) -> bool:
        """Check if alert can be retried."""
        return (
            self.status in (DeliveryStatus.FAILED, DeliveryStatus.RETRYING) and
            self.attempt_count < self.max_attempts
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'alert_id': self.alert_id,
            'channel': self.channel,
            'title': self.title,
            'message': self.message,
            'priority': self.priority,
            'status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'attempts': [a.to_dict() for a in self.attempts],
            'extra_data': self.extra_data,
            'max_attempts': self.max_attempts,
            'next_retry_at': self.next_retry_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertRecord':
        """Create from dictionary."""
        attempts = [
            DeliveryAttempt.from_dict(a) for a in data.get('attempts', [])
        ]
        return cls(
            alert_id=data['alert_id'],
            channel=data['channel'],
            title=data['title'],
            message=data['message'],
            priority=data['priority'],
            status=DeliveryStatus(data['status']),
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            attempts=attempts,
            extra_data=data.get('extra_data', {}),
            max_attempts=data.get('max_attempts', 3),
            next_retry_at=data.get('next_retry_at'),
        )


class DeliveryTracker:
    """
    Tracks alert delivery status and manages retry queue.

    Stores delivery history in SQLite database for persistence and
    efficient querying of failed alerts.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        retention_days: int = 7
    ):
        """
        Initialize delivery tracker.

        Args:
            db_path: Path to SQLite database (defaults to data/delivery_tracker.db)
            retention_days: Days to retain delivery records before cleanup
        """
        if db_path is None:
            data_dir = Path(__file__).parent.parent / 'data'
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / 'delivery_tracker.db')

        self.db_path = db_path
        self.retention_days = retention_days
        self._init_database()

    @contextmanager
    def _get_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_records (
                    alert_id TEXT PRIMARY KEY,
                    channel TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    attempts TEXT NOT NULL DEFAULT '[]',
                    extra_data TEXT NOT NULL DEFAULT '{}',
                    max_attempts INTEGER DEFAULT 3,
                    next_retry_at TEXT
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_status
                ON alert_records(status)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_channel
                ON alert_records(channel)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON alert_records(created_at)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_next_retry
                ON alert_records(next_retry_at)
            ''')

    def create_record(
        self,
        channel: str,
        title: str,
        message: str,
        priority: str = 'normal',
        extra_data: Optional[Dict[str, Any]] = None,
        max_attempts: int = 3
    ) -> AlertRecord:
        """
        Create a new alert record for tracking.

        Args:
            channel: Alert channel name
            title: Alert title
            message: Alert message
            priority: Alert priority level
            extra_data: Additional data for the alert (e.g., discord_fields)
            max_attempts: Maximum delivery attempts

        Returns:
            AlertRecord: Created record
        """
        now = datetime.utcnow().isoformat()
        record = AlertRecord(
            alert_id=str(uuid.uuid4()),
            channel=channel,
            title=title,
            message=message,
            priority=priority,
            status=DeliveryStatus.PENDING,
            created_at=now,
            updated_at=now,
            extra_data=extra_data or {},
            max_attempts=max_attempts
        )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alert_records
                (alert_id, channel, title, message, priority, status,
                 created_at, updated_at, attempts, extra_data, max_attempts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.alert_id,
                record.channel,
                record.title,
                record.message,
                record.priority,
                record.status.value,
                record.created_at,
                record.updated_at,
                json.dumps([]),
                json.dumps(record.extra_data),
                record.max_attempts
            ))

        logger.debug(f"Created alert record: {record.alert_id} for {channel}")
        return record

    def record_attempt(
        self,
        alert_id: str,
        success: bool,
        error_message: Optional[str] = None,
        retry_delay: Optional[float] = None
    ) -> Optional[AlertRecord]:
        """
        Record a delivery attempt for an alert.

        Args:
            alert_id: Alert ID
            success: Whether the attempt was successful
            error_message: Error message if failed
            retry_delay: Delay before next retry if applicable

        Returns:
            AlertRecord: Updated record, or None if not found
        """
        record = self.get_record(alert_id)
        if not record:
            logger.warning(f"Alert record not found: {alert_id}")
            return None

        now = datetime.utcnow()
        attempt = DeliveryAttempt(
            attempt_number=record.attempt_count + 1,
            timestamp=now.isoformat(),
            success=success,
            error_message=error_message,
            retry_delay=retry_delay
        )

        record.attempts.append(attempt)
        record.updated_at = now.isoformat()

        if success:
            record.status = DeliveryStatus.SENT
            record.next_retry_at = None
        elif record.attempt_count >= record.max_attempts:
            record.status = DeliveryStatus.FAILED
            record.next_retry_at = None
            logger.warning(
                f"Alert {alert_id} failed permanently after {record.attempt_count} attempts"
            )
        else:
            record.status = DeliveryStatus.RETRYING
            if retry_delay:
                record.next_retry_at = (
                    now + timedelta(seconds=retry_delay)
                ).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE alert_records
                SET status = ?, updated_at = ?, attempts = ?, next_retry_at = ?
                WHERE alert_id = ?
            ''', (
                record.status.value,
                record.updated_at,
                json.dumps([a.to_dict() for a in record.attempts]),
                record.next_retry_at,
                alert_id
            ))

        logger.info(
            f"Recorded attempt {record.attempt_count} for alert {alert_id}: "
            f"{'success' if success else 'failed'}"
        )
        return record

    def get_record(self, alert_id: str) -> Optional[AlertRecord]:
        """
        Get an alert record by ID.

        Args:
            alert_id: Alert ID

        Returns:
            AlertRecord: Found record, or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM alert_records WHERE alert_id = ?',
                (alert_id,)
            )
            row = cursor.fetchone()

            if row:
                return self._row_to_record(row)
            return None

    def _row_to_record(self, row: sqlite3.Row) -> AlertRecord:
        """Convert database row to AlertRecord."""
        return AlertRecord(
            alert_id=row['alert_id'],
            channel=row['channel'],
            title=row['title'],
            message=row['message'],
            priority=row['priority'],
            status=DeliveryStatus(row['status']),
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            attempts=[
                DeliveryAttempt.from_dict(a)
                for a in json.loads(row['attempts'])
            ],
            extra_data=json.loads(row['extra_data']),
            max_attempts=row['max_attempts'],
            next_retry_at=row['next_retry_at']
        )

    def get_failed_alerts(
        self,
        channel: Optional[str] = None,
        include_retrying: bool = True,
        limit: int = 100
    ) -> List[AlertRecord]:
        """
        Get alerts that failed delivery.

        Args:
            channel: Filter by channel (optional)
            include_retrying: Include alerts in retrying status
            limit: Maximum number of records to return

        Returns:
            List[AlertRecord]: Failed alert records
        """
        statuses = [DeliveryStatus.FAILED.value]
        if include_retrying:
            statuses.append(DeliveryStatus.RETRYING.value)

        placeholders = ','.join('?' * len(statuses))

        with self._get_connection() as conn:
            cursor = conn.cursor()

            if channel:
                cursor.execute(f'''
                    SELECT * FROM alert_records
                    WHERE status IN ({placeholders}) AND channel = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (*statuses, channel, limit))
            else:
                cursor.execute(f'''
                    SELECT * FROM alert_records
                    WHERE status IN ({placeholders})
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (*statuses, limit))

            return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_pending_retries(self, limit: int = 100) -> List[AlertRecord]:
        """
        Get alerts that are due for retry.

        Args:
            limit: Maximum number of records to return

        Returns:
            List[AlertRecord]: Alerts ready for retry
        """
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM alert_records
                WHERE status = ?
                AND (next_retry_at IS NULL OR next_retry_at <= ?)
                ORDER BY created_at ASC
                LIMIT ?
            ''', (DeliveryStatus.RETRYING.value, now, limit))

            return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_alerts_by_status(
        self,
        status: DeliveryStatus,
        channel: Optional[str] = None,
        limit: int = 100
    ) -> List[AlertRecord]:
        """
        Get alerts by status.

        Args:
            status: Delivery status to filter by
            channel: Optional channel filter
            limit: Maximum number of records

        Returns:
            List[AlertRecord]: Matching alert records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if channel:
                cursor.execute('''
                    SELECT * FROM alert_records
                    WHERE status = ? AND channel = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (status.value, channel, limit))
            else:
                cursor.execute('''
                    SELECT * FROM alert_records
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (status.value, limit))

            return [self._row_to_record(row) for row in cursor.fetchall()]

    def mark_for_retry(
        self,
        alert_id: str,
        retry_delay: float = 0
    ) -> Optional[AlertRecord]:
        """
        Mark an alert for retry.

        Args:
            alert_id: Alert ID
            retry_delay: Delay before retry in seconds

        Returns:
            AlertRecord: Updated record, or None if not found
        """
        record = self.get_record(alert_id)
        if not record:
            return None

        if not record.can_retry:
            logger.warning(
                f"Alert {alert_id} cannot be retried "
                f"(status: {record.status}, attempts: {record.attempt_count})"
            )
            return record

        now = datetime.utcnow()
        record.status = DeliveryStatus.RETRYING
        record.updated_at = now.isoformat()
        record.next_retry_at = (
            now + timedelta(seconds=retry_delay)
        ).isoformat() if retry_delay > 0 else now.isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE alert_records
                SET status = ?, updated_at = ?, next_retry_at = ?
                WHERE alert_id = ?
            ''', (
                record.status.value,
                record.updated_at,
                record.next_retry_at,
                alert_id
            ))

        logger.info(f"Marked alert {alert_id} for retry")
        return record

    def reset_for_retry(
        self,
        alert_id: str,
        reset_attempts: bool = False
    ) -> Optional[AlertRecord]:
        """
        Reset an alert for manual retry.

        Args:
            alert_id: Alert ID
            reset_attempts: Whether to reset attempt count

        Returns:
            AlertRecord: Updated record, or None if not found
        """
        record = self.get_record(alert_id)
        if not record:
            return None

        now = datetime.utcnow()
        record.status = DeliveryStatus.RETRYING
        record.updated_at = now.isoformat()
        record.next_retry_at = now.isoformat()

        if reset_attempts:
            record.attempts = []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE alert_records
                SET status = ?, updated_at = ?, next_retry_at = ?, attempts = ?
                WHERE alert_id = ?
            ''', (
                record.status.value,
                record.updated_at,
                record.next_retry_at,
                json.dumps([a.to_dict() for a in record.attempts]),
                alert_id
            ))

        logger.info(f"Reset alert {alert_id} for retry (reset_attempts={reset_attempts})")
        return record

    def get_delivery_stats(
        self,
        since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get delivery statistics.

        Args:
            since: Only count records since this datetime

        Returns:
            Dict: Delivery statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if since:
                since_str = since.isoformat()
                cursor.execute('''
                    SELECT status, COUNT(*) as count
                    FROM alert_records
                    WHERE created_at >= ?
                    GROUP BY status
                ''', (since_str,))
            else:
                cursor.execute('''
                    SELECT status, COUNT(*) as count
                    FROM alert_records
                    GROUP BY status
                ''')

            status_counts = {row['status']: row['count'] for row in cursor.fetchall()}

            if since:
                cursor.execute('''
                    SELECT channel, COUNT(*) as count
                    FROM alert_records
                    WHERE created_at >= ?
                    GROUP BY channel
                ''', (since_str,))
            else:
                cursor.execute('''
                    SELECT channel, COUNT(*) as count
                    FROM alert_records
                    GROUP BY channel
                ''')

            channel_counts = {row['channel']: row['count'] for row in cursor.fetchall()}

            total = sum(status_counts.values())
            sent = status_counts.get(DeliveryStatus.SENT.value, 0)
            failed = status_counts.get(DeliveryStatus.FAILED.value, 0)

            return {
                'total': total,
                'by_status': status_counts,
                'by_channel': channel_counts,
                'success_rate': (sent / total * 100) if total > 0 else 0,
                'failure_rate': (failed / total * 100) if total > 0 else 0,
            }

    def cleanup_old_records(
        self,
        retention_days: Optional[int] = None
    ) -> int:
        """
        Delete old delivery records.

        Args:
            retention_days: Days to retain (uses instance default if not provided)

        Returns:
            int: Number of records deleted
        """
        if retention_days is None:
            retention_days = self.retention_days

        cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                'SELECT COUNT(*) FROM alert_records WHERE created_at < ?',
                (cutoff,)
            )
            count = cursor.fetchone()[0]

            cursor.execute(
                'DELETE FROM alert_records WHERE created_at < ?',
                (cutoff,)
            )

        logger.info(f"Cleaned up {count} delivery records older than {retention_days} days")
        return count

    def delete_record(self, alert_id: str) -> bool:
        """
        Delete a specific alert record.

        Args:
            alert_id: Alert ID to delete

        Returns:
            bool: True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM alert_records WHERE alert_id = ?',
                (alert_id,)
            )
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Deleted alert record: {alert_id}")
        return deleted


_tracker_instance: Optional[DeliveryTracker] = None


def get_delivery_tracker(db_path: Optional[str] = None) -> DeliveryTracker:
    """
    Get or create the global delivery tracker instance.

    Args:
        db_path: Optional database path (only used on first call)

    Returns:
        DeliveryTracker: Tracker instance
    """
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = DeliveryTracker(db_path=db_path)
    return _tracker_instance
