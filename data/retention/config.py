"""
Configuration for data retention policies.

Loads retention settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

from loguru import logger


@dataclass
class RetentionConfig:
    """Configuration for data retention policies."""

    # Retention periods (in days)
    signals_days: int = 90
    trades_days: int = 0  # 0 = keep all (regulatory requirement)
    positions_days_after_close: int = 365
    daily_stats_days: int = 730  # 2 years
    audit_logs_days: int = 2555  # 7 years
    sessions_inactive_days: int = 30
    api_logs_days: int = 90
    ml_predictions_days: int = 180
    market_data_cache_days: int = 7
    order_executions_days: int = 365

    # Archive settings
    archive_path: str = "./archives"
    archive_format: str = "json_gz"  # json, json_gz, parquet
    archive_to_s3: bool = False
    s3_bucket: Optional[str] = None
    s3_prefix: str = "rdt-archives"
    s3_region: str = "us-east-1"

    # Archive database settings
    archive_db_enabled: bool = False
    archive_db_url: Optional[str] = None

    # Cleanup settings
    batch_size: int = 1000
    soft_delete_enabled: bool = True
    soft_delete_field: str = "deleted_at"
    cleanup_schedule: str = "0 3 * * *"  # Daily at 3 AM

    # Safety settings
    dry_run_default: bool = True
    require_confirmation: bool = True
    min_records_threshold: int = 100  # Don't delete if fewer records than this

    # Notification settings
    notify_on_completion: bool = True
    notify_on_error: bool = True
    slack_webhook_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "RetentionConfig":
        """
        Load configuration from environment variables.

        Environment variables:
            RETENTION_SIGNALS_DAYS: Days to keep signals (default: 90)
            RETENTION_TRADES_DAYS: Days to keep trades (default: 0 = all)
            RETENTION_POSITIONS_DAYS: Days after close to keep positions (default: 365)
            RETENTION_DAILY_STATS_DAYS: Days to keep daily stats (default: 730)
            RETENTION_AUDIT_LOGS_DAYS: Days to keep audit logs (default: 2555)
            RETENTION_SESSIONS_DAYS: Days to keep inactive sessions (default: 30)
            RETENTION_API_LOGS_DAYS: Days to keep API logs (default: 90)
            RETENTION_ML_PREDICTIONS_DAYS: Days to keep ML predictions (default: 180)
            RETENTION_MARKET_DATA_CACHE_DAYS: Days to keep market data cache (default: 7)
            RETENTION_ORDER_EXECUTIONS_DAYS: Days to keep order executions (default: 365)

            RETENTION_ARCHIVE_PATH: Path for archive files (default: ./archives)
            RETENTION_ARCHIVE_FORMAT: Archive format (json, json_gz, parquet)
            RETENTION_ARCHIVE_TO_S3: Whether to archive to S3 (default: false)
            RETENTION_S3_BUCKET: S3 bucket name
            RETENTION_S3_PREFIX: S3 key prefix (default: rdt-archives)
            RETENTION_S3_REGION: AWS region (default: us-east-1)

            RETENTION_ARCHIVE_DB_ENABLED: Use separate archive database (default: false)
            RETENTION_ARCHIVE_DB_URL: Archive database connection string

            RETENTION_BATCH_SIZE: Batch size for operations (default: 1000)
            RETENTION_SOFT_DELETE: Enable soft delete (default: true)
            RETENTION_CLEANUP_SCHEDULE: Cron schedule (default: 0 3 * * *)

            RETENTION_DRY_RUN_DEFAULT: Default to dry run (default: true)
            RETENTION_REQUIRE_CONFIRMATION: Require confirmation (default: true)
            RETENTION_MIN_RECORDS_THRESHOLD: Min records before cleanup (default: 100)

            RETENTION_NOTIFY_COMPLETION: Notify on completion (default: true)
            RETENTION_NOTIFY_ERROR: Notify on error (default: true)
            RETENTION_SLACK_WEBHOOK: Slack webhook URL for notifications
        """
        def get_int(key: str, default: int) -> int:
            val = os.environ.get(key)
            if val:
                try:
                    return int(val)
                except ValueError:
                    logger.warning(f"Invalid integer for {key}: {val}, using default {default}")
            return default

        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, "").lower()
            if val in ("true", "1", "yes", "on"):
                return True
            if val in ("false", "0", "no", "off"):
                return False
            return default

        def get_str(key: str, default: Optional[str] = None) -> Optional[str]:
            return os.environ.get(key, default)

        return cls(
            # Retention periods
            signals_days=get_int("RETENTION_SIGNALS_DAYS", 90),
            trades_days=get_int("RETENTION_TRADES_DAYS", 0),
            positions_days_after_close=get_int("RETENTION_POSITIONS_DAYS", 365),
            daily_stats_days=get_int("RETENTION_DAILY_STATS_DAYS", 730),
            audit_logs_days=get_int("RETENTION_AUDIT_LOGS_DAYS", 2555),
            sessions_inactive_days=get_int("RETENTION_SESSIONS_DAYS", 30),
            api_logs_days=get_int("RETENTION_API_LOGS_DAYS", 90),
            ml_predictions_days=get_int("RETENTION_ML_PREDICTIONS_DAYS", 180),
            market_data_cache_days=get_int("RETENTION_MARKET_DATA_CACHE_DAYS", 7),
            order_executions_days=get_int("RETENTION_ORDER_EXECUTIONS_DAYS", 365),

            # Archive settings
            archive_path=get_str("RETENTION_ARCHIVE_PATH", "./archives"),
            archive_format=get_str("RETENTION_ARCHIVE_FORMAT", "json_gz"),
            archive_to_s3=get_bool("RETENTION_ARCHIVE_TO_S3", False),
            s3_bucket=get_str("RETENTION_S3_BUCKET"),
            s3_prefix=get_str("RETENTION_S3_PREFIX", "rdt-archives"),
            s3_region=get_str("RETENTION_S3_REGION", "us-east-1"),

            # Archive database
            archive_db_enabled=get_bool("RETENTION_ARCHIVE_DB_ENABLED", False),
            archive_db_url=get_str("RETENTION_ARCHIVE_DB_URL"),

            # Cleanup settings
            batch_size=get_int("RETENTION_BATCH_SIZE", 1000),
            soft_delete_enabled=get_bool("RETENTION_SOFT_DELETE", True),
            cleanup_schedule=get_str("RETENTION_CLEANUP_SCHEDULE", "0 3 * * *"),

            # Safety settings
            dry_run_default=get_bool("RETENTION_DRY_RUN_DEFAULT", True),
            require_confirmation=get_bool("RETENTION_REQUIRE_CONFIRMATION", True),
            min_records_threshold=get_int("RETENTION_MIN_RECORDS_THRESHOLD", 100),

            # Notifications
            notify_on_completion=get_bool("RETENTION_NOTIFY_COMPLETION", True),
            notify_on_error=get_bool("RETENTION_NOTIFY_ERROR", True),
            slack_webhook_url=get_str("RETENTION_SLACK_WEBHOOK"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "retention_periods": {
                "signals_days": self.signals_days,
                "trades_days": self.trades_days,
                "positions_days_after_close": self.positions_days_after_close,
                "daily_stats_days": self.daily_stats_days,
                "audit_logs_days": self.audit_logs_days,
                "sessions_inactive_days": self.sessions_inactive_days,
                "api_logs_days": self.api_logs_days,
                "ml_predictions_days": self.ml_predictions_days,
                "market_data_cache_days": self.market_data_cache_days,
                "order_executions_days": self.order_executions_days,
            },
            "archive": {
                "path": self.archive_path,
                "format": self.archive_format,
                "s3_enabled": self.archive_to_s3,
                "s3_bucket": self.s3_bucket,
                "s3_prefix": self.s3_prefix,
                "s3_region": self.s3_region,
                "archive_db_enabled": self.archive_db_enabled,
            },
            "cleanup": {
                "batch_size": self.batch_size,
                "soft_delete_enabled": self.soft_delete_enabled,
                "schedule": self.cleanup_schedule,
            },
            "safety": {
                "dry_run_default": self.dry_run_default,
                "require_confirmation": self.require_confirmation,
                "min_records_threshold": self.min_records_threshold,
            },
        }

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            bool: True if valid, raises ValueError otherwise
        """
        # Check archive path is writable
        archive_path = Path(self.archive_path)
        if not self.archive_to_s3:
            try:
                archive_path.mkdir(parents=True, exist_ok=True)
                test_file = archive_path / ".test_write"
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                raise ValueError(f"Archive path {self.archive_path} is not writable: {e}")

        # Check S3 configuration if enabled
        if self.archive_to_s3 and not self.s3_bucket:
            raise ValueError("S3 bucket must be specified when archive_to_s3 is enabled")

        # Check archive database if enabled
        if self.archive_db_enabled and not self.archive_db_url:
            raise ValueError("Archive database URL must be specified when archive_db_enabled is enabled")

        # Check archive format
        valid_formats = {"json", "json_gz", "parquet"}
        if self.archive_format not in valid_formats:
            raise ValueError(f"Invalid archive format: {self.archive_format}. Must be one of {valid_formats}")

        # Check retention periods are non-negative
        for field_name in ["signals_days", "positions_days_after_close", "daily_stats_days",
                          "audit_logs_days", "sessions_inactive_days", "api_logs_days",
                          "ml_predictions_days", "market_data_cache_days", "order_executions_days"]:
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")

        return True


# Global configuration instance
_retention_config: Optional[RetentionConfig] = None


def get_retention_config() -> RetentionConfig:
    """
    Get or create the global retention configuration.

    Returns:
        RetentionConfig: The retention configuration
    """
    global _retention_config
    if _retention_config is None:
        _retention_config = RetentionConfig.from_env()
    return _retention_config


def reset_retention_config():
    """Reset the global retention configuration (useful for testing)."""
    global _retention_config
    _retention_config = None
