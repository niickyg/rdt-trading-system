"""
Data Retention Policies for RDT Trading System.

Defines retention rules for different data types including:
- Retention periods
- Archive vs delete actions
- Regulatory requirements
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

from .config import get_retention_config, RetentionConfig

from loguru import logger


class DataType(str, Enum):
    """Types of data subject to retention policies."""
    SIGNALS = "signals"
    TRADES = "trades"
    POSITIONS = "positions"
    DAILY_STATS = "daily_stats"
    AUDIT_LOGS = "audit_logs"
    SESSIONS = "sessions"
    API_LOGS = "api_logs"
    ML_PREDICTIONS = "ml_predictions"
    MARKET_DATA_CACHE = "market_data_cache"
    ORDER_EXECUTIONS = "order_executions"
    API_USERS = "api_users"
    PAYMENT_HISTORY = "payment_history"


class RetentionAction(str, Enum):
    """Actions to take on expired data."""
    ARCHIVE = "archive"  # Archive before deletion
    DELETE = "delete"  # Delete without archiving
    KEEP = "keep"  # Keep indefinitely (regulatory requirement)
    SOFT_DELETE = "soft_delete"  # Mark as deleted but retain


@dataclass
class RetentionPolicy:
    """
    Defines retention rules for a specific data type.

    Attributes:
        data_type: The type of data this policy applies to
        retention_days: Number of days to retain data (0 = keep forever)
        action: What to do when data expires
        archive_before_delete: Whether to archive before hard delete
        regulatory_requirement: Whether this is a regulatory requirement
        date_field: Name of the date field to use for age calculation
        closed_date_field: For data that has a closed state, the field to check
        description: Human-readable description of this policy
        custom_filter: Optional callable for custom filtering logic
    """
    data_type: DataType
    retention_days: int
    action: RetentionAction
    archive_before_delete: bool = True
    regulatory_requirement: bool = False
    date_field: str = "timestamp"
    closed_date_field: Optional[str] = None
    description: str = ""
    custom_filter: Optional[Callable[[Any], bool]] = None

    def get_cutoff_date(self) -> Optional[datetime]:
        """
        Calculate the cutoff date for this policy.

        Returns:
            datetime: Records older than this should be processed, or None if keep forever
        """
        if self.retention_days == 0:
            return None  # Keep forever
        return datetime.utcnow() - timedelta(days=self.retention_days)

    def should_retain(self, record_date: datetime) -> bool:
        """
        Check if a record should be retained based on its date.

        Args:
            record_date: The date of the record

        Returns:
            bool: True if record should be retained, False if it should be processed
        """
        cutoff = self.get_cutoff_date()
        if cutoff is None:
            return True  # Keep forever
        return record_date > cutoff

    def is_regulatory(self) -> bool:
        """Check if this policy has regulatory requirements."""
        return self.regulatory_requirement

    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary representation."""
        return {
            "data_type": self.data_type.value,
            "retention_days": self.retention_days,
            "retention_human": self._format_retention_period(),
            "action": self.action.value,
            "archive_before_delete": self.archive_before_delete,
            "regulatory_requirement": self.regulatory_requirement,
            "date_field": self.date_field,
            "closed_date_field": self.closed_date_field,
            "description": self.description,
            "cutoff_date": self.get_cutoff_date().isoformat() if self.get_cutoff_date() else None,
        }

    def _format_retention_period(self) -> str:
        """Format retention period as human-readable string."""
        if self.retention_days == 0:
            return "Forever (regulatory requirement)" if self.regulatory_requirement else "Forever"

        if self.retention_days >= 365:
            years = self.retention_days // 365
            return f"{years} year{'s' if years > 1 else ''}"
        elif self.retention_days >= 30:
            months = self.retention_days // 30
            return f"{months} month{'s' if months > 1 else ''}"
        else:
            return f"{self.retention_days} day{'s' if self.retention_days > 1 else ''}"


def get_default_policies(config: Optional[RetentionConfig] = None) -> Dict[DataType, RetentionPolicy]:
    """
    Get the default retention policies for all data types.

    Args:
        config: Optional retention configuration (uses global config if not provided)

    Returns:
        Dict mapping DataType to RetentionPolicy
    """
    if config is None:
        config = get_retention_config()

    policies = {
        # Signals: 90 days active, archive older
        DataType.SIGNALS: RetentionPolicy(
            data_type=DataType.SIGNALS,
            retention_days=config.signals_days,
            action=RetentionAction.ARCHIVE,
            archive_before_delete=True,
            regulatory_requirement=False,
            date_field="timestamp",
            description="Trading signals - archive after 90 days for historical analysis",
        ),

        # Trades: Keep all (regulatory requirement)
        DataType.TRADES: RetentionPolicy(
            data_type=DataType.TRADES,
            retention_days=config.trades_days,  # 0 = keep all
            action=RetentionAction.KEEP,
            archive_before_delete=False,
            regulatory_requirement=True,
            date_field="entry_time",
            description="Trade records - keep all for regulatory compliance (SEC/FINRA)",
        ),

        # Positions: 365 days after close
        DataType.POSITIONS: RetentionPolicy(
            data_type=DataType.POSITIONS,
            retention_days=config.positions_days_after_close,
            action=RetentionAction.ARCHIVE,
            archive_before_delete=True,
            regulatory_requirement=False,
            date_field="entry_time",
            closed_date_field="updated_at",  # Use updated_at as proxy for close date
            description="Closed positions - archive 1 year after closing",
        ),

        # Daily Stats: 2 years
        DataType.DAILY_STATS: RetentionPolicy(
            data_type=DataType.DAILY_STATS,
            retention_days=config.daily_stats_days,
            action=RetentionAction.ARCHIVE,
            archive_before_delete=True,
            regulatory_requirement=False,
            date_field="date",
            description="Daily trading statistics - archive after 2 years",
        ),

        # Audit Logs: 7 years (regulatory requirement)
        DataType.AUDIT_LOGS: RetentionPolicy(
            data_type=DataType.AUDIT_LOGS,
            retention_days=config.audit_logs_days,
            action=RetentionAction.ARCHIVE,
            archive_before_delete=True,
            regulatory_requirement=True,
            date_field="timestamp",
            description="Audit logs - keep 7 years for compliance (SOX/regulatory)",
        ),

        # Sessions: 30 days inactive
        DataType.SESSIONS: RetentionPolicy(
            data_type=DataType.SESSIONS,
            retention_days=config.sessions_inactive_days,
            action=RetentionAction.DELETE,
            archive_before_delete=False,
            regulatory_requirement=False,
            date_field="last_login",
            description="User sessions - delete after 30 days of inactivity",
        ),

        # API Logs: 90 days
        DataType.API_LOGS: RetentionPolicy(
            data_type=DataType.API_LOGS,
            retention_days=config.api_logs_days,
            action=RetentionAction.ARCHIVE,
            archive_before_delete=True,
            regulatory_requirement=False,
            date_field="last_request_at",
            description="API access logs - archive after 90 days",
        ),

        # ML Predictions: 180 days
        DataType.ML_PREDICTIONS: RetentionPolicy(
            data_type=DataType.ML_PREDICTIONS,
            retention_days=config.ml_predictions_days,
            action=RetentionAction.ARCHIVE,
            archive_before_delete=True,
            regulatory_requirement=False,
            date_field="timestamp",
            description="ML model predictions - archive after 6 months",
        ),

        # Market Data Cache: 7 days
        DataType.MARKET_DATA_CACHE: RetentionPolicy(
            data_type=DataType.MARKET_DATA_CACHE,
            retention_days=config.market_data_cache_days,
            action=RetentionAction.DELETE,
            archive_before_delete=False,
            regulatory_requirement=False,
            date_field="expires_at",
            description="Market data cache - delete after 7 days (transient data)",
        ),

        # Order Executions: 365 days
        DataType.ORDER_EXECUTIONS: RetentionPolicy(
            data_type=DataType.ORDER_EXECUTIONS,
            retention_days=config.order_executions_days,
            action=RetentionAction.ARCHIVE,
            archive_before_delete=True,
            regulatory_requirement=False,
            date_field="fill_time",
            description="Order executions - archive after 1 year",
        ),

        # API Users: Keep active, archive inactive
        DataType.API_USERS: RetentionPolicy(
            data_type=DataType.API_USERS,
            retention_days=365,  # 1 year after expiration
            action=RetentionAction.SOFT_DELETE,
            archive_before_delete=True,
            regulatory_requirement=False,
            date_field="expires_at",
            description="API user accounts - soft delete 1 year after expiration",
        ),

        # Payment History: Keep all (financial records)
        DataType.PAYMENT_HISTORY: RetentionPolicy(
            data_type=DataType.PAYMENT_HISTORY,
            retention_days=0,  # Keep all
            action=RetentionAction.KEEP,
            archive_before_delete=False,
            regulatory_requirement=True,
            date_field="created_at",
            description="Payment history - keep all for financial compliance",
        ),
    }

    return policies


def get_policy_for_data_type(
    data_type: DataType,
    config: Optional[RetentionConfig] = None
) -> RetentionPolicy:
    """
    Get the retention policy for a specific data type.

    Args:
        data_type: The type of data
        config: Optional retention configuration

    Returns:
        RetentionPolicy: The policy for this data type
    """
    policies = get_default_policies(config)
    return policies.get(data_type)


def get_policies_summary() -> List[Dict[str, Any]]:
    """
    Get a summary of all retention policies.

    Returns:
        List of policy dictionaries
    """
    policies = get_default_policies()
    return [policy.to_dict() for policy in policies.values()]


def get_regulatory_policies() -> List[RetentionPolicy]:
    """
    Get all policies with regulatory requirements.

    Returns:
        List of policies with regulatory requirements
    """
    policies = get_default_policies()
    return [p for p in policies.values() if p.is_regulatory()]


def get_archivable_policies() -> List[RetentionPolicy]:
    """
    Get all policies that require archiving.

    Returns:
        List of policies that archive data
    """
    policies = get_default_policies()
    return [p for p in policies.values() if p.action == RetentionAction.ARCHIVE]


def validate_policies(policies: Dict[DataType, RetentionPolicy]) -> List[str]:
    """
    Validate a set of retention policies.

    Args:
        policies: Dictionary of policies to validate

    Returns:
        List of validation warnings/errors
    """
    warnings = []

    # Check for missing data types
    for data_type in DataType:
        if data_type not in policies:
            warnings.append(f"Missing policy for data type: {data_type.value}")

    # Check for very short retention periods on important data
    min_recommended = {
        DataType.TRADES: 365 * 7,  # 7 years
        DataType.AUDIT_LOGS: 365 * 7,  # 7 years
        DataType.DAILY_STATS: 365 * 2,  # 2 years
    }

    for data_type, min_days in min_recommended.items():
        if data_type in policies:
            policy = policies[data_type]
            if 0 < policy.retention_days < min_days:
                warnings.append(
                    f"Warning: {data_type.value} has short retention ({policy.retention_days} days), "
                    f"recommended minimum is {min_days} days"
                )

    # Check regulatory policies don't delete
    for data_type, policy in policies.items():
        if policy.regulatory_requirement and policy.action == RetentionAction.DELETE:
            warnings.append(
                f"Warning: {data_type.value} has regulatory requirement but action is DELETE. "
                f"Consider using ARCHIVE instead."
            )

    return warnings


class PolicyManager:
    """
    Manages retention policies for the trading system.

    Provides methods to:
    - Get policies for specific data types
    - Override policies temporarily
    - Export/import policy configurations
    """

    def __init__(self, config: Optional[RetentionConfig] = None):
        """
        Initialize the policy manager.

        Args:
            config: Optional retention configuration
        """
        self.config = config or get_retention_config()
        self._policies = get_default_policies(self.config)
        self._overrides: Dict[DataType, RetentionPolicy] = {}

    def get_policy(self, data_type: DataType) -> RetentionPolicy:
        """
        Get the effective policy for a data type.

        Overrides take precedence over defaults.

        Args:
            data_type: The type of data

        Returns:
            The effective retention policy
        """
        return self._overrides.get(data_type, self._policies.get(data_type))

    def set_override(self, data_type: DataType, policy: RetentionPolicy):
        """
        Set a temporary override for a data type's policy.

        Args:
            data_type: The type of data
            policy: The override policy
        """
        self._overrides[data_type] = policy
        logger.info(f"Set policy override for {data_type.value}: {policy.to_dict()}")

    def clear_override(self, data_type: DataType):
        """
        Clear an override for a data type.

        Args:
            data_type: The type of data
        """
        if data_type in self._overrides:
            del self._overrides[data_type]
            logger.info(f"Cleared policy override for {data_type.value}")

    def clear_all_overrides(self):
        """Clear all policy overrides."""
        self._overrides.clear()
        logger.info("Cleared all policy overrides")

    def get_all_policies(self) -> Dict[DataType, RetentionPolicy]:
        """
        Get all effective policies (including overrides).

        Returns:
            Dictionary of all effective policies
        """
        effective = dict(self._policies)
        effective.update(self._overrides)
        return effective

    def export_policies(self) -> Dict[str, Any]:
        """
        Export all policies to a serializable format.

        Returns:
            Dictionary representation of all policies
        """
        return {
            data_type.value: policy.to_dict()
            for data_type, policy in self.get_all_policies().items()
        }

    def validate(self) -> List[str]:
        """
        Validate all current policies.

        Returns:
            List of validation warnings
        """
        return validate_policies(self.get_all_policies())
