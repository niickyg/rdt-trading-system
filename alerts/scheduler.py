"""
Alert Scheduler - Quiet Hours and Alert Scheduling

Provides functionality to manage alert delivery schedules including:
- Quiet hours (no alerts during specified times)
- Per-channel schedules
- Per-alert-type schedules
- Day-of-week restrictions
- Alert queuing during quiet hours
"""

import os
from datetime import datetime, time, timedelta
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from loguru import logger

try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
    logger.warning("pytz not available, timezone handling will be limited")


class DayOfWeek(str, Enum):
    """Days of the week."""
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"

    @classmethod
    def from_weekday(cls, weekday: int) -> "DayOfWeek":
        """Convert Python weekday (0=Monday) to DayOfWeek."""
        mapping = {
            0: cls.MONDAY,
            1: cls.TUESDAY,
            2: cls.WEDNESDAY,
            3: cls.THURSDAY,
            4: cls.FRIDAY,
            5: cls.SATURDAY,
            6: cls.SUNDAY,
        }
        return mapping[weekday]

    @classmethod
    def all_days(cls) -> List["DayOfWeek"]:
        """Get all days of the week."""
        return list(cls)

    @classmethod
    def weekdays(cls) -> List["DayOfWeek"]:
        """Get weekdays only."""
        return [cls.MONDAY, cls.TUESDAY, cls.WEDNESDAY, cls.THURSDAY, cls.FRIDAY]

    @classmethod
    def weekends(cls) -> List["DayOfWeek"]:
        """Get weekends only."""
        return [cls.SATURDAY, cls.SUNDAY]


@dataclass
class QuietHoursRule:
    """
    Represents a quiet hours rule.

    Attributes:
        start_time: Start of quiet hours (HH:MM format or time object)
        end_time: End of quiet hours (HH:MM format or time object)
        days: Days of week this rule applies to (None = all days)
        channels: Channels this rule applies to (None = all channels)
        alert_types: Alert types this rule applies to (None = all types)
        enabled: Whether this rule is active
        name: Optional name for the rule
    """
    start_time: time
    end_time: time
    days: Optional[List[DayOfWeek]] = None
    channels: Optional[List[str]] = None
    alert_types: Optional[List[str]] = None
    enabled: bool = True
    name: Optional[str] = None
    rule_id: Optional[str] = None

    def __post_init__(self):
        """Convert string times to time objects if needed."""
        if isinstance(self.start_time, str):
            self.start_time = datetime.strptime(self.start_time, "%H:%M").time()
        if isinstance(self.end_time, str):
            self.end_time = datetime.strptime(self.end_time, "%H:%M").time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_time": self.start_time.strftime("%H:%M"),
            "end_time": self.end_time.strftime("%H:%M"),
            "days": [d.value for d in self.days] if self.days else None,
            "channels": self.channels,
            "alert_types": self.alert_types,
            "enabled": self.enabled,
            "name": self.name,
            "rule_id": self.rule_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuietHoursRule":
        """Create from dictionary."""
        days = None
        if data.get("days"):
            days = [DayOfWeek(d) for d in data["days"]]

        return cls(
            start_time=data["start_time"],
            end_time=data["end_time"],
            days=days,
            channels=data.get("channels"),
            alert_types=data.get("alert_types"),
            enabled=data.get("enabled", True),
            name=data.get("name"),
            rule_id=data.get("rule_id"),
        )


@dataclass
class QueuedAlert:
    """
    Represents an alert queued for later delivery.

    Attributes:
        alert_id: Unique identifier for this queued alert
        title: Alert title
        message: Alert message body
        priority: Alert priority level
        channels: Target channels for delivery
        alert_type: Type of alert (trade, signal, system, etc.)
        extra_data: Additional data for the alert
        queued_at: When the alert was queued
        scheduled_for: Earliest time to send the alert
        user_id: Optional user ID for user-specific alerts
        attempts: Number of delivery attempts
    """
    alert_id: str
    title: str
    message: str
    priority: str
    channels: List[str]
    alert_type: str
    queued_at: datetime
    scheduled_for: Optional[datetime] = None
    extra_data: Optional[Dict[str, Any]] = None
    user_id: Optional[int] = None
    attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "priority": self.priority,
            "channels": self.channels,
            "alert_type": self.alert_type,
            "queued_at": self.queued_at.isoformat(),
            "scheduled_for": self.scheduled_for.isoformat() if self.scheduled_for else None,
            "extra_data": self.extra_data,
            "user_id": self.user_id,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueuedAlert":
        """Create from dictionary."""
        return cls(
            alert_id=data["alert_id"],
            title=data["title"],
            message=data["message"],
            priority=data["priority"],
            channels=data["channels"],
            alert_type=data["alert_type"],
            queued_at=datetime.fromisoformat(data["queued_at"]),
            scheduled_for=datetime.fromisoformat(data["scheduled_for"]) if data.get("scheduled_for") else None,
            extra_data=data.get("extra_data"),
            user_id=data.get("user_id"),
            attempts=data.get("attempts", 0),
        )


class AlertScheduler:
    """
    Manages alert schedules and quiet hours.

    Provides functionality to:
    - Define quiet hours rules
    - Check if alerts should be suppressed
    - Queue alerts for later delivery
    - Process queued alerts when quiet hours end
    """

    def __init__(
        self,
        timezone: Optional[str] = None,
        queue_file: Optional[str] = None,
        critical_bypass: bool = True,
    ):
        """
        Initialize the alert scheduler.

        Args:
            timezone: Timezone for schedule evaluation (default: from env or America/New_York)
            queue_file: Path to file for persistent queue storage
            critical_bypass: Whether critical alerts bypass quiet hours
        """
        self._timezone_str = timezone or os.environ.get(
            "QUIET_HOURS_TIMEZONE", "America/New_York"
        )
        self._timezone = self._get_timezone(self._timezone_str)
        self._queue_file = queue_file
        self._critical_bypass = critical_bypass

        self._rules: List[QuietHoursRule] = []
        self._queued_alerts: List[QueuedAlert] = []
        self._rule_counter = 0
        self._alert_counter = 0
        self._lock = threading.RLock()

        # Load default quiet hours from environment
        self._load_default_rules()

        # Load queued alerts from file if exists
        if self._queue_file:
            self._load_queue()

    def _get_timezone(self, tz_str: str):
        """Get timezone object from string."""
        if PYTZ_AVAILABLE:
            try:
                return pytz.timezone(tz_str)
            except Exception:
                logger.warning(f"Invalid timezone {tz_str}, using UTC")
                return pytz.UTC
        return None

    def _get_current_time(self) -> datetime:
        """Get current time in the configured timezone."""
        now = datetime.utcnow()
        if self._timezone and PYTZ_AVAILABLE:
            now = pytz.UTC.localize(now).astimezone(self._timezone)
        return now

    def _load_default_rules(self):
        """Load default quiet hours from environment variables."""
        start = os.environ.get("DEFAULT_QUIET_HOURS_START")
        end = os.environ.get("DEFAULT_QUIET_HOURS_END")

        if start and end:
            try:
                self.set_quiet_hours(
                    start=start,
                    end=end,
                    name="Default Quiet Hours"
                )
                logger.info(f"Loaded default quiet hours: {start} - {end}")
            except Exception as e:
                logger.warning(f"Failed to load default quiet hours: {e}")

    def _load_queue(self):
        """Load queued alerts from file."""
        if not self._queue_file or not os.path.exists(self._queue_file):
            return

        try:
            with open(self._queue_file, "r") as f:
                data = json.load(f)
                self._queued_alerts = [
                    QueuedAlert.from_dict(a) for a in data.get("alerts", [])
                ]
                self._alert_counter = data.get("counter", 0)
            logger.info(f"Loaded {len(self._queued_alerts)} queued alerts")
        except Exception as e:
            logger.warning(f"Failed to load alert queue: {e}")

    def _save_queue(self):
        """Save queued alerts to file."""
        if not self._queue_file:
            return

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self._queue_file), exist_ok=True)

            with open(self._queue_file, "w") as f:
                json.dump({
                    "alerts": [a.to_dict() for a in self._queued_alerts],
                    "counter": self._alert_counter,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save alert queue: {e}")

    def _generate_rule_id(self) -> str:
        """Generate unique rule ID."""
        with self._lock:
            self._rule_counter += 1
            return f"rule_{self._rule_counter}"

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        with self._lock:
            self._alert_counter += 1
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            return f"queued_{timestamp}_{self._alert_counter}"

    def set_quiet_hours(
        self,
        start: str,
        end: str,
        days: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        alert_types: Optional[List[str]] = None,
        enabled: bool = True,
        name: Optional[str] = None,
    ) -> QuietHoursRule:
        """
        Set quiet hours during which alerts will be suppressed.

        Args:
            start: Start time in HH:MM format (e.g., "22:00")
            end: End time in HH:MM format (e.g., "07:00")
            days: Days of week (e.g., ["monday", "tuesday"]), None = all days
            channels: Channels to apply to (e.g., ["email", "sms"]), None = all
            alert_types: Alert types to apply to (e.g., ["signal"]), None = all
            enabled: Whether the rule is active
            name: Optional name for the rule

        Returns:
            QuietHoursRule: The created rule
        """
        with self._lock:
            # Parse days
            parsed_days = None
            if days:
                parsed_days = [DayOfWeek(d.lower()) for d in days]

            rule = QuietHoursRule(
                start_time=start,
                end_time=end,
                days=parsed_days,
                channels=[c.lower() for c in channels] if channels else None,
                alert_types=[t.lower() for t in alert_types] if alert_types else None,
                enabled=enabled,
                name=name,
                rule_id=self._generate_rule_id(),
            )

            self._rules.append(rule)
            logger.info(f"Added quiet hours rule: {start} - {end}")
            return rule

    def remove_quiet_hours(self, rule_id: str) -> bool:
        """
        Remove a quiet hours rule.

        Args:
            rule_id: ID of the rule to remove

        Returns:
            bool: True if rule was removed
        """
        with self._lock:
            for i, rule in enumerate(self._rules):
                if rule.rule_id == rule_id:
                    self._rules.pop(i)
                    logger.info(f"Removed quiet hours rule: {rule_id}")
                    return True
            return False

    def get_rules(self) -> List[QuietHoursRule]:
        """Get all quiet hours rules."""
        with self._lock:
            return list(self._rules)

    def update_rule(
        self,
        rule_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        days: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        alert_types: Optional[List[str]] = None,
        enabled: Optional[bool] = None,
        name: Optional[str] = None,
    ) -> Optional[QuietHoursRule]:
        """
        Update an existing quiet hours rule.

        Args:
            rule_id: ID of the rule to update
            Other args: Fields to update (None = no change)

        Returns:
            QuietHoursRule: Updated rule, or None if not found
        """
        with self._lock:
            for rule in self._rules:
                if rule.rule_id == rule_id:
                    if start is not None:
                        rule.start_time = datetime.strptime(start, "%H:%M").time()
                    if end is not None:
                        rule.end_time = datetime.strptime(end, "%H:%M").time()
                    if days is not None:
                        rule.days = [DayOfWeek(d.lower()) for d in days] if days else None
                    if channels is not None:
                        rule.channels = [c.lower() for c in channels] if channels else None
                    if alert_types is not None:
                        rule.alert_types = [t.lower() for t in alert_types] if alert_types else None
                    if enabled is not None:
                        rule.enabled = enabled
                    if name is not None:
                        rule.name = name
                    return rule
            return None

    def _is_time_in_range(self, check_time: time, start: time, end: time) -> bool:
        """
        Check if a time is within a range, handling overnight spans.

        Args:
            check_time: Time to check
            start: Range start
            end: Range end

        Returns:
            bool: True if check_time is within the range
        """
        if start <= end:
            # Normal range (e.g., 09:00 - 17:00)
            return start <= check_time <= end
        else:
            # Overnight range (e.g., 22:00 - 07:00)
            return check_time >= start or check_time <= end

    def _rule_applies(
        self,
        rule: QuietHoursRule,
        channel: Optional[str] = None,
        alert_type: Optional[str] = None,
    ) -> bool:
        """Check if a rule applies to the given channel and alert type."""
        if not rule.enabled:
            return False

        # Check channel filter
        if rule.channels and channel:
            if channel.lower() not in rule.channels:
                return False

        # Check alert type filter
        if rule.alert_types and alert_type:
            if alert_type.lower() not in rule.alert_types:
                return False

        return True

    def is_quiet_time(
        self,
        channel: Optional[str] = None,
        alert_type: Optional[str] = None,
        priority: Optional[str] = None,
        check_time: Optional[datetime] = None,
    ) -> bool:
        """
        Check if alerts should be suppressed based on quiet hours.

        Args:
            channel: Alert channel to check
            alert_type: Type of alert to check
            priority: Alert priority (critical alerts may bypass)
            check_time: Time to check (default: current time)

        Returns:
            bool: True if alerts should be suppressed
        """
        # Critical alerts bypass quiet hours if configured
        if self._critical_bypass and priority and priority.lower() == "critical":
            return False

        if check_time is None:
            check_time = self._get_current_time()

        current_time = check_time.time()
        current_day = DayOfWeek.from_weekday(check_time.weekday())

        with self._lock:
            for rule in self._rules:
                if not self._rule_applies(rule, channel, alert_type):
                    continue

                # Check day of week
                if rule.days and current_day not in rule.days:
                    continue

                # Check time range
                if self._is_time_in_range(current_time, rule.start_time, rule.end_time):
                    logger.debug(
                        f"Quiet hours active: rule '{rule.name or rule.rule_id}' "
                        f"({rule.start_time} - {rule.end_time})"
                    )
                    return True

        return False

    def get_next_active_time(
        self,
        channel: Optional[str] = None,
        alert_type: Optional[str] = None,
    ) -> Optional[datetime]:
        """
        Get the next time when alerts will be active (not in quiet hours).

        Args:
            channel: Alert channel
            alert_type: Type of alert

        Returns:
            datetime: Next active time, or None if no quiet hours apply
        """
        if not self.is_quiet_time(channel, alert_type):
            return None

        now = self._get_current_time()
        current_time = now.time()
        current_day = DayOfWeek.from_weekday(now.weekday())

        # Find the earliest end time from applicable rules
        earliest_end: Optional[datetime] = None

        with self._lock:
            for rule in self._rules:
                if not self._rule_applies(rule, channel, alert_type):
                    continue

                if rule.days and current_day not in rule.days:
                    continue

                if self._is_time_in_range(current_time, rule.start_time, rule.end_time):
                    # Calculate when this rule ends
                    end_dt = now.replace(
                        hour=rule.end_time.hour,
                        minute=rule.end_time.minute,
                        second=0,
                        microsecond=0,
                    )

                    # Handle overnight rules
                    if rule.start_time > rule.end_time and current_time >= rule.start_time:
                        end_dt += timedelta(days=1)

                    if earliest_end is None or end_dt < earliest_end:
                        earliest_end = end_dt

        return earliest_end

    def queue_for_later(
        self,
        title: str,
        message: str,
        priority: str = "normal",
        channels: Optional[List[str]] = None,
        alert_type: str = "general",
        extra_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        scheduled_for: Optional[datetime] = None,
    ) -> QueuedAlert:
        """
        Queue an alert for delivery after quiet hours end.

        Args:
            title: Alert title
            message: Alert message body
            priority: Alert priority level
            channels: Target channels
            alert_type: Type of alert
            extra_data: Additional alert data
            user_id: Optional user ID
            scheduled_for: Specific time to send (default: next active time)

        Returns:
            QueuedAlert: The queued alert
        """
        with self._lock:
            alert_id = self._generate_alert_id()

            # Determine when to send
            if scheduled_for is None:
                scheduled_for = self.get_next_active_time(
                    channel=channels[0] if channels else None,
                    alert_type=alert_type,
                )

            alert = QueuedAlert(
                alert_id=alert_id,
                title=title,
                message=message,
                priority=priority,
                channels=channels or [],
                alert_type=alert_type,
                queued_at=self._get_current_time(),
                scheduled_for=scheduled_for,
                extra_data=extra_data,
                user_id=user_id,
            )

            self._queued_alerts.append(alert)
            self._save_queue()

            logger.info(
                f"Queued alert '{title}' for delivery "
                f"at {scheduled_for.isoformat() if scheduled_for else 'next active time'}"
            )

            return alert

    def get_queued_alerts(
        self,
        user_id: Optional[int] = None,
        channel: Optional[str] = None,
        alert_type: Optional[str] = None,
        include_scheduled: bool = True,
    ) -> List[QueuedAlert]:
        """
        Get pending queued alerts.

        Args:
            user_id: Filter by user ID
            channel: Filter by channel
            alert_type: Filter by alert type
            include_scheduled: Include alerts scheduled for future

        Returns:
            List[QueuedAlert]: Queued alerts matching filters
        """
        with self._lock:
            alerts = list(self._queued_alerts)

        now = self._get_current_time()

        filtered = []
        for alert in alerts:
            if user_id is not None and alert.user_id != user_id:
                continue
            if channel and channel not in alert.channels:
                continue
            if alert_type and alert.alert_type != alert_type:
                continue
            if not include_scheduled and alert.scheduled_for and alert.scheduled_for > now:
                continue
            filtered.append(alert)

        return filtered

    def get_ready_alerts(self) -> List[QueuedAlert]:
        """
        Get alerts that are ready to be sent.

        Returns:
            List[QueuedAlert]: Alerts ready for delivery
        """
        now = self._get_current_time()

        with self._lock:
            ready = []
            for alert in self._queued_alerts:
                # Check if scheduled time has passed
                if alert.scheduled_for and alert.scheduled_for > now:
                    continue

                # Check if quiet hours still apply
                if self.is_quiet_time(
                    channel=alert.channels[0] if alert.channels else None,
                    alert_type=alert.alert_type,
                    priority=alert.priority,
                ):
                    continue

                ready.append(alert)

            return ready

    def remove_queued_alert(self, alert_id: str) -> bool:
        """
        Remove a queued alert.

        Args:
            alert_id: ID of the alert to remove

        Returns:
            bool: True if alert was removed
        """
        with self._lock:
            for i, alert in enumerate(self._queued_alerts):
                if alert.alert_id == alert_id:
                    self._queued_alerts.pop(i)
                    self._save_queue()
                    return True
            return False

    def process_queued_alerts(self, send_callback) -> List[Dict[str, Any]]:
        """
        Process and send queued alerts when quiet hours end.

        Args:
            send_callback: Function to call with (title, message, priority, channels, extra_data)
                          Should return True if sent successfully

        Returns:
            List[Dict]: Results of processing each alert
        """
        ready_alerts = self.get_ready_alerts()
        results = []

        for alert in ready_alerts:
            try:
                success = send_callback(
                    title=alert.title,
                    message=alert.message,
                    priority=alert.priority,
                    channels=alert.channels,
                    extra_data=alert.extra_data,
                )

                if success:
                    self.remove_queued_alert(alert.alert_id)
                    results.append({
                        "alert_id": alert.alert_id,
                        "status": "sent",
                        "title": alert.title,
                    })
                    logger.info(f"Sent queued alert: {alert.title}")
                else:
                    alert.attempts += 1
                    results.append({
                        "alert_id": alert.alert_id,
                        "status": "failed",
                        "title": alert.title,
                        "attempts": alert.attempts,
                    })
                    logger.warning(f"Failed to send queued alert: {alert.title}")

            except Exception as e:
                alert.attempts += 1
                results.append({
                    "alert_id": alert.alert_id,
                    "status": "error",
                    "title": alert.title,
                    "error": str(e),
                })
                logger.error(f"Error processing queued alert {alert.alert_id}: {e}")

        # Save updated queue
        with self._lock:
            self._save_queue()

        return results

    def clear_queue(self, user_id: Optional[int] = None) -> int:
        """
        Clear all queued alerts.

        Args:
            user_id: If provided, only clear alerts for this user

        Returns:
            int: Number of alerts cleared
        """
        with self._lock:
            if user_id is None:
                count = len(self._queued_alerts)
                self._queued_alerts.clear()
            else:
                original_count = len(self._queued_alerts)
                self._queued_alerts = [
                    a for a in self._queued_alerts if a.user_id != user_id
                ]
                count = original_count - len(self._queued_alerts)

            self._save_queue()
            return count

    def get_schedule_status(self) -> Dict[str, Any]:
        """
        Get current schedule status.

        Returns:
            Dict: Status including active rules, quiet state, and queue info
        """
        now = self._get_current_time()
        is_quiet = self.is_quiet_time()

        return {
            "current_time": now.isoformat(),
            "timezone": self._timezone_str,
            "is_quiet_time": is_quiet,
            "next_active_time": (
                self.get_next_active_time().isoformat()
                if is_quiet else None
            ),
            "active_rules": len([r for r in self._rules if r.enabled]),
            "total_rules": len(self._rules),
            "queued_alerts": len(self._queued_alerts),
            "critical_bypass_enabled": self._critical_bypass,
        }


# Global scheduler instance
_scheduler: Optional[AlertScheduler] = None


def get_alert_scheduler() -> AlertScheduler:
    """
    Get the global alert scheduler instance.

    Returns:
        AlertScheduler: Singleton scheduler instance
    """
    global _scheduler
    if _scheduler is None:
        queue_file = os.environ.get("ALERT_QUEUE_FILE", "data/alerts/queued_alerts.json")
        _scheduler = AlertScheduler(queue_file=queue_file)
    return _scheduler


def set_alert_scheduler(scheduler: AlertScheduler):
    """Set the global alert scheduler instance."""
    global _scheduler
    _scheduler = scheduler
