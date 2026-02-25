"""
Alert Schedule Configuration - Per-User Preferences

Provides functionality to manage per-user alert scheduling preferences including:
- Personal quiet hours settings
- Override for critical alerts
- Timezone handling
- Channel-specific schedules
"""

import os
from datetime import datetime, time, timedelta
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False

from .scheduler import DayOfWeek, QuietHoursRule


@dataclass
class ChannelSchedule:
    """
    Schedule configuration for a specific alert channel.

    Attributes:
        channel: Channel name (email, sms, discord, etc.)
        enabled: Whether the channel is enabled
        quiet_hours_start: Start of quiet hours for this channel
        quiet_hours_end: End of quiet hours for this channel
        active_days: Days when this channel is active
        priority_threshold: Minimum priority to send via this channel
    """
    channel: str
    enabled: bool = True
    quiet_hours_start: Optional[time] = None
    quiet_hours_end: Optional[time] = None
    active_days: Optional[List[DayOfWeek]] = None
    priority_threshold: str = "low"  # low, normal, high, critical

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "channel": self.channel,
            "enabled": self.enabled,
            "quiet_hours_start": self.quiet_hours_start.strftime("%H:%M") if self.quiet_hours_start else None,
            "quiet_hours_end": self.quiet_hours_end.strftime("%H:%M") if self.quiet_hours_end else None,
            "active_days": [d.value for d in self.active_days] if self.active_days else None,
            "priority_threshold": self.priority_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelSchedule":
        """Create from dictionary."""
        return cls(
            channel=data["channel"],
            enabled=data.get("enabled", True),
            quiet_hours_start=(
                datetime.strptime(data["quiet_hours_start"], "%H:%M").time()
                if data.get("quiet_hours_start") else None
            ),
            quiet_hours_end=(
                datetime.strptime(data["quiet_hours_end"], "%H:%M").time()
                if data.get("quiet_hours_end") else None
            ),
            active_days=(
                [DayOfWeek(d) for d in data["active_days"]]
                if data.get("active_days") else None
            ),
            priority_threshold=data.get("priority_threshold", "low"),
        )


@dataclass
class AlertTypePreference:
    """
    Preferences for a specific alert type.

    Attributes:
        alert_type: Type of alert (signal, trade, system, etc.)
        enabled: Whether this alert type is enabled
        channels: Channels to use for this alert type
        quiet_hours_exempt: Whether this type can bypass quiet hours
        min_priority: Minimum priority level to send
    """
    alert_type: str
    enabled: bool = True
    channels: Optional[List[str]] = None
    quiet_hours_exempt: bool = False
    min_priority: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_type": self.alert_type,
            "enabled": self.enabled,
            "channels": self.channels,
            "quiet_hours_exempt": self.quiet_hours_exempt,
            "min_priority": self.min_priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertTypePreference":
        """Create from dictionary."""
        return cls(
            alert_type=data["alert_type"],
            enabled=data.get("enabled", True),
            channels=data.get("channels"),
            quiet_hours_exempt=data.get("quiet_hours_exempt", False),
            min_priority=data.get("min_priority", "low"),
        )


class AlertScheduleConfig:
    """
    Per-user alert schedule configuration.

    Manages user-specific preferences for alert delivery including:
    - Global quiet hours
    - Per-channel schedules
    - Per-alert-type preferences
    - Critical alert overrides
    - Timezone handling
    """

    PRIORITY_LEVELS = ["low", "normal", "high", "critical"]

    def __init__(
        self,
        user_id: int,
        timezone: Optional[str] = None,
    ):
        """
        Initialize alert schedule configuration.

        Args:
            user_id: User ID this configuration belongs to
            timezone: User's timezone (default: America/New_York)
        """
        self.user_id = user_id
        self._timezone_str = timezone or os.environ.get(
            "QUIET_HOURS_TIMEZONE", "America/New_York"
        )
        self._timezone = self._get_timezone(self._timezone_str)

        # Global quiet hours
        self._quiet_hours_enabled = False
        self._quiet_hours_start: Optional[time] = None
        self._quiet_hours_end: Optional[time] = None
        self._quiet_hours_days: Optional[List[DayOfWeek]] = None

        # Critical alert override
        self._override_critical = True

        # Per-channel schedules
        self._channel_schedules: Dict[str, ChannelSchedule] = {}

        # Per-alert-type preferences
        self._alert_type_prefs: Dict[str, AlertTypePreference] = {}

        # DND (Do Not Disturb) mode
        self._dnd_enabled = False
        self._dnd_until: Optional[datetime] = None

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
        """Get current time in the user's timezone."""
        now = datetime.utcnow()
        if self._timezone and PYTZ_AVAILABLE:
            now = pytz.UTC.localize(now).astimezone(self._timezone)
        return now

    def _is_time_in_range(self, check_time: time, start: time, end: time) -> bool:
        """Check if a time is within a range, handling overnight spans."""
        if start <= end:
            return start <= check_time <= end
        else:
            return check_time >= start or check_time <= end

    def _priority_meets_threshold(self, priority: str, threshold: str) -> bool:
        """Check if priority meets or exceeds threshold."""
        try:
            priority_idx = self.PRIORITY_LEVELS.index(priority.lower())
            threshold_idx = self.PRIORITY_LEVELS.index(threshold.lower())
            return priority_idx >= threshold_idx
        except ValueError:
            return True

    @property
    def timezone(self) -> str:
        """Get user's timezone."""
        return self._timezone_str

    @timezone.setter
    def timezone(self, value: str):
        """Set user's timezone."""
        self._timezone_str = value
        self._timezone = self._get_timezone(value)

    def set_quiet_hours(
        self,
        start: str,
        end: str,
        days: Optional[List[str]] = None,
        enabled: bool = True,
    ):
        """
        Set global quiet hours for this user.

        Args:
            start: Start time in HH:MM format (e.g., "22:00")
            end: End time in HH:MM format (e.g., "07:00")
            days: Days to apply (None = all days)
            enabled: Whether quiet hours are enabled
        """
        self._quiet_hours_enabled = enabled
        self._quiet_hours_start = datetime.strptime(start, "%H:%M").time()
        self._quiet_hours_end = datetime.strptime(end, "%H:%M").time()
        self._quiet_hours_days = (
            [DayOfWeek(d.lower()) for d in days] if days else None
        )

        logger.info(
            f"Set quiet hours for user {self.user_id}: "
            f"{start} - {end}, days={days}, enabled={enabled}"
        )

    def disable_quiet_hours(self):
        """Disable quiet hours."""
        self._quiet_hours_enabled = False

    def set_override_critical(self, override: bool):
        """
        Set whether critical alerts bypass quiet hours.

        Args:
            override: True to allow critical alerts during quiet hours
        """
        self._override_critical = override

    @property
    def override_critical(self) -> bool:
        """Get critical alert override setting."""
        return self._override_critical

    def set_channel_schedule(
        self,
        channel: str,
        enabled: bool = True,
        quiet_hours_start: Optional[str] = None,
        quiet_hours_end: Optional[str] = None,
        active_days: Optional[List[str]] = None,
        priority_threshold: str = "low",
    ):
        """
        Set schedule for a specific channel.

        Args:
            channel: Channel name (email, sms, discord, etc.)
            enabled: Whether the channel is enabled
            quiet_hours_start: Channel-specific quiet hours start
            quiet_hours_end: Channel-specific quiet hours end
            active_days: Days when the channel is active
            priority_threshold: Minimum priority for this channel
        """
        schedule = ChannelSchedule(
            channel=channel.lower(),
            enabled=enabled,
            quiet_hours_start=(
                datetime.strptime(quiet_hours_start, "%H:%M").time()
                if quiet_hours_start else None
            ),
            quiet_hours_end=(
                datetime.strptime(quiet_hours_end, "%H:%M").time()
                if quiet_hours_end else None
            ),
            active_days=(
                [DayOfWeek(d.lower()) for d in active_days]
                if active_days else None
            ),
            priority_threshold=priority_threshold.lower(),
        )
        self._channel_schedules[channel.lower()] = schedule

    def get_channel_schedule(self, channel: str) -> Optional[ChannelSchedule]:
        """Get schedule for a channel."""
        return self._channel_schedules.get(channel.lower())

    def set_alert_type_preference(
        self,
        alert_type: str,
        enabled: bool = True,
        channels: Optional[List[str]] = None,
        quiet_hours_exempt: bool = False,
        min_priority: str = "low",
    ):
        """
        Set preferences for a specific alert type.

        Args:
            alert_type: Type of alert (signal, trade, system, etc.)
            enabled: Whether this alert type is enabled
            channels: Specific channels for this alert type
            quiet_hours_exempt: Whether this type bypasses quiet hours
            min_priority: Minimum priority to send
        """
        pref = AlertTypePreference(
            alert_type=alert_type.lower(),
            enabled=enabled,
            channels=[c.lower() for c in channels] if channels else None,
            quiet_hours_exempt=quiet_hours_exempt,
            min_priority=min_priority.lower(),
        )
        self._alert_type_prefs[alert_type.lower()] = pref

    def get_alert_type_preference(self, alert_type: str) -> Optional[AlertTypePreference]:
        """Get preferences for an alert type."""
        return self._alert_type_prefs.get(alert_type.lower())

    def enable_dnd(self, duration_minutes: Optional[int] = None, until: Optional[datetime] = None):
        """
        Enable Do Not Disturb mode.

        Args:
            duration_minutes: Duration in minutes (None = indefinite)
            until: Specific end time for DND
        """
        self._dnd_enabled = True
        if until:
            self._dnd_until = until
        elif duration_minutes:
            self._dnd_until = self._get_current_time() + timedelta(minutes=duration_minutes)
        else:
            self._dnd_until = None

        logger.info(
            f"DND enabled for user {self.user_id} "
            f"until {self._dnd_until.isoformat() if self._dnd_until else 'indefinite'}"
        )

    def disable_dnd(self):
        """Disable Do Not Disturb mode."""
        self._dnd_enabled = False
        self._dnd_until = None

    def is_dnd_active(self) -> bool:
        """Check if DND is currently active."""
        if not self._dnd_enabled:
            return False

        if self._dnd_until is None:
            return True

        now = self._get_current_time()
        if now >= self._dnd_until:
            # DND expired, disable it
            self._dnd_enabled = False
            self._dnd_until = None
            return False

        return True

    def should_suppress_alert(
        self,
        channel: Optional[str] = None,
        alert_type: Optional[str] = None,
        priority: str = "normal",
        check_time: Optional[datetime] = None,
    ) -> bool:
        """
        Check if an alert should be suppressed based on user preferences.

        Args:
            channel: Alert channel
            alert_type: Type of alert
            priority: Alert priority
            check_time: Time to check (default: current time)

        Returns:
            bool: True if the alert should be suppressed
        """
        if check_time is None:
            check_time = self._get_current_time()

        # Check DND mode first
        if self.is_dnd_active():
            # Critical alerts can bypass DND if override is enabled
            if not (self._override_critical and priority.lower() == "critical"):
                return True

        # Check alert type preferences
        if alert_type:
            pref = self._alert_type_prefs.get(alert_type.lower())
            if pref:
                if not pref.enabled:
                    return True
                if not self._priority_meets_threshold(priority, pref.min_priority):
                    return True
                # Check if this alert type is exempt from quiet hours
                if pref.quiet_hours_exempt:
                    return False

        # Check critical override
        if self._override_critical and priority.lower() == "critical":
            return False

        # Check channel-specific schedule
        if channel:
            schedule = self._channel_schedules.get(channel.lower())
            if schedule:
                if not schedule.enabled:
                    return True

                # Check channel priority threshold
                if not self._priority_meets_threshold(priority, schedule.priority_threshold):
                    return True

                # Check channel active days
                current_day = DayOfWeek.from_weekday(check_time.weekday())
                if schedule.active_days and current_day not in schedule.active_days:
                    return True

                # Check channel-specific quiet hours
                if schedule.quiet_hours_start and schedule.quiet_hours_end:
                    current_time = check_time.time()
                    if self._is_time_in_range(
                        current_time,
                        schedule.quiet_hours_start,
                        schedule.quiet_hours_end,
                    ):
                        return True

        # Check global quiet hours
        if self._quiet_hours_enabled and self._quiet_hours_start and self._quiet_hours_end:
            current_time = check_time.time()
            current_day = DayOfWeek.from_weekday(check_time.weekday())

            # Check day restriction
            if self._quiet_hours_days and current_day not in self._quiet_hours_days:
                return False

            if self._is_time_in_range(
                current_time,
                self._quiet_hours_start,
                self._quiet_hours_end,
            ):
                return True

        return False

    def get_available_channels(
        self,
        alert_type: Optional[str] = None,
        priority: str = "normal",
    ) -> List[str]:
        """
        Get channels that are available for sending an alert.

        Args:
            alert_type: Type of alert
            priority: Alert priority

        Returns:
            List[str]: Available channel names
        """
        now = self._get_current_time()
        available = []

        # Get channels from alert type preference if specified
        if alert_type:
            pref = self._alert_type_prefs.get(alert_type.lower())
            if pref and pref.channels:
                channels_to_check = pref.channels
            else:
                channels_to_check = list(self._channel_schedules.keys())
        else:
            channels_to_check = list(self._channel_schedules.keys())

        for channel in channels_to_check:
            if not self.should_suppress_alert(
                channel=channel,
                alert_type=alert_type,
                priority=priority,
                check_time=now,
            ):
                available.append(channel)

        return available

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "timezone": self._timezone_str,
            "quiet_hours": {
                "enabled": self._quiet_hours_enabled,
                "start": self._quiet_hours_start.strftime("%H:%M") if self._quiet_hours_start else None,
                "end": self._quiet_hours_end.strftime("%H:%M") if self._quiet_hours_end else None,
                "days": [d.value for d in self._quiet_hours_days] if self._quiet_hours_days else None,
            },
            "override_critical": self._override_critical,
            "dnd": {
                "enabled": self._dnd_enabled,
                "until": self._dnd_until.isoformat() if self._dnd_until else None,
            },
            "channel_schedules": {
                name: schedule.to_dict()
                for name, schedule in self._channel_schedules.items()
            },
            "alert_type_preferences": {
                name: pref.to_dict()
                for name, pref in self._alert_type_prefs.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertScheduleConfig":
        """Create configuration from dictionary."""
        config = cls(
            user_id=data["user_id"],
            timezone=data.get("timezone"),
        )

        # Load quiet hours
        qh = data.get("quiet_hours", {})
        if qh.get("enabled") and qh.get("start") and qh.get("end"):
            config.set_quiet_hours(
                start=qh["start"],
                end=qh["end"],
                days=qh.get("days"),
                enabled=qh.get("enabled", True),
            )

        config._override_critical = data.get("override_critical", True)

        # Load DND
        dnd = data.get("dnd", {})
        if dnd.get("enabled"):
            config._dnd_enabled = True
            config._dnd_until = (
                datetime.fromisoformat(dnd["until"])
                if dnd.get("until") else None
            )

        # Load channel schedules
        for name, schedule_data in data.get("channel_schedules", {}).items():
            config._channel_schedules[name] = ChannelSchedule.from_dict(schedule_data)

        # Load alert type preferences
        for name, pref_data in data.get("alert_type_preferences", {}).items():
            config._alert_type_prefs[name] = AlertTypePreference.from_dict(pref_data)

        return config


class AlertScheduleConfigManager:
    """
    Manages alert schedule configurations for multiple users.

    Handles loading, saving, and caching of user configurations.
    """

    def __init__(self, storage_backend=None):
        """
        Initialize the configuration manager.

        Args:
            storage_backend: Optional storage backend (defaults to in-memory)
        """
        self._storage = storage_backend
        self._cache: Dict[int, AlertScheduleConfig] = {}

    def get_config(self, user_id: int) -> AlertScheduleConfig:
        """
        Get configuration for a user.

        Args:
            user_id: User ID

        Returns:
            AlertScheduleConfig: User's configuration
        """
        if user_id in self._cache:
            return self._cache[user_id]

        # Try to load from storage
        if self._storage:
            data = self._storage.load(user_id)
            if data:
                config = AlertScheduleConfig.from_dict(data)
                self._cache[user_id] = config
                return config

        # Create new config with defaults
        config = AlertScheduleConfig(user_id=user_id)
        self._cache[user_id] = config
        return config

    def save_config(self, config: AlertScheduleConfig):
        """
        Save configuration.

        Args:
            config: Configuration to save
        """
        self._cache[config.user_id] = config
        if self._storage:
            self._storage.save(config.user_id, config.to_dict())

    def delete_config(self, user_id: int):
        """
        Delete configuration for a user.

        Args:
            user_id: User ID
        """
        self._cache.pop(user_id, None)
        if self._storage:
            self._storage.delete(user_id)

    def get_all_configs(self) -> List[AlertScheduleConfig]:
        """
        Get all cached configurations.

        Returns:
            List[AlertScheduleConfig]: All configurations
        """
        return list(self._cache.values())


# Global configuration manager
_config_manager: Optional[AlertScheduleConfigManager] = None


def get_config_manager() -> AlertScheduleConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = AlertScheduleConfigManager()
    return _config_manager


def get_user_schedule_config(user_id: int) -> AlertScheduleConfig:
    """
    Get schedule configuration for a user.

    Args:
        user_id: User ID

    Returns:
        AlertScheduleConfig: User's schedule configuration
    """
    return get_config_manager().get_config(user_id)
