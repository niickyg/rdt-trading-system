"""
Alert Manager - Unified Multi-Channel Alert System
Provides a single interface to send alerts across multiple notification channels.
With retry logic, delivery tracking, and quiet hours scheduling.
"""

import os
from typing import Optional, List, Dict, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from .pushover_alert import PushoverAlert, PushoverResult
from .discord_alert import DiscordAlert, DiscordResult
from .telegram_alert import TelegramAlert, TelegramResult
from .email_alert import EmailAlert, EmailResult
from .sms_alert import SMSAlert, SMSResult
from .retry import (
    RetryConfig,
    RetryResult,
    RetryExecutor,
    AlertDeliveryError,
    ConfigurationError,
)
from .delivery_tracker import (
    DeliveryTracker,
    DeliveryStatus,
    AlertRecord,
    get_delivery_tracker,
)
from .scheduler import AlertScheduler, get_alert_scheduler, QueuedAlert
from .schedule_config import AlertScheduleConfig, get_user_schedule_config


class AlertChannel(str, Enum):
    """Available alert channels."""
    PUSHOVER = "pushover"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    DESKTOP = "desktop"
    SMS = "sms"
    EMAIL = "email"


class AlertPriority(str, Enum):
    """Alert priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertResult:
    """Result of sending an alert to a channel."""
    channel: str
    success: bool
    error: Optional[str] = None
    attempts: int = 1
    alert_id: Optional[str] = None
    retry_scheduled: bool = False

    @property
    def is_retryable(self) -> bool:
        """Check if the alert can be retried."""
        return not self.success and self.retry_scheduled


@dataclass
class MultiAlertResult:
    """Result of sending an alert to multiple channels."""
    results: List[AlertResult] = field(default_factory=list)

    @property
    def all_success(self) -> bool:
        """Check if all alerts were sent successfully."""
        return all(r.success for r in self.results)

    @property
    def any_success(self) -> bool:
        """Check if any alert was sent successfully."""
        return any(r.success for r in self.results)

    @property
    def success_count(self) -> int:
        """Count of successful alerts."""
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        """Count of failed alerts."""
        return sum(1 for r in self.results if not r.success)

    @property
    def failed_channels(self) -> List[str]:
        """List of channels that failed."""
        return [r.channel for r in self.results if not r.success]

    @property
    def successful_channels(self) -> List[str]:
        """List of channels that succeeded."""
        return [r.channel for r in self.results if r.success]

    @property
    def total_attempts(self) -> int:
        """Total number of delivery attempts across all channels."""
        return sum(r.attempts for r in self.results)

    @property
    def retryable_alerts(self) -> List[AlertResult]:
        """List of alerts that can be retried."""
        return [r for r in self.results if r.is_retryable]


class AlertManager:
    """
    Unified alert manager for multi-channel notifications.

    Provides a single interface to send alerts across multiple channels
    (Pushover, Discord, Telegram, etc.) with graceful error handling,
    retry logic with exponential backoff, delivery tracking, and quiet hours scheduling.

    Configuration is read from environment variables:
        - ALERT_CHANNELS: Comma-separated list of enabled channels
        - Channel-specific credentials (see individual alert classes)
        - DEFAULT_QUIET_HOURS_START/END: Default quiet hours
        - QUIET_HOURS_TIMEZONE: Timezone for quiet hours

    Example:
        manager = AlertManager()
        manager.send_alert("Trade Alert", "AAPL breakout detected", priority="high")
    """

    def __init__(
        self,
        enabled_channels: Optional[List[str]] = None,
        fail_silently: bool = True,
        retry_config: Optional[RetryConfig] = None,
        enable_tracking: bool = True,
        tracker: Optional[DeliveryTracker] = None,
        enable_scheduling: bool = True,
        scheduler: Optional[AlertScheduler] = None,
    ):
        """
        Initialize the alert manager.

        Args:
            enabled_channels: List of channel names to enable (defaults to ALERT_CHANNELS env var)
            fail_silently: If True, don't raise exceptions on alert failures
            retry_config: Configuration for retry behavior (uses defaults if None)
            enable_tracking: Whether to track delivery status in database
            tracker: Optional custom delivery tracker instance
            enable_scheduling: Whether to enable quiet hours scheduling
            scheduler: Optional custom scheduler instance
        """
        self.fail_silently = fail_silently
        self.retry_config = retry_config or RetryConfig()
        self.enable_tracking = enable_tracking
        self.enable_scheduling = enable_scheduling
        self._retry_executor = RetryExecutor(self.retry_config)

        if enable_tracking:
            self._tracker = tracker or get_delivery_tracker()
        else:
            self._tracker = None

        # Initialize scheduler
        if enable_scheduling:
            self._scheduler = scheduler or get_alert_scheduler()
        else:
            self._scheduler = None

        self._pushover: Optional[PushoverAlert] = None
        self._discord: Optional[DiscordAlert] = None
        self._telegram: Optional[TelegramAlert] = None
        self._email: Optional[EmailAlert] = None
        self._sms: Optional[SMSAlert] = None

        # Alert preferences: which channels to use for each alert type
        # Default: all enabled channels for all alert types
        self._alert_preferences: Dict[str, Set[str]] = {}

        if enabled_channels is None:
            channels_env = os.environ.get('ALERT_CHANNELS', '')
            if channels_env:
                enabled_channels = [c.strip().lower() for c in channels_env.split(',') if c.strip()]
            else:
                enabled_channels = []

        self._enabled_channels: Set[str] = set(c.lower() for c in enabled_channels)

        self._initialize_channels()

    def _initialize_channels(self):
        """Initialize configured alert channels."""
        if 'pushover' in self._enabled_channels:
            self._pushover = PushoverAlert()
            if not self._pushover.is_configured:
                logger.warning("Pushover enabled but not properly configured")

        if 'discord' in self._enabled_channels:
            self._discord = DiscordAlert()
            if not self._discord.is_configured:
                logger.warning("Discord enabled but not properly configured")

        if 'telegram' in self._enabled_channels:
            self._telegram = TelegramAlert()
            if not self._telegram.is_configured:
                logger.warning("Telegram enabled but not properly configured")

        if 'email' in self._enabled_channels:
            self._email = EmailAlert()
            if not self._email.is_configured:
                logger.warning("Email enabled but not properly configured")

        if 'sms' in self._enabled_channels:
            self._sms = SMSAlert()
            if not self._sms.is_configured:
                logger.warning("SMS enabled but not properly configured")

    @property
    def enabled_channels(self) -> Set[str]:
        """Get set of enabled channel names."""
        return self._enabled_channels.copy()

    @property
    def configured_channels(self) -> List[str]:
        """Get list of channels that are both enabled and properly configured."""
        configured = []
        if self._pushover and self._pushover.is_configured:
            configured.append('pushover')
        if self._discord and self._discord.is_configured:
            configured.append('discord')
        if self._telegram and self._telegram.is_configured:
            configured.append('telegram')
        if self._email and self._email.is_configured:
            configured.append('email')
        if self._sms and self._sms.is_configured:
            configured.append('sms')
        return configured

    @property
    def tracker(self) -> Optional[DeliveryTracker]:
        """Get the delivery tracker instance."""
        return self._tracker

    def enable_channel(self, channel: str):
        """
        Enable an alert channel.

        Args:
            channel: Channel name to enable
        """
        channel = channel.lower()
        self._enabled_channels.add(channel)

        if channel == 'pushover' and not self._pushover:
            self._pushover = PushoverAlert()
        elif channel == 'discord' and not self._discord:
            self._discord = DiscordAlert()
        elif channel == 'telegram' and not self._telegram:
            self._telegram = TelegramAlert()
        elif channel == 'email' and not self._email:
            self._email = EmailAlert()
        elif channel == 'sms' and not self._sms:
            self._sms = SMSAlert()

    def disable_channel(self, channel: str):
        """
        Disable an alert channel.

        Args:
            channel: Channel name to disable
        """
        self._enabled_channels.discard(channel.lower())

    def is_channel_enabled(self, channel: str) -> bool:
        """Check if a channel is enabled."""
        return channel.lower() in self._enabled_channels

    def is_channel_configured(self, channel: str) -> bool:
        """Check if a channel is both enabled and properly configured."""
        channel = channel.lower()
        if channel == 'pushover':
            return self._pushover is not None and self._pushover.is_configured
        elif channel == 'discord':
            return self._discord is not None and self._discord.is_configured
        elif channel == 'telegram':
            return self._telegram is not None and self._telegram.is_configured
        elif channel == 'email':
            return self._email is not None and self._email.is_configured
        elif channel == 'sms':
            return self._sms is not None and self._sms.is_configured
        return False

    def set_alert_preference(
        self,
        alert_type: str,
        channels: List[str]
    ):
        """
        Set which channels to use for a specific alert type.

        Args:
            alert_type: Type of alert ('trade', 'signal', 'risk', 'system', 'summary')
            channels: List of channel names to use for this alert type
        """
        self._alert_preferences[alert_type.lower()] = set(c.lower() for c in channels)

    def get_channels_for_alert_type(self, alert_type: str) -> Set[str]:
        """
        Get channels configured for a specific alert type.

        Args:
            alert_type: Type of alert

        Returns:
            Set[str]: Channel names to use (falls back to all enabled if not set)
        """
        if alert_type.lower() in self._alert_preferences:
            return self._alert_preferences[alert_type.lower()] & self._enabled_channels
        return self._enabled_channels.copy()

    def _send_to_email_with_retry(
        self,
        title: str,
        message: str,
        priority: str,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> AlertResult:
        """Send alert to Email with retry logic."""
        if not self._email or not self._email.is_configured:
            return AlertResult(
                channel='email',
                success=False,
                error='Email not configured'
            )

        alert_id = None
        if self._tracker:
            record = self._tracker.create_record(
                channel='email',
                title=title,
                message=message,
                priority=priority,
                extra_data=extra_data,
                max_attempts=self.retry_config.max_attempts
            )
            alert_id = record.alert_id

        def send_op():
            return self._email.send_alert_raising(title, message, priority)

        retry_result = self._retry_executor.execute(
            send_op,
            on_retry=self._on_retry
        )

        if self._tracker and alert_id:
            self._tracker.record_attempt(
                alert_id=alert_id,
                success=retry_result.success,
                error_message=retry_result.error_message,
                retry_delay=None
            )

        retry_scheduled = False
        if not retry_result.success and self._tracker and alert_id:
            record = self._tracker.get_record(alert_id)
            if record and record.can_retry:
                retry_scheduled = True

        return AlertResult(
            channel='email',
            success=retry_result.success,
            error=retry_result.error_message,
            attempts=retry_result.attempts,
            alert_id=alert_id,
            retry_scheduled=retry_scheduled
        )

    def _send_to_sms_with_retry(
        self,
        title: str,
        message: str,
        priority: str
    ) -> AlertResult:
        """Send alert to SMS with retry logic."""
        if not self._sms or not self._sms.is_configured:
            return AlertResult(
                channel='sms',
                success=False,
                error='SMS not configured'
            )

        alert_id = None
        if self._tracker:
            record = self._tracker.create_record(
                channel='sms',
                title=title,
                message=message,
                priority=priority,
                max_attempts=self.retry_config.max_attempts
            )
            alert_id = record.alert_id

        def send_op():
            return self._sms.send_alert_raising(title, message, priority)

        retry_result = self._retry_executor.execute(
            send_op,
            on_retry=self._on_retry
        )

        if self._tracker and alert_id:
            self._tracker.record_attempt(
                alert_id=alert_id,
                success=retry_result.success,
                error_message=retry_result.error_message,
                retry_delay=None
            )

        retry_scheduled = False
        if not retry_result.success and self._tracker and alert_id:
            record = self._tracker.get_record(alert_id)
            if record and record.can_retry:
                retry_scheduled = True

        return AlertResult(
            channel='sms',
            success=retry_result.success,
            error=retry_result.error_message,
            attempts=retry_result.attempts,
            alert_id=alert_id,
            retry_scheduled=retry_scheduled
        )

    def _send_to_email(
        self,
        title: str,
        message: str,
        priority: str
    ) -> AlertResult:
        """Send alert to Email (legacy method without retry)."""
        if not self._email or not self._email.is_configured:
            return AlertResult(
                channel='email',
                success=False,
                error='Email not configured'
            )

        try:
            success = self._email.send_alert(title, message, priority)
            return AlertResult(
                channel='email',
                success=success,
                error=None if success else 'Failed to send'
            )
        except Exception as e:
            logger.error(f"Email alert exception: {e}")
            return AlertResult(
                channel='email',
                success=False,
                error=str(e)
            )

    def _send_to_sms(
        self,
        title: str,
        message: str,
        priority: str
    ) -> AlertResult:
        """Send alert to SMS (legacy method without retry)."""
        if not self._sms or not self._sms.is_configured:
            return AlertResult(
                channel='sms',
                success=False,
                error='SMS not configured'
            )

        try:
            success = self._sms.send_alert(title, message, priority)
            return AlertResult(
                channel='sms',
                success=success,
                error=None if success else 'Failed to send'
            )
        except Exception as e:
            logger.error(f"SMS alert exception: {e}")
            return AlertResult(
                channel='sms',
                success=False,
                error=str(e)
            )

    def _on_retry(self, attempt: int, error: Exception, delay: float):
        """Callback for retry attempts - logs retry information."""
        logger.info(
            f"Retry attempt {attempt} scheduled after {delay:.2f}s. "
            f"Error: {error}"
        )

    def _send_to_pushover_with_retry(
        self,
        title: str,
        message: str,
        priority: str,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> AlertResult:
        """Send alert to Pushover with retry logic."""
        if not self._pushover or not self._pushover.is_configured:
            return AlertResult(
                channel='pushover',
                success=False,
                error='Pushover not configured'
            )

        alert_id = None
        if self._tracker:
            record = self._tracker.create_record(
                channel='pushover',
                title=title,
                message=message,
                priority=priority,
                extra_data=extra_data,
                max_attempts=self.retry_config.max_attempts
            )
            alert_id = record.alert_id

        def send_op():
            return self._pushover.send_alert_raising(title, message, priority)

        retry_result = self._retry_executor.execute(
            send_op,
            on_retry=self._on_retry
        )

        if self._tracker and alert_id:
            self._tracker.record_attempt(
                alert_id=alert_id,
                success=retry_result.success,
                error_message=retry_result.error_message,
                retry_delay=None
            )

        retry_scheduled = False
        if not retry_result.success and self._tracker and alert_id:
            record = self._tracker.get_record(alert_id)
            if record and record.can_retry:
                retry_scheduled = True

        return AlertResult(
            channel='pushover',
            success=retry_result.success,
            error=retry_result.error_message,
            attempts=retry_result.attempts,
            alert_id=alert_id,
            retry_scheduled=retry_scheduled
        )

    def _send_to_discord_with_retry(
        self,
        title: str,
        message: str,
        priority: str,
        fields: Optional[List[Dict[str, Any]]] = None
    ) -> AlertResult:
        """Send alert to Discord with retry logic."""
        if not self._discord or not self._discord.is_configured:
            return AlertResult(
                channel='discord',
                success=False,
                error='Discord not configured'
            )

        extra_data = {'fields': fields} if fields else None
        alert_id = None

        if self._tracker:
            record = self._tracker.create_record(
                channel='discord',
                title=title,
                message=message,
                priority=priority,
                extra_data=extra_data,
                max_attempts=self.retry_config.max_attempts
            )
            alert_id = record.alert_id

        def send_op():
            return self._discord.send_alert_raising(title, message, priority, fields)

        retry_result = self._retry_executor.execute(
            send_op,
            on_retry=self._on_retry
        )

        if self._tracker and alert_id:
            self._tracker.record_attempt(
                alert_id=alert_id,
                success=retry_result.success,
                error_message=retry_result.error_message,
                retry_delay=None
            )

        retry_scheduled = False
        if not retry_result.success and self._tracker and alert_id:
            record = self._tracker.get_record(alert_id)
            if record and record.can_retry:
                retry_scheduled = True

        return AlertResult(
            channel='discord',
            success=retry_result.success,
            error=retry_result.error_message,
            attempts=retry_result.attempts,
            alert_id=alert_id,
            retry_scheduled=retry_scheduled
        )

    def _send_to_telegram_with_retry(
        self,
        title: str,
        message: str,
        priority: str
    ) -> AlertResult:
        """Send alert to Telegram with retry logic."""
        if not self._telegram or not self._telegram.is_configured:
            return AlertResult(
                channel='telegram',
                success=False,
                error='Telegram not configured'
            )

        alert_id = None
        if self._tracker:
            record = self._tracker.create_record(
                channel='telegram',
                title=title,
                message=message,
                priority=priority,
                max_attempts=self.retry_config.max_attempts
            )
            alert_id = record.alert_id

        def send_op():
            return self._telegram.send_alert_raising(title, message, priority)

        retry_result = self._retry_executor.execute(
            send_op,
            on_retry=self._on_retry
        )

        if self._tracker and alert_id:
            self._tracker.record_attempt(
                alert_id=alert_id,
                success=retry_result.success,
                error_message=retry_result.error_message,
                retry_delay=None
            )

        retry_scheduled = False
        if not retry_result.success and self._tracker and alert_id:
            record = self._tracker.get_record(alert_id)
            if record and record.can_retry:
                retry_scheduled = True

        return AlertResult(
            channel='telegram',
            success=retry_result.success,
            error=retry_result.error_message,
            attempts=retry_result.attempts,
            alert_id=alert_id,
            retry_scheduled=retry_scheduled
        )

    def _send_to_pushover(
        self,
        title: str,
        message: str,
        priority: str
    ) -> AlertResult:
        """Send alert to Pushover (legacy method without retry)."""
        if not self._pushover or not self._pushover.is_configured:
            return AlertResult(
                channel='pushover',
                success=False,
                error='Pushover not configured'
            )

        try:
            success = self._pushover.send_alert(title, message, priority)
            return AlertResult(
                channel='pushover',
                success=success,
                error=None if success else 'Failed to send'
            )
        except Exception as e:
            logger.error(f"Pushover alert exception: {e}")
            return AlertResult(
                channel='pushover',
                success=False,
                error=str(e)
            )

    def _send_to_discord(
        self,
        title: str,
        message: str,
        priority: str,
        fields: Optional[List[Dict[str, Any]]] = None
    ) -> AlertResult:
        """Send alert to Discord (legacy method without retry)."""
        if not self._discord or not self._discord.is_configured:
            return AlertResult(
                channel='discord',
                success=False,
                error='Discord not configured'
            )

        try:
            success = self._discord.send_alert(title, message, priority, fields)
            return AlertResult(
                channel='discord',
                success=success,
                error=None if success else 'Failed to send'
            )
        except Exception as e:
            logger.error(f"Discord alert exception: {e}")
            return AlertResult(
                channel='discord',
                success=False,
                error=str(e)
            )

    def _send_to_telegram(
        self,
        title: str,
        message: str,
        priority: str
    ) -> AlertResult:
        """Send alert to Telegram (legacy method without retry)."""
        if not self._telegram or not self._telegram.is_configured:
            return AlertResult(
                channel='telegram',
                success=False,
                error='Telegram not configured'
            )

        try:
            success = self._telegram.send_alert(title, message, priority)
            return AlertResult(
                channel='telegram',
                success=success,
                error=None if success else 'Failed to send'
            )
        except Exception as e:
            logger.error(f"Telegram alert exception: {e}")
            return AlertResult(
                channel='telegram',
                success=False,
                error=str(e)
            )

    def send_alert(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        channels: Optional[List[str]] = None,
        discord_fields: Optional[List[Dict[str, Any]]] = None,
        use_retry: bool = True,
        alert_type: str = 'general',
        user_id: Optional[int] = None,
        respect_quiet_hours: bool = True,
        queue_if_quiet: bool = True,
    ) -> MultiAlertResult:
        """
        Send an alert to all enabled channels.

        Args:
            title: Alert title
            message: Alert message body
            priority: Priority level ('low', 'normal', 'high', 'critical')
            channels: Specific channels to send to (defaults to all enabled)
            discord_fields: Optional extra fields for Discord embeds
            use_retry: Whether to use retry logic with exponential backoff
            alert_type: Type of alert (signal, trade, system, etc.)
            user_id: Optional user ID for user-specific schedule checks
            respect_quiet_hours: Whether to check quiet hours before sending
            queue_if_quiet: Whether to queue alert if quiet hours are active

        Returns:
            MultiAlertResult: Results from all attempted channels
        """
        result = MultiAlertResult()

        if channels:
            target_channels = set(c.lower() for c in channels) & self._enabled_channels
        else:
            target_channels = self._enabled_channels

        if not target_channels:
            logger.warning("No alert channels configured or enabled")
            return result

        # Check quiet hours if scheduling is enabled
        if respect_quiet_hours and self._scheduler:
            channels_to_send: Set[str] = set()
            channels_to_queue: Set[str] = set()

            for channel in target_channels:
                # Check user-specific schedule if user_id provided
                if user_id:
                    try:
                        user_config = get_user_schedule_config(user_id)
                        if user_config.should_suppress_alert(
                            channel=channel,
                            alert_type=alert_type,
                            priority=priority,
                        ):
                            if queue_if_quiet:
                                channels_to_queue.add(channel)
                                logger.debug(
                                    f"Alert to {channel} suppressed by user schedule, queuing"
                                )
                            continue
                    except Exception as e:
                        logger.debug(f"Error checking user schedule: {e}")

                # Check global quiet hours
                if self._scheduler.is_quiet_time(
                    channel=channel,
                    alert_type=alert_type,
                    priority=priority,
                ):
                    if queue_if_quiet:
                        channels_to_queue.add(channel)
                        logger.debug(f"Alert to {channel} suppressed by quiet hours, queuing")
                    continue

                channels_to_send.add(channel)

            # Queue alerts for suppressed channels
            if channels_to_queue and queue_if_quiet:
                self._scheduler.queue_for_later(
                    title=title,
                    message=message,
                    priority=priority,
                    channels=list(channels_to_queue),
                    alert_type=alert_type,
                    extra_data={'discord_fields': discord_fields} if discord_fields else None,
                    user_id=user_id,
                )
                logger.info(
                    f"Queued alert '{title}' for channels: {', '.join(channels_to_queue)}"
                )

            target_channels = channels_to_send

            if not target_channels:
                logger.info("All channels suppressed by quiet hours, alert queued")
                return result

        for channel in target_channels:
            if channel == 'pushover':
                if use_retry:
                    result.results.append(
                        self._send_to_pushover_with_retry(title, message, priority)
                    )
                else:
                    result.results.append(
                        self._send_to_pushover(title, message, priority)
                    )
            elif channel == 'discord':
                if use_retry:
                    result.results.append(
                        self._send_to_discord_with_retry(title, message, priority, discord_fields)
                    )
                else:
                    result.results.append(
                        self._send_to_discord(title, message, priority, discord_fields)
                    )
            elif channel == 'telegram':
                if use_retry:
                    result.results.append(
                        self._send_to_telegram_with_retry(title, message, priority)
                    )
                else:
                    result.results.append(
                        self._send_to_telegram(title, message, priority)
                    )
            elif channel == 'email':
                if use_retry:
                    result.results.append(
                        self._send_to_email_with_retry(title, message, priority)
                    )
                else:
                    result.results.append(
                        self._send_to_email(title, message, priority)
                    )
            elif channel == 'sms':
                if use_retry:
                    result.results.append(
                        self._send_to_sms_with_retry(title, message, priority)
                    )
                else:
                    result.results.append(
                        self._send_to_sms(title, message, priority)
                    )

        if result.failure_count > 0:
            failed = ', '.join(result.failed_channels)
            logger.warning(f"Some alert channels failed: {failed}")

        if result.success_count > 0:
            successful = ', '.join(result.successful_channels)
            logger.debug(f"Alert sent to: {successful}")

        return result

    def send_trade_alert(
        self,
        action: str,
        symbol: str,
        price: float,
        quantity: int,
        reason: str,
        priority: str = 'normal',
        use_retry: bool = True
    ) -> MultiAlertResult:
        """
        Send a formatted trade alert to all enabled channels.

        Args:
            action: Trade action ('BUY', 'SELL', 'STOP_LOSS', etc.)
            symbol: Stock symbol
            price: Trade price
            quantity: Number of shares
            reason: Reason for the trade
            priority: Priority level
            use_retry: Whether to use retry logic

        Returns:
            MultiAlertResult: Results from all attempted channels
        """
        title = f"{action.upper()}: {symbol}"
        message = (
            f"Action: {action.upper()}\n"
            f"Symbol: {symbol}\n"
            f"Price: ${price:.2f}\n"
            f"Quantity: {quantity}\n"
            f"Total: ${price * quantity:,.2f}\n"
            f"Reason: {reason}"
        )

        discord_fields = [
            {'name': 'Symbol', 'value': symbol, 'inline': True},
            {'name': 'Price', 'value': f"${price:.2f}", 'inline': True},
            {'name': 'Quantity', 'value': str(quantity), 'inline': True},
            {'name': 'Total Value', 'value': f"${price * quantity:,.2f}", 'inline': True}
        ]

        return self.send_alert(
            title=title,
            message=message,
            priority=priority,
            discord_fields=discord_fields,
            use_retry=use_retry
        )

    def send_system_alert(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        use_retry: bool = True
    ) -> MultiAlertResult:
        """
        Send a system alert (errors, warnings, status updates).

        Args:
            title: Alert title
            message: Alert message
            priority: Priority level
            use_retry: Whether to use retry logic

        Returns:
            MultiAlertResult: Results from all attempted channels
        """
        prefixed_title = f"[System] {title}"
        return self.send_alert(prefixed_title, message, priority, use_retry=use_retry)

    def test_all_channels(self, use_retry: bool = True) -> MultiAlertResult:
        """
        Send a test alert to all enabled channels.

        Args:
            use_retry: Whether to use retry logic

        Returns:
            MultiAlertResult: Results from all attempted channels
        """
        return self.send_alert(
            title="Test Alert",
            message="This is a test alert from the RDT Trading System. "
                    "If you receive this message, your alert channel is working correctly.",
            priority='normal',
            use_retry=use_retry
        )

    def retry_failed_alert(self, alert_id: str) -> Optional[AlertResult]:
        """
        Retry a specific failed alert by ID.

        Args:
            alert_id: The alert ID to retry

        Returns:
            AlertResult: Result of the retry attempt, or None if not found
        """
        if not self._tracker:
            logger.warning("Delivery tracking not enabled")
            return None

        record = self._tracker.get_record(alert_id)
        if not record:
            logger.warning(f"Alert record not found: {alert_id}")
            return None

        if not record.can_retry:
            logger.warning(
                f"Alert {alert_id} cannot be retried "
                f"(status: {record.status}, attempts: {record.attempt_count})"
            )
            return AlertResult(
                channel=record.channel,
                success=False,
                error="Cannot retry: max attempts reached or already sent",
                alert_id=alert_id
            )

        self._tracker.mark_for_retry(alert_id)

        if record.channel == 'pushover':
            return self._send_to_pushover_with_retry(
                record.title, record.message, record.priority
            )
        elif record.channel == 'discord':
            fields = record.extra_data.get('fields') if record.extra_data else None
            return self._send_to_discord_with_retry(
                record.title, record.message, record.priority, fields
            )
        elif record.channel == 'telegram':
            return self._send_to_telegram_with_retry(
                record.title, record.message, record.priority
            )
        elif record.channel == 'email':
            return self._send_to_email_with_retry(
                record.title, record.message, record.priority
            )
        elif record.channel == 'sms':
            return self._send_to_sms_with_retry(
                record.title, record.message, record.priority
            )
        else:
            logger.warning(f"Unknown channel: {record.channel}")
            return AlertResult(
                channel=record.channel,
                success=False,
                error=f"Unknown channel: {record.channel}",
                alert_id=alert_id
            )

    def retry_all_failed(self, channel: Optional[str] = None) -> List[AlertResult]:
        """
        Retry all failed alerts.

        Args:
            channel: Optional channel filter

        Returns:
            List[AlertResult]: Results from retry attempts
        """
        if not self._tracker:
            logger.warning("Delivery tracking not enabled")
            return []

        failed_alerts = self._tracker.get_failed_alerts(channel=channel)
        results = []

        for record in failed_alerts:
            if record.can_retry:
                result = self.retry_failed_alert(record.alert_id)
                if result:
                    results.append(result)

        return results

    def get_delivery_stats(self) -> Dict[str, Any]:
        """
        Get delivery statistics.

        Returns:
            Dict: Delivery statistics or empty dict if tracking disabled
        """
        if not self._tracker:
            return {}
        return self._tracker.get_delivery_stats()

    def get_failed_alerts(
        self,
        channel: Optional[str] = None,
        limit: int = 100
    ) -> List[AlertRecord]:
        """
        Get failed alert records.

        Args:
            channel: Optional channel filter
            limit: Maximum number of records

        Returns:
            List[AlertRecord]: Failed alert records
        """
        if not self._tracker:
            return []
        return self._tracker.get_failed_alerts(channel=channel, limit=limit)

    def cleanup_old_records(self, retention_days: int = 7) -> int:
        """
        Clean up old delivery records.

        Args:
            retention_days: Days to retain records

        Returns:
            int: Number of records deleted
        """
        if not self._tracker:
            return 0
        return self._tracker.cleanup_old_records(retention_days)

    # =========================================================================
    # Scheduler Methods
    # =========================================================================

    @property
    def scheduler(self) -> Optional[AlertScheduler]:
        """Get the scheduler instance."""
        return self._scheduler

    def set_quiet_hours(
        self,
        start: str,
        end: str,
        days: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        alert_types: Optional[List[str]] = None,
        enabled: bool = True,
        name: Optional[str] = None,
    ):
        """
        Set quiet hours during which non-critical alerts will be suppressed.

        Args:
            start: Start time in HH:MM format (e.g., "22:00")
            end: End time in HH:MM format (e.g., "07:00")
            days: Days of week (e.g., ["monday", "tuesday"]), None = all days
            channels: Channels to apply to, None = all
            alert_types: Alert types to apply to, None = all
            enabled: Whether the rule is active
            name: Optional name for the rule
        """
        if not self._scheduler:
            logger.warning("Scheduling not enabled")
            return None
        return self._scheduler.set_quiet_hours(
            start=start,
            end=end,
            days=days,
            channels=channels,
            alert_types=alert_types,
            enabled=enabled,
            name=name,
        )

    def is_quiet_time(
        self,
        channel: Optional[str] = None,
        alert_type: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> bool:
        """
        Check if current time is in quiet hours.

        Args:
            channel: Alert channel to check
            alert_type: Type of alert to check
            priority: Alert priority (critical alerts may bypass)

        Returns:
            bool: True if alerts should be suppressed
        """
        if not self._scheduler:
            return False
        return self._scheduler.is_quiet_time(
            channel=channel,
            alert_type=alert_type,
            priority=priority,
        )

    def get_queued_alerts(
        self,
        user_id: Optional[int] = None,
        channel: Optional[str] = None,
    ) -> List[QueuedAlert]:
        """
        Get pending queued alerts.

        Args:
            user_id: Filter by user ID
            channel: Filter by channel

        Returns:
            List[QueuedAlert]: Queued alerts
        """
        if not self._scheduler:
            return []
        return self._scheduler.get_queued_alerts(
            user_id=user_id,
            channel=channel,
        )

    def process_queued_alerts(self) -> List[Dict[str, Any]]:
        """
        Process and send queued alerts when quiet hours end.

        Returns:
            List[Dict]: Results of processing each alert
        """
        if not self._scheduler:
            return []

        def send_callback(title, message, priority, channels, extra_data):
            discord_fields = extra_data.get('discord_fields') if extra_data else None
            result = self.send_alert(
                title=title,
                message=message,
                priority=priority,
                channels=channels,
                discord_fields=discord_fields,
                respect_quiet_hours=False,  # Don't re-check quiet hours
            )
            return result.any_success

        return self._scheduler.process_queued_alerts(send_callback)

    def get_schedule_status(self) -> Dict[str, Any]:
        """
        Get current schedule status.

        Returns:
            Dict: Status including active rules, quiet state, and queue info
        """
        if not self._scheduler:
            return {'scheduling_enabled': False}

        status = self._scheduler.get_schedule_status()
        status['scheduling_enabled'] = True
        return status

    def clear_queued_alerts(self, user_id: Optional[int] = None) -> int:
        """
        Clear all queued alerts.

        Args:
            user_id: If provided, only clear alerts for this user

        Returns:
            int: Number of alerts cleared
        """
        if not self._scheduler:
            return 0
        return self._scheduler.clear_queue(user_id=user_id)


def get_alert_manager(
    enabled_channels: Optional[List[str]] = None,
    retry_config: Optional[RetryConfig] = None,
    enable_tracking: bool = True
) -> AlertManager:
    """
    Factory function to create an AlertManager instance.

    Args:
        enabled_channels: Optional list of channels to enable
        retry_config: Optional retry configuration
        enable_tracking: Whether to enable delivery tracking

    Returns:
        AlertManager: Configured alert manager instance
    """
    return AlertManager(
        enabled_channels=enabled_channels,
        retry_config=retry_config,
        enable_tracking=enable_tracking
    )
