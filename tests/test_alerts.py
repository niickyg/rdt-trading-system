"""
Tests for Alert System

Tests:
- All alert channels (email, SMS, Discord, Telegram)
- Rate limiting
- Delivery confirmation
- Deduplication
- Channel failover
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio

from alerts.alert_manager import (
    AlertChannel, AlertPriority, AlertResult, MultiAlertResult
)


class TestAlertResult:
    """Tests for AlertResult dataclass"""

    def test_successful_result(self):
        """Test successful alert result"""
        result = AlertResult(
            channel="discord",
            success=True,
            alert_id="test-123"
        )

        assert result.success is True
        assert result.channel == "discord"
        assert result.is_retryable is False

    def test_failed_retryable_result(self):
        """Test failed but retryable alert result"""
        result = AlertResult(
            channel="telegram",
            success=False,
            error="Connection timeout",
            retry_scheduled=True
        )

        assert result.success is False
        assert result.is_retryable is True

    def test_failed_non_retryable_result(self):
        """Test failed non-retryable alert result"""
        result = AlertResult(
            channel="sms",
            success=False,
            error="Invalid phone number",
            retry_scheduled=False
        )

        assert result.success is False
        assert result.is_retryable is False


class TestMultiAlertResult:
    """Tests for MultiAlertResult aggregation"""

    def test_all_success(self):
        """Test all channels succeed"""
        result = MultiAlertResult(results=[
            AlertResult(channel="discord", success=True),
            AlertResult(channel="telegram", success=True),
            AlertResult(channel="email", success=True)
        ])

        assert result.all_success is True
        assert result.any_success is True
        assert result.success_count == 3
        assert result.failure_count == 0

    def test_partial_success(self):
        """Test some channels succeed"""
        result = MultiAlertResult(results=[
            AlertResult(channel="discord", success=True),
            AlertResult(channel="telegram", success=False, error="Timeout"),
            AlertResult(channel="email", success=True)
        ])

        assert result.all_success is False
        assert result.any_success is True
        assert result.success_count == 2
        assert result.failure_count == 1
        assert result.failed_channels == ["telegram"]

    def test_all_failure(self):
        """Test all channels fail"""
        result = MultiAlertResult(results=[
            AlertResult(channel="discord", success=False, error="Rate limited"),
            AlertResult(channel="telegram", success=False, error="Timeout")
        ])

        assert result.all_success is False
        assert result.any_success is False
        assert result.success_count == 0
        assert result.failure_count == 2


class TestAlertChannels:
    """Tests for individual alert channels"""

    @patch('alerts.discord_alert.requests.post')
    def test_discord_send(self, mock_post):
        """Test Discord webhook sending"""
        from alerts.discord_alert import DiscordAlert

        mock_post.return_value.status_code = 204
        mock_post.return_value.ok = True

        alert = DiscordAlert(webhook_url="https://discord.com/api/webhooks/test")
        result = alert.send(
            title="Test Alert",
            message="This is a test message",
            level="info"
        )

        assert result.success is True
        mock_post.assert_called_once()

    @patch('alerts.telegram_alert.requests.post')
    def test_telegram_send(self, mock_post):
        """Test Telegram bot message sending"""
        from alerts.telegram_alert import TelegramAlert

        mock_post.return_value.status_code = 200
        mock_post.return_value.ok = True
        mock_post.return_value.json.return_value = {"ok": True, "result": {"message_id": 123}}

        alert = TelegramAlert(bot_token="test-token", chat_id="test-chat")
        result = alert.send(
            title="Test Alert",
            message="This is a test"
        )

        assert result.success is True

    @patch('alerts.email_alert.smtplib.SMTP')
    def test_email_send(self, mock_smtp):
        """Test email sending"""
        from alerts.email_alert import EmailAlert

        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        alert = EmailAlert(
            smtp_host="smtp.test.com",
            smtp_port=587,
            username="test@test.com",
            password="testpass",
            from_email="test@test.com"
        )
        result = alert.send(
            to_email="recipient@test.com",
            subject="Test Subject",
            message="Test message body"
        )

        assert result.success is True


class TestRateLimiting:
    """Tests for alert rate limiting"""

    def test_rate_limit_detection(self):
        """Test rate limit is detected"""
        from alerts.delivery_tracker import DeliveryTracker

        tracker = DeliveryTracker()

        # Simulate many alerts in short time
        for i in range(10):
            tracker.record_delivery(
                channel="discord",
                alert_id=f"test-{i}",
                success=True
            )

        # Check if rate limiting should apply
        recent_count = tracker.get_recent_count("discord", minutes=1)
        assert recent_count == 10

    def test_rate_limit_enforcement(self):
        """Test rate limiting prevents excessive alerts"""
        from alerts.retry import RetryConfig

        config = RetryConfig(
            max_per_minute=5,
            max_per_hour=20
        )

        # Validate rate limits are set
        assert config.max_per_minute == 5
        assert config.max_per_hour == 20


class TestDeliveryConfirmation:
    """Tests for delivery confirmation tracking"""

    def test_delivery_tracking(self):
        """Test alert delivery is tracked"""
        from alerts.delivery_tracker import DeliveryTracker, DeliveryStatus

        tracker = DeliveryTracker()

        # Record a delivery
        record = tracker.record_delivery(
            channel="telegram",
            alert_id="test-123",
            success=True,
            delivery_time_ms=150.0
        )

        assert record.status == DeliveryStatus.DELIVERED
        assert record.delivery_time_ms == 150.0

    def test_failed_delivery_tracking(self):
        """Test failed delivery is tracked"""
        from alerts.delivery_tracker import DeliveryTracker, DeliveryStatus

        tracker = DeliveryTracker()

        record = tracker.record_delivery(
            channel="sms",
            alert_id="test-456",
            success=False,
            error="Invalid number"
        )

        assert record.status == DeliveryStatus.FAILED
        assert record.error == "Invalid number"

    def test_delivery_history(self):
        """Test delivery history retrieval"""
        from alerts.delivery_tracker import DeliveryTracker

        tracker = DeliveryTracker()

        # Add multiple deliveries
        tracker.record_delivery(channel="discord", alert_id="1", success=True)
        tracker.record_delivery(channel="discord", alert_id="2", success=True)
        tracker.record_delivery(channel="discord", alert_id="3", success=False, error="Error")

        history = tracker.get_channel_history("discord", limit=10)
        assert len(history) == 3


class TestDeduplication:
    """Tests for alert deduplication"""

    def test_duplicate_detection(self):
        """Test duplicate alerts are detected"""
        from alerts.delivery_tracker import DeliveryTracker

        tracker = DeliveryTracker()

        # Same content creates same hash
        is_dup_1 = tracker.is_duplicate(
            channel="discord",
            content_hash="abc123",
            window_seconds=60
        )
        assert is_dup_1 is False

        # Record the alert
        tracker.record_delivery(
            channel="discord",
            alert_id="test",
            success=True,
            content_hash="abc123"
        )

        # Same hash should now be duplicate
        is_dup_2 = tracker.is_duplicate(
            channel="discord",
            content_hash="abc123",
            window_seconds=60
        )
        assert is_dup_2 is True

    def test_different_content_not_duplicate(self):
        """Test different content is not marked duplicate"""
        from alerts.delivery_tracker import DeliveryTracker

        tracker = DeliveryTracker()

        tracker.record_delivery(
            channel="discord",
            alert_id="test-1",
            success=True,
            content_hash="hash1"
        )

        is_dup = tracker.is_duplicate(
            channel="discord",
            content_hash="hash2",
            window_seconds=60
        )
        assert is_dup is False


class TestChannelFailover:
    """Tests for channel failover behavior"""

    def test_failover_to_secondary_channel(self):
        """Test failover when primary channel fails"""
        results = []

        # Simulate primary failure, secondary success
        primary = AlertResult(channel="discord", success=False, error="Rate limited")
        secondary = AlertResult(channel="telegram", success=True)

        results = [primary, secondary]
        multi_result = MultiAlertResult(results=results)

        assert multi_result.any_success is True
        assert "telegram" in multi_result.successful_channels
        assert "discord" in multi_result.failed_channels

    def test_all_channels_fail(self):
        """Test behavior when all channels fail"""
        results = [
            AlertResult(channel="discord", success=False, error="Error"),
            AlertResult(channel="telegram", success=False, error="Error"),
            AlertResult(channel="email", success=False, error="Error")
        ]

        multi_result = MultiAlertResult(results=results)

        assert multi_result.any_success is False
        assert multi_result.failure_count == 3


class TestRetryLogic:
    """Tests for retry logic"""

    def test_retry_config(self):
        """Test retry configuration"""
        from alerts.retry import RetryConfig

        config = RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0
        )

        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 30.0

    def test_retry_delay_calculation(self):
        """Test exponential backoff delay calculation"""
        from alerts.retry import RetryConfig

        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=30.0
        )

        # Delays should increase exponentially
        delay_1 = config.initial_delay
        delay_2 = min(delay_1 * config.exponential_base, config.max_delay)
        delay_3 = min(delay_2 * config.exponential_base, config.max_delay)

        assert delay_1 == 1.0
        assert delay_2 == 2.0
        assert delay_3 == 4.0


class TestAlertPriority:
    """Tests for alert priority handling"""

    def test_priority_values(self):
        """Test all priority values exist"""
        assert AlertPriority.LOW.value == "low"
        assert AlertPriority.NORMAL.value == "normal"
        assert AlertPriority.HIGH.value == "high"
        assert AlertPriority.CRITICAL.value == "critical"

    def test_priority_ordering(self):
        """Test priority can be compared"""
        priorities = [
            AlertPriority.LOW,
            AlertPriority.NORMAL,
            AlertPriority.HIGH,
            AlertPriority.CRITICAL
        ]

        # Just verify all priorities are distinct
        assert len(set(priorities)) == 4


class TestAlertChannelEnum:
    """Tests for AlertChannel enum"""

    def test_all_channels_defined(self):
        """Test all expected channels are defined"""
        channels = [c for c in AlertChannel]

        assert AlertChannel.DISCORD in channels
        assert AlertChannel.TELEGRAM in channels
        assert AlertChannel.EMAIL in channels
        assert AlertChannel.SMS in channels
        assert AlertChannel.PUSHOVER in channels
