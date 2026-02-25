"""
SMS Alert Implementation
Sends alerts via SMS using Twilio.
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from loguru import logger

from .retry import (
    AlertDeliveryError,
    RateLimitError,
    AuthenticationError,
    ConfigurationError,
    NetworkError,
    ServiceUnavailableError,
)


@dataclass
class SMSResult:
    """Result of an SMS send operation."""
    success: bool
    error_message: Optional[str] = None
    error_code: Optional[int] = None
    retry_after: Optional[float] = None
    message_sid: Optional[str] = None

    @property
    def is_retryable(self) -> bool:
        """Check if the error is retryable."""
        if self.success:
            return False
        if self.error_code is not None:
            if self.error_code == 429:
                return True
            if self.error_code >= 500:
                return True
            # Twilio-specific retryable error codes
            if self.error_code in (20429, 20500, 20503):
                return True
        return False


class SMSAlert:
    """
    SMS alert handler using Twilio.

    Sends notifications via SMS using Twilio's REST API.
    Requires TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER
    environment variables.

    Messages are automatically truncated to fit SMS limits (160 characters
    for single SMS, or ~1530 characters for concatenated SMS).

    Priority Levels affect message formatting:
        - 'low': Normal message
        - 'normal': Normal message
        - 'high': Message prefixed with [URGENT]
        - 'critical': Message prefixed with [CRITICAL]
    """

    # SMS character limits
    MAX_SMS_LENGTH = 160  # Single SMS
    MAX_CONCAT_LENGTH = 1530  # ~10 concatenated SMS segments

    # Priority prefixes
    PRIORITY_PREFIX = {
        'low': '',
        'normal': '',
        'high': '[URGENT] ',
        'critical': '[CRITICAL] ',
    }

    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_number: Optional[str] = None,
        to_number: Optional[str] = None
    ):
        """
        Initialize SMS alert handler.

        Args:
            account_sid: Twilio account SID (defaults to TWILIO_ACCOUNT_SID env var)
            auth_token: Twilio auth token (defaults to TWILIO_AUTH_TOKEN env var)
            from_number: Twilio phone number (defaults to TWILIO_FROM_NUMBER env var)
            to_number: Recipient phone number (defaults to TWILIO_TO_NUMBER env var)
        """
        self.account_sid = account_sid or os.environ.get('TWILIO_ACCOUNT_SID', '')
        self.auth_token = auth_token or os.environ.get('TWILIO_AUTH_TOKEN', '')
        self.from_number = from_number or os.environ.get('TWILIO_FROM_NUMBER', '')
        self.to_number = to_number or os.environ.get('TWILIO_TO_NUMBER', '')
        self._client = None

    @property
    def is_configured(self) -> bool:
        """Check if Twilio credentials are configured."""
        return bool(
            self.account_sid and
            self.auth_token and
            self.from_number and
            self.to_number
        )

    def _validate_credentials(self) -> bool:
        """Validate that all credentials are present."""
        if not self.account_sid:
            logger.warning("Twilio account SID not configured (TWILIO_ACCOUNT_SID)")
            return False
        if not self.auth_token:
            logger.warning("Twilio auth token not configured (TWILIO_AUTH_TOKEN)")
            return False
        if not self.from_number:
            logger.warning("Twilio from number not configured (TWILIO_FROM_NUMBER)")
            return False
        if not self.to_number:
            logger.warning("Twilio to number not configured (TWILIO_TO_NUMBER)")
            return False
        return True

    def _get_client(self):
        """Get or create Twilio client."""
        if self._client is None:
            try:
                from twilio.rest import Client
                self._client = Client(self.account_sid, self.auth_token)
            except ImportError:
                raise ConfigurationError(
                    "twilio library not installed. Install with: pip install twilio"
                )
        return self._client

    def _format_message(
        self,
        title: str,
        message: str,
        priority: str,
        max_length: Optional[int] = None
    ) -> str:
        """
        Format and truncate message for SMS.

        Args:
            title: Alert title
            message: Alert message
            priority: Priority level
            max_length: Max message length (default: MAX_CONCAT_LENGTH)

        Returns:
            str: Formatted message
        """
        max_length = max_length or self.MAX_CONCAT_LENGTH
        prefix = self.PRIORITY_PREFIX.get(priority.lower(), '')

        # Format: [PRIORITY] Title: Message
        formatted = f"{prefix}{title}: {message}"

        # Truncate if too long
        if len(formatted) > max_length:
            # Leave room for ellipsis
            formatted = formatted[:max_length - 3] + "..."

        return formatted

    def _format_trade_message(
        self,
        action: str,
        symbol: str,
        price: float,
        quantity: int,
        reason: str,
        priority: str = 'normal'
    ) -> str:
        """
        Format a concise trade message for SMS.

        Args:
            action: Trade action
            symbol: Stock symbol
            price: Trade price
            quantity: Number of shares
            reason: Trade reason
            priority: Priority level

        Returns:
            str: Formatted SMS message
        """
        prefix = self.PRIORITY_PREFIX.get(priority.lower(), '')
        total = price * quantity

        # Concise format for SMS
        msg = (
            f"{prefix}{action.upper()} {symbol}\n"
            f"{quantity} @ ${price:.2f} = ${total:,.2f}\n"
            f"{reason}"
        )

        # Truncate if needed
        if len(msg) > self.MAX_CONCAT_LENGTH:
            # Truncate reason
            available = self.MAX_CONCAT_LENGTH - len(msg) + len(reason) - 3
            reason_truncated = reason[:available] + "..."
            msg = (
                f"{prefix}{action.upper()} {symbol}\n"
                f"{quantity} @ ${price:.2f} = ${total:,.2f}\n"
                f"{reason_truncated}"
            )

        return msg

    def send_alert(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        to_number: Optional[str] = None
    ) -> bool:
        """
        Send an SMS alert.

        Args:
            title: Alert title
            message: Alert message
            priority: Priority level
            to_number: Recipient number (optional, uses default)

        Returns:
            bool: True if sent successfully
        """
        result = self.send_alert_with_result(title, message, priority, to_number)
        return result.success

    def send_alert_with_result(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        to_number: Optional[str] = None
    ) -> SMSResult:
        """
        Send an SMS alert with detailed result.

        Args:
            title: Alert title
            message: Alert message
            priority: Priority level
            to_number: Recipient number (optional)

        Returns:
            SMSResult: Detailed result
        """
        if not self._validate_credentials():
            return SMSResult(
                success=False,
                error_message="Twilio credentials not configured"
            )

        to_number = to_number or self.to_number
        formatted_message = self._format_message(title, message, priority)

        try:
            client = self._get_client()
        except ConfigurationError as e:
            return SMSResult(
                success=False,
                error_message=str(e)
            )

        try:
            twilio_message = client.messages.create(
                body=formatted_message,
                from_=self.from_number,
                to=to_number
            )

            logger.info(f"SMS sent to {to_number}: {twilio_message.sid}")
            return SMSResult(
                success=True,
                message_sid=twilio_message.sid
            )

        except Exception as e:
            # Handle Twilio-specific exceptions
            error_name = type(e).__name__
            error_msg = str(e)

            # Try to extract error code from Twilio exception
            error_code = None
            if hasattr(e, 'code'):
                error_code = e.code
            elif hasattr(e, 'status'):
                error_code = e.status

            if 'Authentication' in error_name or error_code == 20003:
                logger.error(f"Twilio authentication error: {e}")
                return SMSResult(
                    success=False,
                    error_message="Twilio authentication failed",
                    error_code=401
                )

            if error_code == 20429 or 'rate limit' in error_msg.lower():
                logger.warning(f"Twilio rate limit: {e}")
                return SMSResult(
                    success=False,
                    error_message="Twilio rate limit exceeded",
                    error_code=429,
                    retry_after=60.0
                )

            if error_code and error_code >= 500:
                logger.error(f"Twilio server error: {e}")
                return SMSResult(
                    success=False,
                    error_message=f"Twilio server error: {error_code}",
                    error_code=error_code
                )

            if 'Invalid' in error_name or error_code == 21211:
                logger.error(f"Twilio invalid number: {e}")
                return SMSResult(
                    success=False,
                    error_message="Invalid phone number",
                    error_code=400
                )

            if 'connection' in error_msg.lower() or 'timeout' in error_msg.lower():
                logger.error(f"Twilio network error: {e}")
                return SMSResult(
                    success=False,
                    error_message=f"Network error: {error_msg}"
                )

            logger.error(f"Twilio error: {e}")
            return SMSResult(
                success=False,
                error_message=error_msg,
                error_code=error_code
            )

    def send_alert_raising(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        to_number: Optional[str] = None
    ) -> bool:
        """
        Send an SMS alert, raising exceptions on failure.

        This method is designed to work with the retry decorator.

        Args:
            title: Alert title
            message: Alert message
            priority: Priority level
            to_number: Recipient number (optional)

        Returns:
            bool: True if successful

        Raises:
            ConfigurationError: If not configured
            AuthenticationError: If authentication fails
            RateLimitError: If rate limited
            ServiceUnavailableError: If server error
            NetworkError: If connection error
            AlertDeliveryError: For other failures
        """
        if not self._validate_credentials():
            raise ConfigurationError("Twilio credentials not configured")

        result = self.send_alert_with_result(title, message, priority, to_number)

        if result.success:
            return True

        if result.error_code == 401 or result.error_code == 20003:
            raise AuthenticationError(result.error_message or "Authentication failed")

        if result.error_code == 429 or result.error_code == 20429:
            raise RateLimitError(
                result.error_message or "Rate limit exceeded",
                retry_after=result.retry_after
            )

        if result.error_code is not None and result.error_code >= 500:
            raise ServiceUnavailableError(result.error_message or "Server error")

        if "connection" in (result.error_message or "").lower():
            raise NetworkError(result.error_message or "Network error")

        if "timeout" in (result.error_message or "").lower():
            raise NetworkError(result.error_message or "Request timeout")

        raise AlertDeliveryError(
            result.error_message or "Failed to send SMS",
            retryable=result.is_retryable,
            retry_after=result.retry_after
        )

    def send_trade_alert(
        self,
        action: str,
        symbol: str,
        price: float,
        quantity: int,
        reason: str,
        priority: str = 'normal',
        to_number: Optional[str] = None
    ) -> bool:
        """
        Send a formatted trade alert SMS.

        Args:
            action: Trade action (BUY, SELL, etc.)
            symbol: Stock symbol
            price: Trade price
            quantity: Number of shares
            reason: Reason for trade
            priority: Alert priority
            to_number: Recipient number (optional)

        Returns:
            bool: True if sent successfully
        """
        if not self._validate_credentials():
            return False

        to_number = to_number or self.to_number
        formatted_message = self._format_trade_message(
            action, symbol, price, quantity, reason, priority
        )

        try:
            client = self._get_client()
            twilio_message = client.messages.create(
                body=formatted_message,
                from_=self.from_number,
                to=to_number
            )
            logger.info(f"Trade SMS sent to {to_number}: {twilio_message.sid}")
            return True

        except Exception as e:
            logger.error(f"Failed to send trade SMS: {e}")
            return False

    def send_quick_alert(
        self,
        message: str,
        to_number: Optional[str] = None
    ) -> bool:
        """
        Send a quick SMS without formatting.

        Args:
            message: Message to send (will be truncated if too long)
            to_number: Recipient number (optional)

        Returns:
            bool: True if sent successfully
        """
        if not self._validate_credentials():
            return False

        to_number = to_number or self.to_number

        # Truncate if needed
        if len(message) > self.MAX_CONCAT_LENGTH:
            message = message[:self.MAX_CONCAT_LENGTH - 3] + "..."

        try:
            client = self._get_client()
            twilio_message = client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            logger.info(f"Quick SMS sent to {to_number}: {twilio_message.sid}")
            return True

        except Exception as e:
            logger.error(f"Failed to send quick SMS: {e}")
            return False

    def send_batch(
        self,
        messages: List[Dict[str, Any]],
        to_number: Optional[str] = None
    ) -> List[SMSResult]:
        """
        Send multiple SMS messages.

        Args:
            messages: List of message dicts with 'title', 'message', 'priority'
            to_number: Recipient number (optional)

        Returns:
            List[SMSResult]: Results for each message
        """
        results = []
        to_number = to_number or self.to_number

        for msg in messages:
            result = self.send_alert_with_result(
                title=msg.get('title', 'Alert'),
                message=msg.get('message', ''),
                priority=msg.get('priority', 'normal'),
                to_number=to_number
            )
            results.append(result)

        return results

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get Twilio account information (useful for testing configuration).

        Returns:
            dict: Account information if successful, None otherwise
        """
        if not self._validate_credentials():
            return None

        try:
            client = self._get_client()
            account = client.api.accounts(self.account_sid).fetch()

            return {
                'sid': account.sid,
                'friendly_name': account.friendly_name,
                'status': account.status,
                'type': account.type
            }

        except Exception as e:
            logger.error(f"Failed to get Twilio account info: {e}")
            return None

    def test_connection(self) -> bool:
        """
        Test Twilio connection by fetching account info.

        Returns:
            bool: True if connection successful
        """
        return self.get_account_info() is not None
