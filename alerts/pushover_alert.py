"""
Pushover Alert Implementation
Sends push notifications to mobile devices via Pushover API.
"""

import os
from typing import Optional, Tuple
from dataclasses import dataclass
import requests
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
class PushoverResult:
    """Result of a Pushover API call."""
    success: bool
    error_message: Optional[str] = None
    error_code: Optional[int] = None
    retry_after: Optional[float] = None
    request_id: Optional[str] = None

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
        return False


class PushoverAlert:
    """
    Pushover push notification alert handler.

    Sends notifications to mobile devices using the Pushover API.
    Requires PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN environment variables.

    Priority Levels:
        - 'low': Quiet notification (-1)
        - 'normal': Normal priority (0)
        - 'high': High priority, bypasses quiet hours (1)
        - 'critical': Emergency priority, requires acknowledgment (2)
    """

    PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"

    PRIORITY_MAP = {
        'low': -1,
        'normal': 0,
        'high': 1,
        'critical': 2
    }

    def __init__(
        self,
        user_key: Optional[str] = None,
        api_token: Optional[str] = None
    ):
        """
        Initialize Pushover alert handler.

        Args:
            user_key: Pushover user key (defaults to PUSHOVER_USER_KEY env var)
            api_token: Pushover API token (defaults to PUSHOVER_API_TOKEN env var)
        """
        self.user_key = user_key or os.environ.get('PUSHOVER_USER_KEY', '')
        self.api_token = api_token or os.environ.get('PUSHOVER_API_TOKEN', '')
        self._validated = False

    @property
    def is_configured(self) -> bool:
        """Check if Pushover credentials are configured."""
        return bool(self.user_key and self.api_token)

    def _validate_credentials(self) -> bool:
        """Validate that credentials are present."""
        if not self.user_key:
            logger.warning("Pushover user key not configured (PUSHOVER_USER_KEY)")
            return False
        if not self.api_token:
            logger.warning("Pushover API token not configured (PUSHOVER_API_TOKEN)")
            return False
        return True

    def send_alert(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        sound: Optional[str] = None,
        url: Optional[str] = None,
        url_title: Optional[str] = None
    ) -> bool:
        """
        Send a push notification via Pushover.

        Args:
            title: Alert title
            message: Alert message body
            priority: Priority level ('low', 'normal', 'high', 'critical')
            sound: Optional sound name (e.g., 'siren', 'cashregister')
            url: Optional URL to include with notification
            url_title: Optional title for the URL

        Returns:
            bool: True if alert was sent successfully, False otherwise
        """
        result = self.send_alert_with_result(
            title, message, priority, sound, url, url_title
        )
        return result.success

    def send_alert_with_result(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        sound: Optional[str] = None,
        url: Optional[str] = None,
        url_title: Optional[str] = None
    ) -> PushoverResult:
        """
        Send a push notification via Pushover with detailed result.

        Args:
            title: Alert title
            message: Alert message body
            priority: Priority level ('low', 'normal', 'high', 'critical')
            sound: Optional sound name (e.g., 'siren', 'cashregister')
            url: Optional URL to include with notification
            url_title: Optional title for the URL

        Returns:
            PushoverResult: Detailed result of the operation
        """
        if not self._validate_credentials():
            return PushoverResult(
                success=False,
                error_message='Pushover credentials not configured'
            )

        pushover_priority = self.PRIORITY_MAP.get(priority.lower(), 0)

        payload = {
            'token': self.api_token,
            'user': self.user_key,
            'title': title,
            'message': message,
            'priority': pushover_priority,
        }

        if sound:
            payload['sound'] = sound
        else:
            if priority == 'critical':
                payload['sound'] = 'siren'
            elif priority == 'high':
                payload['sound'] = 'persistent'

        if url:
            payload['url'] = url
            if url_title:
                payload['url_title'] = url_title

        if pushover_priority == 2:
            payload['retry'] = 60
            payload['expire'] = 3600

        try:
            response = requests.post(
                self.PUSHOVER_API_URL,
                data=payload,
                timeout=30
            )

            request_id = response.headers.get('X-Request-Id')

            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 1:
                    logger.info(f"Pushover alert sent: {title}")
                    return PushoverResult(
                        success=True,
                        request_id=request_id
                    )
                else:
                    errors = result.get('errors', ['Unknown error'])
                    error_msg = ', '.join(errors)
                    logger.error(f"Pushover API error: {error_msg}")
                    return PushoverResult(
                        success=False,
                        error_message=error_msg,
                        request_id=request_id
                    )

            elif response.status_code == 429:
                retry_after = float(response.headers.get('Retry-After', 60))
                error_msg = 'Rate limit exceeded'
                logger.warning(f"Pushover rate limit, retry after {retry_after}s")
                return PushoverResult(
                    success=False,
                    error_message=error_msg,
                    error_code=429,
                    retry_after=retry_after,
                    request_id=request_id
                )

            elif response.status_code == 401:
                error_msg = 'Invalid API token or user key'
                logger.error(f"Pushover authentication error: {error_msg}")
                return PushoverResult(
                    success=False,
                    error_message=error_msg,
                    error_code=401,
                    request_id=request_id
                )

            elif response.status_code >= 500:
                error_msg = f'Server error: {response.status_code}'
                logger.error(f"Pushover server error: {error_msg}")
                return PushoverResult(
                    success=False,
                    error_message=error_msg,
                    error_code=response.status_code,
                    request_id=request_id
                )

            else:
                error_msg = f'HTTP error {response.status_code}: {response.text}'
                logger.error(f"Pushover HTTP error: {error_msg}")
                return PushoverResult(
                    success=False,
                    error_message=error_msg,
                    error_code=response.status_code,
                    request_id=request_id
                )

        except requests.exceptions.Timeout:
            error_msg = "Request timed out"
            logger.error(f"Pushover request timed out")
            return PushoverResult(
                success=False,
                error_message=error_msg
            )

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {e}"
            logger.error(f"Pushover connection error: {e}")
            return PushoverResult(
                success=False,
                error_message=error_msg
            )

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {e}"
            logger.error(f"Pushover request failed: {e}")
            return PushoverResult(
                success=False,
                error_message=error_msg
            )

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(f"Unexpected error sending Pushover alert: {e}")
            return PushoverResult(
                success=False,
                error_message=error_msg
            )

    def send_alert_raising(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        sound: Optional[str] = None,
        url: Optional[str] = None,
        url_title: Optional[str] = None
    ) -> bool:
        """
        Send a push notification via Pushover, raising exceptions on failure.

        This method is designed to work with the retry decorator.

        Args:
            title: Alert title
            message: Alert message body
            priority: Priority level ('low', 'normal', 'high', 'critical')
            sound: Optional sound name
            url: Optional URL to include
            url_title: Optional title for the URL

        Returns:
            bool: True if successful

        Raises:
            ConfigurationError: If credentials are not configured
            AuthenticationError: If API credentials are invalid
            RateLimitError: If rate limited
            ServiceUnavailableError: If server error
            NetworkError: If connection error
            AlertDeliveryError: For other failures
        """
        if not self._validate_credentials():
            raise ConfigurationError("Pushover credentials not configured")

        result = self.send_alert_with_result(
            title, message, priority, sound, url, url_title
        )

        if result.success:
            return True

        if result.error_code == 401:
            raise AuthenticationError(result.error_message or "Authentication failed")

        if result.error_code == 429:
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
            result.error_message or "Failed to send alert",
            retryable=result.is_retryable,
            retry_after=result.retry_after
        )
