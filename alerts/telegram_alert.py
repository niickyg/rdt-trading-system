"""
Telegram Alert Implementation
Sends alerts via Telegram Bot API.
"""

import os
from typing import Optional
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
class TelegramResult:
    """Result of a Telegram API call."""
    success: bool
    error_message: Optional[str] = None
    error_code: Optional[int] = None
    retry_after: Optional[float] = None
    message_id: Optional[int] = None

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


class TelegramAlert:
    """
    Telegram bot alert handler.

    Sends notifications via Telegram Bot API.
    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.

    Priority Levels:
        - 'low': Normal message, no notification sound
        - 'normal': Normal message with notification
        - 'high': Message with emphasis formatting
        - 'critical': Message with urgent formatting and multiple notifications
    """

    TELEGRAM_API_BASE = "https://api.telegram.org/bot"

    PRIORITY_EMOJI = {
        'low': 'i',
        'normal': '#',
        'high': '!',
        'critical': '!!'
    }

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None
    ):
        """
        Initialize Telegram alert handler.

        Args:
            bot_token: Telegram bot token (defaults to TELEGRAM_BOT_TOKEN env var)
            chat_id: Telegram chat ID (defaults to TELEGRAM_CHAT_ID env var)
        """
        self.bot_token = bot_token or os.environ.get('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = chat_id or os.environ.get('TELEGRAM_CHAT_ID', '')

    @property
    def is_configured(self) -> bool:
        """Check if Telegram credentials are configured."""
        return bool(self.bot_token and self.chat_id)

    def _validate_credentials(self) -> bool:
        """Validate that credentials are present."""
        if not self.bot_token:
            logger.warning("Telegram bot token not configured (TELEGRAM_BOT_TOKEN)")
            return False
        if not self.chat_id:
            logger.warning("Telegram chat ID not configured (TELEGRAM_CHAT_ID)")
            return False
        return True

    def _get_api_url(self, method: str) -> str:
        """Get full API URL for a method."""
        return f"{self.TELEGRAM_API_BASE}{self.bot_token}/{method}"

    def _format_message(
        self,
        title: str,
        message: str,
        priority: str
    ) -> str:
        """
        Format message with priority-based styling.

        Args:
            title: Alert title
            message: Alert message body
            priority: Priority level

        Returns:
            str: Formatted message with HTML markup
        """
        priority_marker = self.PRIORITY_EMOJI.get(priority.lower(), self.PRIORITY_EMOJI['normal'])
        priority_label = priority.upper()

        if priority.lower() == 'critical':
            formatted = (
                f"[{priority_marker}] <b>CRITICAL ALERT</b> [{priority_marker}]\n"
                f"--------------------\n"
                f"<b>{title}</b>\n"
                f"--------------------\n"
                f"{message}\n"
                f"--------------------\n"
                f"<i>RDT Trading System</i>"
            )
        elif priority.lower() == 'high':
            formatted = (
                f"[{priority_marker}] <b>HIGH PRIORITY</b>\n"
                f"<b>{title}</b>\n\n"
                f"{message}\n\n"
                f"<i>RDT Trading System</i>"
            )
        else:
            formatted = (
                f"[{priority_marker}] <b>{title}</b>\n\n"
                f"{message}\n\n"
                f"<i>RDT Trading System</i>"
            )

        return formatted

    def send_alert(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        disable_notification: Optional[bool] = None,
        parse_mode: str = 'HTML'
    ) -> bool:
        """
        Send an alert via Telegram.

        Args:
            title: Alert title
            message: Alert message body
            priority: Priority level ('low', 'normal', 'high', 'critical')
            disable_notification: Whether to send silently (defaults based on priority)
            parse_mode: Message parse mode ('HTML' or 'Markdown')

        Returns:
            bool: True if alert was sent successfully, False otherwise
        """
        result = self.send_alert_with_result(
            title, message, priority, disable_notification, parse_mode
        )
        return result.success

    def send_alert_with_result(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        disable_notification: Optional[bool] = None,
        parse_mode: str = 'HTML'
    ) -> TelegramResult:
        """
        Send an alert via Telegram with detailed result.

        Args:
            title: Alert title
            message: Alert message body
            priority: Priority level
            disable_notification: Whether to send silently
            parse_mode: Message parse mode

        Returns:
            TelegramResult: Detailed result of the operation
        """
        if not self._validate_credentials():
            return TelegramResult(
                success=False,
                error_message='Telegram credentials not configured'
            )

        priority_lower = priority.lower()
        if disable_notification is None:
            disable_notification = priority_lower == 'low'

        formatted_message = self._format_message(title, message, priority)

        payload = {
            'chat_id': self.chat_id,
            'text': formatted_message,
            'parse_mode': parse_mode,
            'disable_notification': disable_notification
        }

        try:
            url = self._get_api_url('sendMessage')
            response = requests.post(url, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    message_id = result.get('result', {}).get('message_id')
                    logger.info(f"Telegram alert sent: {title}")
                    return TelegramResult(
                        success=True,
                        message_id=message_id
                    )
                else:
                    error_code = result.get('error_code')
                    error_description = result.get('description', 'Unknown error')
                    logger.error(f"Telegram API error: {error_description}")

                    if error_code == 429:
                        parameters = result.get('parameters', {})
                        retry_after = parameters.get('retry_after', 60)
                        return TelegramResult(
                            success=False,
                            error_message=error_description,
                            error_code=429,
                            retry_after=float(retry_after)
                        )

                    return TelegramResult(
                        success=False,
                        error_message=error_description,
                        error_code=error_code
                    )

            elif response.status_code == 429:
                try:
                    data = response.json()
                    parameters = data.get('parameters', {})
                    retry_after = parameters.get('retry_after', 60)
                except Exception:
                    retry_after = 60

                error_msg = 'Rate limit exceeded'
                logger.warning(f"Telegram rate limit, retry after {retry_after}s")
                return TelegramResult(
                    success=False,
                    error_message=error_msg,
                    error_code=429,
                    retry_after=float(retry_after)
                )

            elif response.status_code == 401:
                error_msg = 'Invalid bot token'
                logger.error(f"Telegram authentication error: {error_msg}")
                return TelegramResult(
                    success=False,
                    error_message=error_msg,
                    error_code=401
                )

            elif response.status_code == 400:
                try:
                    data = response.json()
                    error_msg = data.get('description', 'Bad request')
                except Exception:
                    error_msg = 'Bad request'
                logger.error(f"Telegram bad request: {error_msg}")
                return TelegramResult(
                    success=False,
                    error_message=error_msg,
                    error_code=400
                )

            elif response.status_code >= 500:
                error_msg = f'Server error: {response.status_code}'
                logger.error(f"Telegram server error: {error_msg}")
                return TelegramResult(
                    success=False,
                    error_message=error_msg,
                    error_code=response.status_code
                )

            else:
                error_msg = f'HTTP error {response.status_code}: {response.text}'
                logger.error(f"Telegram HTTP error: {error_msg}")
                return TelegramResult(
                    success=False,
                    error_message=error_msg,
                    error_code=response.status_code
                )

        except requests.exceptions.Timeout:
            error_msg = "Request timed out"
            logger.error("Telegram request timed out")
            return TelegramResult(
                success=False,
                error_message=error_msg
            )

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {e}"
            logger.error(f"Telegram connection error: {e}")
            return TelegramResult(
                success=False,
                error_message=error_msg
            )

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {e}"
            logger.error(f"Telegram request failed: {e}")
            return TelegramResult(
                success=False,
                error_message=error_msg
            )

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(f"Unexpected error sending Telegram alert: {e}")
            return TelegramResult(
                success=False,
                error_message=error_msg
            )

    def send_alert_raising(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        disable_notification: Optional[bool] = None,
        parse_mode: str = 'HTML'
    ) -> bool:
        """
        Send an alert via Telegram, raising exceptions on failure.

        This method is designed to work with the retry decorator.

        Args:
            title: Alert title
            message: Alert message body
            priority: Priority level
            disable_notification: Whether to send silently
            parse_mode: Message parse mode

        Returns:
            bool: True if successful

        Raises:
            ConfigurationError: If credentials are not configured
            AuthenticationError: If bot token is invalid
            RateLimitError: If rate limited
            ServiceUnavailableError: If server error
            NetworkError: If connection error
            AlertDeliveryError: For other failures
        """
        if not self._validate_credentials():
            raise ConfigurationError("Telegram credentials not configured")

        result = self.send_alert_with_result(
            title, message, priority, disable_notification, parse_mode
        )

        if result.success:
            return True

        if result.error_code == 401:
            raise AuthenticationError(result.error_message or "Invalid bot token")

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

    def send_photo_alert(
        self,
        title: str,
        message: str,
        photo_url: str,
        priority: str = 'normal'
    ) -> bool:
        """
        Send an alert with a photo via Telegram.

        Args:
            title: Alert title
            message: Alert message body (caption)
            photo_url: URL of the photo to send
            priority: Priority level

        Returns:
            bool: True if alert was sent successfully, False otherwise
        """
        if not self._validate_credentials():
            return False

        priority_lower = priority.lower()
        disable_notification = priority_lower == 'low'

        formatted_caption = self._format_message(title, message, priority)

        payload = {
            'chat_id': self.chat_id,
            'photo': photo_url,
            'caption': formatted_caption,
            'parse_mode': 'HTML',
            'disable_notification': disable_notification
        }

        try:
            url = self._get_api_url('sendPhoto')
            response = requests.post(url, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    logger.info(f"Telegram photo alert sent: {title}")
                    return True
                else:
                    error_description = result.get('description', 'Unknown error')
                    logger.error(f"Telegram API error: {error_description}")
                    return False
            else:
                logger.error(
                    f"Telegram HTTP error: {response.status_code} - {response.text}"
                )
                return False

        except requests.exceptions.Timeout:
            logger.error("Telegram photo request timed out")
            return False
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Telegram photo connection error: {e}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Telegram photo request failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram photo alert: {e}")
            return False

    def get_bot_info(self) -> Optional[dict]:
        """
        Get information about the bot (useful for testing configuration).

        Returns:
            dict: Bot information if successful, None otherwise
        """
        if not self.bot_token:
            logger.warning("Telegram bot token not configured")
            return None

        try:
            url = self._get_api_url('getMe')
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    return result.get('result')

            return None

        except Exception as e:
            logger.error(f"Failed to get bot info: {e}")
            return None
