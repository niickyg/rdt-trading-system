"""
Discord Alert Implementation
Sends alerts to Discord channels via webhooks.
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
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
class DiscordResult:
    """Result of a Discord webhook call."""
    success: bool
    error_message: Optional[str] = None
    error_code: Optional[int] = None
    retry_after: Optional[float] = None

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


class DiscordAlert:
    """
    Discord webhook alert handler.

    Sends notifications to Discord channels using webhook URLs.
    Requires DISCORD_WEBHOOK_URL environment variable.

    Priority Levels:
        - 'low': Gray color embed
        - 'normal': Blue color embed
        - 'high': Orange color embed
        - 'critical': Red color embed with @here mention
    """

    COLOR_MAP = {
        'low': 0x95a5a6,
        'normal': 0x3498db,
        'high': 0xf39c12,
        'critical': 0xe74c3c
    }

    PRIORITY_LABELS = {
        'low': 'Low',
        'normal': 'Normal',
        'high': 'High Priority',
        'critical': 'CRITICAL'
    }

    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize Discord alert handler.

        Args:
            webhook_url: Discord webhook URL (defaults to DISCORD_WEBHOOK_URL env var)
        """
        self.webhook_url = webhook_url or os.environ.get('DISCORD_WEBHOOK_URL', '')

    @property
    def is_configured(self) -> bool:
        """Check if Discord webhook is configured."""
        return bool(self.webhook_url)

    def _validate_credentials(self) -> bool:
        """Validate that webhook URL is present."""
        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured (DISCORD_WEBHOOK_URL)")
            return False
        return True

    def _create_embed(
        self,
        title: str,
        message: str,
        priority: str,
        fields: Optional[List[Dict[str, Any]]] = None,
        footer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a Discord embed object.

        Args:
            title: Embed title
            message: Embed description
            priority: Priority level for color coding
            fields: Optional list of field dicts with 'name', 'value', 'inline'
            footer: Optional footer text

        Returns:
            dict: Discord embed object
        """
        color = self.COLOR_MAP.get(priority.lower(), self.COLOR_MAP['normal'])
        priority_label = self.PRIORITY_LABELS.get(priority.lower(), 'Normal')

        embed = {
            'title': title,
            'description': message,
            'color': color,
            'timestamp': datetime.utcnow().isoformat(),
            'author': {
                'name': f'RDT Trading System - {priority_label}'
            }
        }

        if fields:
            embed['fields'] = fields

        if footer:
            embed['footer'] = {'text': footer}
        else:
            embed['footer'] = {'text': 'RDT Trading Alert System'}

        return embed

    def send_alert(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        fields: Optional[List[Dict[str, Any]]] = None,
        mention_here: Optional[bool] = None
    ) -> bool:
        """
        Send an alert to Discord via webhook.

        Args:
            title: Alert title
            message: Alert message body
            priority: Priority level ('low', 'normal', 'high', 'critical')
            fields: Optional list of embed fields [{'name': str, 'value': str, 'inline': bool}]
            mention_here: Whether to mention @here (defaults to True for critical)

        Returns:
            bool: True if alert was sent successfully, False otherwise
        """
        result = self.send_alert_with_result(
            title, message, priority, fields, mention_here
        )
        return result.success

    def send_alert_with_result(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        fields: Optional[List[Dict[str, Any]]] = None,
        mention_here: Optional[bool] = None
    ) -> DiscordResult:
        """
        Send an alert to Discord via webhook with detailed result.

        Args:
            title: Alert title
            message: Alert message body
            priority: Priority level ('low', 'normal', 'high', 'critical')
            fields: Optional list of embed fields
            mention_here: Whether to mention @here

        Returns:
            DiscordResult: Detailed result of the operation
        """
        if not self._validate_credentials():
            return DiscordResult(
                success=False,
                error_message='Discord webhook URL not configured'
            )

        priority_lower = priority.lower()
        if mention_here is None:
            mention_here = priority_lower == 'critical'

        embed = self._create_embed(title, message, priority, fields)

        payload = {
            'embeds': [embed],
            'username': 'RDT Trading Bot'
        }

        if mention_here:
            payload['content'] = '@here'

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=30
            )

            if response.status_code in (200, 204):
                logger.info(f"Discord alert sent: {title}")
                return DiscordResult(success=True)

            elif response.status_code == 429:
                try:
                    data = response.json()
                    retry_after = data.get('retry_after', 60)
                    if isinstance(retry_after, (int, float)):
                        retry_after = float(retry_after) / 1000
                    else:
                        retry_after = 60.0
                except Exception:
                    retry_after = 60.0

                error_msg = 'Rate limit exceeded'
                logger.warning(f"Discord rate limit, retry after {retry_after}s")
                return DiscordResult(
                    success=False,
                    error_message=error_msg,
                    error_code=429,
                    retry_after=retry_after
                )

            elif response.status_code == 401:
                error_msg = 'Invalid webhook URL or unauthorized'
                logger.error(f"Discord authentication error: {error_msg}")
                return DiscordResult(
                    success=False,
                    error_message=error_msg,
                    error_code=401
                )

            elif response.status_code == 404:
                error_msg = 'Webhook not found - URL may be invalid or deleted'
                logger.error(f"Discord webhook not found")
                return DiscordResult(
                    success=False,
                    error_message=error_msg,
                    error_code=404
                )

            elif response.status_code >= 500:
                error_msg = f'Server error: {response.status_code}'
                logger.error(f"Discord server error: {error_msg}")
                return DiscordResult(
                    success=False,
                    error_message=error_msg,
                    error_code=response.status_code
                )

            else:
                error_msg = f'HTTP error {response.status_code}: {response.text}'
                logger.error(f"Discord webhook error: {error_msg}")
                return DiscordResult(
                    success=False,
                    error_message=error_msg,
                    error_code=response.status_code
                )

        except requests.exceptions.Timeout:
            error_msg = "Request timed out"
            logger.error("Discord webhook request timed out")
            return DiscordResult(
                success=False,
                error_message=error_msg
            )

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {e}"
            logger.error(f"Discord webhook connection error: {e}")
            return DiscordResult(
                success=False,
                error_message=error_msg
            )

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {e}"
            logger.error(f"Discord webhook request failed: {e}")
            return DiscordResult(
                success=False,
                error_message=error_msg
            )

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(f"Unexpected error sending Discord alert: {e}")
            return DiscordResult(
                success=False,
                error_message=error_msg
            )

    def send_alert_raising(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        fields: Optional[List[Dict[str, Any]]] = None,
        mention_here: Optional[bool] = None
    ) -> bool:
        """
        Send an alert to Discord, raising exceptions on failure.

        This method is designed to work with the retry decorator.

        Args:
            title: Alert title
            message: Alert message body
            priority: Priority level
            fields: Optional embed fields
            mention_here: Whether to mention @here

        Returns:
            bool: True if successful

        Raises:
            ConfigurationError: If webhook URL is not configured
            AuthenticationError: If webhook is invalid
            RateLimitError: If rate limited
            ServiceUnavailableError: If server error
            NetworkError: If connection error
            AlertDeliveryError: For other failures
        """
        if not self._validate_credentials():
            raise ConfigurationError("Discord webhook URL not configured")

        result = self.send_alert_with_result(
            title, message, priority, fields, mention_here
        )

        if result.success:
            return True

        if result.error_code in (401, 404):
            raise AuthenticationError(result.error_message or "Invalid webhook")

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

    def send_trade_alert(
        self,
        action: str,
        symbol: str,
        price: float,
        quantity: int,
        reason: str,
        priority: str = 'normal'
    ) -> bool:
        """
        Send a formatted trade alert to Discord.

        Args:
            action: Trade action ('BUY', 'SELL', 'STOP_LOSS', etc.)
            symbol: Stock symbol
            price: Trade price
            quantity: Number of shares
            reason: Reason for the trade
            priority: Priority level

        Returns:
            bool: True if alert was sent successfully, False otherwise
        """
        action_emoji = {
            'BUY': ':chart_with_upwards_trend:',
            'SELL': ':chart_with_downwards_trend:',
            'STOP_LOSS': ':octagonal_sign:',
            'TAKE_PROFIT': ':moneybag:'
        }.get(action.upper(), ':bell:')

        title = f"{action_emoji} {action.upper()}: {symbol}"
        message = reason

        fields = [
            {'name': 'Symbol', 'value': symbol, 'inline': True},
            {'name': 'Price', 'value': f"${price:.2f}", 'inline': True},
            {'name': 'Quantity', 'value': str(quantity), 'inline': True},
            {'name': 'Total Value', 'value': f"${price * quantity:,.2f}", 'inline': True}
        ]

        return self.send_alert(title, message, priority, fields)
