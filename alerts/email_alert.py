"""
Email Alert Implementation
Sends alerts via email using multiple providers: SMTP, SendGrid, AWS SES.
"""

import os
import smtplib
import ssl
from abc import ABC, abstractmethod
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from .retry import (
    AlertDeliveryError,
    RateLimitError,
    AuthenticationError,
    ConfigurationError,
    NetworkError,
    ServiceUnavailableError,
)
from .email_templates import EmailTemplates, EmailContent


@dataclass
class EmailResult:
    """Result of an email send operation."""
    success: bool
    error_message: Optional[str] = None
    error_code: Optional[int] = None
    retry_after: Optional[float] = None
    message_id: Optional[str] = None
    provider: Optional[str] = None

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


class EmailProvider(ABC):
    """Abstract base class for email providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if provider is properly configured."""
        pass

    @abstractmethod
    def send(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        plain_text: str,
        from_email: Optional[str] = None,
        from_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None
    ) -> EmailResult:
        """
        Send an email.

        Args:
            to_email: Recipient email address
            subject: Email subject
            html_body: HTML email body
            plain_text: Plain text email body
            from_email: Sender email (optional, uses default)
            from_name: Sender name (optional)
            reply_to: Reply-to email (optional)
            cc: CC recipients (optional)
            bcc: BCC recipients (optional)

        Returns:
            EmailResult: Result of the send operation
        """
        pass


class SMTPProvider(EmailProvider):
    """SMTP email provider."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
        use_ssl: bool = False,
        timeout: int = 30,
        default_from_email: Optional[str] = None,
        default_from_name: str = "RDT Trading System"
    ):
        """
        Initialize SMTP provider.

        Args:
            host: SMTP host (defaults to SMTP_HOST env var)
            port: SMTP port (defaults to SMTP_PORT env var)
            username: SMTP username (defaults to SMTP_USER env var)
            password: SMTP password (defaults to SMTP_PASSWORD env var)
            use_tls: Use STARTTLS (default True)
            use_ssl: Use SSL/TLS (default False, for port 465)
            timeout: Connection timeout in seconds
            default_from_email: Default sender email
            default_from_name: Default sender name
        """
        self.host = host or os.environ.get('SMTP_HOST', 'smtp.gmail.com')
        self.port = int(port or os.environ.get('SMTP_PORT', '587'))
        self.username = username or os.environ.get('SMTP_USER', '')
        self.password = password or os.environ.get('SMTP_PASSWORD', '')
        self.use_tls = use_tls
        self.use_ssl = use_ssl
        self.timeout = timeout
        self.default_from_email = (
            default_from_email or
            os.environ.get('EMAIL_FROM', self.username)
        )
        self.default_from_name = default_from_name

    @property
    def name(self) -> str:
        return "smtp"

    @property
    def is_configured(self) -> bool:
        return bool(self.host and self.port and self.username and self.password)

    def send(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        plain_text: str,
        from_email: Optional[str] = None,
        from_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None
    ) -> EmailResult:
        """Send email via SMTP."""
        if not self.is_configured:
            return EmailResult(
                success=False,
                error_message="SMTP not configured",
                provider=self.name
            )

        from_email = from_email or self.default_from_email
        from_name = from_name or self.default_from_name

        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = formataddr((from_name, from_email))
        msg['To'] = to_email

        if reply_to:
            msg['Reply-To'] = reply_to
        if cc:
            msg['Cc'] = ', '.join(cc)

        # Attach parts
        msg.attach(MIMEText(plain_text, 'plain', 'utf-8'))
        msg.attach(MIMEText(html_body, 'html', 'utf-8'))

        # Build recipient list
        recipients = [to_email]
        if cc:
            recipients.extend(cc)
        if bcc:
            recipients.extend(bcc)

        try:
            # Connect and send
            if self.use_ssl:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(
                    self.host, self.port, timeout=self.timeout, context=context
                ) as server:
                    server.login(self.username, self.password)
                    server.sendmail(from_email, recipients, msg.as_string())
            else:
                with smtplib.SMTP(
                    self.host, self.port, timeout=self.timeout
                ) as server:
                    if self.use_tls:
                        context = ssl.create_default_context()
                        server.starttls(context=context)
                    server.login(self.username, self.password)
                    server.sendmail(from_email, recipients, msg.as_string())

            logger.info(f"Email sent via SMTP to {to_email}")
            return EmailResult(
                success=True,
                provider=self.name,
                message_id=msg['Message-ID']
            )

        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"SMTP authentication failed: {e}"
            logger.error(error_msg)
            return EmailResult(
                success=False,
                error_message=error_msg,
                error_code=535,
                provider=self.name
            )

        except smtplib.SMTPRecipientsRefused as e:
            error_msg = f"Recipients refused: {e}"
            logger.error(error_msg)
            return EmailResult(
                success=False,
                error_message=error_msg,
                error_code=550,
                provider=self.name
            )

        except smtplib.SMTPServerDisconnected as e:
            error_msg = f"Server disconnected: {e}"
            logger.error(error_msg)
            return EmailResult(
                success=False,
                error_message=error_msg,
                error_code=421,
                provider=self.name
            )

        except smtplib.SMTPException as e:
            error_msg = f"SMTP error: {e}"
            logger.error(error_msg)
            return EmailResult(
                success=False,
                error_message=error_msg,
                provider=self.name
            )

        except TimeoutError as e:
            error_msg = f"SMTP connection timed out: {e}"
            logger.error(error_msg)
            return EmailResult(
                success=False,
                error_message=error_msg,
                provider=self.name
            )

        except Exception as e:
            error_msg = f"Unexpected error sending email: {e}"
            logger.error(error_msg)
            return EmailResult(
                success=False,
                error_message=error_msg,
                provider=self.name
            )


class SendGridProvider(EmailProvider):
    """SendGrid email provider."""

    SENDGRID_API_URL = "https://api.sendgrid.com/v3/mail/send"

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_from_email: Optional[str] = None,
        default_from_name: str = "RDT Trading System"
    ):
        """
        Initialize SendGrid provider.

        Args:
            api_key: SendGrid API key (defaults to SENDGRID_API_KEY env var)
            default_from_email: Default sender email
            default_from_name: Default sender name
        """
        self.api_key = api_key or os.environ.get('SENDGRID_API_KEY', '')
        self.default_from_email = (
            default_from_email or
            os.environ.get('SENDGRID_FROM_EMAIL', '')
        )
        self.default_from_name = default_from_name

    @property
    def name(self) -> str:
        return "sendgrid"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.default_from_email)

    def send(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        plain_text: str,
        from_email: Optional[str] = None,
        from_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None
    ) -> EmailResult:
        """Send email via SendGrid API."""
        if not self.is_configured:
            return EmailResult(
                success=False,
                error_message="SendGrid not configured",
                provider=self.name
            )

        try:
            import requests
        except ImportError:
            return EmailResult(
                success=False,
                error_message="requests library not installed",
                provider=self.name
            )

        from_email = from_email or self.default_from_email
        from_name = from_name or self.default_from_name

        # Build request payload
        payload = {
            "personalizations": [
                {
                    "to": [{"email": to_email}]
                }
            ],
            "from": {
                "email": from_email,
                "name": from_name
            },
            "subject": subject,
            "content": [
                {"type": "text/plain", "value": plain_text},
                {"type": "text/html", "value": html_body}
            ]
        }

        if reply_to:
            payload["reply_to"] = {"email": reply_to}

        if cc:
            payload["personalizations"][0]["cc"] = [{"email": e} for e in cc]

        if bcc:
            payload["personalizations"][0]["bcc"] = [{"email": e} for e in bcc]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                self.SENDGRID_API_URL,
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code in (200, 201, 202):
                message_id = response.headers.get('X-Message-Id')
                logger.info(f"Email sent via SendGrid to {to_email}")
                return EmailResult(
                    success=True,
                    provider=self.name,
                    message_id=message_id
                )

            elif response.status_code == 429:
                retry_after = float(response.headers.get('Retry-After', 60))
                error_msg = "SendGrid rate limit exceeded"
                logger.warning(f"{error_msg}, retry after {retry_after}s")
                return EmailResult(
                    success=False,
                    error_message=error_msg,
                    error_code=429,
                    retry_after=retry_after,
                    provider=self.name
                )

            elif response.status_code == 401:
                error_msg = "SendGrid authentication failed"
                logger.error(error_msg)
                return EmailResult(
                    success=False,
                    error_message=error_msg,
                    error_code=401,
                    provider=self.name
                )

            elif response.status_code >= 500:
                error_msg = f"SendGrid server error: {response.status_code}"
                logger.error(error_msg)
                return EmailResult(
                    success=False,
                    error_message=error_msg,
                    error_code=response.status_code,
                    provider=self.name
                )

            else:
                try:
                    error_data = response.json()
                    errors = error_data.get('errors', [])
                    error_msg = '; '.join(e.get('message', '') for e in errors)
                except Exception:
                    error_msg = response.text
                logger.error(f"SendGrid error: {error_msg}")
                return EmailResult(
                    success=False,
                    error_message=error_msg,
                    error_code=response.status_code,
                    provider=self.name
                )

        except requests.exceptions.Timeout:
            error_msg = "SendGrid request timed out"
            logger.error(error_msg)
            return EmailResult(
                success=False,
                error_message=error_msg,
                provider=self.name
            )

        except requests.exceptions.ConnectionError as e:
            error_msg = f"SendGrid connection error: {e}"
            logger.error(error_msg)
            return EmailResult(
                success=False,
                error_message=error_msg,
                provider=self.name
            )

        except Exception as e:
            error_msg = f"SendGrid unexpected error: {e}"
            logger.error(error_msg)
            return EmailResult(
                success=False,
                error_message=error_msg,
                provider=self.name
            )


class AWSSESProvider(EmailProvider):
    """AWS Simple Email Service (SES) provider."""

    def __init__(
        self,
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        default_from_email: Optional[str] = None,
        default_from_name: str = "RDT Trading System"
    ):
        """
        Initialize AWS SES provider.

        Args:
            region: AWS region (defaults to AWS_SES_REGION env var)
            access_key: AWS access key (defaults to AWS_ACCESS_KEY env var)
            secret_key: AWS secret key (defaults to AWS_SECRET_KEY env var)
            default_from_email: Default sender email
            default_from_name: Default sender name
        """
        self.region = region or os.environ.get('AWS_SES_REGION', 'us-east-1')
        self.access_key = access_key or os.environ.get('AWS_ACCESS_KEY', '')
        self.secret_key = secret_key or os.environ.get('AWS_SECRET_KEY', '')
        self.default_from_email = (
            default_from_email or
            os.environ.get('AWS_SES_FROM_EMAIL', '')
        )
        self.default_from_name = default_from_name
        self._client = None

    @property
    def name(self) -> str:
        return "ses"

    @property
    def is_configured(self) -> bool:
        return bool(
            self.region and
            self.access_key and
            self.secret_key and
            self.default_from_email
        )

    def _get_client(self):
        """Get or create boto3 SES client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client(
                    'ses',
                    region_name=self.region,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key
                )
            except ImportError:
                raise ConfigurationError("boto3 library not installed")
        return self._client

    def send(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        plain_text: str,
        from_email: Optional[str] = None,
        from_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None
    ) -> EmailResult:
        """Send email via AWS SES."""
        if not self.is_configured:
            return EmailResult(
                success=False,
                error_message="AWS SES not configured",
                provider=self.name
            )

        try:
            client = self._get_client()
        except ConfigurationError as e:
            return EmailResult(
                success=False,
                error_message=str(e),
                provider=self.name
            )

        from_email = from_email or self.default_from_email
        from_name = from_name or self.default_from_name
        source = formataddr((from_name, from_email))

        # Build destination
        destination = {'ToAddresses': [to_email]}
        if cc:
            destination['CcAddresses'] = cc
        if bcc:
            destination['BccAddresses'] = bcc

        # Build message
        message = {
            'Subject': {'Data': subject, 'Charset': 'UTF-8'},
            'Body': {
                'Text': {'Data': plain_text, 'Charset': 'UTF-8'},
                'Html': {'Data': html_body, 'Charset': 'UTF-8'}
            }
        }

        try:
            kwargs = {
                'Source': source,
                'Destination': destination,
                'Message': message
            }

            if reply_to:
                kwargs['ReplyToAddresses'] = [reply_to]

            response = client.send_email(**kwargs)

            message_id = response.get('MessageId')
            logger.info(f"Email sent via AWS SES to {to_email}")
            return EmailResult(
                success=True,
                provider=self.name,
                message_id=message_id
            )

        except Exception as e:
            error_name = type(e).__name__

            if 'Throttling' in error_name or 'ThrottlingException' in str(e):
                error_msg = "AWS SES rate limit exceeded"
                logger.warning(error_msg)
                return EmailResult(
                    success=False,
                    error_message=error_msg,
                    error_code=429,
                    retry_after=60.0,
                    provider=self.name
                )

            elif 'InvalidClientToken' in error_name or 'SignatureDoesNotMatch' in str(e):
                error_msg = "AWS SES authentication failed"
                logger.error(error_msg)
                return EmailResult(
                    success=False,
                    error_message=error_msg,
                    error_code=401,
                    provider=self.name
                )

            elif 'ServiceUnavailable' in error_name:
                error_msg = "AWS SES service unavailable"
                logger.error(error_msg)
                return EmailResult(
                    success=False,
                    error_message=error_msg,
                    error_code=503,
                    provider=self.name
                )

            else:
                error_msg = f"AWS SES error: {e}"
                logger.error(error_msg)
                return EmailResult(
                    success=False,
                    error_message=error_msg,
                    provider=self.name
                )


class EmailAlert:
    """
    Email alert handler supporting multiple providers.

    Sends notifications via email using SMTP, SendGrid, or AWS SES.
    Provider is selected via EMAIL_PROVIDER environment variable.

    Priority Levels:
        - 'low': Normal email
        - 'normal': Normal email
        - 'high': Email with [URGENT] prefix
        - 'critical': Email with [CRITICAL] prefix and high importance header
    """

    PROVIDERS = {
        'smtp': SMTPProvider,
        'sendgrid': SendGridProvider,
        'ses': AWSSESProvider,
    }

    def __init__(
        self,
        provider: Optional[str] = None,
        to_email: Optional[str] = None,
        **provider_kwargs
    ):
        """
        Initialize email alert handler.

        Args:
            provider: Provider name ('smtp', 'sendgrid', 'ses')
                     Defaults to EMAIL_PROVIDER env var
            to_email: Default recipient email
                     Defaults to EMAIL_TO env var
            **provider_kwargs: Additional provider-specific configuration
        """
        provider_name = (
            provider or
            os.environ.get('EMAIL_PROVIDER', 'smtp')
        ).lower()

        if provider_name not in self.PROVIDERS:
            raise ConfigurationError(
                f"Unknown email provider: {provider_name}. "
                f"Supported: {', '.join(self.PROVIDERS.keys())}"
            )

        self._provider: EmailProvider = self.PROVIDERS[provider_name](**provider_kwargs)
        self.to_email = to_email or os.environ.get('EMAIL_TO', '')

    @property
    def provider_name(self) -> str:
        """Get current provider name."""
        return self._provider.name

    @property
    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return self._provider.is_configured and bool(self.to_email)

    def _validate_credentials(self) -> bool:
        """Validate configuration."""
        if not self._provider.is_configured:
            logger.warning(
                f"Email provider {self._provider.name} not configured"
            )
            return False
        if not self.to_email:
            logger.warning("Email recipient not configured (EMAIL_TO)")
            return False
        return True

    def send_alert(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        to_email: Optional[str] = None
    ) -> bool:
        """
        Send an alert email.

        Args:
            title: Alert title
            message: Alert message
            priority: Priority level ('low', 'normal', 'high', 'critical')
            to_email: Recipient email (optional, uses default)

        Returns:
            bool: True if sent successfully
        """
        result = self.send_alert_with_result(title, message, priority, to_email)
        return result.success

    def send_alert_with_result(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        to_email: Optional[str] = None
    ) -> EmailResult:
        """
        Send an alert email with detailed result.

        Args:
            title: Alert title
            message: Alert message
            priority: Priority level
            to_email: Recipient email (optional)

        Returns:
            EmailResult: Detailed result
        """
        if not self._validate_credentials():
            return EmailResult(
                success=False,
                error_message="Email not configured",
                provider=self._provider.name
            )

        to_email = to_email or self.to_email
        content = EmailTemplates.generic_alert(title, message, priority)

        return self._provider.send(
            to_email=to_email,
            subject=content.subject,
            html_body=content.html_body,
            plain_text=content.plain_text
        )

    def send_alert_raising(
        self,
        title: str,
        message: str,
        priority: str = 'normal',
        to_email: Optional[str] = None
    ) -> bool:
        """
        Send an alert email, raising exceptions on failure.

        Args:
            title: Alert title
            message: Alert message
            priority: Priority level
            to_email: Recipient email (optional)

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
            raise ConfigurationError("Email not configured")

        result = self.send_alert_with_result(title, message, priority, to_email)

        if result.success:
            return True

        if result.error_code == 401 or result.error_code == 535:
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
            result.error_message or "Failed to send email",
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
        strategy: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        priority: str = 'normal',
        to_email: Optional[str] = None
    ) -> bool:
        """
        Send a trade alert email.

        Args:
            action: Trade action (BUY, SELL, etc.)
            symbol: Stock symbol
            price: Trade price
            quantity: Number of shares
            reason: Reason for trade
            strategy: Strategy name (optional)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            priority: Alert priority
            to_email: Recipient email (optional)

        Returns:
            bool: True if sent successfully
        """
        if not self._validate_credentials():
            return False

        to_email = to_email or self.to_email
        content = EmailTemplates.trade_alert(
            action=action,
            symbol=symbol,
            price=price,
            quantity=quantity,
            reason=reason,
            strategy=strategy,
            stop_loss=stop_loss,
            take_profit=take_profit,
            priority=priority
        )

        result = self._provider.send(
            to_email=to_email,
            subject=content.subject,
            html_body=content.html_body,
            plain_text=content.plain_text
        )
        return result.success

    def send_signal_alert(
        self,
        signal_type: str,
        symbol: str,
        price: float,
        indicator: str,
        indicator_value: float,
        recommendation: str,
        confidence: Optional[float] = None,
        priority: str = 'normal',
        additional_data: Optional[Dict[str, Any]] = None,
        to_email: Optional[str] = None
    ) -> bool:
        """
        Send a signal alert email.

        Args:
            signal_type: Type of signal (BULLISH, BEARISH, etc.)
            symbol: Stock symbol
            price: Current price
            indicator: Indicator name
            indicator_value: Indicator value
            recommendation: Action recommendation
            confidence: Confidence level (optional)
            priority: Alert priority
            additional_data: Additional indicator data (optional)
            to_email: Recipient email (optional)

        Returns:
            bool: True if sent successfully
        """
        if not self._validate_credentials():
            return False

        to_email = to_email or self.to_email
        content = EmailTemplates.signal_alert(
            signal_type=signal_type,
            symbol=symbol,
            price=price,
            indicator=indicator,
            indicator_value=indicator_value,
            recommendation=recommendation,
            confidence=confidence,
            priority=priority,
            additional_data=additional_data
        )

        result = self._provider.send(
            to_email=to_email,
            subject=content.subject,
            html_body=content.html_body,
            plain_text=content.plain_text
        )
        return result.success

    def send_daily_summary(
        self,
        date: str,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        total_pnl_percent: float,
        portfolio_value: float,
        starting_value: float,
        trades: Optional[List[Dict[str, Any]]] = None,
        top_winners: Optional[List[Dict[str, Any]]] = None,
        top_losers: Optional[List[Dict[str, Any]]] = None,
        signals_generated: int = 0,
        priority: str = 'normal',
        to_email: Optional[str] = None
    ) -> bool:
        """
        Send daily performance summary email.

        Args:
            date: Summary date
            total_trades: Total number of trades
            winning_trades: Number of winning trades
            losing_trades: Number of losing trades
            total_pnl: Total profit/loss
            total_pnl_percent: Total P&L percentage
            portfolio_value: Current portfolio value
            starting_value: Starting portfolio value
            trades: List of trade details (optional)
            top_winners: Top winning trades (optional)
            top_losers: Top losing trades (optional)
            signals_generated: Number of signals
            priority: Alert priority
            to_email: Recipient email (optional)

        Returns:
            bool: True if sent successfully
        """
        if not self._validate_credentials():
            return False

        to_email = to_email or self.to_email
        content = EmailTemplates.daily_summary(
            date=date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            portfolio_value=portfolio_value,
            starting_value=starting_value,
            trades=trades,
            top_winners=top_winners,
            top_losers=top_losers,
            signals_generated=signals_generated,
            priority=priority
        )

        result = self._provider.send(
            to_email=to_email,
            subject=content.subject,
            html_body=content.html_body,
            plain_text=content.plain_text
        )
        return result.success

    def send_risk_alert(
        self,
        alert_type: str,
        message: str,
        current_value: float,
        threshold_value: float,
        symbol: Optional[str] = None,
        recommendation: Optional[str] = None,
        priority: str = 'high',
        to_email: Optional[str] = None
    ) -> bool:
        """
        Send a risk alert email.

        Args:
            alert_type: Type of risk alert
            message: Alert message
            current_value: Current metric value
            threshold_value: Threshold breached
            symbol: Related symbol (optional)
            recommendation: Recommended action (optional)
            priority: Alert priority
            to_email: Recipient email (optional)

        Returns:
            bool: True if sent successfully
        """
        if not self._validate_credentials():
            return False

        to_email = to_email or self.to_email
        content = EmailTemplates.risk_alert(
            alert_type=alert_type,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            symbol=symbol,
            recommendation=recommendation,
            priority=priority
        )

        result = self._provider.send(
            to_email=to_email,
            subject=content.subject,
            html_body=content.html_body,
            plain_text=content.plain_text
        )
        return result.success

    def send_system_alert(
        self,
        title: str,
        message: str,
        alert_type: str = 'INFO',
        details: Optional[Dict[str, Any]] = None,
        priority: str = 'normal',
        to_email: Optional[str] = None
    ) -> bool:
        """
        Send a system alert email.

        Args:
            title: Alert title
            message: Alert message
            alert_type: Type (INFO, WARNING, ERROR, etc.)
            details: Additional details (optional)
            priority: Alert priority
            to_email: Recipient email (optional)

        Returns:
            bool: True if sent successfully
        """
        if not self._validate_credentials():
            return False

        to_email = to_email or self.to_email
        content = EmailTemplates.system_alert(
            title=title,
            message=message,
            alert_type=alert_type,
            details=details,
            priority=priority
        )

        result = self._provider.send(
            to_email=to_email,
            subject=content.subject,
            html_body=content.html_body,
            plain_text=content.plain_text
        )
        return result.success

    def send_custom_email(
        self,
        subject: str,
        html_body: str,
        plain_text: str,
        to_email: Optional[str] = None,
        from_email: Optional[str] = None,
        from_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None
    ) -> EmailResult:
        """
        Send a custom email with full control over content.

        Args:
            subject: Email subject
            html_body: HTML email body
            plain_text: Plain text email body
            to_email: Recipient email (optional)
            from_email: Sender email (optional)
            from_name: Sender name (optional)
            reply_to: Reply-to email (optional)
            cc: CC recipients (optional)
            bcc: BCC recipients (optional)

        Returns:
            EmailResult: Send result
        """
        if not self._validate_credentials():
            return EmailResult(
                success=False,
                error_message="Email not configured",
                provider=self._provider.name
            )

        to_email = to_email or self.to_email

        return self._provider.send(
            to_email=to_email,
            subject=subject,
            html_body=html_body,
            plain_text=plain_text,
            from_email=from_email,
            from_name=from_name,
            reply_to=reply_to,
            cc=cc,
            bcc=bcc
        )
