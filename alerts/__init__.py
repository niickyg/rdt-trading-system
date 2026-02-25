"""
RDT Trading System - Alert Notification Module

Multi-channel alert system supporting:
- Desktop notifications
- SMS via Twilio
- Email (SMTP, SendGrid, AWS SES)
- Pushover push notifications
- Discord webhooks
- Telegram bot messages

Environment Variables:
    ALERT_CHANNELS: Comma-separated list of enabled channels
                   (e.g., "pushover,discord,telegram,email,sms")

    Email:
        EMAIL_PROVIDER: Provider to use (smtp, sendgrid, ses)
        EMAIL_TO: Recipient email address

        SMTP:
            SMTP_HOST: SMTP server hostname
            SMTP_PORT: SMTP server port (587 for TLS, 465 for SSL)
            SMTP_USER: SMTP username
            SMTP_PASSWORD: SMTP password
            EMAIL_FROM: Sender email address

        SendGrid:
            SENDGRID_API_KEY: SendGrid API key
            SENDGRID_FROM_EMAIL: Verified sender email

        AWS SES:
            AWS_SES_REGION: AWS region
            AWS_ACCESS_KEY: AWS access key
            AWS_SECRET_KEY: AWS secret key
            AWS_SES_FROM_EMAIL: Verified sender email

    SMS (Twilio):
        TWILIO_ACCOUNT_SID: Twilio account SID
        TWILIO_AUTH_TOKEN: Twilio auth token
        TWILIO_FROM_NUMBER: Twilio phone number
        TWILIO_TO_NUMBER: Destination phone number

    Pushover:
        PUSHOVER_USER_KEY: Your Pushover user key
        PUSHOVER_API_TOKEN: Your Pushover application API token

    Discord:
        DISCORD_WEBHOOK_URL: Discord webhook URL for your channel

    Telegram:
        TELEGRAM_BOT_TOKEN: Your Telegram bot token
        TELEGRAM_CHAT_ID: Chat ID to send messages to

    Daily Summary:
        DAILY_SUMMARY_TIME: Time to send daily summary (HH:MM format)

Usage:
    from alerts import AlertManager, get_alert_manager

    # Using the factory function
    manager = get_alert_manager()
    manager.send_alert("Trade Alert", "AAPL breakout detected", priority="high")

    # Or instantiate directly with specific channels
    manager = AlertManager(enabled_channels=['discord', 'telegram', 'email', 'sms'])
    manager.send_trade_alert(
        action='BUY',
        symbol='AAPL',
        price=150.00,
        quantity=10,
        reason='RRS breakout signal'
    )

    # Set alert preferences per alert type
    manager.set_alert_preference('trade', ['email', 'sms', 'telegram'])
    manager.set_alert_preference('signal', ['discord', 'telegram'])
    manager.set_alert_preference('risk', ['email', 'sms'])

    # Individual alert classes
    from alerts import PushoverAlert, DiscordAlert, TelegramAlert, EmailAlert, SMSAlert

    email = EmailAlert()
    email.send_trade_alert(
        action='BUY',
        symbol='AAPL',
        price=150.00,
        quantity=10,
        reason='RRS breakout signal'
    )

    sms = SMSAlert()
    sms.send_trade_alert('BUY', 'AAPL', 150.00, 10, 'RRS breakout')

    # Daily summary emails
    from alerts import setup_daily_summary, record_trade, record_signal

    # Start the daily summary scheduler
    task = setup_daily_summary(schedule_time='16:30')

    # Record trades and signals throughout the day
    record_trade('AAPL', 'BUY', 100, 150.00, pnl=250.00)
    record_signal('AAPL', 'BULLISH', 'RRS', 2.5)

    # Or send summary manually
    task.run_now()
"""

from .notifier import (
    send_desktop_alert,
    send_sms_alert,
    send_email_alert,
    send_alert,
)

from .pushover_alert import PushoverAlert
from .discord_alert import DiscordAlert
from .telegram_alert import TelegramAlert
from .email_alert import EmailAlert, EmailResult
from .sms_alert import SMSAlert, SMSResult
from .email_templates import EmailTemplates, EmailContent
from .alert_manager import (
    AlertManager,
    AlertChannel,
    AlertPriority,
    AlertResult,
    MultiAlertResult,
    get_alert_manager,
)
from .daily_summary_task import (
    DailySummaryTask,
    DailySummaryCollector,
    DailySummaryData,
    setup_daily_summary,
    get_daily_summary_task,
    record_trade,
    record_signal,
)
from .scheduler import (
    AlertScheduler,
    QuietHoursRule,
    QueuedAlert,
    DayOfWeek,
    get_alert_scheduler,
    set_alert_scheduler,
)
from .schedule_config import (
    AlertScheduleConfig,
    AlertScheduleConfigManager,
    ChannelSchedule,
    AlertTypePreference,
    get_config_manager,
    get_user_schedule_config,
)

__all__ = [
    # Legacy notifier functions
    'send_desktop_alert',
    'send_sms_alert',
    'send_email_alert',
    'send_alert',
    # New alert classes
    'PushoverAlert',
    'DiscordAlert',
    'TelegramAlert',
    'EmailAlert',
    'EmailResult',
    'SMSAlert',
    'SMSResult',
    # Email templates
    'EmailTemplates',
    'EmailContent',
    # Alert manager
    'AlertManager',
    'AlertChannel',
    'AlertPriority',
    'AlertResult',
    'MultiAlertResult',
    'get_alert_manager',
    # Daily summary
    'DailySummaryTask',
    'DailySummaryCollector',
    'DailySummaryData',
    'setup_daily_summary',
    'get_daily_summary_task',
    'record_trade',
    'record_signal',
    # Alert scheduling
    'AlertScheduler',
    'QuietHoursRule',
    'QueuedAlert',
    'DayOfWeek',
    'get_alert_scheduler',
    'set_alert_scheduler',
    'AlertScheduleConfig',
    'AlertScheduleConfigManager',
    'ChannelSchedule',
    'AlertTypePreference',
    'get_config_manager',
    'get_user_schedule_config',
]
