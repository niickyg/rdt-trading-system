"""
Alert Notification System
Supports: Desktop notifications, SMS (Twilio), Email
"""

import os
from typing import Dict
from loguru import logger


def send_desktop_alert(message: str):
    """Send desktop notification"""
    try:
        from plyer import notification
        notification.notify(
            title="🔔 RDT Trading Alert",
            message=message,
            app_name="RDT Scanner",
            timeout=10
        )
        logger.info("Desktop alert sent")
    except Exception as e:
        logger.error(f"Failed to send desktop alert: {e}")
        # Fallback: print to console
        print(f"\n{'='*60}\nALERT:\n{message}\n{'='*60}\n")


def send_sms_alert(message: str, config: Dict):
    """Send SMS via Twilio"""
    try:
        from twilio.rest import Client

        account_sid = config.get('twilio_account_sid')
        auth_token = config.get('twilio_auth_token')
        from_number = config.get('twilio_from_number')
        to_number = config.get('twilio_to_number')

        if not all([account_sid, auth_token, from_number, to_number]):
            logger.warning("Twilio credentials missing, skipping SMS")
            return

        client = Client(account_sid, auth_token)
        client.messages.create(
            body=message,
            from_=from_number,
            to=to_number
        )
        logger.info(f"SMS sent to {to_number}")

    except Exception as e:
        logger.error(f"Failed to send SMS: {e}")


def send_email_alert(message: str, config: Dict):
    """Send email alert"""
    try:
        import smtplib
        from email.mime.text import MIMEText

        email_from = config.get('email_from')
        email_to = config.get('email_to')
        email_password = config.get('email_password')

        if not all([email_from, email_to, email_password]):
            logger.warning("Email credentials missing, skipping email")
            return

        msg = MIMEText(message)
        msg['Subject'] = 'RDT Trading Alert'
        msg['From'] = email_from
        msg['To'] = email_to

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(email_from, email_password)
            smtp.send_message(msg)

        logger.info(f"Email sent to {email_to}")

    except Exception as e:
        logger.error(f"Failed to send email: {e}")


def send_alert(message: str, config: Dict):
    """
    Send alert via configured method

    Args:
        message: Alert message
        config: Configuration dict with alert settings
    """
    alert_method = config.get('alert_method', 'desktop').lower()

    if alert_method == 'desktop':
        send_desktop_alert(message)
    elif alert_method == 'twilio' or alert_method == 'sms':
        send_sms_alert(message, config)
    elif alert_method == 'email':
        send_email_alert(message, config)
    elif alert_method == 'all':
        send_desktop_alert(message)
        send_sms_alert(message, config)
        send_email_alert(message, config)
    else:
        logger.info(f"Alert: {message}")
