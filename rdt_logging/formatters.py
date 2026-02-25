"""
RDT Trading System - Log Formatters

Provides structured JSON formatting with:
- Timestamp, level, message, service, correlation_id
- Extra fields support
- Sensitive data masking (passwords, API keys, tokens)
"""

import json
import re
from datetime import datetime, timezone
from typing import Any


# Patterns for sensitive data that should be masked
SENSITIVE_PATTERNS = [
    # API keys and secrets
    (re.compile(r'(api[_-]?key|apikey)["\s:=]+["\']?([a-zA-Z0-9_\-]{16,})["\']?', re.IGNORECASE), r'\1=***MASKED***'),
    (re.compile(r'(secret[_-]?key|secretkey)["\s:=]+["\']?([a-zA-Z0-9_\-]{16,})["\']?', re.IGNORECASE), r'\1=***MASKED***'),
    (re.compile(r'(app[_-]?secret|appsecret)["\s:=]+["\']?([a-zA-Z0-9_\-]{16,})["\']?', re.IGNORECASE), r'\1=***MASKED***'),

    # Passwords
    (re.compile(r'(password|passwd|pwd)["\s:=]+["\']?([^\s"\',}{]+)["\']?', re.IGNORECASE), r'\1=***MASKED***'),

    # Tokens
    (re.compile(r'(token|bearer|jwt|access_token|refresh_token)["\s:=]+["\']?([a-zA-Z0-9_\-\.]{20,})["\']?', re.IGNORECASE), r'\1=***MASKED***'),
    (re.compile(r'Bearer\s+([a-zA-Z0-9_\-\.]{20,})', re.IGNORECASE), r'Bearer ***MASKED***'),

    # Credit card numbers (basic pattern)
    (re.compile(r'\b(\d{4})[- ]?(\d{4})[- ]?(\d{4})[- ]?(\d{4})\b'), r'****-****-****-\4'),

    # AWS keys
    (re.compile(r'(AKIA[0-9A-Z]{16})'), r'***AWS_KEY_MASKED***'),
    (re.compile(r'(aws[_-]?secret[_-]?access[_-]?key)["\s:=]+["\']?([a-zA-Z0-9/+=]{40})["\']?', re.IGNORECASE), r'\1=***MASKED***'),

    # Stripe keys
    (re.compile(r'(sk_live_[a-zA-Z0-9]{24,})'), r'***STRIPE_KEY_MASKED***'),
    (re.compile(r'(sk_test_[a-zA-Z0-9]{24,})'), r'***STRIPE_TEST_KEY_MASKED***'),

    # Database connection strings
    (re.compile(r'(postgresql://[^:]+:)([^@]+)(@.+)'), r'\1***MASKED***\3'),
    (re.compile(r'(mysql://[^:]+:)([^@]+)(@.+)'), r'\1***MASKED***\3'),
    (re.compile(r'(mongodb://[^:]+:)([^@]+)(@.+)'), r'\1***MASKED***\3'),
    (re.compile(r'(redis://:[^@]+)(@.+)'), r'redis://***MASKED***\2'),

    # Generic authorization headers
    (re.compile(r'(authorization)["\s:=]+["\']?([a-zA-Z0-9_\-\.\s]{20,})["\']?', re.IGNORECASE), r'\1=***MASKED***'),

    # Email addresses (partial masking)
    (re.compile(r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'), lambda m: f"{m.group(1)[:2]}***@{m.group(2)}"),

    # Phone numbers
    (re.compile(r'\+?1?[-.\s]?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})'), r'(***) ***-\3'),
]

# Keys that should always be masked in dictionaries
SENSITIVE_KEYS = {
    'password', 'passwd', 'pwd', 'secret', 'token', 'api_key', 'apikey',
    'api_secret', 'apisecret', 'secret_key', 'secretkey', 'access_token',
    'refresh_token', 'bearer', 'authorization', 'auth_token', 'auth',
    'credential', 'credentials', 'private_key', 'privatekey', 'ssn',
    'social_security', 'credit_card', 'card_number', 'cvv', 'cvc',
    'pin', 'account_number', 'routing_number', 'schwab_app_secret',
    'stripe_secret_key', 'stripe_webhook_secret', 'jwt_secret_key',
    'twilio_auth_token', 'email_password', 'telegram_bot_token',
    'discord_webhook_url', 'pushover_api_token', 'ibkr_client_secret',
}


def mask_sensitive_data(data: Any, depth: int = 0, max_depth: int = 10) -> Any:
    """
    Recursively mask sensitive data in strings, dicts, and lists.

    Args:
        data: The data to mask
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        Data with sensitive information masked
    """
    if depth > max_depth:
        return data

    if isinstance(data, str):
        return mask_string(data)
    elif isinstance(data, dict):
        return mask_dict(data, depth, max_depth)
    elif isinstance(data, (list, tuple)):
        return type(data)(mask_sensitive_data(item, depth + 1, max_depth) for item in data)
    else:
        return data


def mask_string(text: str) -> str:
    """Apply all sensitive patterns to mask data in a string."""
    if not text or not isinstance(text, str):
        return text

    masked = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        if callable(replacement):
            masked = pattern.sub(replacement, masked)
        else:
            masked = pattern.sub(replacement, masked)

    return masked


def mask_dict(data: dict, depth: int = 0, max_depth: int = 10) -> dict:
    """Mask sensitive keys and values in a dictionary."""
    if depth > max_depth:
        return data

    result = {}
    for key, value in data.items():
        key_lower = key.lower().replace('-', '_').replace(' ', '_')

        # Check if key is sensitive
        if key_lower in SENSITIVE_KEYS:
            if isinstance(value, str) and len(value) > 0:
                result[key] = '***MASKED***'
            elif value is not None:
                result[key] = '***MASKED***'
            else:
                result[key] = value
        else:
            result[key] = mask_sensitive_data(value, depth + 1, max_depth)

    return result


class JSONFormatter:
    """
    JSON log formatter for structured logging.

    Produces JSON log records with:
    - timestamp: ISO 8601 format with timezone
    - level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - message: The log message
    - service: Service name
    - correlation_id: Request/trace correlation ID
    - logger: Logger name
    - extra: Additional context fields
    """

    def __init__(
        self,
        service_name: str = 'rdt-trading',
        include_timestamp: bool = True,
        include_hostname: bool = True,
        include_process: bool = True,
        mask_sensitive: bool = True,
        extra_fields: dict = None,
    ):
        """
        Initialize the JSON formatter.

        Args:
            service_name: Name of the service for log identification
            include_timestamp: Include ISO timestamp in logs
            include_hostname: Include hostname in logs
            include_process: Include process ID in logs
            mask_sensitive: Apply sensitive data masking
            extra_fields: Additional static fields to include in every log
        """
        self.service_name = service_name
        self.include_timestamp = include_timestamp
        self.include_hostname = include_hostname
        self.include_process = include_process
        self.mask_sensitive = mask_sensitive
        self.extra_fields = extra_fields or {}

        # Cache hostname
        if include_hostname:
            import socket
            self._hostname = socket.gethostname()
        else:
            self._hostname = None

    def format(self, record: dict) -> str:
        """
        Format a log record as JSON.

        This method is designed to work with Loguru's custom format.

        Args:
            record: Loguru record dictionary

        Returns:
            JSON-formatted log string
        """
        # Build the base log structure
        log_entry = {}

        # Timestamp
        if self.include_timestamp:
            # Use record time if available, otherwise current time
            if 'time' in record:
                timestamp = record['time']
                if hasattr(timestamp, 'isoformat'):
                    log_entry['timestamp'] = timestamp.isoformat()
                else:
                    log_entry['timestamp'] = str(timestamp)
            else:
                log_entry['timestamp'] = datetime.now(timezone.utc).isoformat()

        # Core fields — loguru records use named-tuple-like objects, not dicts
        level = record.get('level', 'INFO')
        if hasattr(level, 'name'):
            log_entry['level'] = level.name
        elif isinstance(level, dict):
            log_entry['level'] = level.get('name', 'INFO')
        else:
            log_entry['level'] = str(level)

        log_entry['message'] = str(record.get('message', ''))
        log_entry['service'] = self.service_name
        log_entry['logger'] = record.get('name', 'root')

        # Optional fields
        if self._hostname:
            log_entry['hostname'] = self._hostname

        if self.include_process:
            process = record.get('process')
            thread = record.get('thread')
            log_entry['process_id'] = getattr(process, 'id', None) if process and not isinstance(process, dict) else (process.get('id') if isinstance(process, dict) else None)
            log_entry['thread_id'] = getattr(thread, 'id', None) if thread and not isinstance(thread, dict) else (thread.get('id') if isinstance(thread, dict) else None)

        # Location info
        if 'file' in record or 'function' in record or 'line' in record:
            file_val = record.get('file')
            if hasattr(file_val, 'name'):
                file_name = file_val.name
            elif isinstance(file_val, dict):
                file_name = file_val.get('name')
            else:
                file_name = str(file_val) if file_val else None

            log_entry['location'] = {
                'file': file_name,
                'function': record.get('function'),
                'line': record.get('line'),
            }

        # Exception info
        if record.get('exception'):
            exc = record['exception']
            if hasattr(exc, 'type') and exc.type:
                log_entry['exception'] = {
                    'type': exc.type.__name__ if hasattr(exc.type, '__name__') else str(exc.type),
                    'value': str(exc.value) if exc.value else None,
                    'traceback': exc.traceback if hasattr(exc, 'traceback') else None,
                }

        # Extra context from record
        extra = record.get('extra', {})
        if extra:
            # Extract known context fields
            if 'correlation_id' in extra:
                log_entry['correlation_id'] = extra.pop('correlation_id', None)
            if 'request_id' in extra:
                log_entry['request_id'] = extra.pop('request_id', None)
            if 'user_id' in extra:
                log_entry['user_id'] = extra.pop('user_id', None)
            if 'trace_id' in extra:
                log_entry['trace_id'] = extra.pop('trace_id', None)
            if 'span_id' in extra:
                log_entry['span_id'] = extra.pop('span_id', None)

            # Remaining extra fields
            if extra:
                log_entry['extra'] = extra

        # Add static extra fields
        if self.extra_fields:
            for key, value in self.extra_fields.items():
                if key not in log_entry:
                    log_entry[key] = value

        # Apply sensitive data masking
        if self.mask_sensitive:
            log_entry = mask_sensitive_data(log_entry)

        return json.dumps(log_entry, default=str, ensure_ascii=False)

    def __call__(self, record: dict) -> str:
        """Make the formatter callable for Loguru compatibility."""
        return self.format(record)


class LoguruJSONSink:
    """
    A sink for Loguru that outputs JSON-formatted logs.

    Usage:
        from loguru import logger
        from logging.formatters import LoguruJSONSink

        logger.add(LoguruJSONSink(), format="{message}")
    """

    def __init__(
        self,
        service_name: str = 'rdt-trading',
        mask_sensitive: bool = True,
        output_stream=None,
    ):
        """
        Initialize the JSON sink.

        Args:
            service_name: Name of the service
            mask_sensitive: Apply sensitive data masking
            output_stream: Output stream (defaults to sys.stdout)
        """
        self.formatter = JSONFormatter(
            service_name=service_name,
            mask_sensitive=mask_sensitive,
        )
        self._stream = output_stream

    def write(self, message):
        """Write a log message."""
        import sys
        stream = self._stream or sys.stdout

        # Loguru passes a Message object, extract the record
        record = message.record if hasattr(message, 'record') else {'message': str(message)}

        try:
            json_log = self.formatter.format(record)
            stream.write(json_log + '\n')
            stream.flush()
        except Exception:
            # Fallback to simple output on formatting error
            stream.write(str(message))
            stream.flush()


def create_json_format_string(service_name: str = 'rdt-trading', mask_sensitive: bool = True) -> str:
    """
    Create a Loguru format string that produces JSON output.

    This is a helper for simple JSON logging setup with Loguru.

    Args:
        service_name: Name of the service
        mask_sensitive: Whether to mask sensitive data

    Returns:
        A format function for Loguru
    """
    formatter = JSONFormatter(service_name=service_name, mask_sensitive=mask_sensitive)

    def format_func(record):
        # Escape curly braces so Loguru doesn't interpret them as format placeholders
        json_output = formatter.format(record)
        return json_output.replace('{', '{{').replace('}', '}}') + '\n'

    return format_func
