"""
RDT Trading System - Centralized Logging Configuration

Provides unified logging setup with:
- Multiple outputs: console, file, Loki, Elasticsearch
- Structured JSON logging format
- Log levels configurable via environment
- Context injection (request_id, user_id, service_name)
"""

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from rdt_logging.context import get_full_context, set_service_name
from rdt_logging.formatters import JSONFormatter, LoguruJSONSink, create_json_format_string
from rdt_logging.handlers import (
    BufferedFileHandler,
    LoguruElasticsearchSink,
    LoguruLokiSink,
)


class LogLevel(str, Enum):
    """Log level enumeration."""
    TRACE = 'TRACE'
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    SUCCESS = 'SUCCESS'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'


class LogFormat(str, Enum):
    """Log format enumeration."""
    TEXT = 'text'
    JSON = 'json'
    COLORED = 'colored'


@dataclass
class ConsoleConfig:
    """Console logging configuration."""
    enabled: bool = True
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.COLORED
    colorize: bool = True


@dataclass
class FileConfig:
    """File logging configuration."""
    enabled: bool = True
    level: LogLevel = LogLevel.DEBUG
    path: str = 'logs/rdt-trading.log'
    format: LogFormat = LogFormat.JSON
    rotation: str = '10 MB'
    retention: str = '30 days'
    compression: str = 'gz'
    buffered: bool = False
    buffer_size: int = 10000


@dataclass
class LokiConfig:
    """Loki logging configuration."""
    enabled: bool = False
    url: str = 'http://localhost:3100'
    level: LogLevel = LogLevel.INFO
    labels: Dict[str, str] = field(default_factory=lambda: {'app': 'rdt-trading'})
    tenant_id: Optional[str] = None
    auth_username: Optional[str] = None
    auth_password: Optional[str] = None
    batch_size: int = 100
    flush_interval: float = 5.0


@dataclass
class ElasticsearchConfig:
    """Elasticsearch logging configuration."""
    enabled: bool = False
    url: str = 'http://localhost:9200'
    level: LogLevel = LogLevel.INFO
    index_prefix: str = 'rdt-logs'
    auth_username: Optional[str] = None
    auth_password: Optional[str] = None
    api_key: Optional[str] = None
    use_data_stream: bool = False
    batch_size: int = 100
    flush_interval: float = 5.0
    verify_ssl: bool = True


@dataclass
class LoggingConfig:
    """
    Complete logging configuration.

    Can be loaded from environment variables or passed directly.
    """
    service_name: str = 'rdt-trading'
    default_level: LogLevel = LogLevel.INFO
    console: ConsoleConfig = field(default_factory=ConsoleConfig)
    file: FileConfig = field(default_factory=FileConfig)
    loki: LokiConfig = field(default_factory=LokiConfig)
    elasticsearch: ElasticsearchConfig = field(default_factory=ElasticsearchConfig)
    mask_sensitive: bool = True
    include_context: bool = True
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """
        Load configuration from environment variables.

        Environment variables:
            LOG_LEVEL: Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            LOG_FORMAT: Log format (text, json, colored)
            LOG_TO_FILE: Enable file logging (true/false)
            LOG_FILE_PATH: Path to log file
            LOKI_URL: Loki push URL (enables Loki if set)
            LOKI_TENANT_ID: Loki tenant ID
            LOKI_USERNAME: Loki basic auth username
            LOKI_PASSWORD: Loki basic auth password
            ELASTICSEARCH_URL: Elasticsearch URL (enables ES if set)
            ELASTICSEARCH_INDEX: Elasticsearch index prefix
            ELASTICSEARCH_USERNAME: ES basic auth username
            ELASTICSEARCH_PASSWORD: ES basic auth password
            ELASTICSEARCH_API_KEY: ES API key
            SERVICE_NAME: Service name for logs
        """
        def get_bool(key: str, default: bool = False) -> bool:
            value = os.environ.get(key, str(default)).lower()
            return value in ('true', '1', 'yes', 'on')

        def get_level(key: str, default: LogLevel = LogLevel.INFO) -> LogLevel:
            value = os.environ.get(key, default.value).upper()
            try:
                return LogLevel(value)
            except ValueError:
                return default

        def get_format(key: str, default: LogFormat = LogFormat.COLORED) -> LogFormat:
            value = os.environ.get(key, default.value).lower()
            try:
                return LogFormat(value)
            except ValueError:
                return default

        # Service name
        service_name = os.environ.get('SERVICE_NAME', 'rdt-trading')

        # Default level
        default_level = get_level('LOG_LEVEL', LogLevel.INFO)

        # Console config
        console = ConsoleConfig(
            enabled=get_bool('LOG_TO_CONSOLE', True),
            level=get_level('LOG_CONSOLE_LEVEL', default_level),
            format=get_format('LOG_FORMAT', LogFormat.COLORED),
            colorize=get_bool('LOG_COLORIZE', True),
        )

        # File config
        file = FileConfig(
            enabled=get_bool('LOG_TO_FILE', True),
            level=get_level('LOG_FILE_LEVEL', LogLevel.DEBUG),
            path=os.environ.get('LOG_FILE_PATH', 'logs/rdt-trading.log'),
            format=get_format('LOG_FILE_FORMAT', LogFormat.JSON),
            rotation=os.environ.get('LOG_ROTATION', '10 MB'),
            retention=os.environ.get('LOG_RETENTION', '30 days'),
            compression=os.environ.get('LOG_COMPRESSION', 'gz'),
            buffered=get_bool('LOG_BUFFERED', False),
        )

        # Loki config
        loki_url = os.environ.get('LOKI_URL', '')
        loki_enabled = os.environ.get('LOKI_ENABLED', 'false').lower() == 'true'
        loki = LokiConfig(
            enabled=bool(loki_url) and loki_enabled,
            url=loki_url or 'http://localhost:3100',
            level=get_level('LOKI_LEVEL', default_level),
            labels={'app': service_name, 'env': os.environ.get('RDT_ENV', 'development')},
            tenant_id=os.environ.get('LOKI_TENANT_ID'),
            auth_username=os.environ.get('LOKI_USERNAME'),
            auth_password=os.environ.get('LOKI_PASSWORD'),
            batch_size=int(os.environ.get('LOKI_BATCH_SIZE', '100')),
            flush_interval=float(os.environ.get('LOKI_FLUSH_INTERVAL', '5.0')),
        )

        # Elasticsearch config
        es_url = os.environ.get('ELASTICSEARCH_URL', '')
        elasticsearch = ElasticsearchConfig(
            enabled=bool(es_url),
            url=es_url or 'http://localhost:9200',
            level=get_level('ELASTICSEARCH_LEVEL', default_level),
            index_prefix=os.environ.get('ELASTICSEARCH_INDEX', 'rdt-logs'),
            auth_username=os.environ.get('ELASTICSEARCH_USERNAME'),
            auth_password=os.environ.get('ELASTICSEARCH_PASSWORD'),
            api_key=os.environ.get('ELASTICSEARCH_API_KEY'),
            use_data_stream=get_bool('ELASTICSEARCH_DATA_STREAM', False),
            batch_size=int(os.environ.get('ELASTICSEARCH_BATCH_SIZE', '100')),
            flush_interval=float(os.environ.get('ELASTICSEARCH_FLUSH_INTERVAL', '5.0')),
            verify_ssl=get_bool('ELASTICSEARCH_VERIFY_SSL', True),
        )

        return cls(
            service_name=service_name,
            default_level=default_level,
            console=console,
            file=file,
            loki=loki,
            elasticsearch=elasticsearch,
            mask_sensitive=get_bool('LOG_MASK_SENSITIVE', True),
            include_context=get_bool('LOG_INCLUDE_CONTEXT', True),
            extra_fields={'env': os.environ.get('RDT_ENV', 'development')},
        )


# Global handlers registry for cleanup
_handlers: List[Any] = []


def _create_context_format(config: LoggingConfig, format_type: LogFormat) -> Callable:
    """Create a format function that includes context."""

    if format_type == LogFormat.JSON:
        formatter = JSONFormatter(
            service_name=config.service_name,
            mask_sensitive=config.mask_sensitive,
            extra_fields=config.extra_fields,
        )

        def json_format(record):
            # Inject context into record
            if config.include_context:
                context = get_full_context()
                record['extra'].update(context)
            # Escape curly braces so Loguru doesn't interpret them as format placeholders
            # Also escape angle brackets to prevent loguru color tag parsing
            # (Python 3.14: <module> in function names triggers ValueError)
            json_output = formatter.format(record)
            json_output = json_output.replace('{', '{{').replace('}', '}}')
            json_output = json_output.replace('<', r'\<')
            return json_output + '\n'

        return json_format

    elif format_type == LogFormat.TEXT:
        def text_format(record):
            # Build context string
            context_parts = []
            if config.include_context:
                context = get_full_context()
                if context.get('correlation_id'):
                    context_parts.append(f"[{context['correlation_id'][:8]}]")
                if context.get('user_id'):
                    context_parts.append(f"[user:{context['user_id']}]")

            context_str = ' '.join(context_parts)
            if context_str:
                context_str = f" {context_str}"

            return (
                f"{{time:YYYY-MM-DD HH:mm:ss.SSS}} | {{level: <8}} | "
                f"{{name}}:{{function}}:{{line}}{context_str} - {{message}}\n"
            )

        return text_format

    else:  # COLORED
        def colored_format(record):
            # Build context string
            context_parts = []
            if config.include_context:
                context = get_full_context()
                if context.get('correlation_id'):
                    context_parts.append(f"<cyan>[{context['correlation_id'][:8]}]</cyan>")
                if context.get('user_id'):
                    context_parts.append(f"<yellow>[user:{context['user_id']}]</yellow>")

            context_str = ' '.join(context_parts)
            if context_str:
                context_str = f" {context_str}"

            return (
                f"<green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> | "
                f"<level>{{level: <8}}</level> | "
                f"<cyan>{{name}}:{{function}}:{{line}}</cyan>"
                f"{context_str} - <level>{{message}}</level>\n"
            )

        return colored_format


def configure_logging(config: LoggingConfig = None) -> None:
    """
    Configure centralized logging based on configuration.

    Args:
        config: LoggingConfig instance. If None, loads from environment.
    """
    global _handlers

    if config is None:
        config = LoggingConfig.from_env()

    # Set service name in context
    set_service_name(config.service_name)

    # Remove default logger
    logger.remove()

    # Console handler
    if config.console.enabled:
        format_func = _create_context_format(config, config.console.format)
        logger.add(
            sys.stderr,
            level=config.console.level.value,
            format=format_func,
            colorize=config.console.colorize and config.console.format == LogFormat.COLORED,
            backtrace=True,
            diagnose=False,  # SECURITY: Never enable - exposes local variables including secrets
        )

    # File handler
    if config.file.enabled:
        # Ensure log directory exists
        log_path = Path(config.file.path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if config.file.buffered:
            # Use buffered file handler
            file_handler = BufferedFileHandler(
                filepath=str(log_path),
                max_size=_parse_size(config.file.rotation),
            )
            file_handler.start()
            _handlers.append(file_handler)

            # Force TEXT or JSON format for file handlers — COLORED format contains
            # color tags like <cyan> which loguru parses even with colorize=False,
            # causing ValueError on Python 3.14 when {function} resolves to <module>
            file_format = config.file.format
            if file_format == LogFormat.COLORED:
                file_format = LogFormat.TEXT
            format_func = _create_context_format(config, file_format)
            logger.add(
                file_handler,
                level=config.file.level.value,
                format=format_func,
                colorize=False,
            )
        else:
            # Use Loguru's built-in file rotation
            # Force TEXT or JSON format — see above comment
            file_format = config.file.format
            if file_format == LogFormat.COLORED:
                file_format = LogFormat.TEXT
            format_func = _create_context_format(config, file_format)
            logger.add(
                str(log_path),
                level=config.file.level.value,
                format=format_func,
                rotation=config.file.rotation,
                retention=config.file.retention,
                compression=config.file.compression,
                colorize=False,
                backtrace=True,
                diagnose=False,  # SECURITY: Never enable - exposes local variables including secrets
            )

    # Loki handler
    if config.loki.enabled:
        auth = None
        if config.loki.auth_username and config.loki.auth_password:
            auth = (config.loki.auth_username, config.loki.auth_password)

        loki_sink = LoguruLokiSink(
            url=config.loki.url,
            labels=config.loki.labels,
            auth=auth,
            tenant_id=config.loki.tenant_id,
            batch_size=config.loki.batch_size,
            flush_interval=config.loki.flush_interval,
        )
        loki_sink.start()
        _handlers.append(loki_sink)

        logger.add(
            loki_sink,
            level=config.loki.level.value,
            format="{message}",  # Raw message, formatting done in handler
        )
        logger.info(f"Loki logging enabled: {config.loki.url}")

    # Elasticsearch handler
    if config.elasticsearch.enabled:
        auth = None
        if config.elasticsearch.auth_username and config.elasticsearch.auth_password:
            auth = (config.elasticsearch.auth_username, config.elasticsearch.auth_password)

        es_sink = LoguruElasticsearchSink(
            url=config.elasticsearch.url,
            index_prefix=config.elasticsearch.index_prefix,
            auth=auth,
            api_key=config.elasticsearch.api_key,
            batch_size=config.elasticsearch.batch_size,
            flush_interval=config.elasticsearch.flush_interval,
        )
        es_sink.start()
        _handlers.append(es_sink)

        logger.add(
            es_sink,
            level=config.elasticsearch.level.value,
            format="{message}",  # Raw message, formatting done in handler
        )
        logger.info(f"Elasticsearch logging enabled: {config.elasticsearch.url}")

    logger.info(
        f"Logging configured for service '{config.service_name}' "
        f"(level: {config.default_level.value})"
    )


def _parse_size(size_str: str) -> int:
    """Parse size string like '10 MB' to bytes."""
    size_str = size_str.strip().upper()
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
    }

    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                value = float(size_str[:-len(unit)].strip())
                return int(value * multiplier)
            except ValueError:
                pass

    # Default to 10 MB
    return 10 * 1024 * 1024


def get_logger(name: str = None, **context) -> 'logger':
    """
    Get a logger instance with optional context binding.

    Args:
        name: Logger name (defaults to calling module)
        **context: Additional context to bind to the logger

    Returns:
        Configured logger instance with context
    """
    bound_logger = logger.bind(logger_name=name) if name else logger

    if context:
        bound_logger = bound_logger.bind(**context)

    return bound_logger


def shutdown_logging():
    """Shutdown all logging handlers gracefully."""
    global _handlers

    for handler in _handlers:
        try:
            if hasattr(handler, 'stop'):
                handler.stop()
            elif hasattr(handler, 'close'):
                handler.close()
        except Exception:
            pass

    _handlers.clear()
    logger.info("Logging shutdown complete")


# Register shutdown handler
import atexit
atexit.register(shutdown_logging)


# Convenience function for quick setup
def setup_logging(
    service_name: str = 'rdt-trading',
    level: str = 'INFO',
    json_format: bool = False,
    loki_url: str = None,
    elasticsearch_url: str = None,
) -> None:
    """
    Quick logging setup with common defaults.

    Args:
        service_name: Name of the service
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON format for console output
        loki_url: Optional Loki URL to enable Loki logging
        elasticsearch_url: Optional Elasticsearch URL to enable ES logging
    """
    try:
        log_level = LogLevel(level.upper())
    except ValueError:
        log_level = LogLevel.INFO

    config = LoggingConfig(
        service_name=service_name,
        default_level=log_level,
        console=ConsoleConfig(
            enabled=True,
            level=log_level,
            format=LogFormat.JSON if json_format else LogFormat.COLORED,
        ),
        file=FileConfig(
            enabled=True,
            level=LogLevel.DEBUG,
            path=f'logs/{service_name}.log',
            format=LogFormat.JSON,
        ),
        loki=LokiConfig(
            enabled=bool(loki_url),
            url=loki_url or 'http://localhost:3100',
            level=log_level,
            labels={'app': service_name},
        ),
        elasticsearch=ElasticsearchConfig(
            enabled=bool(elasticsearch_url),
            url=elasticsearch_url or 'http://localhost:9200',
            level=log_level,
            index_prefix=f'{service_name}-logs',
        ),
    )

    configure_logging(config)
