"""
RDT Trading System - Centralized Logging Package

Provides unified logging with support for:
- Multiple outputs: console, file, Loki, Elasticsearch
- Structured JSON logging
- Context injection (request_id, user_id, service_name)
- Sensitive data masking
- Correlation ID propagation

Usage:
    from rdt_logging import configure_logging, get_logger, LogContext

    # Configure logging from environment
    configure_logging()

    # Get a logger with context
    logger = get_logger('my_module')

    # Use context for request tracking
    with LogContext(user_id='123', correlation_id='abc'):
        logger.info("Processing request")
"""

from rdt_logging.config import (
    LoggingConfig,
    LogLevel,
    LogFormat,
    ConsoleConfig,
    FileConfig,
    LokiConfig,
    ElasticsearchConfig,
    configure_logging,
    get_logger,
    setup_logging,
    shutdown_logging,
)
from rdt_logging.context import (
    LogContext,
    AsyncLogContext,
    FlaskRequestContext,
    create_flask_context_middleware,
    get_correlation_id,
    set_correlation_id,
    get_request_id,
    set_request_id,
    get_user_id,
    set_user_id,
    get_full_context,
    clear_context,
    log_context,
    generate_correlation_id,
    generate_request_id,
)
from rdt_logging.formatters import (
    JSONFormatter,
    LoguruJSONSink,
    mask_sensitive_data,
    create_json_format_string,
)
from rdt_logging.handlers import (
    LokiHandler,
    ElasticsearchHandler,
    LoguruLokiSink,
    LoguruElasticsearchSink,
    BufferedFileHandler,
)

__all__ = [
    # Configuration
    'LoggingConfig',
    'LogLevel',
    'LogFormat',
    'ConsoleConfig',
    'FileConfig',
    'LokiConfig',
    'ElasticsearchConfig',
    'configure_logging',
    'get_logger',
    'setup_logging',
    'shutdown_logging',

    # Context
    'LogContext',
    'AsyncLogContext',
    'FlaskRequestContext',
    'create_flask_context_middleware',
    'get_correlation_id',
    'set_correlation_id',
    'get_request_id',
    'set_request_id',
    'get_user_id',
    'set_user_id',
    'get_full_context',
    'clear_context',
    'log_context',
    'generate_correlation_id',
    'generate_request_id',

    # Formatters
    'JSONFormatter',
    'LoguruJSONSink',
    'mask_sensitive_data',
    'create_json_format_string',

    # Handlers
    'LokiHandler',
    'ElasticsearchHandler',
    'LoguruLokiSink',
    'LoguruElasticsearchSink',
    'BufferedFileHandler',
]

__version__ = '1.0.0'
