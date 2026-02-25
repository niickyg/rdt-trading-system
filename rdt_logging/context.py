"""
RDT Trading System - Logging Context Management

Provides:
- LogContext class for adding context to logs
- Request context middleware for Flask
- Correlation ID generation and propagation
- User context injection
- Thread-safe context storage
"""

import contextvars
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Optional

# Context variables for async-safe storage
_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('correlation_id', default=None)
_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('request_id', default=None)
_user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('user_id', default=None)
_user_email: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('user_email', default=None)
_service_name: contextvars.ContextVar[str] = contextvars.ContextVar('service_name', default='rdt-trading')
_extra_context: contextvars.ContextVar[dict] = contextvars.ContextVar('extra_context', default={})

# Thread-local fallback for non-async contexts
_thread_local = threading.local()


def generate_correlation_id() -> str:
    """Generate a unique correlation ID."""
    return str(uuid.uuid4())


def generate_request_id() -> str:
    """Generate a unique request ID with timestamp prefix."""
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
    unique = uuid.uuid4().hex[:12]
    return f"req_{timestamp}_{unique}"


# Correlation ID functions
def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    try:
        return _correlation_id.get()
    except LookupError:
        return getattr(_thread_local, 'correlation_id', None)


def set_correlation_id(correlation_id: Optional[str]) -> None:
    """Set the correlation ID for the current context."""
    try:
        _correlation_id.set(correlation_id)
    except LookupError:
        pass
    _thread_local.correlation_id = correlation_id


# Request ID functions
def get_request_id() -> Optional[str]:
    """Get the current request ID."""
    try:
        return _request_id.get()
    except LookupError:
        return getattr(_thread_local, 'request_id', None)


def set_request_id(request_id: Optional[str]) -> None:
    """Set the request ID for the current context."""
    try:
        _request_id.set(request_id)
    except LookupError:
        pass
    _thread_local.request_id = request_id


# User context functions
def get_user_id() -> Optional[str]:
    """Get the current user ID."""
    try:
        return _user_id.get()
    except LookupError:
        return getattr(_thread_local, 'user_id', None)


def set_user_id(user_id: Optional[str]) -> None:
    """Set the user ID for the current context."""
    try:
        _user_id.set(user_id)
    except LookupError:
        pass
    _thread_local.user_id = user_id


def get_user_email() -> Optional[str]:
    """Get the current user email."""
    try:
        return _user_email.get()
    except LookupError:
        return getattr(_thread_local, 'user_email', None)


def set_user_email(email: Optional[str]) -> None:
    """Set the user email for the current context."""
    try:
        _user_email.set(email)
    except LookupError:
        pass
    _thread_local.user_email = email


# Service name functions
def get_service_name() -> str:
    """Get the current service name."""
    try:
        return _service_name.get()
    except LookupError:
        return getattr(_thread_local, 'service_name', 'rdt-trading')


def set_service_name(name: str) -> None:
    """Set the service name for the current context."""
    try:
        _service_name.set(name)
    except LookupError:
        pass
    _thread_local.service_name = name


# Extra context functions
def get_extra_context() -> dict:
    """Get extra context dictionary."""
    try:
        return _extra_context.get().copy()
    except LookupError:
        return getattr(_thread_local, 'extra_context', {}).copy()


def set_extra_context(context: dict) -> None:
    """Set extra context dictionary."""
    try:
        _extra_context.set(context.copy())
    except LookupError:
        pass
    _thread_local.extra_context = context.copy()


def update_extra_context(**kwargs) -> None:
    """Update extra context with additional key-value pairs."""
    current = get_extra_context()
    current.update(kwargs)
    set_extra_context(current)


def get_full_context() -> dict:
    """Get all context values as a dictionary."""
    context = {
        'correlation_id': get_correlation_id(),
        'request_id': get_request_id(),
        'user_id': get_user_id(),
        'user_email': get_user_email(),
        'service': get_service_name(),
    }
    # Add extra context
    context.update(get_extra_context())
    # Remove None values
    return {k: v for k, v in context.items() if v is not None}


def clear_context() -> None:
    """Clear all context values."""
    set_correlation_id(None)
    set_request_id(None)
    set_user_id(None)
    set_user_email(None)
    set_extra_context({})


@dataclass
class LogContext:
    """
    Context manager for adding structured context to logs.

    Automatically injects context fields into all log messages
    within the context scope.

    Example:
        with LogContext(user_id='123', action='trade'):
            logger.info("Processing trade")  # Includes user_id and action

        # Or use as a decorator
        @LogContext(component='scanner')
        def scan_market():
            logger.info("Scanning")  # Includes component='scanner'
    """

    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    service_name: Optional[str] = None
    auto_generate_correlation_id: bool = False
    auto_generate_request_id: bool = False
    extra: dict = field(default_factory=dict)

    # Store previous context for restoration
    _previous_context: dict = field(default_factory=dict, repr=False)

    def __enter__(self):
        """Enter the context and set context variables."""
        # Save previous context
        self._previous_context = {
            'correlation_id': get_correlation_id(),
            'request_id': get_request_id(),
            'user_id': get_user_id(),
            'user_email': get_user_email(),
            'service_name': get_service_name(),
            'extra': get_extra_context(),
        }

        # Set new context values
        if self.correlation_id:
            set_correlation_id(self.correlation_id)
        elif self.auto_generate_correlation_id and not get_correlation_id():
            set_correlation_id(generate_correlation_id())

        if self.request_id:
            set_request_id(self.request_id)
        elif self.auto_generate_request_id and not get_request_id():
            set_request_id(generate_request_id())

        if self.user_id:
            set_user_id(self.user_id)

        if self.user_email:
            set_user_email(self.user_email)

        if self.service_name:
            set_service_name(self.service_name)

        if self.extra:
            current_extra = get_extra_context()
            current_extra.update(self.extra)
            set_extra_context(current_extra)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore previous context."""
        # Restore previous context
        set_correlation_id(self._previous_context.get('correlation_id'))
        set_request_id(self._previous_context.get('request_id'))
        set_user_id(self._previous_context.get('user_id'))
        set_user_email(self._previous_context.get('user_email'))
        if self._previous_context.get('service_name'):
            set_service_name(self._previous_context['service_name'])
        set_extra_context(self._previous_context.get('extra', {}))
        return False

    def __call__(self, func: Callable) -> Callable:
        """Use LogContext as a decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


@contextmanager
def log_context(**kwargs):
    """
    Simple context manager for adding context to logs.

    Example:
        with log_context(user_id='123', action='buy'):
            logger.info("Executing order")
    """
    ctx = LogContext(
        correlation_id=kwargs.pop('correlation_id', None),
        request_id=kwargs.pop('request_id', None),
        user_id=kwargs.pop('user_id', None),
        user_email=kwargs.pop('user_email', None),
        service_name=kwargs.pop('service_name', None),
        auto_generate_correlation_id=kwargs.pop('auto_generate_correlation_id', False),
        auto_generate_request_id=kwargs.pop('auto_generate_request_id', False),
        extra=kwargs,
    )
    with ctx:
        yield ctx


def inject_context(logger_instance):
    """
    Create a logger wrapper that automatically injects context.

    Args:
        logger_instance: The Loguru logger instance

    Returns:
        A configured logger with context injection
    """
    return logger_instance.bind(**get_full_context())


class FlaskRequestContext:
    """
    Flask middleware for automatic request context management.

    Automatically:
    - Generates/propagates correlation IDs
    - Generates request IDs
    - Extracts user information from session
    - Logs request start/end with timing
    """

    def __init__(
        self,
        app=None,
        service_name: str = 'rdt-trading',
        correlation_id_header: str = 'X-Correlation-ID',
        request_id_header: str = 'X-Request-ID',
        log_requests: bool = True,
        log_responses: bool = True,
        exclude_paths: list = None,
    ):
        """
        Initialize the Flask request context middleware.

        Args:
            app: Flask application instance
            service_name: Name of the service
            correlation_id_header: Header name for correlation ID
            request_id_header: Header name for request ID
            log_requests: Log incoming requests
            log_responses: Log outgoing responses with timing
            exclude_paths: Paths to exclude from logging (e.g., ['/health', '/metrics'])
        """
        self.service_name = service_name
        self.correlation_id_header = correlation_id_header
        self.request_id_header = request_id_header
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.exclude_paths = exclude_paths or ['/health', '/metrics', '/favicon.ico']

        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """Initialize the middleware with a Flask app."""
        from flask import g, request
        from loguru import logger

        set_service_name(self.service_name)

        @app.before_request
        def before_request():
            """Set up request context before each request."""
            import time

            # Skip excluded paths
            if request.path in self.exclude_paths:
                return

            # Store request start time
            g.request_start_time = time.time()

            # Get or generate correlation ID
            correlation_id = request.headers.get(self.correlation_id_header)
            if not correlation_id:
                correlation_id = generate_correlation_id()
            set_correlation_id(correlation_id)

            # Generate request ID
            request_id = request.headers.get(self.request_id_header)
            if not request_id:
                request_id = generate_request_id()
            set_request_id(request_id)

            # Extract user context if available
            try:
                from flask_login import current_user
                if current_user and current_user.is_authenticated:
                    set_user_id(str(current_user.id))
                    if hasattr(current_user, 'email'):
                        set_user_email(current_user.email)
            except (ImportError, RuntimeError):
                pass

            # Set extra context
            update_extra_context(
                method=request.method,
                path=request.path,
                remote_addr=request.remote_addr,
                user_agent=request.headers.get('User-Agent', '')[:100],
            )

            # Log request
            if self.log_requests:
                logger.bind(**get_full_context()).info(
                    f"Request started: {request.method} {request.path}",
                )

        @app.after_request
        def after_request(response):
            """Add context headers to response and log completion."""
            import time

            # Skip excluded paths
            if request.path in self.exclude_paths:
                return response

            # Add correlation ID to response headers
            correlation_id = get_correlation_id()
            if correlation_id:
                response.headers[self.correlation_id_header] = correlation_id

            request_id = get_request_id()
            if request_id:
                response.headers[self.request_id_header] = request_id

            # Log response
            if self.log_responses:
                duration_ms = 0
                if hasattr(g, 'request_start_time'):
                    duration_ms = (time.time() - g.request_start_time) * 1000

                log_data = get_full_context()
                log_data['status_code'] = response.status_code
                log_data['duration_ms'] = round(duration_ms, 2)
                log_data['content_length'] = response.content_length

                # Choose log level based on status code
                if response.status_code >= 500:
                    logger.bind(**log_data).error(
                        f"Request completed: {request.method} {request.path} - {response.status_code}"
                    )
                elif response.status_code >= 400:
                    logger.bind(**log_data).warning(
                        f"Request completed: {request.method} {request.path} - {response.status_code}"
                    )
                else:
                    logger.bind(**log_data).info(
                        f"Request completed: {request.method} {request.path} - {response.status_code}"
                    )

            return response

        @app.teardown_request
        def teardown_request(exception=None):
            """Clean up context after request."""
            if exception:
                from loguru import logger
                logger.bind(**get_full_context()).exception(
                    f"Request failed with exception: {exception}"
                )
            clear_context()


def create_flask_context_middleware(
    app,
    service_name: str = 'rdt-trading',
    log_requests: bool = True,
    log_responses: bool = True,
    exclude_paths: list = None,
) -> FlaskRequestContext:
    """
    Create and initialize Flask request context middleware.

    Args:
        app: Flask application instance
        service_name: Name of the service
        log_requests: Log incoming requests
        log_responses: Log responses with timing
        exclude_paths: Paths to exclude from logging

    Returns:
        Configured FlaskRequestContext instance
    """
    middleware = FlaskRequestContext(
        app=app,
        service_name=service_name,
        log_requests=log_requests,
        log_responses=log_responses,
        exclude_paths=exclude_paths,
    )
    return middleware


class AsyncLogContext:
    """
    Async-compatible context manager for adding context to logs.

    Example:
        async with AsyncLogContext(user_id='123'):
            logger.info("Processing async task")
    """

    def __init__(
        self,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        auto_generate_correlation_id: bool = False,
        **extra,
    ):
        self.correlation_id = correlation_id
        self.request_id = request_id
        self.user_id = user_id
        self.auto_generate_correlation_id = auto_generate_correlation_id
        self.extra = extra
        self._tokens = {}

    async def __aenter__(self):
        """Enter the async context."""
        # Store tokens for restoration
        if self.correlation_id:
            self._tokens['correlation_id'] = _correlation_id.set(self.correlation_id)
        elif self.auto_generate_correlation_id:
            self._tokens['correlation_id'] = _correlation_id.set(generate_correlation_id())

        if self.request_id:
            self._tokens['request_id'] = _request_id.set(self.request_id)

        if self.user_id:
            self._tokens['user_id'] = _user_id.set(self.user_id)

        if self.extra:
            current = _extra_context.get({})
            new_extra = {**current, **self.extra}
            self._tokens['extra'] = _extra_context.set(new_extra)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context and restore previous values."""
        for name, token in self._tokens.items():
            var = {
                'correlation_id': _correlation_id,
                'request_id': _request_id,
                'user_id': _user_id,
                'extra': _extra_context,
            }.get(name)
            if var:
                var.reset(token)
        return False
