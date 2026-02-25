"""
Flask Tracing Middleware for RDT Trading System

Provides automatic request tracing for Flask applications:
- Creates spans for all incoming HTTP requests
- Captures request/response metadata
- Propagates trace context from incoming headers
- Records errors and exceptions
- Integrates with OpenTelemetry Flask instrumentation

Usage:
    from flask import Flask
    from tracing import init_flask_tracing

    app = Flask(__name__)
    init_flask_tracing(app)
"""

import time
from typing import Optional, Dict, Any, List, Callable
from functools import wraps
from loguru import logger

# Flask imports
try:
    from flask import Flask, request, g, Response
    from werkzeug.exceptions import HTTPException
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.debug("Flask not available")

# OpenTelemetry imports
from tracing.tracer import (
    OTEL_AVAILABLE,
    get_tracer,
    get_current_span,
    get_current_trace_id,
    get_current_span_id,
    extract_context,
    inject_context,
    is_tracing_enabled,
    NoOpSpan,
)

if OTEL_AVAILABLE:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from opentelemetry.semconv.trace import SpanAttributes


# Semantic conventions for HTTP spans
HTTP_METHOD = "http.method"
HTTP_URL = "http.url"
HTTP_TARGET = "http.target"
HTTP_HOST = "http.host"
HTTP_SCHEME = "http.scheme"
HTTP_STATUS_CODE = "http.status_code"
HTTP_FLAVOR = "http.flavor"
HTTP_USER_AGENT = "http.user_agent"
HTTP_REQUEST_CONTENT_LENGTH = "http.request_content_length"
HTTP_RESPONSE_CONTENT_LENGTH = "http.response_content_length"
HTTP_ROUTE = "http.route"
HTTP_CLIENT_IP = "http.client_ip"

# Custom attributes
RDT_USER_ID = "rdt.user_id"
RDT_CORRELATION_ID = "rdt.correlation_id"
RDT_REQUEST_ID = "rdt.request_id"


class TracingMiddleware:
    """
    WSGI middleware for tracing HTTP requests.

    Wraps a Flask application to provide automatic span creation
    for all incoming requests.

    Attributes:
        app: The Flask application
        excluded_paths: List of paths to exclude from tracing
        excluded_methods: List of HTTP methods to exclude
    """

    def __init__(
        self,
        app: "Flask",
        excluded_paths: Optional[List[str]] = None,
        excluded_methods: Optional[List[str]] = None,
        record_request_body: bool = False,
        record_response_body: bool = False,
    ):
        """
        Initialize tracing middleware.

        Args:
            app: Flask application
            excluded_paths: Paths to exclude from tracing (e.g., ["/health", "/metrics"])
            excluded_methods: HTTP methods to exclude (e.g., ["OPTIONS"])
            record_request_body: Whether to record request body in span
            record_response_body: Whether to record response body in span
        """
        self.app = app
        self.excluded_paths = excluded_paths or ["/health", "/metrics", "/favicon.ico"]
        self.excluded_methods = excluded_methods or ["OPTIONS"]
        self.record_request_body = record_request_body
        self.record_response_body = record_response_body

        self._setup_hooks()

    def _setup_hooks(self):
        """Set up Flask before/after request hooks."""
        if not FLASK_AVAILABLE:
            return

        @self.app.before_request
        def before_request():
            """Start span for incoming request."""
            if self._should_exclude(request.path, request.method):
                g.tracing_span = None
                return

            # Extract context from incoming headers
            carrier = dict(request.headers)
            ctx = extract_context(carrier)

            # Create span for this request
            tracer = get_tracer("flask")

            span_name = f"{request.method} {request.path}"

            # Start span with extracted context
            if OTEL_AVAILABLE and ctx:
                span = tracer.start_span(
                    span_name,
                    context=ctx,
                    kind=SpanKind.SERVER if OTEL_AVAILABLE else None,
                )
            else:
                span = tracer.start_span(
                    span_name,
                    kind=SpanKind.SERVER if OTEL_AVAILABLE else None,
                )

            # Set request attributes
            if not isinstance(span, NoOpSpan):
                span.set_attribute(HTTP_METHOD, request.method)
                span.set_attribute(HTTP_URL, request.url)
                span.set_attribute(HTTP_TARGET, request.path)
                span.set_attribute(HTTP_HOST, request.host)
                span.set_attribute(HTTP_SCHEME, request.scheme)
                span.set_attribute(HTTP_USER_AGENT, request.user_agent.string or "")
                span.set_attribute(HTTP_CLIENT_IP, request.remote_addr or "")

                if request.content_length:
                    span.set_attribute(HTTP_REQUEST_CONTENT_LENGTH, request.content_length)

                # Add custom RDT attributes
                correlation_id = request.headers.get("X-Correlation-ID")
                if correlation_id:
                    span.set_attribute(RDT_CORRELATION_ID, correlation_id)

                request_id = request.headers.get("X-Request-ID")
                if request_id:
                    span.set_attribute(RDT_REQUEST_ID, request_id)

                # Record query parameters (sanitized)
                if request.args:
                    safe_params = {
                        k: v for k, v in request.args.items()
                        if k.lower() not in ("password", "token", "api_key", "secret")
                    }
                    if safe_params:
                        span.set_attribute("http.query_params", str(safe_params))

            # Store span in Flask's g object
            g.tracing_span = span
            g.tracing_start_time = time.time()

            # Make trace context available
            g.trace_id = get_current_trace_id()
            g.span_id = get_current_span_id()

        @self.app.after_request
        def after_request(response: "Response") -> "Response":
            """Complete span after request."""
            span = getattr(g, "tracing_span", None)
            if span is None:
                return response

            try:
                if not isinstance(span, NoOpSpan):
                    # Set response attributes
                    span.set_attribute(HTTP_STATUS_CODE, response.status_code)

                    if response.content_length:
                        span.set_attribute(HTTP_RESPONSE_CONTENT_LENGTH, response.content_length)

                    # Calculate request duration
                    start_time = getattr(g, "tracing_start_time", None)
                    if start_time:
                        duration_ms = (time.time() - start_time) * 1000
                        span.set_attribute("http.duration_ms", duration_ms)

                    # Set span status based on HTTP status code
                    if response.status_code >= 500:
                        span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                    elif response.status_code >= 400:
                        span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                    else:
                        span.set_status(Status(StatusCode.OK))

                    # Add route if available
                    if request.url_rule:
                        span.set_attribute(HTTP_ROUTE, request.url_rule.rule)

                # End the span
                span.end()

            except Exception as e:
                logger.debug(f"Error completing trace span: {e}")

            # Add trace headers to response
            trace_id = getattr(g, "trace_id", None)
            if trace_id:
                response.headers["X-Trace-ID"] = trace_id

            return response

        @self.app.teardown_request
        def teardown_request(exception: Optional[Exception]):
            """Record exception if request failed."""
            span = getattr(g, "tracing_span", None)
            if span is None or isinstance(span, NoOpSpan):
                return

            try:
                if exception is not None:
                    span.record_exception(exception)
                    span.set_status(Status(StatusCode.ERROR, str(exception)))

                    # Add exception details
                    span.set_attribute("error.type", type(exception).__name__)
                    span.set_attribute("error.message", str(exception))

                    if isinstance(exception, HTTPException):
                        span.set_attribute(HTTP_STATUS_CODE, exception.code)
            except Exception as e:
                logger.debug(f"Error recording exception in span: {e}")

    def _should_exclude(self, path: str, method: str) -> bool:
        """Check if request should be excluded from tracing."""
        if not is_tracing_enabled():
            return True

        if method in self.excluded_methods:
            return True

        for excluded_path in self.excluded_paths:
            if path.startswith(excluded_path):
                return True

        return False


def init_flask_tracing(
    app: "Flask",
    excluded_paths: Optional[List[str]] = None,
    excluded_methods: Optional[List[str]] = None,
    use_opentelemetry_instrumentation: bool = True,
) -> Optional[TracingMiddleware]:
    """
    Initialize tracing for a Flask application.

    Args:
        app: Flask application
        excluded_paths: Paths to exclude from tracing
        excluded_methods: HTTP methods to exclude
        use_opentelemetry_instrumentation: Use OpenTelemetry Flask instrumentation if available

    Returns:
        TracingMiddleware instance or None if tracing disabled
    """
    if not FLASK_AVAILABLE:
        logger.warning("Flask not available, cannot initialize Flask tracing")
        return None

    if not is_tracing_enabled():
        logger.info("Tracing disabled, skipping Flask instrumentation")
        return None

    # Try to use OpenTelemetry Flask instrumentation first
    if use_opentelemetry_instrumentation:
        try:
            from opentelemetry.instrumentation.flask import FlaskInstrumentor

            FlaskInstrumentor().instrument_app(
                app,
                excluded_urls=",".join(excluded_paths or ["/health", "/metrics"]),
            )
            logger.info("OpenTelemetry Flask instrumentation enabled")

            # Still add our middleware for custom attributes
            middleware = TracingMiddleware(
                app,
                excluded_paths=excluded_paths,
                excluded_methods=excluded_methods,
            )
            return middleware

        except ImportError:
            logger.debug("OpenTelemetry Flask instrumentation not available, using custom middleware")

    # Fall back to custom middleware
    middleware = TracingMiddleware(
        app,
        excluded_paths=excluded_paths,
        excluded_methods=excluded_methods,
    )
    logger.info("Custom Flask tracing middleware enabled")
    return middleware


def trace_background_task(
    task_name: str,
    parent_context: Optional[Dict[str, str]] = None,
) -> Callable:
    """
    Decorator for tracing background tasks.

    Creates a new span for the background task, optionally linking
    to a parent context from the originating request.

    Args:
        task_name: Name for the background task span
        parent_context: Optional carrier dict with parent trace context

    Returns:
        Decorator function

    Usage:
        @trace_background_task("process-order", parent_context=context)
        def process_order(order_id):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_tracing_enabled():
                return func(*args, **kwargs)

            tracer = get_tracer()

            # Extract parent context if provided
            ctx = None
            if parent_context:
                ctx = extract_context(parent_context)

            if OTEL_AVAILABLE:
                with tracer.start_as_current_span(
                    task_name,
                    context=ctx,
                    kind=SpanKind.INTERNAL,
                ) as span:
                    span.set_attribute("task.name", task_name)
                    span.set_attribute("task.type", "background")
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            else:
                return func(*args, **kwargs)

        return wrapper
    return decorator


def get_trace_context_for_propagation() -> Dict[str, str]:
    """
    Get current trace context for propagation to background tasks or external services.

    Returns:
        Dictionary with trace context headers

    Usage:
        # In request handler
        context = get_trace_context_for_propagation()

        # Pass to background task
        background_task.delay(data, trace_context=context)
    """
    carrier: Dict[str, str] = {}
    if is_tracing_enabled():
        inject_context(carrier)
    return carrier


def add_user_context(user_id: Optional[str] = None, username: Optional[str] = None):
    """
    Add user context to the current span.

    Args:
        user_id: User ID
        username: Username (optional)
    """
    span = get_current_span()
    if isinstance(span, NoOpSpan):
        return

    if user_id:
        span.set_attribute(RDT_USER_ID, user_id)
        span.set_attribute("enduser.id", user_id)

    if username:
        span.set_attribute("enduser.name", username)


def add_trading_context(
    symbol: Optional[str] = None,
    action: Optional[str] = None,
    quantity: Optional[int] = None,
    price: Optional[float] = None,
):
    """
    Add trading-specific context to the current span.

    Args:
        symbol: Trading symbol
        action: Trading action (buy, sell, etc.)
        quantity: Trade quantity
        price: Trade price
    """
    span = get_current_span()
    if isinstance(span, NoOpSpan):
        return

    if symbol:
        span.set_attribute("trading.symbol", symbol)
    if action:
        span.set_attribute("trading.action", action)
    if quantity is not None:
        span.set_attribute("trading.quantity", quantity)
    if price is not None:
        span.set_attribute("trading.price", price)
