"""
Distributed Tracing Module for RDT Trading System

Provides OpenTelemetry-based distributed tracing with support for:
- Multiple exporters (Jaeger, Zipkin, Console)
- Flask middleware for automatic HTTP request tracing
- Function decorators for custom span creation
- Context propagation helpers for async operations
- Automatic error tracking and recording

Usage:
    from tracing import init_tracing, get_tracer, trace

    # Initialize tracing at application startup
    init_tracing(service_name="rdt-trading-system")

    # Get tracer for manual instrumentation
    tracer = get_tracer()

    # Use decorator for automatic function tracing
    @trace("operation_name")
    def my_function():
        pass
"""

from tracing.config import TracingConfig
from tracing.tracer import (
    init_tracing,
    shutdown_tracing,
    get_tracer,
    get_current_span,
    get_current_trace_id,
    get_current_span_id,
    create_span_context,
    inject_context,
    extract_context,
)
from tracing.decorators import (
    trace,
    trace_async,
    trace_method,
    trace_class,
)
from tracing.middleware import (
    TracingMiddleware,
    init_flask_tracing,
)

__all__ = [
    # Configuration
    "TracingConfig",
    # Tracer functions
    "init_tracing",
    "shutdown_tracing",
    "get_tracer",
    "get_current_span",
    "get_current_trace_id",
    "get_current_span_id",
    "create_span_context",
    "inject_context",
    "extract_context",
    # Decorators
    "trace",
    "trace_async",
    "trace_method",
    "trace_class",
    # Middleware
    "TracingMiddleware",
    "init_flask_tracing",
]

__version__ = "1.0.0"
