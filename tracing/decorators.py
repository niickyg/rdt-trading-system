"""
Tracing Decorators for RDT Trading System

Provides decorators for easy function and method tracing:
- @trace: Synchronous function tracing
- @trace_async: Asynchronous function tracing
- @trace_method: Instance method tracing
- @trace_class: Class-level tracing for all methods

Features:
- Automatic span creation with function name
- Attribute capture from function arguments
- Error recording and status setting
- Support for custom span names and attributes

Usage:
    from tracing import trace, trace_async

    @trace("process-order")
    def process_order(order_id: str):
        pass

    @trace_async("fetch-data", capture_args=["symbol"])
    async def fetch_market_data(symbol: str):
        pass
"""

import asyncio
import functools
import inspect
from typing import Optional, Dict, Any, List, Callable, Union, Type
from loguru import logger

from tracing.tracer import (
    OTEL_AVAILABLE,
    get_tracer,
    get_current_span,
    is_tracing_enabled,
    NoOpSpan,
)

if OTEL_AVAILABLE:
    from opentelemetry.trace import SpanKind, Status, StatusCode


def trace(
    name: Optional[str] = None,
    kind: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    capture_args: Optional[List[str]] = None,
    capture_result: bool = False,
    record_exception: bool = True,
    set_status_on_exception: bool = True,
) -> Callable:
    """
    Decorator for tracing synchronous functions.

    Creates a span that tracks the execution of the decorated function,
    including timing, arguments, results, and exceptions.

    Args:
        name: Span name (defaults to function name)
        kind: Span kind ("internal", "client", "server", "producer", "consumer")
        attributes: Static attributes to add to span
        capture_args: List of argument names to capture as attributes
        capture_result: Whether to capture the return value
        record_exception: Whether to record exceptions
        set_status_on_exception: Whether to set error status on exception

    Returns:
        Decorated function

    Usage:
        @trace("calculate-rrs", capture_args=["symbol"], capture_result=True)
        def calculate_rrs(symbol: str, data: dict) -> float:
            return 2.5
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_tracing_enabled():
                return func(*args, **kwargs)

            tracer = get_tracer()

            # Determine span kind
            span_kind = _get_span_kind(kind)

            with tracer.start_as_current_span(
                span_name,
                kind=span_kind,
                record_exception=record_exception,
                set_status_on_exception=set_status_on_exception,
            ) as span:
                if isinstance(span, NoOpSpan):
                    return func(*args, **kwargs)

                # Add function metadata
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)

                # Add static attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Capture specified arguments
                if capture_args:
                    _capture_function_args(span, func, args, kwargs, capture_args)

                try:
                    result = func(*args, **kwargs)

                    # Capture result if requested
                    if capture_result and result is not None:
                        _capture_result(span, result)

                    if OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.OK))

                    return result

                except Exception as e:
                    if record_exception:
                        span.record_exception(e)
                    if set_status_on_exception and OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise

        return wrapper
    return decorator


def trace_async(
    name: Optional[str] = None,
    kind: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    capture_args: Optional[List[str]] = None,
    capture_result: bool = False,
    record_exception: bool = True,
    set_status_on_exception: bool = True,
) -> Callable:
    """
    Decorator for tracing asynchronous functions.

    Same as @trace but for async functions.

    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        attributes: Static attributes to add to span
        capture_args: List of argument names to capture as attributes
        capture_result: Whether to capture the return value
        record_exception: Whether to record exceptions
        set_status_on_exception: Whether to set error status on exception

    Returns:
        Decorated async function

    Usage:
        @trace_async("fetch-market-data", capture_args=["symbol"])
        async def fetch_data(symbol: str) -> dict:
            pass
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not is_tracing_enabled():
                return await func(*args, **kwargs)

            tracer = get_tracer()
            span_kind = _get_span_kind(kind)

            with tracer.start_as_current_span(
                span_name,
                kind=span_kind,
                record_exception=record_exception,
                set_status_on_exception=set_status_on_exception,
            ) as span:
                if isinstance(span, NoOpSpan):
                    return await func(*args, **kwargs)

                # Add function metadata
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)
                span.set_attribute("code.async", True)

                # Add static attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Capture specified arguments
                if capture_args:
                    _capture_function_args(span, func, args, kwargs, capture_args)

                try:
                    result = await func(*args, **kwargs)

                    # Capture result if requested
                    if capture_result and result is not None:
                        _capture_result(span, result)

                    if OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.OK))

                    return result

                except Exception as e:
                    if record_exception:
                        span.record_exception(e)
                    if set_status_on_exception and OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise

        return wrapper
    return decorator


def trace_method(
    name: Optional[str] = None,
    capture_args: Optional[List[str]] = None,
    capture_result: bool = False,
) -> Callable:
    """
    Decorator for tracing instance methods.

    Similar to @trace but includes class name in span name.

    Args:
        name: Span name (defaults to "ClassName.method_name")
        capture_args: List of argument names to capture
        capture_result: Whether to capture the return value

    Returns:
        Decorated method

    Usage:
        class OrderExecutor:
            @trace_method(capture_args=["order_id"])
            def execute_order(self, order_id: str):
                pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not is_tracing_enabled():
                return func(self, *args, **kwargs)

            # Build span name with class name
            class_name = self.__class__.__name__
            span_name = name or f"{class_name}.{func.__name__}"

            tracer = get_tracer()

            with tracer.start_as_current_span(span_name) as span:
                if isinstance(span, NoOpSpan):
                    return func(self, *args, **kwargs)

                # Add class and method metadata
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.class", class_name)
                span.set_attribute("code.namespace", func.__module__)

                # Capture specified arguments
                if capture_args:
                    _capture_function_args(span, func, args, kwargs, capture_args)

                try:
                    result = func(self, *args, **kwargs)

                    if capture_result and result is not None:
                        _capture_result(span, result)

                    if OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.OK))

                    return result

                except Exception as e:
                    span.record_exception(e)
                    if OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        return wrapper
    return decorator


def trace_class(
    prefix: Optional[str] = None,
    methods: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    capture_args: Optional[List[str]] = None,
) -> Callable[[Type], Type]:
    """
    Class decorator to trace all methods.

    Applies tracing to all public methods of a class (or specified methods).

    Args:
        prefix: Prefix for span names (defaults to class name)
        methods: Specific methods to trace (traces all public if None)
        exclude: Methods to exclude from tracing
        capture_args: Arguments to capture for all methods

    Returns:
        Decorated class

    Usage:
        @trace_class(exclude=["__init__", "_helper"])
        class TradingEngine:
            def execute_trade(self, symbol: str):
                pass
    """
    def decorator(cls: Type) -> Type:
        span_prefix = prefix or cls.__name__
        excluded = set(exclude or [])

        for attr_name in dir(cls):
            # Skip private/magic methods unless explicitly included
            if attr_name.startswith("_") and attr_name not in (methods or []):
                continue

            # Skip excluded methods
            if attr_name in excluded:
                continue

            # Only trace specified methods if provided
            if methods and attr_name not in methods:
                continue

            attr = getattr(cls, attr_name)

            # Only trace callable methods
            if not callable(attr):
                continue

            # Check if it's a method (not a classmethod or staticmethod)
            if isinstance(inspect.getattr_static(cls, attr_name), (staticmethod, classmethod)):
                continue

            # Apply tracing decorator
            span_name = f"{span_prefix}.{attr_name}"

            if asyncio.iscoroutinefunction(attr):
                traced_method = trace_async(
                    name=span_name,
                    capture_args=capture_args,
                )(attr)
            else:
                traced_method = trace(
                    name=span_name,
                    capture_args=capture_args,
                )(attr)

            setattr(cls, attr_name, traced_method)

        return cls

    return decorator


def _get_span_kind(kind: Optional[str]) -> Any:
    """Convert string span kind to SpanKind enum."""
    if not OTEL_AVAILABLE or kind is None:
        return SpanKind.INTERNAL if OTEL_AVAILABLE else None

    kind_map = {
        "internal": SpanKind.INTERNAL,
        "client": SpanKind.CLIENT,
        "server": SpanKind.SERVER,
        "producer": SpanKind.PRODUCER,
        "consumer": SpanKind.CONSUMER,
    }

    return kind_map.get(kind.lower(), SpanKind.INTERNAL)


def _capture_function_args(
    span: Any,
    func: Callable,
    args: tuple,
    kwargs: dict,
    capture_args: List[str],
):
    """Capture specified function arguments as span attributes."""
    try:
        # Get function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Bind arguments to parameters
        bound_args = {}
        for i, arg in enumerate(args):
            if i < len(params):
                bound_args[params[i]] = arg

        bound_args.update(kwargs)

        # Capture specified arguments
        for arg_name in capture_args:
            if arg_name in bound_args:
                value = bound_args[arg_name]
                # Convert to safe attribute value
                attr_value = _to_attribute_value(value)
                if attr_value is not None:
                    span.set_attribute(f"arg.{arg_name}", attr_value)

    except Exception as e:
        logger.debug(f"Error capturing function arguments: {e}")


def _capture_result(span: Any, result: Any):
    """Capture function result as span attribute."""
    try:
        attr_value = _to_attribute_value(result)
        if attr_value is not None:
            span.set_attribute("result", attr_value)
        else:
            span.set_attribute("result.type", type(result).__name__)
    except Exception as e:
        logger.debug(f"Error capturing result: {e}")


def _to_attribute_value(value: Any) -> Optional[Union[str, int, float, bool]]:
    """Convert a value to a valid span attribute value."""
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple)):
        # Convert to string representation for complex lists
        if len(value) <= 10 and all(isinstance(v, (str, int, float)) for v in value):
            return str(list(value))
        return f"[{len(value)} items]"

    if isinstance(value, dict):
        if len(value) <= 5:
            return str(value)
        return f"{{dict with {len(value)} keys}}"

    # For other types, use string representation
    try:
        str_val = str(value)
        if len(str_val) <= 200:
            return str_val
        return str_val[:197] + "..."
    except Exception:
        return type(value).__name__


# Convenience decorators for common use cases

def trace_db_operation(operation: str):
    """
    Decorator for database operations.

    Args:
        operation: Database operation name (e.g., "select", "insert", "update")

    Usage:
        @trace_db_operation("insert")
        def save_trade(trade: dict):
            pass
    """
    return trace(
        name=f"db.{operation}",
        kind="client",
        attributes={"db.system": "postgresql", "db.operation": operation},
    )


def trace_broker_operation(operation: str, broker: str = "unknown"):
    """
    Decorator for broker API operations.

    Args:
        operation: Broker operation name (e.g., "submit_order", "get_positions")
        broker: Broker name (e.g., "schwab", "ibkr")

    Usage:
        @trace_broker_operation("submit_order", broker="schwab")
        def submit_order(order: dict):
            pass
    """
    return trace(
        name=f"broker.{operation}",
        kind="client",
        attributes={"broker.name": broker, "broker.operation": operation},
    )


def trace_ml_operation(operation: str, model: str = "unknown"):
    """
    Decorator for ML operations.

    Args:
        operation: ML operation name (e.g., "predict", "train", "evaluate")
        model: Model name

    Usage:
        @trace_ml_operation("predict", model="xgboost")
        def predict(features: np.ndarray):
            pass
    """
    return trace(
        name=f"ml.{operation}",
        kind="internal",
        attributes={"ml.model": model, "ml.operation": operation},
    )


def trace_signal_operation(operation: str):
    """
    Decorator for signal processing operations.

    Args:
        operation: Signal operation name (e.g., "scan", "filter", "evaluate")

    Usage:
        @trace_signal_operation("scan")
        def scan_market():
            pass
    """
    return trace(
        name=f"signal.{operation}",
        kind="internal",
        attributes={"signal.operation": operation},
    )
