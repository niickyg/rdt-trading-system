"""
OpenTelemetry Tracer Initialization for RDT Trading System

Provides tracer initialization and management with support for:
- Multiple exporters (Jaeger, Zipkin, Console, OTLP)
- Configurable sampling strategies
- Context propagation helpers
- Graceful shutdown

Usage:
    from tracing import init_tracing, get_tracer

    # Initialize at application startup
    init_tracing(service_name="my-service")

    # Get tracer for manual instrumentation
    tracer = get_tracer()
    with tracer.start_as_current_span("my-operation") as span:
        span.set_attribute("key", "value")
        # ... do work
"""

from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from loguru import logger

from tracing.config import TracingConfig, ExporterType, SamplerType

# OpenTelemetry imports with graceful fallback
OTEL_AVAILABLE = False
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider, Span
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.trace import (
        Status,
        StatusCode,
        SpanKind,
        Tracer,
        get_current_span as otel_get_current_span,
    )
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.propagate import set_global_textmap, get_global_textmap, inject, extract
    from opentelemetry.context import Context, get_current, attach, detach
    from opentelemetry.sdk.trace.sampling import (
        Sampler,
        ALWAYS_ON,
        ALWAYS_OFF,
        TraceIdRatioBased,
        ParentBased,
    )
    OTEL_AVAILABLE = True
    logger.info("OpenTelemetry SDK available")
except ImportError as e:
    logger.warning(f"OpenTelemetry not available: {e}. Tracing will be disabled.")
    trace = None
    Tracer = None
    Span = None
    SpanKind = None
    Status = None
    StatusCode = None

# Optional exporter imports
JAEGER_AVAILABLE = False
ZIPKIN_AVAILABLE = False
OTLP_AVAILABLE = False

if OTEL_AVAILABLE:
    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        JAEGER_AVAILABLE = True
    except ImportError:
        logger.debug("Jaeger exporter not available")

    try:
        from opentelemetry.exporter.zipkin.json import ZipkinExporter
        ZIPKIN_AVAILABLE = True
    except ImportError:
        logger.debug("Zipkin exporter not available")

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        OTLP_AVAILABLE = True
    except ImportError:
        logger.debug("OTLP exporter not available")


# Global state
_tracer_provider: Optional["TracerProvider"] = None
_config: Optional[TracingConfig] = None
_initialized: bool = False


class NoOpSpan:
    """No-op span for when tracing is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def set_attribute(self, key: str, value: Any) -> "NoOpSpan":
        return self

    def set_attributes(self, attributes: Dict[str, Any]) -> "NoOpSpan":
        return self

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "NoOpSpan":
        return self

    def record_exception(self, exception: Exception, attributes: Optional[Dict[str, Any]] = None):
        pass

    def set_status(self, status: Any, description: Optional[str] = None):
        pass

    def is_recording(self) -> bool:
        return False

    def get_span_context(self):
        return None

    @property
    def name(self) -> str:
        return "noop"


class NoOpTracer:
    """No-op tracer for when tracing is disabled."""

    def start_span(
        self,
        name: str,
        context: Optional[Any] = None,
        kind: Optional[Any] = None,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[Any] = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> NoOpSpan:
        return NoOpSpan()

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Optional[Any] = None,
        kind: Optional[Any] = None,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[Any] = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ):
        yield NoOpSpan()


def _create_sampler(config: TracingConfig) -> Optional["Sampler"]:
    """Create sampler based on configuration."""
    if not OTEL_AVAILABLE:
        return None

    sampling = config.sampling

    if sampling.sampler == SamplerType.ALWAYS_ON:
        return ALWAYS_ON
    elif sampling.sampler == SamplerType.ALWAYS_OFF:
        return ALWAYS_OFF
    elif sampling.sampler == SamplerType.TRACE_ID_RATIO:
        return TraceIdRatioBased(sampling.ratio)
    elif sampling.sampler == SamplerType.PARENT_BASED_ALWAYS_ON:
        return ParentBased(ALWAYS_ON)
    elif sampling.sampler == SamplerType.PARENT_BASED_ALWAYS_OFF:
        return ParentBased(ALWAYS_OFF)
    elif sampling.sampler == SamplerType.PARENT_BASED_TRACE_ID_RATIO:
        return ParentBased(TraceIdRatioBased(sampling.ratio))
    else:
        return ParentBased(ALWAYS_ON)


def _create_exporter(exporter_type: ExporterType, config: TracingConfig):
    """Create span exporter based on type."""
    if not OTEL_AVAILABLE:
        return None

    if exporter_type == ExporterType.CONSOLE:
        return ConsoleSpanExporter()

    elif exporter_type == ExporterType.JAEGER:
        if not JAEGER_AVAILABLE:
            logger.warning("Jaeger exporter not installed. Install with: pip install opentelemetry-exporter-jaeger")
            return ConsoleSpanExporter()

        jaeger_config = config.jaeger
        if jaeger_config.use_agent:
            return JaegerExporter(
                agent_host_name=jaeger_config.agent_host,
                agent_port=jaeger_config.agent_port,
            )
        else:
            return JaegerExporter(
                collector_endpoint=jaeger_config.endpoint,
            )

    elif exporter_type == ExporterType.ZIPKIN:
        if not ZIPKIN_AVAILABLE:
            logger.warning("Zipkin exporter not installed. Install with: pip install opentelemetry-exporter-zipkin")
            return ConsoleSpanExporter()

        return ZipkinExporter(endpoint=config.zipkin.endpoint)

    elif exporter_type == ExporterType.OTLP:
        if not OTLP_AVAILABLE:
            logger.warning("OTLP exporter not installed. Install with: pip install opentelemetry-exporter-otlp")
            return ConsoleSpanExporter()

        otlp_config = config.otlp
        return OTLPSpanExporter(
            endpoint=otlp_config.endpoint,
            insecure=otlp_config.insecure,
            headers=otlp_config.headers or None,
        )

    elif exporter_type == ExporterType.NONE:
        return None

    else:
        logger.warning(f"Unknown exporter type: {exporter_type}, using console")
        return ConsoleSpanExporter()


def init_tracing(
    service_name: Optional[str] = None,
    config: Optional[TracingConfig] = None,
) -> bool:
    """
    Initialize OpenTelemetry tracing.

    Args:
        service_name: Override service name from config
        config: TracingConfig instance (uses environment if not provided)

    Returns:
        True if initialization succeeded, False otherwise
    """
    global _tracer_provider, _config, _initialized

    if _initialized:
        logger.debug("Tracing already initialized")
        return True

    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available, tracing disabled")
        _initialized = True
        return False

    # Load configuration
    if config is None:
        config = TracingConfig.from_env()

    if service_name:
        config.service_name = service_name

    _config = config

    if not config.enabled:
        logger.info("Tracing disabled by configuration")
        _initialized = True
        return False

    logger.info(f"Initializing tracing: {config}")

    try:
        # Create resource with service information
        resource_attrs = {
            SERVICE_NAME: config.service_name,
            SERVICE_VERSION: config.service_version,
            "service.namespace": config.service_namespace,
            "deployment.environment": config.deployment_environment,
        }
        resource_attrs.update(config.resource_attributes)

        resource = Resource.create(resource_attrs)

        # Create sampler
        sampler = _create_sampler(config)

        # Create tracer provider
        _tracer_provider = TracerProvider(
            resource=resource,
            sampler=sampler,
        )

        # Add exporters
        for exporter_type in config.get_all_exporters():
            exporter = _create_exporter(exporter_type, config)
            if exporter is not None:
                if config.batch_span_processor:
                    processor = BatchSpanProcessor(
                        exporter,
                        max_queue_size=config.max_queue_size,
                        max_export_batch_size=config.max_export_batch_size,
                        export_timeout_millis=config.export_timeout_millis,
                        schedule_delay_millis=config.schedule_delay_millis,
                    )
                else:
                    processor = SimpleSpanProcessor(exporter)

                _tracer_provider.add_span_processor(processor)
                logger.info(f"Added {exporter_type.value} exporter")

        # Set as global tracer provider
        trace.set_tracer_provider(_tracer_provider)

        # Set up context propagation
        set_global_textmap(TraceContextTextMapPropagator())

        _initialized = True
        logger.info("Tracing initialization complete")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize tracing: {e}")
        _initialized = True  # Mark as initialized to prevent retry
        return False


def shutdown_tracing(timeout_millis: int = 30000) -> bool:
    """
    Shutdown tracing and flush pending spans.

    Args:
        timeout_millis: Timeout for flushing pending spans

    Returns:
        True if shutdown succeeded
    """
    global _tracer_provider, _initialized

    if not _initialized or _tracer_provider is None:
        return True

    try:
        logger.info("Shutting down tracing...")
        _tracer_provider.shutdown()
        _tracer_provider = None
        _initialized = False
        logger.info("Tracing shutdown complete")
        return True
    except Exception as e:
        logger.error(f"Error during tracing shutdown: {e}")
        return False


def get_tracer(
    name: Optional[str] = None,
    version: Optional[str] = None,
) -> Any:
    """
    Get a tracer instance.

    Args:
        name: Instrumenting module name (defaults to service name)
        version: Instrumenting module version

    Returns:
        Tracer instance (or NoOpTracer if tracing disabled)
    """
    if not OTEL_AVAILABLE or not _initialized or _tracer_provider is None:
        return NoOpTracer()

    if name is None and _config is not None:
        name = _config.service_name

    return trace.get_tracer(
        instrumenting_module_name=name or "rdt-trading-system",
        instrumenting_library_version=version,
    )


def get_current_span() -> Any:
    """
    Get the current active span.

    Returns:
        Current span or NoOpSpan if none active
    """
    if not OTEL_AVAILABLE or not _initialized:
        return NoOpSpan()

    span = otel_get_current_span()
    if span is None or not span.is_recording():
        return NoOpSpan()
    return span


def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace ID as hex string.

    Returns:
        Trace ID string or None if no active span
    """
    if not OTEL_AVAILABLE or not _initialized:
        return None

    span = otel_get_current_span()
    if span is None:
        return None

    ctx = span.get_span_context()
    if ctx is None or not ctx.is_valid:
        return None

    return format(ctx.trace_id, "032x")


def get_current_span_id() -> Optional[str]:
    """
    Get the current span ID as hex string.

    Returns:
        Span ID string or None if no active span
    """
    if not OTEL_AVAILABLE or not _initialized:
        return None

    span = otel_get_current_span()
    if span is None:
        return None

    ctx = span.get_span_context()
    if ctx is None or not ctx.is_valid:
        return None

    return format(ctx.span_id, "016x")


def create_span_context(
    trace_id: str,
    span_id: str,
    is_remote: bool = True,
) -> Optional[Any]:
    """
    Create a span context from trace/span IDs.

    Args:
        trace_id: Trace ID as hex string
        span_id: Span ID as hex string
        is_remote: Whether this context is from a remote parent

    Returns:
        SpanContext or None if invalid
    """
    if not OTEL_AVAILABLE:
        return None

    try:
        from opentelemetry.trace import SpanContext, TraceFlags

        return SpanContext(
            trace_id=int(trace_id, 16),
            span_id=int(span_id, 16),
            is_remote=is_remote,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
    except Exception as e:
        logger.debug(f"Failed to create span context: {e}")
        return None


def inject_context(carrier: Dict[str, str]) -> Dict[str, str]:
    """
    Inject trace context into a carrier (e.g., HTTP headers).

    Args:
        carrier: Dictionary to inject context into

    Returns:
        Carrier with injected context
    """
    if not OTEL_AVAILABLE or not _initialized:
        return carrier

    inject(carrier)
    return carrier


def extract_context(carrier: Dict[str, str]) -> Any:
    """
    Extract trace context from a carrier (e.g., HTTP headers).

    Args:
        carrier: Dictionary containing trace context

    Returns:
        Context object
    """
    if not OTEL_AVAILABLE or not _initialized:
        return None

    return extract(carrier)


@contextmanager
def span_in_context(span: Any):
    """
    Context manager to set a span as current.

    Args:
        span: Span to set as current

    Yields:
        The span
    """
    if not OTEL_AVAILABLE or not _initialized:
        yield span
        return

    from opentelemetry.trace import use_span
    with use_span(span, end_on_exit=False):
        yield span


def add_span_attribute(key: str, value: Any):
    """
    Add an attribute to the current span.

    Args:
        key: Attribute name
        value: Attribute value
    """
    span = get_current_span()
    if isinstance(span, NoOpSpan):
        return
    span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Add an event to the current span.

    Args:
        name: Event name
        attributes: Optional event attributes
    """
    span = get_current_span()
    if isinstance(span, NoOpSpan):
        return
    span.add_event(name, attributes)


def record_exception(
    exception: Exception,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Record an exception in the current span.

    Args:
        exception: The exception to record
        attributes: Optional additional attributes
    """
    span = get_current_span()
    if isinstance(span, NoOpSpan):
        return
    span.record_exception(exception, attributes)


def set_span_status(
    status_code: str,
    description: Optional[str] = None,
):
    """
    Set the status of the current span.

    Args:
        status_code: "OK", "ERROR", or "UNSET"
        description: Optional status description
    """
    if not OTEL_AVAILABLE or not _initialized:
        return

    span = get_current_span()
    if isinstance(span, NoOpSpan):
        return

    code_map = {
        "OK": StatusCode.OK,
        "ERROR": StatusCode.ERROR,
        "UNSET": StatusCode.UNSET,
    }
    code = code_map.get(status_code.upper(), StatusCode.UNSET)
    span.set_status(Status(code, description))


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled and initialized."""
    return OTEL_AVAILABLE and _initialized and _config is not None and _config.enabled


def get_config() -> Optional[TracingConfig]:
    """Get the current tracing configuration."""
    return _config
