"""
Tracing Configuration for RDT Trading System

Provides configuration management for OpenTelemetry distributed tracing.
Supports environment variables for easy deployment configuration.

Environment Variables:
    OTEL_SERVICE_NAME: Service name for tracing (default: rdt-trading-system)
    OTEL_EXPORTER: Exporter type (jaeger, zipkin, console, otlp, none)
    OTEL_ENABLED: Enable/disable tracing (default: true)

    # Jaeger Configuration
    JAEGER_ENDPOINT: Jaeger collector endpoint (default: http://localhost:14268/api/traces)
    JAEGER_AGENT_HOST: Jaeger agent host for UDP (default: localhost)
    JAEGER_AGENT_PORT: Jaeger agent port (default: 6831)

    # Zipkin Configuration
    ZIPKIN_ENDPOINT: Zipkin collector endpoint (default: http://localhost:9411/api/v2/spans)

    # OTLP Configuration
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (default: http://localhost:4317)
    OTEL_EXPORTER_OTLP_HEADERS: OTLP headers (comma-separated key=value pairs)

    # Sampling Configuration
    OTEL_TRACES_SAMPLER: Sampler type (always_on, always_off, traceidratio, parentbased_always_on)
    OTEL_TRACES_SAMPLER_ARG: Sampler argument (e.g., ratio for traceidratio)

    # Resource Attributes
    OTEL_RESOURCE_ATTRIBUTES: Additional resource attributes (comma-separated key=value)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
from loguru import logger


class ExporterType(str, Enum):
    """Supported tracing exporters."""
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    CONSOLE = "console"
    OTLP = "otlp"
    NONE = "none"


class SamplerType(str, Enum):
    """Supported sampling strategies."""
    ALWAYS_ON = "always_on"
    ALWAYS_OFF = "always_off"
    TRACE_ID_RATIO = "traceidratio"
    PARENT_BASED_ALWAYS_ON = "parentbased_always_on"
    PARENT_BASED_ALWAYS_OFF = "parentbased_always_off"
    PARENT_BASED_TRACE_ID_RATIO = "parentbased_traceidratio"


@dataclass
class JaegerConfig:
    """Jaeger exporter configuration."""
    endpoint: str = "http://localhost:14268/api/traces"
    agent_host: str = "localhost"
    agent_port: int = 6831
    use_agent: bool = False  # Use UDP agent instead of HTTP collector

    @classmethod
    def from_env(cls) -> "JaegerConfig":
        """Create config from environment variables."""
        return cls(
            endpoint=os.environ.get("JAEGER_ENDPOINT", "http://localhost:14268/api/traces"),
            agent_host=os.environ.get("JAEGER_AGENT_HOST", "localhost"),
            agent_port=int(os.environ.get("JAEGER_AGENT_PORT", "6831")),
            use_agent=os.environ.get("JAEGER_USE_AGENT", "false").lower() == "true",
        )


@dataclass
class ZipkinConfig:
    """Zipkin exporter configuration."""
    endpoint: str = "http://localhost:9411/api/v2/spans"

    @classmethod
    def from_env(cls) -> "ZipkinConfig":
        """Create config from environment variables."""
        return cls(
            endpoint=os.environ.get("ZIPKIN_ENDPOINT", "http://localhost:9411/api/v2/spans"),
        )


@dataclass
class OTLPConfig:
    """OTLP exporter configuration."""
    endpoint: str = "http://localhost:4317"
    headers: Dict[str, str] = field(default_factory=dict)
    insecure: bool = True

    @classmethod
    def from_env(cls) -> "OTLPConfig":
        """Create config from environment variables."""
        headers = {}
        headers_str = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
        if headers_str:
            for pair in headers_str.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    headers[key.strip()] = value.strip()

        return cls(
            endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            headers=headers,
            insecure=os.environ.get("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true",
        )


@dataclass
class SamplingConfig:
    """Sampling configuration."""
    sampler: SamplerType = SamplerType.PARENT_BASED_ALWAYS_ON
    ratio: float = 1.0  # For ratio-based samplers

    @classmethod
    def from_env(cls) -> "SamplingConfig":
        """Create config from environment variables."""
        sampler_str = os.environ.get("OTEL_TRACES_SAMPLER", "parentbased_always_on")
        try:
            sampler = SamplerType(sampler_str.lower())
        except ValueError:
            logger.warning(f"Invalid sampler type: {sampler_str}, using parentbased_always_on")
            sampler = SamplerType.PARENT_BASED_ALWAYS_ON

        ratio_str = os.environ.get("OTEL_TRACES_SAMPLER_ARG", "1.0")
        try:
            ratio = float(ratio_str)
        except ValueError:
            logger.warning(f"Invalid sampler ratio: {ratio_str}, using 1.0")
            ratio = 1.0

        return cls(sampler=sampler, ratio=ratio)


@dataclass
class TracingConfig:
    """
    Main tracing configuration.

    Supports multiple exporters and configurable sampling strategies.
    All settings can be overridden via environment variables.

    Usage:
        # Default configuration from environment
        config = TracingConfig.from_env()

        # Custom configuration
        config = TracingConfig(
            service_name="my-service",
            exporter=ExporterType.JAEGER,
            enabled=True,
        )
    """

    # Service identification
    service_name: str = "rdt-trading-system"
    service_version: str = "1.0.0"
    service_namespace: str = "rdt"
    deployment_environment: str = "development"

    # Exporter configuration
    exporter: ExporterType = ExporterType.CONSOLE
    exporters: List[ExporterType] = field(default_factory=list)  # Support multiple exporters

    # Enable/disable tracing
    enabled: bool = True

    # Exporter-specific configs
    jaeger: JaegerConfig = field(default_factory=JaegerConfig)
    zipkin: ZipkinConfig = field(default_factory=ZipkinConfig)
    otlp: OTLPConfig = field(default_factory=OTLPConfig)

    # Sampling configuration
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    # Resource attributes
    resource_attributes: Dict[str, str] = field(default_factory=dict)

    # Instrumentation options
    instrument_flask: bool = True
    instrument_requests: bool = True
    instrument_sqlalchemy: bool = True
    instrument_redis: bool = True

    # Span attributes
    record_exception: bool = True
    set_status_on_exception: bool = True

    # Batch configuration
    batch_span_processor: bool = True
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    export_timeout_millis: int = 30000
    schedule_delay_millis: int = 5000

    @classmethod
    def from_env(cls) -> "TracingConfig":
        """Create configuration from environment variables."""
        # Parse exporter type
        exporter_str = os.environ.get("OTEL_EXPORTER", "console")
        try:
            exporter = ExporterType(exporter_str.lower())
        except ValueError:
            logger.warning(f"Invalid exporter type: {exporter_str}, using console")
            exporter = ExporterType.CONSOLE

        # Parse multiple exporters if specified
        exporters_str = os.environ.get("OTEL_EXPORTERS", "")
        exporters = []
        if exporters_str:
            for exp_str in exporters_str.split(","):
                try:
                    exporters.append(ExporterType(exp_str.strip().lower()))
                except ValueError:
                    logger.warning(f"Invalid exporter in list: {exp_str}")

        # Parse resource attributes
        resource_attrs = {}
        attrs_str = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
        if attrs_str:
            for pair in attrs_str.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    resource_attrs[key.strip()] = value.strip()

        return cls(
            service_name=os.environ.get("OTEL_SERVICE_NAME", "rdt-trading-system"),
            service_version=os.environ.get("OTEL_SERVICE_VERSION", "1.0.0"),
            service_namespace=os.environ.get("OTEL_SERVICE_NAMESPACE", "rdt"),
            deployment_environment=os.environ.get("OTEL_DEPLOYMENT_ENVIRONMENT",
                                                   os.environ.get("FLASK_ENV", "development")),
            exporter=exporter,
            exporters=exporters,
            enabled=os.environ.get("OTEL_ENABLED", "true").lower() == "true",
            jaeger=JaegerConfig.from_env(),
            zipkin=ZipkinConfig.from_env(),
            otlp=OTLPConfig.from_env(),
            sampling=SamplingConfig.from_env(),
            resource_attributes=resource_attrs,
            instrument_flask=os.environ.get("OTEL_INSTRUMENT_FLASK", "true").lower() == "true",
            instrument_requests=os.environ.get("OTEL_INSTRUMENT_REQUESTS", "true").lower() == "true",
            instrument_sqlalchemy=os.environ.get("OTEL_INSTRUMENT_SQLALCHEMY", "true").lower() == "true",
            instrument_redis=os.environ.get("OTEL_INSTRUMENT_REDIS", "true").lower() == "true",
            record_exception=os.environ.get("OTEL_RECORD_EXCEPTION", "true").lower() == "true",
            batch_span_processor=os.environ.get("OTEL_BATCH_PROCESSOR", "true").lower() == "true",
            max_queue_size=int(os.environ.get("OTEL_MAX_QUEUE_SIZE", "2048")),
            max_export_batch_size=int(os.environ.get("OTEL_MAX_EXPORT_BATCH_SIZE", "512")),
            export_timeout_millis=int(os.environ.get("OTEL_EXPORT_TIMEOUT_MILLIS", "30000")),
            schedule_delay_millis=int(os.environ.get("OTEL_SCHEDULE_DELAY_MILLIS", "5000")),
        )

    def get_all_exporters(self) -> List[ExporterType]:
        """Get all configured exporters."""
        if self.exporters:
            return self.exporters
        return [self.exporter]

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "service_name": self.service_name,
            "service_version": self.service_version,
            "service_namespace": self.service_namespace,
            "deployment_environment": self.deployment_environment,
            "exporter": self.exporter.value,
            "exporters": [e.value for e in self.exporters],
            "enabled": self.enabled,
            "jaeger": {
                "endpoint": self.jaeger.endpoint,
                "agent_host": self.jaeger.agent_host,
                "agent_port": self.jaeger.agent_port,
                "use_agent": self.jaeger.use_agent,
            },
            "zipkin": {
                "endpoint": self.zipkin.endpoint,
            },
            "otlp": {
                "endpoint": self.otlp.endpoint,
                "insecure": self.otlp.insecure,
            },
            "sampling": {
                "sampler": self.sampling.sampler.value,
                "ratio": self.sampling.ratio,
            },
            "resource_attributes": self.resource_attributes,
        }

    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"TracingConfig(service={self.service_name}, "
            f"exporter={self.exporter.value}, "
            f"enabled={self.enabled})"
        )
