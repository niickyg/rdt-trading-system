"""
Configuration settings for the RDT Trading System.
Uses Pydantic Settings for environment variable management.
"""

from __future__ import annotations
from enum import Enum
from functools import lru_cache
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.paths import get_project_root


class AlertMethod(str, Enum):
    DESKTOP = "desktop"
    TWILIO = "twilio"
    EMAIL = "email"
    NONE = "none"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class TradingConfig(BaseSettings):
    """Trading risk management configuration."""
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    account_size: float = Field(default=25000.0, ge=0, alias="ACCOUNT_SIZE")
    max_risk_per_trade: float = Field(default=0.015, ge=0, le=1, alias="MAX_RISK_PER_TRADE")  # Was 0.01 (1%)
    max_daily_loss: float = Field(default=0.03, ge=0, le=1, alias="MAX_DAILY_LOSS")
    max_position_size: float = Field(default=0.10, ge=0, le=1, alias="MAX_POSITION_SIZE")
    paper_trading: bool = Field(default=True, alias="PAPER_TRADING")
    auto_trade: bool = Field(default=False, alias="AUTO_TRADE")

    @property
    def max_risk_dollars(self) -> float:
        return self.account_size * self.max_risk_per_trade

    @property
    def max_daily_loss_dollars(self) -> float:
        return self.account_size * self.max_daily_loss


class RRSConfig(BaseSettings):
    """RRS indicator configuration."""
    model_config = SettingsConfigDict(env_prefix="RRS_", extra="ignore")

    strong_threshold: float = Field(default=2.0, alias="RRS_STRONG_THRESHOLD")
    moderate_threshold: float = Field(default=0.5, alias="RRS_MODERATE_THRESHOLD")
    weak_threshold: float = Field(default=-2.0, alias="RRS_WEAK_THRESHOLD")
    atr_period: int = Field(default=14, ge=1, le=200, alias="RRS_ATR_PERIOD")


class ScannerConfig(BaseSettings):
    """Market scanner configuration."""
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    scan_interval_seconds: int = Field(default=60, ge=5, le=3600, alias="SCAN_INTERVAL_SECONDS")
    min_volume: int = Field(default=500000, ge=0, alias="MIN_VOLUME")
    min_price: float = Field(default=5.0, ge=0, alias="MIN_PRICE")
    max_price: float = Field(default=500.0, ge=0, alias="MAX_PRICE")


class MTFConfig(BaseSettings):
    """Multi-Timeframe Analysis configuration."""
    model_config = SettingsConfigDict(env_prefix="MTF_", extra="ignore")

    # Enable/disable multi-timeframe analysis
    # Previous default: True (disabled for performance — MTF adds ~6 min to scan)
    enabled: bool = Field(
        default=False,
        alias="MTF_ENABLED",
        description="Enable multi-timeframe analysis in scanner"
    )

    # Timeframes to analyze (comma-separated)
    timeframes: str = Field(
        default="5m,15m,1h,4h,1d",
        alias="MTF_TIMEFRAMES",
        description="Comma-separated list of timeframes to analyze"
    )

    # Require timeframe alignment for signals
    alignment_required: bool = Field(
        default=False,
        alias="MTF_ALIGNMENT_REQUIRED",
        description="Only generate signals when timeframes are aligned"
    )

    # Minimum alignment percentage (0-1)
    min_alignment_pct: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        alias="MTF_MIN_ALIGNMENT_PCT",
        description="Minimum percentage of timeframes that must agree for alignment"
    )

    # Entry timing threshold for signal boost
    entry_timing_threshold: int = Field(
        default=70,
        ge=0,
        le=100,
        alias="MTF_ENTRY_TIMING_THRESHOLD",
        description="Entry timing score threshold for signal strength boost"
    )

    # Signal strength boost when aligned
    alignment_boost: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        alias="MTF_ALIGNMENT_BOOST",
        description="RRS boost when timeframes are aligned"
    )

    # Cache TTL for MTF data (seconds)
    cache_ttl: int = Field(
        default=60,
        ge=10,
        alias="MTF_CACHE_TTL",
        description="Cache time-to-live for MTF data in seconds"
    )

    @property
    def timeframe_list(self) -> list:
        """Get timeframes as a list."""
        return [tf.strip().lower() for tf in self.timeframes.split(",") if tf.strip()]

    @property
    def is_enabled(self) -> bool:
        """Check if MTF analysis is enabled."""
        return self.enabled


class IntradayConfig(BaseSettings):
    """Intraday (5-minute) analysis and exit management configuration."""
    model_config = SettingsConfigDict(env_prefix="INTRADAY_", extra="ignore")

    enabled: bool = Field(default=True, alias="INTRADAY_ENABLED")
    bars_5m_cache_ttl: int = Field(default=300, ge=30, alias="INTRADAY_BARS_5M_CACHE_TTL")
    rs_loss_threshold: float = Field(default=-0.5, alias="INTRADAY_RS_LOSS_THRESHOLD")
    vwap_confirm_bars: int = Field(default=2, ge=1, alias="INTRADAY_VWAP_CONFIRM_BARS")
    time_stop_minutes: int = Field(default=60, ge=5, alias="INTRADAY_TIME_STOP_MINUTES")
    breakeven_r_threshold: float = Field(default=0.5, ge=0.1, alias="INTRADAY_BREAKEVEN_R_THRESHOLD")
    skip_entry_rrs_threshold: float = Field(default=-1.0, alias="INTRADAY_SKIP_ENTRY_RRS_THRESHOLD")
    bar_refresh_interval: int = Field(default=300, ge=60, alias="INTRADAY_BAR_REFRESH_INTERVAL")


class AlertConfig(BaseSettings):
    """Alert notification configuration."""
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    method: AlertMethod = Field(default=AlertMethod.DESKTOP, alias="ALERT_METHOD")
    twilio_account_sid: str = Field(default="", alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field(default="", alias="TWILIO_AUTH_TOKEN")
    twilio_from_number: str = Field(default="", alias="TWILIO_FROM_NUMBER")
    twilio_to_number: str = Field(default="", alias="TWILIO_TO_NUMBER")
    email_from: str = Field(default="", alias="EMAIL_FROM")
    email_to: str = Field(default="", alias="EMAIL_TO")
    email_password: str = Field(default="", alias="EMAIL_PASSWORD")

    @property
    def is_twilio_configured(self) -> bool:
        return bool(self.twilio_account_sid and self.twilio_auth_token)


class BrokerConfig(BaseSettings):
    """Broker API configuration."""
    model_config = SettingsConfigDict(extra="ignore")

    # Broker type selection
    broker_type: str = Field(default="paper", alias="BROKER_TYPE")

    # Schwab settings
    schwab_app_key: str = Field(default="", alias="SCHWAB_APP_KEY")
    schwab_app_secret: str = Field(default="", alias="SCHWAB_APP_SECRET")
    schwab_callback_url: str = Field(default="https://localhost:8080", alias="SCHWAB_CALLBACK_URL")
    schwab_token_path: str = Field(default="./schwab_token.json", alias="SCHWAB_TOKEN_PATH")

    # IBKR settings
    ibkr_host: str = Field(default="127.0.0.1", alias="IBKR_HOST")
    ibkr_port: int = Field(default=4000, alias="IBKR_PORT")
    ibkr_client_id: int = Field(default=1, alias="IBKR_CLIENT_ID")
    ibkr_timeout: int = Field(default=20, alias="IBKR_TIMEOUT")

    # Backward compat aliases
    @property
    def app_key(self) -> str:
        return self.schwab_app_key

    @property
    def app_secret(self) -> str:
        return self.schwab_app_secret

    @property
    def is_configured(self) -> bool:
        return bool(self.schwab_app_key and self.schwab_app_secret)


class GPUConfig(BaseSettings):
    """GPU configuration for ML model training."""
    model_config = SettingsConfigDict(env_prefix="GPU_", extra="ignore")

    # GPU usage mode: 'auto', 'true'/'yes', 'false'/'no'
    use_gpu: str = Field(
        default="auto",
        alias="USE_GPU",
        description="GPU usage mode: auto (detect), true (force GPU), false (force CPU)"
    )

    # Memory limit: fraction (0-1) for percentage, or MB value
    memory_limit: float = Field(
        default=0.0,
        ge=0.0,
        alias="GPU_MEMORY_LIMIT",
        description="GPU memory limit: 0 for no limit, 0-1 for fraction, >1 for MB"
    )

    # Specific GPU device ID to use
    device_id: int = Field(
        default=-1,
        ge=-1,
        alias="GPU_DEVICE_ID",
        description="GPU device ID to use (-1 for auto-select)"
    )

    # Enable memory growth to prevent OOM
    memory_growth: bool = Field(
        default=True,
        alias="GPU_MEMORY_GROWTH",
        description="Enable GPU memory growth to prevent OOM errors"
    )

    # Enable mixed precision training (float16)
    mixed_precision: bool = Field(
        default=False,
        alias="GPU_MIXED_PRECISION",
        description="Enable mixed precision (float16) training for faster performance"
    )

    # Multi-GPU training strategy
    multi_gpu_strategy: str = Field(
        default="mirrored",
        alias="GPU_MULTI_GPU_STRATEGY",
        description="Multi-GPU strategy: mirrored, multi_worker, parameter_server"
    )

    # Log device placement for debugging
    log_device_placement: bool = Field(
        default=False,
        alias="GPU_LOG_DEVICE_PLACEMENT",
        description="Log TensorFlow device placement for debugging"
    )

    @property
    def should_use_gpu(self) -> bool:
        """Check if GPU should be used based on configuration."""
        return self.use_gpu.lower() in ('auto', 'true', 'yes', '1', 'gpu')

    @property
    def gpu_device_id(self) -> int | None:
        """Get GPU device ID or None for auto-select."""
        return None if self.device_id < 0 else self.device_id

    @property
    def gpu_memory_limit(self) -> float | None:
        """Get GPU memory limit or None for no limit."""
        return None if self.memory_limit == 0 else self.memory_limit


class DataProviderConfig(BaseSettings):
    """Data provider configuration for redundant data fetching."""
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # Provider priority (comma-separated list, first = highest priority)
    provider_priority: str = Field(
        default="ibkr",
        alias="DATA_PROVIDER_PRIORITY",
        description="Comma-separated list of providers in priority order"
    )

    # Enable/disable redundant providers
    use_providers: bool = Field(
        default=True,
        alias="USE_DATA_PROVIDERS",
        description="Enable redundant data providers with fallback"
    )

    # Alpha Vantage configuration
    alpha_vantage_api_key: str = Field(default="", alias="ALPHA_VANTAGE_API_KEY")
    alpha_vantage_premium: bool = Field(default=False, alias="ALPHA_VANTAGE_PREMIUM")
    alpha_vantage_daily_limit: int = Field(default=25, ge=1, alias="ALPHA_VANTAGE_DAILY_LIMIT")

    # Cache configuration
    cache_ttl_seconds: float = Field(
        default=30.0,
        ge=1.0,
        alias="DATA_CACHE_TTL",
        description="Cache time-to-live in seconds"
    )
    cache_max_size: int = Field(
        default=1000,
        ge=100,
        alias="DATA_CACHE_MAX_SIZE",
        description="Maximum cache entries"
    )

    # Circuit breaker configuration
    circuit_failure_threshold: int = Field(
        default=5,
        ge=1,
        alias="CIRCUIT_FAILURE_THRESHOLD",
        description="Failures before circuit opens"
    )
    circuit_recovery_timeout: float = Field(
        default=60.0,
        ge=10.0,
        alias="CIRCUIT_RECOVERY_TIMEOUT",
        description="Seconds before testing failed provider"
    )

    @property
    def is_alpha_vantage_configured(self) -> bool:
        """Check if Alpha Vantage API key is configured."""
        return bool(self.alpha_vantage_api_key)

    @property
    def provider_list(self) -> list:
        """Get provider priority as a list."""
        return [p.strip().lower() for p in self.provider_priority.split(",") if p.strip()]


class Settings(BaseSettings):
    """Main settings aggregating all configuration sections."""
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=True
    )

    trading: TradingConfig = Field(default_factory=TradingConfig)
    rrs: RRSConfig = Field(default_factory=RRSConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    mtf: MTFConfig = Field(default_factory=MTFConfig)
    intraday: IntradayConfig = Field(default_factory=IntradayConfig)
    alert: AlertConfig = Field(default_factory=AlertConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    data_provider: DataProviderConfig = Field(default_factory=DataProviderConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    def __init__(self, **kwargs):
        project_root = get_project_root()
        env_file = project_root / ".env"

        if env_file.exists():
            kwargs.setdefault("_env_file", str(env_file))

        super().__init__(**kwargs)

        env_path = kwargs.get("_env_file", env_file if env_file.exists() else None)
        self.trading = TradingConfig(_env_file=env_path)
        self.rrs = RRSConfig(_env_file=env_path)
        self.scanner = ScannerConfig(_env_file=env_path)
        self.mtf = MTFConfig(_env_file=env_path)
        self.intraday = IntradayConfig(_env_file=env_path)
        self.alert = AlertConfig(_env_file=env_path)
        self.broker = BrokerConfig(_env_file=env_path)
        self.data_provider = DataProviderConfig(_env_file=env_path)
        self.gpu = GPUConfig(_env_file=env_path)

        # Options config (lazy import to avoid circular dependency)
        self._options = None
        try:
            from options.config import OptionsConfig
            self._options = OptionsConfig(_env_file=env_path)
        except ImportError:
            pass

    @property
    def options(self):
        """Options trading configuration (None if module not installed)."""
        return self._options


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the singleton settings instance."""
    return Settings()


settings = get_settings()
