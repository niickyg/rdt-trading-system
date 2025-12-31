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
    max_risk_per_trade: float = Field(default=0.01, ge=0, le=1, alias="MAX_RISK_PER_TRADE")
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
    atr_period: int = Field(default=14, ge=1, le=200, alias="ATR_PERIOD")


class ScannerConfig(BaseSettings):
    """Market scanner configuration."""
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    scan_interval_seconds: int = Field(default=60, ge=10, le=3600, alias="SCAN_INTERVAL_SECONDS")
    min_volume: int = Field(default=500000, ge=0, alias="MIN_VOLUME")
    min_price: float = Field(default=5.0, ge=0, alias="MIN_PRICE")
    max_price: float = Field(default=500.0, ge=0, alias="MAX_PRICE")


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
    model_config = SettingsConfigDict(env_prefix="SCHWAB_", extra="ignore")

    app_key: str = Field(default="", alias="SCHWAB_APP_KEY")
    app_secret: str = Field(default="", alias="SCHWAB_APP_SECRET")
    callback_url: str = Field(default="https://localhost:8080", alias="SCHWAB_CALLBACK_URL")
    token_path: str = Field(default="./schwab_token.json", alias="SCHWAB_TOKEN_PATH")

    @property
    def is_configured(self) -> bool:
        return bool(self.app_key and self.app_secret)


class Settings(BaseSettings):
    """Main settings aggregating all configuration sections."""
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    trading: TradingConfig = Field(default_factory=TradingConfig)
    rrs: RRSConfig = Field(default_factory=RRSConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    alert: AlertConfig = Field(default_factory=AlertConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)

    def __init__(self, **kwargs):
        config_dir = Path(__file__).parent
        project_root = config_dir.parent
        env_file = project_root / ".env"

        if env_file.exists():
            kwargs.setdefault("_env_file", str(env_file))

        super().__init__(**kwargs)

        env_path = kwargs.get("_env_file", env_file if env_file.exists() else None)
        self.trading = TradingConfig(_env_file=env_path)
        self.rrs = RRSConfig(_env_file=env_path)
        self.scanner = ScannerConfig(_env_file=env_path)
        self.alert = AlertConfig(_env_file=env_path)
        self.broker = BrokerConfig(_env_file=env_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the singleton settings instance."""
    return Settings()


settings = get_settings()
