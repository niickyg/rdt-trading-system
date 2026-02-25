"""
Options Trading Configuration for the RDT Trading System.

Pydantic-based settings loaded from environment variables with OPTIONS_ prefix.
"""

from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OptionsMode(str, Enum):
    STOCKS = "stocks"       # Stock trading only (default)
    OPTIONS = "options"     # Options trading only
    BOTH = "both"           # Both stocks and options


class OptionsConfig(BaseSettings):
    """Options trading configuration."""
    model_config = SettingsConfigDict(env_prefix="OPTIONS_", extra="ignore")

    # Master switch
    enabled: bool = Field(
        default=False,
        alias="OPTIONS_ENABLED",
        description="Enable options trading module"
    )
    mode: OptionsMode = Field(
        default=OptionsMode.STOCKS,
        alias="OPTIONS_MODE",
        description="Trading mode: stocks, options, or both"
    )

    # Delta targeting
    long_delta_min: float = Field(default=0.50, alias="OPTIONS_LONG_DELTA_MIN")
    long_delta_max: float = Field(default=0.70, alias="OPTIONS_LONG_DELTA_MAX")
    long_delta_target: float = Field(default=0.60, alias="OPTIONS_LONG_DELTA_TARGET")
    short_leg_delta_min: float = Field(default=0.25, alias="OPTIONS_SHORT_LEG_DELTA_MIN")
    short_leg_delta_max: float = Field(default=0.35, alias="OPTIONS_SHORT_LEG_DELTA_MAX")
    short_leg_delta_target: float = Field(default=0.30, alias="OPTIONS_SHORT_LEG_DELTA_TARGET")
    delta_tolerance: float = Field(default=0.05, alias="OPTIONS_DELTA_TOLERANCE")

    # DTE (days to expiration) settings
    dte_min: int = Field(default=14, alias="OPTIONS_DTE_MIN")
    dte_max: int = Field(default=60, alias="OPTIONS_DTE_MAX")
    dte_target: int = Field(default=35, alias="OPTIONS_DTE_TARGET")

    # Exit management
    profit_target_pct: float = Field(
        default=0.50, alias="OPTIONS_PROFIT_TARGET_PCT",
        description="Close at 50% of max profit for spreads"
    )
    long_option_profit_target_pct: float = Field(
        default=1.00, alias="OPTIONS_LONG_PROFIT_TARGET_PCT",
        description="Close at 100% gain for long options"
    )
    stop_loss_pct: float = Field(
        default=0.50, alias="OPTIONS_STOP_LOSS_PCT",
        description="Stop loss at 50% premium loss"
    )
    time_stop_dte: int = Field(
        default=14, alias="OPTIONS_TIME_STOP_DTE",
        description="Close when DTE drops below this"
    )
    delta_breach_threshold: float = Field(
        default=0.80, alias="OPTIONS_DELTA_BREACH",
        description="Close when net delta exceeds this (deep ITM)"
    )
    iv_crush_threshold: float = Field(
        default=0.20, alias="OPTIONS_IV_CRUSH_THRESHOLD",
        description="Close when IV drops more than 20% from entry"
    )
    roll_dte_threshold: int = Field(
        default=21, alias="OPTIONS_ROLL_DTE",
        description="Consider rolling when DTE < this and profitable"
    )

    # IV thresholds for strategy selection
    iv_rank_low: float = Field(
        default=30.0, alias="OPTIONS_IV_RANK_LOW",
        description="Below this = low IV regime"
    )
    iv_rank_high: float = Field(
        default=50.0, alias="OPTIONS_IV_RANK_HIGH",
        description="Above this = high IV regime"
    )
    iv_rank_very_high: float = Field(
        default=80.0, alias="OPTIONS_IV_RANK_VERY_HIGH",
        description="Above this = very high IV regime (iron condors)"
    )

    # Liquidity filters
    min_open_interest: int = Field(default=100, alias="OPTIONS_MIN_OI")
    max_bid_ask_spread_pct: float = Field(
        default=10.0, alias="OPTIONS_MAX_SPREAD_PCT",
        description="Max bid-ask spread as % of mid price"
    )

    # Spread width limits
    max_spread_width_under_100: float = Field(
        default=5.0, alias="OPTIONS_MAX_SPREAD_WIDTH_UNDER_100",
        description="Max spread width for stocks under $100"
    )
    max_spread_width_over_100: float = Field(
        default=10.0, alias="OPTIONS_MAX_SPREAD_WIDTH_OVER_100",
        description="Max spread width for stocks over $100"
    )

    # Portfolio risk limits
    max_premium_at_risk_pct: float = Field(
        default=0.10, alias="OPTIONS_MAX_PREMIUM_RISK_PCT",
        description="Max total premium at risk as % of account"
    )
    max_portfolio_delta: float = Field(
        default=200.0, alias="OPTIONS_MAX_PORTFOLIO_DELTA",
        description="Max absolute net portfolio delta"
    )
    max_daily_theta_pct: float = Field(
        default=0.005, alias="OPTIONS_MAX_DAILY_THETA_PCT",
        description="Max daily theta as % of account"
    )
    max_positions_per_underlying: int = Field(
        default=2, alias="OPTIONS_MAX_PER_UNDERLYING",
        description="Max options positions per underlying"
    )

    # Cache settings
    greeks_cache_ttl: int = Field(default=60, alias="OPTIONS_GREEKS_CACHE_TTL")
    chain_cache_ttl: int = Field(default=300, alias="OPTIONS_CHAIN_CACHE_TTL")
    rate_limit_ms: int = Field(default=50, alias="OPTIONS_RATE_LIMIT_MS")

    # Order execution
    slippage_ticks: int = Field(
        default=1, alias="OPTIONS_SLIPPAGE_TICKS",
        description="Ticks of slippage for limit order pricing"
    )

    # Force a specific strategy (empty = auto-select)
    force_strategy: str = Field(
        default="", alias="OPTIONS_FORCE_STRATEGY",
        description="Force a specific strategy (e.g., 'long_call', 'bull_call_spread')"
    )

    # Paper trading settings
    risk_free_rate: float = Field(
        default=0.05, alias="OPTIONS_RISK_FREE_RATE",
        description="Risk-free rate for Black-Scholes pricing (paper mode)"
    )
    paper_iv_multiplier: float = Field(
        default=1.1, alias="OPTIONS_PAPER_IV_MULTIPLIER",
        description="HV-to-IV scaling factor for synthetic IV (paper mode)"
    )

    @property
    def is_options_enabled(self) -> bool:
        return self.enabled and self.mode in (OptionsMode.OPTIONS, OptionsMode.BOTH)

    @property
    def is_stocks_enabled(self) -> bool:
        return self.mode in (OptionsMode.STOCKS, OptionsMode.BOTH)

    def get_max_spread_width(self, stock_price: float) -> float:
        if stock_price >= 100:
            return self.max_spread_width_over_100
        return self.max_spread_width_under_100
