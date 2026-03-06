"""
SQLAlchemy 2.0 models for the RDT Trading System.

All models are designed for PostgreSQL compatibility while maintaining
SQLite support for development and testing.
"""

from datetime import datetime, date
from enum import Enum
from typing import Optional

from sqlalchemy import (
    String, Integer, Float, DateTime, Date, Boolean, Text,
    Enum as SQLEnum, Index, CheckConstraint, Numeric, BigInteger,
    UniqueConstraint, ForeignKey,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# =============================================================================
# Enums
# =============================================================================

class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class ExitReason(str, Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    MANUAL = "manual"
    EOD = "end_of_day"
    STALE_RECONCILED = "stale_reconciled"
    INTRADAY_RS_LOSS = "intraday_rs_loss"
    INTRADAY_VWAP = "intraday_vwap"
    INTRADAY_TIME_STOP = "intraday_time_stop"
    INTRADAY_BREAKEVEN = "intraday_breakeven"
    INTRADAY_OTHER = "intraday_other"


class SignalStatus(str, Enum):
    PENDING = "pending"
    TRIGGERED = "triggered"
    EXPIRED = "expired"
    IGNORED = "ignored"


class SubscriptionTierEnum(str, Enum):
    """Subscription tiers for API access"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ELITE = "elite"


class OrderExecutionStatus(str, Enum):
    """Order execution status for tracking fill quality."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SubscriptionStatus(str, Enum):
    """Subscription status for billing."""
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    UNPAID = "unpaid"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    PAUSED = "paused"


# =============================================================================
# Core Trading Models
# =============================================================================

class Trade(Base):
    """Trade records for completed and ongoing trades."""
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    direction: Mapped[str] = mapped_column(SQLEnum(TradeDirection, native_enum=False), nullable=False)
    entry_price: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    exit_price: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    shares: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    pnl: Mapped[Optional[float]] = mapped_column(Numeric(12, 2), nullable=True)
    pnl_percent: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    rrs_at_entry: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    stop_loss: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    take_profit: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    status: Mapped[str] = mapped_column(
        SQLEnum(TradeStatus, native_enum=False),
        default=TradeStatus.OPEN,
        nullable=False
    )
    exit_reason: Mapped[Optional[str]] = mapped_column(
        SQLEnum(ExitReason, native_enum=False),
        nullable=True
    )
    broker_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # MFE/MAE tracking (finalized on POSITION_CLOSED)
    peak_mfe: Mapped[Optional[float]] = mapped_column(Numeric(12, 2), nullable=True)
    peak_mae: Mapped[Optional[float]] = mapped_column(Numeric(12, 2), nullable=True)
    peak_mfe_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    peak_mae_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    peak_mfe_r: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    peak_mae_r: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    bars_to_mfe: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bars_held: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Filter evaluation metadata (for forward-test data collection)
    vix_regime: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    vix_value: Mapped[Optional[float]] = mapped_column(Numeric(8, 2), nullable=True)
    market_regime: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    sector_name: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    sector_rs: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    spy_trend: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    ml_confidence: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    signal_strategy: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    news_sentiment: Mapped[Optional[float]] = mapped_column(Numeric(6, 3), nullable=True)
    news_warning: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    regime_rrs_threshold: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    regime_stop_multiplier: Mapped[Optional[float]] = mapped_column(Numeric(6, 3), nullable=True)
    regime_target_multiplier: Mapped[Optional[float]] = mapped_column(Numeric(6, 3), nullable=True)
    vix_position_size_mult: Mapped[Optional[float]] = mapped_column(Numeric(6, 3), nullable=True)
    sector_boost: Mapped[Optional[float]] = mapped_column(Numeric(6, 3), nullable=True)
    first_hour_filtered: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True, default=False)
    strategy_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, default='rrs_momentum')

    __table_args__ = (
        Index("ix_trades_symbol", "symbol"),
        Index("ix_trades_entry_time", "entry_time"),
        Index("ix_trades_status", "status"),
        Index("ix_trades_symbol_entry_time", "symbol", "entry_time"),
        Index("ix_trades_symbol_status", "symbol", "status"),
        Index("ix_trades_market_regime", "market_regime"),
        Index("ix_trades_vix_regime", "vix_regime"),
        Index("ix_trades_strategy_name", "strategy_name"),
        CheckConstraint("shares > 0", name="ck_trades_shares_positive"),
        CheckConstraint("entry_price > 0", name="ck_trades_entry_price_positive"),
    )


class Position(Base):
    """Current open positions."""
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, unique=True)
    direction: Mapped[str] = mapped_column(SQLEnum(TradeDirection, native_enum=False), nullable=False)
    entry_price: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    shares: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    stop_loss: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    take_profit: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    current_price: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    unrealized_pnl: Mapped[Optional[float]] = mapped_column(Numeric(12, 2), nullable=True)
    rrs_at_entry: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    strategy_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, default='rrs_momentum')
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        onupdate=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_positions_symbol", "symbol"),
        Index("ix_positions_strategy_name", "strategy_name"),
        CheckConstraint("shares > 0", name="ck_positions_shares_positive"),
    )


class OptionsPosition(Base):
    """Current open options positions (paper trading)."""
    __tablename__ = "options_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, unique=True)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # long/short/neutral
    contracts: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    entry_premium: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False, default=0.0)
    total_premium: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False, default=0.0)
    entry_iv: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    entry_delta: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    order_ids: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    legs_json: Mapped[str] = mapped_column(Text, nullable=False)  # JSON: contracts + greeks
    fill_details_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    current_premium: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    unrealized_pnl: Mapped[Optional[float]] = mapped_column(Numeric(14, 2), nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, onupdate=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_options_positions_symbol", "symbol"),
        CheckConstraint("contracts > 0", name="ck_options_positions_contracts_positive"),
    )


class OptionsTrade(Base):
    """Historical options trades (paper trading)."""
    __tablename__ = "options_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)
    contracts: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    entry_premium: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False, default=0.0)
    total_premium: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False, default=0.0)
    exit_premium: Mapped[Optional[float]] = mapped_column(Numeric(14, 2), nullable=True)
    pnl: Mapped[Optional[float]] = mapped_column(Numeric(12, 2), nullable=True)
    pnl_percent: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    entry_iv: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    exit_iv: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    entry_delta: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    legs_json: Mapped[str] = mapped_column(Text, nullable=False)
    fill_details_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    exit_reason: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    status: Mapped[str] = mapped_column(
        SQLEnum(TradeStatus, native_enum=False), default=TradeStatus.OPEN, nullable=False
    )
    order_ids: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array

    __table_args__ = (
        Index("ix_options_trades_symbol", "symbol"),
        Index("ix_options_trades_entry_time", "entry_time"),
        Index("ix_options_trades_status", "status"),
        Index("ix_options_trades_symbol_status", "symbol", "status"),
        CheckConstraint("contracts > 0", name="ck_options_trades_contracts_positive"),
    )


class Signal(Base):
    """Trading signals from the scanner."""
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow
    )
    rrs: Mapped[float] = mapped_column(Numeric(8, 4), nullable=False)
    status: Mapped[str] = mapped_column(
        SQLEnum(SignalStatus, native_enum=False),
        default=SignalStatus.PENDING,
        nullable=False
    )
    direction: Mapped[str] = mapped_column(SQLEnum(TradeDirection, native_enum=False), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    atr: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    daily_strong: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    daily_weak: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    volume: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    market_regime: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    strategy_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, default='rrs_momentum')

    __table_args__ = (
        Index("ix_signals_symbol", "symbol"),
        Index("ix_signals_timestamp", "timestamp"),
        Index("ix_signals_status", "status"),
        Index("ix_signals_symbol_timestamp", "symbol", "timestamp"),
        Index("ix_signals_timestamp_status", "timestamp", "status"),
        Index("ix_signals_strategy_name", "strategy_name"),
    )


class DailyStats(Base):
    """Daily trading statistics and P&L."""
    __tablename__ = "daily_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, unique=True)
    starting_balance: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    ending_balance: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    pnl: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False, default=0.0)
    pnl_percent: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    num_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    winners: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    losers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    win_rate: Mapped[Optional[float]] = mapped_column(Numeric(5, 2), nullable=True)
    avg_win: Mapped[Optional[float]] = mapped_column(Numeric(12, 2), nullable=True)
    avg_loss: Mapped[Optional[float]] = mapped_column(Numeric(12, 2), nullable=True)
    largest_win: Mapped[Optional[float]] = mapped_column(Numeric(12, 2), nullable=True)
    largest_loss: Mapped[Optional[float]] = mapped_column(Numeric(12, 2), nullable=True)
    market_regime: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    __table_args__ = (
        Index("ix_daily_stats_date", "date"),
    )


class WatchlistItem(Base):
    """Symbols on the watchlist for scanning."""
    __tablename__ = "watchlist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, unique=True)
    added_date: Mapped[date] = mapped_column(Date, nullable=False, default=date.today)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    sector: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    market_cap: Mapped[Optional[float]] = mapped_column(Numeric(16, 2), nullable=True)

    __table_args__ = (
        Index("ix_watchlist_symbol", "symbol"),
        Index("ix_watchlist_active", "active"),
    )


# =============================================================================
# Order Execution Models
# =============================================================================

class OrderExecution(Base):
    """
    Model for tracking order executions and fill quality.

    Stores detailed information about each order execution including:
    - Expected vs actual fill prices (slippage tracking)
    - Fill timing metrics
    - Execution status
    """
    __tablename__ = "order_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    side: Mapped[str] = mapped_column(String(20), nullable=False)  # buy, sell, buy_to_cover, sell_short

    # Pricing - using Numeric for precision
    expected_price: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    fill_price: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    slippage: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False, default=0.0)
    slippage_pct: Mapped[float] = mapped_column(Numeric(8, 4), nullable=False, default=0.0)

    # Quantity
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    filled_quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Timing
    order_submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fill_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    time_to_fill_seconds: Mapped[Optional[float]] = mapped_column(Numeric(10, 3), nullable=True)

    # Status
    status: Mapped[str] = mapped_column(
        SQLEnum(OrderExecutionStatus, native_enum=False),
        default=OrderExecutionStatus.FILLED,
        nullable=False
    )

    # Metadata
    broker_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_order_executions_order_id", "order_id"),
        Index("ix_order_executions_symbol", "symbol"),
        Index("ix_order_executions_fill_time", "fill_time"),
        Index("ix_order_executions_status", "status"),
        Index("ix_order_executions_symbol_fill_time", "symbol", "fill_time"),
        Index("ix_order_executions_status_fill_time", "status", "fill_time"),
        CheckConstraint("quantity > 0", name="ck_order_executions_quantity_positive"),
        CheckConstraint("filled_quantity >= 0", name="ck_order_executions_filled_quantity_non_negative"),
    )


# =============================================================================
# User and Authentication Models
# =============================================================================

class APIUser(Base):
    """API user model for authentication persistence."""
    __tablename__ = "api_users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    api_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    api_secret_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    tier: Mapped[str] = mapped_column(
        SQLEnum(SubscriptionTierEnum, native_enum=False),
        default=SubscriptionTierEnum.FREE,
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    rate_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    last_request_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_api_users_email", "email"),
        Index("ix_api_users_api_key", "api_key"),
        Index("ix_api_users_tier", "tier"),
        Index("ix_api_users_is_active", "is_active"),
    )


class User(Base):
    """Dashboard user model for web authentication."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_login_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Stripe integration
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)

    # Relationships
    subscriptions = relationship("Subscription", back_populates="user", cascade="all, delete-orphan")
    payments = relationship("PaymentHistory", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    alert_schedules = relationship("AlertSchedule", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_users_username", "username"),
        Index("ix_users_email", "email"),
        Index("ix_users_is_active", "is_active"),
        Index("ix_users_stripe_customer_id", "stripe_customer_id"),
    )

    def get_id(self) -> str:
        """Return user ID as string for Flask-Login."""
        return str(self.id)

    @property
    def is_authenticated(self) -> bool:
        """Return True if user is authenticated."""
        return True

    @property
    def is_anonymous(self) -> bool:
        """Return False as this is not an anonymous user."""
        return False

    @property
    def user_id(self) -> str:
        """Alias for id, matching APIUserDTO interface for dual-auth compatibility."""
        return str(self.id)

    @property
    def is_expired(self) -> bool:
        """Dashboard users don't expire. Matches APIUserDTO interface."""
        return False

    @property
    def subscription_tier(self):
        """Get the user's active subscription tier.

        Checks the subscriptions relationship for an active subscription
        and returns the corresponding SubscriptionTier from api.v1.auth.
        Defaults to ELITE for admin users, PRO for active users without
        a subscription (dashboard-only users).
        """
        try:
            from api.v1.auth import SubscriptionTier
        except ImportError:
            return None

        # Admin users get full access
        if self.is_admin:
            return SubscriptionTier.ELITE

        # Check for active subscription
        if self.subscriptions:
            for sub in self.subscriptions:
                if hasattr(sub, 'status') and str(sub.status) in ('active', 'trialing', 'SubscriptionStatus.ACTIVE', 'SubscriptionStatus.TRIALING'):
                    plan = sub.plan_id if hasattr(sub, 'plan_id') else 'free'
                    try:
                        return SubscriptionTier(plan)
                    except ValueError:
                        return SubscriptionTier.FREE

        # Dashboard users without subscription default to PRO access
        return SubscriptionTier.PRO


# =============================================================================
# Audit and Logging Models
# =============================================================================

class AuditLog(Base):
    """Audit log for tracking important system events."""
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    entity_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    entity_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    __table_args__ = (
        Index("ix_audit_logs_timestamp", "timestamp"),
        Index("ix_audit_logs_event_type", "event_type"),
        Index("ix_audit_logs_user_id", "user_id"),
        Index("ix_audit_logs_entity", "entity_type", "entity_id"),
    )


# =============================================================================
# Event Persistence Models
# =============================================================================

class EventStatus(str, Enum):
    """Status of persisted events."""
    PENDING = "pending"
    PUBLISHED = "published"
    PROCESSED = "processed"
    FAILED = "failed"


class EventRecord(Base):
    """
    Persisted event record for recovery and audit.

    Events are stored before publishing and marked as processed
    after successful delivery to all subscribers.
    """
    __tablename__ = "event_records"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    data: Mapped[str] = mapped_column(Text, nullable=False)  # JSON serialized
    status: Mapped[str] = mapped_column(
        SQLEnum(EventStatus, native_enum=False),
        nullable=False,
        default=EventStatus.PENDING
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    published_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_event_records_status", "status"),
        Index("ix_event_records_event_type", "event_type"),
        Index("ix_event_records_created_at", "created_at"),
        Index("ix_event_records_status_created", "status", "created_at"),
    )


class WebhookEventStatus(str, Enum):
    """Status of webhook event processing."""
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"
    DUPLICATE = "duplicate"


class WebhookEvent(Base):
    """
    Webhook event tracking for deduplication.

    Stores Stripe webhook events to prevent duplicate processing
    and enable event replay if needed.
    """
    __tablename__ = "webhook_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="stripe")
    payload: Mapped[str] = mapped_column(Text, nullable=False)  # JSON serialized
    status: Mapped[str] = mapped_column(
        SQLEnum(WebhookEventStatus, native_enum=False),
        nullable=False,
        default=WebhookEventStatus.PENDING
    )
    result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON serialized result
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_webhook_events_event_id", "event_id"),
        Index("ix_webhook_events_event_type", "event_type"),
        Index("ix_webhook_events_status", "status"),
        Index("ix_webhook_events_created_at", "created_at"),
    )


# =============================================================================
# Market Data Cache Models
# =============================================================================

class MarketDataCache(Base):
    """Cache for market data to reduce API calls."""
    __tablename__ = "market_data_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    data_type: Mapped[str] = mapped_column(String(32), nullable=False)  # quote, bars, etc.
    timeframe: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # 1m, 5m, 1d, etc.
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    data: Mapped[str] = mapped_column(Text, nullable=False)  # JSON data

    __table_args__ = (
        UniqueConstraint("symbol", "data_type", "timeframe", "timestamp", name="uq_market_data_cache"),
        Index("ix_market_data_cache_symbol", "symbol"),
        Index("ix_market_data_cache_expires", "expires_at"),
        Index("ix_market_data_cache_lookup", "symbol", "data_type", "timeframe"),
    )


# =============================================================================
# Subscription and Billing Models
# =============================================================================

class Subscription(Base):
    """
    Subscription model for tracking user billing and plan status.

    Integrates with Stripe for payment processing.
    """
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Relationship back to user
    user = relationship("User", back_populates="subscriptions")
    payments = relationship("PaymentHistory", back_populates="subscription")

    # Stripe identifiers
    stripe_subscription_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    stripe_customer_id: Mapped[str] = mapped_column(String(255), nullable=False)

    # Plan information
    plan_id: Mapped[str] = mapped_column(String(50), nullable=False)  # basic, pro, elite
    status: Mapped[str] = mapped_column(
        SQLEnum(SubscriptionStatus, native_enum=False),
        default=SubscriptionStatus.INCOMPLETE,
        nullable=False
    )

    # Billing period
    current_period_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    current_period_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Trial information
    trial_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    trial_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Cancellation
    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    canceled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # Note: ix_subscriptions_user_id removed - already created by index=True on user_id column
        Index("ix_subscriptions_stripe_subscription_id", "stripe_subscription_id"),
        Index("ix_subscriptions_stripe_customer_id", "stripe_customer_id"),
        Index("ix_subscriptions_status", "status"),
        Index("ix_subscriptions_plan_id", "plan_id"),
        Index("ix_subscriptions_user_status", "user_id", "status"),
    )

    @property
    def is_active(self) -> bool:
        """Check if subscription is currently active."""
        return self.status in (SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING)

    @property
    def is_trialing(self) -> bool:
        """Check if subscription is in trial period."""
        return self.status == SubscriptionStatus.TRIALING

    @property
    def is_canceled(self) -> bool:
        """Check if subscription is canceled."""
        return self.status == SubscriptionStatus.CANCELED

    @property
    def will_cancel(self) -> bool:
        """Check if subscription will cancel at period end."""
        return self.cancel_at_period_end and not self.is_canceled


class PaymentHistory(Base):
    """
    Payment history for tracking invoices and payments.
    """
    __tablename__ = "payment_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subscription_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("subscriptions.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Relationships
    user = relationship("User", back_populates="payments")
    subscription = relationship("Subscription", back_populates="payments")

    # Stripe identifiers
    stripe_invoice_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    stripe_payment_intent_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Amount
    amount: Mapped[int] = mapped_column(Integer, nullable=False)  # Amount in cents
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="usd")

    # Status
    status: Mapped[str] = mapped_column(String(50), nullable=False)  # paid, failed, refunded, etc.

    # Description
    description: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    paid_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # Note: ix_payment_history_user_id and ix_payment_history_subscription_id removed
        # - already created by index=True on user_id and subscription_id columns
        Index("ix_payment_history_stripe_invoice_id", "stripe_invoice_id"),
        Index("ix_payment_history_status", "status"),
        Index("ix_payment_history_created_at", "created_at"),
    )


# =============================================================================
# Session Management Models
# =============================================================================

class UserSession(Base):
    """
    User session model for tracking active sessions.

    Stores session information for security management including:
    - Session token (hashed)
    - Client information (IP, user agent)
    - Activity timestamps
    - Active/revoked status
    """
    __tablename__ = "user_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Relationship back to user
    user = relationship("User", back_populates="sessions")

    # Session token (stored as hash)
    session_token: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)

    # Client information
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)  # IPv6 max length
    user_agent: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    device_info: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Parsed browser/OS info

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    last_activity: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    __table_args__ = (
        # Note: ix_user_sessions_user_id removed - already created by index=True on user_id column
        Index("ix_user_sessions_session_token", "session_token"),
        Index("ix_user_sessions_is_active", "is_active"),
        Index("ix_user_sessions_user_active", "user_id", "is_active"),
        Index("ix_user_sessions_last_activity", "last_activity"),
    )


# =============================================================================
# Alert Schedule Models
# =============================================================================

class AlertSchedule(Base):
    """
    Alert schedule model for managing quiet hours and delivery schedules.

    Stores user-specific alert scheduling preferences including:
    - Quiet hours (start/end times)
    - Day of week restrictions
    - Channel-specific settings
    - Critical alert override options
    """
    __tablename__ = "alert_schedules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Relationship back to user
    user = relationship("User", back_populates="alert_schedules")

    # Schedule name (for multiple schedules per user)
    name: Mapped[str] = mapped_column(String(100), nullable=False, default="Default")

    # Quiet hours time range (stored as time strings HH:MM)
    start_time: Mapped[str] = mapped_column(String(5), nullable=False)  # e.g., "22:00"
    end_time: Mapped[str] = mapped_column(String(5), nullable=False)    # e.g., "07:00"

    # Timezone for this schedule
    timezone: Mapped[str] = mapped_column(String(50), nullable=False, default="America/New_York")

    # Days of week (comma-separated: "monday,tuesday,wednesday,thursday,friday,saturday,sunday")
    # NULL means all days
    days_of_week: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Channels this schedule applies to (comma-separated: "email,sms,discord,telegram,pushover")
    # NULL means all channels
    channels: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Alert types this schedule applies to (comma-separated: "signal,trade,system,summary")
    # NULL means all alert types
    alert_types: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Critical alert behavior
    override_critical: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Priority threshold (only suppress alerts below this priority)
    # Options: low, normal, high, critical
    priority_threshold: Mapped[str] = mapped_column(String(20), nullable=False, default="critical")

    # Whether this schedule is active
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # Note: ix_alert_schedules_user_id removed - already created by index=True on user_id column
        Index("ix_alert_schedules_is_active", "is_active"),
        Index("ix_alert_schedules_user_active", "user_id", "is_active"),
    )

    def get_days_list(self) -> Optional[list]:
        """Get days of week as a list."""
        if self.days_of_week:
            return [d.strip() for d in self.days_of_week.split(",")]
        return None

    def set_days_list(self, days: Optional[list]):
        """Set days of week from a list."""
        if days:
            self.days_of_week = ",".join(days)
        else:
            self.days_of_week = None

    def get_channels_list(self) -> Optional[list]:
        """Get channels as a list."""
        if self.channels:
            return [c.strip() for c in self.channels.split(",")]
        return None

    def set_channels_list(self, channels: Optional[list]):
        """Set channels from a list."""
        if channels:
            self.channels = ",".join(channels)
        else:
            self.channels = None

    def get_alert_types_list(self) -> Optional[list]:
        """Get alert types as a list."""
        if self.alert_types:
            return [t.strip() for t in self.alert_types.split(",")]
        return None

    def set_alert_types_list(self, types: Optional[list]):
        """Set alert types from a list."""
        if types:
            self.alert_types = ",".join(types)
        else:
            self.alert_types = None


# =============================================================================
# Trade Monitoring & Learning Models
# =============================================================================

class RejectedSignal(Base):
    """
    Rejected trading signals with full context for outcome analysis.

    Stores signals that were rejected by the analyzer along with the reasons
    for rejection. Price outcome columns are filled later by the outcome tracker
    to evaluate whether rejection criteria are too strict.
    """
    __tablename__ = "rejected_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    direction: Mapped[str] = mapped_column(SQLEnum(TradeDirection, native_enum=False), nullable=False)
    rrs: Mapped[float] = mapped_column(Numeric(8, 4), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    rejection_reasons: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array of reasons
    market_regime: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    daily_strong: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    daily_weak: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    atr: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    volume: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    ml_probability: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    ml_confidence: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    strategy_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, default='rrs_momentum')

    # Outcome tracking (filled later by OutcomeTracker)
    price_after_1h: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    price_after_4h: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    price_after_1d: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    would_have_pnl_1h: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    would_have_pnl_4h: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    would_have_pnl_1d: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)

    __table_args__ = (
        Index("ix_rejected_signals_symbol", "symbol"),
        Index("ix_rejected_signals_timestamp", "timestamp"),
        Index("ix_rejected_signals_symbol_timestamp", "symbol", "timestamp"),
        Index("ix_rejected_signals_strategy_name", "strategy_name"),
    )


class EquitySnapshot(Base):
    """
    Equity curve tracking over time.

    Records account equity at regular intervals for drawdown analysis
    and performance visualization.
    """
    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    equity_value: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    cash: Mapped[Optional[float]] = mapped_column(Numeric(14, 2), nullable=True)
    positions_value: Mapped[Optional[float]] = mapped_column(Numeric(14, 2), nullable=True)
    open_positions_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    drawdown_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    high_water_mark: Mapped[Optional[float]] = mapped_column(Numeric(14, 2), nullable=True)

    __table_args__ = (
        Index("ix_equity_snapshots_timestamp", "timestamp"),
    )


class ParameterChange(Base):
    """
    Historical record of adaptive learner parameter changes.

    Tracks every parameter adjustment with the context that drove it,
    enabling analysis of how the system adapts over time.
    """
    __tablename__ = "parameter_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    parameter_name: Mapped[str] = mapped_column(String(64), nullable=False)
    old_value: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    new_value: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    reason: Mapped[str] = mapped_column(String(255), nullable=False)
    trade_count_basis: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    win_rate_at_change: Mapped[Optional[float]] = mapped_column(Numeric(5, 4), nullable=True)
    regime: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    __table_args__ = (
        Index("ix_parameter_changes_timestamp", "timestamp"),
        Index("ix_parameter_changes_parameter_name", "parameter_name"),
    )


class QueuedAlertRecord(Base):
    """
    Queued alert record for alerts pending delivery after quiet hours.

    Stores alerts that were suppressed during quiet hours for later delivery.
    """
    __tablename__ = "queued_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    # Optional user association
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Alert content
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[str] = mapped_column(String(20), nullable=False, default="normal")
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False, default="general")

    # Target channels (comma-separated)
    channels: Mapped[str] = mapped_column(String(200), nullable=False)

    # Extra data (JSON)
    extra_data: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Scheduling
    queued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    scheduled_for: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Delivery tracking
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_attempt_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Status: pending, sent, failed, cancelled
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")

    __table_args__ = (
        Index("ix_queued_alerts_alert_id", "alert_id"),
        Index("ix_queued_alerts_user_id", "user_id"),
        Index("ix_queued_alerts_status", "status"),
        Index("ix_queued_alerts_scheduled_for", "scheduled_for"),
        Index("ix_queued_alerts_user_status", "user_id", "status"),
    )

    def get_channels_list(self) -> list:
        """Get channels as a list."""
        return [c.strip() for c in self.channels.split(",")]

    def set_channels_list(self, channels: list):
        """Set channels from a list."""
        self.channels = ",".join(channels)


# =============================================================================
# ML Training Data Models
# =============================================================================

class IntradayBar(Base):
    """5-minute OHLCV bars for intraday analysis and Entry Timing model."""
    __tablename__ = "intraday_bars"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    vwap: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_intraday_bars_symbol_timestamp"),
        Index("ix_intraday_bars_symbol_timestamp", "symbol", "timestamp"),
    )


class TechnicalIndicator(Base):
    """Daily technical indicators for ML feature persistence."""
    __tablename__ = "technical_indicators"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    rsi_14: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    macd_line: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    macd_signal: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    macd_histogram: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    bb_upper: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    bb_middle: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    bb_lower: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    bb_width: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    ema_9: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    ema_21: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    ema_50: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    ema_200: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    adx: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    obv: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    atr_14: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    close_price: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_technical_indicators_symbol_date"),
        Index("ix_technical_indicators_symbol_date", "symbol", "date"),
    )


class TradeSnapshot(Base):
    """Point-in-time position snapshots for MFE/MAE tracking."""
    __tablename__ = "trade_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("trades.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    current_price: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    unrealized_pnl: Mapped[Optional[float]] = mapped_column(Numeric(12, 2), nullable=True)
    unrealized_pnl_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    unrealized_r: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    mfe: Mapped[Optional[float]] = mapped_column(Numeric(12, 2), nullable=True)
    mae: Mapped[Optional[float]] = mapped_column(Numeric(12, 2), nullable=True)
    mfe_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    mae_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    bars_held: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rsi_at_snapshot: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    distance_to_stop_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    distance_to_target_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)

    __table_args__ = (
        Index("ix_trade_snapshots_trade_id_timestamp", "trade_id", "timestamp"),
    )


class MarketRegimeDaily(Base):
    """Daily market context for regime-aware ML models."""
    __tablename__ = "market_regime_daily"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, unique=True)
    vix_close: Mapped[Optional[float]] = mapped_column(Numeric(8, 2), nullable=True)
    vix_regime: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    spy_close: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    spy_trend: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    spy_above_200ema: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    spy_above_50ema: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    advance_decline_ratio: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    new_highs: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    new_lows: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    breadth_thrust: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    put_call_ratio: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    regime_label: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)

    __table_args__ = (
        Index("ix_market_regime_daily_date", "date"),
    )


class SectorData(Base):
    """Daily sector relative strength for sector rotation analysis."""
    __tablename__ = "sector_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    sector: Mapped[str] = mapped_column(String(30), nullable=False)
    etf_symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    close_price: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    daily_return_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    relative_strength_5d: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    relative_strength_20d: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    relative_strength_60d: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    sector_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint("date", "sector", name="uq_sector_data_date_sector"),
        Index("ix_sector_data_date_sector", "date", "sector"),
    )


class OptionsGreeksHistory(Base):
    """Greeks snapshots during options positions for options ML."""
    __tablename__ = "options_greeks_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    options_trade_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("options_trades.id", ondelete="SET NULL"), nullable=True
    )
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    underlying_price: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    delta: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    gamma: Mapped[Optional[float]] = mapped_column(Numeric(8, 6), nullable=True)
    theta: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    vega: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    iv: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    premium: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    dte: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    moneyness: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    intrinsic_value: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    extrinsic_value: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)

    __table_args__ = (
        Index("ix_options_greeks_history_symbol_timestamp", "symbol", "timestamp"),
    )


class EarningsCalendar(Base):
    """Earnings dates and surprises for event-driven ML features."""
    __tablename__ = "earnings_calendar"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    earnings_date: Mapped[date] = mapped_column(Date, nullable=False)
    timing: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)  # BMO/AMC
    eps_estimate: Mapped[Optional[float]] = mapped_column(Numeric(10, 4), nullable=True)
    eps_actual: Mapped[Optional[float]] = mapped_column(Numeric(10, 4), nullable=True)
    eps_surprise_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    revenue_estimate: Mapped[Optional[float]] = mapped_column(Numeric(16, 2), nullable=True)
    revenue_actual: Mapped[Optional[float]] = mapped_column(Numeric(16, 2), nullable=True)
    revenue_surprise_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    price_change_1d_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    iv_rank_before: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint("symbol", "earnings_date", name="uq_earnings_calendar_symbol_date"),
        Index("ix_earnings_calendar_symbol_date", "symbol", "earnings_date"),
    )


class DailyBar(Base):
    """Daily OHLCV bars cached from IBKR for historical data lookups."""
    __tablename__ = "daily_bars"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    bar_date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint("symbol", "bar_date", name="uq_daily_bars_symbol_date"),
        Index("ix_daily_bars_symbol_date", "symbol", "bar_date"),
    )
