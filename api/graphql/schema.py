"""
GraphQL Schema Definition

Defines the complete GraphQL schema for the RDT Trading System including:
- Object types (Signal, Position, Trade, User, DailyStats, etc.)
- Query type with all resolvers
- Mutation type for creating alerts, closing positions, updating settings
"""

import graphene
from typing import List, Optional

from api.graphql.types import (
    DateTime,
    Decimal,
    DirectionEnum,
    SignalStrengthEnum,
    TradeStatusEnum,
    SignalStatusEnum,
    ExitReasonEnum,
    AlertConditionEnum,
    NotificationMethodEnum,
    SubscriptionTierEnum,
    AlertInput,
    SettingsInput,
    ClosePositionInput,
    SignalFilterInput,
    TradeFilterInput,
    PositionFilterInput,
)
from api.graphql.resolvers import (
    SignalResolver,
    PositionResolver,
    TradeResolver,
    PortfolioResolver,
    MarketResolver,
    ScannerResolver,
    DailyStatsResolver,
    UserResolver,
    AlertMutationResolver,
    PositionMutationResolver,
    SettingsMutationResolver,
)
from api.graphql.auth import require_auth, require_tier, require_feature


# =============================================================================
# Object Types
# =============================================================================

class Signal(graphene.ObjectType):
    """Trading signal from the RRS scanner"""
    id = graphene.Int(description="Unique signal identifier")
    symbol = graphene.String(description="Stock ticker symbol")
    timestamp = DateTime(description="When the signal was generated")
    generated_at = DateTime(description="Alias for timestamp")
    direction = DirectionEnum(description="Signal direction (long/short)")
    strength = SignalStrengthEnum(description="Signal strength based on daily analysis")
    rrs = Decimal(description="Real Relative Strength value")
    price = Decimal(description="Price at signal generation")
    entry_price = Decimal(description="Suggested entry price")
    atr = Decimal(description="Average True Range")
    status = SignalStatusEnum(description="Current signal status")
    daily_strong = graphene.Boolean(description="Daily timeframe showing strength")
    daily_weak = graphene.Boolean(description="Daily timeframe showing weakness")
    strategy = graphene.String(description="Strategy that generated the signal")


class Position(graphene.ObjectType):
    """Open trading position"""
    id = graphene.Int(description="Unique position identifier")
    symbol = graphene.String(description="Stock ticker symbol")
    direction = DirectionEnum(description="Position direction (long/short)")
    entry_price = Decimal(description="Entry price")
    shares = graphene.Int(description="Number of shares")
    entry_time = DateTime(description="When the position was opened")
    stop_loss = Decimal(description="Stop loss price")
    take_profit = Decimal(description="Take profit target")
    current_price = Decimal(description="Current market price")
    pnl = Decimal(description="Unrealized profit/loss in dollars")
    pnl_pct = Decimal(description="Unrealized profit/loss percentage")
    rrs_at_entry = Decimal(description="RRS value when position was opened")


class Trade(graphene.ObjectType):
    """Completed or active trade record"""
    id = graphene.Int(description="Unique trade identifier")
    symbol = graphene.String(description="Stock ticker symbol")
    direction = DirectionEnum(description="Trade direction (long/short)")
    entry_price = Decimal(description="Entry price")
    exit_price = Decimal(description="Exit price (if closed)")
    shares = graphene.Int(description="Number of shares")
    entry_time = DateTime(description="Entry timestamp")
    exit_time = DateTime(description="Exit timestamp (if closed)")
    pnl = Decimal(description="Realized profit/loss in dollars")
    pnl_percent = Decimal(description="Realized profit/loss percentage")
    rrs_at_entry = Decimal(description="RRS value at entry")
    stop_loss = Decimal(description="Stop loss price")
    take_profit = Decimal(description="Take profit target")
    status = TradeStatusEnum(description="Trade status")
    exit_reason = ExitReasonEnum(description="Reason for exit")


class User(graphene.ObjectType):
    """API user information"""
    id = graphene.String(description="User ID")
    email = graphene.String(description="User email address")
    tier = SubscriptionTierEnum(description="Subscription tier")
    created_at = DateTime(description="Account creation timestamp")
    is_active = graphene.Boolean(description="Whether account is active")
    rate_limit = graphene.Int(description="API rate limit per hour")
    features = graphene.JSONString(description="Available features for this tier")


class DailyStats(graphene.ObjectType):
    """Daily trading statistics"""
    date = graphene.String(description="Date (YYYY-MM-DD)")
    starting_balance = Decimal(description="Starting portfolio balance")
    ending_balance = Decimal(description="Ending portfolio balance")
    pnl = Decimal(description="Day's profit/loss")
    pnl_percent = Decimal(description="Day's P&L percentage")
    num_trades = graphene.Int(description="Number of trades")
    winners = graphene.Int(description="Number of winning trades")
    losers = graphene.Int(description="Number of losing trades")
    win_rate = Decimal(description="Win rate percentage")
    avg_win = Decimal(description="Average winning trade")
    avg_loss = Decimal(description="Average losing trade")
    largest_win = Decimal(description="Largest winning trade")
    largest_loss = Decimal(description="Largest losing trade")
    market_regime = graphene.String(description="Market regime for the day")


class MarketStatus(graphene.ObjectType):
    """Current market status"""
    is_open = graphene.Boolean(description="Whether market is currently open")
    status = graphene.String(description="Market status description")
    current_time = DateTime(description="Current market time (Eastern)")
    next_open = DateTime(description="Next market open time")
    next_close = DateTime(description="Next market close time")
    timezone = graphene.String(description="Market timezone")


class ScannerStatus(graphene.ObjectType):
    """RRS Scanner operational status"""
    is_running = graphene.Boolean(description="Whether scanner is running")
    last_scan = DateTime(description="Last scan timestamp")
    symbols_monitored = graphene.Int(description="Number of symbols being scanned")
    active_signals = graphene.Int(description="Number of active signals")
    scan_interval_seconds = graphene.Int(description="Scan interval in seconds")
    scanner_version = graphene.String(description="Scanner version")


class PerformanceStats(graphene.ObjectType):
    """Trading performance statistics"""
    total_signals = graphene.Int(description="Total number of signals/trades")
    wins = graphene.Int(description="Number of winning trades")
    losses = graphene.Int(description="Number of losing trades")
    win_rate = Decimal(description="Win rate (0-1)")
    avg_win_pct = Decimal(description="Average win percentage")
    avg_loss_pct = Decimal(description="Average loss percentage")
    profit_factor = Decimal(description="Profit factor (gross profit / gross loss)")
    total_return_pct = Decimal(description="Total return percentage")
    max_drawdown_pct = Decimal(description="Maximum drawdown percentage")
    total_pnl = Decimal(description="Total P&L in dollars")


class PortfolioSummary(graphene.ObjectType):
    """Portfolio overview"""
    total_value = Decimal(description="Total portfolio value")
    cash_available = Decimal(description="Available cash")
    buying_power = Decimal(description="Buying power")
    day_pnl = Decimal(description="Today's P&L")
    day_pnl_percent = Decimal(description="Today's P&L percentage")
    total_pnl = Decimal(description="Total P&L")
    total_pnl_percent = Decimal(description="Total P&L percentage")
    positions_count = graphene.Int(description="Number of open positions")
    positions = graphene.List(Position, description="List of open positions")
    performance = graphene.Field(PerformanceStats, description="Performance statistics")


class Alert(graphene.ObjectType):
    """Price alert"""
    id = graphene.Int(description="Alert ID")
    user_id = graphene.String(description="Owner user ID")
    symbol = graphene.String(description="Stock symbol")
    price = Decimal(description="Target price")
    condition = AlertConditionEnum(description="Alert condition")
    notification_method = NotificationMethodEnum(description="Notification method")
    message = graphene.String(description="Custom message")
    expires_at = DateTime(description="Expiration time")
    created_at = DateTime(description="Creation time")
    is_active = graphene.Boolean(description="Whether alert is active")
    triggered = graphene.Boolean(description="Whether alert has triggered")


class Settings(graphene.ObjectType):
    """User settings"""
    user_id = graphene.String(description="User ID")
    email_alerts = graphene.Boolean(description="Email alerts enabled")
    push_alerts = graphene.Boolean(description="Push alerts enabled")
    sms_alerts = graphene.Boolean(description="SMS alerts enabled")
    webhook_url = graphene.String(description="Webhook URL")
    default_position_size = Decimal(description="Default position size")
    risk_per_trade = Decimal(description="Risk per trade percentage")
    max_open_positions = graphene.Int(description="Maximum open positions")
    auto_trailing_stop = graphene.Boolean(description="Auto trailing stop enabled")
    trailing_stop_percent = Decimal(description="Trailing stop percentage")
    updated_at = DateTime(description="Last update time")


# =============================================================================
# Query Type
# =============================================================================

class Query(graphene.ObjectType):
    """Root query type"""

    # Signal queries
    signals = graphene.List(
        Signal,
        limit=graphene.Int(default_value=50, description="Maximum signals to return"),
        offset=graphene.Int(default_value=0, description="Number of signals to skip"),
        direction=DirectionEnum(description="Filter by direction"),
        strength=SignalStrengthEnum(description="Filter by strength"),
        symbol=graphene.String(description="Filter by symbol"),
        status=SignalStatusEnum(description="Filter by status"),
        min_rrs=graphene.Float(description="Minimum RRS value"),
        max_rrs=graphene.Float(description="Maximum RRS value"),
        description="Get trading signals with optional filters"
    )

    signal = graphene.Field(
        Signal,
        id=graphene.Int(required=True, description="Signal ID"),
        description="Get a single signal by ID"
    )

    # Position queries
    positions = graphene.List(
        Position,
        status=graphene.String(description="Filter by status"),
        symbol=graphene.String(description="Filter by symbol"),
        direction=DirectionEnum(description="Filter by direction"),
        profitable_only=graphene.Boolean(default_value=False, description="Only profitable"),
        description="Get open positions with optional filters"
    )

    position = graphene.Field(
        Position,
        symbol=graphene.String(required=True, description="Position symbol"),
        description="Get a single position by symbol"
    )

    # Trade queries
    trades = graphene.List(
        Trade,
        days=graphene.Int(default_value=30, description="Days to look back"),
        symbol=graphene.String(description="Filter by symbol"),
        status=TradeStatusEnum(description="Filter by status"),
        direction=DirectionEnum(description="Filter by direction"),
        limit=graphene.Int(default_value=100, description="Maximum trades to return"),
        offset=graphene.Int(default_value=0, description="Number of trades to skip"),
        description="Get trades with optional filters"
    )

    trade = graphene.Field(
        Trade,
        id=graphene.Int(required=True, description="Trade ID"),
        description="Get a single trade by ID"
    )

    # Portfolio query
    portfolio = graphene.Field(
        PortfolioSummary,
        description="Get portfolio summary including positions and performance"
    )

    # Market status query
    market_status = graphene.Field(
        MarketStatus,
        description="Get current market status"
    )

    # Scanner status query
    scanner_status = graphene.Field(
        ScannerStatus,
        description="Get RRS scanner operational status"
    )

    # Daily stats query
    daily_stats = graphene.List(
        DailyStats,
        days=graphene.Int(default_value=30, description="Number of days"),
        start_date=graphene.String(description="Start date (YYYY-MM-DD)"),
        end_date=graphene.String(description="End date (YYYY-MM-DD)"),
        description="Get daily trading statistics"
    )

    # User query
    me = graphene.Field(
        User,
        description="Get current authenticated user"
    )

    # Resolvers
    def resolve_signals(self, info, **kwargs):
        return SignalResolver.resolve_signals(info, **kwargs)

    def resolve_signal(self, info, id):
        return SignalResolver.resolve_signal_by_id(info, id)

    def resolve_positions(self, info, **kwargs):
        return PositionResolver.resolve_positions(info, **kwargs)

    def resolve_position(self, info, symbol):
        return PositionResolver.resolve_position_by_symbol(info, symbol)

    def resolve_trades(self, info, **kwargs):
        return TradeResolver.resolve_trades(info, **kwargs)

    def resolve_trade(self, info, id):
        return TradeResolver.resolve_trade_by_id(info, id)

    def resolve_portfolio(self, info):
        return PortfolioResolver.resolve_portfolio(info)

    def resolve_market_status(self, info):
        return MarketResolver.resolve_market_status(info)

    def resolve_scanner_status(self, info):
        return ScannerResolver.resolve_scanner_status(info)

    def resolve_daily_stats(self, info, **kwargs):
        return DailyStatsResolver.resolve_daily_stats(info, **kwargs)

    def resolve_me(self, info):
        return UserResolver.resolve_me(info)


# =============================================================================
# Mutation Type
# =============================================================================

class CreateAlert(graphene.Mutation):
    """Create a new price alert"""

    class Arguments:
        input = AlertInput(required=True, description="Alert data")

    alert = graphene.Field(Alert, description="Created alert")
    success = graphene.Boolean(description="Whether creation was successful")
    message = graphene.String(description="Status message")

    def mutate(self, info, input):
        try:
            alert = AlertMutationResolver.create_alert(info, input)
            return CreateAlert(alert=alert, success=True, message="Alert created successfully")
        except Exception as e:
            return CreateAlert(alert=None, success=False, message=str(e))


class ClosePosition(graphene.Mutation):
    """Close an open position"""

    class Arguments:
        symbol = graphene.String(required=True, description="Symbol to close")
        price = graphene.Float(required=True, description="Exit price")
        exit_reason = ExitReasonEnum(default_value="manual", description="Exit reason")

    trade = graphene.Field(Trade, description="Closed trade record")
    success = graphene.Boolean(description="Whether close was successful")
    message = graphene.String(description="Status message")

    def mutate(self, info, symbol, price, exit_reason="manual"):
        try:
            reason = exit_reason.value if hasattr(exit_reason, 'value') else exit_reason
            trade = PositionMutationResolver.close_position(info, symbol, price, reason)
            return ClosePosition(trade=trade, success=True, message=f"Position closed for {symbol}")
        except Exception as e:
            return ClosePosition(trade=None, success=False, message=str(e))


class UpdateSettings(graphene.Mutation):
    """Update user settings"""

    class Arguments:
        input = SettingsInput(required=True, description="Settings to update")

    settings = graphene.Field(Settings, description="Updated settings")
    success = graphene.Boolean(description="Whether update was successful")
    message = graphene.String(description="Status message")

    def mutate(self, info, input):
        try:
            settings = SettingsMutationResolver.update_settings(info, input)
            return UpdateSettings(settings=settings, success=True, message="Settings updated")
        except Exception as e:
            return UpdateSettings(settings=None, success=False, message=str(e))


class Mutation(graphene.ObjectType):
    """Root mutation type"""
    create_alert = CreateAlert.Field(description="Create a new price alert")
    close_position = ClosePosition.Field(description="Close an open position")
    update_settings = UpdateSettings.Field(description="Update user settings")


# =============================================================================
# Schema
# =============================================================================

schema = graphene.Schema(
    query=Query,
    mutation=Mutation,
    types=[
        Signal,
        Position,
        Trade,
        User,
        DailyStats,
        MarketStatus,
        ScannerStatus,
        PortfolioSummary,
        Alert,
        Settings,
        PerformanceStats,
    ]
)
