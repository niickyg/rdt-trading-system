"""
GraphQL Custom Types and Input Types

Defines:
- Custom scalars (DateTime, Decimal)
- Enum types (Direction, Strength, OrderStatus)
- Input types for mutations
"""

import graphene
from graphene import Scalar
from datetime import datetime
from decimal import Decimal as PyDecimal


# =============================================================================
# Custom Scalars
# =============================================================================

class DateTime(Scalar):
    """
    Custom DateTime scalar for GraphQL.

    Serializes to ISO 8601 format string.
    Parses ISO 8601 format string to datetime.
    """

    @staticmethod
    def serialize(dt):
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        return dt.isoformat()

    @staticmethod
    def parse_literal(ast, _variables=None):
        from graphql.language import ast as graphql_ast
        if isinstance(ast, graphql_ast.StringValueNode):
            return datetime.fromisoformat(ast.value.replace('Z', '+00:00'))
        return None

    @staticmethod
    def parse_value(value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value.replace('Z', '+00:00'))


class Decimal(Scalar):
    """
    Custom Decimal scalar for GraphQL.

    Used for precise financial calculations.
    """

    @staticmethod
    def serialize(value):
        if value is None:
            return None
        return float(value)

    @staticmethod
    def parse_literal(ast, _variables=None):
        from graphql.language import ast as graphql_ast
        if isinstance(ast, (graphql_ast.FloatValueNode, graphql_ast.IntValueNode)):
            return PyDecimal(str(ast.value))
        if isinstance(ast, graphql_ast.StringValueNode):
            return PyDecimal(ast.value)
        return None

    @staticmethod
    def parse_value(value):
        if value is None:
            return None
        return PyDecimal(str(value))


# =============================================================================
# Enum Types
# =============================================================================

class DirectionEnum(graphene.Enum):
    """Trade direction enum"""
    LONG = "long"
    SHORT = "short"


class SignalStrengthEnum(graphene.Enum):
    """Signal strength enum based on daily analysis"""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class TradeStatusEnum(graphene.Enum):
    """Trade status enum"""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class SignalStatusEnum(graphene.Enum):
    """Signal status enum"""
    PENDING = "pending"
    TRIGGERED = "triggered"
    EXPIRED = "expired"
    IGNORED = "ignored"


class ExitReasonEnum(graphene.Enum):
    """Exit reason enum"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    MANUAL = "manual"
    END_OF_DAY = "end_of_day"


class AlertConditionEnum(graphene.Enum):
    """Price alert condition enum"""
    ABOVE = "above"
    BELOW = "below"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"


class NotificationMethodEnum(graphene.Enum):
    """Notification method enum"""
    EMAIL = "email"
    PUSH = "push"
    SMS = "sms"
    WEBHOOK = "webhook"


class SubscriptionTierEnum(graphene.Enum):
    """Subscription tier enum"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ELITE = "elite"


# =============================================================================
# Input Types for Mutations
# =============================================================================

class AlertInput(graphene.InputObjectType):
    """Input type for creating alerts"""
    symbol = graphene.String(required=True, description="Stock symbol to monitor")
    price = graphene.Float(required=True, description="Target price for alert")
    condition = AlertConditionEnum(required=True, description="Alert trigger condition")
    notification_method = NotificationMethodEnum(
        default_value="email",
        description="How to send the notification"
    )
    message = graphene.String(description="Custom message for the alert")
    expires_at = DateTime(description="When the alert expires (optional)")


class SettingsInput(graphene.InputObjectType):
    """Input type for updating user settings"""
    email_alerts = graphene.Boolean(description="Enable email alerts")
    push_alerts = graphene.Boolean(description="Enable push notifications")
    sms_alerts = graphene.Boolean(description="Enable SMS alerts")
    webhook_url = graphene.String(description="Webhook URL for alerts")
    default_position_size = graphene.Float(description="Default position size in dollars")
    risk_per_trade = graphene.Float(description="Maximum risk per trade as percentage")
    max_open_positions = graphene.Int(description="Maximum number of open positions")
    auto_trailing_stop = graphene.Boolean(description="Enable automatic trailing stops")
    trailing_stop_percent = graphene.Float(description="Default trailing stop percentage")


class ClosePositionInput(graphene.InputObjectType):
    """Input type for closing a position"""
    symbol = graphene.String(required=True, description="Symbol of position to close")
    exit_price = graphene.Float(required=True, description="Exit price")
    exit_reason = ExitReasonEnum(
        default_value="manual",
        description="Reason for closing the position"
    )


class TradeFilterInput(graphene.InputObjectType):
    """Input type for filtering trades"""
    symbol = graphene.String(description="Filter by symbol")
    direction = DirectionEnum(description="Filter by direction")
    status = TradeStatusEnum(description="Filter by status")
    days = graphene.Int(description="Filter by trades within last N days")
    min_pnl = graphene.Float(description="Filter by minimum PnL")
    max_pnl = graphene.Float(description="Filter by maximum PnL")


class SignalFilterInput(graphene.InputObjectType):
    """Input type for filtering signals"""
    symbol = graphene.String(description="Filter by symbol")
    direction = DirectionEnum(description="Filter by direction")
    strength = SignalStrengthEnum(description="Filter by signal strength")
    status = SignalStatusEnum(description="Filter by status")
    min_rrs = graphene.Float(description="Minimum RRS value")
    max_rrs = graphene.Float(description="Maximum RRS value")


class PositionFilterInput(graphene.InputObjectType):
    """Input type for filtering positions"""
    symbol = graphene.String(description="Filter by symbol")
    direction = DirectionEnum(description="Filter by direction")
    profitable_only = graphene.Boolean(description="Only show profitable positions")
