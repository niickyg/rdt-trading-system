"""
Risk Management Data Models
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Dict, List
from enum import Enum


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskViolationType(Enum):
    """Types of risk violations"""
    MAX_POSITION_SIZE = "max_position_size"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_OPEN_POSITIONS = "max_open_positions"
    MAX_SECTOR_EXPOSURE = "max_sector_exposure"
    MAX_CORRELATION = "max_correlation"
    PATTERN_DAY_TRADER = "pdt_rule"
    INSUFFICIENT_BUYING_POWER = "insufficient_buying_power"
    MARKET_CLOSED = "market_closed"
    VOLATILITY_TOO_HIGH = "volatility_too_high"


@dataclass
class RiskCheckResult:
    """Result of a risk check"""
    passed: bool
    violation_type: Optional[RiskViolationType] = None
    message: str = ""
    current_value: Optional[float] = None
    limit_value: Optional[float] = None
    risk_level: RiskLevel = RiskLevel.LOW

    def __bool__(self) -> bool:
        return self.passed


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    shares: int
    position_value: float
    risk_amount: float
    stop_distance: float
    stop_price: float
    target_price: float
    risk_reward_ratio: float
    risk_percent: float
    reason: str = ""


@dataclass
class RiskMetrics:
    """Current risk metrics for the account"""
    # Daily metrics
    daily_pnl: float = 0.0
    daily_pnl_percent: float = 0.0
    daily_trades: int = 0
    daily_wins: int = 0
    daily_losses: int = 0

    # Position metrics
    open_positions: int = 0
    total_exposure: float = 0.0
    exposure_percent: float = 0.0
    largest_position_percent: float = 0.0

    # Risk metrics
    current_drawdown: float = 0.0
    current_drawdown_percent: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0

    # Volatility
    portfolio_beta: float = 1.0
    avg_position_atr_percent: float = 0.0

    # Sector exposure
    sector_exposure: Dict[str, float] = field(default_factory=dict)

    # Calculated at
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    # Per-trade limits
    max_risk_per_trade: float = 0.01  # 1% of account
    max_position_size: float = 0.10  # 10% of account
    min_risk_reward: float = 2.0  # 2:1 R/R minimum

    # Daily limits
    max_daily_loss: float = 0.03  # 3% daily loss limit
    max_daily_trades: int = 10  # Maximum trades per day

    # Portfolio limits
    max_open_positions: int = 5
    max_total_exposure: float = 0.50  # 50% of account
    max_sector_exposure: float = 0.25  # 25% in any sector
    max_correlated_positions: int = 3

    # Drawdown limits
    max_drawdown: float = 0.10  # 10% max drawdown
    drawdown_cooldown_hours: int = 24

    # Volatility limits
    max_position_atr_percent: float = 5.0  # Max 5% ATR for position

    # PDT rule
    day_trade_limit: int = 3  # For accounts < $25k
    pdt_account_minimum: float = 25000.0


@dataclass
class TradeRisk:
    """Risk assessment for a potential trade"""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    stop_price: float
    target_price: float
    shares: int
    position_value: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    risk_percent_of_account: float
    position_percent_of_account: float
    atr: float
    atr_percent: float

    # Risk checks
    checks_passed: List[RiskCheckResult] = field(default_factory=list)
    checks_failed: List[RiskCheckResult] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.checks_failed) == 0

    @property
    def overall_risk_level(self) -> RiskLevel:
        if self.checks_failed:
            return max(c.risk_level for c in self.checks_failed)
        return RiskLevel.LOW


@dataclass
class DailyRiskReport:
    """End of day risk report"""
    date: date
    starting_balance: float
    ending_balance: float
    daily_pnl: float
    daily_pnl_percent: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_percent: float
    risk_violations: List[RiskCheckResult] = field(default_factory=list)
    notes: str = ""
