"""
Options Trading Data Models for the RDT Trading System.

Defines the core data structures used throughout the options module:
contracts, Greeks, legs, strategies, IV analysis, and position sizing results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class OptionRight(str, Enum):
    CALL = "C"
    PUT = "P"


class OptionAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class StrategyDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class IVRegime(str, Enum):
    LOW = "low"           # IV rank < 30
    NORMAL = "normal"     # IV rank 30-50
    HIGH = "high"         # IV rank 50-80
    VERY_HIGH = "very_high"  # IV rank > 80


@dataclass
class OptionContract:
    """Represents a single option contract."""
    symbol: str               # Underlying symbol (e.g., "AAPL")
    expiry: str               # Expiration date YYYYMMDD
    strike: float             # Strike price
    right: OptionRight        # C or P
    exchange: str = "SMART"
    multiplier: int = 100     # Standard options multiplier
    currency: str = "USD"
    con_id: Optional[int] = None  # IBKR contract ID

    @property
    def is_call(self) -> bool:
        return self.right == OptionRight.CALL

    @property
    def is_put(self) -> bool:
        return self.right == OptionRight.PUT

    @property
    def display_name(self) -> str:
        return f"{self.symbol} {self.expiry} {self.strike}{self.right.value}"


@dataclass
class OptionGreeks:
    """Option Greeks snapshot."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    implied_vol: float = 0.0
    underlying_price: float = 0.0
    option_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    open_interest: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def mid_price(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.option_price

    @property
    def spread(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

    @property
    def spread_pct(self) -> float:
        mid = self.mid_price
        if mid > 0:
            return (self.spread / mid) * 100
        return 0.0


@dataclass
class OptionLeg:
    """A single leg in a multi-leg strategy."""
    contract: OptionContract
    action: OptionAction       # BUY or SELL
    quantity: int = 1
    greeks: Optional[OptionGreeks] = None

    @property
    def is_long(self) -> bool:
        return self.action == OptionAction.BUY

    @property
    def is_short(self) -> bool:
        return self.action == OptionAction.SELL

    @property
    def signed_delta(self) -> float:
        if self.greeks is None:
            return 0.0
        sign = 1 if self.is_long else -1
        return self.greeks.delta * self.quantity * sign

    @property
    def premium(self) -> float:
        """Premium per contract (mid price * multiplier)."""
        if self.greeks is None:
            return 0.0
        sign = -1 if self.is_long else 1  # Pay for buys, receive for sells
        return sign * self.greeks.mid_price * self.contract.multiplier * self.quantity


@dataclass
class OptionsStrategy:
    """Complete options strategy with all legs."""
    name: str                 # e.g., "long_call", "bull_call_spread", "iron_condor"
    underlying: str           # Underlying symbol
    direction: StrategyDirection
    legs: List[OptionLeg] = field(default_factory=list)
    max_loss: float = 0.0     # Maximum possible loss (positive number)
    max_profit: float = 0.0   # Maximum possible profit
    breakeven: List[float] = field(default_factory=list)
    net_premium: float = 0.0  # Net premium (negative = debit, positive = credit)
    entry_time: datetime = field(default_factory=datetime.now)

    @property
    def net_delta(self) -> float:
        return sum(leg.signed_delta for leg in self.legs)

    @property
    def net_gamma(self) -> float:
        total = 0.0
        for leg in self.legs:
            if leg.greeks:
                sign = 1 if leg.is_long else -1
                total += leg.greeks.gamma * leg.quantity * sign
        return total

    @property
    def net_theta(self) -> float:
        total = 0.0
        for leg in self.legs:
            if leg.greeks:
                sign = 1 if leg.is_long else -1
                total += leg.greeks.theta * leg.quantity * sign
        return total

    @property
    def net_vega(self) -> float:
        total = 0.0
        for leg in self.legs:
            if leg.greeks:
                sign = 1 if leg.is_long else -1
                total += leg.greeks.vega * leg.quantity * sign
        return total

    @property
    def is_debit(self) -> bool:
        return self.net_premium < 0

    @property
    def is_credit(self) -> bool:
        return self.net_premium > 0

    @property
    def is_defined_risk(self) -> bool:
        return self.max_loss > 0 and self.max_loss < float('inf')

    @property
    def risk_reward_ratio(self) -> float:
        if self.max_loss > 0:
            return self.max_profit / self.max_loss
        return 0.0

    @property
    def num_legs(self) -> int:
        return len(self.legs)

    @property
    def expiry(self) -> str:
        if self.legs:
            return self.legs[0].contract.expiry
        return ""


@dataclass
class IVAnalysis:
    """Implied volatility analysis for strategy selection."""
    symbol: str
    current_iv: float = 0.0       # Current ATM implied volatility
    iv_rank: float = 0.0          # IV rank (0-100) - where current IV sits in 52w range
    iv_percentile: float = 0.0    # IV percentile (0-100) - % of days IV was lower
    hv_20: float = 0.0            # 20-day historical volatility
    iv_high_52w: float = 0.0      # 52-week IV high
    iv_low_52w: float = 0.0       # 52-week IV low
    iv_hv_ratio: float = 0.0      # IV / HV ratio (>1 = IV premium)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def regime(self) -> IVRegime:
        if self.iv_rank > 80:
            return IVRegime.VERY_HIGH
        elif self.iv_rank > 50:
            return IVRegime.HIGH
        elif self.iv_rank > 30:
            return IVRegime.NORMAL
        else:
            return IVRegime.LOW

    @property
    def has_iv_premium(self) -> bool:
        return self.iv_hv_ratio > 1.0


@dataclass
class OptionsPositionSizeResult:
    """Result of options position sizing calculation."""
    strategy_name: str
    contracts: int                 # Number of contracts (or spreads)
    max_risk: float                # Maximum risk in dollars
    premium_cost: float            # Total premium cost (debit strategies)
    premium_received: float = 0.0  # Total premium received (credit strategies)
    buying_power_required: float = 0.0
    risk_percent: float = 0.0     # Percent of account at risk
    reason: str = ""

    @property
    def net_cost(self) -> float:
        return self.premium_cost - self.premium_received
