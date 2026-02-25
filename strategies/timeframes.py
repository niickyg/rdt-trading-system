"""
Multi-Timeframe Trading Module

Supports three trading horizons:
- Short-term: Day trades and 1-5 day swings
- Medium-term: 1-4 week swing trades
- Long-term: 1-3 month position trades

Each timeframe has specific parameters for entry, exit, and position sizing.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger


class Timeframe(Enum):
    """Trading timeframe classifications"""
    SHORT = "short"      # Day trades / 1-5 day swings
    MEDIUM = "medium"    # 1-4 week swings
    LONG = "long"        # 1-3 month positions


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe"""
    timeframe: Timeframe
    name: str
    description: str

    # Holding period
    min_hold_days: int
    max_hold_days: int
    typical_hold_days: int

    # Entry criteria
    min_rrs_threshold: float      # Minimum relative strength
    min_ml_probability: float     # ML confidence threshold
    require_daily_alignment: bool # Must align with daily trend
    require_weekly_alignment: bool # Must align with weekly trend

    # Position sizing
    max_position_pct: float       # Max % of portfolio per trade
    risk_per_trade_pct: float     # % of portfolio risked per trade
    max_positions: int            # Max simultaneous positions

    # Stop/Target multipliers (based on ATR)
    stop_multiplier: float        # ATR multiplier for stop loss
    target_multiplier: float      # ATR multiplier for profit target
    trailing_stop: bool           # Use trailing stop
    trailing_atr_mult: float      # Trailing stop ATR multiplier

    # Exit rules
    time_stop_enabled: bool       # Exit after max hold period
    partial_profit_enabled: bool  # Take partial profits
    partial_profit_pct: float     # % to take at first target

    # Indicators to use
    primary_indicators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "timeframe": self.timeframe.value,
            "name": self.name,
            "min_hold_days": self.min_hold_days,
            "max_hold_days": self.max_hold_days,
            "stop_multiplier": self.stop_multiplier,
            "target_multiplier": self.target_multiplier,
            "max_positions": self.max_positions,
            "risk_per_trade_pct": self.risk_per_trade_pct
        }


# Pre-configured timeframe settings
SHORT_TERM_CONFIG = TimeframeConfig(
    timeframe=Timeframe.SHORT,
    name="Short-Term Swing",
    description="Day trades and 1-5 day swings. Quick profits, tight stops.",

    # Holding period
    min_hold_days=0,      # Can be same day
    max_hold_days=5,
    typical_hold_days=2,

    # Entry criteria - more selective for short-term
    min_rrs_threshold=2.5,        # Strong relative strength required
    min_ml_probability=75.0,      # High ML confidence
    require_daily_alignment=True,
    require_weekly_alignment=False,  # Don't need weekly alignment

    # Position sizing - smaller positions, more trades
    max_position_pct=5.0,         # 5% max per position
    risk_per_trade_pct=0.5,       # 0.5% risk per trade
    max_positions=8,              # Can have more positions

    # Stop/Target - tight for quick profits
    stop_multiplier=1.0,          # 1x ATR stop
    target_multiplier=1.0,        # 1x ATR target (1:1 R:R)
    trailing_stop=False,
    trailing_atr_mult=0.75,

    # Exit rules
    time_stop_enabled=True,       # Exit after 5 days
    partial_profit_enabled=False, # Take full profit
    partial_profit_pct=0.0,

    # Indicators
    primary_indicators=["rrs", "ema_3_8", "volume", "atr"]
)


MEDIUM_TERM_CONFIG = TimeframeConfig(
    timeframe=Timeframe.MEDIUM,
    name="Medium-Term Swing",
    description="1-4 week swing trades. Balanced risk/reward.",

    # Holding period
    min_hold_days=5,
    max_hold_days=28,
    typical_hold_days=14,

    # Entry criteria - moderate selectivity
    min_rrs_threshold=2.0,        # Moderate relative strength
    min_ml_probability=70.0,      # Good ML confidence
    require_daily_alignment=True,
    require_weekly_alignment=True,  # Need weekly trend alignment

    # Position sizing - moderate positions
    max_position_pct=8.0,         # 8% max per position
    risk_per_trade_pct=1.0,       # 1% risk per trade
    max_positions=5,

    # Stop/Target - wider for trend capture
    stop_multiplier=1.5,          # 1.5x ATR stop
    target_multiplier=2.0,        # 2x ATR target (1:1.33 R:R)
    trailing_stop=True,
    trailing_atr_mult=1.5,

    # Exit rules
    time_stop_enabled=True,       # Exit after 28 days
    partial_profit_enabled=True,  # Take partials
    partial_profit_pct=0.5,       # Take 50% at first target

    # Indicators
    primary_indicators=["rrs", "ema_8_21", "macd", "weekly_trend", "volume"]
)


LONG_TERM_CONFIG = TimeframeConfig(
    timeframe=Timeframe.LONG,
    name="Long-Term Position",
    description="1-3 month position trades. Maximum trend capture.",

    # Holding period
    min_hold_days=20,
    max_hold_days=90,
    typical_hold_days=45,

    # Entry criteria - more patient
    min_rrs_threshold=1.5,        # Can enter earlier in move
    min_ml_probability=65.0,      # Lower threshold OK
    require_daily_alignment=True,
    require_weekly_alignment=True,

    # Position sizing - larger positions, fewer trades
    max_position_pct=12.0,        # 12% max per position
    risk_per_trade_pct=1.5,       # 1.5% risk per trade
    max_positions=3,              # Concentrated portfolio

    # Stop/Target - very wide for major moves
    stop_multiplier=2.5,          # 2.5x ATR stop
    target_multiplier=5.0,        # 5x ATR target (1:2 R:R)
    trailing_stop=True,
    trailing_atr_mult=2.0,

    # Exit rules
    time_stop_enabled=True,       # Exit after 90 days
    partial_profit_enabled=True,  # Take partials
    partial_profit_pct=0.33,      # Take 33% at each target

    # Indicators
    primary_indicators=["rrs", "ema_21_50", "weekly_ema", "monthly_trend", "relative_strength"]
)


# All timeframe configs
TIMEFRAME_CONFIGS = {
    Timeframe.SHORT: SHORT_TERM_CONFIG,
    Timeframe.MEDIUM: MEDIUM_TERM_CONFIG,
    Timeframe.LONG: LONG_TERM_CONFIG
}


class TimeframeManager:
    """
    Manages multi-timeframe trading decisions

    Features:
    - Classifies signals by appropriate timeframe
    - Adjusts parameters based on timeframe
    - Tracks positions by timeframe
    - Enforces timeframe-specific rules
    """

    def __init__(
        self,
        enabled_timeframes: List[Timeframe] = None,
        prefer_timeframe: Timeframe = None
    ):
        """
        Initialize timeframe manager

        Args:
            enabled_timeframes: Which timeframes to trade (default: all)
            prefer_timeframe: Preferred timeframe when signal matches multiple
        """
        self.enabled_timeframes = enabled_timeframes or list(Timeframe)
        self.prefer_timeframe = prefer_timeframe or Timeframe.MEDIUM

        # Position tracking by timeframe
        self.positions_by_timeframe: Dict[Timeframe, List[str]] = {
            tf: [] for tf in Timeframe
        }

        logger.info(f"TimeframeManager initialized with: {[tf.value for tf in self.enabled_timeframes]}")

    def classify_signal(
        self,
        signal: Dict,
        weekly_trend: Optional[str] = None,
        monthly_trend: Optional[str] = None
    ) -> Tuple[Timeframe, TimeframeConfig]:
        """
        Classify a signal to the best timeframe

        Args:
            signal: Trading signal data
            weekly_trend: Weekly trend direction ('up', 'down', 'neutral')
            monthly_trend: Monthly trend direction

        Returns:
            (timeframe, config) tuple
        """
        rrs = abs(signal.get("rrs", 0))
        ml_prob = signal.get("ml_probability", 0)
        direction = signal.get("direction", "long")

        # Score each timeframe based on signal characteristics
        scores = {}

        for tf in self.enabled_timeframes:
            config = TIMEFRAME_CONFIGS[tf]
            score = 0

            # Check minimum thresholds
            if rrs < config.min_rrs_threshold:
                continue
            if ml_prob > 0 and ml_prob < config.min_ml_probability:
                continue

            # Check weekly alignment if required
            if config.require_weekly_alignment:
                if weekly_trend:
                    if direction == "long" and weekly_trend != "up":
                        continue
                    if direction == "short" and weekly_trend != "down":
                        continue

            # Score based on RRS strength
            rrs_score = min(rrs / config.min_rrs_threshold, 2.0) * 30
            score += rrs_score

            # Score based on ML probability
            if ml_prob > 0:
                ml_score = min(ml_prob / config.min_ml_probability, 1.5) * 30
                score += ml_score

            # Bonus for weekly trend alignment
            if weekly_trend:
                if direction == "long" and weekly_trend == "up":
                    score += 20
                elif direction == "short" and weekly_trend == "down":
                    score += 20

            # Bonus for monthly trend alignment (favors longer timeframes)
            if monthly_trend:
                if direction == "long" and monthly_trend == "up":
                    score += 15 if tf == Timeframe.LONG else 10
                elif direction == "short" and monthly_trend == "down":
                    score += 15 if tf == Timeframe.LONG else 10

            # Very strong signals favor short-term
            if rrs > 3.0 and ml_prob > 80:
                if tf == Timeframe.SHORT:
                    score += 15

            # Position availability bonus
            config_max = config.max_positions
            current = len(self.positions_by_timeframe[tf])
            if current < config_max:
                availability = (config_max - current) / config_max * 10
                score += availability
            else:
                score -= 50  # Penalty for full positions

            scores[tf] = score

        if not scores:
            # Default to preferred timeframe
            return self.prefer_timeframe, TIMEFRAME_CONFIGS[self.prefer_timeframe]

        # Return highest scoring timeframe
        best_tf = max(scores, key=scores.get)
        return best_tf, TIMEFRAME_CONFIGS[best_tf]

    def get_position_params(
        self,
        timeframe: Timeframe,
        entry_price: float,
        atr: float,
        direction: str = "long"
    ) -> Dict:
        """
        Get position sizing and exit parameters for a timeframe

        Returns:
            Dict with stop_price, target_price, position_pct, etc.
        """
        config = TIMEFRAME_CONFIGS[timeframe]

        if direction == "long":
            stop_price = entry_price - (atr * config.stop_multiplier)
            target_price = entry_price + (atr * config.target_multiplier)
            if config.trailing_stop:
                trailing_stop = entry_price - (atr * config.trailing_atr_mult)
            else:
                trailing_stop = None
        else:
            stop_price = entry_price + (atr * config.stop_multiplier)
            target_price = entry_price - (atr * config.target_multiplier)
            if config.trailing_stop:
                trailing_stop = entry_price + (atr * config.trailing_atr_mult)
            else:
                trailing_stop = None

        return {
            "timeframe": timeframe.value,
            "stop_price": round(stop_price, 2),
            "target_price": round(target_price, 2),
            "stop_multiplier": config.stop_multiplier,
            "target_multiplier": config.target_multiplier,
            "trailing_stop": trailing_stop,
            "trailing_enabled": config.trailing_stop,
            "max_hold_days": config.max_hold_days,
            "time_stop_enabled": config.time_stop_enabled,
            "partial_profit_enabled": config.partial_profit_enabled,
            "partial_profit_pct": config.partial_profit_pct,
            "risk_per_trade_pct": config.risk_per_trade_pct,
            "max_position_pct": config.max_position_pct
        }

    def can_open_position(self, timeframe: Timeframe, symbol: str) -> Tuple[bool, str]:
        """Check if a new position can be opened for this timeframe"""
        config = TIMEFRAME_CONFIGS[timeframe]
        current = len(self.positions_by_timeframe[timeframe])

        if current >= config.max_positions:
            return False, f"{timeframe.value} max positions ({config.max_positions}) reached"

        if symbol in self.positions_by_timeframe[timeframe]:
            return False, f"Already have {timeframe.value} position in {symbol}"

        return True, "OK"

    def register_position(self, timeframe: Timeframe, symbol: str):
        """Register a new position"""
        if symbol not in self.positions_by_timeframe[timeframe]:
            self.positions_by_timeframe[timeframe].append(symbol)
            logger.info(f"Registered {timeframe.value} position: {symbol}")

    def close_position(self, symbol: str, timeframe: Optional[Timeframe] = None):
        """Close a position"""
        if timeframe:
            if symbol in self.positions_by_timeframe[timeframe]:
                self.positions_by_timeframe[timeframe].remove(symbol)
        else:
            # Search all timeframes
            for tf in Timeframe:
                if symbol in self.positions_by_timeframe[tf]:
                    self.positions_by_timeframe[tf].remove(symbol)
                    logger.info(f"Closed {tf.value} position: {symbol}")
                    return

    def should_exit_time_stop(
        self,
        timeframe: Timeframe,
        entry_time: datetime
    ) -> Tuple[bool, str]:
        """Check if position should exit due to time stop"""
        config = TIMEFRAME_CONFIGS[timeframe]

        if not config.time_stop_enabled:
            return False, "Time stop disabled"

        days_held = (datetime.now() - entry_time).days

        if days_held >= config.max_hold_days:
            return True, f"Max hold period ({config.max_hold_days} days) exceeded"

        return False, f"Held {days_held}/{config.max_hold_days} days"

    def get_trailing_stop(
        self,
        timeframe: Timeframe,
        current_price: float,
        highest_price: float,
        atr: float,
        direction: str
    ) -> Optional[float]:
        """Calculate trailing stop price"""
        config = TIMEFRAME_CONFIGS[timeframe]

        if not config.trailing_stop:
            return None

        if direction == "long":
            return round(highest_price - (atr * config.trailing_atr_mult), 2)
        else:
            return round(highest_price + (atr * config.trailing_atr_mult), 2)

    def get_status(self) -> Dict:
        """Get current status"""
        status = {}
        for tf in Timeframe:
            config = TIMEFRAME_CONFIGS[tf]
            positions = self.positions_by_timeframe[tf]
            status[tf.value] = {
                "positions": positions,
                "count": len(positions),
                "max": config.max_positions,
                "available": config.max_positions - len(positions)
            }
        return status


# Global instance
_timeframe_manager: Optional[TimeframeManager] = None


def get_timeframe_manager() -> TimeframeManager:
    """Get global timeframe manager instance"""
    global _timeframe_manager
    if _timeframe_manager is None:
        _timeframe_manager = TimeframeManager()
    return _timeframe_manager


def get_timeframe_config(timeframe: Timeframe) -> TimeframeConfig:
    """Get config for a specific timeframe"""
    return TIMEFRAME_CONFIGS[timeframe]
