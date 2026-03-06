"""
Post-Earnings Announcement Drift (PEAD) Strategy

Stocks with earnings surprises > 5% drift in the surprise direction for 60+ days.
(Ball & Brown 1968, confirmed across decades of research.)

Investors underreact to earnings information. Institutional rebalancing creates
sustained drift.

Entry: Day after earnings, surprise > 3%, gap in direction of surprise, volume > 2x avg
Exit:  10-day hold OR 2 ATR trailing stop
"""

from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from loguru import logger

from strategies.base_strategy import (
    BaseStrategy, StrategySignal, SignalDirection, SignalStrength,
)
from strategies.registry import StrategyRegistry

_REGIME_ALLOCATIONS = {
    'bull_trending': 0.10,
    'low_vol': 0.20,
    'bear_trending': 0.20,
    'high_vol': 0.10,
}


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


class PEADStrategy(BaseStrategy):
    """
    Post-Earnings Announcement Drift Strategy.

    Entry: Day after earnings announcement with surprise > 3%,
           gap in direction of surprise, volume > 2x average.
    Exit:  10-day hold OR 2 ATR trailing stop.

    This is an event-driven strategy. During scanning, it detects large
    overnight gaps with volume surges as potential earnings events.
    """

    def __init__(
        self,
        capital_allocation: float = 0.10,
        max_positions: int = 3,
        risk_per_trade: float = 0.02,
        min_gap_pct: float = 3.0,
        max_gap_pct: float = 25.0,
        volume_mult: float = 2.0,
        hold_days: int = 10,
        trailing_atr_mult: float = 2.0,
    ):
        super().__init__(
            name="pead",
            capital_allocation=capital_allocation,
            max_positions=max_positions,
            risk_per_trade=risk_per_trade,
        )
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.volume_mult = volume_mult
        self.hold_days = hold_days
        self.trailing_atr_mult = trailing_atr_mult

    def scan(
        self,
        stock_data: Dict[str, pd.DataFrame],
        market_data: pd.DataFrame,
        current_date: Optional[date] = None,
    ) -> List[StrategySignal]:
        """
        Scan for post-earnings drift opportunities.

        Detects earnings events by looking for large gaps with high volume.
        Without an earnings calendar API, we proxy earnings announcements as:
        - Overnight gap > min_gap_pct
        - Volume > volume_mult * 20-day average
        This catches most earnings reactions and filters out non-event gaps.
        """
        signals = []

        for symbol, data in stock_data.items():
            if symbol in self.positions:
                continue
            try:
                if current_date:
                    current = data[data.index.date <= current_date]
                else:
                    current = data

                if len(current) < 30:
                    continue

                current = current.copy()
                current.columns = [c.lower() for c in current.columns]

                close = current['close']
                open_price = current['open']
                volume = current['volume']

                prev_close = close.iloc[-2]
                today_open = open_price.iloc[-1]
                today_close = close.iloc[-1]
                today_volume = volume.iloc[-1]

                if prev_close <= 0:
                    continue

                # Calculate gap percentage
                gap_pct = ((today_open - prev_close) / prev_close) * 100

                # Filter: gap must be significant but not insane
                if abs(gap_pct) < self.min_gap_pct or abs(gap_pct) > self.max_gap_pct:
                    continue

                # Volume confirmation (earnings = high volume)
                avg_volume = volume.iloc[-21:-1].mean()
                if pd.isna(avg_volume) or avg_volume <= 0:
                    continue
                if today_volume < avg_volume * self.volume_mult:
                    continue

                # ATR for stops
                atr = _calculate_atr(current)
                current_atr = atr.iloc[-1]
                if pd.isna(current_atr) or current_atr <= 0:
                    continue

                # Direction: follow the gap direction (drift continues)
                if gap_pct > 0:
                    direction = SignalDirection.LONG
                    stop_price = today_close - (current_atr * self.trailing_atr_mult)
                    target_price = today_close + (current_atr * 3.0)
                else:
                    direction = SignalDirection.SHORT
                    stop_price = today_close + (current_atr * self.trailing_atr_mult)
                    target_price = today_close - (current_atr * 3.0)

                # Strength based on surprise magnitude
                abs_gap = abs(gap_pct)
                if abs_gap > 15:
                    strength = SignalStrength.VERY_STRONG
                elif abs_gap > 10:
                    strength = SignalStrength.STRONG
                else:
                    strength = SignalStrength.MODERATE

                signal = StrategySignal(
                    symbol=symbol,
                    direction=direction,
                    strength=strength,
                    strategy_name=self.name,
                    entry_price=today_close,
                    stop_price=stop_price,
                    target_price=target_price,
                    atr=current_atr,
                    risk_per_share=abs(today_close - stop_price),
                    suggested_position_pct=self.risk_per_trade,
                    additional_data={
                        'gap_pct': round(gap_pct, 2),
                        'volume_ratio': round(today_volume / avg_volume, 2),
                    },
                )

                if signal.is_valid:
                    signals.append(signal)

            except Exception as e:
                logger.debug(f"PEAD scan error for {symbol}: {e}")
                continue

        return signals

    def should_exit(
        self,
        symbol: str,
        position: Dict,
        current_data: pd.DataFrame,
        current_date: Optional[date] = None,
    ) -> Optional[str]:
        """Check PEAD exit conditions."""
        try:
            if current_date:
                up_to = current_data[current_data.index.date <= current_date]
            else:
                up_to = current_data

            if len(up_to) < 5:
                return None

            up_to = up_to.copy()
            up_to.columns = [c.lower() for c in up_to.columns]

            low = up_to['low'].iloc[-1]
            high = up_to['high'].iloc[-1]

            direction = position.get('direction', 'long')
            stop_price = position.get('stop_price', 0)

            # Trailing stop check
            if direction == 'long' and stop_price > 0 and low <= stop_price:
                return 'trailing_stop'
            elif direction == 'short' and stop_price > 0 and high >= stop_price:
                return 'trailing_stop'

            # Time-based exit
            entry_date = position.get('entry_date')
            if entry_date and current_date:
                holding_days = (current_date - entry_date).days
                if holding_days >= self.hold_days:
                    return 'time_stop'

        except Exception:
            pass

        return None

    def get_position_params(self, signal: StrategySignal) -> Dict:
        return {
            'stop_atr_mult': self.trailing_atr_mult,
            'target_atr_mult': 3.0,
            'time_stop_days': self.hold_days,
            'risk_per_trade': self.risk_per_trade,
            'trailing_stop': True,
        }

    def get_regime_allocation(self, regime: str) -> float:
        return _REGIME_ALLOCATIONS.get(regime, self.capital_allocation)


def create_default() -> PEADStrategy:
    """Create and register the default PEAD strategy."""
    strategy = PEADStrategy()
    StrategyRegistry.register(strategy.name, strategy)
    return strategy
