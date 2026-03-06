"""
RRS Momentum Strategy

Wraps the existing Real Relative Strength momentum scanner into the
BaseStrategy interface for multi-strategy integration.

This is a thin adapter — all signal logic lives in the existing
MomentumStrategy class and scanner code.
"""

from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from strategies.base_strategy import (
    BaseStrategy, StrategySignal, SignalDirection, SignalStrength,
)
from strategies.registry import StrategyRegistry

# Regime allocation table for RRS Momentum
_REGIME_ALLOCATIONS = {
    'bull_trending': 0.40,
    'low_vol': 0.30,
    'bear_trending': 0.10,
    'high_vol': 0.20,
}


class RRSMomentumStrategy(BaseStrategy):
    """
    Real Relative Strength Momentum Strategy.

    Entry: RRS > threshold + daily chart strength confirmed
    Exit:  Stop loss (ATR-based), take profit (ATR-based), or 10-day time stop
    """

    def __init__(
        self,
        capital_allocation: float = 0.40,
        max_positions: int = 10,
        risk_per_trade: float = 0.03,
        rrs_threshold: float = 1.75,
        stop_atr_mult: float = 0.75,
        target_atr_mult: float = 1.5,
    ):
        super().__init__(
            name="rrs_momentum",
            capital_allocation=capital_allocation,
            max_positions=max_positions,
            risk_per_trade=risk_per_trade,
        )
        self.rrs_threshold = rrs_threshold
        self.stop_atr_mult = stop_atr_mult
        self.target_atr_mult = target_atr_mult

        from shared.indicators.rrs import RRSCalculator, check_daily_strength_relaxed
        self.rrs_calc = RRSCalculator()
        self.check_daily = check_daily_strength_relaxed

    def scan(
        self,
        stock_data: Dict[str, pd.DataFrame],
        market_data: pd.DataFrame,
        current_date: Optional[date] = None,
    ) -> List[StrategySignal]:
        """Scan for RRS momentum signals."""
        signals = []

        if current_date:
            spy_current = market_data[market_data.index.date <= current_date]
        else:
            spy_current = market_data

        if len(spy_current) < 20:
            return signals

        close_col = 'close' if 'close' in spy_current.columns else 'Close'
        spy_close = spy_current[close_col].iloc[-1]
        spy_prev = spy_current[close_col].iloc[-2]

        for symbol, data in stock_data.items():
            if symbol in self.positions:
                continue
            try:
                if current_date:
                    current = data[data.index.date <= current_date]
                else:
                    current = data

                if len(current) < 20:
                    continue

                current = current.copy()
                current.columns = [c.lower() for c in current.columns]

                atr = self.rrs_calc.calculate_atr(current).iloc[-1]
                stock_close = current['close'].iloc[-1]
                stock_prev = current['close'].iloc[-2]

                rrs_result = self.rrs_calc.calculate_rrs_current(
                    stock_data={'current_price': stock_close, 'previous_close': stock_prev},
                    spy_data={'current_price': spy_close, 'previous_close': spy_prev},
                    stock_atr=atr,
                )
                rrs = rrs_result['rrs']

                if abs(rrs) < self.rrs_threshold:
                    continue

                daily_check = self.check_daily(current)

                if rrs > self.rrs_threshold and daily_check['is_strong']:
                    direction = SignalDirection.LONG
                    stop_price = stock_close - (atr * self.stop_atr_mult)
                    target_price = stock_close + (atr * self.target_atr_mult)
                elif rrs < -self.rrs_threshold:
                    from shared.indicators.rrs import check_daily_weakness_relaxed
                    weak_check = check_daily_weakness_relaxed(current)
                    if weak_check['is_weak']:
                        direction = SignalDirection.SHORT
                        stop_price = stock_close + (atr * self.stop_atr_mult)
                        target_price = stock_close - (atr * self.target_atr_mult)
                    else:
                        continue
                else:
                    continue

                if abs(rrs) > 3.0:
                    strength = SignalStrength.VERY_STRONG
                elif abs(rrs) > 2.5:
                    strength = SignalStrength.STRONG
                else:
                    strength = SignalStrength.MODERATE

                signal = StrategySignal(
                    symbol=symbol,
                    direction=direction,
                    strength=strength,
                    strategy_name=self.name,
                    entry_price=stock_close,
                    stop_price=stop_price,
                    target_price=target_price,
                    atr=atr,
                    risk_per_share=abs(stock_close - stop_price),
                    rrs_value=rrs,
                    suggested_position_pct=self.risk_per_trade,
                )

                if signal.is_valid:
                    signals.append(signal)

            except Exception:
                continue

        return signals

    def should_exit(
        self,
        symbol: str,
        position: Dict,
        current_data: pd.DataFrame,
        current_date: Optional[date] = None,
    ) -> Optional[str]:
        """Check exit conditions for RRS momentum."""
        try:
            if current_date:
                data = current_data[current_data.index.date == current_date]
            else:
                data = current_data.tail(1)

            if len(data) == 0:
                return None

            cols = {c.lower(): c for c in data.columns}
            high = data[cols.get('high', 'High')].iloc[0]
            low = data[cols.get('low', 'Low')].iloc[0]

            direction = position.get('direction', 'long')
            stop_price = position['stop_price']
            target_price = position['target_price']

            if direction == 'long':
                if low <= stop_price:
                    return 'stop_loss'
                if high >= target_price:
                    return 'take_profit'
            else:
                if high >= stop_price:
                    return 'stop_loss'
                if low <= target_price:
                    return 'take_profit'

            entry_date = position.get('entry_date')
            if entry_date and current_date:
                holding_days = (current_date - entry_date).days
                if holding_days >= 10:
                    return 'time_stop'

        except Exception:
            pass

        return None

    def get_position_params(self, signal: StrategySignal) -> Dict:
        return {
            'stop_atr_mult': self.stop_atr_mult,
            'target_atr_mult': self.target_atr_mult,
            'time_stop_days': 10,
            'risk_per_trade': self.risk_per_trade,
        }

    def get_regime_allocation(self, regime: str) -> float:
        return _REGIME_ALLOCATIONS.get(regime, self.capital_allocation)


# Auto-register when module is imported
def create_default() -> RRSMomentumStrategy:
    """Create and register the default RRS Momentum strategy."""
    strategy = RRSMomentumStrategy()
    StrategyRegistry.register(strategy.name, strategy)
    return strategy
