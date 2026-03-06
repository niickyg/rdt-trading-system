"""
Trend Following (Breakout) Strategy

Turtle-style 20-day high breakout with volume confirmation.
100+ years of evidence across asset classes. "Crisis alpha" — performs best
when other strategies struggle.

Entry: Price breaks 20-day high + volume > 1.5x 20-day avg + ADX > 25
Exit:  Trailing stop at 2x ATR from high OR 10-day low (chandelier exit)
"""

from datetime import date
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from loguru import logger

from strategies.base_strategy import (
    BaseStrategy, StrategySignal, SignalDirection, SignalStrength,
)
from strategies.registry import StrategyRegistry

_REGIME_ALLOCATIONS = {
    'bull_trending': 0.30,
    'low_vol': 0.10,
    'bear_trending': 0.20,
    'high_vol': 0.20,
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


def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index."""
    high = df['high']
    low = df['low']
    close = df['close']

    # +DM and -DM
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smoothed averages
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr)

    # DX and ADX
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)
    dx = 100.0 * ((plus_di - minus_di).abs() / di_sum)
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return adx


class TrendBreakoutStrategy(BaseStrategy):
    """
    Trend Following Breakout Strategy.

    Entry: Price breaks 20-day high + volume > 1.5x avg + ADX > 25
    Exit:  Trailing 2 ATR chandelier stop OR 10-day low
    """

    def __init__(
        self,
        capital_allocation: float = 0.20,
        max_positions: int = 3,
        risk_per_trade: float = 0.02,
        breakout_period: int = 20,
        volume_mult: float = 1.5,
        adx_threshold: float = 25.0,
        trailing_atr_mult: float = 2.0,
        exit_low_period: int = 10,
    ):
        super().__init__(
            name="trend_breakout",
            capital_allocation=capital_allocation,
            max_positions=max_positions,
            risk_per_trade=risk_per_trade,
        )
        self.breakout_period = breakout_period
        self.volume_mult = volume_mult
        self.adx_threshold = adx_threshold
        self.trailing_atr_mult = trailing_atr_mult
        self.exit_low_period = exit_low_period

    def scan(
        self,
        stock_data: Dict[str, pd.DataFrame],
        market_data: pd.DataFrame,
        current_date: Optional[date] = None,
    ) -> List[StrategySignal]:
        """Scan for breakout signals."""
        signals = []

        for symbol, data in stock_data.items():
            if symbol in self.positions:
                continue
            try:
                if current_date:
                    current = data[data.index.date <= current_date]
                else:
                    current = data

                if len(current) < self.breakout_period + 20:
                    continue

                current = current.copy()
                current.columns = [c.lower() for c in current.columns]

                close = current['close']
                high = current['high']
                volume = current['volume']

                current_close = close.iloc[-1]
                current_high = high.iloc[-1]

                # 20-day high breakout (excluding today)
                period_high = high.iloc[-(self.breakout_period + 1):-1].max()
                if current_high <= period_high:
                    continue

                # Volume confirmation
                avg_volume = volume.iloc[-(self.breakout_period + 1):-1].mean()
                if pd.isna(avg_volume) or avg_volume <= 0:
                    continue
                current_volume = volume.iloc[-1]
                if current_volume < avg_volume * self.volume_mult:
                    continue

                # ADX filter (trending market)
                adx = _calculate_adx(current)
                current_adx = adx.iloc[-1]
                if pd.isna(current_adx) or current_adx < self.adx_threshold:
                    continue

                # ATR for stops
                atr = _calculate_atr(current)
                current_atr = atr.iloc[-1]
                if pd.isna(current_atr) or current_atr <= 0:
                    continue

                # Chandelier stop: 2 ATR below high
                stop_price = current_high - (current_atr * self.trailing_atr_mult)
                # Target: let winners run — set 3 ATR target for R/R calc
                target_price = current_close + (current_atr * 3.0)

                # Strength based on breakout magnitude and ADX
                breakout_pct = (current_high - period_high) / period_high * 100
                if current_adx > 40 and breakout_pct > 2:
                    strength = SignalStrength.VERY_STRONG
                elif current_adx > 30:
                    strength = SignalStrength.STRONG
                else:
                    strength = SignalStrength.MODERATE

                signal = StrategySignal(
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    strength=strength,
                    strategy_name=self.name,
                    entry_price=current_close,
                    stop_price=stop_price,
                    target_price=target_price,
                    atr=current_atr,
                    risk_per_share=abs(current_close - stop_price),
                    suggested_position_pct=self.risk_per_trade,
                    additional_data={
                        'adx': round(current_adx, 2),
                        'breakout_high': round(period_high, 2),
                        'volume_ratio': round(current_volume / avg_volume, 2),
                    },
                )

                if signal.is_valid:
                    signals.append(signal)

            except Exception as e:
                logger.debug(f"Trend breakout scan error for {symbol}: {e}")
                continue

        return signals

    def should_exit(
        self,
        symbol: str,
        position: Dict,
        current_data: pd.DataFrame,
        current_date: Optional[date] = None,
    ) -> Optional[str]:
        """Check trend breakout exit conditions."""
        try:
            if current_date:
                up_to = current_data[current_data.index.date <= current_date]
            else:
                up_to = current_data

            if len(up_to) < self.exit_low_period + 5:
                return None

            up_to = up_to.copy()
            up_to.columns = [c.lower() for c in up_to.columns]

            low = up_to['low'].iloc[-1]
            high = up_to['high']

            # Trailing stop: 2 ATR from highest high since entry
            entry_date = position.get('entry_date')
            if entry_date and current_date:
                since_entry = up_to[up_to.index.date >= entry_date] if hasattr(up_to.index, 'date') else up_to
                if len(since_entry) > 0:
                    highest_high = since_entry['high'].max()
                    atr = _calculate_atr(up_to)
                    current_atr = atr.iloc[-1]
                    if not pd.isna(current_atr) and current_atr > 0:
                        trailing_stop = highest_high - (current_atr * self.trailing_atr_mult)
                        if low <= trailing_stop:
                            return 'trailing_stop'

            # 10-day low exit (chandelier)
            period_low = up_to['low'].iloc[-(self.exit_low_period + 1):-1].min()
            if low <= period_low:
                return 'stop_loss'

        except Exception:
            pass

        return None

    def get_position_params(self, signal: StrategySignal) -> Dict:
        return {
            'stop_atr_mult': self.trailing_atr_mult,
            'target_atr_mult': 3.0,
            'time_stop_days': None,  # No time stop — let winners run
            'risk_per_trade': self.risk_per_trade,
            'trailing_stop': True,
            'exit_low_period': self.exit_low_period,
        }

    def get_regime_allocation(self, regime: str) -> float:
        return _REGIME_ALLOCATIONS.get(regime, self.capital_allocation)


def create_default() -> TrendBreakoutStrategy:
    """Create and register the default trend breakout strategy."""
    strategy = TrendBreakoutStrategy()
    StrategyRegistry.register(strategy.name, strategy)
    return strategy
