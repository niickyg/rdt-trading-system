"""
Gap Fill Reversion Strategy

Stocks gapping 2-4% against trend on no news tend to fill the gap within the
day (65-70% fill rate documented). Overnight gaps from liquidity imbalance,
not information, get faded by market makers.

Entry: Gap down 2-4% at open + no earnings catalyst + stock above 50 SMA + RSI(14) > 40
Exit:  Gap fill (previous close) OR 12:00 PM time stop OR 1 ATR stop loss
Universe: S&P 500, scanned at 9:35 AM ET only
"""

from datetime import date, time as dt_time
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from loguru import logger

from strategies.base_strategy import (
    BaseStrategy, StrategySignal, SignalDirection, SignalStrength,
)
from strategies.registry import StrategyRegistry

_REGIME_ALLOCATIONS = {
    'bull_trending': 0.0,
    'low_vol': 0.0,
    'bear_trending': 0.20,
    'high_vol': 0.40,
}


def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=period).mean()


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


class GapFillStrategy(BaseStrategy):
    """
    Gap Fill Reversion Strategy.

    Entry: Gap down 2-4% at open + stock above 50 SMA + RSI(14) > 40
    Exit:  Gap fill (previous close) OR intraday time stop OR 1 ATR stop loss

    This is an intraday strategy — positions should be closed by EOD.
    """

    def __init__(
        self,
        capital_allocation: float = 0.10,
        max_positions: int = 2,
        risk_per_trade: float = 0.015,
        min_gap_pct: float = 2.0,
        max_gap_pct: float = 4.0,
        sma_period: int = 50,
        rsi_threshold: float = 40.0,
        stop_atr_mult: float = 1.0,
    ):
        super().__init__(
            name="gap_fill",
            capital_allocation=capital_allocation,
            max_positions=max_positions,
            risk_per_trade=risk_per_trade,
        )
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.sma_period = sma_period
        self.rsi_threshold = rsi_threshold
        self.stop_atr_mult = stop_atr_mult

    def scan(
        self,
        stock_data: Dict[str, pd.DataFrame],
        market_data: pd.DataFrame,
        current_date: Optional[date] = None,
    ) -> List[StrategySignal]:
        """
        Scan for gap fill opportunities.

        Looks for stocks that gapped down 2-4% at open with no apparent news
        catalyst (proxied by volume < 3x average — extreme volume suggests news).
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

                if len(current) < self.sma_period + 10:
                    continue

                current = current.copy()
                current.columns = [c.lower() for c in current.columns]

                close = current['close']
                open_price = current['open']
                volume = current['volume']

                prev_close = close.iloc[-2]
                today_open = open_price.iloc[-1]
                today_close = close.iloc[-1]

                if prev_close <= 0:
                    continue

                # Gap down calculation
                gap_pct = ((today_open - prev_close) / prev_close) * 100

                # We want gap downs (negative gap) of 2-4%
                if gap_pct > -self.min_gap_pct or gap_pct < -self.max_gap_pct:
                    continue

                # Volume check: high volume suggests news (skip those)
                avg_volume = volume.iloc[-21:-1].mean()
                if pd.isna(avg_volume) or avg_volume <= 0:
                    continue
                today_volume = volume.iloc[-1]
                # Too high volume = news-driven gap (don't fade it)
                if today_volume > avg_volume * 3.0:
                    continue

                # Stock must be above 50 SMA (bullish bias — healthy stock)
                sma50 = _calculate_sma(close, self.sma_period)
                current_sma50 = sma50.iloc[-1]
                if pd.isna(current_sma50) or today_close <= current_sma50:
                    continue

                # RSI(14) > 40 (not already deeply oversold)
                rsi14 = _calculate_rsi(close, period=14)
                current_rsi14 = rsi14.iloc[-1]
                if pd.isna(current_rsi14) or current_rsi14 < self.rsi_threshold:
                    continue

                # ATR for stop
                atr = _calculate_atr(current)
                current_atr = atr.iloc[-1]
                if pd.isna(current_atr) or current_atr <= 0:
                    continue

                # Target = gap fill (previous close)
                target_price = prev_close
                stop_price = today_open - (current_atr * self.stop_atr_mult)

                # Use today_open as entry (we enter near the open)
                entry_price = today_open

                # Strength based on gap magnitude
                abs_gap = abs(gap_pct)
                if abs_gap > 3.5:
                    strength = SignalStrength.STRONG
                elif abs_gap > 2.5:
                    strength = SignalStrength.MODERATE
                else:
                    strength = SignalStrength.WEAK

                signal = StrategySignal(
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    strength=strength,
                    strategy_name=self.name,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    atr=current_atr,
                    risk_per_share=abs(entry_price - stop_price),
                    suggested_position_pct=self.risk_per_trade,
                    additional_data={
                        'gap_pct': round(gap_pct, 2),
                        'prev_close': round(prev_close, 2),
                        'rsi14': round(current_rsi14, 2),
                        'sma50': round(current_sma50, 2),
                    },
                )

                if signal.is_valid:
                    signals.append(signal)

            except Exception as e:
                logger.debug(f"Gap fill scan error for {symbol}: {e}")
                continue

        return signals

    def should_exit(
        self,
        symbol: str,
        position: Dict,
        current_data: pd.DataFrame,
        current_date: Optional[date] = None,
    ) -> Optional[str]:
        """Check gap fill exit conditions."""
        try:
            if current_date:
                day_data = current_data[current_data.index.date == current_date]
            else:
                day_data = current_data.tail(1)

            if len(day_data) == 0:
                return None

            day_data = day_data.copy()
            day_data.columns = [c.lower() for c in day_data.columns]

            high = day_data['high'].iloc[0]
            low = day_data['low'].iloc[0]

            # Gap fill target
            target_price = position.get('target_price', 0)
            if target_price > 0 and high >= target_price:
                return 'take_profit'

            # Stop loss
            stop_price = position.get('stop_price', 0)
            if stop_price > 0 and low <= stop_price:
                return 'stop_loss'

            # Intraday time stop — exit by end of day
            # In backtesting with daily bars, this triggers on the entry day
            entry_date = position.get('entry_date')
            if entry_date and current_date and current_date > entry_date:
                return 'time_stop'

        except Exception:
            pass

        return None

    def get_position_params(self, signal: StrategySignal) -> Dict:
        return {
            'stop_atr_mult': self.stop_atr_mult,
            'target_atr_mult': None,  # Target is gap fill, not ATR-based
            'time_stop_days': 1,  # Intraday only
            'risk_per_trade': self.risk_per_trade,
            'intraday': True,
        }

    def get_regime_allocation(self, regime: str) -> float:
        return _REGIME_ALLOCATIONS.get(regime, self.capital_allocation)


def create_default() -> GapFillStrategy:
    """Create and register the default gap fill strategy."""
    strategy = GapFillStrategy()
    StrategyRegistry.register(strategy.name, strategy)
    return strategy
