"""
RSI(2) Mean Reversion Strategy

Based on Connors RSI(2) research: buy when RSI(2) < 10, sell when RSI(2) > 95.
~75% historical win rate on S&P 500 stocks (< 10 threshold per Connors research).

Exploits short-term mean reversion in liquid large-cap stocks.
Negatively correlated with momentum (~-0.35), providing portfolio diversification.
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
    'bull_trending': 0.20,
    'low_vol': 0.40,
    'bear_trending': 0.30,
    'high_vol': 0.10,
}


def _calculate_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    """Calculate RSI for a given period."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


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


class RSI2MeanReversionStrategy(BaseStrategy):
    """
    RSI(2) Mean Reversion Strategy.

    Entry: RSI(2) < 10 + price above 200 SMA + relative volume > 0.5
    Exit:  RSI(2) > 65 OR 5-day time stop OR 1.5 ATR stop loss
    Universe: S&P 500 large caps only
    """

    def __init__(
        self,
        capital_allocation: float = 0.20,
        max_positions: int = 3,
        risk_per_trade: float = 0.02,
        rsi_entry_threshold: float = 10.0,
        rsi_exit_threshold: float = 65.0,
        sma_period: int = 200,
        stop_atr_mult: float = 1.5,
        time_stop_days: int = 5,
        min_relative_volume: float = 0.5,
    ):
        super().__init__(
            name="rsi2_mean_reversion",
            capital_allocation=capital_allocation,
            max_positions=max_positions,
            risk_per_trade=risk_per_trade,
        )
        self.rsi_entry_threshold = rsi_entry_threshold
        self.rsi_exit_threshold = rsi_exit_threshold
        self.sma_period = sma_period
        self.stop_atr_mult = stop_atr_mult
        self.time_stop_days = time_stop_days
        self.min_relative_volume = min_relative_volume

    def scan(
        self,
        stock_data: Dict[str, pd.DataFrame],
        market_data: pd.DataFrame,
        current_date: Optional[date] = None,
    ) -> List[StrategySignal]:
        """Scan for RSI(2) mean reversion signals."""
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
                rsi2 = _calculate_rsi(close, period=2)
                sma200 = _calculate_sma(close, self.sma_period)
                atr = _calculate_atr(current)

                current_close = close.iloc[-1]
                current_rsi2 = rsi2.iloc[-1]
                current_sma200 = sma200.iloc[-1]
                current_atr = atr.iloc[-1]

                if pd.isna(current_rsi2) or pd.isna(current_sma200) or pd.isna(current_atr):
                    continue
                if current_atr <= 0:
                    continue

                # LONG entry: RSI(2) < 10 + price above 200 SMA (bullish bias)
                if current_rsi2 >= self.rsi_entry_threshold:
                    continue
                if current_close <= current_sma200:
                    continue

                # Volume filter: at least min_relative_volume of 20-day avg
                vol = current['volume']
                avg_vol = vol.rolling(20).mean().iloc[-1]
                if pd.isna(avg_vol) or avg_vol <= 0:
                    continue
                relative_vol = vol.iloc[-1] / avg_vol
                if relative_vol < self.min_relative_volume:
                    continue

                # Build signal
                stop_price = current_close - (current_atr * self.stop_atr_mult)
                # Target: use RSI exit, but set a nominal target for R/R calc
                target_price = current_close + (current_atr * 2.0)

                # Strength based on how oversold
                if current_rsi2 < 2:
                    strength = SignalStrength.VERY_STRONG
                elif current_rsi2 < 3:
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
                        'rsi2': round(current_rsi2, 2),
                        'sma200': round(current_sma200, 2),
                        'relative_volume': round(relative_vol, 2),
                    },
                )

                if signal.is_valid:
                    signals.append(signal)

            except Exception as e:
                logger.debug(f"RSI2 scan error for {symbol}: {e}")
                continue

        return signals

    def should_exit(
        self,
        symbol: str,
        position: Dict,
        current_data: pd.DataFrame,
        current_date: Optional[date] = None,
    ) -> Optional[str]:
        """Check RSI2 exit conditions."""
        try:
            if current_date:
                up_to = current_data[current_data.index.date <= current_date]
            else:
                up_to = current_data

            if len(up_to) < 3:
                return None

            up_to = up_to.copy()
            up_to.columns = [c.lower() for c in up_to.columns]

            close = up_to['close']
            low = up_to['low'].iloc[-1]
            current_close = close.iloc[-1]

            # RSI(2) exit: take profit when RSI bounces back
            rsi2 = _calculate_rsi(close, period=2)
            current_rsi2 = rsi2.iloc[-1]
            if not pd.isna(current_rsi2) and current_rsi2 > self.rsi_exit_threshold:
                return 'take_profit'

            # Stop loss
            stop_price = position.get('stop_price', 0)
            if stop_price > 0 and low <= stop_price:
                return 'stop_loss'

            # Time stop
            entry_date = position.get('entry_date')
            if entry_date and current_date:
                holding_days = (current_date - entry_date).days
                if holding_days >= self.time_stop_days:
                    return 'time_stop'

        except Exception:
            pass

        return None

    def get_position_params(self, signal: StrategySignal) -> Dict:
        return {
            'stop_atr_mult': self.stop_atr_mult,
            'target_atr_mult': 2.0,
            'time_stop_days': self.time_stop_days,
            'risk_per_trade': self.risk_per_trade,
            'rsi_exit_threshold': self.rsi_exit_threshold,
        }

    def get_regime_allocation(self, regime: str) -> float:
        return _REGIME_ALLOCATIONS.get(regime, self.capital_allocation)


def create_default() -> RSI2MeanReversionStrategy:
    """Create and register the default RSI2 strategy."""
    strategy = RSI2MeanReversionStrategy()
    StrategyRegistry.register(strategy.name, strategy)
    return strategy
