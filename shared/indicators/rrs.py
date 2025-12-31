"""
Real Relative Strength (RRS) Calculator
Based on r/RealDayTrading methodology

Formula: RRS = (PC - expectedPC) / ATR
Where:
- PC = Price Change (percent change of stock)
- expectedPC = Expected Price Change (based on SPY movement)
- ATR = Average True Range (volatility normalization)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class RRSCalculator:
    """Calculate Real Relative Strength for stocks relative to SPY"""

    def __init__(self, atr_period: int = 14):
        """
        Initialize RRS Calculator

        Args:
            atr_period: Period for ATR calculation (default 14)
        """
        self.atr_period = atr_period

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range

        Args:
            df: DataFrame with 'high', 'low', 'close' columns

        Returns:
            Series with ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        # True Range is the maximum of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is the moving average of True Range
        atr = tr.rolling(window=self.atr_period).mean()

        return atr

    def calculate_percent_change(self, df: pd.DataFrame, periods: int = 1) -> pd.Series:
        """
        Calculate percent change over specified periods

        Args:
            df: DataFrame with 'close' column
            periods: Number of periods to look back

        Returns:
            Series with percent change
        """
        return df['close'].pct_change(periods=periods) * 100

    def calculate_rrs(
        self,
        stock_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        periods: int = 1
    ) -> pd.DataFrame:
        """
        Calculate Real Relative Strength

        Args:
            stock_df: DataFrame with stock OHLCV data
            spy_df: DataFrame with SPY OHLCV data
            periods: Lookback period for price change (1 for 1-bar, 5 for 5-bar, etc.)

        Returns:
            DataFrame with RRS calculation and components
        """
        # Calculate ATR for stock
        stock_atr = self.calculate_atr(stock_df)

        # Calculate percent changes
        stock_pc = self.calculate_percent_change(stock_df, periods)
        spy_pc = self.calculate_percent_change(spy_df, periods)

        # Expected price change (what stock "should" do based on SPY)
        # This is simplified - in reality, you'd use beta or historical correlation
        expected_pc = spy_pc  # Assuming beta of 1.0 for simplicity

        # RRS calculation
        rrs = (stock_pc - expected_pc) / stock_atr

        # Create result DataFrame
        result = pd.DataFrame({
            'rrs': rrs,
            'stock_pc': stock_pc,
            'spy_pc': spy_pc,
            'expected_pc': expected_pc,
            'atr': stock_atr,
            'close': stock_df['close']
        })

        return result

    def calculate_rrs_current(
        self,
        stock_data: Dict,
        spy_data: Dict,
        stock_atr: float,
        periods: int = 1
    ) -> Dict:
        """
        Calculate RRS for current/live data (single values)

        Args:
            stock_data: Dict with 'current_price' and 'previous_close'
            spy_data: Dict with 'current_price' and 'previous_close'
            stock_atr: Pre-calculated ATR value for the stock
            periods: Lookback period

        Returns:
            Dict with RRS and components
        """
        # Calculate percent changes
        stock_pc = ((stock_data['current_price'] / stock_data['previous_close']) - 1) * 100
        spy_pc = ((spy_data['current_price'] / spy_data['previous_close']) - 1) * 100

        # Expected price change
        expected_pc = spy_pc

        # RRS
        if stock_atr > 0:
            rrs = (stock_pc - expected_pc) / stock_atr
        else:
            rrs = 0

        # Determine strength/weakness
        if rrs > 2.0:
            status = 'STRONG_RS'
        elif rrs > 0.5:
            status = 'MODERATE_RS'
        elif rrs > -0.5:
            status = 'NEUTRAL'
        elif rrs > -2.0:
            status = 'MODERATE_RW'
        else:
            status = 'STRONG_RW'

        return {
            'rrs': rrs,
            'stock_pc': stock_pc,
            'spy_pc': spy_pc,
            'expected_pc': expected_pc,
            'atr': stock_atr,
            'status': status,
            'current_price': stock_data['current_price']
        }

    def is_relative_strength(self, rrs: float, threshold: float = 0.5) -> bool:
        """Check if RRS indicates relative strength"""
        return rrs > threshold

    def is_relative_weakness(self, rrs: float, threshold: float = -0.5) -> bool:
        """Check if RRS indicates relative weakness"""
        return rrs < threshold

    def get_rrs_interpretation(self, rrs: float) -> str:
        """Get human-readable interpretation of RRS value"""
        if rrs > 2.0:
            return "Very Strong Relative Strength - Institutional buying likely"
        elif rrs > 0.5:
            return "Moderate Relative Strength - Outperforming market"
        elif rrs > -0.5:
            return "Neutral - Moving with market"
        elif rrs > -2.0:
            return "Moderate Relative Weakness - Underperforming market"
        else:
            return "Very Strong Relative Weakness - Institutional selling likely"


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return series.rolling(window=period).mean()


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns

    Returns:
        Series with VWAP values
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()


def check_daily_strength(df: pd.DataFrame) -> Dict:
    """
    Check if daily chart shows strength (RDT criteria)

    Criteria for bullish daily:
    - 3 green days in a row
    - 3 EMA > 8 EMA
    - Last close > 8 EMA

    Args:
        df: DataFrame with daily OHLCV data

    Returns:
        Dict with strength analysis
    """
    # Calculate EMAs
    ema3 = calculate_ema(df['close'], 3)
    ema8 = calculate_ema(df['close'], 8)

    # Check for 3 green days
    last_3_days = df.tail(3)
    three_green = all(last_3_days['close'] > last_3_days['open'])

    # Check EMA alignment
    ema_bullish = ema3.iloc[-1] > ema8.iloc[-1]

    # Check last close vs 8 EMA
    above_ema8 = df['close'].iloc[-1] > ema8.iloc[-1]

    is_strong = three_green and ema_bullish and above_ema8

    return {
        'is_strong': is_strong,
        'three_green_days': three_green,
        'ema3_above_ema8': ema_bullish,
        'above_ema8': above_ema8,
        'ema3': ema3.iloc[-1],
        'ema8': ema8.iloc[-1],
        'current_close': df['close'].iloc[-1]
    }


def check_daily_weakness(df: pd.DataFrame) -> Dict:
    """
    Check if daily chart shows weakness (RDT criteria)

    Criteria for bearish daily:
    - 3 red days in a row
    - 8 EMA > 3 EMA
    - Last close < 8 EMA

    Args:
        df: DataFrame with daily OHLCV data

    Returns:
        Dict with weakness analysis
    """
    # Calculate EMAs
    ema3 = calculate_ema(df['close'], 3)
    ema8 = calculate_ema(df['close'], 8)

    # Check for 3 red days
    last_3_days = df.tail(3)
    three_red = all(last_3_days['close'] < last_3_days['open'])

    # Check EMA alignment
    ema_bearish = ema8.iloc[-1] > ema3.iloc[-1]

    # Check last close vs 8 EMA
    below_ema8 = df['close'].iloc[-1] < ema8.iloc[-1]

    is_weak = three_red and ema_bearish and below_ema8

    return {
        'is_weak': is_weak,
        'three_red_days': three_red,
        'ema8_above_ema3': ema_bearish,
        'below_ema8': below_ema8,
        'ema3': ema3.iloc[-1],
        'ema8': ema8.iloc[-1],
        'current_close': df['close'].iloc[-1]
    }


def check_daily_strength_relaxed(df: pd.DataFrame, require_3_green: bool = False) -> Dict:
    """
    Check if daily chart shows strength with relaxed criteria (based on Hari's actual methodology)

    Relaxed Criteria (any 2 of 3):
    - 3 EMA > 8 EMA OR 8 EMA > 21 EMA
    - Close above 8 EMA
    - Recent higher lows (uptrend)

    Optional: 3 green days (not always required in practice)

    Args:
        df: DataFrame with daily OHLCV data
        require_3_green: Whether to require 3 green days

    Returns:
        Dict with strength analysis
    """
    # Calculate EMAs
    ema3 = calculate_ema(df['close'], 3)
    ema8 = calculate_ema(df['close'], 8)
    ema21 = calculate_ema(df['close'], 21)

    # Check for 3 green days (optional)
    last_3_days = df.tail(3)
    three_green = all(last_3_days['close'] > last_3_days['open'])

    # Check for 2+ green days (more lenient)
    green_days = sum(last_3_days['close'] > last_3_days['open'])
    two_plus_green = green_days >= 2

    # Check EMA alignment - short term
    ema3_above_ema8 = ema3.iloc[-1] > ema8.iloc[-1]

    # Check EMA alignment - medium term (bullish trend)
    ema8_above_ema21 = ema8.iloc[-1] > ema21.iloc[-1]

    # Check last close vs 8 EMA
    above_ema8 = df['close'].iloc[-1] > ema8.iloc[-1]

    # Check for higher lows (last 5 days)
    last_5_lows = df['low'].tail(5)
    higher_lows = len(last_5_lows) >= 3 and last_5_lows.iloc[-1] > last_5_lows.iloc[0]

    # Calculate strength score (0-5)
    strength_score = sum([
        ema3_above_ema8,
        ema8_above_ema21,
        above_ema8,
        higher_lows,
        two_plus_green
    ])

    # Relaxed criteria: score >= 3 (any 3 of 5 conditions)
    is_strong_relaxed = strength_score >= 3

    # Strict criteria: all conditions met
    is_strong_strict = three_green and ema3_above_ema8 and above_ema8

    # Use relaxed or strict based on require_3_green
    is_strong = is_strong_strict if require_3_green else is_strong_relaxed

    return {
        'is_strong': is_strong,
        'is_strong_relaxed': is_strong_relaxed,
        'is_strong_strict': is_strong_strict,
        'strength_score': strength_score,
        'three_green_days': three_green,
        'two_plus_green': two_plus_green,
        'ema3_above_ema8': ema3_above_ema8,
        'ema8_above_ema21': ema8_above_ema21,
        'above_ema8': above_ema8,
        'higher_lows': higher_lows,
        'ema3': ema3.iloc[-1],
        'ema8': ema8.iloc[-1],
        'ema21': ema21.iloc[-1],
        'current_close': df['close'].iloc[-1]
    }


def check_daily_weakness_relaxed(df: pd.DataFrame, require_3_red: bool = False) -> Dict:
    """
    Check if daily chart shows weakness with relaxed criteria

    Relaxed Criteria (any 2 of 3):
    - 8 EMA > 3 EMA OR 21 EMA > 8 EMA
    - Close below 8 EMA
    - Recent lower highs (downtrend)

    Args:
        df: DataFrame with daily OHLCV data
        require_3_red: Whether to require 3 red days

    Returns:
        Dict with weakness analysis
    """
    # Calculate EMAs
    ema3 = calculate_ema(df['close'], 3)
    ema8 = calculate_ema(df['close'], 8)
    ema21 = calculate_ema(df['close'], 21)

    # Check for 3 red days (optional)
    last_3_days = df.tail(3)
    three_red = all(last_3_days['close'] < last_3_days['open'])

    # Check for 2+ red days (more lenient)
    red_days = sum(last_3_days['close'] < last_3_days['open'])
    two_plus_red = red_days >= 2

    # Check EMA alignment - short term
    ema8_above_ema3 = ema8.iloc[-1] > ema3.iloc[-1]

    # Check EMA alignment - medium term (bearish trend)
    ema21_above_ema8 = ema21.iloc[-1] > ema8.iloc[-1]

    # Check last close vs 8 EMA
    below_ema8 = df['close'].iloc[-1] < ema8.iloc[-1]

    # Check for lower highs (last 5 days)
    last_5_highs = df['high'].tail(5)
    lower_highs = len(last_5_highs) >= 3 and last_5_highs.iloc[-1] < last_5_highs.iloc[0]

    # Calculate weakness score (0-5)
    weakness_score = sum([
        ema8_above_ema3,
        ema21_above_ema8,
        below_ema8,
        lower_highs,
        two_plus_red
    ])

    # Relaxed criteria: score >= 3 (any 3 of 5 conditions)
    is_weak_relaxed = weakness_score >= 3

    # Strict criteria: all conditions met
    is_weak_strict = three_red and ema8_above_ema3 and below_ema8

    # Use relaxed or strict based on require_3_red
    is_weak = is_weak_strict if require_3_red else is_weak_relaxed

    return {
        'is_weak': is_weak,
        'is_weak_relaxed': is_weak_relaxed,
        'is_weak_strict': is_weak_strict,
        'weakness_score': weakness_score,
        'three_red_days': three_red,
        'two_plus_red': two_plus_red,
        'ema8_above_ema3': ema8_above_ema3,
        'ema21_above_ema8': ema21_above_ema8,
        'below_ema8': below_ema8,
        'lower_highs': lower_highs,
        'ema3': ema3.iloc[-1],
        'ema8': ema8.iloc[-1],
        'ema21': ema21.iloc[-1],
        'current_close': df['close'].iloc[-1]
    }
