"""
Trend Detection Module for RDT Trading System
Detects trends using multiple methods for robust trend identification.

Methods:
- Moving Average Alignment (EMA 8/21/50/200)
- Higher Highs/Higher Lows (or Lower Highs/Lower Lows)
- ADX Trend Strength
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class TrendDirection(str, Enum):
    """Trend direction enumeration"""
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


@dataclass
class TrendResult:
    """Result of trend detection analysis"""
    direction: TrendDirection
    strength: float  # 0-100 scale
    age: int  # Number of bars trend has been active
    confidence: float  # 0-1 confidence in trend assessment
    details: Dict  # Additional analysis details

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'direction': self.direction.value,
            'strength': round(self.strength, 2),
            'age': self.age,
            'confidence': round(self.confidence, 3),
            'details': self.details
        }


class TrendDetector:
    """
    Multi-method trend detection for robust trend identification.

    Uses three primary methods:
    1. Moving Average Alignment (EMA 8/21/50/200)
    2. Higher Highs/Higher Lows structure
    3. ADX Trend Strength

    Combines all methods to produce a weighted trend assessment.
    """

    def __init__(
        self,
        ema_periods: List[int] = None,
        adx_period: int = 14,
        swing_lookback: int = 5,
        method_weights: Dict[str, float] = None
    ):
        """
        Initialize TrendDetector.

        Args:
            ema_periods: List of EMA periods for alignment analysis [8, 21, 50, 200]
            adx_period: Period for ADX calculation
            swing_lookback: Number of bars to look back for swing detection
            method_weights: Weights for each method (ma_alignment, swing_structure, adx)
        """
        self.ema_periods = ema_periods or [8, 21, 50, 200]
        self.adx_period = adx_period
        self.swing_lookback = swing_lookback
        self.method_weights = method_weights or {
            'ma_alignment': 0.4,
            'swing_structure': 0.35,
            'adx': 0.25
        }

    def detect_trend(self, df: pd.DataFrame) -> TrendResult:
        """
        Detect trend using all methods combined.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            TrendResult with direction, strength, age, confidence, and details
        """
        if df is None or len(df) < max(self.ema_periods):
            return TrendResult(
                direction=TrendDirection.NEUTRAL,
                strength=0,
                age=0,
                confidence=0,
                details={'error': 'Insufficient data'}
            )

        # Normalize column names
        df = self._normalize_columns(df)

        # Run each detection method
        ma_result = self._analyze_ma_alignment(df)
        swing_result = self._analyze_swing_structure(df)
        adx_result = self._analyze_adx(df)

        # Combine results with weights
        combined_score = (
            ma_result['score'] * self.method_weights['ma_alignment'] +
            swing_result['score'] * self.method_weights['swing_structure'] +
            adx_result['score'] * self.method_weights['adx']
        )

        # Determine direction from combined score
        direction = self._score_to_direction(combined_score)

        # Calculate strength (0-100)
        strength = min(100, abs(combined_score) * 50)

        # Calculate confidence based on method agreement
        confidence = self._calculate_confidence(ma_result, swing_result, adx_result)

        # Calculate trend age
        age = self._calculate_trend_age(df, direction)

        details = {
            'ma_alignment': ma_result,
            'swing_structure': swing_result,
            'adx': adx_result,
            'combined_score': round(combined_score, 3)
        }

        return TrendResult(
            direction=direction,
            strength=strength,
            age=age,
            confidence=confidence,
            details=details
        )

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase"""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        return df

    def _analyze_ma_alignment(self, df: pd.DataFrame) -> Dict:
        """
        Analyze Moving Average alignment.

        Perfect bullish alignment: EMA8 > EMA21 > EMA50 > EMA200
        Perfect bearish alignment: EMA8 < EMA21 < EMA50 < EMA200

        Returns dict with score (-2 to +2) and details.
        """
        emas = {}
        for period in self.ema_periods:
            if len(df) >= period:
                emas[period] = df['close'].ewm(span=period, adjust=False).mean().iloc[-1]
            else:
                emas[period] = df['close'].iloc[-1]

        # Check alignment
        sorted_periods = sorted(self.ema_periods)
        bullish_alignments = 0
        bearish_alignments = 0
        total_pairs = 0

        for i in range(len(sorted_periods) - 1):
            short_period = sorted_periods[i]
            long_period = sorted_periods[i + 1]
            total_pairs += 1

            if emas[short_period] > emas[long_period]:
                bullish_alignments += 1
            elif emas[short_period] < emas[long_period]:
                bearish_alignments += 1

        # Check price vs short-term EMA
        price_above_ema8 = df['close'].iloc[-1] > emas[self.ema_periods[0]]

        # Calculate score (-2 to +2)
        if bullish_alignments == total_pairs and price_above_ema8:
            score = 2.0
        elif bullish_alignments >= total_pairs * 0.75 and price_above_ema8:
            score = 1.5
        elif bullish_alignments > bearish_alignments:
            score = 0.5 + (bullish_alignments / total_pairs)
        elif bearish_alignments == total_pairs and not price_above_ema8:
            score = -2.0
        elif bearish_alignments >= total_pairs * 0.75 and not price_above_ema8:
            score = -1.5
        elif bearish_alignments > bullish_alignments:
            score = -0.5 - (bearish_alignments / total_pairs)
        else:
            score = 0.0

        return {
            'score': score,
            'bullish_alignments': bullish_alignments,
            'bearish_alignments': bearish_alignments,
            'total_pairs': total_pairs,
            'price_above_short_ema': price_above_ema8,
            'emas': {str(k): round(v, 2) for k, v in emas.items()}
        }

    def _analyze_swing_structure(self, df: pd.DataFrame) -> Dict:
        """
        Analyze swing structure (Higher Highs/Higher Lows or Lower Highs/Lower Lows).

        Returns dict with score (-2 to +2) and details.
        """
        highs = df['high'].values
        lows = df['low'].values

        # Find swing highs and lows
        swing_highs = self._find_swing_points(highs, self.swing_lookback, 'high')
        swing_lows = self._find_swing_points(lows, self.swing_lookback, 'low')

        # Analyze the structure
        hh_count = 0  # Higher highs
        lh_count = 0  # Lower highs
        hl_count = 0  # Higher lows
        ll_count = 0  # Lower lows

        # Compare last 3 swing highs
        if len(swing_highs) >= 2:
            for i in range(1, min(3, len(swing_highs))):
                if swing_highs[-(i)] > swing_highs[-(i+1)]:
                    hh_count += 1
                elif swing_highs[-(i)] < swing_highs[-(i+1)]:
                    lh_count += 1

        # Compare last 3 swing lows
        if len(swing_lows) >= 2:
            for i in range(1, min(3, len(swing_lows))):
                if swing_lows[-(i)] > swing_lows[-(i+1)]:
                    hl_count += 1
                elif swing_lows[-(i)] < swing_lows[-(i+1)]:
                    ll_count += 1

        # Calculate score
        bullish_points = hh_count + hl_count
        bearish_points = lh_count + ll_count

        if bullish_points >= 3 and bearish_points == 0:
            score = 2.0
        elif bullish_points > bearish_points:
            score = bullish_points / 2
        elif bearish_points >= 3 and bullish_points == 0:
            score = -2.0
        elif bearish_points > bullish_points:
            score = -bearish_points / 2
        else:
            score = 0.0

        return {
            'score': score,
            'higher_highs': hh_count,
            'lower_highs': lh_count,
            'higher_lows': hl_count,
            'lower_lows': ll_count,
            'swing_highs_count': len(swing_highs),
            'swing_lows_count': len(swing_lows)
        }

    def _find_swing_points(self, data: np.ndarray, lookback: int, point_type: str) -> List[float]:
        """
        Find swing points (highs or lows) in price data.

        Args:
            data: Price data array
            lookback: Number of bars to look back for swing detection
            point_type: 'high' or 'low'

        Returns:
            List of swing point values
        """
        swing_points = []

        for i in range(lookback, len(data) - lookback):
            if point_type == 'high':
                # Swing high: highest point in lookback window
                if data[i] == max(data[i - lookback:i + lookback + 1]):
                    swing_points.append(data[i])
            else:
                # Swing low: lowest point in lookback window
                if data[i] == min(data[i - lookback:i + lookback + 1]):
                    swing_points.append(data[i])

        return swing_points

    def _analyze_adx(self, df: pd.DataFrame) -> Dict:
        """
        Analyze ADX (Average Directional Index) for trend strength.

        ADX > 25: Strong trend
        ADX > 50: Very strong trend
        +DI > -DI: Bullish
        -DI > +DI: Bearish

        Returns dict with score (-2 to +2) and details.
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        n = len(df)
        if n < self.adx_period + 1:
            return {'score': 0, 'adx': 0, 'plus_di': 0, 'minus_di': 0, 'trend_strength': 'weak'}

        # Calculate True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        # Calculate +DM and -DM
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Smooth with Wilder's smoothing
        atr = self._wilder_smooth(tr, self.adx_period)
        plus_dm_smooth = self._wilder_smooth(plus_dm, self.adx_period)
        minus_dm_smooth = self._wilder_smooth(minus_dm, self.adx_period)

        # Calculate +DI and -DI
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)

        for i in range(self.adx_period, n):
            if atr[i] > 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]

        # Calculate DX and ADX
        dx = np.zeros(n)
        for i in range(self.adx_period, n):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

        adx = self._wilder_smooth(dx, self.adx_period)

        # Get current values
        current_adx = adx[-1] if len(adx) > 0 else 0
        current_plus_di = plus_di[-1] if len(plus_di) > 0 else 0
        current_minus_di = minus_di[-1] if len(minus_di) > 0 else 0

        # Determine trend strength
        if current_adx > 50:
            trend_strength = 'very_strong'
            strength_multiplier = 1.5
        elif current_adx > 25:
            trend_strength = 'strong'
            strength_multiplier = 1.0
        elif current_adx > 20:
            trend_strength = 'moderate'
            strength_multiplier = 0.6
        else:
            trend_strength = 'weak'
            strength_multiplier = 0.3

        # Calculate direction score
        if current_plus_di > current_minus_di:
            direction_score = (current_plus_di - current_minus_di) / 50  # Normalize
            score = min(2.0, direction_score * strength_multiplier)
        elif current_minus_di > current_plus_di:
            direction_score = (current_minus_di - current_plus_di) / 50
            score = max(-2.0, -direction_score * strength_multiplier)
        else:
            score = 0.0

        return {
            'score': score,
            'adx': round(current_adx, 2),
            'plus_di': round(current_plus_di, 2),
            'minus_di': round(current_minus_di, 2),
            'trend_strength': trend_strength
        }

    def _wilder_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Apply Wilder's smoothing method (used in ATR, ADX calculations)"""
        smoothed = np.zeros(len(data))
        smoothed[:period] = data[:period]

        # Initial value is simple average
        smoothed[period - 1] = np.mean(data[:period])

        # Wilder's smoothing: prev + (current - prev) / period
        for i in range(period, len(data)):
            smoothed[i] = smoothed[i - 1] + (data[i] - smoothed[i - 1]) / period

        return smoothed

    def _score_to_direction(self, score: float) -> TrendDirection:
        """Convert combined score to trend direction"""
        if score >= 1.5:
            return TrendDirection.STRONG_UP
        elif score >= 0.5:
            return TrendDirection.UP
        elif score <= -1.5:
            return TrendDirection.STRONG_DOWN
        elif score <= -0.5:
            return TrendDirection.DOWN
        else:
            return TrendDirection.NEUTRAL

    def _calculate_confidence(
        self,
        ma_result: Dict,
        swing_result: Dict,
        adx_result: Dict
    ) -> float:
        """
        Calculate confidence in trend assessment based on method agreement.

        Returns 0-1 confidence score.
        """
        scores = [ma_result['score'], swing_result['score'], adx_result['score']]

        # Check if all methods agree on direction
        positive = sum(1 for s in scores if s > 0)
        negative = sum(1 for s in scores if s < 0)
        neutral = sum(1 for s in scores if s == 0)

        if positive == 3 or negative == 3:
            # All methods agree
            base_confidence = 0.9
        elif positive == 2 or negative == 2:
            # 2 out of 3 agree
            base_confidence = 0.7
        elif neutral >= 2:
            # Mostly neutral
            base_confidence = 0.5
        else:
            # Mixed signals
            base_confidence = 0.4

        # Adjust for ADX strength
        adx_value = adx_result.get('adx', 0)
        if adx_value > 40:
            adx_boost = 0.1
        elif adx_value > 25:
            adx_boost = 0.05
        else:
            adx_boost = 0

        return min(1.0, base_confidence + adx_boost)

    def _calculate_trend_age(self, df: pd.DataFrame, direction: TrendDirection) -> int:
        """
        Calculate how long the current trend has been active (in bars).

        Uses EMA8 as the short-term trend indicator.
        """
        if len(df) < 2:
            return 0

        ema8 = df['close'].ewm(span=8, adjust=False).mean()

        is_bullish = direction in [TrendDirection.STRONG_UP, TrendDirection.UP]
        age = 0

        for i in range(len(df) - 1, 0, -1):
            if is_bullish:
                # Count bars where price is above EMA8
                if df['close'].iloc[i] > ema8.iloc[i]:
                    age += 1
                else:
                    break
            else:
                # Count bars where price is below EMA8
                if df['close'].iloc[i] < ema8.iloc[i]:
                    age += 1
                else:
                    break

        return age

    def get_ma_values(self, df: pd.DataFrame) -> Dict[int, float]:
        """
        Get current EMA values for all configured periods.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dict mapping period to EMA value
        """
        df = self._normalize_columns(df)
        emas = {}

        for period in self.ema_periods:
            if len(df) >= period:
                emas[period] = df['close'].ewm(span=period, adjust=False).mean().iloc[-1]
            else:
                emas[period] = df['close'].iloc[-1]

        return emas

    def get_adx_components(self, df: pd.DataFrame) -> Dict:
        """
        Get ADX components for external use.

        Returns:
            Dict with adx, plus_di, minus_di values
        """
        df = self._normalize_columns(df)
        return self._analyze_adx(df)
