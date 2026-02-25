"""
Multi-Timeframe Analyzer for RDT Trading System
Analyzes multiple timeframes to identify trend alignment and optimal entry points.

Supported Timeframes: 5m, 15m, 1h, 4h, daily
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

from scanner.trend_detector import TrendDetector, TrendResult, TrendDirection


class Timeframe(str, Enum):
    """Supported timeframes"""
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    DAILY = "1d"

    @classmethod
    def from_string(cls, value: str) -> 'Timeframe':
        """Convert string to Timeframe enum"""
        mapping = {
            '5m': cls.M5, '5min': cls.M5,
            '15m': cls.M15, '15min': cls.M15,
            '1h': cls.H1, '60m': cls.H1, '60min': cls.H1,
            '4h': cls.H4, '240m': cls.H4, '240min': cls.H4,
            '1d': cls.DAILY, 'daily': cls.DAILY, 'd': cls.DAILY, 'day': cls.DAILY
        }
        return mapping.get(value.lower(), cls.DAILY)

    @property
    def yfinance_interval(self) -> str:
        """Get yfinance interval string"""
        return self.value

    @property
    def period_for_data(self) -> str:
        """Get appropriate data period for this timeframe"""
        periods = {
            Timeframe.M5: "5d",
            Timeframe.M15: "10d",
            Timeframe.H1: "30d",
            Timeframe.H4: "60d",
            Timeframe.DAILY: "1y"
        }
        return periods.get(self, "60d")

    @property
    def minutes(self) -> int:
        """Get timeframe in minutes"""
        minutes_map = {
            Timeframe.M5: 5,
            Timeframe.M15: 15,
            Timeframe.H1: 60,
            Timeframe.H4: 240,
            Timeframe.DAILY: 1440
        }
        return minutes_map.get(self, 1440)


@dataclass
class SupportResistanceLevel:
    """Support or resistance level"""
    price: float
    strength: float  # 0-1, how strong the level is (based on touches)
    level_type: str  # 'support' or 'resistance'
    touches: int  # Number of times price touched this level
    last_touch_bars_ago: int  # How many bars ago was the last touch

    def to_dict(self) -> Dict:
        return {
            'price': round(self.price, 2),
            'strength': round(self.strength, 3),
            'type': self.level_type,
            'touches': self.touches,
            'last_touch_bars_ago': self.last_touch_bars_ago
        }


@dataclass
class TimeframeAnalysis:
    """Analysis result for a single timeframe"""
    timeframe: Timeframe
    trend: TrendResult
    support_levels: List[SupportResistanceLevel]
    resistance_levels: List[SupportResistanceLevel]
    bias: str  # 'bullish', 'bearish', 'neutral'
    key_level_nearby: Optional[float]  # Nearest significant level

    def to_dict(self) -> Dict:
        return {
            'timeframe': self.timeframe.value,
            'trend': self.trend.to_dict(),
            'support_levels': [s.to_dict() for s in self.support_levels[:3]],
            'resistance_levels': [r.to_dict() for r in self.resistance_levels[:3]],
            'bias': self.bias,
            'key_level_nearby': round(self.key_level_nearby, 2) if self.key_level_nearby else None
        }


@dataclass
class MTFAnalysisResult:
    """Complete multi-timeframe analysis result"""
    symbol: str
    timestamp: datetime
    timeframe_analyses: Dict[str, TimeframeAnalysis]
    timeframe_alignment: bool
    alignment_direction: str  # 'bullish', 'bearish', 'mixed'
    trend_by_timeframe: Dict[str, str]
    entry_timing_score: int  # 0-100
    recommended_action: str
    key_levels: Dict[str, List[float]]

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'timeframe_alignment': self.timeframe_alignment,
            'alignment_direction': self.alignment_direction,
            'trend_by_timeframe': self.trend_by_timeframe,
            'entry_timing_score': self.entry_timing_score,
            'recommended_action': self.recommended_action,
            'key_levels': self.key_levels,
            'analyses': {k: v.to_dict() for k, v in self.timeframe_analyses.items()}
        }


class TimeframeAnalyzer:
    """
    Multi-timeframe analysis for the RDT Trading System.

    Analyzes trends across multiple timeframes to identify:
    - Trend alignment (all timeframes agreeing)
    - Support and resistance levels
    - Optimal entry timing based on lower timeframe setups
    """

    # Default timeframes to analyze
    DEFAULT_TIMEFRAMES = [Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.DAILY]

    def __init__(
        self,
        timeframes: List[Timeframe] = None,
        trend_detector: TrendDetector = None,
        sr_tolerance: float = 0.002,  # 0.2% tolerance for S/R level clustering
        min_sr_touches: int = 2
    ):
        """
        Initialize TimeframeAnalyzer.

        Args:
            timeframes: List of timeframes to analyze
            trend_detector: TrendDetector instance (created if not provided)
            sr_tolerance: Percentage tolerance for S/R level clustering
            min_sr_touches: Minimum touches for a valid S/R level
        """
        self.timeframes = timeframes or self.DEFAULT_TIMEFRAMES
        self.trend_detector = trend_detector or TrendDetector()
        self.sr_tolerance = sr_tolerance
        self.min_sr_touches = min_sr_touches

        # Data cache for multi-timeframe data
        self._data_cache: Dict[str, Dict[Timeframe, pd.DataFrame]] = {}

    def analyze_trend(
        self,
        symbol: str,
        data_by_timeframe: Dict[Timeframe, pd.DataFrame],
        timeframes: List[Timeframe] = None
    ) -> Dict[str, TrendResult]:
        """
        Analyze trend direction for each timeframe.

        Args:
            symbol: Stock symbol
            data_by_timeframe: Dict mapping Timeframe to DataFrame
            timeframes: Optional list of specific timeframes to analyze

        Returns:
            Dict mapping timeframe string to TrendResult
        """
        timeframes_to_analyze = timeframes or self.timeframes
        results = {}

        for tf in timeframes_to_analyze:
            if tf not in data_by_timeframe:
                logger.debug(f"No data for {symbol} on {tf.value}")
                continue

            df = data_by_timeframe[tf]
            if df is None or df.empty:
                continue

            try:
                trend_result = self.trend_detector.detect_trend(df)
                results[tf.value] = trend_result
            except Exception as e:
                logger.warning(f"Error analyzing trend for {symbol} on {tf.value}: {e}")
                continue

        return results

    def get_support_resistance(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframe: Timeframe = Timeframe.DAILY
    ) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        """
        Calculate support and resistance levels for a symbol.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            timeframe: Timeframe of the data

        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        if df is None or len(df) < 20:
            return [], []

        # Normalize columns
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        current_price = df['close'].iloc[-1]
        highs = df['high'].values
        lows = df['low'].values

        # Find swing points for S/R levels
        support_prices = self._find_sr_levels(lows, 'support')
        resistance_prices = self._find_sr_levels(highs, 'resistance')

        # Cluster nearby levels
        support_levels = self._cluster_and_score_levels(
            support_prices, df, 'support', current_price
        )
        resistance_levels = self._cluster_and_score_levels(
            resistance_prices, df, 'resistance', current_price
        )

        # Sort by strength
        support_levels.sort(key=lambda x: x.strength, reverse=True)
        resistance_levels.sort(key=lambda x: x.strength, reverse=True)

        return support_levels, resistance_levels

    def _find_sr_levels(self, data: np.ndarray, level_type: str) -> List[float]:
        """Find potential support/resistance levels from price data"""
        levels = []
        lookback = 5

        for i in range(lookback, len(data) - lookback):
            if level_type == 'support':
                # Local minimum
                if data[i] == min(data[i - lookback:i + lookback + 1]):
                    levels.append(data[i])
            else:
                # Local maximum
                if data[i] == max(data[i - lookback:i + lookback + 1]):
                    levels.append(data[i])

        return levels

    def _cluster_and_score_levels(
        self,
        prices: List[float],
        df: pd.DataFrame,
        level_type: str,
        current_price: float
    ) -> List[SupportResistanceLevel]:
        """
        Cluster nearby price levels and score them by strength.
        """
        if not prices:
            return []

        # Sort prices
        prices = sorted(prices)

        # Cluster nearby prices
        clusters = []
        current_cluster = [prices[0]]

        for price in prices[1:]:
            # If price is within tolerance of cluster average, add to cluster
            cluster_avg = sum(current_cluster) / len(current_cluster)
            if abs(price - cluster_avg) / cluster_avg <= self.sr_tolerance:
                current_cluster.append(price)
            else:
                if len(current_cluster) >= self.min_sr_touches:
                    clusters.append(current_cluster)
                current_cluster = [price]

        if len(current_cluster) >= self.min_sr_touches:
            clusters.append(current_cluster)

        # Create SupportResistanceLevel objects
        levels = []
        for cluster in clusters:
            avg_price = sum(cluster) / len(cluster)
            touches = len(cluster)

            # Calculate strength based on touches and recency
            # Find last touch
            last_touch_idx = self._find_last_touch(df, avg_price, level_type)
            last_touch_bars_ago = len(df) - 1 - last_touch_idx if last_touch_idx >= 0 else len(df)

            # Strength: more touches = stronger, more recent = stronger
            touch_score = min(1.0, touches / 5)
            recency_score = max(0, 1 - (last_touch_bars_ago / len(df)))
            strength = touch_score * 0.7 + recency_score * 0.3

            # Only include levels near current price (within 10%)
            if abs(avg_price - current_price) / current_price <= 0.10:
                levels.append(SupportResistanceLevel(
                    price=avg_price,
                    strength=strength,
                    level_type=level_type,
                    touches=touches,
                    last_touch_bars_ago=last_touch_bars_ago
                ))

        return levels

    def _find_last_touch(
        self,
        df: pd.DataFrame,
        level_price: float,
        level_type: str
    ) -> int:
        """Find the index of the last time price touched a level"""
        tolerance = level_price * self.sr_tolerance

        for i in range(len(df) - 1, -1, -1):
            if level_type == 'support':
                if abs(df['low'].iloc[i] - level_price) <= tolerance:
                    return i
            else:
                if abs(df['high'].iloc[i] - level_price) <= tolerance:
                    return i

        return -1

    def check_timeframe_alignment(
        self,
        symbol: str,
        data_by_timeframe: Dict[Timeframe, pd.DataFrame],
        required_agreement: float = 0.6
    ) -> Tuple[bool, str, Dict[str, str]]:
        """
        Check if all timeframes are aligned in the same trend direction.

        Args:
            symbol: Stock symbol
            data_by_timeframe: Dict mapping Timeframe to DataFrame
            required_agreement: Percentage of timeframes that must agree (0-1)

        Returns:
            Tuple of (is_aligned, alignment_direction, trend_by_timeframe)
        """
        trend_results = self.analyze_trend(symbol, data_by_timeframe)

        if not trend_results:
            return False, 'unknown', {}

        # Count trend directions
        bullish_count = 0
        bearish_count = 0
        trend_by_tf = {}

        for tf_str, trend_result in trend_results.items():
            direction = trend_result.direction

            if direction in [TrendDirection.STRONG_UP, TrendDirection.UP]:
                bullish_count += 1
                trend_by_tf[tf_str] = 'bullish'
            elif direction in [TrendDirection.STRONG_DOWN, TrendDirection.DOWN]:
                bearish_count += 1
                trend_by_tf[tf_str] = 'bearish'
            else:
                trend_by_tf[tf_str] = 'neutral'

        total = len(trend_results)
        bullish_pct = bullish_count / total if total > 0 else 0
        bearish_pct = bearish_count / total if total > 0 else 0

        if bullish_pct >= required_agreement:
            return True, 'bullish', trend_by_tf
        elif bearish_pct >= required_agreement:
            return True, 'bearish', trend_by_tf
        else:
            return False, 'mixed', trend_by_tf

    def get_entry_timing(
        self,
        symbol: str,
        data_by_timeframe: Dict[Timeframe, pd.DataFrame],
        direction: str = 'long'
    ) -> Dict[str, Any]:
        """
        Analyze lower timeframes for optimal entry timing.

        For longs: Look for pullback to support, bullish momentum on 5m/15m
        For shorts: Look for rally to resistance, bearish momentum on 5m/15m

        Args:
            symbol: Stock symbol
            data_by_timeframe: Dict mapping Timeframe to DataFrame
            direction: 'long' or 'short'

        Returns:
            Dict with entry_timing_score (0-100) and analysis details
        """
        result = {
            'entry_timing_score': 50,
            'recommended_action': 'wait',
            'details': {}
        }

        # Need at least 5m and 15m data for entry timing
        lower_tfs = [Timeframe.M5, Timeframe.M15]
        available_lower = [tf for tf in lower_tfs if tf in data_by_timeframe]

        if not available_lower:
            result['details']['error'] = 'No lower timeframe data available'
            return result

        score = 50  # Start neutral
        factors = []

        for tf in available_lower:
            df = data_by_timeframe[tf]
            if df is None or len(df) < 20:
                continue

            df_norm = df.copy()
            df_norm.columns = [c.lower() for c in df_norm.columns]

            # Check momentum using short EMAs
            ema3 = df_norm['close'].ewm(span=3, adjust=False).mean()
            ema8 = df_norm['close'].ewm(span=8, adjust=False).mean()
            ema21 = df_norm['close'].ewm(span=21, adjust=False).mean()

            current_close = df_norm['close'].iloc[-1]
            prev_close = df_norm['close'].iloc[-2] if len(df_norm) > 1 else current_close

            if direction == 'long':
                # Bullish factors for long entry
                # 1. Price above EMA8
                if current_close > ema8.iloc[-1]:
                    score += 5
                    factors.append(f"{tf.value}: Price above EMA8")

                # 2. EMA3 > EMA8 (short-term momentum)
                if ema3.iloc[-1] > ema8.iloc[-1]:
                    score += 5
                    factors.append(f"{tf.value}: EMA3 > EMA8")

                # 3. EMA8 > EMA21 (trend confirmation)
                if ema8.iloc[-1] > ema21.iloc[-1]:
                    score += 5
                    factors.append(f"{tf.value}: EMA8 > EMA21")

                # 4. Green candle (momentum)
                if current_close > prev_close:
                    score += 5
                    factors.append(f"{tf.value}: Bullish candle")

                # 5. Pullback to EMA8 (ideal entry)
                low = df_norm['low'].iloc[-1]
                if abs(low - ema8.iloc[-1]) / ema8.iloc[-1] < 0.005:  # Within 0.5%
                    score += 10
                    factors.append(f"{tf.value}: Pullback to EMA8")

            else:
                # Bearish factors for short entry
                if current_close < ema8.iloc[-1]:
                    score += 5
                    factors.append(f"{tf.value}: Price below EMA8")

                if ema3.iloc[-1] < ema8.iloc[-1]:
                    score += 5
                    factors.append(f"{tf.value}: EMA3 < EMA8")

                if ema8.iloc[-1] < ema21.iloc[-1]:
                    score += 5
                    factors.append(f"{tf.value}: EMA8 < EMA21")

                if current_close < prev_close:
                    score += 5
                    factors.append(f"{tf.value}: Bearish candle")

                high = df_norm['high'].iloc[-1]
                if abs(high - ema8.iloc[-1]) / ema8.iloc[-1] < 0.005:
                    score += 10
                    factors.append(f"{tf.value}: Rally to EMA8")

        # Cap score at 100
        score = min(100, max(0, score))

        # Determine recommendation
        if score >= 80:
            recommended_action = 'enter_now'
        elif score >= 65:
            recommended_action = 'prepare_entry'
        elif score >= 50:
            recommended_action = 'wait_for_setup'
        else:
            recommended_action = 'no_entry'

        result['entry_timing_score'] = score
        result['recommended_action'] = recommended_action
        result['details']['factors'] = factors
        result['details']['score_breakdown'] = {
            'base_score': 50,
            'adjustments': score - 50
        }

        return result

    def full_analysis(
        self,
        symbol: str,
        data_by_timeframe: Dict[Timeframe, pd.DataFrame]
    ) -> MTFAnalysisResult:
        """
        Perform complete multi-timeframe analysis.

        Args:
            symbol: Stock symbol
            data_by_timeframe: Dict mapping Timeframe to DataFrame

        Returns:
            MTFAnalysisResult with complete analysis
        """
        # Analyze trends for all timeframes
        trend_results = self.analyze_trend(symbol, data_by_timeframe)

        # Check alignment
        is_aligned, alignment_direction, trend_by_tf = self.check_timeframe_alignment(
            symbol, data_by_timeframe
        )

        # Get entry timing based on alignment direction
        entry_direction = 'long' if alignment_direction == 'bullish' else 'short'
        entry_timing = self.get_entry_timing(symbol, data_by_timeframe, entry_direction)

        # Build timeframe analyses
        tf_analyses = {}
        key_supports = []
        key_resistances = []

        for tf in self.timeframes:
            if tf not in data_by_timeframe:
                continue

            df = data_by_timeframe[tf]
            if df is None or df.empty:
                continue

            # Get trend for this timeframe
            trend = trend_results.get(tf.value)
            if trend is None:
                trend = TrendResult(
                    direction=TrendDirection.NEUTRAL,
                    strength=0,
                    age=0,
                    confidence=0,
                    details={}
                )

            # Get S/R levels
            supports, resistances = self.get_support_resistance(symbol, df, tf)

            # Determine bias
            if trend.direction in [TrendDirection.STRONG_UP, TrendDirection.UP]:
                bias = 'bullish'
            elif trend.direction in [TrendDirection.STRONG_DOWN, TrendDirection.DOWN]:
                bias = 'bearish'
            else:
                bias = 'neutral'

            # Find nearest key level
            current_price = df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]
            key_level = self._find_nearest_level(current_price, supports + resistances)

            tf_analyses[tf.value] = TimeframeAnalysis(
                timeframe=tf,
                trend=trend,
                support_levels=supports,
                resistance_levels=resistances,
                bias=bias,
                key_level_nearby=key_level
            )

            # Collect key levels (higher timeframes have priority)
            if tf in [Timeframe.DAILY, Timeframe.H4]:
                key_supports.extend([s.price for s in supports[:2]])
                key_resistances.extend([r.price for r in resistances[:2]])

        # Determine recommended action
        if is_aligned and entry_timing['entry_timing_score'] >= 70:
            recommended_action = f"Strong setup: {alignment_direction.upper()} entry"
        elif is_aligned and entry_timing['entry_timing_score'] >= 50:
            recommended_action = f"Wait for pullback: {alignment_direction.upper()} bias"
        elif is_aligned:
            recommended_action = f"Trend aligned {alignment_direction}, wait for entry"
        else:
            recommended_action = "Mixed signals, no trade"

        return MTFAnalysisResult(
            symbol=symbol,
            timestamp=datetime.now(),
            timeframe_analyses=tf_analyses,
            timeframe_alignment=is_aligned,
            alignment_direction=alignment_direction,
            trend_by_timeframe=trend_by_tf,
            entry_timing_score=entry_timing['entry_timing_score'],
            recommended_action=recommended_action,
            key_levels={
                'support': sorted(set(key_supports))[:3],
                'resistance': sorted(set(key_resistances))[:3]
            }
        )

    def _find_nearest_level(
        self,
        current_price: float,
        levels: List[SupportResistanceLevel]
    ) -> Optional[float]:
        """Find the nearest S/R level to current price"""
        if not levels:
            return None

        nearest = min(levels, key=lambda x: abs(x.price - current_price))
        return nearest.price


def resample_to_timeframe(
    df: pd.DataFrame,
    source_tf: Timeframe,
    target_tf: Timeframe
) -> pd.DataFrame:
    """
    Resample OHLCV data from a lower timeframe to a higher timeframe.

    Args:
        df: Source DataFrame with OHLCV data
        source_tf: Source timeframe
        target_tf: Target timeframe (must be higher than source)

    Returns:
        Resampled DataFrame
    """
    if target_tf.minutes <= source_tf.minutes:
        logger.warning(f"Cannot resample from {source_tf.value} to {target_tf.value}")
        return df

    # Normalize column names
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Determine resample rule
    rule_map = {
        Timeframe.M5: '5min',
        Timeframe.M15: '15min',
        Timeframe.H1: '1h',
        Timeframe.H4: '4h',
        Timeframe.DAILY: '1D'
    }

    rule = rule_map.get(target_tf, '1D')

    # Resample OHLCV
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return resampled
