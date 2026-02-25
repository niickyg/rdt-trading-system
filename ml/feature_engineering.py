"""
Feature Engineering Pipeline for ML-Based Trading System

This module implements a comprehensive feature engineering pipeline that extracts
60+ features from market data including technical indicators, microstructure features,
market regime indicators, and temporal features.

Features are calculated, cached for performance, and stored in TimescaleDB for
historical analysis and model training.
"""

import asyncio
import pickle
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from ml.safe_model_loader import safe_load_model, safe_save_model, ModelSecurityError
from sqlalchemy import create_engine, text, Table, Column, MetaData
from sqlalchemy import Float, String, DateTime, Index, Integer
from sqlalchemy.dialects.postgresql import JSONB

from shared.data_provider import DataProvider
from shared.indicators.rrs import RRSCalculator, calculate_ema, calculate_sma, calculate_vwap


class FeatureEngineer:
    """
    Comprehensive feature engineering pipeline for trading ML models.

    Calculates 60+ features organized into categories:
    - Technical Indicators (20 features): RRS, RSI, MACD, Bollinger Bands, etc.
    - Microstructure Features (15 features): Bid-ask spread, VWAP, momentum, etc.
    - Regime Features (10 features): VIX, SPY trend, sector strength, etc.
    - Temporal Features (15 features): Hour of day, time since open, etc.
    - Derived Features (10 features): Feature interactions and ratios

    Features are cached for performance and can be stored to TimescaleDB.
    """

    def __init__(
        self,
        data_provider: Optional[DataProvider] = None,
        db_url: Optional[str] = None,
        cache_ttl_seconds: int = 300,
        enable_db_storage: bool = False
    ):
        """
        Initialize the Feature Engineer.

        Args:
            data_provider: DataProvider instance for fetching market data
            db_url: PostgreSQL/TimescaleDB connection string
            cache_ttl_seconds: How long to cache features (default 5 minutes)
            enable_db_storage: Whether to store features to database
        """
        self.data_provider = data_provider or DataProvider()
        self.rrs_calculator = RRSCalculator()
        self.cache_ttl = cache_ttl_seconds
        self.enable_db_storage = enable_db_storage

        # Feature cache
        self._feature_cache: Dict[str, Dict] = {}
        self._cache_times: Dict[str, datetime] = {}

        # Database connection
        self.db_url = db_url
        self.engine = None
        self.metadata = MetaData()

        if enable_db_storage and db_url:
            self._init_database()

        # Feature categories and names
        self._init_feature_names()

        logger.info(
            f"FeatureEngineer initialized with {len(self.all_feature_names)} features "
            f"(DB storage: {enable_db_storage})"
        )

    def _init_feature_names(self):
        """Initialize feature name lists for all categories."""
        # Technical indicators (20 features)
        self.technical_features = [
            'rrs', 'rrs_3bar', 'rrs_5bar', 'atr', 'atr_percent',
            'rsi_14', 'rsi_9', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
            'ema_3', 'ema_8', 'ema_21', 'ema_50', 'volume_sma_20'
        ]

        # Microstructure features (15 features)
        self.microstructure_features = [
            'bid_ask_spread', 'vwap', 'vwap_distance', 'vwap_distance_percent',
            'price_momentum_1', 'price_momentum_5', 'price_momentum_15',
            'volume_ratio', 'volume_trend', 'price_range_percent',
            'intraday_high_low_range', 'price_position_in_range',
            'relative_volume', 'tick_direction', 'order_flow_imbalance'
        ]

        # Regime features (10 features)
        self.regime_features = [
            'vix', 'vix_change', 'spy_trend', 'spy_ema_alignment',
            'spy_rsi', 'spy_momentum', 'sector_relative_strength',
            'market_breadth', 'spy_volume_ratio', 'correlation_with_spy'
        ]

        # Temporal features (15 features)
        self.temporal_features = [
            'hour_of_day', 'minute_of_hour', 'day_of_week', 'day_of_month',
            'week_of_year', 'time_since_open_minutes', 'time_until_close_minutes',
            'is_market_open', 'is_pre_market', 'is_after_hours',
            'is_first_hour', 'is_last_hour', 'is_power_hour', 'is_monday', 'is_friday'
        ]

        # Derived/interaction features (10 features)
        self.derived_features = [
            'rrs_rsi_interaction', 'momentum_volume_interaction',
            'volatility_regime_score', 'trend_strength_composite',
            'reversal_probability', 'breakout_probability',
            'risk_reward_ratio', 'sharpe_estimate', 'daily_alignment_score',
            'feature_complexity_score'
        ]

        # Murphy-inspired features (17 features)
        self.murphy_features = [
            # Volume analysis (Murphy Law #10)
            'obv_trend',              # OBV 10-bar slope direction (+1, 0, -1)
            'obv_price_divergence',   # Binary: price trend vs OBV trend disagree
            'volume_climax',          # Binary: volume > 3x 20-day avg

            # ADX indicator selection (Murphy Law #9)
            'adx',                    # ADX value (0-100)
            'adx_rising',             # Binary: ADX higher than 5 bars ago
            'plus_di',                # +DI value
            'minus_di',               # -DI value

            # Moving averages (Murphy Law #6)
            'sma_200',                # 200-day SMA value
            'price_above_sma200',     # Binary: price > 200 SMA
            'golden_cross_state',     # +1 if 50SMA > 200SMA, -1 if below, 0 if within 1%

            # MACD enhancement (Murphy Law #8)
            'macd_histogram_slope',   # 3-bar direction of histogram (+1, -1, 0)
            'macd_histogram_divergence',  # Binary: price vs histogram divergence

            # RRS enhancement (Murphy intermarket)
            'rrs_slope',              # RRS direction over 5 bars
            'rrs_acceleration',       # Change in RRS slope

            # Volatility (Murphy + Carter)
            'bb_squeeze',             # Binary: bb_width below 50-bar low
            'bb_squeeze_duration',    # How many bars squeeze has been active

            # Fibonacci (Murphy Law #4)
            'pullback_depth_pct',     # Retracement % of recent 20-bar move
        ]

        # All features combined
        self.all_feature_names = (
            self.technical_features +
            self.microstructure_features +
            self.regime_features +
            self.temporal_features +
            self.derived_features +
            self.murphy_features
        )

    def _init_database(self):
        """Initialize database connection and create feature storage table."""
        try:
            self.engine = create_engine(self.db_url, pool_pre_ping=True)

            # Define features table
            self.features_table = Table(
                'ml_features',
                self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('symbol', String(10), nullable=False, index=True),
                Column('timestamp', DateTime, nullable=False, index=True),
                Column('features', JSONB, nullable=False),
                Column('feature_version', String(10), nullable=False, default='1.0'),
                Column('created_at', DateTime, nullable=False, default=datetime.utcnow),

                # Indexes for fast queries
                Index('ix_ml_features_symbol_timestamp', 'symbol', 'timestamp'),
                Index('ix_ml_features_timestamp', 'timestamp'),
            )

            # Create table if it doesn't exist
            self.metadata.create_all(self.engine)

            # Create TimescaleDB hypertable (if using TimescaleDB)
            with self.engine.connect() as conn:
                try:
                    conn.execute(text(
                        "SELECT create_hypertable('ml_features', 'timestamp', "
                        "if_not_exists => TRUE, migrate_data => TRUE)"
                    ))
                    conn.commit()
                    logger.info("TimescaleDB hypertable created for ml_features")
                except Exception as e:
                    # Not a TimescaleDB or hypertable already exists
                    logger.debug(f"Hypertable creation skipped: {e}")

            logger.info("Feature storage database initialized")

        except Exception as e:
            logger.error(f"Failed to initialize feature database: {e}")
            self.enable_db_storage = False

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached features are still valid."""
        if key not in self._cache_times:
            return False
        age = (datetime.now() - self._cache_times[key]).total_seconds()
        return age < self.cache_ttl

    async def calculate_features(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Calculate all features for a given symbol.

        Args:
            symbol: Stock ticker symbol
            use_cache: Whether to use cached features if available

        Returns:
            DataFrame with single row containing all features, or None if failed
        """
        cache_key = f"features_{symbol}"

        # Check cache
        if use_cache and self._is_cache_valid(cache_key):
            logger.debug(f"Using cached features for {symbol}")
            return self._feature_cache[cache_key]

        try:
            # Fetch market data
            stock_data = await self.data_provider.get_stock_data(symbol)
            spy_data = await self.data_provider.get_spy_data()

            if not stock_data or not spy_data:
                logger.warning(f"Failed to fetch data for {symbol}")
                return None

            # Calculate all feature categories
            features = {}

            # Technical indicators
            technical = await self._calculate_technical_features(symbol, stock_data, spy_data)
            features.update(technical)

            # Microstructure features
            microstructure = await self._calculate_microstructure_features(symbol, stock_data)
            features.update(microstructure)

            # Regime features
            regime = await self._calculate_regime_features(spy_data, stock_data)
            features.update(regime)

            # Temporal features
            temporal = self._calculate_temporal_features()
            features.update(temporal)

            # Derived/interaction features
            derived = self._calculate_derived_features(features)
            features.update(derived)

            # Murphy-inspired features (must be after technical for rrs, macd, bb references)
            murphy = await self._calculate_murphy_features(symbol, stock_data, spy_data, features)
            features.update(murphy)

            # Create DataFrame
            df = pd.DataFrame([features])
            df['symbol'] = symbol
            df['timestamp'] = datetime.now()

            # Cache the result
            self._feature_cache[cache_key] = df
            self._cache_times[cache_key] = datetime.now()

            # Store to database if enabled
            if self.enable_db_storage:
                await self._store_features_async(symbol, features)

            logger.debug(f"Calculated {len(features)} features for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error calculating features for {symbol}: {e}")
            return None

    async def _calculate_technical_features(
        self,
        symbol: str,
        stock_data: Dict,
        spy_data: Dict
    ) -> Dict[str, float]:
        """Calculate technical indicator features."""
        features = {}

        try:
            daily_df = stock_data['daily_data']
            spy_df = spy_data['daily_data']

            # RRS calculations (1-bar, 3-bar, 5-bar)
            rrs_1bar = self.rrs_calculator.calculate_rrs(daily_df, spy_df, periods=1)
            features['rrs'] = float(rrs_1bar['rrs'].iloc[-1])

            rrs_3bar = self.rrs_calculator.calculate_rrs(daily_df, spy_df, periods=3)
            features['rrs_3bar'] = float(rrs_3bar['rrs'].iloc[-1])

            rrs_5bar = self.rrs_calculator.calculate_rrs(daily_df, spy_df, periods=5)
            features['rrs_5bar'] = float(rrs_5bar['rrs'].iloc[-1])

            # ATR
            atr = self.rrs_calculator.calculate_atr(daily_df)
            features['atr'] = float(atr.iloc[-1])
            features['atr_percent'] = (features['atr'] / stock_data['current_price']) * 100

            # RSI (Relative Strength Index)
            features['rsi_14'] = self._calculate_rsi(daily_df['close'], 14)
            features['rsi_9'] = self._calculate_rsi(daily_df['close'], 9)

            # MACD
            macd_result = self._calculate_macd(daily_df['close'])
            features['macd'] = macd_result['macd']
            features['macd_signal'] = macd_result['signal']
            features['macd_histogram'] = macd_result['histogram']

            # Bollinger Bands
            bb_result = self._calculate_bollinger_bands(daily_df['close'])
            features['bb_upper'] = bb_result['upper']
            features['bb_middle'] = bb_result['middle']
            features['bb_lower'] = bb_result['lower']
            features['bb_width'] = bb_result['width']
            features['bb_percent'] = bb_result['percent_b']

            # EMAs
            features['ema_3'] = float(calculate_ema(daily_df['close'], 3).iloc[-1])
            features['ema_8'] = float(calculate_ema(daily_df['close'], 8).iloc[-1])
            features['ema_21'] = float(calculate_ema(daily_df['close'], 21).iloc[-1])
            features['ema_50'] = float(calculate_ema(daily_df['close'], 50).iloc[-1])

            # Volume SMA
            features['volume_sma_20'] = float(daily_df['volume'].rolling(20).mean().iloc[-1])

        except Exception as e:
            logger.error(f"Error calculating technical features: {e}")
            # Fill with zeros on error
            for name in self.technical_features:
                features.setdefault(name, 0.0)

        return features

    async def _calculate_microstructure_features(
        self,
        symbol: str,
        stock_data: Dict
    ) -> Dict[str, float]:
        """Calculate market microstructure features."""
        features = {}

        try:
            daily_df = stock_data['daily_data']
            current_price = stock_data['current_price']

            # Bid-ask spread (estimated from high-low)
            recent_spread = (daily_df['high'] - daily_df['low']).tail(5).mean()
            features['bid_ask_spread'] = float(recent_spread)

            # VWAP and distance from VWAP
            vwap = calculate_vwap(daily_df)
            features['vwap'] = float(vwap.iloc[-1])
            features['vwap_distance'] = current_price - features['vwap']
            features['vwap_distance_percent'] = (features['vwap_distance'] / features['vwap']) * 100

            # Price momentum (various periods)
            close = daily_df['close']
            features['price_momentum_1'] = float(close.pct_change(1).iloc[-1] * 100)
            features['price_momentum_5'] = float(close.pct_change(5).iloc[-1] * 100)
            features['price_momentum_15'] = float(close.pct_change(15).iloc[-1] * 100)

            # Volume analysis
            volume = daily_df['volume']
            features['volume_ratio'] = float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1])
            features['volume_trend'] = float(volume.rolling(5).mean().iloc[-1] / volume.rolling(20).mean().iloc[-1])

            # Price range analysis
            day_range = daily_df['high'].iloc[-1] - daily_df['low'].iloc[-1]
            features['price_range_percent'] = (day_range / daily_df['close'].iloc[-1]) * 100

            # Intraday position
            intraday_range = daily_df['high'].iloc[-1] - daily_df['low'].iloc[-1]
            features['intraday_high_low_range'] = float(intraday_range)

            if intraday_range > 0:
                features['price_position_in_range'] = (
                    (current_price - daily_df['low'].iloc[-1]) / intraday_range
                )
            else:
                features['price_position_in_range'] = 0.5

            # Relative volume
            features['relative_volume'] = float(
                stock_data['volume'] / daily_df['volume'].rolling(20).mean().iloc[-1]
            )

            # Tick direction (simplified)
            price_change = close.diff().iloc[-1]
            features['tick_direction'] = 1.0 if price_change > 0 else (-1.0 if price_change < 0 else 0.0)

            # Order flow imbalance (estimated)
            upticks = (daily_df['close'] > daily_df['open']).tail(10).sum()
            features['order_flow_imbalance'] = (upticks / 10) * 2 - 1  # Scale to [-1, 1]

        except Exception as e:
            logger.error(f"Error calculating microstructure features: {e}")
            for name in self.microstructure_features:
                features.setdefault(name, 0.0)

        return features

    async def _calculate_regime_features(
        self,
        spy_data: Dict,
        stock_data: Dict
    ) -> Dict[str, float]:
        """Calculate market regime features."""
        features = {}

        try:
            # Fetch VIX data
            loop = asyncio.get_running_loop()
            vix_data = await loop.run_in_executor(
                None,
                self._fetch_vix_sync
            )

            if vix_data is not None:
                features['vix'] = float(vix_data['close'].iloc[-1])
                features['vix_change'] = float(vix_data['close'].pct_change().iloc[-1] * 100)
            else:
                features['vix'] = 15.0  # Default neutral VIX
                features['vix_change'] = 0.0

            # SPY trend analysis
            spy_df = spy_data['daily_data']
            spy_close = spy_df['close']

            spy_ema_8 = calculate_ema(spy_close, 8)
            spy_ema_21 = calculate_ema(spy_close, 21)

            features['spy_trend'] = float(spy_close.pct_change(10).iloc[-1] * 100)
            features['spy_ema_alignment'] = 1.0 if spy_ema_8.iloc[-1] > spy_ema_21.iloc[-1] else 0.0

            # SPY RSI
            features['spy_rsi'] = self._calculate_rsi(spy_close, 14)

            # SPY momentum
            features['spy_momentum'] = float(spy_close.pct_change(5).iloc[-1] * 100)

            # Sector relative strength (simplified - using SPY as proxy)
            stock_return = stock_data['daily_data']['close'].pct_change(10).iloc[-1]
            spy_return = spy_close.pct_change(10).iloc[-1]
            features['sector_relative_strength'] = float((stock_return - spy_return) * 100)

            # Market breadth (estimated from SPY volume)
            spy_volume = spy_df['volume']
            features['market_breadth'] = float(
                spy_volume.iloc[-1] / spy_volume.rolling(20).mean().iloc[-1]
            )

            # SPY volume ratio
            features['spy_volume_ratio'] = float(
                spy_volume.iloc[-1] / spy_volume.rolling(10).mean().iloc[-1]
            )

            # Correlation with SPY (rolling 20-day)
            stock_returns = stock_data['daily_data']['close'].pct_change().tail(20)
            spy_returns = spy_close.pct_change().tail(20)
            features['correlation_with_spy'] = float(stock_returns.corr(spy_returns))

        except Exception as e:
            logger.error(f"Error calculating regime features: {e}")
            for name in self.regime_features:
                features.setdefault(name, 0.0)

        return features

    def _calculate_temporal_features(self) -> Dict[str, float]:
        """Calculate temporal/time-based features."""
        features = {}

        try:
            now = datetime.now()

            # Basic time features
            features['hour_of_day'] = float(now.hour)
            features['minute_of_hour'] = float(now.minute)
            features['day_of_week'] = float(now.weekday())  # 0 = Monday
            features['day_of_month'] = float(now.day)
            features['week_of_year'] = float(now.isocalendar()[1])

            # Market hours features
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

            # Time since market open (in minutes)
            if now >= market_open:
                features['time_since_open_minutes'] = float(
                    (now - market_open).total_seconds() / 60
                )
            else:
                features['time_since_open_minutes'] = 0.0

            # Time until market close (in minutes)
            if now < market_close:
                features['time_until_close_minutes'] = float(
                    (market_close - now).total_seconds() / 60
                )
            else:
                features['time_until_close_minutes'] = 0.0

            # Market session flags
            features['is_market_open'] = 1.0 if market_open <= now <= market_close else 0.0
            features['is_pre_market'] = 1.0 if now.hour < 9 or (now.hour == 9 and now.minute < 30) else 0.0
            features['is_after_hours'] = 1.0 if now.hour >= 16 else 0.0

            # Special time periods
            features['is_first_hour'] = 1.0 if 9 <= now.hour < 10 else 0.0
            features['is_last_hour'] = 1.0 if 15 <= now.hour < 16 else 0.0
            features['is_power_hour'] = 1.0 if 15 <= now.hour < 16 else 0.0

            # Day of week flags
            features['is_monday'] = 1.0 if now.weekday() == 0 else 0.0
            features['is_friday'] = 1.0 if now.weekday() == 4 else 0.0

        except Exception as e:
            logger.error(f"Error calculating temporal features: {e}")
            for name in self.temporal_features:
                features.setdefault(name, 0.0)

        return features

    def _calculate_derived_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate derived features from interactions between base features."""
        derived = {}

        try:
            # RRS and RSI interaction
            rrs = features.get('rrs', 0)
            rsi_14 = features.get('rsi_14', 50)
            derived['rrs_rsi_interaction'] = rrs * (rsi_14 - 50) / 50

            # Momentum and volume interaction
            momentum = features.get('price_momentum_5', 0)
            volume_ratio = features.get('volume_ratio', 1)
            derived['momentum_volume_interaction'] = momentum * np.log1p(volume_ratio)

            # Volatility regime score
            vix = features.get('vix', 15)
            atr_percent = features.get('atr_percent', 2)
            derived['volatility_regime_score'] = (vix / 20) * (atr_percent / 3)

            # Trend strength composite
            ema_3 = features.get('ema_3', 0)
            ema_8 = features.get('ema_8', 0)
            ema_21 = features.get('ema_21', 0)

            if ema_21 > 0:
                trend_alignment = (ema_3 > ema_8 > ema_21) or (ema_3 < ema_8 < ema_21)
                trend_spread = abs((ema_3 - ema_21) / ema_21) * 100
                derived['trend_strength_composite'] = trend_spread if trend_alignment else -trend_spread
            else:
                derived['trend_strength_composite'] = 0.0

            # Reversal probability (based on RSI extremes and MACD)
            # ADX-gated: suppress false overbought/oversold signals in strong trends
            # (Murphy Law #9 - use trend tools when ADX rising, oscillators when flat)
            macd_histogram = features.get('macd_histogram', 0)
            adx_val = features.get('adx', 0)
            adx_rising_val = features.get('adx_rising', 0)

            if rsi_14 > 70 and macd_histogram < 0:
                derived['reversal_probability'] = 0.8
            elif rsi_14 < 30 and macd_histogram > 0:
                derived['reversal_probability'] = 0.8
            else:
                derived['reversal_probability'] = 0.2

            # If ADX > 25 and rising, trend is strong — suppress reversal signal
            if adx_val > 25 and adx_rising_val == 1.0:
                derived['reversal_probability'] = 0.2

            # Breakout probability (based on Bollinger Bands and volume)
            bb_percent = features.get('bb_percent', 0.5)
            if (bb_percent > 0.95 or bb_percent < 0.05) and volume_ratio > 1.5:
                derived['breakout_probability'] = 0.8
            else:
                derived['breakout_probability'] = 0.3

            # Risk-reward ratio estimate
            atr = features.get('atr', 1)
            bb_width = features.get('bb_width', 1)
            if bb_width > 0:
                derived['risk_reward_ratio'] = atr / bb_width
            else:
                derived['risk_reward_ratio'] = 1.0

            # Sharpe ratio estimate (simplified)
            if atr > 0:
                derived['sharpe_estimate'] = abs(momentum) / atr
            else:
                derived['sharpe_estimate'] = 0.0

            # Daily alignment score (EMA alignment + RRS)
            ema_aligned = 1.0 if ema_3 > ema_8 > ema_21 else (-1.0 if ema_3 < ema_8 < ema_21 else 0.0)
            rrs_normalized = np.tanh(rrs / 3)  # Normalize RRS to [-1, 1]
            derived['daily_alignment_score'] = (ema_aligned + rrs_normalized) / 2

            # Feature complexity score (how many strong signals)
            strong_signals = sum([
                abs(rrs) > 2,
                rsi_14 > 60 or rsi_14 < 40,
                abs(macd_histogram) > 0.5,
                volume_ratio > 1.5,
                abs(features.get('vwap_distance_percent', 0)) > 2
            ])
            derived['feature_complexity_score'] = float(strong_signals) / 5

        except Exception as e:
            logger.error(f"Error calculating derived features: {e}")
            for name in self.derived_features:
                derived.setdefault(name, 0.0)

        return derived

    async def _calculate_murphy_features(
        self,
        symbol: str,
        stock_data: Dict,
        spy_data: Dict,
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate Murphy-inspired technical analysis features.

        Based on John Murphy's "Technical Analysis of the Financial Markets":
        - OBV and volume analysis (Law #10)
        - ADX indicator selection (Law #9)
        - 200 SMA and golden cross (Law #6)
        - MACD histogram enhancement (Law #8)
        - RRS slope and acceleration (intermarket)
        - Bollinger Band squeeze (volatility)
        - Fibonacci pullback depth (Law #4)

        Args:
            symbol: Stock ticker symbol
            stock_data: Stock data dictionary from DataProvider
            spy_data: SPY data dictionary from DataProvider
            features: Previously calculated features dict (for referencing rrs, macd, bb, etc.)

        Returns:
            Dictionary of Murphy feature values
        """
        murphy = {}

        try:
            daily_df = stock_data['daily_data']
            close = daily_df['close']
            volume = daily_df['volume']
            high = daily_df['high']
            low = daily_df['low']
            current_price = stock_data['current_price']

            # ---------------------------------------------------------------
            # OBV and volume analysis (Murphy Law #10)
            # ---------------------------------------------------------------
            try:
                obv = (np.sign(close.diff()) * volume).cumsum()

                # OBV trend: sign of slope of last 10 bars
                if len(obv) >= 10:
                    obv_tail = obv.iloc[-10:]
                    x = np.arange(len(obv_tail))
                    obv_vals = obv_tail.values.astype(float)
                    # Simple linear regression slope
                    obv_slope = np.polyfit(x, obv_vals, 1)[0]
                    murphy['obv_trend'] = float(np.sign(obv_slope))
                else:
                    murphy['obv_trend'] = 0.0

                # OBV price divergence: price 10-bar slope vs OBV 10-bar slope disagree
                if len(close) >= 10:
                    close_tail = close.iloc[-10:]
                    x = np.arange(len(close_tail))
                    price_slope = np.polyfit(x, close_tail.values.astype(float), 1)[0]
                    murphy['obv_price_divergence'] = 1.0 if np.sign(price_slope) != np.sign(obv_slope) else 0.0
                else:
                    murphy['obv_price_divergence'] = 0.0

                # Volume climax: volume > 3x 20-day average
                vol_avg_20 = volume.rolling(20).mean().iloc[-1]
                murphy['volume_climax'] = 1.0 if volume.iloc[-1] > 3.0 * vol_avg_20 else 0.0

            except Exception as e:
                logger.debug(f"OBV calculation error for {symbol}: {e}")
                murphy['obv_trend'] = 0.0
                murphy['obv_price_divergence'] = 0.0
                murphy['volume_climax'] = 0.0

            # ---------------------------------------------------------------
            # ADX indicator selection (Murphy Law #9)
            # ---------------------------------------------------------------
            try:
                if len(daily_df) >= 30:
                    # True Range
                    prev_close = close.shift(1)
                    tr1 = high - low
                    tr2 = (high - prev_close).abs()
                    tr3 = (low - prev_close).abs()
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

                    # +DM and -DM
                    up_move = high.diff()
                    down_move = -low.diff()
                    plus_dm = pd.Series(
                        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=close.index
                    )
                    minus_dm = pd.Series(
                        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                        index=close.index
                    )

                    # Smoothed over 14 periods (Wilder's smoothing)
                    period = 14
                    smoothed_tr = tr.rolling(period).sum()
                    smoothed_plus_dm = plus_dm.rolling(period).sum()
                    smoothed_minus_dm = minus_dm.rolling(period).sum()

                    # +DI and -DI
                    plus_di = (smoothed_plus_dm / smoothed_tr) * 100
                    minus_di = (smoothed_minus_dm / smoothed_tr) * 100

                    # DX
                    di_sum = plus_di + minus_di
                    di_diff = (plus_di - minus_di).abs()
                    dx = (di_diff / di_sum.replace(0, np.nan)) * 100

                    # ADX = 14-period smoothed DX
                    adx = dx.rolling(period).mean()

                    murphy['adx'] = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0
                    murphy['plus_di'] = float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else 0.0
                    murphy['minus_di'] = float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 0.0

                    # ADX rising: current ADX > ADX 5 bars ago
                    if len(adx.dropna()) >= 6:
                        murphy['adx_rising'] = 1.0 if adx.iloc[-1] > adx.iloc[-6] else 0.0
                    else:
                        murphy['adx_rising'] = 0.0
                else:
                    murphy['adx'] = 0.0
                    murphy['adx_rising'] = 0.0
                    murphy['plus_di'] = 0.0
                    murphy['minus_di'] = 0.0

            except Exception as e:
                logger.debug(f"ADX calculation error for {symbol}: {e}")
                murphy['adx'] = 0.0
                murphy['adx_rising'] = 0.0
                murphy['plus_di'] = 0.0
                murphy['minus_di'] = 0.0

            # ---------------------------------------------------------------
            # Moving averages (Murphy Law #6)
            # ---------------------------------------------------------------
            try:
                if len(close) >= 200:
                    sma_200 = calculate_sma(close, 200)
                    murphy['sma_200'] = float(sma_200.iloc[-1])
                    murphy['price_above_sma200'] = 1.0 if current_price > sma_200.iloc[-1] else 0.0

                    # Golden cross state: compare 50 SMA vs 200 SMA
                    sma_50 = calculate_sma(close, 50)
                    sma_50_val = float(sma_50.iloc[-1])
                    sma_200_val = float(sma_200.iloc[-1])

                    if sma_200_val > 0:
                        ratio = sma_50_val / sma_200_val
                        if ratio > 1.01:
                            murphy['golden_cross_state'] = 1.0
                        elif ratio < 0.99:
                            murphy['golden_cross_state'] = -1.0
                        else:
                            murphy['golden_cross_state'] = 0.0
                    else:
                        murphy['golden_cross_state'] = 0.0
                else:
                    murphy['sma_200'] = 0.0
                    murphy['price_above_sma200'] = 0.0
                    murphy['golden_cross_state'] = 0.0

            except Exception as e:
                logger.debug(f"200 SMA calculation error for {symbol}: {e}")
                murphy['sma_200'] = 0.0
                murphy['price_above_sma200'] = 0.0
                murphy['golden_cross_state'] = 0.0

            # ---------------------------------------------------------------
            # MACD histogram enhancement (Murphy Law #8)
            # ---------------------------------------------------------------
            try:
                # Recalculate MACD histogram series for slope analysis
                ema_12 = calculate_ema(close, 12)
                ema_26 = calculate_ema(close, 26)
                macd_line = ema_12 - ema_26
                macd_signal = calculate_ema(macd_line, 9)
                histogram = macd_line - macd_signal

                # MACD histogram slope: direction over last 3 bars
                if len(histogram) >= 3:
                    h1 = histogram.iloc[-3]
                    h2 = histogram.iloc[-2]
                    h3 = histogram.iloc[-1]

                    if h3 > h2 > h1:
                        murphy['macd_histogram_slope'] = 1.0
                    elif h3 < h2 < h1:
                        murphy['macd_histogram_slope'] = -1.0
                    else:
                        murphy['macd_histogram_slope'] = 0.0
                else:
                    murphy['macd_histogram_slope'] = 0.0

                # MACD histogram divergence: compare price swing highs vs histogram peaks
                # Look for divergence in last 40 bars
                if len(close) >= 40 and len(histogram) >= 40:
                    lookback = 40
                    price_window = close.iloc[-lookback:]
                    hist_window = histogram.iloc[-lookback:]

                    # Find local peaks (simplified: split window into two halves)
                    half = lookback // 2
                    price_peak1 = price_window.iloc[:half].max()
                    price_peak2 = price_window.iloc[half:].max()
                    hist_peak1 = hist_window.iloc[:half].max()
                    hist_peak2 = hist_window.iloc[half:].max()

                    # Bearish divergence: price higher high, histogram lower high
                    bearish_div = (price_peak2 > price_peak1) and (hist_peak2 < hist_peak1)
                    # Bullish divergence: price lower low, histogram higher low
                    price_trough1 = price_window.iloc[:half].min()
                    price_trough2 = price_window.iloc[half:].min()
                    hist_trough1 = hist_window.iloc[:half].min()
                    hist_trough2 = hist_window.iloc[half:].min()
                    bullish_div = (price_trough2 < price_trough1) and (hist_trough2 > hist_trough1)

                    murphy['macd_histogram_divergence'] = 1.0 if (bearish_div or bullish_div) else 0.0
                else:
                    murphy['macd_histogram_divergence'] = 0.0

            except Exception as e:
                logger.debug(f"MACD histogram enhancement error for {symbol}: {e}")
                murphy['macd_histogram_slope'] = 0.0
                murphy['macd_histogram_divergence'] = 0.0

            # ---------------------------------------------------------------
            # RRS enhancement (Murphy intermarket)
            # ---------------------------------------------------------------
            try:
                # Use pre-calculated RRS values from features dict
                rrs_current = features.get('rrs', 0.0)
                rrs_5bar = features.get('rrs_5bar', None)

                if rrs_5bar is not None:
                    murphy['rrs_slope'] = float(rrs_current - rrs_5bar)
                else:
                    # Compute from data if not available
                    spy_df = spy_data['daily_data']
                    rrs_series = self.rrs_calculator.calculate_rrs(daily_df, spy_df, periods=1)
                    if len(rrs_series['rrs']) >= 6:
                        murphy['rrs_slope'] = float(
                            rrs_series['rrs'].iloc[-1] - rrs_series['rrs'].iloc[-6]
                        )
                    else:
                        murphy['rrs_slope'] = 0.0

                # RRS acceleration: change in slope (compare current slope to 5 bars ago)
                spy_df = spy_data['daily_data']
                rrs_series = self.rrs_calculator.calculate_rrs(daily_df, spy_df, periods=1)
                if len(rrs_series['rrs']) >= 11:
                    rrs_vals = rrs_series['rrs']
                    slope_now = float(rrs_vals.iloc[-1] - rrs_vals.iloc[-6])
                    slope_prev = float(rrs_vals.iloc[-6] - rrs_vals.iloc[-11])
                    murphy['rrs_acceleration'] = slope_now - slope_prev
                else:
                    murphy['rrs_acceleration'] = 0.0

            except Exception as e:
                logger.debug(f"RRS enhancement error for {symbol}: {e}")
                murphy['rrs_slope'] = 0.0
                murphy['rrs_acceleration'] = 0.0

            # ---------------------------------------------------------------
            # Volatility: BB squeeze (Murphy + Carter)
            # ---------------------------------------------------------------
            try:
                bb_width = features.get('bb_width', 0.0)

                # Calculate BB width series for squeeze detection
                sma_20 = calculate_sma(close, 20)
                std_20 = close.rolling(window=20).std()
                bb_upper_series = sma_20 + (std_20 * 2.0)
                bb_lower_series = sma_20 - (std_20 * 2.0)
                bb_width_series = bb_upper_series - bb_lower_series

                if len(bb_width_series.dropna()) >= 50:
                    bb_width_50_low = bb_width_series.rolling(50).min()
                    epsilon = 1e-6
                    murphy['bb_squeeze'] = 1.0 if bb_width <= (bb_width_50_low.iloc[-1] + epsilon) else 0.0

                    # Squeeze duration: count consecutive bars where squeeze is active
                    squeeze_active = bb_width_series <= (bb_width_50_low + epsilon)
                    # Count from the end backward
                    duration = 0
                    for i in range(len(squeeze_active) - 1, -1, -1):
                        if squeeze_active.iloc[i]:
                            duration += 1
                        else:
                            break
                    murphy['bb_squeeze_duration'] = float(duration)
                else:
                    murphy['bb_squeeze'] = 0.0
                    murphy['bb_squeeze_duration'] = 0.0

            except Exception as e:
                logger.debug(f"BB squeeze calculation error for {symbol}: {e}")
                murphy['bb_squeeze'] = 0.0
                murphy['bb_squeeze_duration'] = 0.0

            # ---------------------------------------------------------------
            # Fibonacci pullback depth (Murphy Law #4)
            # ---------------------------------------------------------------
            try:
                if len(close) >= 20:
                    recent_20 = close.iloc[-20:]
                    recent_high = recent_20.max()
                    recent_low = recent_20.min()
                    price_range = recent_high - recent_low

                    if price_range > 0:
                        # Determine trend direction: if close is closer to high, uptrend
                        # Measure retracement from the extreme
                        mid = (recent_high + recent_low) / 2
                        if current_price >= mid:
                            # Uptrend: measure pullback from high
                            pullback = recent_high - current_price
                        else:
                            # Downtrend: measure pullback from low
                            pullback = current_price - recent_low
                        murphy['pullback_depth_pct'] = float(abs(pullback) / price_range * 100)
                    else:
                        murphy['pullback_depth_pct'] = 0.0
                else:
                    murphy['pullback_depth_pct'] = 0.0

            except Exception as e:
                logger.debug(f"Pullback depth calculation error for {symbol}: {e}")
                murphy['pullback_depth_pct'] = 0.0

        except Exception as e:
            logger.error(f"Error calculating Murphy features for {symbol}: {e}")
            for name in self.murphy_features:
                murphy.setdefault(name, 0.0)

        # Ensure all Murphy features have a value
        for name in self.murphy_features:
            murphy.setdefault(name, 0.0)

        return murphy

    # Helper methods for technical calculations

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except Exception:
            return 50.0

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, float]:
        """Calculate MACD indicator."""
        try:
            ema_fast = calculate_ema(prices, fast)
            ema_slow = calculate_ema(prices, slow)

            macd = ema_fast - ema_slow
            macd_signal = calculate_ema(macd, signal)
            histogram = macd - macd_signal

            return {
                'macd': float(macd.iloc[-1]),
                'signal': float(macd_signal.iloc[-1]),
                'histogram': float(histogram.iloc[-1])
            }
        except Exception:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}

    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, float]:
        """Calculate Bollinger Bands."""
        try:
            sma = calculate_sma(prices, period)
            std = prices.rolling(window=period).std()

            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)

            current_price = prices.iloc[-1]
            bb_width = upper.iloc[-1] - lower.iloc[-1]

            # %B indicator
            if bb_width > 0:
                percent_b = (current_price - lower.iloc[-1]) / bb_width
            else:
                percent_b = 0.5

            return {
                'upper': float(upper.iloc[-1]),
                'middle': float(sma.iloc[-1]),
                'lower': float(lower.iloc[-1]),
                'width': float(bb_width),
                'percent_b': float(percent_b)
            }
        except Exception:
            return {
                'upper': 0.0, 'middle': 0.0, 'lower': 0.0,
                'width': 0.0, 'percent_b': 0.5
            }

    def _fetch_vix_sync(self) -> Optional[pd.DataFrame]:
        """Synchronously fetch VIX data."""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            data = vix.history(period="5d", interval="1d")
            if not data.empty:
                data.columns = [c.lower() for c in data.columns]
                return data
        except Exception:
            pass
        return None

    async def _store_features_async(self, symbol: str, features: Dict[str, float]):
        """Store features to database asynchronously."""
        if not self.engine:
            return

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self._store_features_sync,
                symbol,
                features
            )
        except Exception as e:
            logger.error(f"Error storing features to database: {e}")

    def _store_features_sync(self, symbol: str, features: Dict[str, float]):
        """Store features to database synchronously."""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    self.features_table.insert().values(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        features=features,
                        feature_version='1.0'
                    )
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Database insert failed: {e}")

    def get_historical_features(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve historical features from database.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date for query
            end_date: End date for query
            limit: Maximum number of records to return

        Returns:
            DataFrame with historical features or None
        """
        if not self.engine:
            logger.warning("Database not initialized")
            return None

        try:
            query = f"""
                SELECT timestamp, features
                FROM ml_features
                WHERE symbol = :symbol
            """

            params = {'symbol': symbol}

            if start_date:
                query += " AND timestamp >= :start_date"
                params['start_date'] = start_date

            if end_date:
                query += " AND timestamp <= :end_date"
                params['end_date'] = end_date

            query += " ORDER BY timestamp DESC LIMIT :limit"
            params['limit'] = limit

            df = pd.read_sql(query, self.engine, params=params)

            if df.empty:
                return None

            # Expand features JSON into columns
            features_df = pd.json_normalize(df['features'])
            features_df['timestamp'] = df['timestamp'].values

            return features_df

        except Exception as e:
            logger.error(f"Error retrieving historical features: {e}")
            return None

    def clear_cache(self):
        """Clear feature cache."""
        self._feature_cache.clear()
        self._cache_times.clear()
        logger.info("Feature cache cleared")

    def get_feature_names(self, category: Optional[str] = None) -> List[str]:
        """
        Get feature names by category.

        Args:
            category: Feature category ('technical', 'microstructure', 'regime',
                     'temporal', 'derived', or None for all)

        Returns:
            List of feature names
        """
        if category == 'technical':
            return self.technical_features
        elif category == 'microstructure':
            return self.microstructure_features
        elif category == 'regime':
            return self.regime_features
        elif category == 'temporal':
            return self.temporal_features
        elif category == 'derived':
            return self.derived_features
        elif category == 'murphy':
            return self.murphy_features
        else:
            return self.all_feature_names

    async def calculate_batch_features(
        self,
        symbols: List[str],
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate features for multiple symbols in parallel.

        Args:
            symbols: List of stock ticker symbols
            use_cache: Whether to use cached features

        Returns:
            Dictionary mapping symbols to feature DataFrames
        """
        tasks = [self.calculate_features(symbol, use_cache) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        features_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, pd.DataFrame):
                features_dict[symbol] = result
            else:
                logger.warning(f"Failed to calculate features for {symbol}")

        return features_dict

    def save_cache(self, filepath: str):
        """Save feature cache to disk."""
        try:
            cache_data = {
                'features': self._feature_cache,
                'times': self._cache_times
            }
            safe_save_model(cache_data, filepath)
            logger.info(f"Feature cache saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def load_cache(self, filepath: str):
        """Load feature cache from disk."""
        try:
            cache_data = safe_load_model(filepath, allow_unverified=False)
            self._feature_cache = cache_data['features']
            self._cache_times = cache_data['times']
            logger.info(f"Feature cache loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")

    def extract_features(self, signal: Dict) -> Dict:
        """
        Extract basic features from signal data synchronously.

        This is a lightweight method for quick feature extraction from
        signal dictionaries without requiring async data fetching.

        Args:
            signal: Raw signal data dictionary

        Returns:
            Dictionary containing:
            - features: numpy array of features
            - feature_names: list of feature names
            - top_features: list of most important features for this signal
        """
        features = []
        top_features = []

        # Feature 1: RRS (Relative Strength)
        rrs = signal.get('rrs', 0)
        features.append(rrs)
        if abs(rrs) > 2.5:
            top_features.append('rrs')

        # Feature 2: ATR as percent of price
        price = signal.get('price', 1)
        atr = signal.get('atr', 0)
        atr_percent = (atr / price * 100) if price > 0 else 0
        features.append(atr_percent)
        if atr_percent > 2.0:
            top_features.append('atr_percent')

        # Feature 3: Daily alignment - strong
        daily_strong = 1.0 if signal.get('daily_strong', False) else 0.0
        features.append(daily_strong)
        if daily_strong > 0:
            top_features.append('daily_strong')

        # Feature 4: Daily alignment - weak
        daily_weak = 1.0 if signal.get('daily_weak', False) else 0.0
        features.append(daily_weak)
        if daily_weak > 0:
            top_features.append('daily_weak')

        # Feature 5: Price normalized (log scale)
        price_normalized = np.log10(price) if price > 0 else 0
        features.append(price_normalized)

        # Feature 6: Volume ratio (if available)
        volume_ratio = signal.get('volume_ratio', 1.0)
        features.append(volume_ratio)
        if volume_ratio > 1.5:
            top_features.append('volume_ratio')

        # Feature 7: Trend strength (derived from RRS)
        trend_strength = min(abs(rrs) / 5.0, 1.0)  # Normalize to 0-1
        features.append(trend_strength)
        if trend_strength > 0.6:
            top_features.append('trend_strength')

        # Feature 8: Volatility percentile (derived from ATR)
        volatility_percentile = min(atr_percent / 5.0, 1.0)  # Normalize to 0-1
        features.append(volatility_percentile)

        # Default top features if none identified
        if not top_features:
            top_features = ['rrs', 'atr_percent', 'trend_strength']

        feature_names = [
            'rrs', 'atr_percent', 'daily_strong', 'daily_weak',
            'price_normalized', 'volume_ratio', 'trend_strength',
            'volatility_percentile'
        ]

        return {
            'features': np.array(features),
            'feature_names': feature_names,
            'top_features': top_features[:5]  # Limit to top 5
        }


# Convenience function for quick feature calculation
async def calculate_features_for_symbol(
    symbol: str,
    db_url: Optional[str] = None,
    use_cache: bool = True
) -> Optional[pd.DataFrame]:
    """
    Convenience function to calculate features for a single symbol.

    Args:
        symbol: Stock ticker symbol
        db_url: Database connection string (optional)
        use_cache: Whether to use cached features

    Returns:
        DataFrame with features or None
    """
    engineer = FeatureEngineer(db_url=db_url)
    return await engineer.calculate_features(symbol, use_cache=use_cache)
