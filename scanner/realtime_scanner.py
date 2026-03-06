"""
Real-Time RRS Scanner
Scans stocks for relative strength/weakness and sends alerts

This is the SEMI-AUTOMATED system - it scans and alerts, you execute manually
"""

import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import fcntl
import json
import tempfile
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from loguru import logger

from utils.timezone import (
    get_eastern_time,
    format_timestamp,
    is_market_open,
    to_eastern,
    is_premarket,
    is_afterhours,
    is_extended_hours,
    get_extended_hours_session,
    get_market_open_time,
    get_market_close_time,
    is_trading_day,
)

# News sentiment filter (optional)
try:
    from scanner.news_filter import get_news_filter, NewsFilter
    NEWS_FILTER_AVAILABLE = True
except ImportError:
    NEWS_FILTER_AVAILABLE = False
    logger.debug("News sentiment filter not available")

from shared.indicators.rrs import RRSCalculator, check_daily_strength, check_daily_weakness, check_daily_strength_relaxed, check_daily_weakness_relaxed, calculate_vwap
from alerts.notifier import send_alert
from scanner.signal_validator import validate_signal_quality
from scanner.signal_metrics import get_metrics_tracker

# Multi-timeframe analysis
try:
    from scanner.timeframe_analyzer import TimeframeAnalyzer, Timeframe, MTFAnalysisResult
    from scanner.trend_detector import TrendDetector, TrendResult
    MTF_AVAILABLE = True
except ImportError:
    MTF_AVAILABLE = False
    logger.debug("Multi-timeframe analysis modules not available")

# Data providers with automatic fallback
try:
    from data.providers import ProviderManager, ProviderError
    from data.providers.provider_manager import get_provider_manager
    PROVIDERS_AVAILABLE = True
except ImportError:
    PROVIDERS_AVAILABLE = False
    logger.warning("Data providers module not available")

# Module-level SPY price cache (price, timestamp) with 30-second TTL
_spy_cache: Dict = {}
_SPY_CACHE_TTL = 30  # seconds

# WebSocket broadcasting support (optional)
try:
    from web.websocket import (
        broadcast_signal,
        broadcast_scan_started,
        broadcast_scan_progress,
        broadcast_scan_completed,
        broadcast_scan_error
    )
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger.debug("WebSocket broadcasting not available")

# Mean reversion scanner (optional complement to momentum)
try:
    from scanner.mean_reversion_scanner import run_mean_reversion_scan
    MEAN_REVERSION_AVAILABLE = True
except ImportError:
    MEAN_REVERSION_AVAILABLE = False
    logger.debug("Mean reversion scanner not available")

# VIX-based market regime filter (optional)
try:
    from scanner.vix_filter import VIXFilter
    VIX_FILTER_AVAILABLE = True
except ImportError:
    VIX_FILTER_AVAILABLE = False
    logger.debug("VIX filter not available")

# Signal decay predictor (optional -- graceful degradation)
try:
    from ml.signal_decay import SignalDecayPredictor
    DECAY_PREDICTOR_AVAILABLE = True
except ImportError:
    DECAY_PREDICTOR_AVAILABLE = False
    logger.debug("Signal decay predictor not available")

# Regime-adaptive parameter system (optional)
try:
    from scanner.regime_params import RegimeAdaptiveParams
    REGIME_PARAMS_AVAILABLE = True
except ImportError:
    REGIME_PARAMS_AVAILABLE = False
    logger.debug("Regime-adaptive parameters not available")

# Sector strength filter (optional -- RDT "wind at your back")
try:
    from scanner.sector_filter import SectorStrengthFilter
    SECTOR_FILTER_AVAILABLE = True
except Exception as e:
    SECTOR_FILTER_AVAILABLE = False
    logger.warning(f"Sector strength filter not available: {type(e).__name__}: {e}")

# Intermarket analysis (optional -- Murphy framework, leading macro indicator)
try:
    from scanner.intermarket_analyzer import IntermarketAnalyzer
    INTERMARKET_AVAILABLE = True
except ImportError:
    INTERMARKET_AVAILABLE = False
    logger.debug("Intermarket analyzer not available")

# Regime detector for market state detection
try:
    from ml.regime_detector import RegimeDetector as _RegimeDetector
    REGIME_DETECTOR_AVAILABLE = True
except ImportError:
    REGIME_DETECTOR_AVAILABLE = False
    logger.debug("Regime detector not available for scanner")

# Prometheus metrics support (optional)
try:
    from monitoring.metrics import (
        record_signal,
        record_scanner_duration,
        set_market_status,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logger.debug("Prometheus metrics not available")


class RealTimeScanner:
    """Scan stocks in real-time for RRS signals"""

    def __init__(self, config: Dict):
        """
        Initialize scanner

        Args:
            config: Configuration dictionary with settings
        """
        self.config = config
        self.rrs_calc = RRSCalculator(atr_period=config.get('atr_period', 14))
        self.watchlist = self.load_watchlist()
        self.spy_data = None
        self.last_alerts = {}  # Track last alert time to avoid spam

        # First-hour scanning filter (RDT methodology: avoid first 30-60 min)
        self._first_hour_cutoff_minutes = config.get('first_hour_cutoff_minutes', 30)
        self._first_hour_filter_enabled = config.get('first_hour_filter_enabled', True)

        if self._first_hour_filter_enabled:
            logger.info(
                f"First-hour filter enabled: skipping scans for "
                f"{self._first_hour_cutoff_minutes} minutes after market open (RDT methodology)"
            )

        # News sentiment pre-filter
        self._news_filter: Optional[NewsFilter] = None
        self._news_filter_enabled = config.get('news_filter_enabled', True) and NEWS_FILTER_AVAILABLE
        if self._news_filter_enabled:
            try:
                news_cache_ttl = config.get('news_cache_ttl_minutes', 15)
                self._news_filter = get_news_filter(cache_ttl_minutes=news_cache_ttl)
                logger.info(f"News sentiment filter enabled (cache TTL={news_cache_ttl}min)")
            except Exception as e:
                logger.warning(f"Failed to initialize news filter: {e}")
                self._news_filter_enabled = False

        # Extended hours scanning configuration
        self._premarket_scan_enabled = config.get('premarket_scan_enabled', False)
        self._afterhours_scan_enabled = config.get('afterhours_scan_enabled', False)
        self._extended_hours_data_source = config.get('extended_hours_data_source', 'auto')

        if self._premarket_scan_enabled or self._afterhours_scan_enabled:
            logger.info(
                f"Extended hours scanning enabled - "
                f"Pre-market: {self._premarket_scan_enabled}, "
                f"After-hours: {self._afterhours_scan_enabled}"
            )

        # Initialize provider manager for redundant data fetching
        self._provider_manager: Optional[ProviderManager] = None
        self._use_providers = config.get('use_data_providers', True) and PROVIDERS_AVAILABLE

        if self._use_providers:
            try:
                self._provider_manager = get_provider_manager()
                provider_order = self._provider_manager.get_provider_order()
                logger.info(f"Scanner using redundant data providers: {provider_order}")
            except Exception as e:
                logger.warning(f"Failed to initialize provider manager: {e}")
                self._use_providers = False

        # Initialize multi-timeframe analyzer
        # Previous default: True (disabled for performance — MTF adds ~6 min to scan)
        self._mtf_enabled = config.get('mtf_enabled', False) and MTF_AVAILABLE
        self._mtf_analyzer: Optional[TimeframeAnalyzer] = None
        self._mtf_timeframes: List[Timeframe] = []
        self._mtf_alignment_required = config.get('mtf_alignment_required', False)
        self._mtf_alignment_boost = config.get('mtf_alignment_boost', 0.5)
        self._mtf_cache: Dict[str, Dict] = {}  # Cache for MTF data per symbol
        self._mtf_cache_ttl = config.get('mtf_cache_ttl', 60)  # seconds
        self._mtf_cache_max_size = config.get('mtf_cache_max_size', 200)
        self._mtf_cache_last_prune = 0.0  # timestamp of last prune

        if self._mtf_enabled:
            try:
                # Parse timeframes from config
                tf_config = config.get('mtf_timeframes', '5m,15m,1h,4h,1d')
                if isinstance(tf_config, str):
                    tf_strings = [tf.strip() for tf in tf_config.split(',')]
                else:
                    tf_strings = tf_config

                self._mtf_timeframes = [Timeframe.from_string(tf) for tf in tf_strings]

                # Initialize analyzer
                self._mtf_analyzer = TimeframeAnalyzer(
                    timeframes=self._mtf_timeframes,
                    trend_detector=TrendDetector()
                )
                logger.info(f"MTF analysis enabled with timeframes: {[tf.value for tf in self._mtf_timeframes]}")
            except Exception as e:
                logger.warning(f"Failed to initialize MTF analyzer: {e}")
                self._mtf_enabled = False

        # Initialize VIX regime filter (optional)
        self._vix_filter: Optional[VIXFilter] = None
        self._vix_filter_enabled = config.get('vix_filter_enabled', True) and VIX_FILTER_AVAILABLE
        if self._vix_filter_enabled:
            vix_cache_ttl = config.get('vix_cache_ttl', 300)  # 5 minutes
            self._vix_filter = VIXFilter(cache_ttl=vix_cache_ttl)
            logger.info("VIX regime filter enabled")

        # Initialize signal decay predictor (optional)
        self._decay_predictor: Optional[SignalDecayPredictor] = None
        self._decay_predictor_enabled = config.get('decay_predictor_enabled', True) and DECAY_PREDICTOR_AVAILABLE
        if self._decay_predictor_enabled:
            try:
                self._decay_predictor = SignalDecayPredictor()
                self._decay_predictor.load()
                logger.info("Signal decay predictor loaded successfully")
            except FileNotFoundError:
                logger.info("Signal decay predictor model not found — using default TTL (train model first)")
                self._decay_predictor = None
                self._decay_predictor_enabled = False
            except Exception as e:
                logger.warning(f"Failed to load signal decay predictor: {e} — using default TTL")
                self._decay_predictor = None
                self._decay_predictor_enabled = False

        # Initialize regime-adaptive parameter system
        self._regime_params: Optional[RegimeAdaptiveParams] = None
        self._regime_detector_scanner: Optional[_RegimeDetector] = None
        self._regime_params_enabled = (
            config.get('regime_params_enabled', True)
            and REGIME_PARAMS_AVAILABLE
            and REGIME_DETECTOR_AVAILABLE
        )
        if self._regime_params_enabled:
            try:
                self._regime_params = RegimeAdaptiveParams()
                self._regime_detector_scanner = _RegimeDetector()
                logger.info("Regime-adaptive parameter system enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize regime-adaptive params: {e}")
                self._regime_params_enabled = False

        # Initialize sector strength filter (RDT "wind at your back")
        self._sector_filter: Optional[SectorStrengthFilter] = None
        self._sector_filter_enabled = config.get('sector_filter_enabled', True) and SECTOR_FILTER_AVAILABLE
        if self._sector_filter_enabled:
            try:
                sector_cache_ttl = config.get('sector_cache_ttl', 1800)  # 30 minutes
                self._sector_filter = SectorStrengthFilter(cache_ttl=sector_cache_ttl)
                logger.info("Sector strength filter enabled (RDT methodology)")
            except Exception as e:
                logger.warning(f"Failed to initialize sector strength filter: {e}")
                self._sector_filter_enabled = False

        # Initialize intermarket analyzer (Murphy framework -- leading macro indicator)
        self._intermarket_analyzer: Optional[IntermarketAnalyzer] = None
        self._intermarket_enabled = config.get('intermarket_enabled', True) and INTERMARKET_AVAILABLE
        if self._intermarket_enabled:
            try:
                intermarket_cache_ttl = config.get('intermarket_cache_ttl_minutes', 30)
                self._intermarket_analyzer = IntermarketAnalyzer(
                    cache_ttl_minutes=intermarket_cache_ttl
                )
                logger.info("Intermarket analyzer enabled (Murphy framework)")
            except Exception as e:
                logger.warning(f"Failed to initialize intermarket analyzer: {e}")
                self._intermarket_enabled = False

        # Daily SMA filter: require price above/below 50 SMA per direction (RDT methodology)
        self._daily_sma_filter_enabled = config.get('daily_sma_filter_enabled', True)
        self._daily_sma_cache: Dict[str, Dict] = {}  # {symbol: {data, timestamp}}
        self._daily_sma_cache_ttl = 86400  # 24 hours — daily SMAs only change once per day
        if self._daily_sma_filter_enabled:
            logger.info(
                "Daily SMA filter enabled — LONG signals require price > 50 SMA, "
                "SHORT signals require price < 50 SMA (RDT methodology)"
            )

        # SPY Hard Gate: block signals that go against SPY daily trend (RDT "Market First")
        self._spy_gate_enabled = config.get('spy_gate_enabled', True)
        if self._spy_gate_enabled:
            logger.info(
                "SPY hard gate enabled — LONG signals blocked when SPY bearish, "
                "SHORT signals blocked when SPY bullish (RDT: Market First)"
            )

        # VWAP Gate: day trades must respect VWAP (RDT methodology)
        # LONG signals require price > VWAP, SHORT signals require price < VWAP
        self._vwap_filter_enabled = config.get('vwap_filter_enabled', True)
        if self._vwap_filter_enabled:
            logger.info(
                "VWAP filter enabled — LONG signals require price above VWAP, "
                "SHORT signals require price below VWAP (RDT methodology)"
            )

        # Lightweight MTF Gate: resamples existing 5m data to 15m/1h — zero API calls
        # Replaces the old heavy MTF that made per-symbol per-timeframe API calls (~6 min)
        self._mtf_lightweight_enabled = config.get('mtf_lightweight_enabled', True)
        if self._mtf_lightweight_enabled:
            logger.info(
                "Lightweight MTF gate enabled — blocks signals with weak (<3/4) "
                "timeframe alignment (zero extra API calls)"
            )

        logger.info(f"Scanner initialized with {len(self.watchlist)} stocks")

    def _is_first_hour_restricted(self) -> bool:
        """
        Check if scanning should be skipped due to the RDT first-hour rule.

        The r/RealDayTrading methodology recommends avoiding trades during the
        first 30-60 minutes after market open because RS/RW signals are unreliable
        during this period. Also skips scanning after market close.

        Returns:
            True if scanning should be skipped, False if OK to scan.
        """
        if not self._first_hour_filter_enabled:
            return False

        now = get_eastern_time()

        # Not a trading day — let other checks handle this
        if not is_trading_day(now.date()):
            return False

        market_open = get_market_open_time(now.date())
        market_close = get_market_close_time(now.date())

        # Before market open — don't restrict (premarket scanning has its own check)
        if now < market_open:
            return False

        # After market close
        if now >= market_close:
            logger.info("Skipping scan - market is closed (after 4:00 PM ET)")
            return True

        # During first N minutes after open
        from datetime import timedelta
        cutoff_time = market_open + timedelta(minutes=self._first_hour_cutoff_minutes)
        if now < cutoff_time:
            minutes_remaining = int((cutoff_time - now).total_seconds() / 60)
            logger.info(
                f"Skipping scan - first {self._first_hour_cutoff_minutes} minutes "
                f"(RDT methodology). {minutes_remaining} min until scanning starts."
            )
            return True

        return False

    def _apply_news_sentiment(self, signal: Dict) -> Dict:
        """
        Apply news sentiment check to a signal and add warning flags.

        Per RDT methodology, price action is king — news sentiment adds a
        warning flag but does NOT block the signal.

        Args:
            signal: Signal dict (must have 'symbol' and 'direction' keys)

        Returns:
            Signal dict with news_sentiment fields added
        """
        if not self._news_filter_enabled or self._news_filter is None:
            return signal

        symbol = signal.get('symbol', '')
        direction = signal.get('direction', 'long')

        try:
            news_result = self._news_filter.check_symbol(symbol)

            signal['news_sentiment_score'] = news_result['sentiment_score']
            signal['news_headlines'] = news_result['headlines'][:3]  # Keep top 3

            # Flag strongly negative sentiment for LONG signals
            if direction == 'long' and news_result['has_negative_news']:
                signal['news_warning'] = True
                signal['news_warning_reason'] = (
                    f"Strongly negative news sentiment ({news_result['sentiment_score']:.2f})"
                )
                logger.warning(
                    f"[NEWS WARNING] {symbol} LONG signal has negative news "
                    f"(sentiment={news_result['sentiment_score']:.2f}). "
                    f"Signal preserved per RDT methodology (price action > news)."
                )
            # Flag strongly positive sentiment for SHORT signals
            elif direction == 'short' and news_result['sentiment_score'] > 0.5:
                signal['news_warning'] = True
                signal['news_warning_reason'] = (
                    f"Strongly positive news sentiment ({news_result['sentiment_score']:.2f})"
                )
                logger.warning(
                    f"[NEWS WARNING] {symbol} SHORT signal has positive news "
                    f"(sentiment={news_result['sentiment_score']:.2f}). "
                    f"Signal preserved per RDT methodology (price action > news)."
                )
            else:
                signal['news_warning'] = False

        except Exception as e:
            logger.debug(f"News sentiment check failed for {symbol}: {e}")
            signal['news_sentiment_score'] = 0.0
            signal['news_headlines'] = []
            signal['news_warning'] = False

        return signal

    def _apply_sector_filter(self, signal: Dict, direction: str,
                             spy_daily_trend: Optional[Dict] = None) -> Dict:
        """
        Apply sector strength filter to a signal.

        RDT methodology: trade with the sector ("wind at your back").
        - Adjusts RRS based on sector alignment
        - Adds sector metadata to the signal
        - Warns if SPY trend is bearish and signal is long

        Does NOT block signals — price action is still king per RDT.

        Args:
            signal: Signal dict
            direction: 'long' or 'short'
            spy_daily_trend: SPY daily trend dict (or None)

        Returns:
            Signal dict with sector fields added
        """
        if not self._sector_filter_enabled or self._sector_filter is None:
            return signal

        symbol = signal.get('symbol', '')

        try:
            # Get sector strength info
            sector_info = self._sector_filter.get_sector_strength(symbol)
            signal['sector'] = sector_info['sector']
            signal['sector_etf'] = sector_info['sector_etf']
            signal['sector_rs'] = float(round(sector_info['sector_rs'], 4))

            # Get boost/penalty
            boost, reason = self._sector_filter.should_boost_signal(symbol, direction)
            if boost != 0.0:
                original_rrs = signal.get('rrs', 0.0)
                adjusted_rrs = round(original_rrs + boost, 2)
                signal['rrs_original'] = float(round(original_rrs, 2))
                signal['rrs'] = float(adjusted_rrs)
                signal['rrs_boosted'] = True
                signal['sector_boost'] = float(boost)
                signal['sector_boost_reason'] = reason
                logger.debug(
                    f"[SECTOR] {symbol} {direction}: RRS {original_rrs:.2f} -> "
                    f"{adjusted_rrs:.2f} ({boost:+.2f}): {reason}"
                )
            else:
                signal['sector_boost'] = 0.0
                signal['sector_boost_reason'] = reason

            # Add SPY trend metadata
            if spy_daily_trend is not None:
                signal['spy_trend'] = spy_daily_trend['daily_trend']

                # Warn if SPY bearish and signal is long (don't block — RDT: price action is king)
                if direction == 'long' and spy_daily_trend['daily_trend'] == 'bearish':
                    signal['spy_trend_warning'] = True
                    signal['spy_trend_warning_reason'] = (
                        "SPY below 50 & 200 EMA (bearish daily trend) — "
                        "long signal preserved but use caution"
                    )
                    logger.warning(
                        f"[SPY TREND WARNING] {symbol} LONG signal while SPY trend "
                        f"is BEARISH. Signal preserved per RDT methodology."
                    )
                else:
                    signal['spy_trend_warning'] = False
            else:
                signal['spy_trend'] = 'unknown'
                signal['spy_trend_warning'] = False

        except Exception as e:
            logger.debug(f"Sector filter failed for {symbol}: {e}")
            signal['sector'] = 'unknown'
            signal['sector_etf'] = ''
            signal['sector_rs'] = 0.0
            signal['sector_boost'] = 0.0
            signal['spy_trend'] = 'unknown'
            signal['spy_trend_warning'] = False

        return signal

    def _prune_mtf_cache(self):
        """Remove expired entries from MTF cache to prevent unbounded growth."""
        now = time.time()
        # Only prune at most once per TTL period
        if now - self._mtf_cache_last_prune < self._mtf_cache_ttl:
            return
        self._mtf_cache_last_prune = now

        now_dt = get_eastern_time()
        stale_keys = [
            k for k, v in self._mtf_cache.items()
            if (now_dt - v.get('timestamp', datetime.min)).total_seconds() > self._mtf_cache_ttl
        ]
        for k in stale_keys:
            del self._mtf_cache[k]
        if stale_keys:
            logger.debug(f"Pruned {len(stale_keys)} stale MTF cache entries")

        # Also enforce max size: remove oldest half if exceeded
        if len(self._mtf_cache) > self._mtf_cache_max_size:
            sorted_keys = sorted(self._mtf_cache.keys(),
                               key=lambda k: self._mtf_cache[k].get('timestamp', datetime.min))
            for key in sorted_keys[:len(self._mtf_cache) // 2]:
                del self._mtf_cache[key]
            logger.debug(f"Pruned MTF cache to {len(self._mtf_cache)} entries")

    def _should_scan_now(self) -> bool:
        """
        Check if scanning should occur based on current time and configuration.

        Returns:
            True if scanning should proceed, False otherwise.
        """
        session = get_extended_hours_session()

        if session == 'regular':
            return True

        if session == 'closed':
            return False

        if session == 'premarket':
            return self._premarket_scan_enabled

        if session == 'afterhours':
            return self._afterhours_scan_enabled

        return False

    def _get_current_session(self) -> str:
        """
        Get the current trading session.

        Returns:
            One of 'premarket', 'regular', 'afterhours', or 'closed'
        """
        return get_extended_hours_session()

    def _is_extended_hours(self) -> bool:
        """
        Check if currently in extended hours.

        Returns:
            True if in pre-market or after-hours, False otherwise.
        """
        return is_extended_hours()

    def load_watchlist(self) -> List[str]:
        """
        Load watchlist of stocks to scan

        Returns:
            List of ticker symbols
        """
        # High-volume stocks optimized for fast batch scanning (33 stocks in ~30s with batch download)
        watchlist = [
            # Mega-cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            # Semiconductors
            'AMD', 'AVGO', 'QCOM', 'INTC', 'MU',
            # Financials
            'JPM', 'BAC', 'GS', 'MS',
            # Consumer
            'WMT', 'HD', 'COST', 'MCD', 'SBUX',
            # Healthcare
            'UNH', 'JNJ', 'PFE', 'ABBV',
            # Energy/Materials
            'XOM', 'CVX', 'FCX', 'NEM',
            # Consumer Goods
            'PG', 'KO', 'PEP', 'GIS'
        ]
        return watchlist

    def fetch_batch_data_with_providers(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Fetch all stock data using the ProviderManager with automatic fallback.

        Returns:
            Tuple of (batch_5m_data, batch_daily_data) or (None, None) on failure
        """
        symbols = self.watchlist + ['SPY']

        try:
            logger.info(f"Fetching batch data for {len(symbols)} symbols via ProviderManager")

            # Use ProviderManager batch historical
            try:
                batch_5m_data = self._provider_manager.get_batch_historical(symbols, period="1d", interval="5m")
                batch_daily_data = self._provider_manager.get_batch_historical(symbols, period="60d", interval="1d")

                if batch_5m_data and batch_daily_data:
                    # Convert HistoricalData dict to DataFrame format expected by existing code
                    batch_5m = self._convert_historical_dict_to_batch_df(batch_5m_data)
                    batch_daily = self._convert_historical_dict_to_batch_df(batch_daily_data)

                    if not batch_5m.empty and not batch_daily.empty:
                        n5m = len(batch_5m.columns.get_level_values(0).unique()) if isinstance(batch_5m.columns, pd.MultiIndex) else 0
                        nday = len(batch_daily.columns.get_level_values(0).unique()) if isinstance(batch_daily.columns, pd.MultiIndex) else 0
                        logger.info(f"Successfully fetched batch data via ProviderManager (5m: {n5m} symbols, daily: {nday} symbols)")
                        return batch_5m, batch_daily

            except Exception as e:
                logger.warning(f"Batch fetch via provider failed: {e}")

            # Fall back to individual requests through provider manager
            logger.info("Falling back to individual symbol requests via ProviderManager")
            batch_5m, batch_daily = self._fetch_individual_via_providers(symbols)

            if batch_5m is not None and batch_daily is not None:
                return batch_5m, batch_daily

            # If provider manager fails completely, signal to use legacy method
            return None, None

        except Exception as e:
            logger.error(f"ProviderManager batch fetch failed: {e}")
            return None, None

    def _convert_historical_dict_to_batch_df(self, data_dict: Dict) -> pd.DataFrame:
        """
        Convert dictionary of HistoricalData objects to batch DataFrame format.

        Args:
            data_dict: Dict mapping symbols to HistoricalData objects

        Returns:
            Multi-level DataFrame with (symbol, OHLCV) columns
        """
        if not data_dict:
            return pd.DataFrame()

        try:
            # Build multi-level column DataFrame
            frames = {}
            for symbol, hist_data in data_dict.items():
                if hist_data and not hist_data.data.empty:
                    df = hist_data.data.copy()
                    # Normalize column names to lowercase for consistency (Fix 4)
                    df.columns = [c.lower() for c in df.columns]
                    frames[symbol] = df

            if not frames:
                return pd.DataFrame()

            # Concatenate into multi-level columns
            result = pd.concat(frames, axis=1)
            return result

        except Exception as e:
            logger.debug(f"Error converting historical data to batch format: {e}")
            return pd.DataFrame()

    def _fetch_individual_via_providers(self, symbols: List[str]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Fetch data for symbols individually through provider manager.

        Args:
            symbols: List of symbols to fetch

        Returns:
            Tuple of (batch_5m_df, batch_daily_df) or (None, None)
        """
        try:
            frames_5m = {}
            frames_daily = {}

            for symbol in symbols:
                try:
                    # Get 5-minute data
                    hist_5m = self._provider_manager.get_historical(symbol, period="1d", interval="5m")
                    if hist_5m and not hist_5m.data.empty:
                        df_5m = hist_5m.data.copy()
                        df_5m.columns = [c.lower() for c in df_5m.columns]
                        frames_5m[symbol] = df_5m

                    # Get daily data
                    hist_daily = self._provider_manager.get_historical(symbol, period="60d", interval="1d")
                    if hist_daily and not hist_daily.data.empty:
                        df_daily = hist_daily.data.copy()
                        df_daily.columns = [c.lower() for c in df_daily.columns]
                        frames_daily[symbol] = df_daily

                except Exception as e:
                    logger.debug(f"Failed to fetch {symbol} via providers: {e}")
                    continue

            if not frames_5m or not frames_daily:
                return None, None

            batch_5m = pd.concat(frames_5m, axis=1)
            batch_daily = pd.concat(frames_daily, axis=1)

            return batch_5m, batch_daily

        except Exception as e:
            logger.error(f"Individual fetch via providers failed: {e}")
            return None, None

    def fetch_batch_data(self, max_retries: int = 3) -> tuple:
        """
        Fetch all stock data in batch using the ProviderManager with automatic
        failover via ProviderManager.

        Retry schedule uses exponential backoff: 1 s, 2 s, 4 s.

        Args:
            max_retries: Maximum number of retry attempts (default 3)

        Returns:
            Tuple of (batch_5m_data, batch_daily_data) or (None, None) on failure
        """
        symbols = self.watchlist + ['SPY']
        base_delay = 1.0  # seconds; doubled each retry

        for attempt in range(max_retries):
            wait_time = base_delay * (2 ** attempt)  # 1s, 2s, 4s
            try:
                logger.info(
                    f"Batch fetching data for {len(symbols)} symbols via ProviderManager "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

                # Prefer ProviderManager batch historical (handles failover internally)
                if self._use_providers and self._provider_manager:
                    hist_5m = self._provider_manager.get_batch_historical(
                        symbols, period='1d', interval='5m'
                    )
                    hist_daily = self._provider_manager.get_batch_historical(
                        symbols, period='60d', interval='1d'
                    )

                    if hist_5m and hist_daily:
                        batch_5m = self._convert_historical_dict_to_batch_df(hist_5m)
                        batch_daily = self._convert_historical_dict_to_batch_df(hist_daily)

                        if not batch_5m.empty and not batch_daily.empty:
                            logger.info(
                                f"Successfully fetched batch data for {len(symbols)} symbols "
                                f"via ProviderManager"
                            )
                            return batch_5m, batch_daily

                    raise ValueError("ProviderManager returned empty batch data")

            except Exception as e:
                error_msg = str(e).lower()
                if attempt < max_retries - 1:
                    if '401' in error_msg or 'rate' in error_msg or 'too many' in error_msg:
                        logger.warning(
                            f"Rate limited on batch download "
                            f"(attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {wait_time}s..."
                        )
                    else:
                        logger.warning(
                            f"Batch download error (attempt {attempt + 1}/{max_retries}): {e}; "
                            f"retrying in {wait_time}s"
                        )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Batch download failed after {max_retries} attempts: {e}"
                    )

        logger.error("Failed to fetch batch data after all retries")
        return None, None

    def _extract_spy_data(self, batch_5m: pd.DataFrame, batch_daily: pd.DataFrame) -> Dict:
        """
        Extract SPY data from batch download results.

        Args:
            batch_5m: Batch 5-minute data
            batch_daily: Batch daily data

        Returns:
            Dict with SPY data or None if extraction fails
        """
        try:
            if batch_5m is None or batch_daily is None:
                logger.warning("SCAN_WARN[BATCH_DATA_NONE] SPY extraction skipped due to missing batch data")
                return None
            if batch_5m.empty or batch_daily.empty:
                logger.warning("SCAN_WARN[BATCH_DATA_EMPTY] SPY extraction skipped due to empty batch data")
                return None

            # Extract SPY data from batch (multi-level columns: ticker -> OHLCV)
            if 'SPY' in batch_5m.columns.get_level_values(0):
                spy_5m = batch_5m['SPY'].dropna(how='all')
                spy_daily = batch_daily['SPY'].dropna(how='all')
            else:
                # Single ticker case - columns are just OHLCV
                spy_5m = batch_5m.dropna(how='all')
                spy_daily = batch_daily.dropna(how='all')

            if spy_5m.empty or spy_daily.empty:
                logger.error("SPY data is empty in batch")
                return None

            # Normalize column names to lowercase for consistency
            spy_5m.columns = spy_5m.columns.str.lower()
            spy_daily.columns = spy_daily.columns.str.lower()

            if len(spy_daily) < 2:
                logger.warning("Insufficient SPY daily data")
                return None

            self.spy_data = {
                '5m': spy_5m,
                'daily': spy_daily,
                'current_price': spy_5m['close'].iloc[-1],
                'previous_close': spy_daily['close'].iloc[-2]
            }

            return self.spy_data

        except Exception as e:
            logger.error(f"Error extracting SPY data from batch: {e}")
            return None

    def _estimate_beta(self, stock_data: Dict, batch_daily=None) -> Optional[float]:
        """
        Estimate stock beta vs SPY from 30-day daily returns.
        Returns None if insufficient data.
        """
        try:
            import numpy as np
            stock_daily = stock_data.get('daily')
            if stock_daily is None or len(stock_daily) < 30:
                return None

            # Get SPY daily returns
            spy_daily = None
            if batch_daily is not None and 'SPY' in batch_daily.columns.get_level_values(0):
                spy_daily = batch_daily['SPY'].dropna(how='all')
                spy_daily.columns = spy_daily.columns.str.lower()
            elif hasattr(self, 'spy_data') and self.spy_data and 'daily' in self.spy_data:
                spy_daily = self.spy_data['daily']

            if spy_daily is None or len(spy_daily) < 30:
                return None

            # Calculate 30-day returns
            stock_returns = stock_daily['close'].pct_change().iloc[-30:]
            spy_returns = spy_daily['close'].pct_change().iloc[-30:]

            # Align indices
            common_idx = stock_returns.index.intersection(spy_returns.index)
            if len(common_idx) < 20:
                return None

            sr = stock_returns.loc[common_idx].values
            mr = spy_returns.loc[common_idx].values

            # Remove NaN
            mask = ~(np.isnan(sr) | np.isnan(mr))
            sr, mr = sr[mask], mr[mask]
            if len(sr) < 20:
                return None

            # Beta = Cov(stock, market) / Var(market)
            cov = np.cov(sr, mr)
            if cov[1, 1] == 0:
                return None
            beta = cov[0, 1] / cov[1, 1]
            return float(beta)
        except Exception:
            return None

    def _extract_stock_data(self, symbol: str, batch_5m: pd.DataFrame, batch_daily: pd.DataFrame) -> Dict:
        """
        Extract single stock data from batch download results.

        Args:
            symbol: Stock ticker symbol
            batch_5m: Batch 5-minute data
            batch_daily: Batch daily data

        Returns:
            Dict with stock data or None if extraction fails
        """
        try:
            if batch_5m is None or batch_daily is None:
                logger.debug(f"SCAN_WARN[BATCH_DATA_NONE] {symbol}: missing batch data")
                return None
            if batch_5m.empty or batch_daily.empty:
                logger.debug(f"SCAN_WARN[BATCH_DATA_EMPTY] {symbol}: empty batch data")
                return None

            # Check if symbol exists in the batch data
            if symbol not in batch_5m.columns.get_level_values(0):
                logger.debug(f"{symbol} not found in batch data")
                return None
            if symbol not in batch_daily.columns.get_level_values(0):
                logger.debug(f"{symbol} missing in daily batch data")
                return None

            # Extract stock data from multi-level columns
            data_5m = batch_5m[symbol].dropna(how='all')
            data_daily = batch_daily[symbol].dropna(how='all')

            if data_5m.empty or data_daily.empty:
                logger.debug(f"{symbol} has empty data")
                return None

            # Normalize column names to lowercase
            data_5m.columns = data_5m.columns.str.lower()
            data_daily.columns = data_daily.columns.str.lower()

            # Calculate ATR from daily data
            atr_series = self.rrs_calc.calculate_atr(data_daily)
            if atr_series.empty:
                logger.debug(f"{symbol} has no ATR data")
                return None

            current_atr = atr_series.iloc[-1]

            if len(data_daily) < 2:
                logger.warning(f"Insufficient daily data for {symbol}")
                return None

            return {
                '5m': data_5m,
                'daily': data_daily,
                'current_price': data_5m['close'].iloc[-1],
                'previous_close': data_daily['close'].iloc[-2],
                'atr': current_atr,
                'volume': data_daily['volume'].iloc[-1]
            }

        except Exception as e:
            logger.debug(f"Error extracting {symbol} from batch: {e}")
            return None

    def fetch_spy_data(self) -> pd.DataFrame:
        """Fetch current SPY data (legacy method for backward compatibility)"""
        global _spy_cache

        # Return cached SPY data if it is still fresh (< 30 seconds old)
        if _spy_cache and (time.time() - _spy_cache.get('fetched_at', 0)) < _SPY_CACHE_TTL:
            logger.debug("Returning cached SPY data (TTL not expired)")
            self.spy_data = _spy_cache['spy_data']
            return self.spy_data

        try:
            # Try provider manager first if available
            if self._use_providers and self._provider_manager:
                try:
                    spy_5m = self._provider_manager.get_historical('SPY', period='1d', interval='5m')
                    spy_daily = self._provider_manager.get_historical('SPY', period='60d', interval='1d')

                    if spy_5m and spy_daily and not spy_5m.data.empty and not spy_daily.data.empty:
                        # Normalize column names to lowercase for consistency (Fix 4)
                        spy_5m_df = spy_5m.data.copy()
                        spy_daily_df = spy_daily.data.copy()
                        spy_5m_df.columns = spy_5m_df.columns.str.lower()
                        spy_daily_df.columns = spy_daily_df.columns.str.lower()

                        if len(spy_daily_df) < 2:
                            logger.warning("Insufficient SPY daily data from provider")
                        else:
                            self.spy_data = {
                                '5m': spy_5m_df,
                                'daily': spy_daily_df,
                                'current_price': spy_5m_df['close'].iloc[-1],
                                'previous_close': spy_daily_df['close'].iloc[-2]
                            }
                            _spy_cache['spy_data'] = self.spy_data
                            _spy_cache['fetched_at'] = time.time()
                            return self.spy_data

                except Exception as e:
                    logger.error(f"Provider manager SPY fetch failed: {e}")

            logger.error("Failed to fetch SPY data — no providers available")
            return None

        except Exception as e:
            logger.error(f"Error fetching SPY data: {e}")
            return None

    def fetch_stock_data(self, symbol: str) -> Dict:
        """
        Fetch stock data for RRS calculation (legacy method for backward compatibility)

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with stock data or None if error
        """
        try:
            # Try provider manager first if available
            if self._use_providers and self._provider_manager:
                try:
                    hist_5m = self._provider_manager.get_historical(symbol, period='1d', interval='5m')
                    hist_daily = self._provider_manager.get_historical(symbol, period='60d', interval='1d')

                    if hist_5m and hist_daily and not hist_5m.data.empty and not hist_daily.data.empty:
                        data_5m = hist_5m.data.copy()
                        data_daily = hist_daily.data.copy()

                        # Calculate ATR from daily data
                        atr_series = self.rrs_calc.calculate_atr(data_daily)
                        current_atr = atr_series.iloc[-1]

                        if len(data_daily) < 2:
                            logger.warning(f"Insufficient daily data for {symbol} from provider")
                            return None

                        return {
                            '5m': data_5m,
                            'daily': data_daily,
                            'current_price': data_5m['close'].iloc[-1],
                            'previous_close': data_daily['close'].iloc[-2],
                            'atr': current_atr,
                            'volume': data_daily['volume'].iloc[-1]
                        }

                except Exception as e:
                    logger.warning(f"ProviderManager fetch for {symbol} failed: {e}")

            logger.debug(f"No providers available for {symbol}")
            return None

        except Exception as e:
            logger.debug(f"Error fetching {symbol}: {e}")
            return None

    def calculate_stock_rrs(self, symbol: str, stock_data: Dict) -> Dict:
        """
        Calculate RRS for a stock

        Args:
            symbol: Stock ticker
            stock_data: Stock data dict

        Returns:
            Dict with RRS analysis
        """
        try:
            if self.spy_data is None:
                logger.warning(f"No SPY data available, skipping RRS for {symbol}")
                return None

            # Calculate current RRS
            rrs_data = self.rrs_calc.calculate_rrs_current(
                stock_data={
                    'current_price': stock_data['current_price'],
                    'previous_close': stock_data['previous_close']
                },
                spy_data={
                    'current_price': self.spy_data['current_price'],
                    'previous_close': self.spy_data['previous_close']
                },
                stock_atr=stock_data['atr']
            )

            # Check daily chart strength (relaxed criteria for more signals)
            daily_strength = check_daily_strength_relaxed(stock_data['daily'])
            daily_weakness = check_daily_weakness_relaxed(stock_data['daily'])

            return {
                'symbol': symbol,
                'rrs': rrs_data['rrs'],
                'status': rrs_data['status'],
                'stock_pc': rrs_data['stock_pc'],
                'spy_pc': rrs_data['spy_pc'],
                'price': stock_data['current_price'],
                'volume': stock_data['volume'],
                'atr': stock_data['atr'],
                'daily_strong': daily_strength['is_strong'],
                'daily_weak': daily_weakness['is_weak'],
                'ema3': daily_strength['ema3'],
                'ema8': daily_strength['ema8']
            }

        except Exception as e:
            logger.error(f"Error calculating RRS for {symbol}: {e}")
            return None

    def fetch_mtf_data(self, symbol: str) -> Dict[Timeframe, pd.DataFrame]:
        """
        Fetch multi-timeframe data for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict mapping Timeframe to DataFrame
        """
        if not self._mtf_enabled or not MTF_AVAILABLE:
            return {}

        data_by_tf = {}

        for tf in self._mtf_timeframes:
            try:
                period = tf.period_for_data
                interval = tf.yfinance_interval

                if self._use_providers and self._provider_manager:
                    try:
                        hist = self._provider_manager.get_historical(
                            symbol, period=period, interval=interval
                        )
                        if hist and not hist.data.empty:
                            df = hist.data.copy()
                            df.columns = [c.lower() for c in df.columns]
                            data_by_tf[tf] = df
                            continue
                    except Exception as e:
                        logger.debug(f"Provider failed for {symbol} {tf.value}: {e}")

                logger.debug(f"No provider data for {symbol} {tf.value}")

            except Exception as e:
                logger.debug(f"Error fetching {symbol} {tf.value}: {e}")
                continue

        return data_by_tf

    def analyze_mtf(self, symbol: str, stock_data: Dict) -> Optional[Dict]:
        """
        Perform multi-timeframe analysis for a symbol.

        Args:
            symbol: Stock ticker symbol
            stock_data: Existing stock data (contains 5m and daily data)

        Returns:
            Dict with MTF analysis results or None
        """
        if not self._mtf_enabled or not self._mtf_analyzer:
            return None

        try:
            # Check cache
            cache_key = symbol
            cache_entry = self._mtf_cache.get(cache_key)
            if cache_entry:
                cache_time = cache_entry.get('timestamp', datetime.min)
                if (get_eastern_time() - cache_time).total_seconds() < self._mtf_cache_ttl:
                    return cache_entry.get('analysis')

            # Build data dictionary using existing data where possible
            data_by_tf = {}

            # Use existing 5m and daily data if available
            if '5m' in stock_data and stock_data['5m'] is not None:
                df_5m = stock_data['5m'].copy()
                df_5m.columns = [c.lower() for c in df_5m.columns]
                data_by_tf[Timeframe.M5] = df_5m

            if 'daily' in stock_data and stock_data['daily'] is not None:
                df_daily = stock_data['daily'].copy()
                df_daily.columns = [c.lower() for c in df_daily.columns]
                data_by_tf[Timeframe.DAILY] = df_daily

            # Fetch additional timeframes if needed
            for tf in self._mtf_timeframes:
                if tf not in data_by_tf:
                    additional = self.fetch_mtf_data(symbol)
                    for add_tf, add_df in (additional or {}).items():
                        if add_tf not in data_by_tf:
                            data_by_tf[add_tf] = add_df
                    break  # Only need to fetch once

            if len(data_by_tf) < 2:
                return None

            # Perform full MTF analysis
            mtf_result = self._mtf_analyzer.full_analysis(symbol, data_by_tf)

            # Convert to dict for serialization
            analysis = mtf_result.to_dict()

            # Prune stale entries before adding new ones
            self._prune_mtf_cache()

            # Cache the result
            self._mtf_cache[cache_key] = {
                'timestamp': get_eastern_time(),
                'analysis': analysis
            }

            return analysis

        except Exception as e:
            logger.debug(f"MTF analysis error for {symbol}: {e}")
            return None

    def calculate_stock_rrs_with_mtf(self, symbol: str, stock_data: Dict) -> Dict:
        """
        Calculate RRS for a stock with MTF analysis integration.

        Args:
            symbol: Stock ticker
            stock_data: Stock data dict

        Returns:
            Dict with RRS analysis including MTF data
        """
        # Get base RRS analysis
        analysis = self.calculate_stock_rrs(symbol, stock_data)
        if analysis is None:
            return None

        # Add MTF analysis if enabled
        if self._mtf_enabled:
            mtf_analysis = self.analyze_mtf(symbol, stock_data)

            if mtf_analysis:
                # Add MTF fields to analysis
                analysis['timeframe_alignment'] = mtf_analysis.get('timeframe_alignment', False)
                analysis['trend_by_timeframe'] = mtf_analysis.get('trend_by_timeframe', {})
                analysis['entry_timing_score'] = mtf_analysis.get('entry_timing_score', 50)
                analysis['alignment_direction'] = mtf_analysis.get('alignment_direction', 'unknown')
                analysis['mtf_recommended_action'] = mtf_analysis.get('recommended_action', '')
                analysis['key_levels'] = mtf_analysis.get('key_levels', {})

                # Apply alignment boost to RRS if aligned in same direction
                if analysis['timeframe_alignment']:
                    rrs = analysis['rrs']
                    alignment_dir = analysis['alignment_direction']

                    # Boost RRS if alignment matches signal direction
                    if (rrs > 0 and alignment_dir == 'bullish') or \
                       (rrs < 0 and alignment_dir == 'bearish'):
                        boost = self._mtf_alignment_boost
                        if rrs > 0:
                            analysis['rrs'] = rrs + boost
                        else:
                            analysis['rrs'] = rrs - boost
                        analysis['rrs_original'] = rrs
                        analysis['rrs_boosted'] = True
            else:
                # No MTF analysis available - set defaults
                analysis['timeframe_alignment'] = False
                analysis['trend_by_timeframe'] = {}
                analysis['entry_timing_score'] = 50
                analysis['alignment_direction'] = 'unknown'
                analysis['mtf_recommended_action'] = ''
                analysis['key_levels'] = {}
        else:
            # MTF disabled - set defaults
            analysis['timeframe_alignment'] = False
            analysis['trend_by_timeframe'] = {}
            analysis['entry_timing_score'] = 50
            analysis['alignment_direction'] = 'unknown'
            analysis['mtf_recommended_action'] = ''
            analysis['key_levels'] = {}

        return analysis

    def should_alert(self, symbol: str, rrs: float) -> bool:
        """
        Check if we should send an alert for this stock

        Args:
            symbol: Stock ticker
            rrs: RRS value

        Returns:
            bool: True if should alert
        """
        # Prune last_alerts if it grows too large (prevent unbounded growth)
        if len(self.last_alerts) > 500:
            now = get_eastern_time()
            cutoff = now - timedelta(hours=1)
            self.last_alerts = {k: v for k, v in self.last_alerts.items()
                               if isinstance(v, datetime) and v > cutoff}
            # If still too large after pruning, keep only newest 250
            if len(self.last_alerts) > 500:
                sorted_items = sorted(self.last_alerts.items(), key=lambda x: x[1], reverse=True)
                self.last_alerts = dict(sorted_items[:250])

        # Check if we alerted for this stock recently (within 15 minutes)
        if symbol in self.last_alerts:
            time_since_last = (get_eastern_time() - self.last_alerts[symbol]).total_seconds()
            if time_since_last < 900:  # 15 minutes
                return False

        # Alert thresholds
        strong_threshold = self.config.get('rrs_strong_threshold', 2.0)
        weak_threshold = self.config.get('rrs_weak_threshold', -2.0)

        if abs(rrs) >= abs(strong_threshold):
            return True
        elif abs(rrs) >= abs(weak_threshold):
            # Weak signal - filter if below weak_threshold magnitude
            weak_threshold_value = self.config.get('weak_threshold', None)
            if weak_threshold_value is not None and abs(rrs) < abs(weak_threshold_value):
                return False  # Signal too weak per weak_threshold config
            return True

        return False

    def format_alert_message(self, analysis: Dict) -> str:
        """Format alert message"""
        symbol = analysis['symbol']
        rrs = analysis['rrs']
        price = analysis['price']
        status = analysis['status']

        if rrs > 0:
            direction = "LONG"
            signal = "RELATIVE STRENGTH"
        else:
            direction = "SHORT"
            signal = "RELATIVE WEAKNESS"

        daily_context = ""
        if analysis['daily_strong']:
            daily_context = "\nDaily chart: STRONG (3 green days, EMA bullish)"
        elif analysis['daily_weak']:
            daily_context = "\nDaily chart: WEAK (3 red days, EMA bearish)"

        message = f"""
{signal} ALERT

{symbol} @ ${price:.2f}
RRS: {rrs:.2f} ({status})
Direction: {direction}

Stock Change: {analysis['stock_pc']:.2f}%
SPY Change: {analysis['spy_pc']:.2f}%
ATR: ${analysis['atr']:.2f}
{daily_context}

Time: {get_eastern_time().strftime('%I:%M:%S %p ET')}
        """.strip()

        return message

    def _calculate_daily_sma(self, symbol: str, daily_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calculate 50 SMA and 200 SMA for a stock's daily chart.

        Uses cached data when available (daily SMAs change once per day).

        Args:
            symbol: Stock ticker
            daily_df: Pre-fetched daily DataFrame (optional; fetches via ProviderManager if None)

        Returns:
            dict with keys:
                sma_50: float or None
                sma_200: float or None
                current_price: float
                above_sma50: bool
                above_sma200: bool
                daily_sma_filter: str - "strong"/"moderate"/"weak"/"counter_trend"
        """
        now = time.time()

        # Check cache first
        cached = self._daily_sma_cache.get(symbol)
        if cached and (now - cached['timestamp']) < self._daily_sma_cache_ttl:
            return cached['data']

        try:
            if daily_df is not None and not daily_df.empty:
                close_series = daily_df['close'].dropna() if 'close' in daily_df.columns else None
            else:
                close_series = None

            # Need at least 50 data points for the 50 SMA (the required gate).
            # Batch scans typically fetch 60 days which is enough for 50 SMA.
            # For the 200 SMA (bonus classification), we need 200+ days.
            if close_series is None or len(close_series) < 50:
                # Must fetch — don't have enough for even the 50 SMA
                try:
                    if self._use_providers and self._provider_manager:
                        hist_data = self._provider_manager.get_historical(symbol, period='1y', interval='1d')
                        if hist_data and not hist_data.data.empty:
                            close_series = hist_data.data['close'].dropna()
                except Exception as fetch_err:
                    logger.debug(f"Provider 1y fetch for {symbol} SMA failed: {fetch_err}")

            if close_series is None or len(close_series) < 50:
                logger.debug(f"Insufficient daily data for {symbol} SMA (need 50, got {len(close_series) if close_series is not None else 0})")
                return self._sma_fallback(symbol)

            current_price = float(close_series.iloc[-1])

            # Calculate SMAs
            sma_50 = float(close_series.rolling(window=50).mean().iloc[-1])
            sma_200 = float(close_series.rolling(window=200).mean().iloc[-1]) if len(close_series) >= 200 else None

            above_sma50 = current_price > sma_50
            above_sma200 = current_price > sma_200 if sma_200 is not None else True  # Assume above if not enough data

            # Classify SMA filter level
            if above_sma50 and above_sma200:
                daily_sma_filter = 'strong'
            elif above_sma50 and not above_sma200:
                daily_sma_filter = 'moderate'
            elif not above_sma50 and above_sma200:
                daily_sma_filter = 'weak'
            else:
                daily_sma_filter = 'counter_trend'

            result = {
                'sma_50': round(sma_50, 2),
                'sma_200': round(sma_200, 2) if sma_200 is not None else None,
                'current_price': round(current_price, 2),
                'above_sma50': above_sma50,
                'above_sma200': above_sma200,
                'daily_sma_filter': daily_sma_filter,
            }

            # Cache the result
            self._daily_sma_cache[symbol] = {
                'data': result,
                'timestamp': now,
            }

            # Evict old cache entries to prevent unbounded growth
            if len(self._daily_sma_cache) > 300:
                oldest_key = min(self._daily_sma_cache, key=lambda k: self._daily_sma_cache[k]['timestamp'])
                del self._daily_sma_cache[oldest_key]

            return result

        except Exception as e:
            logger.debug(f"Daily SMA calculation failed for {symbol}: {e}")
            return self._sma_fallback(symbol)

    def _sma_fallback(self, symbol: str) -> Dict:
        """Return neutral SMA data when calculation fails."""
        return {
            'sma_50': None,
            'sma_200': None,
            'current_price': 0.0,
            'above_sma50': True,   # Don't block on data failure
            'above_sma200': True,
            'daily_sma_filter': 'unknown',
        }

    def _apply_daily_sma_gate(self, strong_rs: List[Dict], strong_rw: List[Dict]
                              ) -> Tuple[List[Dict], List[Dict]]:
        """
        Daily SMA Gate: filter signals based on 50/200 SMA position.

        RDT methodology:
        - LONG signals: price must be above 50 SMA (required)
        - SHORT signals: price must be below 50 SMA (required)
        - Adds daily_sma_filter classification to each signal

        Args:
            strong_rs: Long signal candidates (each must have 'symbol' and '_raw_stock_data')
            strong_rw: Short signal candidates

        Returns:
            Tuple of (filtered_strong_rs, filtered_strong_rw)
        """
        if not self._daily_sma_filter_enabled:
            return strong_rs, strong_rw

        filtered_rs = []
        blocked_longs = []

        for stock in strong_rs:
            symbol = stock.get('symbol', '?')
            raw_data = stock.get('_raw_stock_data')
            daily_df = raw_data.get('daily') if raw_data else None

            sma_data = self._calculate_daily_sma(symbol, daily_df)
            stock['daily_sma_filter'] = sma_data['daily_sma_filter']
            stock['sma_50'] = sma_data['sma_50']
            stock['sma_200'] = sma_data['sma_200']
            stock['above_sma50'] = sma_data['above_sma50']
            stock['above_sma200'] = sma_data['above_sma200']

            # Don't block if SMA data is unavailable (unknown = pass through)
            if sma_data['daily_sma_filter'] == 'unknown':
                filtered_rs.append(stock)
            elif not sma_data['above_sma50']:
                blocked_longs.append(symbol)
                logger.warning(
                    f"[SMA GATE] Blocked LONG {symbol} — price ${sma_data['current_price']:.2f} "
                    f"below 50 SMA ${sma_data['sma_50']:.2f} "
                    f"(filter={sma_data['daily_sma_filter']})"
                )
            else:
                filtered_rs.append(stock)

        filtered_rw = []
        blocked_shorts = []

        for stock in strong_rw:
            symbol = stock.get('symbol', '?')
            raw_data = stock.get('_raw_stock_data')
            daily_df = raw_data.get('daily') if raw_data else None

            sma_data = self._calculate_daily_sma(symbol, daily_df)
            stock['daily_sma_filter'] = sma_data['daily_sma_filter']
            stock['sma_50'] = sma_data['sma_50']
            stock['sma_200'] = sma_data['sma_200']
            stock['above_sma50'] = sma_data['above_sma50']
            stock['above_sma200'] = sma_data['above_sma200']

            # Don't block if SMA data is unavailable (unknown = pass through)
            if sma_data['daily_sma_filter'] == 'unknown':
                filtered_rw.append(stock)
            elif sma_data['above_sma50']:
                blocked_shorts.append(symbol)
                logger.warning(
                    f"[SMA GATE] Blocked SHORT {symbol} — price ${sma_data['current_price']:.2f} "
                    f"above 50 SMA ${sma_data['sma_50']:.2f} "
                    f"(filter={sma_data['daily_sma_filter']})"
                )
            else:
                filtered_rw.append(stock)

        if blocked_longs:
            logger.info(
                f"[SMA GATE] Blocked {len(blocked_longs)} LONG signal(s) below 50 SMA: "
                f"{', '.join(blocked_longs)}"
            )
        if blocked_shorts:
            logger.info(
                f"[SMA GATE] Blocked {len(blocked_shorts)} SHORT signal(s) above 50 SMA: "
                f"{', '.join(blocked_shorts)}"
            )

        passed_rs = len(filtered_rs)
        passed_rw = len(filtered_rw)
        if passed_rs + passed_rw > 0:
            logger.info(
                f"[SMA GATE] Passed: {passed_rs} LONG, {passed_rw} SHORT signal(s)"
            )

        return filtered_rs, filtered_rw

    # ------------------------------------------------------------------
    # Lightweight Multi-Timeframe (MTF) Alignment
    # ------------------------------------------------------------------
    # Derives 15m and 1h candles by resampling the 5m data already in
    # memory, so this adds ZERO extra API calls.  Daily data is also
    # reused from the batch fetch.  Total cost: < 1 ms per symbol.
    # ------------------------------------------------------------------

    def _calculate_mtf_alignment(
        self,
        df_5m: pd.DataFrame,
        df_daily: pd.DataFrame,
        signal_direction: str,
    ) -> Dict:
        """
        Calculate multi-timeframe trend alignment from existing data.

        Derives 15-minute and 1-hour bars by resampling the 5-minute
        DataFrame.  For each of the four timeframes (5m, 15m, 1h, daily)
        it checks whether price is above the 8-period EMA and whether
        the EMA slope is positive (bullish) or negative (bearish).

        Args:
            df_5m: 5-minute OHLCV DataFrame (columns lowercase)
            df_daily: Daily OHLCV DataFrame (columns lowercase)
            signal_direction: 'long' or 'short'

        Returns:
            Dict with keys:
                mtf_alignment: 'strong' | 'moderate' | 'weak'
                mtf_score: int (0-4, number of aligned timeframes)
                mtf_details: dict mapping timeframe label to trend info
        """
        ema_period = 8

        def _trend_for_series(close: pd.Series, label: str) -> Dict:
            """Determine trend direction from a close-price series."""
            if close is None or len(close) < ema_period + 1:
                return {'label': label, 'trend': 'neutral', 'aligned': False}
            ema = close.ewm(span=ema_period, adjust=False).mean()
            last_price = float(close.iloc[-1])
            last_ema = float(ema.iloc[-1])
            prev_ema = float(ema.iloc[-2])
            slope_up = last_ema > prev_ema
            above_ema = last_price > last_ema

            if above_ema and slope_up:
                trend = 'bullish'
            elif not above_ema and not slope_up:
                trend = 'bearish'
            else:
                trend = 'neutral'

            if signal_direction == 'long':
                aligned = trend == 'bullish'
            else:
                aligned = trend == 'bearish'

            return {
                'label': label,
                'trend': trend,
                'aligned': aligned,
                'price': round(last_price, 2),
                'ema8': round(last_ema, 2),
            }

        details = {}

        # 1. 5-minute trend (already have this data)
        try:
            details['5m'] = _trend_for_series(df_5m['close'], '5m')
        except Exception:
            details['5m'] = {'label': '5m', 'trend': 'neutral', 'aligned': False}

        # 2. 15-minute — resample from 5m
        try:
            df_15m = df_5m.resample('15min').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum',
            }).dropna(subset=['close'])
            details['15m'] = _trend_for_series(df_15m['close'], '15m')
        except Exception:
            details['15m'] = {'label': '15m', 'trend': 'neutral', 'aligned': False}

        # 3. 1-hour — resample from 5m
        try:
            df_1h = df_5m.resample('1h').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum',
            }).dropna(subset=['close'])
            details['1h'] = _trend_for_series(df_1h['close'], '1h')
        except Exception:
            details['1h'] = {'label': '1h', 'trend': 'neutral', 'aligned': False}

        # 4. Daily trend (already have this data)
        try:
            details['daily'] = _trend_for_series(df_daily['close'], 'daily')
        except Exception:
            details['daily'] = {'label': 'daily', 'trend': 'neutral', 'aligned': False}

        # Score: how many of the 4 timeframes agree with signal direction
        aligned_count = sum(1 for v in details.values() if v.get('aligned'))

        if aligned_count >= 4:
            alignment = 'strong'
        elif aligned_count >= 3:
            alignment = 'moderate'
        else:
            alignment = 'weak'

        return {
            'mtf_alignment': alignment,
            'mtf_score': aligned_count,
            'mtf_details': {k: v['trend'] for k, v in details.items()},
        }

    def _apply_mtf_gate(
        self,
        strong_rs: List[Dict],
        strong_rw: List[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        MTF Gate: block signals with weak multi-timeframe alignment.

        Uses _calculate_mtf_alignment() which resamples existing 5m data
        to derive 15m/1h bars — no extra API calls.

        Adds 'mtf_alignment', 'mtf_score', and 'mtf_details' fields to
        each signal dict.  Signals with 'weak' alignment (< 3 of 4
        timeframes agreeing) are removed.

        Fail-open: if raw data is unavailable, the signal passes through
        with mtf_alignment='unknown'.
        """
        if not self._mtf_lightweight_enabled:
            return strong_rs, strong_rw

        filtered_rs = []
        filtered_rw = []
        blocked_longs = []
        blocked_shorts = []

        for stock in strong_rs:
            raw_data = stock.get('_raw_stock_data')
            if raw_data is None:
                stock['mtf_alignment'] = 'unknown'
                stock['mtf_score'] = 0
                stock['mtf_details'] = {}
                filtered_rs.append(stock)
                continue

            df_5m = raw_data.get('5m')
            df_daily = raw_data.get('daily')
            if df_5m is None or df_daily is None or df_5m.empty or df_daily.empty:
                stock['mtf_alignment'] = 'unknown'
                stock['mtf_score'] = 0
                stock['mtf_details'] = {}
                filtered_rs.append(stock)
                continue

            mtf = self._calculate_mtf_alignment(df_5m, df_daily, 'long')
            stock['mtf_alignment'] = mtf['mtf_alignment']
            stock['mtf_score'] = mtf['mtf_score']
            stock['mtf_details'] = mtf['mtf_details']

            if mtf['mtf_alignment'] == 'weak':
                blocked_longs.append(stock['symbol'])
            else:
                filtered_rs.append(stock)

        for stock in strong_rw:
            raw_data = stock.get('_raw_stock_data')
            if raw_data is None:
                stock['mtf_alignment'] = 'unknown'
                stock['mtf_score'] = 0
                stock['mtf_details'] = {}
                filtered_rw.append(stock)
                continue

            df_5m = raw_data.get('5m')
            df_daily = raw_data.get('daily')
            if df_5m is None or df_daily is None or df_5m.empty or df_daily.empty:
                stock['mtf_alignment'] = 'unknown'
                stock['mtf_score'] = 0
                stock['mtf_details'] = {}
                filtered_rw.append(stock)
                continue

            mtf = self._calculate_mtf_alignment(df_5m, df_daily, 'short')
            stock['mtf_alignment'] = mtf['mtf_alignment']
            stock['mtf_score'] = mtf['mtf_score']
            stock['mtf_details'] = mtf['mtf_details']

            if mtf['mtf_alignment'] == 'weak':
                blocked_shorts.append(stock['symbol'])
            else:
                filtered_rw.append(stock)

        if blocked_longs:
            logger.info(
                f"[MTF GATE] Blocked {len(blocked_longs)} LONG signal(s) with weak alignment: "
                f"{', '.join(blocked_longs)}"
            )
        if blocked_shorts:
            logger.info(
                f"[MTF GATE] Blocked {len(blocked_shorts)} SHORT signal(s) with weak alignment: "
                f"{', '.join(blocked_shorts)}"
            )

        passed_rs = len(filtered_rs)
        passed_rw = len(filtered_rw)
        if passed_rs + passed_rw > 0:
            logger.info(
                f"[MTF GATE] Passed: {passed_rs} LONG, {passed_rw} SHORT signal(s)"
            )

        return filtered_rs, filtered_rw

    def _calculate_vwap_for_symbol(self, symbol: str, raw_data: Optional[Dict]) -> Optional[float]:
        """
        Calculate current VWAP from intraday 5-minute data already fetched by the scanner.

        Uses today's data only (VWAP resets daily). Returns None if data is
        insufficient or unavailable (fail-open).

        Args:
            symbol: Stock ticker symbol
            raw_data: The '_raw_stock_data' dict attached to each signal candidate

        Returns:
            Current VWAP value, or None if it cannot be calculated
        """
        if raw_data is None:
            return None

        df_5m = raw_data.get('5m')
        if df_5m is None or df_5m.empty:
            return None

        try:
            # Ensure required columns exist (lowercase — already normalized)
            required = {'high', 'low', 'close', 'volume'}
            if not required.issubset(set(df_5m.columns)):
                logger.debug(f"VWAP: {symbol} missing columns for VWAP calc")
                return None

            # Filter to today's data only (VWAP resets daily)
            today = pd.Timestamp.now(tz='America/New_York').normalize()
            if df_5m.index.tz is None:
                df_today = df_5m[df_5m.index >= today.tz_localize(None)]
            else:
                df_today = df_5m[df_5m.index >= today]

            if df_today.empty or len(df_today) < 2:
                logger.debug(f"VWAP: {symbol} insufficient intraday data ({len(df_today)} bars)")
                return None

            vwap_series = calculate_vwap(df_today)
            vwap_value = float(vwap_series.iloc[-1])

            if pd.isna(vwap_value) or vwap_value <= 0:
                return None

            return vwap_value

        except Exception as e:
            logger.debug(f"VWAP: Failed to calculate for {symbol}: {e}")
            return None

    def _apply_vwap_gate(self, strong_rs: List[Dict], strong_rw: List[Dict]
                         ) -> Tuple[List[Dict], List[Dict]]:
        """
        VWAP Gate: block signals on wrong side of VWAP (RDT day-trade rule).

        - LONG signals: price must be ABOVE VWAP → pass. Below → block.
        - SHORT signals: price must be BELOW VWAP → pass. Above → block.
        - If VWAP cannot be calculated, fail-open (allow signal through).

        Adds 'vwap' and 'above_vwap' fields to each signal dict.

        Args:
            strong_rs: Long signal candidates
            strong_rw: Short signal candidates

        Returns:
            Tuple of (filtered_strong_rs, filtered_strong_rw)
        """
        if not self._vwap_filter_enabled:
            return strong_rs, strong_rw

        filtered_rs = []
        filtered_rw = []
        blocked_longs = []
        blocked_shorts = []

        for stock in strong_rs:
            symbol = stock.get('symbol', '?')
            raw_data = stock.get('_raw_stock_data')
            vwap = self._calculate_vwap_for_symbol(symbol, raw_data)
            current_price = stock.get('price', 0)

            stock['vwap'] = round(vwap, 2) if vwap is not None else None

            if vwap is None:
                # Fail-open: can't calculate VWAP, allow signal through
                stock['above_vwap'] = None
                filtered_rs.append(stock)
                continue

            above_vwap = current_price > vwap
            stock['above_vwap'] = above_vwap

            if above_vwap:
                filtered_rs.append(stock)
            else:
                blocked_longs.append(symbol)
                logger.warning(
                    f"[VWAP GATE] Blocked LONG {symbol} — price ${current_price:.2f} "
                    f"below VWAP ${vwap:.2f}"
                )

        for stock in strong_rw:
            symbol = stock.get('symbol', '?')
            raw_data = stock.get('_raw_stock_data')
            vwap = self._calculate_vwap_for_symbol(symbol, raw_data)
            current_price = stock.get('price', 0)

            stock['vwap'] = round(vwap, 2) if vwap is not None else None

            if vwap is None:
                # Fail-open: can't calculate VWAP, allow signal through
                stock['above_vwap'] = None
                filtered_rw.append(stock)
                continue

            above_vwap = current_price > vwap
            stock['above_vwap'] = above_vwap

            if not above_vwap:
                filtered_rw.append(stock)
            else:
                blocked_shorts.append(symbol)
                logger.warning(
                    f"[VWAP GATE] Blocked SHORT {symbol} — price ${current_price:.2f} "
                    f"above VWAP ${vwap:.2f}"
                )

        if blocked_longs:
            logger.info(
                f"[VWAP GATE] Blocked {len(blocked_longs)} LONG signal(s) below VWAP: "
                f"{', '.join(blocked_longs)}"
            )
        if blocked_shorts:
            logger.info(
                f"[VWAP GATE] Blocked {len(blocked_shorts)} SHORT signal(s) above VWAP: "
                f"{', '.join(blocked_shorts)}"
            )

        passed_rs = len(filtered_rs)
        passed_rw = len(filtered_rw)
        if passed_rs + passed_rw > 0:
            logger.info(
                f"[VWAP GATE] Passed: {passed_rs} LONG, {passed_rw} SHORT signal(s)"
            )

        return filtered_rs, filtered_rw

    def _apply_spy_gate(self, strong_rs: List[Dict], strong_rw: List[Dict],
                        spy_daily_trend: Optional[Dict] = None
                        ) -> Tuple[List[Dict], List[Dict]]:
        """
        SPY Hard Gate: block signals that go against the SPY daily trend.

        RDT "Market First" principle — the single most important rule:
        - SPY BEARISH (below both 50 & 200 EMA): block ALL long signals
        - SPY BULLISH (above both 50 & 200 EMA): block ALL short signals
        - SPY MIXED (above one EMA, below other): allow both, add warning flag

        This does NOT replace the existing advisory warnings in _apply_sector_filter;
        those are still applied to surviving signals for metadata.

        Args:
            strong_rs: Long signal candidates
            strong_rw: Short signal candidates
            spy_daily_trend: SPY daily trend dict from get_spy_daily_trend()

        Returns:
            Tuple of (filtered_strong_rs, filtered_strong_rw)
        """
        if not self._spy_gate_enabled:
            return strong_rs, strong_rw

        if spy_daily_trend is None:
            logger.warning("SPY GATE: No SPY trend data available — allowing all signals through")
            return strong_rs, strong_rw

        trend = spy_daily_trend.get('daily_trend', 'mixed')

        if trend == 'bearish':
            # Block all LONG signals
            blocked_count = len(strong_rs)
            if blocked_count > 0:
                blocked_symbols = [s.get('symbol', '?') for s in strong_rs]
                logger.warning(
                    f"SPY GATE: Blocked {blocked_count} LONG signal(s) — "
                    f"SPY trend is BEARISH (below 50 & 200 EMA). "
                    f"Blocked: {', '.join(blocked_symbols)}"
                )
            strong_rs = []
            if strong_rw:
                logger.info(
                    f"SPY GATE: Allowing {len(strong_rw)} SHORT signal(s) — "
                    f"aligned with BEARISH SPY trend"
                )

        elif trend == 'bullish':
            # Block all SHORT signals
            blocked_count = len(strong_rw)
            if blocked_count > 0:
                blocked_symbols = [s.get('symbol', '?') for s in strong_rw]
                logger.warning(
                    f"SPY GATE: Blocked {blocked_count} SHORT signal(s) — "
                    f"SPY trend is BULLISH (above 50 & 200 EMA). "
                    f"Blocked: {', '.join(blocked_symbols)}"
                )
            strong_rw = []
            if strong_rs:
                logger.info(
                    f"SPY GATE: Allowing {len(strong_rs)} LONG signal(s) — "
                    f"aligned with BULLISH SPY trend"
                )

        else:
            # Mixed trend — allow both directions, flag them
            logger.info(
                f"SPY GATE: SPY trend is MIXED — allowing {len(strong_rs)} LONG "
                f"and {len(strong_rw)} SHORT signal(s) with caution flag"
            )
            for s in strong_rs:
                s['spy_gate_warning'] = True
                s['spy_gate_reason'] = 'SPY trend is MIXED — use caution'
            for s in strong_rw:
                s['spy_gate_warning'] = True
                s['spy_gate_reason'] = 'SPY trend is MIXED — use caution'

        return strong_rs, strong_rw

    def save_signals(self, strong_rs: List[Dict], strong_rw: List[Dict],
                     spy_daily_trend: Optional[Dict] = None):
        """Save signals to JSON file for API access"""
        signals_dir = Path('data/signals')
        signals_dir.mkdir(parents=True, exist_ok=True)

        # SPY Hard Gate: block signals against the market trend BEFORE processing
        strong_rs, strong_rw = self._apply_spy_gate(strong_rs, strong_rw, spy_daily_trend)

        # Daily SMA Gate: block signals on wrong side of 50 SMA
        strong_rs, strong_rw = self._apply_daily_sma_gate(strong_rs, strong_rw)

        # VWAP Gate: day trades must respect VWAP (RDT methodology)
        strong_rs, strong_rw = self._apply_vwap_gate(strong_rs, strong_rw)

        # MTF Gate: block signals with weak multi-timeframe alignment (zero API calls)
        strong_rs, strong_rw = self._apply_mtf_gate(strong_rs, strong_rw)

        # Determine current session for extended hours flagging
        current_session = self._get_current_session()
        is_extended = current_session in ('premarket', 'afterhours')

        # Format signals for API
        all_signals = []

        for stock in strong_rs:
            signal = {
                'symbol': str(stock['symbol']),
                'direction': 'long',
                'strength': 'strong' if stock['rrs'] > 2.5 else 'moderate',
                'rrs': float(round(stock['rrs'], 2)),
                'entry_price': float(round(stock['price'], 2)),
                'stop_price': float(round(stock['price'] - (stock['atr'] * 1.5), 2)),  # Was 1.0x, then 1.5x ATR — backtest 2026-02-19 confirmed 1.5x optimal
                'target_price': float(round(stock['price'] + (stock['atr'] * 2.0), 2)),  # Was 3.0x ATR — backtest showed 2.0x + scaled exits beats 3.0x (3.57% vs 2.21%)
                'atr': float(round(stock['atr'], 2)),
                'stock_change_pct': float(round(stock['stock_pc'], 2)),
                'spy_change_pct': float(round(stock['spy_pc'], 2)),
                'daily_strong': bool(stock.get('daily_strong', False)),
                # Daily SMA filter fields (50/200 SMA)
                'daily_sma_filter': str(stock.get('daily_sma_filter', 'unknown')),
                'sma_50': stock.get('sma_50'),
                'sma_200': stock.get('sma_200'),
                'above_sma50': bool(stock.get('above_sma50', True)),
                'above_sma200': bool(stock.get('above_sma200', True)),
                # VWAP fields (RDT: longs above VWAP, shorts below)
                'vwap': stock.get('vwap'),
                'above_vwap': stock.get('above_vwap'),
                'generated_at': format_timestamp(),
                'strategy': 'RRS_Momentum',
                # Extended hours fields
                'extended_hours': is_extended,
                'session': current_session,
                # MTF fields (lightweight — derived from existing data, zero API calls)
                'mtf_alignment': str(stock.get('mtf_alignment', 'unknown')),
                'mtf_score': int(stock.get('mtf_score', 0)),
                'mtf_details': stock.get('mtf_details', {}),
                # Legacy MTF fields (kept for backward compatibility)
                'timeframe_alignment': bool(stock.get('timeframe_alignment', False)),
                'trend_by_timeframe': stock.get('trend_by_timeframe', {}),
                'entry_timing_score': int(stock.get('entry_timing_score', 50)),
                'alignment_direction': str(stock.get('alignment_direction', 'unknown')),
                'key_levels': stock.get('key_levels', {}),
                # VIX regime fields
                'vix_position_size_multiplier': float(stock.get('vix_position_size_multiplier', 1.0)),
                'vix_regime': str(stock.get('vix_regime', 'normal')),
                # Regime-adaptive fields
                'market_regime': str(stock.get('market_regime', 'unknown')),
                'regime_stop_multiplier': float(stock.get('regime_stop_multiplier', 1.5)),
                'regime_target_multiplier': float(stock.get('regime_target_multiplier', 2.0)),
                'regime_risk_per_trade': float(stock.get('regime_risk_per_trade', 0.015)),
                # Intermarket fields (Murphy framework)
                'intermarket_composite': float(stock.get('intermarket_composite', 0.0)),
                'intermarket_regime': str(stock.get('intermarket_regime', 'neutral')),
                'bonds_stocks_divergence': float(stock.get('bonds_stocks_divergence', 0.0)),
                'risk_on_off_ratio': float(stock.get('risk_on_off_ratio', 0.0)),
                'intermarket_position_size_mult': float(stock.get('intermarket_position_size_mult', 1.0)),
            }
            # Apply regime-adaptive stop/target multipliers if present
            if stock.get('regime_stop_multiplier'):
                signal['stop_price'] = float(round(
                    stock['price'] - (stock['atr'] * stock['regime_stop_multiplier']), 2
                ))
            if stock.get('regime_target_multiplier'):
                signal['target_price'] = float(round(
                    stock['price'] + (stock['atr'] * stock['regime_target_multiplier']), 2
                ))
            # Add original RRS if boosted
            if stock.get('rrs_boosted'):
                signal['rrs_original'] = float(round(stock.get('rrs_original', stock['rrs']), 2))
                signal['rrs_boosted'] = True
            # Predict signal decay / TTL
            if self._decay_predictor_enabled and self._decay_predictor is not None:
                try:
                    raw_stock_data = stock.get('_raw_stock_data')
                    decay_pred = self._decay_predictor.predict_from_signal(signal, stock_data=raw_stock_data)
                    signal['predicted_ttl'] = decay_pred.recommended_ttl
                    signal['decay_rate'] = decay_pred.decay_rate
                    signal['estimated_valid_minutes'] = decay_pred.estimated_valid_minutes
                except Exception as e:
                    logger.debug(f"Decay prediction failed for {signal['symbol']}: {e}")
            # Validate signal quality and annotate with warning flags
            raw_stock_data = stock.get('_raw_stock_data')
            signal = validate_signal_quality(signal, stock_data=raw_stock_data)
            # Apply news sentiment check (warns but does not block — RDT: price action > news)
            signal = self._apply_news_sentiment(signal)
            # Apply sector strength filter (RDT: "wind at your back")
            signal = self._apply_sector_filter(signal, 'long', spy_daily_trend)
            all_signals.append(signal)

        for stock in strong_rw:
            signal = {
                'symbol': str(stock['symbol']),
                'direction': 'short',
                'strength': 'strong' if stock['rrs'] < -2.5 else 'moderate',
                'rrs': float(round(stock['rrs'], 2)),
                'entry_price': float(round(stock['price'], 2)),
                'stop_price': float(round(stock['price'] + (stock['atr'] * 1.5), 2)),  # Was 1.0x, then 1.5x ATR — backtest 2026-02-19 confirmed 1.5x optimal
                'target_price': float(round(stock['price'] - (stock['atr'] * 2.0), 2)),  # Was 3.0x ATR — backtest showed 2.0x + scaled exits beats 3.0x (3.57% vs 2.21%)
                'atr': float(round(stock['atr'], 2)),
                'stock_change_pct': float(round(stock['stock_pc'], 2)),
                'spy_change_pct': float(round(stock['spy_pc'], 2)),
                'daily_weak': bool(stock.get('daily_weak', False)),
                # Daily SMA filter fields (50/200 SMA)
                'daily_sma_filter': str(stock.get('daily_sma_filter', 'unknown')),
                'sma_50': stock.get('sma_50'),
                'sma_200': stock.get('sma_200'),
                'above_sma50': bool(stock.get('above_sma50', True)),
                'above_sma200': bool(stock.get('above_sma200', True)),
                # VWAP fields (RDT: longs above VWAP, shorts below)
                'vwap': stock.get('vwap'),
                'above_vwap': stock.get('above_vwap'),
                'generated_at': format_timestamp(),
                'strategy': 'RRS_Momentum',
                # Extended hours fields
                'extended_hours': is_extended,
                'session': current_session,
                # MTF fields (lightweight — derived from existing data, zero API calls)
                'mtf_alignment': str(stock.get('mtf_alignment', 'unknown')),
                'mtf_score': int(stock.get('mtf_score', 0)),
                'mtf_details': stock.get('mtf_details', {}),
                # Legacy MTF fields (kept for backward compatibility)
                'timeframe_alignment': bool(stock.get('timeframe_alignment', False)),
                'trend_by_timeframe': stock.get('trend_by_timeframe', {}),
                'entry_timing_score': int(stock.get('entry_timing_score', 50)),
                'alignment_direction': str(stock.get('alignment_direction', 'unknown')),
                'key_levels': stock.get('key_levels', {}),
                # VIX regime fields
                'vix_position_size_multiplier': float(stock.get('vix_position_size_multiplier', 1.0)),
                'vix_regime': str(stock.get('vix_regime', 'normal')),
                # Regime-adaptive fields
                'market_regime': str(stock.get('market_regime', 'unknown')),
                'regime_stop_multiplier': float(stock.get('regime_stop_multiplier', 1.5)),
                'regime_target_multiplier': float(stock.get('regime_target_multiplier', 2.0)),
                'regime_risk_per_trade': float(stock.get('regime_risk_per_trade', 0.015)),
                # Intermarket fields (Murphy framework)
                'intermarket_composite': float(stock.get('intermarket_composite', 0.0)),
                'intermarket_regime': str(stock.get('intermarket_regime', 'neutral')),
                'bonds_stocks_divergence': float(stock.get('bonds_stocks_divergence', 0.0)),
                'risk_on_off_ratio': float(stock.get('risk_on_off_ratio', 0.0)),
                'intermarket_position_size_mult': float(stock.get('intermarket_position_size_mult', 1.0)),
            }
            # Apply regime-adaptive stop/target multipliers if present (short direction)
            if stock.get('regime_stop_multiplier'):
                signal['stop_price'] = float(round(
                    stock['price'] + (stock['atr'] * stock['regime_stop_multiplier']), 2
                ))
            if stock.get('regime_target_multiplier'):
                signal['target_price'] = float(round(
                    stock['price'] - (stock['atr'] * stock['regime_target_multiplier']), 2
                ))
            # Add original RRS if boosted
            if stock.get('rrs_boosted'):
                signal['rrs_original'] = float(round(stock.get('rrs_original', stock['rrs']), 2))
                signal['rrs_boosted'] = True
            # Predict signal decay / TTL
            if self._decay_predictor_enabled and self._decay_predictor is not None:
                try:
                    raw_stock_data = stock.get('_raw_stock_data')
                    decay_pred = self._decay_predictor.predict_from_signal(signal, stock_data=raw_stock_data)
                    signal['predicted_ttl'] = decay_pred.recommended_ttl
                    signal['decay_rate'] = decay_pred.decay_rate
                    signal['estimated_valid_minutes'] = decay_pred.estimated_valid_minutes
                except Exception as e:
                    logger.debug(f"Decay prediction failed for {signal['symbol']}: {e}")
            # Validate signal quality and annotate with warning flags
            raw_stock_data = stock.get('_raw_stock_data')
            signal = validate_signal_quality(signal, stock_data=raw_stock_data)
            # Apply news sentiment check (warns but does not block — RDT: price action > news)
            signal = self._apply_news_sentiment(signal)
            # Apply sector strength filter (RDT: "wind at your back")
            signal = self._apply_sector_filter(signal, 'short', spy_daily_trend)
            all_signals.append(signal)

        # Merge with existing signals: preserve original entry prices
        signal_file = signals_dir / 'active_signals.json'
        signal_max_age_minutes = self.config.get('signal_max_age_minutes', 30)  # Was 60 — tighter signal TTL
        new_symbols = {s['symbol']: s for s in all_signals}

        try:
            if signal_file.exists():
                with open(signal_file, 'r') as f:
                    fcntl.flock(f, fcntl.LOCK_SH)  # Shared lock for reading
                    existing = json.load(f)
                    fcntl.flock(f, fcntl.LOCK_UN)
                if isinstance(existing, list):
                    now = get_eastern_time()
                    for old_signal in existing:
                        symbol = old_signal.get('symbol')
                        try:
                            gen_at = datetime.fromisoformat(old_signal['generated_at'])
                            # Ensure both are timezone-aware or both naive (Fix 3)
                            if gen_at.tzinfo is None:
                                from datetime import timezone
                                gen_at = gen_at.replace(tzinfo=timezone.utc)
                            if now.tzinfo is None:
                                from datetime import timezone
                                now = now.replace(tzinfo=timezone.utc)
                            age_minutes = (now - gen_at).total_seconds() / 60
                        except (KeyError, ValueError, TypeError):
                            continue

                        # Use per-signal TTL if available, otherwise global default
                        effective_ttl = old_signal.get('predicted_ttl', signal_max_age_minutes)
                        if age_minutes > effective_ttl:
                            continue  # Expired, drop it

                        if symbol in new_symbols:
                            # Same symbol re-scanned: keep original entry if direction unchanged
                            new_sig = new_symbols[symbol]
                            if old_signal.get('direction') == new_sig.get('direction'):
                                # Drop if signal has deteriorated below threshold (Fix 2)
                                rrs_threshold = self.config.get('rrs_threshold', 2.0)
                                if abs(new_sig['rrs']) < rrs_threshold * 0.75:  # Was 0.5 — drop signals faster when RRS weakens
                                    # Signal deteriorated significantly, remove it
                                    del new_symbols[symbol]
                                    continue
                                # Preserve original entry/stop/target/generated_at
                                # but update live fields like RRS, timing score, trends
                                old_signal['rrs'] = new_sig['rrs']
                                old_signal['entry_timing_score'] = new_sig.get('entry_timing_score', old_signal.get('entry_timing_score'))
                                old_signal['trend_by_timeframe'] = new_sig.get('trend_by_timeframe', old_signal.get('trend_by_timeframe'))
                                old_signal['timeframe_alignment'] = new_sig.get('timeframe_alignment', old_signal.get('timeframe_alignment'))
                                old_signal['alignment_direction'] = new_sig.get('alignment_direction', old_signal.get('alignment_direction'))
                                old_signal['key_levels'] = new_sig.get('key_levels', old_signal.get('key_levels'))
                                # Lightweight MTF fields
                                old_signal['mtf_alignment'] = new_sig.get('mtf_alignment', old_signal.get('mtf_alignment'))
                                old_signal['mtf_score'] = new_sig.get('mtf_score', old_signal.get('mtf_score'))
                                old_signal['mtf_details'] = new_sig.get('mtf_details', old_signal.get('mtf_details'))
                                # Replace new signal with the preserved original
                                new_symbols[symbol] = old_signal
                            # If direction flipped, keep the new signal (new trade idea)
                        else:
                            # Symbol not in new scan, keep old signal if still fresh
                            all_signals.append(old_signal)
        except Exception as e:
            logger.debug(f"Could not load existing signals for merge: {e}")

        # Rebuild all_signals from the (possibly updated) new_symbols map + carried-over old signals
        carried_old = [s for s in all_signals if s.get('symbol') not in new_symbols]
        all_signals = list(new_symbols.values()) + carried_old

        # Sort by absolute RRS value (strongest signals first)
        all_signals.sort(key=lambda x: abs(x.get('rrs', 0)), reverse=True)

        # Save to file
        if all_signals:
            # Atomic write: write to temp file then rename to avoid race conditions
            temp_fd, temp_path = tempfile.mkstemp(
                dir=os.path.dirname(str(signal_file)), suffix='.tmp'
            )
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump(all_signals, f, indent=2, default=str)
                os.replace(temp_path, str(signal_file))  # Atomic on POSIX
            except Exception:
                try:
                    os.unlink(temp_path)  # Clean up temp file on error
                except OSError:
                    pass
                raise
            logger.info(f"Saved {len(all_signals)} signals to {signal_file}")
        else:
            logger.info("No signals found this scan, preserving previous signals")

        # Also append to history
        if all_signals:
            self.append_to_history(all_signals)

    def append_to_history(self, signals: List[Dict]):
        """Append signals to history file"""
        history_file = Path('data/signals/signal_history.json')

        try:
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []

            # Add new signals (avoiding duplicates from same scan)
            existing_keys = {(s['symbol'], s['generated_at'][:16]) for s in history}
            for signal in signals:
                key = (signal['symbol'], signal['generated_at'][:16])
                if key not in existing_keys:
                    history.append(signal)

            # Keep last 30 days of signals
            cutoff = format_timestamp(get_eastern_time() - timedelta(days=30))
            history = [s for s in history if s['generated_at'] > cutoff]

            # Atomic write for history file
            history_dir = os.path.dirname(str(history_file))
            temp_fd, temp_path = tempfile.mkstemp(dir=history_dir, suffix='.tmp')
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump(history, f, indent=2, default=str)
                os.replace(temp_path, str(history_file))
            except Exception:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise

        except Exception as e:
            logger.error(f"Error saving signal history: {e}")

    def get_provider_status(self) -> Dict:
        """
        Get status of data providers.

        Returns:
            Dict with provider status information
        """
        if not self._use_providers or not self._provider_manager:
            return {"status": "disabled", "message": "Provider manager not in use"}

        return self._provider_manager.status()

    def scan_once(self):
        """
        Run a single scan of the watchlist using optimized batch fetching.

        This method fetches ALL stocks in ONE API call using batch download,
        which is significantly faster than fetching stocks one-by-one.
        Target: Complete scan in under 30 seconds for 33 stocks.

        Extended hours scanning is supported when enabled in configuration.
        """
        import time as time_module
        scan_start = time_module.time()

        # RDT first-hour filter: skip scanning during noisy opening period
        if self._is_first_hour_restricted():
            return

        # Prune MTF cache at start of each scan cycle
        self._prune_mtf_cache()

        # Check if we should scan in current session
        session = self._get_current_session()
        is_extended = session in ('premarket', 'afterhours')

        if is_extended:
            if not self._should_scan_now():
                logger.info(
                    f"Extended hours scanning disabled for {session} session. "
                    f"Skipping scan."
                )
                return
            logger.info(f"[EXTENDED_HOURS] Starting {session} scan...")
        else:
            logger.info("Starting optimized batch scan...")

        # Fetch VIX regime at the start of each scan
        vix_regime = None
        if self._vix_filter_enabled and self._vix_filter is not None:
            try:
                vix_regime = self._vix_filter.get_vix_regime()
                logger.info(
                    f"VIX regime: {vix_regime['level'].upper()} "
                    f"(VIX={vix_regime['vix_value']}, "
                    f"pos_mult={vix_regime['position_size_multiplier']}, "
                    f"rrs_adj={vix_regime['rrs_threshold_adjustment']:+.2f})"
                )
            except Exception as e:
                logger.warning(f"VIX regime fetch failed: {e}")

        # Fetch SPY daily trend at scan start (sector filter)
        spy_daily_trend = None
        if self._sector_filter_enabled and self._sector_filter is not None:
            try:
                spy_daily_trend = self._sector_filter.get_spy_daily_trend()
                logger.info(
                    f"SPY daily trend: {spy_daily_trend['daily_trend'].upper()} "
                    f"(50EMA={'above' if spy_daily_trend['above_50ema'] else 'below'}, "
                    f"200EMA={'above' if spy_daily_trend['above_200ema'] else 'below'}, "
                    f"strength={spy_daily_trend['trend_strength']:+.2f}%)"
                )
            except Exception as e:
                logger.warning(f"SPY daily trend fetch failed: {e}")

        # Detect market regime and get adaptive parameters
        regime_adaptive = None
        detected_regime = None
        if self._regime_params_enabled and self._regime_params and self._regime_detector_scanner:
            try:
                # Build a lightweight signal dict for heuristic detection
                # (full HMM detection happens inside detect_regime automatically)
                probe_signal = {
                    'rrs': 0, 'atr': 0, 'price': 1,
                    'daily_strong': False, 'daily_weak': False,
                }
                detected_regime, regime_confidence = (
                    self._regime_detector_scanner.detect_regime(probe_signal)
                )

                # If we have confidence scores from HMM, blend params
                cached_regime = self._regime_detector_scanner.current_regime
                if cached_regime:
                    regime_adaptive = self._regime_params.get_params(cached_regime)
                else:
                    regime_adaptive = self._regime_params.get_params(detected_regime)

                logger.info(
                    f"Regime-adaptive params: regime={regime_adaptive['regime']}, "
                    f"rrs_threshold={regime_adaptive['rrs_threshold']:.2f}, "
                    f"stop_mult={regime_adaptive['stop_multiplier']:.2f}, "
                    f"target_mult={regime_adaptive['target_multiplier']:.2f}, "
                    f"max_positions={regime_adaptive['max_positions']}, "
                    f"risk_per_trade={regime_adaptive['risk_per_trade']:.3f}, "
                    f"prefer_momentum={regime_adaptive['prefer_momentum']}, "
                    f"prefer_mean_reversion={regime_adaptive['prefer_mean_reversion']}"
                )
            except Exception as e:
                logger.warning(f"Regime-adaptive param detection failed: {e}")
                regime_adaptive = None

        # Fetch intermarket signals at scan start (Murphy framework)
        intermarket_signals = None
        if self._intermarket_enabled and self._intermarket_analyzer is not None:
            try:
                intermarket_signals = self._intermarket_analyzer.get_intermarket_signals()
                logger.info(
                    f"Intermarket regime: {intermarket_signals['intermarket_regime'].upper()} "
                    f"(composite={intermarket_signals['intermarket_composite']:+.3f}, "
                    f"bonds_stocks={intermarket_signals['bonds_stocks_divergence']:+.3f}, "
                    f"risk_ratio={intermarket_signals['risk_on_off_ratio']:+.3f}, "
                    f"dollar={intermarket_signals['dollar_trend']:+.3f}, "
                    f"gold={intermarket_signals['gold_signal']:+.3f})"
                )
                # Log intermarket warnings if any
                should_warn, warn_reason = self._intermarket_analyzer.should_warn()
                if should_warn:
                    logger.warning(f"Intermarket warning: {warn_reason}")
            except Exception as e:
                logger.warning(f"Intermarket signal fetch failed: {e}")

        # Broadcast scan started via WebSocket
        if WEBSOCKET_AVAILABLE:
            try:
                broadcast_scan_started(len(self.watchlist))
            except Exception as e:
                logger.debug(f"WebSocket broadcast error: {e}")

        # Fetch batch data via provider manager
        batch_5m = None
        batch_daily = None

        if self._use_providers and self._provider_manager:
            try:
                batch_5m, batch_daily = self.fetch_batch_data_with_providers()
                if batch_5m is not None and batch_daily is not None:
                    logger.info("Using data from ProviderManager")
            except Exception as e:
                logger.warning(f"ProviderManager batch fetch error: {e}")

        # Try legacy batch fetch (also uses ProviderManager internally)
        if batch_5m is None or batch_daily is None:
            batch_5m, batch_daily = self.fetch_batch_data()

        if batch_5m is None or batch_daily is None:
            error_msg = "Failed to fetch batch data from all providers"
            logger.error(error_msg)
            if WEBSOCKET_AVAILABLE:
                try:
                    broadcast_scan_error(error_msg)
                except Exception:
                    pass

            # Graceful degradation: try individual stock fetches
            logger.info("Attempting graceful degradation with individual stock fetches")
            self._scan_individual_stocks()
            return

        # Process SPY first from batch data
        spy_data = self._extract_spy_data(batch_5m, batch_daily)

        if spy_data is None:
            logger.error("Failed to extract SPY data from batch, skipping scan")
            return

        spy_pc = self.spy_data['spy_pc'] = (
            (self.spy_data['current_price'] / self.spy_data['previous_close']) - 1
        ) * 100

        logger.info(f"SPY: ${self.spy_data['current_price']:.2f} ({spy_pc:+.2f}%)")

        # Results storage
        strong_rs = []
        strong_rw = []
        processed_count = 0
        skipped_count = 0
        skip_reasons = {'extract_failed': 0, 'low_volume': 0, 'low_price': 0, 'dead_rvol': 0, 'rrs_failed': 0}

        # Process each stock from the batch data (no API calls, no delays needed)
        for symbol in self.watchlist:
            try:
                # Extract stock data from batch (no API call needed)
                stock_data = self._extract_stock_data(symbol, batch_5m, batch_daily)
                if stock_data is None:
                    skipped_count += 1
                    skip_reasons['extract_failed'] += 1
                    continue

                # Filter by volume and price
                if stock_data['volume'] < self.config.get('min_volume', 500000):
                    skipped_count += 1
                    skip_reasons['low_volume'] += 1
                    continue
                if stock_data['current_price'] < self.config.get('min_price', 5.0):
                    skipped_count += 1
                    skip_reasons['low_price'] += 1
                    continue

                # Relative volume check (soft gate)
                rvol = 1.0  # default if can't calculate
                try:
                    daily_vol = stock_data['daily']['volume']
                    if len(daily_vol) >= 20:
                        avg_20d_vol = daily_vol.iloc[-21:-1].mean()
                        if avg_20d_vol > 0:
                            rvol = float(stock_data['volume'] / avg_20d_vol)
                except Exception:
                    pass
                stock_data['rvol'] = rvol

                if rvol < 0.5:
                    logger.debug(f"{symbol} blocked: rvol={rvol:.2f} < 0.5 (dead volume)")
                    skipped_count += 1
                    skip_reasons['dead_rvol'] += 1
                    continue
                if rvol < 0.8:
                    logger.debug(f"{symbol} low rvol warning: rvol={rvol:.2f} < 0.8")

                # Calculate RRS with MTF analysis
                if self._mtf_enabled:
                    analysis = self.calculate_stock_rrs_with_mtf(symbol, stock_data)
                else:
                    analysis = self.calculate_stock_rrs(symbol, stock_data)

                if analysis is None:
                    skipped_count += 1
                    skip_reasons['rrs_failed'] += 1
                    continue

                processed_count += 1

                # Check for strong signals
                rrs = analysis['rrs']
                # Use regime-adaptive threshold if available, else static config
                if regime_adaptive is not None:
                    threshold = regime_adaptive['rrs_threshold']
                else:
                    threshold = self.config.get('rrs_strong_threshold', 2.0)

                # Apply VIX-based RRS threshold adjustment
                if vix_regime is not None:
                    threshold += vix_regime.get('rrs_threshold_adjustment', 0.0)

                # Apply intermarket-based RRS threshold adjustment (Murphy framework)
                if intermarket_signals is not None:
                    regime_im = intermarket_signals.get('intermarket_regime', 'neutral')
                    if regime_im == 'risk_on':
                        threshold += -0.25
                    elif regime_im == 'risk_off':
                        threshold += 0.50

                # Check if MTF alignment is required
                should_include = True
                if self._mtf_alignment_required and self._mtf_enabled:
                    # Only include if timeframes are aligned
                    if not analysis.get('timeframe_alignment', False):
                        should_include = False

                if should_include:
                    if rrs > threshold:
                        # Attach raw stock_data so save_signals can run quality validation
                        analysis['_raw_stock_data'] = stock_data
                        # Attach relative volume for dashboard display
                        analysis['rvol'] = stock_data.get('rvol', 1.0)
                        if stock_data.get('rvol', 1.0) < 0.8:
                            analysis['low_rvol_warning'] = True
                        # Beta-awareness flag for high-beta names
                        stock_beta = self._estimate_beta(stock_data, batch_daily)
                        if stock_beta is not None:
                            analysis['beta'] = round(stock_beta, 2)
                            # Flag if high beta and RRS only marginally above threshold
                            if stock_beta > 1.5 and rrs < threshold * 1.5:
                                analysis['high_beta_caution'] = True
                        # Attach VIX position size multiplier for downstream use
                        if vix_regime is not None:
                            analysis['vix_position_size_multiplier'] = vix_regime['position_size_multiplier']
                            analysis['vix_regime'] = vix_regime['level']
                        # Attach regime-adaptive metadata
                        if regime_adaptive is not None:
                            analysis['market_regime'] = regime_adaptive['regime']
                            analysis['regime_stop_multiplier'] = regime_adaptive['stop_multiplier']
                            analysis['regime_target_multiplier'] = regime_adaptive['target_multiplier']
                            analysis['regime_risk_per_trade'] = regime_adaptive['risk_per_trade']
                        # Attach intermarket metadata (Murphy framework)
                        if intermarket_signals is not None:
                            analysis['intermarket_composite'] = intermarket_signals['intermarket_composite']
                            analysis['intermarket_regime'] = intermarket_signals['intermarket_regime']
                            analysis['bonds_stocks_divergence'] = intermarket_signals['bonds_stocks_divergence']
                            analysis['risk_on_off_ratio'] = intermarket_signals['risk_on_off_ratio']
                            # Apply intermarket position size multiplier
                            im_regime = intermarket_signals['intermarket_regime']
                            if im_regime == 'risk_on':
                                analysis['intermarket_position_size_mult'] = 1.10
                            elif im_regime == 'risk_off':
                                analysis['intermarket_position_size_mult'] = 0.75
                            else:
                                analysis['intermarket_position_size_mult'] = 1.00
                        strong_rs.append(analysis)
                        if self.should_alert(symbol, rrs):
                            message = self.format_alert_message(analysis)
                            send_alert(message, self.config)
                            self.last_alerts[symbol] = get_eastern_time()

                    elif rrs < -threshold:
                        # Attach raw stock_data so save_signals can run quality validation
                        analysis['_raw_stock_data'] = stock_data
                        # Attach relative volume for dashboard display
                        analysis['rvol'] = stock_data.get('rvol', 1.0)
                        if stock_data.get('rvol', 1.0) < 0.8:
                            analysis['low_rvol_warning'] = True
                        # Beta-awareness flag for high-beta names
                        stock_beta = self._estimate_beta(stock_data, batch_daily)
                        if stock_beta is not None:
                            analysis['beta'] = round(stock_beta, 2)
                            if stock_beta > 1.5 and abs(rrs) < threshold * 1.5:
                                analysis['high_beta_caution'] = True
                        # Attach VIX position size multiplier for downstream use
                        if vix_regime is not None:
                            analysis['vix_position_size_multiplier'] = vix_regime['position_size_multiplier']
                            analysis['vix_regime'] = vix_regime['level']
                        # Attach regime-adaptive metadata
                        if regime_adaptive is not None:
                            analysis['market_regime'] = regime_adaptive['regime']
                            analysis['regime_stop_multiplier'] = regime_adaptive['stop_multiplier']
                            analysis['regime_target_multiplier'] = regime_adaptive['target_multiplier']
                            analysis['regime_risk_per_trade'] = regime_adaptive['risk_per_trade']
                        # Attach intermarket metadata (Murphy framework)
                        if intermarket_signals is not None:
                            analysis['intermarket_composite'] = intermarket_signals['intermarket_composite']
                            analysis['intermarket_regime'] = intermarket_signals['intermarket_regime']
                            analysis['bonds_stocks_divergence'] = intermarket_signals['bonds_stocks_divergence']
                            analysis['risk_on_off_ratio'] = intermarket_signals['risk_on_off_ratio']
                            # Apply intermarket position size multiplier
                            im_regime = intermarket_signals['intermarket_regime']
                            if im_regime == 'risk_on':
                                analysis['intermarket_position_size_mult'] = 1.10
                            elif im_regime == 'risk_off':
                                analysis['intermarket_position_size_mult'] = 0.75
                            else:
                                analysis['intermarket_position_size_mult'] = 1.00
                        strong_rw.append(analysis)
                        if self.should_alert(symbol, rrs):
                            message = self.format_alert_message(analysis)
                            send_alert(message, self.config)
                            self.last_alerts[symbol] = get_eastern_time()

                # NO DELAYS NEEDED - all data already fetched in batch

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                skipped_count += 1
                continue

        # Calculate scan duration
        scan_duration = time_module.time() - scan_start

        # Print summary with session info
        session_prefix = f"[EXTENDED_HOURS:{session.upper()}] " if is_extended else ""
        logger.info(f"{session_prefix}Scan complete in {scan_duration:.1f}s: processed {processed_count}, skipped {skipped_count}")
        if skipped_count > 0:
            logger.info(f"Skip breakdown: {skip_reasons}")
        logger.info(f"{session_prefix}Found {len(strong_rs)} RS stocks, {len(strong_rw)} RW stocks")

        if strong_rs:
            logger.info("=== RELATIVE STRENGTH (Long Candidates) ===")
            for stock in sorted(strong_rs, key=lambda x: x['rrs'], reverse=True)[:10]:
                logger.info(f"  {stock['symbol']}: RRS={stock['rrs']:.2f}, Price=${stock['price']:.2f}")

        if strong_rw:
            logger.info("=== RELATIVE WEAKNESS (Short Candidates) ===")
            for stock in sorted(strong_rw, key=lambda x: x['rrs'])[:10]:
                logger.info(f"  {stock['symbol']}: RRS={stock['rrs']:.2f}, Price=${stock['price']:.2f}")

        # Run mean reversion scanner on the same batch data
        # Regime-adaptive: skip mean reversion if regime does not prefer it
        mr_enabled_by_regime = True
        if regime_adaptive is not None and not regime_adaptive.get('prefer_mean_reversion', True):
            mr_enabled_by_regime = False
            logger.info(
                f"Mean reversion scanner skipped (regime={regime_adaptive['regime']} "
                f"does not prefer mean reversion)"
            )

        mr_signals = []
        if mr_enabled_by_regime and MEAN_REVERSION_AVAILABLE and self.config.get('mean_reversion_enabled', False):
            try:
                mr_signals = run_mean_reversion_scan(
                    symbols=self.watchlist,
                    config=self.config,
                    batch_5m=batch_5m,
                    batch_daily=batch_daily,
                )
                if mr_signals:
                    logger.info(f"Mean reversion scanner found {len(mr_signals)} signals")
                    # Separate into long/short for save_signals compatibility
                    for sig in mr_signals:
                        # Add _raw_stock_data placeholder for signal validator
                        sig['_raw_stock_data'] = None
                        # Map fields for save_signals: use 'price' key for logging
                        sig['price'] = sig.get('entry_price', 0)
                        sig['stock_pc'] = sig.get('stock_change_pct', 0)
                        sig['spy_pc'] = sig.get('spy_change_pct', 0)
                        # Attach regime metadata
                        if regime_adaptive is not None:
                            sig['market_regime'] = regime_adaptive['regime']
                            sig['regime_stop_multiplier'] = regime_adaptive['stop_multiplier']
                            sig['regime_target_multiplier'] = regime_adaptive['target_multiplier']
                            sig['regime_risk_per_trade'] = regime_adaptive['risk_per_trade']
                        if sig['direction'] == 'long':
                            strong_rs.append(sig)
                        else:
                            strong_rw.append(sig)
            except Exception as e:
                logger.warning(f"Mean reversion scan failed: {e}")

        # Apply VIX regime filtering to all collected signals
        if vix_regime is not None and self._vix_filter_enabled and self._vix_filter is not None:
            pre_filter_count = len(strong_rs) + len(strong_rw)
            # Build temporary signal dicts for should_allow_signal check
            strong_rs = [
                s for s in strong_rs
                if self._vix_filter.should_allow_signal({
                    'direction': 'long',
                    'strategy': s.get('strategy', 'RRS_Momentum'),
                    'symbol': s.get('symbol', ''),
                })
            ]
            strong_rw = [
                s for s in strong_rw
                if self._vix_filter.should_allow_signal({
                    'direction': 'short',
                    'strategy': s.get('strategy', 'RRS_Momentum'),
                    'symbol': s.get('symbol', ''),
                })
            ]
            post_filter_count = len(strong_rs) + len(strong_rw)
            filtered_out = pre_filter_count - post_filter_count
            if filtered_out > 0:
                logger.info(
                    f"VIX filter removed {filtered_out} signals "
                    f"(regime={vix_regime['level']})"
                )

        # Save signals for API access
        self.save_signals(strong_rs, strong_rw, spy_daily_trend=spy_daily_trend)

        # Record signal quality metrics
        try:
            all_signals = strong_rs + strong_rw
            get_metrics_tracker().record_scan(all_signals)
        except Exception as e:
            logger.debug(f"Error recording signal metrics: {e}")

        # Record Prometheus metrics
        if METRICS_AVAILABLE:
            try:
                # Record scan duration
                record_scanner_duration(scan_duration, processed_count)

                # Record signals generated
                for stock in strong_rs:
                    strength = 'strong' if stock['rrs'] > 2.5 else 'moderate'
                    record_signal('long', strength)

                for stock in strong_rw:
                    strength = 'strong' if stock['rrs'] < -2.5 else 'moderate'
                    record_signal('short', strength)

                # Update market status
                set_market_status(is_market_open())
            except Exception as e:
                logger.debug(f"Error recording metrics: {e}")

        # Broadcast scan completion via WebSocket
        if WEBSOCKET_AVAILABLE:
            try:
                broadcast_scan_completed(
                    strong_rs=strong_rs,
                    strong_rw=strong_rw,
                    total_scanned=processed_count,
                    duration_seconds=scan_duration
                )
            except Exception as e:
                logger.debug(f"WebSocket broadcast error: {e}")

    def _scan_individual_stocks(self):
        """
        Graceful degradation: scan stocks individually when batch fetch fails.
        This is slower but more resilient.
        """
        import time as time_module
        scan_start = time_module.time()

        # RDT first-hour filter: skip scanning during noisy opening period
        if self._is_first_hour_restricted():
            return

        logger.info("Running individual stock scan (graceful degradation mode)...")

        # Fetch SPY data first
        spy_data = self.fetch_spy_data()
        if spy_data is None:
            logger.error("Failed to fetch SPY data, cannot calculate RRS")
            return

        spy_pc = self.spy_data['spy_pc'] = (
            (self.spy_data['current_price'] / self.spy_data['previous_close']) - 1
        ) * 100

        logger.info(f"SPY: ${self.spy_data['current_price']:.2f} ({spy_pc:+.2f}%)")

        # Results storage
        strong_rs = []
        strong_rw = []
        processed_count = 0
        skipped_count = 0

        # Process each stock individually
        for symbol in self.watchlist:
            try:
                stock_data = self.fetch_stock_data(symbol)
                if stock_data is None:
                    skipped_count += 1
                    continue

                # Filter by volume and price
                if stock_data['volume'] < self.config.get('min_volume', 500000):
                    skipped_count += 1
                    continue
                if stock_data['current_price'] < self.config.get('min_price', 5.0):
                    skipped_count += 1
                    continue

                # Relative volume check (soft gate)
                rvol = 1.0
                try:
                    daily_vol = stock_data['daily']['volume']
                    if len(daily_vol) >= 20:
                        avg_20d_vol = daily_vol.iloc[-21:-1].mean()
                        if avg_20d_vol > 0:
                            rvol = float(stock_data['volume'] / avg_20d_vol)
                except Exception:
                    pass
                stock_data['rvol'] = rvol

                if rvol < 0.5:
                    logger.debug(f"{symbol} blocked: rvol={rvol:.2f} < 0.5 (dead volume)")
                    skipped_count += 1
                    continue
                if rvol < 0.8:
                    logger.debug(f"{symbol} low rvol warning: rvol={rvol:.2f} < 0.8")

                # Calculate RRS with MTF analysis
                if self._mtf_enabled:
                    analysis = self.calculate_stock_rrs_with_mtf(symbol, stock_data)
                else:
                    analysis = self.calculate_stock_rrs(symbol, stock_data)

                if analysis is None:
                    skipped_count += 1
                    continue

                processed_count += 1

                # Check for strong signals
                rrs = analysis['rrs']
                threshold = self.config.get('rrs_strong_threshold', 2.0)

                # Check if MTF alignment is required
                should_include = True
                if self._mtf_alignment_required and self._mtf_enabled:
                    if not analysis.get('timeframe_alignment', False):
                        should_include = False

                if should_include and rrs > threshold:
                    # Attach raw stock_data so save_signals can run quality validation
                    analysis['_raw_stock_data'] = stock_data
                    analysis['rvol'] = stock_data.get('rvol', 1.0)
                    if stock_data.get('rvol', 1.0) < 0.8:
                        analysis['low_rvol_warning'] = True
                    strong_rs.append(analysis)
                    if self.should_alert(symbol, rrs):
                        message = self.format_alert_message(analysis)
                        send_alert(message, self.config)
                        self.last_alerts[symbol] = get_eastern_time()

                elif should_include and rrs < -threshold:
                    # Attach raw stock_data so save_signals can run quality validation
                    analysis['_raw_stock_data'] = stock_data
                    analysis['rvol'] = stock_data.get('rvol', 1.0)
                    if stock_data.get('rvol', 1.0) < 0.8:
                        analysis['low_rvol_warning'] = True
                    strong_rw.append(analysis)
                    if self.should_alert(symbol, rrs):
                        message = self.format_alert_message(analysis)
                        send_alert(message, self.config)
                        self.last_alerts[symbol] = get_eastern_time()

                # Small delay to avoid rate limiting in individual mode
                time_module.sleep(0.1)

            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                skipped_count += 1
                consecutive_errors = getattr(self, '_consecutive_scan_errors', 0) + 1
                self._consecutive_scan_errors = consecutive_errors
                continue

        scan_duration = time_module.time() - scan_start

        # Escalate if many symbols failed — indicates a systematic issue
        error_rate = skipped_count / max(processed_count + skipped_count, 1)
        if error_rate > 0.5 and skipped_count > 10:
            logger.warning(
                f"High scan error rate: {skipped_count}/{processed_count + skipped_count} "
                f"symbols failed ({error_rate:.0%}) — possible provider or network issue"
            )
        self._consecutive_scan_errors = 0

        logger.info(f"Individual scan complete in {scan_duration:.1f}s: processed {processed_count}, skipped {skipped_count}")
        logger.info(f"Found {len(strong_rs)} RS stocks, {len(strong_rw)} RW stocks")

        # Run mean reversion scanner (no batch data in individual mode)
        if MEAN_REVERSION_AVAILABLE and self.config.get('mean_reversion_enabled', False):
            try:
                mr_signals = run_mean_reversion_scan(
                    symbols=self.watchlist,
                    config=self.config,
                    batch_5m=None,
                    batch_daily=None,
                )
                if mr_signals:
                    logger.info(f"Mean reversion scanner found {len(mr_signals)} signals (individual mode)")
                    for sig in mr_signals:
                        sig['_raw_stock_data'] = None
                        sig['price'] = sig.get('entry_price', 0)
                        sig['stock_pc'] = sig.get('stock_change_pct', 0)
                        sig['spy_pc'] = sig.get('spy_change_pct', 0)
                        if sig['direction'] == 'long':
                            strong_rs.append(sig)
                        else:
                            strong_rw.append(sig)
            except Exception as e:
                logger.warning(f"Mean reversion scan failed (individual mode): {e}")

        # Save signals
        self.save_signals(strong_rs, strong_rw)

        # Record signal quality metrics
        try:
            all_signals = strong_rs + strong_rw
            get_metrics_tracker().record_scan(all_signals)
        except Exception as e:
            logger.debug(f"Error recording signal metrics: {e}")

    def run_continuous(self):
        """Run scanner continuously"""
        scan_interval = self.config.get('scan_interval_seconds', 60)

        logger.info(f"Starting continuous scanner (interval: {scan_interval}s)")
        logger.info("Press Ctrl+C to stop")

        # Log provider status at startup
        if self._use_providers and self._provider_manager:
            status = self.get_provider_status()
            logger.info(f"Data providers: {status.get('provider_order', ['yfinance'])}")

        consecutive_failures = 0
        max_backoff = 300  # 5 minutes max backoff

        try:
            while True:
                try:
                    self.scan_once()
                    consecutive_failures = 0
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    consecutive_failures += 1
                    backoff = min(scan_interval * (2 ** (consecutive_failures - 1)), max_backoff)
                    logger.error(
                        f"Unhandled exception in scan_once() (failure #{consecutive_failures}): {e}",
                        exc_info=True
                    )
                    if consecutive_failures >= 5:
                        logger.critical(
                            f"Scanner has failed {consecutive_failures} consecutive times. "
                            f"Backing off {backoff}s. Manual intervention may be needed."
                        )
                    time.sleep(backoff)
                    continue

                logger.info(f"Waiting {scan_interval} seconds until next scan...")
                time.sleep(scan_interval)

        except KeyboardInterrupt:
            logger.info("Scanner stopped by user")


if __name__ == "__main__":
    # Load configuration (in practice, load from .env file)
    config = {
        'atr_period': 14,
        'rrs_strong_threshold': 2.0,
        'rrs_weak_threshold': -2.0,
        'scan_interval_seconds': 300,  # 5 minutes
        'min_volume': 500000,
        'min_price': 5.0,
        'max_price': 500.0,
        'alert_method': 'desktop',
        'use_data_providers': True,  # Enable redundant providers

        # Extended hours scanning settings
        'premarket_scan_enabled': False,     # Enable pre-market scanning (4:00-9:30 AM ET)
        'afterhours_scan_enabled': False,    # Enable after-hours scanning (4:00-8:00 PM ET)
        'extended_hours_data_source': 'auto',  # Data source for extended hours quotes

        # Multi-timeframe analysis settings
        'mtf_enabled': False,  # OLD heavy MTF — disabled (adds ~6 min to scan, per-symbol API calls)
        'mtf_timeframes': '5m,15m,1h,4h,1d',  # Timeframes to analyze (old MTF)
        'mtf_alignment_required': False,  # Require alignment for signals (old MTF)
        'mtf_alignment_boost': 0.5,  # RRS boost when aligned (old MTF)
        'mtf_cache_ttl': 60,  # Cache TTL in seconds (old MTF)

        # Lightweight MTF gate (NEW — resamples existing 5m data, zero API calls)
        'mtf_lightweight_enabled': True,  # Enabled by default — blocks weak (<3/4) alignment

        # Intermarket analysis settings (Murphy framework)
        'intermarket_enabled': True,             # Enable intermarket macro analysis
        'intermarket_cache_ttl_minutes': 30,     # Cache intermarket data for 30 minutes

        # VIX regime filter settings
        'vix_filter_enabled': True,   # Enable VIX-based regime filtering
        'vix_cache_ttl': 300,         # Cache VIX data for 5 minutes

        # RDT first-hour filter (avoid noisy first 30 min after open)
        'first_hour_filter_enabled': True,       # Enable first-hour scanning restriction
        'first_hour_cutoff_minutes': 30,         # Minutes after 9:30 AM ET to wait

        # News sentiment pre-filter
        'news_filter_enabled': True,             # Enable news sentiment warnings on signals
        'news_cache_ttl_minutes': 15,            # Cache news results for 15 minutes
    }

    scanner = RealTimeScanner(config)
    scanner.run_continuous()
