"""
Sector Strength Filter for RDT Trading System

Implements the r/RealDayTrading concept of "wind at your back" -- trading stocks
whose sector is outperforming SPY (for longs) or underperforming SPY (for shorts).

Sector relative strength is measured by comparing each sector ETF's 5-day
percentage change against SPY's 5-day percentage change.

Also provides a SPY daily trend check (50 EMA / 200 EMA) to inform overall
market bias -- RDT methodology favors longs when SPY is above the 50 EMA.

Sector data is cached for 30 minutes since sector rotation is slow-moving.
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from risk.risk_manager import SECTOR_MAP


# ---------------------------------------------------------------------------
# Sector ETF mapping
# ---------------------------------------------------------------------------

# Maps internal sector names (from SECTOR_MAP) to their SPDR sector ETFs
SECTOR_ETF_MAP: Dict[str, str] = {
    'tech': 'XLK',
    'finance': 'XLF',
    'energy': 'XLE',
    'healthcare': 'XLV',
    'industrial': 'XLI',
    'consumer': 'XLY',       # Consumer Discretionary (covers both consumer/consumer goods)
    'materials': 'XLB',
    'real_estate': 'XLRE',
    'communication': 'XLC',
    'utilities': 'XLU',
    'consumer_staples': 'XLP',
}

# All sector ETFs to fetch (including SPY for comparison)
ALL_SECTOR_ETFS = list(set(SECTOR_ETF_MAP.values()))

# Default cache TTL: 30 minutes (sectors rotate slowly)
_SECTOR_CACHE_TTL = 1800  # seconds

# SPY trend cache TTL: 5 minutes
_SPY_TREND_CACHE_TTL = 300  # seconds

# Lookback period for sector RS calculation
_SECTOR_RS_LOOKBACK_DAYS = 5


class SectorStrengthFilter:
    """
    Measures sector relative strength vs SPY and provides signal boost/penalty
    based on whether a stock's sector has wind at its back (or in its face).

    Thread safety: simple cache replacement is atomic enough for the scanner's
    single-threaded scan loop. Worst case is a redundant fetch.
    """

    def __init__(self, cache_ttl: int = _SECTOR_CACHE_TTL,
                 spy_trend_cache_ttl: int = _SPY_TREND_CACHE_TTL):
        self._cache_ttl = cache_ttl
        self._spy_trend_cache_ttl = spy_trend_cache_ttl

        # Sector RS cache: {sector_name: {sector_rs, sector_etf, ...}}
        self._sector_cache: Dict[str, Dict] = {}
        self._sector_cache_at: float = 0.0

        # SPY daily trend cache
        self._spy_trend_cache: Optional[Dict] = None
        self._spy_trend_cache_at: float = 0.0

        logger.info("SectorStrengthFilter initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sector_strength(self, symbol: str) -> Dict:
        """
        Get sector relative strength data for a symbol.

        Args:
            symbol: Stock ticker (e.g. 'AAPL')

        Returns:
            dict with keys:
                sector: str - sector name
                sector_etf: str - sector ETF ticker
                sector_rs: float - sector RS vs SPY (positive = outperforming)
                sector_strong: bool - sector_rs > 0
                sector_weak: bool - sector_rs < -1.0
        """
        sector = SECTOR_MAP.get(symbol, 'other')
        sector_etf = SECTOR_ETF_MAP.get(sector, '')

        # If symbol not in our sector map, return neutral
        if sector == 'other' or not sector_etf:
            return {
                'sector': sector,
                'sector_etf': '',
                'sector_rs': 0.0,
                'sector_strong': False,
                'sector_weak': False,
            }

        # Refresh sector data if cache is stale
        self._refresh_sector_cache()

        # Look up cached result for this sector
        cached = self._sector_cache.get(sector)
        if cached is not None:
            return cached

        # Fallback: sector was not in cache (shouldn't happen after refresh)
        return {
            'sector': sector,
            'sector_etf': sector_etf,
            'sector_rs': 0.0,
            'sector_strong': False,
            'sector_weak': False,
        }

    def should_boost_signal(self, symbol: str, direction: str) -> Tuple[float, str]:
        """
        Determine RRS boost/penalty based on sector alignment with trade direction.

        RDT principle: trade WITH the sector, not against it.
        - Long + strong sector = +0.25 RRS boost (wind at your back)
        - Long + weak sector = -0.5 RRS penalty (fighting the sector)
        - Short + weak sector = +0.25 RRS boost (sector confirms weakness)
        - Short + strong sector = -0.5 RRS penalty (sector is against you)

        Args:
            symbol: Stock ticker
            direction: 'long' or 'short'

        Returns:
            Tuple of (boost: float, reason: str)
        """
        strength = self.get_sector_strength(symbol)
        sector = strength['sector']
        sector_rs = strength['sector_rs']
        sector_strong = strength['sector_strong']
        sector_weak = strength['sector_weak']

        direction_lower = direction.lower()

        if direction_lower == 'long':
            if sector_strong:
                return (0.25, f"Sector {sector} strong (RS={sector_rs:+.2f}): wind at your back")
            elif sector_weak:
                return (-0.5, f"Sector {sector} weak (RS={sector_rs:+.2f}): fighting the sector")
            else:
                return (0.0, f"Sector {sector} neutral (RS={sector_rs:+.2f})")

        elif direction_lower == 'short':
            if sector_weak:
                return (0.25, f"Sector {sector} weak (RS={sector_rs:+.2f}): sector confirms weakness")
            elif sector_strong:
                return (-0.5, f"Sector {sector} strong (RS={sector_rs:+.2f}): sector is against you")
            else:
                return (0.0, f"Sector {sector} neutral (RS={sector_rs:+.2f})")

        # Unknown direction
        return (0.0, f"Unknown direction '{direction}'")

    def get_spy_daily_trend(self) -> Dict:
        """
        Determine SPY's daily trend using 50 EMA and 200 EMA.

        RDT methodology:
        - Above 50 EMA: bullish bias, favor longs
        - Below 50 EMA but above 200 EMA: mixed/cautious
        - Below 200 EMA: bearish bias, favor shorts

        Returns:
            dict with keys:
                above_50ema: bool
                above_200ema: bool
                daily_trend: str - "bullish" / "bearish" / "mixed"
                trend_strength: float - distance from 50 EMA as % of price
        """
        now = time.time()
        if (self._spy_trend_cache is not None
                and (now - self._spy_trend_cache_at) < self._spy_trend_cache_ttl):
            return self._spy_trend_cache

        try:
            result = self._fetch_spy_trend()
            self._spy_trend_cache = result
            self._spy_trend_cache_at = now
            return result
        except Exception as e:
            logger.warning(f"Failed to fetch SPY daily trend: {e}")
            # Return neutral fallback
            fallback = {
                'above_50ema': True,
                'above_200ema': True,
                'daily_trend': 'mixed',
                'trend_strength': 0.0,
            }
            if self._spy_trend_cache is not None:
                return self._spy_trend_cache
            return fallback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_sector_cache(self):
        """Refresh sector RS data if cache is stale."""
        now = time.time()
        if (self._sector_cache
                and (now - self._sector_cache_at) < self._cache_ttl):
            return  # Cache is fresh

        try:
            self._fetch_all_sector_data()
            self._sector_cache_at = now
        except Exception as e:
            logger.warning(f"Failed to refresh sector data: {e}")
            # Keep stale cache if available

    def _fetch_all_sector_data(self):
        """
        Fetch sector ETF + SPY data and compute RS for each sector.

        Uses a single yfinance batch download for efficiency.
        """
        tickers = ALL_SECTOR_ETFS + ['SPY']
        ticker_str = ' '.join(tickers)

        logger.debug(f"Fetching sector ETF data: {ticker_str}")

        # Fetch enough daily data for 5-day percentage change
        # Request extra days to account for weekends/holidays
        data = yf.download(
            ticker_str,
            period='1mo',
            interval='1d',
            progress=False,
            auto_adjust=True,
        )

        if data is None or data.empty:
            logger.warning("Sector ETF batch download returned empty data")
            return

        # Handle yfinance MultiIndex columns: ('Close', 'XLK')
        # For single ticker this won't be MultiIndex, but batch always is
        close_data = None
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                close_data = data['Close']
            elif 'close' in data.columns.get_level_values(0):
                close_data = data['close']
        else:
            # Single ticker edge case (shouldn't happen with batch)
            close_data = data[['Close']] if 'Close' in data.columns else data[['close']]

        if close_data is None or close_data.empty:
            logger.warning("Could not extract Close prices from sector ETF data")
            return

        # Normalize column names to uppercase
        close_data.columns = [str(c).upper() for c in close_data.columns]

        # Calculate SPY 5-day % change
        spy_col = 'SPY'
        if spy_col not in close_data.columns:
            logger.warning("SPY not found in sector ETF batch data")
            return

        spy_series = close_data[spy_col].dropna()
        if len(spy_series) < _SECTOR_RS_LOOKBACK_DAYS + 1:
            logger.warning(f"Insufficient SPY data for {_SECTOR_RS_LOOKBACK_DAYS}-day RS calc")
            return

        spy_current = spy_series.iloc[-1]
        spy_past = spy_series.iloc[-(_SECTOR_RS_LOOKBACK_DAYS + 1)]
        spy_pct_change = ((spy_current / spy_past) - 1) * 100

        # Calculate RS for each sector
        new_cache: Dict[str, Dict] = {}

        for sector_name, etf_ticker in SECTOR_ETF_MAP.items():
            etf_col = etf_ticker.upper()
            if etf_col not in close_data.columns:
                logger.debug(f"Sector ETF {etf_ticker} not found in data")
                continue

            etf_series = close_data[etf_col].dropna()
            if len(etf_series) < _SECTOR_RS_LOOKBACK_DAYS + 1:
                logger.debug(f"Insufficient data for {etf_ticker}")
                continue

            etf_current = etf_series.iloc[-1]
            etf_past = etf_series.iloc[-(_SECTOR_RS_LOOKBACK_DAYS + 1)]
            etf_pct_change = ((etf_current / etf_past) - 1) * 100

            sector_rs = float(etf_pct_change - spy_pct_change)

            # Guard against NaN/inf
            if not np.isfinite(sector_rs):
                sector_rs = 0.0

            new_cache[sector_name] = {
                'sector': sector_name,
                'sector_etf': etf_ticker,
                'sector_rs': round(sector_rs, 4),
                'sector_strong': sector_rs > 0,
                'sector_weak': sector_rs < -1.0,
            }

        if new_cache:
            self._sector_cache = new_cache
            logger.info(
                f"Sector RS updated ({len(new_cache)} sectors): "
                + ", ".join(
                    f"{s['sector_etf']}={s['sector_rs']:+.2f}"
                    for s in sorted(new_cache.values(), key=lambda x: x['sector_rs'], reverse=True)
                )
            )
        else:
            logger.warning("No sector RS data computed")

    def _fetch_spy_trend(self) -> Dict:
        """
        Fetch SPY daily data and compute EMA-based trend.

        Returns dict with trend info.
        """
        logger.debug("Fetching SPY daily trend data")

        spy = yf.download(
            'SPY',
            period='1y',
            interval='1d',
            progress=False,
            auto_adjust=True,
        )

        if spy is None or spy.empty:
            raise ValueError("SPY daily download returned empty data")

        # Handle MultiIndex columns (yfinance quirk even for single symbol)
        if isinstance(spy.columns, pd.MultiIndex):
            if 'Close' in spy.columns.get_level_values(0):
                close_series = spy['Close']
            elif 'close' in spy.columns.get_level_values(0):
                close_series = spy['close']
            else:
                raise ValueError("No Close column in SPY data")
            # For single symbol MultiIndex, flatten
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
        else:
            if 'Close' in spy.columns:
                close_series = spy['Close']
            elif 'close' in spy.columns:
                close_series = spy['close']
            else:
                raise ValueError("No Close column in SPY data")

        close_series = close_series.dropna()

        if len(close_series) < 200:
            logger.warning(f"Only {len(close_series)} days of SPY data, need 200 for full trend")

        current_price = float(close_series.iloc[-1])

        # Calculate EMAs
        ema_50 = float(close_series.ewm(span=50, adjust=False).mean().iloc[-1])
        ema_200 = float(close_series.ewm(span=200, adjust=False).mean().iloc[-1]) if len(close_series) >= 200 else None

        above_50ema = current_price > ema_50
        above_200ema = current_price > ema_200 if ema_200 is not None else True

        # Determine trend
        if above_50ema and above_200ema:
            daily_trend = 'bullish'
        elif not above_50ema and not above_200ema:
            daily_trend = 'bearish'
        else:
            daily_trend = 'mixed'

        # Trend strength: distance from 50 EMA as % of price
        trend_strength = ((current_price - ema_50) / current_price) * 100

        # Guard against NaN
        if not np.isfinite(trend_strength):
            trend_strength = 0.0

        result = {
            'above_50ema': above_50ema,
            'above_200ema': above_200ema,
            'daily_trend': daily_trend,
            'trend_strength': round(trend_strength, 4),
        }

        logger.info(
            f"SPY daily trend: {daily_trend.upper()} "
            f"(price=${current_price:.2f}, 50EMA=${ema_50:.2f}, "
            f"{'200EMA=$' + f'{ema_200:.2f}' if ema_200 else 'no 200EMA'}, "
            f"strength={trend_strength:+.2f}%)"
        )

        return result
