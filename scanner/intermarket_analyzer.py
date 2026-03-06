"""
Intermarket Analysis Module (Murphy Framework)

Implements John Murphy's intermarket analysis framework to provide a LEADING
indicator layer that warns of market regime shifts BEFORE the SPY gate flips.

Key relationships (Murphy's framework):
  - Bonds lead stocks at turning points (TLT vs SPY divergence)
  - Rising dollar is a headwind for equities (UUP trend)
  - Rising gold + falling stocks = flight to safety (GLD signal)
  - Small caps leading = risk-on environment (IWM/SPY ratio)

The intermarket composite adjusts RRS thresholds and position sizing:
  - risk_on:  lower RRS threshold (-0.25), larger positions (1.10x)
  - neutral:  no adjustment
  - risk_off: raise RRS threshold (+0.50), smaller positions (0.75x)

This module is ADVISORY -- it adjusts thresholds and sizing but does NOT
block signals on its own.  All data is fetched from the PostgreSQL daily bar
cache with 30-minute in-memory caching per symbol.
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_TTL = 1800  # 30 minutes in seconds
_DEFAULT_LOOKBACK = 20     # 20 trading days

# Intermarket ETF symbols
_SYMBOLS = {
    'bonds': 'TLT',    # 20+ Year Treasury Bond ETF
    'dollar': 'UUP',   # US Dollar Bull ETF
    'gold': 'GLD',     # Gold ETF
    'smallcap': 'IWM', # Russell 2000 Small Cap ETF
    'benchmark': 'SPY', # S&P 500
}

# Composite signal weights (must sum to 1.0)
_WEIGHTS = {
    'bonds_stocks': 0.35,
    'risk_on_off': 0.30,
    'dollar': 0.20,
    'gold': 0.15,
}

# Regime thresholds
_RISK_ON_THRESHOLD = 0.3
_RISK_OFF_THRESHOLD = -0.3


class IntermarketAnalyzer:
    """
    Analyzes intermarket relationships (bonds, dollar, gold, small caps vs SPY)
    to detect macro regime shifts and provide RRS threshold / position sizing
    adjustments.

    Thread safety: simple cache replacement is atomic enough for the scanner's
    single-threaded scan loop.  Worst case is a redundant fetch.
    """

    def __init__(self, cache_ttl_minutes: int = 30):
        self._cache_ttl = cache_ttl_minutes * 60  # Convert to seconds
        self._lookback = _DEFAULT_LOOKBACK

        # Per-symbol price cache: {symbol: {'data': pd.Series, 'fetched_at': float}}
        self._price_cache: Dict[str, Dict] = {}

        # Historical bar cache (lazy-loaded singleton)
        self._historical_cache = None

        logger.info(
            f"IntermarketAnalyzer initialized "
            f"(cache_ttl={cache_ttl_minutes}min, lookback={self._lookback}d)"
        )

    @property
    def historical_cache(self):
        """Lazy-load the HistoricalBarCache singleton."""
        if self._historical_cache is None:
            from data.database.historical_cache import get_historical_cache
            self._historical_cache = get_historical_cache()
        return self._historical_cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_intermarket_signals(self) -> Dict:
        """
        Compute all intermarket signals.

        Returns a dict with:
            bonds_stocks_divergence : float (-1 to +1)
            dollar_trend            : float (-1 to +1)
            gold_signal             : float (-1 to +1)
            risk_on_off_ratio       : float (-1 to +1)
            intermarket_composite   : float (-1 to +1)
            intermarket_regime      : str ("risk_on" | "neutral" | "risk_off")
        """
        # Fetch all price series (cached)
        prices = {}
        for key, symbol in _SYMBOLS.items():
            prices[key] = self._get_price_series(symbol)

        # Compute individual signals
        bonds_stocks = self._calc_bonds_stocks_divergence(
            prices['bonds'], prices['benchmark']
        )
        dollar = self._calc_dollar_trend(prices['dollar'])
        gold = self._calc_gold_signal(prices['gold'], prices['benchmark'])
        risk_ratio = self._calc_risk_on_off_ratio(
            prices['smallcap'], prices['benchmark']
        )

        # Weighted composite
        composite = (
            _WEIGHTS['bonds_stocks'] * bonds_stocks
            + _WEIGHTS['risk_on_off'] * risk_ratio
            + _WEIGHTS['dollar'] * dollar
            + _WEIGHTS['gold'] * gold
        )
        composite = float(np.clip(composite, -1.0, 1.0))

        # Determine regime
        if composite > _RISK_ON_THRESHOLD:
            regime = "risk_on"
        elif composite < _RISK_OFF_THRESHOLD:
            regime = "risk_off"
        else:
            regime = "neutral"

        signals = {
            'bonds_stocks_divergence': round(bonds_stocks, 4),
            'dollar_trend': round(dollar, 4),
            'gold_signal': round(gold, 4),
            'risk_on_off_ratio': round(risk_ratio, 4),
            'intermarket_composite': round(composite, 4),
            'intermarket_regime': regime,
        }

        logger.info(
            f"Intermarket signals: regime={regime.upper()}, "
            f"composite={composite:+.3f} "
            f"(bonds_stocks={bonds_stocks:+.3f}, risk_ratio={risk_ratio:+.3f}, "
            f"dollar={dollar:+.3f}, gold={gold:+.3f})"
        )

        return signals

    def get_rrs_adjustment(self) -> float:
        """
        Return RRS threshold adjustment based on intermarket composite.

        Returns:
            float: Additive adjustment to RRS threshold.
                risk_on:  -0.25 (lower threshold = more signals)
                neutral:   0.00
                risk_off: +0.50 (raise threshold = fewer, higher quality signals)
        """
        signals = self.get_intermarket_signals()
        regime = signals['intermarket_regime']

        if regime == "risk_on":
            adj = -0.25
        elif regime == "risk_off":
            adj = 0.50
        else:
            adj = 0.0

        logger.debug(f"Intermarket RRS adjustment: {adj:+.2f} (regime={regime})")
        return adj

    def get_position_size_multiplier(self) -> float:
        """
        Return position size multiplier based on intermarket composite.

        Returns:
            float: Multiplicative adjustment to position size.
                risk_on:  1.10
                neutral:  1.00
                risk_off: 0.75
        """
        signals = self.get_intermarket_signals()
        regime = signals['intermarket_regime']

        if regime == "risk_on":
            mult = 1.10
        elif regime == "risk_off":
            mult = 0.75
        else:
            mult = 1.00

        logger.debug(
            f"Intermarket position size multiplier: {mult:.2f} (regime={regime})"
        )
        return mult

    def should_warn(self) -> Tuple[bool, str]:
        """
        Check if any intermarket divergence warrants a warning.

        Returns:
            Tuple of (should_warn: bool, reason: str).
            If no warning needed, returns (False, "").
        """
        signals = self.get_intermarket_signals()
        warnings = []

        # Bond-stock divergence warning
        bsd = signals['bonds_stocks_divergence']
        if bsd <= -0.5:
            warnings.append(
                f"Bonds declining while stocks rising "
                f"(divergence={bsd:+.2f}) -- bonds often lead at turns"
            )
        elif bsd >= 0.5:
            warnings.append(
                f"Bonds rising while stocks falling "
                f"(divergence={bsd:+.2f}) -- potential recovery signal"
            )

        # Dollar headwind warning
        dt = signals['dollar_trend']
        if dt >= 0.5:
            warnings.append(
                f"Dollar rising sharply (trend={dt:+.2f}) -- headwind for equities"
            )

        # Flight to safety warning
        gs = signals['gold_signal']
        if gs <= -0.5:
            warnings.append(
                f"Gold rising while stocks falling "
                f"(signal={gs:+.2f}) -- flight to safety"
            )

        # Risk-off warning
        ror = signals['risk_on_off_ratio']
        if ror <= -0.5:
            warnings.append(
                f"Small caps underperforming large caps "
                f"(ratio={ror:+.2f}) -- risk-off environment"
            )

        # Overall regime warning
        regime = signals['intermarket_regime']
        if regime == "risk_off":
            warnings.append(
                f"Intermarket composite is risk-off "
                f"({signals['intermarket_composite']:+.3f})"
            )

        if warnings:
            reason = "; ".join(warnings)
            logger.warning(f"Intermarket warning: {reason}")
            return (True, reason)

        return (False, "")

    # ------------------------------------------------------------------
    # Signal calculations
    # ------------------------------------------------------------------

    def _calc_bonds_stocks_divergence(
        self,
        bonds: Optional[pd.Series],
        spy: Optional[pd.Series],
    ) -> float:
        """
        TLT 20-day return vs SPY 20-day return.

        - TLT declining while SPY rising = negative divergence (warning: -1)
        - Both rising or both falling = no divergence (0)
        - TLT rising while SPY falling = positive divergence (recovery: +1)
        """
        bond_ret = self._calc_return(bonds)
        spy_ret = self._calc_return(spy)

        if bond_ret is None or spy_ret is None:
            return 0.0

        # Both positive or both negative = no divergence
        if (bond_ret > 0 and spy_ret > 0) or (bond_ret < 0 and spy_ret < 0):
            return 0.0

        # TLT rising, SPY falling = positive divergence (recovery signal)
        if bond_ret > 0 and spy_ret <= 0:
            # Scale by magnitude of divergence
            raw = min(abs(bond_ret) + abs(spy_ret), 10.0) / 10.0
            return float(np.clip(raw, 0.0, 1.0))

        # TLT declining, SPY rising = negative divergence (warning)
        if bond_ret <= 0 and spy_ret > 0:
            raw = min(abs(bond_ret) + abs(spy_ret), 10.0) / 10.0
            return float(np.clip(-raw, -1.0, 0.0))

        return 0.0

    def _calc_dollar_trend(self, dollar: Optional[pd.Series]) -> float:
        """
        UUP 20-day return direction, normalized to -1 (falling fast) to +1 (rising fast).

        Rising dollar = headwind for equities (negative interpretation for stocks).
        We return the raw normalized trend here; the composite weighting handles
        the interpretation.
        """
        ret = self._calc_return(dollar)
        if ret is None:
            return 0.0

        # Normalize: typical UUP 20-day move is roughly -3% to +3%
        # Map that to -1..+1
        normalized = ret / 3.0
        return float(np.clip(normalized, -1.0, 1.0))

    def _calc_gold_signal(
        self,
        gold: Optional[pd.Series],
        spy: Optional[pd.Series],
    ) -> float:
        """
        GLD 20-day return in context of SPY.

        - Rising gold + rising stocks = inflation trade (slight caution): -0.3
        - Rising gold + falling stocks = flight to safety (bearish): -1.0
        - Falling gold + rising stocks = risk-on confidence: +0.5
        - Falling gold + falling stocks = deflationary (neutral): 0.0
        """
        gold_ret = self._calc_return(gold)
        spy_ret = self._calc_return(spy)

        if gold_ret is None or spy_ret is None:
            return 0.0

        if gold_ret > 0 and spy_ret > 0:
            # Inflation trade -- slight caution
            magnitude = min(gold_ret, 5.0) / 5.0
            return float(np.clip(-0.3 * magnitude, -0.3, 0.0))

        if gold_ret > 0 and spy_ret <= 0:
            # Flight to safety -- bearish for equities
            magnitude = min(abs(gold_ret) + abs(spy_ret), 10.0) / 10.0
            return float(np.clip(-magnitude, -1.0, -0.1))

        if gold_ret <= 0 and spy_ret > 0:
            # Risk-on confidence
            magnitude = min(abs(gold_ret) + abs(spy_ret), 10.0) / 10.0
            return float(np.clip(0.5 * magnitude, 0.0, 0.5))

        # Falling gold + falling stocks = deflationary, neutral
        return 0.0

    def _calc_risk_on_off_ratio(
        self,
        smallcap: Optional[pd.Series],
        spy: Optional[pd.Series],
    ) -> float:
        """
        IWM/SPY ratio 20-day change.

        Rising ratio = small caps outperforming = risk-on = bullish.
        Falling ratio = flight to quality = risk-off = bearish.
        """
        if smallcap is None or spy is None:
            return 0.0

        try:
            if len(smallcap) < self._lookback + 1 or len(spy) < self._lookback + 1:
                return 0.0

            # Current ratio
            current_ratio = float(smallcap.iloc[-1] / spy.iloc[-1])
            past_ratio = float(
                smallcap.iloc[-(self._lookback + 1)] / spy.iloc[-(self._lookback + 1)]
            )

            if past_ratio == 0:
                return 0.0

            ratio_change_pct = ((current_ratio / past_ratio) - 1) * 100

            # Normalize: typical 20-day ratio change is roughly -3% to +3%
            normalized = ratio_change_pct / 3.0
            return float(np.clip(normalized, -1.0, 1.0))

        except Exception as e:
            logger.debug(f"Risk on/off ratio calculation failed: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Data fetching and caching
    # ------------------------------------------------------------------

    def _get_price_series(self, symbol: str) -> Optional[pd.Series]:
        """
        Get close price series for a symbol, using cache if fresh.

        Returns:
            pd.Series of close prices, or None on failure.
        """
        now = time.time()

        # Check cache
        cached = self._price_cache.get(symbol)
        if cached is not None and (now - cached['fetched_at']) < self._cache_ttl:
            return cached['data']

        # Fetch fresh data from DB cache
        series = self._fetch_close_prices(symbol)
        if series is not None:
            self._price_cache[symbol] = {
                'data': series,
                'fetched_at': time.time(),
            }
        return series

    def _fetch_close_prices(self, symbol: str) -> Optional[pd.Series]:
        """
        Fetch daily close prices for a symbol from PostgreSQL daily bar cache.

        Requests 90 days of data to ensure enough history for 20-day lookback
        even accounting for weekends/holidays.

        Returns:
            pd.Series of close prices, or None on failure.
        """
        try:
            df = self.historical_cache.get_daily_bars(symbol, lookback_days=90)
            if df is None or df.empty:
                logger.warning(f"No daily bar data in DB cache for {symbol}")
                return None

            close_series = df['close'].dropna()

            if len(close_series) < 2:
                logger.warning(f"Insufficient price data for {symbol}: {len(close_series)} bars")
                return None

            logger.debug(f"Fetched {len(close_series)} daily bars for {symbol} from DB cache")
            return close_series

        except Exception as e:
            logger.error(f"Failed to fetch {symbol} data from DB cache: {e}")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calc_return(self, series: Optional[pd.Series]) -> Optional[float]:
        """
        Calculate percentage return over the lookback period.

        Returns:
            Float percentage return, or None if data insufficient.
        """
        if series is None:
            return None

        if len(series) < self._lookback + 1:
            logger.debug(
                f"Insufficient data for {self._lookback}-day return "
                f"({len(series)} bars)"
            )
            return None

        try:
            current = float(series.iloc[-1])
            past = float(series.iloc[-(self._lookback + 1)])

            if past == 0:
                return None

            ret = ((current / past) - 1) * 100

            if not np.isfinite(ret):
                return None

            return ret

        except Exception as e:
            logger.debug(f"Return calculation failed: {e}")
            return None
