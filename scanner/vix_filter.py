"""
VIX-Based Market Regime Filter

Fetches ^VIX data via yfinance and determines the current market volatility
regime.  Used by the scanner to filter signals and by the risk manager to
adjust position sizes during elevated-volatility environments.

VIX Regimes:
  Low      (< 15)  - Calm market, slightly more aggressive entries allowed
  Normal   (15-20) - Standard parameters
  Elevated (20-25) - Reduce position sizes by 25%, raise RRS threshold +0.5
  High     (25-35) - Reduce position sizes by 50%, raise RRS threshold +1.0,
                      disable mean-reversion signals
  Extreme  (> 35)  - Halt all new entries except momentum shorts
"""

import time
from typing import Dict, Optional

import yfinance as yf
from loguru import logger


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

_REGIMES = [
    # (upper_bound, level, pos_mult, rrs_adj, allow_mr, allow_longs, allow_shorts)
    (15.0,  "low",      1.10, -0.25, True,  True,  True),
    (20.0,  "normal",   1.00,  0.00, True,  True,  True),
    (25.0,  "elevated", 0.75,  0.50, True,  True,  True),
    (35.0,  "high",     0.50,  1.00, False, True,  True),
    (float("inf"), "extreme", 0.0, 2.0, False, False, True),
]

# Cache TTL in seconds
_VIX_CACHE_TTL = 300  # 5 minutes


class VIXFilter:
    """
    Fetches VIX data, determines the current volatility regime, and provides
    helpers to decide whether a given signal should be allowed.

    Instances are safe to share across threads -- the cache is simple (single
    value replaced atomically) and the worst-case race is a redundant fetch.
    """

    def __init__(self, cache_ttl: int = _VIX_CACHE_TTL):
        self._cache_ttl = cache_ttl
        self._cached_vix: Optional[float] = None
        self._cached_at: float = 0.0  # epoch seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_vix_value(self) -> Optional[float]:
        """
        Return the current VIX value, using a cached result if fresh.

        Returns:
            VIX level as a float, or None if the fetch failed.
        """
        now = time.time()
        if self._cached_vix is not None and (now - self._cached_at) < self._cache_ttl:
            return self._cached_vix

        vix = self._fetch_vix()
        if vix is not None:
            self._cached_vix = vix
            self._cached_at = time.time()
        return vix

    def get_vix_regime(self) -> Dict:
        """
        Determine the current VIX regime.

        Returns a dict with:
            level                    : str   ("low" | "normal" | "elevated" | "high" | "extreme")
            vix_value                : float | None
            position_size_multiplier : float (0.0 - 1.10)
            rrs_threshold_adjustment : float
            allow_mean_reversion     : bool
            allow_longs              : bool
            allow_shorts             : bool
        """
        vix = self.get_vix_value()

        if vix is None:
            # Unable to fetch VIX -- fall back to normal (conservative default)
            logger.warning("VIX data unavailable; defaulting to 'normal' regime")
            return self._build_regime("normal", None, 1.0, 0.0, True, True, True)

        for upper, level, pos_mult, rrs_adj, allow_mr, allow_l, allow_s in _REGIMES:
            if vix < upper:
                logger.info(
                    f"VIX regime: {level.upper()} (VIX={vix:.2f}, "
                    f"pos_mult={pos_mult}, rrs_adj={rrs_adj:+.2f})"
                )
                return self._build_regime(level, vix, pos_mult, rrs_adj, allow_mr, allow_l, allow_s)

        # Should never reach here, but just in case
        return self._build_regime("extreme", vix, 0.0, 2.0, False, False, True)

    def should_allow_signal(self, signal_dict: Dict) -> bool:
        """
        Decide whether *signal_dict* is permitted under the current VIX regime.

        Args:
            signal_dict: A signal dictionary as produced by the scanner.
                         Expected keys: 'direction', 'strategy'.

        Returns:
            True if the signal should be kept, False if it should be filtered.
        """
        regime = self.get_vix_regime()
        direction = signal_dict.get("direction", "").lower()
        strategy = signal_dict.get("strategy", "").lower()

        # Extreme regime: only momentum shorts allowed
        if regime["level"] == "extreme":
            if direction == "short" and "momentum" in strategy:
                return True
            logger.debug(
                f"VIX EXTREME: blocking {signal_dict.get('symbol', '?')} "
                f"({direction}/{strategy})"
            )
            return False

        # High regime: block mean-reversion signals
        if regime["level"] == "high":
            if "mean_reversion" in strategy or "mean-reversion" in strategy:
                logger.debug(
                    f"VIX HIGH: blocking mean-reversion signal for "
                    f"{signal_dict.get('symbol', '?')}"
                )
                return False

        # Elevated / normal / low: allow longs & shorts as usual
        if direction == "long" and not regime["allow_longs"]:
            logger.debug(
                f"VIX {regime['level'].upper()}: blocking long for "
                f"{signal_dict.get('symbol', '?')}"
            )
            return False

        if direction == "short" and not regime["allow_shorts"]:
            logger.debug(
                f"VIX {regime['level'].upper()}: blocking short for "
                f"{signal_dict.get('symbol', '?')}"
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_regime(
        level: str,
        vix_value: Optional[float],
        position_size_multiplier: float,
        rrs_threshold_adjustment: float,
        allow_mean_reversion: bool,
        allow_longs: bool,
        allow_shorts: bool,
    ) -> Dict:
        return {
            "level": level,
            "vix_value": vix_value,
            "position_size_multiplier": position_size_multiplier,
            "rrs_threshold_adjustment": rrs_threshold_adjustment,
            "allow_mean_reversion": allow_mean_reversion,
            "allow_longs": allow_longs,
            "allow_shorts": allow_shorts,
        }

    @staticmethod
    def _fetch_vix() -> Optional[float]:
        """
        Fetch the latest ^VIX close price via yfinance.

        Returns:
            Latest VIX value as float, or None on failure.
        """
        try:
            ticker = yf.Ticker("^VIX")
            hist = ticker.history(period="5d", interval="1d")

            if hist is None or hist.empty:
                logger.warning("yfinance returned empty VIX data")
                return None

            # Normalize columns to lowercase
            hist.columns = [c.lower() for c in hist.columns]

            vix_value = float(hist["close"].iloc[-1])
            logger.debug(f"Fetched VIX value: {vix_value:.2f}")
            return vix_value

        except Exception as e:
            logger.error(f"Failed to fetch VIX data: {e}")
            return None
