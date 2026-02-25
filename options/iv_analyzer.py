"""
Implied Volatility Analyzer for the RDT Trading System.

Calculates IV rank, IV percentile, and historical volatility metrics
to inform strategy selection (buy premium when IV is low, sell when high).
"""

import math
import time
from datetime import datetime
from typing import Dict, Optional
from loguru import logger

from options.models import IVAnalysis, OptionRight
from options.chain import OptionsChainManager
from options.chain_provider import ChainProvider


class IVAnalyzer:
    """
    Analyzes implied volatility for options strategy selection.

    Uses a ChainProvider for IV and price history data.
    Fallback: ATM option chain implied volatility.

    Usage:
        iv_analyzer = IVAnalyzer(provider, chain_manager)
        analysis = iv_analyzer.analyze("AAPL")
        print(f"IV Rank: {analysis.iv_rank}, Regime: {analysis.regime}")
    """

    def __init__(self, provider: ChainProvider, chain_manager: OptionsChainManager):
        """
        Args:
            provider: ChainProvider instance for data access
            chain_manager: OptionsChainManager for chain data access
        """
        self._provider = provider
        self._chain_manager = chain_manager

        # Cache: symbol -> IVAnalysis
        self._cache: Dict[str, IVAnalysis] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes

    def analyze(self, symbol: str) -> IVAnalysis:
        """
        Perform full IV analysis for a symbol.

        Args:
            symbol: Underlying symbol

        Returns:
            IVAnalysis with iv_rank, iv_percentile, hv_20, etc.
        """
        now = time.time()
        cached_time = self._cache_time.get(symbol, 0)
        if symbol in self._cache and now - cached_time < self._cache_ttl:
            return self._cache[symbol]

        analysis = self._compute_iv_analysis(symbol)
        self._cache[symbol] = analysis
        self._cache_time[symbol] = now
        return analysis

    def _compute_iv_analysis(self, symbol: str) -> IVAnalysis:
        """Compute IV analysis from provider data."""
        current_iv = 0.0
        iv_history = []
        hv_20 = 0.0

        # Try to get IV history from provider
        try:
            iv_data = self._provider.get_iv_history(symbol)
            if iv_data:
                iv_history = iv_data.get("iv_values", [])
                current_iv = iv_data.get("current_iv", 0.0)
        except Exception as e:
            logger.warning(f"Failed to fetch IV history for {symbol}: {e}")

        # Fallback: get current IV from ATM option
        if current_iv == 0.0:
            current_iv = self._get_atm_iv(symbol)

        # Get historical volatility from price data
        try:
            hv_20 = self._calculate_hv(symbol, period=20)
        except Exception as e:
            logger.warning(f"Failed to calculate HV for {symbol}: {e}")

        # Calculate IV rank and percentile
        iv_rank = 0.0
        iv_percentile = 0.0
        iv_high = current_iv
        iv_low = current_iv

        if iv_history and len(iv_history) >= 20:
            iv_high = max(iv_history)
            iv_low = min(iv_history)

            # IV Rank = (Current - 52w Low) / (52w High - 52w Low)
            iv_range = iv_high - iv_low
            if iv_range > 0:
                iv_rank = ((current_iv - iv_low) / iv_range) * 100
                iv_rank = max(0.0, min(100.0, iv_rank))

            # IV Percentile = % of days current IV was lower
            days_below = sum(1 for iv in iv_history if iv < current_iv)
            iv_percentile = (days_below / len(iv_history)) * 100

        # IV/HV ratio
        iv_hv_ratio = current_iv / hv_20 if hv_20 > 0 else 1.0

        analysis = IVAnalysis(
            symbol=symbol,
            current_iv=current_iv,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            hv_20=hv_20,
            iv_high_52w=iv_high,
            iv_low_52w=iv_low,
            iv_hv_ratio=iv_hv_ratio,
            timestamp=datetime.now(),
        )

        logger.info(
            f"IV Analysis: {symbol} IV={current_iv:.1%} "
            f"Rank={iv_rank:.0f} Pctile={iv_percentile:.0f} "
            f"HV20={hv_20:.1%} Regime={analysis.regime.value}"
        )

        return analysis

    def _get_atm_iv(self, symbol: str) -> float:
        """Get current IV from ATM option as fallback."""
        try:
            underlying_price = self._provider.get_underlying_price(symbol)
            if underlying_price <= 0:
                return 0.0

            # Find ATM strike
            atm_strike = self._chain_manager.get_atm_strike(symbol, underlying_price)

            # Find target expiry
            expiry = self._chain_manager.find_target_expiry(symbol)
            if not expiry:
                return 0.0

            # Get ATM call Greeks
            from options.models import OptionContract
            atm_contract = OptionContract(
                symbol=symbol,
                expiry=expiry,
                strike=atm_strike,
                right=OptionRight.CALL,
            )

            greeks = self._chain_manager.get_greeks(atm_contract)
            if greeks and greeks.implied_vol > 0:
                return greeks.implied_vol

            return 0.0

        except Exception as e:
            logger.warning(f"ATM IV fallback failed for {symbol}: {e}")
            return 0.0

    def _calculate_hv(self, symbol: str, period: int = 20) -> float:
        """
        Calculate historical volatility from price data.

        Args:
            symbol: Underlying symbol
            period: Number of trading days for HV calculation

        Returns:
            Annualized historical volatility (decimal)
        """
        try:
            closes = self._provider.get_price_history(symbol, period_days=period + 10)
            if not closes or len(closes) < period + 1:
                return 0.0

            # Calculate log returns
            log_returns = []
            for i in range(1, len(closes)):
                if closes[i - 1] > 0:
                    log_returns.append(math.log(closes[i] / closes[i - 1]))

            if len(log_returns) < period:
                return 0.0

            # Use most recent 'period' returns
            recent_returns = log_returns[-period:]

            # Standard deviation of returns
            mean = sum(recent_returns) / len(recent_returns)
            variance = sum((r - mean) ** 2 for r in recent_returns) / (len(recent_returns) - 1)
            std_dev = math.sqrt(variance)

            # Annualize (252 trading days)
            hv = std_dev * math.sqrt(252)
            return hv

        except Exception as e:
            logger.warning(f"HV calculation failed for {symbol}: {e}")
            return 0.0

    def clear_cache(self):
        """Clear the IV analysis cache."""
        self._cache.clear()
        self._cache_time.clear()
