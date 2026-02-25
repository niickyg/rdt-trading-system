"""
Options Chain Manager for the RDT Trading System.

Fetches, caches, and queries option chains via a ChainProvider.
Provides delta-based strike selection with liquidity filtering.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger

from options.models import (
    OptionContract, OptionGreeks, OptionRight,
)
from options.config import OptionsConfig
from options.chain_provider import ChainProvider


class OptionsChainManager:
    """
    Manages options chain data with caching and liquidity filtering.

    Usage:
        chain_mgr = OptionsChainManager(provider, config)
        contract = chain_mgr.find_by_delta("AAPL", 0.60, OptionRight.CALL, "20260320")
    """

    def __init__(self, provider: ChainProvider, config: Optional[OptionsConfig] = None):
        """
        Args:
            provider: ChainProvider instance (IBKRChainProvider or PaperChainProvider)
            config: Options configuration
        """
        self._provider = provider
        self._config = config or OptionsConfig()

        # Caches: symbol -> {expiry -> data}
        self._chain_cache: Dict[str, Dict] = {}
        self._chain_cache_time: Dict[str, float] = {}
        self._greeks_cache: Dict[str, Dict[str, OptionGreeks]] = {}
        self._greeks_cache_time: Dict[str, float] = {}

        # Rate limiting
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = (time.time() - self._last_request_time) * 1000  # ms
        if elapsed < self._config.rate_limit_ms:
            time.sleep((self._config.rate_limit_ms - elapsed) / 1000)
        self._last_request_time = time.time()

    def get_expirations(self, symbol: str) -> List[str]:
        """
        Get available expiration dates for a symbol.

        Args:
            symbol: Underlying symbol

        Returns:
            List of expiration dates (YYYYMMDD) sorted ascending
        """
        chain_data = self._get_chain_params(symbol)
        if not chain_data:
            return []
        return sorted(chain_data.get("expirations", []))

    def get_chain(
        self,
        symbol: str,
        expiry: str,
        right: Optional[OptionRight] = None
    ) -> List[OptionContract]:
        """
        Get all option contracts for a symbol/expiry.

        Args:
            symbol: Underlying symbol
            expiry: Expiration date (YYYYMMDD)
            right: Filter by CALL or PUT (None for both)

        Returns:
            List of OptionContract objects
        """
        chain_data = self._get_chain_params(symbol)
        if not chain_data:
            return []

        strikes = chain_data.get("strikes", [])
        contracts = []

        rights = [right] if right else [OptionRight.CALL, OptionRight.PUT]

        for strike in strikes:
            for r in rights:
                contracts.append(OptionContract(
                    symbol=symbol.upper(),
                    expiry=expiry,
                    strike=strike,
                    right=r,
                ))

        return contracts

    def get_atm_strike(self, symbol: str, underlying_price: float) -> float:
        """
        Find the at-the-money strike closest to the underlying price.

        Args:
            symbol: Underlying symbol
            underlying_price: Current underlying price

        Returns:
            Closest strike price
        """
        chain_data = self._get_chain_params(symbol)
        if not chain_data or not chain_data.get("strikes"):
            return round(underlying_price)

        strikes = chain_data["strikes"]
        return min(strikes, key=lambda s: abs(s - underlying_price))

    def find_target_expiry(self, symbol: str) -> Optional[str]:
        """
        Find the expiration closest to the target DTE.

        Returns:
            Expiration date string (YYYYMMDD) or None
        """
        expirations = self.get_expirations(symbol)
        if not expirations:
            return None

        today = datetime.now().date()
        target_date = today + timedelta(days=self._config.dte_target)
        min_date = today + timedelta(days=self._config.dte_min)
        max_date = today + timedelta(days=self._config.dte_max)

        best_expiry = None
        best_diff = float('inf')

        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y%m%d").date()
            except ValueError:
                continue

            if exp_date < min_date or exp_date > max_date:
                continue

            diff = abs((exp_date - target_date).days)
            if diff < best_diff:
                best_diff = diff
                best_expiry = exp_str

        return best_expiry

    def find_by_delta(
        self,
        symbol: str,
        target_delta: float,
        right: OptionRight,
        expiry: str,
        tolerance: Optional[float] = None,
    ) -> Optional[Tuple[OptionContract, OptionGreeks]]:
        """
        Find the option strike closest to a target delta.

        Args:
            symbol: Underlying symbol
            target_delta: Target delta (positive for calls, negative for puts)
            right: CALL or PUT
            expiry: Expiration date (YYYYMMDD)
            tolerance: Max delta deviation (default from config)

        Returns:
            Tuple of (OptionContract, OptionGreeks) or None
        """
        tol = tolerance if tolerance is not None else self._config.delta_tolerance

        chain_data = self._get_chain_params(symbol)
        if not chain_data or not chain_data.get("strikes"):
            logger.warning(f"No chain data for {symbol}")
            return None

        strikes = chain_data["strikes"]
        abs_target = abs(target_delta)

        best_contract = None
        best_greeks = None
        best_diff = float('inf')

        for strike in strikes:
            contract = OptionContract(
                symbol=symbol.upper(),
                expiry=expiry,
                strike=strike,
                right=right,
            )

            greeks = self.get_greeks(contract)
            if greeks is None:
                continue

            # Apply liquidity filters
            if not self._passes_liquidity_filter(greeks):
                continue

            abs_delta = abs(greeks.delta)
            diff = abs(abs_delta - abs_target)

            if diff < best_diff and diff <= tol:
                best_diff = diff
                best_contract = contract
                best_greeks = greeks

        if best_contract is None:
            logger.warning(
                f"No option found for {symbol} {expiry} {right.value} "
                f"delta={target_delta} (tolerance={tol})"
            )
            # Retry with relaxed tolerance
            if tol < 0.15:
                logger.info("Retrying with relaxed tolerance (0.15)")
                return self.find_by_delta(symbol, target_delta, right, expiry, 0.15)

        return (best_contract, best_greeks) if best_contract else None

    def get_greeks(self, contract: OptionContract) -> Optional[OptionGreeks]:
        """
        Get Greeks for a single option contract.

        Args:
            contract: Option contract specification

        Returns:
            OptionGreeks or None if unavailable
        """
        cache_key = contract.display_name
        now = time.time()

        # Check cache
        if cache_key in self._greeks_cache:
            cached_time = self._greeks_cache_time.get(cache_key, 0)
            if now - cached_time < self._config.greeks_cache_ttl:
                return self._greeks_cache[cache_key]

        self._rate_limit()

        try:
            greeks = self._provider.get_greeks(contract)
            if greeks:
                self._greeks_cache[cache_key] = greeks
                self._greeks_cache_time[cache_key] = now
            return greeks

        except Exception as e:
            logger.error(f"Failed to get Greeks for {contract.display_name}: {e}")
            return None

    def get_greeks_batch(
        self,
        contracts: List[OptionContract]
    ) -> Dict[str, OptionGreeks]:
        """
        Get Greeks for multiple contracts efficiently.

        Args:
            contracts: List of option contracts

        Returns:
            Dict mapping contract display_name to OptionGreeks
        """
        results = {}
        to_fetch = []

        now = time.time()

        # Check cache first
        for contract in contracts:
            cache_key = contract.display_name
            if cache_key in self._greeks_cache:
                cached_time = self._greeks_cache_time.get(cache_key, 0)
                if now - cached_time < self._config.greeks_cache_ttl:
                    results[cache_key] = self._greeks_cache[cache_key]
                    continue
            to_fetch.append(contract)

        # Fetch remaining from provider
        if to_fetch:
            self._rate_limit()
            batch_results = self._provider.get_greeks_batch(to_fetch)
            for key, greeks in batch_results.items():
                self._greeks_cache[key] = greeks
                self._greeks_cache_time[key] = time.time()
                results[key] = greeks

        return results

    def _passes_liquidity_filter(self, greeks: OptionGreeks) -> bool:
        """Check if an option passes liquidity requirements."""
        if greeks.open_interest < self._config.min_open_interest:
            return False
        if greeks.spread_pct > self._config.max_bid_ask_spread_pct:
            return False
        return True

    def _get_chain_params(self, symbol: str) -> Optional[Dict]:
        """Get chain parameters (strikes, expirations) with caching."""
        now = time.time()
        cached_time = self._chain_cache_time.get(symbol, 0)

        if symbol in self._chain_cache and now - cached_time < self._config.chain_cache_ttl:
            return self._chain_cache[symbol]

        self._rate_limit()

        try:
            data = self._provider.get_chain_params(symbol)
            if data:
                self._chain_cache[symbol] = data
                self._chain_cache_time[symbol] = now
            return data

        except Exception as e:
            logger.error(f"Failed to get chain params for {symbol}: {e}")
            return None

    def clear_cache(self):
        """Clear all caches."""
        self._chain_cache.clear()
        self._chain_cache_time.clear()
        self._greeks_cache.clear()
        self._greeks_cache_time.clear()
