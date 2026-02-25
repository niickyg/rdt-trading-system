"""
Chain Provider Abstraction for the RDT Trading System.

Provides an ABC so chain.py, iv_analyzer.py, and executor.py don't care
whether data comes from IBKR or is synthetically generated via Black-Scholes.

Implementations:
- IBKRChainProvider: Wraps existing IBKR calls (extracted from chain.py/iv_analyzer.py)
- PaperChainProvider: Generates synthetic chains using BS pricing + yfinance data
"""

import math
import os
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from loguru import logger

from options.models import OptionContract, OptionGreeks, OptionRight
from options.pricing import (
    black_scholes_price, generate_greeks, implied_volatility,
    DEFAULT_RISK_FREE_RATE,
)


class ChainProvider(ABC):
    """Abstract base class for options chain data providers."""

    @abstractmethod
    def get_chain_params(self, symbol: str) -> Optional[Dict]:
        """
        Get chain parameters (strikes, expirations, multiplier).

        Returns:
            Dict with keys: exchange, expirations (list of YYYYMMDD),
            strikes (list of float), multiplier (int)
            Or None if unavailable.
        """

    @abstractmethod
    def get_greeks(self, contract: OptionContract) -> Optional[OptionGreeks]:
        """
        Get Greeks for a single option contract.

        Returns:
            OptionGreeks or None if unavailable.
        """

    @abstractmethod
    def get_greeks_batch(
        self, contracts: List[OptionContract]
    ) -> Dict[str, OptionGreeks]:
        """
        Get Greeks for multiple contracts.

        Returns:
            Dict mapping contract.display_name to OptionGreeks.
        """

    @abstractmethod
    def get_iv_history(self, symbol: str) -> Optional[Dict]:
        """
        Get historical IV data for a symbol.

        Returns:
            Dict with 'iv_values' (list of float) and 'current_iv' (float),
            or None if unavailable.
        """

    @abstractmethod
    def get_price_history(
        self, symbol: str, period_days: int = 30
    ) -> Optional[List[float]]:
        """
        Get historical closing prices.

        Returns:
            List of closing prices (oldest first), or None.
        """

    @abstractmethod
    def get_underlying_price(self, symbol: str) -> float:
        """
        Get the current underlying price.

        Returns:
            Current price, or 0.0 if unavailable.
        """


class IBKRChainProvider(ChainProvider):
    """
    Chain provider backed by Interactive Brokers.

    Wraps the IBKR client calls that were previously embedded in
    chain.py and iv_analyzer.py.
    """

    def __init__(self, ib_client):
        """
        Args:
            ib_client: IBKRClient instance (connected).
        """
        self._ib = ib_client

    def get_chain_params(self, symbol: str) -> Optional[Dict]:
        try:
            from ib_insync import Stock

            ib = self._ib._ib
            contract = Stock(symbol.upper(), "SMART", "USD")
            ib.qualifyContracts(contract)

            chains = ib.reqSecDefOptParams(
                contract.symbol, "", contract.secType, contract.conId
            )
            if not chains:
                logger.warning(f"No option chains found for {symbol}")
                return None

            chain = None
            for c in chains:
                if c.exchange == "SMART":
                    chain = c
                    break
            if chain is None:
                chain = chains[0]

            return {
                "exchange": chain.exchange,
                "expirations": sorted(list(chain.expirations)),
                "strikes": sorted(list(chain.strikes)),
                "multiplier": int(chain.multiplier) if hasattr(chain, "multiplier") else 100,
            }
        except Exception as e:
            logger.error(f"IBKR chain params fetch failed for {symbol}: {e}")
            return None

    def get_greeks(self, contract: OptionContract) -> Optional[OptionGreeks]:
        try:
            from ib_insync import Option as IBOption

            ib = self._ib._ib
            ib_contract = IBOption(
                contract.symbol,
                contract.expiry,
                contract.strike,
                contract.right.value,
                contract.exchange,
                contract.currency,
            )

            qualified = ib.qualifyContracts(ib_contract)
            if not qualified:
                return None

            ticker = ib.reqMktData(ib_contract, snapshot=True)
            ib.sleep(1.0)
            ib.cancelMktData(ib_contract)

            greeks = ticker.modelGreeks
            bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0.0
            ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0.0
            last = ticker.last if ticker.last and ticker.last > 0 else 0.0
            volume = int(ticker.volume) if ticker.volume else 0

            open_interest = 0
            if hasattr(ticker, "callOpenInterest") and contract.right == OptionRight.CALL:
                oi = ticker.callOpenInterest
                open_interest = int(oi) if oi else 0
            elif hasattr(ticker, "putOpenInterest") and contract.right == OptionRight.PUT:
                oi = ticker.putOpenInterest
                open_interest = int(oi) if oi else 0

            if greeks:
                return OptionGreeks(
                    delta=greeks.delta or 0.0,
                    gamma=greeks.gamma or 0.0,
                    theta=greeks.theta or 0.0,
                    vega=greeks.vega or 0.0,
                    implied_vol=greeks.impliedVol or 0.0,
                    underlying_price=greeks.undPrice or 0.0,
                    option_price=greeks.optPrice or last,
                    bid=bid,
                    ask=ask,
                    volume=volume,
                    open_interest=open_interest,
                    timestamp=datetime.now(),
                )
            else:
                if bid > 0 or ask > 0 or last > 0:
                    return OptionGreeks(
                        option_price=last,
                        bid=bid,
                        ask=ask,
                        volume=volume,
                        open_interest=open_interest,
                        timestamp=datetime.now(),
                    )
                return None

        except Exception as e:
            logger.error(f"IBKR Greeks fetch failed for {contract.display_name}: {e}")
            return None

    def get_greeks_batch(
        self, contracts: List[OptionContract]
    ) -> Dict[str, OptionGreeks]:
        results = {}
        for contract in contracts:
            greeks = self.get_greeks(contract)
            if greeks:
                results[contract.display_name] = greeks
        return results

    def get_iv_history(self, symbol: str) -> Optional[Dict]:
        try:
            from ib_insync import Stock

            ib = self._ib._ib
            contract = Stock(symbol.upper(), "SMART", "USD")
            ib.qualifyContracts(contract)

            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="1 Y",
                barSizeSetting="1 day",
                whatToShow="OPTION_IMPLIED_VOLATILITY",
                useRTH=True,
                formatDate=1,
            )
            if not bars:
                return None

            iv_values = [bar.close for bar in bars if bar.close > 0]
            current_iv = iv_values[-1] if iv_values else 0.0
            return {"iv_values": iv_values, "current_iv": current_iv}

        except Exception as e:
            logger.warning(f"IBKR IV history fetch failed for {symbol}: {e}")
            return None

    def get_price_history(
        self, symbol: str, period_days: int = 30
    ) -> Optional[List[float]]:
        try:
            from ib_insync import Stock

            ib = self._ib._ib
            contract = Stock(symbol.upper(), "SMART", "USD")
            ib.qualifyContracts(contract)

            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=f"{period_days + 10} D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            if not bars:
                return None

            return [bar.close for bar in bars if bar.close > 0]

        except Exception as e:
            logger.warning(f"IBKR price history fetch failed for {symbol}: {e}")
            return None

    def get_underlying_price(self, symbol: str) -> float:
        try:
            quote = self._ib.get_quote(symbol)
            return quote.last or quote.mid or 0.0
        except Exception:
            return 0.0


class PaperChainProvider(ChainProvider):
    """
    Chain provider that generates synthetic options data using
    Black-Scholes pricing and yfinance for underlying prices.

    Produces realistic chains with:
    - Strike intervals: $1 (<$50), $2.50 ($50-200), $5 (>$200)
    - Expiries: next 4 weekly + next 3 monthly Fridays
    - Greeks: BS with synthetic IV (HV20 × multiplier + volatility skew)
    - Bid/ask: mid = BS price, spread = max($0.05, 2% of price)
    - OI/volume: random synthetic values
    """

    def __init__(self, risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
                 iv_multiplier: float = 1.1):
        """
        Args:
            risk_free_rate: Risk-free rate for BS pricing.
            iv_multiplier: HV-to-IV scaling factor (default 1.1).
        """
        self._r = risk_free_rate
        self._iv_multiplier = iv_multiplier

        # Cache: symbol -> {price, hv20, price_history, timestamp}
        self._data_cache: Dict[str, Dict] = {}
        self._cache_ttl = 300  # 5 minutes

    def get_chain_params(self, symbol: str) -> Optional[Dict]:
        price = self.get_underlying_price(symbol)
        if price <= 0:
            return None

        strikes = self._generate_strikes(price)
        expirations = self._generate_expirations()

        return {
            "exchange": "PAPER",
            "expirations": expirations,
            "strikes": strikes,
            "multiplier": 100,
        }

    def get_greeks(self, contract: OptionContract) -> Optional[OptionGreeks]:
        data = self._get_symbol_data(contract.symbol)
        if not data or data["price"] <= 0:
            return None

        S = data["price"]
        K = contract.strike
        hv20 = data.get("hv20", 0.25)

        # Calculate time to expiry
        T = self._time_to_expiry(contract.expiry)
        if T <= 0:
            return None

        # Synthetic IV with skew
        sigma = self._synthetic_iv(S, K, hv20, T)

        option_type = contract.right.value  # "C" or "P"
        greeks = generate_greeks(S, K, T, self._r, sigma, option_type)

        # Generate bid/ask spread
        mid = greeks.price
        spread = max(0.05, mid * 0.02)
        bid = max(0.01, round(mid - spread / 2, 2))
        ask = round(mid + spread / 2, 2)

        # Synthetic OI and volume
        oi = random.randint(100, 10000)
        vol = random.randint(10, 1000)

        return OptionGreeks(
            delta=greeks.delta,
            gamma=greeks.gamma,
            theta=greeks.theta,
            vega=greeks.vega,
            implied_vol=sigma,
            underlying_price=S,
            option_price=mid,
            bid=bid,
            ask=ask,
            volume=vol,
            open_interest=oi,
            timestamp=datetime.now(),
        )

    def get_greeks_batch(
        self, contracts: List[OptionContract]
    ) -> Dict[str, OptionGreeks]:
        results = {}
        for contract in contracts:
            greeks = self.get_greeks(contract)
            if greeks:
                results[contract.display_name] = greeks
        return results

    def get_iv_history(self, symbol: str) -> Optional[Dict]:
        """Generate synthetic IV history from price history."""
        prices = self.get_price_history(symbol, period_days=260)
        if not prices or len(prices) < 30:
            return None

        # Calculate rolling HV as proxy for IV history
        iv_values = []
        window = 20
        for i in range(window, len(prices)):
            window_prices = prices[i - window:i + 1]
            hv = self._calculate_hv_from_prices(window_prices)
            # Scale HV to approximate IV
            iv_values.append(hv * self._iv_multiplier)

        current_iv = iv_values[-1] if iv_values else 0.25
        return {"iv_values": iv_values, "current_iv": current_iv}

    def get_price_history(
        self, symbol: str, period_days: int = 30
    ) -> Optional[List[float]]:
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            period = f"{period_days}d" if period_days <= 365 else f"{period_days // 30}mo"
            hist = ticker.history(period=period)

            if hist is None or hist.empty:
                return None

            # Handle MultiIndex columns
            if hasattr(hist.columns, 'nlevels') and hist.columns.nlevels > 1:
                close_col = ("Close", symbol)
                if close_col in hist.columns:
                    closes = hist[close_col].dropna().tolist()
                else:
                    # Try first level
                    closes = hist["Close"].iloc[:, 0].dropna().tolist() if "Close" in hist.columns.get_level_values(0) else []
            else:
                col = "Close" if "Close" in hist.columns else "close"
                closes = hist[col].dropna().tolist() if col in hist.columns else []

            return closes if closes else None

        except Exception as e:
            logger.warning(f"yfinance price history failed for {symbol}: {e}")
            return None

    def get_underlying_price(self, symbol: str) -> float:
        data = self._get_symbol_data(symbol)
        return data["price"] if data else 0.0

    def _get_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Get cached symbol data (price, HV20), refreshing if stale."""
        now = time.time()
        cached = self._data_cache.get(symbol)
        if cached and now - cached.get("timestamp", 0) < self._cache_ttl:
            return cached

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")

            if hist is None or hist.empty:
                return None

            # Handle MultiIndex columns
            if hasattr(hist.columns, 'nlevels') and hist.columns.nlevels > 1:
                close_col = ("Close", symbol)
                if close_col in hist.columns:
                    closes = hist[close_col].dropna().tolist()
                else:
                    closes = hist["Close"].iloc[:, 0].dropna().tolist() if "Close" in hist.columns.get_level_values(0) else []
            else:
                col = "Close" if "Close" in hist.columns else "close"
                closes = hist[col].dropna().tolist() if col in hist.columns else []

            if not closes:
                return None

            price = closes[-1]
            hv20 = self._calculate_hv_from_prices(closes)

            data = {
                "price": price,
                "hv20": hv20,
                "closes": closes,
                "timestamp": now,
            }
            self._data_cache[symbol] = data
            return data

        except Exception as e:
            logger.warning(f"Failed to fetch data for {symbol}: {e}")
            return None

    def _generate_strikes(self, price: float) -> List[float]:
        """Generate realistic strike prices around the underlying price."""
        if price < 50:
            interval = 1.0
        elif price < 200:
            interval = 2.5
        else:
            interval = 5.0

        # Generate strikes ±30% around current price
        low = price * 0.70
        high = price * 1.30

        # Align to interval
        first_strike = math.ceil(low / interval) * interval
        strikes = []
        strike = first_strike
        while strike <= high:
            strikes.append(round(strike, 2))
            strike += interval

        return strikes

    def _generate_expirations(self) -> List[str]:
        """Generate next 4 weekly + next 3 monthly Friday expirations."""
        today = datetime.now().date()
        expirations = set()

        # Next 4 weekly Fridays
        days_until_friday = (4 - today.weekday()) % 7
        if days_until_friday == 0:
            days_until_friday = 7  # Skip this Friday if it's today
        next_friday = today + timedelta(days=days_until_friday)

        for i in range(4):
            exp = next_friday + timedelta(weeks=i)
            expirations.add(exp.strftime("%Y%m%d"))

        # Next 3 monthly expirations (third Friday of each month)
        month = today.month
        year = today.year
        for _ in range(3):
            month += 1
            if month > 12:
                month = 1
                year += 1

            # Find third Friday
            first_day = datetime(year, month, 1).date()
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(weeks=2)
            expirations.add(third_friday.strftime("%Y%m%d"))

        return sorted(expirations)

    def _time_to_expiry(self, expiry_str: str) -> float:
        """Calculate time to expiry in years."""
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y%m%d").date()
            today = datetime.now().date()
            days = (expiry_date - today).days
            return max(0.0, days / 365.0)
        except ValueError:
            return 0.0

    def _synthetic_iv(
        self, S: float, K: float, hv20: float, T: float
    ) -> float:
        """
        Generate synthetic IV with volatility skew.

        Base IV = HV20 × iv_multiplier.
        Skew: OTM options have higher IV (volatility smile).
        """
        base_iv = max(0.10, hv20 * self._iv_multiplier)

        # Moneyness-based skew
        moneyness = math.log(K / S) if S > 0 and K > 0 else 0
        # Parabolic skew: higher IV for deep OTM/ITM
        skew = 0.10 * moneyness * moneyness
        # Put skew: slightly higher IV for puts (negative moneyness)
        if moneyness < 0:
            skew += 0.02 * abs(moneyness)

        # Term structure: shorter-dated options have slightly higher IV
        term_adj = 0.02 * max(0, 0.25 - T) if T < 0.25 else 0

        return max(0.05, base_iv + skew + term_adj)

    @staticmethod
    def _calculate_hv_from_prices(closes: List[float]) -> float:
        """Calculate annualized historical volatility from closing prices."""
        if len(closes) < 3:
            return 0.25  # Default if insufficient data

        log_returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0 and closes[i] > 0:
                log_returns.append(math.log(closes[i] / closes[i - 1]))

        if len(log_returns) < 2:
            return 0.25

        mean = sum(log_returns) / len(log_returns)
        variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
        std_dev = math.sqrt(variance)

        return std_dev * math.sqrt(252)
