"""
Tests for options/chain_provider.py — ChainProvider ABC and PaperChainProvider.

Uses mocked yfinance data to avoid network calls.
"""

import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from options.chain_provider import ChainProvider, PaperChainProvider
from options.models import OptionContract, OptionGreeks, OptionRight


# ---------------------------------------------------------------------------
# Mock yfinance data
# ---------------------------------------------------------------------------

def make_mock_hist(symbol="AAPL", price=150.0, days=30):
    """Create a mock yfinance history DataFrame-like object."""
    import pandas as pd
    import numpy as np

    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    # Generate realistic prices with some volatility
    np.random.seed(42)
    returns = np.random.normal(0, 0.015, days)
    prices = [price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))

    df = pd.DataFrame({'Close': prices}, index=dates)
    return df


class TestPaperChainProviderStrikeGeneration:
    """Test strike generation logic without network calls."""

    def test_strike_intervals_under_50(self):
        provider = PaperChainProvider()
        strikes = provider._generate_strikes(30.0)
        # Should use $1 intervals
        intervals = [strikes[i + 1] - strikes[i] for i in range(len(strikes) - 1)]
        assert all(abs(i - 1.0) < 0.01 for i in intervals)

    def test_strike_intervals_50_to_200(self):
        provider = PaperChainProvider()
        strikes = provider._generate_strikes(100.0)
        intervals = [strikes[i + 1] - strikes[i] for i in range(len(strikes) - 1)]
        assert all(abs(i - 2.5) < 0.01 for i in intervals)

    def test_strike_intervals_over_200(self):
        provider = PaperChainProvider()
        strikes = provider._generate_strikes(300.0)
        intervals = [strikes[i + 1] - strikes[i] for i in range(len(strikes) - 1)]
        assert all(abs(i - 5.0) < 0.01 for i in intervals)

    def test_strikes_surround_price(self):
        provider = PaperChainProvider()
        price = 150.0
        strikes = provider._generate_strikes(price)
        # Price should be within the strike range
        assert min(strikes) < price
        assert max(strikes) > price

    def test_strikes_sorted(self):
        provider = PaperChainProvider()
        strikes = provider._generate_strikes(100.0)
        assert strikes == sorted(strikes)


class TestPaperChainProviderExpirations:
    """Test expiration generation."""

    def test_generates_expirations(self):
        provider = PaperChainProvider()
        expirations = provider._generate_expirations()
        assert len(expirations) >= 4  # At least 4 weeklies

    def test_expirations_sorted(self):
        provider = PaperChainProvider()
        expirations = provider._generate_expirations()
        assert expirations == sorted(expirations)

    def test_expirations_are_valid_dates(self):
        provider = PaperChainProvider()
        expirations = provider._generate_expirations()
        for exp in expirations:
            date = datetime.strptime(exp, "%Y%m%d")
            assert date > datetime.now()

    def test_all_expirations_are_fridays(self):
        provider = PaperChainProvider()
        expirations = provider._generate_expirations()
        for exp in expirations:
            date = datetime.strptime(exp, "%Y%m%d")
            assert date.weekday() == 4, f"{exp} is not a Friday"


class TestPaperChainProviderGreeks:
    """Test Greeks generation with mocked price data."""

    @patch('options.chain_provider.PaperChainProvider._get_symbol_data')
    def test_get_greeks_returns_option_greeks(self, mock_data):
        mock_data.return_value = {"price": 150.0, "hv20": 0.25}
        provider = PaperChainProvider()

        # Create a contract with a future expiry
        future_expiry = (datetime.now() + timedelta(days=30)).strftime("%Y%m%d")
        contract = OptionContract(
            symbol="AAPL", expiry=future_expiry, strike=150.0,
            right=OptionRight.CALL
        )

        greeks = provider.get_greeks(contract)
        assert greeks is not None
        assert isinstance(greeks, OptionGreeks)
        assert greeks.delta > 0  # Call delta is positive
        assert greeks.gamma > 0
        assert greeks.theta < 0  # Time decay
        assert greeks.vega > 0
        assert greeks.implied_vol > 0
        assert greeks.underlying_price == 150.0
        assert greeks.option_price > 0
        assert greeks.bid > 0
        assert greeks.ask > greeks.bid
        assert greeks.open_interest >= 100
        assert greeks.volume >= 10

    @patch('options.chain_provider.PaperChainProvider._get_symbol_data')
    def test_put_greeks(self, mock_data):
        mock_data.return_value = {"price": 150.0, "hv20": 0.25}
        provider = PaperChainProvider()

        future_expiry = (datetime.now() + timedelta(days=30)).strftime("%Y%m%d")
        contract = OptionContract(
            symbol="AAPL", expiry=future_expiry, strike=150.0,
            right=OptionRight.PUT
        )

        greeks = provider.get_greeks(contract)
        assert greeks is not None
        assert greeks.delta < 0  # Put delta is negative

    @patch('options.chain_provider.PaperChainProvider._get_symbol_data')
    def test_expired_contract_returns_none(self, mock_data):
        mock_data.return_value = {"price": 150.0, "hv20": 0.25}
        provider = PaperChainProvider()

        past_expiry = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        contract = OptionContract(
            symbol="AAPL", expiry=past_expiry, strike=150.0,
            right=OptionRight.CALL
        )

        greeks = provider.get_greeks(contract)
        assert greeks is None

    @patch('options.chain_provider.PaperChainProvider._get_symbol_data')
    def test_no_data_returns_none(self, mock_data):
        mock_data.return_value = None
        provider = PaperChainProvider()

        future_expiry = (datetime.now() + timedelta(days=30)).strftime("%Y%m%d")
        contract = OptionContract(
            symbol="AAPL", expiry=future_expiry, strike=150.0,
            right=OptionRight.CALL
        )

        greeks = provider.get_greeks(contract)
        assert greeks is None


class TestPaperChainProviderChainParams:
    @patch('options.chain_provider.PaperChainProvider.get_underlying_price')
    def test_get_chain_params(self, mock_price):
        mock_price.return_value = 150.0
        provider = PaperChainProvider()

        params = provider.get_chain_params("AAPL")
        assert params is not None
        assert "exchange" in params
        assert params["exchange"] == "PAPER"
        assert "expirations" in params
        assert "strikes" in params
        assert "multiplier" in params
        assert params["multiplier"] == 100
        assert len(params["strikes"]) > 0
        assert len(params["expirations"]) >= 4

    @patch('options.chain_provider.PaperChainProvider.get_underlying_price')
    def test_no_price_returns_none(self, mock_price):
        mock_price.return_value = 0.0
        provider = PaperChainProvider()

        params = provider.get_chain_params("INVALID")
        assert params is None


class TestPaperChainProviderBatch:
    @patch('options.chain_provider.PaperChainProvider._get_symbol_data')
    def test_get_greeks_batch(self, mock_data):
        mock_data.return_value = {"price": 150.0, "hv20": 0.25}
        provider = PaperChainProvider()

        future_expiry = (datetime.now() + timedelta(days=30)).strftime("%Y%m%d")
        contracts = [
            OptionContract(symbol="AAPL", expiry=future_expiry, strike=145.0, right=OptionRight.CALL),
            OptionContract(symbol="AAPL", expiry=future_expiry, strike=150.0, right=OptionRight.CALL),
            OptionContract(symbol="AAPL", expiry=future_expiry, strike=155.0, right=OptionRight.CALL),
        ]

        results = provider.get_greeks_batch(contracts)
        assert len(results) == 3
        # Delta should decrease as strike increases (for calls)
        deltas = [results[c.display_name].delta for c in contracts]
        assert deltas[0] > deltas[1] > deltas[2]


class TestSyntheticIV:
    def test_base_iv_scales_hv(self):
        provider = PaperChainProvider(iv_multiplier=1.1)
        iv = provider._synthetic_iv(100, 100, 0.20, 0.25)
        # Base should be approximately 0.20 * 1.1 = 0.22 + skew
        assert 0.20 < iv < 0.35

    def test_otm_has_higher_iv(self):
        provider = PaperChainProvider()
        iv_atm = provider._synthetic_iv(100, 100, 0.25, 0.25)
        iv_otm = provider._synthetic_iv(100, 120, 0.25, 0.25)
        # OTM should have higher IV due to skew
        assert iv_otm > iv_atm

    def test_min_iv_floor(self):
        provider = PaperChainProvider()
        iv = provider._synthetic_iv(100, 100, 0.01, 0.25)
        assert iv >= 0.05


class TestHVCalculation:
    def test_hv_from_prices(self):
        # Generate known prices with ~20% annualized vol
        prices = [100.0]
        daily_vol = 0.20 / math.sqrt(252)
        for i in range(30):
            prices.append(prices[-1] * (1 + daily_vol * (1 if i % 2 == 0 else -1)))

        hv = PaperChainProvider._calculate_hv_from_prices(prices)
        # Should be roughly 20% annualized
        assert 0.10 < hv < 0.40

    def test_insufficient_data(self):
        hv = PaperChainProvider._calculate_hv_from_prices([100, 101])
        assert hv == 0.25  # Default fallback


class TestTimeToExpiry:
    def test_future_date(self):
        provider = PaperChainProvider()
        future = (datetime.now() + timedelta(days=30)).strftime("%Y%m%d")
        T = provider._time_to_expiry(future)
        assert 0.07 < T < 0.10  # ~30/365

    def test_past_date(self):
        provider = PaperChainProvider()
        past = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        T = provider._time_to_expiry(past)
        assert T == 0.0

    def test_invalid_date(self):
        provider = PaperChainProvider()
        T = provider._time_to_expiry("invalid")
        assert T == 0.0
