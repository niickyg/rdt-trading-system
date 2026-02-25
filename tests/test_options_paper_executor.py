"""
Tests for options/paper_executor.py — Paper options executor.

Tests fill simulation, position tracking, and P&L calculation.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from options.paper_executor import PaperOptionsExecutor
from options.models import (
    OptionAction, OptionContract, OptionGreeks, OptionLeg,
    OptionRight, OptionsPositionSizeResult, OptionsStrategy,
    StrategyDirection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_contract(symbol="AAPL", strike=150.0, right=OptionRight.CALL, days=30):
    expiry = (datetime.now() + timedelta(days=days)).strftime("%Y%m%d")
    return OptionContract(symbol=symbol, expiry=expiry, strike=strike, right=right)


def make_greeks(delta=0.55, bid=3.00, ask=3.20, implied_vol=0.25):
    return OptionGreeks(
        delta=delta, gamma=0.03, theta=-0.05, vega=0.12,
        bid=bid, ask=ask, option_price=(bid + ask) / 2,
        implied_vol=implied_vol, underlying_price=150.0,
        volume=500, open_interest=5000,
    )


def make_long_call_strategy(symbol="AAPL"):
    contract = make_contract(symbol=symbol, strike=150.0)
    greeks = make_greeks()
    leg = OptionLeg(contract=contract, action=OptionAction.BUY, quantity=1, greeks=greeks)
    return OptionsStrategy(
        name="long_call",
        underlying=symbol,
        direction=StrategyDirection.LONG,
        legs=[leg],
        max_loss=310.0,  # premium
        max_profit=float('inf'),
        net_premium=-310.0,
    )


def make_bull_call_spread(symbol="AAPL"):
    long_contract = make_contract(symbol=symbol, strike=150.0)
    short_contract = make_contract(symbol=symbol, strike=155.0)
    long_greeks = make_greeks(delta=0.55, bid=3.00, ask=3.20)
    short_greeks = make_greeks(delta=0.35, bid=1.50, ask=1.70)

    long_leg = OptionLeg(contract=long_contract, action=OptionAction.BUY, quantity=1, greeks=long_greeks)
    short_leg = OptionLeg(contract=short_contract, action=OptionAction.SELL, quantity=1, greeks=short_greeks)

    return OptionsStrategy(
        name="bull_call_spread",
        underlying=symbol,
        direction=StrategyDirection.LONG,
        legs=[long_leg, short_leg],
        max_loss=150.0,
        max_profit=350.0,
        net_premium=-150.0,
    )


def make_mock_provider():
    provider = MagicMock()
    provider.get_greeks.return_value = make_greeks()
    return provider


@pytest.fixture(autouse=True)
def no_db_load(monkeypatch):
    """Prevent executor from loading positions from DB during unit tests."""
    monkeypatch.setattr(PaperOptionsExecutor, "_get_repository", lambda self: None)


class TestPaperExecutorExecute:
    def test_execute_single_leg(self):
        provider = make_mock_provider()
        executor = PaperOptionsExecutor(provider)

        strategy = make_long_call_strategy()
        size = OptionsPositionSizeResult(
            strategy_name="long_call", contracts=2, max_risk=620, premium_cost=620
        )

        result = executor.execute_strategy(strategy, size)
        assert result is not None
        assert result["status"] == "Filled"
        assert result["contracts"] == 2
        assert "order_id" in result
        assert len(result["fill_details"]) == 1
        assert result["fill_details"][0]["action"] == "BUY"

    def test_execute_multi_leg(self):
        provider = make_mock_provider()
        executor = PaperOptionsExecutor(provider)

        strategy = make_bull_call_spread()
        size = OptionsPositionSizeResult(
            strategy_name="bull_call_spread", contracts=1, max_risk=150, premium_cost=150
        )

        result = executor.execute_strategy(strategy, size)
        assert result is not None
        assert result["status"] == "Filled"
        assert len(result["fill_details"]) == 2

    def test_zero_contracts_returns_none(self):
        provider = make_mock_provider()
        executor = PaperOptionsExecutor(provider)

        strategy = make_long_call_strategy()
        size = OptionsPositionSizeResult(
            strategy_name="long_call", contracts=0, max_risk=0, premium_cost=0
        )

        result = executor.execute_strategy(strategy, size)
        assert result is None

    def test_tracks_position(self):
        provider = make_mock_provider()
        executor = PaperOptionsExecutor(provider)

        strategy = make_long_call_strategy()
        size = OptionsPositionSizeResult(
            strategy_name="long_call", contracts=1, max_risk=310, premium_cost=310
        )

        executor.execute_strategy(strategy, size)
        pos = executor.get_position("AAPL")
        assert pos is not None
        assert pos["strategy"].name == "long_call"
        assert pos["contracts"] == 1

    def test_missing_greeks_falls_back_to_provider(self):
        provider = make_mock_provider()
        executor = PaperOptionsExecutor(provider)

        # Create strategy with no greeks on leg
        contract = make_contract()
        leg = OptionLeg(contract=contract, action=OptionAction.BUY, quantity=1, greeks=None)
        strategy = OptionsStrategy(
            name="long_call", underlying="AAPL",
            direction=StrategyDirection.LONG, legs=[leg],
        )
        size = OptionsPositionSizeResult(
            strategy_name="long_call", contracts=1, max_risk=310, premium_cost=310
        )

        result = executor.execute_strategy(strategy, size)
        assert result is not None
        provider.get_greeks.assert_called()


class TestPaperExecutorClose:
    def test_close_position(self):
        provider = make_mock_provider()
        executor = PaperOptionsExecutor(provider)

        strategy = make_long_call_strategy()
        size = OptionsPositionSizeResult(
            strategy_name="long_call", contracts=1, max_risk=310, premium_cost=310
        )

        executor.execute_strategy(strategy, size)
        assert executor.get_position("AAPL") is not None

        result = executor.close_position("AAPL")
        assert result is not None
        assert result["status"] == "Filled"
        assert executor.get_position("AAPL") is None

    def test_close_nonexistent_returns_none(self):
        provider = make_mock_provider()
        executor = PaperOptionsExecutor(provider)

        result = executor.close_position("MSFT")
        assert result is None


class TestPaperExecutorRoll:
    def test_roll_position(self):
        provider = make_mock_provider()
        executor = PaperOptionsExecutor(provider)

        strategy = make_long_call_strategy()
        size = OptionsPositionSizeResult(
            strategy_name="long_call", contracts=1, max_risk=310, premium_cost=310
        )

        executor.execute_strategy(strategy, size)

        new_expiry = (datetime.now() + timedelta(days=60)).strftime("%Y%m%d")
        result = executor.roll_position("AAPL", new_expiry)
        assert result is not None
        assert "close" in result
        assert "open" in result
        assert result["new_expiry"] == new_expiry

        # Position should still exist with new expiry
        pos = executor.get_position("AAPL")
        assert pos is not None

    def test_roll_nonexistent_returns_none(self):
        provider = make_mock_provider()
        executor = PaperOptionsExecutor(provider)

        result = executor.roll_position("MSFT", "20261231")
        assert result is None


class TestPaperExecutorPositions:
    def test_get_all_positions(self):
        provider = make_mock_provider()
        executor = PaperOptionsExecutor(provider)

        # Execute two strategies
        for symbol in ["AAPL", "MSFT"]:
            strategy = make_long_call_strategy(symbol)
            size = OptionsPositionSizeResult(
                strategy_name="long_call", contracts=1, max_risk=310, premium_cost=310
            )
            executor.execute_strategy(strategy, size)

        positions = executor.get_all_positions()
        assert len(positions) == 2
        assert "AAPL" in positions
        assert "MSFT" in positions

    def test_positions_have_pnl(self):
        provider = make_mock_provider()
        executor = PaperOptionsExecutor(provider)

        strategy = make_long_call_strategy()
        size = OptionsPositionSizeResult(
            strategy_name="long_call", contracts=1, max_risk=310, premium_cost=310
        )

        executor.execute_strategy(strategy, size)
        positions = executor.get_all_positions()
        pos = positions["AAPL"]
        assert "unrealized_pnl" in pos
        assert "current_value" in pos
