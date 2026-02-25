"""
Tests for options.strategy_selector.StrategySelector.

Covers the full IV-regime x direction decision table, forced strategy override,
fallback paths (no chain data, no expiry), and spread width validation.
"""

import pytest
from unittest.mock import MagicMock, patch

from options.models import (
    IVAnalysis,
    IVRegime,
    OptionAction,
    OptionContract,
    OptionGreeks,
    OptionRight,
    StrategyDirection,
)
from options.config import OptionsConfig
from options.strategy_selector import StrategySelector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPIRY = "20260320"
SYMBOL = "AAPL"
ACCOUNT_SIZE = 25_000.0

BASE_SIGNAL = {
    "symbol": SYMBOL,
    "direction": "long",
    "entry_price": 150.0,
    "atr": 3.5,
}


def _make_signal(direction: str = "long", **overrides) -> dict:
    sig = {**BASE_SIGNAL, "direction": direction}
    sig.update(overrides)
    return sig


def _make_contract(
    strike: float,
    right: OptionRight,
    expiry: str = EXPIRY,
    symbol: str = SYMBOL,
) -> OptionContract:
    return OptionContract(
        symbol=symbol,
        expiry=expiry,
        strike=strike,
        right=right,
    )


def _make_greeks(
    delta: float = 0.60,
    bid: float = 3.00,
    ask: float = 3.20,
    open_interest: int = 500,
    underlying_price: float = 150.0,
) -> OptionGreeks:
    return OptionGreeks(
        delta=delta,
        bid=bid,
        ask=ask,
        open_interest=open_interest,
        underlying_price=underlying_price,
    )


def _make_iv_analysis(iv_rank: float) -> IVAnalysis:
    return IVAnalysis(symbol=SYMBOL, iv_rank=iv_rank)


def _make_chain_result(strike: float, right: OptionRight, delta: float = 0.60,
                       bid: float = 3.00, ask: float = 3.20):
    """Return a (contract, greeks) tuple suitable for find_by_delta mock."""
    return (
        _make_contract(strike, right),
        _make_greeks(delta=delta, bid=bid, ask=ask),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chain_manager():
    mock = MagicMock()
    mock.find_target_expiry.return_value = EXPIRY
    return mock


@pytest.fixture
def iv_analyzer():
    return MagicMock()


@pytest.fixture
def config():
    """Default config with env loading disabled."""
    return OptionsConfig(
        _env_file=None,
        max_spread_width_under_100=5.0,
        max_spread_width_over_100=10.0,
    )


@pytest.fixture
def selector(chain_manager, iv_analyzer, config):
    return StrategySelector(chain_manager, iv_analyzer, config)


# ---------------------------------------------------------------------------
# 1. Decision table: LONG direction
# ---------------------------------------------------------------------------

class TestLongDirection:
    """LONG signal across all IV regimes."""

    def test_long_low_iv_returns_long_call(self, selector, chain_manager, iv_analyzer):
        """IV rank < 30 + LONG => long_call."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=20)
        chain_manager.find_by_delta.return_value = _make_chain_result(
            strike=148.0, right=OptionRight.CALL, delta=0.60, bid=5.00, ask=5.40,
        )

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is not None
        assert result.name == "long_call"
        assert result.direction == StrategyDirection.LONG
        assert len(result.legs) == 1
        assert result.legs[0].contract.right == OptionRight.CALL
        assert result.legs[0].action == OptionAction.BUY
        assert result.max_loss > 0
        assert result.net_premium < 0  # debit

    def test_long_normal_iv_returns_bull_call_spread(self, selector, chain_manager, iv_analyzer):
        """IV rank 30-50 + LONG => bull_call_spread."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=40)

        # Long leg (higher delta, lower strike) then short leg (lower delta, higher strike)
        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=148.0, right=OptionRight.CALL, delta=0.60, bid=5.00, ask=5.40),
            _make_chain_result(strike=155.0, right=OptionRight.CALL, delta=0.30, bid=2.00, ask=2.40),
        ]

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is not None
        assert result.name == "bull_call_spread"
        assert result.direction == StrategyDirection.LONG
        assert len(result.legs) == 2
        assert result.is_debit

    def test_long_high_iv_returns_bull_put_spread(self, selector, chain_manager, iv_analyzer):
        """IV rank 50-80 + LONG => bull_put_spread (credit)."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=65)

        # sell put (higher strike), buy put (lower strike)
        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=148.0, right=OptionRight.PUT, delta=-0.35, bid=3.50, ask=3.80),
            _make_chain_result(strike=142.0, right=OptionRight.PUT, delta=-0.15, bid=1.50, ask=1.80),
        ]

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is not None
        assert result.name == "bull_put_spread"
        assert result.direction == StrategyDirection.LONG
        assert len(result.legs) == 2
        assert result.is_credit

    def test_long_very_high_iv_returns_iron_condor(self, selector, chain_manager, iv_analyzer):
        """IV rank > 80 + LONG => iron_condor (direction irrelevant)."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=85)

        chain_manager.find_by_delta.side_effect = [
            # sell put 0.20 delta
            _make_chain_result(strike=140.0, right=OptionRight.PUT, delta=-0.20, bid=2.00, ask=2.20),
            # buy put 0.10 delta
            _make_chain_result(strike=135.0, right=OptionRight.PUT, delta=-0.10, bid=1.00, ask=1.20),
            # sell call 0.20 delta
            _make_chain_result(strike=160.0, right=OptionRight.CALL, delta=0.20, bid=2.00, ask=2.20),
            # buy call 0.10 delta
            _make_chain_result(strike=165.0, right=OptionRight.CALL, delta=0.10, bid=1.00, ask=1.20),
        ]

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is not None
        assert result.name == "iron_condor"
        assert result.direction == StrategyDirection.NEUTRAL
        assert len(result.legs) == 4
        assert result.is_credit


# ---------------------------------------------------------------------------
# 2. Decision table: SHORT direction
# ---------------------------------------------------------------------------

class TestShortDirection:
    """SHORT signal across all IV regimes."""

    def test_short_low_iv_returns_long_put(self, selector, chain_manager, iv_analyzer):
        """IV rank < 30 + SHORT => long_put."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=15)
        chain_manager.find_by_delta.return_value = _make_chain_result(
            strike=152.0, right=OptionRight.PUT, delta=-0.60, bid=4.80, ask=5.20,
        )

        result = selector.select_strategy(_make_signal("short"), ACCOUNT_SIZE)

        assert result is not None
        assert result.name == "long_put"
        assert result.direction == StrategyDirection.SHORT
        assert len(result.legs) == 1
        assert result.legs[0].contract.right == OptionRight.PUT
        assert result.legs[0].action == OptionAction.BUY
        assert result.net_premium < 0  # debit

    def test_short_normal_iv_returns_bear_put_spread(self, selector, chain_manager, iv_analyzer):
        """IV rank 30-50 + SHORT => bear_put_spread."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=40)

        # Long put (higher strike), short put (lower strike)
        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=152.0, right=OptionRight.PUT, delta=-0.60, bid=5.00, ask=5.40),
            _make_chain_result(strike=145.0, right=OptionRight.PUT, delta=-0.30, bid=2.00, ask=2.40),
        ]

        result = selector.select_strategy(_make_signal("short"), ACCOUNT_SIZE)

        assert result is not None
        assert result.name == "bear_put_spread"
        assert result.direction == StrategyDirection.SHORT
        assert len(result.legs) == 2
        assert result.is_debit

    def test_short_high_iv_returns_bear_call_spread(self, selector, chain_manager, iv_analyzer):
        """IV rank 50-80 + SHORT => bear_call_spread (credit)."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=65)

        # sell call (lower strike), buy call (higher strike)
        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=152.0, right=OptionRight.CALL, delta=0.35, bid=3.50, ask=3.80),
            _make_chain_result(strike=158.0, right=OptionRight.CALL, delta=0.15, bid=1.50, ask=1.80),
        ]

        result = selector.select_strategy(_make_signal("short"), ACCOUNT_SIZE)

        assert result is not None
        assert result.name == "bear_call_spread"
        assert result.direction == StrategyDirection.SHORT
        assert len(result.legs) == 2
        assert result.is_credit

    def test_short_very_high_iv_returns_iron_condor(self, selector, chain_manager, iv_analyzer):
        """IV rank > 80 + SHORT => iron_condor (same as long)."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=90)

        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=140.0, right=OptionRight.PUT, delta=-0.20, bid=2.00, ask=2.20),
            _make_chain_result(strike=135.0, right=OptionRight.PUT, delta=-0.10, bid=1.00, ask=1.20),
            _make_chain_result(strike=160.0, right=OptionRight.CALL, delta=0.20, bid=2.00, ask=2.20),
            _make_chain_result(strike=165.0, right=OptionRight.CALL, delta=0.10, bid=1.00, ask=1.20),
        ]

        result = selector.select_strategy(_make_signal("short"), ACCOUNT_SIZE)

        assert result is not None
        assert result.name == "iron_condor"
        assert result.direction == StrategyDirection.NEUTRAL
        assert len(result.legs) == 4


# ---------------------------------------------------------------------------
# 3. Forced strategy override
# ---------------------------------------------------------------------------

class TestForcedStrategy:
    """Config.force_strategy bypasses the decision table."""

    def test_forced_long_call_ignores_iv(self, chain_manager, iv_analyzer):
        """force_strategy='long_call' builds a long call regardless of IV."""
        cfg = OptionsConfig(_env_file=None, OPTIONS_FORCE_STRATEGY="long_call")
        sel = StrategySelector(chain_manager, iv_analyzer, cfg)

        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=85)  # very high
        chain_manager.find_by_delta.return_value = _make_chain_result(
            strike=148.0, right=OptionRight.CALL, delta=0.60, bid=5.00, ask=5.40,
        )

        result = sel.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is not None
        assert result.name == "long_call"

    def test_forced_bear_put_spread(self, chain_manager, iv_analyzer):
        """force_strategy='bear_put_spread' builds that strategy."""
        cfg = OptionsConfig(_env_file=None, OPTIONS_FORCE_STRATEGY="bear_put_spread")
        sel = StrategySelector(chain_manager, iv_analyzer, cfg)

        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=10)
        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=152.0, right=OptionRight.PUT, delta=-0.60, bid=5.00, ask=5.40),
            _make_chain_result(strike=145.0, right=OptionRight.PUT, delta=-0.30, bid=2.00, ask=2.40),
        ]

        result = sel.select_strategy(_make_signal("short"), ACCOUNT_SIZE)

        assert result is not None
        assert result.name == "bear_put_spread"

    def test_forced_iron_condor(self, chain_manager, iv_analyzer):
        """force_strategy='iron_condor' builds an iron condor."""
        cfg = OptionsConfig(_env_file=None, OPTIONS_FORCE_STRATEGY="iron_condor")
        sel = StrategySelector(chain_manager, iv_analyzer, cfg)

        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=10)
        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=140.0, right=OptionRight.PUT, delta=-0.20, bid=2.00, ask=2.20),
            _make_chain_result(strike=135.0, right=OptionRight.PUT, delta=-0.10, bid=1.00, ask=1.20),
            _make_chain_result(strike=160.0, right=OptionRight.CALL, delta=0.20, bid=2.00, ask=2.20),
            _make_chain_result(strike=165.0, right=OptionRight.CALL, delta=0.10, bid=1.00, ask=1.20),
        ]

        result = sel.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is not None
        assert result.name == "iron_condor"

    def test_forced_unknown_strategy_returns_none(self, chain_manager, iv_analyzer):
        """Unknown force_strategy returns None."""
        cfg = OptionsConfig(_env_file=None, OPTIONS_FORCE_STRATEGY="butterfly_spread")
        sel = StrategySelector(chain_manager, iv_analyzer, cfg)

        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=50)

        result = sel.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is None

    def test_forced_strategy_no_expiry_returns_none(self, chain_manager, iv_analyzer):
        """Forced strategy returns None when no expiry is found."""
        cfg = OptionsConfig(_env_file=None, OPTIONS_FORCE_STRATEGY="long_call")
        sel = StrategySelector(chain_manager, iv_analyzer, cfg)

        chain_manager.find_target_expiry.return_value = None

        result = sel.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is None


# ---------------------------------------------------------------------------
# 4. Fallback: chain manager returns no data
# ---------------------------------------------------------------------------

class TestChainFallback:
    """find_by_delta returning None causes strategy to return None."""

    def test_long_call_no_chain_data(self, selector, chain_manager, iv_analyzer):
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=20)
        chain_manager.find_by_delta.return_value = None

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is None

    def test_long_put_no_chain_data(self, selector, chain_manager, iv_analyzer):
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=15)
        chain_manager.find_by_delta.return_value = None

        result = selector.select_strategy(_make_signal("short"), ACCOUNT_SIZE)

        assert result is None

    def test_bull_call_spread_long_leg_missing(self, selector, chain_manager, iv_analyzer):
        """First find_by_delta call returns None."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=40)
        chain_manager.find_by_delta.return_value = None

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is None

    def test_bull_call_spread_short_leg_missing(self, selector, chain_manager, iv_analyzer):
        """First call succeeds, second returns None."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=40)
        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=148.0, right=OptionRight.CALL, delta=0.60, bid=5.00, ask=5.40),
            None,
        ]

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is None

    def test_bear_call_spread_sell_leg_missing(self, selector, chain_manager, iv_analyzer):
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=65)
        chain_manager.find_by_delta.return_value = None

        result = selector.select_strategy(_make_signal("short"), ACCOUNT_SIZE)

        assert result is None

    def test_iron_condor_partial_legs_missing(self, selector, chain_manager, iv_analyzer):
        """Iron condor returns None when one of four legs is missing."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=85)
        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=140.0, right=OptionRight.PUT, delta=-0.20, bid=2.00, ask=2.20),
            None,  # buy put missing
            _make_chain_result(strike=160.0, right=OptionRight.CALL, delta=0.20, bid=2.00, ask=2.20),
            _make_chain_result(strike=165.0, right=OptionRight.CALL, delta=0.10, bid=1.00, ask=1.20),
        ]

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is None


# ---------------------------------------------------------------------------
# 5. Expiry selection failure
# ---------------------------------------------------------------------------

class TestExpiryFailure:
    """No expiry found => returns None."""

    def test_no_expiry_long(self, selector, chain_manager, iv_analyzer):
        chain_manager.find_target_expiry.return_value = None
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=20)

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is None

    def test_no_expiry_short(self, selector, chain_manager, iv_analyzer):
        chain_manager.find_target_expiry.return_value = None
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=40)

        result = selector.select_strategy(_make_signal("short"), ACCOUNT_SIZE)

        assert result is None

    def test_no_expiry_very_high_iv(self, selector, chain_manager, iv_analyzer):
        """Even very-high IV returns None when no expiry exists."""
        chain_manager.find_target_expiry.return_value = None
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=90)

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is None


# ---------------------------------------------------------------------------
# 6. Spread width validation
# ---------------------------------------------------------------------------

class TestSpreadWidthValidation:
    """Invalid spread widths (zero, negative, too wide) are rejected."""

    def test_bull_call_spread_zero_width_rejected(self, selector, chain_manager, iv_analyzer):
        """Spread with same strike on both legs => width 0 => None."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=40)

        # Both legs at same strike
        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=150.0, right=OptionRight.CALL, delta=0.60, bid=5.00, ask=5.40),
            _make_chain_result(strike=150.0, right=OptionRight.CALL, delta=0.30, bid=2.00, ask=2.40),
        ]

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is None

    def test_bull_call_spread_negative_width_rejected(self, selector, chain_manager, iv_analyzer):
        """Short strike < long strike => negative width => None."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=40)

        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=155.0, right=OptionRight.CALL, delta=0.60, bid=5.00, ask=5.40),
            _make_chain_result(strike=148.0, right=OptionRight.CALL, delta=0.30, bid=2.00, ask=2.40),
        ]

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is None

    def test_bull_call_spread_too_wide_rejected(self, selector, chain_manager, iv_analyzer):
        """Spread wider than config max_spread_width_over_100 => None."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=40)

        # entry_price=150 => max_spread_width_over_100 = 10.0
        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=148.0, right=OptionRight.CALL, delta=0.60, bid=5.00, ask=5.40),
            _make_chain_result(strike=165.0, right=OptionRight.CALL, delta=0.30, bid=2.00, ask=2.40),
        ]

        result = selector.select_strategy(
            _make_signal("long", entry_price=150.0), ACCOUNT_SIZE
        )

        assert result is None  # width 17 > max 10

    def test_bear_put_spread_inverted_strikes_rejected(self, selector, chain_manager, iv_analyzer):
        """Bear put spread: long strike < short strike => negative width => None."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=40)

        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=145.0, right=OptionRight.PUT, delta=-0.60, bid=5.00, ask=5.40),
            _make_chain_result(strike=152.0, right=OptionRight.PUT, delta=-0.30, bid=2.00, ask=2.40),
        ]

        result = selector.select_strategy(_make_signal("short"), ACCOUNT_SIZE)

        assert result is None

    def test_bear_call_spread_inverted_strikes_rejected(self, selector, chain_manager, iv_analyzer):
        """Bear call spread: buy strike < sell strike => negative width => None."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=65)

        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=155.0, right=OptionRight.CALL, delta=0.35, bid=3.50, ask=3.80),
            _make_chain_result(strike=150.0, right=OptionRight.CALL, delta=0.15, bid=1.50, ask=1.80),
        ]

        result = selector.select_strategy(_make_signal("short"), ACCOUNT_SIZE)

        assert result is None

    def test_bull_put_spread_inverted_strikes_rejected(self, selector, chain_manager, iv_analyzer):
        """Bull put spread: sell strike < buy strike => negative width => None."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=65)

        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=140.0, right=OptionRight.PUT, delta=-0.35, bid=1.50, ask=1.80),
            _make_chain_result(strike=148.0, right=OptionRight.PUT, delta=-0.15, bid=3.50, ask=3.80),
        ]

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is None

    def test_iron_condor_invalid_put_wing_rejected(self, selector, chain_manager, iv_analyzer):
        """Iron condor with inverted put wing => None."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=85)

        chain_manager.find_by_delta.side_effect = [
            # sell put at strike LOWER than buy put => negative put width
            _make_chain_result(strike=135.0, right=OptionRight.PUT, delta=-0.20, bid=2.00, ask=2.20),
            _make_chain_result(strike=140.0, right=OptionRight.PUT, delta=-0.10, bid=1.00, ask=1.20),
            _make_chain_result(strike=160.0, right=OptionRight.CALL, delta=0.20, bid=2.00, ask=2.20),
            _make_chain_result(strike=165.0, right=OptionRight.CALL, delta=0.10, bid=1.00, ask=1.20),
        ]

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is None

    def test_spread_width_uses_under_100_limit(self, chain_manager, iv_analyzer):
        """Stock under $100 uses max_spread_width_under_100 (default 5)."""
        cfg = OptionsConfig(
            _env_file=None,
            max_spread_width_under_100=5.0,
            max_spread_width_over_100=10.0,
        )
        sel = StrategySelector(chain_manager, iv_analyzer, cfg)

        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=40)

        # Width of 7 is valid for over-100 but exceeds under-100 limit of 5
        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=48.0, right=OptionRight.CALL, delta=0.60, bid=5.00, ask=5.40),
            _make_chain_result(strike=55.0, right=OptionRight.CALL, delta=0.30, bid=2.00, ask=2.40),
        ]

        result = sel.select_strategy(
            _make_signal("long", entry_price=50.0), ACCOUNT_SIZE
        )

        assert result is None  # width 7 > max 5 for under-100 stocks


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_missing_symbol_returns_none(self, selector):
        result = selector.select_strategy(
            {"direction": "long", "entry_price": 150.0}, ACCOUNT_SIZE
        )
        assert result is None

    def test_empty_symbol_returns_none(self, selector):
        result = selector.select_strategy(
            {"symbol": "", "direction": "long", "entry_price": 150.0}, ACCOUNT_SIZE
        )
        assert result is None

    def test_direction_defaults_to_long(self, selector, chain_manager, iv_analyzer):
        """Missing direction key defaults to 'long'."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=20)
        chain_manager.find_by_delta.return_value = _make_chain_result(
            strike=148.0, right=OptionRight.CALL, delta=0.60, bid=5.00, ask=5.40,
        )

        result = selector.select_strategy(
            {"symbol": SYMBOL, "entry_price": 150.0}, ACCOUNT_SIZE
        )

        assert result is not None
        assert result.name == "long_call"

    def test_iv_rank_boundary_30_is_normal(self, selector, chain_manager, iv_analyzer):
        """IV rank exactly 30 is LOW (regime requires > 30 for NORMAL)."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=30)
        chain_manager.find_by_delta.return_value = _make_chain_result(
            strike=148.0, right=OptionRight.CALL, delta=0.60, bid=5.00, ask=5.40,
        )

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is not None
        assert result.name == "long_call"  # LOW regime => long_call

    def test_iv_rank_boundary_50_is_high(self, selector, chain_manager, iv_analyzer):
        """IV rank exactly 50 is NORMAL (regime requires > 50 for HIGH)."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=50)

        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=148.0, right=OptionRight.CALL, delta=0.60, bid=5.00, ask=5.40),
            _make_chain_result(strike=155.0, right=OptionRight.CALL, delta=0.30, bid=2.00, ask=2.40),
        ]

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is not None
        assert result.name == "bull_call_spread"  # NORMAL regime

    def test_iv_rank_boundary_80_is_very_high(self, selector, chain_manager, iv_analyzer):
        """IV rank exactly 80 is HIGH (regime requires > 80 for VERY_HIGH)."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=80)

        # HIGH regime + long => bull_put_spread
        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=148.0, right=OptionRight.PUT, delta=-0.35, bid=3.50, ask=3.80),
            _make_chain_result(strike=142.0, right=OptionRight.PUT, delta=-0.15, bid=1.50, ask=1.80),
        ]

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is not None
        assert result.name == "bull_put_spread"  # HIGH regime, not VERY_HIGH

    def test_strategy_math_long_call(self, selector, chain_manager, iv_analyzer):
        """Verify premium and breakeven calculations for long call."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=20)
        chain_manager.find_by_delta.return_value = _make_chain_result(
            strike=150.0, right=OptionRight.CALL, delta=0.60, bid=4.00, ask=4.40,
        )

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is not None
        mid = (4.00 + 4.40) / 2  # 4.20
        expected_premium = mid * 100  # 420.0
        assert result.max_loss == pytest.approx(expected_premium)
        assert result.breakeven == [pytest.approx(150.0 + mid)]
        assert result.net_premium == pytest.approx(-expected_premium)

    def test_strategy_math_bull_call_spread(self, selector, chain_manager, iv_analyzer):
        """Verify debit and max_profit for bull call spread."""
        iv_analyzer.analyze.return_value = _make_iv_analysis(iv_rank=40)

        chain_manager.find_by_delta.side_effect = [
            _make_chain_result(strike=148.0, right=OptionRight.CALL, delta=0.60, bid=5.00, ask=5.40),
            _make_chain_result(strike=155.0, right=OptionRight.CALL, delta=0.30, bid=2.00, ask=2.40),
        ]

        result = selector.select_strategy(_make_signal("long"), ACCOUNT_SIZE)

        assert result is not None
        long_mid = (5.00 + 5.40) / 2   # 5.20
        short_mid = (2.00 + 2.40) / 2  # 2.20
        net_debit = (long_mid - short_mid) * 100  # 300.0
        spread_width = 155.0 - 148.0    # 7.0
        max_profit = (spread_width * 100) - net_debit  # 400.0

        assert result.max_loss == pytest.approx(net_debit)
        assert result.max_profit == pytest.approx(max_profit)
