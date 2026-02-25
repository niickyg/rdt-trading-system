"""
Comprehensive tests for options/exit_manager.py

Tests all 6 exit triggers, priority ordering, boundary conditions,
and graceful error handling.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from options.exit_manager import OptionsExitManager, ExitSignal
from options.models import (
    OptionContract, OptionGreeks, OptionLeg, OptionsStrategy,
    OptionRight, OptionAction, StrategyDirection,
)
from options.config import OptionsConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_contract(symbol="AAPL", dte_days=30, strike=150.0, right=OptionRight.CALL):
    """Create an OptionContract with an expiry `dte_days` from today."""
    expiry_date = datetime.now().date() + timedelta(days=dte_days)
    return OptionContract(
        symbol=symbol,
        expiry=expiry_date.strftime("%Y%m%d"),
        strike=strike,
        right=right,
    )


def _make_strategy(
    name="bull_call_spread",
    dte_days=30,
    max_profit=500.0,
    max_loss=500.0,
    is_credit=False,
    symbol="AAPL",
    strike_long=150.0,
    strike_short=155.0,
):
    """Build a two-leg spread strategy."""
    contract_long = _make_contract(symbol=symbol, dte_days=dte_days, strike=strike_long)
    contract_short = _make_contract(symbol=symbol, dte_days=dte_days, strike=strike_short)
    legs = [
        OptionLeg(contract=contract_long, action=OptionAction.BUY, quantity=1),
        OptionLeg(contract=contract_short, action=OptionAction.SELL, quantity=1),
    ]
    net_premium = 1.50 if is_credit else -1.50
    return OptionsStrategy(
        name=name,
        underlying=symbol,
        direction=StrategyDirection.LONG,
        legs=legs,
        max_profit=max_profit,
        max_loss=max_loss,
        net_premium=net_premium,
    )


def _make_long_option_strategy(name="long_call", dte_days=30, symbol="AAPL"):
    """Build a single-leg long option strategy."""
    contract = _make_contract(symbol=symbol, dte_days=dte_days)
    legs = [OptionLeg(contract=contract, action=OptionAction.BUY, quantity=1)]
    return OptionsStrategy(
        name=name,
        underlying=symbol,
        direction=StrategyDirection.LONG,
        legs=legs,
        max_profit=0,
        max_loss=350.0,
        net_premium=-3.50,
    )


def _make_position(strategy, entry_premium=3.50, entry_iv=0.30, entry_delta=0.55, contracts=1):
    """Build the position dict expected by check_exits."""
    return {
        "strategy": strategy,
        "contracts": contracts,
        "entry_premium": entry_premium,
        "entry_iv": entry_iv,
        "entry_delta": entry_delta,
        "entry_time": datetime.now(),
    }


def _build_manager():
    """Return (OptionsExitManager, mock_chain_manager)."""
    mock_chain = MagicMock()
    config = OptionsConfig(
        OPTIONS_ENABLED=True,
        OPTIONS_MODE="options",
    )
    manager = OptionsExitManager(chain_manager=mock_chain, config=config)
    return manager, mock_chain


def _patch_current_value(manager, current_premium, net_delta=0.50, avg_iv=0.30):
    """Patch _get_current_strategy_value to return controlled values."""
    greeks_dict = {"net_delta": net_delta, "avg_iv": avg_iv}
    manager._get_current_strategy_value = MagicMock(
        return_value=(current_premium, greeks_dict)
    )


# ---------------------------------------------------------------------------
# ExitSignal basic tests
# ---------------------------------------------------------------------------

class TestExitSignal:
    def test_repr(self):
        sig = ExitSignal("AAPL", "test reason", priority=1)
        assert "AAPL" in repr(sig)
        assert "test reason" in repr(sig)
        assert "priority=1" in repr(sig)

    def test_default_action_is_close(self):
        sig = ExitSignal("AAPL", "reason", priority=1)
        assert sig.action == "close"

    def test_custom_action(self):
        sig = ExitSignal("AAPL", "roll it", priority=6, action="roll")
        assert sig.action == "roll"

    def test_timestamp_set(self):
        before = datetime.now()
        sig = ExitSignal("AAPL", "reason", priority=1)
        after = datetime.now()
        assert before <= sig.timestamp <= after


# ---------------------------------------------------------------------------
# 1. Profit target tests
# ---------------------------------------------------------------------------

class TestProfitTarget:
    """Priority 1: profit target."""

    # -- Long options: 100% gain --

    def test_long_call_profit_target_fires(self):
        """100% gain on a long call triggers exit."""
        manager, _ = _build_manager()
        strategy = _make_long_option_strategy("long_call")
        position = _make_position(strategy, entry_premium=3.50)
        # current_premium = 7.00 → (7.00 - 3.50) / 3.50 = 100% gain
        _patch_current_value(manager, current_premium=7.00)

        signals = manager.check_exits({"AAPL": position})
        profit_signals = [s for s in signals if s.priority == 1]
        assert len(profit_signals) == 1
        assert "Profit target" in profit_signals[0].reason

    def test_long_put_profit_target_fires(self):
        """100% gain on a long put triggers exit."""
        manager, _ = _build_manager()
        strategy = _make_long_option_strategy("long_put")
        position = _make_position(strategy, entry_premium=2.00)
        # current = 4.00 → 100% gain
        _patch_current_value(manager, current_premium=4.00)

        signals = manager.check_exits({"AAPL": position})
        profit_signals = [s for s in signals if s.priority == 1]
        assert len(profit_signals) == 1

    def test_long_call_just_below_target_does_not_fire(self):
        """99% gain should NOT trigger (target is 100%)."""
        manager, _ = _build_manager()
        strategy = _make_long_option_strategy("long_call")
        position = _make_position(strategy, entry_premium=3.50)
        # 99% gain → 3.50 * 1.99 = 6.965
        _patch_current_value(manager, current_premium=6.965)

        signals = manager.check_exits({"AAPL": position})
        profit_signals = [s for s in signals if s.priority == 1]
        assert len(profit_signals) == 0

    def test_long_option_above_target_fires(self):
        """150% gain (well above 100%) should still trigger."""
        manager, _ = _build_manager()
        strategy = _make_long_option_strategy("long_call")
        position = _make_position(strategy, entry_premium=2.00)
        _patch_current_value(manager, current_premium=5.00)  # 150% gain

        signals = manager.check_exits({"AAPL": position})
        profit_signals = [s for s in signals if s.priority == 1]
        assert len(profit_signals) == 1

    # -- Spreads: 50% of max profit --

    def test_spread_profit_target_fires(self):
        """Debit spread reaching 50% of max profit triggers exit."""
        manager, _ = _build_manager()
        strategy = _make_strategy(
            name="bull_call_spread", max_profit=500.0, max_loss=500.0, is_credit=False,
        )
        position = _make_position(strategy, entry_premium=3.50, contracts=1)
        # Debit spread PnL = (current - entry) * multiplier * contracts
        # Need PnL / max_profit >= 0.50
        # 500 * 0.50 = 250 needed. (entry=3.50, multiplier=100, contracts=1)
        # current_premium = 3.50 + 2.50 = 6.00
        _patch_current_value(manager, current_premium=6.00)

        signals = manager.check_exits({"AAPL": position})
        profit_signals = [s for s in signals if s.priority == 1]
        assert len(profit_signals) == 1
        assert "50%" in profit_signals[0].reason

    def test_spread_just_below_profit_target_does_not_fire(self):
        """Just under 50% of max profit for a spread should NOT trigger."""
        manager, _ = _build_manager()
        strategy = _make_strategy(
            name="bull_call_spread", max_profit=500.0, max_loss=500.0, is_credit=False,
        )
        position = _make_position(strategy, entry_premium=3.50, contracts=1)
        # Need PnL < 250. current = 3.50 + 2.49 = 5.99 → PnL = 249
        _patch_current_value(manager, current_premium=5.99)

        signals = manager.check_exits({"AAPL": position})
        profit_signals = [s for s in signals if s.priority == 1]
        assert len(profit_signals) == 0

    def test_credit_spread_profit_target_fires(self):
        """Credit spread: profit when premium decreases."""
        manager, _ = _build_manager()
        strategy = _make_strategy(
            name="bear_call_spread", max_profit=300.0, max_loss=200.0, is_credit=True,
        )
        position = _make_position(strategy, entry_premium=3.00, contracts=1)
        # Credit PnL = (entry - current) * multiplier * contracts
        # Need PnL / max_profit >= 0.50 → (3.00 - current)*100 / 300 >= 0.50
        # 150 = (3.00 - current)*100 → current = 1.50
        _patch_current_value(manager, current_premium=1.50)

        signals = manager.check_exits({"AAPL": position})
        profit_signals = [s for s in signals if s.priority == 1]
        assert len(profit_signals) == 1

    def test_long_option_zero_entry_premium_no_crash(self):
        """Zero entry premium should not cause division by zero."""
        manager, _ = _build_manager()
        strategy = _make_long_option_strategy("long_call")
        position = _make_position(strategy, entry_premium=0.0)
        _patch_current_value(manager, current_premium=5.00)

        signals = manager.check_exits({"AAPL": position})
        profit_signals = [s for s in signals if s.priority == 1]
        assert len(profit_signals) == 0


# ---------------------------------------------------------------------------
# 2. Stop loss tests
# ---------------------------------------------------------------------------

class TestStopLoss:
    """Priority 2: stop loss at 50% premium loss."""

    def test_long_call_stop_loss_fires(self):
        """50% premium loss on a long call triggers stop."""
        manager, _ = _build_manager()
        strategy = _make_long_option_strategy("long_call")
        position = _make_position(strategy, entry_premium=4.00)
        # loss = (4.00 - 2.00) / 4.00 = 50%
        _patch_current_value(manager, current_premium=2.00)

        signals = manager.check_exits({"AAPL": position})
        stop_signals = [s for s in signals if s.priority == 2]
        assert len(stop_signals) == 1
        assert "Stop loss" in stop_signals[0].reason

    def test_long_call_stop_loss_just_below_does_not_fire(self):
        """49% loss should NOT trigger stop."""
        manager, _ = _build_manager()
        strategy = _make_long_option_strategy("long_call")
        position = _make_position(strategy, entry_premium=4.00)
        # loss = (4.00 - 2.04) / 4.00 = 49% → no trigger
        _patch_current_value(manager, current_premium=2.04)

        signals = manager.check_exits({"AAPL": position})
        stop_signals = [s for s in signals if s.priority == 2]
        assert len(stop_signals) == 0

    def test_spread_stop_loss_fires(self):
        """Spread hitting 50% of max loss triggers stop."""
        manager, _ = _build_manager()
        strategy = _make_strategy(
            name="bull_call_spread", max_profit=500.0, max_loss=500.0, is_credit=False,
        )
        position = _make_position(strategy, entry_premium=5.00, contracts=1)
        # Debit PnL = (current - entry) * 100 * 1 = (2.50 - 5.00)*100 = -250
        # |PnL| / max_loss = 250/500 = 0.50 and PnL < 0
        _patch_current_value(manager, current_premium=2.50)

        signals = manager.check_exits({"AAPL": position})
        stop_signals = [s for s in signals if s.priority == 2]
        assert len(stop_signals) == 1

    def test_spread_stop_loss_just_below_does_not_fire(self):
        """Just under 50% of max loss for a spread should NOT trigger."""
        manager, _ = _build_manager()
        strategy = _make_strategy(
            name="bull_call_spread", max_profit=500.0, max_loss=500.0, is_credit=False,
        )
        position = _make_position(strategy, entry_premium=5.00, contracts=1)
        # PnL = (2.51 - 5.00)*100 = -249 → 249/500 = 0.498
        _patch_current_value(manager, current_premium=2.51)

        signals = manager.check_exits({"AAPL": position})
        stop_signals = [s for s in signals if s.priority == 2]
        assert len(stop_signals) == 0

    def test_long_put_stop_loss_fires(self):
        """Long put with 50% loss triggers stop."""
        manager, _ = _build_manager()
        strategy = _make_long_option_strategy("long_put")
        position = _make_position(strategy, entry_premium=3.00)
        _patch_current_value(manager, current_premium=1.50)  # 50% loss

        signals = manager.check_exits({"AAPL": position})
        stop_signals = [s for s in signals if s.priority == 2]
        assert len(stop_signals) == 1


# ---------------------------------------------------------------------------
# 3. Time stop tests
# ---------------------------------------------------------------------------

class TestTimeStop:
    """Priority 3: DTE < 14."""

    def test_dte_13_triggers(self):
        manager, _ = _build_manager()
        strategy = _make_strategy(dte_days=13)
        position = _make_position(strategy)
        _patch_current_value(manager, current_premium=3.50)

        signals = manager.check_exits({"AAPL": position})
        time_signals = [s for s in signals if s.priority == 3]
        assert len(time_signals) == 1
        assert "Time stop" in time_signals[0].reason

    def test_dte_14_does_not_trigger(self):
        """DTE == 14 is NOT below threshold (config time_stop_dte = 14)."""
        manager, _ = _build_manager()
        strategy = _make_strategy(dte_days=14)
        position = _make_position(strategy)
        _patch_current_value(manager, current_premium=3.50)

        signals = manager.check_exits({"AAPL": position})
        time_signals = [s for s in signals if s.priority == 3]
        assert len(time_signals) == 0

    def test_dte_15_does_not_trigger(self):
        manager, _ = _build_manager()
        strategy = _make_strategy(dte_days=15)
        position = _make_position(strategy)
        _patch_current_value(manager, current_premium=3.50)

        signals = manager.check_exits({"AAPL": position})
        time_signals = [s for s in signals if s.priority == 3]
        assert len(time_signals) == 0

    def test_dte_0_triggers(self):
        """Expiry today triggers time stop."""
        manager, _ = _build_manager()
        strategy = _make_strategy(dte_days=0)
        position = _make_position(strategy)
        _patch_current_value(manager, current_premium=3.50)

        signals = manager.check_exits({"AAPL": position})
        time_signals = [s for s in signals if s.priority == 3]
        assert len(time_signals) == 1

    def test_no_legs_no_time_signal(self):
        """Strategy with no legs returns no time signal."""
        manager, _ = _build_manager()
        strategy = OptionsStrategy(
            name="empty", underlying="AAPL", direction=StrategyDirection.LONG,
            legs=[], max_profit=0, max_loss=0,
        )
        position = _make_position(strategy)
        _patch_current_value(manager, current_premium=3.50)

        signals = manager.check_exits({"AAPL": position})
        time_signals = [s for s in signals if s.priority == 3]
        assert len(time_signals) == 0


# ---------------------------------------------------------------------------
# 4. Delta breach tests
# ---------------------------------------------------------------------------

class TestDeltaBreach:
    """Priority 4: |net delta| > 0.80."""

    def test_positive_delta_above_threshold_fires(self):
        manager, _ = _build_manager()
        strategy = _make_strategy()
        position = _make_position(strategy)
        _patch_current_value(manager, current_premium=3.50, net_delta=0.81)

        signals = manager.check_exits({"AAPL": position})
        delta_signals = [s for s in signals if s.priority == 4]
        assert len(delta_signals) == 1
        assert "Delta breach" in delta_signals[0].reason

    def test_negative_delta_above_threshold_fires(self):
        """Negative delta with |delta| > 0.80 should also trigger."""
        manager, _ = _build_manager()
        strategy = _make_strategy()
        position = _make_position(strategy)
        _patch_current_value(manager, current_premium=3.50, net_delta=-0.85)

        signals = manager.check_exits({"AAPL": position})
        delta_signals = [s for s in signals if s.priority == 4]
        assert len(delta_signals) == 1

    def test_delta_exactly_at_threshold_does_not_fire(self):
        """Exactly 0.80 should NOT trigger (strictly greater than)."""
        manager, _ = _build_manager()
        strategy = _make_strategy()
        position = _make_position(strategy)
        _patch_current_value(manager, current_premium=3.50, net_delta=0.80)

        signals = manager.check_exits({"AAPL": position})
        delta_signals = [s for s in signals if s.priority == 4]
        assert len(delta_signals) == 0

    def test_delta_below_threshold_does_not_fire(self):
        manager, _ = _build_manager()
        strategy = _make_strategy()
        position = _make_position(strategy)
        _patch_current_value(manager, current_premium=3.50, net_delta=0.50)

        signals = manager.check_exits({"AAPL": position})
        delta_signals = [s for s in signals if s.priority == 4]
        assert len(delta_signals) == 0


# ---------------------------------------------------------------------------
# 5. IV crush tests
# ---------------------------------------------------------------------------

class TestIVCrush:
    """Priority 5: IV drops > 20% from entry."""

    def test_iv_crush_fires(self):
        """IV dropping 25% from entry triggers exit."""
        manager, _ = _build_manager()
        strategy = _make_strategy()
        position = _make_position(strategy, entry_iv=0.40)
        # current IV = 0.30 → drop = (0.40-0.30)/0.40 = 25%
        _patch_current_value(manager, current_premium=3.50, avg_iv=0.30)

        signals = manager.check_exits({"AAPL": position})
        iv_signals = [s for s in signals if s.priority == 5]
        assert len(iv_signals) == 1
        assert "IV crush" in iv_signals[0].reason

    def test_iv_crush_just_above_threshold_fires(self):
        """IV drop slightly above 20% triggers exit (>= comparison)."""
        manager, _ = _build_manager()
        strategy = _make_strategy()
        # entry_iv=0.40, current=0.319 → drop = (0.40-0.319)/0.40 = 20.25%
        position = _make_position(strategy, entry_iv=0.40)
        _patch_current_value(manager, current_premium=3.50, avg_iv=0.319)

        signals = manager.check_exits({"AAPL": position})
        iv_signals = [s for s in signals if s.priority == 5]
        assert len(iv_signals) == 1

    def test_iv_crush_just_below_does_not_fire(self):
        """19% drop should NOT trigger."""
        manager, _ = _build_manager()
        strategy = _make_strategy()
        position = _make_position(strategy, entry_iv=0.50)
        # current IV = 0.405 → drop = (0.50-0.405)/0.50 = 19%
        _patch_current_value(manager, current_premium=3.50, avg_iv=0.405)

        signals = manager.check_exits({"AAPL": position})
        iv_signals = [s for s in signals if s.priority == 5]
        assert len(iv_signals) == 0

    def test_iv_crush_zero_entry_iv_no_crash(self):
        """Zero entry IV should not crash."""
        manager, _ = _build_manager()
        strategy = _make_strategy()
        position = _make_position(strategy, entry_iv=0.0)
        _patch_current_value(manager, current_premium=3.50, avg_iv=0.20)

        signals = manager.check_exits({"AAPL": position})
        iv_signals = [s for s in signals if s.priority == 5]
        assert len(iv_signals) == 0

    def test_iv_crush_zero_current_iv_no_crash(self):
        """Zero current IV should not crash."""
        manager, _ = _build_manager()
        strategy = _make_strategy()
        position = _make_position(strategy, entry_iv=0.30)
        _patch_current_value(manager, current_premium=3.50, avg_iv=0.0)

        signals = manager.check_exits({"AAPL": position})
        iv_signals = [s for s in signals if s.priority == 5]
        # current_iv <= 0 guard returns None
        assert len(iv_signals) == 0

    def test_iv_increase_does_not_fire(self):
        """IV going up should never trigger IV crush."""
        manager, _ = _build_manager()
        strategy = _make_strategy()
        position = _make_position(strategy, entry_iv=0.30)
        _patch_current_value(manager, current_premium=3.50, avg_iv=0.50)

        signals = manager.check_exits({"AAPL": position})
        iv_signals = [s for s in signals if s.priority == 5]
        assert len(iv_signals) == 0


# ---------------------------------------------------------------------------
# 6. Roll trigger tests
# ---------------------------------------------------------------------------

class TestRollTrigger:
    """Priority 6: DTE < 21 AND profitable → action="roll"."""

    def test_roll_fires_when_profitable_and_near_expiry(self):
        """DTE=20, current > entry → roll signal."""
        manager, _ = _build_manager()
        strategy = _make_strategy(dte_days=20)
        position = _make_position(strategy, entry_premium=3.00)
        _patch_current_value(manager, current_premium=4.00)

        signals = manager.check_exits({"AAPL": position})
        roll_signals = [s for s in signals if s.priority == 6]
        assert len(roll_signals) == 1
        assert roll_signals[0].action == "roll"
        assert "Roll" in roll_signals[0].reason

    def test_roll_does_not_fire_when_not_profitable(self):
        """DTE < 21 but losing → no roll."""
        manager, _ = _build_manager()
        strategy = _make_strategy(dte_days=15)
        position = _make_position(strategy, entry_premium=3.00)
        _patch_current_value(manager, current_premium=2.50)

        signals = manager.check_exits({"AAPL": position})
        roll_signals = [s for s in signals if s.priority == 6]
        assert len(roll_signals) == 0

    def test_roll_does_not_fire_when_dte_at_threshold(self):
        """DTE == 21 is NOT below threshold → no roll."""
        manager, _ = _build_manager()
        strategy = _make_strategy(dte_days=21)
        position = _make_position(strategy, entry_premium=3.00)
        _patch_current_value(manager, current_premium=4.00)

        signals = manager.check_exits({"AAPL": position})
        roll_signals = [s for s in signals if s.priority == 6]
        assert len(roll_signals) == 0

    def test_roll_does_not_fire_when_dte_above_threshold(self):
        """DTE = 30 (well above 21) → no roll even if profitable."""
        manager, _ = _build_manager()
        strategy = _make_strategy(dte_days=30)
        position = _make_position(strategy, entry_premium=3.00)
        _patch_current_value(manager, current_premium=5.00)

        signals = manager.check_exits({"AAPL": position})
        roll_signals = [s for s in signals if s.priority == 6]
        assert len(roll_signals) == 0

    def test_roll_does_not_fire_when_breakeven(self):
        """current == entry (breakeven) is NOT profitable → no roll."""
        manager, _ = _build_manager()
        strategy = _make_strategy(dte_days=15)
        position = _make_position(strategy, entry_premium=3.00)
        _patch_current_value(manager, current_premium=3.00)

        signals = manager.check_exits({"AAPL": position})
        roll_signals = [s for s in signals if s.priority == 6]
        assert len(roll_signals) == 0


# ---------------------------------------------------------------------------
# Priority ordering tests
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    """Verify signals are returned sorted by priority (ascending)."""

    def test_multiple_triggers_sorted_by_priority(self):
        """When several triggers fire, they come back sorted 1..N."""
        manager, _ = _build_manager()
        # Strategy with DTE=10 (triggers time stop 3 AND roll 6 if profitable)
        strategy = _make_long_option_strategy("long_call", dte_days=10)
        position = _make_position(strategy, entry_premium=2.00, entry_iv=0.50)
        # current=4.00 → 100% gain (profit target, priority 1)
        # net_delta=0.90 (delta breach, priority 4)
        # entry_iv=0.50, avg_iv=0.35 → 30% drop (IV crush, priority 5)
        _patch_current_value(manager, current_premium=4.00, net_delta=0.90, avg_iv=0.35)

        signals = manager.check_exits({"AAPL": position})

        assert len(signals) >= 3
        priorities = [s.priority for s in signals]
        assert priorities == sorted(priorities), "Signals must be sorted by priority ascending"

    def test_profit_target_is_highest_priority(self):
        """Profit target always has priority 1."""
        manager, _ = _build_manager()
        strategy = _make_long_option_strategy("long_call")
        position = _make_position(strategy, entry_premium=2.00)
        _patch_current_value(manager, current_premium=4.00)

        signals = manager.check_exits({"AAPL": position})
        assert signals[0].priority == 1

    def test_stop_loss_priority_is_2(self):
        manager, _ = _build_manager()
        strategy = _make_long_option_strategy("long_call")
        position = _make_position(strategy, entry_premium=4.00)
        _patch_current_value(manager, current_premium=2.00)

        signals = manager.check_exits({"AAPL": position})
        stop_signals = [s for s in signals if s.priority == 2]
        assert len(stop_signals) == 1

    def test_all_six_priorities_correct(self):
        """Verify each trigger produces the correct priority number."""
        manager, _ = _build_manager()
        # Trigger all 6 at once using a long_call with DTE=10
        strategy = _make_long_option_strategy("long_call", dte_days=10)
        position = _make_position(strategy, entry_premium=1.00, entry_iv=0.50)
        # current=2.00 → 100% gain (priority 1)
        # Not a stop loss because it's a gain
        # DTE=10 → time stop (priority 3)
        # net_delta=0.90 → delta breach (priority 4)
        # avg_iv=0.35 → 30% IV drop (priority 5)
        # DTE=10 < 21 and current > entry → roll (priority 6)
        _patch_current_value(manager, current_premium=2.00, net_delta=0.90, avg_iv=0.35)

        signals = manager.check_exits({"AAPL": position})
        priorities = {s.priority for s in signals}

        assert 1 in priorities, "Profit target (1) should fire"
        assert 3 in priorities, "Time stop (3) should fire"
        assert 4 in priorities, "Delta breach (4) should fire"
        assert 5 in priorities, "IV crush (5) should fire"
        assert 6 in priorities, "Roll trigger (6) should fire"
        # Stop loss (2) should NOT fire since position is profitable
        assert 2 not in priorities


# ---------------------------------------------------------------------------
# Multiple triggers on same position
# ---------------------------------------------------------------------------

class TestMultipleTriggers:
    def test_time_stop_and_roll_fire_together(self):
        """DTE=10 triggers both time stop (3) and roll (6) when profitable."""
        manager, _ = _build_manager()
        strategy = _make_strategy(dte_days=10)
        position = _make_position(strategy, entry_premium=3.00)
        _patch_current_value(manager, current_premium=4.00)

        signals = manager.check_exits({"AAPL": position})
        priorities = {s.priority for s in signals}
        assert 3 in priorities
        assert 6 in priorities

    def test_stop_loss_and_delta_breach_together(self):
        """Losing position with high delta fires both."""
        manager, _ = _build_manager()
        strategy = _make_long_option_strategy("long_call")
        position = _make_position(strategy, entry_premium=4.00)
        _patch_current_value(manager, current_premium=2.00, net_delta=0.95)

        signals = manager.check_exits({"AAPL": position})
        priorities = {s.priority for s in signals}
        assert 2 in priorities  # stop loss
        assert 4 in priorities  # delta breach

    def test_signals_from_multiple_positions(self):
        """check_exits handles multiple positions, signals sorted globally."""
        manager, _ = _build_manager()

        strat1 = _make_long_option_strategy("long_call", symbol="AAPL")
        strat2 = _make_long_option_strategy("long_put", symbol="MSFT")

        pos1 = _make_position(strat1, entry_premium=2.00)
        pos2 = _make_position(strat2, entry_premium=3.00)

        # Position 1: profit target (priority 1)
        # Position 2: stop loss (priority 2)
        manager._get_current_strategy_value = MagicMock(
            side_effect=[
                (4.00, {"net_delta": 0.50, "avg_iv": 0.30}),  # AAPL: 100% gain
                (1.50, {"net_delta": 0.50, "avg_iv": 0.30}),  # MSFT: 50% loss
            ]
        )

        signals = manager.check_exits({"AAPL": pos1, "MSFT": pos2})
        assert len(signals) >= 2
        # Globally sorted by priority
        priorities = [s.priority for s in signals]
        assert priorities == sorted(priorities)


# ---------------------------------------------------------------------------
# get_greeks returns None (graceful handling)
# ---------------------------------------------------------------------------

class TestGetGreeksNone:
    """When chain_manager.get_greeks returns None, no signals are generated."""

    def test_get_current_value_returns_none(self):
        """If _get_current_strategy_value returns None, position is skipped."""
        manager, _ = _build_manager()
        strategy = _make_long_option_strategy("long_call")
        position = _make_position(strategy, entry_premium=2.00)
        manager._get_current_strategy_value = MagicMock(return_value=None)

        signals = manager.check_exits({"AAPL": position})
        assert len(signals) == 0

    def test_chain_get_greeks_returns_none_for_one_leg(self):
        """If get_greeks returns None for any leg, _get_current_strategy_value returns None."""
        manager, mock_chain = _build_manager()
        strategy = _make_strategy()
        position = _make_position(strategy, entry_premium=3.00)
        mock_chain.get_greeks.return_value = None

        signals = manager.check_exits({"AAPL": position})
        assert len(signals) == 0

    def test_chain_get_greeks_returns_none_for_second_leg(self):
        """First leg has greeks, second returns None → still returns None overall."""
        manager, mock_chain = _build_manager()
        strategy = _make_strategy()
        position = _make_position(strategy, entry_premium=3.00)

        greeks_ok = OptionGreeks(
            delta=0.55, implied_vol=0.30, bid=3.00, ask=3.20, option_price=3.10,
        )
        mock_chain.get_greeks.side_effect = [greeks_ok, None]

        signals = manager.check_exits({"AAPL": position})
        assert len(signals) == 0

    def test_exception_in_check_position_is_caught(self):
        """If _check_position raises, check_exits continues with other positions."""
        manager, _ = _build_manager()
        strat1 = _make_long_option_strategy("long_call", symbol="AAPL")
        strat2 = _make_long_option_strategy("long_call", symbol="MSFT")
        pos1 = _make_position(strat1, entry_premium=2.00)
        pos2 = _make_position(strat2, entry_premium=2.00)

        call_count = 0

        def side_effect(strategy):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")
            return (4.00, {"net_delta": 0.50, "avg_iv": 0.30})

        manager._get_current_strategy_value = MagicMock(side_effect=side_effect)

        signals = manager.check_exits({"AAPL": pos1, "MSFT": pos2})
        # One position errored, but the other should produce signals
        assert any(s.symbol == "MSFT" for s in signals) or any(s.symbol == "AAPL" for s in signals)


# ---------------------------------------------------------------------------
# _get_current_strategy_value integration-style tests
# ---------------------------------------------------------------------------

class TestGetCurrentStrategyValue:
    """Test the real _get_current_strategy_value (not mocked)."""

    def test_buy_leg_adds_premium(self):
        manager, mock_chain = _build_manager()
        contract = _make_contract()
        strategy = OptionsStrategy(
            name="long_call", underlying="AAPL",
            direction=StrategyDirection.LONG,
            legs=[OptionLeg(contract=contract, action=OptionAction.BUY, quantity=1)],
        )
        mock_chain.get_greeks.return_value = OptionGreeks(
            delta=0.55, implied_vol=0.30, bid=3.00, ask=3.20, option_price=3.10,
        )

        result = manager._get_current_strategy_value(strategy)
        assert result is not None
        premium, greeks = result
        assert premium == pytest.approx(3.10, abs=0.01)  # mid = (3.00+3.20)/2
        assert greeks["net_delta"] == pytest.approx(0.55, abs=0.01)
        assert greeks["avg_iv"] == pytest.approx(0.30, abs=0.01)

    def test_sell_leg_subtracts_premium(self):
        manager, mock_chain = _build_manager()
        contract = _make_contract()
        strategy = OptionsStrategy(
            name="covered_call", underlying="AAPL",
            direction=StrategyDirection.SHORT,
            legs=[OptionLeg(contract=contract, action=OptionAction.SELL, quantity=1)],
        )
        mock_chain.get_greeks.return_value = OptionGreeks(
            delta=0.40, implied_vol=0.25, bid=2.00, ask=2.20, option_price=2.10,
        )

        result = manager._get_current_strategy_value(strategy)
        assert result is not None
        premium, greeks = result
        # SELL leg: total_premium -= mid → -2.10
        assert premium == pytest.approx(-2.10, abs=0.01)
        # SELL leg: delta * qty * -1
        assert greeks["net_delta"] == pytest.approx(-0.40, abs=0.01)

    def test_spread_net_premium_and_delta(self):
        """Two legs: BUY and SELL. Net premium and delta computed correctly."""
        manager, mock_chain = _build_manager()
        strategy = _make_strategy()

        buy_greeks = OptionGreeks(
            delta=0.60, implied_vol=0.30, bid=5.00, ask=5.20, option_price=5.10,
        )
        sell_greeks = OptionGreeks(
            delta=0.35, implied_vol=0.28, bid=2.00, ask=2.20, option_price=2.10,
        )
        mock_chain.get_greeks.side_effect = [buy_greeks, sell_greeks]

        result = manager._get_current_strategy_value(strategy)
        assert result is not None
        premium, greeks = result
        # premium = +5.10 (buy, we hold it) - 2.10 (sell, closing costs) = 3.00
        assert premium == pytest.approx(3.00, abs=0.01)
        # net_delta = 0.60*1*1 + 0.35*1*(-1) = 0.25
        assert greeks["net_delta"] == pytest.approx(0.25, abs=0.01)
        # avg_iv = (0.30 + 0.28) / 2 = 0.29
        assert greeks["avg_iv"] == pytest.approx(0.29, abs=0.01)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_positions_dict(self):
        manager, _ = _build_manager()
        signals = manager.check_exits({})
        assert signals == []

    def test_strategy_with_no_expiry(self):
        """Strategy with empty legs list has no expiry → no time/roll signals."""
        manager, _ = _build_manager()
        strategy = OptionsStrategy(
            name="long_call", underlying="AAPL",
            direction=StrategyDirection.LONG, legs=[],
        )
        position = _make_position(strategy)
        _patch_current_value(manager, current_premium=3.50)

        signals = manager.check_exits({"AAPL": position})
        time_signals = [s for s in signals if s.priority in (3, 6)]
        assert len(time_signals) == 0

    def test_multiple_contracts_scale_pnl(self):
        """contracts > 1 scales PnL for spread profit target."""
        manager, _ = _build_manager()
        strategy = _make_strategy(
            name="bull_call_spread", max_profit=500.0, max_loss=500.0, is_credit=False,
        )
        # 2 contracts → max_profit = 500 * 2 = 1000
        # Need PnL >= 500. PnL = (current - entry) * 100 * 2
        # (entry=3.50, contracts=2): need (current-3.50)*200 >= 500 → current >= 6.00
        position = _make_position(strategy, entry_premium=3.50, contracts=2)
        _patch_current_value(manager, current_premium=6.00)

        signals = manager.check_exits({"AAPL": position})
        profit_signals = [s for s in signals if s.priority == 1]
        assert len(profit_signals) == 1

    def test_signal_symbol_matches_position_key(self):
        """Signal symbol should match the position key."""
        manager, _ = _build_manager()
        strategy = _make_long_option_strategy("long_call", symbol="TSLA")
        position = _make_position(strategy, entry_premium=2.00)
        _patch_current_value(manager, current_premium=4.00)

        signals = manager.check_exits({"TSLA": position})
        assert all(s.symbol == "TSLA" for s in signals)
