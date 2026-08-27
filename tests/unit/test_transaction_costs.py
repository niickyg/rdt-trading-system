"""
Tests for honest transaction-cost modeling and the SPY buy-and-hold benchmark.

These verify the two things the operator mandate requires backtests to measure
and that the historical walk-forward scripts omitted:
  1. Trades cost money (commission + slippage), which reduces net return.
  2. Strategy return can be compared against SPY buy-and-hold.

All tests are self-contained (synthetic in-memory data, no network).
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backtesting.costs import (
    TransactionCostModel,
    ZERO_COST,
    spy_buy_and_hold_return,
    annualize_return,
)


# --------------------------------------------------------------------------
# TransactionCostModel
# --------------------------------------------------------------------------

def test_commission_respects_per_order_minimum():
    m = TransactionCostModel(commission_per_share=0.005, min_commission_per_order=1.0,
                             max_commission_pct_of_notional=1.0, slippage_bps=0.0)
    # 10 shares * $0.005 = $0.05, below the $1.00 minimum
    assert m.commission(10, 100.0) == pytest.approx(1.0)
    # 1000 shares * $0.005 = $5.00, above the minimum
    assert m.commission(1000, 100.0) == pytest.approx(5.0)


def test_commission_capped_at_pct_of_notional():
    m = TransactionCostModel(commission_per_share=0.005, min_commission_per_order=1.0,
                             max_commission_pct_of_notional=0.01, slippage_bps=0.0)
    # 5 shares at $2 => notional $10, cap = $0.10; raw/min would be $1.00
    assert m.commission(5, 2.0) == pytest.approx(0.10)


def test_slippage_is_bps_of_notional_per_fill():
    m = TransactionCostModel(slippage_bps=5.0)
    # 100 shares * $50 = $5000 notional; 5 bps = 0.0005 => $2.50
    assert m.slippage(100, 50.0) == pytest.approx(2.50)


def test_fill_cost_combines_commission_and_slippage():
    m = TransactionCostModel(commission_per_share=0.005, min_commission_per_order=1.0,
                             max_commission_pct_of_notional=1.0, slippage_bps=5.0)
    # 1000 sh @ $50: commission $5.00, slippage 5bps*50000=$25.00 => $30.00
    assert m.fill_cost(1000, 50.0) == pytest.approx(30.0)


def test_zero_and_degenerate_inputs():
    m = TransactionCostModel()
    assert m.fill_cost(0, 100.0) == 0.0
    assert m.fill_cost(100, 0.0) == 0.0
    assert m.fill_cost(-100, 100.0) > 0.0  # sign-agnostic (shorts pay too)
    assert ZERO_COST.fill_cost(1000, 100.0) == 0.0


# --------------------------------------------------------------------------
# SPY buy-and-hold benchmark
# --------------------------------------------------------------------------

def _spy_frame(closes):
    start = date(2024, 1, 2)
    idx = pd.to_datetime([start + timedelta(days=i) for i in range(len(closes))])
    return pd.DataFrame({"Close": closes}, index=idx)


def test_spy_buy_and_hold_basic():
    spy = _spy_frame([100.0, 110.0, 120.0])
    assert spy_buy_and_hold_return(spy) == pytest.approx(0.20)


def test_spy_buy_and_hold_respects_window():
    spy = _spy_frame([100.0, 110.0, 120.0, 130.0])
    r = spy_buy_and_hold_return(spy, start_date=date(2024, 1, 3), end_date=date(2024, 1, 4))
    # closes on 1/3 and 1/4 are 110 and 120
    assert r == pytest.approx(120.0 / 110.0 - 1.0)


def test_spy_buy_and_hold_empty_window_returns_zero():
    spy = _spy_frame([100.0, 110.0])
    assert spy_buy_and_hold_return(spy, start_date=date(2030, 1, 1)) == 0.0


def test_lowercase_close_column_supported():
    spy = _spy_frame([100.0, 150.0])
    spy = spy.rename(columns={"Close": "close"})
    assert spy_buy_and_hold_return(spy) == pytest.approx(0.50)


# --------------------------------------------------------------------------
# annualize_return
# --------------------------------------------------------------------------

def test_annualize_return_one_year():
    assert annualize_return(0.10, 252) == pytest.approx(0.10, abs=1e-9)


def test_annualize_return_half_year_compounds_up():
    # 10% over 126 days annualizes to (1.1^2 - 1) = 21%
    assert annualize_return(0.10, 126) == pytest.approx(0.21, abs=1e-9)


def test_annualize_return_guards_zero_days():
    assert annualize_return(0.10, 0) == 0.0


# --------------------------------------------------------------------------
# Integration: costs flow through the engine's actual fill hooks
# --------------------------------------------------------------------------

def _round_trip(cost_model):
    """
    Drive one entry + one full close through the real engine fill paths and
    return (final_capital, trade_pnl, total_costs). Uses the engine's own
    _enter_position / _close_position so the cost wiring is exercised exactly
    as it is during a backtest, without depending on signal generation.
    """
    from backtesting.engine_enhanced import EnhancedBacktestEngine

    eng = EnhancedBacktestEngine(initial_capital=25000, cost_model=cost_model)
    # Reset run-state the way run() would.
    eng.capital = eng.initial_capital
    eng.positions = {}
    eng.trades = []
    eng.total_costs = 0.0

    eng._enter_position(
        symbol="AAA", direction="long", entry_price=50.0, atr=2.0,
        entry_date=date(2024, 1, 2), rrs=3.0,
    )
    assert "AAA" in eng.positions, "position sizer produced no shares"
    eng._close_position("AAA", exit_price=55.0, exit_date=date(2024, 1, 12),
                        reason="take_profit")
    trade = eng.trades[-1]
    return eng.capital, trade.pnl, eng.total_costs


def test_costs_reduce_returns_through_engine_hooks():
    free_cap, free_pnl, free_costs = _round_trip(None)
    costed_cap, costed_pnl, costed_costs = _round_trip(TransactionCostModel())

    # Zero-cost path charges nothing (legacy behavior preserved).
    assert free_costs == 0.0
    # Costed path charges entry + exit and reduces both capital and trade P&L.
    assert costed_costs > 0.0
    assert costed_cap < free_cap
    assert costed_pnl < free_pnl
    # The capital gap equals the costs charged, and the P&L was reduced by the
    # same total (entry cost + exit cost both land on trade.pnl).
    assert (free_cap - costed_cap) == pytest.approx(costed_costs, rel=1e-9)
    assert (free_pnl - costed_pnl) == pytest.approx(costed_costs, rel=1e-9)
