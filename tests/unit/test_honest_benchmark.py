"""Unit tests for the honest benchmark harness pure logic.

These need no network and no market data — they lock in the cost model, the SPY
buy-and-hold calculation, and the PASS/FAIL verdict logic that the operator
mandate relies on.
"""

import importlib.util
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

# Load scripts/honest_benchmark.py by file path and register it in sys.modules.
# Importing via `from scripts.honest_benchmark import ...` would pull in the
# project's root package __init__ chain (config, utils, ...); the harness's pure
# functions have no such dependencies, so we load the module in isolation.
# Registering in sys.modules is required for the module's dataclasses to resolve
# their (PEP 563 string) annotations.
_HB_PATH = Path(__file__).resolve().parents[2] / "scripts" / "honest_benchmark.py"
_spec = importlib.util.spec_from_file_location("honest_benchmark", _HB_PATH)
honest_benchmark = importlib.util.module_from_spec(_spec)
sys.modules["honest_benchmark"] = honest_benchmark
_spec.loader.exec_module(honest_benchmark)

round_trip_cost = honest_benchmark.round_trip_cost
spy_buy_and_hold_pct = honest_benchmark.spy_buy_and_hold_pct
make_verdict = honest_benchmark.make_verdict
total_costs_for_trades = honest_benchmark.total_costs_for_trades


# --------------------------------------------------------------------------
# round_trip_cost
# --------------------------------------------------------------------------

def test_round_trip_cost_uses_min_commission_for_small_orders():
    # 10 shares * $0.005 = $0.05 < $1 min, so commission floored at $1/leg = $2.
    # slippage: 5bps * (10*100 + 10*110) = 0.0005 * 2100 = $1.05
    cost = round_trip_cost(10, 100.0, 110.0)
    assert cost == pytest.approx(2.0 + 1.05, rel=1e-9)


def test_round_trip_cost_scales_commission_for_large_orders():
    # 1000 shares * $0.005 = $5 > $1 min, so $5/leg = $10 commission.
    # slippage: 5bps * (1000*50 + 1000*52) = 0.0005 * 102000 = $51
    cost = round_trip_cost(1000, 50.0, 52.0)
    assert cost == pytest.approx(10.0 + 51.0, rel=1e-9)


def test_round_trip_cost_zero_shares_is_free():
    assert round_trip_cost(0, 100.0, 100.0) == 0.0


def test_round_trip_cost_handles_short_negative_shares():
    # abs() should make a short (negative shares) cost the same as the long.
    assert round_trip_cost(-100, 20.0, 18.0) == round_trip_cost(100, 20.0, 18.0)


# --------------------------------------------------------------------------
# spy_buy_and_hold_pct
# --------------------------------------------------------------------------

def _spy_df(prices, start="2024-01-01"):
    idx = pd.date_range(start=start, periods=len(prices), freq="D")
    return pd.DataFrame({"Close": prices}, index=idx)


def test_spy_buy_and_hold_basic():
    df = _spy_df([100.0, 105.0, 110.0])
    pct = spy_buy_and_hold_pct(df, date(2024, 1, 1), date(2024, 1, 3))
    assert pct == pytest.approx(10.0)  # 100 -> 110


def test_spy_buy_and_hold_negative():
    df = _spy_df([200.0, 180.0])
    pct = spy_buy_and_hold_pct(df, date(2024, 1, 1), date(2024, 1, 2))
    assert pct == pytest.approx(-10.0)


def test_spy_buy_and_hold_insufficient_data_returns_none():
    df = _spy_df([100.0])
    assert spy_buy_and_hold_pct(df, date(2024, 1, 1), date(2024, 1, 1)) is None


def test_spy_buy_and_hold_lowercase_close_column():
    idx = pd.date_range(start="2024-01-01", periods=2, freq="D")
    df = pd.DataFrame({"close": [100.0, 120.0]}, index=idx)
    assert spy_buy_and_hold_pct(df, date(2024, 1, 1), date(2024, 1, 2)) == pytest.approx(20.0)


# --------------------------------------------------------------------------
# make_verdict
# --------------------------------------------------------------------------

def test_verdict_pass_requires_beating_spy_and_enough_trades():
    # Strategy net +12% on $25k with 40 trades, SPY +8% -> PASS.
    v = make_verdict(
        strategy_gross_return_dollars=3000.0,   # 12%
        total_cost_dollars=0.0,
        initial_capital=25000.0,
        spy_net_pct=8.0,
        num_trades=40,
    )
    assert v.strategy_net_pct == pytest.approx(12.0)
    assert v.edge_pct == pytest.approx(4.0)
    assert v.passes is True


def test_verdict_fails_when_underperforming_spy():
    # Strategy +6.9% (the bot's self-reported best) vs SPY +25% -> FAIL.
    v = make_verdict(
        strategy_gross_return_dollars=1716.0,   # 6.86%
        total_cost_dollars=0.0,
        initial_capital=25000.0,
        spy_net_pct=25.0,
        num_trades=279,
    )
    assert v.passes is False
    assert v.edge_pct < 0


def test_verdict_fails_on_thin_sample_even_if_ahead():
    # Beats SPY but only 2 trades -> cannot establish edge -> FAIL.
    v = make_verdict(
        strategy_gross_return_dollars=5000.0,   # 20%
        total_cost_dollars=0.0,
        initial_capital=25000.0,
        spy_net_pct=10.0,
        num_trades=2,
    )
    assert v.passes is False
    assert "Sample too small" in v.note


def test_verdict_costs_reduce_net_return():
    # $2000 gross but $500 costs on $25k -> net 6%.
    v = make_verdict(
        strategy_gross_return_dollars=2000.0,
        total_cost_dollars=500.0,
        initial_capital=25000.0,
        spy_net_pct=5.0,
        num_trades=50,
    )
    assert v.strategy_gross_pct == pytest.approx(8.0)
    assert v.strategy_cost_pct == pytest.approx(2.0)
    assert v.strategy_net_pct == pytest.approx(6.0)
    assert v.passes is True  # 6% net > 5% SPY, 50 trades


def test_verdict_no_spy_benchmark_is_fail():
    v = make_verdict(
        strategy_gross_return_dollars=1000.0,
        total_cost_dollars=0.0,
        initial_capital=25000.0,
        spy_net_pct=None,
        num_trades=100,
    )
    assert v.passes is False
    assert math.isnan(v.edge_pct)


def test_verdict_rejects_nonpositive_capital():
    with pytest.raises(ValueError):
        make_verdict(1.0, 0.0, 0.0, 5.0, 50)


# --------------------------------------------------------------------------
# total_costs_for_trades
# --------------------------------------------------------------------------

@dataclass
class _FakeTrade:
    shares: float
    entry_price: float
    exit_price: float = None


def test_total_costs_skips_open_trades():
    trades = [
        _FakeTrade(100, 50.0, 55.0),   # counted
        _FakeTrade(100, 50.0, None),   # open, skipped
    ]
    expected = round_trip_cost(100, 50.0, 55.0)
    assert total_costs_for_trades(trades) == pytest.approx(expected)
