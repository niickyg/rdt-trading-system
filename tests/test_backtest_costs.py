"""
Tests for the trading-cost model added to EnhancedBacktestEngine.

The backtest engines historically modeled ZERO trading costs (no commission,
no slippage, no spread), which made reported returns optimistic. These tests
lock in the accounting invariants of the cost model:

  1. Per-side cost = commission (with per-order minimum) + slippage in bps.
  2. A round trip is charged on BOTH entry and exit.
  3. Net trade P&L equals gross P&L minus entry and exit costs.
  4. Capital is conserved: change in engine capital == net trade P&L.
  5. Setting all cost params to 0 reproduces the legacy frictionless numbers.
"""

import math

from backtesting.engine_enhanced import EnhancedBacktestEngine, EnhancedTrade


def _simulate_round_trip(engine, entry_price, exit_price, shares, direction="long"):
    """Manually run one entry->close cycle through the engine's cost paths,
    bypassing signal/sizing so the arithmetic is fully controlled.

    Returns (trade, capital_change).
    """
    from datetime import date

    engine.capital = engine.initial_capital
    engine.total_costs = 0.0
    engine.positions = {}
    engine.trades = []

    entry_cost = engine._side_cost(entry_price, shares)
    trade = EnhancedTrade(
        symbol="TEST",
        direction=direction,
        entry_date=date(2025, 1, 2),
        entry_price=entry_price,
        shares=shares,
        remaining_shares=shares,
        stop_price=entry_price * 0.95,
        original_stop=entry_price * 0.95,
        target_price=entry_price * 1.10,
        entry_cost=entry_cost,
        costs=entry_cost,
    )
    engine.positions["TEST"] = trade
    # Mirror _enter_position capital effects.
    engine.capital -= shares * entry_price
    engine.capital -= entry_cost
    engine.total_costs += entry_cost

    capital_before_cycle = engine.initial_capital
    engine._close_position("TEST", exit_price, date(2025, 1, 10), "take_profit")
    capital_change = engine.capital - capital_before_cycle
    return trade, capital_change


def test_side_cost_formula():
    eng = EnhancedBacktestEngine(
        commission_per_share=0.01,
        min_commission_per_order=1.0,
        slippage_bps=10.0,
    )
    # 100 shares @ $100: commission = max(0.01*100, 1.0) = 1.0;
    # slippage = 100 * (10/10000) * 100 = 10.0  -> total 11.0
    assert math.isclose(eng._side_cost(100.0, 100), 11.0, rel_tol=1e-9)
    # Per-order minimum binds for tiny orders: 1 share @ $50, commission floor 1.0
    # slippage = 50 * 0.001 * 1 = 0.05 -> total 1.05
    assert math.isclose(eng._side_cost(50.0, 1), 1.05, rel_tol=1e-9)
    # Degenerate inputs cost nothing.
    assert eng._side_cost(0.0, 100) == 0.0
    assert eng._side_cost(100.0, 0) == 0.0


def test_round_trip_net_pnl_and_capital_conservation():
    eng = EnhancedBacktestEngine(
        commission_per_share=0.01,
        min_commission_per_order=1.0,
        slippage_bps=10.0,
    )
    trade, capital_change = _simulate_round_trip(
        eng, entry_price=100.0, exit_price=110.0, shares=100, direction="long"
    )
    # gross = (110-100)*100 = 1000
    # entry_cost = 11.0 ; exit_cost = max(1, 0.01*100) + 110*0.001*100 = 1 + 11 = 12.0
    # net = 1000 - 11 - 12 = 977
    assert math.isclose(trade.pnl, 977.0, rel_tol=1e-9)
    assert math.isclose(trade.costs, 23.0, rel_tol=1e-9)
    assert math.isclose(eng.total_costs, 23.0, rel_tol=1e-9)
    # Capital must be conserved: engine capital change == net trade P&L.
    assert math.isclose(capital_change, trade.pnl, rel_tol=1e-9)


def test_short_round_trip_costs_applied():
    eng = EnhancedBacktestEngine(
        commission_per_share=0.0,
        min_commission_per_order=0.0,
        slippage_bps=10.0,
    )
    trade, capital_change = _simulate_round_trip(
        eng, entry_price=100.0, exit_price=90.0, shares=100, direction="short"
    )
    # gross short = (100-90)*100 = 1000
    # entry_cost = 100*0.001*100 = 10 ; exit_cost = 90*0.001*100 = 9
    # net = 1000 - 10 - 9 = 981
    assert math.isclose(trade.pnl, 981.0, rel_tol=1e-9)
    assert math.isclose(trade.costs, 19.0, rel_tol=1e-9)
    assert math.isclose(capital_change, trade.pnl, rel_tol=1e-9)


def test_zero_costs_reproduce_frictionless():
    eng = EnhancedBacktestEngine(
        commission_per_share=0.0,
        min_commission_per_order=0.0,
        slippage_bps=0.0,
    )
    trade, capital_change = _simulate_round_trip(
        eng, entry_price=100.0, exit_price=110.0, shares=100, direction="long"
    )
    # No costs -> net == gross == 1000, matching legacy behavior.
    assert math.isclose(trade.pnl, 1000.0, rel_tol=1e-9)
    assert trade.costs == 0.0
    assert eng.total_costs == 0.0
    assert math.isclose(capital_change, 1000.0, rel_tol=1e-9)


if __name__ == "__main__":
    test_side_cost_formula()
    test_round_trip_net_pnl_and_capital_conservation()
    test_short_round_trip_costs_applied()
    test_zero_costs_reproduce_frictionless()
    print("All cost-model tests passed.")
