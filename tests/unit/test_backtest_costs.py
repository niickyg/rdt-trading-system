"""
Tests for the transaction-cost model added to the enhanced backtest engine.

The engine must:
  1. Compute commission (max(min_commission, shares*per_share)) + slippage
     (price*shares*bps/1e4) per fill.
  2. Default to ZERO cost so historical (gross) backtests reproduce exactly.
  3. Keep capital accounting consistent: final capital == initial + net P&L,
     where net P&L == gross P&L - total costs across entry and all exit fills.
"""
from datetime import date

from backtesting.engine_enhanced import EnhancedBacktestEngine, EnhancedTrade


def test_transaction_cost_math():
    e = EnhancedBacktestEngine(
        commission_per_share=0.005, min_commission=1.0, slippage_bps=5.0
    )
    # 100 sh @ $200: commission max($1, $0.50)=$1; slippage 200*100*0.0005=$10 => $11
    assert abs(e._transaction_cost(100, 200.0) - 11.0) < 1e-9
    # 10 sh @ $50: commission max($1, $0.05)=$1; slippage 50*10*0.0005=$0.25 => $1.25
    assert abs(e._transaction_cost(10, 50.0) - 1.25) < 1e-9
    # non-positive share counts cost nothing
    assert e._transaction_cost(0, 100.0) == 0.0
    assert e._transaction_cost(-5, 100.0) == 0.0


def test_default_engine_is_zero_cost():
    """Default construction must preserve gross reproducibility."""
    e = EnhancedBacktestEngine()
    assert e._transaction_cost(100, 200.0) == 0.0
    assert e._transaction_cost(1, 1000.0) == 0.0


def test_close_accounting_invariant_is_net_of_costs():
    e = EnhancedBacktestEngine(
        commission_per_share=0.005, min_commission=1.0, slippage_bps=5.0
    )
    e.capital = 25000.0
    e.total_costs = 0.0
    e.positions = {}
    e.trades = []

    entry_cost = e._transaction_cost(100, 200.0)  # $11
    trade = EnhancedTrade(
        symbol="X", direction="long", entry_date=date(2025, 1, 1),
        entry_price=200.0, shares=100, remaining_shares=100,
        stop_price=190, original_stop=190, target_price=220,
        total_cost=entry_cost,
    )
    e.positions["X"] = trade
    e.capital -= (200.0 * 100 + entry_cost)
    e.total_costs += entry_cost

    e._close_position("X", 210.0, date(2025, 1, 5), "target")

    gross_pnl = (210.0 - 200.0) * 100  # $1000
    exit_cost = 1.0 + 210.0 * 100 * 0.0005  # $1 commission + $10.50 slippage
    total_cost = entry_cost + exit_cost
    net_pnl = gross_pnl - total_cost

    assert abs(e.trades[0].pnl - net_pnl) < 1e-6
    assert abs(e.capital - (25000.0 + net_pnl)) < 1e-6
    assert abs(e.total_costs - total_cost) < 1e-6


def test_costs_reduce_pnl_relative_to_gross():
    """A round trip with costs must net less than the same trip gross."""
    def roundtrip_pnl(**cost_kwargs):
        e = EnhancedBacktestEngine(**cost_kwargs)
        e.capital = 25000.0
        e.total_costs = 0.0
        e.positions = {}
        e.trades = []
        entry_cost = e._transaction_cost(100, 200.0)
        t = EnhancedTrade(
            symbol="X", direction="long", entry_date=date(2025, 1, 1),
            entry_price=200.0, shares=100, remaining_shares=100,
            stop_price=190, original_stop=190, target_price=220,
            total_cost=entry_cost,
        )
        e.positions["X"] = t
        e.capital -= (200.0 * 100 + entry_cost)
        e.total_costs += entry_cost
        e._close_position("X", 210.0, date(2025, 1, 5), "target")
        return e.trades[0].pnl

    gross = roundtrip_pnl()
    net = roundtrip_pnl(commission_per_share=0.005, min_commission=1.0, slippage_bps=5.0)
    assert gross == 1000.0
    assert net < gross
