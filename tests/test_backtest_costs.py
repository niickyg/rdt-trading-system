"""
Unit tests for the transaction-cost model added to EnhancedBacktestEngine.

These are fully offline (no network / no market data) and validate that
commissions and slippage are:
  1. computed correctly by the cost primitives,
  2. deducted adversely at every fill (entry, scale-out, full close),
  3. accounted for such that final capital == initial capital + net trade P&L
     (capital-conservation invariant), and
  4. a genuine drag: net-of-cost P&L is strictly worse than frictionless P&L.

Run: pytest tests/test_backtest_costs.py -v
"""

from dataclasses import dataclass
from datetime import date

import pytest

from backtesting.engine_enhanced import EnhancedBacktestEngine, EnhancedTrade


# --- Cost primitives -------------------------------------------------------

def _engine(**kw):
    defaults = dict(
        initial_capital=25_000,
        commission_per_share=0.005,
        min_commission=1.0,
        slippage_bps=5.0,
    )
    defaults.update(kw)
    return EnhancedBacktestEngine(**defaults)


def test_slippage_direction():
    e = _engine(slippage_bps=5.0)
    # Buys fill higher, sells fill lower
    assert e._apply_slippage(100.0, "buy") == pytest.approx(100.05)
    assert e._apply_slippage(100.0, "sell") == pytest.approx(99.95)


def test_slippage_disabled():
    e = _engine(slippage_bps=0.0)
    assert e._apply_slippage(100.0, "buy") == 100.0
    assert e._apply_slippage(100.0, "sell") == 100.0


def test_commission_floor_and_per_share():
    e = _engine(commission_per_share=0.005, min_commission=1.0)
    assert e._commission(10) == pytest.approx(1.0)     # 0.05 -> floored to 1.00
    assert e._commission(500) == pytest.approx(2.5)    # 500 * 0.005
    assert e._commission(0) == 0.0


def test_commission_disabled():
    e = _engine(commission_per_share=0.0, min_commission=0.0)
    assert e._commission(500) == 0.0


# --- Fill accounting -------------------------------------------------------

def _place_long(engine, symbol="AAPL", entry=100.0, shares=100, stop=95.0, target=110.0):
    """Directly seat a long position (bypasses signal logic) to test exits."""
    trade = EnhancedTrade(
        symbol=symbol, direction="long", entry_date=date(2025, 1, 2),
        entry_price=entry, shares=shares, remaining_shares=shares,
        stop_price=stop, original_stop=stop, target_price=target,
    )
    engine.positions[symbol] = trade
    # Simulate the capital that would have been spent acquiring the shares
    engine.capital -= entry * shares
    return trade


def test_close_applies_slippage_and_commission():
    e = _engine()
    start_cap = e.capital
    trade = _place_long(e, entry=100.0, shares=100, target=110.0)
    invested = 100.0 * 100

    e._close_position("AAPL", 110.0, date(2025, 1, 10), "take_profit")

    # Exit is a sell -> fills at 110 * (1 - 0.0005) = 109.945
    expected_fill = 110.0 * (1 - 0.0005)
    expected_commission = max(1.0, 100 * 0.005)  # 1.00
    gross = (110.0 - 100.0) * 100                 # 1000 frictionless
    net = (expected_fill - 100.0) * 100 - expected_commission

    assert trade.exit_price == pytest.approx(expected_fill)
    assert trade.pnl == pytest.approx(net)
    assert net < gross                            # costs are a real drag
    assert e.total_commission == pytest.approx(expected_commission)
    assert e.total_slippage_cost == pytest.approx(abs(expected_fill - 110.0) * 100)

    # Capital conservation: cash back = original outlay recovered + net P&L
    assert e.capital == pytest.approx(start_cap - invested + invested + net)
    assert e.capital == pytest.approx(start_cap + net)


def test_zero_cost_matches_frictionless():
    e = _engine(commission_per_share=0.0, min_commission=0.0, slippage_bps=0.0)
    _place_long(e, entry=100.0, shares=100, target=110.0)
    e._close_position("AAPL", 110.0, date(2025, 1, 10), "take_profit")
    trade = e.trades[0]
    assert trade.pnl == pytest.approx(1000.0)   # exactly frictionless
    assert e.total_commission == 0.0
    assert e.total_slippage_cost == 0.0


def test_entry_costs_via_stubbed_sizer():
    """_enter_position should slip the entry and charge commission."""
    e = _engine()

    @dataclass
    class _Sizing:
        shares: int
        stop_price: float
        target_price: float

    # Stub the sizer so we control share count deterministically
    e.position_sizer.calculate_position_size = lambda **kw: _Sizing(
        shares=100, stop_price=95.0, target_price=110.0
    )

    start_cap = e.capital
    e._enter_position("AAPL", "long", entry_price=100.0, atr=2.0,
                      entry_date=date(2025, 1, 2), rrs=2.5)

    trade = e.positions["AAPL"]
    # Long entry is a buy -> fills higher
    assert trade.entry_price == pytest.approx(100.0 * (1 + 0.0005))
    # Entry commission is pre-charged into trade.pnl
    assert trade.pnl == pytest.approx(-1.0)
    # Capital reduced by fill notional + commission
    assert e.capital == pytest.approx(start_cap - (trade.entry_price * 100) - 1.0)


def test_round_trip_capital_conservation_with_costs():
    """After a full round trip, final capital == start + trade.pnl exactly."""
    e = _engine()

    @dataclass
    class _Sizing:
        shares: int
        stop_price: float
        target_price: float

    e.position_sizer.calculate_position_size = lambda **kw: _Sizing(
        shares=100, stop_price=95.0, target_price=110.0
    )

    start_cap = e.capital
    e._enter_position("AAPL", "long", entry_price=100.0, atr=2.0,
                      entry_date=date(2025, 1, 2), rrs=2.5)
    e._close_position("AAPL", 110.0, date(2025, 1, 10), "take_profit")

    trade = e.trades[0]
    # Net P&L must reflect BOTH commissions (entry+exit) and slippage (both sides)
    assert trade.pnl < 1000.0
    assert e.capital == pytest.approx(start_cap + trade.pnl)
    # Two commissions charged (entry + exit)
    assert e.total_commission == pytest.approx(2.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
