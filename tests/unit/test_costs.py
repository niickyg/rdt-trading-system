"""Unit tests for the honest transaction-cost model (backtesting/costs.py)."""

from backtesting.costs import CostModel, apply_costs, net_pnl, summarize


def test_min_commission_floor():
    m = CostModel(commission_per_share=0.005, min_commission_per_order=1.0)
    # 10 shares * $0.005 = $0.05 raw, floored to the $1.00 minimum.
    assert m.commission(price=100.0, shares=10) == 1.0


def test_commission_scales_above_minimum():
    m = CostModel(commission_per_share=0.005, min_commission_per_order=1.0,
                  max_commission_pct=1.0)
    # 1000 shares * $0.005 = $5.00, above the $1 minimum.
    assert m.commission(price=100.0, shares=1000) == 5.0


def test_commission_pct_cap():
    # Penny stock: per-share commission would dwarf trade value; cap at 1%.
    m = CostModel(commission_per_share=0.005, min_commission_per_order=0.0,
                  max_commission_pct=0.01)
    # 1000 shares @ $0.10 = $100 notional; raw comm $5 capped to $1 (1%).
    assert m.commission(price=0.10, shares=1000) == 1.0


def test_execution_friction_is_bps_of_notional():
    m = CostModel(slippage_bps=2.0, half_spread_bps=3.0)  # 5 bps total
    # $10,000 notional * 5bps = $5.00
    assert m.execution_friction(price=100.0, shares=100) == 5.0


def test_round_trip_charges_two_fills():
    m = CostModel(commission_per_share=0.0, min_commission_per_order=1.0,
                  slippage_bps=0.0, half_spread_bps=0.0)
    # Two fills, each at the $1 commission floor => $2.00 round trip.
    assert m.round_trip_cost(entry_price=50.0, exit_price=55.0, shares=10) == 2.0


def test_net_pnl_subtracts_costs():
    m = CostModel(commission_per_share=0.0, min_commission_per_order=1.0,
                  slippage_bps=0.0, half_spread_bps=0.0)
    # Gross $100, costs $2 (two $1 fills) => net $98.
    assert net_pnl(100.0, entry_price=50.0, exit_price=55.0, shares=10, model=m) == 98.0


def test_zero_and_negative_inputs_are_safe():
    m = CostModel()
    assert m.fill_cost(price=0.0, shares=100) == 0.0
    assert m.fill_cost(price=100.0, shares=0) == 0.0
    assert m.commission(price=-5.0, shares=100) == 0.0


def test_apply_and_summarize_over_trades():
    m = CostModel(commission_per_share=0.0, min_commission_per_order=1.0,
                  slippage_bps=0.0, half_spread_bps=0.0)
    trades = [
        {"entry_price": 50.0, "exit_price": 55.0, "shares": 10, "pnl": 50.0},
        {"entry_price": 20.0, "exit_price": 18.0, "shares": 5, "pnl": -10.0},
    ]
    rows = apply_costs(trades, m)
    assert rows[0]["net_pnl"] == 48.0  # 50 gross - 2 cost
    assert rows[1]["net_pnl"] == -12.0  # -10 gross - 2 cost

    s = summarize(trades, m)
    assert s["num_trades"] == 2
    assert s["gross_pnl"] == 40.0
    assert s["total_cost"] == 4.0
    assert s["net_pnl"] == 36.0


def test_costs_can_erase_a_thin_gross_edge():
    """Regression guard: a realistic day-trade cost model materially reduces
    a thin gross edge. Documents WHY gross backtest numbers overstate the edge."""
    m = CostModel()  # defaults
    # ~$2,500 position at $150/share, held for a $5 gross gain.
    cost = m.round_trip_cost(entry_price=150.0, exit_price=150.30, shares=16)
    assert cost > 2.0  # two commission floors plus friction
    net = net_pnl(5.0, entry_price=150.0, exit_price=150.30, shares=16, model=m)
    assert net < 5.0
