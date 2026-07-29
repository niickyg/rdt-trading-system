"""Unit tests for backtesting.benchmark (pure functions, no market data needed)."""

import math
from dataclasses import dataclass

from backtesting.benchmark import (
    CostModel,
    estimate_trade_cost,
    estimate_trading_costs,
    spy_buy_and_hold,
    summarize_vs_benchmark,
)


@dataclass
class _FakeTrade:
    shares: int
    entry_price: float
    exit_price: float


def test_spy_buy_and_hold_basic():
    # 10% price gain over exactly one year (252 bars), no dividends.
    prices = [100.0] + [None] * 250 + [110.0]  # Nones ignored; endpoints matter
    r = spy_buy_and_hold(prices, 25_000, annual_dividend_yield=0.0, trading_days=252)
    assert abs(r["price_return_pct"] - 10.0) < 1e-6
    assert abs(r["total_return_pct"] - 10.0) < 1e-6
    assert abs(r["dollar_profit"] - 2_500.0) < 1e-6
    assert abs(r["annualized_pct"] - 10.0) < 1e-6


def test_spy_buy_and_hold_adds_dividends():
    prices = [100.0, 100.0]  # flat price over 2 years
    r = spy_buy_and_hold(prices, 10_000, annual_dividend_yield=0.013, trading_days=504)
    # 2 years * 1.3% ≈ 2.6% from dividends
    assert abs(r["total_return_pct"] - 2.6) < 0.05
    assert r["dollar_profit"] > 0


def test_spy_buy_and_hold_degrades_gracefully():
    assert spy_buy_and_hold([], 25_000)["dollar_profit"] == 0.0
    assert spy_buy_and_hold([100.0], 25_000)["dollar_profit"] == 0.0
    assert spy_buy_and_hold([100.0, 110.0], 0)["dollar_profit"] == 0.0


def test_estimate_trade_cost_commission_floor_and_slippage():
    model = CostModel(commission_per_share=0.005, min_commission_per_order=1.0, slippage_bps_per_side=2.5)
    # 100 shares @ $50 both sides. Commission per side = max(100*0.005, 1) = $1.
    # Slippage per side = 100*50 * 2.5/10000 = $1.25. Round trip = 2*(1 + 1.25) = $4.50
    cost = estimate_trade_cost(100, 50.0, 50.0, model)
    assert abs(cost - 4.50) < 1e-6


def test_estimate_trade_cost_zero_shares():
    assert estimate_trade_cost(0, 50.0, 50.0, CostModel()) == 0.0


def test_estimate_trading_costs_skips_open_trades():
    trades = [
        _FakeTrade(100, 50.0, 51.0),
        _FakeTrade(100, 50.0, None),  # still open — skipped
    ]
    out = estimate_trading_costs(trades)
    assert out["trades_costed"] == 1
    assert out["total_cost"] > 0


def test_summarize_beats_spy_verdict():
    # Strategy made $9,000 gross on tiny share counts (negligible cost);
    # SPY made ~$2,500. Strategy should win.
    trades = [_FakeTrade(1, 100.0, 100.0) for _ in range(5)]
    prices = [100.0, 110.0]
    out = summarize_vs_benchmark(
        strategy_gross_return=9_000.0,
        trades=trades,
        spy_prices=prices,
        initial_capital=25_000,
        trading_days=252,
    )
    assert out["beats_spy"] is True
    assert out["excess_vs_spy_dollar"] > 0


def test_summarize_loses_to_spy_verdict():
    # Strategy made $1,716 gross (the documented best), SPY made ~$8,600.
    trades = [_FakeTrade(50, 100.0, 101.0) for _ in range(279)]
    prices = [546.26, 740.86]  # actual SPY 2yr window
    out = summarize_vs_benchmark(
        strategy_gross_return=1_716.0,
        trades=trades,
        spy_prices=prices,
        initial_capital=25_000,
        trading_days=504,
    )
    assert out["beats_spy"] is False
    assert out["excess_vs_spy_dollar"] < 0
    # honest costs should eat a meaningful chunk of the $1,716
    assert out["estimated_costs"] > 0


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
