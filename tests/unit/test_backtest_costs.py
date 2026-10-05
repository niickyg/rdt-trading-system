"""
Tests for honest transaction-cost modeling and SPY buy-and-hold benchmarking
in the backtest engine.

Two layers:
  1. Pure-stdlib tests of ``backtesting.costs.CostModel`` (no pandas needed).
  2. Engine integration tests on deterministic synthetic data (no network):
     costs must reduce realized P&L, and every result must carry a SPY
     buy-and-hold benchmark plus alpha.

Rationale: the engine historically filled every order at the exact quoted
price with zero commission and never compared against buy-and-hold, which
overstated the strategy's edge. These tests lock in honest accounting.
"""

import pytest

from backtesting.costs import CostModel, ZERO_COST


# --------------------------------------------------------------------------- #
# Pure cost-model tests (stdlib only)
# --------------------------------------------------------------------------- #

class TestCostModel:
    def test_commission_min_floor(self):
        # 10 shares * $0.005 = $0.05, floored at the $1.00 per-order minimum.
        m = CostModel()
        assert m.commission(10, 100.0) == pytest.approx(1.0)

    def test_commission_per_share_above_floor(self):
        # 1000 shares * $0.005 = $5.00, below the 1% notional cap ($1000).
        m = CostModel()
        assert m.commission(1000, 100.0) == pytest.approx(5.0)

    def test_commission_notional_cap(self):
        # Penny stock: per-share cost would dwarf 1% of notional, so it is
        # capped, but never below the $1 floor.
        m = CostModel()
        # 5000 sh * $0.005 = $25; 1% of (5000*$0.10=$500) = $5 cap -> $5.
        assert m.commission(5000, 0.10) == pytest.approx(5.0)

    def test_commission_zero_shares(self):
        assert CostModel().commission(0, 100.0) == 0.0

    def test_zero_cost_model_is_free(self):
        assert ZERO_COST.commission(1000, 100.0) == 0.0
        assert ZERO_COST.fill_price(100.0, is_buy=True) == 100.0
        assert ZERO_COST.fill_price(100.0, is_buy=False, is_stop=True) == 100.0

    def test_fill_price_buy_is_adverse_higher(self):
        m = CostModel(slippage_bps=10.0, stop_slippage_bps=0.0)
        assert m.fill_price(100.0, is_buy=True) == pytest.approx(100.10)

    def test_fill_price_sell_is_adverse_lower(self):
        m = CostModel(slippage_bps=10.0, stop_slippage_bps=0.0)
        assert m.fill_price(100.0, is_buy=False) == pytest.approx(99.90)

    def test_stop_exit_adds_extra_slippage(self):
        m = CostModel(slippage_bps=2.0, stop_slippage_bps=5.0)
        # Selling on a stop: (2 + 5) bps adverse -> 0.07% lower.
        assert m.fill_price(100.0, is_buy=False, is_stop=True) == pytest.approx(99.93)
        # Non-stop sell: only 2 bps.
        assert m.fill_price(100.0, is_buy=False, is_stop=False) == pytest.approx(99.98)


# --------------------------------------------------------------------------- #
# Engine integration tests (synthetic data, no network)
# --------------------------------------------------------------------------- #

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _make_df(dates, start, daily_ret, vol=0.0015, seed=42):
    rng = np.random.default_rng(seed)
    px = start
    o, h, l, c = [], [], [], []
    for _ in range(len(dates)):
        prev = px
        px = prev * (1 + daily_ret + rng.normal(0, vol))
        op = prev
        hi = max(op, px) * (1 + abs(rng.normal(0, vol)))
        lo = min(op, px) * (1 - abs(rng.normal(0, vol)))
        o.append(op); h.append(hi); l.append(lo); c.append(px)
    return pd.DataFrame(
        {"Open": o, "High": h, "Low": l, "Close": c, "Volume": [1_000_000] * len(dates)},
        index=dates,
    )


@pytest.fixture
def synthetic_market():
    dates = pd.bdate_range("2024-01-01", periods=45)
    spy = _make_df(dates, 500.0, 0.0002)          # ~flat benchmark
    stock = _make_df(dates, 100.0, 0.020)          # steady uptrend -> RS longs
    return dates, spy, {"UPTREND": stock}


def _run(cost_model, synthetic_market):
    from backtesting.engine import BacktestEngine
    dates, spy, stock_data = synthetic_market
    eng = BacktestEngine(
        initial_capital=25000, rrs_threshold=0.9, max_positions=3, cost_model=cost_model
    )
    return eng.run(stock_data, spy, start_date=dates[0].date(), end_date=dates[-1].date())


def test_engine_generates_trades(synthetic_market):
    r = _run(ZERO_COST, synthetic_market)
    assert r.total_trades > 0


def test_costs_reduce_net_pnl(synthetic_market):
    gross = _run(ZERO_COST, synthetic_market)
    net = _run(CostModel(), synthetic_market)
    assert gross.total_costs == pytest.approx(0.0)
    assert net.total_costs > 0
    assert net.total_commission > 0
    assert net.total_slippage > 0
    # Honest costs can only hurt realized P&L.
    assert net.total_return < gross.total_return


def test_benchmark_matches_independent_calc(synthetic_market):
    dates, spy, _ = synthetic_market
    expected = (spy["Close"].iloc[-1] / spy["Close"].iloc[0] - 1) * 100
    r = _run(CostModel(), synthetic_market)
    assert r.benchmark_return_pct == pytest.approx(expected, abs=1e-6)
    assert r.benchmark_return == pytest.approx(25000 * expected / 100, abs=1e-6)


def test_alpha_is_return_minus_benchmark(synthetic_market):
    r = _run(CostModel(), synthetic_market)
    assert r.alpha_vs_benchmark_pct == pytest.approx(
        r.total_return_pct - r.benchmark_return_pct, abs=1e-9
    )
    assert r.beats_benchmark == (r.total_return_pct > r.benchmark_return_pct)


def test_default_engine_uses_costs(synthetic_market):
    """An engine constructed without an explicit cost_model must still charge
    costs -- honesty is the default, not an opt-in."""
    from backtesting.engine import BacktestEngine
    dates, spy, stock_data = synthetic_market
    eng = BacktestEngine(initial_capital=25000, rrs_threshold=0.9, max_positions=3)
    r = eng.run(stock_data, spy, start_date=dates[0].date(), end_date=dates[-1].date())
    if r.total_trades > 0:
        assert r.total_costs > 0


def test_result_to_dict_exposes_new_fields(synthetic_market):
    r = _run(CostModel(), synthetic_market)
    d = r.to_dict()
    for key in (
        "total_costs", "total_commission", "total_slippage",
        "benchmark_return", "benchmark_return_pct",
        "alpha_vs_benchmark_pct", "beats_benchmark",
    ):
        assert key in d
