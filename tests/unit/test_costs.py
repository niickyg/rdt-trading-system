"""Unit tests for the backtest transaction cost model (backtesting/costs.py).

Pure stdlib — runnable without pandas/numpy:
    python3 tests/unit/test_costs.py
or under pytest:
    pytest tests/unit/test_costs.py
"""

import importlib.util
import sys
from pathlib import Path

# Load backtesting/costs.py directly by path so this test needs no pandas/numpy
# (importing the `backtesting` package would trigger its pandas-dependent engine).
_COSTS_PATH = Path(__file__).resolve().parent.parent.parent / "backtesting" / "costs.py"
_spec = importlib.util.spec_from_file_location("_bt_costs", _COSTS_PATH)
_costs = importlib.util.module_from_spec(_spec)
sys.modules["_bt_costs"] = _costs  # required for dataclass introspection
_spec.loader.exec_module(_costs)
CostModel = _costs.CostModel


def test_zero_model_is_free():
    cm = CostModel.zero()
    assert cm.is_zero()
    assert cm.leg_cost(100.0, 100) == 0.0
    assert cm.leg_cost(0.0, 0) == 0.0


def test_degenerate_legs_cost_nothing():
    cm = CostModel.ibkr_retail()
    assert cm.leg_cost(100.0, 0) == 0.0
    assert cm.leg_cost(100.0, -5) == 0.0
    assert cm.leg_cost(0.0, 10) == 0.0
    assert cm.leg_cost(-10.0, 10) == 0.0


def test_commission_minimum_applies_on_small_orders():
    cm = CostModel(commission_per_share=0.0035, min_commission_per_order=0.35, slippage_bps=0.0)
    # 10 shares * 0.0035 = 0.035 -> below the 0.35 minimum, so minimum applies
    assert abs(cm.leg_cost(50.0, 10) - 0.35) < 1e-9


def test_commission_per_share_applies_on_large_orders():
    cm = CostModel(commission_per_share=0.0035, min_commission_per_order=0.35, slippage_bps=0.0)
    # 1000 shares * 0.0035 = 3.50 -> above the minimum
    assert abs(cm.leg_cost(50.0, 1000) - 3.50) < 1e-9


def test_slippage_is_bps_of_notional():
    cm = CostModel(commission_per_share=0.0, min_commission_per_order=0.0, slippage_bps=10.0)
    # 10 bps of (100 shares * $50) = 0.001 * 5000 = $5.00
    assert abs(cm.leg_cost(50.0, 100) - 5.0) < 1e-9


def test_commission_and_slippage_combine():
    cm = CostModel.ibkr_retail()  # 0.0035/share, 0.35 min, 5 bps
    # 200 shares @ $100: commission = max(0.35, 200*0.0035=0.70) = 0.70
    #                    slippage   = 5 bps * (200*100=20000) = 0.0005*20000 = 10.00
    expected = 0.70 + 10.00
    assert abs(cm.leg_cost(100.0, 200) - expected) < 1e-9


def test_round_trip_drag_is_material_vs_thin_edge():
    """Sanity check the magnitude: a ~2.5-leg round trip on a typical position
    is a few dollars — which, times a few hundred trades, is the same order of
    magnitude as the strategy's entire reported 2-year profit."""
    cm = CostModel.ibkr_retail()
    # Representative position: ~30 shares @ $100 ($3k notional), entry + 2 exit legs
    entry = cm.leg_cost(100.0, 30)
    exit1 = cm.leg_cost(101.0, 15)
    exit2 = cm.leg_cost(102.0, 15)
    round_trip = entry + exit1 + exit2
    assert round_trip > 0
    # 279 trades * round_trip should be a meaningful fraction of ~$1,716 profit
    annualized_drag = 279 * round_trip
    assert annualized_drag > 500  # material relative to the gross edge


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
