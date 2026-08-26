"""
Honest transaction-cost modeling for backtests.

WHY THIS EXISTS
---------------
As of 2026-08-26, NONE of the backtest harnesses in this repo
(``scripts/run_walkforward.py``, ``scripts/run_walkforward_v2.py``,
``backtesting/engine*.py``, ``research/backtest_harness.py``) modeled
commissions, slippage, or bid-ask spread. Every profitability figure the
system has ever reported — including the walk-forward table quoted in
CLAUDE.md — is therefore a GROSS number. A day-trading strategy that takes
hundreds of round-trips a year on a $25K account can hand a large fraction of
its gross edge back to costs, so gross numbers systematically overstate the
edge and can flip a "profitable" strategy to a losing one.

This module provides a small, defensible, dependency-free cost model so that
every future profitability claim can be stated NET of honest costs. It is pure
arithmetic (no network, no pandas) and is covered by ``tests/unit/test_costs.py``.

DEFAULTS
--------
Defaults approximate IBKR Pro tiered US-equity pricing plus conservative
execution friction for a retail day-trading account:

* ``commission_per_share`` = $0.005/share (IBKR tiered)
* ``min_commission_per_order`` = $1.00/order (IBKR minimum)
* ``max_commission_pct`` = 1.0% of trade value (IBKR cap)
* ``slippage_bps`` = 2.0 bps per fill (market impact / non-instant fill)
* ``half_spread_bps`` = 3.0 bps per fill (you cross half the quoted spread)

Every fill (one entry OR one exit) pays commission + slippage + half-spread.
A round trip pays for two fills. Tune the parameters per instrument liquidity;
the point is that the number is no longer ZERO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class CostModel:
    """Per-fill transaction cost model. All bps are basis points of notional."""

    commission_per_share: float = 0.005
    min_commission_per_order: float = 1.0
    max_commission_pct: float = 0.01  # cap commission at 1% of trade value (IBKR rule)
    slippage_bps: float = 2.0
    half_spread_bps: float = 3.0

    def commission(self, price: float, shares: float) -> float:
        """Commission for a single fill, honoring the min-per-order and pct cap."""
        shares = abs(shares)
        if shares <= 0 or price <= 0:
            return 0.0
        raw = self.commission_per_share * shares
        capped = min(raw if raw > 0 else 0.0, self.max_commission_pct * price * shares)
        return max(self.min_commission_per_order, capped)

    def execution_friction(self, price: float, shares: float) -> float:
        """Slippage + half-spread cost for a single fill, in dollars."""
        shares = abs(shares)
        if shares <= 0 or price <= 0:
            return 0.0
        notional = price * shares
        bps = (self.slippage_bps + self.half_spread_bps) / 10_000.0
        return notional * bps

    def fill_cost(self, price: float, shares: float) -> float:
        """Total cost of a single fill (one entry OR one exit)."""
        return self.commission(price, shares) + self.execution_friction(price, shares)

    def round_trip_cost(self, entry_price: float, exit_price: float, shares: float) -> float:
        """Total cost of a completed round trip (entry fill + exit fill)."""
        return self.fill_cost(entry_price, shares) + self.fill_cost(exit_price, shares)


# A conservative-but-not-punitive default instance for convenience.
DEFAULT_COST_MODEL = CostModel()


def net_pnl(gross_pnl: float, entry_price: float, exit_price: float,
            shares: float, model: CostModel = DEFAULT_COST_MODEL) -> float:
    """Gross P&L for one round trip, minus honest round-trip costs."""
    return gross_pnl - model.round_trip_cost(entry_price, exit_price, shares)


def apply_costs(trades: Iterable, model: CostModel = DEFAULT_COST_MODEL) -> List[dict]:
    """
    Given an iterable of trade objects, return a list of dicts with gross P&L,
    round-trip cost, and net P&L for each. Works with any object exposing
    ``entry_price``, ``shares`` and one of (``pnl`` / ``gross_pnl``) plus an
    exit price under ``exit_price`` / ``exit`` / ``fill_price`` (falls back to
    entry_price when no exit price is available, which still charges the two
    commissions + friction — the dominant term for small accounts).
    """
    out: List[dict] = []
    for t in trades:
        entry = _getattr(t, ("entry_price", "entry"), 0.0)
        exitp = _getattr(t, ("exit_price", "exit", "fill_price"), entry)
        shares = _getattr(t, ("shares", "quantity", "size"), 0.0)
        gross = _getattr(t, ("pnl", "gross_pnl", "profit"), 0.0)
        cost = model.round_trip_cost(entry, exitp, shares)
        out.append({
            "gross_pnl": gross,
            "cost": cost,
            "net_pnl": gross - cost,
        })
    return out


def summarize(trades: Iterable, model: CostModel = DEFAULT_COST_MODEL) -> dict:
    """Aggregate gross P&L, total cost, and net P&L across all trades."""
    rows = apply_costs(trades, model)
    gross = sum(r["gross_pnl"] for r in rows)
    cost = sum(r["cost"] for r in rows)
    return {
        "num_trades": len(rows),
        "gross_pnl": gross,
        "total_cost": cost,
        "net_pnl": gross - cost,
        "cost_drag_pct_of_gross": (cost / gross * 100.0) if gross else float("inf"),
    }


def _getattr(obj, names, default):
    for n in names:
        if isinstance(obj, dict):
            if n in obj:
                return obj[n]
        elif hasattr(obj, n):
            return getattr(obj, n)
    return default


if __name__ == "__main__":
    # Reproducible illustration of the cost drag on the CLAUDE.md walk-forward
    # "RDT filters" result: gross +$1,716 over 279 trades on a $25K account.
    # We do not have per-trade prices here, so we approximate each trade as a
    # round trip on a ~$2,500 position (10% of a $25K account) at ~$150/share.
    model = DEFAULT_COST_MODEL
    n_trades = 279
    approx_price = 150.0
    approx_shares = 2500.0 / approx_price  # ~16.7 shares
    per_trade_cost = model.round_trip_cost(approx_price, approx_price, approx_shares)
    total_cost = per_trade_cost * n_trades
    gross = 1716.0
    print(f"Approx per-round-trip cost : ${per_trade_cost:,.2f}")
    print(f"Total cost over {n_trades} trades : ${total_cost:,.2f}")
    print(f"Reported GROSS pnl         : ${gross:,.2f}")
    print(f"Estimated NET pnl          : ${gross - total_cost:,.2f}")
    print(f"Net return on $25k         : {(gross - total_cost) / 25000 * 100:,.2f}%")
