"""
Transaction cost modeling and benchmark helpers for honest backtesting.

Background (operator note, 2026-08-27):
The headline walk-forward results historically quoted by this project
(e.g. "6.9% over 2 years" in run_walkforward_v2.py) were computed GROSS of
all transaction costs, and were never compared against a SPY buy-and-hold
benchmark. This module exists so backtests can be evaluated against the only
bar that matters for the operator mandate: positive P&L *net of honest costs*
that also *beats SPY buy-and-hold*.

Everything here is pure and dependency-light so it can be unit-tested without
network access or the full engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class TransactionCostModel:
    """
    Per-fill transaction cost model.

    Defaults approximate Interactive Brokers Pro tiered US equity pricing plus
    a conservative slippage allowance for liquid large caps:

    - commission_per_share: IBKR ~ $0.0035-$0.005 / share.
    - min_commission_per_order: IBKR minimum $1.00 per order.
    - max_commission_pct_of_notional: IBKR caps commission at 1% of trade value.
    - slippage_bps: half-spread + market-impact allowance applied to EACH fill
      (entry and every exit), expressed in basis points of notional. 5 bps per
      side => ~10 bps (0.10%) round trip, a realistic floor for liquid names and
      optimistic for anything less liquid.

    Every fill (entry, each scale-out, final close) incurs one commission and
    one slippage charge. A round-trip therefore pays commission + slippage
    twice, plus once more for each partial scale-out.
    """

    commission_per_share: float = 0.005
    min_commission_per_order: float = 1.0
    max_commission_pct_of_notional: float = 0.01
    slippage_bps: float = 5.0

    def commission(self, shares: float, price: float) -> float:
        """Dollar commission for a single fill of `shares` at `price`."""
        shares = abs(shares)
        if shares <= 0 or price <= 0:
            return 0.0
        notional = shares * price
        raw = shares * self.commission_per_share
        raw = max(raw, self.min_commission_per_order)
        cap = notional * self.max_commission_pct_of_notional
        return min(raw, cap)

    def slippage(self, shares: float, price: float) -> float:
        """Dollar slippage for a single fill of `shares` at `price`."""
        shares = abs(shares)
        if shares <= 0 or price <= 0:
            return 0.0
        notional = shares * price
        return notional * (self.slippage_bps / 10_000.0)

    def fill_cost(self, shares: float, price: float) -> float:
        """Total dollar cost (commission + slippage) for a single fill."""
        return self.commission(shares, price) + self.slippage(shares, price)


ZERO_COST = TransactionCostModel(
    commission_per_share=0.0,
    min_commission_per_order=0.0,
    max_commission_pct_of_notional=0.0,
    slippage_bps=0.0,
)


def _close_column(data: pd.DataFrame) -> str:
    if "Close" in data.columns:
        return "Close"
    if "close" in data.columns:
        return "close"
    raise KeyError("DataFrame has no Close/close column")


def spy_buy_and_hold_return(
    spy_data: pd.DataFrame,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> float:
    """
    Fractional total return of buying SPY at the first available close on/after
    `start_date` and holding to the last available close on/before `end_date`.

    Returns a fraction (0.069 == 6.9%). This is the benchmark the operator
    mandate requires every strategy to beat. Returns 0.0 if the window is empty.
    """
    col = _close_column(spy_data)
    idx_dates = pd.Index([d for d in spy_data.index.date])
    mask = pd.Series(True, index=range(len(spy_data)))
    if start_date is not None:
        mask &= pd.Series(idx_dates >= start_date, index=range(len(spy_data)))
    if end_date is not None:
        mask &= pd.Series(idx_dates <= end_date, index=range(len(spy_data)))

    closes = spy_data[col].to_numpy()
    selected = closes[mask.to_numpy()]
    if len(selected) < 2:
        return 0.0
    first = selected[0]
    last = selected[-1]
    if first <= 0:
        return 0.0
    return float(last / first - 1.0)


def annualize_return(total_return_fraction: float, trading_days: int) -> float:
    """
    Annualize a total fractional return observed over `trading_days` trading
    days (252 trading days per year). Returns a fraction.
    """
    if trading_days <= 0:
        return 0.0
    growth = 1.0 + total_return_fraction
    if growth <= 0:
        return -1.0
    return float(growth ** (252.0 / trading_days) - 1.0)
