"""
Transaction cost and slippage modeling for honest backtest accounting.

The walk-forward backtests historically computed P&L as a frictionless
``(exit_price - entry_price) * shares`` with fills assumed to occur at the
exact stop/target price. Real trading pays commissions on every order and
suffers slippage on every fill — and stop-outs in particular fill *worse*
than their trigger price because the market is moving against you when the
stop trips. Reporting gross returns as if these costs do not exist makes a
marginal-edge strategy look profitable when it may not be.

This module provides a small, dependency-free cost model (pure stdlib) so it
can be unit-tested in isolation and reused by any backtest/reporting layer.

Defaults are calibrated to Interactive Brokers US-equity retail trading on a
~$25K account (the RDT paper account), erring toward realistic-but-not-punitive:

  - Commission: IBKR fixed tier, $0.005/share, $1.00 minimum per order.
    A round trip is two orders (entry + exit).
  - Slippage: 2 basis points (0.02%) of notional per fill for liquid large
    caps (half the typical penny-wide spread on a $200 stock), applied to
    BOTH the entry and the exit.
  - Stop-out penalty: an EXTRA 5 bps on the exit fill when the exit reason is
    a stop, reflecting that protective stops fill through the trigger in fast
    markets and gap opens.

All figures are conservative estimates, not guarantees. The point is to make
net-of-cost performance *measurable* rather than to claim a precise number.

Run ``python backtesting/costs.py`` to execute the self-test.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Realistic per-order commission + per-fill slippage model."""

    commission_per_share: float = 0.005      # IBKR fixed tier
    min_commission_per_order: float = 1.00   # IBKR per-order minimum
    max_commission_pct: float = 0.01         # IBKR caps at 1% of trade value
    slippage_bps: float = 2.0                # per fill, basis points of notional
    stop_extra_bps: float = 5.0              # additional bps when exit is a stop-out

    # ------------------------------------------------------------------ #
    # Commission
    # ------------------------------------------------------------------ #
    def commission(self, price: float, shares: int) -> float:
        """Commission for a single order (one side of a round trip)."""
        shares = abs(int(shares))
        if shares == 0 or price <= 0:
            return 0.0
        raw = shares * self.commission_per_share
        capped = min(raw, self.max_commission_pct * price * shares)
        return max(self.min_commission_per_order, capped)

    # ------------------------------------------------------------------ #
    # Slippage
    # ------------------------------------------------------------------ #
    def slippage(self, price: float, shares: int, is_stop: bool = False) -> float:
        """Slippage cost (always adverse, in dollars) for a single fill."""
        shares = abs(int(shares))
        if shares == 0 or price <= 0:
            return 0.0
        bps = self.slippage_bps + (self.stop_extra_bps if is_stop else 0.0)
        return price * shares * bps / 10_000.0

    # ------------------------------------------------------------------ #
    # Round trip
    # ------------------------------------------------------------------ #
    def round_trip_cost(
        self,
        entry_price: float,
        exit_price: float,
        shares: int,
        is_stop_exit: bool = False,
    ) -> float:
        """
        Total friction (commissions + slippage) for a completed round trip.

        Returned as a positive dollar amount to SUBTRACT from gross P&L.
        """
        shares = abs(int(shares))
        if shares == 0:
            return 0.0
        entry_price = max(0.0, float(entry_price))
        exit_price = max(0.0, float(exit_price))
        commissions = self.commission(entry_price, shares) + self.commission(exit_price, shares)
        slip = (
            self.slippage(entry_price, shares, is_stop=False)
            + self.slippage(exit_price, shares, is_stop=is_stop_exit)
        )
        return commissions + slip


# Shared default instance for convenience.
DEFAULT_COST_MODEL = CostModel()

# Exit reasons that represent a protective stop being hit (worse fills).
_STOP_EXIT_REASONS = {"stop", "stop_loss", "stopped_out", "stop_hit", "trailing_stop"}


def is_stop_exit(exit_reason) -> bool:
    """True when an exit reason string denotes a stop-out."""
    if not exit_reason:
        return False
    r = str(exit_reason).strip().lower().replace(" ", "_").replace("-", "_")
    return any(tok in r for tok in ("stop",)) or r in _STOP_EXIT_REASONS


def total_trade_costs(trades, model: CostModel = DEFAULT_COST_MODEL) -> float:
    """
    Sum round-trip friction across an iterable of trade objects.

    Each trade is expected to expose ``entry_price``, ``exit_price``, ``shares``
    and (optionally) ``exit_reason``. Trades missing prices are skipped.
    """
    total = 0.0
    for t in trades:
        entry = getattr(t, "entry_price", None)
        exit_ = getattr(t, "exit_price", None)
        shares = getattr(t, "shares", 0) or 0
        if entry is None or exit_ is None:
            continue
        total += model.round_trip_cost(
            entry, exit_, shares, is_stop_exit=is_stop_exit(getattr(t, "exit_reason", None))
        )
    return total


def spy_buy_and_hold_return(first_close: float, last_close: float, capital: float) -> float:
    """
    Dollar P&L of buying SPY at ``first_close`` and holding to ``last_close``
    with ``capital`` fully invested. Whole-share purchase, one entry commission
    and one exit commission (negligible on a single position, but included for
    honesty). Returns dollars.
    """
    if first_close <= 0 or capital <= 0:
        return 0.0
    shares = int(capital // first_close)
    if shares == 0:
        return 0.0
    gross = (last_close - first_close) * shares
    friction = DEFAULT_COST_MODEL.commission(first_close, shares) + DEFAULT_COST_MODEL.commission(
        last_close, shares
    )
    return gross - friction


# ---------------------------------------------------------------------------- #
# Self-test (no external dependencies — runnable anywhere)
# ---------------------------------------------------------------------------- #
def _selftest() -> None:
    m = CostModel()

    # Commission: 100 shares * $0.005 = $0.50 -> below $1 min -> $1.00
    assert abs(m.commission(200.0, 100) - 1.00) < 1e-9, m.commission(200.0, 100)
    # 1000 shares * $0.005 = $5.00 -> above min, below 1% cap -> $5.00
    assert abs(m.commission(200.0, 1000) - 5.00) < 1e-9, m.commission(200.0, 1000)
    # Penny stock cap: 1000 sh @ $0.10 -> raw $5, cap = 1% * $100 = $1.00 -> max(min, $1) = $1
    assert abs(m.commission(0.10, 1000) - 1.00) < 1e-9, m.commission(0.10, 1000)

    # Slippage: 100 sh @ $200, 2 bps = 200*100*0.0002 = $4.00
    assert abs(m.slippage(200.0, 100) - 4.00) < 1e-9, m.slippage(200.0, 100)
    # Stop slippage: 2+5=7 bps -> 200*100*0.0007 = $14.00
    assert abs(m.slippage(200.0, 100, is_stop=True) - 14.00) < 1e-9, m.slippage(200.0, 100, True)

    # Round trip, non-stop: comm 1+1=2 ; slip entry 4 + exit(same px) 4 = 8 ; total 10
    rt = m.round_trip_cost(200.0, 200.0, 100, is_stop_exit=False)
    assert abs(rt - 10.00) < 1e-9, rt
    # Round trip, stop exit: comm 2 ; slip entry 4 + exit stop 14 = 18 ; total 20
    rt_stop = m.round_trip_cost(200.0, 200.0, 100, is_stop_exit=True)
    assert abs(rt_stop - 20.00) < 1e-9, rt_stop

    # is_stop_exit classification
    assert is_stop_exit("stop_loss")
    assert is_stop_exit("Trailing Stop")
    assert is_stop_exit("time_stop")  # contains 'stop' -> conservatively penalized
    assert not is_stop_exit("target")
    assert not is_stop_exit(None)

    # SPY buy-and-hold: $25,000 / $400 = 62 shares; move $400->$480 = $80*62 = $4,960 gross
    bh = spy_buy_and_hold_return(400.0, 480.0, 25_000.0)
    # commissions: 62*0.005=$0.31 -> min $1 each side -> $2 total
    assert abs(bh - (80.0 * 62 - 2.0)) < 1e-6, bh

    # total_trade_costs over duck-typed trades
    class T:
        def __init__(self, e, x, s, r):
            self.entry_price, self.exit_price, self.shares, self.exit_reason = e, x, s, r

    trades = [T(200.0, 210.0, 100, "target"), T(100.0, 95.0, 50, "stop_loss")]
    tc = total_trade_costs(trades)
    assert tc > 0, tc

    print("costs.py self-test: ALL PASSED")
    print(f"  round-trip (non-stop, 100sh @ $200): ${rt:.2f}")
    print(f"  round-trip (stop-out, 100sh @ $200): ${rt_stop:.2f}")
    print(f"  SPY buy&hold $25k, $400->$480:       ${bh:,.2f}")


if __name__ == "__main__":
    _selftest()
