"""
Transaction-cost and slippage modeling for honest backtests.

The backtesting engines historically filled every order at the exact quoted
price with zero commission.  That systematically overstates performance:

- Entries and exits always "filled" at the signal/stop/target price, so no
  bid/ask spread crossing or market-order slippage was ever paid.
- Stop-loss exits assumed the stop filled exactly at its trigger, ignoring the
  gap/slippage that stops routinely suffer in fast moves.
- No broker commission was ever deducted.

For a strategy whose headline edge is a few percent over two years across
hundreds of round-trip trades, these omissions are not rounding error -- they
can be the entire "edge".  The mission is positive P&L *net of honest costs*,
so cost accounting belongs in the engine, not in a footnote.

This module is pure stdlib (no pandas/numpy) so it can be unit-tested without
the heavy data stack.  Defaults model Interactive Brokers' US-stock tiered
pricing plus conservative-but-modest slippage; every parameter is overridable.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Models per-trade broker commission and execution slippage.

    Defaults approximate Interactive Brokers (the broker this bot targets):
      * commission_per_share = $0.005, with a $1.00 per-order minimum and a
        cap of 1% of trade notional (IBKR tiered US-stock schedule).
      * slippage_bps = 2.0  -> 0.02% adverse price on every fill (spread
        crossing + routine market-order slippage on liquid large caps).
      * stop_slippage_bps = 5.0 -> an ADDITIONAL 0.05% adverse slippage on
        stop-loss exits, which fill worse than their trigger in practice.

    These are deliberately conservative (small) for liquid S&P names.  Illiquid
    names, options, or larger size would warrant higher values.  Options are
    NOT modeled here -- their per-contract fees and wide spreads are far larger
    and must be handled separately.
    """

    commission_per_share: float = 0.005
    min_commission: float = 1.0
    max_commission_pct: float = 0.01
    slippage_bps: float = 2.0
    stop_slippage_bps: float = 5.0

    def commission(self, shares: float, price: float) -> float:
        """Broker commission for a single fill of ``shares`` at ``price``.

        cost = clamp(per_share_cost, floor=min_commission, cap=1% of notional)
        The per-share cost is capped at 1% of notional first, then floored at
        the per-order minimum, matching IBKR's published rule.
        """
        shares = abs(int(shares))
        if shares == 0:
            return 0.0
        per_share_cost = shares * self.commission_per_share
        notional = shares * abs(price)
        if notional > 0:
            per_share_cost = min(per_share_cost, notional * self.max_commission_pct)
        return max(per_share_cost, self.min_commission)

    def fill_price(self, price: float, is_buy: bool, is_stop: bool = False) -> float:
        """Adverse-adjust a quoted price for slippage.

        Buys fill a touch higher, sells a touch lower -- always against the
        trader.  Stop-loss exits add ``stop_slippage_bps`` on top.
        """
        bps = self.slippage_bps + (self.stop_slippage_bps if is_stop else 0.0)
        frac = bps / 10_000.0
        if is_buy:
            return price * (1.0 + frac)
        return price * (1.0 - frac)


# Reproduces the old zero-cost behaviour.  Use ONLY to compare gross vs net or
# in tests -- never as a realistic default.
ZERO_COST = CostModel(
    commission_per_share=0.0,
    min_commission=0.0,
    max_commission_pct=0.0,
    slippage_bps=0.0,
    stop_slippage_bps=0.0,
)
