"""
Transaction cost model for backtests.

The historical backtest engines modelled fills with ZERO commission and ZERO
slippage: entries filled at the daily close and exits filled at the exact
stop/target price. That is unrealistically optimistic and inflates reported
edge, especially for a strategy with a thin profit factor (~1.1-1.25) and a
few hundred trades per window.

This module provides a small, dependency-free (stdlib-only) cost model so the
backtest can report returns *net of honest costs* — the bar the operator
mandate holds the system to.

Design goals:
  - Pure stdlib (no pandas/numpy) so it is trivially unit-testable.
  - Zero-cost by default -> opting in is explicit and backward compatible.
  - Costs applied per execution LEG (entry fill, each scaled exit, final exit).

Cost components per leg:
  - commission: max(min_commission_per_order, shares * commission_per_share)
  - slippage:   shares * price * (slippage_bps / 10_000)

Slippage is modelled as an adverse cost on every fill. We do NOT try to model
direction here; it is a dollar drag subtracted from P&L, which is the honest
conservative treatment for daily-bar fills that are otherwise assumed perfect.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Per-leg transaction cost model.

    Defaults are all zero, which reproduces the legacy cost-free behaviour.
    Call a factory (e.g. ``CostModel.ibkr_retail()``) for a realistic profile.
    """

    commission_per_share: float = 0.0
    min_commission_per_order: float = 0.0
    slippage_bps: float = 0.0  # basis points of fill notional, per leg

    def leg_cost(self, price: float, shares: int) -> float:
        """Total cost (USD) for a single execution leg of ``shares`` at ``price``.

        Returns 0.0 for non-positive share counts or prices so callers never
        need to guard against degenerate fills.
        """
        if shares <= 0 or price <= 0:
            return 0.0

        commission = max(self.min_commission_per_order, shares * self.commission_per_share)
        slippage = shares * price * (self.slippage_bps / 10_000.0)
        return commission + slippage

    def is_zero(self) -> bool:
        return (
            self.commission_per_share == 0.0
            and self.min_commission_per_order == 0.0
            and self.slippage_bps == 0.0
        )

    # ------------------------------------------------------------------ #
    # Named profiles
    # ------------------------------------------------------------------ #
    @classmethod
    def zero(cls) -> "CostModel":
        """Legacy behaviour: no commission, no slippage."""
        return cls()

    @classmethod
    def ibkr_retail(cls) -> "CostModel":
        """A deliberately *conservative* (i.e. low) retail profile.

        - IBKR Pro tiered US equity commission: ~$0.0035/share, $0.35 order min.
        - 5 bps (0.05%) slippage per leg. For liquid large caps entering at the
          close this is modest; for stop-outs that gap through the stop it is
          arguably generous to the strategy. We keep it low on purpose so the
          net result cannot be dismissed as a pessimistic assumption.
        """
        return cls(
            commission_per_share=0.0035,
            min_commission_per_order=0.35,
            slippage_bps=5.0,
        )

    @classmethod
    def realistic(cls, slippage_bps: float = 10.0) -> "CostModel":
        """A more realistic profile with configurable slippage.

        10 bps/leg reflects that daily-bar stop/target fills are optimistic and
        that short sales incur borrow/locate friction not otherwise modelled.
        """
        return cls(
            commission_per_share=0.0035,
            min_commission_per_order=0.35,
            slippage_bps=slippage_bps,
        )
