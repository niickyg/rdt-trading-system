"""
Benchmark & honest-cost accounting for backtests.

The mandate for this system is explicit: profitability means *actual positive
P&L net of honest costs, beating SPY buy-and-hold*. The existing backtest
engines report gross P&L and never compare against SPY buy-and-hold. This
module supplies the two missing pieces so any backtest can be judged against
the bar that actually matters:

  1. spy_buy_and_hold(...)   — what $1 of capital would have done sitting in SPY
  2. estimate_trading_costs(...) — commissions + slippage the gross number omits
  3. summarize_vs_benchmark(...) — a single honest verdict object

All functions are pure and dependency-free (stdlib only) so they can be unit
tested without market data or heavy libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# SPY buy-and-hold benchmark
# ---------------------------------------------------------------------------

def spy_buy_and_hold(
    spy_prices: Sequence[float],
    initial_capital: float,
    annual_dividend_yield: float = 0.013,
    trading_days: Optional[int] = None,
) -> dict:
    """Return the buy-and-hold outcome for parking `initial_capital` in SPY.

    Args:
        spy_prices: ordered closing prices over the window (first = entry,
            last = exit). Length >= 2.
        initial_capital: dollars deployed at the first price.
        annual_dividend_yield: SPY pays ~1.2-1.4%/yr in dividends that price-only
            return ignores. Added pro-rata so the benchmark is not understated.
            Set to 0.0 for a pure price comparison.
        trading_days: number of trading days the window spans, used to prorate
            dividends and annualize. Defaults to len(spy_prices) if not given.

    Returns:
        dict with price_return_pct, total_return_pct (incl. dividends),
        dollar_profit, ending_value, annualized_pct.
    """
    prices = [float(p) for p in spy_prices if p is not None and _is_finite(p) and p > 0]
    if len(prices) < 2 or initial_capital <= 0:
        return {
            "price_return_pct": 0.0,
            "total_return_pct": 0.0,
            "dollar_profit": 0.0,
            "ending_value": float(initial_capital),
            "annualized_pct": 0.0,
        }

    first, last = prices[0], prices[-1]
    price_ret = (last / first) - 1.0

    n_days = trading_days if trading_days and trading_days > 0 else len(prices)
    years = max(n_days / 252.0, 1e-9)
    div_ret = annual_dividend_yield * years
    total_ret = price_ret + div_ret

    ending = initial_capital * (1.0 + total_ret)
    annualized = ((1.0 + total_ret) ** (1.0 / years)) - 1.0 if total_ret > -1 else -1.0

    return {
        "price_return_pct": price_ret * 100.0,
        "total_return_pct": total_ret * 100.0,
        "dollar_profit": ending - initial_capital,
        "ending_value": ending,
        "annualized_pct": annualized * 100.0,
    }


# ---------------------------------------------------------------------------
# Honest transaction costs
# ---------------------------------------------------------------------------

@dataclass
class CostModel:
    """Conservative-but-honest cost assumptions for a retail IBKR account.

    commission_per_share: IBKR tiered/fixed stock commission. Fixed is $0.005/sh
        (min $1/order). We model per-share and add a per-order floor separately.
    min_commission_per_order: order-level floor.
    slippage_bps_per_side: spread + market-impact cost per fill, in basis points
        of notional. 2-3 bps per side is realistic for liquid large caps with
        marketable-limit orders; illiquid names are worse.
    """

    commission_per_share: float = 0.005
    min_commission_per_order: float = 1.0
    slippage_bps_per_side: float = 2.5


def estimate_trade_cost(
    shares: int,
    entry_price: float,
    exit_price: float,
    model: CostModel,
) -> float:
    """Round-trip cost (entry fill + exit fill) for a single position, in dollars."""
    shares = abs(int(shares))
    if shares <= 0:
        return 0.0

    def _one_side(price: float) -> float:
        notional = shares * max(float(price), 0.0)
        commission = max(shares * model.commission_per_share, model.min_commission_per_order)
        slippage = notional * (model.slippage_bps_per_side / 10_000.0)
        return commission + slippage

    return _one_side(entry_price) + _one_side(exit_price)


def estimate_trading_costs(trades: Iterable, model: Optional[CostModel] = None) -> dict:
    """Total honest costs across a list of trade objects.

    Each trade is expected to expose `shares`, `entry_price`, and `exit_price`
    (the EnhancedTrade dataclass does). Trades with no exit price are skipped.
    """
    model = model or CostModel()
    total = 0.0
    counted = 0
    for t in trades:
        entry = getattr(t, "entry_price", None)
        exit_ = getattr(t, "exit_price", None)
        shares = getattr(t, "shares", 0) or 0
        if entry is None or exit_ is None:
            continue
        total += estimate_trade_cost(shares, entry, exit_, model)
        counted += 1
    return {
        "total_cost": total,
        "trades_costed": counted,
        "avg_cost_per_trade": (total / counted) if counted else 0.0,
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def summarize_vs_benchmark(
    strategy_gross_return: float,
    trades: Sequence,
    spy_prices: Sequence[float],
    initial_capital: float,
    trading_days: Optional[int] = None,
    model: Optional[CostModel] = None,
) -> dict:
    """Produce the one object the mandate cares about: did we beat SPY, net of costs?"""
    costs = estimate_trading_costs(trades, model)
    net_return = strategy_gross_return - costs["total_cost"]
    spy = spy_buy_and_hold(spy_prices, initial_capital, trading_days=trading_days)
    spy_dollar = spy["dollar_profit"]

    return {
        "strategy_gross_dollar": strategy_gross_return,
        "estimated_costs": costs["total_cost"],
        "strategy_net_dollar": net_return,
        "strategy_net_pct": (net_return / initial_capital * 100.0) if initial_capital else 0.0,
        "spy_dollar": spy_dollar,
        "spy_pct": spy["total_return_pct"],
        "excess_vs_spy_dollar": net_return - spy_dollar,
        "beats_spy": net_return > spy_dollar,
        "n_trades_costed": costs["trades_costed"],
    }


def _is_finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))
