#!/usr/bin/env python3
"""
Edge Sensitivity Analysis — no market data required.

Purpose (operator tooling): quantify, from the RDT strategy's OWN reported
per-trade economics, (1) whether those numbers are internally consistent,
(2) how much honest per-trade cost the claimed edge can absorb before going
negative, and (3) how the best-case GROSS result compares to simply holding SPY.

This exists because the committed "profitability" numbers (CLAUDE.md, the 100X
docs) come from a backtest with same-bar look-ahead entries, idealized fills,
and ZERO transaction costs. This script does not re-run that backtest; it stress-
tests the reported result arithmetically. It needs no network and no data files,
so any operator instance can reproduce it.

Sources of the input numbers (all committed in the repo):
  - ACTIONABLE_100X_STRATEGY.md: "Best backtest: 6.8% annual (~$1,700 profit)",
    "profit factor 1.29", "38% win rate", "~215 trades/year", "1% risk ($250)",
    avg win ~$70 / avg loss ~$45, and an explicit "kelly = -0.02  # NEGATIVE!".
  - CLAUDE.md walk-forward table: best net "$1,716 (6.9%)" over 2 years.

Run:  python scripts/edge_sensitivity.py
"""
from __future__ import annotations

# --- Reported inputs (from the repo's own documents) ------------------------
ACCOUNT = 25_000.0
TRADES_PER_YEAR = 215
WIN_RATE = 0.38
CLAIMED_PROFIT_FACTOR = 1.29
CLAIMED_ANNUAL_GROSS = 1_700.0        # best-case, cost-free
RISK_PER_TRADE = 0.01 * ACCOUNT       # $250
DOC_AVG_WIN = 70.0                    # as stated in ACTIONABLE_100X
DOC_AVG_LOSS = 45.0                   # as stated in ACTIONABLE_100X

# SPY total return over candidate windows (approx, for benchmark context).
# 2024 calendar: ~+25%. 2024-2025 blended annualized commonly ~15-25%.
# We show a conservative band so the comparison is not cherry-picked.
SPY_ANNUAL_BAND = (0.10, 0.25)        # 10%..25% annualized buy-and-hold


def line(c="-", n=72):
    print(c * n)


def pf_from_docs():
    """Profit factor implied by the docs' OWN avg win / avg loss / win rate."""
    winners = TRADES_PER_YEAR * WIN_RATE
    losers = TRADES_PER_YEAR * (1 - WIN_RATE)
    gross_profit = winners * DOC_AVG_WIN
    gross_loss = losers * DOC_AVG_LOSS
    pf = gross_profit / gross_loss if gross_loss else float("inf")
    expectancy = (WIN_RATE * DOC_AVG_WIN) - ((1 - WIN_RATE) * DOC_AVG_LOSS)
    return pf, expectancy, gross_profit, gross_loss


def consistent_economics():
    """Back out per-trade economics that ARE consistent with the claimed
    profit factor + win rate + annual gross, so the cost test uses the
    strategy's most FAVORABLE self-description."""
    # gross_profit - gross_loss = ANNUAL_GROSS ; gross_profit/gross_loss = PF
    gross_loss = CLAIMED_ANNUAL_GROSS / (CLAIMED_PROFIT_FACTOR - 1)
    gross_profit = gross_loss * CLAIMED_PROFIT_FACTOR
    winners = TRADES_PER_YEAR * WIN_RATE
    losers = TRADES_PER_YEAR * (1 - WIN_RATE)
    avg_win = gross_profit / winners
    avg_loss = gross_loss / losers
    net_per_trade = CLAIMED_ANNUAL_GROSS / TRADES_PER_YEAR
    return avg_win, avg_loss, net_per_trade, gross_profit, gross_loss


def cost_table(net_per_trade):
    """Net annual P&L as honest round-trip cost per trade rises."""
    print(f"  Gross edge per trade (best case, cost-free): ${net_per_trade:,.2f}")
    print(f"  Break-even round-trip cost per trade:        ${net_per_trade:,.2f}")
    print()
    print(f"  {'round-trip cost/trade':>24} | {'net annual $':>13} | {'net %':>7} | verdict")
    line()
    for cost in (0, 2, 4, net_per_trade, 8, 10, 15):
        net_annual = (net_per_trade - cost) * TRADES_PER_YEAR
        pct = net_annual / ACCOUNT * 100
        verdict = "profitable" if net_annual > 0 else "LOSES MONEY"
        tag = "  <- break-even" if abs(cost - net_per_trade) < 1e-9 else ""
        print(f"  {('$'+format(cost,'.2f')):>24} | {net_annual:>13,.0f} | "
              f"{pct:>6.1f}% | {verdict}{tag}")


def realistic_cost_estimate():
    """A defensible round-trip cost for a ~$12.5k notional momentum trade."""
    # Typical: risk $250 at ~1x ATR stop on a ~$150 stock, ~83 shares, ~$12.5k notional.
    notional = 12_500.0
    shares = 83
    commission = max(1.0, shares * 0.005) * 2                 # IBKR-ish, both sides
    spread = notional * 0.0002 * 2                            # ~2 bp each side
    slippage = notional * 0.0010 * (1 - WIN_RATE)             # gap-through on stop-outs
    total = commission + spread + slippage
    print(f"  Example ~${notional:,.0f} notional momentum trade:")
    print(f"    commission (both sides, IBKR-ish): ${commission:,.2f}")
    print(f"    spread crossed (~2bp each side):   ${spread:,.2f}")
    print(f"    stop-out gap slippage (~10bp*Ploss): ${slippage:,.2f}")
    print(f"    ---------------------------------------------")
    print(f"    realistic round-trip cost/trade:   ${total:,.2f}")
    return total


def spy_comparison():
    lo, hi = SPY_ANNUAL_BAND
    print(f"  Best-case strategy GROSS (cost-free): "
          f"${CLAIMED_ANNUAL_GROSS:,.0f}  ({CLAIMED_ANNUAL_GROSS/ACCOUNT*100:.1f}%/yr)")
    print(f"  SPY buy-and-hold on ${ACCOUNT:,.0f}:   "
          f"${ACCOUNT*lo:,.0f}..${ACCOUNT*hi:,.0f}  ({lo*100:.0f}%..{hi*100:.0f}%/yr)")
    print(f"  --> Even at ZERO trading cost, the strategy underperforms SPY "
          f"buy-and-hold by roughly ${ACCOUNT*lo-CLAIMED_ANNUAL_GROSS:,.0f} "
          f"to ${ACCOUNT*hi-CLAIMED_ANNUAL_GROSS:,.0f} per year.")


def main():
    line("=")
    print("  RDT EDGE SENSITIVITY — reported numbers stress-tested arithmetically")
    print("  (no market data; reproducible by any operator instance)")
    line("=")

    print("\n[1] Are the documents' own per-trade numbers internally consistent?")
    pf, exp, gp, gl = pf_from_docs()
    print(f"  Using the docs' stated avg win ${DOC_AVG_WIN:.0f} / avg loss "
          f"${DOC_AVG_LOSS:.0f} / win rate {WIN_RATE:.0%} / {TRADES_PER_YEAR} trades:")
    print(f"    implied gross profit ${gp:,.0f}, gross loss ${gl:,.0f}")
    print(f"    implied PROFIT FACTOR = {pf:.3f}   (docs CLAIM {CLAIMED_PROFIT_FACTOR})")
    print(f"    implied per-trade expectancy = ${exp:+.2f}  (NEGATIVE => no edge)")
    print(f"  >>> The docs' own avg-win/avg-loss imply PF {pf:.2f}, not "
          f"{CLAIMED_PROFIT_FACTOR}, and a NEGATIVE expectancy. The stated")
    print(f"      numbers are mutually inconsistent — a red flag on the whole result.")

    print("\n[2] Give the strategy its MOST favorable self-description (PF 1.29).")
    avg_win, avg_loss, net_pt, gp2, gl2 = consistent_economics()
    print(f"  Consistent with PF {CLAIMED_PROFIT_FACTOR} + {WIN_RATE:.0%} WR + "
          f"${CLAIMED_ANNUAL_GROSS:,.0f}/yr gross:")
    print(f"    avg win ${avg_win:,.2f}, avg loss ${avg_loss:,.2f}, "
          f"net ${net_pt:,.2f}/trade")

    print("\n[3] How much honest cost can that best-case edge absorb?")
    cost_table(net_pt)

    print("\n[4] Is that break-even cost realistic to exceed?")
    realistic = realistic_cost_estimate()
    net_after = (net_pt - realistic) * TRADES_PER_YEAR
    print(f"  Net annual after realistic cost: ${net_after:,.0f} "
          f"({net_after/ACCOUNT*100:.1f}%/yr)  "
          f"{'PROFITABLE' if net_after>0 else '=> NET LOSS'}")

    print("\n[5] Even cost-free, does it beat SPY buy-and-hold?")
    spy_comparison()

    print()
    line("=")
    print("  CONCLUSION: The reported edge is (a) internally inconsistent, "
          "(b) thin\n  enough that realistic costs likely flip it negative, and "
          "(c) below SPY\n  buy-and-hold even at zero cost. No honest evidence of "
          "an edge vs SPY.")
    line("=")


if __name__ == "__main__":
    main()
