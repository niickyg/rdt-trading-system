"""
Honest overlay on the repo's walk-forward V2 gross results:
  1. SPY buy-and-hold benchmark over the identical span (the mission benchmark).
  2. Transaction-cost sweep applied to each strategy config.

Gross results are taken verbatim from the walk-forward V2 run (zero-cost engine).
"""
import yfinance as yf
import pandas as pd
from datetime import date

CAPITAL = 25_000.0

# From the walk-forward V2 AGGREGATE output (gross, zero-cost):
CONFIGS = {
    "A) Baseline":    {"gross_ret_pct": 7.43, "gross_ret_usd": 1856.69, "trades": 234},
    "B) Old Filters": {"gross_ret_pct": 5.06, "gross_ret_usd": 1264.02, "trades": 427},
    "C) RDT Filters": {"gross_ret_pct": 7.68, "gross_ret_usd": 1920.13, "trades": 280},
}

WINDOWS = [
    ("2024-07-12", "2024-11-08"),
    ("2024-11-11", "2025-03-17"),
    ("2025-03-18", "2025-07-18"),
    ("2025-07-21", "2025-11-17"),
    ("2025-11-18", "2026-03-09"),
    ("2026-03-10", "2026-07-24"),
]
SPAN_START, SPAN_END = WINDOWS[0][0], WINDOWS[-1][1]

def close_series():
    df = yf.download("SPY", start="2024-07-01", end="2026-07-26",
                     progress=False, auto_adjust=True)
    c = df["Close"]
    if hasattr(c, "columns"):
        c = c.iloc[:, 0]
    return c

spy = close_series()

def ret_between(s, d0, d1):
    sub = s[(s.index.date >= date.fromisoformat(d0)) & (s.index.date <= date.fromisoformat(d1))]
    return (float(sub.iloc[-1]) / float(sub.iloc[0]) - 1.0) * 100.0

# SPY B&H over the full contiguous span
sub = spy[(spy.index.date >= date.fromisoformat(SPAN_START)) & (spy.index.date <= date.fromisoformat(SPAN_END))]
spy_full_pct = (float(sub.iloc[-1]) / float(sub.iloc[0]) - 1.0) * 100.0
spy_full_usd = CAPITAL * spy_full_pct / 100.0
n_days = len(sub)
years = n_days / 252.0
spy_annualized = ((1 + spy_full_pct/100.0) ** (1/years) - 1) * 100.0

# SPY per-window, summed (apples-to-apples with strategy's summed-window % metric)
per_win = [ret_between(spy, a, b) for a, b in WINDOWS]
spy_sum_windows = sum(per_win)

print("="*88)
print("SPY BUY-AND-HOLD BENCHMARK  (span %s -> %s, %d trading days, %.2f yrs)"
      % (SPAN_START, SPAN_END, n_days, years))
print("="*88)
print(f"  SPY continuous B&H:        {spy_full_pct:+.2f}%   (${spy_full_usd:+,.0f} on ${CAPITAL:,.0f})")
print(f"  SPY annualized:            {spy_annualized:+.2f}%")
print(f"  SPY sum-of-6-windows:      {spy_sum_windows:+.2f}%   (matches strategy's summed-window metric)")
print("  Per window:  " + "  ".join(f"{p:+.1f}%" for p in per_win))
print()

print("="*88)
print("STRATEGY NET OF HONEST TRANSACTION COSTS  (cost applied per round-trip trade)")
print("="*88)
print("  Gross figures are from the zero-cost walk-forward engine.")
print("  Round-trip cost = IBKR commission (~$1-2) + half-spread/slippage both sides.")
print("  NOTE: scaled exits add extra partial fills, so costs here are mildly understated.\n")
costs = [2, 4, 6, 8, 10]
hdr = "  %-16s %10s %8s | " % ("Config", "GrossRet", "Trades") + " ".join(f"${c}/rt" for c in costs)
print(hdr)
print("  " + "-"*(len(hdr)-2))
for name, d in CONFIGS.items():
    row = "  %-16s %9.2f%% %8d | " % (name, d["gross_ret_pct"], d["trades"])
    cells = []
    for c in costs:
        net_usd = d["gross_ret_usd"] - d["trades"] * c
        net_pct = net_usd / CAPITAL * 100.0
        cells.append(f"{net_pct:+6.2f}%")
    print(row + " ".join(cells))
print()
print("  (Each cell = total net return %% over the full ~2yr span after that per-trade cost.)")
print()

print("="*88)
print("HEAD-TO-HEAD  (best strategy config C vs SPY, full span, net of cost)")
print("="*88)
c = CONFIGS["C) RDT Filters"]
for rt in [2, 4, 6, 8]:
    net_usd = c["gross_ret_usd"] - c["trades"] * rt
    net_pct = net_usd / CAPITAL * 100.0
    ann = ((1 + net_pct/100.0) ** (1/years) - 1) * 100.0
    gap = ann - spy_annualized
    print(f"  @ ${rt}/round-trip:  strategy {net_pct:+.2f}% total / {ann:+.2f}% ann"
          f"   vs SPY {spy_annualized:+.2f}% ann   -> gap {gap:+.2f} pts/yr")
