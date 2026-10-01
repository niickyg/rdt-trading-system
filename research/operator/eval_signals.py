"""
Out-of-sample evaluation of RDT raw signals (Feb-Mar 2026).
For each signal, walk forward daily OHLC and determine whether the stop or
target was hit first. Compute R-multiples and $ P&L per share with honest costs.
Today is 2026-10-01, so all forward outcomes are fully knowable.
"""
import json, warnings, sys
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
import yfinance as yf
warnings.filterwarnings("ignore")

FWD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10   # time-stop window
SLIPPAGE_FRAC_ATR = 0.05   # slippage per fill = 5% of ATR (entry+exit = 10%)
COMMISSION_PER_SHARE = 0.005

sig = json.load(open("data/signals/signal_history.json"))
syms = sorted(set(s["symbol"] for s in sig))

# Download daily OHLC once per symbol for the whole window + forward buffer
print(f"Downloading {len(syms)} symbols...", file=sys.stderr)
data = yf.download(syms, start="2026-02-01", end="2026-05-01",
                   progress=False, auto_adjust=True, group_by="ticker")

def ohlc(sym):
    try:
        df = data[sym].dropna()
        return df
    except Exception:
        return None

results = []
skipped = 0
for s in sig:
    sym = s["symbol"]; direction = s["direction"]
    entry = float(s["entry_price"]); stop = float(s["stop_price"]); target = float(s["target_price"])
    atr = float(s.get("atr") or 0)
    gen = s["generated_at"][:10]
    df = ohlc(sym)
    if df is None or df.empty:
        skipped += 1; continue
    # start the trading day AFTER signal generation
    try:
        gd = pd.Timestamp(gen)
    except Exception:
        skipped += 1; continue
    fwd = df[df.index > gd].head(FWD_DAYS)
    if fwd.empty:
        skipped += 1; continue

    slip = SLIPPAGE_FRAC_ATR * atr
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        skipped += 1; continue

    outcome = None; exit_px = None
    for _, row in fwd.iterrows():
        hi, lo = row["High"], row["Low"]
        if direction == "long":
            hit_stop = lo <= stop
            hit_tgt = hi >= target
        else:
            hit_stop = hi >= stop
            hit_tgt = lo <= target
        if hit_stop and hit_tgt:
            outcome = "stop"; exit_px = stop  # conservative: assume stop first
            break
        if hit_stop:
            outcome = "stop"; exit_px = stop; break
        if hit_tgt:
            outcome = "target"; exit_px = target; break
    if outcome is None:
        outcome = "time"; exit_px = fwd.iloc[-1]["Close"]

    # directional P&L per share incl slippage on entry+exit + commission
    if direction == "long":
        gross = exit_px - entry
    else:
        gross = entry - exit_px
    net = gross - 2 * slip - 2 * COMMISSION_PER_SHARE
    r_mult = net / risk_per_share
    results.append({
        "sym": sym, "dir": direction, "rrs": s.get("rrs"), "outcome": outcome,
        "gross": gross, "net": net, "r": r_mult, "risk": risk_per_share,
        "daily_strong": s.get("daily_strong"),
    })

df = pd.DataFrame(results)
print(f"\n=== RAW SIGNAL EVALUATION (fwd={FWD_DAYS}d, slip={SLIPPAGE_FRAC_ATR}*ATR/side) ===")
print(f"evaluated: {len(df)}  skipped: {skipped}")

def summarize(sub, label):
    if len(sub) == 0:
        print(f"{label}: no trades"); return
    wins = (sub["net"] > 0).sum()
    wr = wins / len(sub) * 100
    avg_r = sub["r"].mean()
    tot_r = sub["r"].sum()
    gp = sub.loc[sub.net > 0, "net"].sum(); gl = -sub.loc[sub.net <= 0, "net"].sum()
    pf = gp / gl if gl > 0 else float("inf")
    oc = sub["outcome"].value_counts().to_dict()
    print(f"{label:16s} n={len(sub):5d} win%={wr:5.1f} avgR={avg_r:+.3f} totR={tot_r:+8.1f} PF={pf:4.2f} outcomes={oc}")

summarize(df, "ALL")
summarize(df[df.dir == "long"], "LONG")
summarize(df[df.dir == "short"], "SHORT")
print("\n-- by RRS strength bucket (|rrs|) --")
df["arrs"] = df["rrs"].abs()
for lo, hi in [(0,2),(2,3),(3,5),(5,99)]:
    summarize(df[(df.arrs>=lo)&(df.arrs<hi)], f"|rrs| {lo}-{hi}")
print("\n-- daily_strong filter --")
summarize(df[df.daily_strong == True], "daily_strong")

# Compare to SPY buy&hold over same ~1 month window for each trade's capital
print("\n-- benchmark: SPY over window --")
spy = yf.download("SPY", start="2026-02-01", end="2026-05-01", progress=False, auto_adjust=True)
spy_close = spy["Close"].squeeze()
print(f"SPY 2026-02-03 -> 2026-03-19: {float(spy_close.iloc[1]):.2f} -> {float(spy_close.loc['2026-03-19']):.2f} "
      f"({(float(spy_close.loc['2026-03-19'])/float(spy_close.iloc[1])-1)*100:+.2f}%)")
