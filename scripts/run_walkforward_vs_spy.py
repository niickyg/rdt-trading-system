#!/usr/bin/env python3
"""
Walk-forward backtest vs SPY buy-and-hold benchmark.

Reuses the repo's OWN walk-forward V2 engines + data loader (run_walkforward_v2),
then adds the one comparison that script never makes: SPY buy-and-hold over the
identical windows and the identical contiguous span. This is the benchmark the
operator mandate measures against ("beating SPY buy-and-hold").

Created by operator run 001 (2026-11-18). Progress prints to stderr; a machine-
readable result block prints between ===JSON_START===/===JSON_END=== on stdout.

Usage:
    python scripts/run_walkforward_vs_spy.py
"""
import sys, json, time
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from loguru import logger
logger.remove(); logger.add(sys.stderr, level="ERROR")

import run_walkforward_v2 as wf  # the repo's canonical script as a module


def spy_bh_return(spy_data, start_d, end_d, capital):
    close_col = 'Close' if 'Close' in spy_data.columns else 'close'
    s = spy_data[(spy_data.index.date >= start_d) & (spy_data.index.date <= end_d)]
    if len(s) < 2:
        return 0.0, 0.0
    p0 = float(s[close_col].iloc[0]); p1 = float(s[close_col].iloc[-1])
    pct = (p1 / p0 - 1.0)
    return capital * pct, pct * 100.0


def main():
    t0 = time.time()
    stock_data, spy_data, vix_data, sector_etf_data = wf.load_all_data()
    all_dates = sorted(spy_data.index.date)
    out = {"windows": [], "data_days": len(all_dates),
           "data_range": [str(all_dates[0]), str(all_dates[-1])],
           "watchlist_n": len(stock_data), "capital": wf.INITIAL_CAPITAL}

    agg = {k: {"ret": 0.0, "trades": 0, "wins": 0, "gross_w": 0.0, "gross_l": 0.0}
           for k in ("baseline", "old", "rdt", "spy")}

    for win in wf.WALK_FORWARD_WINDOWS:
        si = min(win["start_day"], len(all_dates) - 1)
        ei = min(win["end_day"], len(all_dates) - 1)
        w_start, w_end = all_dates[si], all_dates[ei]

        bl = wf.run_baseline(stock_data, spy_data, w_start, w_end, win["name"])
        ol = wf.run_old_filtered(stock_data, spy_data, vix_data, sector_etf_data, w_start, w_end, win["name"])
        rd = wf.run_rdt_filtered(stock_data, spy_data, vix_data, sector_etf_data, w_start, w_end, win["name"])
        spy_ret, spy_pct = spy_bh_return(spy_data, w_start, w_end, wf.INITIAL_CAPITAL)

        rec = {"name": win["name"], "start": str(w_start), "end": str(w_end),
               "baseline": {"ret": bl.result.total_return, "pct": bl.result.total_return_pct,
                            "trades": bl.result.total_trades, "wr": bl.result.win_rate * 100},
               "old": {"ret": ol.result.total_return, "pct": ol.result.total_return_pct,
                       "trades": ol.result.total_trades, "wr": ol.result.win_rate * 100},
               "rdt": {"ret": rd.result.total_return, "pct": rd.result.total_return_pct,
                       "trades": rd.result.total_trades, "wr": rd.result.win_rate * 100},
               "spy_bh": {"ret": spy_ret, "pct": spy_pct}}
        out["windows"].append(rec)

        for key, res in (("baseline", bl), ("old", ol), ("rdt", rd)):
            agg[key]["ret"] += res.result.total_return
            agg[key]["trades"] += res.result.total_trades
            for t in res.result.trades:
                if t.pnl > 0:
                    agg[key]["wins"] += 1; agg[key]["gross_w"] += t.pnl
                else:
                    agg[key]["gross_l"] += abs(t.pnl)
        agg["spy"]["ret"] += spy_ret

        print(f"{win['name']}: base {bl.result.total_return_pct:+.2f}% | old {ol.result.total_return_pct:+.2f}% | "
              f"rdt {rd.result.total_return_pct:+.2f}% | SPY B&H {spy_pct:+.2f}%", file=sys.stderr)

    # contiguous SPY hold across full span (day of first window start -> last window end)
    full_start = all_dates[min(wf.WALK_FORWARD_WINDOWS[0]["start_day"], len(all_dates)-1)]
    full_end = all_dates[min(wf.WALK_FORWARD_WINDOWS[-1]["end_day"], len(all_dates)-1)]
    spy_full_ret, spy_full_pct = spy_bh_return(spy_data, full_start, full_end, wf.INITIAL_CAPITAL)
    span_days = (full_end - full_start).days

    def finalize(a):
        pf = (a["gross_w"] / a["gross_l"]) if a["gross_l"] > 0 else None
        wr = (a["wins"] / a["trades"] * 100) if a["trades"] else 0.0
        pct = a["ret"] / wf.INITIAL_CAPITAL * 100
        return {"ret": a["ret"], "pct": pct, "trades": a["trades"], "win_rate": wr, "profit_factor": pf}

    out["aggregate_sum_of_windows"] = {k: finalize(agg[k]) for k in ("baseline", "old", "rdt")}
    out["aggregate_sum_of_windows"]["spy_sum"] = {"ret": agg["spy"]["ret"],
                                                  "pct": agg["spy"]["ret"] / wf.INITIAL_CAPITAL * 100}
    out["spy_contiguous_hold"] = {"start": str(full_start), "end": str(full_end),
                                  "span_days": span_days, "ret": spy_full_ret, "pct": spy_full_pct,
                                  "annualized_pct": (spy_full_pct * 365.0 / span_days) if span_days else None}
    # annualize strategy on same contiguous calendar span
    for k in ("baseline", "old", "rdt"):
        p = out["aggregate_sum_of_windows"][k]["pct"]
        out["aggregate_sum_of_windows"][k]["annualized_pct_same_span"] = p * 365.0 / span_days if span_days else None

    out["elapsed_s"] = round(time.time() - t0, 1)
    print("===JSON_START===")
    print(json.dumps(out, indent=2, default=float))
    print("===JSON_END===")


if __name__ == "__main__":
    main()
