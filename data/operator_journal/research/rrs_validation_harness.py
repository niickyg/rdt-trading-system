#!/usr/bin/env python3.13
"""
RRS validation harness — reproducible, dependency-light backtest of the core
Real Relative Strength signal as a DAILY SWING strategy, used by the autonomous
operator to sanity-check profitability claims independently of the production
intraday engine (which needs a DB, IBKR, and intraday bars).

Created: 2026-10-21 (operator run 001). See the journal entry of the same date.

What it does
------------
Implements the repo's own RRS formula (shared/indicators/rrs.py) on DAILY bars:
    RRS = (stock_pct_change - spy_pct_change) / atr_pct
Entry (long only): RRS > threshold AND close > 50-SMA AND SPY "bullish"
  (SPY hard gate: close > 50 EMA and > 200 EMA — the repo's gate #1).
Exit: ATR stop / ATR target / time stop, whichever first (stop assumed to hit
  before target when both are touched intrabar — conservative).
Sizing: fixed fractional risk per trade off the ATR stop distance.
Costs: configurable basis points round-trip (slippage + spread; IBKR commission
  ~$0 at this size).

IMPORTANT CAVEATS (do not overclaim from this tool):
  1. SURVIVORSHIP BIAS. The default universe is today's S&P-500 mega-caps, which
     are survivors. Long-only results are biased upward. A point-in-time universe
     is required before trusting absolute numbers.
  2. DAILY PROXY. The production system trades 5-minute bars with many more
     filters. This tool validates the SIGNAL premise, not the exact production
     implementation.
  3. HINDSIGHT. Parameters explored here were chosen while seeing the data.
     True out-of-sample walk-forward (optimize on window A, test on window B)
     is the required next step before any production change.

Usage:
    python3.13 rrs_validation_harness.py --start 2022-01-01 --end 2025-10-20
    python3.13 rrs_validation_harness.py --config trend_ride --risk 0.01
"""
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

# Repo core watchlist (config/watchlists.py :: SP500_TOP_50)
UNIVERSE = [
    'AAPL','MSFT','GOOGL','GOOG','AMZN','NVDA','META','TSLA','AVGO','ADBE','CRM',
    'AMD','INTC','CSCO','ORCL','QCOM','TXN','AMAT','MU','INTU','NOW','IBM','ADI',
    'LRCX','JPM','V','MA','BAC','WFC','GS','MS','AXP','BLK','C','UNH','JNJ','PFE',
    'ABBV','MRK','LLY','TMO','ABT','DHR','BMY','HD','MCD','NKE','SBUX','TGT','LOW',
    'COST','WMT','PG','KO',
]

CONFIGS = {
    # name           thr   stopATR tgtATR time_stop max_concurrent
    'base':       dict(thr=2.0,  stopA=1.5, tgtA=3.0, tstop=10, maxc=8),
    'thr1_75':    dict(thr=1.75, stopA=1.5, tgtA=3.0, tstop=10, maxc=8),
    'trend_ride': dict(thr=2.0,  stopA=2.0, tgtA=6.0, tstop=20, maxc=8),
}


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _atr(d, n=14):
    h, l, c = d['high'], d['low'], d['close']
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def load(universe, warmup_start, end):
    raw = yf.download(universe + ['SPY'], start=warmup_start, end=end,
                      progress=False, auto_adjust=True)

    def gs(sym):
        try:
            return pd.DataFrame({
                'open': raw['Open'][sym], 'high': raw['High'][sym],
                'low': raw['Low'][sym], 'close': raw['Close'][sym],
            }).dropna()
        except Exception:
            return None

    spy = gs('SPY')
    spy['ret'] = spy['close'].pct_change() * 100
    spy['bull'] = (spy['close'] > _ema(spy['close'], 50)) & (spy['close'] > _ema(spy['close'], 200))

    frames = {}
    for sym in universe:
        d = gs(sym)
        if d is None or len(d) < 250:
            continue
        d['pc'] = d['close'].pct_change() * 100
        d['atr'] = _atr(d)
        d['atrp'] = (d['atr'] / d['close']) * 100
        d['sma50'] = d['close'].rolling(50).mean()
        d = d.join(spy['ret'].rename('spy_pc'))
        d['rrs'] = ((d['pc'] - d['spy_pc']) / d['atrp'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        frames[sym] = d
    return spy, frames


def backtest(spy, frames, start, end, thr=2.0, stopA=1.5, tgtA=3.0, tstop=10,
             maxc=8, risk=0.01, use_spygate=True, acct=25000.0, cost_bps=10.0):
    dates = spy.loc[start:end].index
    op, closed = {}, []
    eq = peak = acct
    mdd = 0.0
    for dt in dates:
        bull = bool(spy.at[dt, 'bull']) if dt in spy.index else False
        for s in list(op):
            d = frames[s]
            if dt not in d.index:
                continue
            b = d.loc[dt]
            ep = None
            if b['low'] <= op[s]['stop']:
                ep = op[s]['stop']
            elif b['high'] >= op[s]['tgt']:
                ep = op[s]['tgt']
            else:
                op[s]['d'] += 1
                if op[s]['d'] >= tstop:
                    ep = b['close']
            if ep is not None:
                g = (ep - op[s]['e']) * op[s]['sh']
                c = (op[s]['e'] + ep) * op[s]['sh'] * (cost_bps / 2 / 10000)
                eq += g - c
                closed.append(g - c)
                peak = max(peak, eq)
                mdd = max(mdd, (peak - eq) / peak * 100)
                del op[s]
        if (bull or not use_spygate) and len(op) < maxc:
            cand = []
            for s, d in frames.items():
                if s in op or dt not in d.index:
                    continue
                b = d.loc[dt]
                if pd.isna(b['rrs']) or pd.isna(b['atr']) or pd.isna(b['sma50']):
                    continue
                if b['rrs'] > thr and b['close'] > b['sma50'] and b['atr'] > 0:
                    cand.append((b['rrs'], s, b))
            cand.sort(key=lambda x: x[0], reverse=True)
            for _, s, b in cand:
                if len(op) >= maxc:
                    break
                e = float(b['close'])
                st = e - stopA * float(b['atr'])
                tg = e + tgtA * float(b['atr'])
                if e - st <= 0:
                    continue
                sh = int(eq * risk / (e - st))
                if sh < 1:
                    continue
                op[s] = {'e': e, 'stop': st, 'tgt': tg, 'sh': sh, 'd': 0}
    for s, p in op.items():
        last = frames[s].loc[:end].iloc[-1]
        ep = float(last['close'])
        g = (ep - p['e']) * p['sh']
        c = (p['e'] + ep) * p['sh'] * (cost_bps / 2 / 10000)
        eq += g - c
        closed.append(g - c)
    t = pd.Series(closed, dtype=float)
    n = len(t)
    wins = int((t > 0).sum())
    gw = t[t > 0].sum()
    gl = -t[t < 0].sum()
    pf = gw / gl if gl > 0 else float('inf')
    return {
        'trades': n, 'win_rate': (wins / n * 100 if n else 0.0), 'pf': pf,
        'total_pct': (eq / acct - 1) * 100, 'net': eq - acct, 'max_dd': mdd,
    }


def spy_bh(spy, start, end):
    c = spy.loc[start:end, 'close']
    return (float(c.iloc[-1]) / float(c.iloc[0]) - 1) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2022-01-01')
    ap.add_argument('--end', default='2025-10-20')
    ap.add_argument('--warmup-start', default=None, help='data start for indicator warmup (default: start - ~7mo)')
    ap.add_argument('--config', default='all', choices=list(CONFIGS) + ['all'])
    ap.add_argument('--risk', type=float, default=0.01)
    ap.add_argument('--cost-bps', type=float, default=10.0)
    args = ap.parse_args()

    warmup = args.warmup_start or (pd.Timestamp(args.start) - pd.Timedelta(days=220)).strftime('%Y-%m-%d')
    spy, frames = load(UNIVERSE, warmup, args.end)
    bh = spy_bh(spy, args.start, args.end)
    print(f"Window {args.start}..{args.end} | universe {len(frames)} | risk {args.risk*100:.0f}% | {args.cost_bps}bps rt")
    print(f"SPY buy & hold: {bh:+.1f}%\n")
    print(f"{'config':<14}{'n':>5}{'win%':>7}{'PF':>6}{'tot%':>9}{'maxDD%':>8}{'vs B&H':>9}")
    names = list(CONFIGS) if args.config == 'all' else [args.config]
    for name in names:
        r = backtest(spy, frames, args.start, args.end, risk=args.risk,
                     cost_bps=args.cost_bps, **CONFIGS[name])
        vs = r['total_pct'] - bh
        print(f"{name:<14}{r['trades']:>5}{r['win_rate']:>6.1f}%{r['pf']:>6.2f}"
              f"{r['total_pct']:>+8.1f}%{r['max_dd']:>7.1f}%{vs:>+8.1f}%")
    print("\nCAVEATS: survivorship bias (fixed mega-cap universe), daily proxy of an "
          "intraday system, hindsight params. Walk-forward OOS required before production use.")


if __name__ == '__main__':
    main()
