#!/usr/bin/env python3.13
"""
Independent, honest backtest of the RDT 'Config C' daily-bar strategy.

Goals the committed harness does NOT address:
  1. Transaction costs (commission + slippage) -> net P&L
  2. SPY buy-and-hold benchmark over the IDENTICAL window
  3. No lookahead: signal at close[T], enter at open[T+1], exits via intrabar H/L

This is a corroborating independent estimate, deliberately faithful to
shared/indicators/rrs.py (RRS formula, relaxed strength/weakness) and the
scanner gates (SPY 50/200 EMA gate, stock 50-SMA gate). MTF/VIX/sector/regime
adjustments are omitted -> if anything this trades MORE than live Config C,
making the gross-return estimate an optimistic upper bound.
"""
import warnings, sys
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

WATCHLIST = ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','JPM','V','JNJ',
             'UNH','HD','PG','MA','DIS','PYPL','ADBE','CRM','NFLX','INTC','AMD',
             'CSCO','PEP','KO','MRK','ABT','TMO','COST','AVGO','TXN']

START = "2024-02-01"     # live window referenced by CLAUDE.md
END   = "2025-10-22"
WARMUP_START = "2023-01-01"   # for 200-period EMA/SMA

INITIAL = 25000.0
RRS_THR = 2.0
STOP_ATR = 1.5
TGT_ATR = 2.0
RISK_PCT = 0.015
MAX_POS = 8
MAX_HOLD = 12

def dl(tickers):
    df = yf.download(tickers, start=WARMUP_START, end=END, progress=False,
                     auto_adjust=True, group_by="ticker", threads=True)
    out = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if t in df.columns.get_level_values(0):
                sub = df[t].dropna(how="all")
                sub = sub.rename(columns=str.lower)
                if len(sub) > 50:
                    out[t] = sub
    else:
        out[tickers[0]] = df.rename(columns=str.lower)
    return out

def atr14(d):
    h,l,c = d['high'],d['low'],d['close']
    tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(14).mean()

def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def relaxed_strong(d):
    c=d['close']; e3,e8,e21=ema(c,3),ema(c,8),ema(c,21)
    l3=d.tail(3); green=int((l3['close']>l3['open']).sum())
    l5lows=d['low'].tail(5)
    score = sum([int(e3.iloc[-1]>e8.iloc[-1]), int(e8.iloc[-1]>e21.iloc[-1]),
             int(c.iloc[-1]>e8.iloc[-1]),
             int(len(l5lows)>=3 and l5lows.iloc[-1]>l5lows.iloc[0]), int(green>=2)])
    return score>=3

def relaxed_weak(d):
    c=d['close']; e3,e8,e21=ema(c,3),ema(c,8),ema(c,21)
    l3=d.tail(3); red=int((l3['close']<l3['open']).sum())
    l5h=d['high'].tail(5)
    score = sum([int(e8.iloc[-1]>e3.iloc[-1]), int(e21.iloc[-1]>e8.iloc[-1]),
             int(c.iloc[-1]<e8.iloc[-1]),
             int(len(l5h)>=3 and l5h.iloc[-1]<l5h.iloc[0]), int(red>=2)])
    return score>=3

MAX_POS_PCT = 0.25   # cap any single position at 25% of equity (repo also caps)

def run(data, spy, comm_per_share, slip_bps):
    dates = [d for d in spy.index if START <= d.strftime("%Y-%m-%d") <= END]
    # precompute spy EMAs on full series
    spy_c = spy['close']; spy_e50=ema(spy_c,50); spy_e200=ema(spy_c,200)
    # precompute per-stock
    atr = {t: atr14(d) for t,d in data.items()}
    sma50 = {t: d['close'].rolling(50).mean() for t,d in data.items()}
    cash = INITIAL
    positions = {}   # sym -> dict
    trades = []
    slip = slip_bps/10000.0
    idx_map = {t: {d:i for i,d in enumerate(d_.index)} for t,d_ in data.items()}
    spy_idx = {d:i for i,d in enumerate(spy.index)}

    def cost(shares, price):
        return shares*comm_per_share + shares*price*slip

    def close_pos(p, exit_px):
        """Return cash delta and realized pnl for closing position p at exit_px."""
        sh=p['shares']; xc=cost(sh,exit_px)
        if p['dir']=='long':
            cash_delta = sh*exit_px - xc
            pnl = (exit_px-p['entry_price'])*sh - p['entry_cost'] - xc
        else:
            cash_delta = 2*sh*p['entry_price'] - sh*exit_px - xc  # release margin + pnl
            pnl = (p['entry_price']-exit_px)*sh - p['entry_cost'] - xc
        return cash_delta, pnl

    def mark(p, cl):
        sh=p['shares']
        if p['dir']=='long': return sh*cl                 # long position market value
        return sh*(2*p['entry_price']-cl)                 # short: reserved margin + unrealized

    for di,today in enumerate(dates):
        # ----- manage open positions using today's H/L (exits) -----
        for sym in list(positions.keys()):
            p = positions[sym]; d = data[sym]
            if today not in idx_map[sym]: continue
            i = idx_map[sym][today]
            hi,lo,cl = d['high'].iloc[i], d['low'].iloc[i], d['close'].iloc[i]
            p['held'] += 1
            exit_px=None
            if p['dir']=='long':
                if lo<=p['stop']: exit_px=p['stop']
                elif hi>=p['tgt']: exit_px=p['tgt']
            else:
                if hi>=p['stop']: exit_px=p['stop']
                elif lo<=p['tgt']: exit_px=p['tgt']
            if exit_px is None and p['held']>=MAX_HOLD: exit_px=cl
            if exit_px is not None:
                cd,pnl=close_pos(p,exit_px)
                cash += cd; trades.append(pnl); del positions[sym]

        # ----- equity for sizing (mark to today close) -----
        equity = cash
        for sym,p in positions.items():
            if today in idx_map[sym]:
                cl=data[sym]['close'].iloc[idx_map[sym][today]]
                equity += mark(p, cl)
        # ----- scan for new signals at close[today], enter at open[tomorrow] -----
        if di+1>=len(dates): continue
        tmrw = dates[di+1]
        si = spy_idx.get(today)
        if si is None or si<200: continue
        spy_px=spy_c.iloc[si]; spy_prev=spy_c.iloc[si-1]
        spy_pc=(spy_px/spy_prev-1)*100
        allow_long = (spy_px>spy_e50.iloc[si]) and (spy_px>spy_e200.iloc[si])
        allow_short= (spy_px<spy_e50.iloc[si]) and (spy_px<spy_e200.iloc[si])

        for sym,d in data.items():
            if sym in positions: continue
            if len(positions)>=MAX_POS: break
            if today not in idx_map[sym]: continue
            i=idx_map[sym][today]
            if i<200: continue
            a=atr[sym].iloc[i]
            if not np.isfinite(a) or a<=0: continue
            c0=d['close'].iloc[i]; cprev=d['close'].iloc[i-1]
            if c0<=0 or cprev<=0: continue
            stock_pc=(c0/cprev-1)*100
            atr_pct=(a/c0)*100
            if atr_pct<=0: continue
            rrs=(stock_pc-spy_pc)/atr_pct
            if not np.isfinite(rrs): continue
            direction = 'long' if rrs>0 else 'short'
            if abs(rrs)<RRS_THR: continue
            if direction=='long' and not allow_long: continue
            if direction=='short' and not allow_short: continue
            s50=sma50[sym].iloc[i]
            if direction=='long' and not (c0>s50): continue
            if direction=='short' and not (c0<s50): continue
            hist=d.iloc[:i+1]
            if direction=='long' and not relaxed_strong(hist): continue
            if direction=='short' and not relaxed_weak(hist): continue
            # enter at tomorrow open
            if tmrw not in idx_map[sym]: continue
            j=idx_map[sym][tmrw]
            entry=d['open'].iloc[j]
            if not np.isfinite(entry) or entry<=0: continue
            stop_dist=STOP_ATR*a
            if stop_dist<=0: continue
            risk_dollars=equity*RISK_PCT
            shares=int(risk_dollars/stop_dist)
            # cap position notional at MAX_POS_PCT of equity
            max_sh_by_pct=int((equity*MAX_POS_PCT)/entry)
            shares=min(shares,max_sh_by_pct)
            # no leverage: notional+cost must fit remaining cash
            max_sh_by_cash=int((cash*0.98)/entry)
            shares=min(shares,max_sh_by_cash)
            if shares<1: continue
            notional=shares*entry; ec=cost(shares,entry)
            if notional+ec>cash: continue
            cash-=notional+ec   # long buy OR short margin reserve (symmetric buying-power use)
            if direction=='long':
                stop=entry-stop_dist; tgt=entry+TGT_ATR*a
            else:
                stop=entry+stop_dist; tgt=entry-TGT_ATR*a
            positions[sym]=dict(dir=direction,shares=shares,entry_price=entry,
                entry_cost=ec,stop=stop,tgt=tgt,held=0)
    # liquidate remaining at last close
    last=dates[-1]
    for sym,p in list(positions.items()):
        cl=data[sym]['close'].iloc[idx_map[sym][last]] if last in idx_map[sym] else p['entry_price']
        cd,pnl=close_pos(p,cl); cash+=cd; trades.append(pnl); del positions[sym]
    final=cash
    trades=np.array(trades)
    n=len(trades); wins=(trades>0).sum()
    wr=wins/n*100 if n else 0
    gp=trades[trades>0].sum(); gl=abs(trades[trades<=0].sum())
    pf=gp/gl if gl>0 else float('inf')
    ret=(final/INITIAL-1)*100
    return dict(n=n,wr=wr,pf=pf,ret=ret,final=final,avg=trades.mean() if n else 0)

if __name__=="__main__":
    print("Downloading data...",flush=True)
    allt=WATCHLIST+['SPY']
    data=dl(allt)
    spy=data.pop('SPY')
    print(f"Loaded {len(data)} stocks + SPY")
    # restrict spy index to business days present
    first_trade_date=[d for d in spy.index if d.strftime('%Y-%m-%d')>=START][0]
    last_date=[d for d in spy.index if d.strftime('%Y-%m-%d')<=END][-1]
    spy_bh=(spy['close'].loc[:last_date].iloc[-1]/spy['close'].loc[first_trade_date]-1)*100
    days=(last_date-first_trade_date).days
    yrs=days/365.25
    spy_ann=((1+spy_bh/100)**(1/yrs)-1)*100
    print(f"\nWindow: {first_trade_date.date()} -> {last_date.date()}  ({days} days, {yrs:.2f} yr)")
    print(f"SPY buy-and-hold (div-adjusted): {spy_bh:+.1f}% total,  {spy_ann:+.1f}% annualized\n")
    print(f"{'Scenario':<34}{'Trades':>7}{'WinR':>7}{'PF':>7}{'TotRet':>9}{'Final$':>11}{'AnnRet':>8}")
    scen=[("Gross (no costs)",0.0,0.0),
          ("Realistic ($0.005/sh + 2bps)",0.005,2.0),
          ("Higher ($0.01/sh + 5bps)",0.01,5.0)]
    for name,c,s in scen:
        r=run(data,spy,c,s)
        ann=((r['final']/INITIAL)**(1/yrs)-1)*100
        print(f"{name:<34}{r['n']:>7}{r['wr']:>6.1f}%{r['pf']:>7.2f}{r['ret']:>+8.1f}%{r['final']:>11,.0f}{ann:>+7.1f}%")
    print(f"\nBenchmark to beat: SPY {spy_ann:+.1f}% annualized ({spy_bh:+.1f}% total)")
