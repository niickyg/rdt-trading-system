import json, warnings, sys
import numpy as np, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")
sig=json.load(open("data/signals/signal_history.json"))
syms=sorted(set(s["symbol"] for s in sig))
data=yf.download(syms,start="2026-02-01",end="2026-05-01",progress=False,auto_adjust=True,group_by="ticker")
SLIP=0.10  # fraction of ATR per side (round-trip 0.20 ATR) - honest/conservative
COMM=0.005

def run(stop_mult, tcap, longs_only):
    rows=[]
    for s in sig:
        if longs_only and s["direction"]!="long": continue
        try: df=data[s["symbol"]].dropna()
        except: continue
        d=s["direction"]; entry=float(s["entry_price"]); stop0=float(s["stop_price"]); tgt=float(s["target_price"]); atr=float(s.get("atr") or 0)
        risk0=abs(entry-stop0)
        if risk0<=0: continue
        # widen stop
        if d=="long": stop=entry-stop_mult*risk0
        else: stop=entry+stop_mult*risk0
        gd=pd.Timestamp(s["generated_at"][:10]); fwd=df[df.index>gd].head(tcap)
        if fwd.empty: continue
        oc=None; ex=None
        for _,r in fwd.iterrows():
            hi,lo=r["High"],r["Low"]
            hs = lo<=stop if d=="long" else hi>=stop
            ht = hi>=tgt if d=="long" else lo<=tgt
            if hs and ht: oc,ex="stop",stop; break
            if hs: oc,ex="stop",stop; break
            if ht: oc,ex="target",tgt; break
        if oc is None: oc,ex="time",fwd.iloc[-1]["Close"]
        gross = (ex-entry) if d=="long" else (entry-ex)
        net = gross - 2*SLIP*atr - 2*COMM
        rows.append({"net":net,"risk":risk0,"r":net/risk0,"oc":oc,"entry":entry})
    df=pd.DataFrame(rows)
    if not len(df): return None
    wr=(df.net>0).mean()*100; gp=df.loc[df.net>0,"net"].sum(); gl=-df.loc[df.net<=0,"net"].sum()
    pf=gp/gl if gl>0 else 99
    # $ expectancy assuming fixed $ risk per trade of $100 (R-based), equal weight
    exp_per_trade_R=df.r.mean()
    return dict(n=len(df),wr=wr,avgR=exp_per_trade_R,totR=df.r.sum(),pf=pf,oc=df.oc.value_counts().to_dict())

print("Testing exit-rule variants (SLIP=0.10*ATR/side, longs-only=RRS has long edge):\n")
for lo in [True, False]:
  print(f"--- longs_only={lo} ---")
  for sm in [1.0, 2.0, 3.0]:
    for tc in [5,10]:
        r=run(sm,tc,lo)
        if r: print(f" stop={sm}xRisk tcap={tc}d  n={r['n']:4d} win%={r['wr']:4.1f} avgR={r['avgR']:+.3f} totR={r['totR']:+7.1f} PF={r['pf']:.2f}  {r['oc']}")
  print()
