import json, warnings, sys
import numpy as np, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")
FWD=int(sys.argv[1]) if len(sys.argv)>1 else 5
sig=json.load(open("data/signals/signal_history.json"))
syms=sorted(set(s["symbol"] for s in sig))
data=yf.download(syms,start="2026-02-01",end="2026-05-01",progress=False,auto_adjust=True,group_by="ticker")
rows=[]
for s in sig:
    try: df=data[s["symbol"]].dropna()
    except: continue
    gd=pd.Timestamp(s["generated_at"][:10])
    fwd=df[df.index>gd].head(FWD+1)
    if len(fwd)<2: continue
    entry=fwd.iloc[0]["Open"]; exitp=fwd.iloc[-1]["Close"]
    if entry<=0: continue
    raw=(exitp-entry)/entry  # pct move of the stock
    signed=raw if s["direction"]=="long" else -raw   # directional return as traded
    rows.append({"dir":s["direction"],"rrs":s.get("rrs"),"signed":signed*100,"raw":raw*100})
df=pd.DataFrame(rows)
print(f"=== PURE {FWD}d FORWARD RETURN (no stop/target), entry=next open, exit=close ===  n={len(df)}")
def st(sub,l):
    if not len(sub): print(l,"none");return
    print(f"{l:14s} n={len(sub):5d} mean_signed_ret={sub.signed.mean():+.3f}%  win%={(sub.signed>0).mean()*100:4.1f}  median={sub.signed.median():+.3f}%")
st(df,"ALL(as traded)"); st(df[df.dir=='long'],"LONG"); st(df[df.dir=='short'],"SHORT")
df["arrs"]=df["rrs"].abs()
for lo,hi in [(2,3),(3,5),(5,99)]:
    st(df[(df.arrs>=lo)&(df.arrs<hi)],f"|rrs|{lo}-{hi}")
