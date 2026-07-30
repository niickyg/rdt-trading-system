#!/usr/bin/env python3
"""Per-trade alpha vs SPY over matched holding windows.

Same first-touch exit logic as first_touch.py, but instead of R-multiples we
compute each trade's realized directional % return on the underlying and compare
it to SPY's close-to-close return over the SAME entry->exit dates (SPY held long,
i.e. buy-and-hold). alpha = trade_return - spy_return.

This isolates whether the signals added timing/selection edge BEYOND simply being
long in a rising market. Costs: subtract a flat friction of 0.10% of notional per
trade (round-trip commission+slippage on the underlying).
"""
import json, os, sys

SP = os.path.dirname(os.path.abspath(__file__))
HORIZON = int(sys.argv[1]) if len(sys.argv) > 1 else 10
FRICTION_PCT = 0.10  # % of notional, round trip

setups = json.load(open(os.path.join(SP, 'setups.json')))
spy = json.load(open(os.path.join(SP, 'prices', 'SPY.json')))
spy_close = {t[:10]: c for t, c in zip(spy['time'], spy['close'])}

def load_bars(sym):
    p = os.path.join(SP, 'prices', sym + '.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    bars = [{'date': d['time'][i][:10], 'high': d['high'][i],
             'low': d['low'][i], 'close': d['close'][i]}
            for i in range(len(d['time']))]
    bars.sort(key=lambda b: b['date'])
    return bars

rows = []
for s in setups:
    bars = load_bars(s['symbol'])
    if not bars:
        continue
    entry, stop, target, direction = s['entry'], s['stop'], s['target'], s['direction']
    if entry <= 0 or stop <= 0 or target <= 0 or abs(entry-stop) <= 0:
        continue
    start = next((i for i, b in enumerate(bars) if b['date'] >= s['date']), None)
    if start is None:
        continue
    window = bars[start:start+HORIZON]
    if not window:
        continue
    entry_date = window[0]['date']
    exit_price = None; exit_date = None
    for b in window:
        if direction == 'long':
            hs, ht = b['low'] <= stop, b['high'] >= target
        else:
            hs, ht = b['high'] >= stop, b['low'] <= target
        if hs and ht:
            exit_price, exit_date = stop, b['date']; break   # conservative
        if hs:
            exit_price, exit_date = stop, b['date']; break
        if ht:
            exit_price, exit_date = target, b['date']; break
    if exit_price is None:
        exit_price, exit_date = window[-1]['close'], window[-1]['date']
    # realized directional return on underlying, net of friction
    if direction == 'long':
        ret = (exit_price - entry) / entry
    else:
        ret = (entry - exit_price) / entry
    ret -= FRICTION_PCT / 100.0
    # SPY matched-window return (buy-and-hold long)
    se, sx = spy_close.get(entry_date), spy_close.get(exit_date)
    if se is None or sx is None:
        continue
    spy_ret = (sx - se) / se
    rows.append({'sym': s['symbol'], 'dir': direction, 'entry_date': entry_date,
                 'exit_date': exit_date, 'ret': ret, 'spy_ret': spy_ret,
                 'alpha': ret - spy_ret})

def summ(name, rr):
    if not rr:
        print(f"{name}: n=0"); return
    n = len(rr)
    mret = sum(x['ret'] for x in rr)/n
    mspy = sum(x['spy_ret'] for x in rr)/n
    mal = sum(x['alpha'] for x in rr)/n
    pos = sum(1 for x in rr if x['alpha'] > 0)/n
    posret = sum(1 for x in rr if x['ret'] > 0)/n
    print(f"{name}: n={n}  mean_trade_ret={mret*100:+.2f}%  mean_SPY_ret={mspy*100:+.2f}%  "
          f"mean_ALPHA={mal*100:+.2f}%  %win(ret>0)={posret*100:.0f}%  %beat_SPY={pos*100:.0f}%")

print(f"=== Alpha vs SPY, matched windows (HORIZON={HORIZON}d, friction={FRICTION_PCT}%/trade) ===")
summ("ALL   ", rows)
summ("LONG  ", [x for x in rows if x['dir'] == 'long'])
summ("SHORT ", [x for x in rows if x['dir'] == 'short'])
# average holding length in trading days (approx via index distance not available; skip)
json.dump(rows, open(os.path.join(SP, 'bench_h%d.json' % HORIZON), 'w'), indent=0)
