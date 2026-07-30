#!/usr/bin/env python3
"""First-touch forward test of RDT signals against real IBKR daily bars.

For each distinct setup, assume a fill at entry_price on the first daily bar whose
date >= the signal date, then scan that bar and subsequent bars (up to HORIZON
trading days) for the first touch of stop or target. Conservative tie-break: if a
single bar's range spans BOTH stop and target, count it as a STOP (worst case).
If neither is touched within HORIZON bars, exit at the last bar's close (M2M).

Outputs win rate, average R, expectancy, and a $ P&L translation using a fixed
fractional-risk model. Also reports the count of ambiguous (both-in-one-bar) bars
so the tie-break's impact is visible.
"""
import json, sys, os

SP = os.path.dirname(os.path.abspath(__file__))
HORIZON = int(sys.argv[1]) if len(sys.argv) > 1 else 10
RISK_PER_TRADE = 0.015   # 1.5% of equity risked per trade (per CLAUDE.md Config C)
EQUITY = 25000.0
SLIPPAGE_R = 0.02        # slippage+commission drag, expressed as fraction of 1R, per side-ish
COST_R = 0.05            # total round-trip friction charged against every trade, in R units

setups = json.load(open(os.path.join(SP, 'setups.json')))

def load_bars(sym):
    p = os.path.join(SP, 'prices', sym + '.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    bars = []
    for i, t in enumerate(d['time']):
        bars.append({
            'date': t[:10],
            'open': d['open'][i], 'high': d['high'][i],
            'low': d['low'][i], 'close': d['close'][i],
        })
    bars.sort(key=lambda b: b['date'])
    return bars

results = []
missing = []
ambiguous_bars = 0

for s in setups:
    bars = load_bars(s['symbol'])
    if not bars:
        missing.append(s['symbol'])
        continue
    entry = s['entry']; stop = s['stop']; target = s['target']
    direction = s['direction']
    if entry <= 0 or stop <= 0 or target <= 0:
        continue
    risk = abs(entry - stop)
    if risk <= 0:
        continue
    reward_R = abs(target - entry) / risk   # planned R:R
    # find start index: first bar with date >= signal date
    start = None
    for i, b in enumerate(bars):
        if b['date'] >= s['date']:
            start = i
            break
    if start is None:
        missing.append(s['symbol'] + ':nostart')
        continue
    window = bars[start:start + HORIZON]
    if not window:
        continue
    outcome = None  # 'target' | 'stop' | 'timeout'
    r = None
    for b in window:
        hi, lo = b['high'], b['low']
        if direction == 'long':
            hit_stop = lo <= stop
            hit_tgt = hi >= target
        else:
            hit_stop = hi >= stop
            hit_tgt = lo <= target
        if hit_stop and hit_tgt:
            global_amb = True
            ambiguous_bars += 1
            outcome = 'stop'; r = -1.0   # conservative
            break
        if hit_stop:
            outcome = 'stop'; r = -1.0; break
        if hit_tgt:
            outcome = 'target'; r = reward_R; break
    if outcome is None:
        # mark to close of last bar
        last = window[-1]['close']
        if direction == 'long':
            r = (last - entry) / risk
        else:
            r = (entry - last) / risk
        outcome = 'timeout'
    r_net = r - COST_R
    results.append({'sym': s['symbol'], 'dir': direction, 'date': s['date'],
                    'outcome': outcome, 'r': r, 'r_net': r_net,
                    'reward_R': reward_R})

n = len(results)
if n == 0:
    print("NO RESULTS. missing:", missing)
    sys.exit(1)

wins = [x for x in results if x['r'] > 0]
losses = [x for x in results if x['r'] <= 0]
tgt = sum(1 for x in results if x['outcome'] == 'target')
stp = sum(1 for x in results if x['outcome'] == 'stop')
tmo = sum(1 for x in results if x['outcome'] == 'timeout')
avg_r = sum(x['r'] for x in results) / n
avg_r_net = sum(x['r_net'] for x in results) / n
win_rate = len(wins) / n
gross_win = sum(x['r'] for x in wins)
gross_loss = -sum(x['r'] for x in losses)
pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

# $ translation: each trade risks RISK_PER_TRADE * EQUITY = 1R in dollars
dollar_per_R = RISK_PER_TRADE * EQUITY
total_dollar_net = sum(x['r_net'] for x in results) * dollar_per_R

print(f"=== First-touch forward test (HORIZON={HORIZON} trading days) ===")
print(f"Setups tested: {n}   (missing price data: {sorted(set(missing))})")
print(f"Outcomes: target={tgt}  stop={stp}  timeout(M2M)={tmo}")
print(f"Ambiguous (both stop&target in one bar, counted stop): {ambiguous_bars}")
print(f"Win rate: {win_rate*100:.1f}%")
print(f"Avg R (gross): {avg_r:+.3f}    Avg R (net of {COST_R}R cost): {avg_r_net:+.3f}")
print(f"Profit factor (gross): {pf:.2f}")
print(f"Avg planned reward:risk: {sum(x['reward_R'] for x in results)/n:.2f}")
print(f"$ per 1R (at {RISK_PER_TRADE*100:.1f}% of ${EQUITY:,.0f}): ${dollar_per_R:.2f}")
print(f"Total NET P&L over {n} trades: ${total_dollar_net:+,.2f}")
print(f"  (this is cumulative if each trade risked 1.5% of a static ${EQUITY:,.0f})")
# by direction
for d in ('long', 'short'):
    dd = [x for x in results if x['dir'] == d]
    if dd:
        wr = sum(1 for x in dd if x['r'] > 0)/len(dd)
        ar = sum(x['r_net'] for x in dd)/len(dd)
        print(f"  {d}: n={len(dd)} winrate={wr*100:.1f}% avgRnet={ar:+.3f}")
json.dump(results, open(os.path.join(SP, 'results_h%d.json' % HORIZON), 'w'), indent=0)
