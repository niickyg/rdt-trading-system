# Out-of-Sample Evaluation of Recorded Signals

Signals file window: 2026-02-03 .. 2026-03-05
Raw signals: 1986  |  De-duplicated trades: 50  (one position per symbol at a time)

## Assumptions
- Entry: next-day open after signal | Max hold: 10 days | Cost: 10.0 bps round-trip | Ties -> stop
- Portfolio: 1.0% risk/trade, max 8 concurrent, $25,000 start

## Trade statistics (per-trade, equal weight in R)
- Trades: 50  (37 long / 13 short)
- Win rate: 64.0%
- Expectancy: +0.284 R per trade
- Profit factor: 1.81
- Avg win: +0.99 R | Avg loss: -0.97 R
- Outcomes: {'stop': 21, 'target': 25, 'time': 4}
- LONG expectancy: +0.413 R (n=37, win 67.6%)
- SHORT expectancy: -0.085 R (n=13, win 53.8%)

## Portfolio simulation vs SPY buy-and-hold
- Trades actually taken (concurrency cap): 28/50
- Final equity: $24,875  (-0.50%)
- SPY buy-and-hold 2026-02-04..2026-03-19: -4.43%
- **Strategy minus SPY: +3.92%**

## Selection policy at the position cap (why arrival order matters)
On 36 same-day signals vs a 8-slot cap, *which* signals get taken dominates the result:
- Arbitrary arrival order (current bot behavior): -0.50%
- RRS-priority (strongest RRS gets the slot): +1.61%
- RRS-priority + long-only: +5.42%
- (SPY buy-and-hold same window: -4.43%)

## Verdict: strategy BEATS SPY buy-and-hold over this window (net of 10.0 bps costs).
