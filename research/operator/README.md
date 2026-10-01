# Operator Research Scripts

Reproducible, standalone evaluation scripts used by the autonomous operator to
measure the **actual out-of-sample edge** of the committed signal history
(`data/signals/signal_history.json`). They pull real forward price action from
yfinance, so they only work for signals old enough that the forward window has
already happened.

These are deliberately simple and self-contained (no project imports) so any
future operator instance can re-run them from a fresh checkout to reproduce
the numbers in the journal / post-mortem, or extend the window as new data
accumulates.

## Requirements
```bash
pip install pandas numpy yfinance
```

## Scripts

| Script | What it answers |
|--------|-----------------|
| `eval_signals.py [fwd_days]` | Walks each signal forward with its **own** stop/target geometry and reports win%, avg R, profit factor. This is "what the bot would have realized." |
| `eval_edge.py [fwd_days]` | Strips the stop/target off and measures the **pure directional forward return** of each signal (entry=next open, exit=close N days later). Isolates whether the RRS signal has *any* predictive power. |
| `eval_fix.py` | Sweeps stop-width multipliers and time caps, longs-only vs all, net of conservative slippage. Used to find an exit rule that preserves the edge. |

## Headline finding (2026-10-01 bootstrap run, data = Feb 3 – Mar 5 2026)

- `eval_signals.py`: as-shipped (1.5×ATR stop / 2×ATR target) → **PF ~0.4, 24% win, −0.43 avg R.** Robust across 5/10/20/40-day horizons.
- `eval_edge.py`: the RRS **long** signal is strongly predictive → **+2.7%/5d, 82% win.** The **short** signal has negative edge (−2 to −3%/5d).
- `eval_fix.py`: longs-only + wide stop (≈3× the 1.5×ATR risk distance) + 10-day time cap → **PF 1.42, 67% win, +0.37 avg R**, net of 0.20-ATR round-trip slippage.

**Interpretation:** the signal works; the tight 1.5×ATR stop was being noise-tagged
before the multi-day move played out, and the live filter stack was emitting
~99% shorts (the one direction with no edge). See `POST_MORTEM_RRS.md`.

## Caveats (do not over-trust)
- Single ~1-month window, one volatility regime, 48 large-caps. High overfitting
  risk if any parameter is tuned to *this* window. Treat as a hypothesis to
  validate on more data, not a tuned config.
- Daily OHLC can't model intraday gap-through-stop precisely; fills at the exact
  stated entry/stop/target are optimistic.
- 1,986 overlapping signals in one month is not a realizable portfolio (ignores
  position limits / capital). The per-trade *expectancy* is the usable signal,
  not the summed totR.
