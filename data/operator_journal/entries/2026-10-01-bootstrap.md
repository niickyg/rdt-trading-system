# 2026-10-01 — Bootstrap: journal created; RRS long edge found, tight stops + short-gating identified as the P&L killers

## State on arrival
Fresh checkout. **None of the operator infrastructure existed** — no
`data/operator_journal/`, no `MANDATE.md`, no `POST_MORTEM_RRS.md` anywhere in
git history (checked `git log --all`). Current branch `claude/adoring-feynman-sh4qht`
is identical to `main`. This is the first operator instance; there was no
predecessor to read.

Data present: `data/signals/signal_history.json` (1,986 raw signals, Feb 3 – Mar
5 2026), `active_signals.json` (1), `signal_metrics.json`. The metrics file is
the only trace of live behavior: **880 scans, 120 emitted signals (119 short / 1
long), and just 2 tracked outcomes** (1 target hit, 1 stop out). Last scan
2026-03-05 — data is ~7 months stale. No trade DB in the checkout (lives on the
user's machine, not available to a remote agent).

## Objective
Two things, given the bootstrap situation: (1) stand up the journal/mandate/
post-mortem infrastructure every future run depends on; (2) do a *genuine*
first profitability assessment from the one evidence source available — and since
the signals are from Feb–Mar and today is October, their forward outcomes are
fully knowable, so this can be a real out-of-sample test, not a guess.

## What I did
- Created `data/operator_journal/` (MANDATE.md, entries/, LATEST.md).
- Authored `MANDATE.md` v0 (bootstrap draft, flagged for human ratification) —
  transcribed the mission + hard constraints from the launch prompt and wrote the
  per-run Protocol and journal template.
- Authored `POST_MORTEM_RRS.md` v0 with the full evidence below.
- Wrote and committed 3 reproducible evaluation scripts under `research/operator/`
  (`eval_signals.py`, `eval_edge.py`, `eval_fix.py`) + README, so the numbers can
  be re-derived and extended from any fresh checkout.
- Read the relevant scanner code to confirm the mechanism (stop geometry at
  `realtime_scanner.py:2078`; SPY hard gate `:331`; 50-SMA long-block `:1510`).

## Evidence / results
Installed pandas/numpy/yfinance; yfinance works through the proxy. All numbers
reproduce via `python research/operator/eval_*.py`.

1. **As shipped (1.5×ATR stop / 2×ATR target):** PF **0.42**, 24% win, **−0.43
   avg R**; 75% of trades stopped out. Robust across 5/10/20/40-day horizons.
   SPY benchmark over the window: **−4.3%**.
2. **Pure directional edge (no stop, 5-day hold):** LONG **+2.69%/82% win**;
   SHORT −2.0% (loses). The RRS long signal is genuinely predictive.
3. **Exit-rule sweep, longs-only, net of 0.20-ATR round-trip slippage:** 1×risk
   stop → PF 0.37; 2× → 0.73; **3×risk + 10-day cap → PF 1.42, 67% win, +0.37R.**
4. **Filter inversion:** raw scan 85% long, but emitted set 99% short. In the
   Feb–Mar decline the SPY-bearish gate + 50-SMA gate blocked ~all longs and
   passed shorts — routing capital into the one direction with negative edge.

Caveats (also in the post-mortem): one ~1-month window, one regime, 48 large-caps,
daily-OHLC fills optimistic, overlapping signals ≠ realizable portfolio. These are
**hypotheses to validate on more data, not a config to ship.**

## Decision & rationale
I deliberately made **no change to the live strategy, risk params, or filter
logic** this session. The finding is strong but rests on a single month; shipping
a 3×ATR stop now would be overfitting, and changing exit/risk geometry blind
violates the mandate's "evidence before code" floor. The right move for a
stateless first run is to lock in the infrastructure + the reproducible evidence
base so the *next* instance can validate the lever on 2024–2025 data and then
change code with a human in the loop. The mission is profit, and the highest-
leverage path to it is now identified and documented rather than guessed at.

## Risk/safety flags
- **Did NOT touch `risk/`.** No live-surface changes. No `AUTO_TRADE`, no broker
  credentials, nothing executed.
- `MANDATE.md` is a bootstrap draft I authored for myself in the absence of one —
  flagged at the top for the human owner to ratify or rewrite. Transcribing my
  own constitution is inherently circular; the human should confirm it.
- **Branch:** the launch prompt asked for `operator/2026-10-01`, but this
  session's harness git configuration pins work to `claude/adoring-feynman-sh4qht`
  and instructs never to push elsewhere without explicit permission. I pushed to
  the harness branch to avoid violating that rule. Human should reconcile the two
  branch conventions.

## Recommendation for next instance
**One experiment, do this before anything else:** validate the wide-stop +
long-bias hypothesis **out-of-sample on 2024–2025** (multiple regimes), using the
`research/operator/` scripts adapted to pull historical signals or regenerate
them. If PF > 1 holds across regimes (especially bull markets, where the SPY gate
behaves differently), then — with a human in the loop — change
`realtime_scanner.py` stop geometry (or add a time-based exit) and reconsider the
short book / SPY gate. If it does **not** hold, document that and we're back to
"no robust edge → recommend wind-down."
Secondary: wire up / fix **outcome tracking** (2 outcomes / 120 signals means the
live system is flying blind) so future changes get real P&L attribution.

## Open questions
- Is the long edge regime-dependent (does it survive a bull market where the SPY
  gate would *pass* longs)?
- Is the short book's negative edge structural or just this down-month sample?
- Where does the real trade/P&L history live, and can a future run get read-only
  access to it rather than inferring from signal records?
