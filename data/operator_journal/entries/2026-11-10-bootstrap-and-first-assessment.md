# 2026-11-10 — Bootstrap of operator journal + first honest assessment

**Instance:** Claude Opus 4.8, cloud remote agent
**Branch:** `claude/adoring-feynman-qturqr` (harness-pinned; task asked for `operator/2026-11-10`, see note)
**Verdict:** The RDT strategy has a **small, cost-fragile per-signal edge** but at a realistic
portfolio level over a fair window it **underperformed SPY buy-and-hold** (−0.84% vs +4.22%).
No evidence the bot is profitable net of honest costs. **Recommend: measure before optimizing,
and treat the current strategy as not-yet-proven — do not deploy, do not scale risk.**

---

## Context read

I was instructed to read `MANDATE.md`, `LATEST.md`, `POST_MORTEM_RRS.md`, `CLAUDE.md`, and the 3
most recent journal entries. **Reality: none of the operator-journal files existed** — no
`data/operator_journal/` directory, no `MANDATE.md`, no `LATEST.md`, no `POST_MORTEM_RRS.md`,
anywhere in HEAD, on `main`, or in git history. This is the **bootstrap session**: I am
effectively the first operator instance.

I did **not** fabricate the missing history. I read what actually exists: `CLAUDE.md` (detailed
architecture), `README.md`, `ACTIONABLE_100X_STRATEGY.md`, the trained-model metrics, and the
signal history/metrics. I reconstructed `MANDATE.md` from the authorized scheduled-task prompt
(the only available source of the constitution) and flagged it as bootstrap-created.

**Branch note:** the task prompt asked for an `operator/2026-11-10` branch, but the harness
pins this session to `claude/adoring-feynman-qturqr` with "NEVER push to a different branch
without explicit permission." The harness pin is the stronger constraint and is wired to the
review flow, so I used it. Future instances: follow whatever branch the harness pins.

## Assessment (evidence, with numbers)

**1. The project's own honest math already concedes a marginal edge.**
`ACTIONABLE_100X_STRATEGY.md` states: profit factor **1.29**, win rate **38%**, best backtest
**~6.8% annual**, and Kelly criterion **≈ −0.02 (negative)**. It then concludes "trading alone
cannot achieve 100% returns" and pivots to a **signal-service / SaaS** plan (subscriptions,
Stripe, API access). The most recent git commit is a "SaaS product overhaul." **This is mission
drift**: subscription revenue is not trading P&L and does not make the *bot* profitable.

**2. The ML stack has no demonstrated out-of-sample edge.**
`models/ensemble/metrics.json`: cross-validated ensemble AUC **0.543** (0.5 = random) while
train AUC is **0.993** — severe overfitting. `train_precision/recall/f1 = 0.0` indicates a
broken label/threshold. `models/training_metrics.json` (regime detector): silhouette
**−0.087** (worse than random clustering); 1030 of 1056 samples collapse into one regime. These
layers should be treated as advisory noise, consistent with CLAUDE.md's own "ML is advisory-only".

**3. There is essentially no realized track record.**
`data/signals/signal_metrics.json`: 880 scans, 120 signals, but **only 2 tracked outcomes**
(1 win, 1 loss). `data/signals/signal_history.json`: **1986 signals** (Feb 3 – Mar 5 2026, 48
symbols, 1687 long / 299 short) with **zero** outcome fields. The bot generates signals and does
not record what happens to them. You cannot claim or improve profitability without this.

**4. Independent replay of the bot's OWN signals (new work this session).**
I built `scripts/replay_signal_outcomes.py` and replayed the 1986 signals (deduped to 88 unique
symbol/date/direction trades) against real daily OHLC bars (first-touch, stop-first tie-break).

| View | Result |
|---|---|
| Per-signal, raw, fwd=10d | WR 45.5%, expectancy **+0.30R**, PF **1.58** |
| Expectancy consistency (fwd 5/10/20d) | +0.19 / +0.30 / +0.32 R — stable (reassuring) |
| Cost sensitivity | 0.10R friction → +0.20R, PF 1.35; **0.30R friction → +0.00R, PF 1.00 (edge gone)** |
| Realistic portfolio (max 8 positions, 1% risk, 0.10R cost) | only **25/88** signals fit; **−0.84%** over window |
| SPY buy & hold, matched full window (Feb 3 – Apr 30) | **+4.22%** |

The headline: a genuine but **small and cost-fragile** per-signal edge that **evaporates** under
realistic friction and **does not survive** portfolio capacity limits. Over a fair window the
strategy **lost to SPY buy-and-hold**. (An earlier naive sim showed "+29% vs SPY −6%", but that
assumed zero costs, no position cap, and measured SPY at its drawdown trough — all three flatter
the strategy and are not honest.)

**Caveat on the replay:** the live strategy is intraday (5-min entries); intraday history is
unavailable ~8 months later, so this is a **daily-bar swing approximation**, directional
evidence only, N=88, one month. It is not proof the strategy is bad — it is proof that **nobody
has shown it is good**, and the best available fresh evidence is "roughly break-even, behind SPY."

## What I changed

All additive and reviewable. **Did not touch `risk/`, brokers, credentials, or any trading
execution path. AUTO_TRADE untouched (stays false).**

- **Bootstrapped the operator journal:** `data/operator_journal/{MANDATE.md, README.md,
  LATEST.md, entries/}` so future stateless instances have continuity and a constitution.
- **Added `scripts/replay_signal_outcomes.py`** — the measurement tool the project lacks. It
  replays `signal_history.json` against market data with configurable holding window, per-trade
  cost, and a position-capped portfolio sim, and prints the SPY B&H comparison. This is the first
  honest way to ask "did the signals make money?"

## How I verified

- `py_compile` passes on the new script.
- Ran the script end-to-end against live yfinance daily data; output reproduced above and saved
  to the scratchpad `final_result.json`. Deps (pandas/numpy/yfinance) are under `python3.13` in
  this environment, not the project's `python3` (3.11); the script degrades gracefully with a
  clear error if deps are missing.

## Honest verdict vs. the mission

**The bot is not demonstrably profitable, and does not beat SPY buy-and-hold** on the only fresh,
cost-aware evidence available. The edge is marginal (per-signal ~0.2R net of modest cost) and
disappears under realistic portfolio constraints. The ML layers add no out-of-sample edge. The
project has drifted toward monetizing signals rather than proving the trade.

This is **not** a recommendation to wind down yet — the sample is one month and the measurement
infrastructure to decide properly did not exist until today. But it **is** a clear signal to stop
adding features/SaaS surface and start measuring, and to **not deploy live or increase risk**.

## Handoff for the next instance

1. **Run the new tool first:** `python3 scripts/replay_signal_outcomes.py` (use `python3.13` here
   if 3.11 lacks deps). Re-run with `--cost 0.05/0.10/0.20` to see edge fragility yourself.
2. **Biggest gap = no realized-outcome tracking.** Before any optimization, wire the bot to
   persist each signal's actual outcome (target/stop/time, realized R) into `signal_history.json`
   or the DB. Without this, every "improvement" is unfalsifiable. (I did not modify the live
   tracking path because I cannot test it end-to-end here — propose it for human review.)
3. **Get a bigger sample.** One month (Feb 2026) is not enough. Extend the replay across more
   history if signal logs exist, or generate signals over 12+ months of data and replay them.
4. **Resist mission drift.** Do not build more SaaS. The question is edge, not packaging.
5. **Decision gate for a future session:** if, with ≥6 months of signals and honest 0.10R costs,
   the position-capped portfolio still does not clear SPY buy-and-hold, recommend wind-down of the
   automated-trading thesis to the human.
6. **Do NOT touch `risk/`, brokers, or `AUTO_TRADE`** without flagging. They remain untouched.
