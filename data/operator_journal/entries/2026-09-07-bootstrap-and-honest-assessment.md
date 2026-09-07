# 2026-09-07 — Bootstrap + first honest edge assessment

- **Operator:** autonomous instance #1 (Claude Code, remote cloud checkout)
- **Branch:** `claude/adoring-feynman-9j4ymz` (see "Branch note" below)
- **Live infra reachable:** No — no live DB, broker, or container. Committed
  repo snapshots + fetched market data only.

---

## TL;DR

The operator journal / mandate infrastructure **did not exist** in the repo, so I
bootstrapped it and then did the first thing the mandate actually asks for: a
ground-truth, honest read of whether this bot has a profitable edge.

**It does not — not on the evidence available.** Every independent measurement
lands in the same place: a marginal-to-negative edge, no out-of-sample validation,
and ML/regime layers that add nothing. On realistic trade accounting the one month
of real signals the bot produced **lost money** (−0.072R expectancy, profit factor
0.89). **Recommendation: do NOT deploy real capital. Escalate the edge question.**

---

## What I found

### 0. The mandate/journal did not exist
`data/operator_journal/`, `MANDATE.md`, `LATEST.md`, and `POST_MORTEM_RRS.md` are
referenced by my task prompt but are absent from the repo and its entire git
history. I am effectively instance #1. I created `MANDATE.md` (constitution,
derived from the standing task instructions), the `entries/` dir, and this entry.
`POST_MORTEM_RRS.md` still does not exist — its content appears to be partly
captured by `ACTIONABLE_100X_STRATEGY.md`.

### 1. The ML layer has no edge (and is badly overfit)
`models/ensemble/metrics.json`:
- Cross-validated AUC ≈ **0.54** (xgb 0.540, rf 0.546, ensemble 0.543). 0.50 = random.
- Train AUC **0.99** vs CV 0.54 → severe overfitting.
- train precision/recall/F1 = **0.0** → it collapses to predicting the majority class.

`models/training_metrics.json` (regime detector):
- silhouette **−0.087** (worse than random clustering).
- 1030 / 1056 samples labelled `low_volatility` → one-regime collapse. Effectively broken.

Conclusion: the "ML-enhanced signal validation" and "regime detection" in the
architecture are not contributing measurable edge. CLAUDE.md already half-admits
this ("Rule-based filters provide all measurable improvement; ML is advisory-only").

### 2. The bot does not measure its own trades
`data/signals/signal_metrics.json`: **880 scans, 120 signals, total_outcomes = 2.**
`data/signals/signal_history.json`: **1,986 signals, ZERO tracked outcomes.**
`agents/outcome_tracker.py` only tracks *rejected* signals (to test filter
strictness); nothing closes the loop on executed-trade P&L in the committed data.

**You cannot make a bot profitable if it never measures its trades.** This is the
single biggest systemic gap: the system flies blind.

### 3. Ground-truth evaluation of the real signals (new tool this session)
I wrote `scripts/evaluate_signals.py` (stdlib-only, fetches real daily bars from
Yahoo via the proxy, simulates each signal against its own entry/stop/target with a
deliberately *pessimistic* intraday path assumption, nets 5bps costs). Yahoo serves
this environment's 2026 data, so the 1,986 signals (Feb 3 – Mar 5 2026) **are
resolvable**. Full run across all 48 symbols:

| Accounting | Trades | Win rate | Expectancy | Profit factor |
|---|---|---|---|---|
| Per raw signal (inflated — re-counts one position daily) | 1986 | 37.4% | **+0.034R** | 1.053 |
| **Non-overlapping (realistic — one position/symbol)** | **48** | **31.2%** | **−0.072R** | **0.893** |

The "positive" per-signal number is an **artifact of duplication** (DOW alone was
re-flagged 93 times in a month). Counted honestly — one position per symbol at a
time — the strategy has a **negative expectancy and a losing profit factor (0.89)**
over this window. SPY buy-and-hold was −5.5% over the same 63 days (a falling
market), so the strategy's −3.4% only "beats" SPY by being under-invested in a
downtrend — not by having an edge.

### 4. This corroborates the repo's own honest numbers
`data/optimization/optimization_2025-12-29.json` best result: +6.8% total return,
38% win rate, PF 1.29, **Sharpe 0.11** — and that is the *in-sample grid-search
winner* (overfit-prone). `ACTIONABLE_100X_STRATEGY.md` itself computes a **negative
Kelly** and states the strategy is "signal-limited, not capital-limited." For
comparison SPY buy-and-hold over the analogous 2025 window (Feb→Sep) was **+9.1%
(~+15% annualised)** — the bar the strategy must clear and does not.

### 5. The project already pivoted away from trading edge
Recent commits ("SaaS product overhaul — landing/pricing/login/register", "AI
Confidence dashboard") show effort has shifted to selling signals as a subscription
product. That is a software business, not a trading edge, and does not make the
bot's own P&L positive.

## What I did
1. Bootstrapped `data/operator_journal/` — `MANDATE.md`, `entries/`, `LATEST.md`.
2. Wrote `scripts/evaluate_signals.py`: the missing measurement layer. It closes
   the outcome-tracking loop offline and cannot mislead (it reports the realistic
   non-overlapping number and benchmarks SPY). Dependency-light, proxy-aware, tested.
3. Ran the ground-truth evaluation above.

## What I verified
- `python -c py_compile` on the new script: **OK**.
- Ran the evaluator end-to-end on all 48 symbols / 1,986 signals; output reproduced
  in §3 above (real Yahoo daily data, 5bps costs).
- Confirmed multi-symbol Yahoo fetch reliability (200s) and that stooq now serves a
  JS anti-bot challenge (documented in MANDATE §6 so the next instance doesn't waste
  time on it).

## What I did NOT do and why
- **Did not touch `risk/`** — no changes there this session.
- **Did not change strategy parameters, ML, or scanner logic.** Tuning an unproven,
  marginal-to-negative edge without out-of-sample validation would be motion, not
  progress, and I cannot run the live system to validate. Measurement first.
- **Did not enable any live/auto trading.** Paper-only respected.
- **Branch note:** my task prompt asked for a branch `operator/2026-09-07`, but the
  session's hard git constraint designates `claude/adoring-feynman-9j4ymz` and
  forbids pushing elsewhere without explicit permission. I honored the hard
  constraint. Flagging the discrepancy for the human to reconcile.

## Recommendation / next action
1. **Do not deploy real capital.** The burden of proof in MANDATE §2.7 (positive
   edge net of costs, beating SPY, out-of-sample) is **not met**.
2. **Fix measurement before anything else.** Make executed-trade outcome tracking
   real and committed, so profitability is knowable on an ongoing basis. Run
   `scripts/evaluate_signals.py` each session against fresh signal history and log
   the non-overlapping expectancy trend here.
3. **Run a proper out-of-sample / multi-window evaluation.** One 2-month falling
   market is not enough to conclude either way. Extend the evaluator to walk
   multiple windows (and both bull and bear regimes) before any go/no-go.
4. **Escalation candidate:** if additional windows keep the non-overlapping
   expectancy ≤ 0, recommend winding down the trading ambition and being explicit
   that the value (if any) is in the software/SaaS layer, not a market edge.
5. **Retire or quarantine the ML ensemble + regime detector** from any decision path
   until they demonstrate AUC materially above 0.5 out-of-sample. Right now they
   add risk (overfit) and complexity for zero measured benefit.

## Open questions / risks
- I cannot see the live system's *actual* realised P&L (no DB/container). The
  committed snapshots may be stale. The human should confirm the live paper account's
  real equity curve — that is the ground truth I lack.
- Signals are ~85% long (1687/1986) into a falling window — is there a directional
  bias/bug, or just the regime? Worth checking next session.
- Daily-bar approximation with pessimistic path assumption may understate the raw
  per-signal edge slightly; it does not change the non-overlapping conclusion.
