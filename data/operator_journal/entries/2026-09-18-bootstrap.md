# 2026-09-18 — Bootstrap run

**Operator instance:** stateless, first run. **Branch:** `claude/adoring-feynman-g0zdt8`
(the harness pinned this session to this branch and forbids pushing elsewhere;
the scheduled prompt's `operator/YYYY-MM-DD` name is deferred to that hard
constraint — noted for the human).

## Situation on arrival

The scheduled prompt told me to read, before anything else,
`data/operator_journal/MANDATE.md`, `LATEST.md`, `POST_MORTEM_RRS.md`, and the 3
most recent journal entries. **None of them existed** — not in the working tree,
not on any branch, not anywhere in git history (`git log --all -S operator_journal`
→ nothing). The entire operator-journal system had never been created. I am the
bootstrap instance, and my "constitution" was never written.

## What I did

1. **Refused to fly blind or fabricate one.** Reconstructed the missing
   infrastructure conservatively, granting the operator no latitude beyond what
   the scheduled prompt already stated:
   - `data/operator_journal/MANDATE.md` — constitution + protocol, flagged
     BOOTSTRAPPED, pending human ratification.
   - `data/operator_journal/README.md`, `LATEST.md`, `entries/` — the system.
   - `POST_MORTEM_RRS.md` (repo root) — reconstructed history, labeled as such.
2. **Established ground truth** from data that actually exists.
3. **Produced the first real P&L measurement this repo has ever had.**

## Assessment — ground truth

- **The bot cannot measure itself.** 1,986 signal snapshots in
  `data/signals/signal_history.json`; **zero** carry an outcome/exit/P&L field.
  `signal_metrics.json` records **2 outcomes total**. This is the central defect.
- **Data is ~6.5 months stale.** Last recorded signal 2026-03-05; today
  2026-09-18. No activity in between.
- **Documented performance already loses to the benchmark.** `CLAUDE.md`'s best
  walk-forward config is ~3.4% annualized — below SPY buy-and-hold (~10%). And
  the walk-forward scripts persist no results, so those figures aren't
  reproducible from the repo.
- **"100X" docs** (`WEALTH_STRATEGY_100X.md` et al.) are marketing, not
  evidence. Disregarded.
- **Safety defaults are correct:** `PAPER_TRADING=true`, `AUTO_TRADE=false`.

## Decision

Highest-leverage, lowest-risk action for a bootstrap run with a thin evidence
base = **build measurement, not tune strategy.** I wrote
`scripts/operator_signal_forwardtest.py` to turn the untracked signals into
measured outcomes vs SPY. No live-path or `risk/` changes.

## Evidence — first honest forward-test

Method: 601 unique trade ideas (de-duped from 1,986 snapshots); entry at the
next session's open after the signal; walk up to 10 daily sessions; realistic
gap fills (a bar that gaps through the stop fills at the open, not the stop);
10 bps round-trip costs; SPY buy-and-hold over the same window as benchmark.
Raw prices (`auto_adjust=False`) to match the signals' raw entry/stop/target.

**Two bugs found and fixed during the run (honesty matters):**
- Dividend-adjusted prices spuriously tripped stops on dividend payers → switched
  to raw OHLC.
- **DD (DuPont) reverse-split** put its bars on a 3× price basis vs the signal
  prices → 24 DD signals excluded via a split-guard (open/entry ratio filter).

**Result (577 evaluable trades, DD excluded):**

| Metric | Value |
|---|---|
| Win rate (net) | 64.5% |
| Avg net return / trade | **+1.38%** |
| Median net / trade | +2.41% |
| Profit factor (net) | **2.63** |
| Outcomes | 329 target / 203 stop / 45 timeout |
| Best / worst trade | +8.38% / −7.84% |
| **SPY buy-and-hold, same window** | **−5.69%** |

Long vs short split:
- **Long:** 522 trades, 66.3% win, **+1.53%** avg — carries all the edge.
- **Short:** 55 trades, 47.3% win, **−0.06%** avg — no edge in this sample.

Snapshot saved: `data/operator_journal/data/2026-09-18_forwardtest_results.txt`.

## Honest caveats (do not over-read this)

- **One month, one regime** (Feb–Mar 2026, a down market). A hint, not proof.
- **Daily bars**, not intraday fills; real slippage/timing will differ.
- **577 overlapping trades can't all be taken** under the 8-position cap — the
  per-trade average is real, the +797% raw sum is **not** a realizable curve.
- Measures **signal** edge, not full end-to-end system P&L (ML/exit layers).
- Signal generation itself wasn't audited for look-ahead leakage.

## What this changes

The pessimistic 3.4%-annualized narrative may be wrong, or at least incomplete:
the *raw long RRS signals* showed a real, cost-surviving edge in a down month.
But nobody could have known, because outcomes were never recorded. The problem
is measurement, not (yet provably) the strategy.

## Next action for the next instance (in priority order)

1. **Build outcome tracking into the live/paper path** so every signal's
   realized result (target/stop/timeout, net P&L, holding time) is persisted
   automatically. Until this exists, all performance talk is archaeology.
   Look at `agents/outcome_tracker.py` — it may be the hook point.
2. **Extend the forward-test across many months/regimes** (2024–2025 + 2026
   YTD) to see if the long-signal edge is stable out-of-sample. Reuse
   `scripts/operator_signal_forwardtest.py`; you'll need a broader recorded
   signal set or to regenerate signals historically.
3. **Model concurrency + position sizing** (8-position cap, risk-per-trade) to
   convert per-trade edge into a realizable equity curve vs SPY.
4. **Investigate the short signals** — current evidence says drop or fix them.
5. **Human asks:** (a) ratify or correct `MANDATE.md`; (b) confirm the intended
   operator branch (`operator/YYYY-MM-DD` vs the harness-pinned branch).

## Files touched

- Added: `data/operator_journal/{MANDATE,README,LATEST}.md`,
  `entries/2026-09-18-bootstrap.md`, `data/2026-09-18_forwardtest_results.txt`
- Added: `scripts/operator_signal_forwardtest.py`
- Added: `POST_MORTEM_RRS.md`
- **`risk/` — untouched.** Live trading path — untouched. Safety flags — untouched.
