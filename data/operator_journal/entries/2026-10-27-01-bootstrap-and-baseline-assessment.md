# 2026-10-27 — Entry 01: Bootstrap + Baseline Profitability Assessment

- **Operator run:** rdt-bot-operator (scheduled), fired 2026-10-27 12:54 UTC
- **Branch:** `operator/2026-10-27` (based on `origin/main` @ `ae4350a`)
- **Start SHA:** `ae4350afa293a7a104d7bd66f238eb669151460b`
- **Instance:** first run with no prior journal (see Flags)

---

## What I found before I could start (critical)

The scheduled task's non-negotiable first action was to read
`data/operator_journal/MANDATE.md`, `LATEST.md`, `POST_MORTEM_RRS.md`, and the 3
most recent journal entries. **None of these existed** — not on disk, not in
`.gitignore`, not on any branch, not anywhere in git history (`git log --all`).
`main`, `claude/adoring-feynman-x5v26o`, and this operator branch all point at the
same commit `ae4350a`.

So this is either the genuine first run of the operator, or the schedule was
activated before its journal infrastructure was ever committed. I could not
follow "the Protocol in MANDATE.md precisely" because there was no MANDATE.

**Decision:** rather than invent a trading protocol and treat it as my own
constitution, I **bootstrapped the journal infrastructure** by faithfully
transcribing the constitution from the scheduler prompt itself (the only
authoritative source) into `MANDATE.md`, and reconstructed `POST_MORTEM_RRS.md`
from primary sources already in the repo. Both files are marked
**"BOOTSTRAPPED — PENDING HUMAN CONFIRMATION"** at the top. **The human must
review and correct them.** I made no trading-logic changes this session.

## Assessment — does the bot make money / beat SPY?

I did not trust the docs; I ran a fresh walk-forward backtest on current data and
benchmarked it against SPY buy-and-hold over the identical window.

**Fresh walk-forward** (`scripts/run_walkforward.py`, 30 mega-caps + SPY + 11
sector ETFs + VIX, daily 2025-08-28→2026-10-26; full log saved at
`data/operator_journal/walkforward_2026-10-27.log`):

| Window | Baseline | Filtered |
|---|---|---|
| W1 2026-01-08→05-18 | +5.76% | +5.82% |
| W2 2026-05-19→09-25 | +0.01% | −0.42% |
| W3 2026-09-28→10-26 | −0.87% | −0.06% |
| **Aggregate (201 td)** | **+4.90% (~6.1% ann.)** | **+5.35% (~6.7% ann.)** |

**SPY buy-and-hold, same window: +11.87% (~15.1% annualized).** (Computed
directly from the cached SPY parquet: 684.17 → 765.37.)

**Verdict: the bot underperforms SPY buy-and-hold by more than 2x over the same
recent window.** Supporting evidence:

1. **One-window edge.** The entire ~5–6% came from W1 (Jan–May). The strategy was
   flat-to-negative for the next ~5 months while SPY kept rising. Not robust.
2. **No cost model.** `backtesting/` has zero commission/slippage terms (grep
   confirmed). The ~6.7% is **gross**; net of honest costs on 90–170 trades/yr it
   is lower. Watchlist is 30 surviving mega-caps (survivorship bias).
3. **Filters hurt quality.** Filtered vs baseline: profit factor 1.34 vs 1.54,
   max drawdown doubled ($900 vs $450), worst day worse. Filters mostly traded
   more, not better.
4. **ML is noise.** `models/ensemble/metrics.json`: CV AUC 0.54 (random = 0.50),
   train AUC 0.99 (overfit). Regime detector silhouette −0.087 (worse than
   random), degenerate states. CLAUDE.md agrees: "ML is advisory-only."
5. **Live signals are degenerate.** `signal_metrics.json`: 119 short : 1 long,
   and only **2 signals ever had a tracked outcome.** Effectively no live record.
6. **The project's own docs concede it.** The "100X/WEALTH" docs admit trading
   caps at ~7% and reach their target only by selling signals to subscribers +
   adding leverage — i.e. not by trading alpha.

This matches the scheduler mandate's trigger clause: *"If the evidence keeps
saying no strategy works, say so in the journal and recommend escalation or
wind-down."*

## Decision & rationale

- **Did:** bootstrapped `MANDATE.md`, `POST_MORTEM_RRS.md`, this journal, and
  `LATEST.md`; ran and preserved a fresh, cost-aware-caveated backtest + SPY
  benchmark as the first real datum in the journal.
- **Did NOT:** change any strategy, filter, risk, or model code. On a first run,
  with no confirmed mandate and a strategy whose problem is *absence of edge*
  (not a mis-tuned parameter), a blind code change would add risk and noise, not
  value. The honest, high-leverage action was to establish state + a truthful
  baseline and put a clear question to the human.

## Verification

- `git log --all` / `.gitignore` / disk search: confirmed journal truly absent.
- Backtest ran to completion (exit summary present in log); SPY benchmark
  computed independently from the cached parquet with an isolated interpreter.
- Confirmed absence of cost modeling via grep over `backtesting/`.
- No source files modified → nothing to py_compile. Only added markdown + a log.

## Recommendation for the next instance

1. **Resolve the mandate question first.** Confirm with the human whether the
   bootstrapped `MANDATE.md`/`POST_MORTEM_RRS.md` are acceptable, and whether a
   real mandate exists elsewhere. Do not do strategy surgery until the
   constitution is confirmed.
2. **Make the backtest honest before trusting any number:** add commission +
   slippage to `backtesting/engine_enhanced.py`, test on a non-survivorship
   universe, and always print the SPY buy-and-hold benchmark alongside results.
   This is the single highest-value code change and is cheap.
3. **Run the falsification, not another optimization.** Segment results by regime
   (bull/bear/chop) to test the one surviving hypothesis — that the strategy
   earns its keep only in down/choppy markets. If it fails that too, escalate a
   wind-down recommendation to the human explicitly.
4. **Stop adding complexity/leverage.** The evidence says neither creates edge.

## Flags

- **ESCALATION:** On current evidence the bot does not beat SPY buy-and-hold and
  may be net-unprofitable after costs. Not a wind-down call yet (one more honest,
  cost-aware, regime-segmented pass is owed), but the human should be aware.
- **MANDATE MISSING:** constitution and journal were absent; I bootstrapped them.
  Needs human confirmation.
- **JOURNAL PERSISTENCE:** this journal lives on `operator/2026-10-27`. It will
  only survive to the next run if the human merges it somewhere the next fresh
  checkout will see (e.g. `main`). Flagging until resolved.
- **risk/ directory:** NOT touched this session.
- No live-trading settings touched. PAPER_TRADING / AUTO_TRADE unchanged.
