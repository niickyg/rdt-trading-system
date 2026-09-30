# 2026-09-30 — First operator session: infrastructure bootstrap + honest baseline assessment

**Operator:** autonomous run (stateless).
**Branch:** `claude/adoring-feynman-fo2odx` (see "Branch discrepancy" below).
**Verdict this session: RED** — best honest evidence shows the strategy does
**not** beat SPY buy-and-hold; it loses to it by roughly 5x over the tested
window, before honest costs. There is essentially no live track record.

---

## 0. Situation on arrival

The scheduled task told me to read `data/operator_journal/MANDATE.md`,
`data/operator_journal/LATEST.md`, `POST_MORTEM_RRS.md`, and the 3 most recent
journal entries. **None of these files existed** — not in the working tree, not
on `main`, not anywhere in git history (`git log --all -- ...` returns nothing).

So either this is the genuine first run, or prior runs never committed/pushed
their journals. I proceeded as a bootstrap:

- Created `data/operator_journal/` + `entries/`.
- Reconstructed `MANDATE.md` from the constraints embedded in the scheduled-task
  prompt (provenance noted at the top of that file). It invents no new authority.
- Wrote this entry and `LATEST.md`.

**Branch discrepancy (for the human):** the scheduled prompt asks for a branch
`operator/2026-09-30`. The harness-level git requirements for this session
explicitly designate `claude/adoring-feynman-fo2odx` and say never to push to a
different branch without explicit permission. The harness rule wins, so I stayed
on and pushed to `claude/adoring-feynman-fo2odx`. If you want the
`operator/YYYY-MM-DD` convention, grant it explicitly and I'll follow it.

## 1. Is there a real P&L track record? — No.

From `data/signals/signal_metrics.json`:

- `total_scans`: 880, `total_signals`: 120
- **`total_outcomes`: 2** (1 target hit, 1 stop-out)
- `last_scan_at`: **2026-03-05** — the signal data is ~7 months stale.
- Signal direction is wildly skewed: **119 short / 1 long**.

Two tracked outcomes is not a track record; it's noise. We cannot make any
profitability claim from live/paper trading. A 119:1 short bias also suggests
either a scanner bias or a market regime in which a short-heavy book is
dangerous — worth a dedicated audit later, but moot until the core edge exists.

## 2. Best available backtest — and what it really says

The only quantitative performance evidence is the 2-year walk-forward backtest
summarized in `CLAUDE.md` (`scripts/run_walkforward_v2.py`), window ~Feb 2024 –
Nov 2025, $25K capital:

| Metric | Baseline | Old Filters | **RDT Filters** |
|---|---|---|---|
| Total Return | 3.3% | 5.1% | **6.9% ($1,716)** |
| Annualized | 1.6% | 2.5% | **3.4%** |
| Win Rate | 47.7% | 47.9% | **49.5%** |
| Profit Factor | 1.12 | 1.12 | **1.24** |

**Two cost-honesty problems make even these numbers optimistic:**

1. `scripts/run_walkforward_v2.py` models **no commissions and no slippage** at
   all (grep confirms). The headline 6.9% / 3.4%-annualized is a **gross** figure.
2. `scripts/run_backtest.py` *claims* 0.1% slippage but **does not apply it** to
   fills — lines 186-188: "we document the expected slippage impact rather than
   modifying data." Entries/exits use daily close prices. VWAP and first-hour
   filters (2 of the 4 RDT gates) **cannot even be simulated** on daily bars, so
   the backtest isn't testing the live system.

With ~279 round-trip trades, honest costs (commission + realistic slippage,
call it ~0.1–0.2% round trip) plausibly erase 1–3 percentage points of the
already-thin 6.9%, pushing net return toward low-single-digits or negative.

## 3. The benchmark: SPY buy-and-hold over the same window

Pulled from IBKR (`get_price_history`, SPY contract 756733, weekly):

- 2024-02-05 close: **501.20**
- 2025-11-24 close: **683.39**
- Price return: **+36.4%**; add ~2% dividends → **~+38% total**, **~19%/yr**.

**Head-to-head, same window:**

| | Strategy (RDT, gross) | SPY buy-and-hold |
|---|---|---|
| Total return | ~6.9% (before costs) | ~+38% |
| Annualized | ~3.4% (before costs) | ~19% |

SPY buy-and-hold beat the strategy by **~5x**, on absolute return, over the exact
window the strategy was tuned on — and the strategy figure is pre-cost. On
risk-adjusted terms the gap narrows (the strategy's worst day was -$257 vs
baseline -$404) but does not close: a ~19%/yr benchmark vs a ~3.4%/yr gross
strategy is not a Sharpe story that rescues it over this period.

## 4. The project's own documents already admit this

`ACTIONABLE_100X_STRATEGY.md`, `WEALTH_STRATEGY_100X.md`, `WEALTH_OPTIMIZATION.md`:

- They state the trading strategy "mathematically caps returns around 7%" at
  reasonable risk, with a 37–38% win rate and PF ~1.29.
- Their path to the "100X" / "$25k→$50k" goal is **not trading** — it's pivoting
  40–60% of the target onto **selling a signal service / API / education**. That
  is a SaaS business plan, not evidence the bot is profitable.
- `QUICK_START_100X.md` recommends `MAX_RISK_PER_TRADE=0.03` (3%), above the
  documented Config C of 1.5% — i.e. raising risk to chase returns, which the
  mandate explicitly distrusts. **Flagging: do not adopt this without proof.**

Note the win-rate inconsistency across artifacts (backtest says ~49.5%, the
optimization docs say ~38%). That itself signals the numbers are not being
tracked from one honest, reproducible source of truth.

## 5. Verdict and recommendation

**RED.** The best honest evidence says this strategy does not beat SPY
buy-and-hold; over its own tuning window it lost to it ~5x, before costs, and
there is no live track record to argue otherwise. Continuing to add features,
filters, or ML layers is not warranted — the project has already accreted many
(87 ML features, 4 filter gates, intermarket analysis, options) and the return
is still ~3.4%/yr gross.

**Recommended escalation to the human — pick a lane:**

1. **Wind down the trading premise.** For a $25K account with no demonstrated
   edge, SPY (or a simple momentum-on-index) buy-and-hold dominates on return,
   effort, and risk. This is the evidence-based default.
2. **If trading continues, the only justified next work is proof, not features:**
   - Add honest commission + slippage to `run_walkforward_v2.py` and re-run.
   - Validate strictly **out-of-sample** (tune on 2024, test untouched on
     2025–2026), including the Feb–Mar 2026 window where the live scanner went
     119:1 short.
   - Compare net-of-cost equity curve directly against SPY on the same axes.
   - If net-of-cost, out-of-sample return does not clear SPY, stop.
3. **The "signal service" business** is a separate product decision, not a
   profitability result for the bot. Don't let it launder the trading verdict.

## 6. What I did NOT do (and why)

- Did **not** touch `risk/` (no change needed; would require flagging).
- Did **not** change trading logic or parameters. Optimizing a strategy that
  loses to buy-and-hold is motion without progress; the mandate calls for the
  honest verdict first.
- Did **not** run the walk-forward myself this session (network/data + engine
  time). The cost-honesty gaps are visible in the code and the conclusion is
  already decisive from the repo's own numbers + the IBKR SPY benchmark. Adding
  honest costs to the backtest is the concrete first task I'm leaving my
  successor — it can only make the RED verdict more certain.

## 7. Next session should start here

- Read this entry first.
- If the human has chosen a lane (wind-down vs prove-it), follow it.
- If no direction yet: implement honest cost modeling in
  `scripts/run_walkforward_v2.py`, re-run, and record the net-of-cost number
  against SPY. Do not add features.
