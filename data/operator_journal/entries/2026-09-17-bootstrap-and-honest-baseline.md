# 2026-09-17 — Bootstrap + honest baseline vs SPY

**Operator instance:** #1 (first run). **Branch:** `operator/2026-09-17`.
**Session type:** Bootstrap the journal + establish ground-truth assessment.

---

## What I found on arrival

The scheduled prompt told me to read `data/operator_journal/MANDATE.md`,
`LATEST.md`, `POST_MORTEM_RRS.md`, and the 3 most recent journal entries **before
doing anything else.** None of those files existed. This is the first operator
run; there is no prior state, no journal, no mandate, no post-mortem. So Step 0
this session was to **create the constitution the prompt assumes exists**, faithful
to the constraints handed down in the prompt (paper-only, no AUTO_TRADE, don't
touch `risk/` silently, human merges, journal every session, honesty over
optimism, no revenue-theater drift).

Files created this session:
- `data/operator_journal/MANDATE.md` — the constitution + protocol.
- `POST_MORTEM_RRS.md` — history of why the bot is where it is.
- `data/operator_journal/entries/` — this entry.
- `data/operator_journal/LATEST.md` — pointer to this entry.
- `scripts/benchmark_vs_spy.py` — new tool: one-command "strategy vs SPY b&h, net
  of a cost band." (Read-only; modifies no engine code.)

## What I assessed (with fresh evidence, not docs)

Installed `yfinance`/`pandas`/`pyarrow` (not preinstalled) and ran the project's own
`scripts/run_walkforward_v2.py` on live data (2024-01 → 2026-09), then computed the
SPY buy-and-hold benchmark over the **same** windows. yfinance works from this
environment via the HTTPS proxy.

### The numbers (fresh, this session)

Walk-forward, 6 windows, 504–510 traded days, $25k, best config (RDT filters):

| | Baseline (no filters) | Old filters | **RDT filters (best)** |
|---|---|---|---|
| Total return | $1,458 (5.8%) | $1,270 (5.1%) | **$1,848 (7.4%)** |
| Annualized | 2.9% | 2.5% | **3.7%** |
| Win rate | 46.2% | 45.1% | 46.3% |
| Profit factor | 1.22 | 1.13 | 1.25 |
| Trades | 236 | 432 | 281 |

**SPY buy-and-hold, same window:** **+47.4% (+$11,845 on $25k)**, ~20–24% annualized.

### The verdict (`benchmark_vs_spy.py` output)

```
--- RDT STRATEGY (best config), net of cost band ---
 Cost/round-trip     Net $     Net %   vs SPY in-window
             $0    $+1,848    +7.39%           $-9,997
             $2    $+1,286    +5.14%          $-10,559
             $5      $+443    +1.77%          $-11,402
            $10      $-962    -3.85%          $-12,807   <- goes NEGATIVE
            $20    $-3,772   -15.09%          $-15,617
```

**The strategy loses to SPY buy-and-hold by ~$10,000 even at ZERO transaction
cost.** And this is generous: **no backtest engine models any cost** —
`grep -ic slippage|commission|spread` returns 0 across `engine.py`,
`engine_enhanced.py`, `engine_intraday.py`. At a modest $10/round-trip the strategy
is outright negative.

## What this means

1. **The core RRS signal is weak, not the filters.** The unfiltered baseline still
   makes only 2.9%/yr. Years of filter-tuning moved the number from ~2.8% → ~3.7%;
   none of it approaches the benchmark. This is consistent with the project's own
   admission that Kelly is ≈ −0.02 (`ACTIONABLE_100X_STRATEGY.md`).
2. **The comparison is fair.** The strategy was exposed to the same ~504 market
   days SPY b&h was measured over (in-window calc), so this is not a bull-market
   artifact of picking a lucky span — it's the same span, same market.
3. **Prior effort drifted off-mission.** The most recent commits are a "SaaS
   product overhaul" (landing/pricing/onboarding). Selling a negative-edge signal
   to subscribers monetizes a product that loses money. Documented in the
   post-mortem; flagged here as the trap to avoid.

## Decision this session

Per the mandate, I did **not** manufacture a tuning change to nudge a metric — that
would violate "no manufactured progress." A bootstrap session's highest-value output
is (a) the honest infrastructure + baseline, and (b) a reusable measurement tool so
every future instance is forced to look at the benchmark. Both delivered.

I deliberately did **not** touch `risk/`. I changed no engine/strategy logic. The
only code added is a read-only benchmark script.

## Recommendation for instance #2 (and escalation note to the human)

The evidence is close to a wind-down trigger but not conclusive: extensive *tuning*
has been tried, but a few *structurally different* hypotheses have not been honestly
tested. Before recommending wind-down, instance #2 should test **exactly one** of
these, each falsifiable in a backtest with the SPY benchmark shown:

1. **SPY-trend participation vs stock-picking.** The one robust fact is SPY rose
   ~47%. Test a trivial rule: hold SPY (or go flat) based on the 50/200 EMA trend
   gate the bot already computes. If dumb trend-participation beats the RRS
   stock-picking, that's the real finding — the machinery is destroying value
   versus just riding the index.
2. **Does RRS have any forward IC?** Use `research/factor_tester.py` to check
   whether raw RRS predicts forward returns at all. If IC ≈ 0, no gating can help
   and wind-down is the honest call.
3. **Longs-only in uptrends.** The short side may be a structural drag in a bull
   market. Measure long-only.

**Escalation flag for the human:** As of this baseline, the bot does not come close
to its mandate (beat SPY b&h net of costs) and is negative under realistic costs. If
instances #2–#3 also fail to find a structurally different edge, the mandate-compliant
recommendation is to **wind down live pursuit of this strategy** and either (a) pivot
the account to index participation, or (b) stop. Do not deploy real capital to this
strategy in its current form. Continue paper-only.

## Reproduce

```bash
pip install yfinance pandas numpy pyarrow loguru pydantic pydantic-settings
python scripts/run_walkforward_v2.py     # strategy numbers
python scripts/benchmark_vs_spy.py       # strategy vs SPY, net of cost band
```

## Safety checklist

- [x] Paper-only. AUTO_TRADE untouched. No broker credentials touched.
- [x] No real orders (did not use IBKR MCP write tools).
- [x] `risk/` untouched.
- [x] No merge to main. Work pushed to `operator/2026-09-17` for human review.
- [x] Journal entry committed; `LATEST.md` updated.
