# POST-MORTEM: RDT Trading System (RRS Strategy)

> **Status of this document.** The autonomous operator task expects a pre-existing
> `POST_MORTEM_RRS.md`. None existed in this repository on any branch or anywhere in
> git history as of 2026-11-16. This post-mortem was **reconstructed from hard evidence
> gathered during the 2026-11-16 operator session** — the live IBKR paper account
> performance series, the live account summary, and the strategy documents checked into
> the repo. It is factual where it cites those sources and explicitly marked where it
> infers. It should be corrected/extended by the human if a fuller history exists
> off-repo.

---

## 1. The bottom line (ground truth, 2026-11-16)

Pulled live from the connected IBKR paper account this session:

| Metric | Value | Source |
|--------|-------|--------|
| Net liquidation value | **$5** | `get_account_summary` |
| Open positions | **0** | `get_account_positions` |
| Trades in last 90 days | **0** | `get_account_trades(DAYS_90)` |
| Performance since inception (2026-02-25) | **−61.5% TWR** | `get_pa_performance_all_periods` |
| SPY buy-and-hold, same window (~686 → 754) | **≈ +10%** | `get_price_history(SPY)` |

**The bot underperformed a passive SPY hold by ~70 percentage points, then went
dormant.** The account has sat untouched at ~$5 since early August 2026 (≈3.5 months).

### NAV trajectory (from the TWR series)

```
2026-02-26   $50      ← inception funding
2026-03-03   $21      ← down 58% in the first week
2026-03-20   $521     ← ~$500 cash top-up added
2026-03-24   $521     ← peak
... steady bleed ...   $506.5 → $492 → $477.5
~2026-08     $5        ← drops to $5 and stays flat to today
```

The drop from ~$477 to $5 does not show up as a proportional TWR cliff, which means it
was most likely a **cash withdrawal** (money out), not a single trading loss. The
**trading** performance — the part the strategy is responsible for — is the −61.5% TWR
bleed from $50/$521 downward. Either way the account is now non-functional for trading
and has generated no activity for months.

---

## 2. Why it lost (root cause)

The repository's own strategy analysis already diagnosed the core problem before this
session. From `ACTIONABLE_100X_STRATEGY.md` and `WEALTH_STRATEGY_100X.md` (both checked
into the repo):

- Best honest backtest: **~6.8% annual return**, profit factor ~1.29, win rate ~38%.
- **Kelly criterion computed as slightly NEGATIVE** (`kelly ≈ -0.02`). The authors
  state plainly: *"the current edge is marginal … increasing position size actually
  increases risk of ruin without improving returns."*

A strategy with a negative/zero Kelly has **no exploitable edge net of costs**. That is
the whole story: the RRS momentum strategy, as implemented, does not reliably beat its
own transaction costs and slippage. The walk-forward table in `CLAUDE.md` shows the
"best" configuration returning 6.9% over two years (3.4% annualized) — below SPY's
long-run average and far below the risk taken.

### The aggravating factor: chasing 100x

Rather than accept a no-edge result, the repo contains two documents
(`ACTIONABLE_100X_STRATEGY.md`, `WEALTH_STRATEGY_100X.md`, plus `QUICK_START_100X.md`,
`WEALTH_OPTIMIZATION.md`) pushing for a **14.6x improvement to a 100% annual return**
via leverage, position concentration, options, crypto/futures expansion, and a paid
signal-service side business. Applying leverage and concentration to a negative-Kelly
strategy is a textbook path to ruin — and the account trajectory ($521 → $5) is
consistent with exactly that outcome. **This directly contradicts the r/RealDayTrading
philosophy of taking small, high-probability, risk-managed edges and compounding
slowly.**

---

## 3. What the governance system looked like (and didn't)

The operator mandate assumes a persistent journal at `data/operator_journal/`
(`MANDATE.md`, `LATEST.md`, `entries/`) to carry memory and hard constraints across
stateless sessions. **None of it existed in the repo as of 2026-11-16.** A stateless
operator with no constitution, no history, and an unfunded dead account is structurally
unable to operate safely — it is the governance gap, as much as the strategy, that
needs fixing first. The 2026-11-16 session bootstrapped a *proposed* scaffold (see
`data/operator_journal/MANDATE.md`) pending human ratification.

---

## 4. Honest recommendation

1. **Do not add capital or leverage to this strategy as-is.** The edge is, by the
   repo's own math, not positive after costs. Re-funding and re-enabling it would
   repeat the −61.5% result.
2. **Delete or quarantine the "100X" / "wealth" documents** as operating guidance. They
   point the system at its most dangerous failure mode.
3. **Re-establish a positive, costed edge in research *before* any live/paper trading
   resumes.** Burden of proof: out-of-sample, walk-forward, net of realistic commission
   and slippage, with a *positive* Kelly and a Sharpe that beats SPY buy-and-hold on a
   risk-adjusted basis. If that cannot be shown, the honest conclusion is to **wind the
   trading down** and keep only the research harness.
4. **Human decision required** on (a) ratifying the operator mandate, (b) whether to
   re-fund the paper account for forward testing, and (c) wind-down vs. continued
   research. See the 2026-11-16 journal entry.

---

*Reconstructed 2026-11-16 by the autonomous operator session from live account data and
in-repo documents. Correct freely if off-repo history exists.*
