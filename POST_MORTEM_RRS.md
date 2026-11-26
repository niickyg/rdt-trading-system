# POST-MORTEM: RRS Strategy — Why the Bot Is Where It Is

> **STATUS: RECONSTRUCTED FROM REPO EVIDENCE — 2026-11-26.**
>
> The scheduled operator task instructed instances to read `POST_MORTEM_RRS.md`
> as "the history of why the bot is in its current state." No such file existed
> in the repository (any branch, any point in git history). This is a
> reconstruction built only from committed artifacts: `CLAUDE.md`, the
> root-level strategy docs, `.env.example`, and the git log. It is **not** an
> authoritative historical record. Treat its claims as *sourced inferences* and
> verify against primary data before acting on them.

## What the committed evidence actually shows

**The core strategy and its honest ceiling.** The system trades Real Relative
Strength (RRS) momentum per the r/RealDayTrading methodology, with a stack of
"market first" filter gates (SPY gate, 50/200 SMA, VWAP, multi-timeframe). The
project's own documented, honest numbers:

- `CLAUDE.md` walk-forward backtest (2 years, $25K, 6 quarterly windows,
  Feb 2024 – Nov 2025), best configuration ("RDT Filters"):
  **+$1,716 total (6.9%) over 2 years → ~3.4% annualized**, 49.5% win rate,
  profit factor 1.24, worst day −$257. ~98% of raw signals are filtered out.
- `WEALTH_OPTIMIZATION.md` baseline: **2.85% annual (~$712)**, 33.7% win rate,
  profit factor 1.23.
- `ACTIONABLE_100X_STRATEGY.md` / `WEALTH_STRATEGY_100X.md`:
  **~6.8% annual (~$1,711)**, 37–38% win rate, profit factor ~1.29.

**The uncomfortable conclusion these numbers imply.** The best honestly reported
result is roughly **3–7% annualized**. SPY buy-and-hold has historically
returned on the order of ~10%/yr. So on the repo's own evidence, **the strategy
has not been shown to beat buy-and-hold net of costs.** The strategy docs
themselves concede it is "signal-limited, not capital-limited" — i.e. simply
raising risk does not fix the edge, it only raises variance.

**The drift that followed.** Rather than concluding the edge was thin, the repo
accumulated aspirational documents aiming to turn $25K into $50K (100%/yr):
`ACTIONABLE_100X_STRATEGY.md`, `QUICK_START_100X.md`, `WEALTH_STRATEGY_100X.md`,
`WEALTH_OPTIMIZATION.md`, and `DEPLOYMENT_SUMMARY.md` (dated 2025-12-29,
"Mode: AGGRESSIVE," proposing 3% risk/trade, 6% daily loss, 20% position size).
These plans lean heavily on **non-trading revenue** (selling a "signal service,"
API access, education) to hit the headline number — an admission that the
*trading* edge alone cannot. The committed `.env.example` still uses the
conservative `MAX_RISK_PER_TRADE=0.01`; the aggressive config appears to live
only in docs, not in committed configuration.

**Development then stopped.** The last code commit is **2026-03-06** (a "SaaS
product overhaul"). This reconstruction is being written **2026-11-26** — about
8.5 months later. No commits in between.

**No persistent results.** `data/trades/` is `.gitignore`d, and the operator
runs on cloud checkouts with no access to the user's live DB. So **no real P&L
can survive to a fresh checkout**, and none is present in this one. Every
performance figure above is a *backtest*, not realized trading.

## The through-line

1. The RRS strategy's honest edge is small and **not demonstrated to beat SPY
   net of costs**.
2. Instead of accepting that, the project pivoted toward aggressive risk and a
   "sell signals to others" business model to manufacture the target return.
3. The governance/journal framework meant to keep an autonomous operator honest
   was **never committed**, so each stateless run starts blind.

## What a future instance should do with this

- Treat "beats SPY buy-and-hold, net of honest costs" as the only success bar.
- Do not trust the "100X"/"WEALTH" docs as strategy; treat them as a cautionary
  artifact of goal-induced drift.
- Get real, reproducible, out-of-sample P&L before changing anything. If it
  can't be produced, that is itself the finding — escalate.
