# POST-MORTEM: RRS Strategy — Why The Bot Is Where It Is

> Bootstrapped by operator instance #001 (2026-08-10). The scheduled prompt
> referenced this file as pre-existing history; it did not exist. This is an
> honest reconstruction from the codebase, `CLAUDE.md`, and live data pulled
> this session. Future instances: append, correct, and date your additions.

## The one-paragraph summary

The system implements the r/RealDayTrading "Real Relative Strength" methodology
with enormous engineering breadth — a scanner with 4 sequential filter gates,
VIX/sector/regime/intermarket overlays, 87 ML features, an options module, a
multi-broker execution layer, and a walk-forward backtester. Despite all of it,
**the strategy's own best documented backtest returns ~3.4%/yr, while SPY buy-
and-hold returned ~18.6%/yr over the same recent period.** The bot has been
optimizing the wrong objective (beat a no-filter baseline, maximize win-rate)
instead of the only one that matters (beat buy-and-hold net of costs).

## Evidence (instance #001, real numbers)

- **SPY buy-and-hold** (real IBKR monthly closes, conid 756733):
  Aug 2024 $563.68 → Aug 2026 $773.26 = **+37.2% total, ~18.6%/yr**.
- **Bot documented "best" (RDT filters), 2yr walk-forward** (`CLAUDE.md`):
  $1,716 on $25K = **6.9% total, 3.4%/yr**, 279 trades, PF 1.24, WR 49.5%.
- **Gap:** holding SPY beat the best active config by **~$7,579** over 2 years —
  and the backtest does not appear to subtract commissions/slippage, so the real
  gap on 279 round-trip trades is larger.
- **Live paper account** (IBKR MCP this session): `net_liquidation = $5`, zero
  positions. Not the $25K DUP995654 account `CLAUDE.md` describes. The bot is
  effectively unfunded and not trading. Whatever is "live" is dead capital.

## Root causes

1. **Wrong benchmark.** The backtester compared filtered vs unfiltered strategy
   variants and never once computed SPY buy-and-hold. "RDT filters beat old
   filters by $449" is true and irrelevant when both lose to holding the index.
   (Instance #001 added a buy-and-hold benchmark to `run_walkforward_v2.py`.)
2. **Feature-count as progress.** The history (see `CLAUDE.md` Murphy sections,
   audit sections) is a march of added indicators, gates, and ML features. None
   of it is validated against the buy-and-hold bar. The ML section even admits
   the models are "advisory-only" and the exit predictor is "43.3% accuracy —
   SKIP."
3. **Low, positive-but-tiny edge on a mean-market.** WR ~49.5% and PF ~1.24 in a
   backtest that ignores costs is, realistically, a coin flip once you pay the
   spread. Day-trading single-name momentum during a strong-trend bull market
   underperforms simply owning the trend.
4. **Structural drag.** Active trading incurs costs (commissions, slippage,
   taxes, PDT constraints on <$25K accounts) that buy-and-hold does not. The
   strategy must overcome ~18.6%/yr PLUS those costs to win. Nothing in the
   evidence suggests it can.

## What this means for the mission

The mandate bar is "beat SPY buy-and-hold net of costs." On all evidence
available at instance #001, the strategy does not, and the accumulated
complexity has been chasing the wrong target. Before adding anything, a future
instance should **prove there is any real, cost-surviving edge at all** — e.g.
run the (now benchmark-aware) walk-forward with realistic per-trade costs and
see whether ANY config clears buy-and-hold. If nothing does across honest tests,
the correct recommendation is wind-down of active trading, not another filter.

## Open threads for the next instance

- Re-run `scripts/run_walkforward_v2.py` (now prints the SPY benchmark) once a
  reliable data feed is wired (yfinance is 429-throttled; consider feeding the
  backtester from the IBKR MCP `get_price_history`).
- Add explicit commission + slippage to the backtest engine and re-check whether
  the tiny edge survives costs. If not, that is the headline finding.
- Reconcile the $5 live account vs the documented $25K account. The bot may not
  be running at all.
