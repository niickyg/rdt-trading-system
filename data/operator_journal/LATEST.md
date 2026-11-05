# LATEST — most recent operator run

**Points to:** [entries/2026-11-05-genesis.md](entries/2026-11-05-genesis.md)
**Date:** 2026-11-05
**Instance:** GENESIS (first run — bootstrapped the operator infrastructure)

## TL;DR for the next stateless instance

- **Infrastructure did not exist before today.** I created `MANDATE.md`,
  `POST_MORTEM_RRS.md`, this journal, and the first entry. Read the MANDATE first;
  it is a genesis draft awaiting human ratification.
- **Honest verdict:** the system does **not** beat SPY buy-and-hold. Best
  documented out-of-sample result is +6.9%/2yr (and that is **gross** — the
  engine models zero commission/slippage/spread). SPY returned **+41%** over the
  same window. At realistic 10–20 bps costs the strategy's edge shrinks to ~+2.7%
  down to **negative**.
- **Do NOT chase thresholds.** Per the mandate, no strategy change is
  recommendable until it is measured out-of-sample **net of honest costs**.

## Start-here agenda (full detail in the entry)

1. Add transaction-cost accounting to `backtesting/engine_intraday.py` + the
   walk-forward scripts. Foundation for everything.
2. Reproduce the 2yr walk-forward **net of costs** vs the SPY benchmark. Use
   `/usr/local/bin/python3.13` (PATH `python3` can't import pandas here; deps also
   drift from the repo pins — see entry §4).
3. If net-of-cost OOS return ≤ 0 or ≪ SPY: find a genuinely new edge or recommend
   wind-down. Don't keep tuning.
4. Quantify (then likely remove) the ML/Murphy/options machinery `CLAUDE.md`
   admits adds no measurable edge.

## Hard constraints (never violate — see MANDATE §3)

Paper only · never `AUTO_TRADE=true` · never place live broker orders · flag any
`risk/` change in the journal · commit a journal entry every session · dated
`operator/` branch, never merge to main.
