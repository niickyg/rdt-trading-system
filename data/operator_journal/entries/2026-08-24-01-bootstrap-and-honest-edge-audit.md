# 2026-08-24 — Session 01 — Bootstrap + honest edge audit

## Prime-directive status
**Not closer to beating SPY — because we now know the prior "closeness" was an
illusion.** The bot's headline profitability numbers come from a backtest with
same-bar look-ahead, idealized fills, zero costs, and a survivor-picked universe.
Stress-tested arithmetically, the reported edge is internally inconsistent,
break-even at ~$8/trade of cost, and below SPY buy-and-hold even at zero cost.
Honest status: **no demonstrated edge over SPY net of costs.**

## Ground truth at start
- Branch: `claude/adoring-feynman-bulskd` (harness-designated), clean tree.
- **No operator journal existed** — no MANDATE.md, LATEST.md, POST_MORTEM_RRS.md,
  or `entries/` anywhere in git history or on `main`. This is the bootstrap run.
- LATEST said nothing (didn't exist). The scheduled mandate defined the mission:
  make the bot actually profitable net of costs vs SPY, or honestly say it can't.
- Most important open question: **is there ANY defensible evidence this bot beats
  SPY buy-and-hold net of honest costs?**

## Objective (this session)
1. Bootstrap the operator-journal infrastructure (constitution + history + memory).
2. Answer the open question with evidence, not charity.
Falsifiable success criterion: produce a reproducible artifact that either
demonstrates an honest edge or demonstrates its absence. → Achieved (absence).

## What I did
- Created `data/operator_journal/` with **MANDATE.md** (the constitution: prime
  directive, hard constraints, honest-cost definition, per-session protocol,
  anti-patterns, journal template) and this `entries/` log.
- Wrote **POST_MORTEM_RRS.md** — durable history of why the bot is where it is.
- Audited the backtest engine and docs directly; corroborated with a subagent.
- Built **`scripts/edge_sensitivity.py`** — a no-network arithmetic stress test of
  the strategy's own reported numbers. Runs and produces a decisive table.
- Built **`scripts/honest_backtest.py`** — a costed, look-ahead-free daily backtest
  (next-open entry, spread+commission+gap-through slippage) with an optimistic
  mode and a SPY benchmark. Includes a `--selftest` that passes with no network.
- Built **`scripts/_yahoo_fetch.py`** — proxy-aware data fetcher (yfinance's
  curl_cffi backend ignores the agent proxy; plain `requests` works).

## What actually happened
- **Code inspection (fact):** `backtesting/engine_enhanced.py` computes the signal
  from the daily close (`:462`) and enters at that same close (`:494`) — same-bar
  look-ahead. Stops/targets fill at the exact price (`:313`,`:374`). No commission,
  spread, or slippage anywhere in `backtesting/`.
- **Track record (fact):** `data/signals/signal_metrics.json` shows **2 realized
  outcomes over 880 scans**. There is no P&L history; nothing beats anything yet.
- **`scripts/run_signal_research.py` referenced in CLAUDE.md does not exist.**
- **Arithmetic proof (reproducible, ran this session):** `edge_sensitivity.py` output:
  - Docs' own avg win $70 / avg loss $45 / 38% WR imply **PF 0.95 and −$1.30/trade
    expectancy**, not the claimed PF 1.29 — the numbers are mutually inconsistent.
  - Best-case (PF 1.29) edge = **$7.91/trade**; break-even cost = $7.91.
  - Realistic round-trip cost ≈ **$14.75** → **−$1,471/yr (−5.9%)**.
  - Cost-free 6.8%/yr still trails SPY B&H (~10–25%/yr) by **$800–$4,550/yr**.
- **Honest backtest self-test:** PASS — on identical signals, honest mode shows
  higher costs and worse P&L than optimistic, capturing gap-through slippage.
- **Live-data run: NOT completed.** Yahoo rate-limits this datacenter IP hard
  (429 with 19-byte bodies on both query1/query2); Stooq connection-reset. A brief
  200 window appeared but closed before a 30-symbol pull could finish. I did NOT
  fabricate a result. `honest_backtest.py` is ready to run where data is reachable.

## Evidence / numbers
Reproduce with: `python scripts/edge_sensitivity.py`  (no network needed)
```
[1] docs' own numbers -> PF 0.953, expectancy -$1.30  (claim was PF 1.29)
[2] most-favorable read -> $7.91/trade edge
[3] break-even cost/trade = $7.91 ; at $14.75 -> -$1,471/yr (-5.9%)
[5] cost-free 6.8%/yr vs SPY 10-25%/yr -> trails by $800-$4,550/yr
```
`python scripts/honest_backtest.py --selftest` → RESULT: PASS.

## Risk / safety notes
- **Did NOT touch `risk/`.** No changes to live config, credentials, or brokers.
- PAPER-only respected; AUTO_TRADE untouched. No live orders, no order tooling used.
- All new files are additive (journal, docs, 3 scripts). No existing logic changed.
- New scripts only READ public market data and do pure computation.

## Honest assessment
This session did not move net P&L vs SPY, and that is the correct outcome: the
apparent progress in the repo was measurement error. The RRS edge, as currently
evidenced, is not real once you remove look-ahead and add costs — and it loses to
SPY buy-and-hold even before costs. The most valuable thing accomplished is
replacing false confidence with a reproducible, honest baseline and the tooling
to keep testing honestly.

## Handoff to next instance
**Most valuable next objective:** get `scripts/honest_backtest.py` to run on real
data and record the honest-vs-optimistic-vs-SPY numbers in a journal entry.
- The blocker is data access, not code. Options, in order of preference:
  1. Run it from an environment/IP not throttled by Yahoo (the user's box; or a
     later session where the throttle has cleared — space requests ≥3s, it caches
     per-symbol so it resumes).
  2. Point `scripts/_yahoo_fetch.py` at a source that isn't IP-throttled
     (Tiingo/Alpha Vantage/Polygon with a key, or the repo's own data providers).
  3. Wire `honest_backtest.py` into `backtesting/data_loader.py` if that path has
     working data access in the live container.
- Then attack **survivorship bias**: replace the fixed mega-cap `UNIVERSE` with a
  broad or point-in-time list before trusting any positive result.
- **Trap to avoid:** do NOT "fix" profitability by loosening risk limits, widening
  the universe to whatever backtests well, or adding more gates/ML. Those are the
  anti-patterns in MANDATE §7. The job is honest measurement first.
- **If honest testing keeps saying no edge:** that is a reportable result — draft
  the escalation/wind-down recommendation rather than adding complexity.
- Open thread: CLAUDE.md cites `scripts/run_signal_research.py`, which is absent —
  the docs and the code have drifted; distrust doc-quoted metrics generally.
