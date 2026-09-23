# LATEST — Operator Journal Pointer

> Always reflects the most recent operator run. Full detail in the linked entry.

## → Most recent: [Run 001 — 2026-09-23](entries/2026-09-23-run-001.md)

**Branch:** claude/adoring-feynman-7y4p0p · **Model:** claude-opus-4-8 (1M)

### Bottom line
First operator session. Bootstrapped the entire journal system (it did not exist) and
established an honest baseline.

**Verdict: no demonstrated edge over SPY buy-and-hold.**
- Strategy (RDT Filters), fresh reproduction: **~6% annualized gross, ~3–5% net of costs.**
- **SPY buy-and-hold, same window: ~18.5% annualized.** Net edge vs benchmark ≈ **−13 pp/yr.**
- The backtest engine models **zero** commission/slippage — all historical repo numbers are
  gross. Costs shrink the edge but are not the main problem; the gross edge already loses to
  buy-and-hold in the bull-market data available.

### Do NOT (already disproven)
- Add more filters/ML/features — the bot lacks an *edge*, not features.
- Raise risk-per-trade to lever returns — Kelly is ~0/negative.
- Re-run cost overlays expecting a different answer.

### Next instance should answer ONE question
**Is there any regime (flat/bear, e.g. 2022 or 2018-Q4) where the strategy beats SPY net of
costs?**
- If no → recommend wind-down / fundamental pivot to the human (MANDATE §7).
- If yes → build & prove a **regime-gated overlay** (hold SPY in bull; RRS engine only when
  SPY < 200-day) vs pure buy-and-hold. That is the only currently-visible path to the goal.

### Tooling added this run
- `scripts/run_cost_overlay.py` — reproducible net-of-cost vs SPY benchmark.
- `POST_MORTEM_RRS.md`, `data/operator_journal/MANDATE.md` — read these first.
