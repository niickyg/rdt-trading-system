# LATEST

Points to the most recent operator session. Read this, then the entry it names,
then follow `MANDATE.md` §4 Protocol.

---

## Most recent: 2026-09-07 — Bootstrap + first honest edge assessment
**File:** `entries/2026-09-07-bootstrap-and-honest-assessment.md`
**Operator:** instance #1 · **Branch:** `claude/adoring-feynman-9j4ymz`

### Bottom line
The journal/mandate did not exist — I bootstrapped it. Then I measured the edge for
real. **No profitable edge on the evidence:**

- ML ensemble CV AUC ≈ 0.54 (random), train AUC 0.99 (overfit), precision/recall 0.
- Regime detector broken (silhouette −0.087, one-regime collapse).
- 1,986 generated signals, **0 tracked outcomes** — the bot never measures itself.
- Ground-truth eval of the real signals (new `scripts/evaluate_signals.py`), on
  **realistic non-overlapping accounting: expectancy −0.072R, profit factor 0.89**
  (losing). The "+0.034R" per-raw-signal number is a duplication artifact.
- Repo's own best backtest: Sharpe 0.11, negative Kelly. SPY B&H over the analog
  window: ~+9–15% annualised — the unmet bar.

### Do NOT
- Deploy real capital (MANDATE §2.7 bar unmet). Enable auto/live trading. Trust the
  ML/regime layers in any decision path.

### Next instance, start here
1. Confirm the live paper account's real equity curve (I couldn't see it).
2. Make executed-trade outcome tracking real + committed (biggest systemic gap).
3. Extend `scripts/evaluate_signals.py` to multiple windows / regimes before any
   go/no-go. Log the non-overlapping expectancy trend in each entry.
4. If expectancy stays ≤ 0 across windows → recommend wind-down of the trading
   ambition; the value, if any, is in the software layer, not a market edge.

### Tooling added
- `scripts/evaluate_signals.py` — stdlib-only, proxy-aware signal→realised-P&L
  evaluator vs SPY. Run: `python scripts/evaluate_signals.py --cost-bps 5`.
