# Operator Journal — 2026-08-11

**Instance:** first recorded run (see caveat #1). **Branch:** `claude/adoring-feynman-9xknok`
**Focus:** Establish ground truth — does the bot's RRS signal actually make money vs SPY?

---

## TL;DR (read this if nothing else)

1. **The operator-journal infrastructure did not exist.** `MANDATE.md`, `LATEST.md`,
   `POST_MORTEM_RRS.md`, and `data/operator_journal/entries/` were absent from every
   branch and from all of git history. I bootstrapped `MANDATE.md` + this journal.
2. **I ran the first real backtest of the bot's own signals against real forward
   prices** (86 clean signals, Feb–Aug 2026, IBKR data). Result:
   - There **is** a genuine long-side edge: **+1.85% alpha over 5 days vs SPY, 75% of
     longs beat SPY.** Real Relative Strength does work on the long side, in this sample.
   - **But the strategy as actually tradeable does NOT beat SPY buy-and-hold.** A
     realistic $25k account (max 8 concurrent, no leverage, compounded) returned
     **+1% to +3%** while **SPY buy-and-hold returned +12.1%** over the same window.
   - **Shorts are pure drag** (-1.29% alpha, 40% win) and should be dropped.
3. **Verdict on the mandate's core test (beat SPY net of costs): currently FAILING.**
   Not a wind-down call yet — the sample is one ~1-week episode — but the burden of
   proof is now clearly on generating a longer, honest track record.

---

## What I found, in order

### Caveat #1 — no journal, no history
The scheduled task told me to read `MANDATE.md`, `LATEST.md`, `POST_MORTEM_RRS.md` and
the 3 latest entries "before doing anything else." **None existed.** `git log --all`
confirms they were never committed. The "stateless operator with a persistent journal"
system was never actually wired up in the repo. I reconstructed `MANDATE.md` faithfully
from the constraints stated in the task prompt and started this journal so the next
instance has continuity. If an authoritative original surfaces, it wins.

### Caveat #2 — the data-timeline trap (important for every future instance)
- The bot lives in a **2026 market timeline**. The **IBKR MCP server**
  (`get_price_history`) serves real 2026 OHLC that matches the signals.
- **Public yfinance/Yahoo reachable from this sandbox is ~1 year behind** (returns
  2025 data) and does NOT correspond to the 2026 signals. I nearly backtested against
  it — the entry prices didn't match, which is what exposed the trap. **Do not use
  yfinance here for 2026 signals.** Use IBKR.

### Caveat #3 — corporate-action contamination
First cut of the backtest showed an absurd **+100% return** and a **+5%/day** average
move. Cause: **DuPont (DD)** — the signal entry (~$45) is at 1/3 the scale of DD's
actual Feb-2026 IBKR series (~$150), a split/spinoff data break. Two DD "trades" faked
+240% each. I added a **contamination guard** (drop any symbol whose signal entry is
>1.5x off its own monthly median). Only DD was affected. All numbers below EXCLUDE DD.

### The data
- `data/signals/signal_history.json`: 1,986 raw signals → **88 distinct**
  (symbol/date/direction) → **86 after dropping DD**.
- Concentrated in **6 trading days**, essentially all **Feb 3–6, 2026** (+4 in March).
  **83% are longs.** This is a single market episode, not a diversified sample.
- Forward prices: IBKR daily bars for all 48 symbols + SPY, Feb→Aug 2026. Saved to
  `data/operator_journal/data/ibkr_prices_2026-02_to_08.json` (reproducibility; avoids
  re-fetching ~96 MCP calls). Deduped signals in `.../signals_deduped_88.json`.

### The numbers (net of $1/side commission + 5bps/side slippage)

**Per-trade barrier-touch** (looks great, but see portfolio reality below):
| max_hold | win% | PF | expectancy |
|---|---|---|---|
| 5d | 65% | 2.76 | +0.60R |
| 10d | 62% | 2.92 | +0.74R |
| 20d | 59% | 2.71 | +0.73R |

**Alpha vs beta (5-day, signal return − SPY over identical days):**
| side | n | signal | beta(SPY) | **alpha** | alpha-win% |
|---|---|---|---|---|---|
| long | 71 | +2.59% | +0.74% | **+1.85%** | 75% |
| short | 15 | −1.72% | −0.43% | **−1.29%** | 40% |

**Realistic portfolio (max 8 concurrent, no leverage, compounded) vs SPY +12.11%:**
| config | return | verdict |
|---|---|---|
| all signals, barrier 10d | +1.1% | lags SPY |
| long-only, barrier 10d | +1.3% | lags SPY |
| long-only, 5d time-exit | +2.8% | lags SPY |
| long-only, 5d exit, 12 slots (best found) | +3.2% | lags SPY |

### Why real long alpha still loses to SPY
Signals arrive in **infrequent clusters**, so a capacity-capped account can only take
~13–26 of 86 trades and **sits mostly in cash**. In a rising market (SPY +12% over the
window) being under-deployed is a massive opportunity cost. The per-trade edge is real
but too small and too sparse to compound past a fully-invested benchmark. `|RRS|` did
**not** predict outcome (Pearson ≈ −0.02), so it is a poor position-selection ranker.

---

## What I changed this session
- **Added `data/operator_journal/MANDATE.md`** (bootstrap reconstruction).
- **Added `scripts/operator_backtest.py`** — provider-agnostic, honest evaluation
  harness (barrier + raw-edge + alpha-vs-beta + capacity-constrained portfolio sim +
  SPY benchmark + contamination guard). Compiles clean; reproduces every number above.
  Run: `python scripts/operator_backtest.py --prices <ohlc.json>`.
- **Committed reproducibility data** under `data/operator_journal/data/`.
- **Did NOT touch `risk/`, broker code, or any live/trading parameter.** No strategy
  parameters were changed — deliberately, because tuning on a 6-day sample is
  overfitting, not edge. PAPER-only respected; AUTO_TRADE untouched.

## Recommendation to the next instance (do these, in order)
1. **Fix the measurement gap first.** The system persisted only 6 days of signals with
   ~zero outcome tracking (2 outcomes in `signal_metrics.json`). You cannot make this
   profitable if you cannot measure it over time. Wire outcome logging (entry→exit→P&L,
   alpha vs SPY) into the live loop, and/or generate a **multi-month, multi-regime**
   signal set via IBKR history so `operator_backtest.py` has real breadth to chew on.
   **One bullish week proves nothing.**
2. **Drop / disable the short side** unless a longer sample rehabilitates it. Here it
   subtracted alpha in every cut. (Consistent with RDT "trade with the market" in a
   bull regime — but confirm on more data before committing a code change.)
3. **Attack under-deployment, not entry tuning.** The bottleneck is idle cash, not
   signal quality. Options: broaden the watchlist / relax the RRS gate to fire more
   consistently; OR accept this is a "cash + occasional trade" profile that can only
   beat SPY in flat/down markets, and benchmark **regime-conditionally**.
4. **Find a better position-selection ranker than |RRS|** (it's ~uncorrelated with
   outcome). The 5-day alpha lives on the long side; test ranking by alpha-predictive
   features on a larger sample.

## Mandate status
- Core test (beat SPY buy-and-hold, net of costs): **FAILING** on available evidence.
- Escalation/wind-down trigger: **NOT YET** — sample too small (one episode) for a
  definitive no, and a real per-trade long edge exists. Re-evaluate after step #1
  produces a multi-regime sample. If a broad, honest sample still lags SPY, that is the
  wind-down signal — say so plainly.
