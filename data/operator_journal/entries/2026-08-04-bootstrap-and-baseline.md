# 2026-08-04 — Bot −61.5% vs SPY +10.3%; account at $5; dormant since March. Bootstrapped the journal.

First operator run. **The mandate, journal, and post-mortem the scheduled task told me to read
did not exist** — this is genuinely the first instance. So this session (a) built the governance
infrastructure future instances depend on, and (b) established the honest baseline from live data.

## Scoreboard (measured this session)
- **Bot TWR since inception (2026-02-26): −61.5%**  `[LIVE, get_pa_performance_all_periods]`
- **SPY same window (2026-02-26 → 2026-08-04): +10.3%**  `[LIVE, 756733: $689.30 → $760.12]`
- **Verdict: beating SPY? NO.** Trailing buy-and-hold by ~72 points.
- Account: NAV **$5**, **0** open positions, running? **NO** (last scan 2026-03-05).

## What I found
- **Live paper account is effectively blown up and abandoned.** NAV path (TWR, cash-flow-insensitive):
  $50 → $21 in month one (−58%), topped up to ~$521, bled to ~$477 through spring, collapsed to
  **$5** in early July, flat since. `[LIVE]`
- **Measurement loop is broken.** 1,986 signals logged, 880 scans, but only **2** outcomes ever
  tracked (1 win / 1 loss). `[LIVE, signal_metrics.json]` The bot almost never checked if it was right.
- **Backtest vs reality gap.** Repo headlines +6.9%/2yr for RDT filters `[REPO-CLAIM]`; reality is
  −61.5%. Nobody reconciled them because of the broken measurement loop.
- **Short-heavy into a rally.** 119/120 tracked signals were shorts `[LIVE]` while SPY rose ~10%.
- **Env constraint:** yfinance is network-blocked here (SSL reset) — repo backtest/training scripts
  that depend on it **cannot run in this remote agent.** But **IBKR MCP price history works** and is
  the path forward (verified: pulled 6mo SPY daily + 13 single-name histories).

Full analysis in the new `POST_MORTEM_RRS.md`.

## What I did
**No strategy/alpha code was changed — deliberately.** With the measurement loop broken and no
ability to backtest-validate here yet, shipping a speculative alpha tweak would violate the
discipline this system needs (MANDATE §2.7). Instead I delivered the missing foundation:
- Created `data/operator_journal/MANDATE.md` — the constitution: mission (beat SPY, not metrics),
  hard safety constraints (paper-only, risk/ protected, no live orders), the environment reality
  (yfinance blocked, IBKR MCP is the data path), and a 6-step per-session protocol. **Flagged as
  operator-drafted and requiring human ratification.**
- Created `POST_MORTEM_RRS.md` — the honest history from live data.
- Created this entry, `LATEST.md`, and `data/operator_journal/README.md`.

## Verification
- Every performance number is from a live MCP query made this session, not from repo docs. SPY
  same-window return computed from IBKR bars ($689.30 close 2026-02-26 → $760.12 last 2026-08-04).
- yfinance block confirmed by direct test (`curl (35) Recv failure`). IBKR `get_price_history`
  confirmed working on SPY and 13 single names.
- No `.py` files changed, so no compile/test step applies. Docs only.

## Risk-directory touched? **NO.**

## Recommendation / hand-off to next instance (do this first, it's now unblocked)
**Run the honest edge test the bot never ran.** You do NOT need yfinance — use IBKR MCP
`get_price_history`. Concrete recipe:
1. Load `data/signals/signal_history.json` (1,986 signals, dated 2026-02-03 → 2026-03-05, each with
   symbol, direction, entry_price, stop_price, target_price).
2. For each signal, pull daily bars from its `generated_at` date forward ~10–15 trading days via
   `get_price_history` (period SIX_MONTHS covers the whole window). Classify: did **stop** hit
   first, **target** first, or neither by horizon end; record forward return.
3. Subtract honest costs (commission + slippage) per trade. Compute realized expectancy, win rate,
   and — critically — **compare the resulting equity curve to SPY over the same dates.**
4. **Pre-resolved contract_ids** (US primary listings, so you can skip `search_contracts`):
   SPY 756733 · CCL 878372298 · DOW 356576040 · EXC 11000 · OXY 10880 · PYPL 199169591 ·
   T(AT&T) 37018770 · PFE 11031 · BAC 10098 · BMY 5111 · C(Citi) 87335484 · LOW 9199 ·
   MRNA 344809106 · ROST 273939. Resolve the rest with `search_contracts` (pick country_code US,
   primary exchange, exact symbol match).
5. If the naked signal has no positive cost-adjusted expectancy, that's the answer — the 8-layer
   filter stack cannot rescue a signal with no edge. If it does, test whether each filter layer
   *adds* to expectancy out of sample; drop the ones that don't.

Second priority: **fix the outcome-tracking loop** so a deployed bot actually records target/stop
outcomes (only 2 in history is the core disease). Third: reconcile or retire the +6.9% backtest.

Do **not** add features. Do **not** re-enable trading. Answer "does the signal make money vs SPY?"

## Open questions / unknowns
- Why did NAV collapse from ~$477 to $5 in July — withdrawal, or a final losing burst? `get_account_trades`
  returned empty for recent quarters (paper fills may not surface via that endpoint); couldn't confirm.
- Is the owner's live bot config the same as this repo's defaults? Can't see their env from here.
- **Branch note:** the remote-agent harness designated branch `claude/adoring-feynman-tw4kb1`, but the
  scheduled task instructed `operator/YYYY-MM-DD`. I followed the task → work is on **`operator/2026-08-04`**.
  If the owner's automation watches the other branch, look here instead.
