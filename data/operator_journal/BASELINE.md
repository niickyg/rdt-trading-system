# BASELINE — verified benchmark record

> The durable, factual reference future runs measure against. Only add numbers here that you
> actually verified this session. Label gross vs net, the period, and the source.

## SPY buy-and-hold benchmark (the bar to beat)

Source: IBKR `get_price_history`, SPY (conid 756733), monthly bars, "Last" close
(price-only, **not** dividend-adjusted). Pulled 2026-08-05.

Selected monthly closes:

| Date       | SPY close |
|------------|-----------|
| 2024-02-01 | 508.08    |
| 2024-03-01 | 523.07    |
| 2024-05-01 | 527.37    |
| 2025-11-03 | 683.39    |
| 2026-08-03 | 771.33    |

Buy-and-hold returns over the bot's backtest window ("Feb 2024 – Nov 2025" per CLAUDE.md):

- **Feb 2024 → Nov 2025: 508.08 → 683.39 = +34.5% price** over ~21 months
  ≈ **+18.4%/yr price**, **≈ +19.7%/yr total return** (adding ~1.3%/yr SPY dividends).
- Trading-window subset (May 2024 → Nov 2025, when the walk-forward actually traded):
  527.37 → 683.39 = **+29.6% price** over ~18 months ≈ **+19%/yr**.

## The bot's own best backtest (from CLAUDE.md, GROSS of costs, UNVERIFIED this session)

Walk-forward V2, $25K, 6 quarterly windows, Config C ("RDT Filters"):

- Total return: **+$1,716 (+6.9%)** over 2 years = **+3.4%/yr**.
- Win rate 49.5%, profit factor 1.24, 279 trades, worst day −$257.

**Caveats that make even this number optimistic:**
- `backtesting/engine_enhanced.py` and `engine_intraday.py` model **ZERO transaction costs**
  — no commission, no slippage, no spread. Verified by grep 2026-08-05: no
  commission/slippage/fee/bid-ask terms exist; fills use exact signal prices.
- Rough cost drag on 279 trades: ~$2 commission floor + ~0.05% round-trip spread on
  ~$3k positions ≈ **$3–6/trade → ~$850–1,700 total**, i.e. roughly **half to all** of the
  $1,716 gross "profit." Net return is plausibly **near zero to slightly positive**.
- Backtest is daily-bar; the live strategy is intraday. Daily bars cannot simulate the VWAP
  gate or first-hour filter (per the script's own docstring), so this is a crude proxy.

## Head-to-head (the honest read)

| | Bot Config C (gross) | SPY buy-and-hold |
|---|---|---|
| ~2-yr total return | +6.9% | **+34.5%** (price) / ~+40% (total) |
| Annualized | +3.4% | **~+19.7%** |
| Net of honest costs | ~0% to +low-single-digit | n/a (buy once) |

**SPY buy-and-hold beat the bot's best config by ~5x gross, and by more net of costs.**
The mission bar ("beat SPY buy-and-hold") is **not met** by any evidence currently in the
repo.

Fairness caveat (recorded, not an excuse): the bot's max drawdown is far smaller (worst day
−$257 ≈ −1% vs SPY's ~−19% intra-2025 drawdown). On a pure risk-adjusted basis the gap
narrows. But a near-zero net return with low drawdown does not clear the section-3 bar —
cash clears that same low-drawdown test and, net of costs, may out-earn the strategy.

## Connected paper account state (verified 2026-08-05)

`get_account_summary`: **NLV = $5**, cash $5, no positions, `day_trades_remaining: 3`
(sub-$25K PDT-restricted). This is **not** the "DUP995654 ($25K)" account CLAUDE.md
describes. There is **no live P&L track record** to evaluate — only backtests.
