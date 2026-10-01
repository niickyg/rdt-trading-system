# LATEST — pointer to the most recent operator session

**Most recent entry:** [`entries/2026-10-01-bootstrap.md`](entries/2026-10-01-bootstrap.md)
**Date:** 2026-10-01
**Instance:** #1 (bootstrap)

## 30-second summary for the next instance

- This was the **first** run. There was no journal, mandate, or post-mortem —
  I created all of them. Read `MANDATE.md`, then `POST_MORTEM_RRS.md`, then the
  bootstrap entry above.
- **No verified live P&L exists** (2 tracked outcomes / 120 signals). I assessed
  from the committed signal history (Feb–Mar 2026) out-of-sample instead.
- **Headline finding:** the RRS **long** signal is genuinely predictive
  (+2.7%/5d, 82% win), but the shipped **1.5×ATR stop** turns it into a loser
  (PF 0.42), and the live filters emit **~99% shorts** (the direction with *no*
  edge). Wide stop + long bias flips the long book to **PF 1.42** on this sample.
- I changed **no strategy/risk/filter code** — the finding rests on one month;
  shipping it now would be overfitting. Evidence + infra only.

## Your job (highest leverage first)

1. **Validate** the wide-stop + long-bias hypothesis on **2024–2025 multi-regime**
   data before touching `realtime_scanner.py`. Scripts: `research/operator/`.
2. If it holds across regimes → propose the stop/exit change with a human in the
   loop. If it doesn't → document and consider wind-down.
3. Wire up/fix **outcome tracking** so the live system stops flying blind.

## Do not forget
- PAPER ONLY. Don't touch `risk/` without flagging. End with a committed entry +
  update this file. Don't merge to main.
- `MANDATE.md` is a bootstrap draft — ask the human to ratify it.
- Branch note: launch prompt wants `operator/YYYY-MM-DD`; this session's harness
  pinned `claude/adoring-feynman-sh4qht`. Reconcile with the human.
