# OPERATOR MANDATE

> ⚠️ **RECONSTRUCTED BOOTSTRAP — 2026-10-07.** The original `MANDATE.md`
> referenced by the operator's scheduled prompt **did not exist anywhere in
> this repository or its git history** when the 2026-10-07 session ran. This
> file was reconstructed by transcribing the constraints and mission that the
> scheduled prompt itself states (which are authoritative, operator-authorized
> content) plus a minimal protocol. **The human owner should verify this file
> and replace it with the canonical mandate if one exists elsewhere.** Until
> then, treat the "Hard constraints" and "Mission" sections as authoritative
> (they are quoted from the operator's own prompt) and the "Protocol" section
> as a sensible default, not gospel.

## Mission

Make this trading bot **profitable**. Not optimize metrics. Not follow a
methodology for its own sake. **Actual positive P&L net of honest costs,
beating SPY buy-and-hold.** If the evidence keeps saying no strategy works, say
so in the journal and **recommend escalation or wind-down.**

The underlying philosophy derives from r/RealDayTrading and its founders'
teachings — but the philosophy is a means, not the objective. The objective is
net profit that beats buy-and-hold.

## Hard constraints (absolute)

- **PAPER TRADING ONLY.** Never enable `AUTO_TRADE=true`. Never modify live
  broker credentials.
- **Never touch the `risk/` directory** without explicitly flagging it in the
  session's journal entry.
- **Every session must end with a committed journal entry** in
  `data/operator_journal/entries/`.
- **Update `data/operator_journal/LATEST.md`** to point to the new entry.
- You are **stateless** across sessions. The journal is your only memory.
  Write entries so a future instance with zero prior context can continue.

## Environment reality (remote agent)

You run as a remote Claude Code agent. You HAVE: a fresh git checkout, standard
tools, the ability to commit/push, and (this session confirmed) a **read-capable
IBKR connector** for market data. You do NOT have: the user's live bot
container, their local PostgreSQL/TimescaleDB, the ability to restart their
services, or memory across sessions. Work model: **research → code → test →
commit → push → journal.** The human reviews and merges; that gap is a safety
feature.

## Protocol (default — verify against canonical mandate when available)

1. **Read first, fully:** this MANDATE, `LATEST.md`, `POST_MORTEM_RRS.md`,
   `CLAUDE.md`, and the 3 most recent `entries/`.
2. **Establish ground truth before acting.** What is the *actual* P&L? What
   does the evidence say vs SPY buy-and-hold? Distrust self-reported metrics;
   prefer measured outcomes. If you cannot see live P&L, say so — that is a
   blocker to the mission and should be escalated.
3. **Decide honestly.** If there is no evidence of net-of-cost outperformance,
   do not hill-climb internal metrics. Either run a genuine measurement, or
   escalate the strategic question to the owner.
4. **Change the minimum needed.** Small, focused, reviewable commits. Never
   change trading/risk behavior speculatively. Test what you change
   (`python -c "import py_compile; py_compile.compile('file.py', doraise=True)"`
   at minimum; run real tests when feasible).
5. **Verify.** Re-read your own diff adversarially. Did you change risk? Did you
   leave the repo in a worse state?
6. **Journal every session** — what you assessed, the numbers, what you decided
   and why, what you changed, and a concrete starting point for the next
   instance. Then update `LATEST.md`, commit, and push to the designated branch.

## Stop conditions (escalate to human, do not proceed autonomously)

- No visibility into live/paper P&L (cannot measure the mission).
- Evidence repeatedly shows the strategy does not beat buy-and-hold net of
  costs → recommend wind-down or a reframing of the objective.
- Any action that would require touching live credentials, enabling auto-trade,
  or modifying `risk/` logic in a way not clearly warranted and flagged.
