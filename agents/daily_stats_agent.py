"""
Daily Stats Agent

Computes and persists daily trading statistics and equity snapshots.
Runs once at market close (triggered by MARKET_CLOSE event) and can
also snapshot equity intraday on a schedule.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from loguru import logger

from agents.base import ScheduledAgent
from agents.events import Event, EventType


class DailyStatsAgent(ScheduledAgent):
    """
    Agent that computes daily P&L stats and tracks equity over time.

    Responsibilities:
    - Calculate daily stats (win rate, avg win/loss, etc.) at market close
    - Upsert into daily_stats table
    - Save equity snapshots for drawdown analysis
    - Optionally snapshot intraday equity on schedule
    """

    def __init__(
        self,
        account_size: float = 25000.0,
        intraday_snapshot_interval: float = 3600.0,  # 1 hour
        **kwargs
    ):
        super().__init__(
            name="DailyStatsAgent",
            interval_seconds=intraday_snapshot_interval,
            **kwargs
        )
        self.account_size = account_size
        self._trades_repo = None

    @property
    def trades_repo(self):
        """Lazy-load trades repository."""
        if self._trades_repo is None:
            from data.database import get_trades_repository
            self._trades_repo = get_trades_repository()
        return self._trades_repo

    async def initialize(self):
        """Initialize daily stats agent."""
        logger.info("DailyStatsAgent initialized")

    async def cleanup(self):
        """Cleanup daily stats agent."""
        pass

    def get_subscribed_events(self) -> List[EventType]:
        """Subscribe to market close for daily stats computation."""
        return [EventType.MARKET_CLOSE]

    async def handle_event(self, event: Event):
        """Handle market close event to compute daily stats."""
        if event.event_type == EventType.MARKET_CLOSE:
            await self._compute_daily_stats()

    async def run_scheduled_task(self):
        """Periodically snapshot equity (intraday)."""
        await self._save_equity_snapshot()

    async def _compute_daily_stats(self):
        """Compute and save daily trading statistics."""
        try:
            today = date.today()

            # Get today's closed trades
            trades = self.trades_repo.get_trades(status='closed', days=1)

            # Filter to trades closed today
            today_trades = []
            for t in trades:
                exit_time = t.get('exit_time')
                if exit_time:
                    if isinstance(exit_time, str):
                        exit_dt = datetime.fromisoformat(exit_time)
                    else:
                        exit_dt = exit_time
                    if exit_dt.date() == today:
                        today_trades.append(t)

            # Get starting balance from yesterday's ending balance
            latest_stats = self.trades_repo.get_latest_daily_stats()
            if latest_stats and latest_stats.get('date') != today.isoformat():
                starting_balance = latest_stats.get('ending_balance', self.account_size)
            else:
                starting_balance = self.account_size

            # Calculate stats
            num_trades = len(today_trades)
            winners = [t for t in today_trades if t.get('pnl') and float(t['pnl']) > 0]
            losers = [t for t in today_trades if t.get('pnl') and float(t['pnl']) <= 0]

            total_pnl = sum(float(t.get('pnl', 0)) for t in today_trades)
            ending_balance = starting_balance + total_pnl

            win_pnls = [float(t['pnl']) for t in winners]
            loss_pnls = [float(t['pnl']) for t in losers]

            stats_data = {
                'date': today,
                'starting_balance': starting_balance,
                'ending_balance': ending_balance,
                'pnl': total_pnl,
                'pnl_percent': (total_pnl / starting_balance * 100) if starting_balance > 0 else 0,
                'num_trades': num_trades,
                'winners': len(winners),
                'losers': len(losers),
                'win_rate': (len(winners) / num_trades * 100) if num_trades > 0 else None,
                'avg_win': (sum(win_pnls) / len(win_pnls)) if win_pnls else None,
                'avg_loss': (sum(loss_pnls) / len(loss_pnls)) if loss_pnls else None,
                'largest_win': max(win_pnls) if win_pnls else None,
                'largest_loss': min(loss_pnls) if loss_pnls else None,
                'market_regime': self._get_current_regime(),
            }

            result = self.trades_repo.save_daily_stats(stats_data)
            if result:
                logger.info(
                    f"Daily stats saved: {today} | PnL=${total_pnl:.2f} | "
                    f"Trades={num_trades} | Win Rate={stats_data['win_rate']:.0f}%"
                    if stats_data['win_rate'] else
                    f"Daily stats saved: {today} | PnL=${total_pnl:.2f} | Trades={num_trades}"
                )

            # Also save equity snapshot at close
            await self._save_equity_snapshot(ending_balance=ending_balance)

        except Exception as e:
            logger.error(f"Error computing daily stats: {e}")

    async def _save_equity_snapshot(self, ending_balance: Optional[float] = None):
        """Save an equity snapshot."""
        try:
            # Get current equity
            if ending_balance is not None:
                equity = ending_balance
            else:
                # Estimate from positions
                positions = self.trades_repo.get_open_positions()
                positions_value = sum(
                    float(p.get('pnl', 0) or 0) for p in positions
                )
                # Get latest daily stats for base balance
                latest = self.trades_repo.get_latest_daily_stats()
                base = latest.get('ending_balance', self.account_size) if latest else self.account_size
                equity = base + positions_value

            # Get high water mark
            hwm = self.trades_repo.get_high_water_mark()
            if equity > hwm:
                hwm = equity

            drawdown_pct = ((hwm - equity) / hwm * 100) if hwm > 0 else 0

            positions = self.trades_repo.get_open_positions()

            snapshot_data = {
                'timestamp': datetime.utcnow(),
                'equity_value': equity,
                'cash': equity - sum(
                    float(p.get('entry_price', 0)) * int(p.get('shares', 0))
                    for p in positions
                ) if positions else equity,
                'positions_value': sum(
                    float(p.get('pnl', 0) or 0) for p in positions
                ) if positions else 0,
                'open_positions_count': len(positions),
                'drawdown_pct': drawdown_pct,
                'high_water_mark': hwm,
            }

            self.trades_repo.save_equity_snapshot(snapshot_data)

        except Exception as e:
            logger.error(f"Error saving equity snapshot: {e}")

    def _get_current_regime(self) -> Optional[str]:
        """Get current market regime from adaptive learner if available."""
        try:
            from agents.adaptive_learner import get_adaptive_learner
            learner = get_adaptive_learner()
            return learner.current_regime
        except Exception:
            return None
