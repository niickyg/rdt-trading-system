"""
Daily Summary Email Task
Aggregates daily trading activity and sends performance summary email.
"""

import os
import asyncio
import schedule
import time
import threading
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from loguru import logger

from .email_alert import EmailAlert


@dataclass
class DailySummaryData:
    """Container for daily summary data."""
    date: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    portfolio_value: float = 0.0
    starting_value: float = 0.0
    signals_generated: int = 0
    trades: List[Dict[str, Any]] = field(default_factory=list)
    top_winners: List[Dict[str, Any]] = field(default_factory=list)
    top_losers: List[Dict[str, Any]] = field(default_factory=list)
    positions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DailySummaryCollector:
    """
    Collects trading data throughout the day for the daily summary.

    This class tracks trades, signals, and portfolio changes during the
    trading day and provides methods to aggregate this data for the summary.
    """

    def __init__(self):
        """Initialize the collector."""
        self._trades: List[Dict[str, Any]] = []
        self._signals: List[Dict[str, Any]] = []
        self._portfolio_snapshots: List[Dict[str, Any]] = []
        self._start_of_day_value: Optional[float] = None
        self._lock = threading.Lock()

    def reset(self):
        """Reset all collected data for a new day."""
        with self._lock:
            self._trades = []
            self._signals = []
            self._portfolio_snapshots = []
            self._start_of_day_value = None

    def set_start_of_day_value(self, value: float):
        """
        Set the portfolio value at start of day.

        Args:
            value: Portfolio value at market open
        """
        with self._lock:
            self._start_of_day_value = value

    def record_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        pnl: Optional[float] = None,
        return_pct: Optional[float] = None,
        strategy: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a trade execution.

        Args:
            symbol: Stock symbol
            action: Trade action (BUY, SELL, etc.)
            quantity: Number of shares
            price: Execution price
            pnl: Realized P&L (optional)
            return_pct: Return percentage (optional)
            strategy: Strategy name (optional)
            timestamp: Trade timestamp (optional)
        """
        with self._lock:
            self._trades.append({
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'value': quantity * price,
                'pnl': pnl,
                'return_pct': return_pct,
                'strategy': strategy,
                'timestamp': (timestamp or datetime.now()).isoformat()
            })

    def record_signal(
        self,
        symbol: str,
        signal_type: str,
        indicator: str,
        indicator_value: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a generated signal.

        Args:
            symbol: Stock symbol
            signal_type: Signal type (BULLISH, BEARISH, etc.)
            indicator: Indicator name
            indicator_value: Indicator value
            timestamp: Signal timestamp (optional)
        """
        with self._lock:
            self._signals.append({
                'symbol': symbol,
                'signal_type': signal_type,
                'indicator': indicator,
                'indicator_value': indicator_value,
                'timestamp': (timestamp or datetime.now()).isoformat()
            })

    def record_portfolio_snapshot(
        self,
        value: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a portfolio value snapshot.

        Args:
            value: Current portfolio value
            timestamp: Snapshot timestamp (optional)
        """
        with self._lock:
            self._portfolio_snapshots.append({
                'value': value,
                'timestamp': (timestamp or datetime.now()).isoformat()
            })

    def get_summary_data(self, summary_date: Optional[str] = None) -> DailySummaryData:
        """
        Aggregate collected data into summary format.

        Args:
            summary_date: Date string for the summary (defaults to today)

        Returns:
            DailySummaryData: Aggregated summary data
        """
        summary_date = summary_date or date.today().strftime('%Y-%m-%d')

        with self._lock:
            # Calculate trade statistics
            closed_trades = [t for t in self._trades if t.get('pnl') is not None]
            winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]

            total_pnl = sum(t.get('pnl', 0) for t in closed_trades)

            # Get portfolio values
            starting_value = self._start_of_day_value or 0.0
            if self._portfolio_snapshots:
                current_value = self._portfolio_snapshots[-1]['value']
            else:
                current_value = starting_value

            # Calculate return percentage
            if starting_value > 0:
                total_pnl_percent = (current_value - starting_value) / starting_value * 100
            else:
                total_pnl_percent = 0.0

            # Sort trades by P&L for top winners/losers
            sorted_winners = sorted(
                winning_trades,
                key=lambda x: x.get('pnl', 0),
                reverse=True
            )[:5]

            sorted_losers = sorted(
                losing_trades,
                key=lambda x: x.get('pnl', 0)
            )[:5]

            return DailySummaryData(
                date=summary_date,
                total_trades=len(closed_trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                total_pnl=total_pnl,
                total_pnl_percent=total_pnl_percent,
                portfolio_value=current_value,
                starting_value=starting_value,
                signals_generated=len(self._signals),
                trades=self._trades.copy(),
                top_winners=sorted_winners,
                top_losers=sorted_losers
            )


class DailySummaryTask:
    """
    Scheduled task for sending daily summary emails.

    Can be run via:
    - Manual trigger: task.run_now()
    - Schedule-based: task.start_scheduler()
    - Async: await task.run_async()
    """

    DEFAULT_SCHEDULE_TIME = "16:30"  # After market close

    def __init__(
        self,
        email_alert: Optional[EmailAlert] = None,
        collector: Optional[DailySummaryCollector] = None,
        schedule_time: Optional[str] = None,
        data_provider: Optional[Callable[[], DailySummaryData]] = None,
        to_email: Optional[str] = None
    ):
        """
        Initialize daily summary task.

        Args:
            email_alert: EmailAlert instance (creates one if not provided)
            collector: DailySummaryCollector instance (creates one if not provided)
            schedule_time: Time to send summary (HH:MM format, default 16:30)
            data_provider: Optional callback to fetch summary data
            to_email: Override recipient email
        """
        self._email_alert = email_alert or EmailAlert()
        self._collector = collector or DailySummaryCollector()
        self._schedule_time = schedule_time or os.environ.get(
            'DAILY_SUMMARY_TIME', self.DEFAULT_SCHEDULE_TIME
        )
        self._data_provider = data_provider
        self._to_email = to_email
        self._scheduler_running = False
        self._scheduler_thread: Optional[threading.Thread] = None

    @property
    def collector(self) -> DailySummaryCollector:
        """Get the data collector."""
        return self._collector

    @property
    def is_configured(self) -> bool:
        """Check if email is configured."""
        return self._email_alert.is_configured

    def _get_summary_data(self) -> DailySummaryData:
        """
        Get summary data from data provider or collector.

        Returns:
            DailySummaryData: Summary data
        """
        if self._data_provider:
            return self._data_provider()
        return self._collector.get_summary_data()

    def send_summary(self, data: Optional[DailySummaryData] = None) -> bool:
        """
        Send the daily summary email.

        Args:
            data: Summary data (optional, will be collected if not provided)

        Returns:
            bool: True if sent successfully
        """
        if not self.is_configured:
            logger.warning("Email not configured for daily summary")
            return False

        data = data or self._get_summary_data()

        success = self._email_alert.send_daily_summary(
            date=data.date,
            total_trades=data.total_trades,
            winning_trades=data.winning_trades,
            losing_trades=data.losing_trades,
            total_pnl=data.total_pnl,
            total_pnl_percent=data.total_pnl_percent,
            portfolio_value=data.portfolio_value,
            starting_value=data.starting_value,
            trades=data.trades,
            top_winners=data.top_winners,
            top_losers=data.top_losers,
            signals_generated=data.signals_generated,
            to_email=self._to_email
        )

        if success:
            logger.info(f"Daily summary email sent for {data.date}")
            # Reset collector for next day
            self._collector.reset()
        else:
            logger.error(f"Failed to send daily summary email for {data.date}")

        return success

    def run_now(self) -> bool:
        """
        Run the summary task immediately.

        Returns:
            bool: True if sent successfully
        """
        return self.send_summary()

    async def run_async(self) -> bool:
        """
        Run the summary task asynchronously.

        Returns:
            bool: True if sent successfully
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.send_summary)

    def _scheduler_loop(self):
        """Background scheduler loop."""
        while self._scheduler_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def start_scheduler(self):
        """
        Start the background scheduler.

        Schedules the summary to be sent at the configured time daily.
        """
        if self._scheduler_running:
            logger.warning("Scheduler already running")
            return

        # Clear any existing scheduled jobs
        schedule.clear('daily_summary')

        # Schedule the summary
        schedule.every().day.at(self._schedule_time).do(
            self.send_summary
        ).tag('daily_summary')

        logger.info(f"Daily summary scheduled for {self._schedule_time}")

        # Start background thread
        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="DailySummaryScheduler"
        )
        self._scheduler_thread.start()

    def stop_scheduler(self):
        """Stop the background scheduler."""
        if not self._scheduler_running:
            return

        self._scheduler_running = False
        schedule.clear('daily_summary')

        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
            self._scheduler_thread = None

        logger.info("Daily summary scheduler stopped")

    def reschedule(self, new_time: str):
        """
        Reschedule the daily summary to a new time.

        Args:
            new_time: New time in HH:MM format
        """
        self._schedule_time = new_time

        if self._scheduler_running:
            schedule.clear('daily_summary')
            schedule.every().day.at(new_time).do(
                self.send_summary
            ).tag('daily_summary')
            logger.info(f"Daily summary rescheduled to {new_time}")


def create_daily_summary_from_db(
    db_session,
    summary_date: Optional[date] = None
) -> DailySummaryData:
    """
    Create summary data from database records using real SQLAlchemy queries.

    Args:
        db_session: SQLAlchemy session
        summary_date: Date to summarize (defaults to today)

    Returns:
        DailySummaryData: Summary data from database
    """
    from sqlalchemy import func, and_, cast, Date
    from data.database.models import Trade, Signal, Position, TradeStatus

    summary_date = summary_date or date.today()
    date_str = summary_date.strftime('%Y-%m-%d')
    day_start = datetime.combine(summary_date, datetime.min.time())
    day_end = datetime.combine(summary_date, datetime.max.time())

    try:
        # Trades closed today
        closed_trades = db_session.query(Trade).filter(
            and_(
                Trade.status == TradeStatus.CLOSED,
                Trade.exit_time >= day_start,
                Trade.exit_time <= day_end
            )
        ).all()

        total_trades = len(closed_trades)
        winning_trades = sum(1 for t in closed_trades if (t.pnl or 0) > 0)
        losing_trades = sum(1 for t in closed_trades if (t.pnl or 0) <= 0)
        total_pnl = sum(float(t.pnl or 0) for t in closed_trades)

        trade_details = []
        for t in closed_trades:
            pnl = float(t.pnl or 0)
            trade_details.append({
                'symbol': t.symbol,
                'direction': str(t.direction.value) if t.direction else 'LONG',
                'pnl': pnl,
                'pnl_percent': float(t.pnl_percent or 0),
                'strategy': t.strategy_name or 'unknown',
                'exit_reason': str(t.exit_reason.value) if t.exit_reason else 'unknown',
            })

        # Sort for top winners/losers
        sorted_trades = sorted(trade_details, key=lambda x: x['pnl'], reverse=True)
        top_winners = sorted_trades[:3] if sorted_trades else []
        top_losers = sorted_trades[-3:][::-1] if sorted_trades else []

        # Signals generated today
        signals_count = db_session.query(func.count(Signal.id)).filter(
            and_(
                Signal.timestamp >= day_start,
                Signal.timestamp <= day_end
            )
        ).scalar() or 0

        # Open positions
        open_positions = db_session.query(Position).all()
        positions_data = []
        portfolio_value = 0.0
        for p in open_positions:
            price = float(p.current_price or p.entry_price or 0)
            value = price * (p.shares or 0)
            portfolio_value += value
            positions_data.append({
                'symbol': p.symbol,
                'direction': str(p.direction.value) if p.direction else 'LONG',
                'shares': p.shares,
                'entry_price': float(p.entry_price or 0),
                'current_price': price,
                'unrealized_pnl': float(p.unrealized_pnl or 0),
                'strategy': p.strategy_name or 'unknown',
            })

        # P&L by strategy
        strategy_pnl = {}
        for t in trade_details:
            strat = t['strategy']
            strategy_pnl[strat] = strategy_pnl.get(strat, 0) + t['pnl']

        return DailySummaryData(
            date=date_str,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            total_pnl_percent=(total_pnl / 25000.0) * 100 if total_pnl else 0.0,
            portfolio_value=portfolio_value,
            starting_value=25000.0,
            signals_generated=signals_count,
            trades=trade_details,
            top_winners=top_winners,
            top_losers=top_losers,
            positions=positions_data,
            metadata={'strategy_pnl': strategy_pnl},
        )

    except Exception as e:
        logger.error(f"Error creating daily summary from DB: {e}")
        return DailySummaryData(
            date=date_str,
            metadata={'error': str(e)}
        )


# Convenience function for quick setup
def setup_daily_summary(
    schedule_time: Optional[str] = None,
    start_scheduler: bool = True
) -> DailySummaryTask:
    """
    Quick setup for daily summary task.

    Args:
        schedule_time: Time to send summary (HH:MM format)
        start_scheduler: Whether to start the background scheduler

    Returns:
        DailySummaryTask: Configured task instance
    """
    task = DailySummaryTask(schedule_time=schedule_time)

    if not task.is_configured:
        logger.warning(
            "Daily summary email not fully configured. "
            "Set EMAIL_PROVIDER and EMAIL_TO environment variables."
        )

    if start_scheduler:
        task.start_scheduler()

    return task


# Global task instance for easy access
_daily_summary_task: Optional[DailySummaryTask] = None


def get_daily_summary_task() -> DailySummaryTask:
    """
    Get or create the global daily summary task instance.

    Returns:
        DailySummaryTask: Task instance
    """
    global _daily_summary_task
    if _daily_summary_task is None:
        _daily_summary_task = DailySummaryTask()
    return _daily_summary_task


def record_trade(
    symbol: str,
    action: str,
    quantity: int,
    price: float,
    pnl: Optional[float] = None,
    **kwargs
):
    """
    Convenience function to record a trade to the global collector.

    Args:
        symbol: Stock symbol
        action: Trade action
        quantity: Number of shares
        price: Execution price
        pnl: Realized P&L (optional)
        **kwargs: Additional trade data
    """
    task = get_daily_summary_task()
    task.collector.record_trade(
        symbol=symbol,
        action=action,
        quantity=quantity,
        price=price,
        pnl=pnl,
        **kwargs
    )


def record_signal(
    symbol: str,
    signal_type: str,
    indicator: str,
    indicator_value: float,
    **kwargs
):
    """
    Convenience function to record a signal to the global collector.

    Args:
        symbol: Stock symbol
        signal_type: Signal type
        indicator: Indicator name
        indicator_value: Indicator value
        **kwargs: Additional signal data
    """
    task = get_daily_summary_task()
    task.collector.record_signal(
        symbol=symbol,
        signal_type=signal_type,
        indicator=indicator,
        indicator_value=indicator_value,
        **kwargs
    )
