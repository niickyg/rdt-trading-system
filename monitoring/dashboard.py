"""
Trading Dashboard
Real-time monitoring of the trading system
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger

from agents.events import EventType, Event, subscribe, get_event_bus


@dataclass
class DashboardState:
    """Current dashboard state"""
    # System status
    system_status: str = "stopped"
    agents_running: int = 0
    uptime_seconds: float = 0

    # Market
    spy_price: float = 0
    spy_change_pct: float = 0
    market_status: str = "closed"

    # Scanning
    last_scan_time: Optional[datetime] = None
    signals_today: int = 0
    strong_rs_count: int = 0
    strong_rw_count: int = 0

    # Trading
    trades_today: int = 0
    pending_setups: int = 0
    orders_pending: int = 0

    # Portfolio
    open_positions: int = 0
    total_exposure: float = 0
    unrealized_pnl: float = 0
    realized_pnl: float = 0

    # Risk
    daily_pnl: float = 0
    daily_pnl_pct: float = 0
    current_drawdown: float = 0
    daily_limit_pct: float = 0
    trading_halted: bool = False

    # Recent activity
    recent_signals: List[Dict] = field(default_factory=list)
    recent_trades: List[Dict] = field(default_factory=list)
    recent_events: List[Dict] = field(default_factory=list)


class TradingDashboard:
    """
    Real-time trading system dashboard

    Provides monitoring and visualization of:
    - System status and health
    - Active positions and P&L
    - Recent signals and trades
    - Risk metrics
    """

    def __init__(self, max_history: int = 50):
        self.state = DashboardState()
        self.max_history = max_history
        self._running = False

        # Subscribe to events
        self._setup_subscriptions()

        logger.info("Dashboard initialized")

    def _setup_subscriptions(self):
        """Subscribe to relevant events"""
        subscribe(EventType.SYSTEM_START, self._on_system_start)
        subscribe(EventType.SYSTEM_STOP, self._on_system_stop)
        subscribe(EventType.SCAN_COMPLETED, self._on_scan_completed)
        subscribe(EventType.SIGNAL_FOUND, self._on_signal_found)
        subscribe(EventType.SETUP_VALID, self._on_setup_valid)
        subscribe(EventType.ORDER_FILLED, self._on_order_filled)
        subscribe(EventType.POSITION_OPENED, self._on_position_opened)
        subscribe(EventType.POSITION_CLOSED, self._on_position_closed)
        subscribe(EventType.TRADING_HALTED, self._on_trading_halted)
        subscribe(EventType.MARKET_OPEN, self._on_market_open)
        subscribe(EventType.MARKET_CLOSE, self._on_market_close)
        subscribe(EventType.PNL_UPDATED, self._on_pnl_updated)

    def _add_event(self, event_type: str, data: Dict):
        """Add event to recent history"""
        entry = {
            "type": event_type,
            "time": datetime.now().strftime("%H:%M:%S"),
            "data": data
        }
        self.state.recent_events.insert(0, entry)
        if len(self.state.recent_events) > self.max_history:
            self.state.recent_events.pop()

    # Event handlers
    def _on_system_start(self, event: Event):
        self.state.system_status = "running"
        self._add_event("system_start", event.data)

    def _on_system_stop(self, event: Event):
        self.state.system_status = "stopped"
        self._add_event("system_stop", event.data)

    def _on_scan_completed(self, event: Event):
        self.state.last_scan_time = datetime.now()
        self.state.strong_rs_count = event.data.get("strong_rs_count", 0)
        self.state.strong_rw_count = event.data.get("strong_rw_count", 0)

    def _on_signal_found(self, event: Event):
        self.state.signals_today += 1

        signal = {
            "symbol": event.data.get("symbol"),
            "direction": event.data.get("direction"),
            "rrs": event.data.get("rrs"),
            "price": event.data.get("price"),
            "time": datetime.now().strftime("%H:%M:%S")
        }
        self.state.recent_signals.insert(0, signal)
        if len(self.state.recent_signals) > self.max_history:
            self.state.recent_signals.pop()

        self._add_event("signal", event.data)

    def _on_setup_valid(self, event: Event):
        self.state.pending_setups += 1
        self._add_event("setup_valid", event.data)

    def _on_order_filled(self, event: Event):
        self.state.trades_today += 1
        self.state.orders_pending = max(0, self.state.orders_pending - 1)

        trade = {
            "symbol": event.data.get("symbol"),
            "direction": event.data.get("direction"),
            "shares": event.data.get("shares"),
            "price": event.data.get("fill_price"),
            "time": datetime.now().strftime("%H:%M:%S")
        }
        self.state.recent_trades.insert(0, trade)
        if len(self.state.recent_trades) > self.max_history:
            self.state.recent_trades.pop()

        self._add_event("order_filled", event.data)

    def _on_position_opened(self, event: Event):
        self.state.open_positions += 1
        self.state.pending_setups = max(0, self.state.pending_setups - 1)
        self._add_event("position_opened", event.data)

    def _on_position_closed(self, event: Event):
        self.state.open_positions = max(0, self.state.open_positions - 1)
        pnl = event.data.get("pnl", 0)
        self.state.realized_pnl += pnl
        self._add_event("position_closed", event.data)

    def _on_trading_halted(self, event: Event):
        self.state.trading_halted = True
        self._add_event("trading_halted", event.data)

    def _on_market_open(self, event: Event):
        self.state.market_status = "open"
        # Reset daily counters
        self.state.signals_today = 0
        self.state.trades_today = 0
        self.state.daily_pnl = 0
        self.state.realized_pnl = 0

    def _on_market_close(self, event: Event):
        self.state.market_status = "closed"

    def _on_pnl_updated(self, event: Event):
        self.state.unrealized_pnl = event.data.get("unrealized", 0)
        self.state.daily_pnl = event.data.get("daily_pnl", 0)

    def update_portfolio(
        self,
        positions: int,
        exposure: float,
        unrealized: float,
        realized: float
    ):
        """Update portfolio metrics"""
        self.state.open_positions = positions
        self.state.total_exposure = exposure
        self.state.unrealized_pnl = unrealized
        self.state.realized_pnl = realized

    def update_risk(
        self,
        daily_pnl: float,
        daily_pnl_pct: float,
        drawdown: float,
        limit_pct: float
    ):
        """Update risk metrics"""
        self.state.daily_pnl = daily_pnl
        self.state.daily_pnl_pct = daily_pnl_pct
        self.state.current_drawdown = drawdown
        self.state.daily_limit_pct = limit_pct

    def update_spy(self, price: float, change_pct: float):
        """Update SPY data"""
        self.state.spy_price = price
        self.state.spy_change_pct = change_pct

    def get_state(self) -> Dict:
        """Get current dashboard state as dict"""
        return {
            "system": {
                "status": self.state.system_status,
                "agents_running": self.state.agents_running,
                "uptime_seconds": self.state.uptime_seconds,
                "trading_halted": self.state.trading_halted
            },
            "market": {
                "status": self.state.market_status,
                "spy_price": self.state.spy_price,
                "spy_change_pct": self.state.spy_change_pct
            },
            "scanning": {
                "last_scan": self.state.last_scan_time.isoformat() if self.state.last_scan_time else None,
                "signals_today": self.state.signals_today,
                "strong_rs": self.state.strong_rs_count,
                "strong_rw": self.state.strong_rw_count
            },
            "trading": {
                "trades_today": self.state.trades_today,
                "pending_setups": self.state.pending_setups,
                "orders_pending": self.state.orders_pending
            },
            "portfolio": {
                "open_positions": self.state.open_positions,
                "total_exposure": self.state.total_exposure,
                "unrealized_pnl": self.state.unrealized_pnl,
                "realized_pnl": self.state.realized_pnl,
                "total_pnl": self.state.unrealized_pnl + self.state.realized_pnl
            },
            "risk": {
                "daily_pnl": self.state.daily_pnl,
                "daily_pnl_pct": self.state.daily_pnl_pct,
                "current_drawdown": self.state.current_drawdown,
                "daily_limit_pct": self.state.daily_limit_pct
            },
            "recent_signals": self.state.recent_signals[:10],
            "recent_trades": self.state.recent_trades[:10],
            "recent_events": self.state.recent_events[:20]
        }

    def render_console(self) -> str:
        """Render dashboard for console output"""
        state = self.get_state()

        lines = [
            "=" * 60,
            "RDT TRADING SYSTEM DASHBOARD",
            "=" * 60,
            "",
            f"System: {state['system']['status'].upper()}  "
            f"Market: {state['market']['status'].upper()}  "
            f"SPY: ${state['market']['spy_price']:.2f} ({state['market']['spy_change_pct']:+.2f}%)",
            "",
            "--- Portfolio ---",
            f"Positions: {state['portfolio']['open_positions']}  "
            f"Exposure: ${state['portfolio']['total_exposure']:,.0f}",
            f"Unrealized P&L: ${state['portfolio']['unrealized_pnl']:+,.2f}  "
            f"Realized: ${state['portfolio']['realized_pnl']:+,.2f}",
            "",
            "--- Risk ---",
            f"Daily P&L: ${state['risk']['daily_pnl']:+,.2f} ({state['risk']['daily_pnl_pct']:+.2f}%)  "
            f"Drawdown: ${state['risk']['current_drawdown']:,.0f}",
            f"{'*** TRADING HALTED ***' if state['system']['trading_halted'] else ''}",
            "",
            "--- Today ---",
            f"Signals: {state['scanning']['signals_today']}  "
            f"Trades: {state['trading']['trades_today']}  "
            f"RS: {state['scanning']['strong_rs']} / RW: {state['scanning']['strong_rw']}",
            "",
            "--- Recent Signals ---",
        ]

        for sig in state['recent_signals'][:5]:
            lines.append(
                f"  {sig['time']} {sig['symbol']} {sig['direction'].upper()} "
                f"RRS={sig['rrs']:.2f} @ ${sig['price']:.2f}"
            )

        if state['recent_trades']:
            lines.append("")
            lines.append("--- Recent Trades ---")
            for trade in state['recent_trades'][:5]:
                lines.append(
                    f"  {trade['time']} {trade['symbol']} {trade['direction'].upper()} "
                    f"{trade['shares']} @ ${trade['price']:.2f}"
                )

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    async def run_console_loop(self, refresh_seconds: float = 5.0):
        """Run dashboard in console with periodic refresh"""
        self._running = True
        try:
            while self._running:
                # Clear screen
                print("\033[H\033[J", end="")
                print(self.render_console())
                await asyncio.sleep(refresh_seconds)
        except KeyboardInterrupt:
            pass

    def stop(self):
        """Stop the dashboard loop"""
        self._running = False
