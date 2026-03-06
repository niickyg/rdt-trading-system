"""
Orchestrator Agent
Coordinates all trading agents and manages system lifecycle
"""

import asyncio
from datetime import datetime, time, timezone, timedelta
from typing import Dict, List, Optional
from loguru import logger

from agents.base import BaseAgent, AgentState
from agents.events import Event, EventType, EventBus, get_event_bus
from agents.scanner_agent import ScannerAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.executor_agent import ExecutorAgent
from agents.adaptive_learner import AdaptiveLearner
from agents.daily_stats_agent import DailyStatsAgent
from agents.outcome_tracker import OutcomeTracker
from agents.data_collection_agent import DataCollectionAgent
from strategies.registry import StrategyRegistry


class Orchestrator(BaseAgent):
    """
    Master orchestrator for the trading system

    Responsibilities:
    - Initialize and manage all agents
    - Handle market open/close schedules
    - Coordinate agent communication
    - Monitor system health
    - Handle graceful shutdown
    """

    def __init__(
        self,
        broker,
        risk_manager,
        data_provider,
        watchlist: List[str],
        config: Optional[Dict] = None,
        auto_trade: bool = False,
        **kwargs
    ):
        super().__init__(name="Orchestrator", config=config, **kwargs)

        self.broker = broker
        self.risk_manager = risk_manager
        self.data_provider = data_provider
        self.watchlist = watchlist
        self.auto_trade = auto_trade

        # Agent references
        self.scanner: Optional[ScannerAgent] = None
        self.analyzer: Optional[AnalyzerAgent] = None
        self.executor: Optional[ExecutorAgent] = None
        self.adaptive_learner: Optional[AdaptiveLearner] = None
        self.daily_stats: Optional[DailyStatsAgent] = None
        self.outcome_tracker: Optional[OutcomeTracker] = None
        self.data_collection: Optional[DataCollectionAgent] = None

        # Market schedule (Eastern Time)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)

        # System state
        self.system_started_at: Optional[datetime] = None
        self.is_market_hours = False
        self._halting = False

    async def initialize(self):
        """Initialize the orchestrator and all agents"""
        logger.info("Initializing trading system...")

        # Start event bus
        await self.event_bus.start()

        # Create agents
        self.scanner = ScannerAgent(
            watchlist=self.watchlist,
            data_provider=self.data_provider,
            scan_interval=self.config.get("scan_interval", 60),
            rrs_threshold=self.config.get("rrs_threshold", 2.0),
            event_bus=self.event_bus
        )

        self.analyzer = AnalyzerAgent(
            risk_manager=self.risk_manager,
            event_bus=self.event_bus
        )

        self.executor = ExecutorAgent(
            broker=self.broker,
            auto_execute=self.auto_trade,
            event_bus=self.event_bus
        )

        # Create adaptive learner for self-improvement
        self.adaptive_learner = AdaptiveLearner(
            window_size=self.config.get("learning_window", 20),
            adjustment_frequency=self.config.get("adjustment_frequency", 5),
            learning_rate=self.config.get("learning_rate", 0.1),
            event_bus=self.event_bus
        )

        # Create daily stats agent (runs on schedule + MARKET_CLOSE event)
        account_size = self.config.get("account_size", 25000.0)
        self.daily_stats = DailyStatsAgent(
            account_size=account_size,
            intraday_snapshot_interval=self.config.get("equity_snapshot_interval", 3600.0),
            event_bus=self.event_bus
        )

        # Create outcome tracker (checks rejected signals hourly)
        self.outcome_tracker = OutcomeTracker(
            check_interval=self.config.get("outcome_check_interval", 3600.0),
            lookback_hours=self.config.get("outcome_lookback_hours", 48),
            event_bus=self.event_bus
        )

        # Create data collection agent for ML training data
        self.data_collection = DataCollectionAgent(
            data_provider=self.data_provider,
            watchlist=self.watchlist,
            event_bus=self.event_bus
        )

        # Register trading strategies
        self._register_strategies()

        # Start all agents
        await self.scanner.start()
        await self.analyzer.start()
        await self.executor.start()
        await self.adaptive_learner.start()
        await self.daily_stats.start()
        await self.outcome_tracker.start()
        await self.data_collection.start()

        self.system_started_at = datetime.now()
        self.metrics.custom_metrics["agents_active"] = 7

        logger.info("All agents started successfully")

    async def cleanup(self):
        """Cleanup and stop all agents"""
        logger.info("Shutting down trading system...")

        # Stop agents in reverse order
        if self.data_collection:
            await self.data_collection.stop()
        if self.outcome_tracker:
            await self.outcome_tracker.stop()
        if self.daily_stats:
            await self.daily_stats.stop()
        if self.adaptive_learner:
            await self.adaptive_learner.stop()
        if self.executor:
            await self.executor.stop()
        if self.analyzer:
            await self.analyzer.stop()
        if self.scanner:
            await self.scanner.stop()

        # Stop event bus
        await self.event_bus.stop()

        logger.info("Trading system shut down complete")

    def get_subscribed_events(self) -> List[EventType]:
        """Events orchestrator listens to"""
        return [
            EventType.SYSTEM_ERROR,
            EventType.RISK_CHECK_FAILED,
            EventType.TRADING_HALTED,
            EventType.DAILY_LIMIT_HIT
        ]

    async def handle_event(self, event: Event):
        """Handle system-level events"""
        if event.event_type == EventType.SYSTEM_ERROR:
            await self._handle_error(event)

        elif event.event_type == EventType.TRADING_HALTED:
            # Don't re-publish — just act on external halt events
            await self._halt_trading(event.data.get("reason", "Unknown"), republish=False)

        elif event.event_type == EventType.DAILY_LIMIT_HIT:
            await self._halt_trading("Daily loss limit exceeded")

        elif event.event_type == EventType.RISK_CHECK_FAILED:
            logger.warning(f"Risk check failed: {event.data}")

    async def _handle_error(self, event: Event):
        """Handle system errors"""
        error = event.data.get("error", "Unknown error")
        agent = event.data.get("agent", "Unknown")
        logger.error(f"System error from {agent}: {error}")

        self.metrics.errors += 1

    async def _halt_trading(self, reason: str, republish: bool = True):
        """Halt all trading activity"""
        if self._halting:
            return
        self._halting = True
        logger.warning(f"HALTING TRADING: {reason}")

        # Pause executor
        if self.executor:
            self.executor.pause()

        # Pause scanner
        if self.scanner:
            self.scanner.pause()

        # Only publish if we're the originator (not re-handling an external event)
        if republish:
            await self.publish(EventType.TRADING_HALTED, {
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

    async def resume_trading(self):
        """Resume trading after halt"""
        logger.info("Resuming trading")
        self._halting = False

        if self.scanner:
            self.scanner.resume()
        if self.executor:
            self.executor.resume()

    def check_market_hours(self) -> bool:
        """Check if market is currently open (uses Eastern Time)"""
        from utils.timezone import is_market_open
        return is_market_open()

    async def emit_market_open(self):
        """Emit market open event"""
        await self.publish(EventType.MARKET_OPEN, {
            "timestamp": datetime.now().isoformat()
        })
        self.is_market_hours = True
        logger.info("Market OPEN")

    async def emit_market_close(self):
        """Emit market close event"""
        await self.publish(EventType.MARKET_CLOSE, {
            "timestamp": datetime.now().isoformat()
        })
        self.is_market_hours = False
        logger.info("Market CLOSED")

    def _register_strategies(self):
        """
        Register trading strategies based on config.

        Control which strategies are active via config dict or env vars:
            STRATEGY_RSI2_ENABLED=true
            STRATEGY_TREND_ENABLED=true
            STRATEGY_PEAD_ENABLED=true
            STRATEGY_GAP_FILL_ENABLED=true

        RRS Momentum is always registered (core strategy).
        """
        import os

        # Always register the core RRS Momentum strategy
        from strategies.rrs_momentum import create_default as create_rrs
        create_rrs()

        # Optional strategies — enabled via config or env vars
        strategy_map = {
            'rsi2': {
                'env': 'STRATEGY_RSI2_ENABLED',
                'config_key': 'strategy_rsi2_enabled',
                'factory': 'strategies.rsi2_mean_reversion:create_default',
            },
            'trend': {
                'env': 'STRATEGY_TREND_ENABLED',
                'config_key': 'strategy_trend_enabled',
                'factory': 'strategies.trend_breakout:create_default',
            },
            'pead': {
                'env': 'STRATEGY_PEAD_ENABLED',
                'config_key': 'strategy_pead_enabled',
                'factory': 'strategies.pead:create_default',
            },
            'gap_fill': {
                'env': 'STRATEGY_GAP_FILL_ENABLED',
                'config_key': 'strategy_gap_fill_enabled',
                'factory': 'strategies.gap_fill:create_default',
            },
        }

        for name, info in strategy_map.items():
            # Check env var first, then config dict
            env_val = os.environ.get(info['env'], '').lower()
            config_val = self.config.get(info['config_key'], False)
            enabled = env_val in ('true', '1', 'yes') or config_val

            if enabled:
                try:
                    module_path, func_name = info['factory'].split(':')
                    import importlib
                    mod = importlib.import_module(module_path)
                    factory = getattr(mod, func_name)
                    strategy = factory()
                    logger.info(f"Strategy '{strategy.name}' enabled")
                except Exception as e:
                    logger.error(f"Failed to register strategy '{name}': {e}")
            else:
                logger.debug(f"Strategy '{name}' disabled (set {info['env']}=true to enable)")

        active = StrategyRegistry.get_active()
        logger.info(
            f"Registered {StrategyRegistry.count()} strategies, "
            f"{len(active)} active: {[s.name for s in active]}"
        )

    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            "orchestrator": self.state.value,
            "scanner": self.scanner.state.value if self.scanner else "not_started",
            "analyzer": self.analyzer.state.value if self.analyzer else "not_started",
            "executor": self.executor.state.value if self.executor else "not_started",
            "adaptive_learner": self.adaptive_learner.state.value if self.adaptive_learner else "not_started",
            "daily_stats": self.daily_stats.state.value if self.daily_stats else "not_started",
            "outcome_tracker": self.outcome_tracker.state.value if self.outcome_tracker else "not_started",
            "data_collection": self.data_collection.state.value if self.data_collection else "not_started",
            "market_hours": self.is_market_hours,
            "auto_trade": self.auto_trade,
            "strategies": {
                "registered": StrategyRegistry.get_names(),
                "active": [s.name for s in StrategyRegistry.get_active()],
            },
            "uptime": (datetime.now() - self.system_started_at).total_seconds() if self.system_started_at else 0
        }

    def get_all_metrics(self) -> Dict:
        """Get metrics from all agents"""
        metrics = {
            "orchestrator": self.get_metrics(),
            "scanner": self.scanner.get_metrics() if self.scanner else {},
            "analyzer": self.analyzer.get_metrics() if self.analyzer else {},
            "executor": self.executor.get_metrics() if self.executor else {}
        }

        # Add adaptive learner parameters if available
        if self.adaptive_learner:
            metrics["adaptive_learner"] = self.adaptive_learner.get_current_parameters()

        return metrics


async def _market_close_watchdog(orchestrator: Orchestrator):
    """
    Background task that monitors the clock and manages market open/close lifecycle.
    Emits MARKET_OPEN at 9:30 AM ET and MARKET_CLOSE at 4:00 PM ET (or 1:00 PM on
    early-close days). Runs continuously so the orchestrator survives across trading days.
    Checks every 60 seconds.
    """
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    eastern = ZoneInfo("America/New_York")
    market_open = time(9, 30)
    market_open_emitted = False
    market_close_handled_today = False
    last_date = None

    # Import early-close support
    try:
        from utils.timezone import is_early_close_day
    except ImportError:
        def is_early_close_day(dt=None):
            return False

    while orchestrator.is_running:
        now_et = datetime.now(eastern)
        today = now_et.date()

        # Reset flags at the start of a new calendar day
        if last_date is not None and today != last_date:
            market_open_emitted = False
            market_close_handled_today = False
        last_date = today

        # Determine market close time (1:00 PM on early-close days, 4:00 PM otherwise)
        if is_early_close_day(now_et):
            market_close = time(13, 0)
        else:
            market_close = time(16, 0)

        # Emit MARKET_OPEN once when 9:30 AM ET is reached
        if not market_open_emitted and now_et.time() >= market_open and now_et.time() < market_close:
            market_open_emitted = True
            logger.info("Market open detected (9:30 AM ET) — emitting MARKET_OPEN event")
            await orchestrator.emit_market_open()

        # Handle market close once per day (agents pause, but orchestrator keeps running)
        if not market_close_handled_today and now_et.time() >= market_close:
            market_close_handled_today = True
            close_label = "1:00 PM" if market_close == time(13, 0) else "4:00 PM"
            logger.info(f"Market close reached ({close_label} ET) — cancelling orders and pausing agents")

            # Cancel open orders
            if orchestrator.executor and hasattr(orchestrator.executor, 'broker'):
                try:
                    open_orders = orchestrator.broker.get_open_orders()
                    for order in open_orders:
                        try:
                            orchestrator.broker.cancel_order(order.order_id)
                            logger.info(f"Cancelled open order: {order.order_id}")
                        except Exception as e:
                            logger.warning(f"Failed to cancel order {order.order_id}: {e}")
                except Exception as e:
                    logger.warning(f"Could not retrieve open orders for cleanup: {e}")

            # Emit market close event (agents will pause themselves)
            await orchestrator.emit_market_close()

        await asyncio.sleep(60)


import threading

_running_orchestrator: Optional[Orchestrator] = None
_orchestrator_lock = threading.Lock()


def set_running_orchestrator(orch: Optional[Orchestrator]):
    """Set the module-level orchestrator reference (thread-safe)."""
    global _running_orchestrator
    with _orchestrator_lock:
        _running_orchestrator = orch


def get_running_orchestrator() -> Optional[Orchestrator]:
    """Get the running orchestrator instance (thread-safe)."""
    with _orchestrator_lock:
        return _running_orchestrator


async def run_trading_system(
    broker,
    risk_manager,
    data_provider,
    watchlist: List[str],
    config: Dict = None,
    auto_trade: bool = False
):
    """
    Main entry point to run the trading system

    Args:
        broker: Broker instance
        risk_manager: RiskManager instance
        data_provider: Data provider instance
        watchlist: List of symbols to scan
        config: Configuration dictionary
        auto_trade: Enable automatic trade execution
    """
    orchestrator = Orchestrator(
        broker=broker,
        risk_manager=risk_manager,
        data_provider=data_provider,
        watchlist=watchlist,
        config=config or {},
        auto_trade=auto_trade
    )

    watchdog_task = None
    try:
        await orchestrator.start()
        set_running_orchestrator(orchestrator)

        # Start market close watchdog (runs continuously across trading days)
        watchdog_task = asyncio.create_task(_market_close_watchdog(orchestrator))

        # Run until interrupted — watchdog keeps orchestrator alive across days
        while orchestrator.is_running:
            await asyncio.sleep(1)

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown requested")

    finally:
        set_running_orchestrator(None)
        if watchdog_task and not watchdog_task.done():
            watchdog_task.cancel()
        await orchestrator.stop()
