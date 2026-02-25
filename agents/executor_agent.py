"""
Executor Agent
Executes trades through the broker
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

from agents.base import BaseAgent, AgentState
from agents.events import Event, EventType
from brokers.base import AbstractBroker, OrderSide, OrderType
from data.database import get_trades_repository

# Trailing stop support
try:
    from trading.advanced_orders import TrailingStopOrder, AdvancedOrderManager
    TRAILING_STOP_AVAILABLE = True
except ImportError:
    TRAILING_STOP_AVAILABLE = False
    logger.warning("Advanced orders module not available for trailing stops")

# ML-based exit predictor
try:
    from ml.exit_predictor import ExitPredictor, ExitPrediction, EXIT_STRATEGIES
    EXIT_PREDICTOR_AVAILABLE = True
except ImportError:
    EXIT_PREDICTOR_AVAILABLE = False
    logger.debug("ExitPredictor not available")

# ML-based entry timing predictor
try:
    from ml.entry_timing import EntryTimingModel, EntryTimingPrediction
    ENTRY_TIMING_AVAILABLE = True
except ImportError:
    ENTRY_TIMING_AVAILABLE = False
    logger.debug("EntryTimingModel not available")

# Dynamic position sizing -- record trade outcomes for rolling Kelly stats
try:
    from ml.dynamic_sizer import get_dynamic_sizer
    DYNAMIC_SIZER_AVAILABLE = True
except ImportError:
    DYNAMIC_SIZER_AVAILABLE = False

# Options trading support
try:
    from options.config import OptionsConfig
    from options.chain import OptionsChainManager
    from options.chain_provider import ChainProvider, IBKRChainProvider, PaperChainProvider
    from options.iv_analyzer import IVAnalyzer
    from options.strategy_selector import StrategySelector
    from options.position_sizer import OptionsPositionSizer
    from options.executor import OptionsExecutor, IBKROptionsExecutor
    from options.paper_executor import PaperOptionsExecutor
    from options.exit_manager import OptionsExitManager
    from options.risk import OptionsRiskManager
    from options.pricing import DEFAULT_RISK_FREE_RATE
    OPTIONS_AVAILABLE = True
except ImportError:
    OPTIONS_AVAILABLE = False
    logger.debug("Options trading module not available")

# Limit order timeout for entry timing pullback orders (30 minutes)
ENTRY_TIMING_LIMIT_TIMEOUT_MINUTES = 30


class ExecutorAgent(BaseAgent):
    """
    Trade execution agent

    Responsibilities:
    - Execute trades from approved setups
    - Place stop-loss orders
    - Place take-profit orders
    - Handle order confirmations
    - Track execution quality
    """

    def __init__(
        self,
        broker: AbstractBroker,
        auto_execute: bool = False,
        **kwargs
    ):
        super().__init__(name="ExecutorAgent", **kwargs)

        self.broker = broker
        self.auto_execute = auto_execute

        # Pending setups awaiting execution
        self.pending_setups: Dict[str, Dict] = {}

        # Active orders
        self.active_orders: Dict[str, Dict] = {}

        # Execution metrics
        self.orders_placed = 0
        self.orders_filled = 0
        self.orders_rejected = 0

        # Trailing stop configuration (enabled by default when available)
        self.trailing_stop_enabled = TRAILING_STOP_AVAILABLE
        self.trailing_stop_activation_r = 1.0  # Activate after +1R profit
        self.trailing_stop_atr_multiplier = 0.75  # Trail at 0.75x ATR
        self._position_risk: Dict[str, Dict] = {}  # symbol -> {entry, stop, atr, direction}
        self._trailing_stops: Dict[str, "TrailingStopOrder"] = {}  # symbol -> TrailingStopOrder

        # ML-based exit predictor
        self._exit_predictor: Optional["ExitPredictor"] = None
        self._exit_predictor_enabled = EXIT_PREDICTOR_AVAILABLE
        if self._exit_predictor_enabled:
            try:
                self._exit_predictor = ExitPredictor()
                self._exit_predictor.load()
                logger.info("ExitPredictor loaded successfully — dynamic exit strategies enabled")
            except FileNotFoundError:
                logger.info(
                    "ExitPredictor model not found — using default exit strategies. "
                    "Train with: ml/exit_predictor.py generate_exit_training_data()"
                )
                self._exit_predictor = None
                self._exit_predictor_enabled = False
            except Exception as e:
                logger.warning(f"Failed to load ExitPredictor: {e}")
                self._exit_predictor = None
                self._exit_predictor_enabled = False

        # ML-based entry timing predictor
        self._entry_timing: Optional["EntryTimingModel"] = None
        self._entry_timing_enabled = ENTRY_TIMING_AVAILABLE
        if self._entry_timing_enabled:
            try:
                self._entry_timing = EntryTimingModel()
                self._entry_timing.load()
                logger.info("EntryTimingModel loaded successfully — smart entry timing enabled")
            except FileNotFoundError:
                logger.info(
                    "EntryTimingModel not found — using immediate market orders. "
                    "Train with: ml/entry_timing.py generate_entry_timing_data()"
                )
                self._entry_timing = None
                self._entry_timing_enabled = False
            except Exception as e:
                logger.warning(f"Failed to load EntryTimingModel: {e}")
                self._entry_timing = None
                self._entry_timing_enabled = False

        # Pending limit orders from entry timing (symbol -> {order_id, setup, placed_at})
        self._entry_timing_limits: Dict[str, Dict] = {}

        # Options trading components
        self._options_enabled = False
        self._strategy_selector: Optional["StrategySelector"] = None
        self._options_executor: Optional["OptionsExecutor"] = None
        self._options_exit_manager: Optional["OptionsExitManager"] = None
        self._options_risk_manager: Optional["OptionsRiskManager"] = None
        self._options_position_sizer: Optional["OptionsPositionSizer"] = None

        if OPTIONS_AVAILABLE:
            try:
                from config.settings import get_settings
                settings = get_settings()
                options_config = getattr(settings, 'options', None)
                if options_config and options_config.enabled:
                    # Detect broker type and create appropriate provider + executor
                    broker_type = getattr(settings, 'broker_type', 'paper')
                    if hasattr(settings, 'trading'):
                        broker_type = getattr(settings.trading, 'broker_type', broker_type)

                    is_paper = broker_type == 'paper' or getattr(settings, 'paper_trading', True)

                    if is_paper:
                        # Paper trading: use BS-based synthetic chain provider
                        r = float(os.environ.get("OPTIONS_RISK_FREE_RATE", str(DEFAULT_RISK_FREE_RATE)))
                        iv_mult = float(os.environ.get("OPTIONS_PAPER_IV_MULTIPLIER", "1.1"))
                        provider = PaperChainProvider(risk_free_rate=r, iv_multiplier=iv_mult)
                        broker_executor = PaperOptionsExecutor(provider, options_config, risk_free_rate=r)
                        logger.info("Options: using PaperChainProvider + PaperOptionsExecutor")
                    else:
                        # IBKR: use real IBKR chain provider
                        provider = IBKRChainProvider(broker)
                        broker_executor = IBKROptionsExecutor(broker, options_config)
                        logger.info("Options: using IBKRChainProvider + IBKROptionsExecutor")

                    chain_mgr = OptionsChainManager(provider, options_config)
                    iv_analyzer = IVAnalyzer(provider, chain_mgr)
                    self._strategy_selector = StrategySelector(chain_mgr, iv_analyzer, options_config)
                    self._options_executor = OptionsExecutor(broker_executor, options_config)
                    self._options_exit_manager = OptionsExitManager(chain_mgr, options_config)
                    self._options_risk_manager = OptionsRiskManager(chain_mgr, options_config)
                    self._options_position_sizer = OptionsPositionSizer(options_config)
                    self._options_enabled = True
                    logger.info("Options trading enabled — strategy selector + executor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize options trading: {e}")
                self._options_enabled = False

        # Database repository for trade persistence
        self._trades_repo = None

        # Periodic position monitoring interval (seconds)
        self._monitor_interval = 30
        self._monitor_task: Optional[asyncio.Task] = None

    @property
    def trades_repo(self):
        """Lazy-load trades repository."""
        if self._trades_repo is None:
            self._trades_repo = get_trades_repository()
        return self._trades_repo

    async def initialize(self):
        """Initialize executor"""
        logger.info(f"Executor initialized (auto_execute={self.auto_execute})")
        self.metrics.custom_metrics["orders_placed"] = 0
        self.metrics.custom_metrics["fill_rate"] = 0

    async def start(self):
        """Start executor and launch periodic position monitor."""
        await super().start()
        self._monitor_task = asyncio.create_task(self._position_monitor_loop())
        logger.info(f"Position monitor loop started (interval={self._monitor_interval}s)")

    async def stop(self):
        """Stop executor and cancel periodic monitor."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        await super().stop()

    async def _position_monitor_loop(self):
        """
        Periodic loop that checks trailing stops, options exits,
        and entry timing timeouts.

        Runs every _monitor_interval seconds alongside the event-driven
        handler. This mirrors the ScheduledAgent pattern used by ScannerAgent.
        """
        while self.state == AgentState.RUNNING:
            try:
                await self._run_periodic_checks()
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                self.metrics.errors += 1
            await asyncio.sleep(self._monitor_interval)

    async def _run_periodic_checks(self):
        """
        Run all periodic position management checks.

        Called every _monitor_interval seconds. Handles:
        1. Trailing stop activation for positions reaching +1R
        2. Trailing stop price updates for already-active trails
        3. Options position exit checks (expiry, profit target, stop loss)
        4. Entry timing limit order timeout cancellation
        """
        # Collect current prices for all tracked positions
        price_updates: Dict[str, float] = {}
        for symbol in list(self._position_risk.keys()):
            try:
                quote = self.broker.get_quote(symbol)
                if quote and hasattr(quote, 'last') and quote.last:
                    price_updates[symbol] = quote.last
            except Exception as e:
                logger.debug(f"Could not fetch price for {symbol}: {e}")

        # 1. Check trailing stop activation for each position
        for symbol, price in price_updates.items():
            self.check_trailing_stop_activation(symbol, price)

        # 2. Update active trailing stops with latest prices
        self.update_trailing_stops(price_updates if price_updates else None)

        # 3. Check options positions for exit triggers
        await self.check_options_exits()

        # 4. Check entry timing limit orders for timeouts
        await self.check_entry_timing_timeouts()

    async def cleanup(self):
        """Cleanup executor"""
        # Cancel any pending entry timing limit orders
        for symbol in list(self._entry_timing_limits.keys()):
            limit_data = self._entry_timing_limits.pop(symbol)
            try:
                self.broker.cancel_order(limit_data['order_id'])
            except Exception as e:
                logger.debug(f"Error canceling entry timing limit for {symbol}: {e}")

        # Cancel any pending orders on shutdown
        for order_id in list(self.active_orders.keys()):
            try:
                self.broker.cancel_order(order_id)
            except Exception as e:
                logger.error(f"Error canceling order {order_id}: {e}")

    def get_subscribed_events(self) -> List[EventType]:
        """Events this agent listens to"""
        return [
            EventType.SETUP_VALID,
            EventType.ORDER_REQUESTED,
            EventType.POSITION_CLOSED
        ]

    async def handle_event(self, event: Event):
        """Handle incoming events"""
        if event.event_type == EventType.SETUP_VALID:
            await self.handle_valid_setup(event.data)

        elif event.event_type == EventType.ORDER_REQUESTED:
            await self.execute_order(event.data)

        elif event.event_type == EventType.POSITION_CLOSED:
            # Clean up any related orders and update database
            symbol = event.data.get("symbol")
            await self._handle_position_closed(event.data)
            await self.cancel_orders_for_symbol(symbol)

    async def _handle_position_closed(self, data: Dict):
        """Handle position close event and update database"""
        symbol = data.get("symbol")
        exit_price = data.get("exit_price")
        exit_reason = data.get("exit_reason", "manual")
        exit_time = datetime.now()

        if not symbol:
            logger.warning("Position closed event missing symbol")
            return

        # Close the trade in database
        if exit_price:
            closed_trade = self.trades_repo.close_trade_by_symbol(
                symbol=symbol,
                exit_price=exit_price,
                exit_reason=exit_reason,
                exit_time=exit_time
            )
            if closed_trade:
                logger.info(f"Trade closed in database: {symbol} PnL=${closed_trade.get('pnl', 0):.2f}")

                # Record trade result for dynamic position sizer's rolling stats
                self._record_dynamic_sizer_outcome(symbol, closed_trade, data)

        # Remove position from database
        self.trades_repo.close_position(symbol)
        logger.info(f"Position removed from database: {symbol}")

        # Clean up options executor in-memory positions
        if self._options_enabled and self._options_executor:
            if symbol in self._options_executor._positions:
                del self._options_executor._positions[symbol]
                logger.info(f"Options in-memory position cleaned up: {symbol}")

        # Clean up entry timing limit tracking
        self._entry_timing_limits.pop(symbol, None)

        # Clean up trailing stop tracking
        self._position_risk.pop(symbol, None)
        trail = self._trailing_stops.pop(symbol, None)
        if trail:
            try:
                trail.cancel()
            except Exception as e:
                logger.debug(f"Error cancelling trailing stop for {symbol}: {e}")

    def _apply_exit_prediction(self, setup: Dict) -> Dict:
        """
        Apply ML-based exit prediction to adjust the setup's exit strategy.

        If the ExitPredictor is available and trained, it predicts whether the
        trade is likely a quick_scalp, swing, or runner, and adjusts the
        stop/target/trailing parameters accordingly.

        Args:
            setup: Trade setup dictionary with at least symbol, direction,
                   entry_price, stop_price, target_price, atr.

        Returns:
            Updated setup dict (modified in place and returned).
        """
        if not self._exit_predictor_enabled or self._exit_predictor is None:
            return setup

        try:
            # Build a signal-like dict for feature extraction
            signal_data = {
                'rrs': setup.get('rrs', 0),
                'atr': setup.get('atr', 0),
                'price': setup.get('entry_price', 0),
                'entry_price': setup.get('entry_price', 0),
                'direction': setup.get('direction', 'long'),
                'volume_ratio': setup.get('volume_ratio', 1.0),
                'daily_strong': setup.get('daily_strong', False),
                'daily_weak': setup.get('daily_weak', False),
                'rsi_14': setup.get('rsi_14', 50),
                'daily_strength_score': setup.get('daily_strength_score', 0),
                'volume_trend': setup.get('volume_trend', 1.0),
                'market_regime': setup.get('market_regime', 0),
            }

            # Pass raw stock data if available for richer feature extraction
            stock_data = setup.get('_raw_stock_data')
            features = self._exit_predictor.extract_features(signal_data, stock_data)
            prediction = self._exit_predictor.predict(features)

            entry_price = setup.get('entry_price', 0)
            atr = setup.get('atr', 0)
            direction = setup.get('direction', 'long')

            if entry_price <= 0 or atr <= 0:
                return setup

            stop_distance = atr * 0.75  # Standard 0.75x ATR stop

            # Apply predicted strategy to target and trailing stop params
            if prediction.strategy == 'quick_scalp':
                target_distance = stop_distance * prediction.recommended_target_r
                setup['exit_strategy'] = 'quick_scalp'
                setup['trailing_stop_enabled'] = False

            elif prediction.strategy == 'swing':
                target_distance = stop_distance * prediction.recommended_target_r
                setup['exit_strategy'] = 'swing'
                setup['trailing_stop_enabled'] = True
                setup['trailing_stop_activation_r'] = prediction.trail_activation_r
                setup['trailing_stop_atr_multiplier'] = prediction.recommended_trail_r

            elif prediction.strategy == 'runner':
                target_distance = stop_distance * prediction.recommended_target_r
                setup['exit_strategy'] = 'runner'
                setup['trailing_stop_enabled'] = True
                setup['trailing_stop_activation_r'] = prediction.trail_activation_r
                setup['trailing_stop_atr_multiplier'] = prediction.recommended_trail_r

            else:
                return setup

            # Update target price
            if direction == 'long':
                setup['target_price'] = entry_price + target_distance
            else:
                setup['target_price'] = entry_price - target_distance

            # Store prediction metadata
            setup['exit_prediction'] = {
                'strategy': prediction.strategy,
                'mfe_class': prediction.mfe_class,
                'confidence': prediction.confidence,
                'class_probabilities': prediction.class_probabilities,
                'target_r': prediction.recommended_target_r,
                'trail_r': prediction.recommended_trail_r,
                'hold_estimate_hours': prediction.hold_estimate_hours,
            }

            logger.info(
                f"ExitPredictor: {setup.get('symbol')} -> {prediction.strategy} "
                f"(confidence={prediction.confidence:.2f}, "
                f"target={prediction.recommended_target_r}R, "
                f"trail={prediction.recommended_trail_r}R)"
            )

        except Exception as e:
            logger.warning(f"ExitPredictor failed for {setup.get('symbol')}: {e}")
            # Fall through with original setup -- no changes

        return setup

    async def handle_valid_setup(self, setup: Dict):
        """Handle a valid setup from analyzer"""
        symbol = setup.get("symbol")

        # Apply ML-based exit prediction to optimize exit strategy
        setup = self._apply_exit_prediction(setup)

        if self.auto_execute:
            logger.info(f"Auto-executing: {symbol}")
            await self.execute_trade(setup)
        else:
            # Store for manual execution
            self.pending_setups[symbol] = setup
            logger.info(f"Setup pending manual execution: {symbol}")

            await self.publish(EventType.ORDER_REQUESTED, {
                "symbol": symbol,
                "setup": setup,
                "awaiting_confirmation": True
            })

    def _apply_entry_timing(self, setup: Dict) -> Optional["EntryTimingPrediction"]:
        """
        Apply ML-based entry timing prediction to decide HOW to enter.

        If the EntryTimingModel is available and trained, it predicts whether to
        enter immediately (market order), wait for a pullback (limit order), or
        skip the trade entirely.

        Requires '_bars_5m' key in setup (DataFrame of recent 5m bars).  If not
        present, the model is skipped and execution proceeds normally.

        Args:
            setup: Trade setup dictionary.

        Returns:
            EntryTimingPrediction if the model ran, or None to proceed normally.
        """
        if not self._entry_timing_enabled or self._entry_timing is None:
            return None

        bars_5m = setup.get('_bars_5m')
        if bars_5m is None or (hasattr(bars_5m, '__len__') and len(bars_5m) < 3):
            logger.debug(
                f"EntryTiming: skipping for {setup.get('symbol')} — no 5m bars provided"
            )
            return None

        try:
            signal_data = {
                'direction': setup.get('direction', 'long'),
            }
            features = self._entry_timing.extract_features(bars_5m, signal_data)
            prediction = self._entry_timing.predict(features)

            logger.info(
                f"EntryTiming: {setup.get('symbol')} -> {prediction.entry_action} "
                f"(confidence={prediction.confidence:.2f}, "
                f"pullback={prediction.expected_pullback_pct:.2f}%)"
            )

            # Store prediction metadata on setup
            setup['entry_timing_prediction'] = {
                'action': prediction.entry_action,
                'expected_pullback_pct': prediction.expected_pullback_pct,
                'confidence': prediction.confidence,
                'class_probabilities': prediction.class_probabilities,
            }

            return prediction

        except Exception as e:
            logger.warning(f"EntryTiming failed for {setup.get('symbol')}: {e}")
            return None

    async def execute_trade(self, setup: Dict):
        """Execute a trade setup"""
        symbol = setup.get("symbol")
        direction = setup.get("direction")
        shares = setup.get("shares")
        entry_price = setup.get("entry_price")
        stop_price = setup.get("stop_price")
        target_price = setup.get("target_price")

        # --- Entry timing check ---
        timing = self._apply_entry_timing(setup)
        if timing is not None:
            if timing.entry_action == 'skip':
                logger.info(
                    f"EntryTiming SKIP: {symbol} — model recommends skipping "
                    f"(confidence={timing.confidence:.2f})"
                )
                await self._order_rejected(setup, "Entry timing model recommends skip")
                return

            if timing.entry_action == 'wait_for_pullback':
                pullback_pct = timing.expected_pullback_pct / 100.0
                if direction == 'long':
                    limit_price = round(entry_price * (1 - pullback_pct), 2)
                else:
                    limit_price = round(entry_price * (1 + pullback_pct), 2)

                logger.info(
                    f"EntryTiming WAIT: {symbol} — placing limit order @ "
                    f"${limit_price:.2f} (pullback {timing.expected_pullback_pct:.2f}% "
                    f"from ${entry_price:.2f}), timeout {ENTRY_TIMING_LIMIT_TIMEOUT_MINUTES}min"
                )
                await self._place_entry_timing_limit(setup, limit_price)
                return

            # timing.entry_action == 'enter_now' -> fall through to market order
            logger.info(
                f"EntryTiming ENTER_NOW: {symbol} — proceeding with market order "
                f"(confidence={timing.confidence:.2f})"
            )

        # --- Options trading path ---
        if self._options_enabled and self._strategy_selector:
            options_result = await self._execute_options_trade(setup)
            if options_result is not None:
                return  # Options path handled the trade (success or rejection)
            # If options_result is None, fall through to stock execution
            logger.info(f"Options path returned None for {symbol} — falling back to stock execution")

        logger.info(f"Executing: {direction.upper()} {shares} {symbol} @ ${entry_price:.2f}")

        try:
            # Determine order side
            if direction == "long":
                entry_side = OrderSide.BUY
                exit_side = OrderSide.SELL
            else:
                entry_side = OrderSide.SELL_SHORT
                exit_side = OrderSide.BUY_TO_COVER

            # Place bracket order (entry + stop-loss + take-profit as OCO)
            try:
                entry_order, tp_order, sl_order = self.broker.place_bracket_order(
                    symbol=symbol,
                    side=entry_side,
                    quantity=shares,
                    entry_price=None,
                    take_profit_price=target_price,
                    stop_loss_price=stop_price,
                    entry_type=OrderType.MARKET
                )
            except Exception:
                # Fallback to individual orders if bracket order fails for any reason
                logger.warning(f"Bracket order not supported, falling back to individual orders for {symbol}")
                entry_order = self.broker.place_order(
                    symbol=symbol, side=entry_side,
                    quantity=shares, order_type=OrderType.MARKET
                )
                tp_order = None
                sl_order = None
                if entry_order:
                    sl_order = self.broker.place_order(
                        symbol=symbol, side=exit_side,
                        quantity=shares, order_type=OrderType.STOP,
                        stop_price=stop_price
                    )
                    tp_order = self.broker.place_order(
                        symbol=symbol, side=exit_side,
                        quantity=shares, order_type=OrderType.LIMIT,
                        price=target_price
                    )

            if entry_order is None:
                await self._order_rejected(setup, "Entry order failed")
                return

            self.orders_placed += 1
            self.active_orders[entry_order.order_id] = {
                "order": entry_order,
                "type": "entry",
                "setup": setup
            }

            await self.publish(EventType.ORDER_PLACED, {
                "order_id": entry_order.order_id,
                "symbol": symbol,
                "side": entry_side.value,
                "shares": shares,
                "order_type": "market",
                "timestamp": datetime.now().isoformat()
            })

            stop_order = sl_order
            if stop_order:
                self.active_orders[stop_order.order_id] = {
                    "order": stop_order,
                    "type": "stop_loss",
                    "setup": setup
                }
                logger.info(f"Stop-loss placed: {symbol} @ ${stop_price:.2f}")

            target_order = tp_order
            if target_order:
                self.active_orders[target_order.order_id] = {
                    "order": target_order,
                    "type": "take_profit",
                    "setup": setup
                }
                logger.info(f"Take-profit placed: {symbol} @ ${target_price:.2f}")

            # Simulate immediate fill for market order (paper trading)
            await self._handle_fill(entry_order, setup)

            # Update metrics
            self._update_metrics()

        except Exception as e:
            logger.error(f"Execution error for {symbol}: {e}")
            await self._order_rejected(setup, str(e))

    async def _handle_fill(self, order, setup: Dict):
        """Handle order fill"""
        self.orders_filled += 1

        fill_time = datetime.now()

        await self.publish(EventType.ORDER_FILLED, {
            "order_id": order.order_id,
            "symbol": setup["symbol"],
            "direction": setup["direction"],
            "shares": setup["shares"],
            "fill_price": setup["entry_price"],  # Would be actual fill price
            "timestamp": fill_time.isoformat()
        })

        # Save trade to database (including filter evaluation metadata)
        trade_data = {
            "symbol": setup["symbol"],
            "direction": setup["direction"],
            "entry_price": setup["entry_price"],
            "shares": setup["shares"],
            "stop_loss": setup.get("stop_price"),
            "take_profit": setup.get("target_price"),
            "rrs_at_entry": setup.get("rrs"),
            "entry_time": fill_time,
            # Filter evaluation metadata from signal/analyzer pipeline
            "vix_regime": setup.get("vix_regime"),
            "vix_value": setup.get("vix_value"),
            "market_regime": setup.get("market_regime"),
            "sector_name": setup.get("sector"),  # signal uses 'sector' key
            "sector_rs": setup.get("sector_rs"),
            "spy_trend": setup.get("spy_trend"),
            "ml_confidence": setup.get("ml_probability"),  # analyzer uses 'ml_probability'
            "signal_strategy": setup.get("strategy"),  # signal uses 'strategy' key
            "news_sentiment": setup.get("news_sentiment_score"),  # signal uses 'news_sentiment_score'
            "news_warning": setup.get("news_warning"),
            "regime_rrs_threshold": setup.get("regime_rrs_threshold"),
            "regime_stop_multiplier": setup.get("regime_stop_multiplier"),
            "regime_target_multiplier": setup.get("regime_target_multiplier"),
            "vix_position_size_mult": setup.get("vix_position_size_multiplier"),  # signal uses longer name
            "sector_boost": setup.get("sector_boost"),
            "first_hour_filtered": setup.get("first_hour_filtered", False),
        }
        saved_trade = self.trades_repo.save_trade(trade_data)
        if saved_trade:
            # Log metadata fields that were captured for forward-test analysis
            meta_fields = [k for k in (
                'vix_regime', 'market_regime', 'sector_name', 'ml_confidence',
                'signal_strategy', 'spy_trend'
            ) if trade_data.get(k) is not None]
            meta_summary = ", ".join(f"{k}={trade_data[k]}" for k in meta_fields)
            logger.info(
                f"Trade saved to database: {setup['symbol']} (ID: {saved_trade.get('id')})"
                f"{' | metadata: ' + meta_summary if meta_summary else ''}"
            )

        # Save position to database
        position_data = {
            "symbol": setup["symbol"],
            "direction": setup["direction"],
            "entry_price": setup["entry_price"],
            "shares": setup["shares"],
            "stop_price": setup.get("stop_price"),
            "target_price": setup.get("target_price"),
            "rrs": setup.get("rrs"),
            "entry_time": fill_time
        }
        saved_position = self.trades_repo.save_position(position_data)
        if saved_position:
            logger.info(f"Position saved to database: {setup['symbol']}")

        # Track position risk data for trailing stop activation
        # Use exit prediction parameters if available, otherwise defaults
        if self.trailing_stop_enabled and setup.get("atr"):
            exit_pred = setup.get('exit_prediction', {})
            self._position_risk[setup["symbol"]] = {
                'entry_price': setup["entry_price"],
                'stop_price': setup["stop_price"],
                'target_price': setup["target_price"],
                'atr': setup["atr"],
                'direction': setup["direction"],
                'shares': setup["shares"],
                'exit_strategy': setup.get('exit_strategy', 'default'),
                'trail_activation_r': setup.get(
                    'trailing_stop_activation_r',
                    self.trailing_stop_activation_r
                ),
                'trail_atr_multiplier': setup.get(
                    'trailing_stop_atr_multiplier',
                    self.trailing_stop_atr_multiplier
                ),
            }

        # Publish position opened
        await self.publish(EventType.POSITION_OPENED, {
            "symbol": setup["symbol"],
            "direction": setup["direction"],
            "shares": setup["shares"],
            "entry_price": setup["entry_price"],
            "stop_price": setup["stop_price"],
            "target_price": setup["target_price"],
            "rrs": setup.get("rrs"),
            "timestamp": fill_time.isoformat()
        })

    async def _place_entry_timing_limit(self, setup: Dict, limit_price: float):
        """
        Place a limit order for entry timing 'wait_for_pullback' action.

        The limit order will be automatically cancelled after
        ENTRY_TIMING_LIMIT_TIMEOUT_MINUTES if not filled.

        Args:
            setup: Trade setup dictionary.
            limit_price: Limit price for the entry order.
        """
        symbol = setup.get("symbol")
        direction = setup.get("direction")
        shares = setup.get("shares")

        try:
            if direction == "long":
                entry_side = OrderSide.BUY
            else:
                entry_side = OrderSide.SELL_SHORT

            limit_order = self.broker.place_order(
                symbol=symbol,
                side=entry_side,
                quantity=shares,
                order_type=OrderType.LIMIT,
                price=limit_price
            )

            if limit_order is None:
                await self._order_rejected(setup, "Entry timing limit order failed")
                return

            self.orders_placed += 1

            # Track the limit order for timeout cancellation
            self._entry_timing_limits[symbol] = {
                'order_id': limit_order.order_id,
                'setup': setup,
                'limit_price': limit_price,
                'placed_at': datetime.now(),
                'timeout_at': datetime.now() + timedelta(
                    minutes=ENTRY_TIMING_LIMIT_TIMEOUT_MINUTES
                ),
            }

            self.active_orders[limit_order.order_id] = {
                "order": limit_order,
                "type": "entry_timing_limit",
                "setup": setup
            }

            await self.publish(EventType.ORDER_PLACED, {
                "order_id": limit_order.order_id,
                "symbol": symbol,
                "side": entry_side.value,
                "shares": shares,
                "order_type": "limit",
                "limit_price": limit_price,
                "entry_timing_action": "wait_for_pullback",
                "timeout_minutes": ENTRY_TIMING_LIMIT_TIMEOUT_MINUTES,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(
                f"Entry timing limit order placed: {symbol} {direction} "
                f"{shares} shares @ ${limit_price:.2f} "
                f"(timeout: {ENTRY_TIMING_LIMIT_TIMEOUT_MINUTES} min)"
            )

            self._update_metrics()

        except Exception as e:
            logger.error(f"Failed to place entry timing limit order for {symbol}: {e}")
            await self._order_rejected(setup, f"Entry timing limit order error")

    async def check_entry_timing_timeouts(self):
        """
        Check for entry timing limit orders that have timed out and cancel them.

        Call this periodically (e.g., every price update cycle or every minute).
        """
        now = datetime.now()
        timed_out = []

        for symbol, limit_data in self._entry_timing_limits.items():
            if now >= limit_data['timeout_at']:
                timed_out.append(symbol)

        for symbol in timed_out:
            limit_data = self._entry_timing_limits.pop(symbol)
            order_id = limit_data['order_id']

            try:
                self.broker.cancel_order(order_id)
                self.active_orders.pop(order_id, None)

                elapsed = (now - limit_data['placed_at']).total_seconds() / 60
                logger.info(
                    f"Entry timing limit order CANCELLED (timeout): {symbol} "
                    f"@ ${limit_data['limit_price']:.2f} after {elapsed:.0f} min"
                )

                await self.publish(EventType.ORDER_CANCELLED, {
                    "order_id": order_id,
                    "symbol": symbol,
                    "reason": "entry_timing_timeout",
                    "elapsed_minutes": elapsed,
                    "timestamp": now.isoformat()
                })

            except Exception as e:
                logger.error(
                    f"Error cancelling timed-out entry timing order for {symbol}: {e}"
                )

    async def _execute_options_trade(self, setup: Dict) -> Optional[bool]:
        """
        Execute a trade through the options pipeline.

        Returns:
            True if options order placed successfully,
            False if options rejected (don't fall back to stocks),
            None if options not applicable (fall through to stocks).
        """
        symbol = setup.get("symbol", "")
        direction = setup.get("direction", "long")

        try:
            # 1. Select strategy based on signal + IV regime
            strategy = self._strategy_selector.select_strategy(setup, self._get_account_size())
            if strategy is None:
                logger.warning(f"Options: no suitable strategy for {symbol} — skipping options")
                return None  # Fall through to stock execution

            # 2. Size the position
            account_size = self._get_account_size()
            from config.settings import get_settings
            max_risk = get_settings().trading.max_risk_per_trade

            size_result = self._options_position_sizer.calculate(
                strategy, account_size, max_risk
            )

            if size_result.contracts <= 0:
                logger.warning(
                    f"Options: cannot size {strategy.name} for {symbol} — "
                    f"{size_result.reason}"
                )
                await self._order_rejected(setup, f"Options sizing failed: {size_result.reason}")
                return False

            # 3. Portfolio risk check
            existing_positions = self._options_executor.get_all_positions()
            risk_check = self._options_risk_manager.validate_new_trade(
                strategy, size_result, existing_positions, account_size
            )

            if not risk_check:
                logger.warning(
                    f"Options: risk check failed for {symbol} — {risk_check.reason}"
                )
                await self._order_rejected(setup, f"Options risk check: {risk_check.reason}")
                return False

            if risk_check.warnings:
                for w in risk_check.warnings:
                    logger.warning(f"Options risk warning ({symbol}): {w}")

            # 4. Execute the strategy
            order_result = self._options_executor.execute_strategy(strategy, size_result)

            if order_result is None:
                logger.error(f"Options: execution failed for {symbol}")
                await self._order_rejected(setup, "Options execution failed")
                return False

            # 5. Publish order placed event
            self.orders_placed += 1
            await self.publish(EventType.ORDER_PLACED, {
                "order_id": order_result.get("order_id", ""),
                "symbol": symbol,
                "side": direction,
                "order_type": "options",
                "options_strategy": strategy.name,
                "contracts": size_result.contracts,
                "max_risk": size_result.max_risk,
                "net_premium": strategy.net_premium,
                "timestamp": datetime.now().isoformat(),
            })

            logger.info(
                f"OPTIONS TRADE EXECUTED: {strategy.name} {symbol} "
                f"{size_result.contracts}x, max_risk=${size_result.max_risk:.2f}, "
                f"net_premium=${strategy.net_premium:.2f}"
            )

            # Persist options trade to database
            fill_time = datetime.now()
            trade_data = {
                "symbol": symbol,
                "direction": direction,
                "entry_price": abs(strategy.net_premium) / max(size_result.contracts, 1),
                "shares": size_result.contracts,  # contracts mapped to shares column
                "stop_loss": strategy.max_loss / max(size_result.contracts, 1) if strategy.max_loss else None,
                "take_profit": strategy.max_profit / max(size_result.contracts, 1) if strategy.max_profit else None,
                "entry_time": fill_time,
                "signal_strategy": f"options_{strategy.name}",
                "vix_regime": setup.get("vix_regime"),
                "vix_value": setup.get("vix_value"),
                "market_regime": setup.get("market_regime"),
                "sector_name": setup.get("sector"),
                "ml_confidence": setup.get("ml_probability"),
            }
            saved_trade = self.trades_repo.save_trade(trade_data)
            if saved_trade:
                logger.info(
                    f"Options trade saved to database: {symbol} "
                    f"(ID: {saved_trade.get('id')}, strategy: {strategy.name})"
                )

            # Persist options position to database
            position_data = {
                "symbol": symbol,
                "direction": direction,
                "entry_price": abs(strategy.net_premium) / max(size_result.contracts, 1),
                "shares": size_result.contracts,
                "stop_price": strategy.max_loss / max(size_result.contracts, 1) if strategy.max_loss else None,
                "target_price": strategy.max_profit / max(size_result.contracts, 1) if strategy.max_profit else None,
                "entry_time": fill_time,
            }
            saved_position = self.trades_repo.save_position(position_data)
            if saved_position:
                logger.info(f"Options position saved to database: {symbol}")

            self._update_metrics()
            return True

        except Exception as e:
            logger.error(f"Options execution error for {symbol}: {e}")
            return None  # Fall through to stock execution on unexpected error

    def _get_account_size(self) -> float:
        """Get current account size from settings."""
        try:
            from config.settings import get_settings
            return get_settings().trading.account_size
        except Exception:
            return 25000.0

    async def check_options_exits(self):
        """
        Check all open options positions for exit triggers.
        Call this periodically (e.g., every price update cycle).
        """
        if not self._options_enabled or not self._options_exit_manager:
            return

        try:
            positions = self._options_executor.get_all_positions()
            if not positions:
                return

            exit_signals = self._options_exit_manager.check_exits(positions)

            for signal in exit_signals:
                if signal.action == "close":
                    logger.info(f"Options exit triggered: {signal.symbol} — {signal.reason}")

                    # Capture position data before closing (for DB persistence)
                    opt_position = self._options_executor.get_position(signal.symbol)
                    entry_premium = opt_position.get("entry_premium", 0) if opt_position else 0

                    self._options_executor.close_position(signal.symbol)

                    # Map options exit reason to ExitReason enum values
                    reason_lower = signal.reason.lower()
                    if "profit" in reason_lower or "target" in reason_lower:
                        db_exit_reason = "take_profit"
                    elif "stop" in reason_lower and "time" not in reason_lower:
                        db_exit_reason = "stop_loss"
                    elif "time" in reason_lower or "dte" in reason_lower:
                        db_exit_reason = "end_of_day"
                    else:
                        db_exit_reason = "manual"

                    # Close trade in database
                    exit_time = datetime.now()
                    closed_trade = self.trades_repo.close_trade_by_symbol(
                        symbol=signal.symbol,
                        exit_price=entry_premium,
                        exit_reason=db_exit_reason,
                        exit_time=exit_time,
                    )
                    if closed_trade:
                        logger.info(
                            f"Options trade closed in database: {signal.symbol} "
                            f"PnL=${closed_trade.get('pnl', 0):.2f} reason={db_exit_reason}"
                        )

                    # Remove position from database
                    self.trades_repo.close_position(signal.symbol)
                    logger.info(f"Options position removed from database: {signal.symbol}")

                    await self.publish(EventType.POSITION_CLOSED, {
                        "symbol": signal.symbol,
                        "exit_reason": f"options_{signal.reason}",
                        "exit_price": entry_premium,
                        "timestamp": datetime.now().isoformat(),
                    })
                elif signal.action == "roll":
                    logger.info(f"Options roll recommended: {signal.symbol} — {signal.reason}")
                    # Roll is advisory — log but don't auto-execute

        except Exception as e:
            logger.error(f"Options exit check failed: {e}")

    async def _order_rejected(self, setup: Dict, reason: str):
        """Handle order rejection"""
        self.orders_rejected += 1
        self._update_metrics()

        await self.publish(EventType.ORDER_REJECTED, {
            "symbol": setup["symbol"],
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })

    async def cancel_orders_for_symbol(self, symbol: str):
        """Cancel all pending orders for a symbol"""
        orders_to_cancel = []

        for order_id, order_data in self.active_orders.items():
            if order_data["setup"].get("symbol") == symbol:
                orders_to_cancel.append(order_id)

        for order_id in orders_to_cancel:
            try:
                self.broker.cancel_order(order_id)
                del self.active_orders[order_id]

                await self.publish(EventType.ORDER_CANCELLED, {
                    "order_id": order_id,
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error canceling order {order_id}: {e}")

    async def execute_order(self, order_request: Dict):
        """Execute a manual order request"""
        if "setup" in order_request:
            setup = order_request["setup"]
            await self.execute_trade(setup)

    def confirm_pending_setup(self, symbol: str) -> bool:
        """Confirm and execute a pending setup"""
        if symbol in self.pending_setups:
            setup = self.pending_setups.pop(symbol)
            # Use event loop to execute
            import asyncio
            asyncio.create_task(self.execute_trade(setup))
            return True
        return False

    def reject_pending_setup(self, symbol: str):
        """Reject a pending setup"""
        if symbol in self.pending_setups:
            del self.pending_setups[symbol]
            logger.info(f"Rejected pending setup: {symbol}")

    def _update_metrics(self):
        """Update executor metrics"""
        self.metrics.custom_metrics["orders_placed"] = self.orders_placed
        self.metrics.custom_metrics["orders_filled"] = self.orders_filled
        self.metrics.custom_metrics["orders_rejected"] = self.orders_rejected

        if self.orders_placed > 0:
            fill_rate = (self.orders_filled / self.orders_placed) * 100
            self.metrics.custom_metrics["fill_rate"] = round(fill_rate, 1)

    def check_trailing_stop_activation(self, symbol: str, current_price: float):
        """
        Check if a position has reached +1R profit and activate trailing stop.

        Args:
            symbol: Stock ticker symbol
            current_price: Current market price
        """
        if not self.trailing_stop_enabled or not TRAILING_STOP_AVAILABLE:
            return

        # Skip if trailing stop already active for this symbol
        if symbol in self._trailing_stops:
            return

        risk_data = self._position_risk.get(symbol)
        if not risk_data:
            return

        entry_price = risk_data['entry_price']
        stop_price = risk_data['stop_price']
        atr = risk_data['atr']
        direction = risk_data['direction']
        initial_risk = abs(entry_price - stop_price)  # 1R

        # Use per-position activation R if set by ExitPredictor, else global default
        activation_r = risk_data.get('trail_activation_r', self.trailing_stop_activation_r)
        trail_multiplier = risk_data.get('trail_atr_multiplier', self.trailing_stop_atr_multiplier)

        # Skip trailing stop for quick_scalp strategy
        if risk_data.get('exit_strategy') == 'quick_scalp':
            return

        # Check if position has reached activation threshold
        if direction == 'long':
            unrealized = current_price - entry_price
            activation_threshold = entry_price + (initial_risk * activation_r)
            if current_price < activation_threshold:
                return
            exit_side = OrderSide.SELL
        else:
            unrealized = entry_price - current_price
            activation_threshold = entry_price - (initial_risk * activation_r)
            if current_price > activation_threshold:
                return
            exit_side = OrderSide.BUY_TO_COVER

        # Position has reached activation threshold -- activate trailing stop
        trail_amount = atr * trail_multiplier
        shares = risk_data.get('shares', 0)

        if shares <= 0 or trail_amount <= 0:
            return

        try:
            trailing_stop = TrailingStopOrder(
                broker=self.broker,
                symbol=symbol,
                side=exit_side,
                quantity=shares,
                trail_amount=trail_amount,
            )
            trailing_stop.create()
            self._trailing_stops[symbol] = trailing_stop

            logger.info(
                f"TRAILING STOP ACTIVATED: {symbol} {direction} "
                f"trail=${trail_amount:.2f} (0.75x ATR=${atr:.2f}) "
                f"after reaching +1R (${unrealized:.2f} profit)"
            )
        except Exception as e:
            logger.error(f"Failed to create trailing stop for {symbol}: {e}")

    def update_trailing_stops(self, price_updates: Optional[Dict[str, float]] = None):
        """
        Update all active trailing stops with current prices.
        Call this periodically (e.g., every price update cycle).

        Args:
            price_updates: Dict of symbol -> current_price (optional)
        """
        if not self.trailing_stop_enabled or not TRAILING_STOP_AVAILABLE:
            return

        for symbol in list(self._trailing_stops.keys()):
            trail = self._trailing_stops[symbol]
            try:
                current_price = price_updates.get(symbol) if price_updates else None
                _, triggered = trail.update_trail(current_price)
                if triggered:
                    logger.info(f"Trailing stop triggered for {symbol}")
                    del self._trailing_stops[symbol]
                    # Clean up position risk tracking
                    self._position_risk.pop(symbol, None)
            except Exception as e:
                logger.error(f"Error updating trailing stop for {symbol}: {e}")

    def _record_dynamic_sizer_outcome(
        self, symbol: str, closed_trade: Dict, event_data: Dict
    ):
        """Record trade outcome in the dynamic sizer for Kelly/win-rate tracking."""
        if not DYNAMIC_SIZER_AVAILABLE:
            return
        try:
            sizer = get_dynamic_sizer()
            pnl = closed_trade.get("pnl", 0)
            risk_amount = closed_trade.get("risk_amount")
            entry_price = closed_trade.get("entry_price", 0)
            stop_price = closed_trade.get("stop_loss") or closed_trade.get("stop_price", 0)

            # Calculate R-multiple
            if risk_amount and risk_amount > 0:
                pnl_r = pnl / risk_amount
            elif entry_price and stop_price:
                initial_risk = abs(entry_price - stop_price)
                pnl_r = pnl / initial_risk if initial_risk > 0 else 0.0
            else:
                pnl_r = 0.0

            direction = closed_trade.get("direction", event_data.get("direction", ""))
            sizer.record_trade_result(
                pnl_r=pnl_r,
                win=pnl > 0,
                symbol=symbol,
                direction=direction,
            )
            logger.debug(
                f"Dynamic sizer recorded: {symbol} pnl_r={pnl_r:.2f} win={pnl > 0}"
            )
        except Exception as e:
            logger.debug(f"Failed to record dynamic sizer outcome: {e}")

    def set_auto_execute(self, enabled: bool):
        """Enable or disable auto-execution"""
        self.auto_execute = enabled
        logger.info(f"Auto-execute {'enabled' if enabled else 'disabled'}")
