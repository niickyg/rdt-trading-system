"""
Fully Automated RDT Trading Bot
WARNING: Use with extreme caution. Test extensively in paper trading first.

This bot:
1. Scans for RRS signals
2. Checks daily chart strength
3. Automatically enters trades
4. Manages position sizing
5. Sets stops and targets
6. Monitors and exits positions

Uses the unified broker interface to support:
- Paper trading (simulation)
- Schwab live trading
- Interactive Brokers

Supports multiple trading accounts:
- Specify account_id to trade on a specific account
- Tracks trades by account for performance analysis

Example:
    # Paper trading mode (safe)
    config = {
        'account_size': 25000,
        'paper_trading': True,
        'auto_trade': False
    }
    bot = TradingBot(config)
    bot.run()

    # Multi-account trading
    config = {
        'account_id': 'acc_123456789',  # Specific account
        'user_id': 'user123',
        'paper_trading': False,
        'auto_trade': True
    }
    bot = TradingBot(config)
    bot.run()
"""

import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from brokers import (
    get_broker, get_broker_from_config,
    BrokerInterface, OrderSide, OrderType, OrderStatus,
    BrokerError
)

# Account management support
try:
    from accounts import AccountManager
    ACCOUNT_MANAGER_AVAILABLE = True
except ImportError:
    ACCOUNT_MANAGER_AVAILABLE = False
    logger.debug("AccountManager not available - using direct broker")

from scanner.realtime_scanner import RealTimeScanner
from shared.indicators.rrs import RRSCalculator
from utils.timezone import (
    get_eastern_time,
    format_timestamp,
    is_market_open,
    is_trading_day,
    is_premarket,
    is_afterhours,
    is_extended_hours,
    get_extended_hours_session,
)

# Order monitoring imports
try:
    from trading.order_monitor import OrderMonitor, get_order_monitor, OrderState
    ORDER_MONITOR_AVAILABLE = True
except ImportError:
    ORDER_MONITOR_AVAILABLE = False
    logger.warning("OrderMonitor not available")

try:
    from trading.execution_tracker import ExecutionTracker, get_execution_tracker
    EXECUTION_TRACKER_AVAILABLE = True
except ImportError:
    EXECUTION_TRACKER_AVAILABLE = False
    logger.warning("ExecutionTracker not available")

# Prometheus metrics support
try:
    from monitoring.metrics import (
        record_execution_metrics,
        rdt_order_fill_time_seconds,
        rdt_order_slippage_pct,
        rdt_order_fill_rate,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logger.debug("Prometheus execution metrics not available")


class TradingBot:
    """
    Fully automated trading bot with broker integration.

    Supports multiple brokers through unified interface:
    - Paper trading for safe testing
    - Schwab for live trading
    - Interactive Brokers for live trading

    Safety features:
    - Paper trading mode by default
    - Auto-trade disabled by default
    - Daily loss limits
    - Position size limits
    - Comprehensive logging
    """

    def __init__(self, config: Dict, account_id: Optional[str] = None):
        """
        Initialize trading bot.

        Args:
            config: Configuration dictionary with:
                - paper_trading: Use paper broker (default: True)
                - auto_trade: Enable automatic order execution (default: False)
                - broker_type: "paper", "schwab", or "ibkr" (default: "paper")
                - account_size: Account size for risk calculations
                - max_risk_per_trade: Max risk per trade as decimal (default: 0.01)
                - max_daily_loss: Max daily loss as decimal (default: 0.03)
                - max_position_size: Max position size as decimal (default: 0.10)
                - extended_hours_enabled: Enable extended hours trading (default: False)
                - extended_hours_position_size_pct: Position size multiplier for extended hours (default: 0.5)
                - premarket_start: Earliest pre-market trading time as "HH:MM" (default: "07:00")
                - afterhours_end: Latest after-hours trading time as "HH:MM" (default: "18:00")
                - user_id: User ID for account manager (optional)
                - account_id: Trading account ID (optional, uses default if not specified)
                - For Schwab: app_key, app_secret, callback_url
                - For IBKR: host, port, client_id
            account_id: Optional account ID to trade on (overrides config)
        """
        self.config = config
        self.paper_trading = config.get('paper_trading', True)
        self.auto_trade = config.get('auto_trade', False)

        # Account management
        self.account_id = account_id or config.get('account_id')
        self.user_id = config.get('user_id')
        self._account_manager: Optional[AccountManager] = None

        # Extended hours configuration
        self.extended_hours_enabled = config.get('extended_hours_enabled', False)
        self.extended_hours_position_size_pct = config.get('extended_hours_position_size_pct', 0.5)
        self.premarket_start = config.get('premarket_start', '07:00')  # Default 7:00 AM ET
        self.afterhours_end = config.get('afterhours_end', '18:00')    # Default 6:00 PM ET

        # Parse time strings
        self._premarket_start_hour, self._premarket_start_minute = self._parse_time(self.premarket_start)
        self._afterhours_end_hour, self._afterhours_end_minute = self._parse_time(self.afterhours_end)

        # Extended hours trade tracking
        self.extended_hours_trades_today: List[dict] = []

        # Bracket order and trailing stop configuration
        self.use_bracket_orders = config.get('use_bracket_orders', False)
        self.use_trailing_stops = config.get('use_trailing_stops', False)
        self.trailing_stop_activation_pct = config.get('trailing_stop_activation_pct', 2.0)  # Activate after 2% profit
        self.trailing_stop_trail_pct = config.get('trailing_stop_trail_pct', 1.5)  # Trail by 1.5%

        # Advanced order manager
        self._advanced_order_manager = None

        # Safety warnings
        if not self.paper_trading:
            logger.warning("LIVE TRADING MODE - REAL MONEY AT RISK")
        else:
            logger.info("Paper trading mode - no real money")

        if not self.auto_trade:
            logger.info("Scan-only mode (auto_trade=False)")
        else:
            logger.warning("FULL AUTOMATION ENABLED - orders will execute automatically")

        if self.extended_hours_enabled:
            logger.info(
                f"Extended hours trading ENABLED - "
                f"Pre-market from {self.premarket_start} ET, "
                f"After-hours until {self.afterhours_end} ET, "
                f"Position size: {self.extended_hours_position_size_pct * 100:.0f}% of regular"
            )
        else:
            logger.info("Extended hours trading disabled")

        # Log bracket/trailing stop configuration
        if self.use_bracket_orders:
            logger.info("Bracket orders ENABLED for entries")
        if self.use_trailing_stops:
            logger.info(
                f"Trailing stops ENABLED - Activation: {self.trailing_stop_activation_pct}% profit, "
                f"Trail: {self.trailing_stop_trail_pct}%"
            )

        # Initialize broker
        self.broker: BrokerInterface = self._init_broker()

        # Initialize scanner and RRS calculator
        self.scanner = RealTimeScanner(config)
        self.rrs_calc = RRSCalculator()

        # Trading state
        self.positions: Dict[str, dict] = {}  # Tracked positions with stops/targets
        self.daily_pnl = 0.0
        self.account_size = config.get('account_size', 25000)
        self.max_daily_loss = config.get('max_daily_loss', 0.03)
        self.trades_today: List[dict] = []

        # Performance tracking
        self.session_start_equity = 0.0
        self.total_trades = 0
        self.winning_trades = 0

        # Order monitoring and execution tracking
        self.order_monitor: Optional[OrderMonitor] = None
        self.execution_tracker: Optional[ExecutionTracker] = None
        self._init_order_monitoring()

        # Configuration for execution quality alerts
        self.high_slippage_threshold_pct = config.get('high_slippage_threshold_pct', 0.5)
        self.stuck_order_timeout_seconds = config.get('stuck_order_timeout_seconds', 60)
        self.fill_confirmation_timeout_seconds = config.get('fill_confirmation_timeout_seconds', 30)

        logger.info(f"TradingBot initialized - Account: ${self.account_size:,.2f}")

    def _parse_time(self, time_str: str) -> tuple:
        """Parse a time string like '07:00' into (hour, minute) tuple."""
        try:
            parts = time_str.split(':')
            return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            logger.warning(f"Invalid time format: {time_str}, using 00:00")
            return 0, 0

    def _is_within_extended_hours_window(self) -> bool:
        """
        Check if current time is within the configured extended hours trading window.

        Returns:
            True if within the allowed extended hours window, False otherwise.
        """
        if not self.extended_hours_enabled:
            return False

        now = get_eastern_time()
        current_hour = now.hour
        current_minute = now.minute
        current_time_minutes = current_hour * 60 + current_minute

        session = get_extended_hours_session()

        if session == 'premarket':
            # Check if after configured premarket start time
            premarket_start_minutes = self._premarket_start_hour * 60 + self._premarket_start_minute
            return current_time_minutes >= premarket_start_minutes

        elif session == 'afterhours':
            # Check if before configured afterhours end time
            afterhours_end_minutes = self._afterhours_end_hour * 60 + self._afterhours_end_minute
            return current_time_minutes <= afterhours_end_minutes

        return False

    def _get_current_session(self) -> str:
        """
        Get the current trading session.

        Returns:
            One of 'premarket', 'regular', 'afterhours', or 'closed'
        """
        return get_extended_hours_session()

    def _should_trade_now(self) -> bool:
        """
        Check if trading should occur based on current time and configuration.

        Returns:
            True if trading is allowed, False otherwise.
        """
        session = self._get_current_session()

        if session == 'regular':
            return True

        if session == 'closed':
            return False

        # Extended hours - check configuration
        if not self.extended_hours_enabled:
            return False

        return self._is_within_extended_hours_window()

    def _log_extended_hours_activity(self, activity_type: str, details: Dict):
        """
        Log extended hours trading activity separately.

        Args:
            activity_type: Type of activity (e.g., 'signal', 'order', 'fill')
            details: Dictionary with activity details
        """
        session = self._get_current_session()
        timestamp = format_timestamp()

        log_entry = {
            'timestamp': timestamp,
            'session': session,
            'activity_type': activity_type,
            **details
        }

        # Log to standard logger with EXTENDED_HOURS prefix
        logger.info(f"[EXTENDED_HOURS] {activity_type.upper()}: {details}")

        # Track extended hours trades separately
        if activity_type == 'order':
            self.extended_hours_trades_today.append(log_entry)

    def _init_broker(self) -> BrokerInterface:
        """Initialize the appropriate broker based on configuration."""
        # Try to use account manager if available and user_id is set
        if ACCOUNT_MANAGER_AVAILABLE and self.user_id:
            try:
                self._account_manager = AccountManager(user_id=self.user_id)

                # Get broker from account manager
                broker = self._account_manager.get_broker(
                    account_id=self.account_id,
                    connect=False  # We'll connect later
                )

                # If we got a specific account, log it
                if self.account_id:
                    account = self._account_manager.get_account(self.account_id)
                    logger.info(f"Using account: {account.name} ({account.id})")
                else:
                    default_account = self._account_manager.get_default_account()
                    if default_account:
                        self.account_id = default_account.id
                        logger.info(f"Using default account: {default_account.name} ({default_account.id})")

                return broker

            except Exception as e:
                logger.warning(f"Account manager failed: {e}. Falling back to direct broker.")

        # Fallback to direct broker initialization
        broker_type = self.config.get('broker_type', 'paper')

        # Force paper trading if paper_trading flag is set
        if self.paper_trading:
            broker_type = 'paper'

        try:
            if broker_type == 'paper':
                broker = get_broker(
                    'paper',
                    initial_balance=self.config.get('account_size', 25000.0),
                    slippage_pct=self.config.get('slippage_pct', 0.001),
                    realistic_fills=True
                )
            elif broker_type == 'schwab':
                broker = get_broker(
                    'schwab',
                    app_key=self.config.get('schwab_app_key', self.config.get('app_key', '')),
                    app_secret=self.config.get('schwab_app_secret', self.config.get('app_secret', '')),
                    callback_url=self.config.get('schwab_callback_url', 'https://localhost:8080'),
                    account_number=self.config.get('account_number'),
                    token_path=self.config.get('token_path')
                )
            elif broker_type == 'ibkr':
                broker = get_broker(
                    'ibkr',
                    host=self.config.get('ibkr_host', '127.0.0.1'),
                    port=self.config.get('ibkr_port', 7497),
                    client_id=self.config.get('ibkr_client_id', 1),
                    paper_trading=self.paper_trading
                )
            else:
                logger.warning(f"Unknown broker type: {broker_type}, using paper")
                broker = get_broker('paper')

            logger.info(f"Broker initialized: {broker_type}")
            return broker

        except Exception as e:
            logger.error(f"Failed to initialize broker: {e}")
            logger.info("Falling back to paper trading")
            return get_broker('paper')

    def _init_order_monitoring(self) -> None:
        """Initialize order monitoring and execution tracking."""
        if ORDER_MONITOR_AVAILABLE:
            self.order_monitor = OrderMonitor(
                stuck_order_threshold_seconds=self.config.get('stuck_order_timeout_seconds', 60),
                high_slippage_threshold_pct=self.config.get('high_slippage_threshold_pct', 0.5),
                on_fill=self._on_order_fill,
                on_complete=self._on_order_complete,
                on_stuck_order=self._on_stuck_order,
                on_high_slippage=self._on_high_slippage,
                on_rejection=self._on_order_rejection
            )
            self.order_monitor.start()
            logger.info("OrderMonitor initialized and started")
        else:
            logger.warning("OrderMonitor not available - order monitoring disabled")

        if EXECUTION_TRACKER_AVAILABLE:
            self.execution_tracker = get_execution_tracker()
            logger.info("ExecutionTracker initialized")
        else:
            logger.warning("ExecutionTracker not available - execution tracking disabled")

    def _on_order_fill(self, order, fill: Dict) -> None:
        """Callback when an order fill occurs."""
        logger.info(
            f"Fill received: {order.symbol} {fill['fill_quantity']} @ ${fill['fill_price']:.2f}"
        )

    def _on_order_complete(self, order) -> None:
        """Callback when an order completes."""
        if order.state == OrderState.FILLED:
            logger.info(
                f"Order completed: {order.order_id} {order.symbol} "
                f"filled {order.filled_quantity} @ ${order.avg_fill_price:.2f} "
                f"(slippage: {order.slippage_pct:.4f}%)"
            )

            # Record execution for tracking
            if self.execution_tracker and order.submitted_at:
                self.execution_tracker.record_execution(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    expected_price=order.expected_price,
                    fill_price=order.avg_fill_price,
                    quantity=order.quantity,
                    filled_quantity=order.filled_quantity,
                    fill_time=order.completed_at or datetime.utcnow(),
                    order_submitted_at=order.submitted_at,
                    status="filled"
                )

    def _on_stuck_order(self, order) -> None:
        """Callback when an order is detected as stuck."""
        logger.warning(
            f"STUCK ORDER ALERT: {order.order_id} {order.symbol} "
            f"{order.side} {order.quantity} has been pending for "
            f"{order.time_since_submission:.1f} seconds"
        )
        # Could trigger additional alerts here (email, SMS, etc.)

    def _on_high_slippage(self, order) -> None:
        """Callback when high slippage is detected."""
        logger.warning(
            f"HIGH SLIPPAGE ALERT: {order.order_id} {order.symbol} "
            f"slippage: {order.slippage_pct:.4f}% (${order.slippage:.4f})"
        )
        # Could trigger additional alerts here

    def _on_order_rejection(self, order) -> None:
        """Callback when an order is rejected."""
        logger.error(
            f"ORDER REJECTED: {order.order_id} {order.symbol} - {order.error_message}"
        )

    def connect_broker(self) -> bool:
        """
        Connect to the broker.

        Returns:
            True if connected successfully.
        """
        try:
            if self.broker.connect():
                logger.info("Broker connected successfully")

                # Get initial account state
                account = self.broker.get_account()
                self.session_start_equity = account.equity
                self.account_size = account.equity

                logger.info(f"Account equity: ${account.equity:,.2f}")
                logger.info(f"Buying power: ${account.buying_power:,.2f}")

                # Sync existing positions
                self._sync_positions()

                return True
            else:
                logger.error("Broker connection failed")
                return False

        except BrokerError as e:
            logger.error(f"Broker connection error: {e}")
            return False

    def _sync_positions(self):
        """Synchronize positions from broker."""
        try:
            broker_positions = self.broker.get_positions()

            for symbol, pos in broker_positions.items():
                if symbol not in self.positions:
                    # Add position to tracking (without stop/target)
                    self.positions[symbol] = {
                        'direction': 'long' if pos.quantity > 0 else 'short',
                        'entry_price': pos.avg_cost,
                        'shares': abs(pos.quantity),
                        'stop_loss': None,
                        'take_profit': None,
                        'entry_time': format_timestamp(),
                        'rrs': 0,
                        'synced': True
                    }
                    logger.info(f"Synced position: {symbol} - {pos.quantity} shares @ ${pos.avg_cost:.2f}")

        except Exception as e:
            logger.warning(f"Failed to sync positions: {e}")

    def calculate_position_size(self, price: float, atr: float, direction: str) -> int:
        """
        Calculate position size based on ATR and risk management.

        Position size is reduced during extended hours due to lower liquidity.

        Args:
            price: Current stock price
            atr: Average True Range
            direction: 'long' or 'short'

        Returns:
            Number of shares to trade
        """
        # Get current buying power from broker
        try:
            account = self.broker.get_account()
            available_funds = account.buying_power
        except Exception:
            available_funds = self.account_size

        # Risk amount (configurable % of account)
        risk_pct = self.config.get('max_risk_per_trade', 0.01)
        risk_amount = self.account_size * risk_pct

        # Stop distance (1.5x ATR by default)
        stop_multiplier = self.config.get('stop_atr_multiplier', 1.5)
        stop_distance = atr * stop_multiplier

        # Position size based on risk
        if stop_distance > 0:
            shares = int(risk_amount / stop_distance)
        else:
            shares = 0

        # Check maximum position size (% of account)
        max_position_pct = self.config.get('max_position_size', 0.10)
        max_position_value = self.account_size * max_position_pct
        max_shares = int(max_position_value / price)

        # Can't exceed available funds
        affordable_shares = int(available_funds / price)

        # Use the smallest of all constraints
        shares = min(shares, max_shares, affordable_shares)

        # Adjust position size for extended hours (lower liquidity)
        session = self._get_current_session()
        if session in ('premarket', 'afterhours') and self.extended_hours_enabled:
            original_shares = shares
            shares = int(shares * self.extended_hours_position_size_pct)
            logger.info(
                f"Extended hours position sizing: {original_shares} -> {shares} shares "
                f"({self.extended_hours_position_size_pct * 100:.0f}% of regular size)"
            )

        # Minimum 1 share if we have any size
        if shares > 0 and shares < 1:
            shares = 1

        return max(shares, 0)

    def enter_trade(self, symbol: str, analysis: Dict, direction: str):
        """
        Enter a trade.

        Args:
            symbol: Stock ticker
            analysis: RRS analysis dict with price, atr, rrs
            direction: 'long' or 'short'
        """
        try:
            price = analysis['price']
            atr = analysis['atr']
            rrs = analysis.get('rrs', 0)

            # Calculate position size
            shares = self.calculate_position_size(price, atr, direction)

            if shares == 0:
                logger.warning(f"Position size = 0 for {symbol}, skipping")
                return

            # Calculate stops and targets
            stop_multiplier = self.config.get('stop_atr_multiplier', 1.5)
            target_multiplier = self.config.get('target_atr_multiplier', 3.0)

            if direction == 'long':
                stop_loss = price - (atr * stop_multiplier)
                take_profit = price + (atr * target_multiplier)
                side = OrderSide.BUY
            else:  # short
                stop_loss = price + (atr * stop_multiplier)
                take_profit = price - (atr * target_multiplier)
                side = OrderSide.SELL_SHORT

            # Position value
            position_value = price * shares

            # Log trade signal
            logger.info(f"""
{'='*60}
TRADE SIGNAL: {direction.upper()} {symbol}
{'='*60}
Price: ${price:.2f}
Shares: {shares}
Position Value: ${position_value:,.2f}
Stop Loss: ${stop_loss:.2f} ({((stop_loss/price - 1) * 100):.2f}%)
Take Profit: ${take_profit:.2f} ({((take_profit/price - 1) * 100):.2f}%)
RRS: {rrs:.2f}
Auto-Trade: {self.auto_trade}
{'='*60}
            """)

            if self.auto_trade:
                # Execute trade via broker with order monitoring
                try:
                    # Determine current session for extended hours handling
                    session = self._get_current_session()
                    is_extended = session in ('premarket', 'afterhours')

                    # Generate unique order ID
                    order_id = f"{symbol}_{int(time.time() * 1000)}"

                    # Extended hours requires limit orders
                    if is_extended and self.extended_hours_enabled:
                        # Use limit order for extended hours
                        order_type = OrderType.LIMIT
                        order_type_str = 'limit'
                        # Set limit price slightly above/below market for quick fill
                        if direction == 'long':
                            limit_price = price * 1.002  # 0.2% above for buy
                        else:
                            limit_price = price * 0.998  # 0.2% below for sell

                        self._log_extended_hours_activity('signal', {
                            'symbol': symbol,
                            'direction': direction,
                            'price': price,
                            'limit_price': limit_price,
                            'shares': shares,
                            'rrs': rrs
                        })
                    else:
                        order_type = OrderType.MARKET
                        order_type_str = 'market'
                        limit_price = None

                    # Track order in monitor before submission
                    if self.order_monitor:
                        self.order_monitor.track_order(
                            order_id=order_id,
                            symbol=symbol,
                            side=side.value,
                            quantity=shares,
                            order_type=order_type_str,
                            expected_price=price
                        )

                    # Place order with broker (with session parameter for extended hours)
                    # Use bracket order if enabled and not in extended hours
                    if self.use_bracket_orders and not is_extended:
                        try:
                            entry_order, tp_order, sl_order = self.broker.place_bracket_order(
                                symbol=symbol,
                                side=side,
                                quantity=shares,
                                entry_price=limit_price,
                                take_profit_price=take_profit,
                                stop_loss_price=stop_loss,
                                entry_type=order_type
                            )
                            order = entry_order
                            logger.info(
                                f"Bracket order placed: Entry={entry_order.order_id}, "
                                f"TP={tp_order.order_id}, SL={sl_order.order_id}"
                            )
                        except NotImplementedError:
                            # Broker doesn't support native brackets, fall back to regular order
                            logger.info("Broker doesn't support native brackets, using regular order")
                            order = self.broker.place_order(
                                symbol=symbol,
                                side=side,
                                quantity=shares,
                                order_type=order_type,
                                price=limit_price,
                                session=session if is_extended else 'regular'
                            )
                    else:
                        order = self.broker.place_order(
                            symbol=symbol,
                            side=side,
                            quantity=shares,
                            order_type=order_type,
                            price=limit_price,
                            session=session if is_extended else 'regular'
                        )

                    # Log extended hours order
                    if is_extended and self.extended_hours_enabled:
                        self._log_extended_hours_activity('order', {
                            'order_id': order.order_id,
                            'symbol': symbol,
                            'side': side.value,
                            'quantity': shares,
                            'order_type': order_type_str,
                            'limit_price': limit_price,
                            'session': session
                        })

                    # Update order monitor with broker order ID
                    if self.order_monitor:
                        self.order_monitor.order_submitted(
                            order_id=order_id,
                            broker_order_id=order.order_id
                        )

                    # Wait for and verify fill confirmation
                    actual_price = self._wait_for_fill_confirmation(
                        order_id=order_id,
                        broker_order=order,
                        symbol=symbol,
                        shares=shares,
                        expected_price=price
                    )

                    if actual_price is None:
                        logger.error(f"Failed to confirm fill for {symbol}")
                        return

                    logger.info(f"Order FILLED: {shares} {symbol} @ ${actual_price:.2f}")

                    # Log execution quality metrics
                    slippage = actual_price - price
                    slippage_pct = (slippage / price) * 100
                    self._log_execution_quality(
                        symbol=symbol,
                        expected_price=price,
                        actual_price=actual_price,
                        slippage=slippage,
                        slippage_pct=slippage_pct
                    )

                    # Alert on poor execution (high slippage)
                    if abs(slippage_pct) >= self.high_slippage_threshold_pct:
                        logger.warning(
                            f"HIGH SLIPPAGE on {symbol}: {slippage_pct:.4f}% "
                            f"(${slippage:.4f})"
                        )

                    # Update stop loss based on actual fill
                    if direction == 'long':
                        stop_loss = actual_price - (atr * stop_multiplier)
                        take_profit = actual_price + (atr * target_multiplier)
                    else:
                        stop_loss = actual_price + (atr * stop_multiplier)
                        take_profit = actual_price - (atr * target_multiplier)

                    # Place stop loss order (skip if bracket order was used)
                    if not self.use_bracket_orders:
                        self._place_stop_loss(symbol, shares, direction, stop_loss, entry_price=actual_price)

                except BrokerError as e:
                    logger.error(f"Order execution failed: {e}")
                    # Record rejection in order monitor
                    if self.order_monitor:
                        self.order_monitor.order_rejected(order_id, str(e))
                    return
            else:
                logger.info("AUTO_TRADE=False - Trade NOT executed (signal only)")

            # Track position (whether executed or not for monitoring)
            self.positions[symbol] = {
                'direction': direction,
                'entry_price': price,
                'shares': shares,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': format_timestamp(),
                'rrs': rrs,
                'executed': self.auto_trade
            }

            self.total_trades += 1
            self.trades_today.append({
                'symbol': symbol,
                'direction': direction,
                'entry_price': price,
                'shares': shares,
                'time': format_timestamp(),
                'account_id': self.account_id  # Track which account
            })

        except Exception as e:
            logger.error(f"Error entering trade for {symbol}: {e}")

    def _place_stop_loss(
        self,
        symbol: str,
        shares: int,
        direction: str,
        stop_price: float,
        entry_price: Optional[float] = None
    ):
        """
        Place a stop loss order.

        If trailing stops are enabled and an activation price is set,
        will place a trailing stop instead of a regular stop.

        Args:
            symbol: Stock ticker
            shares: Number of shares
            direction: 'long' or 'short'
            stop_price: Initial stop price
            entry_price: Entry price for calculating trailing stop activation
        """
        try:
            if direction == 'long':
                side = OrderSide.SELL
            else:
                side = OrderSide.BUY_TO_COVER

            # Check if we should use trailing stop instead
            if self.use_trailing_stops and entry_price is not None:
                # Calculate activation price (when profit reaches threshold)
                if direction == 'long':
                    activation_price = entry_price * (1 + self.trailing_stop_activation_pct / 100)
                else:
                    activation_price = entry_price * (1 - self.trailing_stop_activation_pct / 100)

                try:
                    order = self.broker.place_trailing_stop(
                        symbol=symbol,
                        side=side,
                        quantity=shares,
                        trail_percent=self.trailing_stop_trail_pct,
                        activation_price=activation_price,
                        time_in_force="GTC"
                    )
                    logger.info(
                        f"Trailing stop placed: {symbol} - Trail: {self.trailing_stop_trail_pct}%, "
                        f"Activates at ${activation_price:.2f} (order: {order.order_id})"
                    )
                    return
                except NotImplementedError:
                    # Broker doesn't support native trailing stops, use regular stop
                    logger.info("Broker doesn't support native trailing stops, using regular stop")

            # Place regular stop loss
            order = self.broker.place_order(
                symbol=symbol,
                side=side,
                quantity=shares,
                order_type=OrderType.STOP,
                stop_price=stop_price
            )

            logger.info(f"Stop loss placed: {symbol} @ ${stop_price:.2f} (order: {order.order_id})")

        except BrokerError as e:
            logger.warning(f"Failed to place stop loss for {symbol}: {e}")

    def _wait_for_fill_confirmation(
        self,
        order_id: str,
        broker_order: 'Order',
        symbol: str,
        shares: int,
        expected_price: float
    ) -> Optional[float]:
        """
        Wait for fill confirmation with timeout.

        Args:
            order_id: Internal order ID for monitoring
            broker_order: Order object from broker
            symbol: Stock symbol
            shares: Number of shares
            expected_price: Expected fill price

        Returns:
            Actual fill price if filled, None if failed/timeout
        """
        timeout = self.fill_confirmation_timeout_seconds
        start_time = time.time()
        poll_interval = 0.5  # Poll every 500ms

        # If already filled (common for market orders in paper trading)
        if broker_order.status == OrderStatus.FILLED:
            actual_price = broker_order.avg_fill_price or expected_price

            # Update order monitor
            if self.order_monitor:
                self.order_monitor.order_fill(
                    order_id=order_id,
                    fill_price=actual_price,
                    fill_quantity=shares
                )

            return actual_price

        # Poll for fill status
        while (time.time() - start_time) < timeout:
            try:
                # Get updated order status from broker
                updated_order = self.broker.get_order_status(broker_order.order_id)

                if updated_order is None:
                    logger.warning(f"Order {broker_order.order_id} not found")
                    time.sleep(poll_interval)
                    continue

                if updated_order.status == OrderStatus.FILLED:
                    actual_price = updated_order.avg_fill_price or expected_price

                    # Update order monitor
                    if self.order_monitor:
                        self.order_monitor.order_fill(
                            order_id=order_id,
                            fill_price=actual_price,
                            fill_quantity=updated_order.filled_quantity
                        )

                    return actual_price

                elif updated_order.status == OrderStatus.PARTIALLY_FILLED:
                    # Update order monitor with partial fill
                    if self.order_monitor and updated_order.avg_fill_price:
                        self.order_monitor.order_fill(
                            order_id=order_id,
                            fill_price=updated_order.avg_fill_price,
                            fill_quantity=updated_order.filled_quantity
                        )

                elif updated_order.status == OrderStatus.REJECTED:
                    if self.order_monitor:
                        self.order_monitor.order_rejected(
                            order_id=order_id,
                            reason=updated_order.error_message or "Rejected by broker"
                        )
                    return None

                elif updated_order.status == OrderStatus.CANCELLED:
                    if self.order_monitor:
                        self.order_monitor.order_cancelled(
                            order_id=order_id,
                            reason="Cancelled"
                        )
                    return None

            except Exception as e:
                logger.error(f"Error checking order status: {e}")

            time.sleep(poll_interval)

        # Timeout - order is stuck
        logger.warning(f"Fill confirmation timeout for {symbol} after {timeout}s")

        return None

    def _log_execution_quality(
        self,
        symbol: str,
        expected_price: float,
        actual_price: float,
        slippage: float,
        slippage_pct: float
    ) -> None:
        """
        Log execution quality metrics.

        Args:
            symbol: Stock symbol
            expected_price: Expected fill price
            actual_price: Actual fill price
            slippage: Slippage in dollars
            slippage_pct: Slippage as percentage
        """
        # Classify execution quality
        abs_slippage = abs(slippage_pct)
        if abs_slippage < 0.05:
            quality = "EXCELLENT"
        elif abs_slippage < 0.1:
            quality = "GOOD"
        elif abs_slippage < 0.25:
            quality = "FAIR"
        elif abs_slippage < 0.5:
            quality = "POOR"
        else:
            quality = "VERY POOR"

        logger.info(
            f"Execution Quality [{quality}]: {symbol} - "
            f"Expected: ${expected_price:.2f}, Actual: ${actual_price:.2f}, "
            f"Slippage: ${slippage:.4f} ({slippage_pct:.4f}%)"
        )

        # Update Prometheus metrics if available
        if METRICS_AVAILABLE:
            try:
                record_execution_metrics(
                    symbol=symbol,
                    side='buy',  # Will be updated based on actual side
                    slippage_pct=slippage_pct,
                    fill_time_seconds=0,  # Not tracked here
                    fill_rate=100.0
                )
            except Exception as e:
                logger.debug(f"Error recording execution metrics: {e}")

    def get_execution_stats(self) -> Dict:
        """
        Get execution statistics from the execution tracker.

        Returns:
            Dictionary with execution statistics
        """
        if not self.execution_tracker:
            return {"error": "ExecutionTracker not available"}

        return self.execution_tracker.get_overall_stats()

    def get_order_monitor_metrics(self) -> Dict:
        """
        Get order monitoring metrics.

        Returns:
            Dictionary with order monitoring metrics
        """
        if not self.order_monitor:
            return {"error": "OrderMonitor not available"}

        return self.order_monitor.get_metrics()

    def check_entry_conditions(self, analysis: Dict) -> Optional[str]:
        """
        Check if trade setup meets entry conditions.

        Args:
            analysis: RRS analysis dict

        Returns:
            'long', 'short', or None
        """
        rrs = analysis.get('rrs', 0)
        daily_strong = analysis.get('daily_strong', False)
        daily_weak = analysis.get('daily_weak', False)

        threshold = self.config.get('rrs_strong_threshold', 2.0)

        # Long setup: Strong RRS + Strong daily chart
        if rrs > threshold and daily_strong:
            return 'long'

        # Short setup: Weak RRS + Weak daily chart
        if rrs < -threshold and daily_weak:
            return 'short'

        return None

    def check_daily_loss_limit(self) -> bool:
        """
        Check if daily loss limit has been hit.

        Returns:
            True if should stop trading
        """
        try:
            account = self.broker.get_account()
            current_equity = account.equity
            daily_pnl = current_equity - self.session_start_equity
            self.daily_pnl = daily_pnl

            max_loss = self.account_size * self.max_daily_loss

            if daily_pnl < -max_loss:
                logger.error(f"DAILY LOSS LIMIT HIT: ${daily_pnl:,.2f} / ${-max_loss:,.2f}")
                return True

            return False

        except Exception as e:
            logger.warning(f"Could not check daily loss limit: {e}")
            return False

    def monitor_positions(self):
        """Monitor and manage open positions."""
        for symbol, position in list(self.positions.items()):
            try:
                # Get current price from broker
                quote = self.broker.get_quote(symbol)
                current_price = quote.last

                stop_loss = position.get('stop_loss')
                take_profit = position.get('take_profit')

                # Check stops/targets
                if position['direction'] == 'long':
                    if stop_loss and current_price <= stop_loss:
                        logger.warning(f"Stop loss triggered for {symbol}")
                        self.exit_position(symbol, current_price, 'stop_loss')
                    elif take_profit and current_price >= take_profit:
                        logger.info(f"Take profit triggered for {symbol}")
                        self.exit_position(symbol, current_price, 'take_profit')
                else:  # short
                    if stop_loss and current_price >= stop_loss:
                        logger.warning(f"Stop loss triggered for {symbol}")
                        self.exit_position(symbol, current_price, 'stop_loss')
                    elif take_profit and current_price <= take_profit:
                        logger.info(f"Take profit triggered for {symbol}")
                        self.exit_position(symbol, current_price, 'take_profit')

            except Exception as e:
                logger.error(f"Error monitoring {symbol}: {e}")

    def exit_position(self, symbol: str, exit_price: float, reason: str):
        """
        Exit a position.

        Args:
            symbol: Stock ticker
            exit_price: Exit price
            reason: Reason for exit
        """
        if symbol not in self.positions:
            logger.warning(f"No tracked position for {symbol}")
            return

        position = self.positions[symbol]

        # Calculate P&L
        if position['direction'] == 'long':
            pnl = (exit_price - position['entry_price']) * position['shares']
        else:
            pnl = (position['entry_price'] - exit_price) * position['shares']

        pnl_percent = (pnl / (position['entry_price'] * position['shares'])) * 100

        logger.info(f"""
{'='*60}
EXITING POSITION
{'='*60}
Symbol: {symbol}
Direction: {position['direction']}
Entry: ${position['entry_price']:.2f}
Exit: ${exit_price:.2f}
Shares: {position['shares']}
P&L: ${pnl:,.2f} ({pnl_percent:+.2f}%)
Reason: {reason}
{'='*60}
        """)

        # Track winning trade
        if pnl > 0:
            self.winning_trades += 1

        # Execute exit order if auto_trade is enabled
        if self.auto_trade and position.get('executed', False):
            try:
                if position['direction'] == 'long':
                    side = OrderSide.SELL
                else:
                    side = OrderSide.BUY_TO_COVER

                order = self.broker.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=position['shares'],
                    order_type=OrderType.MARKET
                )

                logger.info(f"Exit order: {order.status.value}")

            except BrokerError as e:
                logger.error(f"Failed to exit position: {e}")

        # Update daily P&L
        self.daily_pnl += pnl

        # Remove position from tracking
        del self.positions[symbol]

    def get_status(self) -> dict:
        """Get current bot status."""
        try:
            account = self.broker.get_account()
            positions = self.broker.get_positions()

            win_rate = (
                (self.winning_trades / self.total_trades * 100)
                if self.total_trades > 0 else 0
            )

            status = {
                'connected': self.broker.is_connected,
                'paper_trading': self.paper_trading,
                'auto_trade': self.auto_trade,
                'account_equity': account.equity,
                'buying_power': account.buying_power,
                'daily_pnl': self.daily_pnl,
                'positions_count': len(positions),
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': win_rate,
                'time': format_timestamp()
            }

            # Add account info if using account manager
            if self.account_id:
                status['account_id'] = self.account_id
                if self._account_manager:
                    try:
                        acc = self._account_manager.get_account(self.account_id)
                        status['account_name'] = acc.name
                        status['broker_type'] = acc.broker_type.value
                    except Exception:
                        pass

            return status
        except Exception as e:
            return {'error': str(e)}

    def run(self):
        """Run the trading bot."""
        logger.info("Trading Bot Starting...")
        logger.info(f"Current time (ET): {get_eastern_time().strftime('%Y-%m-%d %I:%M:%S %p ET')}")
        logger.info(f"Account Size: ${self.account_size:,.2f}")
        logger.info(f"Max Risk Per Trade: {self.config.get('max_risk_per_trade', 0.01)*100}%")
        logger.info(f"Max Daily Loss: {self.max_daily_loss*100}%")

        # Connect to broker
        if not self.connect_broker():
            logger.error("Failed to connect to broker. Exiting.")
            return

        try:
            while True:
                # Check if it's a trading day
                if not is_trading_day():
                    logger.info(f"Not a trading day. Waiting... (ET: {get_eastern_time().strftime('%I:%M %p')})")
                    time.sleep(300)
                    continue

                # Check trading session
                session = self._get_current_session()

                if session == 'closed':
                    logger.info(f"Market closed. Waiting... (ET: {get_eastern_time().strftime('%I:%M %p')})")
                    time.sleep(60)
                    continue

                # Check if we should trade in current session
                if not self._should_trade_now():
                    if session in ('premarket', 'afterhours'):
                        if not self.extended_hours_enabled:
                            logger.info(
                                f"Extended hours ({session}) - trading disabled. "
                                f"Waiting... (ET: {get_eastern_time().strftime('%I:%M %p')})"
                            )
                        else:
                            logger.info(
                                f"Extended hours ({session}) - outside configured window. "
                                f"Waiting... (ET: {get_eastern_time().strftime('%I:%M %p')})"
                            )
                    time.sleep(60)
                    continue

                # Log session information
                if session != 'regular':
                    logger.info(f"[EXTENDED_HOURS] Trading in {session} session")

                # Check daily loss limit
                if self.check_daily_loss_limit():
                    logger.error("Daily loss limit exceeded - stopping bot")
                    break

                # Run scanner
                self.scanner.scan_once()

                # Monitor existing positions
                self.monitor_positions()

                # Process any pending orders (for paper trading)
                if hasattr(self.broker, 'process_pending_orders'):
                    self.broker.process_pending_orders()

                # Log status periodically
                status = self.get_status()
                session_info = f" [{session}]" if session != 'regular' else ""
                logger.debug(
                    f"Status{session_info}: Equity=${status.get('account_equity', 0):,.2f}, "
                    f"P&L=${status.get('daily_pnl', 0):,.2f}"
                )

                # Wait before next scan
                time.sleep(self.config.get('scan_interval_seconds', 60))

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown of the bot."""
        logger.info("Shutting down trading bot...")

        # Stop order monitoring
        if self.order_monitor:
            self.order_monitor.stop()
            logger.info("OrderMonitor stopped")

        # Log final status
        status = self.get_status()
        logger.info(f"""
{'='*60}
SESSION SUMMARY
{'='*60}
Total Trades: {status.get('total_trades', 0)}
Winning Trades: {status.get('winning_trades', 0)}
Win Rate: {status.get('win_rate', 0):.1f}%
Daily P&L: ${status.get('daily_pnl', 0):,.2f}
Final Equity: ${status.get('account_equity', 0):,.2f}
{'='*60}
        """)

        # Log execution quality summary
        if self.execution_tracker:
            exec_stats = self.execution_tracker.get_overall_stats()
            slippage = exec_stats.get('slippage', {})
            logger.info(f"""
{'='*60}
EXECUTION QUALITY SUMMARY
{'='*60}
Total Executions: {exec_stats.get('total_executions', 0)}
Avg Slippage: {slippage.get('avg_slippage_pct', 0):.4f}%
Max Slippage: {slippage.get('max_slippage_pct', 0):.4f}%
Quality Breakdown: {slippage.get('quality_breakdown', {})}
{'='*60}
            """)

        # Disconnect broker
        try:
            self.broker.disconnect()
            logger.info("Broker disconnected")
        except Exception as e:
            logger.warning(f"Error disconnecting broker: {e}")


if __name__ == "__main__":
    # Default configuration - SAFE settings
    config = {
        # Account settings
        'account_size': 25000,
        'max_risk_per_trade': 0.01,  # 1%
        'max_daily_loss': 0.03,  # 3%
        'max_position_size': 0.10,  # 10%

        # RRS settings
        'atr_period': 14,
        'rrs_strong_threshold': 2.0,
        'stop_atr_multiplier': 1.5,
        'target_atr_multiplier': 3.0,

        # Scanning
        'scan_interval_seconds': 300,

        # Broker settings
        'broker_type': 'paper',  # 'paper', 'schwab', 'ibkr'
        'paper_trading': True,   # SAFE: Use paper trading
        'auto_trade': False,     # SAFE: Don't auto-execute

        # Extended hours trading settings
        'extended_hours_enabled': False,           # SAFE: Disabled by default
        'extended_hours_position_size_pct': 0.5,   # 50% of regular position size
        'premarket_start': '07:00',                # Start trading at 7:00 AM ET
        'afterhours_end': '18:00',                 # Stop trading at 6:00 PM ET

        # Extended hours scanning settings
        'premarket_scan_enabled': False,           # Enable pre-market scanning
        'afterhours_scan_enabled': False,          # Enable after-hours scanning

        # For Schwab (if needed)
        # 'schwab_app_key': 'your_key',
        # 'schwab_app_secret': 'your_secret',

        # For IBKR (if needed)
        # 'ibkr_host': '127.0.0.1',
        # 'ibkr_port': 7497,  # 7497=paper, 7496=live

        # Alert settings
        'alert_method': 'desktop'
    }

    bot = TradingBot(config)
    bot.run()
