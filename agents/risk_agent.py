"""
Risk Agent
Real-time risk monitoring and trade validation with database integration

Responsibilities:
- Validate all trades against comprehensive risk limits
- Monitor portfolio risk in real-time with VaR calculations
- Enforce circuit breakers and daily loss limits
- Calculate dynamic position sizes based on portfolio state
- Track sector concentration and position correlations
- Monitor exposure and enforce diversification limits
- Publish risk alerts and warnings
- Load and track real positions from the database
- Calculate actual P&L from real trade history

Risk Checks Implemented:
1. Position size limits (max % of account)
2. Risk per trade limits (max $ at risk)
3. Risk/reward ratio requirements (min 2:1)
4. Daily loss limits with circuit breakers
5. Maximum drawdown monitoring
6. Portfolio VaR tracking
7. Sector concentration limits (max 25% per sector)
8. Position correlation limits (max correlated positions)
9. Maximum open positions enforcement
10. Buying power validation

Dynamic Position Sizing:
- Reduces position size when portfolio VaR is elevated
- Scales down after daily losses (50% reduction after 50% of daily limit)
- Adjusts for portfolio concentration (reduces size when >70% of max positions)
- Uses ATR-based stops and targets

Event-Driven Architecture:
- Subscribes to: SIGNAL_FOUND, SETUP_VALID, ORDER_REQUESTED, POSITION_OPENED/CLOSED, MARKET_DATA
- Publishes: RISK_CHECK_PASSED/FAILED, RISK_ALERT, TRADING_HALTED, PORTFOLIO_UPDATED

Database Integration:
- Loads positions from TradesRepository on initialization
- Tracks real trade P&L from database
- Syncs position state with database
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger

try:
    import yfinance as yf
except ImportError:
    yf = None
    logger.warning("yfinance not installed - market prices will not be available")

from agents.base import BaseAgent
from agents.events import Event, EventType
from risk.risk_manager import RiskManager
from risk.position_sizer import PositionSizer
from risk.models import (
    RiskCheckResult, RiskMetrics, RiskLevel,
    TradeRisk, RiskViolationType
)
from data.database.trades_repository import get_trades_repository, TradesRepository


class PositionRiskMetrics:
    """Risk metrics for a single position"""

    def __init__(
        self,
        symbol: str,
        direction: str,
        shares: int,
        entry_price: float,
        current_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float]
    ):
        self.symbol = symbol
        self.direction = direction
        self.shares = shares
        self.entry_price = entry_price
        self.current_price = current_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # Calculate derived metrics
        self.position_value = current_price * shares
        self.cost_basis = entry_price * shares

        # P&L calculations
        if direction == "long":
            self.unrealized_pnl = (current_price - entry_price) * shares
        else:
            self.unrealized_pnl = (entry_price - current_price) * shares

        self.unrealized_pnl_percent = (self.unrealized_pnl / self.cost_basis) * 100 if self.cost_basis > 0 else 0

        # Risk at stop
        if stop_loss:
            if direction == "long":
                self.risk_at_stop = (entry_price - stop_loss) * shares
            else:
                self.risk_at_stop = (stop_loss - entry_price) * shares
            self.stop_distance_percent = abs(entry_price - stop_loss) / entry_price * 100
        else:
            self.risk_at_stop = 0
            self.stop_distance_percent = 0

        # Reward at target
        if take_profit:
            if direction == "long":
                self.reward_at_target = (take_profit - entry_price) * shares
            else:
                self.reward_at_target = (entry_price - take_profit) * shares
            self.target_distance_percent = abs(take_profit - entry_price) / entry_price * 100
        else:
            self.reward_at_target = 0
            self.target_distance_percent = 0

        # Risk/Reward
        self.risk_reward_ratio = self.reward_at_target / self.risk_at_stop if self.risk_at_stop > 0 else 0

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_value": self.position_value,
            "cost_basis": self.cost_basis,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_percent": self.unrealized_pnl_percent,
            "risk_at_stop": self.risk_at_stop,
            "stop_distance_percent": self.stop_distance_percent,
            "reward_at_target": self.reward_at_target,
            "target_distance_percent": self.target_distance_percent,
            "risk_reward_ratio": self.risk_reward_ratio
        }


class PortfolioLimitsStatus:
    """Status of all portfolio risk limits"""

    def __init__(self):
        self.limits_breached: List[RiskCheckResult] = []
        self.limits_warning: List[RiskCheckResult] = []
        self.limits_ok: List[RiskCheckResult] = []
        self.is_trading_allowed = True
        self.halt_reason = ""

    def add_check(self, check: RiskCheckResult, warning_threshold: float = 0.8):
        if not check.passed:
            self.limits_breached.append(check)
            self.is_trading_allowed = False
            if not self.halt_reason:
                self.halt_reason = check.message
        elif check.current_value and check.limit_value:
            ratio = check.current_value / check.limit_value if check.limit_value > 0 else 0
            if ratio >= warning_threshold:
                self.limits_warning.append(check)
            else:
                self.limits_ok.append(check)
        else:
            self.limits_ok.append(check)

    def to_dict(self) -> Dict:
        return {
            "is_trading_allowed": self.is_trading_allowed,
            "halt_reason": self.halt_reason,
            "limits_breached": [{"message": c.message, "type": c.violation_type.value if c.violation_type else None} for c in self.limits_breached],
            "limits_warning": [{"message": c.message, "type": c.violation_type.value if c.violation_type else None} for c in self.limits_warning],
            "limits_ok_count": len(self.limits_ok)
        }


class RiskAgent(BaseAgent):
    """
    Real-time risk monitoring agent with database integration

    Responsibilities:
    - Validate all trades against risk limits
    - Monitor portfolio risk in real-time
    - Enforce circuit breakers and daily loss limits
    - Calculate dynamic position sizes
    - Track exposure and concentration
    - Publish risk alerts and warnings
    - Load real positions from database
    - Calculate actual P&L from real trades
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        trades_repository: Optional[TradesRepository] = None,
        **kwargs
    ):
        super().__init__(name="RiskAgent", **kwargs)

        self.risk_manager = risk_manager
        self.position_sizer = risk_manager.position_sizer

        # Database integration
        self._trades_repo = trades_repository

        # Real-time tracking
        self.active_positions: Dict[str, Dict] = {}
        self.pending_orders: Dict[str, Dict] = {}
        self.risk_violations_today: List[RiskCheckResult] = []

        # Market price cache
        self._price_cache: Dict[str, Dict] = {}
        self._price_cache_time: Dict[str, datetime] = {}
        self._price_cache_ttl = 30  # seconds

        # VaR calculation tracking
        self.portfolio_var: float = 0.0
        self.last_var_update: Optional[datetime] = None

        # Sector and correlation tracking
        self.sector_exposure: Dict[str, float] = {}  # sector -> total value
        self.position_correlations: Dict[str, List[str]] = {}  # symbol -> correlated symbols

        # Circuit breaker state
        self.circuit_breaker_triggered = False
        self.circuit_breaker_reason = ""

        # Alert thresholds (% of limit before warning)
        self.warning_threshold = 0.80  # 80% of limit

        # Today's trade tracking
        self._todays_closed_trades: List[Dict] = []

    @property
    def trades_repo(self) -> TradesRepository:
        """Lazy-load trades repository"""
        if self._trades_repo is None:
            self._trades_repo = get_trades_repository()
        return self._trades_repo

    async def initialize(self):
        """Initialize risk agent and load portfolio state from database"""
        logger.info("Initializing Risk Agent")

        # Initialize metrics
        self.metrics.custom_metrics.update({
            "trades_approved": 0,
            "trades_rejected": 0,
            "approval_rate": 0.0,
            "risk_alerts_sent": 0,
            "circuit_breakers_triggered": 0,
            "avg_position_size": 0.0,
            "current_var": 0.0,
            "sector_violations": 0,
            "correlation_violations": 0,
            "daily_pnl": 0.0,
            "total_exposure": 0.0
        })

        # Load real portfolio state from database
        await self.load_portfolio_state()

        logger.info(f"Risk limits: {self.risk_manager.limits}")
        logger.info(f"Loaded {len(self.active_positions)} open positions from database")

    async def cleanup(self):
        """Cleanup risk agent"""
        # Generate final risk report
        report = self.risk_manager.generate_daily_report()
        logger.info(f"Daily Risk Report: P&L=${report.daily_pnl:,.2f} "
                   f"({report.daily_pnl_percent:.2f}%), "
                   f"Trades={report.total_trades}, "
                   f"Win Rate={report.win_rate*100:.1f}%")

    def get_subscribed_events(self) -> List[EventType]:
        """Events this agent listens to"""
        return [
            EventType.SIGNAL_FOUND,
            EventType.SETUP_VALID,
            EventType.ORDER_REQUESTED,
            EventType.POSITION_OPENED,
            EventType.POSITION_CLOSED,
            EventType.MARKET_DATA,
            EventType.MARKET_OPEN,
            EventType.MARKET_CLOSE
        ]

    async def handle_event(self, event: Event):
        """Handle incoming events"""
        try:
            if event.event_type == EventType.SIGNAL_FOUND:
                await self.validate_signal(event.data)

            elif event.event_type == EventType.SETUP_VALID:
                await self.validate_setup(event.data)

            elif event.event_type == EventType.ORDER_REQUESTED:
                await self.approve_order(event.data)

            elif event.event_type == EventType.POSITION_OPENED:
                await self.track_position_opened(event.data)

            elif event.event_type == EventType.POSITION_CLOSED:
                await self.track_position_closed(event.data)

            elif event.event_type == EventType.MARKET_DATA:
                await self.update_var_calculations(event.data)

            elif event.event_type == EventType.MARKET_OPEN:
                await self.on_market_open()

            elif event.event_type == EventType.MARKET_CLOSE:
                await self.on_market_close()

        except Exception as e:
            logger.error(f"Risk agent error handling {event.event_type}: {e}")
            await self.publish(EventType.SYSTEM_ERROR, {
                "agent": self.name,
                "error": str(e),
                "event_type": event.event_type.value
            })

    # ==================== Database Integration ====================

    async def load_portfolio_state(self):
        """
        Load current positions from database and calculate exposure.

        This method:
        - Loads all open positions from TradesRepository
        - Fetches current market prices
        - Calculates unrealized P&L
        - Updates risk manager with position data
        - Calculates total portfolio exposure
        """
        logger.info("Loading portfolio state from database...")

        try:
            # Get open positions from database
            positions = self.trades_repo.get_open_positions()
            logger.info(f"Found {len(positions)} open positions in database")

            # Clear current tracking
            self.active_positions.clear()
            self.risk_manager.open_positions.clear()
            self.sector_exposure.clear()

            # Load each position
            for pos in positions:
                symbol = pos['symbol']
                entry_price = pos['entry_price']
                shares = pos['shares']
                direction = pos['direction']
                stop_loss = pos.get('stop_loss')
                take_profit = pos.get('take_profit')

                # Get current market price
                current_price = await self._get_current_price(symbol)
                if current_price is None:
                    current_price = entry_price  # Fallback to entry price

                # Calculate unrealized P&L
                if direction == 'long':
                    unrealized_pnl = (current_price - entry_price) * shares
                else:
                    unrealized_pnl = (entry_price - current_price) * shares

                position_value = current_price * shares

                # Build position dict
                position_data = {
                    "symbol": symbol,
                    "direction": direction,
                    "shares": shares,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "stop_price": stop_loss,
                    "target_price": take_profit,
                    "position_value": position_value,
                    "cost_basis": entry_price * shares,
                    "unrealized_pnl": unrealized_pnl,
                    "sector": pos.get('sector'),
                    "opened_at": pos.get('entry_time'),
                    "db_id": pos.get('id')
                }

                # Track position
                self.active_positions[symbol] = position_data
                self.risk_manager.add_position(symbol, position_data)

                logger.debug(f"Loaded position: {symbol} - {shares} shares @ ${entry_price:.2f}, "
                            f"current ${current_price:.2f}, P&L ${unrealized_pnl:.2f}")

            # Calculate today's closed trade P&L
            await self._load_todays_closed_trades()

            # Update exposure metrics
            await self._update_exposure_metrics()

            logger.info(f"Portfolio loaded: {len(self.active_positions)} positions, "
                       f"Total exposure: ${sum(p['position_value'] for p in self.active_positions.values()):,.2f}")

        except Exception as e:
            logger.error(f"Error loading portfolio state: {e}")
            raise

    async def _load_todays_closed_trades(self):
        """Load today's closed trades from database for P&L calculation"""
        try:
            # Get trades closed today
            trades = self.trades_repo.get_trades(status='closed', days=1)

            # Filter to only today's closed trades
            today = date.today()
            self._todays_closed_trades = []

            for trade in trades:
                exit_time_str = trade.get('exit_time')
                if exit_time_str:
                    if isinstance(exit_time_str, str):
                        exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00'))
                    else:
                        exit_time = exit_time_str

                    if exit_time.date() == today:
                        self._todays_closed_trades.append(trade)

            # Update risk manager with today's P&L
            daily_pnl = sum(t.get('pnl', 0) or 0 for t in self._todays_closed_trades)
            self.risk_manager.daily_pnl = daily_pnl
            self.risk_manager.daily_trades = len(self._todays_closed_trades)
            self.risk_manager.daily_wins = sum(1 for t in self._todays_closed_trades if (t.get('pnl') or 0) > 0)
            self.risk_manager.daily_losses = sum(1 for t in self._todays_closed_trades if (t.get('pnl') or 0) <= 0)

            logger.info(f"Loaded {len(self._todays_closed_trades)} closed trades today, P&L: ${daily_pnl:.2f}")

        except Exception as e:
            logger.error(f"Error loading today's closed trades: {e}")

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol using yfinance"""
        # Check cache first
        cache_key = symbol
        if cache_key in self._price_cache:
            cache_time = self._price_cache_time.get(cache_key)
            if cache_time and (datetime.now() - cache_time).total_seconds() < self._price_cache_ttl:
                return self._price_cache[cache_key].get('price')

        if yf is None:
            logger.warning(f"Cannot fetch price for {symbol} - yfinance not available")
            return None

        try:
            loop = asyncio.get_running_loop()
            price = await loop.run_in_executor(None, self._fetch_price_sync, symbol)

            if price:
                self._price_cache[cache_key] = {'price': price, 'symbol': symbol}
                self._price_cache_time[cache_key] = datetime.now()

            return price

        except Exception as e:
            logger.debug(f"Error fetching price for {symbol}: {e}")
            return None

    def _fetch_price_sync(self, symbol: str) -> Optional[float]:
        """Synchronous price fetch"""
        try:
            ticker = yf.Ticker(symbol)
            # Try to get the most recent price
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])

            # Fallback to daily data
            hist = ticker.history(period="5d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])

            return None
        except Exception as e:
            logger.debug(f"Error in sync price fetch for {symbol}: {e}")
            return None

    def calculate_daily_pnl(self) -> Dict:
        """
        Calculate today's P&L from closed trades.

        Returns:
            Dict with:
            - realized_pnl: P&L from closed trades
            - unrealized_pnl: P&L from open positions
            - total_pnl: Sum of realized and unrealized
            - num_trades: Number of closed trades today
            - win_rate: Percentage of winning trades
        """
        # Realized P&L from closed trades
        realized_pnl = sum(t.get('pnl', 0) or 0 for t in self._todays_closed_trades)
        num_trades = len(self._todays_closed_trades)
        winners = sum(1 for t in self._todays_closed_trades if (t.get('pnl') or 0) > 0)
        win_rate = winners / num_trades if num_trades > 0 else 0

        # Unrealized P&L from open positions
        unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in self.active_positions.values())

        return {
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": realized_pnl + unrealized_pnl,
            "num_trades": num_trades,
            "winners": winners,
            "losers": num_trades - winners,
            "win_rate": win_rate,
            "daily_pnl_percent": (realized_pnl / self.risk_manager.daily_start_balance) * 100 if self.risk_manager.daily_start_balance > 0 else 0
        }

    def get_position_risk(self, symbol: str) -> Optional[PositionRiskMetrics]:
        """
        Get risk metrics for a specific position.

        Args:
            symbol: Stock ticker symbol

        Returns:
            PositionRiskMetrics object with risk calculations, or None if position not found
        """
        if symbol not in self.active_positions:
            return None

        pos = self.active_positions[symbol]

        return PositionRiskMetrics(
            symbol=symbol,
            direction=pos.get('direction', 'long'),
            shares=pos.get('shares', 0),
            entry_price=pos.get('entry_price', 0),
            current_price=pos.get('current_price', pos.get('entry_price', 0)),
            stop_loss=pos.get('stop_price'),
            take_profit=pos.get('target_price')
        )

    def check_portfolio_limits(self) -> PortfolioLimitsStatus:
        """
        Check if any risk limits are breached.

        Returns:
            PortfolioLimitsStatus with details on all limit checks
        """
        status = PortfolioLimitsStatus()

        # 1. Daily loss limit
        daily_pnl_data = self.calculate_daily_pnl()
        max_daily_loss = self.risk_manager.current_balance * self.risk_manager.limits.max_daily_loss
        current_loss = abs(min(0, daily_pnl_data['realized_pnl']))

        status.add_check(RiskCheckResult(
            passed=current_loss < max_daily_loss,
            violation_type=RiskViolationType.MAX_DAILY_LOSS if current_loss >= max_daily_loss else None,
            message=f"Daily loss: ${current_loss:,.0f} / ${max_daily_loss:,.0f} limit",
            current_value=current_loss,
            limit_value=max_daily_loss,
            risk_level=RiskLevel.CRITICAL if current_loss >= max_daily_loss else RiskLevel.LOW
        ), self.warning_threshold)

        # 2. Max open positions
        current_positions = len(self.active_positions)
        max_positions = self.risk_manager.limits.max_open_positions

        status.add_check(RiskCheckResult(
            passed=current_positions < max_positions,
            violation_type=RiskViolationType.MAX_OPEN_POSITIONS if current_positions >= max_positions else None,
            message=f"Open positions: {current_positions} / {max_positions} max",
            current_value=current_positions,
            limit_value=max_positions,
            risk_level=RiskLevel.MEDIUM if current_positions >= max_positions else RiskLevel.LOW
        ), self.warning_threshold)

        # 3. Total exposure
        total_exposure = sum(p.get('position_value', 0) for p in self.active_positions.values())
        max_exposure = self.risk_manager.current_balance * self.risk_manager.limits.max_total_exposure

        status.add_check(RiskCheckResult(
            passed=total_exposure <= max_exposure,
            violation_type=RiskViolationType.MAX_POSITION_SIZE if total_exposure > max_exposure else None,
            message=f"Total exposure: ${total_exposure:,.0f} / ${max_exposure:,.0f} max",
            current_value=total_exposure,
            limit_value=max_exposure,
            risk_level=RiskLevel.HIGH if total_exposure > max_exposure else RiskLevel.LOW
        ), self.warning_threshold)

        # 4. Drawdown limit
        dd_percent = (self.risk_manager.current_drawdown / self.risk_manager.peak_balance) if self.risk_manager.peak_balance > 0 else 0
        max_dd = self.risk_manager.limits.max_drawdown

        status.add_check(RiskCheckResult(
            passed=dd_percent < max_dd,
            violation_type=RiskViolationType.MAX_DRAWDOWN if dd_percent >= max_dd else None,
            message=f"Drawdown: {dd_percent*100:.1f}% / {max_dd*100:.0f}% max",
            current_value=dd_percent * 100,
            limit_value=max_dd * 100,
            risk_level=RiskLevel.CRITICAL if dd_percent >= max_dd else RiskLevel.LOW
        ), self.warning_threshold)

        # 5. Portfolio VaR
        var_percent = (self.portfolio_var / self.risk_manager.current_balance) * 100 if self.risk_manager.current_balance > 0 else 0
        max_var = self.risk_manager.limits.max_risk_per_trade * len(self.active_positions) * 100

        if max_var > 0:
            status.add_check(RiskCheckResult(
                passed=var_percent <= max_var,
                message=f"Portfolio VaR: {var_percent:.1f}% / {max_var:.1f}% max",
                current_value=var_percent,
                limit_value=max_var,
                risk_level=RiskLevel.MEDIUM if var_percent > max_var else RiskLevel.LOW
            ), self.warning_threshold)

        # 6. Sector concentration
        for sector, value in self.sector_exposure.items():
            sector_percent = value / self.risk_manager.current_balance if self.risk_manager.current_balance > 0 else 0
            max_sector = self.risk_manager.limits.max_sector_exposure

            status.add_check(RiskCheckResult(
                passed=sector_percent <= max_sector,
                violation_type=RiskViolationType.MAX_SECTOR_EXPOSURE if sector_percent > max_sector else None,
                message=f"Sector {sector}: {sector_percent*100:.1f}% / {max_sector*100:.0f}% max",
                current_value=sector_percent * 100,
                limit_value=max_sector * 100,
                risk_level=RiskLevel.MEDIUM if sector_percent > max_sector else RiskLevel.LOW
            ), self.warning_threshold)

        # 7. Trading halted check
        if self.risk_manager.trading_halted or self.circuit_breaker_triggered:
            status.is_trading_allowed = False
            status.halt_reason = self.risk_manager.halt_reason or self.circuit_breaker_reason

        return status

    async def refresh_position_prices(self):
        """Refresh current prices for all open positions"""
        logger.debug("Refreshing position prices...")

        for symbol, pos in self.active_positions.items():
            current_price = await self._get_current_price(symbol)
            if current_price:
                old_price = pos.get('current_price', pos['entry_price'])
                pos['current_price'] = current_price
                pos['position_value'] = current_price * pos['shares']

                # Recalculate unrealized P&L
                if pos['direction'] == 'long':
                    pos['unrealized_pnl'] = (current_price - pos['entry_price']) * pos['shares']
                else:
                    pos['unrealized_pnl'] = (pos['entry_price'] - current_price) * pos['shares']

                if abs(current_price - old_price) > 0.01:
                    logger.debug(f"Updated {symbol} price: ${old_price:.2f} -> ${current_price:.2f}")

        # Update portfolio VaR with new prices
        await self.update_var_calculations({})

    # ==================== Signal/Setup Validation ====================

    async def validate_signal(self, signal: Dict):
        """
        Validate a trading signal before analysis
        Quick pre-check to filter out obvious violations
        """
        symbol = signal.get("symbol")

        # Check if trading is halted
        if self.risk_manager.trading_halted:
            logger.warning(f"Signal rejected - trading halted: {symbol}")
            await self._publish_rejection(symbol, "Trading halted", signal)
            return

        # Check max positions
        if len(self.active_positions) >= self.risk_manager.limits.max_open_positions:
            logger.warning(f"Signal rejected - max positions reached: {symbol}")
            await self._publish_rejection(symbol, "Max positions reached", signal)
            return

        # Check daily loss limit
        if self.risk_manager.check_daily_loss_limit():
            logger.warning(f"Signal rejected - daily loss limit exceeded: {symbol}")
            await self._trigger_circuit_breaker("Daily loss limit exceeded")
            return

        # Signal passes pre-checks
        logger.debug(f"Signal pre-check passed: {symbol}")

    async def validate_setup(self, setup: Dict):
        """
        Validate a complete trade setup
        Includes position sizing and full risk analysis
        """
        symbol = setup.get("symbol")
        direction = setup.get("direction", "long")
        entry_price = setup.get("entry_price")
        atr = setup.get("atr")
        sector = setup.get("sector")  # Optional sector classification
        correlated_symbols = setup.get("correlated_symbols")  # Optional correlation data

        if not all([entry_price, atr]):
            logger.error(f"Invalid setup data for {symbol}")
            return

        logger.info(f"Validating setup: {symbol} {direction.upper()}")

        # Check portfolio limits first
        limits_status = self.check_portfolio_limits()
        if not limits_status.is_trading_allowed:
            logger.warning(f"Setup rejected - portfolio limits breached: {limits_status.halt_reason}")
            await self._publish_rejection(symbol, limits_status.halt_reason, setup)
            return

        # Calculate position size with risk adjustment
        position_size = self._calculate_dynamic_position_size(
            entry_price=entry_price,
            atr=atr,
            direction=direction,
            symbol=symbol
        )

        if position_size.shares == 0:
            logger.warning(f"Position size is 0 for {symbol}: {position_size.reason}")
            await self._publish_rejection(symbol, position_size.reason, setup)
            return

        # Run full risk validation
        trade_risk = self.risk_manager.validate_trade(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            shares=position_size.shares,
            stop_price=position_size.stop_price,
            target_price=position_size.target_price,
            atr=atr
        )

        # Additional risk checks: sector concentration and correlation
        sector_check = await self._check_sector_concentration(
            symbol=symbol,
            position_value=position_size.position_value,
            sector=sector
        )
        if not sector_check.passed:
            trade_risk.checks_failed.append(sector_check)

        correlation_check = await self._check_position_correlation(
            symbol=symbol,
            correlated_symbols=correlated_symbols
        )
        if not correlation_check.passed:
            trade_risk.checks_failed.append(correlation_check)

        # Check if trade passes all risk checks
        if trade_risk.is_valid:
            logger.info(f"Risk check PASSED: {symbol} - {position_size.shares} shares, "
                       f"Risk=${trade_risk.risk_amount:.0f} ({trade_risk.risk_percent_of_account:.2f}%), "
                       f"R/R={trade_risk.risk_reward_ratio:.2f}")

            await self._publish_approval(symbol, trade_risk, position_size, setup)

            # Check for warnings (approaching limits)
            await self._check_warning_thresholds(trade_risk)

        else:
            # Trade failed risk checks
            failed_checks = ", ".join([c.message for c in trade_risk.checks_failed])
            logger.warning(f"Risk check FAILED: {symbol} - {failed_checks}")

            await self._publish_rejection(symbol, failed_checks, setup, trade_risk)

            # Check for critical violations
            await self._handle_critical_violations(trade_risk)

    async def approve_order(self, order_request: Dict):
        """
        Final approval before order execution
        Last checkpoint before money moves
        """
        symbol = order_request.get("symbol")
        setup = order_request.get("setup", {})

        # Re-validate that conditions haven't changed
        if self.risk_manager.trading_halted:
            logger.error(f"Order rejected - trading halted: {symbol}")
            await self._publish_rejection(symbol, self.risk_manager.halt_reason, setup)
            return

        # Check daily loss limit again
        if self.risk_manager.check_daily_loss_limit():
            await self._trigger_circuit_breaker("Daily loss limit exceeded")
            return

        # Check portfolio limits
        limits_status = self.check_portfolio_limits()
        if not limits_status.is_trading_allowed:
            logger.error(f"Order rejected - portfolio limits: {limits_status.halt_reason}")
            await self._publish_rejection(symbol, limits_status.halt_reason, setup)
            return

        # Track pending order
        self.pending_orders[symbol] = {
            "setup": setup,
            "timestamp": datetime.now()
        }

        logger.info(f"Order approved for execution: {symbol}")

    # ==================== Position Tracking ====================

    async def track_position_opened(self, position: Dict):
        """Track a new position and sync with database"""
        symbol = position.get("symbol")
        shares = position.get("shares")
        entry_price = position.get("entry_price")
        stop_price = position.get("stop_price")
        direction = position.get("direction")
        sector = position.get("sector")
        correlated_symbols = position.get("correlated_symbols")

        position_value = entry_price * shares

        # Add to tracking
        self.active_positions[symbol] = {
            "symbol": symbol,
            "direction": direction,
            "shares": shares,
            "entry_price": entry_price,
            "current_price": entry_price,
            "stop_price": stop_price,
            "target_price": position.get("target_price"),
            "position_value": position_value,
            "cost_basis": position_value,
            "sector": sector,
            "correlated_symbols": correlated_symbols,
            "opened_at": datetime.now(),
            "unrealized_pnl": 0.0
        }

        # Update risk manager
        self.risk_manager.add_position(symbol, self.active_positions[symbol])

        # Update sector exposure tracking
        self._update_sector_exposure(symbol, sector, position_value, add=True)

        # Update correlation tracking
        self._update_correlation_tracking(symbol, correlated_symbols, add=True)

        # Remove from pending
        self.pending_orders.pop(symbol, None)

        logger.info(f"Position tracked: {symbol} - {shares} shares @ ${entry_price:.2f}" +
                   (f" [Sector: {sector}]" if sector else ""))

        # Update metrics
        await self._update_exposure_metrics()

        # Check portfolio risk
        await self._check_portfolio_risk()

    async def track_position_closed(self, position: Dict):
        """Track a closed position and update P&L"""
        symbol = position.get("symbol")
        pnl = position.get("pnl", 0.0)
        exit_price = position.get("exit_price")

        if symbol not in self.active_positions:
            logger.warning(f"Position not found in tracking: {symbol}")
            return

        pos = self.active_positions[symbol]
        logger.info(f"Position closed: {symbol} - P&L=${pnl:.2f} "
                   f"(Entry=${pos['entry_price']:.2f}, Exit=${exit_price:.2f})")

        # Add to today's closed trades
        self._todays_closed_trades.append({
            "symbol": symbol,
            "pnl": pnl,
            "exit_price": exit_price,
            "entry_price": pos['entry_price'],
            "shares": pos['shares'],
            "direction": pos['direction'],
            "exit_time": datetime.now().isoformat()
        })

        # Record trade
        is_day_trade = position.get("is_day_trade", False)
        self.risk_manager.record_trade(pnl, is_day_trade)

        # Update sector exposure tracking (remove)
        self._update_sector_exposure(
            symbol=symbol,
            sector=pos.get("sector"),
            position_value=pos["position_value"],
            add=False
        )

        # Update correlation tracking (remove)
        self._update_correlation_tracking(
            symbol=symbol,
            correlated_symbols=pos.get("correlated_symbols"),
            add=False
        )

        # Remove from tracking
        del self.active_positions[symbol]
        self.risk_manager.remove_position(symbol)

        # Update metrics
        await self._update_exposure_metrics()

        # Publish P&L update
        await self.publish(EventType.PNL_UPDATED, {
            "symbol": symbol,
            "pnl": pnl,
            "daily_pnl": self.risk_manager.daily_pnl,
            "balance": self.risk_manager.current_balance
        })

        # Check if we need to halt trading
        if self.risk_manager.check_daily_loss_limit():
            await self._trigger_circuit_breaker("Daily loss limit exceeded")

    # ==================== VaR Calculations ====================

    async def update_var_calculations(self, market_data: Dict):
        """
        Update portfolio Value at Risk calculations
        Based on current positions and market volatility
        """
        if not self.active_positions:
            self.portfolio_var = 0.0
            return

        # Simple VaR calculation: sum of position risks
        # In production, would use historical simulation or parametric VaR
        total_risk = 0.0

        for symbol, position in self.active_positions.items():
            stop_price = position.get("stop_price")
            entry_price = position.get("entry_price")
            current_price = position.get("current_price", entry_price)

            if stop_price:
                # Risk to stop
                stop_distance = abs(current_price - stop_price)
                position_risk = stop_distance * position["shares"]
            else:
                # Estimate risk as 2% of position value
                position_risk = position.get("position_value", 0) * 0.02

            total_risk += position_risk

        self.portfolio_var = total_risk
        self.last_var_update = datetime.now()

        # Update metrics
        var_percent = (self.portfolio_var / self.risk_manager.current_balance) * 100 if self.risk_manager.current_balance > 0 else 0
        self.metrics.custom_metrics["current_var"] = var_percent

        # Check if VaR is too high
        max_var_percent = self.risk_manager.limits.max_risk_per_trade * len(self.active_positions) * 100
        if var_percent > max_var_percent and max_var_percent > 0:
            await self.publish(EventType.RISK_ALERT, {
                "type": "high_var",
                "message": f"Portfolio VaR elevated: {var_percent:.2f}%",
                "var": self.portfolio_var,
                "var_percent": var_percent
            })

    # ==================== Sector & Correlation Checks ====================

    async def _check_sector_concentration(
        self,
        symbol: str,
        position_value: float,
        sector: Optional[str] = None
    ) -> RiskCheckResult:
        """
        Check if adding this position would violate sector concentration limits

        Args:
            symbol: Stock symbol
            position_value: Proposed position value
            sector: Sector classification (if known)

        Returns:
            RiskCheckResult indicating if check passed
        """
        if sector is None:
            # If sector not provided, try to infer or skip check
            # In production, would query from market data
            return RiskCheckResult(
                passed=True,
                message="Sector unknown - check skipped"
            )

        # Calculate new sector exposure
        current_sector_value = self.sector_exposure.get(sector, 0.0)
        new_sector_value = current_sector_value + position_value

        # Check against limit
        max_sector_value = self.risk_manager.current_balance * self.risk_manager.limits.max_sector_exposure
        passed = new_sector_value <= max_sector_value

        sector_percent = (new_sector_value / self.risk_manager.current_balance) * 100 if self.risk_manager.current_balance > 0 else 0
        limit_percent = self.risk_manager.limits.max_sector_exposure * 100

        if not passed:
            logger.warning(f"Sector concentration limit exceeded: {sector} would be {sector_percent:.1f}% (limit: {limit_percent:.0f}%)")
            self.metrics.custom_metrics["sector_violations"] += 1

        return RiskCheckResult(
            passed=passed,
            violation_type=RiskViolationType.MAX_SECTOR_EXPOSURE if not passed else None,
            message=f"Sector {sector}: {sector_percent:.1f}% / {limit_percent:.0f}% max",
            current_value=new_sector_value,
            limit_value=max_sector_value,
            risk_level=RiskLevel.HIGH if not passed else RiskLevel.LOW
        )

    async def _check_position_correlation(
        self,
        symbol: str,
        correlated_symbols: Optional[List[str]] = None
    ) -> RiskCheckResult:
        """
        Check if adding this position would create too many correlated positions

        Args:
            symbol: Stock symbol
            correlated_symbols: List of symbols correlated with this one

        Returns:
            RiskCheckResult indicating if check passed
        """
        if not correlated_symbols:
            # No correlation data available
            return RiskCheckResult(
                passed=True,
                message="Correlation data unavailable - check skipped"
            )

        # Count how many correlated positions we already have
        correlated_count = 0
        correlated_list = []

        for existing_symbol in self.active_positions.keys():
            if existing_symbol in correlated_symbols:
                correlated_count += 1
                correlated_list.append(existing_symbol)

        # Check against limit
        max_correlated = self.risk_manager.limits.max_correlated_positions
        passed = correlated_count < max_correlated

        if not passed:
            logger.warning(f"Correlation limit exceeded: {symbol} correlated with {correlated_count} existing positions: {correlated_list}")
            self.metrics.custom_metrics["correlation_violations"] += 1

        return RiskCheckResult(
            passed=passed,
            violation_type=RiskViolationType.MAX_CORRELATION if not passed else None,
            message=f"Correlated positions: {correlated_count} / {max_correlated} max (with: {', '.join(correlated_list[:3])})",
            current_value=correlated_count,
            limit_value=max_correlated,
            risk_level=RiskLevel.MEDIUM if not passed else RiskLevel.LOW
        )

    def _update_sector_exposure(self, symbol: str, sector: Optional[str], position_value: float, add: bool = True):
        """
        Update sector exposure tracking

        Args:
            symbol: Stock symbol
            sector: Sector classification
            position_value: Position value
            add: True to add, False to remove
        """
        if not sector:
            return

        if add:
            self.sector_exposure[sector] = self.sector_exposure.get(sector, 0.0) + position_value
            logger.debug(f"Added to {sector} exposure: ${position_value:,.0f} (total: ${self.sector_exposure[sector]:,.0f})")
        else:
            if sector in self.sector_exposure:
                self.sector_exposure[sector] = max(0.0, self.sector_exposure[sector] - position_value)
                logger.debug(f"Removed from {sector} exposure: ${position_value:,.0f} (total: ${self.sector_exposure[sector]:,.0f})")

                # Clean up empty sectors
                if self.sector_exposure[sector] == 0.0:
                    del self.sector_exposure[sector]

    def _update_correlation_tracking(self, symbol: str, correlated_symbols: Optional[List[str]] = None, add: bool = True):
        """
        Update position correlation tracking

        Args:
            symbol: Stock symbol
            correlated_symbols: List of correlated symbols
            add: True to add, False to remove
        """
        if not correlated_symbols:
            return

        if add:
            self.position_correlations[symbol] = correlated_symbols
            logger.debug(f"Tracking correlations for {symbol}: {len(correlated_symbols)} symbols")
        else:
            if symbol in self.position_correlations:
                del self.position_correlations[symbol]
                logger.debug(f"Removed correlation tracking for {symbol}")

    # ==================== Circuit Breakers ====================

    async def _trigger_circuit_breaker(self, reason: str):
        """Trigger emergency trading halt"""
        if self.circuit_breaker_triggered:
            return  # Already triggered

        self.circuit_breaker_triggered = True
        self.circuit_breaker_reason = reason
        self.risk_manager.halt_trading(reason)

        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")

        self.metrics.custom_metrics["circuit_breakers_triggered"] += 1

        await self.publish(EventType.TRADING_HALTED, {
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "daily_pnl": self.risk_manager.daily_pnl,
            "open_positions": len(self.active_positions)
        })

    async def reset_circuit_breaker(self):
        """Reset circuit breaker (manual intervention)"""
        self.circuit_breaker_triggered = False
        self.circuit_breaker_reason = ""
        self.risk_manager.resume_trading()

        logger.warning("Circuit breaker reset - trading resumed")

        await self.publish(EventType.SYSTEM_START, {
            "message": "Trading resumed after circuit breaker reset"
        })

    # ==================== Position Sizing ====================

    def _calculate_dynamic_position_size(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        symbol: str
    ):
        """
        Calculate position size with dynamic risk adjustment

        Reduces position size if:
        - Portfolio VaR is elevated
        - Multiple positions open
        - Recent losses
        """
        # Base risk percentage
        base_risk = self.risk_manager.limits.max_risk_per_trade

        # Adjust for portfolio state
        risk_adjustment = 1.0

        # 1. Reduce if portfolio VaR is high
        if len(self.active_positions) > 0:
            var_percent = (self.portfolio_var / self.risk_manager.current_balance) if self.risk_manager.current_balance > 0 else 0
            max_var = self.risk_manager.limits.max_risk_per_trade * 5  # 5 positions worth

            if var_percent > max_var:
                var_adjustment = max_var / var_percent
                risk_adjustment *= var_adjustment
                logger.info(f"Position size reduced due to high VaR: {var_adjustment:.2f}x")

        # 2. Reduce if recent losses
        if self.risk_manager.daily_pnl < 0:
            loss_percent = abs(self.risk_manager.daily_pnl / self.risk_manager.daily_start_balance) if self.risk_manager.daily_start_balance > 0 else 0
            max_loss_allowed = self.risk_manager.limits.max_daily_loss

            if loss_percent > max_loss_allowed * 0.5:  # Past 50% of daily loss limit
                loss_adjustment = 0.5  # Cut position size in half
                risk_adjustment *= loss_adjustment
                logger.info(f"Position size reduced due to daily losses: {loss_adjustment:.2f}x")

        # 3. Reduce if approaching max positions
        positions_ratio = len(self.active_positions) / self.risk_manager.limits.max_open_positions if self.risk_manager.limits.max_open_positions > 0 else 0
        if positions_ratio > 0.7:  # Over 70% of max positions
            position_adjustment = 1.0 - (positions_ratio - 0.7) * 0.5
            risk_adjustment *= position_adjustment

        # Calculate final risk percentage
        adjusted_risk = base_risk * risk_adjustment

        logger.debug(f"Position sizing for {symbol}: "
                    f"Base={base_risk*100:.1f}%, "
                    f"Adjusted={adjusted_risk*100:.1f}%, "
                    f"Factor={risk_adjustment:.2f}x")

        # Use position sizer with adjusted risk
        return self.position_sizer.calculate_position_size(
            account_size=self.risk_manager.current_balance,
            entry_price=entry_price,
            atr=atr,
            direction=direction,
            custom_risk_percent=adjusted_risk
        )

    # ==================== Event Publishing ====================

    async def _publish_approval(
        self,
        symbol: str,
        trade_risk: TradeRisk,
        position_size,
        setup: Dict
    ):
        """Publish risk check passed event"""
        self.metrics.custom_metrics["trades_approved"] += 1
        self._update_approval_rate()

        await self.publish(EventType.RISK_CHECK_PASSED, {
            "symbol": symbol,
            "shares": position_size.shares,
            "position_value": position_size.position_value,
            "risk_amount": trade_risk.risk_amount,
            "risk_percent": trade_risk.risk_percent_of_account,
            "stop_price": position_size.stop_price,
            "target_price": position_size.target_price,
            "risk_reward_ratio": trade_risk.risk_reward_ratio,
            "risk_level": trade_risk.overall_risk_level.value,
            "sector": setup.get("sector"),
            "correlated_symbols": setup.get("correlated_symbols"),
            "timestamp": datetime.now().isoformat()
        })

    async def _publish_rejection(
        self,
        symbol: str,
        reason: str,
        setup: Dict,
        trade_risk: Optional[TradeRisk] = None
    ):
        """Publish risk check failed event"""
        self.metrics.custom_metrics["trades_rejected"] += 1
        self._update_approval_rate()

        violation_types = []
        if trade_risk:
            violation_types = [c.violation_type.value for c in trade_risk.checks_failed
                             if c.violation_type]

        await self.publish(EventType.RISK_CHECK_FAILED, {
            "symbol": symbol,
            "reason": reason,
            "violations": violation_types,
            "risk_level": trade_risk.overall_risk_level.value if trade_risk else "high",
            "timestamp": datetime.now().isoformat()
        })

    # ==================== Warning System ====================

    async def _check_warning_thresholds(self, trade_risk: TradeRisk):
        """Check if we're approaching risk limits"""
        warnings = []

        # Check daily loss approaching limit
        if self.risk_manager.daily_pnl < 0:
            loss_percent = abs(self.risk_manager.daily_pnl / self.risk_manager.daily_start_balance) if self.risk_manager.daily_start_balance > 0 else 0
            limit_percent = self.risk_manager.limits.max_daily_loss

            if loss_percent > limit_percent * self.warning_threshold:
                warnings.append({
                    "type": "daily_loss_warning",
                    "message": f"Daily loss at {loss_percent*100:.1f}% (limit: {limit_percent*100:.0f}%)",
                    "current": loss_percent,
                    "limit": limit_percent
                })

        # Check drawdown
        dd_percent = self.risk_manager.current_drawdown / self.risk_manager.peak_balance if self.risk_manager.peak_balance > 0 else 0
        dd_limit = self.risk_manager.limits.max_drawdown

        if dd_percent > dd_limit * self.warning_threshold:
            warnings.append({
                "type": "drawdown_warning",
                "message": f"Drawdown at {dd_percent*100:.1f}% (limit: {dd_limit*100:.0f}%)",
                "current": dd_percent,
                "limit": dd_limit
            })

        # Check position concentration
        if trade_risk.position_percent_of_account > self.risk_manager.limits.max_position_size * self.warning_threshold * 100:
            warnings.append({
                "type": "position_size_warning",
                "message": f"Large position: {trade_risk.position_percent_of_account:.1f}% of account",
                "current": trade_risk.position_percent_of_account / 100,
                "limit": self.risk_manager.limits.max_position_size
            })

        # Check sector concentration
        for sector, value in self.sector_exposure.items():
            sector_percent = value / self.risk_manager.current_balance if self.risk_manager.current_balance > 0 else 0
            limit_percent = self.risk_manager.limits.max_sector_exposure

            if sector_percent > limit_percent * self.warning_threshold:
                warnings.append({
                    "type": "sector_concentration_warning",
                    "message": f"Sector {sector} at {sector_percent*100:.1f}% (limit: {limit_percent*100:.0f}%)",
                    "current": sector_percent,
                    "limit": limit_percent,
                    "sector": sector
                })

        # Publish warnings
        for warning in warnings:
            logger.warning(f"Risk warning: {warning['message']}")

            self.metrics.custom_metrics["risk_alerts_sent"] += 1

            await self.publish(EventType.RISK_ALERT, {
                "symbol": trade_risk.symbol,
                "warning": warning,
                "timestamp": datetime.now().isoformat()
            })

    async def _handle_critical_violations(self, trade_risk: TradeRisk):
        """Handle critical risk violations"""
        for check in trade_risk.checks_failed:
            if check.risk_level == RiskLevel.CRITICAL:
                logger.critical(f"Critical risk violation: {check.message}")

                # Store violation
                self.risk_violations_today.append(check)

                # Trigger circuit breaker for certain violations
                if check.violation_type in [
                    RiskViolationType.MAX_DAILY_LOSS,
                    RiskViolationType.MAX_DRAWDOWN
                ]:
                    await self._trigger_circuit_breaker(check.message)

    # ==================== Metrics & Reporting ====================

    async def _update_exposure_metrics(self):
        """Update portfolio exposure metrics"""
        if not self.active_positions:
            self.metrics.custom_metrics["avg_position_size"] = 0.0
            self.metrics.custom_metrics["total_exposure"] = 0.0
            self.metrics.custom_metrics["daily_pnl"] = self.risk_manager.daily_pnl
            return

        total_exposure = sum(p["position_value"] for p in self.active_positions.values())
        avg_position = total_exposure / len(self.active_positions)

        self.metrics.custom_metrics["avg_position_size"] = avg_position
        self.metrics.custom_metrics["total_exposure"] = total_exposure
        self.metrics.custom_metrics["daily_pnl"] = self.risk_manager.daily_pnl

        # Calculate sector exposure percentages
        sector_exposure_pct = {}
        if self.risk_manager.current_balance > 0:
            for sector, value in self.sector_exposure.items():
                sector_exposure_pct[sector] = round((value / self.risk_manager.current_balance) * 100, 2)

        # Publish portfolio update
        await self.publish(EventType.PORTFOLIO_UPDATED, {
            "open_positions": len(self.active_positions),
            "total_exposure": total_exposure,
            "exposure_percent": (total_exposure / self.risk_manager.current_balance) * 100 if self.risk_manager.current_balance > 0 else 0,
            "daily_pnl": self.risk_manager.daily_pnl,
            "balance": self.risk_manager.current_balance,
            "sector_exposure": sector_exposure_pct,
            "total_sectors": len(self.sector_exposure)
        })

    def _update_approval_rate(self):
        """Update trade approval rate metric"""
        approved = self.metrics.custom_metrics["trades_approved"]
        rejected = self.metrics.custom_metrics["trades_rejected"]
        total = approved + rejected

        if total > 0:
            rate = (approved / total) * 100
            self.metrics.custom_metrics["approval_rate"] = round(rate, 1)

    async def _check_portfolio_risk(self):
        """Comprehensive portfolio risk check"""
        if not self.active_positions:
            return

        metrics = self.risk_manager.get_metrics()

        # Log current state
        logger.debug(f"Portfolio risk: "
                    f"{metrics.open_positions} positions, "
                    f"{metrics.exposure_percent:.1f}% exposure, "
                    f"Daily P&L=${metrics.daily_pnl:.2f}")

        # Check for concentration risk
        if metrics.exposure_percent > self.risk_manager.limits.max_total_exposure * 100:
            await self.publish(EventType.RISK_ALERT, {
                "type": "high_exposure",
                "message": f"Portfolio exposure at {metrics.exposure_percent:.1f}%",
                "exposure_percent": metrics.exposure_percent
            })

    # ==================== Daily Lifecycle ====================

    async def on_market_open(self):
        """Reset daily tracking at market open"""
        logger.info("Market open - resetting daily risk metrics")

        self.risk_manager.reset_daily()
        self.risk_violations_today.clear()
        self._todays_closed_trades.clear()

        # Reload portfolio state from database
        await self.load_portfolio_state()

        # Note: We don't reset sector/correlation tracking as positions may carry overnight
        # Only reset if no positions are open
        if not self.active_positions:
            self.sector_exposure.clear()
            self.position_correlations.clear()
            logger.info("Cleared sector and correlation tracking (no overnight positions)")

        # Reset circuit breaker if it was triggered yesterday
        if self.circuit_breaker_triggered:
            logger.info("Auto-resetting circuit breaker for new trading day")
            await self.reset_circuit_breaker()

    async def on_market_close(self):
        """Generate end-of-day risk report"""
        logger.info("Market close - generating risk report")

        # Refresh prices one last time
        await self.refresh_position_prices()

        report = self.risk_manager.generate_daily_report()

        # Log summary
        logger.info(f"=== Daily Risk Report ===")
        logger.info(f"P&L: ${report.daily_pnl:,.2f} ({report.daily_pnl_percent:.2f}%)")
        logger.info(f"Trades: {report.total_trades} (W:{report.winning_trades} L:{report.losing_trades})")
        logger.info(f"Win Rate: {report.win_rate*100:.1f}%")
        logger.info(f"Max Drawdown: ${report.max_drawdown:,.2f} ({report.max_drawdown_percent:.2f}%)")
        logger.info(f"Violations: {len(self.risk_violations_today)}")
        logger.info(f"========================")

        # Publish daily report
        await self.publish(EventType.DAILY_LIMIT_HIT if report.daily_pnl < 0 else EventType.PNL_UPDATED, {
            "report": {
                "date": report.date.isoformat(),
                "daily_pnl": report.daily_pnl,
                "daily_pnl_percent": report.daily_pnl_percent,
                "total_trades": report.total_trades,
                "win_rate": report.win_rate,
                "max_drawdown": report.max_drawdown,
                "violations": len(self.risk_violations_today)
            }
        })

    # ==================== Public Interface ====================

    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        return self.risk_manager.get_metrics()

    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status with real data"""
        metrics = self.get_risk_metrics()
        pnl_data = self.calculate_daily_pnl()
        limits_status = self.check_portfolio_limits()

        # Calculate sector exposure percentages
        sector_exposure_pct = {}
        if self.risk_manager.current_balance > 0:
            for sector, value in self.sector_exposure.items():
                sector_exposure_pct[sector] = (value / self.risk_manager.current_balance) * 100

        # Build position summaries
        positions_summary = []
        for symbol, pos in self.active_positions.items():
            positions_summary.append({
                "symbol": symbol,
                "direction": pos.get('direction'),
                "shares": pos.get('shares'),
                "entry_price": pos.get('entry_price'),
                "current_price": pos.get('current_price'),
                "unrealized_pnl": pos.get('unrealized_pnl'),
                "stop_loss": pos.get('stop_price'),
                "take_profit": pos.get('target_price')
            })

        return {
            "balance": self.risk_manager.current_balance,
            "daily_pnl": pnl_data['realized_pnl'],
            "unrealized_pnl": pnl_data['unrealized_pnl'],
            "total_pnl": pnl_data['total_pnl'],
            "daily_pnl_percent": pnl_data['daily_pnl_percent'],
            "open_positions": metrics.open_positions,
            "exposure_percent": metrics.exposure_percent,
            "drawdown_percent": metrics.current_drawdown_percent,
            "portfolio_var": self.portfolio_var,
            "trading_halted": self.risk_manager.trading_halted,
            "circuit_breaker": self.circuit_breaker_triggered,
            "positions": positions_summary,
            "sector_exposure": sector_exposure_pct,
            "total_sectors": len(self.sector_exposure),
            "limits_status": limits_status.to_dict(),
            "todays_trades": len(self._todays_closed_trades),
            "win_rate": pnl_data['win_rate']
        }

    def is_trade_allowed(self) -> Tuple[bool, str]:
        """
        Quick check if new trades are allowed
        Returns: (allowed, reason)
        """
        if self.risk_manager.trading_halted:
            return False, self.risk_manager.halt_reason

        if self.circuit_breaker_triggered:
            return False, self.circuit_breaker_reason

        if len(self.active_positions) >= self.risk_manager.limits.max_open_positions:
            return False, "Max positions reached"

        if self.risk_manager.check_daily_loss_limit():
            return False, "Daily loss limit exceeded"

        # Check portfolio limits
        limits_status = self.check_portfolio_limits()
        if not limits_status.is_trading_allowed:
            return False, limits_status.halt_reason

        return True, "OK"

    async def pre_trade_check(self, symbol: str, direction: str, entry_price: float, shares: int, stop_price: float) -> Tuple[bool, str, Dict]:
        """
        Pre-trade risk check for integration with trading workflow.

        Args:
            symbol: Stock ticker
            direction: 'long' or 'short'
            entry_price: Proposed entry price
            shares: Number of shares
            stop_price: Stop loss price

        Returns:
            Tuple of (allowed, reason, details)
        """
        # Basic trading allowed check
        allowed, reason = self.is_trade_allowed()
        if not allowed:
            return False, reason, {}

        # Calculate position metrics
        position_value = entry_price * shares
        risk_amount = abs(entry_price - stop_price) * shares

        # Check position size
        max_position = self.risk_manager.current_balance * self.risk_manager.limits.max_position_size
        if position_value > max_position:
            return False, f"Position value ${position_value:,.0f} exceeds max ${max_position:,.0f}", {}

        # Check risk per trade
        max_risk = self.risk_manager.current_balance * self.risk_manager.limits.max_risk_per_trade
        if risk_amount > max_risk:
            return False, f"Risk ${risk_amount:,.0f} exceeds max ${max_risk:,.0f}", {}

        # Check buying power
        current_exposure = sum(p.get('position_value', 0) for p in self.active_positions.values())
        available = self.risk_manager.current_balance - current_exposure
        if position_value > available:
            return False, f"Insufficient buying power: ${available:,.0f} available", {}

        return True, "OK", {
            "position_value": position_value,
            "risk_amount": risk_amount,
            "risk_percent": (risk_amount / self.risk_manager.current_balance) * 100,
            "available_buying_power": available
        }

    async def post_trade_update(self, symbol: str, trade_data: Dict):
        """
        Update risk tracking after trade execution.

        Args:
            symbol: Stock ticker
            trade_data: Trade execution details
        """
        if trade_data.get('action') == 'open':
            await self.track_position_opened({
                "symbol": symbol,
                "shares": trade_data.get('shares'),
                "entry_price": trade_data.get('entry_price'),
                "stop_price": trade_data.get('stop_loss'),
                "target_price": trade_data.get('take_profit'),
                "direction": trade_data.get('direction', 'long'),
                "sector": trade_data.get('sector')
            })
        elif trade_data.get('action') == 'close':
            await self.track_position_closed({
                "symbol": symbol,
                "pnl": trade_data.get('pnl', 0),
                "exit_price": trade_data.get('exit_price'),
                "is_day_trade": trade_data.get('is_day_trade', False)
            })

        # Refresh portfolio state
        await self._update_exposure_metrics()

    def get_all_position_risks(self) -> Dict[str, PositionRiskMetrics]:
        """Get risk metrics for all open positions"""
        risks = {}
        for symbol in self.active_positions:
            risk = self.get_position_risk(symbol)
            if risk:
                risks[symbol] = risk
        return risks
