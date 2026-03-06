"""
Paper Trading Broker for simulation and testing.

This module provides a simulated broker that tracks virtual positions
and P&L without executing real trades. Useful for:
- Strategy testing without real money
- Development and debugging
- Demo environments
"""

import uuid
import random
import threading
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from brokers.broker_interface import (
    BrokerInterface, Quote, Order, Position, AccountInfo,
    OrderSide, OrderType, OrderStatus,
    BrokerError, OrderError, InsufficientFundsError, PositionError
)


class PaperBroker(BrokerInterface):
    """
    Paper trading broker for simulation and testing.

    Simulates order execution with realistic features:
    - Slippage simulation
    - Commission tracking
    - Position management
    - P&L calculation

    Example:
        broker = PaperBroker(initial_balance=50000)
        broker.connect()
        order = broker.place_order("AAPL", OrderSide.BUY, 10)
        print(f"P&L: ${broker.get_account().daily_pnl:.2f}")
    """

    def __init__(
        self,
        initial_balance: float = 25000.0,
        slippage_pct: float = 0.001,
        commission_per_trade: float = 0.0,
        latency_ms: int = 50,
        realistic_fills: bool = True,
        data_provider=None
    ):
        """
        Initialize paper trading broker.

        Args:
            initial_balance: Starting account balance.
            slippage_pct: Slippage percentage (0.001 = 0.1%).
            commission_per_trade: Commission per trade.
            latency_ms: Simulated latency in milliseconds.
            realistic_fills: If True, apply slippage and partial fills.
            data_provider: Optional DataProvider for cached price lookups.
        """
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.slippage_pct = slippage_pct
        self.commission_per_trade = commission_per_trade
        self.latency_ms = latency_ms
        self.realistic_fills = realistic_fills
        self._data_provider = data_provider

        # Positions: symbol -> {quantity, avg_cost, side}
        self._positions: Dict[str, dict] = {}
        self._lock = threading.Lock()  # Guards _positions, _orders, _pending_orders

        # Orders
        self._orders: Dict[str, Order] = {}
        self._pending_orders: Dict[str, Order] = {}

        # Trade history
        self._order_history: List[Order] = []
        self._trade_log: List[dict] = []

        # State
        self._connected = False
        self._quote_cache: Dict[str, Quote] = {}
        self._realized_pnl = 0.0
        self._realized_pnl_today = 0.0
        self._total_commissions = 0.0

        logger.info(
            f"PaperBroker initialized with ${initial_balance:,.2f} balance"
        )

    # ==================== Connection Methods ====================

    def connect(self) -> bool:
        """Connect to paper trading (always succeeds)."""
        self._connected = True
        logger.info("Paper trading broker connected")
        return True

    def disconnect(self) -> None:
        """Disconnect from paper trading."""
        self._connected = False
        logger.info("Paper trading broker disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    # ==================== Account Methods ====================

    def get_account(self) -> AccountInfo:
        """Get account information."""
        if not self._connected:
            raise BrokerError("Not connected")

        positions = self.get_positions()
        positions_value = sum(p.market_value for p in positions.values())
        total_value = self.cash + positions_value
        # daily_pnl should be: realized_pnl_today + unrealized_pnl
        daily_pnl = self._realized_pnl_today + sum(p.unrealized_pnl for p in positions.values())

        return AccountInfo(
            account_id="PAPER_ACCOUNT",
            buying_power=self.cash,
            cash=self.cash,
            equity=total_value,
            day_trades_remaining=999,  # No PDT restrictions in paper
            pattern_day_trader=False,
            positions_value=positions_value,
            daily_pnl=daily_pnl
        )

    # ==================== Quote Methods ====================

    def _try_data_provider_price(self, symbol: str) -> Optional[Quote]:
        """Try to get a quote from the DataProvider's batch cache."""
        if self._data_provider is None:
            return None
        try:
            cached = self._data_provider.get_cached_price(symbol)
            if cached is not None:
                price = cached['price']
                return Quote(
                    symbol=symbol,
                    bid=price * 0.999,
                    ask=price * 1.001,
                    last=price,
                    volume=cached.get('volume', 0),
                    timestamp=datetime.now(),
                    high=cached.get('high', price),
                    low=cached.get('low', price),
                    open=cached.get('open', price),
                    prev_close=cached.get('prev_close', price)
                )
        except Exception as e:
            logger.debug(f"DataProvider cache miss for {symbol}: {e}")
        return None

    def get_quote(self, symbol: str) -> Quote:
        """Get current quote for a symbol.

        Tries the DataProvider (IBKR) up to 3 times, then falls back to
        the local quote cache.  Never uses yfinance or dummy prices.
        """
        symbol = symbol.upper()

        # Retry DataProvider (IBKR) up to 3 times with short back-off
        for attempt in range(3):
            dp_quote = self._try_data_provider_price(symbol)
            if dp_quote is not None:
                self._quote_cache[symbol] = dp_quote
                return dp_quote
            if attempt < 2:
                import time as _time
                _time.sleep(0.3 * (attempt + 1))

        # Return last-known cached quote if available
        if symbol in self._quote_cache:
            cached = self._quote_cache[symbol]
            logger.debug(f"Using cached quote for {symbol} (IBKR unavailable)")
            return Quote(
                symbol=cached.symbol,
                bid=cached.bid,
                ask=cached.ask,
                last=cached.last,
                volume=cached.volume,
                timestamp=datetime.now(),
                high=cached.high,
                low=cached.low,
                open=cached.open,
                prev_close=cached.prev_close
            )

        # No data at all — return zero-price so callers can guard
        logger.warning(f"No IBKR quote available for {symbol} after 3 retries, no cache")
        return Quote(
            symbol=symbol,
            bid=0.0,
            ask=0.0,
            last=0.0,
            volume=0,
            timestamp=datetime.now()
        )

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple symbols."""
        return {symbol: self.get_quote(symbol) for symbol in symbols}

    # ==================== Position Methods ====================

    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        if not self._connected:
            raise BrokerError("Not connected")

        result = {}
        with self._lock:
            positions_snapshot = dict(self._positions)
        for symbol, pos in positions_snapshot.items():
            if pos['quantity'] == 0:
                continue

            quote = self.get_quote(symbol)
            current_price = quote.last if quote.last > 0 else pos['avg_cost']
            quantity = pos['quantity']
            avg_cost = pos['avg_cost']

            market_value = abs(quantity) * current_price
            cost_basis = abs(quantity) * avg_cost

            if quantity > 0:  # Long position
                unrealized_pnl = (current_price - avg_cost) * quantity
            else:  # Short position
                unrealized_pnl = (avg_cost - current_price) * abs(quantity)

            unrealized_pnl_pct = (
                (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
            )

            result[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=avg_cost,
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                cost_basis=cost_basis
            )

        return result

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        positions = self.get_positions()
        return positions.get(symbol.upper())

    # ==================== Order Methods ====================

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "DAY",
        session: str = "regular"
    ) -> Order:
        """Place an order (paper trading).

        Args:
            symbol: Stock ticker symbol.
            side: Buy or sell direction.
            quantity: Number of shares.
            order_type: MARKET (immediate fill), LIMIT (fill at price or better),
                STOP (trigger at stop_price then fill at market), or
                STOP_LIMIT (trigger at stop_price then fill at limit price or better).
            price: Limit price — required for LIMIT and STOP_LIMIT orders.
            stop_price: Stop trigger price — required for STOP and STOP_LIMIT orders.
            time_in_force: Order duration (DAY, GTC, etc.).
            session: Trading session ('regular', 'premarket', 'afterhours', 'extended').
                Accepted for interface parity; paper broker ignores session restrictions.

        Returns:
            Order object. Market orders are filled immediately. Limit/stop orders
            are stored as pending and filled when check_pending_orders() is called
            and price conditions are met.
        """
        if not self._connected:
            raise BrokerError("Not connected")

        symbol = symbol.upper()

        # Validate order
        is_valid, error_msg = self.validate_order(
            symbol, side, quantity, order_type, price, stop_price
        )
        if not is_valid:
            raise OrderError(error_msg)

        # Create order
        order_id = f"paper_{uuid.uuid4().hex[:8]}"
        quote = self.get_quote(symbol)

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            status=OrderStatus.PENDING,
            time_in_force=time_in_force
        )

        self._orders[order_id] = order

        # Simulate immediate fill for market orders
        if order_type == OrderType.MARKET:
            fill_price = self._calculate_fill_price(quote, side)
            self._execute_fill(order, fill_price)
        elif order_type == OrderType.LIMIT:
            # Check if limit can be filled immediately
            if side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
                if price and price >= quote.ask:
                    fill_price = self._calculate_fill_price(quote, side)
                    self._execute_fill(order, fill_price)
                else:
                    self._pending_orders[order_id] = order
                    order.status = OrderStatus.OPEN
            else:  # SELL or SELL_SHORT
                if price and price <= quote.bid:
                    fill_price = self._calculate_fill_price(quote, side)
                    self._execute_fill(order, fill_price)
                else:
                    self._pending_orders[order_id] = order
                    order.status = OrderStatus.OPEN
        else:
            # Stop and stop-limit orders go to pending
            self._pending_orders[order_id] = order
            order.status = OrderStatus.OPEN

        logger.info(
            f"Paper order: {side.value} {quantity} {symbol} @ "
            f"{order.avg_fill_price or price or 'market'} - {order.status.value}"
        )

        return order

    def _calculate_fill_price(self, quote: Quote, side: OrderSide) -> float:
        """Calculate fill price with slippage."""
        if side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
            base_price = quote.ask if quote.ask > 0 else quote.last
        else:
            base_price = quote.bid if quote.bid > 0 else quote.last

        if self.realistic_fills:
            slippage = base_price * self.slippage_pct * random.uniform(0.5, 1.5)
            if side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
                return base_price + slippage
            else:
                return base_price - slippage

        return base_price

    def _execute_fill(self, order: Order, fill_price: float):
        """Execute order fill and update positions."""
        with self._lock:
            self._execute_fill_locked(order, fill_price)

    def _execute_fill_locked(self, order: Order, fill_price: float):
        """Execute order fill and update positions (must hold self._lock)."""
        symbol = order.symbol
        quantity = order.quantity
        side = order.side
        trade_pnl = 0.0  # Track P&L for this trade

        # Check buying power for buys
        if side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
            cost = fill_price * quantity + self.commission_per_trade
            if cost > self.cash:
                order.status = OrderStatus.REJECTED
                order.error_message = "Insufficient funds"
                logger.warning(f"Order rejected: insufficient funds for {symbol}")
                return

        # Check margin for short selling (150% initial margin)
        if side == OrderSide.SELL_SHORT:
            margin_required = fill_price * quantity * 1.5
            if margin_required > self.cash:
                order.status = OrderStatus.REJECTED
                order.error_message = f"Insufficient margin: need ${margin_required:,.2f}, have ${self.cash:,.2f}"
                logger.warning(f"Order rejected: insufficient margin for short {symbol}")
                return

        # Validate short position exists for covers
        if side == OrderSide.BUY_TO_COVER:
            if symbol not in self._positions or self._positions[symbol].get('side') != 'short':
                order.status = OrderStatus.REJECTED
                order.error_message = "No short position to cover"
                logger.warning(f"Order rejected: no short position in {symbol}")
                return

        # Check position for sells
        if side == OrderSide.SELL:
            if symbol not in self._positions:
                order.status = OrderStatus.REJECTED
                order.error_message = "No position to sell"
                logger.warning(f"Order rejected: no position in {symbol}")
                return

            pos = self._positions[symbol]
            if pos['quantity'] < quantity:
                order.status = OrderStatus.REJECTED
                order.error_message = f"Position size ({pos['quantity']}) < order quantity ({quantity})"
                logger.warning(f"Order rejected: insufficient shares in {symbol}")
                return

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = quantity
        order.avg_fill_price = fill_price
        order.filled_at = datetime.now()

        # Apply commission
        self._total_commissions += self.commission_per_trade

        # Update positions and cash
        if side == OrderSide.BUY:
            cost = fill_price * quantity + self.commission_per_trade
            self.cash -= cost

            if symbol in self._positions:
                pos = self._positions[symbol]
                if pos.get('side') == 'short':
                    # This is a cover (buying to close short position)
                    short_qty = abs(pos['quantity'])
                    covered_qty = min(quantity, short_qty)
                    remaining_buy = quantity - covered_qty

                    # Calculate realized P&L on covered shares
                    pnl = (pos['avg_cost'] - fill_price) * covered_qty
                    self._realized_pnl += pnl
                    self._realized_pnl_today += pnl
                    trade_pnl = pnl

                    pos['quantity'] += covered_qty  # quantity is negative for shorts
                    if pos['quantity'] == 0:
                        del self._positions[symbol]

                    # If buying more than the short, open a long with the remainder
                    if remaining_buy > 0:
                        self._positions[symbol] = {
                            'quantity': remaining_buy,
                            'avg_cost': fill_price,
                            'side': 'long'
                        }
                else:
                    # Adding to existing long position
                    total_shares = pos['quantity'] + quantity
                    pos['avg_cost'] = (
                        (pos['avg_cost'] * pos['quantity'] + fill_price * quantity)
                        / total_shares
                    )
                    pos['quantity'] = total_shares
            else:
                self._positions[symbol] = {
                    'quantity': quantity,
                    'avg_cost': fill_price,
                    'side': 'long'
                }

        elif side == OrderSide.SELL:
            pos = self._positions[symbol]
            proceeds = fill_price * quantity - self.commission_per_trade
            self.cash += proceeds

            # Calculate realized P&L
            pnl = (fill_price - pos['avg_cost']) * quantity
            self._realized_pnl += pnl
            self._realized_pnl_today += pnl
            trade_pnl = pnl

            pos['quantity'] -= quantity
            if pos['quantity'] == 0:
                del self._positions[symbol]

        elif side == OrderSide.SELL_SHORT:
            # Short selling — reject if there is a conflicting long position
            if symbol in self._positions and self._positions[symbol].get('side') == 'long':
                order.status = OrderStatus.REJECTED
                order.error_message = (
                    f"Cannot sell short {symbol}: existing long position "
                    f"({self._positions[symbol]['quantity']} shares). "
                    f"Close the long position first."
                )
                logger.warning(
                    f"Order rejected: SELL_SHORT on {symbol} conflicts with "
                    f"existing long position of {self._positions[symbol]['quantity']} shares"
                )
                return

            proceeds = fill_price * quantity - self.commission_per_trade
            self.cash += proceeds

            if symbol in self._positions:
                pos = self._positions[symbol]
                old_qty = abs(pos['quantity'])
                new_qty = quantity
                # Recalculate weighted average cost
                pos['avg_cost'] = (
                    (pos['avg_cost'] * old_qty + fill_price * new_qty)
                    / (old_qty + new_qty)
                )
                pos['quantity'] = pos['quantity'] - quantity
                if pos['quantity'] == 0:
                    del self._positions[symbol]
            else:
                self._positions[symbol] = {
                    'quantity': -quantity,
                    'avg_cost': fill_price,
                    'side': 'short'
                }

        elif side == OrderSide.BUY_TO_COVER:
            # Cover short position
            cost = fill_price * quantity + self.commission_per_trade
            self.cash -= cost

            pos = self._positions[symbol]
            # Calculate realized P&L on short cover
            pnl = (pos['avg_cost'] - fill_price) * quantity
            self._realized_pnl += pnl
            self._realized_pnl_today += pnl
            trade_pnl = pnl

            pos['quantity'] += quantity
            if pos['quantity'] == 0:
                del self._positions[symbol]

        # Record trade
        self._order_history.append(order)
        trade_entry = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side.value,
            'quantity': quantity,
            'price': fill_price,
            'order_id': order.order_id,
            'commission': self.commission_per_trade
        }
        if trade_pnl != 0:
            trade_entry['pnl'] = trade_pnl
        self._trade_log.append(trade_entry)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not self._connected:
            raise BrokerError("Not connected")

        if order_id in self._pending_orders:
            order = self._pending_orders[order_id]
            if order.status in (OrderStatus.PENDING, OrderStatus.OPEN):
                order.status = OrderStatus.CANCELLED
                del self._pending_orders[order_id]
                logger.info(f"Order cancelled: {order_id}")
                return True

        if order_id in self._orders:
            order = self._orders[order_id]
            if order.status in (OrderStatus.PENDING, OrderStatus.OPEN):
                order.status = OrderStatus.CANCELLED
                logger.info(f"Order cancelled: {order_id}")
                return True

        return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status."""
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        return [
            o for o in self._pending_orders.values()
            if o.status in (OrderStatus.PENDING, OrderStatus.OPEN)
        ]

    # ==================== Paper Trading Specific ====================

    def reset(self):
        """Reset paper account to initial state."""
        self.cash = self.initial_balance
        self._positions.clear()
        self._orders.clear()
        self._pending_orders.clear()
        self._order_history.clear()
        self._trade_log.clear()
        self._realized_pnl = 0.0
        self._realized_pnl_today = 0.0
        self._total_commissions = 0.0
        logger.info("Paper trading account reset")

    def process_pending_orders(self):
        """
        Process pending limit/stop orders against current prices.
        Call this periodically to simulate order fills.

        Fill logic by order type:
        - LIMIT buy:  fills when ask <= limit_price (price at or below limit).
        - LIMIT sell: fills when bid >= limit_price (price at or above limit).
        - STOP sell/short: triggers when last <= stop_price; fills at market.
        - STOP buy/cover:  triggers when last >= stop_price; fills at market.
        - STOP_LIMIT sell: triggers when last <= stop_price; fills only if
          bid >= limit_price (limit or better), otherwise stays pending.
        - STOP_LIMIT buy:  triggers when last >= stop_price; fills only if
          ask <= limit_price (limit or better), otherwise stays pending.
        """
        orders_to_fill = []

        for order_id, order in list(self._pending_orders.items()):
            quote = self.get_quote(order.symbol)

            if order.order_type == OrderType.LIMIT:
                if order.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
                    # Buy limit: fill when market ask is at or below our limit price
                    if order.price and quote.ask <= order.price:
                        orders_to_fill.append((order, quote.ask))
                else:
                    # Sell limit: fill when market bid is at or above our limit price
                    if order.price and quote.bid >= order.price:
                        orders_to_fill.append((order, quote.bid))

            elif order.order_type == OrderType.STOP:
                if order.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
                    # Sell stop: trigger when price falls to or below stop_price
                    if order.stop_price and quote.last <= order.stop_price:
                        orders_to_fill.append(
                            (order, self._calculate_fill_price(quote, order.side))
                        )
                else:
                    # Buy stop: trigger when price rises to or above stop_price
                    if order.stop_price and quote.last >= order.stop_price:
                        orders_to_fill.append(
                            (order, self._calculate_fill_price(quote, order.side))
                        )

            elif order.order_type == OrderType.STOP_LIMIT:
                # Mark the order as stop-triggered using a transient attribute so
                # we can move to the limit-fill phase on subsequent ticks.
                if not getattr(order, '_stop_triggered', False):
                    # Phase 1: wait for the stop trigger price to be hit.
                    if order.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
                        triggered = order.stop_price and quote.last <= order.stop_price
                    else:
                        triggered = order.stop_price and quote.last >= order.stop_price

                    if triggered:
                        object.__setattr__(order, '_stop_triggered', True) \
                            if hasattr(order, '__slots__') \
                            else setattr(order, '_stop_triggered', True)
                        logger.info(
                            f"Stop-limit order {order.order_id} stop triggered at "
                            f"{quote.last} (stop={order.stop_price})"
                        )

                # Phase 2 (stop already triggered): attempt limit fill.
                if getattr(order, '_stop_triggered', False):
                    if order.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
                        # Sell stop-limit: fill if bid >= limit_price (limit or better)
                        if order.price and quote.bid >= order.price:
                            orders_to_fill.append((order, quote.bid))
                    else:
                        # Buy stop-limit: fill if ask <= limit_price (limit or better)
                        if order.price and quote.ask <= order.price:
                            orders_to_fill.append((order, quote.ask))

        for order, fill_price in orders_to_fill:
            del self._pending_orders[order.order_id]
            self._execute_fill(order, fill_price)
            logger.info(f"Pending order filled: {order.order_id} @ {fill_price}")

    def check_pending_orders(self, current_prices: Dict[str, float]) -> None:
        """
        Process pending orders using a caller-supplied price map.

        This is a convenience wrapper around :meth:`process_pending_orders` for
        callers that already have current prices and want to avoid extra quote
        fetches.  Prices are written into the quote cache before the standard
        pending-order sweep runs.

        Args:
            current_prices: Mapping of uppercase symbol -> current last price.
                For each symbol a synthetic Quote is synthesised with bid = price
                * 0.999 and ask = price * 1.001 so limit/stop comparisons remain
                consistent with the standard quote format.
        """
        for symbol, last_price in current_prices.items():
            symbol = symbol.upper()
            if last_price and last_price > 0:
                synthetic = Quote(
                    symbol=symbol,
                    bid=round(last_price * 0.999, 4),
                    ask=round(last_price * 1.001, 4),
                    last=float(last_price),
                    volume=0,
                    timestamp=datetime.now()
                )
                self._quote_cache[symbol] = synthetic

        self.process_pending_orders()

    def get_trade_history(self) -> List[dict]:
        """Get trade history log."""
        return self._trade_log.copy()

    def get_performance_summary(self) -> dict:
        """Get paper trading performance summary."""
        account = self.get_account()
        positions = self.get_positions()

        total_trades = len(self._trade_log)
        winning_trades = sum(
            1 for t in self._trade_log
            if t.get('pnl', 0) > 0
        )

        return {
            'initial_balance': self.initial_balance,
            'current_balance': account.equity,
            'cash': self.cash,
            'positions_value': account.positions_value,
            'realized_pnl': self._realized_pnl,
            'unrealized_pnl': sum(p.unrealized_pnl for p in positions.values()),
            'total_pnl': account.daily_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_commissions': self._total_commissions,
            'return_pct': ((account.equity - self.initial_balance) / self.initial_balance * 100)
        }

    @property
    def positions(self) -> Dict[str, dict]:
        """Raw position data (for backward compatibility)."""
        return self._positions

    @property
    def orders(self) -> Dict[str, Order]:
        """All orders (for backward compatibility)."""
        return self._orders

    @property
    def order_history(self) -> List[Order]:
        """Order history (for backward compatibility)."""
        return self._order_history

    @property
    def trade_log(self) -> List[dict]:
        """Trade log (for backward compatibility)."""
        return self._trade_log

    # ==================== Advanced Order Methods ====================

    def place_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_price: Optional[float],
        take_profit_price: float,
        stop_loss_price: float,
        entry_type: OrderType = OrderType.LIMIT
    ) -> tuple:
        """
        Place a bracket order (entry + profit target + stop loss).

        Simulates bracket order behavior by tracking linked orders.

        Args:
            symbol: Stock ticker symbol
            side: OrderSide.BUY for long, OrderSide.SELL_SHORT for short
            quantity: Number of shares
            entry_price: Entry limit price (None for market order)
            take_profit_price: Profit target price
            stop_loss_price: Stop loss trigger price
            entry_type: Entry order type (MARKET or LIMIT)

        Returns:
            Tuple of (entry_order, take_profit_order, stop_loss_order)
        """
        if not self._connected:
            raise BrokerError("Not connected")

        symbol = symbol.upper()
        bracket_id = f"bracket_{uuid.uuid4().hex[:8]}"

        # Determine exit side
        if side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
            exit_side = OrderSide.SELL
        else:
            exit_side = OrderSide.BUY_TO_COVER

        # Create entry order
        entry_order = self.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=entry_type,
            price=entry_price
        )

        # Create exit orders (linked to bracket)
        tp_order_id = f"{bracket_id}_tp"
        sl_order_id = f"{bracket_id}_sl"

        take_profit_order = Order(
            order_id=tp_order_id,
            symbol=symbol,
            side=exit_side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price=take_profit_price,
            status=OrderStatus.PENDING
        )

        stop_loss_order = Order(
            order_id=sl_order_id,
            symbol=symbol,
            side=exit_side,
            quantity=quantity,
            order_type=OrderType.STOP,
            stop_price=stop_loss_price,
            status=OrderStatus.PENDING
        )

        # If entry is filled, activate exit orders
        if entry_order.status == OrderStatus.FILLED:
            take_profit_order.status = OrderStatus.OPEN
            stop_loss_order.status = OrderStatus.OPEN
            self._pending_orders[tp_order_id] = take_profit_order
            self._pending_orders[sl_order_id] = stop_loss_order

            # Store OCO link
            if not hasattr(self, '_oco_groups'):
                self._oco_groups = {}
            self._oco_groups[tp_order_id] = [sl_order_id]
            self._oco_groups[sl_order_id] = [tp_order_id]

        self._orders[tp_order_id] = take_profit_order
        self._orders[sl_order_id] = stop_loss_order

        # Store bracket tracking info
        if not hasattr(self, '_bracket_orders'):
            self._bracket_orders = {}
        self._bracket_orders[bracket_id] = {
            'entry_order_id': entry_order.order_id,
            'tp_order_id': tp_order_id,
            'sl_order_id': sl_order_id,
            'symbol': symbol,
            'quantity': quantity
        }

        logger.info(
            f"Paper bracket order: {symbol} - Entry: {entry_price or 'MKT'}, "
            f"TP: {take_profit_price}, SL: {stop_loss_price}"
        )

        return entry_order, take_profit_order, stop_loss_order

    def place_trailing_stop(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        trail_amount: Optional[float] = None,
        trail_percent: Optional[float] = None,
        activation_price: Optional[float] = None,
        time_in_force: str = "GTC"
    ) -> Order:
        """
        Place a trailing stop order.

        Simulates trailing stop by tracking reference price and adjusting stop.

        Args:
            symbol: Stock ticker symbol
            side: SELL for long exit, BUY_TO_COVER for short exit
            quantity: Number of shares
            trail_amount: Trail by fixed dollar amount
            trail_percent: Trail by percentage
            activation_price: Price at which trailing begins (optional)
            time_in_force: Order duration

        Returns:
            Order object with trailing stop details
        """
        if not self._connected:
            raise BrokerError("Not connected")

        if trail_amount is None and trail_percent is None:
            raise OrderError("Either trail_amount or trail_percent must be specified")

        symbol = symbol.upper()
        order_id = f"paper_trail_{uuid.uuid4().hex[:8]}"

        # Get current quote
        quote = self.get_quote(symbol)
        current_price = quote.last

        # Calculate initial stop price
        if trail_percent is not None:
            trail_value = current_price * (trail_percent / 100.0)
        else:
            trail_value = trail_amount

        if side in (OrderSide.SELL, OrderSide.SELL_SHORT):
            # Long exit - stop below price
            initial_stop = current_price - trail_value
        else:
            # Short exit - stop above price
            initial_stop = current_price + trail_value

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.TRAILING_STOP,
            stop_price=initial_stop,
            status=OrderStatus.OPEN,
            time_in_force=time_in_force
        )

        self._orders[order_id] = order
        self._pending_orders[order_id] = order

        # Store trailing stop tracking info
        if not hasattr(self, '_trailing_stops'):
            self._trailing_stops = {}

        self._trailing_stops[order_id] = {
            'trail_amount': trail_amount,
            'trail_percent': trail_percent,
            'activation_price': activation_price,
            'reference_price': current_price,
            'current_stop': initial_stop,
            'is_activated': activation_price is None,
            'is_long_exit': side in (OrderSide.SELL, OrderSide.SELL_SHORT)
        }

        logger.info(
            f"Paper trailing stop: {symbol} - Trail: "
            f"${trail_amount or 'N/A'} / {trail_percent or 'N/A'}%, "
            f"Initial stop: {initial_stop:.2f}"
        )

        return order

    def place_oco_order(
        self,
        symbol: str,
        orders: List[Dict]
    ) -> List[Order]:
        """
        Place OCO (one-cancels-other) orders.

        Args:
            symbol: Stock ticker symbol
            orders: List of order specifications

        Returns:
            List of Order objects linked as OCO
        """
        if not self._connected:
            raise BrokerError("Not connected")

        if len(orders) < 2:
            raise OrderError("OCO requires at least 2 orders")

        symbol = symbol.upper()
        oco_group_id = f"oco_{uuid.uuid4().hex[:8]}"

        result_orders = []
        order_ids = []

        for spec in orders:
            order = self.place_order(
                symbol=symbol,
                side=spec.get("side", OrderSide.SELL),
                quantity=spec.get("quantity", 1),
                order_type=spec.get("order_type", OrderType.MARKET),
                price=spec.get("price"),
                stop_price=spec.get("stop_price"),
                time_in_force=spec.get("time_in_force", "GTC")
            )
            result_orders.append(order)
            order_ids.append(order.order_id)

        # Store OCO links
        if not hasattr(self, '_oco_groups'):
            self._oco_groups = {}

        for i, order_id in enumerate(order_ids):
            # Link each order to all others in the group
            other_ids = [oid for j, oid in enumerate(order_ids) if j != i]
            self._oco_groups[order_id] = other_ids

        logger.info(f"Paper OCO order: {symbol} with {len(result_orders)} orders")

        return result_orders

    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        quantity: Optional[int] = None
    ) -> bool:
        """
        Modify an existing order.

        Args:
            order_id: Order ID to modify
            price: New limit price (optional)
            stop_price: New stop trigger price (optional)
            quantity: New quantity (optional)

        Returns:
            True if modification successful
        """
        if not self._connected:
            raise BrokerError("Not connected")

        if order_id not in self._orders:
            logger.warning(f"Order {order_id} not found for modification")
            return False

        order = self._orders[order_id]

        if not order.is_active:
            logger.warning(f"Order {order_id} is not active (status: {order.status.value})")
            return False

        if price is not None:
            order.price = price

        if stop_price is not None:
            order.stop_price = stop_price

        if quantity is not None:
            order.quantity = quantity

        # Update trailing stop tracking if applicable
        if hasattr(self, '_trailing_stops') and order_id in self._trailing_stops:
            if stop_price is not None:
                self._trailing_stops[order_id]['current_stop'] = stop_price

        logger.info(f"Paper order {order_id} modified: price={price}, stop={stop_price}, qty={quantity}")
        return True

    def process_advanced_orders(self):
        """
        Process advanced orders (trailing stops, OCO).
        Call this periodically to update trailing stops and check OCO fills.
        """
        # Process trailing stops
        if hasattr(self, '_trailing_stops'):
            for order_id, trail_info in list(self._trailing_stops.items()):
                if order_id not in self._pending_orders:
                    continue

                order = self._pending_orders[order_id]
                quote = self.get_quote(order.symbol)
                current_price = quote.last

                # Check activation
                if not trail_info['is_activated']:
                    activation = trail_info['activation_price']
                    if trail_info['is_long_exit']:
                        if current_price >= activation:
                            trail_info['is_activated'] = True
                            trail_info['reference_price'] = current_price
                            logger.info(f"Trailing stop activated at {current_price}")
                    else:
                        if current_price <= activation:
                            trail_info['is_activated'] = True
                            trail_info['reference_price'] = current_price
                            logger.info(f"Trailing stop activated at {current_price}")

                if not trail_info['is_activated']:
                    continue

                # Update trail
                if trail_info['is_long_exit']:
                    # Long exit - track highest price, stop trails below
                    if current_price > trail_info['reference_price']:
                        trail_info['reference_price'] = current_price

                        # Calculate new stop
                        if trail_info['trail_percent'] is not None:
                            trail_value = current_price * (trail_info['trail_percent'] / 100.0)
                        else:
                            trail_value = trail_info['trail_amount']

                        new_stop = current_price - trail_value

                        # Only move stop up, never down
                        if new_stop > trail_info['current_stop']:
                            trail_info['current_stop'] = new_stop
                            order.stop_price = new_stop
                            logger.debug(f"Trail updated: stop={new_stop:.2f}")

                    # Check if triggered
                    if current_price <= trail_info['current_stop']:
                        del self._pending_orders[order_id]
                        fill_price = self._calculate_fill_price(quote, order.side)
                        self._execute_fill(order, fill_price)
                        del self._trailing_stops[order_id]
                        logger.info(f"Trailing stop triggered at {current_price}")
                else:
                    # Short exit - track lowest price, stop trails above
                    if current_price < trail_info['reference_price']:
                        trail_info['reference_price'] = current_price

                        if trail_info['trail_percent'] is not None:
                            trail_value = current_price * (trail_info['trail_percent'] / 100.0)
                        else:
                            trail_value = trail_info['trail_amount']

                        new_stop = current_price + trail_value

                        # Only move stop down, never up
                        if new_stop < trail_info['current_stop']:
                            trail_info['current_stop'] = new_stop
                            order.stop_price = new_stop
                            logger.debug(f"Trail updated: stop={new_stop:.2f}")

                    # Check if triggered
                    if current_price >= trail_info['current_stop']:
                        del self._pending_orders[order_id]
                        fill_price = self._calculate_fill_price(quote, order.side)
                        self._execute_fill(order, fill_price)
                        del self._trailing_stops[order_id]
                        logger.info(f"Trailing stop triggered at {current_price}")

        # Process OCO orders - check for fills and cancel others
        if hasattr(self, '_oco_groups'):
            filled_orders = []
            for order_id, linked_ids in list(self._oco_groups.items()):
                if order_id in self._orders:
                    order = self._orders[order_id]
                    if order.status == OrderStatus.FILLED:
                        filled_orders.append(order_id)

            # Cancel linked orders for filled ones
            for filled_id in filled_orders:
                if filled_id in self._oco_groups:
                    linked_ids = self._oco_groups[filled_id]
                    if isinstance(linked_ids, str):
                        linked_ids = [linked_ids]

                    for linked_id in linked_ids:
                        if linked_id in self._pending_orders:
                            self.cancel_order(linked_id)
                            logger.info(f"OCO: Cancelled {linked_id} due to fill of {filled_id}")

                    del self._oco_groups[filled_id]
