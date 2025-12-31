"""
Paper trading broker for simulation and testing.
"""

import uuid
import random
from datetime import datetime
from typing import Dict, List, Optional

import yfinance as yf

from ..base import (
    AbstractBroker, Quote, Order, Position, AccountInfo,
    OrderSide, OrderType, OrderStatus
)


class PaperBroker(AbstractBroker):
    """Paper trading broker for simulation."""

    def __init__(
        self,
        initial_balance: float = 25000.0,
        slippage_pct: float = 0.001,
        latency_ms: int = 50
    ):
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.slippage_pct = slippage_pct
        self.latency_ms = latency_ms

        self.positions: Dict[str, dict] = {}
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.trade_log: List[dict] = []

        self.connected = False
        self._quote_cache: Dict[str, Quote] = {}

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def get_account(self) -> AccountInfo:
        positions = self.get_positions()
        total_position_value = sum(p.market_value for p in positions.values())
        total_value = self.cash + total_position_value
        daily_pnl = total_value - self.initial_balance

        return AccountInfo(
            account_id="PAPER_ACCOUNT",
            cash_available=self.cash,
            buying_power=self.cash,
            total_value=total_value,
            positions=positions,
            daily_pnl=daily_pnl
        )

    def get_quote(self, symbol: str) -> Quote:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            last = info.get('regularMarketPrice', info.get('previousClose', 0))
            bid = info.get('bid', last * 0.999)
            ask = info.get('ask', last * 1.001)
            volume = info.get('regularMarketVolume', 0)

            quote = Quote(
                symbol=symbol,
                bid=bid or last * 0.999,
                ask=ask or last * 1.001,
                last=last,
                volume=volume,
                timestamp=datetime.now()
            )
            self._quote_cache[symbol] = quote
            return quote
        except Exception:
            # Return cached quote or dummy
            if symbol in self._quote_cache:
                return self._quote_cache[symbol]
            return Quote(symbol=symbol, bid=100, ask=100.10, last=100.05, volume=0, timestamp=datetime.now())

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        return {symbol: self.get_quote(symbol) for symbol in symbols}

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply simulated slippage to price."""
        slippage = price * self.slippage_pct * random.uniform(0.5, 1.5)
        if side in (OrderSide.BUY, OrderSide.COVER):
            return price + slippage
        else:
            return price - slippage

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        order_id = str(uuid.uuid4())[:8]
        quote = self.get_quote(symbol)

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            status=OrderStatus.PENDING
        )
        self.orders[order_id] = order

        # Simulate immediate fill for market orders
        if order_type == OrderType.MARKET:
            fill_price = quote.ask if side in (OrderSide.BUY, OrderSide.COVER) else quote.bid
            fill_price = self._apply_slippage(fill_price, side)
            self._fill_order(order, fill_price)

        return order

    def _fill_order(self, order: Order, fill_price: float):
        """Execute order fill."""
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.filled_at = datetime.now()

        # Update positions and cash
        if order.side == OrderSide.BUY:
            cost = fill_price * order.quantity
            if cost > self.cash:
                order.status = OrderStatus.REJECTED
                return

            self.cash -= cost
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                total_shares = pos['quantity'] + order.quantity
                pos['avg_cost'] = (pos['avg_cost'] * pos['quantity'] + fill_price * order.quantity) / total_shares
                pos['quantity'] = total_shares
            else:
                self.positions[order.symbol] = {
                    'quantity': order.quantity,
                    'avg_cost': fill_price,
                    'side': 'long'
                }

        elif order.side == OrderSide.SELL:
            if order.symbol not in self.positions:
                order.status = OrderStatus.REJECTED
                return

            pos = self.positions[order.symbol]
            if pos['quantity'] < order.quantity:
                order.status = OrderStatus.REJECTED
                return

            proceeds = fill_price * order.quantity
            self.cash += proceeds
            pos['quantity'] -= order.quantity

            if pos['quantity'] == 0:
                del self.positions[order.symbol]

        self.order_history.append(order)
        self.trade_log.append({
            'timestamp': datetime.now(),
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': fill_price,
            'order_id': order.order_id
        })

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)

    def get_positions(self) -> Dict[str, Position]:
        result = {}
        for symbol, pos in self.positions.items():
            quote = self.get_quote(symbol)
            current_price = quote.last
            market_value = pos['quantity'] * current_price
            cost_basis = pos['quantity'] * pos['avg_cost']
            unrealized_pnl = market_value - cost_basis
            pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0

            result[symbol] = Position(
                symbol=symbol,
                quantity=pos['quantity'],
                avg_cost=pos['avg_cost'],
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=pnl_pct
            )
        return result

    def get_position(self, symbol: str) -> Optional[Position]:
        positions = self.get_positions()
        return positions.get(symbol)

    def reset(self):
        """Reset paper account to initial state."""
        self.cash = self.initial_balance
        self.positions.clear()
        self.orders.clear()
        self.order_history.clear()
        self.trade_log.clear()
