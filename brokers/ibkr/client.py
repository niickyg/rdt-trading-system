"""
Interactive Brokers Client
Implements AbstractBroker interface using ib_insync
"""

from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

try:
    from ib_insync import IB, Stock, Order as IBOrder, MarketOrder, LimitOrder, StopOrder
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logger.warning("ib_insync not installed. Install with: pip install ib_insync")

from brokers.base import (
    AbstractBroker, Order, Position, Quote, AccountInfo,
    OrderSide, OrderType, OrderStatus
)


class IBKRClient(AbstractBroker):
    """
    Interactive Brokers client using ib_insync library

    Supports:
    - Paper trading (port 7497)
    - Live trading (port 7496)
    - Stocks and options
    - Multiple order types
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,  # 7497 = paper, 7496 = live
        client_id: int = 1,
        paper_trading: bool = True
    ):
        """
        Initialize IBKR client

        Args:
            host: TWS/Gateway host (default: localhost)
            port: TWS/Gateway port (7497 for paper, 7496 for live)
            client_id: Unique client ID
            paper_trading: Whether using paper account
        """
        if not IB_AVAILABLE:
            raise ImportError(
                "ib_insync is required for IBKR integration. "
                "Install with: pip install ib_insync"
            )

        self.host = host
        self.port = port
        self.client_id = client_id
        self.paper_trading = paper_trading
        self.ib = IB()
        self._connected = False

        logger.info(
            f"Initialized IBKR client: "
            f"{'Paper' if paper_trading else 'Live'} trading on port {port}"
        )

    def connect(self) -> bool:
        """Connect to TWS/Gateway"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self._connected = True
            logger.info(f"Connected to IBKR on {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from TWS/Gateway"""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    def get_account(self) -> AccountInfo:
        """Get account information"""
        if not self._connected:
            raise ConnectionError("Not connected to IBKR")

        # Get account summary
        account_values = self.ib.accountSummary()

        # Parse account values
        cash_available = 0.0
        buying_power = 0.0
        total_value = 0.0

        for value in account_values:
            if value.tag == "AvailableFunds":
                cash_available = float(value.value)
            elif value.tag == "BuyingPower":
                buying_power = float(value.value)
            elif value.tag == "NetLiquidation":
                total_value = float(value.value)

        # Get positions
        positions = self.get_positions()

        # Calculate daily P&L
        pnl_values = [v for v in account_values if v.tag == "DailyPnL"]
        daily_pnl = float(pnl_values[0].value) if pnl_values else 0.0

        account_id = account_values[0].account if account_values else "IBKR"

        return AccountInfo(
            account_id=account_id,
            cash_available=cash_available,
            buying_power=buying_power,
            total_value=total_value,
            positions=positions,
            daily_pnl=daily_pnl
        )

    def get_quote(self, symbol: str) -> Quote:
        """Get current quote for a symbol"""
        if not self._connected:
            raise ConnectionError("Not connected to IBKR")

        # Create stock contract
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)

        # Request market data
        ticker = self.ib.reqMktData(contract, snapshot=True)
        self.ib.sleep(1)  # Wait for data

        return Quote(
            symbol=symbol,
            bid=ticker.bid if ticker.bid and ticker.bid > 0 else ticker.last,
            ask=ticker.ask if ticker.ask and ticker.ask > 0 else ticker.last,
            last=ticker.last if ticker.last else 0.0,
            volume=ticker.volume if ticker.volume else 0,
            timestamp=datetime.now()
        )

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple symbols"""
        quotes = {}
        for symbol in symbols:
            try:
                quotes[symbol] = self.get_quote(symbol)
            except Exception as e:
                logger.warning(f"Failed to get quote for {symbol}: {e}")
        return quotes

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """Place an order"""
        if not self._connected:
            raise ConnectionError("Not connected to IBKR")

        # Create contract
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)

        # Map order side
        action = "BUY" if side in [OrderSide.BUY, OrderSide.COVER] else "SELL"

        # Create order based on type
        if order_type == OrderType.MARKET:
            ib_order = MarketOrder(action, quantity)
        elif order_type == OrderType.LIMIT:
            if price is None:
                raise ValueError("Limit order requires price")
            ib_order = LimitOrder(action, quantity, price)
        elif order_type == OrderType.STOP:
            if stop_price is None:
                raise ValueError("Stop order requires stop_price")
            ib_order = StopOrder(action, quantity, stop_price)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

        # Place order
        trade = self.ib.placeOrder(contract, ib_order)

        # Wait for order to be submitted
        self.ib.sleep(0.5)

        # Create our Order object
        order = Order(
            order_id=str(trade.order.orderId),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            status=self._map_order_status(trade.orderStatus.status),
            filled_quantity=trade.orderStatus.filled,
            avg_fill_price=trade.orderStatus.avgFillPrice if trade.orderStatus.avgFillPrice > 0 else None,
            created_at=datetime.now()
        )

        logger.info(
            f"Placed {order_type.value} {side.value} order: "
            f"{quantity} {symbol} @ {price or 'market'}"
        )

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self._connected:
            raise ConnectionError("Not connected to IBKR")

        try:
            # Find the trade
            trades = self.ib.trades()
            for trade in trades:
                if str(trade.order.orderId) == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Cancelled order {order_id}")
                    return True

            logger.warning(f"Order {order_id} not found")
            return False
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order status"""
        if not self._connected:
            raise ConnectionError("Not connected to IBKR")

        trades = self.ib.trades()
        for trade in trades:
            if str(trade.order.orderId) == order_id:
                return Order(
                    order_id=order_id,
                    symbol=trade.contract.symbol,
                    side=OrderSide.BUY if trade.order.action == "BUY" else OrderSide.SELL,
                    order_type=self._map_order_type(trade.order.orderType),
                    quantity=int(trade.order.totalQuantity),
                    price=trade.order.lmtPrice if hasattr(trade.order, 'lmtPrice') else None,
                    stop_price=trade.order.auxPrice if hasattr(trade.order, 'auxPrice') else None,
                    status=self._map_order_status(trade.orderStatus.status),
                    filled_quantity=trade.orderStatus.filled,
                    avg_fill_price=trade.orderStatus.avgFillPrice if trade.orderStatus.avgFillPrice > 0 else None
                )

        return None

    def get_positions(self) -> Dict[str, Position]:
        """Get all open positions"""
        if not self._connected:
            raise ConnectionError("Not connected to IBKR")

        positions = {}
        ib_positions = self.ib.positions()

        for pos in ib_positions:
            symbol = pos.contract.symbol
            quantity = int(pos.position)
            avg_cost = pos.avgCost

            # Get current price
            ticker = self.ib.reqMktData(pos.contract, snapshot=True)
            self.ib.sleep(0.5)
            current_price = ticker.last if ticker.last else avg_cost

            market_value = current_price * abs(quantity)
            unrealized_pnl = (current_price - avg_cost) * quantity
            unrealized_pnl_pct = (unrealized_pnl / (avg_cost * abs(quantity))) * 100 if avg_cost > 0 else 0

            positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=avg_cost,
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct
            )

        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol"""
        positions = self.get_positions()
        return positions.get(symbol)

    def _map_order_status(self, ib_status: str) -> OrderStatus:
        """Map IBKR order status to our OrderStatus"""
        status_map = {
            "Submitted": OrderStatus.PENDING,
            "PreSubmitted": OrderStatus.PENDING,
            "Filled": OrderStatus.FILLED,
            "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "PendingCancel": OrderStatus.PENDING,
            "Inactive": OrderStatus.REJECTED,
            "ApiCancelled": OrderStatus.CANCELLED
        }
        return status_map.get(ib_status, OrderStatus.PENDING)

    def _map_order_type(self, ib_order_type: str) -> OrderType:
        """Map IBKR order type to our OrderType"""
        type_map = {
            "MKT": OrderType.MARKET,
            "LMT": OrderType.LIMIT,
            "STP": OrderType.STOP,
            "STP LMT": OrderType.STOP_LIMIT
        }
        return type_map.get(ib_order_type, OrderType.MARKET)
