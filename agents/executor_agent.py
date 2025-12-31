"""
Executor Agent
Executes trades through the broker
"""

from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from agents.base import BaseAgent
from agents.events import Event, EventType
from brokers.base import AbstractBroker, OrderSide, OrderType


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

    async def initialize(self):
        """Initialize executor"""
        logger.info(f"Executor initialized (auto_execute={self.auto_execute})")
        self.metrics.custom_metrics["orders_placed"] = 0
        self.metrics.custom_metrics["fill_rate"] = 0

    async def cleanup(self):
        """Cleanup executor"""
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
            # Clean up any related orders
            symbol = event.data.get("symbol")
            await self.cancel_orders_for_symbol(symbol)

    async def handle_valid_setup(self, setup: Dict):
        """Handle a valid setup from analyzer"""
        symbol = setup.get("symbol")

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

    async def execute_trade(self, setup: Dict):
        """Execute a trade setup"""
        symbol = setup.get("symbol")
        direction = setup.get("direction")
        shares = setup.get("shares")
        entry_price = setup.get("entry_price")
        stop_price = setup.get("stop_price")
        target_price = setup.get("target_price")

        logger.info(f"Executing: {direction.upper()} {shares} {symbol} @ ${entry_price:.2f}")

        try:
            # Determine order side
            if direction == "long":
                entry_side = OrderSide.BUY
                exit_side = OrderSide.SELL
            else:
                entry_side = OrderSide.SELL_SHORT
                exit_side = OrderSide.BUY_TO_COVER

            # Place entry order (market)
            entry_order = self.broker.place_order(
                symbol=symbol,
                side=entry_side,
                quantity=shares,
                order_type=OrderType.MARKET
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

            # Place stop-loss order
            stop_order = self.broker.place_order(
                symbol=symbol,
                side=exit_side,
                quantity=shares,
                order_type=OrderType.STOP,
                stop_price=stop_price
            )

            if stop_order:
                self.active_orders[stop_order.order_id] = {
                    "order": stop_order,
                    "type": "stop_loss",
                    "setup": setup
                }
                logger.info(f"Stop-loss placed: {symbol} @ ${stop_price:.2f}")

            # Place take-profit order (optional - could use trailing stop instead)
            target_order = self.broker.place_order(
                symbol=symbol,
                side=exit_side,
                quantity=shares,
                order_type=OrderType.LIMIT,
                price=target_price
            )

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

        await self.publish(EventType.ORDER_FILLED, {
            "order_id": order.order_id,
            "symbol": setup["symbol"],
            "direction": setup["direction"],
            "shares": setup["shares"],
            "fill_price": setup["entry_price"],  # Would be actual fill price
            "timestamp": datetime.now().isoformat()
        })

        # Publish position opened
        await self.publish(EventType.POSITION_OPENED, {
            "symbol": setup["symbol"],
            "direction": setup["direction"],
            "shares": setup["shares"],
            "entry_price": setup["entry_price"],
            "stop_price": setup["stop_price"],
            "target_price": setup["target_price"],
            "rrs": setup.get("rrs"),
            "timestamp": datetime.now().isoformat()
        })

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

    def set_auto_execute(self, enabled: bool):
        """Enable or disable auto-execution"""
        self.auto_execute = enabled
        logger.info(f"Auto-execute {'enabled' if enabled else 'disabled'}")
