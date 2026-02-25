"""
Options Order Executor for the RDT Trading System.

Handles placing single-leg and multi-leg (combo/spread) option orders.
Delegates to a broker-specific executor (IBKR or Paper).
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from loguru import logger

from options.models import (
    OptionContract, OptionLeg, OptionsStrategy,
    OptionAction, OptionsPositionSizeResult,
)
from options.config import OptionsConfig


class OptionsExecutor:
    """
    Executes options orders through a broker-specific executor.

    Supports:
    - Single-leg orders (long calls/puts)
    - Multi-leg combo orders (spreads, iron condors)
    - Position closing and rolling

    The broker_executor parameter can be:
    - IBKROptionsExecutor (for live/paper IBKR trading)
    - PaperOptionsExecutor (for standalone paper trading)
    """

    def __init__(self, broker_executor, config: Optional[OptionsConfig] = None):
        """
        Args:
            broker_executor: Broker-specific executor with execute_strategy(),
                close_position(), roll_position(), get_position(),
                get_all_positions() methods.
            config: Options configuration
        """
        self._executor = broker_executor
        self._config = config or OptionsConfig()

    @property
    def _positions(self):
        """Proxy access to underlying executor's positions for compatibility."""
        return self._executor._positions

    def execute_strategy(
        self,
        strategy: OptionsStrategy,
        size_result: OptionsPositionSizeResult,
    ) -> Optional[Dict]:
        """
        Execute a full options strategy (single or multi-leg).

        Args:
            strategy: The OptionsStrategy to execute
            size_result: Position sizing result with contract count

        Returns:
            Dict with order details or None on failure
        """
        if size_result.contracts <= 0:
            logger.warning(f"Zero contracts for {strategy.name} — skipping")
            return None

        try:
            result = self._executor.execute_strategy(strategy, size_result)

            if result:
                logger.info(
                    f"Options order executed: {strategy.name} {strategy.underlying} "
                    f"{size_result.contracts}x"
                )

            return result

        except Exception as e:
            logger.error(f"Failed to execute strategy: {e}")
            return None

    def close_position(self, symbol: str) -> Optional[Dict]:
        """
        Close an existing options position.

        Args:
            symbol: Underlying symbol

        Returns:
            Dict with close order details or None
        """
        try:
            result = self._executor.close_position(symbol)
            if result:
                logger.info(f"Closed options position: {symbol}")
            return result
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return None

    def roll_position(
        self, symbol: str, new_expiry: str
    ) -> Optional[Dict]:
        """
        Roll a position to a new expiration.

        Args:
            symbol: Underlying symbol
            new_expiry: New expiration date (YYYYMMDD)

        Returns:
            Dict with roll details or None
        """
        try:
            result = self._executor.roll_position(symbol, new_expiry)
            if result:
                logger.info(f"Rolled {symbol} to expiry {new_expiry}")
            return result
        except Exception as e:
            logger.error(f"Failed to roll position for {symbol}: {e}")
            return None

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get tracked options position for a symbol."""
        return self._executor.get_position(symbol)

    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all tracked options positions."""
        return self._executor.get_all_positions()


class IBKROptionsExecutor:
    """
    IBKR-specific options executor.

    Handles placing single-leg and multi-leg combo orders through
    the IBKR API using ib_insync.
    """

    def __init__(self, ib_client, config: Optional[OptionsConfig] = None):
        self._ib = ib_client
        self._config = config or OptionsConfig()
        self._positions: Dict[str, Dict] = {}

    def execute_strategy(
        self,
        strategy: OptionsStrategy,
        size_result: OptionsPositionSizeResult,
    ) -> Optional[Dict]:
        contracts = size_result.contracts

        if strategy.num_legs == 1:
            return self._execute_single_leg(strategy, contracts)
        else:
            return self._execute_combo(strategy, contracts)

    def _execute_single_leg(
        self, strategy: OptionsStrategy, contracts: int
    ) -> Optional[Dict]:
        leg = strategy.legs[0]
        contract = leg.contract
        greeks = leg.greeks

        if not greeks:
            logger.error(f"No greeks for {contract.display_name}")
            return None

        mid_price = greeks.mid_price
        tick_size = 0.01
        slippage = tick_size * self._config.slippage_ticks

        if leg.action == OptionAction.BUY:
            limit_price = round(mid_price + slippage, 2)
            action = "BUY"
        else:
            limit_price = round(max(0.01, mid_price - slippage), 2)
            action = "SELL"

        try:
            order_result = self._place_option_order(
                contract, action, contracts, limit_price
            )

            if order_result:
                self._positions[strategy.underlying] = {
                    "strategy": strategy,
                    "contracts": contracts,
                    "order_ids": [order_result["order_id"]],
                    "entry_time": datetime.now(),
                    "entry_premium": mid_price,
                    "entry_iv": greeks.implied_vol,
                    "entry_delta": greeks.delta,
                }

                logger.info(
                    f"Options order placed: {action} {contracts}x "
                    f"{contract.display_name} @ ${limit_price:.2f}"
                )

            return order_result

        except Exception as e:
            logger.error(f"Failed to execute single-leg order: {e}")
            return None

    def _execute_combo(
        self, strategy: OptionsStrategy, contracts: int
    ) -> Optional[Dict]:
        try:
            order_result = self._place_combo_order(strategy, contracts)

            if order_result:
                entry_premium = abs(strategy.net_premium) / strategy.legs[0].contract.multiplier

                self._positions[strategy.underlying] = {
                    "strategy": strategy,
                    "contracts": contracts,
                    "order_ids": order_result.get("order_ids", []),
                    "entry_time": datetime.now(),
                    "entry_premium": entry_premium,
                    "entry_iv": self._avg_iv(strategy),
                    "entry_delta": strategy.net_delta,
                }

                logger.info(
                    f"Combo order placed: {strategy.name} {strategy.underlying} "
                    f"{contracts}x, net_premium=${strategy.net_premium:.2f}"
                )

            return order_result

        except Exception as e:
            logger.error(f"Failed to execute combo order: {e}")
            return None

    def _place_option_order(
        self,
        contract: OptionContract,
        action: str,
        quantity: int,
        limit_price: float,
    ) -> Optional[Dict]:
        try:
            from ib_insync import Option as IBOption, LimitOrder

            ib = self._ib._ib

            ib_contract = IBOption(
                contract.symbol,
                contract.expiry,
                contract.strike,
                contract.right.value,
                contract.exchange,
                contract.currency,
            )

            qualified = ib.qualifyContracts(ib_contract)
            if not qualified:
                logger.error(f"Failed to qualify contract: {contract.display_name}")
                return None

            order = LimitOrder(action, quantity, limit_price)
            order.tif = "DAY"

            trade = ib.placeOrder(ib_contract, order)
            ib.sleep(0.5)

            order_id = str(trade.order.orderId)

            return {
                "order_id": order_id,
                "contract": contract.display_name,
                "action": action,
                "quantity": quantity,
                "limit_price": limit_price,
                "status": trade.orderStatus.status,
            }

        except Exception as e:
            logger.error(f"IBKR option order failed: {e}")
            return None

    def _place_combo_order(
        self, strategy: OptionsStrategy, contracts: int
    ) -> Optional[Dict]:
        try:
            from ib_insync import (
                Contract as IBContract,
                Option as IBOption,
                LimitOrder,
                ComboLeg,
            )

            ib = self._ib._ib

            combo_legs = []
            for leg in strategy.legs:
                ib_opt = IBOption(
                    leg.contract.symbol,
                    leg.contract.expiry,
                    leg.contract.strike,
                    leg.contract.right.value,
                    leg.contract.exchange,
                    leg.contract.currency,
                )

                qualified = ib.qualifyContracts(ib_opt)
                if not qualified:
                    logger.error(f"Failed to qualify: {leg.contract.display_name}")
                    return None

                action = "BUY" if leg.action == OptionAction.BUY else "SELL"
                combo_leg = ComboLeg(
                    conId=ib_opt.conId,
                    ratio=leg.quantity,
                    action=action,
                    exchange=leg.contract.exchange,
                )
                combo_legs.append(combo_leg)

            bag = IBContract()
            bag.symbol = strategy.underlying
            bag.secType = "BAG"
            bag.currency = "USD"
            bag.exchange = "SMART"
            bag.comboLegs = combo_legs

            net_price = 0.0
            for leg in strategy.legs:
                if leg.greeks:
                    mid = leg.greeks.mid_price
                    if leg.action == OptionAction.BUY:
                        net_price -= mid
                    else:
                        net_price += mid

            tick_size = 0.01
            slippage = tick_size * self._config.slippage_ticks
            if net_price < 0:
                limit_price = round(net_price - slippage, 2)
            else:
                limit_price = round(net_price + slippage, 2)

            ib_limit = round(-limit_price, 2)

            order = LimitOrder("BUY", contracts, ib_limit)
            order.tif = "DAY"

            trade = ib.placeOrder(bag, order)
            ib.sleep(1.0)

            order_id = str(trade.order.orderId)

            logger.info(
                f"Combo order placed: {strategy.name} {strategy.underlying} "
                f"{contracts}x @ net ${ib_limit:.2f}, order_id={order_id}"
            )

            return {
                "order_id": order_id,
                "order_ids": [order_id],
                "strategy": strategy.name,
                "contracts": contracts,
                "net_limit_price": ib_limit,
                "status": trade.orderStatus.status,
            }

        except Exception as e:
            logger.error(f"IBKR combo order failed: {e}")
            return None

    def close_position(self, symbol: str) -> Optional[Dict]:
        position = self._positions.get(symbol)
        if not position:
            logger.warning(f"No options position found for {symbol}")
            return None

        strategy = position["strategy"]
        contracts = position["contracts"]

        close_legs = []
        for leg in strategy.legs:
            close_action = OptionAction.SELL if leg.action == OptionAction.BUY else OptionAction.BUY
            close_legs.append(OptionLeg(
                contract=leg.contract,
                action=close_action,
                quantity=leg.quantity,
                greeks=leg.greeks,
            ))

        close_strategy = OptionsStrategy(
            name=f"close_{strategy.name}",
            underlying=symbol,
            direction=strategy.direction,
            legs=close_legs,
        )

        if len(close_legs) == 1:
            result = self._execute_single_leg(close_strategy, contracts)
        else:
            result = self._execute_combo(close_strategy, contracts)

        if result:
            del self._positions[symbol]
            logger.info(f"Closed options position: {symbol} {strategy.name}")

        return result

    def roll_position(
        self, symbol: str, new_expiry: str
    ) -> Optional[Dict]:
        position = self._positions.get(symbol)
        if not position:
            logger.warning(f"No options position to roll for {symbol}")
            return None

        strategy = position["strategy"]
        contracts = position["contracts"]

        close_result = self.close_position(symbol)
        if not close_result:
            logger.error(f"Failed to close position for roll: {symbol}")
            return None

        new_legs = []
        for leg in strategy.legs:
            new_contract = OptionContract(
                symbol=leg.contract.symbol,
                expiry=new_expiry,
                strike=leg.contract.strike,
                right=leg.contract.right,
                exchange=leg.contract.exchange,
                multiplier=leg.contract.multiplier,
            )
            new_legs.append(OptionLeg(
                contract=new_contract,
                action=leg.action,
                quantity=leg.quantity,
            ))

        new_strategy = OptionsStrategy(
            name=strategy.name,
            underlying=symbol,
            direction=strategy.direction,
            legs=new_legs,
        )

        size_result = OptionsPositionSizeResult(
            strategy_name=new_strategy.name,
            contracts=contracts,
            max_risk=0,
            premium_cost=0,
            reason="Roll position",
        )

        open_result = self.execute_strategy(new_strategy, size_result)

        if open_result:
            logger.info(
                f"Rolled {symbol} {strategy.name} to expiry {new_expiry}"
            )

        return {
            "close": close_result,
            "open": open_result,
            "new_expiry": new_expiry,
        }

    def get_position(self, symbol: str) -> Optional[Dict]:
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Dict]:
        return dict(self._positions)

    def _avg_iv(self, strategy: OptionsStrategy) -> float:
        ivs = [leg.greeks.implied_vol for leg in strategy.legs if leg.greeks and leg.greeks.implied_vol > 0]
        return sum(ivs) / len(ivs) if ivs else 0.0
