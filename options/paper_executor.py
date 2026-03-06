"""
Paper Options Executor for the RDT Trading System.

Simulates options order fills for paper trading mode.
Tracks positions with live P&L recalculation via Black-Scholes.
Positions are persisted to the database so they survive restarts.
"""

import json
import random
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from options.models import (
    OptionAction, OptionContract, OptionGreeks, OptionLeg, OptionRight,
    OptionsStrategy, OptionsPositionSizeResult, StrategyDirection,
)
from options.config import OptionsConfig
from options.pricing import black_scholes_price, generate_greeks, DEFAULT_RISK_FREE_RATE
from options.chain_provider import ChainProvider


def _legs_to_json(strategy: OptionsStrategy) -> str:
    """Serialize strategy legs to JSON for DB storage."""
    legs = []
    for leg in strategy.legs:
        leg_data = {
            "symbol": leg.contract.symbol,
            "expiry": leg.contract.expiry,
            "strike": leg.contract.strike,
            "right": leg.contract.right.value,
            "exchange": leg.contract.exchange,
            "multiplier": leg.contract.multiplier,
            "action": leg.action.value,
            "quantity": leg.quantity,
        }
        if leg.greeks:
            leg_data["greeks"] = {
                "delta": leg.greeks.delta,
                "gamma": leg.greeks.gamma,
                "theta": leg.greeks.theta,
                "vega": leg.greeks.vega,
                "implied_vol": leg.greeks.implied_vol,
                "underlying_price": leg.greeks.underlying_price,
                "option_price": leg.greeks.option_price,
                "bid": leg.greeks.bid,
                "ask": leg.greeks.ask,
                "volume": leg.greeks.volume,
                "open_interest": leg.greeks.open_interest,
            }
        legs.append(leg_data)
    return json.dumps(legs)


def _legs_from_json(
    legs_json: str, strategy_name: str, underlying: str, direction: str,
    max_loss: float = 0.0, max_profit: float = 0.0,
    breakeven: Optional[List[float]] = None, net_premium: float = 0.0,
) -> Optional[OptionsStrategy]:
    """Reconstruct an OptionsStrategy from stored JSON legs."""
    try:
        legs_data = json.loads(legs_json)
    except (json.JSONDecodeError, TypeError):
        logger.error(f"Invalid legs_json for {underlying}")
        return None

    legs = []
    for ld in legs_data:
        contract = OptionContract(
            symbol=ld["symbol"],
            expiry=ld["expiry"],
            strike=ld["strike"],
            right=OptionRight(ld["right"]),
            exchange=ld.get("exchange", "SMART"),
            multiplier=ld.get("multiplier", 100),
        )
        greeks = None
        if "greeks" in ld:
            g = ld["greeks"]
            greeks = OptionGreeks(
                delta=g["delta"],
                gamma=g["gamma"],
                theta=g["theta"],
                vega=g["vega"],
                implied_vol=g["implied_vol"],
                underlying_price=g["underlying_price"],
                option_price=g["option_price"],
                bid=g["bid"],
                ask=g["ask"],
                volume=g.get("volume", 0),
                open_interest=g.get("open_interest", 0),
            )
        legs.append(OptionLeg(
            contract=contract,
            action=OptionAction(ld["action"]),
            quantity=ld.get("quantity", 1),
            greeks=greeks,
        ))

    return OptionsStrategy(
        name=strategy_name,
        underlying=underlying,
        direction=StrategyDirection(direction),
        legs=legs,
        max_loss=max_loss,
        max_profit=max_profit,
        breakeven=breakeven if breakeven is not None else [],
        net_premium=net_premium,
    )


class PaperOptionsExecutor:
    """
    Simulates options order execution for paper trading.

    Features:
    - Fill simulation at mid price + random slippage
    - Position tracking with live P&L via BS repricing
    - Close and roll support
    - Database persistence (positions survive restarts)
    """

    def __init__(
        self,
        provider: ChainProvider,
        config: Optional[OptionsConfig] = None,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ):
        self._provider = provider
        self._config = config or OptionsConfig()
        self._r = risk_free_rate

        # Active positions: symbol -> position dict
        self._positions: Dict[str, Dict] = {}

        # Load persisted positions from DB
        self._load_positions()

    def _get_repository(self):
        """Lazy-load the trades repository to avoid circular imports."""
        try:
            from data.database.trades_repository import get_trades_repository
            return get_trades_repository()
        except Exception as e:
            logger.debug(f"Could not load trades repository: {e}")
            return None

    def _load_positions(self):
        """Restore persisted positions from the database on startup."""
        repo = self._get_repository()
        if not repo:
            return

        try:
            rows = repo.get_all_options_positions()
            for row in rows:
                symbol = row['symbol']
                breakeven = []
                if row.get('breakeven_json'):
                    try:
                        breakeven = json.loads(row['breakeven_json'])
                    except (json.JSONDecodeError, TypeError):
                        pass
                strategy = _legs_from_json(
                    row['legs_json'],
                    row['strategy_name'],
                    symbol,
                    row['direction'],
                    max_loss=row.get('max_loss', 0.0),
                    max_profit=row.get('max_profit', 0.0),
                    breakeven=breakeven,
                    net_premium=row.get('net_premium', 0.0),
                )
                if not strategy:
                    logger.warning(f"Could not restore options position for {symbol}")
                    continue

                fill_details = []
                if row.get('fill_details_json'):
                    try:
                        fill_details = json.loads(row['fill_details_json'])
                    except (json.JSONDecodeError, TypeError):
                        pass

                order_ids = []
                if row.get('order_ids'):
                    try:
                        order_ids = json.loads(row['order_ids'])
                    except (json.JSONDecodeError, TypeError):
                        pass

                entry_time = row.get('entry_time')
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)

                self._positions[symbol] = {
                    "strategy": strategy,
                    "contracts": row['contracts'],
                    "order_ids": order_ids,
                    "entry_time": entry_time or datetime.now(),
                    "entry_premium": row.get('entry_premium', 0),
                    "entry_iv": row.get('entry_iv', 0),
                    "entry_delta": row.get('entry_delta', 0),
                    "fill_details": fill_details,
                    "total_premium": row.get('total_premium', 0),
                }

            if rows:
                logger.info(f"Restored {len(rows)} options position(s) from database")
        except Exception as e:
            logger.error(f"Error loading options positions from DB: {e}")

    def _persist_position(self, symbol: str):
        """Save the in-memory position for `symbol` to the database."""
        repo = self._get_repository()
        if not repo:
            return

        position = self._positions.get(symbol)
        if not position:
            return

        try:
            strategy = position["strategy"]
            repo.save_options_position({
                'symbol': symbol,
                'strategy_name': strategy.name,
                'direction': strategy.direction.value,
                'contracts': position['contracts'],
                'entry_time': position.get('entry_time', datetime.utcnow()),
                'entry_premium': position.get('entry_premium', 0),
                'total_premium': position.get('total_premium', 0),
                'entry_iv': position.get('entry_iv'),
                'entry_delta': position.get('entry_delta'),
                'order_ids': json.dumps(position.get('order_ids', [])),
                'legs_json': _legs_to_json(strategy),
                'fill_details_json': json.dumps(position.get('fill_details', [])),
                'max_loss': strategy.max_loss,
                'max_profit': strategy.max_profit,
                'net_premium': strategy.net_premium,
                'breakeven_json': json.dumps(strategy.breakeven),
            })
        except Exception as e:
            logger.error(f"Error persisting options position for {symbol}: {e}")

    def _delete_persisted_position(self, symbol: str):
        """Remove a position from the database."""
        repo = self._get_repository()
        if not repo:
            return
        try:
            repo.close_options_position(symbol)
        except Exception as e:
            logger.error(f"Error deleting persisted options position for {symbol}: {e}")

    def _save_trade_record(self, symbol: str, position: Dict, pnl: float, exit_reason: str = "manual"):
        """Write a closed-trade record to the database for history."""
        repo = self._get_repository()
        if not repo:
            return

        try:
            strategy = position["strategy"]
            total_prem = abs(position.get("total_premium", 0))
            current_value = self._get_position_value_from_data(position)
            pnl_pct = (pnl / total_prem * 100) if total_prem > 0.01 else 0.0

            repo.save_options_trade({
                'symbol': symbol,
                'strategy_name': strategy.name,
                'direction': strategy.direction.value,
                'contracts': position['contracts'],
                'entry_time': position.get('entry_time', datetime.utcnow()),
                'exit_time': datetime.utcnow(),
                'entry_premium': position.get('entry_premium', 0),
                'total_premium': position.get('total_premium', 0),
                'exit_premium': current_value,
                'pnl': pnl,
                'pnl_percent': pnl_pct,
                'entry_iv': position.get('entry_iv'),
                'exit_iv': self._avg_iv(strategy),
                'entry_delta': position.get('entry_delta'),
                'legs_json': _legs_to_json(strategy),
                'fill_details_json': json.dumps(position.get('fill_details', [])),
                'exit_reason': exit_reason,
                'status': 'closed',
                'order_ids': json.dumps(position.get('order_ids', [])),
            })
        except Exception as e:
            logger.error(f"Error saving options trade record for {symbol}: {e}")

    def execute_strategy(
        self,
        strategy: OptionsStrategy,
        size_result: OptionsPositionSizeResult,
    ) -> Optional[Dict]:
        """
        Execute a strategy with simulated fills.

        Returns:
            Dict with order details, or None on failure.
        """
        if size_result.contracts <= 0:
            logger.warning(f"Zero contracts for {strategy.name} — skipping")
            return None

        contracts = size_result.contracts
        order_id = str(uuid.uuid4())[:8]

        # Calculate fill prices for each leg
        fill_details = []
        total_premium = 0.0

        for leg in strategy.legs:
            greeks = leg.greeks
            if not greeks:
                # Try to get greeks from provider
                greeks = self._provider.get_greeks(leg.contract)
                if not greeks:
                    logger.error(f"No greeks for {leg.contract.display_name}")
                    return None

            mid = greeks.mid_price
            spread = greeks.spread

            # Simulate slippage: 0-2% of spread
            slippage = random.uniform(0, 0.02) * spread

            if leg.action == OptionAction.BUY:
                fill_price = round(mid + slippage, 2)
                total_premium -= fill_price * leg.contract.multiplier * leg.quantity
            else:
                fill_price = round(max(0.01, mid - slippage), 2)
                total_premium += fill_price * leg.contract.multiplier * leg.quantity

            fill_details.append({
                "contract": leg.contract.display_name,
                "action": leg.action.value,
                "fill_price": fill_price,
                "quantity": leg.quantity,
            })

        # Track position
        # total_premium is per-set (1 contract set), includes multiplier
        # entry_premium: per-share net cost for exit_manager compatibility
        multiplier = strategy.legs[0].contract.multiplier if strategy.legs else 100
        entry_premium = abs(total_premium) / multiplier if multiplier > 0 else 0
        avg_iv = self._avg_iv(strategy)
        # Store actual total cost across all contracts
        actual_total_premium = total_premium * contracts

        self._positions[strategy.underlying] = {
            "strategy": strategy,
            "contracts": contracts,
            "order_ids": [order_id],
            "entry_time": datetime.now(),
            "entry_premium": entry_premium,
            "entry_iv": avg_iv,
            "entry_delta": strategy.net_delta,
            "fill_details": fill_details,
            "total_premium": actual_total_premium,
        }

        # Persist to DB
        self._persist_position(strategy.underlying)

        logger.info(
            f"PAPER OPTIONS FILL: {strategy.name} {strategy.underlying} "
            f"{contracts}x, premium=${total_premium:.2f}, order_id={order_id}"
        )

        return {
            "order_id": order_id,
            "order_ids": [order_id],
            "strategy": strategy.name,
            "contracts": contracts,
            "fill_details": fill_details,
            "net_premium": total_premium,
            "status": "Filled",
        }

    def close_position(self, symbol: str) -> Optional[Dict]:
        """Close an existing paper options position."""
        position = self._positions.get(symbol)
        if not position:
            logger.warning(f"No paper options position for {symbol}")
            return None

        strategy = position["strategy"]
        contracts = position["contracts"]

        # Get current value to calculate P&L
        current_value = self._get_position_value(symbol)
        total_premium = position.get("total_premium", 0)

        close_id = str(uuid.uuid4())[:8]

        # PnL = current_value + total_premium (total_premium is negative for debits)
        pnl = (current_value + total_premium) if current_value is not None else 0.0

        # Save trade record before removing position
        self._save_trade_record(symbol, position, pnl, exit_reason="manual")

        del self._positions[symbol]

        # Remove from DB
        self._delete_persisted_position(symbol)

        logger.info(
            f"PAPER OPTIONS CLOSE: {symbol} {strategy.name} "
            f"P&L=${pnl:.2f}, order_id={close_id}"
        )

        return {
            "order_id": close_id,
            "symbol": symbol,
            "strategy": strategy.name,
            "pnl": pnl,
            "status": "Filled",
        }

    def roll_position(
        self, symbol: str, new_expiry: str
    ) -> Optional[Dict]:
        """Roll a position to a new expiration."""
        position = self._positions.get(symbol)
        if not position:
            logger.warning(f"No paper options position to roll for {symbol}")
            return None

        strategy = position["strategy"]
        contracts = position["contracts"]

        # Close existing
        close_result = self.close_position(symbol)
        if not close_result:
            return None

        # Build new strategy with updated expiry
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
            # Get fresh greeks for new contract
            new_greeks = self._provider.get_greeks(new_contract)
            new_legs.append(OptionLeg(
                contract=new_contract,
                action=leg.action,
                quantity=leg.quantity,
                greeks=new_greeks,
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
            logger.info(f"PAPER ROLL: {symbol} {strategy.name} to {new_expiry}")

        return {
            "close": close_result,
            "open": open_result,
            "new_expiry": new_expiry,
        }

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get tracked position for a symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all positions with live P&L."""
        result = {}
        for symbol, position in self._positions.items():
            pos_copy = dict(position)
            current_value = self._get_position_value(symbol)
            if current_value is not None:
                total_premium = position.get("total_premium", 0)
                pos_copy["current_value"] = current_value
                # PnL = current_value + total_premium (total_premium is negative for debits)
                pos_copy["unrealized_pnl"] = current_value + total_premium
            result[symbol] = pos_copy
        return result

    def _get_position_value(self, symbol: str) -> Optional[float]:
        """Calculate current position value using provider greeks."""
        position = self._positions.get(symbol)
        if not position:
            return None
        return self._get_position_value_from_data(position)

    def _get_position_value_from_data(self, position: Dict) -> Optional[float]:
        """Calculate position value from position data dict."""
        strategy = position["strategy"]
        contracts = position["contracts"]
        total_value = 0.0

        for leg in strategy.legs:
            greeks = self._provider.get_greeks(leg.contract)
            if not greeks:
                return None

            mid = greeks.mid_price
            if leg.action == OptionAction.BUY:
                total_value += mid * leg.contract.multiplier * leg.quantity
            else:
                total_value -= mid * leg.contract.multiplier * leg.quantity

        return total_value * contracts

    def _avg_iv(self, strategy: OptionsStrategy) -> float:
        """Calculate average IV across strategy legs."""
        ivs = [
            leg.greeks.implied_vol
            for leg in strategy.legs
            if leg.greeks and leg.greeks.implied_vol > 0
        ]
        return sum(ivs) / len(ivs) if ivs else 0.0
