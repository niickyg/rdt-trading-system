"""
Options Position Sizer for the RDT Trading System.

Calculates the number of contracts based on defined risk per trade,
max loss per contract/spread, and buying power requirements.
"""

import math
from typing import Optional
from loguru import logger

from options.models import OptionsStrategy, OptionsPositionSizeResult
from options.config import OptionsConfig


class OptionsPositionSizer:
    """
    Calculates position size (number of contracts) for options strategies.

    For defined-risk strategies (spreads, iron condors):
        contracts = floor(max_risk_dollars / max_loss_per_contract)

    For undefined-risk (long options):
        contracts = floor(max_risk_dollars / premium_per_contract)
    """

    def __init__(self, config: Optional[OptionsConfig] = None):
        self._config = config or OptionsConfig()

    def calculate(
        self,
        strategy: OptionsStrategy,
        account_size: float,
        max_risk_per_trade: float = 0.015,
    ) -> OptionsPositionSizeResult:
        """
        Calculate number of contracts for a strategy.

        Args:
            strategy: The selected OptionsStrategy
            account_size: Total account value
            max_risk_per_trade: Max risk as fraction of account (default 1.5%)

        Returns:
            OptionsPositionSizeResult with contracts, risk, and cost
        """
        max_risk_dollars = account_size * max_risk_per_trade

        if strategy.max_loss <= 0:
            logger.warning(f"Strategy {strategy.name} has zero max_loss")
            return OptionsPositionSizeResult(
                strategy_name=strategy.name,
                contracts=0,
                max_risk=0,
                premium_cost=0,
                reason="Invalid max loss"
            )

        # Calculate contracts from risk budget
        contracts = math.floor(max_risk_dollars / strategy.max_loss)

        # Minimum 1 contract if we can afford the risk
        if contracts == 0 and max_risk_dollars >= strategy.max_loss:
            contracts = 1

        # Enforce minimum
        if contracts < 1:
            return OptionsPositionSizeResult(
                strategy_name=strategy.name,
                contracts=0,
                max_risk=0,
                premium_cost=0,
                reason=f"Max loss per contract (${strategy.max_loss:.2f}) exceeds risk budget (${max_risk_dollars:.2f})"
            )

        # Calculate costs
        total_max_risk = strategy.max_loss * contracts

        if strategy.is_debit:
            premium_cost = abs(strategy.net_premium) * contracts
            premium_received = 0.0
        else:
            premium_cost = 0.0
            premium_received = strategy.net_premium * contracts

        # Buying power: for spreads, it's max_loss * contracts
        # For long options, it's the premium paid
        if strategy.is_defined_risk:
            buying_power = total_max_risk
        else:
            buying_power = premium_cost

        risk_percent = total_max_risk / account_size if account_size > 0 else 0

        result = OptionsPositionSizeResult(
            strategy_name=strategy.name,
            contracts=contracts,
            max_risk=total_max_risk,
            premium_cost=premium_cost,
            premium_received=premium_received,
            buying_power_required=buying_power,
            risk_percent=risk_percent,
            reason=f"Based on {max_risk_per_trade*100:.1f}% risk ({contracts} contracts)",
        )

        logger.info(
            f"Position size: {strategy.name} {strategy.underlying} "
            f"{contracts} contracts, max_risk=${total_max_risk:.2f} "
            f"({risk_percent*100:.1f}% of account)"
        )

        return result
