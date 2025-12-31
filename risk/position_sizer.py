"""
Position Sizing Calculator
Calculates optimal position sizes based on risk parameters
"""

from typing import Optional, Dict
from loguru import logger

from risk.models import PositionSizeResult, RiskLimits


class PositionSizer:
    """
    Calculate position sizes based on risk management rules

    Implements the RDT methodology:
    - Risk 1-2% per trade
    - Use ATR for stop placement
    - Maintain 2:1+ risk/reward ratio
    """

    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        """
        Initialize position sizer

        Args:
            risk_limits: Risk limit configuration
        """
        self.limits = risk_limits or RiskLimits()

    def calculate_position_size(
        self,
        account_size: float,
        entry_price: float,
        atr: float,
        direction: str = "long",
        stop_multiplier: float = 1.5,
        target_multiplier: float = 3.0,
        custom_risk_percent: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Calculate position size based on ATR and risk parameters

        Args:
            account_size: Total account value
            entry_price: Planned entry price
            atr: Average True Range of the stock
            direction: 'long' or 'short'
            stop_multiplier: ATR multiplier for stop loss (default 1.5x)
            target_multiplier: ATR multiplier for target (default 3x)
            custom_risk_percent: Override default risk percentage

        Returns:
            PositionSizeResult with calculated values
        """
        # Risk amount (default 1% of account)
        risk_percent = custom_risk_percent or self.limits.max_risk_per_trade
        risk_amount = account_size * risk_percent

        # Stop distance based on ATR
        stop_distance = atr * stop_multiplier

        # Calculate stop and target prices
        if direction.lower() == "long":
            stop_price = entry_price - stop_distance
            target_price = entry_price + (atr * target_multiplier)
        else:  # short
            stop_price = entry_price + stop_distance
            target_price = entry_price - (atr * target_multiplier)

        # Position size = Risk Amount / Stop Distance
        if stop_distance <= 0:
            logger.warning("Invalid stop distance, returning 0 shares")
            return PositionSizeResult(
                shares=0,
                position_value=0,
                risk_amount=risk_amount,
                stop_distance=stop_distance,
                stop_price=stop_price,
                target_price=target_price,
                risk_reward_ratio=0,
                risk_percent=risk_percent,
                reason="Invalid stop distance"
            )

        shares = int(risk_amount / stop_distance)

        # Check maximum position size constraint
        max_position_value = account_size * self.limits.max_position_size
        max_shares_by_position = int(max_position_value / entry_price)

        # Use smaller of the two
        if shares > max_shares_by_position:
            shares = max_shares_by_position
            reason = f"Limited by max position size ({self.limits.max_position_size*100}%)"
        else:
            reason = "Based on risk per trade"

        # Ensure at least 1 share if viable
        if shares == 0 and risk_amount >= stop_distance:
            shares = 1

        # Final calculations
        position_value = entry_price * shares
        actual_risk = stop_distance * shares
        reward = abs(target_price - entry_price) * shares
        risk_reward_ratio = reward / actual_risk if actual_risk > 0 else 0

        return PositionSizeResult(
            shares=shares,
            position_value=position_value,
            risk_amount=actual_risk,
            stop_distance=stop_distance,
            stop_price=round(stop_price, 2),
            target_price=round(target_price, 2),
            risk_reward_ratio=round(risk_reward_ratio, 2),
            risk_percent=risk_percent,
            reason=reason
        )

    def calculate_with_fixed_stop(
        self,
        account_size: float,
        entry_price: float,
        stop_price: float,
        target_price: Optional[float] = None,
        custom_risk_percent: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Calculate position size with a fixed stop price

        Args:
            account_size: Total account value
            entry_price: Planned entry price
            stop_price: Fixed stop loss price
            target_price: Optional target price
            custom_risk_percent: Override default risk percentage

        Returns:
            PositionSizeResult with calculated values
        """
        risk_percent = custom_risk_percent or self.limits.max_risk_per_trade
        risk_amount = account_size * risk_percent

        # Determine direction
        direction = "long" if stop_price < entry_price else "short"

        # Stop distance
        stop_distance = abs(entry_price - stop_price)

        if stop_distance <= 0:
            return PositionSizeResult(
                shares=0,
                position_value=0,
                risk_amount=risk_amount,
                stop_distance=0,
                stop_price=stop_price,
                target_price=target_price or entry_price,
                risk_reward_ratio=0,
                risk_percent=risk_percent,
                reason="Stop price equals entry price"
            )

        # Calculate shares
        shares = int(risk_amount / stop_distance)

        # Apply max position size constraint
        max_position_value = account_size * self.limits.max_position_size
        max_shares_by_position = int(max_position_value / entry_price)

        if shares > max_shares_by_position:
            shares = max_shares_by_position
            reason = f"Limited by max position size"
        else:
            reason = "Based on risk per trade"

        # Calculate target if not provided (2:1 R/R)
        if target_price is None:
            target_distance = stop_distance * self.limits.min_risk_reward
            if direction == "long":
                target_price = entry_price + target_distance
            else:
                target_price = entry_price - target_distance

        # Final calculations
        position_value = entry_price * shares
        actual_risk = stop_distance * shares
        reward = abs(target_price - entry_price) * shares
        risk_reward_ratio = reward / actual_risk if actual_risk > 0 else 0

        return PositionSizeResult(
            shares=shares,
            position_value=position_value,
            risk_amount=actual_risk,
            stop_distance=stop_distance,
            stop_price=round(stop_price, 2),
            target_price=round(target_price, 2),
            risk_reward_ratio=round(risk_reward_ratio, 2),
            risk_percent=risk_percent,
            reason=reason
        )

    def scale_position(
        self,
        current_shares: int,
        scale_factor: float,
        min_shares: int = 1
    ) -> int:
        """
        Scale a position up or down

        Args:
            current_shares: Current number of shares
            scale_factor: Factor to scale by (0.5 = half, 2.0 = double)
            min_shares: Minimum shares to maintain

        Returns:
            New share count
        """
        new_shares = int(current_shares * scale_factor)
        return max(new_shares, min_shares)

    def calculate_add_to_position(
        self,
        account_size: float,
        current_position: Dict,
        add_price: float,
        atr: float,
        max_adds: int = 2
    ) -> Optional[PositionSizeResult]:
        """
        Calculate shares to add to a winning position

        Args:
            account_size: Total account value
            current_position: Current position info (shares, avg_price, entry_price)
            add_price: Price to add at
            atr: Current ATR
            max_adds: Maximum number of adds allowed

        Returns:
            PositionSizeResult for the add, or None if not advisable
        """
        current_shares = current_position.get("shares", 0)
        entry_price = current_position.get("entry_price", add_price)
        adds_count = current_position.get("adds_count", 0)

        # Check if we've hit max adds
        if adds_count >= max_adds:
            logger.info(f"Max adds ({max_adds}) reached")
            return None

        # Only add to winning positions
        direction = current_position.get("direction", "long")
        if direction == "long" and add_price <= entry_price:
            logger.info("Cannot add to losing long position")
            return None
        elif direction == "short" and add_price >= entry_price:
            logger.info("Cannot add to losing short position")
            return None

        # Calculate reduced size for add (typically 50% of original)
        add_risk_percent = self.limits.max_risk_per_trade * 0.5
        risk_amount = account_size * add_risk_percent
        stop_distance = atr * 1.5
        add_shares = int(risk_amount / stop_distance)

        # Don't add more than current position
        add_shares = min(add_shares, current_shares)

        if add_shares == 0:
            return None

        # Calculate new stop (based on new average)
        new_total_shares = current_shares + add_shares
        new_avg_price = (
            (current_shares * current_position.get("avg_price", entry_price)) +
            (add_shares * add_price)
        ) / new_total_shares

        if direction == "long":
            stop_price = new_avg_price - (atr * 1.5)
            target_price = add_price + (atr * 3.0)
        else:
            stop_price = new_avg_price + (atr * 1.5)
            target_price = add_price - (atr * 3.0)

        return PositionSizeResult(
            shares=add_shares,
            position_value=add_shares * add_price,
            risk_amount=add_shares * stop_distance,
            stop_distance=stop_distance,
            stop_price=round(stop_price, 2),
            target_price=round(target_price, 2),
            risk_reward_ratio=2.0,
            risk_percent=add_risk_percent,
            reason=f"Add #{adds_count + 1} to winning position"
        )
