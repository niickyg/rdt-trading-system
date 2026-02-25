"""
Position Sizing Calculator
Calculates optimal position sizes based on risk parameters
"""

from datetime import datetime
from typing import Optional, Dict
from loguru import logger

from risk.models import PositionSizeResult, RiskLimits

# Dynamic position sizing (optional)
try:
    from ml.dynamic_sizer import DynamicPositionSizer, SizingInput, get_dynamic_sizer
    DYNAMIC_SIZER_AVAILABLE = True
except ImportError:
    DYNAMIC_SIZER_AVAILABLE = False
    logger.debug("Dynamic position sizer not available")


class PositionSizer:
    """
    Calculate position sizes based on risk management rules

    Implements the RDT methodology:
    - Risk 1-2% per trade
    - Use ATR for stop placement
    - Maintain 2:1+ risk/reward ratio

    Supports optional ML-based dynamic sizing that adjusts risk per trade
    based on signal conviction (0.5x-2.0x of base risk).
    """

    def __init__(
        self,
        risk_limits: Optional[RiskLimits] = None,
        dynamic_sizing_enabled: bool = True,
    ):
        """
        Initialize position sizer

        Args:
            risk_limits: Risk limit configuration
            dynamic_sizing_enabled: Enable ML-based dynamic position sizing
        """
        self.limits = risk_limits or RiskLimits()

        # Dynamic sizing
        self.dynamic_sizing_enabled = dynamic_sizing_enabled and DYNAMIC_SIZER_AVAILABLE
        self._dynamic_sizer: Optional["DynamicPositionSizer"] = None
        if self.dynamic_sizing_enabled:
            self._dynamic_sizer = get_dynamic_sizer(
                base_risk=self.limits.max_risk_per_trade,
                enabled=True,
            )
            logger.info("Dynamic position sizing enabled")

    def calculate_position_size(
        self,
        account_size: float,
        entry_price: float,
        atr: float,
        direction: str = "long",
        stop_multiplier: float = 1.5,  # 1.5x ATR stop (unchanged)
        target_multiplier: float = 2.5,  # Was 1.0 — wider target for better R:R
        custom_risk_percent: Optional[float] = None,
        signal_features: Optional[Dict] = None,
    ) -> PositionSizeResult:
        """
        Calculate position size based on ATR and risk parameters

        Args:
            account_size: Total account value
            entry_price: Planned entry price
            atr: Average True Range of the stock
            direction: 'long' or 'short'
            stop_multiplier: ATR multiplier for stop loss (default 1.5x ATR)
            target_multiplier: ATR multiplier for target (default 2.5x ATR)
            custom_risk_percent: Override default risk percentage
            signal_features: Optional dict with signal quality data for dynamic
                sizing. Keys: ml_confidence, rrs, quality_warnings,
                market_regime, symbol, portfolio_heat, sector_concentration.

        Returns:
            PositionSizeResult with calculated values
        """
        # Risk amount (default 1% of account)
        risk_percent = custom_risk_percent or self.limits.max_risk_per_trade

        # --- Dynamic sizing adjustment ---
        dynamic_result = None
        if (
            self.dynamic_sizing_enabled
            and self._dynamic_sizer is not None
            and signal_features is not None
            and custom_risk_percent is None  # Don't override explicit custom risk
        ):
            try:
                sizing_input = self._build_sizing_input(signal_features, direction)
                dynamic_result = self._dynamic_sizer.calculate_multiplier(sizing_input)
                risk_percent = dynamic_result.effective_risk_pct

                # Hard cap: never exceed 2x base risk
                max_allowed = self.limits.max_risk_per_trade * self._dynamic_sizer.MAX_MULTIPLIER
                risk_percent = min(risk_percent, max_allowed)
            except Exception as e:
                logger.warning(f"Dynamic sizing failed, using base risk: {e}")
                risk_percent = self.limits.max_risk_per_trade

        risk_amount = account_size * risk_percent

        # Guard against invalid entry price (Fix 5)
        if entry_price <= 0:
            logger.warning(f"Invalid entry price: {entry_price}")
            return PositionSizeResult(
                shares=0, position_value=0, risk_amount=risk_amount,
                stop_distance=0, stop_price=0,
                target_price=0, risk_reward_ratio=0,
                risk_percent=risk_percent, reason="Invalid entry price"
            )

        # Enforce minimum ATR threshold (percentage-based) (Fix 1)
        min_atr_pct = 0.05  # Minimum 0.05% of price
        min_atr = max(0.10, entry_price * min_atr_pct / 100) if entry_price > 0 else 0.10
        if atr < min_atr:
            logger.warning(f"ATR too low ({atr:.4f}), using minimum of ${min_atr:.2f} for ${entry_price:.2f} stock")
            atr = min_atr

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

    @property
    def dynamic_sizer(self) -> Optional["DynamicPositionSizer"]:
        """Access the underlying DynamicPositionSizer (or None)."""
        return self._dynamic_sizer

    def _build_sizing_input(
        self, signal_features: Dict, direction: str
    ) -> "SizingInput":
        """
        Build a SizingInput from a signal_features dict.

        Expected keys in signal_features (all optional with defaults):
            ml_confidence (float 0-100), rrs (float), quality_warnings (list),
            market_regime (str), symbol (str), portfolio_heat (float 0-1),
            sector_concentration (int), time_of_day_hour (int 0-23).
        """
        from ml.dynamic_sizer import SizingInput

        # Get recent stats from the dynamic sizer's trade history
        stats = self._dynamic_sizer.get_recent_stats(n=10)

        return SizingInput(
            ml_confidence=signal_features.get("ml_confidence", 50.0),
            rrs_strength=signal_features.get("rrs", 0.0),
            quality_warnings=signal_features.get("quality_warnings", []),
            market_regime=signal_features.get("market_regime", "unknown"),
            direction=direction,
            recent_win_rate=stats["win_rate"],
            recent_avg_win=stats["avg_win_r"],
            recent_avg_loss=stats["avg_loss_r"],
            recent_trade_count=stats["trade_count"],
            portfolio_heat=signal_features.get("portfolio_heat", 0.0),
            sector_concentration=signal_features.get("sector_concentration", 0),
            symbol=signal_features.get("symbol", ""),
            time_of_day_hour=signal_features.get(
                "time_of_day_hour",
                datetime.now().hour,  # default to current hour
            ),
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
