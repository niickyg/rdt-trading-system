"""
Options Exit Manager for the RDT Trading System.

Monitors open options positions and triggers exits based on:
1. Profit target (50% max profit for spreads, 100% gain for long options)
2. Stop loss (50% premium loss)
3. Time stop (DTE < 14 — accelerating theta decay)
4. Delta breach (net delta > 0.80 — deep ITM)
5. IV crush (IV drops > 20% from entry)
6. Roll trigger (DTE < 21 and profitable)
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from loguru import logger
from utils.timezone import get_eastern_time

from options.models import OptionAction, OptionsStrategy, IVRegime
from options.config import OptionsConfig
from options.chain import OptionsChainManager


class ExitSignal:
    """Represents an exit trigger with reason and priority."""

    def __init__(self, symbol: str, reason: str, priority: int, action: str = "close"):
        """
        Args:
            symbol: Underlying symbol
            reason: Human-readable exit reason
            priority: Priority (1=highest). Used to resolve multiple triggers.
            action: "close" or "roll"
        """
        self.symbol = symbol
        self.reason = reason
        self.priority = priority
        self.action = action
        self.timestamp = get_eastern_time()

    def __repr__(self):
        return f"ExitSignal({self.symbol}, {self.reason}, priority={self.priority})"


class OptionsExitManager:
    """
    Monitors options positions and generates exit signals.

    Call check_exits() periodically (e.g., every price update cycle).
    """

    def __init__(
        self,
        chain_manager: OptionsChainManager,
        config: Optional[OptionsConfig] = None,
    ):
        self._chain = chain_manager
        self._config = config or OptionsConfig()

    def check_exits(self, positions: Dict[str, Dict]) -> List[ExitSignal]:
        """
        Check all open options positions for exit triggers.

        Args:
            positions: Dict from OptionsExecutor.get_all_positions()
                Each value has: strategy, contracts, entry_time,
                entry_premium, entry_iv, entry_delta

        Returns:
            List of ExitSignal objects (sorted by priority, highest first)
        """
        signals = []

        for symbol, position in positions.items():
            try:
                pos_signals = self._check_position(symbol, position)
                signals.extend(pos_signals)
            except Exception as e:
                logger.error(f"Exit check failed for {symbol}: {e}")

        # Sort by priority (lower number = higher priority)
        signals.sort(key=lambda s: s.priority)
        return signals

    def _check_position(self, symbol: str, position: Dict) -> List[ExitSignal]:
        """Check a single position for all exit triggers."""
        signals = []
        strategy: OptionsStrategy = position["strategy"]
        contracts = position["contracts"]
        entry_premium = position.get("entry_premium", 0)
        entry_iv = position.get("entry_iv", 0)

        # Get current Greeks for all legs
        current_value = self._get_current_strategy_value(strategy)
        if current_value is None:
            return signals

        current_premium, current_greeks = current_value

        # 1. Profit target check
        profit_signal = self._check_profit_target(
            symbol, strategy, entry_premium, current_premium, contracts
        )
        if profit_signal:
            signals.append(profit_signal)

        # 2. Stop loss check
        loss_signal = self._check_stop_loss(
            symbol, strategy, entry_premium, current_premium, contracts
        )
        if loss_signal:
            signals.append(loss_signal)

        # 3. Time stop (DTE check)
        time_signal = self._check_time_stop(symbol, strategy)
        if time_signal:
            signals.append(time_signal)

        # 4. Delta breach
        delta_signal = self._check_delta_breach(symbol, current_greeks)
        if delta_signal:
            signals.append(delta_signal)

        # 5. IV crush
        iv_signal = self._check_iv_crush(symbol, entry_iv, current_greeks)
        if iv_signal:
            signals.append(iv_signal)

        # 6. Roll trigger (lower priority — only if profitable and near expiry)
        roll_signal = self._check_roll_trigger(
            symbol, strategy, entry_premium, current_premium
        )
        if roll_signal:
            signals.append(roll_signal)

        return signals

    def _check_profit_target(
        self, symbol: str, strategy: OptionsStrategy,
        entry_premium: float, current_premium: float, contracts: int
    ) -> Optional[ExitSignal]:
        """Check if profit target is reached."""
        if strategy.name in ("long_call", "long_put"):
            # Long options: close at 100% gain
            target_pct = self._config.long_option_profit_target_pct
            if entry_premium > 0:
                pnl_pct = (current_premium - entry_premium) / entry_premium
                if pnl_pct >= target_pct:
                    return ExitSignal(
                        symbol, f"Profit target reached ({pnl_pct:.0%} gain)", priority=1
                    )
        else:
            # Spreads: close at 50% of max profit
            target_pct = self._config.profit_target_pct
            if strategy.max_profit > 0:
                current_pnl = self._calculate_pnl(strategy, entry_premium, current_premium, contracts)
                max_profit = strategy.max_profit * contracts
                if max_profit > 0 and current_pnl / max_profit >= target_pct:
                    return ExitSignal(
                        symbol, f"Profit target ({target_pct:.0%} of max) reached", priority=1
                    )

        return None

    def _check_stop_loss(
        self, symbol: str, strategy: OptionsStrategy,
        entry_premium: float, current_premium: float, contracts: int
    ) -> Optional[ExitSignal]:
        """Check if stop loss is triggered."""
        stop_pct = self._config.stop_loss_pct

        if strategy.name in ("long_call", "long_put"):
            # Long options: stop at 50% premium loss
            if entry_premium > 0:
                loss_pct = (entry_premium - current_premium) / entry_premium
                if loss_pct >= stop_pct:
                    return ExitSignal(
                        symbol, f"Stop loss triggered ({loss_pct:.0%} premium loss)", priority=2
                    )
        else:
            # Spreads: stop at 50% of max loss (or premium loss)
            current_pnl = self._calculate_pnl(strategy, entry_premium, current_premium, contracts)
            max_loss = strategy.max_loss * contracts
            if max_loss > 0 and abs(current_pnl) / max_loss >= stop_pct and current_pnl < 0:
                return ExitSignal(
                    symbol, f"Stop loss ({stop_pct:.0%} of max loss) triggered", priority=2
                )

        return None

    def _check_time_stop(
        self, symbol: str, strategy: OptionsStrategy
    ) -> Optional[ExitSignal]:
        """Check if DTE is below time stop threshold."""
        if not strategy.legs:
            return None

        expiry_str = strategy.expiry
        if not expiry_str:
            return None

        try:
            expiry_date = datetime.strptime(expiry_str, "%Y%m%d").date()
            today = get_eastern_time().date()
            dte = (expiry_date - today).days

            if dte < self._config.time_stop_dte:
                return ExitSignal(
                    symbol, f"Time stop: {dte} DTE (threshold: {self._config.time_stop_dte})",
                    priority=3
                )
        except ValueError:
            pass

        return None

    def _check_delta_breach(
        self, symbol: str, current_greeks: Dict
    ) -> Optional[ExitSignal]:
        """Check if net delta exceeds threshold (deep ITM)."""
        net_delta = current_greeks.get("net_delta", 0)
        threshold = self._config.delta_breach_threshold

        if abs(net_delta) > threshold:
            return ExitSignal(
                symbol, f"Delta breach: |{net_delta:.2f}| > {threshold}",
                priority=4
            )

        return None

    def _check_iv_crush(
        self, symbol: str, entry_iv: float, current_greeks: Dict
    ) -> Optional[ExitSignal]:
        """Check if IV has dropped significantly from entry (for long premium positions)."""
        if entry_iv <= 0:
            return None

        current_iv = current_greeks.get("avg_iv", 0)
        if current_iv <= 0:
            return None

        iv_change = (entry_iv - current_iv) / entry_iv
        threshold = self._config.iv_crush_threshold

        if iv_change >= threshold:
            return ExitSignal(
                symbol, f"IV crush: IV dropped {iv_change:.0%} from entry",
                priority=5
            )

        return None

    def _check_roll_trigger(
        self, symbol: str, strategy: OptionsStrategy,
        entry_premium: float, current_premium: float
    ) -> Optional[ExitSignal]:
        """Check if position should be rolled (near expiry + profitable)."""
        if not strategy.legs:
            return None

        expiry_str = strategy.expiry
        if not expiry_str:
            return None

        try:
            expiry_date = datetime.strptime(expiry_str, "%Y%m%d").date()
            today = get_eastern_time().date()
            dte = (expiry_date - today).days

            if dte < self._config.roll_dte_threshold:
                # Only recommend roll if profitable (handles both credit and debit strategies)
                pnl = self._calculate_pnl(strategy, entry_premium, current_premium, 1)
                if pnl > 0:
                    return ExitSignal(
                        symbol, f"Roll recommended: {dte} DTE, currently profitable",
                        priority=6, action="roll"
                    )
        except ValueError:
            pass

        return None

    def _get_current_strategy_value(
        self, strategy: OptionsStrategy
    ) -> Optional[Tuple[float, Dict]]:
        """
        Get current value and Greeks for a strategy.

        Returns:
            Tuple of (current_premium_per_contract, greeks_dict) or None
        """
        total_premium = 0.0
        total_delta = 0.0
        total_iv = 0.0
        iv_count = 0

        for leg in strategy.legs:
            greeks = self._chain.get_greeks(leg.contract)
            if greeks is None:
                logger.warning(f"Cannot get current Greeks for {leg.contract.display_name}")
                return None

            mid = greeks.mid_price
            if leg.action == OptionAction.BUY:
                total_premium += mid  # We hold this, it's worth mid
            else:
                total_premium -= mid  # We're short, closing costs mid

            sign = 1 if leg.action == OptionAction.BUY else -1
            total_delta += greeks.delta * leg.quantity * sign

            if greeks.implied_vol > 0:
                total_iv += greeks.implied_vol
                iv_count += 1

        avg_iv = total_iv / iv_count if iv_count > 0 else 0

        return total_premium, {
            "net_delta": total_delta,
            "avg_iv": avg_iv,
        }

    def _calculate_pnl(
        self, strategy: OptionsStrategy,
        entry_premium: float, current_premium: float, contracts: int
    ) -> float:
        """Calculate current P&L for a strategy.

        For debit strategies: we paid entry_premium, position is now worth current_premium.
            PnL = (current_premium - entry_premium) * multiplier * contracts
        For credit strategies: we received entry_premium, current_premium is the net
            position value (negative because we'd pay to close).
            PnL = (entry_premium + current_premium) * multiplier * contracts
        """
        multiplier = strategy.legs[0].contract.multiplier if strategy.legs else 100

        if strategy.is_credit:
            # Credit spread: received entry_premium, current_premium is negative (liability)
            pnl = (entry_premium + current_premium) * multiplier * contracts
        else:
            # Debit spread: paid entry_premium, current_premium is current value
            pnl = (current_premium - entry_premium) * multiplier * contracts

        return pnl
