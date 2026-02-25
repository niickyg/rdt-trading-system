"""
Options Portfolio Risk Manager for the RDT Trading System.

Enforces portfolio-level risk limits for options positions:
- Total premium at risk < 10% of account
- |Net portfolio delta| < 200
- Daily theta < 0.5% of account
- Max 2 options positions per underlying
- Expiration clustering warnings
"""

from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from loguru import logger

from options.models import OptionsStrategy, OptionsPositionSizeResult
from options.config import OptionsConfig
from options.chain import OptionsChainManager


class RiskCheckResult:
    """Result of a portfolio risk check."""

    def __init__(self, passed: bool, reason: str = "", warnings: Optional[List[str]] = None):
        self.passed = passed
        self.reason = reason
        self.warnings = warnings or []

    def __bool__(self):
        return self.passed

    def __repr__(self):
        status = "PASS" if self.passed else f"FAIL: {self.reason}"
        return f"RiskCheck({status})"


class OptionsRiskManager:
    """
    Portfolio-level risk management for options positions.

    Call validate_new_trade() before executing any new options strategy.
    Call get_portfolio_risk() for a dashboard risk summary.
    """

    def __init__(
        self,
        chain_manager: OptionsChainManager,
        config: Optional[OptionsConfig] = None,
    ):
        self._chain = chain_manager
        self._config = config or OptionsConfig()

    def validate_new_trade(
        self,
        strategy: OptionsStrategy,
        size_result: OptionsPositionSizeResult,
        existing_positions: Dict[str, Dict],
        account_size: float,
    ) -> RiskCheckResult:
        """
        Validate a new options trade against portfolio risk limits.

        Args:
            strategy: Proposed strategy
            size_result: Position sizing result
            existing_positions: Current options positions from OptionsExecutor
            account_size: Current account value

        Returns:
            RiskCheckResult (truthy if approved, falsy if rejected)
        """
        warnings = []

        # 1. Check max positions per underlying
        underlying = strategy.underlying
        underlying_count = sum(
            1 for pos in existing_positions.values()
            if pos["strategy"].underlying == underlying
        )
        if underlying_count >= self._config.max_positions_per_underlying:
            return RiskCheckResult(
                False,
                f"Max {self._config.max_positions_per_underlying} positions "
                f"per underlying ({underlying} has {underlying_count})"
            )

        # 2. Check total premium at risk
        total_risk = self._total_premium_at_risk(existing_positions)
        new_risk = size_result.max_risk
        total_after = total_risk + new_risk
        max_allowed = account_size * self._config.max_premium_at_risk_pct

        if total_after > max_allowed:
            return RiskCheckResult(
                False,
                f"Total premium at risk ${total_after:.0f} would exceed "
                f"limit ${max_allowed:.0f} ({self._config.max_premium_at_risk_pct*100:.0f}% of account)"
            )

        # 3. Check net portfolio delta
        current_delta = self._net_portfolio_delta(existing_positions)
        new_delta = strategy.net_delta * size_result.contracts
        total_delta = current_delta + new_delta

        if abs(total_delta) > self._config.max_portfolio_delta:
            return RiskCheckResult(
                False,
                f"|Net delta| {abs(total_delta):.0f} would exceed "
                f"limit {self._config.max_portfolio_delta:.0f}"
            )

        # 4. Check daily theta
        current_theta = self._total_daily_theta(existing_positions)
        new_theta = strategy.net_theta * size_result.contracts
        total_theta = current_theta + new_theta
        max_theta = account_size * self._config.max_daily_theta_pct

        if abs(total_theta) > max_theta:
            return RiskCheckResult(
                False,
                f"|Daily theta| ${abs(total_theta):.2f} would exceed "
                f"limit ${max_theta:.2f}"
            )

        # 5. Expiration clustering warning
        exp_warning = self._check_expiration_clustering(strategy, existing_positions)
        if exp_warning:
            warnings.append(exp_warning)

        # 6. Risk/reward sanity check
        if strategy.is_defined_risk and strategy.risk_reward_ratio < 0.5:
            warnings.append(
                f"Low risk/reward ratio: {strategy.risk_reward_ratio:.2f}"
            )

        if warnings:
            logger.warning(f"Options risk warnings for {underlying}: {'; '.join(warnings)}")

        return RiskCheckResult(True, warnings=warnings)

    def get_portfolio_risk(
        self,
        positions: Dict[str, Dict],
        account_size: float,
    ) -> Dict:
        """
        Get portfolio-level risk metrics for dashboard display.

        Args:
            positions: Current options positions
            account_size: Current account value

        Returns:
            Dict with risk metrics
        """
        total_risk = self._total_premium_at_risk(positions)
        net_delta = self._net_portfolio_delta(positions)
        daily_theta = self._total_daily_theta(positions)
        position_count = len(positions)

        # Expiration distribution
        exp_dist = Counter()
        for pos in positions.values():
            strategy = pos["strategy"]
            if strategy.expiry:
                exp_dist[strategy.expiry] += 1

        return {
            "total_premium_at_risk": total_risk,
            "premium_risk_pct": total_risk / account_size if account_size > 0 else 0,
            "net_portfolio_delta": net_delta,
            "daily_theta": daily_theta,
            "theta_pct": abs(daily_theta) / account_size if account_size > 0 else 0,
            "position_count": position_count,
            "expiration_distribution": dict(exp_dist),
            "limits": {
                "max_premium_risk_pct": self._config.max_premium_at_risk_pct,
                "max_portfolio_delta": self._config.max_portfolio_delta,
                "max_daily_theta_pct": self._config.max_daily_theta_pct,
                "max_per_underlying": self._config.max_positions_per_underlying,
            },
        }

    def _total_premium_at_risk(self, positions: Dict[str, Dict]) -> float:
        """Calculate total premium at risk across all positions."""
        total = 0.0
        for pos in positions.values():
            strategy = pos["strategy"]
            contracts = pos["contracts"]
            total += strategy.max_loss * contracts
        return total

    def _net_portfolio_delta(self, positions: Dict[str, Dict]) -> float:
        """Calculate net portfolio delta across all positions."""
        total = 0.0
        for pos in positions.values():
            strategy = pos["strategy"]
            contracts = pos["contracts"]
            total += strategy.net_delta * contracts
        return total

    def _total_daily_theta(self, positions: Dict[str, Dict]) -> float:
        """Calculate total daily theta across all positions."""
        total = 0.0
        for pos in positions.values():
            strategy = pos["strategy"]
            contracts = pos["contracts"]
            # Theta is already per-day per-contract
            multiplier = strategy.legs[0].contract.multiplier if strategy.legs else 100
            total += strategy.net_theta * contracts * multiplier
        return total

    def _check_expiration_clustering(
        self, new_strategy: OptionsStrategy, positions: Dict[str, Dict]
    ) -> Optional[str]:
        """Check for concentration of positions in a single expiration."""
        new_expiry = new_strategy.expiry
        if not new_expiry:
            return None

        count = 1  # Include the new trade
        for pos in positions.values():
            if pos["strategy"].expiry == new_expiry:
                count += 1

        if count >= 3:
            return f"Expiration clustering: {count} positions expiring {new_expiry}"

        return None
