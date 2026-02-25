"""
Portfolio Aggregator for Multiple Trading Accounts.

Provides aggregated views of positions and performance across all accounts.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

from accounts.account_manager import AccountManager
from brokers import Position


@dataclass
class AggregatedPosition:
    """
    Represents a position aggregated across multiple accounts.

    Attributes:
        symbol: Stock ticker
        total_quantity: Total shares across all accounts
        total_market_value: Total current market value
        total_cost_basis: Total cost basis
        total_unrealized_pnl: Total unrealized P&L
        total_realized_pnl: Total realized P&L
        weighted_avg_cost: Volume-weighted average cost
        accounts: List of account positions
    """
    symbol: str
    total_quantity: int = 0
    total_market_value: float = 0.0
    total_cost_basis: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    weighted_avg_cost: float = 0.0
    current_price: float = 0.0
    accounts: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.total_cost_basis > 0:
            return (self.total_unrealized_pnl / self.total_cost_basis) * 100
        return 0.0

    @property
    def is_long(self) -> bool:
        """Check if net position is long."""
        return self.total_quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if net position is short."""
        return self.total_quantity < 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "total_quantity": self.total_quantity,
            "total_market_value": self.total_market_value,
            "total_cost_basis": self.total_cost_basis,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "total_realized_pnl": self.total_realized_pnl,
            "weighted_avg_cost": self.weighted_avg_cost,
            "current_price": self.current_price,
            "direction": "long" if self.is_long else ("short" if self.is_short else "flat"),
            "accounts": self.accounts
        }


@dataclass
class AggregatedPerformance:
    """
    Aggregated performance metrics across accounts.

    Attributes:
        total_equity: Total equity across all accounts
        total_cash: Total cash across all accounts
        total_positions_value: Total value of all positions
        total_daily_pnl: Total daily P&L
        total_unrealized_pnl: Total unrealized P&L
        total_realized_pnl: Total realized P&L
        total_buying_power: Total buying power
        accounts: Performance breakdown by account
    """
    total_equity: float = 0.0
    total_cash: float = 0.0
    total_positions_value: float = 0.0
    total_daily_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    total_buying_power: float = 0.0
    accounts: List[Dict[str, Any]] = field(default_factory=list)
    calculated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.total_realized_pnl + self.total_unrealized_pnl

    @property
    def daily_pnl_pct(self) -> float:
        """Calculate daily P&L percentage."""
        if self.total_equity > 0:
            return (self.total_daily_pnl / self.total_equity) * 100
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_equity": self.total_equity,
            "total_cash": self.total_cash,
            "total_positions_value": self.total_positions_value,
            "total_daily_pnl": self.total_daily_pnl,
            "daily_pnl_pct": self.daily_pnl_pct,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_realized_pnl": self.total_realized_pnl,
            "total_pnl": self.total_pnl,
            "total_buying_power": self.total_buying_power,
            "accounts_count": len(self.accounts),
            "accounts": self.accounts,
            "calculated_at": self.calculated_at.isoformat()
        }


class PortfolioAggregator:
    """
    Aggregates portfolio data across multiple trading accounts.

    Provides:
    - Combined portfolio value
    - Aggregated positions (same symbol across accounts)
    - Combined performance metrics
    - Account-filtered views

    Example:
        manager = AccountManager(user_id="user123")
        aggregator = PortfolioAggregator(manager)

        # Get total portfolio value
        total = aggregator.get_total_value()

        # Get combined positions
        positions = aggregator.get_combined_positions()

        # Get performance by specific accounts
        perf = aggregator.get_combined_performance(
            account_ids=["acc_123", "acc_456"]
        )
    """

    def __init__(self, account_manager: AccountManager):
        """
        Initialize portfolio aggregator.

        Args:
            account_manager: AccountManager instance to use
        """
        self.account_manager = account_manager
        self._cache_timeout = 60  # Cache timeout in seconds
        self._position_cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None

        logger.info("PortfolioAggregator initialized")

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache_time is None:
            return False
        elapsed = (datetime.utcnow() - self._cache_time).total_seconds()
        return elapsed < self._cache_timeout

    def _invalidate_cache(self) -> None:
        """Invalidate the cache."""
        self._position_cache = {}
        self._cache_time = None

    def get_total_value(
        self,
        account_ids: Optional[List[str]] = None,
        include_inactive: bool = False
    ) -> float:
        """
        Get total portfolio value across accounts.

        Args:
            account_ids: Optional list of account IDs to include
            include_inactive: Whether to include inactive accounts

        Returns:
            Total equity value
        """
        performance = self.get_combined_performance(
            account_ids=account_ids,
            include_inactive=include_inactive
        )
        return performance.total_equity

    def get_combined_positions(
        self,
        account_ids: Optional[List[str]] = None,
        include_inactive: bool = False,
        symbols: Optional[List[str]] = None
    ) -> List[AggregatedPosition]:
        """
        Get combined positions across accounts.

        Positions in the same symbol across different accounts are aggregated.

        Args:
            account_ids: Optional list of account IDs to include
            include_inactive: Whether to include inactive accounts
            symbols: Optional filter by specific symbols

        Returns:
            List of AggregatedPosition instances
        """
        # Get accounts to include
        if account_ids:
            accounts = [
                self.account_manager.get_account(aid)
                for aid in account_ids
            ]
        else:
            accounts = (
                self.account_manager.get_all_accounts()
                if include_inactive
                else self.account_manager.get_active_accounts()
            )

        # Aggregate positions by symbol
        position_map: Dict[str, AggregatedPosition] = {}

        for account in accounts:
            try:
                broker = self.account_manager.get_broker(account.id, connect=True)
                positions = broker.get_positions()

                for symbol, pos in positions.items():
                    # Filter by symbols if specified
                    if symbols and symbol not in symbols:
                        continue

                    if symbol not in position_map:
                        position_map[symbol] = AggregatedPosition(symbol=symbol)

                    agg = position_map[symbol]

                    # Aggregate quantities and values
                    agg.total_quantity += pos.quantity
                    agg.total_market_value += pos.market_value
                    agg.total_cost_basis += pos.cost_basis
                    agg.total_unrealized_pnl += pos.unrealized_pnl
                    agg.total_realized_pnl += pos.realized_pnl
                    agg.current_price = pos.current_price  # Use latest price

                    # Track per-account details
                    agg.accounts.append({
                        "account_id": account.id,
                        "account_name": account.name,
                        "quantity": pos.quantity,
                        "avg_cost": pos.avg_cost,
                        "market_value": pos.market_value,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "unrealized_pnl_pct": pos.unrealized_pnl_pct
                    })

            except Exception as e:
                logger.warning(f"Failed to get positions for account {account.name}: {e}")
                continue

        # Calculate weighted average costs
        for symbol, agg in position_map.items():
            if agg.total_quantity != 0:
                agg.weighted_avg_cost = agg.total_cost_basis / abs(agg.total_quantity)

        # Sort by total market value (descending)
        sorted_positions = sorted(
            position_map.values(),
            key=lambda x: abs(x.total_market_value),
            reverse=True
        )

        return sorted_positions

    def get_combined_performance(
        self,
        account_ids: Optional[List[str]] = None,
        include_inactive: bool = False
    ) -> AggregatedPerformance:
        """
        Get combined performance metrics across accounts.

        Args:
            account_ids: Optional list of account IDs to include
            include_inactive: Whether to include inactive accounts

        Returns:
            AggregatedPerformance instance
        """
        # Get accounts to include
        if account_ids:
            accounts = [
                self.account_manager.get_account(aid)
                for aid in account_ids
            ]
        else:
            accounts = (
                self.account_manager.get_all_accounts()
                if include_inactive
                else self.account_manager.get_active_accounts()
            )

        performance = AggregatedPerformance()

        for account in accounts:
            try:
                broker = self.account_manager.get_broker(account.id, connect=True)
                account_info = broker.get_account()
                positions = broker.get_positions()

                # Calculate position totals
                positions_unrealized = sum(p.unrealized_pnl for p in positions.values())
                positions_realized = sum(p.realized_pnl for p in positions.values())

                # Aggregate totals
                performance.total_equity += account_info.equity
                performance.total_cash += account_info.cash
                performance.total_positions_value += account_info.positions_value
                performance.total_daily_pnl += account_info.daily_pnl
                performance.total_unrealized_pnl += positions_unrealized
                performance.total_realized_pnl += positions_realized
                performance.total_buying_power += account_info.buying_power

                # Add account details
                performance.accounts.append({
                    "account_id": account.id,
                    "account_name": account.name,
                    "broker_type": account.broker_type.value,
                    "equity": account_info.equity,
                    "cash": account_info.cash,
                    "positions_value": account_info.positions_value,
                    "daily_pnl": account_info.daily_pnl,
                    "unrealized_pnl": positions_unrealized,
                    "buying_power": account_info.buying_power,
                    "positions_count": len(positions)
                })

            except Exception as e:
                logger.warning(
                    f"Failed to get performance for account {account.name}: {e}"
                )
                performance.accounts.append({
                    "account_id": account.id,
                    "account_name": account.name,
                    "broker_type": account.broker_type.value,
                    "error": str(e)
                })

        performance.calculated_at = datetime.utcnow()
        return performance

    def get_position_by_symbol(
        self,
        symbol: str,
        account_ids: Optional[List[str]] = None
    ) -> Optional[AggregatedPosition]:
        """
        Get aggregated position for a specific symbol.

        Args:
            symbol: Stock ticker
            account_ids: Optional list of account IDs to include

        Returns:
            AggregatedPosition or None if no position
        """
        positions = self.get_combined_positions(
            account_ids=account_ids,
            symbols=[symbol.upper()]
        )

        return positions[0] if positions else None

    def get_account_positions(self, account_id: str) -> List[Dict[str, Any]]:
        """
        Get positions for a specific account.

        Args:
            account_id: Account ID

        Returns:
            List of position dictionaries
        """
        account = self.account_manager.get_account(account_id)
        broker = self.account_manager.get_broker(account_id, connect=True)
        positions = broker.get_positions()

        return [
            {
                "symbol": symbol,
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                "realized_pnl": pos.realized_pnl,
                "cost_basis": pos.cost_basis,
                "is_long": pos.is_long,
                "is_short": pos.is_short
            }
            for symbol, pos in positions.items()
        ]

    def get_portfolio_allocation(
        self,
        account_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get portfolio allocation breakdown.

        Args:
            account_ids: Optional list of account IDs to include

        Returns:
            Dictionary with allocation details
        """
        performance = self.get_combined_performance(account_ids=account_ids)
        positions = self.get_combined_positions(account_ids=account_ids)

        total_equity = performance.total_equity
        if total_equity <= 0:
            return {
                "cash_pct": 100.0,
                "positions_pct": 0.0,
                "positions": [],
                "by_account": {}
            }

        # Position allocations
        position_allocations = []
        for pos in positions:
            allocation_pct = (abs(pos.total_market_value) / total_equity) * 100
            position_allocations.append({
                "symbol": pos.symbol,
                "market_value": pos.total_market_value,
                "allocation_pct": allocation_pct,
                "direction": "long" if pos.is_long else "short"
            })

        # Sort by allocation
        position_allocations.sort(key=lambda x: x["allocation_pct"], reverse=True)

        # Account allocations
        account_allocations = {}
        for acc in performance.accounts:
            if "equity" in acc:
                account_allocations[acc["account_id"]] = {
                    "account_name": acc["account_name"],
                    "equity": acc["equity"],
                    "allocation_pct": (acc["equity"] / total_equity) * 100
                }

        cash_pct = (performance.total_cash / total_equity) * 100
        positions_pct = (performance.total_positions_value / total_equity) * 100

        return {
            "total_equity": total_equity,
            "cash": performance.total_cash,
            "cash_pct": cash_pct,
            "positions_value": performance.total_positions_value,
            "positions_pct": positions_pct,
            "positions": position_allocations,
            "by_account": account_allocations,
            "calculated_at": datetime.utcnow().isoformat()
        }

    def get_daily_summary(
        self,
        account_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get daily portfolio summary.

        Args:
            account_ids: Optional list of account IDs to include

        Returns:
            Dictionary with daily summary
        """
        performance = self.get_combined_performance(account_ids=account_ids)
        positions = self.get_combined_positions(account_ids=account_ids)

        # Calculate position metrics
        long_positions = [p for p in positions if p.is_long]
        short_positions = [p for p in positions if p.is_short]

        long_value = sum(p.total_market_value for p in long_positions)
        short_value = sum(abs(p.total_market_value) for p in short_positions)

        # Winners and losers
        winners = [p for p in positions if p.total_unrealized_pnl > 0]
        losers = [p for p in positions if p.total_unrealized_pnl < 0]

        return {
            "date": date.today().isoformat(),
            "total_equity": performance.total_equity,
            "daily_pnl": performance.total_daily_pnl,
            "daily_pnl_pct": performance.daily_pnl_pct,
            "unrealized_pnl": performance.total_unrealized_pnl,
            "positions": {
                "total": len(positions),
                "long": len(long_positions),
                "short": len(short_positions),
                "long_value": long_value,
                "short_value": short_value,
                "net_exposure": long_value - short_value
            },
            "performance": {
                "winners": len(winners),
                "losers": len(losers),
                "best_performer": (
                    max(positions, key=lambda p: p.unrealized_pnl_pct).symbol
                    if positions else None
                ),
                "worst_performer": (
                    min(positions, key=lambda p: p.unrealized_pnl_pct).symbol
                    if positions else None
                )
            },
            "accounts": len(performance.accounts),
            "buying_power": performance.total_buying_power,
            "calculated_at": datetime.utcnow().isoformat()
        }
