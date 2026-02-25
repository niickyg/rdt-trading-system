"""
Execution Tracker for the RDT Trading System.

Provides fill quality analysis with:
- Slippage tracking (expected vs actual fill price)
- Fill rate monitoring
- Partial fill tracking
- Database persistence for execution history
- Statistical analysis of execution quality
- Distributed tracing integration
"""

import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

# Import distributed tracing
try:
    from tracing import trace, get_tracer, get_current_span
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    # Create no-op decorators
    def trace(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def get_tracer():
        return None
    def get_current_span():
        return None

# Database support
try:
    from data.database import get_db_manager
    from data.database.models import OrderExecution, OrderExecutionStatus
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logger.warning("Database not available for execution persistence")

# Prometheus metrics support
try:
    from monitoring.metrics import (
        rdt_order_fill_time_seconds,
        rdt_order_slippage_pct,
        rdt_order_fill_rate,
        record_execution_metrics
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logger.debug("Prometheus execution metrics not available")


class ExecutionQuality(str, Enum):
    """Execution quality classification."""
    EXCELLENT = "excellent"  # < 0.05% slippage
    GOOD = "good"           # 0.05-0.1% slippage
    FAIR = "fair"           # 0.1-0.25% slippage
    POOR = "poor"           # 0.25-0.5% slippage
    VERY_POOR = "very_poor" # > 0.5% slippage


@dataclass
class ExecutionRecord:
    """Record of an order execution."""
    order_id: str
    symbol: str
    side: str  # buy, sell, buy_to_cover, sell_short
    expected_price: float
    fill_price: float
    quantity: int
    filled_quantity: int
    fill_time: datetime
    order_submitted_at: datetime
    status: str  # filled, partial, cancelled, rejected

    # Calculated fields
    slippage: float = field(init=False)
    slippage_pct: float = field(init=False)
    slippage_bps: float = field(init=False)  # Basis points
    fill_rate: float = field(init=False)
    time_to_fill_seconds: float = field(init=False)
    quality: ExecutionQuality = field(init=False)

    def __post_init__(self):
        """Calculate derived fields."""
        # Slippage calculations
        # Positive slippage always means unfavorable
        if self.side in ('buy', 'BUY', 'BUY_TO_COVER', 'buy_to_cover'):
            self.slippage = self.fill_price - self.expected_price
        else:
            self.slippage = self.expected_price - self.fill_price
        if self.expected_price != 0:
            self.slippage_pct = (self.slippage / self.expected_price) * 100
            self.slippage_bps = self.slippage_pct * 100  # Basis points
        else:
            self.slippage_pct = 0.0
            self.slippage_bps = 0.0

        # Fill rate
        self.fill_rate = (self.filled_quantity / self.quantity * 100) if self.quantity > 0 else 0.0

        # Time to fill
        self.time_to_fill_seconds = (self.fill_time - self.order_submitted_at).total_seconds()

        # Classify quality based on absolute slippage percentage
        abs_slippage = abs(self.slippage_pct)
        if abs_slippage < 0.05:
            self.quality = ExecutionQuality.EXCELLENT
        elif abs_slippage < 0.1:
            self.quality = ExecutionQuality.GOOD
        elif abs_slippage < 0.25:
            self.quality = ExecutionQuality.FAIR
        elif abs_slippage < 0.5:
            self.quality = ExecutionQuality.POOR
        else:
            self.quality = ExecutionQuality.VERY_POOR

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "expected_price": self.expected_price,
            "fill_price": self.fill_price,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "fill_time": self.fill_time.isoformat(),
            "order_submitted_at": self.order_submitted_at.isoformat(),
            "status": self.status,
            "slippage": round(self.slippage, 4),
            "slippage_pct": round(self.slippage_pct, 4),
            "slippage_bps": round(self.slippage_bps, 2),
            "fill_rate": round(self.fill_rate, 2),
            "time_to_fill_seconds": round(self.time_to_fill_seconds, 3),
            "quality": self.quality.value
        }


@dataclass
class SlippageStats:
    """Statistical summary of slippage."""
    count: int
    avg_slippage_pct: float
    median_slippage_pct: float
    std_slippage_pct: float
    min_slippage_pct: float
    max_slippage_pct: float
    avg_slippage_bps: float
    positive_slippage_count: int  # Paid more than expected
    negative_slippage_count: int  # Paid less than expected (favorable)
    zero_slippage_count: int
    quality_breakdown: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "avg_slippage_pct": round(self.avg_slippage_pct, 4),
            "median_slippage_pct": round(self.median_slippage_pct, 4),
            "std_slippage_pct": round(self.std_slippage_pct, 4),
            "min_slippage_pct": round(self.min_slippage_pct, 4),
            "max_slippage_pct": round(self.max_slippage_pct, 4),
            "avg_slippage_bps": round(self.avg_slippage_bps, 2),
            "positive_slippage_count": self.positive_slippage_count,
            "negative_slippage_count": self.negative_slippage_count,
            "zero_slippage_count": self.zero_slippage_count,
            "quality_breakdown": self.quality_breakdown
        }


class ExecutionTracker:
    """
    Tracks and analyzes order execution quality.

    Features:
    - Records executions with slippage calculations
    - Provides statistical analysis of fill quality
    - Persists execution history to database
    - Integrates with Prometheus metrics

    Usage:
        tracker = ExecutionTracker()

        # Record an execution
        record = tracker.record_execution(
            order_id="order123",
            symbol="AAPL",
            side="buy",
            expected_price=185.50,
            fill_price=185.52,
            quantity=100,
            filled_quantity=100,
            fill_time=datetime.utcnow(),
            order_submitted_at=submitted_time
        )

        # Get statistics
        stats = tracker.get_slippage_stats()
        fill_rate = tracker.get_fill_rate()
    """

    def __init__(self, persist_to_db: bool = True, max_history: int = 10000):
        """
        Initialize Execution Tracker.

        Args:
            persist_to_db: Whether to save executions to database
            max_history: Maximum number of executions to keep in memory
        """
        self._executions: List[ExecutionRecord] = []
        self._persist_to_db = persist_to_db and DATABASE_AVAILABLE
        self._max_history = max_history

        # Per-symbol tracking
        self._symbol_executions: Dict[str, List[ExecutionRecord]] = {}

        # Load recent executions from database
        if self._persist_to_db:
            self._load_from_db()

        logger.info("ExecutionTracker initialized")

    def _load_from_db(self) -> None:
        """Load recent executions from database."""
        try:
            db_manager = get_db_manager()
            with db_manager.get_session() as session:
                from sqlalchemy import desc
                executions = session.query(OrderExecution).order_by(
                    desc(OrderExecution.fill_time)
                ).limit(self._max_history).all()

                for ex in executions:
                    record = ExecutionRecord(
                        order_id=ex.order_id,
                        symbol=ex.symbol,
                        side=ex.side,
                        expected_price=ex.expected_price,
                        fill_price=ex.fill_price,
                        quantity=ex.quantity,
                        filled_quantity=ex.filled_quantity,
                        fill_time=ex.fill_time,
                        order_submitted_at=ex.order_submitted_at,
                        status=ex.status.value if hasattr(ex.status, 'value') else ex.status
                    )
                    self._executions.append(record)

                    # Update per-symbol tracking
                    if record.symbol not in self._symbol_executions:
                        self._symbol_executions[record.symbol] = []
                    self._symbol_executions[record.symbol].append(record)

                logger.info(f"Loaded {len(executions)} executions from database")
        except Exception as e:
            logger.error(f"Error loading executions from database: {e}")

    def _save_to_db(self, record: ExecutionRecord) -> bool:
        """Save execution record to database."""
        if not self._persist_to_db:
            return False

        try:
            db_manager = get_db_manager()
            with db_manager.get_session() as session:
                # Map status string to enum
                from data.database.models import OrderExecutionStatus
                status_map = {
                    'filled': OrderExecutionStatus.FILLED,
                    'partial': OrderExecutionStatus.PARTIAL_FILL,
                    'cancelled': OrderExecutionStatus.CANCELLED,
                    'rejected': OrderExecutionStatus.REJECTED
                }
                status_enum = status_map.get(record.status, OrderExecutionStatus.FILLED)

                execution = OrderExecution(
                    order_id=record.order_id,
                    symbol=record.symbol,
                    side=record.side,
                    expected_price=record.expected_price,
                    fill_price=record.fill_price,
                    quantity=record.quantity,
                    filled_quantity=record.filled_quantity,
                    slippage=record.slippage,
                    slippage_pct=record.slippage_pct,
                    fill_time=record.fill_time,
                    order_submitted_at=record.order_submitted_at,
                    time_to_fill_seconds=record.time_to_fill_seconds,
                    status=status_enum
                )
                session.add(execution)
                return True
        except Exception as e:
            logger.error(f"Error saving execution to database: {e}")
            return False

    @trace("execution.record", capture_args=["order_id", "symbol", "side"], attributes={"component": "execution_tracker"})
    def record_execution(
        self,
        order_id: str,
        symbol: str,
        side: str,
        expected_price: float,
        fill_price: float,
        quantity: int,
        filled_quantity: int,
        fill_time: datetime,
        order_submitted_at: datetime,
        status: str = "filled"
    ) -> ExecutionRecord:
        """
        Record an order execution.

        Args:
            order_id: Unique order identifier
            symbol: Stock symbol
            side: Order side (buy, sell, etc.)
            expected_price: Price expected at order submission
            fill_price: Actual average fill price
            quantity: Total order quantity
            filled_quantity: Quantity actually filled
            fill_time: Time of fill completion
            order_submitted_at: Time order was submitted
            status: Execution status (filled, partial, cancelled, rejected)

        Returns:
            ExecutionRecord with calculated metrics
        """
        record = ExecutionRecord(
            order_id=order_id,
            symbol=symbol.upper(),
            side=side.lower(),
            expected_price=expected_price,
            fill_price=fill_price,
            quantity=quantity,
            filled_quantity=filled_quantity,
            fill_time=fill_time,
            order_submitted_at=order_submitted_at,
            status=status
        )

        # Add to in-memory history
        self._executions.append(record)

        # Trim history if needed
        if len(self._executions) > self._max_history:
            self._executions = self._executions[-self._max_history:]

        # Update per-symbol tracking
        if record.symbol not in self._symbol_executions:
            self._symbol_executions[record.symbol] = []
        self._symbol_executions[record.symbol].append(record)

        # Persist to database
        self._save_to_db(record)

        # Update Prometheus metrics
        if METRICS_AVAILABLE:
            try:
                record_execution_metrics(
                    symbol=record.symbol,
                    side=record.side,
                    slippage_pct=record.slippage_pct,
                    fill_time_seconds=record.time_to_fill_seconds,
                    fill_rate=record.fill_rate
                )
            except Exception as e:
                logger.debug(f"Error updating execution metrics: {e}")

        # Add tracing attributes
        span = get_current_span()
        if span:
            span.set_attribute("execution.order_id", order_id)
            span.set_attribute("execution.symbol", symbol)
            span.set_attribute("execution.side", side)
            span.set_attribute("execution.expected_price", expected_price)
            span.set_attribute("execution.fill_price", fill_price)
            span.set_attribute("execution.quantity", quantity)
            span.set_attribute("execution.filled_quantity", filled_quantity)
            span.set_attribute("execution.slippage_pct", record.slippage_pct)
            span.set_attribute("execution.slippage_bps", record.slippage_bps)
            span.set_attribute("execution.quality", record.quality.value)
            span.set_attribute("execution.time_to_fill_seconds", record.time_to_fill_seconds)

        logger.info(
            f"Recorded execution: {order_id} {side} {filled_quantity}/{quantity} "
            f"{symbol} @ ${fill_price:.2f} (expected: ${expected_price:.2f}, "
            f"slippage: {record.slippage_pct:.4f}%, quality: {record.quality.value})"
        )

        return record

    def get_slippage_stats(
        self,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        days: Optional[int] = None,
        status: Optional[str] = None
    ) -> SlippageStats:
        """
        Get slippage statistics.

        Args:
            symbol: Filter by symbol
            side: Filter by side (buy, sell)
            days: Only consider executions within this many days
            status: Filter by status

        Returns:
            SlippageStats with statistical summary
        """
        # Filter executions
        executions = self._filter_executions(symbol, side, days, status)

        if not executions:
            return SlippageStats(
                count=0,
                avg_slippage_pct=0.0,
                median_slippage_pct=0.0,
                std_slippage_pct=0.0,
                min_slippage_pct=0.0,
                max_slippage_pct=0.0,
                avg_slippage_bps=0.0,
                positive_slippage_count=0,
                negative_slippage_count=0,
                zero_slippage_count=0,
                quality_breakdown={}
            )

        slippage_values = [ex.slippage_pct for ex in executions]

        # Calculate statistics
        avg_slippage = statistics.mean(slippage_values)
        median_slippage = statistics.median(slippage_values)
        std_slippage = statistics.stdev(slippage_values) if len(slippage_values) > 1 else 0.0
        min_slippage = min(slippage_values)
        max_slippage = max(slippage_values)

        # Count by direction
        positive = sum(1 for s in slippage_values if s > 0.0001)  # Small threshold for "zero"
        negative = sum(1 for s in slippage_values if s < -0.0001)
        zero = len(slippage_values) - positive - negative

        # Quality breakdown
        quality_breakdown = {q.value: 0 for q in ExecutionQuality}
        for ex in executions:
            quality_breakdown[ex.quality.value] += 1

        return SlippageStats(
            count=len(executions),
            avg_slippage_pct=avg_slippage,
            median_slippage_pct=median_slippage,
            std_slippage_pct=std_slippage,
            min_slippage_pct=min_slippage,
            max_slippage_pct=max_slippage,
            avg_slippage_bps=avg_slippage * 100,
            positive_slippage_count=positive,
            negative_slippage_count=negative,
            zero_slippage_count=zero,
            quality_breakdown=quality_breakdown
        )

    def get_fill_rate(
        self,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get fill rate statistics.

        Args:
            symbol: Filter by symbol
            side: Filter by side
            days: Only consider executions within this many days

        Returns:
            Dictionary with fill rate statistics
        """
        executions = self._filter_executions(symbol, side, days)

        if not executions:
            return {
                "count": 0,
                "fully_filled_count": 0,
                "partial_fill_count": 0,
                "cancelled_count": 0,
                "rejected_count": 0,
                "avg_fill_rate_pct": 0.0,
                "full_fill_rate_pct": 0.0,
                "total_quantity_ordered": 0,
                "total_quantity_filled": 0,
                "overall_fill_rate_pct": 0.0
            }

        # Count by status
        fully_filled = sum(1 for ex in executions if ex.fill_rate >= 100.0)
        partial = sum(1 for ex in executions if 0 < ex.fill_rate < 100.0)
        cancelled = sum(1 for ex in executions if ex.status == "cancelled")
        rejected = sum(1 for ex in executions if ex.status == "rejected")

        # Fill rate calculations
        fill_rates = [ex.fill_rate for ex in executions if ex.fill_rate > 0]
        avg_fill_rate = statistics.mean(fill_rates) if fill_rates else 0.0

        total_ordered = sum(ex.quantity for ex in executions)
        total_filled = sum(ex.filled_quantity for ex in executions)
        overall_fill_rate = (total_filled / total_ordered * 100) if total_ordered > 0 else 0.0

        full_fill_rate = (fully_filled / len(executions) * 100) if executions else 0.0

        return {
            "count": len(executions),
            "fully_filled_count": fully_filled,
            "partial_fill_count": partial,
            "cancelled_count": cancelled,
            "rejected_count": rejected,
            "avg_fill_rate_pct": round(avg_fill_rate, 2),
            "full_fill_rate_pct": round(full_fill_rate, 2),
            "total_quantity_ordered": total_ordered,
            "total_quantity_filled": total_filled,
            "overall_fill_rate_pct": round(overall_fill_rate, 2)
        }

    def get_time_to_fill_stats(
        self,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get time-to-fill statistics.

        Args:
            symbol: Filter by symbol
            side: Filter by side
            days: Only consider executions within this many days

        Returns:
            Dictionary with time-to-fill statistics
        """
        executions = self._filter_executions(symbol, side, days, status="filled")

        if not executions:
            return {
                "count": 0,
                "avg_seconds": 0.0,
                "median_seconds": 0.0,
                "min_seconds": 0.0,
                "max_seconds": 0.0,
                "std_seconds": 0.0,
                "under_1_second_count": 0,
                "under_5_seconds_count": 0,
                "under_30_seconds_count": 0,
                "over_60_seconds_count": 0
            }

        fill_times = [ex.time_to_fill_seconds for ex in executions]

        under_1 = sum(1 for t in fill_times if t < 1.0)
        under_5 = sum(1 for t in fill_times if t < 5.0)
        under_30 = sum(1 for t in fill_times if t < 30.0)
        over_60 = sum(1 for t in fill_times if t >= 60.0)

        return {
            "count": len(fill_times),
            "avg_seconds": round(statistics.mean(fill_times), 3),
            "median_seconds": round(statistics.median(fill_times), 3),
            "min_seconds": round(min(fill_times), 3),
            "max_seconds": round(max(fill_times), 3),
            "std_seconds": round(statistics.stdev(fill_times), 3) if len(fill_times) > 1 else 0.0,
            "under_1_second_count": under_1,
            "under_5_seconds_count": under_5,
            "under_30_seconds_count": under_30,
            "over_60_seconds_count": over_60
        }

    def get_execution_history(
        self,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        days: Optional[int] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get execution history.

        Args:
            symbol: Filter by symbol
            side: Filter by side
            days: Only consider executions within this many days
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of execution records as dictionaries
        """
        executions = self._filter_executions(symbol, side, days)

        # Sort by fill time descending (most recent first)
        executions.sort(key=lambda x: x.fill_time, reverse=True)

        # Apply pagination
        paginated = executions[offset:offset + limit]

        return [ex.to_dict() for ex in paginated]

    def get_symbol_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get execution statistics grouped by symbol.

        Returns:
            Dictionary mapping symbols to their stats
        """
        results = {}

        for symbol, executions in self._symbol_executions.items():
            if not executions:
                continue

            slippage_values = [ex.slippage_pct for ex in executions]
            fill_rates = [ex.fill_rate for ex in executions]
            fill_times = [ex.time_to_fill_seconds for ex in executions]

            results[symbol] = {
                "execution_count": len(executions),
                "avg_slippage_pct": round(statistics.mean(slippage_values), 4),
                "avg_fill_rate_pct": round(statistics.mean(fill_rates), 2),
                "avg_fill_time_seconds": round(statistics.mean(fill_times), 3),
                "total_volume": sum(ex.filled_quantity for ex in executions),
                "quality_distribution": self._get_quality_distribution(executions)
            }

        return results

    def get_recent_poor_executions(
        self,
        threshold_pct: float = 0.25,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent executions with slippage above threshold.

        Args:
            threshold_pct: Slippage threshold (absolute value)
            limit: Maximum number of records to return

        Returns:
            List of poor execution records
        """
        poor_executions = [
            ex for ex in self._executions
            if abs(ex.slippage_pct) >= threshold_pct
        ]

        # Sort by fill time descending
        poor_executions.sort(key=lambda x: x.fill_time, reverse=True)

        return [ex.to_dict() for ex in poor_executions[:limit]]

    def _filter_executions(
        self,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        days: Optional[int] = None,
        status: Optional[str] = None
    ) -> List[ExecutionRecord]:
        """Filter executions based on criteria."""
        executions = self._executions.copy()

        # Filter by symbol
        if symbol:
            symbol = symbol.upper()
            executions = [ex for ex in executions if ex.symbol == symbol]

        # Filter by side
        if side:
            side = side.lower()
            executions = [ex for ex in executions if ex.side == side]

        # Filter by days
        if days:
            cutoff = datetime.utcnow() - timedelta(days=days)
            executions = [ex for ex in executions if ex.fill_time >= cutoff]

        # Filter by status
        if status:
            status = status.lower()
            executions = [ex for ex in executions if ex.status == status]

        return executions

    def _get_quality_distribution(
        self,
        executions: List[ExecutionRecord]
    ) -> Dict[str, int]:
        """Get quality distribution for a list of executions."""
        distribution = {q.value: 0 for q in ExecutionQuality}
        for ex in executions:
            distribution[ex.quality.value] += 1
        return distribution

    def get_overall_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive execution statistics.

        Returns:
            Dictionary with all execution metrics
        """
        slippage_stats = self.get_slippage_stats()
        fill_rate_stats = self.get_fill_rate()
        time_stats = self.get_time_to_fill_stats()

        return {
            "total_executions": len(self._executions),
            "slippage": slippage_stats.to_dict(),
            "fill_rate": fill_rate_stats,
            "time_to_fill": time_stats,
            "symbols_tracked": len(self._symbol_executions),
            "recent_poor_executions_count": len(self.get_recent_poor_executions())
        }


# Global execution tracker instance
_execution_tracker: Optional[ExecutionTracker] = None


def get_execution_tracker() -> ExecutionTracker:
    """Get or create the global ExecutionTracker instance."""
    global _execution_tracker
    if _execution_tracker is None:
        _execution_tracker = ExecutionTracker()
    return _execution_tracker
