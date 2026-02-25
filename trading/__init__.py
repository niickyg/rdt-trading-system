"""
Trading module for the RDT Trading System.

Provides:
- Position tracking and management
- Order lifecycle monitoring
- Execution quality tracking and analysis
"""

from .position_tracker import PositionTracker, get_position_tracker
from .order_monitor import OrderMonitor, OrderState, MonitoredOrder, get_order_monitor
from .execution_tracker import (
    ExecutionTracker,
    ExecutionRecord,
    ExecutionQuality,
    SlippageStats,
    get_execution_tracker
)

__all__ = [
    # Position tracking
    'PositionTracker',
    'get_position_tracker',

    # Order monitoring
    'OrderMonitor',
    'OrderState',
    'MonitoredOrder',
    'get_order_monitor',

    # Execution tracking
    'ExecutionTracker',
    'ExecutionRecord',
    'ExecutionQuality',
    'SlippageStats',
    'get_execution_tracker',
]
