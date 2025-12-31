"""
Trading Agents Module
Multi-agent system for autonomous trading
"""

from agents.events import (
    EventType,
    Event,
    EventBus,
    get_event_bus,
    subscribe,
    emit
)
from agents.base import (
    BaseAgent,
    ScheduledAgent,
    AgentState,
    AgentMetrics
)
from agents.scanner_agent import ScannerAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.executor_agent import ExecutorAgent
from agents.orchestrator import Orchestrator, run_trading_system

__all__ = [
    # Events
    "EventType",
    "Event",
    "EventBus",
    "get_event_bus",
    "subscribe",
    "emit",
    # Base
    "BaseAgent",
    "ScheduledAgent",
    "AgentState",
    "AgentMetrics",
    # Agents
    "ScannerAgent",
    "AnalyzerAgent",
    "ExecutorAgent",
    "Orchestrator",
    # Runner
    "run_trading_system"
]
