"""
Base Agent Class
Foundation for all trading agents in the system
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from loguru import logger

from agents.events import EventBus, Event, EventType, get_event_bus


class AgentState(Enum):
    """Agent lifecycle states"""
    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    events_processed: int = 0
    events_published: int = 0
    errors: int = 0
    last_activity: Optional[datetime] = None
    uptime_seconds: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents

    Provides:
    - Lifecycle management (start, stop, pause, resume)
    - Event subscription and publishing
    - Logging and metrics
    - Error handling
    """

    def __init__(
        self,
        name: str,
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize base agent

        Args:
            name: Unique agent name
            event_bus: Event bus for communication (uses global if not provided)
            config: Agent configuration dictionary
        """
        self.name = name
        self.event_bus = event_bus or get_event_bus()
        self.config = config or {}

        self.state = AgentState.CREATED
        self.metrics = AgentMetrics()
        self._start_time: Optional[datetime] = None
        self._subscriptions: List[EventType] = []

        logger.info(f"Agent created: {self.name}")

    @property
    def is_running(self) -> bool:
        return self.state == AgentState.RUNNING

    # ==================== Lifecycle Methods ====================

    async def start(self):
        """Start the agent"""
        if self.state == AgentState.RUNNING:
            logger.warning(f"{self.name}: Already running")
            return

        logger.info(f"{self.name}: Starting...")
        self.state = AgentState.INITIALIZING

        try:
            # Initialize agent
            await self.initialize()

            # Subscribe to events
            self._setup_subscriptions()

            self.state = AgentState.RUNNING
            self._start_time = datetime.now()

            # Publish start event
            await self.publish(EventType.SYSTEM_START, {
                "agent": self.name,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"{self.name}: Started successfully")

        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"{self.name}: Failed to start - {e}")
            raise

    async def stop(self):
        """Stop the agent"""
        if self.state in (AgentState.STOPPED, AgentState.STOPPING):
            return

        logger.info(f"{self.name}: Stopping...")
        self.state = AgentState.STOPPING

        try:
            # Cleanup
            await self.cleanup()

            # Unsubscribe from events
            self._remove_subscriptions()

            self.state = AgentState.STOPPED

            # Update uptime
            if self._start_time:
                self.metrics.uptime_seconds = (
                    datetime.now() - self._start_time
                ).total_seconds()

            logger.info(f"{self.name}: Stopped")

        except Exception as e:
            logger.error(f"{self.name}: Error during stop - {e}")
            self.state = AgentState.ERROR

    def pause(self):
        """Pause the agent"""
        if self.state == AgentState.RUNNING:
            self.state = AgentState.PAUSED
            logger.info(f"{self.name}: Paused")

    def resume(self):
        """Resume a paused agent"""
        if self.state == AgentState.PAUSED:
            self.state = AgentState.RUNNING
            logger.info(f"{self.name}: Resumed")

    # ==================== Abstract Methods ====================

    @abstractmethod
    async def initialize(self):
        """Initialize agent resources - must be implemented by subclasses"""
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup agent resources - must be implemented by subclasses"""
        pass

    @abstractmethod
    def get_subscribed_events(self) -> List[EventType]:
        """Return list of event types this agent subscribes to"""
        pass

    @abstractmethod
    async def handle_event(self, event: Event):
        """Handle an incoming event - must be implemented by subclasses"""
        pass

    # ==================== Event Methods ====================

    def _setup_subscriptions(self):
        """Subscribe to events defined by get_subscribed_events"""
        self._subscriptions = self.get_subscribed_events()
        for event_type in self._subscriptions:
            self.event_bus.subscribe(event_type, self._on_event)
        logger.debug(f"{self.name}: Subscribed to {len(self._subscriptions)} event types")

    def _remove_subscriptions(self):
        """Unsubscribe from all events"""
        for event_type in self._subscriptions:
            self.event_bus.unsubscribe(event_type, self._on_event)
        self._subscriptions = []

    async def _on_event(self, event: Event):
        """Internal event handler wrapper"""
        if self.state != AgentState.RUNNING:
            return

        try:
            self.metrics.last_activity = datetime.now()
            self.metrics.events_processed += 1
            await self.handle_event(event)

        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"{self.name}: Error handling event {event.event_type} - {e}")
            await self.publish(EventType.SYSTEM_ERROR, {
                "agent": self.name,
                "error": str(e),
                "event": str(event)
            })

    async def publish(self, event_type: EventType, data: Dict = None):
        """Publish an event"""
        event = Event(
            event_type=event_type,
            data=data or {},
            source=self.name
        )
        await self.event_bus.publish(event)
        self.metrics.events_published += 1

    # ==================== Utility Methods ====================

    def get_metrics(self) -> Dict:
        """Get agent metrics as dictionary"""
        return {
            "name": self.name,
            "state": self.state.value,
            "events_processed": self.metrics.events_processed,
            "events_published": self.metrics.events_published,
            "errors": self.metrics.errors,
            "last_activity": self.metrics.last_activity.isoformat() if self.metrics.last_activity else None,
            "uptime_seconds": self.metrics.uptime_seconds,
            **self.metrics.custom_metrics
        }

    def log(self, level: str, message: str):
        """Log with agent name prefix"""
        getattr(logger, level)(f"[{self.name}] {message}")


class ScheduledAgent(BaseAgent):
    """
    Agent that runs on a schedule

    Extends BaseAgent with periodic task execution
    """

    def __init__(
        self,
        name: str,
        interval_seconds: float = 60.0,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.interval_seconds = interval_seconds
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the scheduled agent"""
        await super().start()

        # Start the scheduled task
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        """Stop the scheduled agent"""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        await super().stop()

    async def _run_loop(self):
        """Main loop that runs the scheduled task"""
        while self.state == AgentState.RUNNING:
            try:
                await self.run_scheduled_task()
            except Exception as e:
                self.metrics.errors += 1
                logger.error(f"{self.name}: Scheduled task error - {e}")

            await asyncio.sleep(self.interval_seconds)

    @abstractmethod
    async def run_scheduled_task(self):
        """Task to run on schedule - must be implemented by subclasses"""
        pass
