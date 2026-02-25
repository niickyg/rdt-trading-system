"""
Broker Failover Manager

Manages failover between primary, secondary, and paper trading brokers.
Provides automatic failover on broker failures with health monitoring.

Features:
- Primary/secondary/paper broker management
- Automatic failover on persistent failures
- Health status tracking
- Status reporting
- Manual failover control
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from brokers.broker_interface import (
    BrokerInterface, Order, Position, Quote, AccountInfo,
    OrderSide, OrderType, BrokerError, ConnectionError
)


class BrokerRole(str, Enum):
    """Broker roles in the failover hierarchy"""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    PAPER = "paper"


class BrokerHealthStatus(str, Enum):
    """Health status of a broker"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Intermittent failures
    UNHEALTHY = "unhealthy"  # Persistent failures
    DISCONNECTED = "disconnected"


@dataclass
class BrokerHealth:
    """Health information for a broker"""
    role: BrokerRole
    status: BrokerHealthStatus = BrokerHealthStatus.DISCONNECTED
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    total_failures: int = 0
    total_requests: int = 0
    latency_ms: float = 0.0
    error_message: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.total_requests - self.total_failures) / self.total_requests

    def record_success(self, latency_ms: float = 0.0):
        """Record successful request"""
        self.total_requests += 1
        self.consecutive_failures = 0
        self.last_success = datetime.now()
        self.last_check = datetime.now()
        self.latency_ms = latency_ms
        self.error_message = None

        if self.consecutive_failures == 0:
            self.status = BrokerHealthStatus.HEALTHY

    def record_failure(self, error: str):
        """Record failed request"""
        self.total_requests += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_check = datetime.now()
        self.error_message = error

        if self.consecutive_failures >= 5:
            self.status = BrokerHealthStatus.UNHEALTHY
        elif self.consecutive_failures >= 2:
            self.status = BrokerHealthStatus.DEGRADED


class BrokerFailoverManager:
    """
    Manages failover between multiple brokers.

    Maintains primary, secondary, and paper trading brokers with automatic
    failover on persistent failures. Tracks health status and provides
    status reporting.

    Example:
        manager = BrokerFailoverManager()
        manager.set_broker(BrokerRole.PRIMARY, ibkr_client)
        manager.set_broker(BrokerRole.SECONDARY, schwab_client)
        manager.set_broker(BrokerRole.PAPER, paper_client)

        # Get the best available broker
        broker = manager.get_active_broker()

        # Execute with failover
        result = manager.execute_with_failover(
            lambda b: b.get_quote("AAPL")
        )
    """

    # Thresholds for failover decisions
    FAILOVER_THRESHOLD = 3  # Consecutive failures before failover
    RECOVERY_THRESHOLD = 5  # Consecutive successes before recovery
    HEALTH_CHECK_INTERVAL = 60  # Seconds between health checks

    def __init__(
        self,
        failover_to_paper: bool = True,
        auto_recovery: bool = True
    ):
        """
        Initialize failover manager.

        Args:
            failover_to_paper: Allow failover to paper broker
            auto_recovery: Automatically try to recover failed brokers
        """
        self._brokers: Dict[BrokerRole, BrokerInterface] = {}
        self._health: Dict[BrokerRole, BrokerHealth] = {
            role: BrokerHealth(role=role) for role in BrokerRole
        }
        self._active_role: Optional[BrokerRole] = None
        self._failover_to_paper = failover_to_paper
        self._auto_recovery = auto_recovery
        self._lock = threading.RLock()

        # Recovery monitoring
        self._recovery_thread: Optional[threading.Thread] = None
        self._stop_recovery = threading.Event()

        logger.info("BrokerFailoverManager initialized")

    def set_broker(self, role: BrokerRole, broker: BrokerInterface) -> None:
        """
        Set a broker for a specific role.

        Args:
            role: The role for this broker
            broker: The broker instance
        """
        with self._lock:
            self._brokers[role] = broker
            self._health[role] = BrokerHealth(role=role)

            # If this is the first broker or primary, make it active
            if self._active_role is None or role == BrokerRole.PRIMARY:
                if broker.is_connected:
                    self._active_role = role
                    self._health[role].status = BrokerHealthStatus.HEALTHY
                    logger.info(f"Set {role.value} broker as active")

    def get_broker(self, role: BrokerRole) -> Optional[BrokerInterface]:
        """Get broker by role"""
        return self._brokers.get(role)

    @property
    def active_role(self) -> Optional[BrokerRole]:
        """Get the currently active broker role"""
        return self._active_role

    @property
    def active_broker(self) -> Optional[BrokerInterface]:
        """Get the currently active broker"""
        if self._active_role:
            return self._brokers.get(self._active_role)
        return None

    def get_active_broker(self) -> Optional[BrokerInterface]:
        """
        Get the best available broker.

        Returns the active broker if healthy, otherwise attempts failover.

        Returns:
            Active broker or None if no healthy brokers
        """
        with self._lock:
            if self._active_role:
                broker = self._brokers.get(self._active_role)
                if broker and broker.is_connected:
                    return broker

            # Try to failover
            return self._attempt_failover()

    def _attempt_failover(self) -> Optional[BrokerInterface]:
        """
        Attempt to failover to the next available broker.

        Tries brokers in order: PRIMARY -> SECONDARY -> PAPER

        Returns:
            New active broker or None
        """
        failover_order = [BrokerRole.PRIMARY, BrokerRole.SECONDARY]
        if self._failover_to_paper:
            failover_order.append(BrokerRole.PAPER)

        for role in failover_order:
            if role == self._active_role:
                continue

            broker = self._brokers.get(role)
            if broker:
                try:
                    if not broker.is_connected:
                        if broker.connect():
                            self._active_role = role
                            self._health[role].status = BrokerHealthStatus.HEALTHY
                            logger.warning(f"Failover: Now using {role.value} broker")
                            return broker
                    else:
                        self._active_role = role
                        self._health[role].status = BrokerHealthStatus.HEALTHY
                        logger.warning(f"Failover: Now using {role.value} broker")
                        return broker
                except Exception as e:
                    logger.error(f"Failover to {role.value} failed: {e}")
                    self._health[role].record_failure(str(e))

        logger.error("All brokers unavailable - no failover possible")
        return None

    def execute_with_failover(
        self,
        operation: callable,
        max_retries: int = 2
    ) -> Any:
        """
        Execute an operation with automatic failover on failure.

        Args:
            operation: Function that takes a broker and returns result
            max_retries: Maximum retry attempts per broker

        Returns:
            Result of the operation

        Raises:
            BrokerError: If all brokers fail
        """
        last_error = None
        tried_roles = set()

        while True:
            broker = self.get_active_broker()
            if broker is None:
                raise ConnectionError("No brokers available")

            role = self._active_role
            if role in tried_roles:
                # Already tried this broker, no more options
                break
            tried_roles.add(role)

            for attempt in range(max_retries):
                try:
                    start_time = time.time()
                    result = operation(broker)
                    latency_ms = (time.time() - start_time) * 1000

                    # Record success
                    self._health[role].record_success(latency_ms)
                    return result

                except Exception as e:
                    last_error = e
                    self._health[role].record_failure(str(e))
                    logger.warning(
                        f"Broker {role.value} operation failed "
                        f"(attempt {attempt + 1}/{max_retries}): {e}"
                    )

                    # Check if we should failover
                    if self._health[role].consecutive_failures >= self.FAILOVER_THRESHOLD:
                        logger.warning(
                            f"Broker {role.value} exceeded failure threshold, "
                            f"attempting failover"
                        )
                        self._active_role = None  # Force failover on next get
                        break

        raise BrokerError(f"All brokers failed: {last_error}")

    def force_failover(self, to_role: Optional[BrokerRole] = None) -> bool:
        """
        Force failover to a specific broker or next available.

        Args:
            to_role: Specific role to failover to, or None for next available

        Returns:
            True if failover successful
        """
        with self._lock:
            if to_role:
                broker = self._brokers.get(to_role)
                if broker and broker.is_connected:
                    old_role = self._active_role
                    self._active_role = to_role
                    self._health[to_role].status = BrokerHealthStatus.HEALTHY
                    logger.info(
                        f"Forced failover from {old_role.value if old_role else 'none'} "
                        f"to {to_role.value}"
                    )
                    return True
                return False
            else:
                self._active_role = None
                result = self._attempt_failover()
                return result is not None

    def get_health_status(self) -> Dict[str, Dict]:
        """
        Get health status for all brokers.

        Returns:
            Dictionary with health info for each broker role
        """
        status = {}
        for role, health in self._health.items():
            broker = self._brokers.get(role)
            status[role.value] = {
                "configured": broker is not None,
                "connected": broker.is_connected if broker else False,
                "active": role == self._active_role,
                "status": health.status.value,
                "consecutive_failures": health.consecutive_failures,
                "total_failures": health.total_failures,
                "total_requests": health.total_requests,
                "success_rate": f"{health.success_rate * 100:.1f}%",
                "latency_ms": health.latency_ms,
                "last_success": health.last_success.isoformat() if health.last_success else None,
                "last_check": health.last_check.isoformat() if health.last_check else None,
                "error_message": health.error_message,
            }
        return status

    def print_status(self) -> None:
        """Print status report to console"""
        status = self.get_health_status()
        print("\n=== Broker Failover Status ===")
        print(f"Active Broker: {self._active_role.value if self._active_role else 'None'}")
        print()

        for role, info in status.items():
            active = " [ACTIVE]" if info["active"] else ""
            print(f"{role.upper()}{active}:")
            print(f"  Configured: {info['configured']}")
            print(f"  Connected: {info['connected']}")
            print(f"  Status: {info['status']}")
            print(f"  Success Rate: {info['success_rate']}")
            print(f"  Latency: {info['latency_ms']:.1f}ms")
            if info['error_message']:
                print(f"  Last Error: {info['error_message']}")
            print()

    def start_recovery_monitor(self) -> None:
        """Start background recovery monitoring thread"""
        if self._recovery_thread and self._recovery_thread.is_alive():
            return

        self._stop_recovery.clear()
        self._recovery_thread = threading.Thread(
            target=self._recovery_loop,
            daemon=True,
            name="BrokerRecoveryMonitor"
        )
        self._recovery_thread.start()
        logger.info("Started broker recovery monitor")

    def stop_recovery_monitor(self) -> None:
        """Stop background recovery monitoring"""
        self._stop_recovery.set()
        if self._recovery_thread:
            self._recovery_thread.join(timeout=5)
        logger.info("Stopped broker recovery monitor")

    def _recovery_loop(self) -> None:
        """Background loop to check and recover failed brokers"""
        while not self._stop_recovery.is_set():
            try:
                self._check_broker_health()
                self._attempt_broker_recovery()
            except Exception as e:
                logger.error(f"Recovery monitor error: {e}")

            self._stop_recovery.wait(self.HEALTH_CHECK_INTERVAL)

    def _check_broker_health(self) -> None:
        """Check health of all configured brokers"""
        for role, broker in self._brokers.items():
            if broker is None:
                continue

            try:
                start_time = time.time()
                if broker.is_connected:
                    # Try a simple operation to verify connectivity
                    broker.get_account()
                    latency_ms = (time.time() - start_time) * 1000
                    self._health[role].record_success(latency_ms)
                else:
                    self._health[role].status = BrokerHealthStatus.DISCONNECTED
            except Exception as e:
                self._health[role].record_failure(str(e))

    def _attempt_broker_recovery(self) -> None:
        """Attempt to recover unhealthy brokers"""
        if not self._auto_recovery:
            return

        # Try to recover primary broker first
        primary_health = self._health.get(BrokerRole.PRIMARY)
        primary_broker = self._brokers.get(BrokerRole.PRIMARY)

        if (primary_health and primary_broker and
            primary_health.status in (BrokerHealthStatus.UNHEALTHY, BrokerHealthStatus.DISCONNECTED)):

            try:
                logger.info("Attempting to recover primary broker...")
                if not primary_broker.is_connected:
                    primary_broker.connect()

                if primary_broker.is_connected:
                    primary_health.status = BrokerHealthStatus.HEALTHY
                    primary_health.consecutive_failures = 0
                    logger.info("Primary broker recovered")

                    # If we're not using primary, try to failback
                    if self._active_role != BrokerRole.PRIMARY:
                        logger.info("Failing back to primary broker")
                        self._active_role = BrokerRole.PRIMARY

            except Exception as e:
                logger.warning(f"Primary broker recovery failed: {e}")

    def shutdown(self) -> None:
        """Shutdown all brokers and cleanup"""
        self.stop_recovery_monitor()

        for role, broker in self._brokers.items():
            if broker:
                try:
                    broker.disconnect()
                    logger.info(f"Disconnected {role.value} broker")
                except Exception as e:
                    logger.error(f"Error disconnecting {role.value} broker: {e}")

        self._brokers.clear()
        self._active_role = None
        logger.info("Broker failover manager shutdown complete")


# Singleton instance
_failover_manager: Optional[BrokerFailoverManager] = None


def get_failover_manager() -> BrokerFailoverManager:
    """Get or create the global failover manager instance"""
    global _failover_manager
    if _failover_manager is None:
        _failover_manager = BrokerFailoverManager()
    return _failover_manager
