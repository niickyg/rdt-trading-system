"""
Locust Load Testing Configuration for RDT Trading System

This module defines user behaviors for load testing:
- APIUser: Tests REST API endpoints
- WebSocketUser: Tests WebSocket connections
- DashboardUser: Simulates dashboard usage patterns

Usage:
    # Start Locust web UI
    locust -f tests/performance/locustfile.py --host=http://localhost:5000

    # Headless mode (for CI/CD)
    locust -f tests/performance/locustfile.py --host=http://localhost:5000 \
           --headless -u 50 -r 5 -t 60s

    # With custom configuration
    python scripts/run_load_test.py --users 100 --spawn-rate 10 --duration 300
"""

import os
import json
import random
import time
from typing import Dict, List, Optional, Any

from locust import HttpUser, TaskSet, task, between, events
from locust.exception import RescheduleTask
from locust.env import Environment

# Try to import WebSocket support
try:
    import socketio
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False

# Try to import the config
try:
    from tests.performance.config import (
        get_performance_config,
        API_ENDPOINTS,
        TEST_SYMBOLS,
        SMALL_WATCHLIST,
    )
except ImportError:
    # Fallback if running standalone
    API_ENDPOINTS = {
        'health': {'method': 'GET', 'path': '/api/v1/health', 'auth_required': False},
        'status': {'method': 'GET', 'path': '/api/v1/status', 'auth_required': False},
        'dashboard': {'method': 'GET', 'path': '/api/v1/dashboard', 'auth_required': False},
        'signals_current': {'method': 'GET', 'path': '/api/v1/signals/current', 'auth_required': True},
        'positions': {'method': 'GET', 'path': '/api/v1/positions', 'auth_required': True},
    }
    TEST_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
    SMALL_WATCHLIST = TEST_SYMBOLS[:5]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_api_key() -> str:
    """Get API key from environment or use test key."""
    return os.getenv('TEST_API_KEY', 'test-api-key-for-load-testing')


def get_auth_headers() -> Dict[str, str]:
    """Get authentication headers."""
    return {
        'X-API-Key': get_api_key(),
        'Content-Type': 'application/json',
    }


# =============================================================================
# API USER - Tests REST API endpoints
# =============================================================================

class APIUserTasks(TaskSet):
    """
    Task set for API user behavior.

    Simulates typical API usage patterns:
    - Frequent: Health checks, signal queries
    - Medium: Position queries, dashboard loads
    - Rare: Backtests, order placement
    """

    def on_start(self):
        """Called when user starts."""
        self.api_key = get_api_key()
        self.auth_headers = get_auth_headers()

    @task(10)
    def get_health(self):
        """Health check - most common request."""
        with self.client.get(
            '/api/v1/health',
            name='GET /api/v1/health',
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get('status') != 'healthy':
                    response.failure(f"Unexpected status: {data.get('status')}")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(8)
    def get_status(self):
        """System status - frequent request."""
        with self.client.get(
            '/api/v1/status',
            name='GET /api/v1/status',
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if 'status' not in data:
                    response.failure("Missing status field")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(15)
    def get_current_signals(self):
        """Get current signals - most common authenticated request."""
        with self.client.get(
            '/api/v1/signals/current',
            headers=self.auth_headers,
            name='GET /api/v1/signals/current',
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if 'signals' not in data and not isinstance(data, list):
                    response.failure("Invalid response format")
            elif response.status_code == 401:
                response.failure("Authentication failed")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    def get_signal_history(self):
        """Get signal history."""
        params = {
            'limit': random.choice([10, 25, 50]),
            'direction': random.choice(['all', 'long', 'short']),
        }
        with self.client.get(
            '/api/v1/signals/history',
            headers=self.auth_headers,
            params=params,
            name='GET /api/v1/signals/history',
            catch_response=True
        ) as response:
            if response.status_code not in [200, 401, 404]:
                response.failure(f"Status code: {response.status_code}")

    @task(8)
    def get_positions(self):
        """Get current positions."""
        with self.client.get(
            '/api/v1/positions',
            headers=self.auth_headers,
            name='GET /api/v1/positions',
            catch_response=True
        ) as response:
            if response.status_code not in [200, 401, 404]:
                response.failure(f"Status code: {response.status_code}")

    @task(3)
    def get_performance(self):
        """Get performance metrics."""
        params = {'days': random.choice([7, 30, 90])}
        with self.client.get(
            '/api/v1/performance',
            headers=self.auth_headers,
            params=params,
            name='GET /api/v1/performance',
            catch_response=True
        ) as response:
            if response.status_code not in [200, 401, 404]:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def get_quote(self):
        """Get stock quote."""
        symbol = random.choice(TEST_SYMBOLS)
        with self.client.get(
            f'/api/v1/quote/{symbol}',
            headers=self.auth_headers,
            name='GET /api/v1/quote/{symbol}',
            catch_response=True
        ) as response:
            if response.status_code not in [200, 401, 404, 500]:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def run_backtest(self):
        """Run a backtest - heavy operation, should be rare."""
        payload = {
            'symbols': random.sample(SMALL_WATCHLIST, 3),
            'start_date': '2024-01-01',
            'end_date': '2024-03-01',
            'initial_capital': 100000,
        }
        with self.client.post(
            '/api/v1/backtest',
            headers=self.auth_headers,
            json=payload,
            name='POST /api/v1/backtest',
            catch_response=True,
            timeout=30  # Backtests can take longer
        ) as response:
            if response.status_code not in [200, 202, 401, 404, 500, 503]:
                response.failure(f"Status code: {response.status_code}")


class APIUser(HttpUser):
    """
    Simulates an API user making REST requests.

    This user type represents programmatic API access,
    such as from trading bots or integration systems.
    """
    tasks = [APIUserTasks]
    wait_time = between(0.5, 2.0)  # Wait between requests
    weight = 3  # Higher weight = more users of this type


# =============================================================================
# DASHBOARD USER - Simulates web dashboard usage
# =============================================================================

class DashboardUserTasks(TaskSet):
    """
    Task set for dashboard user behavior.

    Simulates typical dashboard usage:
    - Initial page load (dashboard data)
    - Periodic refreshes
    - Signal browsing
    - Position monitoring
    """

    def on_start(self):
        """Called when user starts - simulates initial page load."""
        self.api_key = get_api_key()
        self.auth_headers = get_auth_headers()

        # Simulate initial dashboard load
        self.load_dashboard()

    def load_dashboard(self):
        """Load complete dashboard data."""
        self.client.get('/api/v1/dashboard', name='GET /api/v1/dashboard (initial)')

    @task(10)
    def refresh_dashboard(self):
        """Refresh dashboard data - most common action."""
        with self.client.get(
            '/api/v1/dashboard',
            name='GET /api/v1/dashboard',
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    def check_signals(self):
        """Check current signals."""
        with self.client.get(
            '/api/v1/signals/current',
            headers=self.auth_headers,
            name='GET /api/v1/signals/current (dashboard)',
            catch_response=True
        ) as response:
            if response.status_code not in [200, 401]:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    def check_positions(self):
        """Check open positions."""
        with self.client.get(
            '/api/v1/positions',
            headers=self.auth_headers,
            name='GET /api/v1/positions (dashboard)',
            catch_response=True
        ) as response:
            if response.status_code not in [200, 401]:
                response.failure(f"Status code: {response.status_code}")

    @task(3)
    def view_performance(self):
        """View performance stats."""
        self.client.get(
            '/api/v1/performance?days=30',
            headers=self.auth_headers,
            name='GET /api/v1/performance (dashboard)'
        )

    @task(2)
    def browse_signal_history(self):
        """Browse signal history."""
        page = random.randint(1, 5)
        self.client.get(
            f'/api/v1/signals/history?page={page}&limit=20',
            headers=self.auth_headers,
            name='GET /api/v1/signals/history (dashboard)'
        )

    @task(1)
    def view_system_status(self):
        """View system status."""
        self.client.get('/api/v1/status', name='GET /api/v1/status (dashboard)')


class DashboardUser(HttpUser):
    """
    Simulates a user viewing the web dashboard.

    Dashboard users have longer wait times (reading content)
    and focus on data visualization endpoints.
    """
    tasks = [DashboardUserTasks]
    wait_time = between(2.0, 5.0)  # Longer waits (reading dashboard)
    weight = 2  # Medium weight


# =============================================================================
# WEBSOCKET USER - Tests WebSocket connections
# =============================================================================

if SOCKETIO_AVAILABLE:
    class WebSocketUser(HttpUser):
        """
        Simulates a WebSocket user for real-time data.

        Tests:
        - Connection establishment
        - Room subscriptions
        - Message receiving
        - Reconnection handling
        """
        wait_time = between(1.0, 3.0)
        weight = 1  # Lower weight (fewer WebSocket connections)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sio: Optional[socketio.Client] = None
            self.connected = False
            self.messages_received = 0
            self.subscribed_rooms: List[str] = []

        def on_start(self):
            """Connect to WebSocket on user start."""
            self.connect_websocket()

        def on_stop(self):
            """Disconnect WebSocket on user stop."""
            self.disconnect_websocket()

        def connect_websocket(self):
            """Establish WebSocket connection."""
            start_time = time.time()
            try:
                self.sio = socketio.Client()

                @self.sio.on('connect')
                def on_connect():
                    self.connected = True

                @self.sio.on('disconnect')
                def on_disconnect():
                    self.connected = False

                @self.sio.on('signal')
                def on_signal(data):
                    self.messages_received += 1

                @self.sio.on('position')
                def on_position(data):
                    self.messages_received += 1

                @self.sio.on('scanner')
                def on_scanner(data):
                    self.messages_received += 1

                # Connect
                ws_url = self.host.replace('http://', 'ws://').replace('https://', 'wss://')
                self.sio.connect(ws_url, wait_timeout=10)

                response_time = (time.time() - start_time) * 1000
                events.request.fire(
                    request_type='WebSocket',
                    name='connect',
                    response_time=response_time,
                    response_length=0,
                    exception=None,
                    context={}
                )

            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                events.request.fire(
                    request_type='WebSocket',
                    name='connect',
                    response_time=response_time,
                    response_length=0,
                    exception=e,
                    context={}
                )

        def disconnect_websocket(self):
            """Disconnect from WebSocket."""
            if self.sio and self.connected:
                try:
                    self.sio.disconnect()
                except Exception:
                    pass

        @task(5)
        def subscribe_to_signals(self):
            """Subscribe to signals room."""
            if not self.sio or not self.connected:
                self.connect_websocket()
                return

            start_time = time.time()
            try:
                self.sio.emit('subscribe', {'room': 'signals'})
                if 'signals' not in self.subscribed_rooms:
                    self.subscribed_rooms.append('signals')

                response_time = (time.time() - start_time) * 1000
                events.request.fire(
                    request_type='WebSocket',
                    name='subscribe_signals',
                    response_time=response_time,
                    response_length=0,
                    exception=None,
                    context={}
                )
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                events.request.fire(
                    request_type='WebSocket',
                    name='subscribe_signals',
                    response_time=response_time,
                    response_length=0,
                    exception=e,
                    context={}
                )

        @task(3)
        def subscribe_to_positions(self):
            """Subscribe to positions room."""
            if not self.sio or not self.connected:
                return

            start_time = time.time()
            try:
                self.sio.emit('subscribe', {'room': 'positions'})
                if 'positions' not in self.subscribed_rooms:
                    self.subscribed_rooms.append('positions')

                response_time = (time.time() - start_time) * 1000
                events.request.fire(
                    request_type='WebSocket',
                    name='subscribe_positions',
                    response_time=response_time,
                    response_length=0,
                    exception=None,
                    context={}
                )
            except Exception as e:
                events.request.fire(
                    request_type='WebSocket',
                    name='subscribe_positions',
                    response_time=response_time,
                    response_length=0,
                    exception=e,
                    context={}
                )

        @task(2)
        def send_ping(self):
            """Send ping to keep connection alive."""
            if not self.sio or not self.connected:
                return

            start_time = time.time()
            try:
                self.sio.emit('ping')
                response_time = (time.time() - start_time) * 1000
                events.request.fire(
                    request_type='WebSocket',
                    name='ping',
                    response_time=response_time,
                    response_length=0,
                    exception=None,
                    context={}
                )
            except Exception as e:
                events.request.fire(
                    request_type='WebSocket',
                    name='ping',
                    response_time=response_time,
                    response_length=0,
                    exception=e,
                    context={}
                )

        @task(1)
        def get_rooms(self):
            """Get current room subscriptions."""
            if not self.sio or not self.connected:
                return

            start_time = time.time()
            try:
                self.sio.emit('get_rooms')
                response_time = (time.time() - start_time) * 1000
                events.request.fire(
                    request_type='WebSocket',
                    name='get_rooms',
                    response_time=response_time,
                    response_length=0,
                    exception=None,
                    context={}
                )
            except Exception as e:
                events.request.fire(
                    request_type='WebSocket',
                    name='get_rooms',
                    response_time=response_time,
                    response_length=0,
                    exception=e,
                    context={}
                )

else:
    # Fallback if socketio not available
    class WebSocketUser(HttpUser):
        """Placeholder WebSocket user when python-socketio not installed."""
        wait_time = between(1.0, 3.0)
        weight = 0  # Disabled

        @task
        def placeholder(self):
            """Placeholder task."""
            pass


# =============================================================================
# MIXED USER - Combination of behaviors
# =============================================================================

class MixedUserTasks(TaskSet):
    """
    Mixed user behavior combining API and dashboard patterns.

    Represents a power user who uses both the API and dashboard.
    """

    def on_start(self):
        """Initialize user."""
        self.api_key = get_api_key()
        self.auth_headers = get_auth_headers()

    @task(5)
    def api_health_check(self):
        """Quick health check."""
        self.client.get('/api/v1/health', name='GET /api/v1/health (mixed)')

    @task(8)
    def get_signals(self):
        """Get current signals."""
        self.client.get(
            '/api/v1/signals/current',
            headers=self.auth_headers,
            name='GET /api/v1/signals/current (mixed)'
        )

    @task(6)
    def get_dashboard(self):
        """Load dashboard."""
        self.client.get('/api/v1/dashboard', name='GET /api/v1/dashboard (mixed)')

    @task(4)
    def get_positions(self):
        """Get positions."""
        self.client.get(
            '/api/v1/positions',
            headers=self.auth_headers,
            name='GET /api/v1/positions (mixed)'
        )

    @task(2)
    def get_quote(self):
        """Get a quote."""
        symbol = random.choice(TEST_SYMBOLS)
        self.client.get(
            f'/api/v1/quote/{symbol}',
            headers=self.auth_headers,
            name='GET /api/v1/quote/{symbol} (mixed)'
        )


class MixedUser(HttpUser):
    """Mixed behavior user."""
    tasks = [MixedUserTasks]
    wait_time = between(1.0, 3.0)
    weight = 2


# =============================================================================
# EVENT HANDLERS
# =============================================================================

@events.test_start.add_listener
def on_test_start(environment: Environment, **kwargs):
    """Called when test starts."""
    print("\n" + "=" * 60)
    print("RDT Trading System Load Test Starting")
    print(f"Target Host: {environment.host}")
    print(f"User Classes: APIUser, DashboardUser, WebSocketUser, MixedUser")
    print("=" * 60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment: Environment, **kwargs):
    """Called when test stops."""
    print("\n" + "=" * 60)
    print("Load Test Complete")
    print("=" * 60 + "\n")

    # Print summary if stats available
    if environment.stats:
        print(f"Total Requests: {environment.stats.total.num_requests}")
        print(f"Failed Requests: {environment.stats.total.num_failures}")
        print(f"Average Response Time: {environment.stats.total.avg_response_time:.2f}ms")
        print(f"Median Response Time: {environment.stats.total.median_response_time:.2f}ms")
        print(f"95th Percentile: {environment.stats.total.get_response_time_percentile(0.95):.2f}ms")
        print(f"99th Percentile: {environment.stats.total.get_response_time_percentile(0.99):.2f}ms")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Called on each request."""
    if exception:
        print(f"Request failed: {request_type} {name} - {exception}")


# =============================================================================
# CUSTOM SHAPE (Optional)
# =============================================================================

try:
    from locust import LoadTestShape

    class StagesShape(LoadTestShape):
        """
        Custom load test shape with ramp-up, steady state, and ramp-down.

        Stages:
        1. Ramp up to peak load
        2. Maintain steady state
        3. Ramp down
        4. Cool down with minimal load
        """

        stages = [
            {"duration": 30, "users": 10, "spawn_rate": 1},    # Warm up
            {"duration": 60, "users": 50, "spawn_rate": 5},    # Ramp up
            {"duration": 120, "users": 100, "spawn_rate": 10}, # Peak load
            {"duration": 60, "users": 50, "spawn_rate": 5},    # Ramp down
            {"duration": 30, "users": 10, "spawn_rate": 1},    # Cool down
        ]

        def tick(self):
            """Return the user count and spawn rate for the current time."""
            run_time = self.get_run_time()

            for stage in self.stages:
                run_time -= stage["duration"]
                if run_time < 0:
                    return (stage["users"], stage["spawn_rate"])

            return None  # Stop the test

except ImportError:
    pass  # LoadTestShape not available in older Locust versions


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("This file should be run with Locust:")
    print("  locust -f tests/performance/locustfile.py --host=http://localhost:5000")
    print("\nOr use the load test script:")
    print("  python scripts/run_load_test.py --users 50 --duration 60")
