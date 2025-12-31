"""
RDT Trading System API

REST API for signal service monetization.
Provides endpoints for:
- Real-time trading signals
- Historical signal performance
- Backtest execution
- Account management
"""

from api.v1.app import create_app
from api.v1.auth import require_api_key, require_subscription

__all__ = ['create_app', 'require_api_key', 'require_subscription']
