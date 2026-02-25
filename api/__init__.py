"""
RDT Trading System API

REST API for signal service monetization.
Provides endpoints for:
- Real-time trading signals
- Historical signal performance
- Backtest execution
- Account management
"""

from api.v1.routes import api_bp
from api.v1.auth import require_api_key, require_subscription, init_api_auth

__all__ = ['api_bp', 'require_api_key', 'require_subscription', 'init_api_auth']
