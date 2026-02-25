"""
Web routes for RDT Trading System.

Contains blueprint-based route modules for:
- Billing and subscription management
"""

from web.routes.billing import billing_bp

__all__ = ['billing_bp']
