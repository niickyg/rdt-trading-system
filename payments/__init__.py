"""
Payments module for RDT Trading System.

Provides Stripe integration for subscription management including:
- Customer management
- Subscription lifecycle
- Checkout sessions
- Webhook handling
- Billing portal
"""

from payments.stripe_client import StripeClient
from payments.plans import (
    SUBSCRIPTION_PLANS,
    get_plan_by_id,
    get_plan_by_price_id,
    get_plan_features,
    get_plan_rate_limits,
)
from payments.webhooks import WebhookHandler

__all__ = [
    'StripeClient',
    'WebhookHandler',
    'SUBSCRIPTION_PLANS',
    'get_plan_by_id',
    'get_plan_by_price_id',
    'get_plan_features',
    'get_plan_rate_limits',
]
