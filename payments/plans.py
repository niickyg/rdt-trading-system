"""
Subscription Plans Configuration for RDT Trading System.

Defines subscription tiers, pricing, features, and rate limits.
Maps to Stripe price IDs for billing integration.
"""

import os
from typing import Optional, Dict, Any, List
from enum import Enum


class PlanTier(str, Enum):
    """Subscription tier enumeration."""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ELITE = "elite"


# Stripe Price IDs - Configure in environment variables
# In development, use Stripe test mode prices
# In production, use Stripe live mode prices
STRIPE_PRICE_IDS = {
    PlanTier.BASIC: os.environ.get('STRIPE_PRICE_BASIC', 'price_basic_monthly'),
    PlanTier.PRO: os.environ.get('STRIPE_PRICE_PRO', 'price_pro_monthly'),
    PlanTier.ELITE: os.environ.get('STRIPE_PRICE_ELITE', 'price_elite_monthly'),
}

# Annual price IDs (optional)
STRIPE_ANNUAL_PRICE_IDS = {
    PlanTier.BASIC: os.environ.get('STRIPE_PRICE_BASIC_ANNUAL', 'price_basic_annual'),
    PlanTier.PRO: os.environ.get('STRIPE_PRICE_PRO_ANNUAL', 'price_pro_annual'),
    PlanTier.ELITE: os.environ.get('STRIPE_PRICE_ELITE_ANNUAL', 'price_elite_annual'),
}


# =============================================================================
# Subscription Plans Definition
# =============================================================================

SUBSCRIPTION_PLANS: Dict[str, Dict[str, Any]] = {
    PlanTier.FREE: {
        'id': PlanTier.FREE,
        'name': 'Free',
        'description': 'Limited access to explore the platform',
        'price_monthly': 0,
        'price_annual': 0,
        'stripe_price_id': None,
        'stripe_annual_price_id': None,
        'trial_days': 0,
        'popular': False,
        'features': [
            'View delayed signals (1 hour delay)',
            '7-day signal history',
            'Basic email alerts',
            'Community access',
            'Educational content',
        ],
        'feature_flags': {
            'realtime_signals': False,
            'api_access': False,
            'websocket_access': False,
            'backtesting': False,
            'custom_alerts': False,
            'leveraged_etf_signals': False,
            'options_signals': False,
            'consulting': False,
            'white_label': False,
        },
        'rate_limits': {
            'api_requests_per_minute': 0,
            'api_requests_per_day': 0,
            'signals_per_day': 5,
            'history_days': 7,
            'websocket_connections': 0,
        },
    },

    PlanTier.BASIC: {
        'id': PlanTier.BASIC,
        'name': 'Basic',
        'description': 'Essential trading signals for active traders',
        'price_monthly': 49,
        'price_annual': 490,  # ~2 months free
        'stripe_price_id': STRIPE_PRICE_IDS[PlanTier.BASIC],
        'stripe_annual_price_id': STRIPE_ANNUAL_PRICE_IDS[PlanTier.BASIC],
        'trial_days': 7,
        'popular': False,
        'features': [
            'Daily email alerts',
            '30-day signal history',
            'Custom price alerts (5 max)',
            'Email support',
            '~5-10 signals per week',
            'Basic performance analytics',
        ],
        'feature_flags': {
            'realtime_signals': False,
            'api_access': False,
            'websocket_access': False,
            'backtesting': False,
            'custom_alerts': True,
            'leveraged_etf_signals': False,
            'options_signals': False,
            'consulting': False,
            'white_label': False,
        },
        'rate_limits': {
            'api_requests_per_minute': 0,
            'api_requests_per_day': 0,
            'signals_per_day': 20,
            'history_days': 30,
            'websocket_connections': 0,
            'custom_alerts_max': 5,
        },
    },

    PlanTier.PRO: {
        'id': PlanTier.PRO,
        'name': 'Pro',
        'description': 'Full-featured access for serious traders',
        'price_monthly': 149,
        'price_annual': 1490,  # ~2 months free
        'stripe_price_id': STRIPE_PRICE_IDS[PlanTier.PRO],
        'stripe_annual_price_id': STRIPE_ANNUAL_PRICE_IDS[PlanTier.PRO],
        'trial_days': 7,
        'popular': True,
        'features': [
            'Real-time signal alerts',
            '1-year signal history',
            'Full REST API access',
            'Custom backtesting',
            'WebSocket streaming',
            'Priority email support',
            'Leveraged ETF signals',
            'Custom alerts (50 max)',
            'Performance dashboard',
        ],
        'feature_flags': {
            'realtime_signals': True,
            'api_access': True,
            'websocket_access': True,
            'backtesting': True,
            'custom_alerts': True,
            'leveraged_etf_signals': True,
            'options_signals': False,
            'consulting': False,
            'white_label': False,
        },
        'rate_limits': {
            'api_requests_per_minute': 60,
            'api_requests_per_day': 10000,
            'signals_per_day': 100,
            'history_days': 365,
            'websocket_connections': 3,
            'custom_alerts_max': 50,
        },
    },

    PlanTier.ELITE: {
        'id': PlanTier.ELITE,
        'name': 'Elite',
        'description': 'Premium access with personalized support',
        'price_monthly': 499,
        'price_annual': 4990,  # ~2 months free
        'stripe_price_id': STRIPE_PRICE_IDS[PlanTier.ELITE],
        'stripe_annual_price_id': STRIPE_ANNUAL_PRICE_IDS[PlanTier.ELITE],
        'trial_days': 14,
        'popular': False,
        'features': [
            'Everything in Pro',
            'Unlimited signal history',
            'Strategy consulting (2hr/mo)',
            '1-on-1 support calls',
            'Custom integrations',
            'White-label options',
            'Options signals (coming soon)',
            'Unlimited custom alerts',
            'Dedicated account manager',
            'Early access to new features',
        ],
        'feature_flags': {
            'realtime_signals': True,
            'api_access': True,
            'websocket_access': True,
            'backtesting': True,
            'custom_alerts': True,
            'leveraged_etf_signals': True,
            'options_signals': True,
            'consulting': True,
            'white_label': True,
        },
        'rate_limits': {
            'api_requests_per_minute': 300,
            'api_requests_per_day': 100000,
            'signals_per_day': -1,  # unlimited
            'history_days': -1,  # unlimited
            'websocket_connections': 10,
            'custom_alerts_max': -1,  # unlimited
        },
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_plan_by_id(plan_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a subscription plan by its ID.

    Args:
        plan_id: Plan identifier (basic, pro, elite)

    Returns:
        Plan dictionary or None if not found
    """
    plan_id = plan_id.lower()
    if plan_id in SUBSCRIPTION_PLANS:
        return SUBSCRIPTION_PLANS[plan_id]
    # Try to match by PlanTier enum
    try:
        tier = PlanTier(plan_id)
        return SUBSCRIPTION_PLANS.get(tier)
    except ValueError:
        return None


def get_plan_by_price_id(price_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a subscription plan by its Stripe price ID.

    Args:
        price_id: Stripe price ID

    Returns:
        Plan dictionary or None if not found
    """
    for plan in SUBSCRIPTION_PLANS.values():
        if plan.get('stripe_price_id') == price_id:
            return plan
        if plan.get('stripe_annual_price_id') == price_id:
            return plan
    return None


def get_plan_features(plan_id: str) -> List[str]:
    """
    Get the feature list for a plan.

    Args:
        plan_id: Plan identifier

    Returns:
        List of feature strings
    """
    plan = get_plan_by_id(plan_id)
    if plan:
        return plan.get('features', [])
    return []


def get_plan_rate_limits(plan_id: str) -> Dict[str, int]:
    """
    Get the rate limits for a plan.

    Args:
        plan_id: Plan identifier

    Returns:
        Dictionary of rate limits
    """
    plan = get_plan_by_id(plan_id)
    if plan:
        return plan.get('rate_limits', {})
    return {}


def get_feature_flags(plan_id: str) -> Dict[str, bool]:
    """
    Get the feature flags for a plan.

    Args:
        plan_id: Plan identifier

    Returns:
        Dictionary of feature flags
    """
    plan = get_plan_by_id(plan_id)
    if plan:
        return plan.get('feature_flags', {})
    return {}


def has_feature(plan_id: str, feature: str) -> bool:
    """
    Check if a plan has a specific feature enabled.

    Args:
        plan_id: Plan identifier
        feature: Feature name to check

    Returns:
        True if feature is enabled, False otherwise
    """
    flags = get_feature_flags(plan_id)
    return flags.get(feature, False)


def get_trial_days(plan_id: str) -> int:
    """
    Get the trial period days for a plan.

    Args:
        plan_id: Plan identifier

    Returns:
        Number of trial days (0 if no trial)
    """
    plan = get_plan_by_id(plan_id)
    if plan:
        return plan.get('trial_days', 0)
    return 0


def get_price_id(plan_id: str, annual: bool = False) -> Optional[str]:
    """
    Get the Stripe price ID for a plan.

    Args:
        plan_id: Plan identifier
        annual: If True, return annual price ID

    Returns:
        Stripe price ID or None if not found
    """
    plan = get_plan_by_id(plan_id)
    if plan:
        if annual:
            return plan.get('stripe_annual_price_id')
        return plan.get('stripe_price_id')
    return None


def get_all_plans() -> Dict[str, Dict[str, Any]]:
    """
    Get all subscription plans.

    Returns:
        Dictionary of all plans
    """
    return SUBSCRIPTION_PLANS


def get_paid_plans() -> Dict[str, Dict[str, Any]]:
    """
    Get only paid subscription plans.

    Returns:
        Dictionary of paid plans (excludes free tier)
    """
    return {
        k: v for k, v in SUBSCRIPTION_PLANS.items()
        if v.get('price_monthly', 0) > 0
    }


def compare_plans(plan_id_1: str, plan_id_2: str) -> int:
    """
    Compare two plans by tier level.

    Args:
        plan_id_1: First plan identifier
        plan_id_2: Second plan identifier

    Returns:
        -1 if plan_1 < plan_2, 0 if equal, 1 if plan_1 > plan_2
    """
    tier_order = [PlanTier.FREE, PlanTier.BASIC, PlanTier.PRO, PlanTier.ELITE]

    try:
        tier_1 = PlanTier(plan_id_1.lower())
        tier_2 = PlanTier(plan_id_2.lower())

        idx_1 = tier_order.index(tier_1)
        idx_2 = tier_order.index(tier_2)

        if idx_1 < idx_2:
            return -1
        elif idx_1 > idx_2:
            return 1
        return 0
    except (ValueError, AttributeError):
        return 0


def is_upgrade(current_plan: str, new_plan: str) -> bool:
    """
    Check if changing plans would be an upgrade.

    Args:
        current_plan: Current plan identifier
        new_plan: New plan identifier

    Returns:
        True if new_plan is higher tier than current_plan
    """
    return compare_plans(current_plan, new_plan) < 0


def is_downgrade(current_plan: str, new_plan: str) -> bool:
    """
    Check if changing plans would be a downgrade.

    Args:
        current_plan: Current plan identifier
        new_plan: New plan identifier

    Returns:
        True if new_plan is lower tier than current_plan
    """
    return compare_plans(current_plan, new_plan) > 0
