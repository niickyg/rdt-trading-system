"""
API Authentication and Authorization

Handles API key validation and subscription tier checking
for the signal service monetization.
"""

from functools import wraps
from datetime import datetime, timedelta
from typing import Optional, Dict
from enum import Enum
import hashlib
import secrets
from dataclasses import dataclass


class SubscriptionTier(str, Enum):
    """Subscription tiers for signal service"""
    FREE = "free"           # Limited access
    BASIC = "basic"         # $49/month - Daily alerts
    PRO = "pro"             # $149/month - Real-time + API
    ELITE = "elite"         # $499/month - Full access + support


@dataclass
class APIUser:
    """API user model"""
    user_id: str
    email: str
    api_key: str
    subscription_tier: SubscriptionTier
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    rate_limit: int = 100  # Requests per hour
    requests_this_hour: int = 0
    last_request_reset: datetime = None

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    @property
    def can_make_request(self) -> bool:
        if not self.is_active or self.is_expired:
            return False

        # Reset hourly counter if needed
        if self.last_request_reset is None or \
           (datetime.now() - self.last_request_reset).total_seconds() > 3600:
            self.requests_this_hour = 0
            self.last_request_reset = datetime.now()

        return self.requests_this_hour < self.rate_limit


# Tier-based rate limits
TIER_RATE_LIMITS = {
    SubscriptionTier.FREE: 10,      # 10 requests/hour
    SubscriptionTier.BASIC: 100,    # 100 requests/hour
    SubscriptionTier.PRO: 1000,     # 1000 requests/hour
    SubscriptionTier.ELITE: 10000,  # 10000 requests/hour (essentially unlimited)
}

# Tier-based feature access
TIER_FEATURES = {
    SubscriptionTier.FREE: {
        'real_time_signals': False,
        'historical_signals': True,
        'backtest_api': False,
        'websocket': False,
        'custom_alerts': False,
        'api_access': False,
        'signal_history_days': 7,
    },
    SubscriptionTier.BASIC: {
        'real_time_signals': False,
        'historical_signals': True,
        'backtest_api': False,
        'websocket': False,
        'custom_alerts': True,
        'api_access': False,
        'signal_history_days': 30,
    },
    SubscriptionTier.PRO: {
        'real_time_signals': True,
        'historical_signals': True,
        'backtest_api': True,
        'websocket': True,
        'custom_alerts': True,
        'api_access': True,
        'signal_history_days': 365,
    },
    SubscriptionTier.ELITE: {
        'real_time_signals': True,
        'historical_signals': True,
        'backtest_api': True,
        'websocket': True,
        'custom_alerts': True,
        'api_access': True,
        'signal_history_days': -1,  # Unlimited
    },
}


class APIKeyManager:
    """Manage API keys and user authentication"""

    def __init__(self):
        # In production, this would be a database
        self._users: Dict[str, APIUser] = {}
        self._api_keys: Dict[str, str] = {}  # api_key -> user_id

    def generate_api_key(self) -> str:
        """Generate a secure API key"""
        return f"rdt_{secrets.token_urlsafe(32)}"

    def create_user(
        self,
        email: str,
        tier: SubscriptionTier = SubscriptionTier.FREE,
        duration_days: Optional[int] = None
    ) -> APIUser:
        """Create a new API user"""
        user_id = hashlib.sha256(email.encode()).hexdigest()[:16]
        api_key = self.generate_api_key()

        expires_at = None
        if duration_days:
            expires_at = datetime.now() + timedelta(days=duration_days)

        user = APIUser(
            user_id=user_id,
            email=email,
            api_key=api_key,
            subscription_tier=tier,
            created_at=datetime.now(),
            expires_at=expires_at,
            rate_limit=TIER_RATE_LIMITS[tier]
        )

        self._users[user_id] = user
        self._api_keys[api_key] = user_id

        return user

    def get_user_by_api_key(self, api_key: str) -> Optional[APIUser]:
        """Get user by API key"""
        user_id = self._api_keys.get(api_key)
        if user_id:
            return self._users.get(user_id)
        return None

    def validate_api_key(self, api_key: str) -> tuple[bool, Optional[str]]:
        """
        Validate an API key

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not api_key:
            return False, "API key required"

        user = self.get_user_by_api_key(api_key)
        if not user:
            return False, "Invalid API key"

        if not user.is_active:
            return False, "API key has been deactivated"

        if user.is_expired:
            return False, "Subscription has expired"

        if not user.can_make_request:
            return False, "Rate limit exceeded"

        # Increment request counter
        user.requests_this_hour += 1

        return True, None

    def check_feature_access(
        self,
        api_key: str,
        feature: str
    ) -> tuple[bool, Optional[str]]:
        """Check if user has access to a feature"""
        user = self.get_user_by_api_key(api_key)
        if not user:
            return False, "Invalid API key"

        features = TIER_FEATURES.get(user.subscription_tier, {})
        has_access = features.get(feature, False)

        if not has_access:
            return False, f"Feature '{feature}' requires upgrade to higher tier"

        return True, None

    def upgrade_user(self, user_id: str, new_tier: SubscriptionTier):
        """Upgrade a user's subscription"""
        if user_id in self._users:
            self._users[user_id].subscription_tier = new_tier
            self._users[user_id].rate_limit = TIER_RATE_LIMITS[new_tier]


# Global instance
api_key_manager = APIKeyManager()


def require_api_key(f):
    """Decorator to require valid API key"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get API key from header or query param
        from flask import request, jsonify

        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')

        is_valid, error = api_key_manager.validate_api_key(api_key)
        if not is_valid:
            return jsonify({'error': error, 'status': 'unauthorized'}), 401

        # Add user to request context
        user = api_key_manager.get_user_by_api_key(api_key)
        request.api_user = user

        return f(*args, **kwargs)

    return decorated_function


def require_subscription(min_tier: SubscriptionTier):
    """Decorator to require minimum subscription tier"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import request, jsonify

            user = getattr(request, 'api_user', None)
            if not user:
                return jsonify({'error': 'Authentication required'}), 401

            tier_order = [
                SubscriptionTier.FREE,
                SubscriptionTier.BASIC,
                SubscriptionTier.PRO,
                SubscriptionTier.ELITE
            ]

            user_tier_idx = tier_order.index(user.subscription_tier)
            required_tier_idx = tier_order.index(min_tier)

            if user_tier_idx < required_tier_idx:
                return jsonify({
                    'error': f'This endpoint requires {min_tier.value} subscription or higher',
                    'current_tier': user.subscription_tier.value,
                    'upgrade_url': '/api/v1/billing/upgrade'
                }), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def require_feature(feature: str):
    """Decorator to require specific feature access"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import request, jsonify

            api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
            has_access, error = api_key_manager.check_feature_access(api_key, feature)

            if not has_access:
                return jsonify({
                    'error': error,
                    'feature': feature,
                    'upgrade_url': '/api/v1/billing/upgrade'
                }), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator
