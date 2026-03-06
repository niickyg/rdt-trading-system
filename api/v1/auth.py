"""
API Authentication and Authorization

Handles API key validation and subscription tier checking
for the signal service monetization.

Uses database persistence for API keys and users.
"""

from functools import wraps
from datetime import datetime, timedelta
from typing import Optional, Dict
from enum import Enum
import hashlib
import secrets
from dataclasses import dataclass

# Database imports
from data.database.connection import get_db_manager
from data.database.models import APIUser as APIUserModel, SubscriptionTierEnum

from loguru import logger


class SubscriptionTier(str, Enum):
    """Subscription tiers for signal service"""
    FREE = "free"           # Limited access
    BASIC = "basic"         # $49/month - Daily alerts
    PRO = "pro"             # $149/month - Real-time + API
    ELITE = "elite"         # $499/month - Full access + support


@dataclass
class APIUserDTO:
    """API user data transfer object (in-memory representation)"""
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


# Alias for backward compatibility
APIUser = APIUserDTO


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
    """Manage API keys and user authentication with database persistence"""

    # NOTE: API keys should be created via the API or CLI tools
    # DO NOT add default/test API keys here

    def __init__(self):
        # In-memory cache for rate limiting (requests_this_hour, last_request_reset)
        # Key: user_id, Value: dict with rate limit tracking
        self._rate_limit_cache: Dict[str, dict] = {}
        self._db_manager = None
        self._initialized = False

    def _get_db_manager(self):
        """Lazy load database manager"""
        if self._db_manager is None:
            self._db_manager = get_db_manager()
        return self._db_manager

    def initialize(self):
        """Initialize the API key manager.

        NOTE: API keys should be created via:
        - POST /api/v1/auth/register endpoint
        - scripts/create_api_user.py CLI tool
        - Admin dashboard

        DO NOT create default API keys with hardcoded credentials.
        """
        if self._initialized:
            return

        db_manager = self._get_db_manager()

        # Ensure tables are created
        db_manager.create_tables()

        logger.info("API Key Manager initialized")
        logger.info("Create API users via /api/v1/auth/register or scripts/create_api_user.py")

        self._initialized = True

    def generate_api_key(self) -> str:
        """Generate a secure API key"""
        return f"rdt_{secrets.token_urlsafe(32)}"

    def _model_to_dto(self, db_user: APIUserModel) -> APIUserDTO:
        """Convert database model to DTO"""
        # Get rate limit cache for this user
        cache = self._rate_limit_cache.get(db_user.id, {
            'requests_this_hour': 0,
            'last_request_reset': None
        })

        return APIUserDTO(
            user_id=db_user.id,
            email=db_user.email,
            api_key=f"rdt_***{db_user.api_secret_hash[-4:] if db_user.api_secret_hash else '****'}",
            subscription_tier=SubscriptionTier(db_user.tier.value if hasattr(db_user.tier, 'value') else db_user.tier),
            created_at=db_user.created_at,
            expires_at=db_user.expires_at,
            is_active=db_user.is_active,
            rate_limit=db_user.rate_limit,
            requests_this_hour=cache.get('requests_this_hour', 0),
            last_request_reset=cache.get('last_request_reset')
        )

    def create_user(
        self,
        email: str,
        tier: SubscriptionTier = SubscriptionTier.FREE,
        duration_days: Optional[int] = None
    ) -> APIUserDTO:
        """Create a new API user and persist to database"""
        user_id = hashlib.sha256(email.encode()).hexdigest()[:16]
        api_key = self.generate_api_key()

        expires_at = None
        if duration_days:
            expires_at = datetime.now() + timedelta(days=duration_days)

        # Hash the API key before storing
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        db_manager = self._get_db_manager()

        with db_manager.get_session() as session:
            # Check if user already exists
            existing = session.query(APIUserModel).filter_by(email=email).first()
            if existing:
                raise ValueError(f"User with email {email} already exists")

            db_user = APIUserModel(
                id=user_id,
                email=email,
                api_key=api_key_hash,
                api_secret_hash=api_key_hash[:8],  # Store first 8 chars of the HASH for display identification
                tier=SubscriptionTierEnum(tier.value),
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                is_active=True,
                rate_limit=TIER_RATE_LIMITS[tier]
            )
            session.add(db_user)

            logger.info(f"Created new API user: {email} with tier {tier.value}")

            # Return DTO
            return APIUserDTO(
                user_id=user_id,
                email=email,
                api_key=api_key,
                subscription_tier=tier,
                created_at=db_user.created_at,
                expires_at=expires_at,
                is_active=True,
                rate_limit=TIER_RATE_LIMITS[tier]
            )

    def get_user_by_api_key(self, api_key: str) -> Optional[APIUserDTO]:
        """Get user by API key from database"""
        if not api_key:
            return None

        # Hash the incoming key before lookup
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        db_manager = self._get_db_manager()

        with db_manager.get_session() as session:
            db_user = session.query(APIUserModel).filter_by(api_key=api_key_hash).first()
            if db_user:
                return self._model_to_dto(db_user)
        return None

    def get_user_by_email(self, email: str) -> Optional[APIUserDTO]:
        """Get user by email from database"""
        if not email:
            return None

        db_manager = self._get_db_manager()

        with db_manager.get_session() as session:
            db_user = session.query(APIUserModel).filter_by(email=email).first()
            if db_user:
                return self._model_to_dto(db_user)
        return None

    def validate_api_key(self, api_key: str) -> tuple[bool, Optional[str]]:
        """
        Validate an API key against database

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

        # Check rate limit using in-memory cache
        cache = self._rate_limit_cache.get(user.user_id, {
            'requests_this_hour': 0,
            'last_request_reset': None
        })

        # Reset hourly counter if needed
        if cache['last_request_reset'] is None or \
           (datetime.now() - cache['last_request_reset']).total_seconds() > 3600:
            cache['requests_this_hour'] = 0
            cache['last_request_reset'] = datetime.now()

        if cache['requests_this_hour'] >= user.rate_limit:
            return False, "Rate limit exceeded"

        # Increment request counter
        cache['requests_this_hour'] += 1
        self._rate_limit_cache[user.user_id] = cache

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
        """Upgrade a user's subscription in database"""
        db_manager = self._get_db_manager()

        with db_manager.get_session() as session:
            db_user = session.query(APIUserModel).filter_by(id=user_id).first()
            if db_user:
                db_user.tier = SubscriptionTierEnum(new_tier.value)
                db_user.rate_limit = TIER_RATE_LIMITS[new_tier]
                logger.info(f"Upgraded user {user_id} to tier {new_tier.value}")

    def deactivate_user(self, user_id: str):
        """Deactivate a user's API access"""
        db_manager = self._get_db_manager()

        with db_manager.get_session() as session:
            db_user = session.query(APIUserModel).filter_by(id=user_id).first()
            if db_user:
                db_user.is_active = False
                logger.info(f"Deactivated user {user_id}")

    def regenerate_api_key(self, user_id: str) -> Optional[str]:
        """Regenerate API key for a user"""
        db_manager = self._get_db_manager()
        new_api_key = self.generate_api_key()

        with db_manager.get_session() as session:
            db_user = session.query(APIUserModel).filter_by(id=user_id).first()
            if db_user:
                new_api_key_hash = hashlib.sha256(new_api_key.encode()).hexdigest()
                db_user.api_key = new_api_key_hash
                db_user.api_secret_hash = new_api_key_hash[:8]
                logger.info(f"Regenerated API key for user {user_id}")
                return new_api_key
        return None

    def list_users(self) -> list[APIUserDTO]:
        """List all users from database"""
        db_manager = self._get_db_manager()

        with db_manager.get_session() as session:
            db_users = session.query(APIUserModel).all()
            return [self._model_to_dto(u) for u in db_users]


# Global instance
api_key_manager = APIKeyManager()


def init_api_auth():
    """Initialize API authentication system.

    Should be called when the API server starts.
    Creates database tables and default test API key if needed.
    """
    api_key_manager.initialize()
    return api_key_manager


def require_api_key(f):
    """Decorator to require valid API key or authenticated Flask-Login session"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from flask import request, jsonify

        api_key = request.headers.get('X-API-Key')

        if api_key:
            is_valid, error = api_key_manager.validate_api_key(api_key)
            if not is_valid:
                return jsonify({'error': error, 'status': 'unauthorized'}), 401

            # Add user to request context
            user = api_key_manager.get_user_by_api_key(api_key)
            request.api_user = user
            return f(*args, **kwargs)

        # Fallback: accept authenticated Flask-Login sessions (dashboard users)
        try:
            from flask_login import current_user
            if current_user and current_user.is_authenticated:
                request.api_user = current_user
                return f(*args, **kwargs)
        except ImportError:
            pass

        return jsonify({'error': 'API key required. Send via X-API-Key header.', 'status': 'unauthorized'}), 401

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

            # First try API key
            api_key = request.headers.get('X-API-Key')
            if api_key:
                has_access, error = api_key_manager.check_feature_access(api_key, feature)
                if not has_access:
                    return jsonify({
                        'error': error,
                        'feature': feature,
                        'upgrade_url': '/api/v1/billing/upgrade'
                    }), 403
                return f(*args, **kwargs)

            # Fallback: check Flask-Login session user set by require_api_key
            user = getattr(request, 'api_user', None)
            if user:
                # Dashboard users (Flask-Login User model) get full feature access
                if getattr(user, 'is_authenticated', False):
                    return f(*args, **kwargs)

            return jsonify({
                'error': f"Feature '{feature}' requires upgrade to higher tier",
                'feature': feature,
                'upgrade_url': '/api/v1/billing/upgrade'
            }), 403

        return decorated_function
    return decorator

def require_admin(f):
    """Decorator to require admin privileges"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from flask import request, jsonify

        user = getattr(request, 'api_user', None)
        if not user:
            return jsonify({'error': 'Authentication required'}), 401

        if user.subscription_tier != SubscriptionTier.ELITE:
            return jsonify({'error': 'Admin access required', 'code': 'FORBIDDEN'}), 403

        return f(*args, **kwargs)
    return decorated_function
