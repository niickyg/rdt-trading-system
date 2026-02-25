"""
GraphQL Authentication Middleware

Handles API key validation and user context for GraphQL requests.
"""

from functools import wraps
from typing import Optional, Dict, Any
from flask import request, g
from loguru import logger


class GraphQLAuthMiddleware:
    """
    Middleware for GraphQL authentication.

    Extracts API key from request headers and validates it,
    adding the user to the GraphQL context.
    """

    def __init__(self):
        self._api_key_manager = None

    @property
    def api_key_manager(self):
        """Lazy load API key manager to avoid circular imports"""
        if self._api_key_manager is None:
            from api.v1.auth import api_key_manager
            self._api_key_manager = api_key_manager
        return self._api_key_manager

    def get_api_key(self) -> Optional[str]:
        """Extract API key from request headers or query params"""
        # Try Authorization header first (Bearer token style)
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            return auth_header[7:]

        # Try X-API-Key header
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return api_key

        return None

    def authenticate(self) -> Dict[str, Any]:
        """
        Authenticate the request and return context data.

        Returns:
            Dict with user info and authentication status
        """
        api_key = self.get_api_key()

        context = {
            'authenticated': False,
            'user': None,
            'api_key': api_key,
            'error': None,
            'tier': None,
            'features': {},
        }

        if not api_key:
            context['error'] = 'API key required. Provide via X-API-Key header or Authorization Bearer header.'
            return context

        # Validate the API key
        is_valid, error = self.api_key_manager.validate_api_key(api_key)

        if not is_valid:
            context['error'] = error
            return context

        # Get user details
        user = self.api_key_manager.get_user_by_api_key(api_key)
        if user:
            context['authenticated'] = True
            context['user'] = user
            context['tier'] = user.subscription_tier.value
            context['features'] = self._get_tier_features(user.subscription_tier)

        return context

    def _get_tier_features(self, tier) -> Dict[str, bool]:
        """Get feature flags for subscription tier"""
        from api.v1.auth import TIER_FEATURES
        return TIER_FEATURES.get(tier, {})

    def resolve(self, next_resolver, root, info, **args):
        """
        Middleware resolver that adds authentication context.

        This is called by graphene for each field resolution.
        """
        return next_resolver(root, info, **args)


# Global middleware instance
graphql_auth = GraphQLAuthMiddleware()


def get_graphql_context() -> Dict[str, Any]:
    """
    Get the GraphQL context with authentication info.

    Call this at the start of GraphQL request handling.
    """
    return graphql_auth.authenticate()


def require_auth(resolver_func):
    """
    Decorator to require authentication for a resolver.

    Usage:
        @require_auth
        def resolve_signals(self, info, **kwargs):
            user = info.context.get('user')
            # ... resolver logic
    """
    @wraps(resolver_func)
    def wrapper(self, info, **kwargs):
        context = info.context
        if not context.get('authenticated'):
            error_msg = context.get('error', 'Authentication required')
            raise Exception(f"Unauthorized: {error_msg}")
        return resolver_func(self, info, **kwargs)
    return wrapper


def require_tier(min_tier: str):
    """
    Decorator to require minimum subscription tier for a resolver.

    Usage:
        @require_tier('pro')
        def resolve_advanced_feature(self, info, **kwargs):
            # ... resolver logic
    """
    tier_order = ['free', 'basic', 'pro', 'elite']

    def decorator(resolver_func):
        @wraps(resolver_func)
        def wrapper(self, info, **kwargs):
            context = info.context
            if not context.get('authenticated'):
                error_msg = context.get('error', 'Authentication required')
                raise Exception(f"Unauthorized: {error_msg}")

            user_tier = context.get('tier', 'free')
            user_tier_idx = tier_order.index(user_tier) if user_tier in tier_order else 0
            required_tier_idx = tier_order.index(min_tier) if min_tier in tier_order else 0

            if user_tier_idx < required_tier_idx:
                raise Exception(
                    f"This feature requires {min_tier} tier or higher. "
                    f"Your current tier: {user_tier}"
                )

            return resolver_func(self, info, **kwargs)
        return wrapper
    return decorator


def require_feature(feature_name: str):
    """
    Decorator to require a specific feature for a resolver.

    Usage:
        @require_feature('websocket')
        def resolve_realtime_data(self, info, **kwargs):
            # ... resolver logic
    """
    def decorator(resolver_func):
        @wraps(resolver_func)
        def wrapper(self, info, **kwargs):
            context = info.context
            if not context.get('authenticated'):
                error_msg = context.get('error', 'Authentication required')
                raise Exception(f"Unauthorized: {error_msg}")

            features = context.get('features', {})
            if not features.get(feature_name, False):
                raise Exception(
                    f"This feature ({feature_name}) is not available in your subscription tier. "
                    "Please upgrade to access this feature."
                )

            return resolver_func(self, info, **kwargs)
        return wrapper
    return decorator
