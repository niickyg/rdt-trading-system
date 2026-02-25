"""
RDT Trading System - Web Push Notifications Module

Provides:
- VAPID key generation and management
- Push notification subscription management
- Send notification functions for different trading events
- Database storage for subscriptions
"""

import os
import json
import base64
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

from loguru import logger

# Optional: pywebpush for sending push notifications
try:
    from pywebpush import webpush, WebPushException
    WEBPUSH_AVAILABLE = True
except ImportError:
    WEBPUSH_AVAILABLE = False
    logger.warning("pywebpush not available. Install with: pip install pywebpush")

# Optional: cryptography for VAPID key generation
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography not available for key generation")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PushSubscription:
    """Push notification subscription data"""
    endpoint: str
    keys: Dict[str, str]
    user_id: Optional[int] = None
    created_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'endpoint': self.endpoint,
            'keys': self.keys,
            'expirationTime': None
        }


@dataclass
class NotificationPayload:
    """Push notification payload"""
    title: str
    body: str
    icon: str = '/static/images/icons/icon-192x192.png'
    badge: str = '/static/images/icons/icon-96x96.png'
    tag: str = 'rdt-notification'
    data: Optional[Dict[str, Any]] = None
    actions: Optional[List[Dict[str, str]]] = None
    requireInteraction: bool = False

    def to_json(self) -> str:
        payload = {
            'title': self.title,
            'body': self.body,
            'icon': self.icon,
            'badge': self.badge,
            'tag': self.tag,
        }
        if self.data:
            payload['data'] = self.data
        if self.actions:
            payload['actions'] = self.actions
        if self.requireInteraction:
            payload['requireInteraction'] = True
        return json.dumps(payload)


# ============================================================================
# VAPID Key Management
# ============================================================================

class VAPIDKeyManager:
    """Manages VAPID keys for push notifications"""

    def __init__(self, private_key: Optional[str] = None, public_key: Optional[str] = None):
        self._private_key = private_key or os.environ.get('VAPID_PRIVATE_KEY')
        self._public_key = public_key or os.environ.get('VAPID_PUBLIC_KEY')
        self._claims_email = os.environ.get('VAPID_CLAIMS_EMAIL', 'mailto:admin@rdttrading.com')

    @property
    def public_key(self) -> Optional[str]:
        """Get the VAPID public key"""
        return self._public_key

    @property
    def private_key(self) -> Optional[str]:
        """Get the VAPID private key"""
        return self._private_key

    @property
    def claims(self) -> Dict[str, str]:
        """Get VAPID claims for push notifications"""
        return {'sub': self._claims_email}

    def is_configured(self) -> bool:
        """Check if VAPID keys are configured"""
        return bool(self._private_key and self._public_key)

    @staticmethod
    def generate_keys() -> Dict[str, str]:
        """Generate new VAPID key pair"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography package required for key generation")

        # Generate EC key pair
        private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())

        # Get private key bytes (raw format for WebPush)
        private_bytes = private_key.private_numbers().private_value.to_bytes(32, 'big')
        private_key_b64 = base64.urlsafe_b64encode(private_bytes).rstrip(b'=').decode('ascii')

        # Get public key bytes (uncompressed point format)
        public_key = private_key.public_key()
        public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint
        )
        public_key_b64 = base64.urlsafe_b64encode(public_bytes).rstrip(b'=').decode('ascii')

        return {
            'private_key': private_key_b64,
            'public_key': public_key_b64
        }


# ============================================================================
# Subscription Storage (In-Memory with optional database persistence)
# ============================================================================

class SubscriptionStore:
    """Store for push notification subscriptions"""

    def __init__(self):
        self._subscriptions: Dict[str, PushSubscription] = {}
        self._user_subscriptions: Dict[int, List[str]] = {}

    def add(self, subscription: PushSubscription) -> bool:
        """Add a new subscription"""
        endpoint = subscription.endpoint

        if endpoint in self._subscriptions:
            # Update existing subscription
            self._subscriptions[endpoint] = subscription
            logger.debug(f"Updated subscription: {endpoint[:50]}...")
            return True

        # Add new subscription
        self._subscriptions[endpoint] = subscription

        # Track by user if user_id is provided
        if subscription.user_id:
            if subscription.user_id not in self._user_subscriptions:
                self._user_subscriptions[subscription.user_id] = []
            self._user_subscriptions[subscription.user_id].append(endpoint)

        logger.info(f"Added new push subscription: {endpoint[:50]}...")
        return True

    def remove(self, endpoint: str) -> bool:
        """Remove a subscription"""
        if endpoint not in self._subscriptions:
            return False

        subscription = self._subscriptions.pop(endpoint)

        # Remove from user tracking
        if subscription.user_id and subscription.user_id in self._user_subscriptions:
            self._user_subscriptions[subscription.user_id] = [
                e for e in self._user_subscriptions[subscription.user_id]
                if e != endpoint
            ]

        logger.info(f"Removed push subscription: {endpoint[:50]}...")
        return True

    def get(self, endpoint: str) -> Optional[PushSubscription]:
        """Get a subscription by endpoint"""
        return self._subscriptions.get(endpoint)

    def get_by_user(self, user_id: int) -> List[PushSubscription]:
        """Get all subscriptions for a user"""
        endpoints = self._user_subscriptions.get(user_id, [])
        return [self._subscriptions[e] for e in endpoints if e in self._subscriptions]

    def get_all(self) -> List[PushSubscription]:
        """Get all active subscriptions"""
        return [s for s in self._subscriptions.values() if s.is_active]

    def mark_inactive(self, endpoint: str) -> None:
        """Mark a subscription as inactive (failed delivery)"""
        if endpoint in self._subscriptions:
            self._subscriptions[endpoint].is_active = False

    def count(self) -> int:
        """Get total number of active subscriptions"""
        return len([s for s in self._subscriptions.values() if s.is_active])


# ============================================================================
# Push Notification Service
# ============================================================================

class PushNotificationService:
    """Service for sending push notifications"""

    def __init__(self,
                 vapid_manager: Optional[VAPIDKeyManager] = None,
                 subscription_store: Optional[SubscriptionStore] = None):
        self.vapid = vapid_manager or VAPIDKeyManager()
        self.subscriptions = subscription_store or SubscriptionStore()

    def is_available(self) -> bool:
        """Check if push notifications are available"""
        return WEBPUSH_AVAILABLE and self.vapid.is_configured()

    def subscribe(self,
                  endpoint: str,
                  keys: Dict[str, str],
                  user_id: Optional[int] = None) -> bool:
        """Subscribe to push notifications"""
        subscription = PushSubscription(
            endpoint=endpoint,
            keys=keys,
            user_id=user_id,
            created_at=datetime.utcnow(),
            is_active=True
        )
        return self.subscriptions.add(subscription)

    def unsubscribe(self, endpoint: str) -> bool:
        """Unsubscribe from push notifications"""
        return self.subscriptions.remove(endpoint)

    def send_to_endpoint(self,
                         endpoint: str,
                         payload: NotificationPayload) -> bool:
        """Send notification to a specific endpoint"""
        if not self.is_available():
            logger.warning("Push notifications not available")
            return False

        subscription = self.subscriptions.get(endpoint)
        if not subscription:
            logger.warning(f"Subscription not found: {endpoint[:50]}...")
            return False

        return self._send_notification(subscription, payload)

    def send_to_user(self,
                     user_id: int,
                     payload: NotificationPayload) -> int:
        """Send notification to all devices of a user"""
        if not self.is_available():
            return 0

        subscriptions = self.subscriptions.get_by_user(user_id)
        success_count = 0

        for subscription in subscriptions:
            if self._send_notification(subscription, payload):
                success_count += 1

        return success_count

    def broadcast(self, payload: NotificationPayload) -> int:
        """Send notification to all subscribers"""
        if not self.is_available():
            return 0

        subscriptions = self.subscriptions.get_all()
        success_count = 0

        for subscription in subscriptions:
            if self._send_notification(subscription, payload):
                success_count += 1

        logger.info(f"Broadcast sent to {success_count}/{len(subscriptions)} subscribers")
        return success_count

    def _send_notification(self,
                          subscription: PushSubscription,
                          payload: NotificationPayload) -> bool:
        """Internal method to send a single notification"""
        try:
            response = webpush(
                subscription_info=subscription.to_dict(),
                data=payload.to_json(),
                vapid_private_key=self.vapid.private_key,
                vapid_claims=self.vapid.claims
            )

            # Update last used timestamp
            subscription.last_used = datetime.utcnow()
            logger.debug(f"Push sent successfully: {subscription.endpoint[:50]}...")
            return True

        except WebPushException as e:
            logger.error(f"Push failed: {e}")

            # Handle expired/invalid subscriptions
            if e.response and e.response.status_code in (404, 410):
                logger.info("Subscription expired, marking inactive")
                self.subscriptions.mark_inactive(subscription.endpoint)

            return False
        except Exception as e:
            logger.error(f"Push error: {e}")
            return False


# ============================================================================
# Trading-Specific Notification Functions
# ============================================================================

def create_signal_notification(
    symbol: str,
    direction: str,
    strength: str,
    price: float
) -> NotificationPayload:
    """Create notification for a new trading signal"""
    direction_emoji = "LONG" if direction.upper() == "LONG" else "SHORT"

    return NotificationPayload(
        title=f"New Signal: {symbol} {direction_emoji}",
        body=f"{strength} {direction.lower()} signal at ${price:.2f}",
        tag=f"signal-{symbol}",
        data={
            'type': 'signal',
            'symbol': symbol,
            'direction': direction,
            'url': '/dashboard/signals'
        },
        actions=[
            {'action': 'view', 'title': 'View Signal'},
            {'action': 'dismiss', 'title': 'Dismiss'}
        ],
        requireInteraction=True
    )


def create_alert_notification(
    symbol: str,
    alert_type: str,
    trigger_price: float,
    current_price: float
) -> NotificationPayload:
    """Create notification for a triggered price alert"""
    return NotificationPayload(
        title=f"Alert: {symbol} {alert_type.upper()}",
        body=f"Price reached ${current_price:.2f} (target: ${trigger_price:.2f})",
        tag=f"alert-{symbol}",
        data={
            'type': 'alert',
            'symbol': symbol,
            'url': '/dashboard/alerts'
        },
        actions=[
            {'action': 'view', 'title': 'View Alert'},
            {'action': 'dismiss', 'title': 'Dismiss'}
        ],
        requireInteraction=True
    )


def create_position_notification(
    symbol: str,
    event: str,
    pnl: Optional[float] = None
) -> NotificationPayload:
    """Create notification for position events"""
    if event == 'stop_hit':
        title = f"Stop Hit: {symbol}"
        body = f"Stop loss triggered" + (f" - P/L: ${pnl:.2f}" if pnl else "")
    elif event == 'target_hit':
        title = f"Target Hit: {symbol}"
        body = f"Profit target reached" + (f" - P/L: ${pnl:.2f}" if pnl else "")
    elif event == 'closed':
        title = f"Position Closed: {symbol}"
        body = f"Position closed" + (f" - P/L: ${pnl:.2f}" if pnl else "")
    else:
        title = f"Position Update: {symbol}"
        body = event

    return NotificationPayload(
        title=title,
        body=body,
        tag=f"position-{symbol}",
        data={
            'type': 'position',
            'symbol': symbol,
            'url': '/dashboard/positions'
        },
        actions=[
            {'action': 'view', 'title': 'View Position'},
            {'action': 'dismiss', 'title': 'Dismiss'}
        ]
    )


def create_system_notification(
    title: str,
    body: str,
    url: str = '/dashboard'
) -> NotificationPayload:
    """Create notification for system events"""
    return NotificationPayload(
        title=title,
        body=body,
        tag='system',
        data={
            'type': 'system',
            'url': url
        }
    )


# ============================================================================
# Global Instance
# ============================================================================

_push_service: Optional[PushNotificationService] = None


def get_push_service() -> PushNotificationService:
    """Get the global push notification service instance"""
    global _push_service
    if _push_service is None:
        _push_service = PushNotificationService()
    return _push_service


def init_push_notifications(private_key: Optional[str] = None,
                           public_key: Optional[str] = None) -> PushNotificationService:
    """Initialize the push notification service with VAPID keys"""
    global _push_service
    vapid = VAPIDKeyManager(private_key=private_key, public_key=public_key)
    _push_service = PushNotificationService(vapid_manager=vapid)

    if _push_service.is_available():
        logger.info("Push notification service initialized")
    else:
        logger.warning("Push notifications not fully configured")

    return _push_service
