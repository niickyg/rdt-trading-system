"""
Stripe Webhook Handler for RDT Trading System.

Processes Stripe webhook events for subscription lifecycle management:
- Checkout session completion
- Subscription created/updated/deleted
- Invoice paid/payment failed

Includes event deduplication to prevent duplicate processing.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, Callable

import stripe
from loguru import logger
from sqlalchemy.orm import Session

# Import database models
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database.models import User, Subscription, SubscriptionStatus, WebhookEvent, WebhookEventStatus
from payments.plans import get_plan_by_price_id, PlanTier


class WebhookHandler:
    """
    Handler for Stripe webhook events.

    Processes subscription-related events and updates the database accordingly.
    Also supports sending notifications for important events.
    """

    def __init__(
        self,
        db_session: Session,
        email_callback: Optional[Callable[[str, str, Dict], None]] = None,
        alert_callback: Optional[Callable[[str, Dict], None]] = None
    ):
        """
        Initialize the webhook handler.

        Args:
            db_session: SQLAlchemy database session
            email_callback: Optional callback for sending emails
                           (email, template_name, context) -> None
            alert_callback: Optional callback for sending alerts/notifications
                           (event_type, data) -> None
        """
        self.db_session = db_session
        self.email_callback = email_callback
        self.alert_callback = alert_callback

        # Map event types to handlers
        self.event_handlers: Dict[str, Callable] = {
            'checkout.session.completed': self._handle_checkout_completed,
            'customer.subscription.created': self._handle_subscription_created,
            'customer.subscription.updated': self._handle_subscription_updated,
            'customer.subscription.deleted': self._handle_subscription_deleted,
            'invoice.paid': self._handle_invoice_paid,
            'invoice.payment_failed': self._handle_invoice_payment_failed,
            'customer.created': self._handle_customer_created,
            'customer.updated': self._handle_customer_updated,
        }

    def handle_event(self, event: stripe.Event) -> Dict[str, Any]:
        """
        Process a Stripe webhook event with deduplication.

        Checks if the event has already been processed before handling.
        Records all events for audit trail.

        Args:
            event: Verified Stripe event object

        Returns:
            Dictionary with processing result
        """
        event_type = event.type
        event_id = event.id
        event_data = event.data.object

        logger.info(f"Processing webhook event: {event_type} (id: {event_id})")

        # Check for duplicate event
        if self._is_duplicate_event(event_id):
            logger.info(f"Duplicate webhook event detected: {event_id}")
            return {
                'status': 'duplicate',
                'event_type': event_type,
                'event_id': event_id,
                'reason': 'already_processed'
            }

        # Record event before processing (for recovery)
        webhook_event = self._record_event(event)

        handler = self.event_handlers.get(event_type)

        if handler:
            try:
                result = handler(event_data)
                logger.info(f"Successfully processed event: {event_type}")

                # Mark event as processed
                self._mark_event_processed(webhook_event, result)

                return {
                    'status': 'processed',
                    'event_type': event_type,
                    'event_id': event_id,
                    'result': result
                }
            except Exception as e:
                logger.error(f"Error processing event {event_type}: {e}")

                # Mark event as failed
                self._mark_event_failed(webhook_event, str(e))

                return {
                    'status': 'error',
                    'event_type': event_type,
                    'event_id': event_id,
                    'error': str(e)
                }
        else:
            logger.debug(f"No handler for event type: {event_type}")

            # Mark as processed even if no handler (prevents reprocessing)
            self._mark_event_processed(webhook_event, {'action': 'no_handler'})

            return {
                'status': 'ignored',
                'event_type': event_type,
                'event_id': event_id,
                'reason': 'no_handler'
            }

    def _is_duplicate_event(self, event_id: str) -> bool:
        """
        Check if event has already been processed.

        Args:
            event_id: Stripe event ID

        Returns:
            True if event already exists and was processed
        """
        existing = self.db_session.query(WebhookEvent).filter_by(
            event_id=event_id
        ).first()

        if existing and existing.status in (WebhookEventStatus.PROCESSED, WebhookEventStatus.DUPLICATE):
            return True

        return False

    def _record_event(self, event: stripe.Event) -> WebhookEvent:
        """
        Record webhook event to database.

        Args:
            event: Stripe event object

        Returns:
            WebhookEvent database record
        """
        webhook_event = WebhookEvent(
            event_id=event.id,
            event_type=event.type,
            source="stripe",
            payload=json.dumps(event.data.object, default=str),
            status=WebhookEventStatus.PENDING,
            created_at=datetime.utcnow()
        )

        self.db_session.add(webhook_event)
        self.db_session.commit()

        return webhook_event

    def _mark_event_processed(self, webhook_event: WebhookEvent, result: Dict[str, Any]) -> None:
        """
        Mark webhook event as successfully processed.

        Args:
            webhook_event: WebhookEvent database record
            result: Processing result
        """
        webhook_event.status = WebhookEventStatus.PROCESSED
        webhook_event.result = json.dumps(result, default=str)
        webhook_event.processed_at = datetime.utcnow()
        self.db_session.commit()

    def _mark_event_failed(self, webhook_event: WebhookEvent, error: str) -> None:
        """
        Mark webhook event as failed.

        Args:
            webhook_event: WebhookEvent database record
            error: Error message
        """
        webhook_event.status = WebhookEventStatus.FAILED
        webhook_event.error_message = error
        webhook_event.retry_count += 1
        webhook_event.processed_at = datetime.utcnow()
        self.db_session.commit()

    def _send_email(self, email: str, template: str, context: Dict[str, Any]) -> None:
        """Send email notification if callback is configured."""
        if self.email_callback:
            try:
                self.email_callback(email, template, context)
            except Exception as e:
                logger.error(f"Failed to send email notification: {e}")

    def _send_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Send alert notification if callback is configured."""
        if self.alert_callback:
            try:
                self.alert_callback(alert_type, data)
            except Exception as e:
                logger.error(f"Failed to send alert notification: {e}")

    def _get_user_by_stripe_customer(self, customer_id: str) -> Optional[User]:
        """Find user by Stripe customer ID."""
        return self.db_session.query(User).filter_by(
            stripe_customer_id=customer_id
        ).first()

    def _get_user_by_email(self, email: str) -> Optional[User]:
        """Find user by email."""
        return self.db_session.query(User).filter_by(email=email).first()

    def _get_subscription_by_stripe_id(self, subscription_id: str) -> Optional[Subscription]:
        """Find subscription by Stripe subscription ID."""
        return self.db_session.query(Subscription).filter_by(
            stripe_subscription_id=subscription_id
        ).first()

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _handle_checkout_completed(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle checkout.session.completed event.

        This is triggered when a customer completes the Stripe Checkout flow.
        """
        customer_id = session.get('customer')
        customer_email = session.get('customer_email') or session.get('customer_details', {}).get('email')
        subscription_id = session.get('subscription')
        mode = session.get('mode')

        logger.info(f"Checkout completed: customer={customer_id}, subscription={subscription_id}")

        if mode != 'subscription':
            logger.info(f"Checkout mode is {mode}, not subscription - skipping")
            return {'action': 'skipped', 'reason': 'not_subscription'}

        # Find or link user
        user = self._get_user_by_stripe_customer(customer_id)

        if not user and customer_email:
            user = self._get_user_by_email(customer_email)
            if user:
                # Link Stripe customer to existing user
                user.stripe_customer_id = customer_id
                self.db_session.commit()
                logger.info(f"Linked Stripe customer {customer_id} to user {user.id}")

        if not user:
            logger.warning(f"No user found for checkout: customer={customer_id}, email={customer_email}")
            return {
                'action': 'user_not_found',
                'customer_id': customer_id,
                'email': customer_email
            }

        # Send welcome email
        self._send_email(
            email=user.email,
            template='subscription_welcome',
            context={
                'user_name': user.username,
                'subscription_id': subscription_id
            }
        )

        self._send_alert('checkout_completed', {
            'user_id': user.id,
            'email': user.email,
            'subscription_id': subscription_id
        })

        return {
            'action': 'checkout_processed',
            'user_id': user.id,
            'subscription_id': subscription_id
        }

    def _handle_subscription_created(self, subscription: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle customer.subscription.created event.

        Creates a new subscription record in the database.
        """
        stripe_sub_id = subscription.get('id')
        customer_id = subscription.get('customer')
        status = subscription.get('status')
        current_period_start = subscription.get('current_period_start')
        current_period_end = subscription.get('current_period_end')
        trial_end = subscription.get('trial_end')

        # Get price and plan info
        items = subscription.get('items', {}).get('data', [])
        price_id = items[0].get('price', {}).get('id') if items else None
        plan = get_plan_by_price_id(price_id) if price_id else None
        plan_id = plan.get('id') if plan else PlanTier.BASIC

        logger.info(f"Subscription created: {stripe_sub_id} for customer {customer_id}")

        # Find user
        user = self._get_user_by_stripe_customer(customer_id)
        if not user:
            logger.warning(f"No user found for customer: {customer_id}")
            return {'action': 'user_not_found', 'customer_id': customer_id}

        # Check for existing subscription
        existing = self._get_subscription_by_stripe_id(stripe_sub_id)
        if existing:
            logger.info(f"Subscription {stripe_sub_id} already exists")
            return {'action': 'already_exists', 'subscription_id': existing.id}

        # Create subscription record
        sub = Subscription(
            user_id=user.id,
            stripe_subscription_id=stripe_sub_id,
            stripe_customer_id=customer_id,
            plan_id=str(plan_id),
            status=self._map_stripe_status(status),
            current_period_start=datetime.fromtimestamp(current_period_start) if current_period_start else None,
            current_period_end=datetime.fromtimestamp(current_period_end) if current_period_end else None,
            trial_end=datetime.fromtimestamp(trial_end) if trial_end else None,
            cancel_at_period_end=subscription.get('cancel_at_period_end', False),
        )

        self.db_session.add(sub)
        self.db_session.commit()

        logger.info(f"Created subscription record: id={sub.id}, user={user.id}, plan={plan_id}")

        # Send confirmation email
        self._send_email(
            email=user.email,
            template='subscription_created',
            context={
                'user_name': user.username,
                'plan_name': plan.get('name', 'Unknown') if plan else 'Unknown',
                'trial_end': datetime.fromtimestamp(trial_end).strftime('%B %d, %Y') if trial_end else None
            }
        )

        return {
            'action': 'subscription_created',
            'subscription_id': sub.id,
            'user_id': user.id,
            'plan_id': str(plan_id)
        }

    def _handle_subscription_updated(self, subscription: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle customer.subscription.updated event.

        Updates the subscription record with new status, plan, or period.
        """
        stripe_sub_id = subscription.get('id')
        customer_id = subscription.get('customer')
        status = subscription.get('status')
        current_period_start = subscription.get('current_period_start')
        current_period_end = subscription.get('current_period_end')
        cancel_at_period_end = subscription.get('cancel_at_period_end', False)

        # Get price and plan info
        items = subscription.get('items', {}).get('data', [])
        price_id = items[0].get('price', {}).get('id') if items else None
        plan = get_plan_by_price_id(price_id) if price_id else None
        plan_id = plan.get('id') if plan else None

        logger.info(f"Subscription updated: {stripe_sub_id} status={status}")

        # Find existing subscription
        sub = self._get_subscription_by_stripe_id(stripe_sub_id)
        if not sub:
            logger.warning(f"Subscription not found: {stripe_sub_id}")
            # Try to create it
            return self._handle_subscription_created(subscription)

        # Track changes for notification
        old_plan = sub.plan_id
        old_status = sub.status
        plan_changed = plan_id and str(plan_id) != sub.plan_id

        # Update subscription
        sub.status = self._map_stripe_status(status)
        sub.current_period_start = datetime.fromtimestamp(current_period_start) if current_period_start else sub.current_period_start
        sub.current_period_end = datetime.fromtimestamp(current_period_end) if current_period_end else sub.current_period_end
        sub.cancel_at_period_end = cancel_at_period_end
        sub.updated_at = datetime.utcnow()

        if plan_id:
            sub.plan_id = str(plan_id)

        self.db_session.commit()

        logger.info(f"Updated subscription: {stripe_sub_id}")

        # Send notifications for significant changes
        user = self._get_user_by_stripe_customer(customer_id)

        if plan_changed and user:
            self._send_email(
                email=user.email,
                template='subscription_plan_changed',
                context={
                    'user_name': user.username,
                    'old_plan': old_plan,
                    'new_plan': plan.get('name', str(plan_id)) if plan else str(plan_id)
                }
            )

        if cancel_at_period_end and not sub.cancel_at_period_end and user:
            self._send_email(
                email=user.email,
                template='subscription_canceling',
                context={
                    'user_name': user.username,
                    'end_date': sub.current_period_end.strftime('%B %d, %Y') if sub.current_period_end else 'Unknown'
                }
            )

        return {
            'action': 'subscription_updated',
            'subscription_id': sub.id,
            'status': status,
            'plan_changed': plan_changed
        }

    def _handle_subscription_deleted(self, subscription: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle customer.subscription.deleted event.

        Marks the subscription as canceled in the database.
        """
        stripe_sub_id = subscription.get('id')
        customer_id = subscription.get('customer')

        logger.info(f"Subscription deleted: {stripe_sub_id}")

        # Find subscription
        sub = self._get_subscription_by_stripe_id(stripe_sub_id)
        if not sub:
            logger.warning(f"Subscription not found for deletion: {stripe_sub_id}")
            return {'action': 'not_found', 'subscription_id': stripe_sub_id}

        # Update status
        sub.status = SubscriptionStatus.CANCELED
        sub.canceled_at = datetime.utcnow()
        sub.updated_at = datetime.utcnow()

        self.db_session.commit()

        logger.info(f"Marked subscription as canceled: {sub.id}")

        # Send cancellation email
        user = self._get_user_by_stripe_customer(customer_id)
        if user:
            self._send_email(
                email=user.email,
                template='subscription_canceled',
                context={
                    'user_name': user.username
                }
            )

            self._send_alert('subscription_canceled', {
                'user_id': user.id,
                'email': user.email,
                'subscription_id': sub.id
            })

        return {
            'action': 'subscription_canceled',
            'subscription_id': sub.id
        }

    def _handle_invoice_paid(self, invoice: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle invoice.paid event.

        Confirms payment and updates subscription status if needed.
        """
        invoice_id = invoice.get('id')
        customer_id = invoice.get('customer')
        subscription_id = invoice.get('subscription')
        amount_paid = invoice.get('amount_paid', 0)
        billing_reason = invoice.get('billing_reason')

        logger.info(f"Invoice paid: {invoice_id} for ${amount_paid/100:.2f}")

        if not subscription_id:
            logger.info("Invoice not associated with subscription - skipping")
            return {'action': 'skipped', 'reason': 'no_subscription'}

        # Find subscription
        sub = self._get_subscription_by_stripe_id(subscription_id)
        if sub:
            # Ensure subscription is active
            if sub.status != SubscriptionStatus.ACTIVE:
                sub.status = SubscriptionStatus.ACTIVE
                sub.updated_at = datetime.utcnow()
                self.db_session.commit()

        # Send receipt email
        user = self._get_user_by_stripe_customer(customer_id)
        if user and amount_paid > 0:
            self._send_email(
                email=user.email,
                template='invoice_paid',
                context={
                    'user_name': user.username,
                    'amount': f"${amount_paid/100:.2f}",
                    'invoice_id': invoice_id,
                    'billing_reason': billing_reason
                }
            )

        return {
            'action': 'invoice_processed',
            'invoice_id': invoice_id,
            'amount': amount_paid
        }

    def _handle_invoice_payment_failed(self, invoice: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle invoice.payment_failed event.

        Notifies the user of payment failure and may update subscription status.
        """
        invoice_id = invoice.get('id')
        customer_id = invoice.get('customer')
        subscription_id = invoice.get('subscription')
        attempt_count = invoice.get('attempt_count', 1)
        next_attempt = invoice.get('next_payment_attempt')

        logger.warning(f"Invoice payment failed: {invoice_id} (attempt {attempt_count})")

        # Find subscription
        if subscription_id:
            sub = self._get_subscription_by_stripe_id(subscription_id)
            if sub:
                sub.status = SubscriptionStatus.PAST_DUE
                sub.updated_at = datetime.utcnow()
                self.db_session.commit()

        # Send payment failed email
        user = self._get_user_by_stripe_customer(customer_id)
        if user:
            next_attempt_date = None
            if next_attempt:
                next_attempt_date = datetime.fromtimestamp(next_attempt).strftime('%B %d, %Y')

            self._send_email(
                email=user.email,
                template='payment_failed',
                context={
                    'user_name': user.username,
                    'attempt_count': attempt_count,
                    'next_attempt_date': next_attempt_date
                }
            )

            self._send_alert('payment_failed', {
                'user_id': user.id,
                'email': user.email,
                'invoice_id': invoice_id,
                'attempt_count': attempt_count
            })

        return {
            'action': 'payment_failure_handled',
            'invoice_id': invoice_id,
            'attempt_count': attempt_count
        }

    def _handle_customer_created(self, customer: Dict[str, Any]) -> Dict[str, Any]:
        """Handle customer.created event."""
        customer_id = customer.get('id')
        email = customer.get('email')

        logger.info(f"Customer created: {customer_id} ({email})")

        # Link to existing user if email matches
        if email:
            user = self._get_user_by_email(email)
            if user and not user.stripe_customer_id:
                user.stripe_customer_id = customer_id
                self.db_session.commit()
                logger.info(f"Linked new Stripe customer to user {user.id}")

        return {
            'action': 'customer_created',
            'customer_id': customer_id,
            'email': email
        }

    def _handle_customer_updated(self, customer: Dict[str, Any]) -> Dict[str, Any]:
        """Handle customer.updated event."""
        customer_id = customer.get('id')
        email = customer.get('email')

        logger.info(f"Customer updated: {customer_id}")

        # Could update user email if needed
        return {
            'action': 'customer_updated',
            'customer_id': customer_id
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _map_stripe_status(self, stripe_status: str) -> SubscriptionStatus:
        """
        Map Stripe subscription status to internal status.

        Args:
            stripe_status: Stripe status string

        Returns:
            SubscriptionStatus enum value
        """
        status_map = {
            'active': SubscriptionStatus.ACTIVE,
            'trialing': SubscriptionStatus.TRIALING,
            'past_due': SubscriptionStatus.PAST_DUE,
            'canceled': SubscriptionStatus.CANCELED,
            'unpaid': SubscriptionStatus.UNPAID,
            'incomplete': SubscriptionStatus.INCOMPLETE,
            'incomplete_expired': SubscriptionStatus.INCOMPLETE_EXPIRED,
            'paused': SubscriptionStatus.PAUSED,
        }
        return status_map.get(stripe_status, SubscriptionStatus.INCOMPLETE)
