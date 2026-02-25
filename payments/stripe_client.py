"""
Stripe Client for RDT Trading System.

Handles all Stripe API operations including:
- Customer creation and management
- Subscription lifecycle
- Checkout session creation
- Billing portal sessions
"""

import os
from typing import Optional, Dict, Any, List
from datetime import datetime

import stripe
from loguru import logger


class StripeError(Exception):
    """Custom exception for Stripe-related errors."""

    def __init__(self, message: str, code: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class StripeClient:
    """
    Client for Stripe API operations.

    Provides methods for:
    - Customer management (create, retrieve, update)
    - Subscription management (create, update, cancel, retrieve)
    - Checkout session creation
    - Billing portal session creation
    """

    def __init__(self, secret_key: Optional[str] = None, webhook_secret: Optional[str] = None):
        """
        Initialize the Stripe client.

        Args:
            secret_key: Stripe secret key. Defaults to STRIPE_SECRET_KEY env var.
            webhook_secret: Stripe webhook secret. Defaults to STRIPE_WEBHOOK_SECRET env var.
        """
        self.secret_key = secret_key or os.environ.get('STRIPE_SECRET_KEY')
        self.webhook_secret = webhook_secret or os.environ.get('STRIPE_WEBHOOK_SECRET')

        if not self.secret_key:
            logger.warning("STRIPE_SECRET_KEY not set - Stripe operations will fail")
        else:
            stripe.api_key = self.secret_key
            logger.info("Stripe client initialized")

    def _handle_stripe_error(self, error: Exception) -> None:
        """
        Handle Stripe API errors and convert to StripeError.

        Args:
            error: The original Stripe error

        Raises:
            StripeError: Converted error with details
        """
        if isinstance(error, stripe.error.CardError):
            logger.error(f"Card error: {error.user_message}")
            raise StripeError(
                message=error.user_message,
                code=error.code,
                status_code=error.http_status
            )
        elif isinstance(error, stripe.error.RateLimitError):
            logger.error("Stripe rate limit exceeded")
            raise StripeError(
                message="Too many requests to payment service. Please try again.",
                code="rate_limit",
                status_code=429
            )
        elif isinstance(error, stripe.error.InvalidRequestError):
            logger.error(f"Invalid Stripe request: {error.user_message}")
            raise StripeError(
                message=str(error.user_message),
                code=error.code,
                status_code=error.http_status
            )
        elif isinstance(error, stripe.error.AuthenticationError):
            logger.error("Stripe authentication failed - check API key")
            raise StripeError(
                message="Payment service authentication failed",
                code="auth_error",
                status_code=401
            )
        elif isinstance(error, stripe.error.APIConnectionError):
            logger.error("Could not connect to Stripe API")
            raise StripeError(
                message="Could not connect to payment service",
                code="connection_error",
                status_code=503
            )
        elif isinstance(error, stripe.error.StripeError):
            logger.error(f"Stripe error: {error}")
            raise StripeError(
                message="Payment service error. Please try again.",
                code="stripe_error",
                status_code=500
            )
        else:
            logger.error(f"Unexpected error: {error}")
            raise StripeError(
                message="An unexpected error occurred",
                code="unknown_error",
                status_code=500
            )

    # =========================================================================
    # Customer Management
    # =========================================================================

    def create_customer(
        self,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None
    ) -> stripe.Customer:
        """
        Create a new Stripe customer.

        Args:
            email: Customer email address
            name: Customer name (optional)
            metadata: Additional metadata to store with customer
            idempotency_key: Unique key for request idempotency (optional)

        Returns:
            stripe.Customer: The created customer object

        Raises:
            StripeError: If customer creation fails
        """
        try:
            customer_data = {
                'email': email,
            }
            if name:
                customer_data['name'] = name
            if metadata:
                customer_data['metadata'] = metadata

            # Add idempotency key if provided
            create_kwargs = {}
            if idempotency_key:
                create_kwargs['idempotency_key'] = idempotency_key

            customer = stripe.Customer.create(**customer_data, **create_kwargs)
            logger.info(f"Created Stripe customer: {customer.id} for {email}")
            return customer

        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    def get_customer(self, customer_id: str) -> Optional[stripe.Customer]:
        """
        Retrieve a Stripe customer by ID.

        Args:
            customer_id: Stripe customer ID

        Returns:
            stripe.Customer or None if not found

        Raises:
            StripeError: If retrieval fails (except for not found)
        """
        try:
            customer = stripe.Customer.retrieve(customer_id)
            return customer
        except stripe.error.InvalidRequestError as e:
            if 'No such customer' in str(e):
                logger.warning(f"Customer not found: {customer_id}")
                return None
            self._handle_stripe_error(e)
        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    def update_customer(
        self,
        customer_id: str,
        email: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> stripe.Customer:
        """
        Update a Stripe customer.

        Args:
            customer_id: Stripe customer ID
            email: New email address (optional)
            name: New name (optional)
            metadata: Metadata to merge with existing (optional)

        Returns:
            stripe.Customer: The updated customer object

        Raises:
            StripeError: If update fails
        """
        try:
            update_data = {}
            if email:
                update_data['email'] = email
            if name:
                update_data['name'] = name
            if metadata:
                update_data['metadata'] = metadata

            customer = stripe.Customer.modify(customer_id, **update_data)
            logger.info(f"Updated Stripe customer: {customer_id}")
            return customer

        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    def get_or_create_customer(
        self,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> stripe.Customer:
        """
        Get existing customer by email or create a new one.

        Args:
            email: Customer email address
            name: Customer name (optional)
            metadata: Metadata for new customer (optional)

        Returns:
            stripe.Customer: Existing or new customer object

        Raises:
            StripeError: If operation fails
        """
        try:
            # Search for existing customer by email
            customers = stripe.Customer.list(email=email, limit=1)

            if customers.data:
                logger.info(f"Found existing customer for {email}")
                return customers.data[0]

            # Create new customer
            return self.create_customer(email=email, name=name, metadata=metadata)

        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    # =========================================================================
    # Subscription Management
    # =========================================================================

    def create_subscription(
        self,
        customer_id: str,
        price_id: str,
        trial_days: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None
    ) -> stripe.Subscription:
        """
        Create a new subscription for a customer.

        Args:
            customer_id: Stripe customer ID
            price_id: Stripe price ID for the subscription
            trial_days: Number of trial days (optional)
            metadata: Additional metadata (optional)
            idempotency_key: Unique key for request idempotency (optional)

        Returns:
            stripe.Subscription: The created subscription object

        Raises:
            StripeError: If subscription creation fails
        """
        try:
            sub_data = {
                'customer': customer_id,
                'items': [{'price': price_id}],
                'payment_behavior': 'default_incomplete',
                'expand': ['latest_invoice.payment_intent'],
            }

            if trial_days and trial_days > 0:
                sub_data['trial_period_days'] = trial_days

            if metadata:
                sub_data['metadata'] = metadata

            # Add idempotency key if provided
            create_kwargs = {}
            if idempotency_key:
                create_kwargs['idempotency_key'] = idempotency_key

            subscription = stripe.Subscription.create(**sub_data, **create_kwargs)
            logger.info(f"Created subscription {subscription.id} for customer {customer_id}")
            return subscription

        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    def get_subscription(self, subscription_id: str) -> Optional[stripe.Subscription]:
        """
        Retrieve a subscription by ID.

        Args:
            subscription_id: Stripe subscription ID

        Returns:
            stripe.Subscription or None if not found

        Raises:
            StripeError: If retrieval fails (except for not found)
        """
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            return subscription
        except stripe.error.InvalidRequestError as e:
            if 'No such subscription' in str(e):
                logger.warning(f"Subscription not found: {subscription_id}")
                return None
            self._handle_stripe_error(e)
        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    def get_customer_subscriptions(
        self,
        customer_id: str,
        status: Optional[str] = None
    ) -> List[stripe.Subscription]:
        """
        Get all subscriptions for a customer.

        Args:
            customer_id: Stripe customer ID
            status: Filter by status (active, canceled, etc.)

        Returns:
            List of subscription objects

        Raises:
            StripeError: If retrieval fails
        """
        try:
            params = {'customer': customer_id, 'limit': 100}
            if status:
                params['status'] = status

            subscriptions = stripe.Subscription.list(**params)
            return subscriptions.data

        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    def update_subscription(
        self,
        subscription_id: str,
        price_id: Optional[str] = None,
        proration_behavior: str = 'create_prorations',
        metadata: Optional[Dict[str, str]] = None
    ) -> stripe.Subscription:
        """
        Update a subscription (e.g., change plan).

        Args:
            subscription_id: Stripe subscription ID
            price_id: New price ID to switch to (optional)
            proration_behavior: How to handle prorations
            metadata: Metadata to update (optional)

        Returns:
            stripe.Subscription: The updated subscription object

        Raises:
            StripeError: If update fails
        """
        try:
            update_data = {}

            if price_id:
                # Get current subscription to find item ID
                subscription = stripe.Subscription.retrieve(subscription_id)
                if subscription.items.data:
                    item_id = subscription.items.data[0].id
                    update_data['items'] = [{
                        'id': item_id,
                        'price': price_id
                    }]
                    update_data['proration_behavior'] = proration_behavior

            if metadata:
                update_data['metadata'] = metadata

            if update_data:
                subscription = stripe.Subscription.modify(subscription_id, **update_data)
                logger.info(f"Updated subscription: {subscription_id}")
                return subscription

            return stripe.Subscription.retrieve(subscription_id)

        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    def cancel_subscription(
        self,
        subscription_id: str,
        immediately: bool = False
    ) -> stripe.Subscription:
        """
        Cancel a subscription.

        Args:
            subscription_id: Stripe subscription ID
            immediately: If True, cancel immediately. If False, cancel at period end.

        Returns:
            stripe.Subscription: The canceled subscription object

        Raises:
            StripeError: If cancellation fails
        """
        try:
            if immediately:
                subscription = stripe.Subscription.cancel(subscription_id)
                logger.info(f"Canceled subscription immediately: {subscription_id}")
            else:
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True
                )
                logger.info(f"Scheduled subscription cancellation at period end: {subscription_id}")

            return subscription

        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    def reactivate_subscription(self, subscription_id: str) -> stripe.Subscription:
        """
        Reactivate a subscription that was scheduled for cancellation.

        Args:
            subscription_id: Stripe subscription ID

        Returns:
            stripe.Subscription: The reactivated subscription object

        Raises:
            StripeError: If reactivation fails
        """
        try:
            subscription = stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=False
            )
            logger.info(f"Reactivated subscription: {subscription_id}")
            return subscription

        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    # =========================================================================
    # Checkout Sessions
    # =========================================================================

    def create_checkout_session(
        self,
        price_id: str,
        success_url: str,
        cancel_url: str,
        customer_id: Optional[str] = None,
        customer_email: Optional[str] = None,
        trial_days: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        allow_promotion_codes: bool = True
    ) -> stripe.checkout.Session:
        """
        Create a Stripe Checkout session for subscription signup.

        Args:
            price_id: Stripe price ID for the subscription
            success_url: URL to redirect on successful payment
            cancel_url: URL to redirect on cancellation
            customer_id: Existing customer ID (optional)
            customer_email: Email to prefill (optional, ignored if customer_id set)
            trial_days: Number of trial days (optional)
            metadata: Session metadata (optional)
            allow_promotion_codes: Allow promo codes (default True)

        Returns:
            stripe.checkout.Session: The checkout session object

        Raises:
            StripeError: If session creation fails
        """
        try:
            session_data = {
                'mode': 'subscription',
                'line_items': [{
                    'price': price_id,
                    'quantity': 1,
                }],
                'success_url': success_url,
                'cancel_url': cancel_url,
                'allow_promotion_codes': allow_promotion_codes,
            }

            if customer_id:
                session_data['customer'] = customer_id
            elif customer_email:
                session_data['customer_email'] = customer_email

            if trial_days and trial_days > 0:
                session_data['subscription_data'] = {
                    'trial_period_days': trial_days
                }

            if metadata:
                session_data['metadata'] = metadata
                if 'subscription_data' in session_data:
                    session_data['subscription_data']['metadata'] = metadata
                else:
                    session_data['subscription_data'] = {'metadata': metadata}

            session = stripe.checkout.Session.create(**session_data)
            logger.info(f"Created checkout session: {session.id}")
            return session

        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    def get_checkout_session(self, session_id: str) -> Optional[stripe.checkout.Session]:
        """
        Retrieve a checkout session by ID.

        Args:
            session_id: Stripe checkout session ID

        Returns:
            stripe.checkout.Session or None if not found

        Raises:
            StripeError: If retrieval fails (except for not found)
        """
        try:
            session = stripe.checkout.Session.retrieve(session_id)
            return session
        except stripe.error.InvalidRequestError as e:
            if 'No such checkout' in str(e):
                logger.warning(f"Checkout session not found: {session_id}")
                return None
            self._handle_stripe_error(e)
        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    # =========================================================================
    # Billing Portal
    # =========================================================================

    def create_portal_session(
        self,
        customer_id: str,
        return_url: str
    ) -> stripe.billing_portal.Session:
        """
        Create a Stripe Billing Portal session for customer self-service.

        Args:
            customer_id: Stripe customer ID
            return_url: URL to return to after portal session

        Returns:
            stripe.billing_portal.Session: The portal session object

        Raises:
            StripeError: If session creation fails
        """
        try:
            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url
            )
            logger.info(f"Created portal session for customer: {customer_id}")
            return session

        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    # =========================================================================
    # Webhook Verification
    # =========================================================================

    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str
    ) -> stripe.Event:
        """
        Verify a webhook signature and construct the event.

        Args:
            payload: Raw request body
            signature: Stripe-Signature header value

        Returns:
            stripe.Event: The verified webhook event

        Raises:
            StripeError: If verification fails
        """
        if not self.webhook_secret:
            raise StripeError(
                message="Webhook secret not configured",
                code="webhook_config_error",
                status_code=500
            )

        try:
            event = stripe.Webhook.construct_event(
                payload,
                signature,
                self.webhook_secret
            )
            return event

        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Webhook signature verification failed: {e}")
            raise StripeError(
                message="Invalid webhook signature",
                code="invalid_signature",
                status_code=400
            )

    # =========================================================================
    # Invoices
    # =========================================================================

    def get_customer_invoices(
        self,
        customer_id: str,
        limit: int = 10
    ) -> List[stripe.Invoice]:
        """
        Get invoices for a customer.

        Args:
            customer_id: Stripe customer ID
            limit: Maximum number of invoices to return

        Returns:
            List of invoice objects

        Raises:
            StripeError: If retrieval fails
        """
        try:
            invoices = stripe.Invoice.list(customer=customer_id, limit=limit)
            return invoices.data

        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    def get_upcoming_invoice(self, customer_id: str) -> Optional[stripe.Invoice]:
        """
        Get the upcoming invoice for a customer.

        Args:
            customer_id: Stripe customer ID

        Returns:
            stripe.Invoice or None if no upcoming invoice

        Raises:
            StripeError: If retrieval fails
        """
        try:
            invoice = stripe.Invoice.upcoming(customer=customer_id)
            return invoice
        except stripe.error.InvalidRequestError as e:
            if 'No upcoming invoices' in str(e):
                return None
            self._handle_stripe_error(e)
        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    # =========================================================================
    # Refunds
    # =========================================================================

    def create_refund(
        self,
        payment_intent_id: Optional[str] = None,
        charge_id: Optional[str] = None,
        amount: Optional[int] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None
    ) -> stripe.Refund:
        """
        Create a refund for a payment.

        Must provide either payment_intent_id or charge_id.

        Args:
            payment_intent_id: Stripe PaymentIntent ID (optional)
            charge_id: Stripe Charge ID (optional)
            amount: Amount to refund in cents (optional, full refund if not specified)
            reason: Reason for refund: duplicate, fraudulent, or requested_by_customer
            metadata: Additional metadata (optional)
            idempotency_key: Unique key for request idempotency (optional)

        Returns:
            stripe.Refund: The created refund object

        Raises:
            StripeError: If refund creation fails
        """
        if not payment_intent_id and not charge_id:
            raise StripeError(
                message="Either payment_intent_id or charge_id must be provided",
                code="invalid_params",
                status_code=400
            )

        try:
            refund_data = {}

            if payment_intent_id:
                refund_data['payment_intent'] = payment_intent_id
            elif charge_id:
                refund_data['charge'] = charge_id

            if amount:
                refund_data['amount'] = amount

            if reason:
                refund_data['reason'] = reason

            if metadata:
                refund_data['metadata'] = metadata

            # Add idempotency key if provided
            create_kwargs = {}
            if idempotency_key:
                create_kwargs['idempotency_key'] = idempotency_key

            refund = stripe.Refund.create(**refund_data, **create_kwargs)
            logger.info(f"Created refund {refund.id} for amount {refund.amount}")
            return refund

        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    def get_refund(self, refund_id: str) -> Optional[stripe.Refund]:
        """
        Retrieve a refund by ID.

        Args:
            refund_id: Stripe refund ID

        Returns:
            stripe.Refund or None if not found

        Raises:
            StripeError: If retrieval fails (except for not found)
        """
        try:
            refund = stripe.Refund.retrieve(refund_id)
            return refund
        except stripe.error.InvalidRequestError as e:
            if 'No such refund' in str(e):
                logger.warning(f"Refund not found: {refund_id}")
                return None
            self._handle_stripe_error(e)
        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)

    def list_refunds(
        self,
        payment_intent_id: Optional[str] = None,
        charge_id: Optional[str] = None,
        limit: int = 10
    ) -> List[stripe.Refund]:
        """
        List refunds, optionally filtered by payment intent or charge.

        Args:
            payment_intent_id: Filter by payment intent (optional)
            charge_id: Filter by charge (optional)
            limit: Maximum number of refunds to return

        Returns:
            List of refund objects

        Raises:
            StripeError: If retrieval fails
        """
        try:
            params = {'limit': limit}

            if payment_intent_id:
                params['payment_intent'] = payment_intent_id
            elif charge_id:
                params['charge'] = charge_id

            refunds = stripe.Refund.list(**params)
            return refunds.data

        except stripe.error.StripeError as e:
            self._handle_stripe_error(e)
