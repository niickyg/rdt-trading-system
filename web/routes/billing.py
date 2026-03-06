"""
Billing routes for RDT Trading System.

Provides endpoints for:
- Billing dashboard
- Checkout session creation
- Stripe customer portal
- Stripe webhooks
"""

import os
import sys
from datetime import datetime
from typing import Optional

from flask import Blueprint, render_template, redirect, url_for, request, jsonify, flash, current_app
from flask_login import login_required, current_user
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from payments.stripe_client import StripeClient, StripeError
from payments.plans import (
    get_plan_by_id,
    get_plan_by_price_id,
    get_price_id,
    get_trial_days,
    get_paid_plans,
    SUBSCRIPTION_PLANS,
)
from data.database.models import User, Subscription, SubscriptionStatus, PaymentHistory, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# Create blueprint
billing_bp = Blueprint('billing', __name__, url_prefix='/billing')

# Initialize Stripe client (singleton)
_stripe_client: Optional[StripeClient] = None


def get_stripe_client() -> StripeClient:
    """Get or create the Stripe client instance."""
    global _stripe_client
    if _stripe_client is None:
        _stripe_client = StripeClient()
    return _stripe_client


def get_db_path():
    """Get the database path."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    db_dir = os.path.join(base_dir, 'data', 'database')
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, 'rdt_auth.db')


def get_db_session():
    """Get database session."""
    db_path = get_db_path()
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def get_user_subscription(user_id: int) -> Optional[Subscription]:
    """Get the active subscription for a user."""
    session = get_db_session()
    return session.query(Subscription).filter(
        Subscription.user_id == user_id,
        Subscription.status.in_([
            SubscriptionStatus.ACTIVE,
            SubscriptionStatus.TRIALING,
            SubscriptionStatus.PAST_DUE
        ])
    ).first()


def get_base_url() -> str:
    """Get the base URL for the application."""
    return os.environ.get('APP_BASE_URL', request.host_url.rstrip('/'))


# =============================================================================
# Billing Dashboard
# =============================================================================

@billing_bp.route('/')
@login_required
def billing_dashboard():
    """
    Billing dashboard showing current subscription and options.
    """
    # Get user's current subscription
    subscription = get_user_subscription(current_user.id)

    # Get plan details
    current_plan = None
    if subscription:
        current_plan = get_plan_by_id(subscription.plan_id)

    # Get payment history
    db_session = get_db_session()
    payment_history = db_session.query(PaymentHistory).filter(
        PaymentHistory.user_id == current_user.id
    ).order_by(PaymentHistory.created_at.desc()).limit(10).all()

    # Get available plans for upgrade/downgrade
    available_plans = get_paid_plans()

    # Get upcoming invoice if applicable
    upcoming_invoice = None
    if current_user.stripe_customer_id:
        try:
            stripe_client = get_stripe_client()
            upcoming_invoice = stripe_client.get_upcoming_invoice(current_user.stripe_customer_id)
        except StripeError:
            pass

    return render_template(
        'billing.html',
        user=current_user,
        subscription=subscription,
        current_plan=current_plan,
        available_plans=available_plans,
        payment_history=payment_history,
        upcoming_invoice=upcoming_invoice
    )


# =============================================================================
# Checkout
# =============================================================================

@billing_bp.route('/checkout', methods=['POST'])
@login_required
def create_checkout():
    """
    Create a Stripe Checkout session for subscription.
    """
    plan_id = request.form.get('plan_id') or request.json.get('plan_id')
    annual = request.form.get('annual', 'false').lower() == 'true'

    if not plan_id:
        flash('Please select a plan.', 'error')
        return redirect(url_for('billing.billing_dashboard'))

    plan = get_plan_by_id(plan_id)
    if not plan:
        flash('Invalid plan selected.', 'error')
        return redirect(url_for('billing.billing_dashboard'))

    # Get price ID
    price_id = get_price_id(plan_id, annual=annual)
    if not price_id:
        flash('Plan pricing not configured.', 'error')
        return redirect(url_for('billing.billing_dashboard'))

    try:
        stripe_client = get_stripe_client()
        base_url = get_base_url()

        # Get or create Stripe customer
        customer_id = current_user.stripe_customer_id
        if not customer_id:
            customer = stripe_client.get_or_create_customer(
                email=current_user.email,
                name=current_user.username,
                metadata={'user_id': str(current_user.id)}
            )
            customer_id = customer.id

            # Update user with Stripe customer ID
            db_session = get_db_session()
            user = db_session.query(User).filter_by(id=current_user.id).first()
            if user:
                user.stripe_customer_id = customer_id
                db_session.commit()

        # Create checkout session
        session = stripe_client.create_checkout_session(
            price_id=price_id,
            success_url=f"{base_url}/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{base_url}/billing/cancel",
            customer_id=customer_id,
            trial_days=get_trial_days(plan_id),
            metadata={
                'user_id': str(current_user.id),
                'plan_id': plan_id
            }
        )

        logger.info(f"Created checkout session {session.id} for user {current_user.id}")

        # Redirect to Stripe Checkout
        return redirect(session.url)

    except StripeError as e:
        logger.error(f"Checkout error for user {current_user.id}: {e.message}")
        flash(f'Unable to create checkout session: {e.message}', 'error')
        return redirect(url_for('billing.billing_dashboard'))


@billing_bp.route('/checkout/<plan_id>')
@login_required
def checkout_plan(plan_id: str):
    """
    Direct checkout URL for a specific plan.
    """
    plan = get_plan_by_id(plan_id)
    if not plan:
        flash('Invalid plan selected.', 'error')
        return redirect(url_for('pricing'))

    annual = request.args.get('annual', 'false').lower() == 'true'

    return render_template(
        'checkout.html',
        plan=plan,
        plan_id=plan_id,
        annual=annual,
        user=current_user
    )


@billing_bp.route('/success')
@login_required
def checkout_success():
    """
    Handle successful checkout completion.
    """
    session_id = request.args.get('session_id')

    if session_id:
        try:
            stripe_client = get_stripe_client()
            session = stripe_client.get_checkout_session(session_id)

            if session:
                logger.info(f"Checkout success for session {session_id}")

                # Get subscription details
                subscription_id = session.subscription
                customer_id = session.customer

                # Update user's Stripe customer ID if not set
                if customer_id and not current_user.stripe_customer_id:
                    db_session = get_db_session()
                    user = db_session.query(User).filter_by(id=current_user.id).first()
                    if user:
                        user.stripe_customer_id = customer_id
                        db_session.commit()

        except StripeError as e:
            logger.error(f"Error retrieving checkout session: {e.message}")

    flash('Welcome! Your subscription is now active.', 'success')
    return render_template('billing_success.html', user=current_user)


@billing_bp.route('/cancel')
def checkout_cancel():
    """
    Handle checkout cancellation.
    """
    flash('Checkout was cancelled. No charges were made.', 'info')
    return redirect(url_for('pricing'))


# =============================================================================
# Customer Portal
# =============================================================================

@billing_bp.route('/portal')
@login_required
def customer_portal():
    """
    Redirect to Stripe Customer Portal for self-service management.
    """
    if not current_user.stripe_customer_id:
        flash('No billing account found. Please subscribe to a plan first.', 'warning')
        return redirect(url_for('billing.billing_dashboard'))

    try:
        stripe_client = get_stripe_client()
        base_url = get_base_url()

        session = stripe_client.create_portal_session(
            customer_id=current_user.stripe_customer_id,
            return_url=f"{base_url}/billing"
        )

        logger.info(f"Created portal session for user {current_user.id}")
        return redirect(session.url)

    except StripeError as e:
        logger.error(f"Portal error for user {current_user.id}: {e.message}")
        flash(f'Unable to access billing portal: {e.message}', 'error')
        return redirect(url_for('billing.billing_dashboard'))


# =============================================================================
# Subscription Management
# =============================================================================

@billing_bp.route('/subscription/cancel', methods=['POST'])
@login_required
def cancel_subscription():
    """
    Cancel the current subscription (at period end).
    """
    subscription = get_user_subscription(current_user.id)

    if not subscription:
        flash('No active subscription found.', 'warning')
        return redirect(url_for('billing.billing_dashboard'))

    immediately = request.form.get('immediately', 'false').lower() == 'true'

    try:
        stripe_client = get_stripe_client()
        stripe_client.cancel_subscription(
            subscription_id=subscription.stripe_subscription_id,
            immediately=immediately
        )

        # Update local record
        db_session = get_db_session()
        sub = db_session.query(Subscription).filter_by(id=subscription.id).first()
        if sub:
            if immediately:
                sub.status = SubscriptionStatus.CANCELED
                sub.canceled_at = datetime.utcnow()
            else:
                sub.cancel_at_period_end = True
            sub.updated_at = datetime.utcnow()
            db_session.commit()

        if immediately:
            flash('Your subscription has been canceled.', 'info')
        else:
            flash('Your subscription will be canceled at the end of the billing period.', 'info')

        logger.info(f"User {current_user.id} canceled subscription {subscription.id}")

    except StripeError as e:
        logger.error(f"Cancel error for user {current_user.id}: {e.message}")
        flash(f'Unable to cancel subscription: {e.message}', 'error')

    return redirect(url_for('billing.billing_dashboard'))


@billing_bp.route('/subscription/reactivate', methods=['POST'])
@login_required
def reactivate_subscription():
    """
    Reactivate a subscription scheduled for cancellation.
    """
    subscription = get_user_subscription(current_user.id)

    if not subscription:
        flash('No subscription found.', 'warning')
        return redirect(url_for('billing.billing_dashboard'))

    if not subscription.cancel_at_period_end:
        flash('Subscription is not scheduled for cancellation.', 'info')
        return redirect(url_for('billing.billing_dashboard'))

    try:
        stripe_client = get_stripe_client()
        stripe_client.reactivate_subscription(subscription.stripe_subscription_id)

        # Update local record
        db_session = get_db_session()
        sub = db_session.query(Subscription).filter_by(id=subscription.id).first()
        if sub:
            sub.cancel_at_period_end = False
            sub.updated_at = datetime.utcnow()
            db_session.commit()

        flash('Your subscription has been reactivated.', 'success')
        logger.info(f"User {current_user.id} reactivated subscription {subscription.id}")

    except StripeError as e:
        logger.error(f"Reactivate error for user {current_user.id}: {e.message}")
        flash(f'Unable to reactivate subscription: {e.message}', 'error')

    return redirect(url_for('billing.billing_dashboard'))


@billing_bp.route('/subscription/change', methods=['POST'])
@login_required
def change_subscription():
    """
    Change subscription plan (upgrade/downgrade).
    """
    new_plan_id = request.form.get('plan_id')

    if not new_plan_id:
        flash('Please select a plan.', 'error')
        return redirect(url_for('billing.billing_dashboard'))

    subscription = get_user_subscription(current_user.id)

    if not subscription:
        flash('No active subscription. Please subscribe first.', 'warning')
        return redirect(url_for('billing.billing_dashboard'))

    new_plan = get_plan_by_id(new_plan_id)
    if not new_plan:
        flash('Invalid plan selected.', 'error')
        return redirect(url_for('billing.billing_dashboard'))

    new_price_id = get_price_id(new_plan_id)
    if not new_price_id:
        flash('Plan pricing not configured.', 'error')
        return redirect(url_for('billing.billing_dashboard'))

    try:
        stripe_client = get_stripe_client()
        stripe_client.update_subscription(
            subscription_id=subscription.stripe_subscription_id,
            price_id=new_price_id,
            proration_behavior='create_prorations'
        )

        # Update local record
        db_session = get_db_session()
        sub = db_session.query(Subscription).filter_by(id=subscription.id).first()
        if sub:
            sub.plan_id = new_plan_id
            sub.updated_at = datetime.utcnow()
            db_session.commit()

        flash(f'Successfully changed to {new_plan["name"]} plan.', 'success')
        logger.info(f"User {current_user.id} changed plan to {new_plan_id}")

    except StripeError as e:
        logger.error(f"Plan change error for user {current_user.id}: {e.message}")
        flash(f'Unable to change plan: {e.message}', 'error')

    return redirect(url_for('billing.billing_dashboard'))


# =============================================================================
# Webhook Endpoint
# =============================================================================

@billing_bp.route('/webhook', methods=['POST'])
def stripe_webhook():
    """
    Handle Stripe webhook events.

    This endpoint is called by Stripe to notify of subscription and payment events.
    Must be exempt from CSRF protection.
    """
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')

    if not sig_header:
        logger.warning("Webhook received without signature")
        return jsonify({'error': 'Missing signature'}), 400

    try:
        stripe_client = get_stripe_client()
        event = stripe_client.verify_webhook_signature(payload, sig_header)

    except StripeError as e:
        logger.error(f"Webhook signature verification failed: {e.message}")
        return jsonify({'error': 'Webhook verification failed'}), 400

    # Import webhook handler
    from payments.webhooks import WebhookHandler

    # Create handler with database session
    db_session = get_db_session()
    handler = WebhookHandler(
        db_session=db_session,
        email_callback=send_billing_email,
        alert_callback=send_billing_alert
    )

    # Process the event
    result = handler.handle_event(event)

    logger.info(f"Webhook processed: {event.type} -> {result.get('status')}")

    return jsonify(result), 200


# =============================================================================
# API Endpoints
# =============================================================================

@billing_bp.route('/api/subscription')
@login_required
def api_subscription_status():
    """
    API endpoint to get current subscription status.
    """
    subscription = get_user_subscription(current_user.id)

    if not subscription:
        return jsonify({
            'has_subscription': False,
            'plan': None,
            'status': None
        })

    plan = get_plan_by_id(subscription.plan_id)

    return jsonify({
        'has_subscription': True,
        'plan': {
            'id': subscription.plan_id,
            'name': plan.get('name') if plan else subscription.plan_id,
            'features': plan.get('features', []) if plan else []
        },
        'status': subscription.status.value if hasattr(subscription.status, 'value') else subscription.status,
        'is_active': subscription.is_active,
        'is_trialing': subscription.is_trialing,
        'will_cancel': subscription.will_cancel,
        'current_period_end': subscription.current_period_end.isoformat() if subscription.current_period_end else None,
        'trial_end': subscription.trial_end.isoformat() if subscription.trial_end else None
    })


@billing_bp.route('/api/plans')
def api_plans():
    """
    API endpoint to get available subscription plans.
    """
    plans = get_paid_plans()

    return jsonify({
        'plans': [
            {
                'id': plan_id,
                'name': plan['name'],
                'description': plan.get('description', ''),
                'price_monthly': plan['price_monthly'],
                'price_annual': plan.get('price_annual', plan['price_monthly'] * 12),
                'features': plan['features'],
                'popular': plan.get('popular', False),
                'trial_days': plan.get('trial_days', 0)
            }
            for plan_id, plan in plans.items()
        ]
    })


# =============================================================================
# Helper Functions
# =============================================================================

def send_billing_email(email: str, template: str, context: dict) -> None:
    """
    Send billing-related email notification.

    Uses the existing email alert infrastructure from the alerts module.

    Args:
        email: Recipient email address
        template: Email template name (subscription_created, payment_failed, etc.)
        context: Template context with variables
    """
    logger.info(f"Sending billing email: template={template}, to={email}")

    try:
        from alerts.email_alert import get_email_manager
        from alerts.email_templates import EmailTemplates

        email_manager = get_email_manager()

        # Build subject and body based on template
        subjects = {
            'subscription_created': 'Welcome to RDT Trading! Your subscription is active',
            'subscription_canceled': 'Your RDT Trading subscription has been canceled',
            'payment_failed': 'Action required: Payment failed for your RDT Trading subscription',
            'invoice_paid': 'Payment received - Thank you!',
            'trial_ending': 'Your RDT Trading trial ends soon',
            'plan_changed': 'Your subscription plan has been updated',
        }

        subject = subjects.get(template, f'RDT Trading - {template.replace("_", " ").title()}')

        # Generate HTML and plain text bodies
        html_body, plain_text = _generate_billing_email_content(template, context)

        # Send using the email manager
        result = email_manager.send(
            to_email=email,
            subject=subject,
            html_body=html_body,
            plain_text=plain_text,
            from_name='RDT Trading'
        )

        if result.success:
            logger.info(f"Billing email sent successfully to {email}")
        else:
            logger.error(f"Failed to send billing email: {result.error_message}")

    except ImportError:
        # Fallback to basic SMTP if alerts module not available
        logger.warning("Email manager not available, using fallback SMTP")
        _send_billing_email_smtp(email, template, context)
    except Exception as e:
        logger.error(f"Error sending billing email: {e}")


def _generate_billing_email_content(template: str, context: dict) -> tuple:
    """Generate HTML and plain text email content for billing templates."""

    plan_name = context.get('plan_name', 'your plan')
    amount = context.get('amount', 0)
    currency = context.get('currency', 'USD').upper()
    period_end = context.get('period_end', '')
    customer_name = context.get('customer_name', 'Valued Customer')

    templates = {
        'subscription_created': {
            'html': f'''
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <h2 style="color: #2563eb;">Welcome to RDT Trading!</h2>
                <p>Hi {customer_name},</p>
                <p>Thank you for subscribing to <strong>{plan_name}</strong>!</p>
                <p>Your subscription is now active and you have full access to all features included in your plan.</p>
                <h3>What's Next?</h3>
                <ul>
                    <li>Set up your broker connections in Settings</li>
                    <li>Configure your risk parameters</li>
                    <li>Start scanning for trading signals</li>
                </ul>
                <p>If you have any questions, reply to this email or visit our support center.</p>
                <p>Happy trading!</p>
                <p style="color: #666;">The RDT Trading Team</p>
            </body>
            </html>
            ''',
            'text': f'''Welcome to RDT Trading!

Hi {customer_name},

Thank you for subscribing to {plan_name}!

Your subscription is now active and you have full access to all features included in your plan.

What's Next?
- Set up your broker connections in Settings
- Configure your risk parameters
- Start scanning for trading signals

If you have any questions, reply to this email.

Happy trading!
The RDT Trading Team'''
        },
        'subscription_canceled': {
            'html': f'''
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <h2>Subscription Canceled</h2>
                <p>Hi {customer_name},</p>
                <p>Your RDT Trading subscription ({plan_name}) has been canceled.</p>
                <p>You will continue to have access until <strong>{period_end}</strong>.</p>
                <p>We're sorry to see you go. If you change your mind, you can resubscribe anytime.</p>
                <p style="color: #666;">The RDT Trading Team</p>
            </body>
            </html>
            ''',
            'text': f'''Subscription Canceled

Hi {customer_name},

Your RDT Trading subscription ({plan_name}) has been canceled.
You will continue to have access until {period_end}.

We're sorry to see you go. If you change your mind, you can resubscribe anytime.

The RDT Trading Team'''
        },
        'payment_failed': {
            'html': f'''
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <h2 style="color: #dc2626;">Payment Failed</h2>
                <p>Hi {customer_name},</p>
                <p>We were unable to process your payment of <strong>{currency} {amount/100:.2f}</strong> for {plan_name}.</p>
                <p>Please update your payment method to avoid service interruption.</p>
                <p><a href="https://app.rdttrading.com/billing" style="background-color: #2563eb; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Update Payment Method</a></p>
                <p style="color: #666;">The RDT Trading Team</p>
            </body>
            </html>
            ''',
            'text': f'''Payment Failed

Hi {customer_name},

We were unable to process your payment of {currency} {amount/100:.2f} for {plan_name}.

Please update your payment method to avoid service interruption.

Update your payment method at: https://app.rdttrading.com/billing

The RDT Trading Team'''
        },
        'invoice_paid': {
            'html': f'''
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <h2 style="color: #16a34a;">Payment Received</h2>
                <p>Hi {customer_name},</p>
                <p>Thank you! We've received your payment of <strong>{currency} {amount/100:.2f}</strong>.</p>
                <p>Your {plan_name} subscription is active.</p>
                <p style="color: #666;">The RDT Trading Team</p>
            </body>
            </html>
            ''',
            'text': f'''Payment Received

Hi {customer_name},

Thank you! We've received your payment of {currency} {amount/100:.2f}.
Your {plan_name} subscription is active.

The RDT Trading Team'''
        },
    }

    template_content = templates.get(template, {
        'html': f'<html><body><p>{template.replace("_", " ").title()}</p></body></html>',
        'text': template.replace('_', ' ').title()
    })

    return template_content['html'], template_content['text']


def _send_billing_email_smtp(email: str, template: str, context: dict) -> None:
    """Fallback SMTP email sending."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import os

    smtp_host = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
    smtp_port = int(os.environ.get('SMTP_PORT', '465'))
    smtp_user = os.environ.get('SMTP_USER')
    smtp_password = os.environ.get('SMTP_PASSWORD')

    if not all([smtp_user, smtp_password]):
        logger.warning("SMTP credentials not configured, skipping email")
        return

    try:
        html_body, plain_text = _generate_billing_email_content(template, context)

        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"RDT Trading - {template.replace('_', ' ').title()}"
        msg['From'] = smtp_user
        msg['To'] = email

        msg.attach(MIMEText(plain_text, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))

        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

        logger.info(f"Billing email sent via SMTP to {email}")

    except Exception as e:
        logger.error(f"SMTP email failed: {e}")


def send_billing_alert(alert_type: str, data: dict) -> None:
    """
    Send billing-related alert notification to admin channels.

    Uses Discord and/or other configured admin notification channels.

    Args:
        alert_type: Type of billing event (payment_failed, subscription_created, etc.)
        data: Event data containing details
    """
    logger.info(f"Billing alert: type={alert_type}, data={data}")

    # Build alert message
    emojis = {
        'subscription_created': '🎉',
        'subscription_canceled': '👋',
        'payment_failed': '⚠️',
        'invoice_paid': '💰',
        'trial_ending': '⏰',
        'plan_changed': '🔄',
        'refund_issued': '💸',
    }

    emoji = emojis.get(alert_type, '📢')
    customer_email = data.get('customer_email', 'Unknown')
    plan_name = data.get('plan_name', 'Unknown')
    amount = data.get('amount', 0)
    currency = data.get('currency', 'USD').upper()

    # Format message based on type
    messages = {
        'subscription_created': f"New subscription! {customer_email} subscribed to {plan_name}",
        'subscription_canceled': f"Subscription canceled: {customer_email} ({plan_name})",
        'payment_failed': f"PAYMENT FAILED: {customer_email} - {currency} {amount/100:.2f}",
        'invoice_paid': f"Payment received: {customer_email} - {currency} {amount/100:.2f}",
        'trial_ending': f"Trial ending: {customer_email}",
        'plan_changed': f"Plan changed: {customer_email} -> {plan_name}",
        'refund_issued': f"Refund issued: {customer_email} - {currency} {amount/100:.2f}",
    }

    message = f"{emoji} {messages.get(alert_type, f'Billing event: {alert_type}')}"

    # Try Discord first
    try:
        from alerts.discord_alert import DiscordAlert
        import os

        webhook_url = os.environ.get('DISCORD_ADMIN_WEBHOOK') or os.environ.get('DISCORD_WEBHOOK')
        if webhook_url:
            discord = DiscordAlert(webhook_url=webhook_url)

            # Build embed for rich message
            embed = {
                'title': f'Billing: {alert_type.replace("_", " ").title()}',
                'description': message,
                'color': 0x10B981 if 'paid' in alert_type or 'created' in alert_type else 0xEF4444 if 'failed' in alert_type else 0x6B7280,
                'fields': [
                    {'name': 'Customer', 'value': customer_email, 'inline': True},
                    {'name': 'Plan', 'value': plan_name, 'inline': True},
                ],
                'timestamp': data.get('timestamp', None),
            }

            if amount:
                embed['fields'].append({'name': 'Amount', 'value': f'{currency} {amount/100:.2f}', 'inline': True})

            result = discord.send(
                title=f'Billing Alert: {alert_type}',
                message=message,
                color='success' if 'paid' in alert_type or 'created' in alert_type else 'danger' if 'failed' in alert_type else 'warning',
            )

            if result.success:
                logger.info(f"Billing alert sent to Discord")
            else:
                logger.warning(f"Discord alert failed: {result.error_message}")

    except ImportError:
        logger.debug("Discord alert module not available")
    except Exception as e:
        logger.warning(f"Error sending Discord alert: {e}")

    # Also try Slack if configured
    try:
        import os
        import requests

        slack_webhook = os.environ.get('SLACK_BILLING_WEBHOOK') or os.environ.get('SLACK_WEBHOOK')
        if slack_webhook:
            payload = {
                'text': message,
                'username': 'RDT Billing',
                'icon_emoji': ':money_with_wings:',
            }
            response = requests.post(slack_webhook, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("Billing alert sent to Slack")
            else:
                logger.warning(f"Slack alert failed: {response.status_code}")

    except Exception as e:
        logger.debug(f"Slack alert not sent: {e}")
