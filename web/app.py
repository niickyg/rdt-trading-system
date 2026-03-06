"""
RDT Trading Signal Service - Flask Web Application

Provides:
- Landing page for customer acquisition
- Dashboard for signal viewing
- API integration for programmatic access
- Subscription management via Stripe
- User authentication with Flask-Login
- WebSocket streaming for real-time updates
- Prometheus metrics endpoint for monitoring
- Centralized logging with ELK/Loki support
- Distributed tracing with OpenTelemetry
"""

import os
import sys
import time
import threading
import collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, redirect, url_for, request, Response, g, send_from_directory
from flask_cors import CORS
from flask_login import login_required, current_user
from flask_wtf.csrf import CSRFProtect, CSRFError
from loguru import logger

from utils.timezone import format_timestamp
from utils.secrets import get_flask_secret_key, InsecureSecretWarning

# Import distributed tracing
try:
    from tracing import (
        init_tracing,
        init_flask_tracing,
        get_tracer,
        get_current_trace_id,
        get_trace_context_for_propagation,
    )
    TRACING_AVAILABLE = True
except ImportError as e:
    TRACING_AVAILABLE = False
    logger.warning(f"Distributed tracing not available: {e}")

# Import centralized logging
try:
    from rdt_logging import (
        configure_logging,
        create_flask_context_middleware,
        get_logger,
        LogContext,
        get_correlation_id,
    )
    CENTRALIZED_LOGGING_AVAILABLE = True
    # Initialize centralized logging immediately so all subsequent imports and
    # startup code benefit from the configured handlers/formatters.
    configure_logging()
except ImportError as e:
    CENTRALIZED_LOGGING_AVAILABLE = False
    logger.warning(f"Centralized logging not available: {e}")

# Import Prometheus metrics
try:
    from monitoring.metrics import (
        get_metrics,
        get_metrics_content_type,
        create_metrics_middleware,
        update_websocket_metrics,
    )
    METRICS_AVAILABLE = True
except ImportError as e:
    METRICS_AVAILABLE = False
    logger.warning(f"Prometheus metrics not available: {e}")

# Create Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Configure CORS with restricted origins
# SECURITY: Do NOT use CORS(app) without origin restrictions in production
cors_origins = [
    origin.strip() for origin in
    os.environ.get('CORS_ORIGINS', 'http://localhost:5000,http://127.0.0.1:5000').split(',')
    if origin.strip()
]
CORS(app,
     origins=cors_origins,
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization', 'X-API-Key', 'X-Requested-With'],
     supports_credentials=True,
     max_age=3600
)

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Configuration - Use secure secret key management
# This will:
# - Raise an error in production if SECRET_KEY is not set or is insecure
# - Warn in development if using default key
# - Validate key strength in production
try:
    app.config['SECRET_KEY'] = get_flask_secret_key()
except Exception as e:
    logger.error(f"SECRET_KEY configuration error: {e}")
    raise

app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)
app.config['SESSION_PROTECTION'] = 'basic'  # 'strong' causes random logouts on IP/UA changes

# Session cookie security settings
app.config['SESSION_COOKIE_SECURE'] = not app.debug  # HTTPS only in production
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection
app.config['REMEMBER_COOKIE_SECURE'] = not app.debug
app.config['REMEMBER_COOKIE_HTTPONLY'] = True
app.config['REMEMBER_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)  # Session timeout


@app.before_request
def _make_session_permanent():
    """Mark every session as permanent so PERMANENT_SESSION_LIFETIME applies."""
    from flask import session
    session.permanent = True


# Initialize WebSocket support
try:
    from web.websocket import init_websocket, get_room_stats
    # Use threading mode for maximum compatibility (eventlet requires early monkey patching)
    async_mode = os.environ.get('WEBSOCKET_ASYNC_MODE', 'threading')
    socketio = init_websocket(app, async_mode=async_mode)
    logger.info(f"WebSocket support initialized (mode: {async_mode})")
except ImportError as e:
    socketio = None
    logger.warning(f"Could not import websocket module: {e}")

# Initialize Dashboard Authentication
try:
    from web.auth import init_auth
    init_auth(app)
    logger.info("Dashboard authentication initialized")
except ImportError as e:
    logger.warning(f"Could not import auth module: {e}")

# Import and register API blueprint
try:
    from api.v1.routes import api_bp
    from api.v1.auth import init_api_auth
    app.register_blueprint(api_bp)
    logger.info("API v1 blueprint registered")

    # Exempt API routes from CSRF protection (they use API key authentication)
    csrf.exempt(api_bp)
    logger.info("API v1 blueprint exempted from CSRF protection")

    # Initialize API authentication (creates database tables and default test key)
    init_api_auth()
    logger.info("API authentication initialized")
except ImportError as e:
    logger.warning(f"Could not import API blueprint: {e}")

# Import and register Options API blueprint
try:
    from web.routes.options import options_bp
    app.register_blueprint(options_bp)
    # Options routes use @login_required (session auth), so CSRF stays enabled
    logger.info("Options API blueprint registered")
except ImportError as e:
    logger.warning(f"Could not import Options blueprint: {e}")

# Import and register Dashboard Data blueprint (session-auth AJAX routes)
try:
    from web.routes.dashboard_data import dashboard_data_bp
    app.register_blueprint(dashboard_data_bp)
    logger.info("Dashboard data blueprint registered")
except ImportError as e:
    logger.warning(f"Could not import Dashboard data blueprint: {e}")

# Import and register GraphQL blueprint
try:
    from api.graphql import graphql_bp
    app.register_blueprint(graphql_bp)
    logger.info("GraphQL blueprint registered at /graphql")

    # Exempt GraphQL routes from CSRF protection (uses API key authentication)
    csrf.exempt(graphql_bp)
    logger.info("GraphQL blueprint exempted from CSRF protection")
except ImportError as e:
    logger.warning(f"Could not import GraphQL blueprint: {e}")

# Initialize Prometheus metrics middleware
if METRICS_AVAILABLE:
    create_metrics_middleware(app)
    logger.info("Prometheus metrics middleware initialized")

# Initialize distributed tracing
if TRACING_AVAILABLE:
    # Initialize OpenTelemetry tracing
    tracing_initialized = init_tracing(service_name="rdt-trading-system")
    if tracing_initialized:
        # Add Flask tracing middleware
        init_flask_tracing(
            app,
            excluded_paths=["/health", "/health/live", "/health/ready", "/health/detailed", "/metrics", "/favicon.ico", "/static"],
            excluded_methods=["OPTIONS"],
        )
        logger.info("Distributed tracing initialized with OpenTelemetry")
    else:
        logger.info("Tracing initialization skipped (disabled or not configured)")

# Initialize request validation middleware
try:
    from web.middleware import init_request_validation
    init_request_validation(app, {
        'max_content_length': 10 * 1024 * 1024,  # 10 MB
        'json_max_length': 1 * 1024 * 1024,  # 1 MB for JSON
    })
    logger.info("Request validation middleware initialized")
except ImportError as e:
    logger.warning(f"Could not import middleware: {e}")


# =============================================================================
# WEB ROUTE RATE LIMITING
# =============================================================================

# In-memory store: { ip_address: deque([timestamp, ...]) }
# Uses a sliding window of 60 seconds, limit of 60 requests per window.
_WEB_RATE_LIMIT_REQUESTS = 60   # max requests per window
_WEB_RATE_LIMIT_WINDOW = 60     # window size in seconds
_web_rate_limit_store: dict = {}
_web_rate_limit_lock = threading.Lock()

# Paths that are excluded from web rate limiting
_WEB_RATE_LIMIT_SKIP_PREFIXES = (
    '/static/',
    '/health',
    '/metrics',
    '/favicon.ico',
    '/api/v1/',      # API v1 has its own per-key rate limiting
    '/api/graphql',  # GraphQL uses API key auth
)


@app.before_request
def web_rate_limit():
    """
    Apply rate limiting to web/dashboard routes.

    Limits each client IP to 60 requests per 60-second sliding window.
    API routes (/api/v1/, /api/graphql) are excluded — they have their own
    per-API-key rate limiting. Static files and health checks are also excluded.
    """
    path = request.path

    # Skip excluded paths
    for prefix in _WEB_RATE_LIMIT_SKIP_PREFIXES:
        if path.startswith(prefix):
            return None

    # Identify client by IP (X-Forwarded-For respected for reverse-proxy setups)
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if client_ip:
        # X-Forwarded-For may be a comma-separated list; take the first entry
        client_ip = client_ip.split(',')[0].strip()

    now = time.time()
    window_start = now - _WEB_RATE_LIMIT_WINDOW

    with _web_rate_limit_lock:
        if client_ip not in _web_rate_limit_store:
            _web_rate_limit_store[client_ip] = collections.deque()

        timestamps = _web_rate_limit_store[client_ip]

        # Evict timestamps outside the current window
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        # Periodically purge stale IPs to prevent unbounded dict growth
        if len(_web_rate_limit_store) > 10000:
            stale_ips = [ip for ip, dq in _web_rate_limit_store.items()
                         if not dq or dq[-1] < window_start]
            for ip in stale_ips:
                del _web_rate_limit_store[ip]

        if len(timestamps) >= _WEB_RATE_LIMIT_REQUESTS:
            logger.warning(
                f"Web rate limit exceeded: {client_ip} made {len(timestamps)} "
                f"requests in {_WEB_RATE_LIMIT_WINDOW}s window (path: {path})"
            )
            retry_after = int(_WEB_RATE_LIMIT_WINDOW - (now - timestamps[0])) + 1
            response = jsonify({
                'error': 'Too many requests',
                'message': f'Rate limit of {_WEB_RATE_LIMIT_REQUESTS} requests per '
                           f'{_WEB_RATE_LIMIT_WINDOW} seconds exceeded.',
                'retry_after': retry_after,
                'code': 'RATE_LIMIT_EXCEEDED',
            })
            response.status_code = 429
            response.headers['Retry-After'] = str(retry_after)
            response.headers['X-RateLimit-Limit'] = str(_WEB_RATE_LIMIT_REQUESTS)
            response.headers['X-RateLimit-Remaining'] = '0'
            response.headers['X-RateLimit-Reset'] = str(int(timestamps[0] + _WEB_RATE_LIMIT_WINDOW))
            return response

        timestamps.append(now)
        return None


def _check_admin_auth():
    """Check for admin API key authentication."""
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return False
    try:
        from api.v1.auth import api_key_manager, SubscriptionTier
        user = api_key_manager.get_user_by_api_key(api_key)
        return user and user.is_active and not user.is_expired and user.subscription_tier == SubscriptionTier.ELITE
    except Exception:
        return False


# =============================================================================
# PUBLIC ROUTES
# =============================================================================

@app.route('/')
def landing():
    """Landing page for signal service"""
    total_signals = None
    total_trades = None
    try:
        from data.database.models import Trade, Signal
        from data.database.db_manager import get_db_manager
        db = get_db_manager()
        with db.session() as session:
            total_trades_val = session.query(Trade).filter(Trade.status == 'CLOSED').count()
            total_trades = f"{total_trades_val:,}" if total_trades_val else None
            total_signals_val = session.query(Signal).count()
            total_signals = f"{total_signals_val:,}" if total_signals_val else None
    except Exception:
        pass
    return render_template('landing.html',
                         title='RDT Trading Signals',
                         total_signals=total_signals,
                         total_trades=total_trades)


@app.route('/pricing')
def pricing():
    """Pricing page with subscription tiers"""
    plans = {
        'basic': {
            'name': 'Basic',
            'price': 49,
            'features': [
                'Daily email alerts',
                '30-day signal history',
                'Custom price alerts',
                'Email support',
                '~5-10 signals per week'
            ]
        },
        'pro': {
            'name': 'Pro',
            'price': 149,
            'popular': True,
            'features': [
                'Real-time signal alerts',
                '1-year signal history',
                'Full REST API access',
                'Custom backtesting',
                'WebSocket streaming',
                'Priority support',
                'Leveraged ETF signals'
            ]
        },
        'elite': {
            'name': 'Elite',
            'price': 499,
            'features': [
                'Everything in Pro',
                'Unlimited history',
                'Strategy consulting (2hr/mo)',
                '1-on-1 support calls',
                'Custom integrations',
                'White-label options',
                'Options signals (coming soon)'
            ]
        }
    }
    return render_template('pricing.html', plans=plans)


@app.route('/features')
def features():
    """Features page explaining the RRS methodology"""
    return render_template('features.html')


@app.route('/performance')
def performance():
    """Historical performance page"""
    # Would pull from actual backtest/live results
    stats = {
        'total_trades': 215,
        'win_rate': 38.1,
        'profit_factor': 1.29,
        'avg_holding_days': 4.4,
        'annual_return': 6.8,
        'max_drawdown': 2.4
    }
    return render_template('performance.html', stats=stats)


# =============================================================================
# PWA ROUTES
# =============================================================================

@app.route('/sw.js')
def service_worker():
    """Serve service worker from root for proper scope"""
    return send_from_directory(
        app.static_folder,
        'sw.js',
        mimetype='application/javascript'
    )


@app.route('/offline')
def offline():
    """Offline fallback page"""
    return render_template('offline.html')


# =============================================================================
# AUTHENTICATED ROUTES (Dashboard) - Protected with @login_required
# =============================================================================

@app.route('/dashboard')
@login_required
def dashboard():
    """Main trading dashboard"""
    from web.routes.dashboard_data import (
        get_open_stock_positions, get_open_options_positions,
        get_trade_stats, get_market_status, get_recent_signals,
    )
    return render_template('dashboard.html', user=current_user,
        positions=get_open_stock_positions(),
        options_positions=get_open_options_positions(),
        stats=get_trade_stats(),
        market_status=get_market_status(),
        recent_signals=get_recent_signals(limit=10))


@app.route('/dashboard/signals')
@login_required
def dashboard_signals():
    """Current active signals view"""
    from web.routes.dashboard_data import get_recent_signals
    return render_template('dashboard_signals.html', user=current_user,
        signals=get_recent_signals(limit=100))


@app.route('/dashboard/history')
@login_required
def dashboard_history():
    """Signal history view"""
    from web.routes.dashboard_data import get_closed_trades, get_trade_stats, get_trade_stats_by_strategy
    return render_template('dashboard_history.html', user=current_user,
        trades=get_closed_trades(days=30),
        trade_stats=get_trade_stats(),
        strategy_stats=get_trade_stats_by_strategy())


@app.route('/dashboard/settings')
@login_required
def dashboard_settings():
    """User settings page"""
    return render_template('dashboard_settings.html', user=current_user)


@app.route('/dashboard/scanner')
@login_required
def dashboard_scanner():
    """Live RRS scanner view"""
    return render_template('dashboard_scanner.html', user=current_user)


@app.route('/dashboard/backtest')
@login_required
def dashboard_backtest():
    """Strategy backtesting view"""
    return render_template('dashboard_backtest.html', user=current_user)


@app.route('/dashboard/alerts')
@login_required
def dashboard_alerts():
    """Alert management view"""
    return render_template('dashboard_alerts.html', user=current_user)


@app.route('/dashboard/positions')
@login_required
def dashboard_positions():
    """Open positions view - full-featured position tracker"""
    from web.routes.dashboard_data import (
        get_open_stock_positions, get_open_options_positions,
    )
    return render_template('dashboard_positions.html', user=current_user,
        positions=get_open_stock_positions(),
        options_positions=get_open_options_positions())


@app.route('/dashboard/ml')
@login_required
def dashboard_ml():
    """ML model monitoring dashboard"""
    ml_data = {
        'model_status': None,
        'drift_status': None,
        'performance_status': None,
        'recent_predictions': [],
        'feature_drift': [],
        'error': None,
    }

    try:
        from ml.model_monitor import get_monitor_registry
        registry = get_monitor_registry()
        all_monitors = registry.get_all_monitors()

        if all_monitors:
            # Use the first registered monitor (typically the ensemble monitor)
            monitor_name = next(iter(all_monitors))
            monitor = all_monitors[monitor_name]

            # Model status
            current_status = monitor.get_current_status()
            ml_data['model_status'] = current_status

            # Drift report
            drift_report = monitor.get_drift_report()
            if drift_report:
                ml_data['drift_status'] = drift_report.to_dict()
                ml_data['feature_drift'] = monitor.get_feature_drift_status()

            # Performance metrics
            ml_data['performance_status'] = monitor.get_performance_status()

            # Recent predictions from the monitoring store
            try:
                raw_preds = monitor.monitoring_store.get_recent_predictions()
                recent = []
                for p in raw_preds[-20:]:
                    recent.append({
                        'timestamp': p.timestamp.isoformat(),
                        'predicted_class': p.predicted_class,
                        'confidence': round(float(p.prediction) * 100, 1),
                        'model_version': p.model_version,
                        'metadata': p.metadata or {},
                    })
                ml_data['recent_predictions'] = list(reversed(recent))
            except Exception as pred_err:
                logger.debug(f"Could not retrieve recent predictions: {pred_err}")
        else:
            ml_data['error'] = 'no_monitors'
    except Exception as e:
        logger.warning(f"ML dashboard data collection failed: {e}")
        ml_data['error'] = 'ml_data_unavailable'

    # Also try to get ensemble model info directly
    try:
        from ml.ensemble import StackedEnsemble
        # Check if there is a saved ensemble on disk
        import os
        ensemble_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'models', 'ensemble'
        )
        ensemble_meta_path = os.path.join(ensemble_path, 'ensemble_meta.pkl')
        ml_data['ensemble_path_exists'] = os.path.isfile(ensemble_meta_path)
        ml_data['ensemble_path'] = ensemble_path
    except Exception:
        ml_data['ensemble_path_exists'] = False
        ml_data['ensemble_path'] = None

    return render_template('dashboard_ml.html', user=current_user, ml_data=ml_data)


@app.route('/dashboard/options')
@login_required
def dashboard_options():
    """Options trading dashboard"""
    from web.routes.dashboard_data import get_open_options_positions
    return render_template('dashboard_options.html', user=current_user,
        options_positions=get_open_options_positions())


@app.route('/dashboard/strategies')
@login_required
def dashboard_strategies():
    """Strategy performance & control page"""
    return render_template('dashboard_strategies.html', user=current_user)


@app.route('/dashboard/agents')
@login_required
def dashboard_agents():
    """Agent health monitor page"""
    return render_template('dashboard_agents.html', user=current_user)


@app.route('/dashboard/confidence')
@login_required
def dashboard_confidence():
    """AI Signal Confidence analysis page"""
    return render_template('dashboard_confidence.html', user=current_user)


@app.route('/dashboard/journal')
@login_required
def dashboard_journal():
    """Trading journal and analytics page"""
    return render_template('dashboard_journal.html', user=current_user)


@app.route('/dashboard/settings/security')
@login_required
def dashboard_settings_security():
    """Security settings page with session management"""
    return render_template('settings_security.html', user=current_user)


@app.route('/positions')
@login_required
def positions_tracker():
    """Full-featured position tracker view (alias for dashboard/positions)"""
    from web.routes.dashboard_data import (
        get_open_stock_positions, get_open_options_positions,
    )
    return render_template('positions.html', user=current_user,
        positions=get_open_stock_positions(),
        options_positions=get_open_options_positions())


# =============================================================================
# CHECKOUT & BILLING
# =============================================================================

@app.route('/checkout/<plan>')
@login_required
def checkout(plan):
    """Initiate checkout for a subscription plan"""
    valid_plans = ['basic', 'pro', 'elite']
    if plan not in valid_plans:
        return redirect(url_for('pricing'))

    stripe_secret = os.environ.get('STRIPE_SECRET_KEY')
    if not stripe_secret:
        logger.warning("STRIPE_SECRET_KEY not configured - checkout unavailable")
        return render_template('checkout.html', plan=plan, stripe_unavailable=True)

    try:
        from payments.plans import get_price_id, get_trial_days
        from payments.stripe_client import StripeClient, StripeError

        # Map plan name to Stripe price ID
        price_id = get_price_id(plan)
        if not price_id:
            logger.error(f"No Stripe price ID configured for plan: {plan}")
            return render_template('checkout.html', plan=plan, stripe_unavailable=True)

        # Build success/cancel URLs
        success_url = url_for('billing_success', _external=True) + '?session_id={CHECKOUT_SESSION_ID}'
        cancel_url = url_for('billing_cancel', _external=True)

        # Get or create Stripe customer for the logged-in user
        client = StripeClient(secret_key=stripe_secret)

        customer_id = getattr(current_user, 'stripe_customer_id', None)
        customer_email = getattr(current_user, 'email', None)

        # Create the checkout session
        trial_days = get_trial_days(plan)
        session = client.create_checkout_session(
            price_id=price_id,
            success_url=success_url,
            cancel_url=cancel_url,
            customer_id=customer_id if customer_id else None,
            customer_email=customer_email if not customer_id else None,
            trial_days=trial_days if trial_days > 0 else None,
            metadata={
                'user_id': str(current_user.id),
                'plan': plan,
            }
        )

        logger.info(f"Created Stripe checkout session {session.id} for user {current_user.id}, plan={plan}")
        return redirect(session.url)

    except Exception as e:
        logger.error(f"Failed to create Stripe checkout session for plan={plan}: {e}")
        return render_template('checkout.html', plan=plan, stripe_error='Payment processing error. Please try again.')


@app.route('/billing/success')
def billing_success():
    """Post-checkout success page"""
    return render_template('billing_success.html')


@app.route('/billing/cancel')
def billing_cancel():
    """Post-checkout cancel page"""
    return redirect(url_for('pricing'))


# =============================================================================
# WEBHOOKS
# =============================================================================

@app.route('/webhooks/stripe', methods=['POST'])
@csrf.exempt
def stripe_webhook():
    """Handle Stripe webhook events"""
    webhook_secret = os.environ.get('STRIPE_WEBHOOK_SECRET')
    if not webhook_secret:
        logger.error("STRIPE_WEBHOOK_SECRET not configured")
        return jsonify({'error': 'Webhook not configured'}), 500

    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')

    if not sig_header:
        return jsonify({'error': 'Missing signature'}), 400

    # Verify webhook signature and construct event
    try:
        import stripe
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except ValueError:
        logger.warning("Stripe webhook received invalid payload")
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError as e:
        logger.warning(f"Stripe webhook signature verification failed: {e}")
        return jsonify({'error': 'Invalid signature'}), 400
    except Exception as e:
        logger.warning(f"Stripe webhook error during signature verification: {e}")
        return jsonify({'error': 'Invalid signature'}), 400

    logger.info(f"Stripe webhook received: {event['type']} (id: {event['id']})")

    # Process the event using WebhookHandler with a database session
    try:
        from web.auth import get_db_session
        from payments.webhooks import WebhookHandler

        db_session = get_db_session()
        handler = WebhookHandler(db_session=db_session)
        result = handler.handle_event(event)

        logger.info(f"Stripe webhook processed: {event['type']} -> {result.get('status', 'unknown')}")
        return jsonify({'status': 'received', 'result': result.get('status')}), 200

    except Exception as e:
        logger.error(f"Error processing Stripe webhook event {event['type']}: {e}")
        # Return 200 to prevent Stripe from retrying — the event has been
        # received; any DB issues should be resolved separately.
        return jsonify({'status': 'received', 'error': 'processing_error'}), 200


# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@app.route('/health')
@app.route('/health/live')
def health_live():
    """
    Kubernetes liveness probe.

    Returns 200 if the application process is alive.
    Does not check dependencies - only that the app can respond.
    """
    response = {
        'status': 'alive',
        'timestamp': format_timestamp()
    }
    if TRACING_AVAILABLE:
        trace_id = get_current_trace_id()
        if trace_id:
            response['trace_id'] = trace_id
    return jsonify(response), 200


@app.route('/health/ready')
def health_ready():
    """
    Kubernetes readiness probe.

    Returns 200 if the application is ready to serve traffic.
    Checks database connectivity and broker status.
    """
    checks = {
        'database': False,
        'broker': False
    }
    all_ready = True

    # Check database
    try:
        from data.database.connection import get_db_manager
        db = get_db_manager()
        if db.check_connection():
            checks['database'] = True
        else:
            all_ready = False
    except Exception as e:
        logger.warning(f"Health check - database failed: {e}")
        all_ready = False

    # Check broker (optional - not all environments have broker)
    try:
        from brokers.failover_manager import get_failover_manager
        manager = get_failover_manager()
        if manager.active_broker and manager.active_broker.is_connected:
            checks['broker'] = True
        else:
            # Broker not connected is not critical for readiness
            checks['broker'] = None  # Unknown/not configured
    except Exception:
        checks['broker'] = None  # Not configured

    status_code = 200 if all_ready else 503
    response = {
        'status': 'ready' if all_ready else 'not_ready',
        'checks': checks,
        'timestamp': format_timestamp()
    }

    if TRACING_AVAILABLE:
        trace_id = get_current_trace_id()
        if trace_id:
            response['trace_id'] = trace_id

    return jsonify(response), status_code


@app.route('/health/detailed')
def health_detailed():
    """
    Detailed health check for diagnostics.

    Returns comprehensive health information including:
    - Database status and pool stats
    - Broker connection status
    - Memory usage
    - Event bus status
    - Recent errors
    """
    if not _check_admin_auth():
        return jsonify({'error': 'Admin authentication required'}), 401

    import psutil
    import os

    health = {
        'status': 'healthy',
        'timestamp': format_timestamp(),
        'components': {}
    }
    issues = []

    # Database health
    try:
        from data.database.connection import get_db_manager
        db = get_db_manager()
        db_connected = db.check_connection()

        # Get pool stats if available
        pool_stats = {}
        try:
            pool = db.engine.pool
            pool_stats = {
                'size': pool.size(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
                'checked_in': pool.checkedin()
            }
        except Exception:
            pass

        health['components']['database'] = {
            'status': 'healthy' if db_connected else 'unhealthy',
            'connected': db_connected,
            'pool': pool_stats
        }
        if not db_connected:
            issues.append('Database connection failed')
    except Exception as e:
        logger.error(f"Health check database error: {e}")
        health['components']['database'] = {
            'status': 'unhealthy',
            'error': 'connection_failed'
        }
        issues.append('Database error')

    # Broker health
    try:
        from brokers.failover_manager import get_failover_manager
        manager = get_failover_manager()
        broker_status = manager.get_health_status()

        health['components']['broker'] = {
            'status': 'healthy' if manager.active_broker else 'degraded',
            'active_role': manager.active_role.value if manager.active_role else None,
            'details': broker_status
        }
        if not manager.active_broker:
            issues.append('No active broker')
    except Exception as e:
        logger.error(f"Health check broker error: {e}")
        health['components']['broker'] = {
            'status': 'unknown',
            'error': 'check_failed'
        }

    # Event bus health
    try:
        from agents.events import get_event_bus, PersistentEventBus
        event_bus = get_event_bus()
        event_info = {
            'status': 'running' if event_bus._running else 'stopped',
            'subscribers': len(event_bus._subscribers),
            'history_size': len(event_bus._event_history)
        }

        if isinstance(event_bus, PersistentEventBus):
            event_info['persistent'] = True
            event_info['unprocessed_count'] = event_bus.get_unprocessed_count()
            if event_info['unprocessed_count'] > 50:
                issues.append(f"Event backlog: {event_info['unprocessed_count']} unprocessed")

        health['components']['event_bus'] = event_info
    except Exception as e:
        logger.error(f"Health check event bus error: {e}")
        health['components']['event_bus'] = {
            'status': 'unknown',
            'error': 'check_failed'
        }

    # System resources
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        health['components']['system'] = {
            'memory_mb': round(memory_info.rss / 1024 / 1024, 2),
            'memory_percent': round(process.memory_percent(), 2),
            'cpu_percent': round(process.cpu_percent(), 2),
            'threads': process.num_threads(),
            'open_files': len(process.open_files())
        }

        if memory_info.rss > 1024 * 1024 * 1024:  # > 1GB
            issues.append('High memory usage')
    except Exception as e:
        logger.error(f"Health check system resources error: {e}")
        health['components']['system'] = {
            'status': 'unknown',
            'error': 'check_failed'
        }

    # Determine overall status
    if issues:
        health['status'] = 'degraded'
        health['issues'] = issues

    if TRACING_AVAILABLE:
        trace_id = get_current_trace_id()
        if trace_id:
            health['trace_id'] = trace_id

    return jsonify(health), 200


@app.route('/api/status')
def api_status():
    """API status endpoint"""
    return jsonify({
        'api_version': '1.0.0',
        'status': 'operational',
        'scanner_active': True,
        'last_scan': format_timestamp()
    })


# =============================================================================
# SESSION MANAGEMENT API
# =============================================================================

@app.route('/api/sessions')
@login_required
def get_sessions():
    """Get all active sessions for the current user"""
    try:
        from web.session_manager import get_session_manager
        from web.auth import get_session_token

        session_manager = get_session_manager()
        current_token = get_session_token()

        sessions = session_manager.get_user_sessions(
            user_id=current_user.id,
            current_token=current_token
        )

        return jsonify({
            'sessions': [
                {
                    'id': s.id,
                    'ip_masked': s.ip_masked,
                    'user_agent': s.user_agent,
                    'device_info': s.device_info,
                    'created_at': s.created_at.isoformat() if s.created_at else None,
                    'last_activity': s.last_activity.isoformat() if s.last_activity else None,
                    'is_current': s.is_current,
                    'is_active': s.is_active
                }
                for s in sessions
            ],
            'total': len(sessions)
        })
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        return jsonify({'error': 'Failed to retrieve sessions'}), 500


@app.route('/api/sessions/revoke/<int:session_id>', methods=['POST'])
@login_required
def revoke_session(session_id):
    """Revoke a specific session"""
    try:
        from web.session_manager import get_session_manager
        from web.auth import get_session_token

        session_manager = get_session_manager()
        current_token = get_session_token()

        # Get current session ID to prevent revoking current session
        sessions = session_manager.get_user_sessions(
            user_id=current_user.id,
            current_token=current_token
        )

        # Check if trying to revoke current session
        for s in sessions:
            if s.id == session_id and s.is_current:
                return jsonify({'error': 'Cannot revoke current session'}), 400

        # Revoke the session
        success = session_manager.revoke_session(session_id, current_user.id)

        if success:
            logger.info(f"Session {session_id} revoked by user {current_user.id}")
            return jsonify({'success': True, 'message': 'Session revoked'})
        else:
            return jsonify({'error': 'Session not found or already revoked'}), 404

    except Exception as e:
        logger.error(f"Error revoking session: {e}")
        return jsonify({'error': 'Failed to revoke session'}), 500


@app.route('/api/sessions/revoke-all', methods=['POST'])
@login_required
def revoke_all_sessions():
    """Revoke all sessions except the current one"""
    try:
        from web.session_manager import get_session_manager
        from web.auth import get_session_token

        session_manager = get_session_manager()
        current_token = get_session_token()

        count = session_manager.revoke_all_except_current(
            user_id=current_user.id,
            current_token=current_token
        )

        logger.info(f"User {current_user.id} revoked {count} sessions")
        return jsonify({
            'success': True,
            'revoked_count': count,
            'message': f'{count} session(s) revoked'
        })

    except Exception as e:
        logger.error(f"Error revoking all sessions: {e}")
        return jsonify({'error': 'Failed to revoke sessions'}), 500


@app.route('/api/change-password', methods=['POST'])
@login_required
def api_change_password():
    """Change the current user's password"""
    try:
        from werkzeug.security import check_password_hash
        from web.auth import change_password, get_db_session
        from data.database.models import User

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        current_password = data.get('current_password')
        new_password = data.get('new_password')

        if not current_password or not new_password:
            return jsonify({'error': 'Current password and new password are required'}), 400

        # Validate new password strength
        from web.auth import validate_password
        is_valid, error_msg = validate_password(new_password)
        if not is_valid:
            return jsonify({'error': error_msg}), 400

        # Verify current password
        session = get_db_session()
        user = session.query(User).filter_by(id=current_user.id).first()

        if not user or not check_password_hash(user.password_hash, current_password):
            return jsonify({'error': 'Current password is incorrect'}), 400

        # Change password
        change_password(current_user.id, new_password)

        logger.info(f"Password changed for user {current_user.id}")
        return jsonify({'success': True, 'message': 'Password changed successfully'})

    except Exception as e:
        logger.error(f"Error changing password: {e}")
        return jsonify({'error': 'Failed to change password'}), 500


# =============================================================================
# PUSH NOTIFICATION API
# =============================================================================

@app.route('/api/push/vapid-key')
def get_vapid_key():
    """Get the VAPID public key for push notifications"""
    try:
        from web.push_notifications import get_push_service
        push_service = get_push_service()

        if not push_service.vapid.public_key:
            return jsonify({
                'error': 'Push notifications not configured',
                'publicKey': None
            }), 503

        return jsonify({
            'publicKey': push_service.vapid.public_key
        })
    except Exception as e:
        logger.error(f"Error getting VAPID key: {e}")
        return jsonify({'error': 'Failed to get VAPID key'}), 500


@app.route('/api/push/subscribe', methods=['POST'])
@login_required
def push_subscribe():
    """Subscribe to push notifications"""
    try:
        from web.push_notifications import get_push_service

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No subscription data provided'}), 400

        endpoint = data.get('endpoint')
        keys = data.get('keys', {})

        if not endpoint or not keys.get('p256dh') or not keys.get('auth'):
            return jsonify({'error': 'Invalid subscription data'}), 400

        push_service = get_push_service()
        success = push_service.subscribe(
            endpoint=endpoint,
            keys=keys,
            user_id=current_user.id
        )

        if success:
            logger.info(f"User {current_user.id} subscribed to push notifications")
            return jsonify({'success': True, 'message': 'Subscribed successfully'})
        else:
            return jsonify({'error': 'Failed to subscribe'}), 500

    except Exception as e:
        logger.error(f"Error subscribing to push: {e}")
        return jsonify({'error': 'Failed to subscribe'}), 500


@app.route('/api/push/unsubscribe', methods=['POST'])
@login_required
def push_unsubscribe():
    """Unsubscribe from push notifications"""
    try:
        from web.push_notifications import get_push_service

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        endpoint = data.get('endpoint')
        if not endpoint:
            return jsonify({'error': 'Endpoint is required'}), 400

        push_service = get_push_service()
        success = push_service.unsubscribe(endpoint)

        if success:
            logger.info(f"User {current_user.id} unsubscribed from push notifications")
            return jsonify({'success': True, 'message': 'Unsubscribed successfully'})
        else:
            return jsonify({'error': 'Subscription not found'}), 404

    except Exception as e:
        logger.error(f"Error unsubscribing from push: {e}")
        return jsonify({'error': 'Failed to unsubscribe'}), 500


@app.route('/api/push/test', methods=['POST'])
@login_required
def push_test():
    """Send a test push notification to the current user"""
    try:
        from web.push_notifications import get_push_service, create_system_notification

        push_service = get_push_service()

        if not push_service.is_available():
            return jsonify({
                'error': 'Push notifications not available',
                'message': 'VAPID keys may not be configured'
            }), 503

        notification = create_system_notification(
            title='Test Notification',
            body='Push notifications are working correctly!',
            url='/dashboard'
        )

        count = push_service.send_to_user(current_user.id, notification)

        if count > 0:
            return jsonify({
                'success': True,
                'message': f'Notification sent to {count} device(s)'
            })
        else:
            return jsonify({
                'error': 'No subscriptions found',
                'message': 'Enable notifications first'
            }), 404

    except Exception as e:
        logger.error(f"Error sending test notification: {e}")
        return jsonify({'error': 'Failed to send notification'}), 500


@app.route('/api/websocket/status')
def websocket_status():
    """WebSocket status and statistics endpoint"""
    if not _check_admin_auth():
        return jsonify({'error': 'Admin authentication required'}), 401

    if socketio is None:
        return jsonify({
            'status': 'unavailable',
            'message': 'WebSocket support not initialized'
        }), 503

    try:
        stats = get_room_stats()
        return jsonify({
            'status': 'operational',
            'connected_clients': stats.get('total_connected', 0),
            'room_stats': {
                'signals': stats.get('signals', 0),
                'positions': stats.get('positions', 0),
                'scanner': stats.get('scanner', 0),
                'alerts': stats.get('alerts', 0),
                'prices': stats.get('prices', 0)
            },
            'timestamp': format_timestamp()
        })
    except Exception as e:
        logger.error(f"Error getting WebSocket status: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

@app.route('/metrics')
def prometheus_metrics():
    """
    Prometheus metrics endpoint for scraping.

    Returns Prometheus-formatted metrics for:
    - Trading signals and executions
    - Portfolio and positions
    - Scanner performance
    - API request tracking
    - ML model predictions
    - Alert and broker operations
    - WebSocket connections
    """
    if not _check_admin_auth():
        return jsonify({'error': 'Admin authentication required'}), 401

    if not METRICS_AVAILABLE:
        return jsonify({
            'error': 'Prometheus metrics not available',
            'message': 'prometheus-client package may not be installed'
        }), 503

    try:
        # Update WebSocket metrics before returning
        if socketio is not None:
            try:
                stats = get_room_stats()
                update_websocket_metrics(
                    total_connections=stats.get('total_connected', 0),
                    room_counts={
                        'signals': stats.get('signals', 0),
                        'positions': stats.get('positions', 0),
                        'scanner': stats.get('scanner', 0),
                        'alerts': stats.get('alerts', 0),
                        'prices': stats.get('prices', 0)
                    }
                )
            except Exception as e:
                logger.debug(f"Error updating WebSocket metrics: {e}")

        # Generate and return Prometheus metrics
        metrics_output = get_metrics()
        return Response(
            metrics_output,
            mimetype=get_metrics_content_type()
        )
    except Exception as e:
        logger.error(f"Error generating Prometheus metrics: {e}")
        return jsonify({
            'error': 'Failed to generate metrics',
            'message': 'Internal server error'
        }), 500


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(CSRFError)
def handle_csrf_error(error):
    """Handle CSRF validation errors"""
    logger.warning(f"CSRF error: {error.description}")
    # Check if this is an API request
    if request.path.startswith('/api/'):
        return jsonify({
            'error': 'CSRF token missing or invalid',
            'code': 'CSRF_ERROR',
            'message': 'CSRF token missing or invalid'
        }), 400
    # For web requests, show error page or redirect to login
    return render_template('error.html',
                         error_title='Session Expired',
                         error_message='Your session has expired or the form token is invalid. Please try again.',
                         show_login=True), 400


@app.errorhandler(400)
def bad_request(error):
    """Handle 400 Bad Request errors"""
    logger.warning(f"Bad request: {error}")
    if request.path.startswith('/api/'):
        return jsonify({
            'error': 'Bad request',
            'message': 'Invalid request',
            'code': 'BAD_REQUEST'
        }), 400
    return render_template('error.html',
                         error_title='Bad Request',
                         error_message='The request could not be understood by the server.'), 400


@app.errorhandler(413)
def request_too_large(error):
    """Handle 413 Request Entity Too Large errors"""
    logger.warning(f"Request too large: {error}")
    if request.path.startswith('/api/'):
        return jsonify({
            'error': 'Request too large',
            'message': 'The request body exceeds the maximum allowed size.',
            'code': 'REQUEST_TOO_LARGE'
        }), 413
    return render_template('error.html',
                         error_title='Request Too Large',
                         error_message='The uploaded data exceeds the maximum allowed size.'), 413


@app.errorhandler(415)
def unsupported_media_type(error):
    """Handle 415 Unsupported Media Type errors"""
    logger.warning(f"Unsupported media type: {error}")
    return jsonify({
        'error': 'Unsupported media type',
        'message': 'The Content-Type of the request is not supported.',
        'code': 'UNSUPPORTED_MEDIA_TYPE'
    }), 415


@app.errorhandler(422)
def unprocessable_entity(error):
    """Handle 422 Unprocessable Entity errors (validation errors)"""
    logger.warning(f"Validation error: {error}")
    if request.path.startswith('/api/'):
        return jsonify({
            'error': 'Validation error',
            'message': 'The submitted data failed validation',
            'code': 'VALIDATION_ERROR'
        }), 422
    return render_template('error.html',
                         error_title='Validation Error',
                         error_message='The submitted data failed validation.'), 422


@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found errors"""
    if request.path.startswith('/api/'):
        return jsonify({
            'error': 'Not found',
            'message': 'The requested resource was not found.',
            'code': 'NOT_FOUND'
        }), 404
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 Internal Server errors"""
    logger.error(f"Server error: {error}")
    if request.path.startswith('/api/'):
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred.',
            'code': 'INTERNAL_ERROR'
        }), 500
    return render_template('500.html'), 500


# =============================================================================
# INITIALIZE TRADING COMPONENTS
# =============================================================================

def initialize_trading_components():
    """Initialize all trading components using the dedicated module."""
    if os.environ.get('RDT_SKIP_TRADING_INIT', '').lower() in ('true', '1', 'yes'):
        logger.info("Dashboard-only mode: skipping trading component initialization")
        return {}
    try:
        from web.trading_init import initialize_all_components
        components = initialize_all_components()
        return components
    except Exception as e:
        logger.error(f"Failed to initialize trading components: {e}")
        return {}


# Initialize on module load — but skip in werkzeug reloader parent process.
# With debug=True, werkzeug runs this module twice: once as the parent (which just
# watches for file changes) and once as the child (WERKZEUG_RUN_MAIN='true') which
# actually serves requests. Only the child should connect to IBKR to avoid
# duplicate client_id conflicts.
_is_reloader_parent_init = (
    __name__ == '__main__'
    and os.environ.get('WERKZEUG_RUN_MAIN') != 'true'
)
if _is_reloader_parent_init:
    logger.info("Werkzeug reloader parent: deferring trading init to child process")
    _trading_components = {}
else:
    _trading_components = initialize_trading_components()

# Start the real-time scanner in a background thread.
# When werkzeug reloader is active (debug=True), both parent and child load this module.
# Only start the scanner in the reloader child (WERKZEUG_RUN_MAIN='true') or when
# imported by a WSGI server (not running as __main__).
_scanner = _trading_components.get('realtime_scanner')
_scanner_thread = None
if _scanner is not None:
    _is_reloader_parent = (
        __name__ == '__main__'
        and os.environ.get('WERKZEUG_RUN_MAIN') != 'true'
    )
    if not _is_reloader_parent:
        _scanner_thread = threading.Thread(
            target=_scanner.run_continuous, daemon=True
        )
        _scanner_thread.start()
        logger.info("Real-time scanner started in background thread")
    else:
        logger.info("Deferring scanner start to werkzeug child process")


# Start the agent orchestrator (ScannerAgent → AnalyzerAgent → ExecutorAgent pipeline)
# in a background thread with its own asyncio event loop.
def _start_agent_system():
    """Run the trading agent orchestrator in a background asyncio event loop."""
    import asyncio
    import nest_asyncio
    from agents.orchestrator import run_trading_system
    from web.trading_init import load_configuration, get_watchlist

    broker = _trading_components.get('broker')
    risk_manager = _trading_components.get('risk_manager')
    data_provider = _trading_components.get('data_provider')

    if not all([broker, risk_manager, data_provider]):
        logger.warning("Cannot start agent system: missing core components")
        return

    config = load_configuration()
    watchlist = get_watchlist()
    auto_trade = config.get('auto_trade', False)

    # Apply nest_asyncio so ib_insync can run its event loop inside ours
    nest_asyncio.apply()

    logger.info(f"Starting agent system (auto_trade={auto_trade}, {len(watchlist)} symbols)")
    try:
        asyncio.run(run_trading_system(
            broker=broker,
            risk_manager=risk_manager,
            data_provider=data_provider,
            watchlist=watchlist,
            config=config,
            auto_trade=auto_trade
        ))
    except Exception as e:
        logger.error(f"Agent system exited: {e}")


_agent_thread = None
if _trading_components.get('broker') is not None:
    _is_reloader_parent = (
        __name__ == '__main__'
        and os.environ.get('WERKZEUG_RUN_MAIN') != 'true'
    )
    if not _is_reloader_parent:
        _agent_thread = threading.Thread(target=_start_agent_system, daemon=True)
        _agent_thread.start()
        logger.info("Agent system (orchestrator) started in background thread")
    else:
        logger.info("Deferring agent system start to werkzeug child process")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting RDT Signal Service web app on port {port}")

    # Default to debug=True for local development (set FLASK_DEBUG=false for production)
    debug_mode = os.environ.get('FLASK_DEBUG', 'true').lower() in ('true', '1', 'yes')

    if socketio is not None:
        # Use SocketIO run for WebSocket support
        logger.info("Starting with WebSocket support enabled")
        socketio.run(app, host='0.0.0.0', port=port, debug=debug_mode, allow_unsafe_werkzeug=True)
    else:
        # Fallback to standard Flask run
        logger.warning("Starting without WebSocket support")
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
