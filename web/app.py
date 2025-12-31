"""
RDT Trading Signal Service - Flask Web Application

Provides:
- Landing page for customer acquisition
- Dashboard for signal viewing
- API integration for programmatic access
- Subscription management via Stripe
"""

import os
from datetime import datetime
from flask import Flask, render_template, jsonify, redirect, url_for, request
from flask_cors import CORS
from loguru import logger

# Create Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Import and register API blueprint
try:
    from api.v1.routes import api_bp
    app.register_blueprint(api_bp)
    logger.info("API v1 blueprint registered")
except ImportError as e:
    logger.warning(f"Could not import API blueprint: {e}")


# =============================================================================
# PUBLIC ROUTES
# =============================================================================

@app.route('/')
def landing():
    """Landing page for signal service"""
    return render_template('landing.html',
                         title='RDT Trading Signals',
                         tagline='Professional-grade trading signals powered by Real Relative Strength')


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
# AUTHENTICATED ROUTES (Dashboard)
# =============================================================================

@app.route('/dashboard')
def dashboard():
    """Main trading dashboard"""
    # In production, would check authentication
    return render_template('dashboard.html')


@app.route('/dashboard/signals')
def dashboard_signals():
    """Current active signals view"""
    return render_template('dashboard_signals.html')


@app.route('/dashboard/history')
def dashboard_history():
    """Signal history view"""
    return render_template('dashboard_history.html')


@app.route('/dashboard/settings')
def dashboard_settings():
    """User settings page"""
    return render_template('dashboard_settings.html')


# =============================================================================
# CHECKOUT & BILLING
# =============================================================================

@app.route('/checkout/<plan>')
def checkout(plan):
    """Initiate checkout for a subscription plan"""
    valid_plans = ['basic', 'pro', 'elite']
    if plan not in valid_plans:
        return redirect(url_for('pricing'))

    # In production, would create Stripe checkout session
    return render_template('checkout.html', plan=plan)


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
def stripe_webhook():
    """Handle Stripe webhook events"""
    # In production, verify webhook signature and process events
    return jsonify({'status': 'received'}), 200


# =============================================================================
# API STATUS (Simple health endpoints)
# =============================================================================

@app.route('/health')
def health():
    """Simple health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/status')
def api_status():
    """API status endpoint"""
    return jsonify({
        'api_version': '1.0.0',
        'status': 'operational',
        'scanner_active': True,
        'last_scan': datetime.now().isoformat()
    })


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {error}")
    return render_template('500.html'), 500


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Development server
    logger.info("Starting RDT Signal Service web app")
    app.run(host='0.0.0.0', port=8080, debug=True)
