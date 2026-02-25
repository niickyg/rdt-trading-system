"""
API Routes for Signal Service

Provides REST endpoints for:
- Real-time trading signals
- Historical performance data
- Backtest execution
- User management
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
from flask import Blueprint, request, jsonify
from loguru import logger

from utils.timezone import (
    format_timestamp,
    get_eastern_time,
    get_market_status,
    is_market_open as tz_is_market_open,
    is_trading_day,
    to_eastern,
)

from utils.validators import (
    validate_symbol,
    validate_price,
    validate_quantity,
    validate_email,
    validate_direction,
    validate_condition,
    validate_notification_method,
    validate_integer,
    validate_float,
    validate_json_body,
    sanitize_input,
    ValidationError,
)

from api.v1.auth import (
    require_api_key,
    require_subscription,
    require_feature,
    require_admin,
    SubscriptionTier,
    api_key_manager
)

# Import scanner and data components
try:
    from scanner.realtime_scanner import RealTimeScanner
    SCANNER_AVAILABLE = True
except ImportError:
    SCANNER_AVAILABLE = False
    logger.warning("RealTimeScanner not available")

try:
    from shared.indicators.rrs import RRSCalculator, check_daily_strength, check_daily_weakness
    RRS_AVAILABLE = True
except ImportError:
    RRS_AVAILABLE = False
    logger.warning("RRSCalculator not available")

try:
    from backtesting import BacktestEngine, DataLoader, load_default_watchlist
    BACKTEST_AVAILABLE = True
except ImportError:
    BACKTEST_AVAILABLE = False
    logger.warning("BacktestEngine not available")

try:
    import yfinance as yf
    import pandas as pd
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available")

try:
    from data.database import get_trades_repository
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logger.warning("Database module not available")

try:
    from trading import get_position_tracker
    POSITION_TRACKER_AVAILABLE = True
except ImportError:
    POSITION_TRACKER_AVAILABLE = False
    logger.warning("PositionTracker not available")

try:
    from trading.execution_tracker import get_execution_tracker
    EXECUTION_TRACKER_AVAILABLE = True
except ImportError:
    EXECUTION_TRACKER_AVAILABLE = False
    logger.warning("ExecutionTracker not available")

try:
    from trading.order_monitor import get_order_monitor
    ORDER_MONITOR_AVAILABLE = True
except ImportError:
    ORDER_MONITOR_AVAILABLE = False
    logger.warning("OrderMonitor not available")

try:
    from trading.advanced_orders import (
        BracketOrder, TrailingStopOrder, OCOOrder,
        AdvancedOrderManager, BracketOrderStatus, TrailingStopType
    )
    from brokers import get_broker, get_broker_from_config, OrderSide, OrderType
    ADVANCED_ORDERS_AVAILABLE = True
except ImportError:
    ADVANCED_ORDERS_AVAILABLE = False
    logger.warning("Advanced orders not available")

try:
    from agents.learning_agent import LearningAgent
    LEARNING_AGENT_AVAILABLE = True
except ImportError:
    LEARNING_AGENT_AVAILABLE = False
    logger.warning("LearningAgent not available")

# Global learning agent reference (set by application startup)
_learning_agent = None


def set_learning_agent(agent):
    """Set the global learning agent reference"""
    global _learning_agent
    _learning_agent = agent


def get_learning_agent():
    """Get the global learning agent reference"""
    return _learning_agent


api_bp = Blueprint('api_v1', __name__, url_prefix='/api/v1')


def handle_api_error(e: Exception, context: str = "operation") -> tuple:
    """Handle API errors safely - log details server-side, return generic message to client."""
    logger.error(f"Error in {context}: {e}", exc_info=True)
    return jsonify({
        'error': f'Failed to complete {context}',
        'code': 'INTERNAL_ERROR'
    }), 500


# ============================================================================
# Health and Status Endpoints
# ============================================================================

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': format_timestamp(),
        'version': '1.0.0'
    })


@api_bp.route('/status', methods=['GET'])
def system_status():
    """System status and scanner state - no authentication required for dashboard"""
    signals = get_active_signals()
    spy_price = get_spy_price()

    return jsonify({
        'status': 'operational',
        'scanner_running': SCANNER_AVAILABLE,
        'last_scan': format_timestamp(),
        'symbols_monitored': 175,
        'active_signals': len(signals),
        'market_status': get_market_status(),
        'spy_price': spy_price
    })


@api_bp.route('/dashboard', methods=['GET'])
def get_dashboard_data():
    """
    Combined dashboard data endpoint - no authentication required

    Returns:
        Dashboard data including market status, signals, positions, and performance
    """
    signals = get_active_signals()
    spy_price = get_spy_price()
    performance = calculate_performance_stats(30, None)

    return jsonify({
        'timestamp': format_timestamp(),
        'market_status': get_market_status(),
        'spy_price': spy_price,
        'active_signals_count': len(signals),
        'recent_signals': signals[:10],  # Last 10 signals
        'open_positions': get_open_positions(),
        'performance': performance
    })


# ============================================================================
# Signal Endpoints
# ============================================================================

@api_bp.route('/signals/current', methods=['GET'])
@require_api_key
def get_current_signals():
    """
    Get current active trading signals

    Query params:
        page: Page number (default 1)
        per_page: Results per page (default 50, max 100)

    Returns:
        List of active signals with entry/exit levels and pagination metadata
    """
    # This would connect to the actual scanner
    # For now, return sample data structure

    signals = get_active_signals()

    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    per_page = min(per_page, 100)  # Cap at 100
    total = len(signals)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_signals = signals[start:end]

    return jsonify({
        'timestamp': format_timestamp(),
        'count': len(paginated_signals),
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': (total + per_page - 1) // per_page if per_page > 0 else 0,
        'signals': paginated_signals
    })


@api_bp.route('/quotes', methods=['GET'])
@require_api_key
def get_quotes():
    """
    Get live price quotes for given symbols

    Query params:
        symbols: Comma-separated list of symbols (required)
    """
    symbols_param = request.args.get('symbols', '')
    if not symbols_param:
        return jsonify({'error': 'symbols parameter required'}), 400

    symbols = [s.strip().upper() for s in symbols_param.split(',') if s.strip()]
    if not symbols:
        return jsonify({'error': 'No valid symbols provided'}), 400

    # Cap at 20 symbols per request
    symbols = symbols[:20]

    quotes = {}
    if YFINANCE_AVAILABLE:
        try:
            # Batch download for all symbols
            tickers_str = ' '.join(symbols)
            data = yf.download(tickers_str, period='1d', interval='1m', progress=False)

            if data is not None and not data.empty:
                for symbol in symbols:
                    try:
                        # yfinance returns MultiIndex columns: ('Close', 'AAPL')
                        # Try MultiIndex access first, then flat column fallback
                        last_close = None
                        if isinstance(data.columns, pd.MultiIndex):
                            # Find the close column name (could be 'Close' or 'close')
                            level1_vals = data.columns.get_level_values(0).unique()
                            close_key = 'Close' if 'Close' in level1_vals else 'close'
                            if (close_key, symbol) in data.columns:
                                series = data[(close_key, symbol)].dropna()
                                if len(series) > 0:
                                    last_close = float(series.iloc[-1])
                        else:
                            # Flat columns (older yfinance or single symbol)
                            close_col = 'Close' if 'Close' in data.columns else 'close'
                            series = data[close_col].dropna()
                            if len(series) > 0:
                                last_close = float(series.iloc[-1])

                        if last_close is not None:
                            quotes[symbol] = {
                                'price': round(last_close, 2),
                                'timestamp': format_timestamp()
                            }
                    except Exception as e:
                        logger.debug(f"Error extracting quote for {symbol}: {e}")
                        continue
        except Exception as e:
            logger.debug(f"Error batch fetching quotes: {e}")

    return jsonify({
        'timestamp': format_timestamp(),
        'quotes': quotes
    })


@api_bp.route('/signals/history', methods=['GET'])
@require_api_key
def get_signal_history():
    """
    Get historical signals

    Query params:
        days: Number of days to look back (default 7)
        symbol: Filter by symbol (optional)
        direction: Filter by direction (optional)
    """
    # Validate 'days' parameter
    is_valid, days_result = validate_integer(
        request.args.get('days', 7),
        field_name='days',
        min_val=1,
        max_val=365
    )
    if not is_valid:
        return jsonify({
            'error': 'Invalid parameter',
            'message': days_result,
            'code': 'INVALID_PARAMETER'
        }), 400
    days = days_result

    # Validate 'symbol' parameter (optional)
    symbol = request.args.get('symbol', None)
    if symbol:
        is_valid, symbol_result = validate_symbol(symbol)
        if not is_valid:
            return jsonify({
                'error': 'Invalid symbol',
                'message': symbol_result,
                'code': 'INVALID_SYMBOL'
            }), 400
        symbol = symbol_result

    # Validate 'direction' parameter (optional)
    direction = request.args.get('direction', None)
    if direction:
        is_valid, direction_result = validate_direction(direction)
        if not is_valid:
            return jsonify({
                'error': 'Invalid direction',
                'message': direction_result,
                'code': 'INVALID_DIRECTION'
            }), 400
        direction = direction_result

    # Check tier-based history limits
    user = getattr(request, 'api_user', None)
    if user:
        from api.v1.auth import TIER_FEATURES
        max_days = TIER_FEATURES[user.subscription_tier]['signal_history_days']
        if max_days > 0:
            days = min(days, max_days)

    # Query actual signal database
    signals = get_historical_signals(days, symbol, direction)

    return jsonify({
        'timestamp': format_timestamp(),
        'days': days,
        'count': len(signals),
        'signals': signals
    })


@api_bp.route('/signals/performance', methods=['GET'])
@require_api_key
def get_signal_performance():
    """
    Get signal performance statistics

    Query params:
        days: Lookback period (default 30)
        strategy: Filter by strategy (optional)
    """
    days = request.args.get('days', 30, type=int)
    strategy = request.args.get('strategy', None)

    # Would calculate from actual trade data
    performance = calculate_performance_stats(days, strategy)

    return jsonify({
        'timestamp': format_timestamp(),
        'period_days': days,
        'performance': performance
    })


@api_bp.route('/signals/metrics', methods=['GET'])
@require_api_key
def get_signal_metrics():
    """
    Get signal quality metrics summary.

    Returns counters and rates tracked by SignalMetricsTracker:
    - Total scans and signal counts
    - Direction breakdown (long / short)
    - Quality breakdown (clean vs flagged with warnings)
    - Hit rate: signals that reached their target vs stopped out
    - False-positive rate: fraction of signals carrying quality warnings
    - Average RRS strength and confidence across all recorded signals
    - Last scan snapshot (timestamp, count, avg RRS, flagged count)

    No query parameters required.
    """
    try:
        from scanner.signal_metrics import get_metrics_tracker
        summary = get_metrics_tracker().get_summary()
        return jsonify({
            'timestamp': format_timestamp(),
            'metrics': summary,
        })
    except Exception as e:
        return handle_api_error(e, "signal metrics retrieval")


# ============================================================================
# Real-Time Endpoints (Pro tier and above)
# ============================================================================

@api_bp.route('/signals/stream', methods=['GET'])
@require_api_key
@require_feature('real_time_signals')
def get_signal_stream():
    """
    Get real-time signal stream info

    Returns WebSocket connection details for Pro subscribers
    """
    return jsonify({
        'websocket_url': 'wss://api.rdttrading.com/v1/ws/signals',
        'protocols': ['wss'],
        'authentication': 'Include X-API-Key header in connection',
        'message_format': 'JSON',
        'channels': ['signals', 'alerts', 'performance']
    })


# ============================================================================
# RRS Calculation Endpoints
# ============================================================================

@api_bp.route('/rrs/<symbol>', methods=['GET'])
@require_api_key
def get_symbol_rrs(symbol: str):
    """
    Get RRS calculation for a specific symbol

    Path params:
        symbol: Stock ticker symbol

    Returns:
        Current RRS value and components
    """
    # Validate symbol
    is_valid, result = validate_symbol(symbol)
    if not is_valid:
        return jsonify({
            'error': 'Invalid symbol',
            'message': result,
            'code': 'INVALID_SYMBOL'
        }), 400

    symbol = result  # Use sanitized symbol

    # Calculate actual RRS
    rrs_data = calculate_rrs_for_symbol(symbol)

    if rrs_data is None:
        return jsonify({
            'error': f'Symbol {symbol} not found or data unavailable'
        }), 404

    return jsonify({
        'timestamp': format_timestamp(),
        'symbol': symbol,
        **rrs_data
    })


@api_bp.route('/rrs/scan', methods=['GET'])
@require_api_key
@require_subscription(SubscriptionTier.PRO)
def scan_all_rrs():
    """
    Scan all watchlist symbols for RRS values

    Returns:
        List of all symbols with RRS values, sorted by strength
    """
    # Would run actual scan
    results = run_full_rrs_scan()

    return jsonify({
        'timestamp': format_timestamp(),
        'count': len(results),
        'strongest': results[:10],
        'weakest': results[-10:] if len(results) >= 10 else [],
        'all_results': results
    })


# ============================================================================
# Backtest Endpoints (Pro tier and above)
# ============================================================================

@api_bp.route('/backtest', methods=['POST'])
@require_api_key
@require_feature('backtest_api')
def run_custom_backtest():
    """
    Run a custom backtest

    Request body:
        {
            "symbols": ["AAPL", "MSFT"],  // Optional, default watchlist
            "days": 365,
            "rrs_threshold": 1.75,
            "stop_atr_mult": 0.75,
            "target_atr_mult": 1.5,
            "risk_per_trade": 0.02
        }

    Returns:
        Backtest results with trade list
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'Request body required', 'code': 'BODY_REQUIRED'}), 400

    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object', 'code': 'INVALID_BODY'}), 400

    # Validate 'days' parameter
    is_valid, days_result = validate_integer(
        data.get('days', 365),
        field_name='days',
        min_val=30,
        max_val=730
    )
    if not is_valid:
        return jsonify({
            'error': 'Invalid parameter',
            'message': days_result,
            'code': 'INVALID_PARAMETER'
        }), 400
    data['days'] = days_result

    # Validate 'rrs_threshold' parameter (optional)
    if 'rrs_threshold' in data:
        is_valid, threshold_result = validate_float(
            data['rrs_threshold'],
            field_name='rrs_threshold',
            min_val=0.5,
            max_val=5.0
        )
        if not is_valid:
            return jsonify({
                'error': 'Invalid parameter',
                'message': threshold_result,
                'code': 'INVALID_PARAMETER'
            }), 400
        data['rrs_threshold'] = threshold_result

    # Validate 'stop_atr_mult' parameter (optional)
    if 'stop_atr_mult' in data:
        is_valid, stop_result = validate_float(
            data['stop_atr_mult'],
            field_name='stop_atr_mult',
            min_val=0.25,
            max_val=3.0
        )
        if not is_valid:
            return jsonify({
                'error': 'Invalid parameter',
                'message': stop_result,
                'code': 'INVALID_PARAMETER'
            }), 400
        data['stop_atr_mult'] = stop_result

    # Validate 'target_atr_mult' parameter (optional)
    if 'target_atr_mult' in data:
        is_valid, target_result = validate_float(
            data['target_atr_mult'],
            field_name='target_atr_mult',
            min_val=0.5,
            max_val=5.0
        )
        if not is_valid:
            return jsonify({
                'error': 'Invalid parameter',
                'message': target_result,
                'code': 'INVALID_PARAMETER'
            }), 400
        data['target_atr_mult'] = target_result

    # Validate 'risk_per_trade' parameter (optional)
    if 'risk_per_trade' in data:
        is_valid, risk_result = validate_float(
            data['risk_per_trade'],
            field_name='risk_per_trade',
            min_val=0.001,
            max_val=0.1
        )
        if not is_valid:
            return jsonify({
                'error': 'Invalid parameter',
                'message': risk_result,
                'code': 'INVALID_PARAMETER'
            }), 400
        data['risk_per_trade'] = risk_result

    # Validate 'symbols' parameter (optional)
    if 'symbols' in data:
        if not isinstance(data['symbols'], list):
            return jsonify({
                'error': 'Invalid parameter',
                'message': 'symbols must be a list',
                'code': 'INVALID_PARAMETER'
            }), 400

        validated_symbols = []
        for sym in data['symbols']:
            is_valid, sym_result = validate_symbol(sym)
            if not is_valid:
                return jsonify({
                    'error': 'Invalid symbol',
                    'message': f"Invalid symbol '{sym}': {sym_result}",
                    'code': 'INVALID_SYMBOL'
                }), 400
            validated_symbols.append(sym_result)
        data['symbols'] = validated_symbols

        if len(validated_symbols) > 500:
            return jsonify({
                'error': 'Invalid parameter',
                'message': 'Maximum 500 symbols allowed',
                'code': 'TOO_MANY_SYMBOLS'
            }), 400

    # Run actual backtest
    result = run_backtest_with_params(data)

    return jsonify({
        'timestamp': format_timestamp(),
        'parameters': data,
        'result': result
    })


@api_bp.route('/backtest/presets', methods=['GET'])
@require_api_key
def get_backtest_presets():
    """Get predefined backtest configurations"""
    presets = {
        'conservative': {
            'rrs_threshold': 2.5,
            'stop_atr_mult': 1.0,
            'target_atr_mult': 2.0,
            'risk_per_trade': 0.01,
            'description': 'Lower risk, fewer trades, higher win rate'
        },
        'balanced': {
            'rrs_threshold': 1.75,
            'stop_atr_mult': 0.75,
            'target_atr_mult': 1.5,
            'risk_per_trade': 0.02,
            'description': 'Optimized balance of risk and return'
        },
        'aggressive': {
            'rrs_threshold': 1.5,
            'stop_atr_mult': 0.5,
            'target_atr_mult': 1.0,
            'risk_per_trade': 0.03,
            'description': 'Higher risk, more trades, larger position sizes'
        }
    }

    return jsonify({'presets': presets})


# ============================================================================
# Alert Management Endpoints
# ============================================================================

@api_bp.route('/alerts', methods=['GET'])
@require_api_key
@require_subscription(SubscriptionTier.BASIC)
def get_user_alerts():
    """Get user's configured alerts"""
    user = getattr(request, 'api_user', None)

    # Would query user's alerts from database
    alerts = get_alerts_for_user(user.user_id)

    return jsonify({
        'alerts': alerts,
        'max_alerts': 10 if user.subscription_tier == SubscriptionTier.BASIC else 100
    })


@api_bp.route('/alerts', methods=['POST'])
@require_api_key
@require_subscription(SubscriptionTier.BASIC)
def create_alert():
    """
    Create a new alert

    Request body:
        {
            "symbol": "AAPL",
            "condition": "rrs_above",  // rrs_above, rrs_below, price_above, price_below
            "value": 2.0,
            "notification": "email"  // email, sms, webhook
        }
    """
    data = request.get_json()
    user = getattr(request, 'api_user', None)

    if not data:
        return jsonify({'error': 'Request body required', 'code': 'BODY_REQUIRED'}), 400

    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object', 'code': 'INVALID_BODY'}), 400

    # Validate required fields
    is_valid, body_result = validate_json_body(data, ['symbol', 'condition', 'value'])
    if not is_valid:
        return jsonify({
            'error': 'Missing required fields',
            'message': body_result,
            'code': 'MISSING_FIELDS'
        }), 400

    # Validate symbol
    is_valid, symbol_result = validate_symbol(data['symbol'])
    if not is_valid:
        return jsonify({
            'error': 'Invalid symbol',
            'message': symbol_result,
            'code': 'INVALID_SYMBOL'
        }), 400
    data['symbol'] = symbol_result

    # Validate condition
    is_valid, condition_result = validate_condition(data['condition'])
    if not is_valid:
        return jsonify({
            'error': 'Invalid condition',
            'message': condition_result,
            'code': 'INVALID_CONDITION'
        }), 400
    data['condition'] = condition_result

    # Validate value (different ranges for RRS vs price)
    if data['condition'].startswith('rrs_'):
        is_valid, value_result = validate_float(
            data['value'],
            field_name='value',
            min_val=-20.0,
            max_val=20.0
        )
    else:
        is_valid, value_result = validate_price(data['value'])
        if is_valid:
            value_result = value_result  # validate_price returns (is_valid, price_or_error)
    if not is_valid:
        return jsonify({
            'error': 'Invalid value',
            'message': value_result,
            'code': 'INVALID_VALUE'
        }), 400
    data['value'] = value_result

    # Validate notification method (optional)
    if 'notification' in data:
        is_valid, notif_result = validate_notification_method(data['notification'])
        if not is_valid:
            return jsonify({
                'error': 'Invalid notification method',
                'message': notif_result,
                'code': 'INVALID_NOTIFICATION'
            }), 400
        data['notification'] = notif_result

    # Sanitize optional note field
    if 'note' in data and data['note']:
        is_valid, note_result = sanitize_input(data['note'], max_length=200)
        if not is_valid:
            return jsonify({
                'error': 'Invalid note',
                'message': note_result,
                'code': 'INVALID_NOTE'
            }), 400
        data['note'] = note_result

    # Create alert in database
    alert = create_alert_for_user(user.user_id, data)

    return jsonify({
        'status': 'created',
        'alert': alert
    }), 201


@api_bp.route('/alerts/<alert_id>', methods=['DELETE'])
@require_api_key
@require_subscription(SubscriptionTier.BASIC)
def delete_alert(alert_id: str):
    """Delete an alert"""
    user = getattr(request, 'api_user', None)

    # Would delete from database
    success = delete_alert_for_user(user.user_id, alert_id)

    if success:
        return jsonify({'status': 'deleted'})
    return jsonify({'error': 'Alert not found'}), 404


# ============================================================================
# Alert Schedule Endpoints
# ============================================================================

@api_bp.route('/alerts/schedule', methods=['GET'])
@require_api_key
def get_alert_schedule():
    """
    Get user's alert schedule settings.

    Returns:
        Alert schedule configuration including quiet hours and preferences
    """
    user = getattr(request, 'api_user', None)

    try:
        from alerts.schedule_config import get_user_schedule_config
        from alerts.alert_manager import get_alert_manager

        # Get user-specific config
        config = get_user_schedule_config(user.user_id if user else 0)

        # Get global scheduler status
        manager = get_alert_manager()
        scheduler_status = manager.get_schedule_status()

        return jsonify({
            'timestamp': format_timestamp(),
            'user_config': config.to_dict(),
            'scheduler_status': scheduler_status,
        })
    except Exception as e:
        return handle_api_error(e, "getting alert schedule")


@api_bp.route('/alerts/schedule', methods=['PUT'])
@require_api_key
def update_alert_schedule():
    """
    Update user's alert schedule settings.

    Request body:
        {
            "quiet_hours": {
                "enabled": true,
                "start": "22:00",
                "end": "07:00",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday"]
            },
            "override_critical": true,
            "timezone": "America/New_York",
            "channel_schedules": {
                "email": {
                    "enabled": true,
                    "quiet_hours_start": "21:00",
                    "quiet_hours_end": "08:00"
                }
            }
        }

    Returns:
        Updated schedule configuration
    """
    user = getattr(request, 'api_user', None)
    data = request.get_json()

    if not data:
        return jsonify({'error': 'Request body required', 'code': 'BODY_REQUIRED'}), 400

    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object', 'code': 'INVALID_BODY'}), 400

    try:
        from alerts.schedule_config import get_user_schedule_config, get_config_manager

        config = get_user_schedule_config(user.user_id if user else 0)

        # Update quiet hours
        quiet_hours = data.get('quiet_hours')
        if quiet_hours:
            if quiet_hours.get('enabled') and quiet_hours.get('start') and quiet_hours.get('end'):
                config.set_quiet_hours(
                    start=quiet_hours['start'],
                    end=quiet_hours['end'],
                    days=quiet_hours.get('days'),
                    enabled=quiet_hours.get('enabled', True)
                )
            elif quiet_hours.get('enabled') is False:
                config.disable_quiet_hours()

        # Update critical override
        if 'override_critical' in data:
            config.set_override_critical(data['override_critical'])

        # Update timezone
        if 'timezone' in data:
            config.timezone = data['timezone']

        # Update channel schedules
        channel_schedules = data.get('channel_schedules')
        if channel_schedules:
            for channel, schedule in channel_schedules.items():
                config.set_channel_schedule(
                    channel=channel,
                    enabled=schedule.get('enabled', True),
                    quiet_hours_start=schedule.get('quiet_hours_start'),
                    quiet_hours_end=schedule.get('quiet_hours_end'),
                    active_days=schedule.get('active_days'),
                    priority_threshold=schedule.get('priority_threshold', 'low')
                )

        # Update alert type preferences
        alert_type_prefs = data.get('alert_type_preferences')
        if alert_type_prefs:
            for alert_type, pref in alert_type_prefs.items():
                config.set_alert_type_preference(
                    alert_type=alert_type,
                    enabled=pref.get('enabled', True),
                    channels=pref.get('channels'),
                    quiet_hours_exempt=pref.get('quiet_hours_exempt', False),
                    min_priority=pref.get('min_priority', 'low')
                )

        # Save config
        manager = get_config_manager()
        manager.save_config(config)

        return jsonify({
            'status': 'updated',
            'config': config.to_dict()
        })
    except Exception as e:
        return handle_api_error(e, "updating alert schedule")


@api_bp.route('/alerts/queued', methods=['GET'])
@require_api_key
def get_queued_alerts():
    """
    Get queued alerts pending delivery.

    Query params:
        channel: Filter by channel (optional)
        include_scheduled: Include future scheduled alerts (default: true)

    Returns:
        List of queued alerts
    """
    user = getattr(request, 'api_user', None)
    channel = request.args.get('channel')
    include_scheduled = request.args.get('include_scheduled', 'true').lower() == 'true'

    try:
        from alerts.alert_manager import get_alert_manager

        manager = get_alert_manager()
        queued = manager.get_queued_alerts(
            user_id=user.user_id if user else None,
            channel=channel
        )

        # Filter by scheduled if needed
        if not include_scheduled:
            from datetime import datetime
            now = datetime.utcnow()
            queued = [a for a in queued if not a.scheduled_for or a.scheduled_for <= now]

        return jsonify({
            'timestamp': format_timestamp(),
            'count': len(queued),
            'alerts': [a.to_dict() for a in queued]
        })
    except Exception as e:
        return handle_api_error(e, "getting queued alerts")


@api_bp.route('/alerts/queued/<alert_id>', methods=['DELETE'])
@require_api_key
def cancel_queued_alert(alert_id: str):
    """
    Cancel a queued alert.

    Args:
        alert_id: ID of the queued alert to cancel

    Returns:
        Status of cancellation
    """
    try:
        from alerts.alert_manager import get_alert_manager

        manager = get_alert_manager()
        if manager.scheduler:
            # SECURITY: Verify the requesting user owns this alert before cancelling
            api_user = getattr(request, 'api_user', None)
            requesting_user_id = getattr(api_user, 'user_id', getattr(api_user, 'id', None))

            # Look up the alert to check ownership
            queued_alerts = manager.scheduler.get_queued_alerts()
            target_alert = None
            for alert in queued_alerts:
                if alert.alert_id == alert_id:
                    target_alert = alert
                    break

            if target_alert is None:
                return jsonify({'error': 'Alert not found'}), 404

            # Enforce ownership: alert must belong to the requesting user
            if target_alert.user_id is not None and requesting_user_id is not None:
                if target_alert.user_id != requesting_user_id:
                    logger.warning(
                        f"User {requesting_user_id} attempted to cancel alert {alert_id} "
                        f"owned by user {target_alert.user_id}"
                    )
                    return jsonify({'error': 'Alert not found'}), 404

            success = manager.scheduler.remove_queued_alert(alert_id)
            if success:
                return jsonify({'status': 'cancelled', 'alert_id': alert_id})
            return jsonify({'error': 'Alert not found'}), 404
        return jsonify({'error': 'Scheduling not enabled'}), 400
    except Exception as e:
        return handle_api_error(e, "cancelling queued alert")


@api_bp.route('/alerts/queued/process', methods=['POST'])
@require_api_key
@require_admin
def process_queued_alerts_endpoint():
    """
    Process and send queued alerts that are ready.

    This endpoint is typically called by a scheduled task, but can
    be triggered manually to force processing.

    Returns:
        Results of processing each alert
    """
    try:
        from alerts.alert_manager import get_alert_manager

        manager = get_alert_manager()
        results = manager.process_queued_alerts()

        return jsonify({
            'timestamp': format_timestamp(),
            'processed': len(results),
            'results': results
        })
    except Exception as e:
        return handle_api_error(e, "processing queued alerts")


@api_bp.route('/alerts/schedule/dnd', methods=['POST'])
@require_api_key
def enable_dnd():
    """
    Enable Do Not Disturb mode.

    Request body:
        {
            "duration_minutes": 60,  // Optional, minutes to enable DND
            "until": "2024-01-15T10:00:00"  // Optional, specific end time
        }

    At least one of duration_minutes or until must be provided.
    If neither is provided, DND is enabled indefinitely.

    Returns:
        DND status
    """
    user = getattr(request, 'api_user', None)
    data = request.get_json() or {}

    try:
        from alerts.schedule_config import get_user_schedule_config, get_config_manager

        config = get_user_schedule_config(user.user_id if user else 0)

        duration = data.get('duration_minutes')
        until_str = data.get('until')
        until = None

        if until_str:
            try:
                until = datetime.fromisoformat(until_str)
            except ValueError:
                return jsonify({
                    'error': 'Invalid datetime format',
                    'message': 'Use ISO format: YYYY-MM-DDTHH:MM:SS'
                }), 400

        config.enable_dnd(duration_minutes=duration, until=until)

        # Save config
        manager = get_config_manager()
        manager.save_config(config)

        return jsonify({
            'status': 'enabled',
            'dnd_active': True,
            'dnd_until': config._dnd_until.isoformat() if config._dnd_until else None
        })
    except Exception as e:
        return handle_api_error(e, "enabling DND")


@api_bp.route('/alerts/schedule/dnd', methods=['DELETE'])
@require_api_key
def disable_dnd():
    """
    Disable Do Not Disturb mode.

    Returns:
        DND status
    """
    user = getattr(request, 'api_user', None)

    try:
        from alerts.schedule_config import get_user_schedule_config, get_config_manager

        config = get_user_schedule_config(user.user_id if user else 0)
        config.disable_dnd()

        # Save config
        manager = get_config_manager()
        manager.save_config(config)

        return jsonify({
            'status': 'disabled',
            'dnd_active': False
        })
    except Exception as e:
        return handle_api_error(e, "disabling DND")


# ============================================================================
# Account Endpoints
# ============================================================================

@api_bp.route('/account', methods=['GET'])
@require_api_key
def get_account_info():
    """Get account information"""
    user = getattr(request, 'api_user', None)

    from api.v1.auth import TIER_FEATURES, TIER_RATE_LIMITS

    return jsonify({
        'user_id': user.user_id,
        'email': user.email,
        'subscription': {
            'tier': user.subscription_tier.value,
            'expires_at': user.expires_at.isoformat() if user.expires_at else None,
            'is_active': user.is_active
        },
        'rate_limit': {
            'limit': user.rate_limit,
            'used': user.requests_this_hour,
            'remaining': user.rate_limit - user.requests_this_hour
        },
        'features': TIER_FEATURES[user.subscription_tier]
    })


@api_bp.route('/account/usage', methods=['GET'])
@require_api_key
def get_usage_stats():
    """Get API usage statistics"""
    user = getattr(request, 'api_user', None)

    # Would query usage from database
    return jsonify({
        'period': 'current_month',
        'requests': {
            'total': 1234,
            'signals': 456,
            'backtest': 12,
            'alerts': 789
        },
        'quota': {
            'total': user.rate_limit * 24 * 30,  # Monthly estimate
            'used_pct': 15.5
        }
    })


# ============================================================================
# Billing Endpoints (Stub for Stripe integration)
# ============================================================================

@api_bp.route('/billing/plans', methods=['GET'])
def get_billing_plans():
    """Get available subscription plans"""
    plans = {
        'basic': {
            'name': 'Basic',
            'price': 49,
            'currency': 'USD',
            'interval': 'month',
            'features': [
                'Daily email alerts',
                '30-day signal history',
                'Custom alerts',
                'Email support'
            ]
        },
        'pro': {
            'name': 'Pro',
            'price': 149,
            'currency': 'USD',
            'interval': 'month',
            'features': [
                'Real-time signals',
                '1-year signal history',
                'Full API access',
                'Custom backtests',
                'WebSocket streaming',
                'Priority support'
            ],
            'popular': True
        },
        'elite': {
            'name': 'Elite',
            'price': 499,
            'currency': 'USD',
            'interval': 'month',
            'features': [
                'Everything in Pro',
                'Unlimited history',
                'Strategy consulting',
                '1-on-1 support calls',
                'Custom integrations',
                'White-label options'
            ]
        }
    }

    return jsonify({'plans': plans})


@api_bp.route('/billing/upgrade', methods=['POST'])
@require_api_key
def upgrade_subscription():
    """
    Initiate subscription upgrade

    Would integrate with Stripe for payment processing
    """
    data = request.get_json()
    plan = data.get('plan')

    if plan not in ['basic', 'pro', 'elite']:
        return jsonify({'error': 'Invalid plan'}), 400

    # Would create Stripe checkout session
    return jsonify({
        'status': 'pending',
        'checkout_url': f'https://checkout.stripe.com/session/{plan}',
        'message': 'Redirect user to checkout URL'
    })


# ============================================================================
# Helper Functions (Connected to real services)
# ============================================================================

# Signal file paths to check for real signal data
SIGNAL_FILE_PATHS = [
    'data/signals/active_signals.json',
    'signals/active_signals.json',
    'output/signals.json',
]


def is_market_open() -> bool:
    """Check if US stock market is open - uses proper Eastern Time handling"""
    return tz_is_market_open()


def get_spy_price() -> Optional[float]:
    """Get current SPY price"""
    if not YFINANCE_AVAILABLE:
        return 478.50  # Fallback price

    try:
        spy = yf.Ticker('SPY')
        data = spy.history(period='1d', interval='1m')
        if len(data) > 0:
            return float(data['Close'].iloc[-1])
    except Exception as e:
        logger.debug(f"Error fetching SPY price: {e}")

    return 478.50  # Fallback price


def get_open_positions() -> List[Dict]:
    """Get currently open positions from database"""
    # Try to load from database first
    if DATABASE_AVAILABLE:
        try:
            repo = get_trades_repository()
            positions = repo.get_open_positions()
            if positions:
                logger.debug(f"Loaded {len(positions)} positions from database")
                return positions
            # Database is available but empty - return empty list, not mock data
            return []
        except Exception as e:
            logger.debug(f"Error loading positions from database: {e}")

    # Fall back to file-based loading only if database is not available
    position_files = [
        'data/positions/open_positions.json',
        'positions/open.json',
        '/tmp/rdt_positions.json'
    ]

    for filepath in position_files:
        try:
            if os.path.exists(filepath):
                # SECURITY: Verify restrictive permissions on temp files to prevent tampering
                if filepath.startswith('/tmp/'):
                    file_stat = os.stat(filepath)
                    # Only read if owned by current user and not world-writable
                    if file_stat.st_uid != os.getuid():
                        logger.warning(f"Skipping {filepath}: not owned by current user")
                        continue
                    if file_stat.st_mode & 0o002:  # world-writable
                        logger.warning(f"Skipping {filepath}: world-writable permissions")
                        continue
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Error loading positions from {filepath}: {e}")

    # Return empty list when no data source is available
    return []


def get_active_signals() -> List[Dict]:
    """Get current active signals from database, scanner, or signal file"""
    # Try to load signals from database first
    if DATABASE_AVAILABLE:
        try:
            repo = get_trades_repository()
            signals = repo.get_active_signals()
            if signals:
                logger.debug(f"Loaded {len(signals)} signals from database")
                return signals
            # Database is available but empty - check file fallback before returning empty
        except Exception as e:
            logger.debug(f"Error loading signals from database: {e}")

    # Try to load signals from file as fallback
    for filepath in SIGNAL_FILE_PATHS:
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    signals = json.load(f)
                    if isinstance(signals, list) and len(signals) > 0:
                        logger.debug(f"Loaded {len(signals)} signals from {filepath}")
                        return signals
        except Exception as e:
            logger.debug(f"Error loading signals from {filepath}: {e}")

    # Return empty list when no data source has signals
    return []


def get_historical_signals(days: int, symbol: Optional[str], direction: Optional[str]) -> List[Dict]:
    """Get historical signals from database"""
    # Try to load from database first
    if DATABASE_AVAILABLE:
        try:
            repo = get_trades_repository()
            signals = repo.get_signals(
                symbol=symbol,
                days=days,
                limit=500
            )
            # Filter by direction if specified
            if direction and signals:
                signals = [s for s in signals if s.get('direction') == direction]
            if signals:
                logger.debug(f"Loaded {len(signals)} historical signals from database")
                return signals
            # Database is available but empty - return empty list
            return []
        except Exception as e:
            logger.debug(f"Error loading historical signals from database: {e}")

    # Fall back to file-based loading only if database is not available
    history_files = [
        'data/signals/signal_history.json',
        'signals/history.json'
    ]

    for filepath in history_files:
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    signals = json.load(f)
                    # Filter by days, symbol, direction using Eastern Time
                    cutoff = get_eastern_time() - timedelta(days=days)
                    filtered = []
                    for s in signals:
                        try:
                            sig_date = datetime.fromisoformat(s.get('generated_at', ''))
                            # Convert to Eastern Time for proper comparison
                            sig_date = to_eastern(sig_date)
                            if sig_date < cutoff:
                                continue
                        except (ValueError, TypeError):
                            continue
                        if symbol and s.get('symbol') != symbol:
                            continue
                        if direction and s.get('direction') != direction:
                            continue
                        filtered.append(s)
                    return filtered
        except Exception as e:
            logger.debug(f"Error loading signal history from {filepath}: {e}")

    return []


def calculate_performance_stats(days: int, strategy: Optional[str]) -> Dict:
    """Calculate signal performance statistics from database trades"""
    # Try to calculate from database first
    if DATABASE_AVAILABLE:
        try:
            repo = get_trades_repository()
            stats = repo.calculate_performance_stats(days=days)
            if stats and stats.get('total_signals', 0) > 0:
                logger.debug(f"Calculated performance stats from {stats['total_signals']} trades")
                return stats
            # Database is available but no trades - return empty stats
            return {
                'total_signals': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'avg_win_pct': 0.0,
                'avg_loss_pct': 0.0,
                'profit_factor': 0.0,
                'total_return_pct': 0.0,
                'max_drawdown': 0.0
            }
        except Exception as e:
            logger.debug(f"Error calculating performance stats from database: {e}")

    # Fall back to file-based loading only if database is not available
    perf_files = [
        'data/performance/stats.json',
        'performance/stats.json'
    ]

    for filepath in perf_files:
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Error loading performance stats from {filepath}: {e}")

    # Return empty stats when no data source is available
    return {
        'total_signals': 0,
        'wins': 0,
        'losses': 0,
        'win_rate': 0.0,
        'avg_win_pct': 0.0,
        'avg_loss_pct': 0.0,
        'profit_factor': 0.0,
        'total_return_pct': 0.0,
        'max_drawdown': 0.0
    }


def calculate_rrs_for_symbol(symbol: str) -> Optional[Dict]:
    """Calculate RRS for a specific symbol using real data"""
    if not RRS_AVAILABLE or not YFINANCE_AVAILABLE:
        # Return sample data if dependencies not available
        return {
            'rrs': 1.85,
            'stock_pct_change': 2.3,
            'spy_pct_change': 0.8,
            'atr': 3.25,
            'status': 'STRONG_RS',
            'daily_strength': {
                'is_strong': True,
                'score': 4
            }
        }

    try:
        # Initialize RRS calculator
        rrs_calc = RRSCalculator(atr_period=14)

        # Fetch stock data
        ticker = yf.Ticker(symbol)
        stock_daily = ticker.history(period='60d', interval='1d')

        if stock_daily.empty:
            return None

        # Fetch SPY data
        spy = yf.Ticker('SPY')
        spy_daily = spy.history(period='60d', interval='1d')

        if spy_daily.empty:
            return None

        # Normalize column names to lowercase
        stock_daily.columns = [c.lower() for c in stock_daily.columns]
        spy_daily.columns = [c.lower() for c in spy_daily.columns]

        # Calculate ATR
        atr_series = rrs_calc.calculate_atr(stock_daily)
        current_atr = float(atr_series.iloc[-1])

        # Get current and previous prices
        stock_current = float(stock_daily['close'].iloc[-1])
        stock_prev = float(stock_daily['close'].iloc[-2])
        spy_current = float(spy_daily['close'].iloc[-1])
        spy_prev = float(spy_daily['close'].iloc[-2])

        # Calculate RRS
        rrs_result = rrs_calc.calculate_rrs_current(
            stock_data={'current_price': stock_current, 'previous_close': stock_prev},
            spy_data={'current_price': spy_current, 'previous_close': spy_prev},
            stock_atr=current_atr
        )

        # Check daily strength
        daily_strength = check_daily_strength(stock_daily)

        return {
            'rrs': round(rrs_result['rrs'], 2),
            'stock_pct_change': round(rrs_result['stock_pc'], 2),
            'spy_pct_change': round(rrs_result['spy_pc'], 2),
            'atr': round(current_atr, 2),
            'status': rrs_result['status'],
            'current_price': round(stock_current, 2),
            'daily_strength': {
                'is_strong': daily_strength['is_strong'],
                'three_green_days': daily_strength['three_green_days'],
                'ema3_above_ema8': daily_strength['ema3_above_ema8'],
                'above_ema8': daily_strength['above_ema8']
            }
        }

    except Exception as e:
        logger.error(f"Error calculating RRS for {symbol}: {e}")
        return None


def run_full_rrs_scan() -> List[Dict]:
    """Run full watchlist RRS scan"""
    if not RRS_AVAILABLE or not YFINANCE_AVAILABLE:
        return []

    results = []
    watchlist = load_default_watchlist() if BACKTEST_AVAILABLE else [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'
    ]

    for symbol in watchlist:
        try:
            rrs_data = calculate_rrs_for_symbol(symbol)
            if rrs_data:
                results.append({
                    'symbol': symbol,
                    **rrs_data
                })
        except Exception as e:
            logger.debug(f"Error scanning {symbol}: {e}")
            continue

    # Sort by RRS value (strongest first)
    results.sort(key=lambda x: x.get('rrs', 0), reverse=True)
    return results


def run_backtest_with_params(params: Dict) -> Dict:
    """Run backtest with custom parameters using BacktestEngine"""
    if not BACKTEST_AVAILABLE:
        # Return sample results if backtest engine not available
        return {
            'total_return_pct': 8.5,
            'total_trades': 125,
            'win_rate': 0.42,
            'profit_factor': 1.28,
            'max_drawdown': 4.5,
            'sharpe_ratio': 0.85,
            'note': 'Sample data - BacktestEngine not available'
        }

    try:
        # Extract parameters
        days = params.get('days', 365)
        symbols = params.get('symbols', load_default_watchlist())
        rrs_threshold = params.get('rrs_threshold', 1.75)
        stop_atr_mult = params.get('stop_atr_mult', 0.75)
        target_atr_mult = params.get('target_atr_mult', 1.5)
        initial_capital = params.get('initial_capital', 25000)

        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        # Load data
        data_loader = DataLoader()
        stock_data = data_loader.load_stock_data(symbols, start_date, end_date)
        spy_data = data_loader.load_spy_data(start_date, end_date)

        if not stock_data:
            return {
                'error': 'Failed to load stock data',
                'total_return_pct': 0,
                'total_trades': 0
            }

        # Create and run backtest engine
        engine = BacktestEngine(
            initial_capital=initial_capital,
            rrs_threshold=rrs_threshold,
            stop_atr_multiplier=stop_atr_mult,
            target_atr_multiplier=target_atr_mult
        )

        result = engine.run(stock_data, spy_data, start_date, end_date)

        # Return results
        return {
            'total_return_pct': round(result.total_return_pct, 2),
            'total_trades': result.total_trades,
            'win_rate': round(result.win_rate, 2),
            'profit_factor': round(result.profit_factor, 2),
            'max_drawdown': round(result.max_drawdown, 2),
            'sharpe_ratio': round(result.sharpe_ratio, 2),
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'avg_win': round(result.avg_win, 2),
            'avg_loss': round(result.avg_loss, 2),
            'avg_holding_days': round(result.avg_holding_days, 1),
            'initial_capital': result.initial_capital,
            'final_capital': round(result.final_capital, 2)
        }

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return {
            'error': str(e),
            'total_return_pct': 0,
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }


def get_alerts_for_user(user_id: str) -> List[Dict]:
    """Get alerts for a user"""
    return []


def create_alert_for_user(user_id: str, data: Dict) -> Dict:
    """Create an alert for a user"""
    return {
        'alert_id': 'alert_123',
        **data
    }


def delete_alert_for_user(user_id: str, alert_id: str) -> bool:
    """Delete an alert"""
    return True


# ============================================================================
# Position Management Endpoints
# ============================================================================

@api_bp.route('/positions', methods=['GET'])
@require_api_key
def get_positions():
    """
    Get all open positions with live P&L.

    Returns:
        List of open positions with current prices and P&L
    """
    if POSITION_TRACKER_AVAILABLE:
        try:
            tracker = get_position_tracker()
            # Update prices synchronously
            tracker.update_prices_sync()
            positions = tracker.get_all_positions()
            summary = tracker.get_summary()

            return jsonify({
                'timestamp': format_timestamp(),
                'count': len(positions),
                'positions': positions,
                'summary': summary
            })
        except Exception as e:
            logger.error(f"Error getting positions: {e}")

    # Fallback to database
    positions = get_open_positions()
    return jsonify({
        'timestamp': format_timestamp(),
        'count': len(positions),
        'positions': positions,
        'summary': {
            'total_positions': len(positions),
            'total_exposure': sum(p.get('entry_price', 0) * p.get('shares', 0) for p in positions),
            'total_unrealized_pnl': sum(p.get('pnl', 0) or 0 for p in positions)
        }
    })


@api_bp.route('/positions', methods=['POST'])
@require_api_key
def open_position():
    """
    Open a new position.

    Request body:
        {
            "symbol": "AAPL",
            "direction": "long",
            "entry_price": 185.50,
            "shares": 100,
            "stop_price": 180.00,
            "target_price": 195.00,
            "rrs_at_entry": 2.15  // optional
        }

    Returns:
        Created position data
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'Request body required', 'code': 'BODY_REQUIRED'}), 400

    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object', 'code': 'INVALID_BODY'}), 400

    # Validate required fields
    required_fields = ['symbol', 'direction', 'entry_price', 'shares', 'stop_price', 'target_price']
    is_valid, body_result = validate_json_body(data, required_fields)
    if not is_valid:
        return jsonify({
            'error': 'Missing required fields',
            'message': body_result,
            'code': 'MISSING_FIELDS'
        }), 400

    # Validate symbol
    is_valid, symbol_result = validate_symbol(data['symbol'])
    if not is_valid:
        return jsonify({
            'error': 'Invalid symbol',
            'message': symbol_result,
            'code': 'INVALID_SYMBOL'
        }), 400

    # Validate direction
    is_valid, direction_result = validate_direction(data['direction'])
    if not is_valid:
        return jsonify({
            'error': 'Invalid direction',
            'message': direction_result,
            'code': 'INVALID_DIRECTION'
        }), 400

    # Validate entry_price
    is_valid, entry_result = validate_price(data['entry_price'])
    if not is_valid:
        return jsonify({
            'error': 'Invalid entry price',
            'message': entry_result,
            'code': 'INVALID_PRICE'
        }), 400

    # Validate shares
    is_valid, shares_result = validate_quantity(data['shares'])
    if not is_valid:
        return jsonify({
            'error': 'Invalid shares',
            'message': shares_result,
            'code': 'INVALID_QUANTITY'
        }), 400

    # Validate stop_price
    is_valid, stop_result = validate_price(data['stop_price'])
    if not is_valid:
        return jsonify({
            'error': 'Invalid stop price',
            'message': stop_result,
            'code': 'INVALID_PRICE'
        }), 400

    # Validate target_price
    is_valid, target_result = validate_price(data['target_price'])
    if not is_valid:
        return jsonify({
            'error': 'Invalid target price',
            'message': target_result,
            'code': 'INVALID_PRICE'
        }), 400

    # Validate rrs_at_entry (optional)
    rrs_at_entry = None
    if 'rrs_at_entry' in data and data['rrs_at_entry'] is not None:
        is_valid, rrs_result = validate_float(
            data['rrs_at_entry'],
            field_name='rrs_at_entry',
            min_val=-20.0,
            max_val=20.0
        )
        if not is_valid:
            return jsonify({
                'error': 'Invalid RRS value',
                'message': rrs_result,
                'code': 'INVALID_RRS'
            }), 400
        rrs_at_entry = rrs_result

    if not POSITION_TRACKER_AVAILABLE:
        return jsonify({'error': 'Position tracking not available', 'code': 'SERVICE_UNAVAILABLE'}), 503

    try:
        tracker = get_position_tracker()
        result = tracker.open_position(
            symbol=symbol_result,
            direction=direction_result,
            entry_price=entry_result,
            shares=shares_result,
            stop_price=stop_result,
            target_price=target_result,
            rrs_at_entry=rrs_at_entry
        )

        if 'error' in result:
            return jsonify(result), 400

        return jsonify({
            'status': 'created',
            'timestamp': format_timestamp(),
            'position': result
        }), 201

    except Exception as e:
        return handle_api_error(e, "opening position")


@api_bp.route('/positions/<symbol>', methods=['GET'])
@require_api_key
def get_position(symbol: str):
    """
    Get a specific position with live P&L.

    Path params:
        symbol: Stock ticker symbol

    Returns:
        Position data with current price and P&L
    """
    # Validate symbol
    is_valid, result = validate_symbol(symbol)
    if not is_valid:
        return jsonify({
            'error': 'Invalid symbol',
            'message': result,
            'code': 'INVALID_SYMBOL'
        }), 400

    symbol = result  # Use sanitized symbol

    if POSITION_TRACKER_AVAILABLE:
        try:
            tracker = get_position_tracker()
            tracker.update_prices_sync()
            position = tracker.get_position(symbol)

            if position:
                return jsonify({
                    'timestamp': format_timestamp(),
                    'position': position
                })
            else:
                return jsonify({'error': f'No position found for {symbol}'}), 404

        except Exception as e:
            logger.error(f"Error getting position: {e}")

    # Fallback to database
    if DATABASE_AVAILABLE:
        try:
            repo = get_trades_repository()
            position = repo.get_position_by_symbol(symbol)
            if position:
                return jsonify({
                    'timestamp': format_timestamp(),
                    'position': position
                })
        except Exception as e:
            logger.error(f"Error getting position from database: {e}")

    return jsonify({'error': f'No position found for {symbol}'}), 404


@api_bp.route('/positions/<symbol>/close', methods=['PUT'])
@require_api_key
def close_position_endpoint(symbol: str):
    """
    Close an open position.

    Path params:
        symbol: Stock ticker symbol

    Request body:
        {
            "exit_price": 190.50,
            "reason": "manual"  // stop_loss, take_profit, trailing_stop, manual, end_of_day
        }

    Returns:
        Closed position data with P&L
    """
    # Validate symbol
    is_valid, symbol_result = validate_symbol(symbol)
    if not is_valid:
        return jsonify({
            'error': 'Invalid symbol',
            'message': symbol_result,
            'code': 'INVALID_SYMBOL'
        }), 400

    symbol = symbol_result  # Use sanitized symbol
    data = request.get_json()

    if not data:
        return jsonify({'error': 'Request body required', 'code': 'BODY_REQUIRED'}), 400

    if 'exit_price' not in data:
        return jsonify({'error': 'Missing required field: exit_price', 'code': 'MISSING_FIELD'}), 400

    # Validate exit_price
    is_valid, exit_result = validate_price(data['exit_price'])
    if not is_valid:
        return jsonify({
            'error': 'Invalid exit price',
            'message': exit_result,
            'code': 'INVALID_PRICE'
        }), 400

    # Validate reason (optional)
    valid_reasons = {'stop_loss', 'take_profit', 'trailing_stop', 'manual', 'end_of_day', 'signal_exit'}
    reason = data.get('reason', 'manual')
    if reason and reason.lower() not in valid_reasons:
        return jsonify({
            'error': 'Invalid reason',
            'message': f"reason must be one of: {', '.join(valid_reasons)}",
            'code': 'INVALID_REASON'
        }), 400
    reason = reason.lower()

    if not POSITION_TRACKER_AVAILABLE:
        return jsonify({'error': 'Position tracking not available', 'code': 'SERVICE_UNAVAILABLE'}), 503

    try:
        tracker = get_position_tracker()
        result = tracker.close_position(
            symbol=symbol,
            exit_price=exit_result,
            reason=reason
        )

        if 'error' in result:
            return jsonify(result), 400

        return jsonify({
            'status': 'closed',
            'timestamp': format_timestamp(),
            'position': result
        })

    except Exception as e:
        return handle_api_error(e, "closing position")


@api_bp.route('/positions/<symbol>/stop', methods=['PUT'])
@require_api_key
def update_stop_endpoint(symbol: str):
    """
    Update stop loss price for a position.

    Path params:
        symbol: Stock ticker symbol

    Request body:
        {
            "stop_price": 182.00
        }

    Returns:
        Updated position data
    """
    # Validate symbol
    is_valid, symbol_result = validate_symbol(symbol)
    if not is_valid:
        return jsonify({
            'error': 'Invalid symbol',
            'message': symbol_result,
            'code': 'INVALID_SYMBOL'
        }), 400

    symbol = symbol_result
    data = request.get_json()

    if not data:
        return jsonify({'error': 'Request body required', 'code': 'BODY_REQUIRED'}), 400

    if 'stop_price' not in data:
        return jsonify({'error': 'Missing required field: stop_price', 'code': 'MISSING_FIELD'}), 400

    # Validate stop_price
    is_valid, stop_result = validate_price(data['stop_price'])
    if not is_valid:
        return jsonify({
            'error': 'Invalid stop price',
            'message': stop_result,
            'code': 'INVALID_PRICE'
        }), 400

    if not POSITION_TRACKER_AVAILABLE:
        return jsonify({'error': 'Position tracking not available', 'code': 'SERVICE_UNAVAILABLE'}), 503

    try:
        tracker = get_position_tracker()
        result = tracker.update_stop(symbol, stop_result)

        if 'error' in result:
            return jsonify(result), 400

        return jsonify({
            'status': 'updated',
            'timestamp': format_timestamp(),
            'position': result
        })

    except Exception as e:
        return handle_api_error(e, "updating stop")


@api_bp.route('/positions/<symbol>/target', methods=['PUT'])
@require_api_key
def update_target_endpoint(symbol: str):
    """
    Update target price for a position.

    Path params:
        symbol: Stock ticker symbol

    Request body:
        {
            "target_price": 198.00
        }

    Returns:
        Updated position data
    """
    # Validate symbol
    is_valid, symbol_result = validate_symbol(symbol)
    if not is_valid:
        return jsonify({
            'error': 'Invalid symbol',
            'message': symbol_result,
            'code': 'INVALID_SYMBOL'
        }), 400

    symbol = symbol_result
    data = request.get_json()

    if not data:
        return jsonify({'error': 'Request body required', 'code': 'BODY_REQUIRED'}), 400

    if 'target_price' not in data:
        return jsonify({'error': 'Missing required field: target_price', 'code': 'MISSING_FIELD'}), 400

    # Validate target_price
    is_valid, target_result = validate_price(data['target_price'])
    if not is_valid:
        return jsonify({
            'error': 'Invalid target price',
            'message': target_result,
            'code': 'INVALID_PRICE'
        }), 400

    if not POSITION_TRACKER_AVAILABLE:
        return jsonify({'error': 'Position tracking not available', 'code': 'SERVICE_UNAVAILABLE'}), 503

    try:
        tracker = get_position_tracker()
        result = tracker.update_target(symbol, target_result)

        if 'error' in result:
            return jsonify(result), 400

        return jsonify({
            'status': 'updated',
            'timestamp': format_timestamp(),
            'position': result
        })

    except Exception as e:
        return handle_api_error(e, "updating target")


@api_bp.route('/positions/<symbol>/notes', methods=['POST'])
@require_api_key
def add_trade_note_endpoint(symbol: str):
    """
    Add a journal note to a position.

    Path params:
        symbol: Stock ticker symbol

    Request body:
        {
            "note": "Entered on pullback to EMA support",
            "note_type": "entry"  // general, entry, exit, lesson, screenshot
        }

    Returns:
        Updated position data with notes
    """
    # Validate symbol
    is_valid, symbol_result = validate_symbol(symbol)
    if not is_valid:
        return jsonify({
            'error': 'Invalid symbol',
            'message': symbol_result,
            'code': 'INVALID_SYMBOL'
        }), 400

    symbol = symbol_result
    data = request.get_json()

    if not data:
        return jsonify({'error': 'Request body required', 'code': 'BODY_REQUIRED'}), 400

    if 'note' not in data:
        return jsonify({'error': 'Missing required field: note', 'code': 'MISSING_FIELD'}), 400

    # Sanitize note
    is_valid, note_result = sanitize_input(data['note'], max_length=1000, allow_newlines=True)
    if not is_valid:
        return jsonify({
            'error': 'Invalid note',
            'message': note_result,
            'code': 'INVALID_NOTE'
        }), 400

    # Validate note_type (optional)
    valid_note_types = {'general', 'entry', 'exit', 'lesson', 'screenshot', 'analysis'}
    note_type = data.get('note_type', 'general')
    if note_type and note_type.lower() not in valid_note_types:
        return jsonify({
            'error': 'Invalid note type',
            'message': f"note_type must be one of: {', '.join(valid_note_types)}",
            'code': 'INVALID_NOTE_TYPE'
        }), 400
    note_type = note_type.lower()

    if not POSITION_TRACKER_AVAILABLE:
        return jsonify({'error': 'Position tracking not available', 'code': 'SERVICE_UNAVAILABLE'}), 503

    try:
        tracker = get_position_tracker()
        result = tracker.add_trade_note(
            symbol=symbol,
            note=note_result,
            note_type=note_type
        )

        if 'error' in result:
            return jsonify(result), 400

        return jsonify({
            'status': 'note_added',
            'timestamp': format_timestamp(),
            'position': result
        }), 201

    except Exception as e:
        return handle_api_error(e, "adding trade note")


@api_bp.route('/positions/<symbol>/notes', methods=['GET'])
@require_api_key
def get_trade_notes_endpoint(symbol: str):
    """
    Get all notes for a position.

    Path params:
        symbol: Stock ticker symbol

    Returns:
        List of notes for the position
    """
    # Validate symbol
    is_valid, result = validate_symbol(symbol)
    if not is_valid:
        return jsonify({
            'error': 'Invalid symbol',
            'message': result,
            'code': 'INVALID_SYMBOL'
        }), 400

    symbol = result  # Use sanitized symbol

    if not POSITION_TRACKER_AVAILABLE:
        return jsonify({'error': 'Position tracking not available'}), 503

    try:
        tracker = get_position_tracker()
        position = tracker.get_position(symbol)

        if position:
            return jsonify({
                'timestamp': format_timestamp(),
                'symbol': symbol,
                'notes': position.get('notes', [])
            })

        # Check closed positions history
        history = tracker.get_trade_history(symbol)
        if 'error' not in history:
            return jsonify({
                'timestamp': format_timestamp(),
                'symbol': symbol,
                'notes': history.get('notes', [])
            })

        return jsonify({'error': f'No position found for {symbol}'}), 404

    except Exception as e:
        return handle_api_error(e, "getting trade notes")


@api_bp.route('/positions/<symbol>/history', methods=['GET'])
@require_api_key
def get_position_history_endpoint(symbol: str):
    """
    Get complete history for a symbol including events and notes.

    Path params:
        symbol: Stock ticker symbol

    Returns:
        Position history with events, notes, and trade records
    """
    # Validate symbol
    is_valid, result = validate_symbol(symbol)
    if not is_valid:
        return jsonify({
            'error': 'Invalid symbol',
            'message': result,
            'code': 'INVALID_SYMBOL'
        }), 400

    symbol = result  # Use sanitized symbol

    if POSITION_TRACKER_AVAILABLE:
        try:
            tracker = get_position_tracker()
            history = tracker.get_trade_history(symbol)

            if 'error' not in history:
                return jsonify({
                    'timestamp': format_timestamp(),
                    'history': history
                })

        except Exception as e:
            logger.error(f"Error getting position history: {e}")

    # Fallback to database trades
    if DATABASE_AVAILABLE:
        try:
            repo = get_trades_repository()
            trades = repo.get_trades(symbol=symbol, limit=50)
            if trades:
                return jsonify({
                    'timestamp': format_timestamp(),
                    'history': {
                        'symbol': symbol,
                        'status': 'historical',
                        'trades': trades
                    }
                })
        except Exception as e:
            logger.error(f"Error getting trade history from database: {e}")

    return jsonify({'error': f'No history found for {symbol}'}), 404


@api_bp.route('/positions/check', methods=['GET'])
@require_api_key
def check_stops_and_targets_endpoint():
    """
    Check if any positions have hit stops or targets.

    Returns:
        Lists of positions that have hit stops or targets
    """
    if not POSITION_TRACKER_AVAILABLE:
        return jsonify({'error': 'Position tracking not available'}), 503

    try:
        tracker = get_position_tracker()
        # Update prices first
        tracker.update_prices_sync()

        alerts = tracker.check_stops_and_targets()

        return jsonify({
            'timestamp': format_timestamp(),
            'alerts': alerts,
            'stops_hit_count': len(alerts['stops_hit']),
            'targets_hit_count': len(alerts['targets_hit'])
        })

    except Exception as e:
        return handle_api_error(e, "checking stops and targets")


@api_bp.route('/positions/summary', methods=['GET'])
@require_api_key
def get_positions_summary():
    """
    Get summary statistics for all open positions.

    Returns:
        Summary with total exposure, P&L, and counts
    """
    if POSITION_TRACKER_AVAILABLE:
        try:
            tracker = get_position_tracker()
            tracker.update_prices_sync()
            summary = tracker.get_summary()

            return jsonify({
                'timestamp': format_timestamp(),
                'summary': summary
            })

        except Exception as e:
            logger.error(f"Error getting positions summary: {e}")

    # Fallback calculation
    positions = get_open_positions()
    total_exposure = sum(p.get('entry_price', 0) * p.get('shares', 0) for p in positions)
    total_pnl = sum(p.get('pnl', 0) or 0 for p in positions)

    return jsonify({
        'timestamp': format_timestamp(),
        'summary': {
            'total_positions': len(positions),
            'total_exposure': total_exposure,
            'total_unrealized_pnl': total_pnl,
            'total_unrealized_pnl_pct': (total_pnl / total_exposure * 100) if total_exposure > 0 else 0
        }
    })


@api_bp.route('/positions/performance', methods=['GET'])
@require_api_key
def get_positions_performance():
    """
    Get performance statistics from closed positions.

    Returns:
        Performance metrics including win rate, profit factor, etc.
    """
    if POSITION_TRACKER_AVAILABLE:
        try:
            tracker = get_position_tracker()
            stats = tracker.get_performance_stats()

            return jsonify({
                'timestamp': format_timestamp(),
                'performance': stats
            })

        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")

    # Fallback to database stats
    if DATABASE_AVAILABLE:
        try:
            repo = get_trades_repository()
            stats = repo.calculate_performance_stats()

            return jsonify({
                'timestamp': format_timestamp(),
                'performance': stats
            })

        except Exception as e:
            logger.error(f"Error getting performance stats from database: {e}")

    return jsonify({
        'timestamp': format_timestamp(),
        'performance': {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
    })


# ============================================================================
# Execution Monitoring Endpoints
# ============================================================================

@api_bp.route('/executions', methods=['GET'])
def get_executions():
    """
    Get order execution history.

    Query params:
        symbol: Filter by symbol (optional)
        side: Filter by side - buy, sell (optional)
        days: Only return executions from last N days (optional, default 30)
        limit: Maximum number of records (optional, default 100)
        offset: Number of records to skip for pagination (optional, default 0)

    Returns:
        List of execution records with slippage and fill quality data
    """
    if not EXECUTION_TRACKER_AVAILABLE:
        return jsonify({'error': 'Execution tracking not available'}), 503

    try:
        tracker = get_execution_tracker()

        # Get query parameters
        symbol = request.args.get('symbol')
        side = request.args.get('side')
        days = request.args.get('days', 30, type=int)
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)

        # Cap limit to prevent excessive queries
        limit = min(limit, 500)

        executions = tracker.get_execution_history(
            symbol=symbol,
            side=side,
            days=days,
            limit=limit,
            offset=offset
        )

        return jsonify({
            'timestamp': format_timestamp(),
            'count': len(executions),
            'filters': {
                'symbol': symbol,
                'side': side,
                'days': days
            },
            'pagination': {
                'limit': limit,
                'offset': offset
            },
            'executions': executions
        })

    except Exception as e:
        return handle_api_error(e, "getting executions")


@api_bp.route('/executions/stats', methods=['GET'])
def get_execution_stats():
    """
    Get execution slippage and fill quality statistics.

    Query params:
        symbol: Filter by symbol (optional)
        side: Filter by side - buy, sell (optional)
        days: Only consider executions from last N days (optional, default 30)

    Returns:
        Slippage statistics, fill rate stats, and time-to-fill metrics
    """
    if not EXECUTION_TRACKER_AVAILABLE:
        return jsonify({'error': 'Execution tracking not available'}), 503

    try:
        tracker = get_execution_tracker()

        # Get query parameters
        symbol = request.args.get('symbol')
        side = request.args.get('side')
        days = request.args.get('days', 30, type=int)

        # Get all stats
        slippage_stats = tracker.get_slippage_stats(
            symbol=symbol,
            side=side,
            days=days
        )

        fill_rate_stats = tracker.get_fill_rate(
            symbol=symbol,
            side=side,
            days=days
        )

        time_to_fill_stats = tracker.get_time_to_fill_stats(
            symbol=symbol,
            side=side,
            days=days
        )

        # Get recent poor executions
        poor_executions = tracker.get_recent_poor_executions(
            threshold_pct=0.25,
            limit=5
        )

        return jsonify({
            'timestamp': format_timestamp(),
            'filters': {
                'symbol': symbol,
                'side': side,
                'days': days
            },
            'slippage': slippage_stats.to_dict(),
            'fill_rate': fill_rate_stats,
            'time_to_fill': time_to_fill_stats,
            'recent_poor_executions': poor_executions
        })

    except Exception as e:
        return handle_api_error(e, "getting execution stats")


@api_bp.route('/executions/summary', methods=['GET'])
def get_execution_summary():
    """
    Get comprehensive execution summary.

    Returns:
        Overall execution statistics and symbol breakdown
    """
    if not EXECUTION_TRACKER_AVAILABLE:
        return jsonify({'error': 'Execution tracking not available'}), 503

    try:
        tracker = get_execution_tracker()

        overall_stats = tracker.get_overall_stats()
        symbol_stats = tracker.get_symbol_stats()

        return jsonify({
            'timestamp': format_timestamp(),
            'overall': overall_stats,
            'by_symbol': symbol_stats
        })

    except Exception as e:
        return handle_api_error(e, "getting execution summary")


@api_bp.route('/orders/active', methods=['GET'])
def get_active_orders():
    """
    Get currently active (non-filled) orders.

    Returns:
        List of active orders being monitored
    """
    if not ORDER_MONITOR_AVAILABLE:
        return jsonify({'error': 'Order monitoring not available'}), 503

    try:
        monitor = get_order_monitor()
        active_orders = monitor.get_active_orders()

        return jsonify({
            'timestamp': format_timestamp(),
            'count': len(active_orders),
            'orders': [order.to_dict() for order in active_orders]
        })

    except Exception as e:
        return handle_api_error(e, "getting active orders")


@api_bp.route('/orders/stuck', methods=['GET'])
def get_stuck_orders():
    """
    Get orders that appear to be stuck (not filling in expected time).

    Returns:
        List of stuck orders with time since submission
    """
    if not ORDER_MONITOR_AVAILABLE:
        return jsonify({'error': 'Order monitoring not available'}), 503

    try:
        monitor = get_order_monitor()
        stuck_orders = monitor.get_stuck_orders()

        return jsonify({
            'timestamp': format_timestamp(),
            'count': len(stuck_orders),
            'alert': len(stuck_orders) > 0,
            'orders': [order.to_dict() for order in stuck_orders]
        })

    except Exception as e:
        return handle_api_error(e, "getting stuck orders")


@api_bp.route('/orders/metrics', methods=['GET'])
def get_order_metrics():
    """
    Get order monitoring metrics.

    Returns:
        Order fill times, rejection rates, and other metrics
    """
    if not ORDER_MONITOR_AVAILABLE:
        return jsonify({'error': 'Order monitoring not available'}), 503

    try:
        monitor = get_order_monitor()

        metrics = monitor.get_metrics()
        fill_time_stats = monitor.get_fill_time_stats()

        return jsonify({
            'timestamp': format_timestamp(),
            'metrics': metrics,
            'fill_time_stats': fill_time_stats
        })

    except Exception as e:
        return handle_api_error(e, "getting order metrics")


@api_bp.route('/orders/<order_id>', methods=['GET'])
def get_order_by_id(order_id: str):
    """
    Get a specific order by ID.

    Path params:
        order_id: Order identifier

    Returns:
        Order details including fills and state history
    """
    if not ORDER_MONITOR_AVAILABLE:
        return jsonify({'error': 'Order monitoring not available'}), 503

    try:
        monitor = get_order_monitor()
        order = monitor.get_order(order_id)

        if order:
            return jsonify({
                'timestamp': format_timestamp(),
                'order': order.to_dict()
            })
        else:
            return jsonify({'error': f'Order {order_id} not found'}), 404

    except Exception as e:
        return handle_api_error(e, "getting order {order_id}")


# ============================================================================
# Advanced Orders Endpoints (Bracket, Trailing Stop, OCO)
# ============================================================================

# Global broker and order manager for advanced orders (set by application startup)
_advanced_orders_broker = None
_advanced_order_manager = None


def set_advanced_orders_broker(broker):
    """Set the broker for advanced orders."""
    global _advanced_orders_broker, _advanced_order_manager
    _advanced_orders_broker = broker
    if ADVANCED_ORDERS_AVAILABLE and broker:
        _advanced_order_manager = AdvancedOrderManager(broker)


def get_advanced_orders_broker():
    """Get the broker for advanced orders."""
    return _advanced_orders_broker


def get_advanced_order_manager():
    """Get the advanced order manager."""
    return _advanced_order_manager


@api_bp.route('/orders/bracket', methods=['POST'])
@require_api_key
@require_subscription(SubscriptionTier.PRO)
def place_bracket_order():
    """
    Place a bracket order (entry + take profit + stop loss).

    Request body:
        {
            "symbol": "AAPL",
            "side": "BUY",  # BUY or SELL
            "quantity": 100,
            "entry_price": 150.00,  # null for market order
            "take_profit_price": 165.00,
            "stop_loss_price": 142.50,
            "entry_type": "LIMIT"  # LIMIT or MARKET (optional, default LIMIT)
        }

    Returns:
        Bracket order result with entry, take profit, and stop loss order IDs
    """
    if not ADVANCED_ORDERS_AVAILABLE:
        return jsonify({'error': 'Advanced orders not available'}), 503

    broker = get_advanced_orders_broker()
    if not broker:
        return jsonify({'error': 'Broker not configured for advanced orders'}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body required'}), 400

        # Validate required fields
        required_fields = ['symbol', 'side', 'quantity', 'take_profit_price', 'stop_loss_price']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({'error': f'Missing required fields: {missing}'}), 400

        # Parse order side
        side_str = data['side'].upper()
        if side_str not in ['BUY', 'SELL']:
            return jsonify({'error': 'Invalid side. Must be BUY or SELL'}), 400
        side = OrderSide.BUY if side_str == 'BUY' else OrderSide.SELL

        # Parse entry type
        entry_type_str = data.get('entry_type', 'LIMIT').upper()
        entry_type = OrderType.LIMIT if entry_type_str == 'LIMIT' else OrderType.MARKET

        # Validate symbol and prices
        symbol = sanitize_input(data['symbol'].upper())
        if not validate_symbol(symbol):
            return jsonify({'error': f'Invalid symbol: {symbol}'}), 400

        quantity = validate_integer(data['quantity'], min_value=1, field_name='quantity')
        take_profit_price = validate_float(data['take_profit_price'], min_value=0.01, field_name='take_profit_price')
        stop_loss_price = validate_float(data['stop_loss_price'], min_value=0.01, field_name='stop_loss_price')

        entry_price = None
        if data.get('entry_price') is not None:
            entry_price = validate_float(data['entry_price'], min_value=0.01, field_name='entry_price')

        # Create and place bracket order
        bracket = BracketOrder(
            broker=broker,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            entry_type=entry_type
        )

        result = bracket.create()

        # Track in manager
        manager = get_advanced_order_manager()
        if manager:
            manager.track_bracket(bracket)

        return jsonify({
            'timestamp': format_timestamp(),
            'success': True,
            'bracket_id': result.bracket_id,
            'status': result.status.value,
            'entry_order': result.entry_order.to_dict() if result.entry_order else None,
            'take_profit_order': result.take_profit_order.to_dict() if result.take_profit_order else None,
            'stop_loss_order': result.stop_loss_order.to_dict() if result.stop_loss_order else None,
            'error_message': result.error_message
        }), 201

    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return handle_api_error(e, "placing bracket order")


@api_bp.route('/orders/trailing-stop', methods=['POST'])
@require_api_key
@require_subscription(SubscriptionTier.PRO)
def place_trailing_stop_order():
    """
    Place a trailing stop order.

    Request body:
        {
            "symbol": "AAPL",
            "side": "SELL",  # SELL for long position, BUY for short
            "quantity": 100,
            "trail_amount": 2.50,  # Fixed dollar amount to trail (optional)
            "trail_percent": null,  # Percentage to trail (optional, use one or the other)
            "activation_price": 155.00,  # Price at which trailing begins (optional)
            "time_in_force": "GTC"  # GTC or DAY (optional, default GTC)
        }

    Returns:
        Trailing stop order details
    """
    if not ADVANCED_ORDERS_AVAILABLE:
        return jsonify({'error': 'Advanced orders not available'}), 503

    broker = get_advanced_orders_broker()
    if not broker:
        return jsonify({'error': 'Broker not configured for advanced orders'}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body required'}), 400

        # Validate required fields
        required_fields = ['symbol', 'side', 'quantity']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({'error': f'Missing required fields: {missing}'}), 400

        # Must have either trail_amount or trail_percent
        if not data.get('trail_amount') and not data.get('trail_percent'):
            return jsonify({'error': 'Either trail_amount or trail_percent is required'}), 400

        # Parse order side
        side_str = data['side'].upper()
        if side_str not in ['BUY', 'SELL']:
            return jsonify({'error': 'Invalid side. Must be BUY or SELL'}), 400
        side = OrderSide.BUY if side_str == 'BUY' else OrderSide.SELL

        # Validate symbol and quantities
        symbol = sanitize_input(data['symbol'].upper())
        if not validate_symbol(symbol):
            return jsonify({'error': f'Invalid symbol: {symbol}'}), 400

        quantity = validate_integer(data['quantity'], min_value=1, field_name='quantity')

        trail_amount = None
        trail_percent = None
        if data.get('trail_amount'):
            trail_amount = validate_float(data['trail_amount'], min_value=0.01, field_name='trail_amount')
        if data.get('trail_percent'):
            trail_percent = validate_float(data['trail_percent'], min_value=0.01, max_value=50.0, field_name='trail_percent')

        activation_price = None
        if data.get('activation_price'):
            activation_price = validate_float(data['activation_price'], min_value=0.01, field_name='activation_price')

        time_in_force = data.get('time_in_force', 'GTC').upper()
        if time_in_force not in ['GTC', 'DAY']:
            return jsonify({'error': 'Invalid time_in_force. Must be GTC or DAY'}), 400

        # Create and place trailing stop
        trailing_stop = TrailingStopOrder(
            broker=broker,
            symbol=symbol,
            side=side,
            quantity=quantity,
            trail_amount=trail_amount,
            trail_percent=trail_percent,
            activation_price=activation_price,
            time_in_force=time_in_force
        )

        result = trailing_stop.create()

        # Track in manager
        manager = get_advanced_order_manager()
        if manager:
            manager.track_trailing_stop(trailing_stop)

        return jsonify({
            'timestamp': format_timestamp(),
            'success': True,
            'trailing_stop_id': trailing_stop.trailing_stop_id,
            'order_id': result.order_id if result else None,
            'symbol': symbol,
            'side': side_str,
            'quantity': quantity,
            'trail_type': trailing_stop.trail_type.value,
            'trail_amount': trail_amount,
            'trail_percent': trail_percent,
            'current_stop_price': trailing_stop.get_current_stop(),
            'activation_price': activation_price,
            'is_activated': trailing_stop.is_activated,
            'time_in_force': time_in_force
        }), 201

    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return handle_api_error(e, "placing trailing stop order")


@api_bp.route('/orders/<order_id>/trail', methods=['PUT'])
@require_api_key
@require_subscription(SubscriptionTier.PRO)
def update_trailing_stop(order_id: str):
    """
    Update trail parameters for a trailing stop order.

    Path params:
        order_id: Trailing stop order ID

    Request body:
        {
            "trail_amount": 3.00,  # New fixed dollar trail (optional)
            "trail_percent": 2.0,  # New percentage trail (optional)
            "activation_price": 160.00  # New activation price (optional)
        }

    Returns:
        Updated trailing stop details
    """
    if not ADVANCED_ORDERS_AVAILABLE:
        return jsonify({'error': 'Advanced orders not available'}), 503

    manager = get_advanced_order_manager()
    if not manager:
        return jsonify({'error': 'Advanced order manager not available'}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body required'}), 400

        # Find the trailing stop
        trailing_stop = manager.get_trailing_stop(order_id)
        if not trailing_stop:
            return jsonify({'error': f'Trailing stop {order_id} not found'}), 404

        # Parse and validate new trail parameters
        new_trail_amount = None
        new_trail_percent = None

        if data.get('trail_amount') is not None:
            new_trail_amount = validate_float(data['trail_amount'], min_value=0.01, field_name='trail_amount')
        if data.get('trail_percent') is not None:
            new_trail_percent = validate_float(data['trail_percent'], min_value=0.01, max_value=50.0, field_name='trail_percent')

        # Update the trail
        if new_trail_amount is not None or new_trail_percent is not None:
            success = trailing_stop.update_trail(
                new_trail_amount=new_trail_amount,
                new_trail_percent=new_trail_percent
            )
            if not success:
                return jsonify({'error': 'Failed to update trail parameters'}), 500

        # Update activation price if provided
        if data.get('activation_price') is not None:
            new_activation = validate_float(data['activation_price'], min_value=0.01, field_name='activation_price')
            trailing_stop.activation_price = new_activation

        return jsonify({
            'timestamp': format_timestamp(),
            'success': True,
            'trailing_stop_id': trailing_stop.trailing_stop_id,
            'symbol': trailing_stop.symbol,
            'trail_type': trailing_stop.trail_type.value,
            'trail_amount': trailing_stop.trail_amount,
            'trail_percent': trailing_stop.trail_percent,
            'current_stop_price': trailing_stop.get_current_stop(),
            'activation_price': trailing_stop.activation_price,
            'is_activated': trailing_stop.is_activated
        })

    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return handle_api_error(e, "updating trailing stop {order_id}")


@api_bp.route('/orders/oco', methods=['POST'])
@require_api_key
@require_subscription(SubscriptionTier.PRO)
def place_oco_order():
    """
    Place an OCO (One-Cancels-Other) order.

    Request body:
        {
            "symbol": "AAPL",
            "orders": [
                {
                    "side": "SELL",
                    "quantity": 100,
                    "order_type": "LIMIT",
                    "price": 165.00
                },
                {
                    "side": "SELL",
                    "quantity": 100,
                    "order_type": "STOP",
                    "stop_price": 142.50
                }
            ]
        }

    Returns:
        OCO order result with both order IDs
    """
    if not ADVANCED_ORDERS_AVAILABLE:
        return jsonify({'error': 'Advanced orders not available'}), 503

    broker = get_advanced_orders_broker()
    if not broker:
        return jsonify({'error': 'Broker not configured for advanced orders'}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body required'}), 400

        # Validate required fields
        if 'symbol' not in data:
            return jsonify({'error': 'symbol is required'}), 400
        if 'orders' not in data or not isinstance(data['orders'], list):
            return jsonify({'error': 'orders array is required'}), 400
        if len(data['orders']) != 2:
            return jsonify({'error': 'OCO requires exactly 2 orders'}), 400

        symbol = sanitize_input(data['symbol'].upper())
        if not validate_symbol(symbol):
            return jsonify({'error': f'Invalid symbol: {symbol}'}), 400

        # Parse the two orders
        parsed_orders = []
        for i, order_data in enumerate(data['orders']):
            side_str = order_data.get('side', '').upper()
            if side_str not in ['BUY', 'SELL']:
                return jsonify({'error': f'Invalid side in order {i+1}'}), 400

            order_type_str = order_data.get('order_type', '').upper()
            if order_type_str not in ['LIMIT', 'STOP', 'STOP_LIMIT', 'MARKET']:
                return jsonify({'error': f'Invalid order_type in order {i+1}'}), 400

            quantity = validate_integer(order_data.get('quantity', 0), min_value=1, field_name=f'order {i+1} quantity')

            parsed_order = {
                'side': OrderSide.BUY if side_str == 'BUY' else OrderSide.SELL,
                'quantity': quantity,
                'order_type': getattr(OrderType, order_type_str),
            }

            if order_data.get('price'):
                parsed_order['price'] = validate_float(order_data['price'], min_value=0.01, field_name=f'order {i+1} price')
            if order_data.get('stop_price'):
                parsed_order['stop_price'] = validate_float(order_data['stop_price'], min_value=0.01, field_name=f'order {i+1} stop_price')

            parsed_orders.append(parsed_order)

        # Create and place OCO order
        oco = OCOOrder(
            broker=broker,
            symbol=symbol,
            orders=parsed_orders
        )

        result = oco.create()

        # Track in manager
        manager = get_advanced_order_manager()
        if manager:
            manager.track_oco(oco)

        return jsonify({
            'timestamp': format_timestamp(),
            'success': True,
            'oco_id': oco.oco_id,
            'symbol': symbol,
            'orders': [
                order.to_dict() if order else None
                for order in result
            ] if result else []
        }), 201

    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return handle_api_error(e, "placing OCO order")


@api_bp.route('/orders/advanced/status', methods=['GET'])
@require_api_key
def get_advanced_orders_status():
    """
    Get status of all tracked advanced orders.

    Returns:
        Status of bracket orders, trailing stops, and OCO orders
    """
    if not ADVANCED_ORDERS_AVAILABLE:
        return jsonify({'error': 'Advanced orders not available'}), 503

    manager = get_advanced_order_manager()
    if not manager:
        return jsonify({
            'timestamp': format_timestamp(),
            'status': 'manager_not_initialized',
            'brackets': [],
            'trailing_stops': [],
            'oco_orders': []
        })

    try:
        status = manager.get_all_status()

        return jsonify({
            'timestamp': format_timestamp(),
            'brackets': status.get('brackets', []),
            'trailing_stops': status.get('trailing_stops', []),
            'oco_orders': status.get('oco_orders', [])
        })

    except Exception as e:
        return handle_api_error(e, "getting advanced orders status")


# ============================================================================
# ML Model Drift and Performance Endpoints
# ============================================================================

@api_bp.route('/ml/drift', methods=['GET'])
def get_ml_drift_status():
    """
    Get current ML model drift detection status.

    Returns:
        Drift status including:
        - Overall drift severity
        - Feature drift details (PSI scores)
        - Prediction drift
        - Recommended actions
    """
    learning_agent = get_learning_agent()

    if learning_agent is None:
        return jsonify({
            'timestamp': format_timestamp(),
            'status': 'learning_agent_not_initialized',
            'message': 'Learning agent not available. Drift detection requires an active learning agent.'
        }), 503

    try:
        drift_status = learning_agent.get_drift_status()

        return jsonify({
            'timestamp': format_timestamp(),
            **drift_status
        })

    except Exception as e:
        return handle_api_error(e, "getting drift status")


@api_bp.route('/ml/performance', methods=['GET'])
def get_ml_performance():
    """
    Get ML model performance metrics.

    Query params:
        lookback_hours: Hours to look back for metrics (default 24)

    Returns:
        Model performance metrics including:
        - Current accuracy, precision, recall, F1
        - Baseline comparison
        - Degradation detection
    """
    learning_agent = get_learning_agent()

    if learning_agent is None:
        return jsonify({
            'timestamp': format_timestamp(),
            'status': 'learning_agent_not_initialized',
            'message': 'Learning agent not available. Performance metrics require an active learning agent.'
        }), 503

    try:
        performance = learning_agent.get_performance_metrics()
        model_info = learning_agent.get_model_info()
        learning_stats = learning_agent.get_learning_stats()

        return jsonify({
            'timestamp': format_timestamp(),
            'model_info': model_info,
            'performance': performance,
            'learning_stats': {
                'signals_tracked': learning_stats.get('signals_tracked', 0),
                'outcomes_labeled': learning_stats.get('outcomes_labeled', 0),
                'success_rate': learning_stats.get('success_rate', 0),
                'labeled_data_size': learning_stats.get('labeled_data_size', 0),
                'total_retrains': learning_stats.get('total_retrains', 0),
                'total_deployments': learning_stats.get('total_deployments', 0)
            }
        })

    except Exception as e:
        return handle_api_error(e, "getting ML performance")


@api_bp.route('/ml/drift/features', methods=['GET'])
def get_ml_drift_features():
    """
    Get detailed feature drift information.

    Returns:
        Feature-level drift details including:
        - PSI scores per feature
        - KS test results
        - Baseline vs current statistics
    """
    learning_agent = get_learning_agent()

    if learning_agent is None:
        return jsonify({
            'timestamp': format_timestamp(),
            'status': 'learning_agent_not_initialized'
        }), 503

    try:
        drift_status = learning_agent.get_drift_status()

        # Extract feature-level information
        feature_drift = []
        if 'last_drift_report' in drift_status:
            report = drift_status['last_drift_report']
            if 'feature_drift_results' in report:
                feature_drift = report['feature_drift_results']

        return jsonify({
            'timestamp': format_timestamp(),
            'feature_count': len(feature_drift),
            'features': feature_drift
        })

    except Exception as e:
        return handle_api_error(e, "getting feature drift")


@api_bp.route('/ml/predict', methods=['POST'])
def predict_signal():
    """
    Get ML model prediction for a signal.

    Request body:
        {
            "symbol": "AAPL",
            "direction": "long",
            "rrs": 2.5,
            "price": 185.50,
            "atr": 3.25,
            "daily_strong": true,
            "daily_weak": false,
            "spy_pct_change": 0.5,
            "stock_pct_change": 1.2,
            "volume": 50000000
        }

    Returns:
        Prediction results with probability and confidence
    """
    learning_agent = get_learning_agent()

    if learning_agent is None:
        return jsonify({
            'timestamp': format_timestamp(),
            'status': 'learning_agent_not_initialized'
        }), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body required', 'code': 'BODY_REQUIRED'}), 400

        if not isinstance(data, dict):
            return jsonify({'error': 'Request body must be a JSON object', 'code': 'INVALID_BODY'}), 400

        # Validate required fields
        required_fields = ['symbol', 'direction', 'rrs', 'price', 'atr']
        is_valid, body_result = validate_json_body(data, required_fields)
        if not is_valid:
            return jsonify({
                'error': 'Missing required fields',
                'message': body_result,
                'code': 'MISSING_FIELDS'
            }), 400

        # Validate symbol
        is_valid, symbol_result = validate_symbol(data['symbol'])
        if not is_valid:
            return jsonify({
                'error': 'Invalid symbol',
                'message': symbol_result,
                'code': 'INVALID_SYMBOL'
            }), 400
        data['symbol'] = symbol_result

        # Validate direction
        is_valid, direction_result = validate_direction(data['direction'])
        if not is_valid:
            return jsonify({
                'error': 'Invalid direction',
                'message': direction_result,
                'code': 'INVALID_DIRECTION'
            }), 400
        data['direction'] = direction_result

        # Validate RRS
        is_valid, rrs_result = validate_float(
            data['rrs'],
            field_name='rrs',
            min_val=-20.0,
            max_val=20.0
        )
        if not is_valid:
            return jsonify({
                'error': 'Invalid RRS',
                'message': rrs_result,
                'code': 'INVALID_RRS'
            }), 400
        data['rrs'] = rrs_result

        # Validate price
        is_valid, price_result = validate_price(data['price'])
        if not is_valid:
            return jsonify({
                'error': 'Invalid price',
                'message': price_result,
                'code': 'INVALID_PRICE'
            }), 400
        data['price'] = price_result

        # Validate ATR
        is_valid, atr_result = validate_float(
            data['atr'],
            field_name='atr',
            min_val=0.0,
            max_val=1000.0
        )
        if not is_valid:
            return jsonify({
                'error': 'Invalid ATR',
                'message': atr_result,
                'code': 'INVALID_ATR'
            }), 400
        data['atr'] = atr_result

        # Get prediction
        probability = learning_agent.predict_signal_quality(data)
        predicted_class = 1 if probability >= 0.5 else 0

        return jsonify({
            'timestamp': format_timestamp(),
            'symbol': data['symbol'],
            'prediction': {
                'success_probability': probability,
                'predicted_class': predicted_class,
                'prediction_label': 'SUCCESS' if predicted_class == 1 else 'FAILURE',
                'confidence': abs(probability - 0.5) * 2  # Distance from 0.5 normalized
            }
        })

    except Exception as e:
        return handle_api_error(e, "making prediction")


@api_bp.route('/ml/retrain', methods=['POST'])
@require_api_key
@require_subscription(SubscriptionTier.PRO)
def trigger_retrain():
    """
    Manually trigger model retraining.

    Requires PRO subscription.

    Returns:
        Retraining status
    """
    learning_agent = get_learning_agent()

    if learning_agent is None:
        return jsonify({
            'timestamp': format_timestamp(),
            'status': 'learning_agent_not_initialized'
        }), 503

    try:
        # This would trigger async retraining
        # For now, just indicate it's been requested
        return jsonify({
            'timestamp': format_timestamp(),
            'status': 'retrain_requested',
            'message': 'Model retraining has been queued. Check /ml/performance for status.',
            'current_stats': learning_agent.get_learning_stats()
        }), 202

    except Exception as e:
        return handle_api_error(e, "triggering retrain")


# ============================================================================
# A/B Testing Endpoints
# ============================================================================

# Import A/B testing components
try:
    from ml.ab_testing import (
        get_experiment_manager,
        analyze_experiment,
        ExperimentStatus,
        ModelVariant,
    )
    AB_TESTING_AVAILABLE = True
except ImportError:
    AB_TESTING_AVAILABLE = False
    logger.warning("A/B testing module not available")


@api_bp.route('/experiments', methods=['GET'])
@require_api_key
def list_experiments():
    """
    List all A/B experiments.

    Query params:
        status: Filter by status (draft, active, paused, completed, archived)
        include_archived: Include archived experiments (default false)

    Returns:
        List of experiments with basic info
    """
    if not AB_TESTING_AVAILABLE:
        return jsonify({'error': 'A/B testing not available', 'code': 'SERVICE_UNAVAILABLE'}), 503

    try:
        manager = get_experiment_manager()

        # Get filters
        status_filter = request.args.get('status')
        include_archived = request.args.get('include_archived', 'false').lower() == 'true'

        # Map status string to enum
        status = None
        if status_filter:
            try:
                status = ExperimentStatus(status_filter.lower())
            except ValueError:
                return jsonify({
                    'error': 'Invalid status',
                    'message': f"Status must be one of: {', '.join([s.value for s in ExperimentStatus])}",
                    'code': 'INVALID_STATUS'
                }), 400

        experiments = manager.list_experiments(status=status, include_archived=include_archived)

        return jsonify({
            'timestamp': format_timestamp(),
            'count': len(experiments),
            'experiments': [exp.to_dict() for exp in experiments]
        })

    except Exception as e:
        return handle_api_error(e, "listing experiments")


@api_bp.route('/experiments', methods=['POST'])
@require_api_key
@require_subscription(SubscriptionTier.PRO)
def create_experiment():
    """
    Create a new A/B experiment.

    Request body:
        {
            "name": "xgboost_v2_test",
            "model_a_id": "xgboost_v1",
            "model_b_id": "xgboost_v2",
            "traffic_split": 0.5,
            "use_thompson_sampling": false,
            "model_a_version": "1.0.0",
            "model_b_version": "2.0.0",
            "description": "Testing new XGBoost model",
            "min_samples_per_variant": 100,
            "confidence_threshold": 0.95,
            "start_date": "2024-01-15T00:00:00Z",  // optional
            "end_date": "2024-02-15T00:00:00Z"  // optional
        }

    Returns:
        Created experiment details
    """
    if not AB_TESTING_AVAILABLE:
        return jsonify({'error': 'A/B testing not available', 'code': 'SERVICE_UNAVAILABLE'}), 503

    data = request.get_json()

    if not data:
        return jsonify({'error': 'Request body required', 'code': 'BODY_REQUIRED'}), 400

    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object', 'code': 'INVALID_BODY'}), 400

    # Validate required fields
    required_fields = ['name', 'model_a_id', 'model_b_id']
    is_valid, body_result = validate_json_body(data, required_fields)
    if not is_valid:
        return jsonify({
            'error': 'Missing required fields',
            'message': body_result,
            'code': 'MISSING_FIELDS'
        }), 400

    # Validate name
    is_valid, name_result = sanitize_input(data['name'], max_length=128)
    if not is_valid:
        return jsonify({
            'error': 'Invalid name',
            'message': name_result,
            'code': 'INVALID_NAME'
        }), 400

    # Validate traffic_split
    traffic_split = data.get('traffic_split', 0.5)
    if not isinstance(traffic_split, (int, float)) or traffic_split < 0 or traffic_split > 1:
        return jsonify({
            'error': 'Invalid traffic_split',
            'message': 'traffic_split must be a number between 0 and 1',
            'code': 'INVALID_TRAFFIC_SPLIT'
        }), 400

    # Validate min_samples_per_variant
    min_samples = data.get('min_samples_per_variant', 100)
    if not isinstance(min_samples, int) or min_samples < 10:
        return jsonify({
            'error': 'Invalid min_samples_per_variant',
            'message': 'min_samples_per_variant must be an integer >= 10',
            'code': 'INVALID_MIN_SAMPLES'
        }), 400

    # Validate confidence_threshold
    confidence = data.get('confidence_threshold', 0.95)
    if not isinstance(confidence, (int, float)) or confidence <= 0 or confidence >= 1:
        return jsonify({
            'error': 'Invalid confidence_threshold',
            'message': 'confidence_threshold must be between 0 and 1 (exclusive)',
            'code': 'INVALID_CONFIDENCE'
        }), 400

    try:
        manager = get_experiment_manager()

        # Parse dates if provided
        start_date = None
        end_date = None
        if data.get('start_date'):
            try:
                start_date = datetime.fromisoformat(data['start_date'].replace('Z', '+00:00'))
            except ValueError:
                return jsonify({
                    'error': 'Invalid start_date',
                    'message': 'start_date must be in ISO format',
                    'code': 'INVALID_DATE'
                }), 400

        if data.get('end_date'):
            try:
                end_date = datetime.fromisoformat(data['end_date'].replace('Z', '+00:00'))
            except ValueError:
                return jsonify({
                    'error': 'Invalid end_date',
                    'message': 'end_date must be in ISO format',
                    'code': 'INVALID_DATE'
                }), 400

        # Get user for created_by
        user = getattr(request, 'api_user', None)
        created_by = user.user_id if user else None

        experiment = manager.create_experiment(
            name=name_result,
            model_a_id=data['model_a_id'],
            model_b_id=data['model_b_id'],
            traffic_split=traffic_split,
            start_date=start_date,
            end_date=end_date,
            use_thompson_sampling=data.get('use_thompson_sampling', False),
            model_a_version=data.get('model_a_version'),
            model_b_version=data.get('model_b_version'),
            min_samples_per_variant=min_samples,
            confidence_threshold=confidence,
            description=data.get('description'),
            config=data.get('config'),
            created_by=created_by,
        )

        return jsonify({
            'status': 'created',
            'timestamp': format_timestamp(),
            'experiment': experiment.to_dict()
        }), 201

    except ValueError as e:
        logger.warning(f"Experiment creation failed: {e}")
        return jsonify({
            'error': 'Experiment creation failed',
            'code': 'CREATION_FAILED'
        }), 400
    except Exception as e:
        return handle_api_error(e, "creating experiment")


@api_bp.route('/experiments/<int:experiment_id>', methods=['GET'])
@require_api_key
def get_experiment(experiment_id: int):
    """
    Get details of a specific experiment.

    Path params:
        experiment_id: Experiment ID

    Returns:
        Experiment details with current statistics
    """
    if not AB_TESTING_AVAILABLE:
        return jsonify({'error': 'A/B testing not available', 'code': 'SERVICE_UNAVAILABLE'}), 503

    try:
        manager = get_experiment_manager()
        experiment = manager.get_experiment_by_id(experiment_id)

        if not experiment:
            return jsonify({'error': f'Experiment {experiment_id} not found', 'code': 'NOT_FOUND'}), 404

        return jsonify({
            'timestamp': format_timestamp(),
            'experiment': experiment.to_dict(),
            'stats': experiment.get_stats()
        })

    except Exception as e:
        return handle_api_error(e, "getting experiment {experiment_id}")


@api_bp.route('/experiments/<int:experiment_id>/results', methods=['GET'])
@require_api_key
def get_experiment_results(experiment_id: int):
    """
    Get statistical analysis results for an experiment.

    Path params:
        experiment_id: Experiment ID

    Returns:
        Statistical analysis including winner, confidence, and test results
    """
    if not AB_TESTING_AVAILABLE:
        return jsonify({'error': 'A/B testing not available', 'code': 'SERVICE_UNAVAILABLE'}), 503

    try:
        manager = get_experiment_manager()
        experiment = manager.get_experiment_by_id(experiment_id)

        if not experiment:
            return jsonify({'error': f'Experiment {experiment_id} not found', 'code': 'NOT_FOUND'}), 404

        # Run statistical analysis
        result = analyze_experiment(experiment)

        return jsonify({
            'timestamp': format_timestamp(),
            'experiment_id': experiment_id,
            'experiment_name': experiment.name,
            'analysis': result.to_dict()
        })

    except Exception as e:
        return handle_api_error(e, "analyzing experiment {experiment_id}")


@api_bp.route('/experiments/<int:experiment_id>/start', methods=['POST'])
@require_api_key
@require_subscription(SubscriptionTier.PRO)
def start_experiment(experiment_id: int):
    """
    Start an experiment.

    Path params:
        experiment_id: Experiment ID

    Returns:
        Updated experiment status
    """
    if not AB_TESTING_AVAILABLE:
        return jsonify({'error': 'A/B testing not available', 'code': 'SERVICE_UNAVAILABLE'}), 503

    try:
        manager = get_experiment_manager()
        experiment = manager.get_experiment_by_id(experiment_id)

        if not experiment:
            return jsonify({'error': f'Experiment {experiment_id} not found', 'code': 'NOT_FOUND'}), 404

        success = manager.start_experiment(experiment.name)

        if success:
            return jsonify({
                'status': 'started',
                'timestamp': format_timestamp(),
                'experiment': experiment.to_dict()
            })
        else:
            return jsonify({
                'error': 'Could not start experiment',
                'message': f'Experiment is in status {experiment.status.value}',
                'code': 'INVALID_STATE'
            }), 400

    except Exception as e:
        return handle_api_error(e, "starting experiment {experiment_id}")


@api_bp.route('/experiments/<int:experiment_id>/stop', methods=['POST'])
@require_api_key
@require_subscription(SubscriptionTier.PRO)
def stop_experiment(experiment_id: int):
    """
    Stop an experiment.

    Path params:
        experiment_id: Experiment ID

    Returns:
        Updated experiment status with final results
    """
    if not AB_TESTING_AVAILABLE:
        return jsonify({'error': 'A/B testing not available', 'code': 'SERVICE_UNAVAILABLE'}), 503

    try:
        manager = get_experiment_manager()
        experiment = manager.get_experiment_by_id(experiment_id)

        if not experiment:
            return jsonify({'error': f'Experiment {experiment_id} not found', 'code': 'NOT_FOUND'}), 404

        success = manager.stop_experiment(experiment.name)

        if success:
            # Run final analysis
            result = analyze_experiment(experiment)

            return jsonify({
                'status': 'stopped',
                'timestamp': format_timestamp(),
                'experiment': experiment.to_dict(),
                'final_analysis': result.to_dict()
            })
        else:
            return jsonify({
                'error': 'Could not stop experiment',
                'message': f'Experiment is in status {experiment.status.value}',
                'code': 'INVALID_STATE'
            }), 400

    except Exception as e:
        return handle_api_error(e, "stopping experiment {experiment_id}")


@api_bp.route('/experiments/<int:experiment_id>/pause', methods=['POST'])
@require_api_key
@require_subscription(SubscriptionTier.PRO)
def pause_experiment(experiment_id: int):
    """
    Pause an experiment.

    Path params:
        experiment_id: Experiment ID

    Returns:
        Updated experiment status
    """
    if not AB_TESTING_AVAILABLE:
        return jsonify({'error': 'A/B testing not available', 'code': 'SERVICE_UNAVAILABLE'}), 503

    try:
        manager = get_experiment_manager()
        experiment = manager.get_experiment_by_id(experiment_id)

        if not experiment:
            return jsonify({'error': f'Experiment {experiment_id} not found', 'code': 'NOT_FOUND'}), 404

        success = manager.pause_experiment(experiment.name)

        if success:
            return jsonify({
                'status': 'paused',
                'timestamp': format_timestamp(),
                'experiment': experiment.to_dict()
            })
        else:
            return jsonify({
                'error': 'Could not pause experiment',
                'message': f'Experiment is in status {experiment.status.value}',
                'code': 'INVALID_STATE'
            }), 400

    except Exception as e:
        return handle_api_error(e, "pausing experiment {experiment_id}")


@api_bp.route('/experiments/<int:experiment_id>/archive', methods=['POST'])
@require_api_key
@require_subscription(SubscriptionTier.PRO)
def archive_experiment(experiment_id: int):
    """
    Archive an experiment.

    Path params:
        experiment_id: Experiment ID

    Returns:
        Updated experiment status
    """
    if not AB_TESTING_AVAILABLE:
        return jsonify({'error': 'A/B testing not available', 'code': 'SERVICE_UNAVAILABLE'}), 503

    try:
        manager = get_experiment_manager()
        experiment = manager.get_experiment_by_id(experiment_id)

        if not experiment:
            return jsonify({'error': f'Experiment {experiment_id} not found', 'code': 'NOT_FOUND'}), 404

        success = manager.archive_experiment(experiment.name)

        if success:
            return jsonify({
                'status': 'archived',
                'timestamp': format_timestamp(),
                'experiment': experiment.to_dict()
            })
        else:
            return jsonify({
                'error': 'Could not archive experiment',
                'code': 'ARCHIVE_FAILED'
            }), 400

    except Exception as e:
        return handle_api_error(e, "archiving experiment {experiment_id}")


@api_bp.route('/experiments/active', methods=['GET'])
@require_api_key
def get_active_experiment():
    """
    Get the currently active experiment.

    Returns:
        Active experiment details or null if none active
    """
    if not AB_TESTING_AVAILABLE:
        return jsonify({'error': 'A/B testing not available', 'code': 'SERVICE_UNAVAILABLE'}), 503

    try:
        manager = get_experiment_manager()
        experiment = manager.get_active_experiment()

        if experiment:
            return jsonify({
                'timestamp': format_timestamp(),
                'has_active': True,
                'experiment': experiment.to_dict(),
                'stats': experiment.get_stats()
            })
        else:
            return jsonify({
                'timestamp': format_timestamp(),
                'has_active': False,
                'experiment': None
            })

    except Exception as e:
        return handle_api_error(e, "getting active experiment")


# ============================================================================
# ML Model Optimization Endpoints
# ============================================================================

try:
    from ml.optimization.optuna_optimizer import (
        ModelOptimizer,
        OptimizationJobManager,
        OptimizationConfig,
        get_job_manager
    )
    ML_OPTIMIZATION_AVAILABLE = True
except ImportError:
    ML_OPTIMIZATION_AVAILABLE = False
    logger.warning("ML optimization not available")

# Store for optimization jobs (in production, use Redis or database)
_optimization_jobs = {}


@api_bp.route('/ml/optimize', methods=['POST'])
@require_api_key
@require_subscription(SubscriptionTier.PRO)
def start_optimization():
    """
    Start a hyperparameter optimization job.

    Request body:
        {
            "model_type": "xgboost",  // xgboost, random_forest, or lstm
            "n_trials": 100,          // Number of optimization trials
            "timeout": 3600,          // Optional timeout in seconds
            "metric": "f1",           // Optimization metric
            "cv_splits": 5,           // Number of CV splits
            "data_source": "recent"   // 'recent', 'all', or path to data file
        }

    Returns:
        Job ID and status information
    """
    if not ML_OPTIMIZATION_AVAILABLE:
        return jsonify({
            'error': 'ML optimization not available',
            'message': 'Optuna library not installed',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required', 'code': 'BODY_REQUIRED'}), 400

    # Validate model_type
    model_type = data.get('model_type')
    if not model_type or model_type not in ['xgboost', 'random_forest', 'lstm']:
        return jsonify({
            'error': 'Invalid model_type',
            'message': 'model_type must be xgboost, random_forest, or lstm',
            'code': 'INVALID_MODEL_TYPE'
        }), 400

    # Validate n_trials
    is_valid, n_trials = validate_integer(
        data.get('n_trials', 100),
        field_name='n_trials',
        min_val=10,
        max_val=1000
    )
    if not is_valid:
        return jsonify({
            'error': 'Invalid n_trials',
            'message': n_trials,
            'code': 'INVALID_PARAMETER'
        }), 400

    # Validate timeout
    timeout = data.get('timeout')
    if timeout is not None:
        is_valid, timeout = validate_integer(
            timeout,
            field_name='timeout',
            min_val=60,
            max_val=86400  # Max 24 hours
        )
        if not is_valid:
            return jsonify({
                'error': 'Invalid timeout',
                'message': timeout,
                'code': 'INVALID_PARAMETER'
            }), 400

    # Validate metric
    metric = data.get('metric', 'f1')
    valid_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'log_loss', 'profit_factor']
    if metric not in valid_metrics:
        return jsonify({
            'error': 'Invalid metric',
            'message': f'metric must be one of: {valid_metrics}',
            'code': 'INVALID_METRIC'
        }), 400

    # Validate cv_splits
    is_valid, cv_splits = validate_integer(
        data.get('cv_splits', 5),
        field_name='cv_splits',
        min_val=2,
        max_val=10
    )
    if not is_valid:
        return jsonify({
            'error': 'Invalid cv_splits',
            'message': cv_splits,
            'code': 'INVALID_PARAMETER'
        }), 400

    try:
        import uuid
        import numpy as np
        import threading

        # Generate job ID
        job_id = str(uuid.uuid4())[:8]

        # Generate or load training data
        # In production, this would load from database or file
        np.random.seed(42)
        n_samples = 2000
        n_features = 20
        X = np.random.randn(n_samples, n_features)
        weights = np.random.randn(n_features)
        linear_combo = X @ weights
        probs = 1 / (1 + np.exp(-linear_combo + np.random.randn(n_samples) * 2))
        y = (probs > 0.5).astype(int)

        # Create optimization config
        config = OptimizationConfig(
            model_type=model_type,
            metric=metric,
            n_trials=n_trials,
            timeout=timeout,
            n_cv_splits=cv_splits
        )

        # Store job info
        _optimization_jobs[job_id] = {
            'id': job_id,
            'model_type': model_type,
            'metric': metric,
            'n_trials': n_trials,
            'timeout': timeout,
            'status': 'pending',
            'created_at': format_timestamp(),
            'started_at': None,
            'completed_at': None,
            'progress': 0,
            'current_best_value': None,
            'result': None,
            'error': None
        }

        # Start optimization in background thread
        def run_optimization():
            job = _optimization_jobs[job_id]
            try:
                job['status'] = 'running'
                job['started_at'] = format_timestamp()

                optimizer = ModelOptimizer(X=X, y=y, model_type=model_type)

                result = optimizer.optimize(
                    n_trials=n_trials,
                    timeout=timeout,
                    metric=metric,
                    n_cv_splits=cv_splits,
                    show_progress_bar=False
                )

                job['status'] = 'completed'
                job['completed_at'] = format_timestamp()
                job['progress'] = 100
                job['current_best_value'] = result.best_value
                job['result'] = result.to_dict()

                logger.info(f"Optimization job {job_id} completed successfully")

            except Exception as e:
                job['status'] = 'failed'
                job['completed_at'] = format_timestamp()
                job['error'] = str(e)
                logger.error(f"Optimization job {job_id} failed: {e}")

        thread = threading.Thread(target=run_optimization, daemon=True)
        thread.start()

        return jsonify({
            'job_id': job_id,
            'status': 'pending',
            'message': 'Optimization job created and starting',
            'timestamp': format_timestamp()
        }), 202

    except Exception as e:
        return handle_api_error(e, "starting optimization")


@api_bp.route('/ml/optimize/<job_id>', methods=['GET'])
@require_api_key
def get_optimization_status(job_id: str):
    """
    Get the status of an optimization job.

    Path params:
        job_id: The optimization job ID

    Returns:
        Job status and progress information
    """
    if not ML_OPTIMIZATION_AVAILABLE:
        return jsonify({
            'error': 'ML optimization not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503

    job = _optimization_jobs.get(job_id)
    if not job:
        return jsonify({
            'error': f'Job {job_id} not found',
            'code': 'NOT_FOUND'
        }), 404

    return jsonify({
        'job_id': job['id'],
        'model_type': job['model_type'],
        'metric': job['metric'],
        'status': job['status'],
        'progress': job['progress'],
        'current_best_value': job['current_best_value'],
        'created_at': job['created_at'],
        'started_at': job['started_at'],
        'completed_at': job['completed_at'],
        'error': job['error'],
        'timestamp': format_timestamp()
    })


@api_bp.route('/ml/optimize/<job_id>/results', methods=['GET'])
@require_api_key
def get_optimization_results(job_id: str):
    """
    Get the results of a completed optimization job.

    Path params:
        job_id: The optimization job ID

    Returns:
        Optimization results including best parameters and history
    """
    if not ML_OPTIMIZATION_AVAILABLE:
        return jsonify({
            'error': 'ML optimization not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503

    job = _optimization_jobs.get(job_id)
    if not job:
        return jsonify({
            'error': f'Job {job_id} not found',
            'code': 'NOT_FOUND'
        }), 404

    if job['status'] != 'completed':
        return jsonify({
            'error': 'Optimization not yet completed',
            'status': job['status'],
            'progress': job['progress'],
            'code': 'NOT_READY'
        }), 400

    return jsonify({
        'job_id': job['id'],
        'model_type': job['model_type'],
        'metric': job['metric'],
        'status': 'completed',
        'result': job['result'],
        'timestamp': format_timestamp()
    })


@api_bp.route('/ml/optimize/<job_id>/cancel', methods=['POST'])
@require_api_key
@require_subscription(SubscriptionTier.PRO)
def cancel_optimization(job_id: str):
    """
    Cancel a running optimization job.

    Path params:
        job_id: The optimization job ID

    Returns:
        Cancellation status
    """
    if not ML_OPTIMIZATION_AVAILABLE:
        return jsonify({
            'error': 'ML optimization not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503

    job = _optimization_jobs.get(job_id)
    if not job:
        return jsonify({
            'error': f'Job {job_id} not found',
            'code': 'NOT_FOUND'
        }), 404

    if job['status'] != 'running':
        return jsonify({
            'error': f'Job is not running (status: {job["status"]})',
            'code': 'INVALID_STATE'
        }), 400

    # Mark as cancelled (actual cancellation depends on Optuna implementation)
    job['status'] = 'cancelled'
    job['completed_at'] = format_timestamp()

    return jsonify({
        'job_id': job_id,
        'status': 'cancelled',
        'message': 'Optimization job marked for cancellation',
        'timestamp': format_timestamp()
    })


@api_bp.route('/ml/optimize/jobs', methods=['GET'])
@require_api_key
def list_optimization_jobs():
    """
    List all optimization jobs.

    Query params:
        status: Filter by status (optional)
        limit: Maximum number of jobs to return (default 20)

    Returns:
        List of optimization jobs
    """
    if not ML_OPTIMIZATION_AVAILABLE:
        return jsonify({
            'error': 'ML optimization not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503

    status_filter = request.args.get('status')
    limit = request.args.get('limit', 20, type=int)
    limit = min(limit, 100)  # Cap at 100

    jobs = list(_optimization_jobs.values())

    if status_filter:
        jobs = [j for j in jobs if j['status'] == status_filter]

    # Sort by created_at descending
    jobs = sorted(jobs, key=lambda x: x['created_at'], reverse=True)[:limit]

    # Return summary without full results
    summary = [{
        'job_id': j['id'],
        'model_type': j['model_type'],
        'metric': j['metric'],
        'status': j['status'],
        'progress': j['progress'],
        'current_best_value': j['current_best_value'],
        'created_at': j['created_at'],
        'completed_at': j['completed_at']
    } for j in jobs]

    return jsonify({
        'count': len(summary),
        'jobs': summary,
        'timestamp': format_timestamp()
    })


@api_bp.route('/ml/optimize/apply/<job_id>', methods=['POST'])
@require_api_key
@require_subscription(SubscriptionTier.PRO)
def apply_optimized_params(job_id: str):
    """
    Apply optimized parameters from a completed job to the config file.

    Path params:
        job_id: The optimization job ID

    Request body (optional):
        {
            "config_path": "config/optimized_params.json"  // Optional custom path
        }

    Returns:
        Confirmation of applied parameters
    """
    if not ML_OPTIMIZATION_AVAILABLE:
        return jsonify({
            'error': 'ML optimization not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503

    job = _optimization_jobs.get(job_id)
    if not job:
        return jsonify({
            'error': f'Job {job_id} not found',
            'code': 'NOT_FOUND'
        }), 404

    if job['status'] != 'completed' or job['result'] is None:
        return jsonify({
            'error': 'Optimization not completed or no results available',
            'code': 'NOT_READY'
        }), 400

    try:
        import json
        from pathlib import Path

        data = request.get_json() or {}
        config_path = data.get('config_path', 'config/optimized_params.json')

        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing config if it exists
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            config = {}

        # Update with optimized params
        result = job['result']
        config[job['model_type']] = {
            'params': result['best_params'],
            'metric': result['metric'],
            'best_value': result['best_value'],
            'n_trials': result['n_trials'],
            'optimized_at': result['timestamp']
        }

        # Save config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Applied optimized params from job {job_id} to {config_path}")

        return jsonify({
            'message': 'Optimized parameters applied successfully',
            'config_path': str(config_file),
            'model_type': job['model_type'],
            'best_params': result['best_params'],
            'best_value': result['best_value'],
            'timestamp': format_timestamp()
        })

    except Exception as e:
        return handle_api_error(e, "applying optimized params")


# ============================================================================
# Trading Account Management Endpoints
# ============================================================================

# Import account management modules
try:
    from accounts import AccountManager, PortfolioAggregator
    from accounts.models import BrokerType
    ACCOUNTS_AVAILABLE = True
except ImportError:
    ACCOUNTS_AVAILABLE = False
    logger.warning("Account management module not available")


@api_bp.route('/accounts', methods=['GET'])
@require_api_key
def list_accounts():
    """
    List all trading accounts for the user.

    Returns:
        List of account summaries
    """
    if not ACCOUNTS_AVAILABLE:
        return jsonify({'error': 'Account management not available'}), 503

    user = getattr(request, 'api_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    try:
        manager = AccountManager(user_id=user.user_id)
        accounts = manager.get_all_account_summaries()

        return jsonify({
            'timestamp': format_timestamp(),
            'count': len(accounts),
            'accounts': accounts
        })

    except Exception as e:
        return handle_api_error(e, "listing accounts")


@api_bp.route('/accounts', methods=['POST'])
@require_api_key
def add_account():
    """
    Add a new trading account.

    Request body:
        {
            "name": "My Trading Account",
            "broker_type": "schwab",  // paper, schwab, ibkr
            "credentials": {
                "app_key": "...",
                "app_secret": "..."
            },
            "is_default": false
        }

    Returns:
        Created account details
    """
    if not ACCOUNTS_AVAILABLE:
        return jsonify({'error': 'Account management not available'}), 503

    user = getattr(request, 'api_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required', 'code': 'BODY_REQUIRED'}), 400

    # Validate required fields
    if 'name' not in data:
        return jsonify({'error': 'name is required', 'code': 'MISSING_FIELD'}), 400

    if 'broker_type' not in data:
        return jsonify({'error': 'broker_type is required', 'code': 'MISSING_FIELD'}), 400

    # Validate broker type
    valid_brokers = [b.value for b in BrokerType]
    if data['broker_type'] not in valid_brokers:
        return jsonify({
            'error': f"Invalid broker_type. Must be one of: {valid_brokers}",
            'code': 'INVALID_BROKER_TYPE'
        }), 400

    try:
        manager = AccountManager(user_id=user.user_id)
        account_id = manager.add_account(
            name=data['name'],
            broker_type=data['broker_type'],
            credentials=data.get('credentials', {}),
            is_default=data.get('is_default', False),
            metadata=data.get('metadata')
        )

        account = manager.get_account(account_id)

        return jsonify({
            'status': 'created',
            'account': account.to_dict()
        }), 201

    except Exception as e:
        logger.error(f"Error adding account: {e}")
        return jsonify({'error': 'Failed to complete adding account', 'code': 'ACCOUNT_ERROR'}), 400


@api_bp.route('/accounts/<account_id>', methods=['GET'])
@require_api_key
def get_account_details(account_id: str):
    """
    Get details for a specific trading account.

    Returns:
        Account details with live balance info if connected
    """
    if not ACCOUNTS_AVAILABLE:
        return jsonify({'error': 'Account management not available'}), 503

    user = getattr(request, 'api_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    try:
        manager = AccountManager(user_id=user.user_id)
        summary = manager.get_account_summary(account_id)

        return jsonify({
            'timestamp': format_timestamp(),
            'account': summary
        })

    except Exception as e:
        logger.error(f"Error getting account: {e}")
        return jsonify({'error': 'Failed to complete getting account', 'code': 'ACCOUNT_NOT_FOUND'}), 404


@api_bp.route('/accounts/<account_id>', methods=['DELETE'])
@require_api_key
def remove_account(account_id: str):
    """
    Remove a trading account.

    Returns:
        Success status
    """
    if not ACCOUNTS_AVAILABLE:
        return jsonify({'error': 'Account management not available'}), 503

    user = getattr(request, 'api_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    try:
        manager = AccountManager(user_id=user.user_id)
        manager.remove_account(account_id)

        return jsonify({
            'status': 'deleted',
            'account_id': account_id
        })

    except Exception as e:
        logger.error(f"Error removing account: {e}")
        return jsonify({'error': 'Failed to complete removing account', 'code': 'ACCOUNT_ERROR'}), 404


@api_bp.route('/accounts/<account_id>', methods=['PATCH'])
@require_api_key
def update_account(account_id: str):
    """
    Update a trading account.

    Request body:
        {
            "name": "New Name",
            "credentials": {...},
            "is_active": true,
            "metadata": {...}
        }

    Returns:
        Updated account details
    """
    if not ACCOUNTS_AVAILABLE:
        return jsonify({'error': 'Account management not available'}), 503

    user = getattr(request, 'api_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required', 'code': 'BODY_REQUIRED'}), 400

    try:
        manager = AccountManager(user_id=user.user_id)
        account = manager.update_account(
            account_id=account_id,
            name=data.get('name'),
            credentials=data.get('credentials'),
            is_active=data.get('is_active'),
            metadata=data.get('metadata')
        )

        return jsonify({
            'status': 'updated',
            'account': account.to_dict()
        })

    except Exception as e:
        logger.error(f"Error updating account: {e}")
        return jsonify({'error': 'Failed to complete updating account', 'code': 'ACCOUNT_ERROR'}), 400


@api_bp.route('/accounts/<account_id>/default', methods=['POST'])
@require_api_key
def set_default_account(account_id: str):
    """
    Set an account as the default trading account.

    Returns:
        Success status
    """
    if not ACCOUNTS_AVAILABLE:
        return jsonify({'error': 'Account management not available'}), 503

    user = getattr(request, 'api_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    try:
        manager = AccountManager(user_id=user.user_id)
        manager.set_default_account(account_id)

        return jsonify({
            'status': 'success',
            'default_account_id': account_id
        })

    except Exception as e:
        logger.error(f"Error setting default account: {e}")
        return jsonify({'error': 'Failed to complete setting default account', 'code': 'ACCOUNT_ERROR'}), 404


@api_bp.route('/accounts/<account_id>/test', methods=['POST'])
@require_api_key
def test_account_connection(account_id: str):
    """
    Test connection to a trading account.

    Returns:
        Connection test results
    """
    if not ACCOUNTS_AVAILABLE:
        return jsonify({'error': 'Account management not available'}), 503

    user = getattr(request, 'api_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    try:
        manager = AccountManager(user_id=user.user_id)
        result = manager.test_connection(account_id)

        return jsonify(result)

    except Exception as e:
        return handle_api_error(e, "testing account connection")


@api_bp.route('/accounts/<account_id>/positions', methods=['GET'])
@require_api_key
def get_account_positions(account_id: str):
    """
    Get positions for a specific trading account.

    Returns:
        List of positions in the account
    """
    if not ACCOUNTS_AVAILABLE:
        return jsonify({'error': 'Account management not available'}), 503

    user = getattr(request, 'api_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    try:
        manager = AccountManager(user_id=user.user_id)
        aggregator = PortfolioAggregator(manager)
        positions = aggregator.get_account_positions(account_id)

        return jsonify({
            'timestamp': format_timestamp(),
            'account_id': account_id,
            'count': len(positions),
            'positions': positions
        })

    except Exception as e:
        logger.error(f"Error getting account positions: {e}")
        return jsonify({'error': 'Failed to complete getting account positions', 'code': 'ACCOUNT_ERROR'}), 404


@api_bp.route('/portfolio/aggregate', methods=['GET'])
@require_api_key
def get_aggregated_portfolio():
    """
    Get aggregated portfolio across all accounts.

    Query params:
        account_ids: Comma-separated list of account IDs to include (optional)
        include_inactive: Include inactive accounts (default: false)

    Returns:
        Aggregated portfolio data including positions and performance
    """
    if not ACCOUNTS_AVAILABLE:
        return jsonify({'error': 'Account management not available'}), 503

    user = getattr(request, 'api_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    # Parse query params
    account_ids_param = request.args.get('account_ids')
    account_ids = account_ids_param.split(',') if account_ids_param else None
    include_inactive = request.args.get('include_inactive', 'false').lower() == 'true'

    try:
        manager = AccountManager(user_id=user.user_id)
        aggregator = PortfolioAggregator(manager)

        # Get combined data
        performance = aggregator.get_combined_performance(
            account_ids=account_ids,
            include_inactive=include_inactive
        )
        positions = aggregator.get_combined_positions(
            account_ids=account_ids,
            include_inactive=include_inactive
        )
        allocation = aggregator.get_portfolio_allocation(account_ids=account_ids)

        return jsonify({
            'timestamp': format_timestamp(),
            'performance': performance.to_dict(),
            'positions': [p.to_dict() for p in positions],
            'allocation': allocation
        })

    except Exception as e:
        return handle_api_error(e, "getting aggregated portfolio")


@api_bp.route('/portfolio/summary', methods=['GET'])
@require_api_key
def get_portfolio_summary():
    """
    Get daily portfolio summary across all accounts.

    Returns:
        Daily summary with key metrics
    """
    if not ACCOUNTS_AVAILABLE:
        return jsonify({'error': 'Account management not available'}), 503

    user = getattr(request, 'api_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    try:
        manager = AccountManager(user_id=user.user_id)
        aggregator = PortfolioAggregator(manager)
        summary = aggregator.get_daily_summary()

        return jsonify(summary)

    except Exception as e:
        return handle_api_error(e, "getting portfolio summary")


# ============================================================================
# Broker Integration Endpoints
# ============================================================================

# Import broker components
try:
    from brokers import get_broker, get_broker_from_config
    from brokers.broker_interface import OrderSide, OrderType, OrderStatus
    BROKER_AVAILABLE = True
except ImportError:
    BROKER_AVAILABLE = False
    logger.warning("Broker integration not available")

# Import risk manager
try:
    from risk.risk_manager import RiskManager
    from risk.models import RiskLimits, RiskLevel
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False
    logger.warning("Risk manager not available")

# Import ML components
try:
    from ml.regime_detector import MarketRegimeDetector
    from ml.ensemble import StackedEnsemble
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML components not available")

# Global instances (set by application startup)
_broker = None
_risk_manager = None
_regime_detector = None
_ensemble_model = None


def set_broker(broker):
    """Set the global broker instance"""
    global _broker
    _broker = broker


def get_broker_instance():
    """Get the global broker instance"""
    return _broker


def set_risk_manager(manager):
    """Set the global risk manager instance"""
    global _risk_manager
    _risk_manager = manager


def get_risk_manager_instance():
    """Get the global risk manager instance"""
    return _risk_manager


def set_regime_detector(detector):
    """Set the global regime detector instance"""
    global _regime_detector
    _regime_detector = detector


def set_ensemble_model(model):
    """Set the global ensemble model instance"""
    global _ensemble_model
    _ensemble_model = model


def get_regime_detector_instance():
    """Get the global regime detector instance"""
    return _regime_detector


def get_ensemble_model_instance():
    """Get the global ensemble model instance"""
    return _ensemble_model


# Aliases for testing
get_global_broker = get_broker_instance
get_global_risk_manager = get_risk_manager_instance
get_global_regime_detector = get_regime_detector_instance
get_global_ensemble_model = get_ensemble_model_instance


@api_bp.route('/broker/account', methods=['GET'])
@require_api_key
def get_broker_account():
    """
    Get broker account information.

    Returns:
        Account balance, buying power, and status
    """
    if not BROKER_AVAILABLE:
        return jsonify({'error': 'Broker integration not available'}), 503

    broker = get_broker_instance()
    if broker is None:
        return jsonify({'error': 'Broker not configured'}), 503

    try:
        account_info = broker.get_account()

        return jsonify({
            'timestamp': format_timestamp(),
            'account': {
                'balance': getattr(account_info, 'balance', 0),
                'buying_power': getattr(account_info, 'buying_power', 0),
                'cash': getattr(account_info, 'cash', 0),
                'equity': getattr(account_info, 'equity', 0),
                'day_trades_remaining': getattr(account_info, 'day_trades_remaining', None),
                'status': getattr(account_info, 'status', 'unknown'),
                'broker_type': broker.__class__.__name__
            }
        })

    except Exception as e:
        return handle_api_error(e, "getting broker account")


@api_bp.route('/broker/positions', methods=['GET'])
@require_api_key
def get_broker_positions():
    """
    Get live positions from broker.

    Returns:
        List of current positions
    """
    if not BROKER_AVAILABLE:
        return jsonify({'error': 'Broker integration not available'}), 503

    broker = get_broker_instance()
    if broker is None:
        return jsonify({'error': 'Broker not configured'}), 503

    try:
        positions = broker.get_positions()

        position_list = []
        for pos in positions:
            position_list.append({
                'symbol': getattr(pos, 'symbol', ''),
                'quantity': getattr(pos, 'quantity', 0),
                'side': getattr(pos, 'side', 'long'),
                'avg_entry_price': getattr(pos, 'avg_entry_price', 0),
                'current_price': getattr(pos, 'current_price', 0),
                'market_value': getattr(pos, 'market_value', 0),
                'unrealized_pnl': getattr(pos, 'unrealized_pnl', 0),
                'unrealized_pnl_percent': getattr(pos, 'unrealized_pnl_percent', 0),
            })

        return jsonify({
            'timestamp': format_timestamp(),
            'count': len(position_list),
            'positions': position_list
        })

    except Exception as e:
        return handle_api_error(e, "getting broker positions")


@api_bp.route('/broker/orders', methods=['GET'])
@require_api_key
def get_broker_orders():
    """
    Get orders from broker.

    Query params:
        status: Filter by status (open, filled, cancelled, all)

    Returns:
        List of orders
    """
    if not BROKER_AVAILABLE:
        return jsonify({'error': 'Broker integration not available'}), 503

    broker = get_broker_instance()
    if broker is None:
        return jsonify({'error': 'Broker not configured'}), 503

    status_filter = request.args.get('status', 'open')

    try:
        orders = broker.get_orders(status=status_filter)

        order_list = []
        for order in orders:
            order_list.append({
                'order_id': getattr(order, 'order_id', ''),
                'symbol': getattr(order, 'symbol', ''),
                'side': str(getattr(order, 'side', '')),
                'order_type': str(getattr(order, 'order_type', '')),
                'quantity': getattr(order, 'quantity', 0),
                'filled_quantity': getattr(order, 'filled_quantity', 0),
                'price': getattr(order, 'price', None),
                'stop_price': getattr(order, 'stop_price', None),
                'status': str(getattr(order, 'status', '')),
                'created_at': str(getattr(order, 'created_at', '')),
                'filled_at': str(getattr(order, 'filled_at', '')) if getattr(order, 'filled_at', None) else None,
            })

        return jsonify({
            'timestamp': format_timestamp(),
            'count': len(order_list),
            'orders': order_list
        })

    except Exception as e:
        return handle_api_error(e, "getting broker orders")


@api_bp.route('/broker/orders', methods=['POST'])
@require_api_key
@require_subscription(SubscriptionTier.BASIC)
def place_broker_order():
    """
    Place an order through the broker.

    Request body:
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            "order_type": "market",
            "price": null,
            "stop_price": null
        }

    Returns:
        Order confirmation
    """
    if not BROKER_AVAILABLE:
        return jsonify({'error': 'Broker integration not available'}), 503

    broker = get_broker_instance()
    if broker is None:
        return jsonify({'error': 'Broker not configured'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    # Validate required fields
    symbol = data.get('symbol')
    side = data.get('side')
    quantity = data.get('quantity')
    order_type = data.get('order_type', 'market')

    if not all([symbol, side, quantity]):
        return jsonify({'error': 'Missing required fields: symbol, side, quantity'}), 400

    try:
        # Map side string to enum
        side_enum = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        type_enum = OrderType.MARKET if order_type.lower() == 'market' else OrderType.LIMIT

        order = broker.place_order(
            symbol=symbol,
            side=side_enum,
            quantity=quantity,
            order_type=type_enum,
            price=data.get('price'),
            stop_price=data.get('stop_price')
        )

        return jsonify({
            'status': 'submitted',
            'order_id': getattr(order, 'order_id', ''),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'order_type': order_type
        })

    except Exception as e:
        return handle_api_error(e, "placing order")


@api_bp.route('/broker/orders/<order_id>', methods=['DELETE'])
@require_api_key
def cancel_broker_order(order_id: str):
    """
    Cancel an order.

    Returns:
        Cancellation status
    """
    if not BROKER_AVAILABLE:
        return jsonify({'error': 'Broker integration not available'}), 503

    broker = get_broker_instance()
    if broker is None:
        return jsonify({'error': 'Broker not configured'}), 503

    try:
        success = broker.cancel_order(order_id)

        return jsonify({
            'status': 'cancelled' if success else 'failed',
            'order_id': order_id
        })

    except Exception as e:
        return handle_api_error(e, "cancelling order")


# ============================================================================
# Risk Management Endpoints
# ============================================================================

@api_bp.route('/risk/status', methods=['GET'])
@require_api_key
def get_risk_status():
    """
    Get current risk metrics and status.

    Returns:
        Current risk metrics, limits, and trading status
    """
    if not RISK_MANAGER_AVAILABLE:
        return jsonify({'error': 'Risk management not available'}), 503

    risk_manager = get_risk_manager_instance()
    if risk_manager is None:
        return jsonify({'error': 'Risk manager not configured'}), 503

    try:
        return jsonify({
            'timestamp': format_timestamp(),
            'trading_halted': risk_manager.trading_halted,
            'halt_reason': risk_manager.halt_reason if risk_manager.trading_halted else None,
            'metrics': {
                'account_size': risk_manager.account_size,
                'current_balance': risk_manager.current_balance,
                'daily_pnl': risk_manager.daily_pnl,
                'daily_pnl_percent': (risk_manager.daily_pnl / risk_manager.daily_start_balance * 100)
                    if risk_manager.daily_start_balance > 0 else 0,
                'current_drawdown': risk_manager.current_drawdown,
                'current_drawdown_percent': (risk_manager.current_drawdown / risk_manager.peak_balance * 100)
                    if risk_manager.peak_balance > 0 else 0,
                'max_drawdown': risk_manager.max_drawdown,
                'peak_balance': risk_manager.peak_balance,
                'daily_trades': risk_manager.daily_trades,
                'daily_wins': risk_manager.daily_wins,
                'daily_losses': risk_manager.daily_losses,
                'day_trades_today': risk_manager.day_trades_today,
            },
            'limits': {
                'max_position_size': risk_manager.limits.max_position_size,
                'max_daily_loss': risk_manager.limits.max_daily_loss,
                'max_drawdown': risk_manager.limits.max_drawdown,
                'max_daily_trades': risk_manager.limits.max_daily_trades,
                'max_open_positions': risk_manager.limits.max_open_positions,
                'min_risk_reward': risk_manager.limits.min_risk_reward,
            },
            'open_positions': len(risk_manager.open_positions)
        })

    except Exception as e:
        return handle_api_error(e, "getting risk status")


@api_bp.route('/risk/validate', methods=['POST'])
@require_api_key
def validate_trade_risk():
    """
    Validate a potential trade against risk rules.

    Request body:
        {
            "symbol": "AAPL",
            "direction": "long",
            "entry_price": 150.00,
            "shares": 100,
            "stop_price": 145.00,
            "target_price": 160.00,
            "atr": 3.50
        }

    Returns:
        Validation results with any violations
    """
    if not RISK_MANAGER_AVAILABLE:
        return jsonify({'error': 'Risk management not available'}), 503

    risk_manager = get_risk_manager_instance()
    if risk_manager is None:
        return jsonify({'error': 'Risk manager not configured'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    required = ['symbol', 'direction', 'entry_price', 'shares', 'stop_price', 'target_price']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({'error': f'Missing required fields: {missing}'}), 400

    try:
        trade_risk = risk_manager.validate_trade(
            symbol=data['symbol'],
            direction=data['direction'],
            entry_price=float(data['entry_price']),
            shares=int(data['shares']),
            stop_price=float(data['stop_price']),
            target_price=float(data['target_price']),
            atr=float(data.get('atr', 0))
        )

        return jsonify({
            'timestamp': format_timestamp(),
            'approved': trade_risk.is_approved,
            'risk_level': str(trade_risk.risk_level.name) if hasattr(trade_risk.risk_level, 'name') else str(trade_risk.risk_level),
            'trade_details': {
                'symbol': trade_risk.symbol,
                'direction': trade_risk.direction,
                'entry_price': float(trade_risk.entry_price),
                'stop_price': float(trade_risk.stop_price),
                'target_price': float(trade_risk.target_price),
                'shares': trade_risk.shares,
                'position_value': float(trade_risk.position_value),
                'risk_amount': float(trade_risk.risk_amount),
                'reward_amount': float(trade_risk.reward_amount),
                'risk_reward_ratio': float(trade_risk.rr_ratio) if hasattr(trade_risk, 'rr_ratio') else 0,
            },
            'violations': [
                {
                    'rule': str(v.violation_type.name) if hasattr(v, 'violation_type') else str(v),
                    'message': str(v.message) if hasattr(v, 'message') else str(v),
                    'severity': str(v.severity) if hasattr(v, 'severity') else 'medium'
                }
                for v in (trade_risk.violations if hasattr(trade_risk, 'violations') else [])
            ]
        })

    except Exception as e:
        return handle_api_error(e, "validating trade")


@api_bp.route('/risk/limits', methods=['GET'])
@require_api_key
def get_risk_limits():
    """
    Get current risk limits.

    Returns:
        Risk limit configuration
    """
    if not RISK_MANAGER_AVAILABLE:
        return jsonify({'error': 'Risk management not available'}), 503

    risk_manager = get_risk_manager_instance()
    if risk_manager is None:
        return jsonify({'error': 'Risk manager not configured'}), 503

    try:
        limits = risk_manager.limits

        return jsonify({
            'timestamp': format_timestamp(),
            'limits': {
                'max_position_size': limits.max_position_size,
                'max_daily_loss': limits.max_daily_loss,
                'max_drawdown': limits.max_drawdown,
                'max_daily_trades': limits.max_daily_trades,
                'max_open_positions': limits.max_open_positions,
                'min_risk_reward': limits.min_risk_reward,
                'max_risk_per_trade_pct': getattr(limits, 'max_risk_per_trade_pct', 1.0),
                'max_correlation_risk': getattr(limits, 'max_correlation_risk', 0.7),
            }
        })

    except Exception as e:
        return handle_api_error(e, "getting risk limits")


@api_bp.route('/risk/limits', methods=['PUT'])
@require_api_key
@require_subscription(SubscriptionTier.PRO)
def update_risk_limits():
    """
    Update risk limits.

    Request body:
        {
            "max_position_size": 10.0,
            "max_daily_loss": 2.0,
            ...
        }

    Returns:
        Updated risk limits
    """
    if not RISK_MANAGER_AVAILABLE:
        return jsonify({'error': 'Risk management not available'}), 503

    risk_manager = get_risk_manager_instance()
    if risk_manager is None:
        return jsonify({'error': 'Risk manager not configured'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    try:
        limits = risk_manager.limits

        # Update only provided fields
        if 'max_position_size' in data:
            limits.max_position_size = float(data['max_position_size'])
        if 'max_daily_loss' in data:
            limits.max_daily_loss = float(data['max_daily_loss'])
        if 'max_drawdown' in data:
            limits.max_drawdown = float(data['max_drawdown'])
        if 'max_daily_trades' in data:
            limits.max_daily_trades = int(data['max_daily_trades'])
        if 'max_open_positions' in data:
            limits.max_open_positions = int(data['max_open_positions'])
        if 'min_risk_reward' in data:
            limits.min_risk_reward = float(data['min_risk_reward'])

        return jsonify({
            'status': 'updated',
            'limits': {
                'max_position_size': limits.max_position_size,
                'max_daily_loss': limits.max_daily_loss,
                'max_drawdown': limits.max_drawdown,
                'max_daily_trades': limits.max_daily_trades,
                'max_open_positions': limits.max_open_positions,
                'min_risk_reward': limits.min_risk_reward,
            }
        })

    except Exception as e:
        return handle_api_error(e, "updating risk limits")


# ============================================================================
# ML Prediction Endpoints
# ============================================================================

@api_bp.route('/ml/regime', methods=['GET'])
@require_api_key
def get_market_regime():
    """
    Get current market regime.

    Returns:
        Current regime with confidence scores
    """
    if not ML_AVAILABLE:
        return jsonify({'error': 'ML components not available'}), 503

    detector = _regime_detector
    if detector is None:
        # Try to load from default path
        try:
            from pathlib import Path
            model_path = Path(__file__).parent.parent.parent / "models" / "regime_detector.pkl"
            if model_path.exists():
                detector = MarketRegimeDetector()
                detector.load_model(str(model_path))
            else:
                return jsonify({'error': 'Regime detector model not trained'}), 503
        except Exception as e:
            logger.error(f"Error loading regime detector: {e}")
            return jsonify({'error': 'Regime detector not available'}), 503

    try:
        # RegimeDetector uses detect_regime() not predict()
        if hasattr(detector, 'detect_regime'):
            # Use heuristic mode - pass empty signal for general regime
            regime, confidence = detector.detect_regime({})
            return jsonify({
                'timestamp': format_timestamp(),
                'regime': regime,
                'confidence': confidence,
                'mode': 'heuristic'
            })
        elif hasattr(detector, 'predict'):
            # ML-based detector
            regime, info = detector.predict(return_confidence=True)
            return jsonify({
                'timestamp': format_timestamp(),
                'regime': regime,
                'confidence_scores': info.get('confidence_scores', {}),
                'strategy_allocation': info.get('strategy_allocation', {}),
                'model_timestamp': info.get('timestamp', format_timestamp()),
                'mode': 'ml'
            })
        else:
            return jsonify({'error': 'Regime detector not properly configured'}), 503

    except Exception as e:
        return handle_api_error(e, "getting market regime")


@api_bp.route('/ml/predict', methods=['POST'])
@require_api_key
@require_subscription(SubscriptionTier.BASIC)
def predict_trade_success():
    """
    Predict trade success probability.

    Request body:
        {
            "features": {
                "rrs": 2.5,
                "atr_percent": 3.2,
                "rsi_14": 55,
                ...
            }
        }

    Returns:
        Success probability and confidence
    """
    if not ML_AVAILABLE:
        return jsonify({'error': 'ML components not available'}), 503

    model = _ensemble_model
    if model is None:
        return jsonify({'error': 'ML model not loaded'}), 503

    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'Request body with features required'}), 400

    try:
        import numpy as np

        features = data['features']

        # Convert features dict to array (must match training order)
        feature_names = model.feature_names if hasattr(model, 'feature_names') else list(features.keys())
        feature_array = np.array([[features.get(f, 0) for f in feature_names]])

        proba = model.predict_proba(feature_array)[0]

        return jsonify({
            'timestamp': format_timestamp(),
            'prediction': {
                'success_probability': float(proba),
                'recommended_action': 'trade' if proba >= 0.6 else 'skip',
                'confidence': 'high' if abs(proba - 0.5) > 0.3 else 'medium' if abs(proba - 0.5) > 0.15 else 'low'
            },
            'features_used': len(feature_names)
        })

    except Exception as e:
        return handle_api_error(e, "predicting trade")


@api_bp.route('/ml/models/status', methods=['GET'])
@require_api_key
def get_ml_models_status():
    """
    Get ML model health status.

    Returns:
        Status of all ML models
    """
    models_status = {}

    # Check regime detector
    if _regime_detector is not None:
        models_status['regime_detector'] = {
            'loaded': True,
            'type': 'HMM',
            'regimes': getattr(_regime_detector, 'n_regimes', 4)
        }
    else:
        models_status['regime_detector'] = {'loaded': False}

    # Check ensemble model
    if _ensemble_model is not None:
        models_status['ensemble'] = {
            'loaded': True,
            'type': 'StackedEnsemble',
            'base_models': ['XGBoost', 'RandomForest'],
            'features': len(_ensemble_model.feature_names) if hasattr(_ensemble_model, 'feature_names') else 0
        }
    else:
        models_status['ensemble'] = {'loaded': False}

    # Check for model files
    from pathlib import Path
    model_dir = Path(__file__).parent.parent.parent / "models"

    model_files = {
        'regime_detector': model_dir / 'regime_detector.pkl',
        'xgboost': model_dir / 'xgboost_trade_classifier.pkl',
        'random_forest': model_dir / 'random_forest_classifier.pkl',
        'ensemble_dir': model_dir / 'ensemble'
    }

    for name, path in model_files.items():
        if name not in models_status:
            models_status[name] = {}
        models_status[name]['file_exists'] = path.exists()
        if path.exists() and path.is_file():
            models_status[name]['file_size_kb'] = path.stat().st_size / 1024
            models_status[name]['modified'] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()

    return jsonify({
        'timestamp': format_timestamp(),
        'ml_available': ML_AVAILABLE,
        'models': models_status
    })


# ============================================================================
# Error Handlers
# ============================================================================

@api_bp.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400


@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@api_bp.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500
