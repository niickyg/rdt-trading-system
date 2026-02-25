"""
Prometheus Metrics for RDT Trading System

Provides comprehensive metrics collection for monitoring:
- Trading signals and executions
- Portfolio and positions
- Scanner performance
- API request tracking
- ML model predictions
- Alert and broker operations
- WebSocket connections
"""

from typing import Optional
from functools import wraps
import time

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY
)
from loguru import logger


# =============================================================================
# METRIC REGISTRY
# =============================================================================

# Use the default registry for standard Prometheus integration
# This allows the /metrics endpoint to serve all metrics automatically
metrics_registry = REGISTRY


# =============================================================================
# TRADING SIGNAL METRICS
# =============================================================================

# Counter: Total signals generated
# Labels: direction (long/short), strength (strong/moderate/weak)
rdt_signals_generated_total = Counter(
    'rdt_signals_generated_total',
    'Total number of trading signals generated',
    ['direction', 'strength']
)

# Counter: Total trades executed
# Labels: direction (long/short), result (win/loss/breakeven/pending)
rdt_trades_executed_total = Counter(
    'rdt_trades_executed_total',
    'Total number of trades executed',
    ['direction', 'result']
)


# =============================================================================
# PORTFOLIO METRICS
# =============================================================================

# Gauge: Current number of open positions
rdt_positions_open = Gauge(
    'rdt_positions_open',
    'Current number of open positions'
)

# Gauge: Current portfolio value in dollars
rdt_portfolio_value = Gauge(
    'rdt_portfolio_value',
    'Current total portfolio value in USD'
)

# Gauge: Today's P&L in dollars
rdt_daily_pnl = Gauge(
    'rdt_daily_pnl',
    'Today\'s profit and loss in USD'
)

# Gauge: Unrealized P&L in dollars
rdt_unrealized_pnl = Gauge(
    'rdt_unrealized_pnl',
    'Total unrealized profit and loss in USD'
)

# Gauge: Total exposure in dollars
rdt_total_exposure = Gauge(
    'rdt_total_exposure',
    'Total market exposure in USD'
)


# =============================================================================
# SCANNER METRICS
# =============================================================================

# Histogram: Scanner execution duration in seconds
rdt_scanner_duration_seconds = Histogram(
    'rdt_scanner_duration_seconds',
    'Time spent scanning symbols in seconds',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
)

# Gauge: Number of symbols in watchlist being scanned
rdt_scanner_symbols_count = Gauge(
    'rdt_scanner_symbols_count',
    'Number of symbols in the current watchlist'
)

# Counter: Total scans completed
rdt_scans_completed_total = Counter(
    'rdt_scans_completed_total',
    'Total number of scan cycles completed'
)


# =============================================================================
# API METRICS
# =============================================================================

# Counter: Total API requests
# Labels: endpoint, method, status
rdt_api_requests_total = Counter(
    'rdt_api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

# Histogram: API request latency in seconds
rdt_api_request_duration_seconds = Histogram(
    'rdt_api_request_duration_seconds',
    'API request latency in seconds',
    ['endpoint', 'method'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)


# =============================================================================
# ML MODEL METRICS
# =============================================================================

# Counter: Total ML model predictions
# Labels: model (xgboost/random_forest/lstm/ensemble)
rdt_model_predictions_total = Counter(
    'rdt_model_predictions_total',
    'Total number of ML model predictions',
    ['model']
)

# Histogram: ML model prediction confidence scores
# Labels: model
rdt_model_prediction_confidence = Histogram(
    'rdt_model_prediction_confidence',
    'ML model prediction confidence scores',
    ['model'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

# Histogram: ML model inference time in seconds
rdt_model_inference_duration_seconds = Histogram(
    'rdt_model_inference_duration_seconds',
    'ML model inference time in seconds',
    ['model'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)


# =============================================================================
# ML MODEL DRIFT METRICS
# =============================================================================

# Gauge: PSI score by feature for drift detection
# Labels: model, feature
rdt_model_drift_psi = Gauge(
    'rdt_model_drift_psi',
    'Population Stability Index (PSI) score for drift detection',
    ['model', 'feature']
)

# Gauge: Rolling model accuracy
rdt_model_performance_accuracy = Gauge(
    'rdt_model_performance_accuracy',
    'Rolling model accuracy',
    ['model']
)

# Gauge: Rolling model precision
rdt_model_performance_precision = Gauge(
    'rdt_model_performance_precision',
    'Rolling model precision',
    ['model']
)

# Gauge: Rolling model recall
rdt_model_performance_recall = Gauge(
    'rdt_model_performance_recall',
    'Rolling model recall',
    ['model']
)

# Gauge: Rolling model F1 score
rdt_model_performance_f1 = Gauge(
    'rdt_model_performance_f1',
    'Rolling model F1 score',
    ['model']
)

# Gauge: Overall drift severity (0=none, 1=low, 2=medium, 3=high, 4=critical)
rdt_model_drift_severity = Gauge(
    'rdt_model_drift_severity',
    'Overall drift severity level (0=none, 1=low, 2=medium, 3=high, 4=critical)',
    ['model']
)

# Counter: Total model retraining triggers
rdt_model_retrain_triggered_total = Counter(
    'rdt_model_retrain_triggered_total',
    'Total number of model retraining triggers',
    ['model', 'reason']
)

# Gauge: Number of features with detected drift
rdt_model_features_with_drift = Gauge(
    'rdt_model_features_with_drift',
    'Number of features with detected drift',
    ['model']
)

# Gauge: Predictions since last retrain
rdt_model_predictions_since_retrain = Gauge(
    'rdt_model_predictions_since_retrain',
    'Number of predictions since last model retrain',
    ['model']
)


# =============================================================================
# ALERT METRICS
# =============================================================================

# Counter: Total alerts sent
# Labels: channel (pushover/discord/telegram/email/sms), status (success/failed)
rdt_alert_sent_total = Counter(
    'rdt_alert_sent_total',
    'Total number of alerts sent',
    ['channel', 'status']
)


# =============================================================================
# BROKER METRICS
# =============================================================================

# Counter: Total broker orders
# Labels: type (market/limit/stop/stop_limit), status (filled/rejected/cancelled/pending)
rdt_broker_orders_total = Counter(
    'rdt_broker_orders_total',
    'Total number of broker orders',
    ['type', 'status']
)

# Gauge: Current cash balance
rdt_broker_cash_balance = Gauge(
    'rdt_broker_cash_balance',
    'Current cash balance at broker'
)

# Gauge: Current buying power
rdt_broker_buying_power = Gauge(
    'rdt_broker_buying_power',
    'Current buying power at broker'
)


# =============================================================================
# WEBSOCKET METRICS
# =============================================================================

# Gauge: Active WebSocket connections
rdt_websocket_connections = Gauge(
    'rdt_websocket_connections',
    'Number of active WebSocket connections'
)

# Gauge: WebSocket clients per room
rdt_websocket_room_clients = Gauge(
    'rdt_websocket_room_clients',
    'Number of clients subscribed to each WebSocket room',
    ['room']
)

# Counter: WebSocket messages sent
rdt_websocket_messages_total = Counter(
    'rdt_websocket_messages_total',
    'Total WebSocket messages sent',
    ['room', 'event_type']
)


# =============================================================================
# ORDER EXECUTION METRICS
# =============================================================================

# Histogram: Order fill time in seconds
rdt_order_fill_time_seconds = Histogram(
    'rdt_order_fill_time_seconds',
    'Time from order submission to fill completion in seconds',
    ['symbol', 'side', 'order_type'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
)

# Histogram: Order slippage in percentage
rdt_order_slippage_pct = Histogram(
    'rdt_order_slippage_pct',
    'Order slippage as percentage of expected price',
    ['symbol', 'side'],
    buckets=[-1.0, -0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

# Gauge: Order fill rate percentage
rdt_order_fill_rate = Gauge(
    'rdt_order_fill_rate',
    'Order fill rate as percentage (filled_quantity / total_quantity)',
    ['symbol', 'side']
)

# Counter: Total executions by quality
rdt_executions_by_quality_total = Counter(
    'rdt_executions_by_quality_total',
    'Total executions categorized by execution quality',
    ['quality']  # excellent, good, fair, poor, very_poor
)

# Gauge: Average slippage for recent executions
rdt_avg_slippage_pct = Gauge(
    'rdt_avg_slippage_pct',
    'Average slippage percentage for recent executions',
    ['symbol']
)

# Counter: Stuck orders detected
rdt_stuck_orders_total = Counter(
    'rdt_stuck_orders_total',
    'Total number of stuck orders detected',
    ['symbol']
)

# Counter: Order rejections
rdt_order_rejections_total = Counter(
    'rdt_order_rejections_total',
    'Total number of rejected orders',
    ['symbol', 'reason']
)

# Gauge: Active orders count
rdt_active_orders = Gauge(
    'rdt_active_orders',
    'Number of currently active (unfilled) orders'
)


# =============================================================================
# A/B TESTING METRICS
# =============================================================================

# Counter: Total predictions by experiment and variant
rdt_ab_predictions_total = Counter(
    'rdt_ab_predictions_total',
    'Total A/B test predictions',
    ['experiment', 'variant']
)

# Counter: Total outcomes by experiment, variant, and outcome type
rdt_ab_outcomes_total = Counter(
    'rdt_ab_outcomes_total',
    'Total A/B test outcomes',
    ['experiment', 'variant', 'outcome']
)

# Gauge: Win rate by experiment and variant
rdt_ab_win_rate = Gauge(
    'rdt_ab_win_rate',
    'Win rate for A/B experiment variant',
    ['experiment', 'variant']
)

# Gauge: Average P&L by experiment and variant
rdt_ab_avg_pnl = Gauge(
    'rdt_ab_avg_pnl',
    'Average P&L for A/B experiment variant',
    ['experiment', 'variant']
)

# Gauge: Prediction accuracy by experiment and variant
rdt_ab_accuracy = Gauge(
    'rdt_ab_accuracy',
    'Prediction accuracy for A/B experiment variant',
    ['experiment', 'variant']
)

# Gauge: Thompson sampling probability (for MAB experiments)
rdt_ab_thompson_probability = Gauge(
    'rdt_ab_thompson_probability',
    'Thompson sampling selection probability',
    ['experiment', 'variant']
)

# Gauge: Active experiments count
rdt_ab_active_experiments = Gauge(
    'rdt_ab_active_experiments',
    'Number of currently active A/B experiments'
)

# Gauge: Experiment sample count
rdt_ab_sample_count = Gauge(
    'rdt_ab_sample_count',
    'Number of samples in A/B experiment variant',
    ['experiment', 'variant']
)

# Histogram: Confidence level distribution
rdt_ab_confidence = Histogram(
    'rdt_ab_confidence',
    'Statistical confidence level for A/B experiments',
    ['experiment'],
    buckets=[0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
)


# =============================================================================
# SYSTEM METRICS
# =============================================================================

# Gauge: System uptime in seconds
rdt_system_uptime_seconds = Gauge(
    'rdt_system_uptime_seconds',
    'System uptime in seconds'
)

# Gauge: Market status (1 = open, 0 = closed)
rdt_market_status = Gauge(
    'rdt_market_status',
    'Market status (1 = open, 0 = closed)'
)

# Counter: System errors
rdt_system_errors_total = Counter(
    'rdt_system_errors_total',
    'Total system errors',
    ['component', 'error_type']
)


# =============================================================================
# GPU METRICS (for ML Training)
# =============================================================================

# Gauge: GPU utilization percentage
rdt_gpu_utilization_percent = Gauge(
    'rdt_gpu_utilization_percent',
    'GPU utilization percentage',
    ['device']
)

# Gauge: GPU memory used in MB
rdt_gpu_memory_used_mb = Gauge(
    'rdt_gpu_memory_used_mb',
    'GPU memory used in MB',
    ['device']
)

# Gauge: GPU total memory in MB
rdt_gpu_memory_total_mb = Gauge(
    'rdt_gpu_memory_total_mb',
    'GPU total memory in MB',
    ['device']
)

# Gauge: GPU temperature in Celsius
rdt_gpu_temperature_celsius = Gauge(
    'rdt_gpu_temperature_celsius',
    'GPU temperature in Celsius',
    ['device']
)

# Gauge: GPU memory utilization percentage
rdt_gpu_memory_utilization_percent = Gauge(
    'rdt_gpu_memory_utilization_percent',
    'GPU memory utilization as percentage of total',
    ['device']
)

# Counter: Training epochs completed on GPU
rdt_gpu_training_epochs_total = Counter(
    'rdt_gpu_training_epochs_total',
    'Total training epochs completed on GPU',
    ['model', 'device']
)

# Histogram: GPU training batch time in seconds
rdt_gpu_training_batch_seconds = Histogram(
    'rdt_gpu_training_batch_seconds',
    'Time per training batch on GPU in seconds',
    ['model', 'device'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Gauge: GPU power usage in watts (if available)
rdt_gpu_power_watts = Gauge(
    'rdt_gpu_power_watts',
    'GPU power usage in watts',
    ['device']
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_metrics() -> bytes:
    """
    Generate Prometheus metrics output.

    Returns:
        bytes: Prometheus-formatted metrics text
    """
    return generate_latest(metrics_registry)


def get_metrics_content_type() -> str:
    """
    Get the content type for Prometheus metrics.

    Returns:
        str: Content type string
    """
    return CONTENT_TYPE_LATEST


def record_signal(direction: str, strength: str) -> None:
    """
    Record a trading signal generation.

    Args:
        direction: Signal direction ('long' or 'short')
        strength: Signal strength ('strong', 'moderate', 'weak')
    """
    rdt_signals_generated_total.labels(
        direction=direction.lower(),
        strength=strength.lower()
    ).inc()


def record_trade(direction: str, result: str) -> None:
    """
    Record a trade execution.

    Args:
        direction: Trade direction ('long' or 'short')
        result: Trade result ('win', 'loss', 'breakeven', 'pending')
    """
    rdt_trades_executed_total.labels(
        direction=direction.lower(),
        result=result.lower()
    ).inc()


def record_api_request(endpoint: str, method: str, status_code: int, duration: float) -> None:
    """
    Record an API request.

    Args:
        endpoint: API endpoint path
        method: HTTP method
        status_code: Response status code
        duration: Request duration in seconds
    """
    # Normalize endpoint by removing specific IDs
    normalized_endpoint = _normalize_endpoint(endpoint)

    rdt_api_requests_total.labels(
        endpoint=normalized_endpoint,
        method=method.upper(),
        status=str(status_code)
    ).inc()

    rdt_api_request_duration_seconds.labels(
        endpoint=normalized_endpoint,
        method=method.upper()
    ).observe(duration)


def _normalize_endpoint(endpoint: str) -> str:
    """
    Normalize API endpoint by replacing variable path segments.

    Args:
        endpoint: Raw endpoint path

    Returns:
        str: Normalized endpoint path
    """
    import re

    # Replace common variable patterns
    # /api/v1/positions/AAPL -> /api/v1/positions/{symbol}
    # /api/v1/alerts/alert_123 -> /api/v1/alerts/{id}

    endpoint = re.sub(r'/positions/[A-Z]+', '/positions/{symbol}', endpoint)
    endpoint = re.sub(r'/rrs/[A-Z]+', '/rrs/{symbol}', endpoint)
    endpoint = re.sub(r'/alerts/[a-zA-Z0-9_-]+', '/alerts/{id}', endpoint)

    return endpoint


def record_alert(channel: str, success: bool) -> None:
    """
    Record an alert send attempt.

    Args:
        channel: Alert channel name
        success: Whether the send was successful
    """
    rdt_alert_sent_total.labels(
        channel=channel.lower(),
        status='success' if success else 'failed'
    ).inc()


def record_broker_order(order_type: str, status: str) -> None:
    """
    Record a broker order.

    Args:
        order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
        status: Order status ('filled', 'rejected', 'cancelled', 'pending')
    """
    rdt_broker_orders_total.labels(
        type=order_type.lower(),
        status=status.lower()
    ).inc()


def record_model_prediction(model: str, confidence: float, duration: float) -> None:
    """
    Record an ML model prediction.

    Args:
        model: Model name
        confidence: Prediction confidence score (0-1)
        duration: Inference time in seconds
    """
    rdt_model_predictions_total.labels(model=model.lower()).inc()
    rdt_model_prediction_confidence.labels(model=model.lower()).observe(confidence)
    rdt_model_inference_duration_seconds.labels(model=model.lower()).observe(duration)


def update_drift_metrics(
    model: str,
    feature_psi_scores: dict = None,
    accuracy: float = None,
    precision: float = None,
    recall: float = None,
    f1_score: float = None,
    drift_severity: int = None,
    features_with_drift: int = None,
    predictions_since_retrain: int = None
) -> None:
    """
    Update ML model drift metrics.

    Args:
        model: Model name
        feature_psi_scores: Dictionary mapping feature names to PSI scores
        accuracy: Current rolling accuracy
        precision: Current rolling precision
        recall: Current rolling recall
        f1_score: Current rolling F1 score
        drift_severity: Severity level (0=none, 1=low, 2=medium, 3=high, 4=critical)
        features_with_drift: Number of features with detected drift
        predictions_since_retrain: Number of predictions since last retrain
    """
    model_lower = model.lower()

    if feature_psi_scores:
        for feature_name, psi_score in feature_psi_scores.items():
            rdt_model_drift_psi.labels(model=model_lower, feature=feature_name).set(psi_score)

    if accuracy is not None:
        rdt_model_performance_accuracy.labels(model=model_lower).set(accuracy)

    if precision is not None:
        rdt_model_performance_precision.labels(model=model_lower).set(precision)

    if recall is not None:
        rdt_model_performance_recall.labels(model=model_lower).set(recall)

    if f1_score is not None:
        rdt_model_performance_f1.labels(model=model_lower).set(f1_score)

    if drift_severity is not None:
        rdt_model_drift_severity.labels(model=model_lower).set(drift_severity)

    if features_with_drift is not None:
        rdt_model_features_with_drift.labels(model=model_lower).set(features_with_drift)

    if predictions_since_retrain is not None:
        rdt_model_predictions_since_retrain.labels(model=model_lower).set(predictions_since_retrain)


def record_model_retrain(model: str, reason: str) -> None:
    """
    Record a model retraining trigger.

    Args:
        model: Model name
        reason: Reason for retraining (e.g., 'drift_detected', 'scheduled', 'manual')
    """
    rdt_model_retrain_triggered_total.labels(model=model.lower(), reason=reason.lower()).inc()


def record_websocket_message(room: str, event_type: str) -> None:
    """
    Record a WebSocket message.

    Args:
        room: Room name
        event_type: Event type
    """
    rdt_websocket_messages_total.labels(
        room=room,
        event_type=event_type
    ).inc()


def update_portfolio_metrics(
    open_positions: int,
    portfolio_value: float,
    daily_pnl: float,
    unrealized_pnl: float = 0.0,
    total_exposure: float = 0.0
) -> None:
    """
    Update portfolio-related metrics.

    Args:
        open_positions: Number of open positions
        portfolio_value: Total portfolio value
        daily_pnl: Today's P&L
        unrealized_pnl: Unrealized P&L
        total_exposure: Total market exposure
    """
    rdt_positions_open.set(open_positions)
    rdt_portfolio_value.set(portfolio_value)
    rdt_daily_pnl.set(daily_pnl)
    rdt_unrealized_pnl.set(unrealized_pnl)
    rdt_total_exposure.set(total_exposure)


def update_broker_metrics(cash_balance: float, buying_power: float) -> None:
    """
    Update broker account metrics.

    Args:
        cash_balance: Current cash balance
        buying_power: Current buying power
    """
    rdt_broker_cash_balance.set(cash_balance)
    rdt_broker_buying_power.set(buying_power)


def update_websocket_metrics(total_connections: int, room_counts: dict) -> None:
    """
    Update WebSocket connection metrics.

    Args:
        total_connections: Total number of connections
        room_counts: Dict of room name to client count
    """
    rdt_websocket_connections.set(total_connections)

    for room, count in room_counts.items():
        rdt_websocket_room_clients.labels(room=room).set(count)


def record_scanner_duration(duration: float, symbols_count: int) -> None:
    """
    Record scanner execution metrics.

    Args:
        duration: Scan duration in seconds
        symbols_count: Number of symbols scanned
    """
    rdt_scanner_duration_seconds.observe(duration)
    rdt_scanner_symbols_count.set(symbols_count)
    rdt_scans_completed_total.inc()


def record_system_error(component: str, error_type: str) -> None:
    """
    Record a system error.

    Args:
        component: Component where error occurred
        error_type: Type of error
    """
    rdt_system_errors_total.labels(
        component=component,
        error_type=error_type
    ).inc()


def set_market_status(is_open: bool) -> None:
    """
    Set market status metric.

    Args:
        is_open: Whether market is open
    """
    rdt_market_status.set(1 if is_open else 0)


def record_execution_metrics(
    symbol: str,
    side: str,
    slippage_pct: float,
    fill_time_seconds: float,
    fill_rate: float,
    order_type: str = 'market'
) -> None:
    """
    Record order execution metrics.

    Args:
        symbol: Stock symbol
        side: Order side (buy, sell)
        slippage_pct: Slippage as percentage
        fill_time_seconds: Time to fill in seconds
        fill_rate: Fill rate as percentage (0-100)
        order_type: Order type (market, limit, etc.)
    """
    symbol = symbol.upper()
    side = side.lower()
    order_type = order_type.lower()

    # Record fill time
    rdt_order_fill_time_seconds.labels(
        symbol=symbol,
        side=side,
        order_type=order_type
    ).observe(fill_time_seconds)

    # Record slippage
    rdt_order_slippage_pct.labels(
        symbol=symbol,
        side=side
    ).observe(slippage_pct)

    # Record fill rate
    rdt_order_fill_rate.labels(
        symbol=symbol,
        side=side
    ).set(fill_rate)

    # Classify execution quality and increment counter
    abs_slippage = abs(slippage_pct)
    if abs_slippage < 0.05:
        quality = 'excellent'
    elif abs_slippage < 0.1:
        quality = 'good'
    elif abs_slippage < 0.25:
        quality = 'fair'
    elif abs_slippage < 0.5:
        quality = 'poor'
    else:
        quality = 'very_poor'

    rdt_executions_by_quality_total.labels(quality=quality).inc()


def record_stuck_order(symbol: str) -> None:
    """
    Record a stuck order detection.

    Args:
        symbol: Stock symbol
    """
    rdt_stuck_orders_total.labels(symbol=symbol.upper()).inc()


def record_order_rejection(symbol: str, reason: str) -> None:
    """
    Record an order rejection.

    Args:
        symbol: Stock symbol
        reason: Rejection reason
    """
    rdt_order_rejections_total.labels(
        symbol=symbol.upper(),
        reason=reason.lower()
    ).inc()


def update_active_orders_count(count: int) -> None:
    """
    Update the active orders count.

    Args:
        count: Number of active orders
    """
    rdt_active_orders.set(count)


def update_avg_slippage(symbol: str, avg_slippage_pct: float) -> None:
    """
    Update average slippage for a symbol.

    Args:
        symbol: Stock symbol
        avg_slippage_pct: Average slippage percentage
    """
    rdt_avg_slippage_pct.labels(symbol=symbol.upper()).set(avg_slippage_pct)


# =============================================================================
# A/B TESTING HELPER FUNCTIONS
# =============================================================================

def record_ab_prediction(experiment: str, variant: str) -> None:
    """
    Record an A/B test prediction.

    Args:
        experiment: Experiment name
        variant: Variant name (control/treatment)
    """
    rdt_ab_predictions_total.labels(
        experiment=experiment.lower(),
        variant=variant.lower()
    ).inc()


def record_ab_outcome(experiment: str, variant: str, outcome: str) -> None:
    """
    Record an A/B test outcome.

    Args:
        experiment: Experiment name
        variant: Variant name (control/treatment)
        outcome: Outcome type (win/loss/breakeven)
    """
    rdt_ab_outcomes_total.labels(
        experiment=experiment.lower(),
        variant=variant.lower(),
        outcome=outcome.lower()
    ).inc()


def update_ab_experiment_metrics(
    experiment: str,
    control_stats: dict = None,
    treatment_stats: dict = None,
    confidence: float = None,
    is_active: bool = True,
) -> None:
    """
    Update A/B experiment metrics.

    Args:
        experiment: Experiment name
        control_stats: Stats dict for control variant
        treatment_stats: Stats dict for treatment variant
        confidence: Statistical confidence level
        is_active: Whether experiment is active
    """
    exp_lower = experiment.lower()

    if control_stats:
        rdt_ab_win_rate.labels(experiment=exp_lower, variant='control').set(
            control_stats.get('win_rate', 0)
        )
        rdt_ab_avg_pnl.labels(experiment=exp_lower, variant='control').set(
            control_stats.get('avg_pnl', 0)
        )
        rdt_ab_accuracy.labels(experiment=exp_lower, variant='control').set(
            control_stats.get('accuracy', 0)
        )
        rdt_ab_sample_count.labels(experiment=exp_lower, variant='control').set(
            control_stats.get('total_outcomes', 0)
        )

    if treatment_stats:
        rdt_ab_win_rate.labels(experiment=exp_lower, variant='treatment').set(
            treatment_stats.get('win_rate', 0)
        )
        rdt_ab_avg_pnl.labels(experiment=exp_lower, variant='treatment').set(
            treatment_stats.get('avg_pnl', 0)
        )
        rdt_ab_accuracy.labels(experiment=exp_lower, variant='treatment').set(
            treatment_stats.get('accuracy', 0)
        )
        rdt_ab_sample_count.labels(experiment=exp_lower, variant='treatment').set(
            treatment_stats.get('total_outcomes', 0)
        )

    if confidence is not None:
        rdt_ab_confidence.labels(experiment=exp_lower).observe(confidence)


def update_ab_thompson_probabilities(
    experiment: str,
    control_prob: float,
    treatment_prob: float,
) -> None:
    """
    Update Thompson sampling probabilities for an experiment.

    Args:
        experiment: Experiment name
        control_prob: Selection probability for control
        treatment_prob: Selection probability for treatment
    """
    exp_lower = experiment.lower()
    rdt_ab_thompson_probability.labels(experiment=exp_lower, variant='control').set(control_prob)
    rdt_ab_thompson_probability.labels(experiment=exp_lower, variant='treatment').set(treatment_prob)


def set_active_experiments_count(count: int) -> None:
    """
    Set the number of active A/B experiments.

    Args:
        count: Number of active experiments
    """
    rdt_ab_active_experiments.set(count)


# =============================================================================
# DECORATORS
# =============================================================================

def track_scanner_duration(func):
    """
    Decorator to track scanner execution duration.

    Usage:
        @track_scanner_duration
        def run_scan(symbols):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            # Try to get symbols count from args or result
            symbols_count = 0
            if args and hasattr(args[0], '__len__'):
                symbols_count = len(args[0])
            elif 'symbols' in kwargs and hasattr(kwargs['symbols'], '__len__'):
                symbols_count = len(kwargs['symbols'])
            record_scanner_duration(duration, symbols_count)
    return wrapper


def track_model_prediction(model_name: str):
    """
    Decorator to track ML model predictions.

    Usage:
        @track_model_prediction('xgboost')
        def predict(self, features):
            ...

    Args:
        model_name: Name of the ML model
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Try to extract confidence from result
            confidence = 0.5
            if isinstance(result, dict):
                confidence = result.get('confidence', result.get('probability', 0.5))
            elif isinstance(result, (tuple, list)) and len(result) >= 2:
                confidence = result[1] if isinstance(result[1], (int, float)) else 0.5

            record_model_prediction(model_name, confidence, duration)
            return result
        return wrapper
    return decorator


# =============================================================================
# FLASK MIDDLEWARE
# =============================================================================

class MetricsMiddleware:
    """
    WSGI middleware for tracking request metrics.

    Usage:
        app.wsgi_app = MetricsMiddleware(app.wsgi_app)
    """

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        # Skip metrics endpoint to avoid recursion
        path = environ.get('PATH_INFO', '')
        if path == '/metrics':
            return self.app(environ, start_response)

        start_time = time.time()

        def custom_start_response(status, response_headers, exc_info=None):
            # Record metrics after getting response status
            duration = time.time() - start_time
            status_code = int(status.split(' ')[0])
            method = environ.get('REQUEST_METHOD', 'GET')

            record_api_request(path, method, status_code, duration)

            return start_response(status, response_headers, exc_info)

        return self.app(environ, custom_start_response)


def create_metrics_middleware(app):
    """
    Create Flask before/after request handlers for metrics.

    Args:
        app: Flask application instance
    """
    from flask import g, request

    @app.before_request
    def before_request():
        g.start_time = time.time()

    @app.after_request
    def after_request(response):
        # Skip metrics endpoint
        if request.path == '/metrics':
            return response

        duration = time.time() - getattr(g, 'start_time', time.time())
        record_api_request(
            request.path,
            request.method,
            response.status_code,
            duration
        )

        return response

    return app


# =============================================================================
# GPU METRICS HELPER FUNCTIONS
# =============================================================================

def update_gpu_metrics(
    device: str,
    utilization_percent: float = None,
    memory_used_mb: float = None,
    memory_total_mb: float = None,
    temperature_celsius: float = None,
    power_watts: float = None
) -> None:
    """
    Update GPU metrics for Prometheus.

    Args:
        device: GPU device identifier (e.g., 'gpu_0', 'gpu_1')
        utilization_percent: GPU utilization percentage (0-100)
        memory_used_mb: GPU memory used in MB
        memory_total_mb: GPU total memory in MB
        temperature_celsius: GPU temperature in Celsius
        power_watts: GPU power usage in watts
    """
    if utilization_percent is not None:
        rdt_gpu_utilization_percent.labels(device=device).set(utilization_percent)

    if memory_used_mb is not None:
        rdt_gpu_memory_used_mb.labels(device=device).set(memory_used_mb)

    if memory_total_mb is not None:
        rdt_gpu_memory_total_mb.labels(device=device).set(memory_total_mb)

    if temperature_celsius is not None:
        rdt_gpu_temperature_celsius.labels(device=device).set(temperature_celsius)

    if memory_used_mb is not None and memory_total_mb is not None and memory_total_mb > 0:
        utilization = (memory_used_mb / memory_total_mb) * 100
        rdt_gpu_memory_utilization_percent.labels(device=device).set(utilization)

    if power_watts is not None:
        rdt_gpu_power_watts.labels(device=device).set(power_watts)


def record_gpu_training_epoch(model: str, device: str) -> None:
    """
    Record a training epoch completed on GPU.

    Args:
        model: Model name (e.g., 'lstm', 'ensemble')
        device: GPU device identifier
    """
    rdt_gpu_training_epochs_total.labels(model=model.lower(), device=device).inc()


def record_gpu_training_batch(model: str, device: str, duration: float) -> None:
    """
    Record training batch time on GPU.

    Args:
        model: Model name
        device: GPU device identifier
        duration: Batch training time in seconds
    """
    rdt_gpu_training_batch_seconds.labels(model=model.lower(), device=device).observe(duration)


def collect_gpu_metrics_from_nvidia_smi() -> None:
    """
    Collect GPU metrics from nvidia-smi and update Prometheus metrics.

    This function queries nvidia-smi for GPU utilization, memory, temperature,
    and power metrics and updates the corresponding Prometheus gauges.
    """
    import subprocess

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    device_id = parts[0]
                    device_label = f'gpu_{device_id}'

                    try:
                        update_gpu_metrics(
                            device=device_label,
                            utilization_percent=float(parts[1]) if parts[1] else None,
                            memory_used_mb=float(parts[2]) if parts[2] else None,
                            memory_total_mb=float(parts[3]) if parts[3] else None,
                            temperature_celsius=float(parts[4]) if parts[4] else None,
                            power_watts=float(parts[5]) if len(parts) > 5 and parts[5] else None
                        )
                    except (ValueError, IndexError):
                        pass

    except FileNotFoundError:
        # nvidia-smi not available (no NVIDIA GPU or not installed)
        pass
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
    except Exception as e:
        logger.debug(f"Error collecting GPU metrics: {e}")


logger.info("Prometheus metrics module initialized")
