"""
Trading Components Initialization

Comprehensive initialization of all trading system components:

INFRASTRUCTURE:
- Configuration settings
- Database connection
- Event bus for agent communication
- Data providers with caching

TRADING CORE:
- Broker (Paper/Schwab/IBKR)
- Risk Manager with limits
- Position Sizer
- Position Tracker
- Order Monitor
- Execution Tracker

ML/ANALYSIS:
- Ensemble Model (XGBoost + RF + Meta-learner)
- Regime Detector
- Feature Engineering Pipeline
- Model Monitor
- Drift Detector

AGENTS:
- Scanner Agent
- Analyzer Agent
- Executor Agent
- Risk Agent
- Learning Agent
- Adaptive Learner
- Orchestrator

MONITORING:
- Alert Manager (multi-channel)
- Prometheus Metrics
- Trades Repository
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent


def get_model_path(model_name: str) -> Path:
    """Get path to a model file."""
    return PROJECT_ROOT / "models" / model_name


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_configuration() -> Dict[str, Any]:
    """
    Load all configuration from environment variables.

    Returns:
        Dict with all configuration values
    """
    config = {
        # Trading mode
        'paper_trading': os.environ.get('PAPER_TRADING', 'true').lower() == 'true',
        'auto_trade': os.environ.get('AUTO_TRADE', 'false').lower() == 'true',
        'account_size': float(os.environ.get('ACCOUNT_SIZE', 25000)),

        # Broker settings
        'broker_type': os.environ.get('BROKER_TYPE', 'paper'),

        # Risk limits
        'max_risk_per_trade': float(os.environ.get('MAX_RISK_PER_TRADE', 0.02)),
        'max_position_size': float(os.environ.get('MAX_POSITION_SIZE', 0.15)),
        'max_daily_loss': float(os.environ.get('MAX_DAILY_LOSS', 0.05)),
        'max_daily_trades': int(os.environ.get('MAX_DAILY_TRADES', 20)),
        'max_open_positions': int(os.environ.get('MAX_OPEN_POSITIONS', 5)),
        'min_risk_reward': float(os.environ.get('MIN_RISK_REWARD', 2.0)),
        'max_drawdown': float(os.environ.get('MAX_DRAWDOWN', 0.10)),

        # Scanner settings
        'scan_interval': int(os.environ.get('SCAN_INTERVAL', 60)),
        'rrs_threshold': float(os.environ.get('RRS_THRESHOLD', 2.0)),
        'min_volume': int(os.environ.get('MIN_VOLUME', 500000)),
        'min_price': float(os.environ.get('MIN_PRICE', 5.0)),
        'max_price': float(os.environ.get('MAX_PRICE', 500.0)),

        # ML settings
        'ml_confidence_threshold': float(os.environ.get('ML_CONFIDENCE_THRESHOLD', 0.72)),

        # Alert settings
        'alerts_enabled': os.environ.get('ALERTS_ENABLED', 'false').lower() == 'true',
        'alert_channels': os.environ.get('ALERT_CHANNELS', 'desktop').split(','),

        # Database
        'database_url': os.environ.get('DATABASE_URL', 'postgresql://rdt:rdt@localhost:5432/rdt_trading'),

        # Data provider
        'data_cache_ttl': int(os.environ.get('DATA_CACHE_TTL', 30)),
    }

    # Validate critical config values
    account_size = config.get('account_size', 25000)
    if account_size <= 0:
        raise ValueError(f"Invalid ACCOUNT_SIZE: {account_size}, must be positive")

    max_risk = config.get('max_risk_per_trade', 0.02)
    if not 0 < max_risk < 1:
        raise ValueError(f"MAX_RISK_PER_TRADE must be between 0 and 1, got {max_risk}")

    max_pos = config.get('max_position_size', 0.15)
    if not 0 < max_pos <= 1:
        raise ValueError(f"MAX_POSITION_SIZE must be between 0 and 1, got {max_pos}")

    max_daily_loss = config.get('max_daily_loss', 0.05)
    if not 0 < max_daily_loss < 1:
        raise ValueError(f"MAX_DAILY_LOSS must be between 0 and 1, got {max_daily_loss}")

    logger.debug(f"Configuration loaded: {len(config)} settings")
    return config


def get_watchlist() -> List[str]:
    """
    Load watchlist from environment or use default.

    Returns:
        List of stock symbols to scan
    """
    try:
        from config.watchlists import get_watchlist_by_name

        watchlist_name = os.environ.get('WATCHLIST', 'core')
        watchlist = get_watchlist_by_name(watchlist_name)
        logger.info(f"Loaded '{watchlist_name}' watchlist: {len(watchlist)} symbols")
        return watchlist

    except Exception as e:
        # Default watchlist
        default = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
                   'AMD', 'NFLX', 'CRM', 'SHOP', 'SQ', 'ROKU', 'SNOW', 'PLTR']
        logger.warning(f"Using default watchlist ({len(default)} symbols): {e}")
        return default


# =============================================================================
# INFRASTRUCTURE
# =============================================================================

def initialize_database():
    """
    Initialize database connection and run migrations if needed.

    Returns:
        DatabaseManager instance or None
    """
    try:
        from data.database.connection import get_db_manager, DatabaseManager

        db_manager = get_db_manager()

        # Check connection health
        if hasattr(db_manager, 'health_check'):
            if db_manager.health_check():
                logger.info("Database connection healthy")
            else:
                logger.warning("Database health check failed")

        logger.info("Database manager initialized")
        return db_manager

    except Exception as e:
        logger.critical(f"CRITICAL: Database initialization failed: {e}")
        raise  # Database is critical - don't swallow


def initialize_event_bus():
    """
    Initialize the event bus for agent communication.

    Returns:
        EventBus instance or None
    """
    try:
        from agents.events import EventBus

        event_bus = EventBus()
        logger.info("Event bus initialized")
        return event_bus

    except Exception as e:
        logger.debug(f"Event bus not available: {e}")
        return None


def initialize_data_provider(cache_ttl: int = 30):
    """
    Initialize the market data provider with caching.

    Args:
        cache_ttl: Cache time-to-live in seconds

    Returns:
        DataProvider instance or None
    """
    try:
        from shared.data_provider import DataProvider

        provider = DataProvider(cache_ttl_seconds=cache_ttl)
        logger.info(f"Data provider initialized (cache TTL: {cache_ttl}s)")
        return provider

    except Exception as e:
        logger.error(f"Failed to initialize data provider: {e}")
        return None


# =============================================================================
# BROKER
# =============================================================================

def initialize_broker(paper_trading: bool = True, initial_balance: float = 25000.0):
    """
    Initialize and connect the trading broker.

    Args:
        paper_trading: Use paper trading mode (default: True for safety)
        initial_balance: Initial balance for paper trading

    Returns:
        Connected broker instance or None
    """
    try:
        from brokers import get_broker

        if paper_trading:
            broker = get_broker("paper", initial_balance=initial_balance)
            logger.info(f"Paper broker created with ${initial_balance:,.2f} balance")
        else:
            # Load broker config from environment
            broker_type = os.environ.get('BROKER_TYPE', 'paper')
            if broker_type == 'schwab':
                broker = get_broker("schwab", from_env=True)
                logger.info("Schwab broker created from environment config")
            elif broker_type == 'ibkr':
                broker = get_broker("ibkr", from_env=True)
                logger.info("IBKR broker created from environment config")
            else:
                broker = get_broker("paper", initial_balance=initial_balance)
                logger.warning(f"Unknown broker type '{broker_type}', using paper broker")

        # Connect the broker
        broker.connect()
        logger.info(f"Broker connected: {broker.__class__.__name__}")
        return broker

    except Exception as e:
        logger.critical(f"CRITICAL: Broker initialization failed: {e}")
        raise  # Broker is critical - don't swallow


# =============================================================================
# RISK MANAGEMENT
# =============================================================================

def initialize_risk_manager(account_size: float = 25000.0, config: Dict = None):
    """
    Initialize the risk manager with limits.

    Args:
        account_size: Account size for risk calculations
        config: Optional configuration dict

    Returns:
        RiskManager instance or None
    """
    try:
        from risk import RiskManager, RiskLimits

        config = config or {}

        risk_limits = RiskLimits(
            max_risk_per_trade=config.get('max_risk_per_trade', 0.02),
            max_position_size=config.get('max_position_size', 0.15),
            max_daily_loss=config.get('max_daily_loss', 0.05),
            max_daily_trades=config.get('max_daily_trades', 20),
            max_open_positions=config.get('max_open_positions', 5),
            min_risk_reward=config.get('min_risk_reward', 2.0),
            max_drawdown=config.get('max_drawdown', 0.10),
        )

        risk_manager = RiskManager(
            account_size=account_size,
            risk_limits=risk_limits
        )

        logger.info(f"Risk manager initialized: ${account_size:,.2f} account")
        logger.debug(f"  Max risk/trade: {risk_limits.max_risk_per_trade:.1%}")
        logger.debug(f"  Max position: {risk_limits.max_position_size:.1%}")
        logger.debug(f"  Max daily loss: {risk_limits.max_daily_loss:.1%}")

        return risk_manager

    except Exception as e:
        logger.error(f"Failed to initialize risk manager: {e}")
        return None


def initialize_position_sizer(account_size: float = 25000.0, risk_per_trade: float = 0.02):
    """
    Initialize the position sizer for calculating trade sizes.

    Args:
        account_size: Account size for calculations
        risk_per_trade: Max risk per trade as decimal

    Returns:
        PositionSizer instance or None
    """
    try:
        from risk.position_sizer import PositionSizer

        # Try different initialization signatures
        try:
            sizer = PositionSizer(account_size=account_size, risk_per_trade=risk_per_trade)
        except TypeError:
            try:
                sizer = PositionSizer(account_size, risk_per_trade)
            except TypeError:
                sizer = PositionSizer()

        logger.info("Position sizer initialized")
        return sizer

    except Exception as e:
        logger.debug(f"Position sizer not available: {e}")
        return None


# =============================================================================
# TRADING TRACKING
# =============================================================================

def initialize_position_tracker():
    """
    Initialize the position tracker for real-time position management.

    Returns:
        PositionTracker instance or None
    """
    try:
        from trading.position_tracker import get_position_tracker

        tracker = get_position_tracker()
        logger.info("Position tracker initialized")
        return tracker

    except Exception as e:
        logger.debug(f"Position tracker not available: {e}")
        return None


def reconcile_positions(broker, position_tracker) -> None:
    """
    Reconcile broker positions with the internal position tracker.

    Called once during startup — after both the broker and position tracker
    are initialized — to ensure the system's view of open positions matches
    what the broker actually holds.  Any discrepancies are logged and
    corrected in-place; errors are caught and logged so they never prevent
    startup from completing.

    Args:
        broker: Connected BrokerInterface instance.
        position_tracker: PositionTracker instance.
    """
    if broker is None or position_tracker is None:
        logger.debug("Skipping position reconciliation: broker or position tracker not available")
        return

    try:
        logger.info("Starting position reconciliation between broker and tracker...")
        result = position_tracker.reconcile_positions(broker)

        if result.get("errors"):
            for err in result["errors"]:
                logger.warning(f"Reconciliation error: {err}")

    except Exception as e:
        # Reconciliation is a safety net — never let it block startup.
        logger.error(f"Position reconciliation failed (non-fatal): {e}")


def initialize_order_monitor():
    """
    Initialize the order monitor for tracking order lifecycle.

    Returns:
        OrderMonitor instance or None
    """
    try:
        from trading.order_monitor import get_order_monitor

        monitor = get_order_monitor()
        logger.info("Order monitor initialized")
        return monitor

    except Exception as e:
        logger.debug(f"Order monitor not available: {e}")
        return None


def initialize_execution_tracker():
    """
    Initialize the execution tracker for quality analysis.

    Returns:
        ExecutionTracker instance or None
    """
    try:
        from trading.execution_tracker import get_execution_tracker

        tracker = get_execution_tracker()
        logger.info("Execution tracker initialized")
        return tracker

    except Exception as e:
        logger.debug(f"Execution tracker not available: {e}")
        return None


def initialize_trades_repository():
    """
    Initialize the trades repository for database operations.

    Returns:
        TradesRepository instance or None
    """
    try:
        from data.database.trades_repository import get_trades_repository

        repo = get_trades_repository()
        logger.info("Trades repository initialized")
        return repo

    except Exception as e:
        logger.debug(f"Trades repository not available: {e}")
        return None


# =============================================================================
# ML / ANALYSIS
# =============================================================================

def initialize_regime_detector():
    """
    Initialize the market regime detector.

    Returns:
        RegimeDetector instance or None
    """
    try:
        from ml.regime_detector import RegimeDetector

        model_path = get_model_path("regime_detector.pkl")
        detector = RegimeDetector(model_path=str(model_path) if model_path.exists() else None)

        if model_path.exists():
            logger.info("Regime detector initialized with trained model")
        else:
            logger.info("Regime detector initialized (heuristic mode)")

        return detector

    except Exception as e:
        logger.error(f"Failed to initialize regime detector: {e}")
        return None


def initialize_ensemble_model():
    """
    Initialize the ML ensemble model for trade predictions.

    Returns:
        Ensemble model instance or None
    """
    try:
        import joblib
        import numpy as np

        model_dir = get_model_path("ensemble")

        if not model_dir.exists():
            logger.warning(f"Ensemble model directory not found: {model_dir}")
            return None

        class EnsembleModel:
            """Wrapper for the stacked ensemble model."""

            def __init__(self, model_dir: Path):
                self.model_dir = model_dir
                self.xgb_model = None
                self.rf_model = None
                self.meta_learner = None
                self.feature_names = None
                self.is_loaded = False
                self.load_models()

            def load_models(self):
                """Load all model components."""
                try:
                    xgb_path = self.model_dir / "xgboost_model.pkl"
                    rf_path = self.model_dir / "random_forest_model.pkl"
                    meta_path = self.model_dir / "meta_learner.pkl"
                    features_path = self.model_dir / "feature_names.json"

                    if xgb_path.exists():
                        self.xgb_model = joblib.load(xgb_path)
                        logger.debug("XGBoost model loaded")

                    if rf_path.exists():
                        self.rf_model = joblib.load(rf_path)
                        logger.debug("Random Forest model loaded")

                    if meta_path.exists():
                        self.meta_learner = joblib.load(meta_path)
                        logger.debug("Meta-learner loaded")

                    if features_path.exists():
                        import json
                        with open(features_path) as f:
                            self.feature_names = json.load(f)
                        logger.debug(f"Feature names loaded: {len(self.feature_names)} features")

                    self.is_loaded = (self.xgb_model is not None or
                                      self.rf_model is not None)

                except Exception as e:
                    logger.error(f"Error loading ensemble components: {e}")
                    self.is_loaded = False

            def predict_proba(self, features):
                """Predict success probability for given features."""
                if not self.is_loaded:
                    return [0.5]

                try:
                    if isinstance(features, dict):
                        if self.feature_names:
                            features = [features.get(f, 0) for f in self.feature_names]
                        else:
                            features = list(features.values())

                    features = np.array(features)
                    if len(features.shape) == 1:
                        features = features.reshape(1, -1)

                    predictions = []

                    if self.xgb_model is not None:
                        xgb_pred = self.xgb_model.predict_proba(features)[:, 1]
                        predictions.append(xgb_pred)

                    if self.rf_model is not None:
                        rf_pred = self.rf_model.predict_proba(features)[:, 1]
                        predictions.append(rf_pred)

                    if not predictions:
                        return [0.5]

                    if self.meta_learner is not None and len(predictions) == 2:
                        meta_features = np.column_stack(predictions)
                        final_pred = self.meta_learner.predict_proba(meta_features)[:, 1]
                        return final_pred.tolist()

                    avg_pred = np.mean(predictions, axis=0)
                    return avg_pred.tolist()

                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    return [0.5]

            def get_model_info(self):
                """Get information about loaded models."""
                return {
                    'xgboost': {
                        'loaded': self.xgb_model is not None,
                        'type': type(self.xgb_model).__name__ if self.xgb_model else None
                    },
                    'random_forest': {
                        'loaded': self.rf_model is not None,
                        'type': type(self.rf_model).__name__ if self.rf_model else None
                    },
                    'meta_learner': {
                        'loaded': self.meta_learner is not None,
                        'type': type(self.meta_learner).__name__ if self.meta_learner else None
                    },
                    'feature_count': len(self.feature_names) if self.feature_names else 0,
                    'is_loaded': self.is_loaded
                }

        ensemble = EnsembleModel(model_dir)

        if ensemble.is_loaded:
            info = ensemble.get_model_info()
            logger.info("Ensemble model initialized:")
            logger.info(f"  XGBoost: {'✓' if info['xgboost']['loaded'] else '✗'}")
            logger.info(f"  Random Forest: {'✓' if info['random_forest']['loaded'] else '✗'}")
            logger.info(f"  Meta-learner: {'✓' if info['meta_learner']['loaded'] else '✗'}")
            logger.info(f"  Features: {info['feature_count']}")
            return ensemble
        else:
            logger.warning("Ensemble model not loaded")
            return None

    except Exception as e:
        logger.error(f"Failed to initialize ensemble model: {e}")
        return None


def initialize_feature_pipeline():
    """
    Initialize the feature engineering pipeline.

    Returns:
        FeatureEngineer instance or None
    """
    try:
        from ml.feature_engineering import FeatureEngineer

        pipeline = FeatureEngineer()

        # Get feature count if available
        feature_count = getattr(pipeline, 'feature_count', None)
        if feature_count is None:
            feature_count = getattr(pipeline, 'num_features', len(getattr(pipeline, 'feature_names', [])))

        logger.info(f"Feature pipeline initialized ({feature_count} features)")
        return pipeline

    except Exception as e:
        logger.debug(f"Feature pipeline not available: {e}")
        return None


def initialize_model_monitor():
    """
    Initialize the model monitor for tracking ML performance.

    Returns:
        ModelMonitor instance or None
    """
    try:
        from ml.model_monitor import get_monitor_registry

        monitor = get_monitor_registry()
        logger.info("Model monitor initialized")
        return monitor

    except Exception as e:
        logger.debug(f"Model monitor not available: {e}")
        return None


def initialize_drift_detector():
    """
    Initialize the drift detector for model degradation detection.

    Returns:
        DriftDetector instance or None
    """
    try:
        from ml.drift_detector import DriftDetector

        detector = DriftDetector()
        logger.info("Drift detector initialized")
        return detector

    except Exception as e:
        logger.debug(f"Drift detector not available: {e}")
        return None


# =============================================================================
# AGENTS
# =============================================================================

def initialize_learning_agent():
    """
    Initialize the learning agent for trade analysis.

    Returns:
        LearningAgent instance or None
    """
    try:
        from agents.learning_agent import LearningAgent

        agent = LearningAgent()
        logger.info("Learning agent initialized")
        return agent

    except Exception as e:
        logger.debug(f"Learning agent not available: {e}")
        return None


def initialize_scanner_agent(data_provider, event_bus, watchlist: List[str], config: Dict = None):
    """
    Initialize the scanner agent for finding signals.

    Args:
        data_provider: DataProvider instance
        event_bus: EventBus instance
        watchlist: List of symbols to scan
        config: Optional configuration

    Returns:
        ScannerAgent instance or None
    """
    try:
        from agents.scanner_agent import ScannerAgent

        config = config or {}

        agent = ScannerAgent(
            watchlist=watchlist,
            data_provider=data_provider,
            scan_interval=config.get('scan_interval', 60),
            rrs_threshold=config.get('rrs_threshold', 2.0),
            event_bus=event_bus
        )
        logger.info(f"Scanner agent initialized ({len(watchlist)} symbols)")
        return agent

    except Exception as e:
        logger.debug(f"Scanner agent not available: {e}")
        return None


def initialize_analyzer_agent(risk_manager, event_bus, config: Dict = None):
    """
    Initialize the analyzer agent for signal validation.

    Args:
        risk_manager: RiskManager instance
        event_bus: EventBus instance
        config: Optional configuration

    Returns:
        AnalyzerAgent instance or None
    """
    try:
        from agents.analyzer_agent import AnalyzerAgent

        agent = AnalyzerAgent(
            risk_manager=risk_manager,
            event_bus=event_bus
        )
        logger.info("Analyzer agent initialized")
        return agent

    except Exception as e:
        logger.debug(f"Analyzer agent not available: {e}")
        return None


def initialize_executor_agent(broker, event_bus, auto_execute: bool = False):
    """
    Initialize the executor agent for trade execution.

    Args:
        broker: Broker instance
        event_bus: EventBus instance
        auto_execute: Enable automatic execution

    Returns:
        ExecutorAgent instance or None
    """
    try:
        from agents.executor_agent import ExecutorAgent

        agent = ExecutorAgent(
            broker=broker,
            auto_execute=auto_execute,
            event_bus=event_bus
        )
        logger.info(f"Executor agent initialized (auto_execute: {auto_execute})")
        return agent

    except Exception as e:
        logger.debug(f"Executor agent not available: {e}")
        return None


def initialize_risk_agent(risk_manager, event_bus):
    """
    Initialize the risk agent for real-time monitoring.

    Args:
        risk_manager: RiskManager instance
        event_bus: EventBus instance

    Returns:
        RiskAgent instance or None
    """
    try:
        from agents.risk_agent import RiskAgent

        agent = RiskAgent(
            risk_manager=risk_manager,
            event_bus=event_bus
        )
        logger.info("Risk agent initialized")
        return agent

    except Exception as e:
        logger.debug(f"Risk agent not available: {e}")
        return None


def initialize_adaptive_learner(event_bus, config: Dict = None):
    """
    Initialize the adaptive learner for parameter tuning.

    Args:
        event_bus: EventBus instance
        config: Optional configuration

    Returns:
        AdaptiveLearner instance or None
    """
    try:
        from agents.adaptive_learner import AdaptiveLearner

        config = config or {}

        agent = AdaptiveLearner(
            window_size=config.get('window_size', 20),
            adjustment_frequency=config.get('adjustment_frequency', 5),
            learning_rate=config.get('learning_rate', 0.1),
            event_bus=event_bus
        )
        logger.info("Adaptive learner initialized")
        return agent

    except Exception as e:
        logger.debug(f"Adaptive learner not available: {e}")
        return None


def initialize_orchestrator(broker, risk_manager, data_provider, watchlist: List[str],
                           config: Dict = None, auto_trade: bool = False):
    """
    Initialize the orchestrator to coordinate all agents.

    Args:
        broker: Broker instance
        risk_manager: RiskManager instance
        data_provider: DataProvider instance
        watchlist: List of symbols
        config: Optional configuration
        auto_trade: Enable automatic trading

    Returns:
        Orchestrator instance or None
    """
    try:
        from agents.orchestrator import Orchestrator

        orchestrator = Orchestrator(
            broker=broker,
            risk_manager=risk_manager,
            data_provider=data_provider,
            watchlist=watchlist,
            config=config or {},
            auto_trade=auto_trade
        )
        logger.info("Orchestrator initialized")
        return orchestrator

    except Exception as e:
        logger.debug(f"Orchestrator not available: {e}")
        return None


# =============================================================================
# ALERTS & MONITORING
# =============================================================================

def initialize_alert_manager(config: Dict = None):
    """
    Initialize the alert manager for notifications.

    Args:
        config: Alert configuration

    Returns:
        AlertManager instance or None
    """
    try:
        from alerts.alert_manager import AlertManager

        config = config or {}
        enabled_channels = config.get('alert_channels', ['desktop'])

        manager = AlertManager(enabled_channels=enabled_channels)
        logger.info(f"Alert manager initialized (channels: {enabled_channels})")
        return manager

    except Exception as e:
        logger.debug(f"Alert manager not available: {e}")
        return None


def initialize_prometheus_metrics():
    """
    Initialize Prometheus metrics for monitoring.

    Returns:
        Metrics registry or None
    """
    try:
        from monitoring.metrics import get_metrics_registry

        registry = get_metrics_registry()
        logger.info("Prometheus metrics initialized")
        return registry

    except Exception as e:
        logger.debug(f"Prometheus metrics not available: {e}")
        return None


# =============================================================================
# SCANNER
# =============================================================================

def initialize_realtime_scanner(config: Dict = None, watchlist: List[str] = None):
    """
    Initialize the real-time scanner for market scanning.

    Args:
        config: Scanner configuration
        watchlist: List of stock symbols to scan

    Returns:
        RealTimeScanner instance or None
    """
    try:
        from scanner.realtime_scanner import RealTimeScanner

        config = config or {}

        scanner = RealTimeScanner(config={
            'atr_period': config.get('atr_period', 14),
            'rrs_strong_threshold': config.get('rrs_strong_threshold', config.get('rrs_threshold', 1.75)),
            'scan_interval_seconds': config.get('scan_interval', 60),
            'min_volume': config.get('min_volume', 500000),
            'min_price': config.get('min_price', 5.0),
            'max_price': config.get('max_price', 500.0),
        })
        if watchlist:
            scanner.watchlist = watchlist
        logger.info(f"Real-time scanner initialized with {len(scanner.watchlist)} symbols")
        return scanner

    except Exception as e:
        logger.debug(f"Real-time scanner not available: {e}")
        return None


# =============================================================================
# MAIN INITIALIZATION
# =============================================================================

def initialize_all_components(include_agents: bool = False) -> Dict[str, Any]:
    """
    Initialize all trading components and register them with the API.

    Args:
        include_agents: Also initialize agent system (requires async)

    Returns:
        Dict of initialized components
    """
    logger.info("=" * 70)
    logger.info("Initializing RDT Trading System Components")
    logger.info("=" * 70)

    # Load configuration
    config = load_configuration()
    watchlist = get_watchlist()

    components = {
        # Configuration
        'config': config,
        'watchlist': watchlist,

        # Infrastructure
        'database': None,
        'event_bus': None,
        'data_provider': None,

        # Broker & Risk
        'broker': None,
        'risk_manager': None,
        'position_sizer': None,

        # Trading Tracking
        'position_tracker': None,
        'order_monitor': None,
        'execution_tracker': None,
        'trades_repository': None,

        # ML/Analysis
        'regime_detector': None,
        'ensemble_model': None,
        'feature_pipeline': None,
        'model_monitor': None,
        'drift_detector': None,

        # Agents (optional)
        'learning_agent': None,
        'scanner_agent': None,
        'analyzer_agent': None,
        'executor_agent': None,
        'risk_agent': None,
        'adaptive_learner': None,
        'orchestrator': None,

        # Monitoring
        'alert_manager': None,
        'prometheus_metrics': None,
        'realtime_scanner': None,
    }

    logger.info(f"Mode: {'Paper Trading' if config['paper_trading'] else 'LIVE TRADING'}")
    logger.info(f"Account Size: ${config['account_size']:,.2f}")
    logger.info(f"Auto Trade: {config['auto_trade']}")
    logger.info(f"Watchlist: {len(watchlist)} symbols")
    logger.info("-" * 70)

    # Initialize infrastructure
    logger.info("Initializing Infrastructure...")
    components['database'] = initialize_database()
    components['event_bus'] = initialize_event_bus()
    components['data_provider'] = initialize_data_provider(config['data_cache_ttl'])

    # Initialize broker & risk
    logger.info("Initializing Broker & Risk Management...")
    components['broker'] = initialize_broker(
        paper_trading=config['paper_trading'],
        initial_balance=config['account_size']
    )
    components['risk_manager'] = initialize_risk_manager(
        account_size=config['account_size'],
        config=config
    )
    components['position_sizer'] = initialize_position_sizer(config['account_size'])

    # Initialize trading tracking
    logger.info("Initializing Trading Tracking...")
    components['position_tracker'] = initialize_position_tracker()
    components['order_monitor'] = initialize_order_monitor()
    components['execution_tracker'] = initialize_execution_tracker()
    components['trades_repository'] = initialize_trades_repository()

    # Reconcile broker positions with internal tracker after both are ready
    logger.info("Reconciling broker and tracker positions...")
    reconcile_positions(components['broker'], components['position_tracker'])

    # Initialize ML/Analysis
    logger.info("Initializing ML & Analysis...")
    components['regime_detector'] = initialize_regime_detector()
    components['ensemble_model'] = initialize_ensemble_model()
    components['feature_pipeline'] = initialize_feature_pipeline()
    components['model_monitor'] = initialize_model_monitor()
    components['drift_detector'] = initialize_drift_detector()

    # Initialize agents (if requested)
    if include_agents:
        logger.info("Initializing Agent System...")
        components['learning_agent'] = initialize_learning_agent()

        if components['data_provider'] and components['event_bus']:
            components['scanner_agent'] = initialize_scanner_agent(
                components['data_provider'],
                components['event_bus'],
                watchlist,
                config
            )

        if components['risk_manager'] and components['event_bus']:
            components['analyzer_agent'] = initialize_analyzer_agent(
                components['risk_manager'],
                components['event_bus'],
                config
            )
            components['risk_agent'] = initialize_risk_agent(
                components['risk_manager'],
                components['event_bus']
            )

        if components['broker'] and components['event_bus']:
            components['executor_agent'] = initialize_executor_agent(
                components['broker'],
                components['event_bus'],
                config['auto_trade']
            )

        if components['event_bus']:
            components['adaptive_learner'] = initialize_adaptive_learner(
                components['event_bus'],
                config
            )

        if all([components['broker'], components['risk_manager'],
                components['data_provider']]):
            components['orchestrator'] = initialize_orchestrator(
                components['broker'],
                components['risk_manager'],
                components['data_provider'],
                watchlist,
                config,
                config['auto_trade']
            )

    # Initialize monitoring
    logger.info("Initializing Monitoring...")
    if config['alerts_enabled']:
        components['alert_manager'] = initialize_alert_manager(config)
    components['prometheus_metrics'] = initialize_prometheus_metrics()
    components['realtime_scanner'] = initialize_realtime_scanner(config, watchlist)

    # Register core components with API routes
    try:
        from api.v1.routes import (
            set_broker, set_risk_manager,
            set_regime_detector, set_ensemble_model,
            set_learning_agent
        )

        if components['broker']:
            set_broker(components['broker'])

        if components['risk_manager']:
            set_risk_manager(components['risk_manager'])

        if components['regime_detector']:
            set_regime_detector(components['regime_detector'])

        if components['ensemble_model']:
            set_ensemble_model(components['ensemble_model'])

        if components['learning_agent']:
            set_learning_agent(components['learning_agent'])

        logger.info("Components registered with API")

    except ImportError as e:
        logger.warning(f"Could not register components with API: {e}")

    # Summary
    logger.info("-" * 70)
    logger.info("Component Status Summary:")
    logger.info("-" * 70)

    categories = {
        'Infrastructure': ['database', 'event_bus', 'data_provider'],
        'Broker & Risk': ['broker', 'risk_manager', 'position_sizer'],
        'Trading Tracking': ['position_tracker', 'order_monitor', 'execution_tracker', 'trades_repository'],
        'ML & Analysis': ['regime_detector', 'ensemble_model', 'feature_pipeline', 'model_monitor', 'drift_detector'],
        'Agents': ['learning_agent', 'scanner_agent', 'analyzer_agent', 'executor_agent', 'risk_agent', 'adaptive_learner', 'orchestrator'],
        'Monitoring': ['alert_manager', 'prometheus_metrics', 'realtime_scanner'],
    }

    for category, items in categories.items():
        ready = sum(1 for item in items if components.get(item) is not None)
        total = len(items)
        logger.info(f"  {category:20s}: {ready}/{total} ready")

    logger.info("=" * 70)

    return components


def get_component_status() -> Dict[str, bool]:
    """Get status of all initialized components."""
    try:
        from api.v1.routes import (
            get_broker_instance, get_risk_manager_instance,
            get_regime_detector_instance, get_ensemble_model_instance,
            get_learning_agent
        )

        return {
            'broker': get_broker_instance() is not None,
            'risk_manager': get_risk_manager_instance() is not None,
            'regime_detector': get_regime_detector_instance() is not None,
            'ensemble_model': get_ensemble_model_instance() is not None,
            'learning_agent': get_learning_agent() is not None,
        }
    except Exception:
        return {}


async def start_trading_system(components: Dict[str, Any]) -> bool:
    """
    Start the full trading system with all agents.

    Args:
        components: Dict from initialize_all_components()

    Returns:
        True if started successfully
    """
    try:
        # Start event bus
        if components.get('event_bus'):
            await components['event_bus'].start()
            logger.info("Event bus started")

        # Start orchestrator (coordinates all agents)
        if components.get('orchestrator'):
            await components['orchestrator'].start()
            logger.info("Orchestrator and all agents started")
            return True
        else:
            logger.warning("Orchestrator not available - agents not started")
            return False

    except Exception as e:
        logger.error(f"Failed to start trading system: {e}")
        return False


async def stop_trading_system(components: Dict[str, Any]):
    """
    Gracefully stop the trading system.

    Args:
        components: Dict from initialize_all_components()
    """
    try:
        # Stop orchestrator (stops all agents)
        if components.get('orchestrator'):
            await components['orchestrator'].stop()
            logger.info("Orchestrator stopped")

        # Stop event bus
        if components.get('event_bus'):
            await components['event_bus'].stop()
            logger.info("Event bus stopped")

        logger.info("Trading system stopped gracefully")

    except Exception as e:
        logger.error(f"Error stopping trading system: {e}")
