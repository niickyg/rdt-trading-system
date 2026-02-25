"""
Learning Agent
Continuously improves the trading system through machine learning
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import json
import numpy as np
from loguru import logger
from ml.safe_model_loader import safe_load_model, safe_save_model, ModelSecurityError

from agents.base import BaseAgent
from agents.events import Event, EventType
from data.database.connection import get_db_manager
from data.database.models import Signal, Trade, TradeStatus
from ml import StackedEnsemble
from ml.model_monitor import ModelMonitor, AlertConfig, MonitoringConfig
from ml.drift_detector import DriftThresholds, DriftSeverity

try:
    from monitoring.metrics import (
        update_drift_metrics,
        record_model_retrain,
        record_model_prediction
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


@dataclass
class SignalFeatures:
    """Features extracted from a trading signal"""
    # Signal data
    symbol: str
    direction: str
    rrs: float
    price: float
    atr: float
    atr_percent: float

    # Daily chart indicators
    daily_strong: bool
    daily_weak: bool
    ema3: Optional[float] = None
    ema8: Optional[float] = None

    # Market context
    spy_pct_change: Optional[float] = None
    stock_pct_change: Optional[float] = None
    volume: Optional[int] = None

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for ML model"""
        return np.array([
            self.rrs,
            self.atr_percent,
            1.0 if self.daily_strong else 0.0,
            1.0 if self.daily_weak else 0.0,
            1.0 if self.direction == "long" else -1.0,
            self.spy_pct_change or 0.0,
            self.stock_pct_change or 0.0,
            np.log(self.volume) if self.volume and self.volume > 0 else 0.0,
        ])

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "rrs": self.rrs,
            "price": self.price,
            "atr": self.atr,
            "atr_percent": self.atr_percent,
            "daily_strong": self.daily_strong,
            "daily_weak": self.daily_weak,
            "ema3": self.ema3,
            "ema8": self.ema8,
            "spy_pct_change": self.spy_pct_change,
            "stock_pct_change": self.stock_pct_change,
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class LabeledSignal:
    """Signal with outcome label"""
    features: SignalFeatures
    label: int  # 1 = success (reached 2R), 0 = failure
    position_id: Optional[int] = None
    outcome_timestamp: Optional[datetime] = None
    realized_pnl: Optional[float] = None
    r_multiple: Optional[float] = None  # Actual R achieved


@dataclass
class ModelMetrics:
    """Metrics for model performance"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    training_samples: int = 0
    validation_samples: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class SimpleTradingModel:
    """
    Simple trading signal classifier

    Uses a basic scoring model that can be easily retrained.
    In production, this would be replaced with more sophisticated models
    (e.g., XGBoost, LightGBM, Neural Networks)
    """

    def __init__(self):
        # Feature weights (learned from data)
        self.weights = np.array([
            1.0,   # RRS strength
            -0.5,  # ATR percent (lower is better)
            0.8,   # Daily strong
            0.8,   # Daily weak
            0.0,   # Direction (neutral)
            0.3,   # SPY momentum
            0.5,   # Stock momentum
            0.2,   # Volume (log)
        ])
        self.threshold = 0.5
        self.version = "1.0.0"
        self.trained_at: Optional[datetime] = None
        self.training_samples = 0

    def predict(self, features: np.ndarray) -> float:
        """
        Predict probability of success

        Args:
            features: Feature vector

        Returns:
            Probability between 0 and 1
        """
        score = np.dot(features, self.weights)
        # Sigmoid activation
        probability = 1 / (1 + np.exp(-score))
        return probability

    def predict_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """Predict for multiple samples"""
        scores = np.dot(features_batch, self.weights)
        probabilities = 1 / (1 + np.exp(-scores))
        return probabilities

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the model

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            Training metrics
        """
        if len(X) < 10:
            logger.warning("Insufficient training samples")
            return {"error": "Insufficient samples"}

        # Simple gradient descent for logistic regression
        learning_rate = 0.01
        epochs = 100

        for epoch in range(epochs):
            # Predictions
            predictions = self.predict_batch(X)

            # Gradient
            error = predictions - y
            gradient = np.dot(X.T, error) / len(X)

            # Update weights
            self.weights -= learning_rate * gradient

        self.trained_at = datetime.now()
        self.training_samples = len(X)

        # Calculate metrics
        final_predictions = self.predict_batch(X)
        final_labels = (final_predictions >= self.threshold).astype(int)

        accuracy = np.mean(final_labels == y)

        return {
            "accuracy": accuracy,
            "samples": len(X),
            "weights": self.weights.tolist()
        }

    def save(self, filepath: Path):
        """Save model to disk"""
        safe_save_model(self, str(filepath))

    @staticmethod
    def load(filepath: Path) -> 'SimpleTradingModel':
        """Load model from disk"""
        return safe_load_model(str(filepath), allow_unverified=False)


class LearningAgent(BaseAgent):
    """
    Learning agent that continuously improves the trading system

    Responsibilities:
    - Track all trade outcomes
    - Label signals as success/failure
    - Accumulate labeled training data
    - Retrain model periodically
    - Validate and deploy improved models
    - Publish model update events
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        retrain_threshold: int = 100,  # Retrain after N new trades
        retrain_interval_days: int = 7,  # Or weekly
        lookback_months: int = 6,  # Use last 6 months of data
        min_improvement: float = 0.02,  # Require 2% improvement to deploy
        **kwargs
    ):
        super().__init__(name="LearningAgent", **kwargs)

        # Configuration
        self.model_dir = model_dir or Path.home() / ".rdt-trading" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.retrain_threshold = retrain_threshold
        self.retrain_interval_days = retrain_interval_days
        self.lookback_months = lookback_months
        self.min_improvement = min_improvement

        # Database
        self.db_manager = get_db_manager()

        # Model - use sophisticated StackedEnsemble with XGBoost and Random Forest
        self.current_model: Optional[StackedEnsemble] = None
        self.current_model_metrics: Optional[ModelMetrics] = None

        # Model monitoring for drift detection
        self.model_monitor: Optional[ModelMonitor] = None
        self._drift_check_interval = 100  # Check drift every N predictions
        self._predictions_since_drift_check = 0

        # Data storage
        self.pending_signals: Dict[str, SignalFeatures] = {}  # symbol -> features
        self.labeled_data: List[LabeledSignal] = []
        self.new_labels_count = 0

        # Training tracking
        self.last_retrain_time: Optional[datetime] = None
        self.total_retrains = 0
        self.total_deployments = 0

        # Metrics
        self.signals_tracked = 0
        self.outcomes_labeled = 0
        self.success_rate = 0.0

    async def initialize(self):
        """Initialize learning agent"""
        logger.info("Initializing Learning Agent...")

        # Load existing model if available
        self._load_current_model()

        # Load historical data from database
        await self._load_historical_data()

        # Initialize model monitor for drift detection
        self._initialize_model_monitor()

        # Initialize metrics
        self.metrics.custom_metrics.update({
            "signals_tracked": 0,
            "outcomes_labeled": 0,
            "success_rate": 0.0,
            "labeled_data_size": len(self.labeled_data),
            "total_retrains": 0,
            "total_deployments": 0,
            "model_version": self.current_model.version if self.current_model else "none",
            "drift_severity": "none",
            "drift_check_count": 0
        })

        logger.info(
            f"Learning Agent initialized - "
            f"Loaded {len(self.labeled_data)} historical labels"
        )

    def _initialize_model_monitor(self):
        """Initialize model monitoring for drift detection"""
        try:
            # Configure drift thresholds
            drift_thresholds = DriftThresholds(
                psi_low=0.1,
                psi_medium=0.2,
                psi_high=0.25,
                accuracy_degradation_warning=0.05,
                accuracy_degradation_critical=0.10,
                min_samples_feature_drift=100,
                min_samples_performance=30
            )

            # Configure alerts
            alert_config = AlertConfig(
                enabled=True,
                alert_on_high=True,
                alert_on_critical=True,
                cooldown_minutes=60,
                alert_callback=self._on_drift_alert
            )

            # Configure monitoring
            monitoring_config = MonitoringConfig(
                check_interval_minutes=30,
                lookback_hours=24,
                min_predictions_for_check=50,
                enable_background_monitoring=False,  # Manual checks
                persistence_path=self.model_dir / "monitoring"
            )

            # Create monitor
            model_version = self.current_model.version if self.current_model else "1.0.0"
            self.model_monitor = ModelMonitor(
                model_name="stacked_ensemble",
                model_version=model_version,
                drift_thresholds=drift_thresholds,
                alert_config=alert_config,
                monitoring_config=monitoring_config
            )

            # Set baseline from labeled data if available
            if self.labeled_data and len(self.labeled_data) >= 50:
                X = np.array([ls.features.to_feature_vector() for ls in self.labeled_data])
                y = np.array([ls.label for ls in self.labeled_data])
                feature_names = [
                    'rrs', 'atr_percent', 'daily_strong', 'daily_weak',
                    'direction', 'spy_pct_change', 'stock_pct_change', 'volume_log'
                ]
                self.model_monitor.set_baseline(X, y, feature_names)

            # Set baseline performance from current model metrics
            if self.current_model_metrics:
                self.model_monitor.set_baseline_performance({
                    'accuracy': self.current_model_metrics.accuracy,
                    'precision': self.current_model_metrics.precision,
                    'recall': self.current_model_metrics.recall,
                    'f1_score': self.current_model_metrics.f1_score
                })

            # Attach monitor to model if loaded
            if self.current_model:
                self.current_model.set_model_monitor(self.model_monitor)

            logger.info("Model monitor initialized for drift detection")

        except Exception as e:
            logger.warning(f"Could not initialize model monitor: {e}")
            self.model_monitor = None

    def _on_drift_alert(self, report):
        """Callback when drift is detected"""
        logger.warning(
            f"DRIFT ALERT: severity={report.overall_severity.value}, "
            f"features_with_drift={report.features_with_drift}, "
            f"action={report.recommended_action}"
        )

        # Update Prometheus metrics
        if METRICS_AVAILABLE:
            severity_map = {
                DriftSeverity.NONE: 0,
                DriftSeverity.LOW: 1,
                DriftSeverity.MEDIUM: 2,
                DriftSeverity.HIGH: 3,
                DriftSeverity.CRITICAL: 4
            }

            # Get PSI scores for features
            psi_scores = {
                r.feature_name: r.psi_score
                for r in report.feature_drift_results
            }

            # Get current performance
            current_metrics = {}
            for r in report.performance_drift_results:
                current_metrics[r.metric_name] = r.current_value

            update_drift_metrics(
                model="stacked_ensemble",
                feature_psi_scores=psi_scores,
                accuracy=current_metrics.get('accuracy'),
                precision=current_metrics.get('precision'),
                recall=current_metrics.get('recall'),
                f1_score=current_metrics.get('f1_score'),
                drift_severity=severity_map[report.overall_severity],
                features_with_drift=report.features_with_drift
            )

    async def cleanup(self):
        """Cleanup learning agent"""
        # Save current model
        if self.current_model:
            self._save_model(self.current_model, "current_model.pkl")

        # Save labeled data
        self._save_labeled_data()

        logger.info("Learning Agent cleanup complete")

    def get_subscribed_events(self) -> List[EventType]:
        """Events this agent listens to"""
        return [
            EventType.SIGNAL_FOUND,      # Track new signals
            EventType.POSITION_OPENED,   # Link signal to position
            EventType.POSITION_CLOSED,   # Label outcome
        ]

    async def handle_event(self, event: Event):
        """Handle incoming events"""
        try:
            if event.event_type == EventType.SIGNAL_FOUND:
                await self._on_signal_found(event.data)

            elif event.event_type == EventType.POSITION_OPENED:
                await self._on_position_opened(event.data)

            elif event.event_type == EventType.POSITION_CLOSED:
                await self._on_position_closed(event.data)

        except Exception as e:
            logger.error(f"Error handling event: {e}")
            self.metrics.errors += 1

    async def _on_signal_found(self, signal_data: Dict):
        """Track a new signal"""
        try:
            # Extract features
            features = self._extract_features(signal_data)

            # Store pending signal
            self.pending_signals[features.symbol] = features

            # Save to database
            await self._save_signal_to_db(features)

            self.signals_tracked += 1
            self.metrics.custom_metrics["signals_tracked"] = self.signals_tracked

            logger.debug(f"Tracked signal: {features.symbol} RRS={features.rrs:.2f}")

        except Exception as e:
            logger.error(f"Error tracking signal: {e}")

    async def _on_position_opened(self, position_data: Dict):
        """Link position to signal"""
        symbol = position_data.get("symbol")

        if symbol in self.pending_signals:
            # Signal was acted upon - keep tracking
            logger.debug(f"Position opened for tracked signal: {symbol}")
        else:
            # Position opened without our signal tracking
            # Could happen if signal came from another source
            logger.debug(f"Position opened for non-tracked signal: {symbol}")

    async def _on_position_closed(self, position_data: Dict):
        """Label the outcome when position closes"""
        try:
            symbol = position_data.get("symbol")

            # Check if we have the signal
            if symbol not in self.pending_signals:
                logger.debug(f"Position closed for non-tracked symbol: {symbol}")
                return

            features = self.pending_signals.pop(symbol)

            # Extract outcome
            pnl = position_data.get("pnl", 0.0)
            entry_price = position_data.get("entry_price")
            exit_price = position_data.get("exit_price")
            shares = position_data.get("shares", 1)
            direction = features.direction

            # Calculate R-multiple
            stop_distance = features.atr  # Using ATR as stop distance
            if stop_distance > 0:
                if direction == "long":
                    r_multiple = (exit_price - entry_price) / stop_distance
                else:
                    r_multiple = (entry_price - exit_price) / stop_distance
            else:
                r_multiple = 0.0

            # Label: success if reached 2R or better
            label = 1 if r_multiple >= 2.0 else 0

            # Create labeled signal
            labeled_signal = LabeledSignal(
                features=features,
                label=label,
                outcome_timestamp=datetime.now(),
                realized_pnl=pnl,
                r_multiple=r_multiple
            )

            # Store labeled data
            self.labeled_data.append(labeled_signal)
            self.new_labels_count += 1
            self.outcomes_labeled += 1

            # Update success rate
            self._update_success_rate()

            # Update metrics
            self.metrics.custom_metrics["outcomes_labeled"] = self.outcomes_labeled
            self.metrics.custom_metrics["success_rate"] = round(self.success_rate, 3)
            self.metrics.custom_metrics["labeled_data_size"] = len(self.labeled_data)

            logger.info(
                f"Labeled outcome: {symbol} "
                f"Label={'SUCCESS' if label == 1 else 'FAILURE'} "
                f"R={r_multiple:.2f} PnL=${pnl:+,.2f}"
            )

            # Check if we should retrain
            await self._check_retrain_trigger()

        except Exception as e:
            logger.error(f"Error labeling outcome: {e}")

    def _extract_features(self, signal_data: Dict) -> SignalFeatures:
        """Extract features from signal data"""
        price = signal_data.get("price", 0.0)
        atr = signal_data.get("atr", 0.0)
        atr_percent = (atr / price * 100) if price > 0 else 0.0

        return SignalFeatures(
            symbol=signal_data.get("symbol", ""),
            direction=signal_data.get("direction", ""),
            rrs=signal_data.get("rrs", 0.0),
            price=price,
            atr=atr,
            atr_percent=atr_percent,
            daily_strong=signal_data.get("daily_strong", False),
            daily_weak=signal_data.get("daily_weak", False),
            ema3=signal_data.get("ema3"),
            ema8=signal_data.get("ema8"),
            spy_pct_change=signal_data.get("spy_pct_change"),
            stock_pct_change=signal_data.get("stock_pct_change"),
            volume=signal_data.get("volume"),
            timestamp=datetime.now()
        )

    async def _check_retrain_trigger(self):
        """Check if we should trigger model retraining"""
        should_retrain = False
        reason = ""

        # Trigger 1: Sufficient new data
        if self.new_labels_count >= self.retrain_threshold:
            should_retrain = True
            reason = f"New labels threshold reached ({self.new_labels_count})"

        # Trigger 2: Time-based (weekly)
        if self.last_retrain_time:
            days_since_retrain = (datetime.now() - self.last_retrain_time).days
            if days_since_retrain >= self.retrain_interval_days:
                should_retrain = True
                reason = f"Weekly retrain interval ({days_since_retrain} days)"
        elif len(self.labeled_data) >= 50:  # Initial training threshold
            should_retrain = True
            reason = "Initial model training"

        if should_retrain:
            logger.info(f"Triggering model retrain: {reason}")
            await self._retrain_model()

    async def _retrain_model(self):
        """Retrain the trading model"""
        try:
            logger.info("Starting model retraining...")

            # Get training data
            cutoff_date = datetime.now() - timedelta(days=self.lookback_months * 30)
            training_data = [
                ls for ls in self.labeled_data
                if ls.features.timestamp >= cutoff_date
            ]

            if len(training_data) < 20:
                logger.warning(
                    f"Insufficient training data ({len(training_data)} samples). "
                    f"Need at least 20."
                )
                return

            # Split train/validation (80/20)
            split_idx = int(len(training_data) * 0.8)
            train_data = training_data[:split_idx]
            val_data = training_data[split_idx:]

            # Prepare features and labels
            X_train = np.array([ls.features.to_feature_vector() for ls in train_data])
            y_train = np.array([ls.label for ls in train_data])

            X_val = np.array([ls.features.to_feature_vector() for ls in val_data])
            y_val = np.array([ls.label for ls in val_data])

            # Train new StackedEnsemble model with XGBoost and Random Forest
            new_model = StackedEnsemble(use_xgboost=True, use_random_forest=True, use_lstm=False)

            # Combine train and val for StackedEnsemble (it does its own validation split)
            X_full = np.vstack([X_train, X_val])
            y_full = np.hstack([y_train, y_val])

            # Train the stacked ensemble
            ensemble_metrics = new_model.train(X_full, y_full, validation_split=0.2)

            # Convert ensemble metrics to our ModelMetrics format
            new_metrics = ModelMetrics(
                accuracy=ensemble_metrics.accuracy,
                precision=ensemble_metrics.precision,
                recall=ensemble_metrics.recall,
                f1_score=ensemble_metrics.f1_score,
                roc_auc=ensemble_metrics.auc,
                training_samples=len(X_train),
                validation_samples=len(X_val),
                timestamp=datetime.now()
            )

            logger.info(
                f"New model trained - "
                f"Accuracy: {new_metrics.accuracy:.3f}, "
                f"Precision: {new_metrics.precision:.3f}, "
                f"F1: {new_metrics.f1_score:.3f}"
            )

            # Compare with current model
            deploy = self._should_deploy_model(new_model, new_metrics)

            if deploy:
                await self._deploy_model(new_model, new_metrics)
            else:
                logger.info("New model did not meet deployment criteria")

            # Update tracking
            self.last_retrain_time = datetime.now()
            self.new_labels_count = 0
            self.total_retrains += 1
            self.metrics.custom_metrics["total_retrains"] = self.total_retrains

        except Exception as e:
            logger.error(f"Error during model retraining: {e}")

    def _validate_model(
        self,
        model: StackedEnsemble,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> ModelMetrics:
        """Validate model performance using StackedEnsemble"""
        # Get predictions from StackedEnsemble
        predictions = model.predict_success_probability(X_val)
        predicted_labels = model.predict(X_val)

        # Calculate metrics
        tp = np.sum((predicted_labels == 1) & (y_val == 1))
        fp = np.sum((predicted_labels == 1) & (y_val == 0))
        tn = np.sum((predicted_labels == 0) & (y_val == 0))
        fn = np.sum((predicted_labels == 0) & (y_val == 1))

        accuracy = (tp + tn) / len(y_val) if len(y_val) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Simple AUC approximation
        sorted_idx = np.argsort(predictions)[::-1]
        sorted_labels = y_val[sorted_idx]
        cumsum = np.cumsum(sorted_labels)
        roc_auc = np.sum(cumsum) / (np.sum(y_val) * len(y_val)) if np.sum(y_val) > 0 else 0.5

        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            validation_samples=len(y_val),
            timestamp=datetime.now()
        )

    def _should_deploy_model(
        self,
        new_model: StackedEnsemble,
        new_metrics: ModelMetrics
    ) -> bool:
        """Determine if new model should be deployed"""
        # If no current model, deploy if accuracy > 50%
        if self.current_model is None:
            return new_metrics.accuracy > 0.5

        # Compare with current model
        if self.current_model_metrics is None:
            return True

        # Check for improvement
        improvement = new_metrics.f1_score - self.current_model_metrics.f1_score

        if improvement >= self.min_improvement:
            logger.info(
                f"Model improvement: {improvement:.3f} "
                f"(threshold: {self.min_improvement})"
            )
            return True

        logger.info(
            f"Model improvement insufficient: {improvement:.3f} "
            f"< {self.min_improvement}"
        )
        return False

    async def _deploy_model(
        self,
        model: StackedEnsemble,
        metrics: ModelMetrics
    ):
        """Deploy a new StackedEnsemble model"""
        try:
            # Save old model as backup
            if self.current_model:
                backup_path = self.model_dir / f"model_backup_{datetime.now():%Y%m%d_%H%M%S}.pkl"
                self._save_model(self.current_model, backup_path)

            # Deploy new model
            self.current_model = model
            self.current_model_metrics = metrics

            # Save as current
            self._save_model(model, "current_model.pkl")
            self._save_metrics(metrics, "current_metrics.json")

            self.total_deployments += 1
            self.metrics.custom_metrics["total_deployments"] = self.total_deployments
            self.metrics.custom_metrics["model_version"] = model.version

            logger.info(
                f"Model deployed successfully - "
                f"Version: {model.version}, "
                f"Accuracy: {metrics.accuracy:.3f}"
            )

            # Publish update event
            await self.publish(EventType.SYSTEM_MODEL_UPDATED, {
                "model_version": model.version,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "training_samples": metrics.training_samples,
                "validation_samples": metrics.validation_samples,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Error deploying model: {e}")

    def _update_success_rate(self):
        """Update overall success rate"""
        if self.labeled_data:
            successes = sum(1 for ls in self.labeled_data if ls.label == 1)
            self.success_rate = successes / len(self.labeled_data)

    def _load_current_model(self):
        """Load existing StackedEnsemble model from disk"""
        model_path = self.model_dir / "stacked_ensemble"
        metrics_path = self.model_dir / "current_metrics.json"

        try:
            if model_path.exists():
                self.current_model = StackedEnsemble(use_xgboost=True, use_random_forest=True, use_lstm=False)
                self.current_model.load(str(model_path))
                logger.info(f"Loaded StackedEnsemble model (trained: {self.current_model.is_trained})")

            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    # Handle timestamp field if present
                    if 'timestamp' in data and isinstance(data['timestamp'], str):
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    self.current_model_metrics = ModelMetrics(**data)
                logger.info(f"Loaded model metrics: Accuracy={self.current_model_metrics.accuracy:.3f}")

        except Exception as e:
            logger.warning(f"Could not load model: {e}")

    def _save_model(self, model: StackedEnsemble, filename: str):
        """Save StackedEnsemble model to disk"""
        try:
            # StackedEnsemble saves to a directory, not a single file
            if filename == "current_model.pkl":
                filepath = self.model_dir / "stacked_ensemble"
            else:
                # For backup, use the provided path as a directory
                filepath = self.model_dir / filename.replace(".pkl", "_ensemble")
            model.save(str(filepath))
            logger.debug(f"StackedEnsemble model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def _save_metrics(self, metrics: ModelMetrics, filename: str):
        """Save metrics to disk"""
        try:
            filepath = self.model_dir / filename
            with open(filepath, 'w') as f:
                json.dump({
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score,
                    "roc_auc": metrics.roc_auc,
                    "training_samples": metrics.training_samples,
                    "validation_samples": metrics.validation_samples,
                    "timestamp": metrics.timestamp.isoformat()
                }, f, indent=2)
            logger.debug(f"Metrics saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    async def _load_historical_data(self):
        """Load historical labeled data from database"""
        try:
            # Query database for historical trades
            with self.db_manager.get_session() as session:
                # Get closed trades from last N months
                cutoff_date = datetime.now() - timedelta(days=self.lookback_months * 30)

                trades = session.query(Trade).filter(
                    Trade.status == TradeStatus.CLOSED,
                    Trade.entry_time >= cutoff_date
                ).all()

                logger.info(f"Found {len(trades)} historical trades")

                # Convert to labeled signals (if we have the signal data)
                # This is a simplified version - in production you'd join with signals table
                for trade in trades:
                    # Reconstruct features (simplified - would need signal table join)
                    features = SignalFeatures(
                        symbol=trade.symbol,
                        direction=trade.direction,
                        rrs=trade.rrs_at_entry or 0.0,
                        price=trade.entry_price,
                        atr=0.0,  # Would need to fetch
                        atr_percent=0.0,
                        daily_strong=trade.direction == "long",
                        daily_weak=trade.direction == "short",
                        timestamp=trade.entry_time
                    )

                    # Calculate R-multiple from trade
                    if trade.pnl is not None and trade.stop_loss:
                        stop_distance = abs(trade.entry_price - trade.stop_loss)
                        if stop_distance > 0:
                            r_multiple = abs(trade.pnl) / (stop_distance * trade.shares)
                            if trade.pnl < 0:
                                r_multiple = -r_multiple
                        else:
                            r_multiple = 0.0

                        # Label
                        label = 1 if r_multiple >= 2.0 else 0

                        labeled_signal = LabeledSignal(
                            features=features,
                            label=label,
                            position_id=trade.id,
                            outcome_timestamp=trade.exit_time,
                            realized_pnl=trade.pnl,
                            r_multiple=r_multiple
                        )

                        self.labeled_data.append(labeled_signal)

                logger.info(f"Loaded {len(self.labeled_data)} labeled samples")

        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")

    async def _save_signal_to_db(self, features: SignalFeatures):
        """Save signal to database"""
        try:
            with self.db_manager.get_session() as session:
                signal = Signal(
                    symbol=features.symbol,
                    timestamp=features.timestamp,
                    rrs=features.rrs,
                    direction=features.direction,
                    price=features.price,
                    atr=features.atr,
                    daily_strong=features.daily_strong,
                    daily_weak=features.daily_weak,
                )
                session.add(signal)
                session.commit()

        except Exception as e:
            logger.debug(f"Could not save signal to database: {e}")

    def _save_labeled_data(self):
        """Save labeled data to disk for backup"""
        try:
            filepath = self.model_dir / "labeled_data.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(self.labeled_data, f)
            logger.info(f"Saved {len(self.labeled_data)} labeled samples")
        except Exception as e:
            logger.error(f"Error saving labeled data: {e}")

    def get_model_info(self) -> Dict:
        """Get current StackedEnsemble model information"""
        if not self.current_model:
            return {"status": "no_model"}

        info = {
            "type": "StackedEnsemble",
            "is_trained": self.current_model.is_trained,
            "n_base_models": self.current_model.n_base_models,
            "model_weights": self.current_model.model_weights,
            "use_xgboost": self.current_model.use_xgboost,
            "use_random_forest": self.current_model.use_random_forest,
            "use_lstm": self.current_model.use_lstm,
        }

        if self.current_model_metrics:
            info.update({
                "accuracy": self.current_model_metrics.accuracy,
                "precision": self.current_model_metrics.precision,
                "recall": self.current_model_metrics.recall,
                "f1_score": self.current_model_metrics.f1_score,
            })

        return info

    def predict_signal_quality(self, signal_data: Dict) -> float:
        """
        Predict quality/success probability of a signal using StackedEnsemble

        Args:
            signal_data: Signal data dictionary

        Returns:
            Probability of success (0-1)
        """
        if not self.current_model or not self.current_model.is_trained:
            return 0.5  # No trained model, return neutral

        try:
            import time
            start_time = time.time()

            features = self._extract_features(signal_data)
            feature_vector = features.to_feature_vector()
            # StackedEnsemble expects 2D array
            feature_array = feature_vector.reshape(1, -1)
            probability = self.current_model.predict_success_probability(feature_array)[0]

            # Record prediction timing for metrics
            inference_time = time.time() - start_time
            if METRICS_AVAILABLE:
                record_model_prediction(
                    model="stacked_ensemble",
                    confidence=float(probability),
                    duration=inference_time
                )

            # Track predictions for drift checking
            self._predictions_since_drift_check += 1

            # Periodically check for drift
            if self._predictions_since_drift_check >= self._drift_check_interval:
                self._check_drift_async()
                self._predictions_since_drift_check = 0

            return float(probability)

        except Exception as e:
            logger.error(f"Error predicting signal quality: {e}")
            return 0.5

    def _check_drift_async(self):
        """Check for drift (non-blocking)"""
        if self.model_monitor is None:
            return

        try:
            report = self.model_monitor.check_drift()

            if report:
                # Update metrics
                self.metrics.custom_metrics["drift_severity"] = report.overall_severity.value
                self.metrics.custom_metrics["drift_check_count"] = \
                    self.metrics.custom_metrics.get("drift_check_count", 0) + 1

                # Check if retraining is recommended due to drift
                if report.requires_action:
                    logger.warning(
                        f"Drift detected - Triggering retrain: {report.recommended_action}"
                    )

                    # Record retrain trigger
                    if METRICS_AVAILABLE:
                        record_model_retrain(
                            model="stacked_ensemble",
                            reason="drift_detected"
                        )

                    # We don't await here to avoid blocking
                    # The next _check_retrain_trigger will handle it
                    self.new_labels_count = self.retrain_threshold  # Force retrain check

        except Exception as e:
            logger.debug(f"Error checking drift: {e}")

    def get_drift_status(self) -> Dict[str, Any]:
        """
        Get current drift detection status.

        Returns:
            Dictionary with drift status information
        """
        if self.model_monitor is None:
            return {"status": "monitor_not_initialized"}

        status = self.model_monitor.get_current_status()
        report = self.model_monitor.get_drift_report()

        if report:
            status["last_drift_report"] = report.to_dict()

        return status

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current model performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        if self.model_monitor is None:
            return {"status": "monitor_not_initialized"}

        return self.model_monitor.get_performance_status()

    def get_learning_stats(self) -> Dict:
        """Get learning statistics"""
        return {
            "signals_tracked": self.signals_tracked,
            "outcomes_labeled": self.outcomes_labeled,
            "success_rate": self.success_rate,
            "labeled_data_size": len(self.labeled_data),
            "pending_signals": len(self.pending_signals),
            "new_labels_since_retrain": self.new_labels_count,
            "total_retrains": self.total_retrains,
            "total_deployments": self.total_deployments,
            "last_retrain": self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            "model_info": self.get_model_info()
        }
