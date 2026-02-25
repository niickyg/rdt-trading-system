"""
ML Model Monitor for RDT Trading System

Provides continuous monitoring of ML model performance including:
- Prediction distribution tracking
- Feature distribution comparison vs training data
- Automatic drift detection and alerting
- Integration with Prometheus metrics
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from threading import Lock, Thread
import time
from pathlib import Path

import numpy as np
import threading
from loguru import logger

from ml.monitoring_store import MonitoringStore
from ml.drift_detector import (
    DriftDetector,
    DriftThresholds,
    DriftReport,
    DriftSeverity
)


@dataclass
class AlertConfig:
    """Configuration for drift alerts"""
    enabled: bool = True
    alert_on_medium: bool = False  # Alert on medium severity
    alert_on_high: bool = True  # Alert on high severity
    alert_on_critical: bool = True  # Alert on critical severity
    cooldown_minutes: int = 60  # Minimum time between alerts
    alert_callback: Optional[Callable[[DriftReport], None]] = None


@dataclass
class MonitoringConfig:
    """Configuration for model monitoring"""
    check_interval_minutes: int = 15  # How often to run drift checks
    lookback_hours: int = 24  # How far back to look for drift
    min_predictions_for_check: int = 50  # Minimum predictions before checking
    enable_background_monitoring: bool = True
    persistence_path: Optional[Path] = None


class ModelMonitor:
    """
    Continuous monitoring for ML models in the RDT Trading System.

    Features:
    - Tracks prediction distributions over time
    - Compares feature distributions vs training data
    - Detects drift using PSI and statistical tests
    - Alerts when drift exceeds thresholds
    - Integrates with Prometheus metrics

    Usage:
        monitor = ModelMonitor(
            model_name="stacked_ensemble",
            baseline_features=X_train,
            baseline_labels=y_train
        )

        # Record each prediction
        monitor.record_prediction(features, probability, predicted_class)

        # Record outcomes when known
        monitor.record_outcome(prediction_time, actual_class)

        # Check drift status
        if monitor.should_retrain():
            # Trigger retraining
            pass
    """

    def __init__(
        self,
        model_name: str,
        model_version: str = "1.0.0",
        baseline_features: Optional[np.ndarray] = None,
        baseline_labels: Optional[np.ndarray] = None,
        baseline_performance: Optional[Dict[str, float]] = None,
        feature_names: Optional[List[str]] = None,
        drift_thresholds: Optional[DriftThresholds] = None,
        alert_config: Optional[AlertConfig] = None,
        monitoring_config: Optional[MonitoringConfig] = None
    ):
        """
        Initialize model monitor.

        Args:
            model_name: Name of the model being monitored
            model_version: Current version of the model
            baseline_features: Training features for baseline distributions
            baseline_labels: Training labels for baseline distributions
            baseline_performance: Baseline performance metrics
            feature_names: Names of features
            drift_thresholds: Custom drift detection thresholds
            alert_config: Alert configuration
            monitoring_config: Monitoring configuration
        """
        self.model_name = model_name
        self.model_version = model_version
        self.feature_names = feature_names

        # Configuration
        self.drift_thresholds = drift_thresholds or DriftThresholds()
        self.alert_config = alert_config or AlertConfig()
        self.monitoring_config = monitoring_config or MonitoringConfig()

        # Initialize monitoring store
        persistence_path = self.monitoring_config.persistence_path
        if persistence_path:
            persistence_path = persistence_path / model_name

        self.monitoring_store = MonitoringStore(
            persistence_path=persistence_path,
            auto_persist=True
        )

        # Set baseline if provided
        if baseline_features is not None and baseline_labels is not None:
            self.set_baseline(baseline_features, baseline_labels, feature_names)

        # Initialize drift detector
        self.drift_detector = DriftDetector(
            monitoring_store=self.monitoring_store,
            thresholds=self.drift_thresholds,
            baseline_performance=baseline_performance
        )

        # Alert tracking
        self._last_alert_time: Optional[datetime] = None
        self._alert_history: List[Tuple[datetime, DriftSeverity]] = []

        # Background monitoring
        self._monitoring_thread: Optional[Thread] = None
        self._stop_monitoring = False
        self._lock = Lock()

        # Drift state
        self._last_drift_report: Optional[DriftReport] = None
        self._retrain_recommended = False

        # Statistics
        self._predictions_since_retrain = 0
        self._total_predictions = 0
        self._total_outcomes = 0

        logger.info(
            f"ModelMonitor initialized for {model_name} v{model_version}"
        )

    def set_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """
        Set baseline distributions from training data.

        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Feature names
        """
        if feature_names:
            self.feature_names = feature_names

        self.monitoring_store.set_baseline_from_training_data(
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names or self.feature_names
        )

        # Calculate baseline performance if not provided
        # (This would need actual model predictions)

        logger.info(
            f"Baseline set from {len(X_train)} training samples with "
            f"{X_train.shape[1]} features"
        )

    def set_baseline_performance(self, metrics: Dict[str, float]):
        """
        Set baseline performance metrics.

        Args:
            metrics: Dictionary with accuracy, precision, recall, f1_score
        """
        self.drift_detector.set_baseline_performance(metrics)

    def record_prediction(
        self,
        features: np.ndarray,
        prediction: float,
        predicted_class: int,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a model prediction for monitoring.

        Args:
            features: Feature vector used for prediction
            prediction: Raw prediction probability
            predicted_class: Predicted class label
            timestamp: Prediction timestamp
            metadata: Additional metadata

        Returns:
            Prediction ID
        """
        prediction_id = self.monitoring_store.record_prediction(
            features=features,
            prediction=prediction,
            predicted_class=predicted_class,
            model_version=self.model_version,
            feature_names=self.feature_names,
            timestamp=timestamp,
            metadata=metadata
        )

        with self._lock:
            self._predictions_since_retrain += 1
            self._total_predictions += 1

        return prediction_id

    def record_outcome(
        self,
        prediction_timestamp: datetime,
        actual_class: int,
        actual_value: Optional[float] = None,
        outcome_timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Record the actual outcome for a prediction.

        Args:
            prediction_timestamp: When the prediction was made
            actual_class: Actual class label
            actual_value: Actual continuous value (if applicable)
            outcome_timestamp: When outcome was observed

        Returns:
            True if matching prediction was found
        """
        result = self.monitoring_store.record_outcome(
            prediction_timestamp=prediction_timestamp,
            actual_class=actual_class,
            actual_value=actual_value,
            outcome_timestamp=outcome_timestamp
        )

        with self._lock:
            self._total_outcomes += 1

        return result

    def check_drift(
        self,
        force: bool = False
    ) -> Optional[DriftReport]:
        """
        Run drift detection.

        Args:
            force: Force check even if insufficient data

        Returns:
            DriftReport or None if check was skipped
        """
        # Check if we have enough data
        if not force:
            predictions = self.monitoring_store.get_recent_predictions()
            if len(predictions) < self.monitoring_config.min_predictions_for_check:
                logger.debug(
                    f"Skipping drift check: only {len(predictions)} predictions "
                    f"(need {self.monitoring_config.min_predictions_for_check})"
                )
                return None

        # Run drift check
        lookback = datetime.now() - timedelta(hours=self.monitoring_config.lookback_hours)
        report = self.drift_detector.run_full_drift_check(since=lookback)

        with self._lock:
            self._last_drift_report = report
            self._retrain_recommended = report.requires_action

        # Handle alerts
        if self.alert_config.enabled:
            self._handle_alerts(report)

        return report

    def should_retrain(self) -> bool:
        """
        Check if model retraining is recommended.

        Returns:
            True if retraining is recommended based on drift detection
        """
        with self._lock:
            return self._retrain_recommended

    def mark_retrained(self, new_version: str):
        """
        Mark that the model has been retrained.

        Args:
            new_version: New model version after retraining
        """
        with self._lock:
            self.model_version = new_version
            self._predictions_since_retrain = 0
            self._retrain_recommended = False

        logger.info(f"Model marked as retrained to version {new_version}")

    def get_current_status(self) -> Dict[str, Any]:
        """
        Get current monitoring status.

        Returns:
            Dictionary with monitoring status
        """
        with self._lock:
            status = {
                'model_name': self.model_name,
                'model_version': self.model_version,
                'total_predictions': self._total_predictions,
                'total_outcomes': self._total_outcomes,
                'predictions_since_retrain': self._predictions_since_retrain,
                'retrain_recommended': self._retrain_recommended,
                'last_drift_check': self._last_drift_report.timestamp.isoformat() if self._last_drift_report else None,
                'overall_drift_severity': self._last_drift_report.overall_severity.value if self._last_drift_report else 'unknown',
                'monitoring_active': self._monitoring_thread is not None and self._monitoring_thread.is_alive()
            }

        # Add monitoring store summary
        status['store_summary'] = self.monitoring_store.get_summary()

        return status

    def get_drift_report(self) -> Optional[DriftReport]:
        """Get the most recent drift report."""
        return self._last_drift_report

    def get_feature_drift_status(self) -> List[Dict[str, Any]]:
        """
        Get current feature drift status.

        Returns:
            List of feature drift info
        """
        if not self._last_drift_report:
            return []

        return [
            {
                'feature': r.feature_name,
                'psi_score': r.psi_score,
                'drift_detected': r.drift_detected,
                'severity': r.severity.value
            }
            for r in self._last_drift_report.feature_drift_results
        ]

    def get_performance_status(self) -> Dict[str, Any]:
        """
        Get current performance status.

        Returns:
            Dictionary with performance metrics
        """
        current_metrics = self.monitoring_store.get_performance_metrics(
            since=datetime.now() - timedelta(hours=self.monitoring_config.lookback_hours)
        )

        result = {
            'current_metrics': current_metrics,
            'degradation_detected': False,
            'degraded_metrics': []
        }

        if self._last_drift_report:
            degraded = [
                r.metric_name
                for r in self._last_drift_report.performance_drift_results
                if r.degradation_detected
            ]
            result['degradation_detected'] = len(degraded) > 0
            result['degraded_metrics'] = degraded

        return result

    def start_background_monitoring(self):
        """Start background monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Background monitoring already running")
            return

        self._stop_monitoring = False
        self._monitoring_thread = Thread(
            target=self._background_monitor_loop,
            daemon=True,
            name=f"ModelMonitor-{self.model_name}"
        )
        self._monitoring_thread.start()
        logger.info(f"Started background monitoring for {self.model_name}")

    def stop_background_monitoring(self):
        """Stop background monitoring thread."""
        self._stop_monitoring = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
            self._monitoring_thread = None
        logger.info(f"Stopped background monitoring for {self.model_name}")

    def _background_monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_monitoring:
            try:
                self.check_drift()
            except Exception as e:
                logger.error(f"Error in background drift check: {e}")

            # Sleep with periodic check for stop signal
            for _ in range(self.monitoring_config.check_interval_minutes * 60):
                if self._stop_monitoring:
                    break
                time.sleep(1)

    def _handle_alerts(self, report: DriftReport):
        """Handle drift alerts based on configuration."""
        severity = report.overall_severity

        # Check if we should alert
        should_alert = False
        if severity == DriftSeverity.CRITICAL and self.alert_config.alert_on_critical:
            should_alert = True
        elif severity == DriftSeverity.HIGH and self.alert_config.alert_on_high:
            should_alert = True
        elif severity == DriftSeverity.MEDIUM and self.alert_config.alert_on_medium:
            should_alert = True

        if not should_alert:
            return

        # Check cooldown
        now = datetime.now()
        if self._last_alert_time:
            cooldown = timedelta(minutes=self.alert_config.cooldown_minutes)
            if now - self._last_alert_time < cooldown:
                logger.debug(f"Alert suppressed due to cooldown")
                return

        # Record alert
        self._last_alert_time = now
        self._alert_history.append((now, severity))

        # Trigger callback if configured
        if self.alert_config.alert_callback:
            try:
                self.alert_config.alert_callback(report)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        logger.warning(
            f"DRIFT ALERT [{severity.value.upper()}]: {self.model_name} - "
            f"{report.recommended_action}"
        )

    def get_prometheus_metrics(self) -> Dict[str, float]:
        """
        Get metrics formatted for Prometheus.

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        # Basic counts
        metrics['predictions_total'] = float(self._total_predictions)
        metrics['outcomes_total'] = float(self._total_outcomes)
        metrics['predictions_since_retrain'] = float(self._predictions_since_retrain)
        metrics['retrain_recommended'] = 1.0 if self._retrain_recommended else 0.0

        # Drift metrics from detector
        if self._last_drift_report:
            drift_metrics = self.drift_detector.get_drift_metrics_for_prometheus()
            metrics.update(drift_metrics)

        return metrics

    def cleanup(self):
        """Cleanup resources."""
        self.stop_background_monitoring()
        self.monitoring_store.persist()
        logger.info(f"ModelMonitor cleanup complete for {self.model_name}")


class ModelMonitorRegistry:
    """
    Registry for managing multiple model monitors.

    Usage:
        registry = ModelMonitorRegistry()
        registry.register_monitor(monitor)
        all_status = registry.get_all_status()
    """

    def __init__(self):
        self._monitors: Dict[str, ModelMonitor] = {}
        self._lock = Lock()

    def register_monitor(self, monitor: ModelMonitor):
        """Register a model monitor."""
        with self._lock:
            self._monitors[monitor.model_name] = monitor
        logger.info(f"Registered monitor for {monitor.model_name}")

    def unregister_monitor(self, model_name: str):
        """Unregister a model monitor."""
        with self._lock:
            if model_name in self._monitors:
                self._monitors[model_name].cleanup()
                del self._monitors[model_name]

    def get_monitor(self, model_name: str) -> Optional[ModelMonitor]:
        """Get a specific monitor."""
        with self._lock:
            return self._monitors.get(model_name)

    def get_all_monitors(self) -> Dict[str, ModelMonitor]:
        """Get all monitors."""
        with self._lock:
            return dict(self._monitors)

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all monitors."""
        with self._lock:
            return {
                name: monitor.get_current_status()
                for name, monitor in self._monitors.items()
            }

    def check_all_drift(self) -> Dict[str, Optional[DriftReport]]:
        """Run drift checks on all monitors."""
        results = {}
        with self._lock:
            for name, monitor in self._monitors.items():
                results[name] = monitor.check_drift()
        return results

    def any_retrain_recommended(self) -> bool:
        """Check if any model needs retraining."""
        with self._lock:
            return any(m.should_retrain() for m in self._monitors.values())

    def get_all_prometheus_metrics(self) -> Dict[str, float]:
        """Get Prometheus metrics from all monitors."""
        metrics = {}
        with self._lock:
            for name, monitor in self._monitors.items():
                model_metrics = monitor.get_prometheus_metrics()
                for metric_name, value in model_metrics.items():
                    metrics[f'{name}_{metric_name}'] = value
        return metrics

    def cleanup_all(self):
        """Cleanup all monitors."""
        with self._lock:
            for monitor in self._monitors.values():
                monitor.cleanup()
            self._monitors.clear()


# Global registry instance
_global_registry: Optional[ModelMonitorRegistry] = None
_global_registry_lock = threading.Lock()


def get_monitor_registry() -> ModelMonitorRegistry:
    """Get the global model monitor registry (thread-safe)."""
    global _global_registry
    if _global_registry is None:
        with _global_registry_lock:
            if _global_registry is None:
                _global_registry = ModelMonitorRegistry()
    return _global_registry
