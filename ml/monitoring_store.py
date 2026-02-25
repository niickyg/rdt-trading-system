"""
Monitoring Data Store for ML Model Drift Detection

Provides persistent storage for:
- Model predictions and actuals
- Feature distributions
- Windowed statistics (hourly, daily, weekly)
- Historical trend data for drift analysis
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from threading import Lock
import json
import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from ml.safe_model_loader import safe_load_model, safe_save_model, ModelSecurityError


@dataclass
class PredictionRecord:
    """Single prediction record for monitoring"""
    timestamp: datetime
    features: np.ndarray
    prediction: float
    predicted_class: int
    model_version: str
    feature_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeRecord:
    """Outcome record linking prediction to actual result"""
    prediction_timestamp: datetime
    outcome_timestamp: datetime
    actual_class: int
    actual_value: Optional[float] = None
    prediction: Optional[float] = None
    predicted_class: Optional[int] = None
    is_correct: Optional[bool] = None


@dataclass
class WindowStats:
    """Statistics for a time window"""
    window_start: datetime
    window_end: datetime
    window_type: str  # 'hourly', 'daily', 'weekly'

    # Prediction statistics
    prediction_count: int = 0
    mean_prediction: float = 0.0
    std_prediction: float = 0.0
    prediction_distribution: Dict[int, int] = field(default_factory=dict)

    # Performance statistics (if outcomes available)
    outcome_count: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Feature statistics
    feature_means: Optional[np.ndarray] = None
    feature_stds: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'window_start': self.window_start.isoformat(),
            'window_end': self.window_end.isoformat(),
            'window_type': self.window_type,
            'prediction_count': self.prediction_count,
            'mean_prediction': self.mean_prediction,
            'std_prediction': self.std_prediction,
            'prediction_distribution': self.prediction_distribution,
            'outcome_count': self.outcome_count,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'feature_means': self.feature_means.tolist() if self.feature_means is not None else None,
            'feature_stds': self.feature_stds.tolist() if self.feature_stds is not None else None,
        }


class MonitoringStore:
    """
    Storage for ML model monitoring data.

    Maintains:
    - Rolling window of recent predictions
    - Outcome records for performance tracking
    - Pre-computed window statistics
    - Historical baseline distributions

    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        max_predictions: int = 10000,
        max_outcomes: int = 5000,
        persistence_path: Optional[Path] = None,
        auto_persist: bool = True,
        persist_interval_minutes: int = 30
    ):
        """
        Initialize monitoring store.

        Args:
            max_predictions: Maximum predictions to keep in memory
            max_outcomes: Maximum outcomes to keep in memory
            persistence_path: Path for persisting data to disk
            auto_persist: Whether to auto-persist periodically
            persist_interval_minutes: How often to auto-persist
        """
        self.max_predictions = max_predictions
        self.max_outcomes = max_outcomes
        self.persistence_path = persistence_path
        self.auto_persist = auto_persist
        self.persist_interval_minutes = persist_interval_minutes

        # Thread safety
        self._lock = Lock()

        # Data storage using deques for efficient rolling windows
        self._predictions: deque[PredictionRecord] = deque(maxlen=max_predictions)
        self._outcomes: deque[OutcomeRecord] = deque(maxlen=max_outcomes)

        # Baseline distributions (from training data)
        self._baseline_feature_distributions: Optional[Dict[str, Dict]] = None
        self._baseline_prediction_distribution: Optional[Dict[int, float]] = None
        self._baseline_timestamp: Optional[datetime] = None

        # Pre-computed window statistics
        self._hourly_stats: Dict[str, WindowStats] = {}
        self._daily_stats: Dict[str, WindowStats] = {}
        self._weekly_stats: Dict[str, WindowStats] = {}

        # Feature names for consistent tracking
        self._feature_names: Optional[List[str]] = None
        self._n_features: Optional[int] = None

        # Tracking
        self._last_persist_time: Optional[datetime] = None
        self._total_predictions_recorded: int = 0
        self._total_outcomes_recorded: int = 0

        # Load persisted data if available
        if persistence_path:
            self._load_persisted_data()

        logger.info(
            f"MonitoringStore initialized (max_predictions={max_predictions}, "
            f"max_outcomes={max_outcomes})"
        )

    def record_prediction(
        self,
        features: np.ndarray,
        prediction: float,
        predicted_class: int,
        model_version: str,
        feature_names: Optional[List[str]] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a model prediction for monitoring.

        Args:
            features: Feature vector used for prediction
            prediction: Raw prediction value (probability)
            predicted_class: Predicted class label
            model_version: Version of model that made prediction
            feature_names: Names of features (optional)
            timestamp: Prediction timestamp (defaults to now)
            metadata: Additional metadata to store

        Returns:
            Unique identifier for this prediction
        """
        timestamp = timestamp or datetime.now()

        with self._lock:
            # Store feature names if provided and not yet set
            if feature_names and self._feature_names is None:
                self._feature_names = feature_names
                self._n_features = len(feature_names)
            elif self._n_features is None and features is not None:
                self._n_features = len(features)

            record = PredictionRecord(
                timestamp=timestamp,
                features=features.copy() if isinstance(features, np.ndarray) else np.array(features),
                prediction=prediction,
                predicted_class=predicted_class,
                model_version=model_version,
                feature_names=feature_names or self._feature_names,
                metadata=metadata or {}
            )

            self._predictions.append(record)
            self._total_predictions_recorded += 1

            # Check if we should auto-persist
            if self.auto_persist:
                self._check_auto_persist()

        prediction_id = f"pred_{timestamp.timestamp()}_{self._total_predictions_recorded}"

        logger.debug(
            f"Recorded prediction: class={predicted_class}, prob={prediction:.4f}, "
            f"version={model_version}"
        )

        return prediction_id

    def record_outcome(
        self,
        prediction_timestamp: datetime,
        actual_class: int,
        actual_value: Optional[float] = None,
        outcome_timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Record the actual outcome for a previous prediction.

        Args:
            prediction_timestamp: Timestamp of the original prediction
            actual_class: Actual class label
            actual_value: Actual continuous value (if applicable)
            outcome_timestamp: When outcome was observed (defaults to now)

        Returns:
            True if matching prediction was found and updated
        """
        outcome_timestamp = outcome_timestamp or datetime.now()

        with self._lock:
            # Find matching prediction
            matching_prediction = None
            for pred in reversed(self._predictions):
                if abs((pred.timestamp - prediction_timestamp).total_seconds()) < 1:
                    matching_prediction = pred
                    break

            outcome = OutcomeRecord(
                prediction_timestamp=prediction_timestamp,
                outcome_timestamp=outcome_timestamp,
                actual_class=actual_class,
                actual_value=actual_value,
                prediction=matching_prediction.prediction if matching_prediction else None,
                predicted_class=matching_prediction.predicted_class if matching_prediction else None,
                is_correct=matching_prediction.predicted_class == actual_class if matching_prediction else None
            )

            self._outcomes.append(outcome)
            self._total_outcomes_recorded += 1

        logger.debug(
            f"Recorded outcome: actual_class={actual_class}, "
            f"is_correct={outcome.is_correct}"
        )

        return matching_prediction is not None

    def set_baseline_distributions(
        self,
        feature_distributions: Dict[str, Dict],
        prediction_distribution: Dict[int, float],
        feature_names: Optional[List[str]] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Set baseline distributions from training data.

        Args:
            feature_distributions: Dict mapping feature names to their distribution stats
                Each entry should have: {'mean': float, 'std': float, 'min': float, 'max': float}
            prediction_distribution: Dict mapping class labels to their proportions
            feature_names: List of feature names
            timestamp: When baseline was computed
        """
        with self._lock:
            self._baseline_feature_distributions = feature_distributions
            self._baseline_prediction_distribution = prediction_distribution
            self._baseline_timestamp = timestamp or datetime.now()

            if feature_names:
                self._feature_names = feature_names
                self._n_features = len(feature_names)

        logger.info(
            f"Set baseline distributions for {len(feature_distributions)} features, "
            f"{len(prediction_distribution)} classes"
        )

    def set_baseline_from_training_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """
        Compute and set baseline distributions from training data.

        Args:
            X_train: Training features array
            y_train: Training labels array
            feature_names: Optional feature names
        """
        n_features = X_train.shape[1]

        # Compute feature distributions
        feature_distributions = {}
        names = feature_names or [f"feature_{i}" for i in range(n_features)]

        for i, name in enumerate(names):
            feature_values = X_train[:, i]
            feature_distributions[name] = {
                'mean': float(np.mean(feature_values)),
                'std': float(np.std(feature_values)),
                'min': float(np.min(feature_values)),
                'max': float(np.max(feature_values)),
                'percentiles': {
                    '5': float(np.percentile(feature_values, 5)),
                    '25': float(np.percentile(feature_values, 25)),
                    '50': float(np.percentile(feature_values, 50)),
                    '75': float(np.percentile(feature_values, 75)),
                    '95': float(np.percentile(feature_values, 95)),
                }
            }

        # Compute prediction/class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        prediction_distribution = {int(c): count / total for c, count in zip(unique, counts)}

        self.set_baseline_distributions(
            feature_distributions=feature_distributions,
            prediction_distribution=prediction_distribution,
            feature_names=names
        )

    def get_recent_predictions(
        self,
        n: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[PredictionRecord]:
        """
        Get recent prediction records.

        Args:
            n: Number of recent predictions to return
            since: Return predictions since this timestamp

        Returns:
            List of prediction records
        """
        with self._lock:
            predictions = list(self._predictions)

        if since:
            predictions = [p for p in predictions if p.timestamp >= since]

        if n:
            predictions = predictions[-n:]

        return predictions

    def get_recent_outcomes(
        self,
        n: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[OutcomeRecord]:
        """
        Get recent outcome records.

        Args:
            n: Number of recent outcomes to return
            since: Return outcomes since this timestamp

        Returns:
            List of outcome records
        """
        with self._lock:
            outcomes = list(self._outcomes)

        if since:
            outcomes = [o for o in outcomes if o.outcome_timestamp >= since]

        if n:
            outcomes = outcomes[-n:]

        return outcomes

    def get_window_stats(
        self,
        window_type: str = 'hourly',
        n_windows: int = 24
    ) -> List[WindowStats]:
        """
        Get pre-computed window statistics.

        Args:
            window_type: Type of window ('hourly', 'daily', 'weekly')
            n_windows: Number of windows to return

        Returns:
            List of WindowStats objects
        """
        # First, compute fresh stats
        self._compute_window_stats(window_type)

        with self._lock:
            if window_type == 'hourly':
                stats_dict = self._hourly_stats
            elif window_type == 'daily':
                stats_dict = self._daily_stats
            elif window_type == 'weekly':
                stats_dict = self._weekly_stats
            else:
                raise ValueError(f"Unknown window type: {window_type}")

            # Sort by window start and return most recent
            sorted_stats = sorted(
                stats_dict.values(),
                key=lambda s: s.window_start,
                reverse=True
            )

            return sorted_stats[:n_windows]

    def _compute_window_stats(self, window_type: str):
        """Compute statistics for a given window type."""
        now = datetime.now()

        if window_type == 'hourly':
            delta = timedelta(hours=1)
            window_start = now.replace(minute=0, second=0, microsecond=0)
        elif window_type == 'daily':
            delta = timedelta(days=1)
            window_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif window_type == 'weekly':
            delta = timedelta(weeks=1)
            # Start from Monday
            days_since_monday = now.weekday()
            window_start = (now - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        else:
            return

        window_end = window_start + delta
        window_key = window_start.isoformat()

        with self._lock:
            # Get predictions in this window
            window_predictions = [
                p for p in self._predictions
                if window_start <= p.timestamp < window_end
            ]

            # Get outcomes in this window
            window_outcomes = [
                o for o in self._outcomes
                if window_start <= o.outcome_timestamp < window_end
            ]

            if not window_predictions:
                return

            # Compute prediction statistics
            predictions = np.array([p.prediction for p in window_predictions])
            predicted_classes = [p.predicted_class for p in window_predictions]

            class_counts = {}
            for c in predicted_classes:
                class_counts[c] = class_counts.get(c, 0) + 1

            # Compute feature statistics
            features = np.array([p.features for p in window_predictions])
            feature_means = np.mean(features, axis=0)
            feature_stds = np.std(features, axis=0)

            # Compute performance metrics if outcomes available
            accuracy = precision = recall = f1 = 0.0
            if window_outcomes:
                correct = sum(1 for o in window_outcomes if o.is_correct)
                accuracy = correct / len(window_outcomes) if window_outcomes else 0.0

                # Binary classification metrics
                tp = sum(1 for o in window_outcomes if o.is_correct and o.predicted_class == 1)
                fp = sum(1 for o in window_outcomes if not o.is_correct and o.predicted_class == 1)
                fn = sum(1 for o in window_outcomes if not o.is_correct and o.predicted_class == 0)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            stats = WindowStats(
                window_start=window_start,
                window_end=window_end,
                window_type=window_type,
                prediction_count=len(window_predictions),
                mean_prediction=float(np.mean(predictions)),
                std_prediction=float(np.std(predictions)),
                prediction_distribution=class_counts,
                outcome_count=len(window_outcomes),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                feature_means=feature_means,
                feature_stds=feature_stds
            )

            # Store in appropriate dict
            if window_type == 'hourly':
                self._hourly_stats[window_key] = stats
            elif window_type == 'daily':
                self._daily_stats[window_key] = stats
            elif window_type == 'weekly':
                self._weekly_stats[window_key] = stats

    def get_feature_distributions(
        self,
        since: Optional[datetime] = None
    ) -> Optional[Dict[str, Dict]]:
        """
        Get current feature distributions from recent predictions.

        Args:
            since: Only use predictions since this timestamp

        Returns:
            Dictionary mapping feature names to distribution stats
        """
        predictions = self.get_recent_predictions(since=since)

        if not predictions:
            return None

        features = np.array([p.features for p in predictions])
        n_features = features.shape[1]
        names = self._feature_names or [f"feature_{i}" for i in range(n_features)]

        distributions = {}
        for i, name in enumerate(names):
            feature_values = features[:, i]
            distributions[name] = {
                'mean': float(np.mean(feature_values)),
                'std': float(np.std(feature_values)),
                'min': float(np.min(feature_values)),
                'max': float(np.max(feature_values)),
                'count': len(feature_values)
            }

        return distributions

    def get_prediction_distribution(
        self,
        since: Optional[datetime] = None
    ) -> Optional[Dict[int, float]]:
        """
        Get current prediction class distribution.

        Args:
            since: Only use predictions since this timestamp

        Returns:
            Dictionary mapping class labels to proportions
        """
        predictions = self.get_recent_predictions(since=since)

        if not predictions:
            return None

        predicted_classes = [p.predicted_class for p in predictions]
        unique, counts = np.unique(predicted_classes, return_counts=True)
        total = len(predicted_classes)

        return {int(c): count / total for c, count in zip(unique, counts)}

    def get_baseline_distributions(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Get baseline distributions.

        Returns:
            Tuple of (feature_distributions, prediction_distribution)
        """
        with self._lock:
            return (
                self._baseline_feature_distributions,
                self._baseline_prediction_distribution
            )

    def get_performance_metrics(
        self,
        since: Optional[datetime] = None,
        n_recent: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from recorded outcomes.

        Args:
            since: Only use outcomes since this timestamp
            n_recent: Only use n most recent outcomes

        Returns:
            Dictionary with performance metrics
        """
        outcomes = self.get_recent_outcomes(n=n_recent, since=since)

        if not outcomes:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'sample_count': 0
            }

        # Filter to only outcomes with predictions
        valid_outcomes = [o for o in outcomes if o.is_correct is not None]

        if not valid_outcomes:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'sample_count': 0
            }

        correct = sum(1 for o in valid_outcomes if o.is_correct)
        accuracy = correct / len(valid_outcomes)

        # Binary classification metrics
        tp = sum(1 for o in valid_outcomes if o.is_correct and o.predicted_class == 1)
        fp = sum(1 for o in valid_outcomes if not o.is_correct and o.predicted_class == 1)
        fn = sum(1 for o in valid_outcomes if not o.is_correct and o.predicted_class == 0)
        tn = sum(1 for o in valid_outcomes if o.is_correct and o.predicted_class == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'sample_count': len(valid_outcomes)
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of monitoring store state.

        Returns:
            Dictionary with summary information
        """
        with self._lock:
            return {
                'total_predictions_recorded': self._total_predictions_recorded,
                'total_outcomes_recorded': self._total_outcomes_recorded,
                'current_prediction_buffer_size': len(self._predictions),
                'current_outcome_buffer_size': len(self._outcomes),
                'max_predictions': self.max_predictions,
                'max_outcomes': self.max_outcomes,
                'has_baseline': self._baseline_feature_distributions is not None,
                'baseline_timestamp': self._baseline_timestamp.isoformat() if self._baseline_timestamp else None,
                'n_features': self._n_features,
                'feature_names': self._feature_names,
                'last_persist_time': self._last_persist_time.isoformat() if self._last_persist_time else None
            }

    def _check_auto_persist(self):
        """Check if we should auto-persist data."""
        if not self.persistence_path:
            return

        now = datetime.now()
        if self._last_persist_time is None:
            should_persist = True
        else:
            minutes_elapsed = (now - self._last_persist_time).total_seconds() / 60
            should_persist = minutes_elapsed >= self.persist_interval_minutes

        if should_persist:
            self._persist_data()

    def _persist_data(self):
        """Persist data to disk."""
        if not self.persistence_path:
            return

        try:
            self.persistence_path.mkdir(parents=True, exist_ok=True)

            # Persist predictions
            predictions_path = self.persistence_path / 'predictions.pkl'
            safe_save_model(list(self._predictions), str(predictions_path))

            # Persist outcomes
            outcomes_path = self.persistence_path / 'outcomes.pkl'
            safe_save_model(list(self._outcomes), str(outcomes_path))

            # Persist baseline and metadata
            metadata = {
                'baseline_feature_distributions': self._baseline_feature_distributions,
                'baseline_prediction_distribution': self._baseline_prediction_distribution,
                'baseline_timestamp': self._baseline_timestamp,
                'feature_names': self._feature_names,
                'n_features': self._n_features,
                'total_predictions_recorded': self._total_predictions_recorded,
                'total_outcomes_recorded': self._total_outcomes_recorded
            }
            metadata_path = self.persistence_path / 'metadata.json'
            with open(metadata_path, 'w') as f:
                # Convert datetime objects for JSON
                json_metadata = {
                    k: (v.isoformat() if isinstance(v, datetime) else v)
                    for k, v in metadata.items()
                }
                json.dump(json_metadata, f, indent=2)

            self._last_persist_time = datetime.now()
            logger.debug(f"Persisted monitoring data to {self.persistence_path}")

        except Exception as e:
            logger.error(f"Failed to persist monitoring data: {e}")

    def _load_persisted_data(self):
        """Load persisted data from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            # Load predictions
            predictions_path = self.persistence_path / 'predictions.pkl'
            if predictions_path.exists():
                loaded_predictions = safe_load_model(str(predictions_path), allow_unverified=False)
                for pred in loaded_predictions:
                    self._predictions.append(pred)

            # Load outcomes
            outcomes_path = self.persistence_path / 'outcomes.pkl'
            if outcomes_path.exists():
                loaded_outcomes = safe_load_model(str(outcomes_path), allow_unverified=False)
                for outcome in loaded_outcomes:
                    self._outcomes.append(outcome)

            # Load metadata
            metadata_path = self.persistence_path / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self._baseline_feature_distributions = metadata.get('baseline_feature_distributions')
                    self._baseline_prediction_distribution = metadata.get('baseline_prediction_distribution')
                    baseline_ts = metadata.get('baseline_timestamp')
                    self._baseline_timestamp = datetime.fromisoformat(baseline_ts) if baseline_ts else None
                    self._feature_names = metadata.get('feature_names')
                    self._n_features = metadata.get('n_features')
                    self._total_predictions_recorded = metadata.get('total_predictions_recorded', 0)
                    self._total_outcomes_recorded = metadata.get('total_outcomes_recorded', 0)

            logger.info(
                f"Loaded persisted data: {len(self._predictions)} predictions, "
                f"{len(self._outcomes)} outcomes"
            )

        except Exception as e:
            logger.error(f"Failed to load persisted monitoring data: {e}")

    def clear(self):
        """Clear all stored data."""
        with self._lock:
            self._predictions.clear()
            self._outcomes.clear()
            self._hourly_stats.clear()
            self._daily_stats.clear()
            self._weekly_stats.clear()
            self._total_predictions_recorded = 0
            self._total_outcomes_recorded = 0

        logger.info("MonitoringStore cleared")

    def persist(self):
        """Manually trigger data persistence."""
        with self._lock:
            self._persist_data()
