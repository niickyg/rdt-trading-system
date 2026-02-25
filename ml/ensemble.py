"""
Meta-Learner Ensemble for Trade Success Prediction

Combines XGBoost, Random Forest, and LSTM models using a Logistic Regression
meta-learner to produce final probability predictions.

GPU Acceleration:
- Automatically detects CUDA/MPS GPU availability
- Configures TensorFlow/Keras for GPU training
- Supports memory growth to prevent OOM errors
- Falls back gracefully to CPU if no GPU available
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, precision_score,
        recall_score, f1_score, confusion_matrix, log_loss
    )
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn joblib")

from ml.safe_model_loader import safe_load_model, safe_save_model, ModelSecurityError

# GPU utilities for accelerated training
try:
    from ml.gpu_utils import (
        is_gpu_available,
        is_cuda_available,
        is_mps_available,
        configure_gpu,
        get_gpu_info,
        get_optimal_batch_size,
        setup_gpu_for_training,
        get_gpu_summary,
        update_gpu_metrics,
        GPU_METRICS_AVAILABLE
    )
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    GPU_METRICS_AVAILABLE = False
    logger.debug("GPU utilities not available")

# Import base models
try:
    from ml.models.xgboost_model import XGBoostTradeClassifier
    from ml.models.random_forest_model import RandomForestTradeClassifier
    from ml.models.lstm_model import LSTMTradePredictor
    BASE_MODELS_AVAILABLE = True
except ImportError:
    BASE_MODELS_AVAILABLE = False
    logger.warning("Base models not available. Ensemble cannot function.")


@dataclass
class EnsembleMetrics:
    """Validation metrics for Ensemble model"""
    accuracy: float = 0.0
    auc: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    log_loss: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    model_weights: Dict[str, float] = field(default_factory=dict)
    base_model_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'accuracy': self.accuracy,
            'auc': self.auc,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'log_loss': self.log_loss,
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            'model_weights': self.model_weights,
            'base_model_performance': self.base_model_performance,
            'timestamp': self.timestamp.isoformat()
        }


class StackedEnsemble:
    """
    Stacked Ensemble using Logistic Regression as meta-learner

    Combines predictions from:
    1. XGBoost classifier
    2. Random Forest classifier
    3. LSTM time-series predictor

    The meta-learner (Logistic Regression) learns optimal weights
    for combining base model predictions.

    Architecture:
    - Level 0 (Base models): XGBoost, Random Forest, LSTM
    - Level 1 (Meta-learner): Logistic Regression
    - Output: Final probability of trade success
    """

    def __init__(
        self,
        use_xgboost: bool = True,
        use_random_forest: bool = True,
        use_lstm: bool = False,  # LSTM requires sequential data
        meta_learner_C: float = 1.0,
        random_state: int = 42,
        # GPU configuration
        use_gpu: str = "auto",
        gpu_memory_limit: Optional[float] = None,
        gpu_device_id: Optional[int] = None,
        gpu_memory_growth: bool = True,
        enable_mixed_precision: bool = False
    ):
        """
        Initialize Stacked Ensemble

        Args:
            use_xgboost: Whether to use XGBoost in ensemble
            use_random_forest: Whether to use Random Forest in ensemble
            use_lstm: Whether to use LSTM in ensemble
            meta_learner_C: Regularization strength for meta-learner
            random_state: Random seed for reproducibility
            use_gpu: GPU usage mode ('auto', 'true', 'false')
            gpu_memory_limit: GPU memory limit in MB or fraction (0-1)
            gpu_device_id: Specific GPU device ID to use
            gpu_memory_growth: Enable memory growth to prevent OOM
            enable_mixed_precision: Enable mixed precision (float16) training
        """
        if not SKLEARN_AVAILABLE or not BASE_MODELS_AVAILABLE:
            raise ImportError("Required dependencies not available")

        self.use_xgboost = use_xgboost
        self.use_random_forest = use_random_forest
        self.use_lstm = use_lstm
        self.meta_learner_C = meta_learner_C
        self.random_state = random_state

        # GPU configuration
        self.use_gpu = use_gpu
        self.gpu_memory_limit = gpu_memory_limit
        self.gpu_device_id = gpu_device_id
        self.gpu_memory_growth = gpu_memory_growth
        self.enable_mixed_precision = enable_mixed_precision
        self._gpu_configured = False
        self._gpu_info: Dict[str, Any] = {}

        # Configure GPU if LSTM is used
        if use_lstm:
            self._configure_gpu()

        # Base models
        self.xgboost_model: Optional[XGBoostTradeClassifier] = None
        self.rf_model: Optional[RandomForestTradeClassifier] = None
        self.lstm_model: Optional[LSTMTradePredictor] = None

        # Meta-learner
        self.meta_learner: Optional[LogisticRegression] = None
        self.is_trained = False

        # Feature names
        self.feature_names: List[str] = []
        self.n_base_models = sum([use_xgboost, use_random_forest, use_lstm])

        # Metrics
        self.val_metrics: Optional[EnsembleMetrics] = None

        # Model weights (learned by meta-learner)
        self.model_weights: Dict[str, float] = {}

        # Baseline distributions for drift detection (set after training)
        self.baseline_feature_means: Optional[np.ndarray] = None
        self.baseline_feature_stds: Optional[np.ndarray] = None
        self.baseline_prediction_distribution: Optional[Dict[int, float]] = None
        self._training_X: Optional[np.ndarray] = None  # Keep reference for baseline

        # Model monitoring hook
        self._model_monitor = None
        self.version: str = "1.0.0"

        gpu_status = "GPU enabled" if self._gpu_configured else "CPU mode"
        logger.info(
            f"Initialized StackedEnsemble with {self.n_base_models} base models "
            f"(XGBoost: {use_xgboost}, RF: {use_random_forest}, LSTM: {use_lstm}) - {gpu_status}"
        )

    def _configure_gpu(self) -> bool:
        """
        Configure GPU for LSTM training.

        Returns:
            bool: True if GPU was configured successfully
        """
        if not GPU_UTILS_AVAILABLE:
            logger.info("GPU utilities not available, using CPU")
            return False

        try:
            result = setup_gpu_for_training(
                use_gpu=self.use_gpu,
                memory_limit=self.gpu_memory_limit,
                device_id=self.gpu_device_id,
                enable_mixed_prec=self.enable_mixed_precision
            )

            self._gpu_configured = result.get('gpu_configured', False)
            self._gpu_info = result

            if self._gpu_configured:
                device = result.get('device', 'GPU')
                logger.info(f"GPU configured successfully: {device}")

                # Log GPU info
                gpu_infos = get_gpu_info()
                for gpu in gpu_infos:
                    logger.info(
                        f"  GPU {gpu.device_id}: {gpu.name} "
                        f"({gpu.memory_total_mb:.0f}MB total, {gpu.memory_free_mb:.0f}MB free)"
                    )
            else:
                logger.info("No GPU available or GPU disabled, using CPU")

            return self._gpu_configured

        except Exception as e:
            logger.warning(f"Error configuring GPU: {e}. Falling back to CPU.")
            self._gpu_configured = False
            return False

    def get_gpu_status(self) -> Dict[str, Any]:
        """
        Get current GPU configuration status.

        Returns:
            Dict with GPU status information
        """
        if not GPU_UTILS_AVAILABLE:
            return {
                'gpu_available': False,
                'gpu_configured': False,
                'reason': 'GPU utilities not available'
            }

        status = get_gpu_summary()
        status['gpu_configured'] = self._gpu_configured
        status['config'] = {
            'use_gpu': self.use_gpu,
            'memory_limit': self.gpu_memory_limit,
            'device_id': self.gpu_device_id,
            'memory_growth': self.gpu_memory_growth,
            'mixed_precision': self.enable_mixed_precision
        }
        return status

    def _initialize_base_models(self, **model_kwargs):
        """Initialize base models with configurations"""
        if self.use_xgboost:
            xgb_kwargs = model_kwargs.get('xgboost', {})
            self.xgboost_model = XGBoostTradeClassifier(
                random_state=self.random_state,
                **xgb_kwargs
            )
            logger.info("Initialized XGBoost model")

        if self.use_random_forest:
            rf_kwargs = model_kwargs.get('random_forest', {})
            self.rf_model = RandomForestTradeClassifier(
                random_state=self.random_state,
                **rf_kwargs
            )
            logger.info("Initialized Random Forest model")

        if self.use_lstm:
            lstm_kwargs = model_kwargs.get('lstm', {})
            self.lstm_model = LSTMTradePredictor(
                random_state=self.random_state,
                **lstm_kwargs
            )
            logger.info("Initialized LSTM model")

    def _create_meta_learner(self) -> LogisticRegression:
        """Create Logistic Regression meta-learner"""
        return LogisticRegression(
            C=self.meta_learner_C,
            random_state=self.random_state,
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced'
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_lstm: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        validation_split: float = 0.2,
        cv_folds: int = 5,
        lstm_batch_size: Optional[int] = None,
        **model_kwargs
    ) -> EnsembleMetrics:
        """
        Train stacked ensemble with cross-validation

        Args:
            X: Feature array for tree models (n_samples, n_features)
            y: Target array (n_samples,) - binary 0/1
            X_lstm: Optional sequences for LSTM (n_samples, seq_len, n_features)
            feature_names: Optional feature names
            validation_split: Fraction for final validation
            cv_folds: Number of cross-validation folds for meta-learner
            lstm_batch_size: Batch size for LSTM training (None for auto)
            **model_kwargs: Additional kwargs for base models

        Returns:
            Validation metrics
        """
        logger.info(f"Training StackedEnsemble with {len(X)} samples")

        # Log GPU status for LSTM training
        if self.use_lstm and GPU_UTILS_AVAILABLE:
            gpu_status = self.get_gpu_status()
            if gpu_status.get('gpu_configured'):
                logger.info(f"LSTM training will use GPU: {gpu_status.get('device_strategy', 'unknown')}")

                # Auto-calculate batch size if not specified
                if lstm_batch_size is None and X_lstm is not None:
                    seq_len = X_lstm.shape[1] if len(X_lstm.shape) > 2 else 20
                    n_features = X_lstm.shape[2] if len(X_lstm.shape) > 2 else X_lstm.shape[1]
                    lstm_batch_size = get_optimal_batch_size(
                        model_size_mb=10.0,  # Approximate LSTM model size
                        sequence_length=seq_len,
                        n_features=n_features
                    )
                    logger.info(f"Auto-selected LSTM batch size: {lstm_batch_size}")
            else:
                logger.info("LSTM training will use CPU")

        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Initialize base models
        self._initialize_base_models(**model_kwargs)

        # Split data for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if X_lstm is not None:
            X_lstm_train, X_lstm_val = X_lstm[:split_idx], X_lstm[split_idx:]
        else:
            X_lstm_train, X_lstm_val = None, None

        # Step 1: Train base models
        logger.info("Step 1: Training base models...")
        base_predictions_train = self._train_base_models(
            X_train, y_train, X_lstm_train
        )
        base_predictions_val = self._get_base_predictions(
            X_val, X_lstm_val
        )

        logger.info(f"Base model predictions shape: {base_predictions_train.shape}")

        # Step 2: Train meta-learner on base model predictions
        logger.info("Step 2: Training meta-learner...")
        self.meta_learner = self._create_meta_learner()

        # Cross-validation for meta-learner
        cv_scores = cross_val_score(
            self.meta_learner,
            base_predictions_train,
            y_train,
            cv=cv_folds,
            scoring='roc_auc'
        )
        logger.info(f"Meta-learner CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Train final meta-learner
        self.meta_learner.fit(base_predictions_train, y_train)
        self.is_trained = True

        # Extract model weights from meta-learner coefficients
        self._update_model_weights()

        # Step 3: Calculate validation metrics
        self.val_metrics = self._calculate_metrics(
            base_predictions_val, y_val,
            X_val, y_val, X_lstm_val
        )

        # Step 4: Store baseline distributions for drift detection
        self._store_baseline_distributions(X, y)

        logger.info(
            f"Ensemble training complete. Val Accuracy: {self.val_metrics.accuracy:.4f}, "
            f"AUC: {self.val_metrics.auc:.4f}"
        )

        return self.val_metrics

    def _train_base_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_lstm: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Train all base models and return their predictions

        Returns:
            Stacked predictions (n_samples, n_base_models)
        """
        predictions = []

        # Train XGBoost
        if self.use_xgboost and self.xgboost_model is not None:
            logger.info("Training XGBoost...")
            self.xgboost_model.train(X, y, feature_names=self.feature_names)
            pred = self.xgboost_model.predict_success_probability(X)
            predictions.append(pred)
            logger.info(f"XGBoost trained. Val AUC: {self.xgboost_model.val_metrics.auc:.4f}")

        # Train Random Forest
        if self.use_random_forest and self.rf_model is not None:
            logger.info("Training Random Forest...")
            self.rf_model.train(X, y, feature_names=self.feature_names)
            pred = self.rf_model.predict_success_probability(X)
            predictions.append(pred)
            logger.info(f"Random Forest trained. Val AUC: {self.rf_model.val_metrics.auc:.4f}")

        # Train LSTM
        if self.use_lstm and self.lstm_model is not None and X_lstm is not None:
            logger.info("Training LSTM...")
            self.lstm_model.train(X_lstm, y)
            pred_proba = self.lstm_model.predict_proba(X_lstm)
            # For binary classification from 3-class output, use "up" probability
            pred = pred_proba[:, 0]  # Probability of "up"
            predictions.append(pred)
            logger.info(f"LSTM trained. Val Accuracy: {self.lstm_model.val_metrics.accuracy:.4f}")

        # Stack predictions horizontally
        stacked_predictions = np.column_stack(predictions)
        return stacked_predictions

    def _get_base_predictions(
        self,
        X: np.ndarray,
        X_lstm: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get predictions from all trained base models

        Returns:
            Stacked predictions (n_samples, n_base_models)
        """
        if not self.is_trained:
            raise ValueError("Base models not trained yet")

        predictions = []

        # XGBoost predictions
        if self.use_xgboost and self.xgboost_model is not None:
            pred = self.xgboost_model.predict_success_probability(X)
            predictions.append(pred)

        # Random Forest predictions
        if self.use_random_forest and self.rf_model is not None:
            pred = self.rf_model.predict_success_probability(X)
            predictions.append(pred)

        # LSTM predictions
        if self.use_lstm and self.lstm_model is not None and X_lstm is not None:
            pred_proba = self.lstm_model.predict_proba(X_lstm)
            pred = pred_proba[:, 0]  # Probability of "up"
            predictions.append(pred)

        return np.column_stack(predictions)

    def predict_proba(
        self,
        X: np.ndarray,
        X_lstm: Optional[np.ndarray] = None,
        record_for_monitoring: bool = True
    ) -> np.ndarray:
        """
        Predict probability of trade success

        Args:
            X: Feature array for tree models
            X_lstm: Optional sequences for LSTM
            record_for_monitoring: Whether to record predictions for drift monitoring

        Returns:
            Probability array (n_samples, 2) for [prob_failure, prob_success]
        """
        if not self.is_trained or self.meta_learner is None:
            raise ValueError("Ensemble not trained. Call train() first.")

        # Get base model predictions
        base_predictions = self._get_base_predictions(X, X_lstm)

        # Meta-learner prediction
        probas = self.meta_learner.predict_proba(base_predictions)

        # Record predictions for monitoring if monitor is attached
        if record_for_monitoring and self._model_monitor is not None:
            try:
                for i in range(len(X)):
                    prediction = probas[i, 1]  # Success probability
                    predicted_class = 1 if prediction >= 0.5 else 0
                    self._model_monitor.record_prediction(
                        features=X[i],
                        prediction=float(prediction),
                        predicted_class=predicted_class
                    )
            except Exception as e:
                logger.debug(f"Failed to record predictions for monitoring: {e}")

        return probas

    def predict(
        self,
        X: np.ndarray,
        X_lstm: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict class labels

        Args:
            X: Feature array for tree models
            X_lstm: Optional sequences for LSTM

        Returns:
            Predicted labels (0 or 1)
        """
        probas = self.predict_proba(X, X_lstm)
        return (probas[:, 1] > 0.5).astype(int)

    def predict_success_probability(
        self,
        X: np.ndarray,
        X_lstm: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict probability of trade success (reaching 2R)

        Args:
            X: Feature array for tree models
            X_lstm: Optional sequences for LSTM

        Returns:
            Success probabilities (n_samples,)
        """
        probas = self.predict_proba(X, X_lstm)
        return probas[:, 1]  # Probability of class 1 (success)

    def predict_with_failover(
        self,
        X: np.ndarray,
        X_lstm: Optional[np.ndarray] = None,
        fallback_probability: float = 0.5,
        min_models_required: int = 1
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict with automatic failover when models fail.

        This method catches individual model failures and:
        1. Uses remaining healthy models when some fail
        2. Adjusts weights proportionally for available models
        3. Falls back to heuristic probability when all models fail

        Args:
            X: Feature array for tree models
            X_lstm: Optional sequences for LSTM
            fallback_probability: Default probability when all models fail
            min_models_required: Minimum models needed (below this, use fallback)

        Returns:
            Tuple of (probabilities, status_dict) where status_dict contains:
                - models_used: List of models that contributed
                - models_failed: List of models that failed
                - used_fallback: Whether fallback was used
                - adjusted_weights: Weights used for prediction
        """
        n_samples = len(X)
        predictions = []
        model_names = []
        failed_models = []

        status = {
            "models_used": [],
            "models_failed": [],
            "used_fallback": False,
            "adjusted_weights": {},
            "error_messages": {}
        }

        # Try XGBoost
        if self.use_xgboost and self.xgboost_model is not None:
            try:
                pred = self.xgboost_model.predict_success_probability(X)
                predictions.append(pred)
                model_names.append('xgboost')
                status["models_used"].append('xgboost')
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}")
                failed_models.append('xgboost')
                status["models_failed"].append('xgboost')
                status["error_messages"]['xgboost'] = str(e)

        # Try Random Forest
        if self.use_random_forest and self.rf_model is not None:
            try:
                pred = self.rf_model.predict_success_probability(X)
                predictions.append(pred)
                model_names.append('random_forest')
                status["models_used"].append('random_forest')
            except Exception as e:
                logger.warning(f"Random Forest prediction failed: {e}")
                failed_models.append('random_forest')
                status["models_failed"].append('random_forest')
                status["error_messages"]['random_forest'] = str(e)

        # Try LSTM
        if self.use_lstm and self.lstm_model is not None and X_lstm is not None:
            try:
                pred_proba = self.lstm_model.predict_proba(X_lstm)
                pred = pred_proba[:, 0]  # Probability of "up"
                predictions.append(pred)
                model_names.append('lstm')
                status["models_used"].append('lstm')
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
                failed_models.append('lstm')
                status["models_failed"].append('lstm')
                status["error_messages"]['lstm'] = str(e)

        # Check if we have enough models
        if len(predictions) < min_models_required:
            logger.warning(
                f"Only {len(predictions)} models available, "
                f"minimum required is {min_models_required}. Using fallback."
            )
            status["used_fallback"] = True
            return np.full(n_samples, fallback_probability), status

        # If all models failed, use fallback
        if len(predictions) == 0:
            logger.error("All models failed. Using fallback probability.")
            status["used_fallback"] = True
            return np.full(n_samples, fallback_probability), status

        # Calculate adjusted weights for available models
        if self.meta_learner is not None and len(predictions) > 1:
            # Get original weights and adjust for available models
            original_weights = self.model_weights.copy() if self.model_weights else {}

            # Extract weights for available models and renormalize
            available_weights = []
            for name in model_names:
                w = original_weights.get(name, 1.0 / len(model_names))
                available_weights.append(w)

            # Normalize to sum to 1
            total_weight = sum(available_weights)
            if total_weight > 0:
                available_weights = [w / total_weight for w in available_weights]
            else:
                available_weights = [1.0 / len(model_names)] * len(model_names)

            status["adjusted_weights"] = dict(zip(model_names, available_weights))

            # Weighted average of predictions
            stacked = np.column_stack(predictions)
            probabilities = np.average(stacked, axis=1, weights=available_weights)

        elif len(predictions) == 1:
            # Single model, use directly
            probabilities = predictions[0]
            status["adjusted_weights"] = {model_names[0]: 1.0}
        else:
            # Simple average
            stacked = np.column_stack(predictions)
            probabilities = np.mean(stacked, axis=1)
            status["adjusted_weights"] = {name: 1.0/len(model_names) for name in model_names}

        # Log status
        if failed_models:
            logger.info(
                f"Prediction with failover: used {model_names}, "
                f"failed: {failed_models}"
            )

        return probabilities, status

    def _heuristic_prediction(
        self,
        X: np.ndarray,
        rrs_column_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Fallback heuristic prediction when models fail.

        Uses simple rules based on RRS (Relative Strength) if available,
        otherwise returns neutral probability.

        Args:
            X: Feature array
            rrs_column_idx: Index of RRS column in features (if known)

        Returns:
            Heuristic probabilities
        """
        n_samples = len(X)

        # If we know the RRS column, use it for simple heuristic
        if rrs_column_idx is not None and rrs_column_idx < X.shape[1]:
            rrs = X[:, rrs_column_idx]
            # Simple sigmoid-like mapping: RRS > 2 = higher prob, < -2 = lower prob
            probabilities = 1 / (1 + np.exp(-rrs))
            return probabilities

        # Default: neutral probability
        return np.full(n_samples, 0.5)

    def _update_model_weights(self):
        """Extract and normalize model weights from meta-learner"""
        if self.meta_learner is None:
            return

        # Get coefficients from logistic regression
        coefficients = self.meta_learner.coef_[0]

        # Apply softmax to get normalized weights
        weights = np.exp(coefficients) / np.sum(np.exp(coefficients))

        model_names = []
        if self.use_xgboost:
            model_names.append('xgboost')
        if self.use_random_forest:
            model_names.append('random_forest')
        if self.use_lstm:
            model_names.append('lstm')

        self.model_weights = dict(zip(model_names, weights.tolist()))

        logger.info(f"Model weights: {self.model_weights}")

    def _store_baseline_distributions(self, X: np.ndarray, y: np.ndarray):
        """
        Store baseline feature and prediction distributions for drift detection.

        Args:
            X: Training features
            y: Training labels
        """
        # Store feature distributions
        self.baseline_feature_means = np.mean(X, axis=0)
        self.baseline_feature_stds = np.std(X, axis=0)
        self._training_X = X.copy()  # Keep copy for detailed baseline

        # Store prediction/label distribution
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        self.baseline_prediction_distribution = {
            int(c): count / total for c, count in zip(unique, counts)
        }

        logger.info(
            f"Stored baseline distributions: {len(self.feature_names)} features, "
            f"class distribution: {self.baseline_prediction_distribution}"
        )

    def get_baseline_distributions(self) -> Dict[str, Any]:
        """
        Get baseline distributions for drift detection.

        Returns:
            Dictionary containing feature and prediction baselines
        """
        if self.baseline_feature_means is None:
            return {}

        feature_distributions = {}
        for i, name in enumerate(self.feature_names):
            feature_distributions[name] = {
                'mean': float(self.baseline_feature_means[i]),
                'std': float(self.baseline_feature_stds[i]),
            }

            # Add percentiles if training data is available
            if self._training_X is not None:
                feature_values = self._training_X[:, i]
                feature_distributions[name].update({
                    'min': float(np.min(feature_values)),
                    'max': float(np.max(feature_values)),
                    'percentiles': {
                        '5': float(np.percentile(feature_values, 5)),
                        '25': float(np.percentile(feature_values, 25)),
                        '50': float(np.percentile(feature_values, 50)),
                        '75': float(np.percentile(feature_values, 75)),
                        '95': float(np.percentile(feature_values, 95)),
                    }
                })

        return {
            'feature_distributions': feature_distributions,
            'prediction_distribution': self.baseline_prediction_distribution,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names
        }

    def set_model_monitor(self, monitor):
        """
        Set model monitor for tracking predictions.

        Args:
            monitor: ModelMonitor instance
        """
        self._model_monitor = monitor
        logger.info(f"Model monitor attached to StackedEnsemble")

    def _calculate_metrics(
        self,
        base_predictions: np.ndarray,
        y_true: np.ndarray,
        X_original: np.ndarray,
        y_original: np.ndarray,
        X_lstm: Optional[np.ndarray] = None
    ) -> EnsembleMetrics:
        """Calculate comprehensive metrics for ensemble"""
        if self.meta_learner is None:
            return EnsembleMetrics()

        # Ensemble predictions
        y_pred_proba = self.meta_learner.predict_proba(base_predictions)
        y_pred = self.meta_learner.predict(base_predictions)

        # Calculate base model performance
        base_performance = {}

        if self.use_xgboost and self.xgboost_model is not None:
            xgb_pred = self.xgboost_model.predict(X_original)
            base_performance['xgboost'] = {
                'accuracy': accuracy_score(y_original, xgb_pred),
                'auc': self.xgboost_model.val_metrics.auc if self.xgboost_model.val_metrics else 0.0
            }

        if self.use_random_forest and self.rf_model is not None:
            rf_pred = self.rf_model.predict(X_original)
            base_performance['random_forest'] = {
                'accuracy': accuracy_score(y_original, rf_pred),
                'auc': self.rf_model.val_metrics.auc if self.rf_model.val_metrics else 0.0
            }

        if self.use_lstm and self.lstm_model is not None and X_lstm is not None:
            lstm_pred = self.lstm_model.predict(X_lstm)
            # Map 3-class to binary (0=up=success, 1,2=neutral/down=failure)
            lstm_pred_binary = (lstm_pred == 0).astype(int)
            base_performance['lstm'] = {
                'accuracy': accuracy_score(y_original, lstm_pred_binary),
                'auc': 0.0  # Would need proper calculation for multi-class
            }

        return EnsembleMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            auc=roc_auc_score(y_true, y_pred_proba[:, 1]),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1_score=f1_score(y_true, y_pred, zero_division=0),
            log_loss=log_loss(y_true, y_pred_proba),
            confusion_matrix=confusion_matrix(y_true, y_pred),
            model_weights=self.model_weights,
            base_model_performance=base_performance
        )

    def get_base_model_predictions(
        self,
        X: np.ndarray,
        X_lstm: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get individual predictions from each base model

        Args:
            X: Feature array for tree models
            X_lstm: Optional sequences for LSTM

        Returns:
            Dictionary mapping model names to prediction arrays
        """
        predictions = {}

        if self.use_xgboost and self.xgboost_model is not None:
            predictions['xgboost'] = self.xgboost_model.predict_success_probability(X)

        if self.use_random_forest and self.rf_model is not None:
            predictions['random_forest'] = self.rf_model.predict_success_probability(X)

        if self.use_lstm and self.lstm_model is not None and X_lstm is not None:
            lstm_proba = self.lstm_model.predict_proba(X_lstm)
            predictions['lstm'] = lstm_proba[:, 0]

        return predictions

    def get_feature_importance(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance from all base models

        Returns:
            Dictionary mapping model names to feature importance lists
        """
        importance = {}

        if self.use_xgboost and self.xgboost_model is not None:
            importance['xgboost'] = self.xgboost_model.get_top_features(10)

        if self.use_random_forest and self.rf_model is not None:
            importance['random_forest'] = self.rf_model.get_top_features(10)

        return importance

    def save(self, path: str):
        """
        Save ensemble and all base models

        Args:
            path: Base path for saving (will create subdirectory)
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Nothing to save.")

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save base models
        if self.use_xgboost and self.xgboost_model is not None:
            self.xgboost_model.save(str(save_path / "xgboost_model.pkl"))

        if self.use_random_forest and self.rf_model is not None:
            self.rf_model.save(str(save_path / "random_forest_model.pkl"))

        if self.use_lstm and self.lstm_model is not None:
            self.lstm_model.save(str(save_path / "lstm_model"))

        # Save meta-learner and ensemble metadata
        ensemble_data = {
            'meta_learner': self.meta_learner,
            'model_weights': self.model_weights,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'val_metrics': self.val_metrics,
            'config': {
                'use_xgboost': self.use_xgboost,
                'use_random_forest': self.use_random_forest,
                'use_lstm': self.use_lstm,
                'meta_learner_C': self.meta_learner_C,
                'random_state': self.random_state,
                'n_base_models': self.n_base_models
            }
        }

        ensemble_path = save_path / "ensemble_meta.pkl"
        safe_save_model(ensemble_data, str(ensemble_path))

        logger.info(f"Ensemble saved to {save_path}")

    def load(self, path: str):
        """
        Load ensemble and all base models

        Args:
            path: Base path for loading
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Ensemble path not found: {load_path}")

        # Load ensemble metadata
        ensemble_path = load_path / "ensemble_meta.pkl"
        if not ensemble_path.exists():
            raise FileNotFoundError(f"Ensemble metadata not found: {ensemble_path}")

        ensemble_data = safe_load_model(str(ensemble_path), allow_unverified=False)

        self.meta_learner = ensemble_data['meta_learner']
        self.model_weights = ensemble_data['model_weights']
        self.feature_names = ensemble_data['feature_names']
        self.is_trained = ensemble_data['is_trained']
        self.val_metrics = ensemble_data.get('val_metrics')

        config = ensemble_data.get('config', {})
        self.use_xgboost = config.get('use_xgboost', self.use_xgboost)
        self.use_random_forest = config.get('use_random_forest', self.use_random_forest)
        self.use_lstm = config.get('use_lstm', self.use_lstm)
        self.meta_learner_C = config.get('meta_learner_C', self.meta_learner_C)
        self.random_state = config.get('random_state', self.random_state)
        self.n_base_models = config.get('n_base_models', self.n_base_models)

        # Load base models
        if self.use_xgboost:
            xgb_path = load_path / "xgboost_model.pkl"
            if xgb_path.exists():
                self.xgboost_model = XGBoostTradeClassifier()
                self.xgboost_model.load(str(xgb_path))

        if self.use_random_forest:
            rf_path = load_path / "random_forest_model.pkl"
            if rf_path.exists():
                self.rf_model = RandomForestTradeClassifier()
                self.rf_model.load(str(rf_path))

        if self.use_lstm:
            lstm_path = load_path / "lstm_model"
            if lstm_path.with_suffix('.h5').exists():
                self.lstm_model = LSTMTradePredictor()
                self.lstm_model.load(str(lstm_path))

        logger.info(f"Ensemble loaded from {load_path}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        if self.val_metrics is None:
            return {}

        summary = {
            'ensemble': self.val_metrics.to_dict(),
            'model_weights': self.model_weights,
            'n_base_models': self.n_base_models,
            'is_trained': self.is_trained,
            'n_features': len(self.feature_names)
        }

        # Add base model summaries
        if self.use_xgboost and self.xgboost_model is not None:
            summary['xgboost'] = self.xgboost_model.get_metrics_summary()

        if self.use_random_forest and self.rf_model is not None:
            summary['random_forest'] = self.rf_model.get_metrics_summary()

        if self.use_lstm and self.lstm_model is not None:
            summary['lstm'] = self.lstm_model.get_metrics_summary()

        return summary

    def load_optimized_params(
        self,
        config_path: str = "config/optimized_params.json"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load optimized parameters from config file and apply to models.

        Loads hyperparameters that were optimized using Optuna and configures
        the base models accordingly.

        Args:
            config_path: Path to the optimized parameters JSON file

        Returns:
            Dictionary of loaded parameters per model type

        Example:
            ```python
            ensemble = StackedEnsemble()
            params = ensemble.load_optimized_params("config/optimized_params.json")
            ensemble.train(X, y)  # Will use optimized params
            ```
        """
        import json
        from pathlib import Path

        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Optimized params file not found: {config_path}")
            return {}

        with open(config_file, 'r') as f:
            config = json.load(f)

        loaded_params = {}

        # Load XGBoost params
        if self.use_xgboost and 'xgboost' in config:
            xgb_config = config['xgboost']
            params = xgb_config.get('params', {})
            loaded_params['xgboost'] = params

            # Update XGBoost model parameters if model exists
            if self.xgboost_model is not None:
                self.xgboost_model.n_estimators = params.get('n_estimators', self.xgboost_model.n_estimators)
                self.xgboost_model.max_depth = params.get('max_depth', self.xgboost_model.max_depth)
                self.xgboost_model.learning_rate = params.get('learning_rate', self.xgboost_model.learning_rate)

            logger.info(f"Loaded optimized XGBoost params: {params}")

        # Load Random Forest params
        if self.use_random_forest and 'random_forest' in config:
            rf_config = config['random_forest']
            params = rf_config.get('params', {})
            loaded_params['random_forest'] = params

            # Update Random Forest model parameters if model exists
            if self.rf_model is not None:
                self.rf_model.n_estimators = params.get('n_estimators', self.rf_model.n_estimators)
                self.rf_model.max_depth = params.get('max_depth', self.rf_model.max_depth)
                self.rf_model.min_samples_split = params.get('min_samples_split', self.rf_model.min_samples_split)
                self.rf_model.min_samples_leaf = params.get('min_samples_leaf', self.rf_model.min_samples_leaf)

            logger.info(f"Loaded optimized Random Forest params: {params}")

        # Load LSTM params
        if self.use_lstm and 'lstm' in config:
            lstm_config = config['lstm']
            params = lstm_config.get('params', {})
            loaded_params['lstm'] = params

            # Update LSTM model parameters if model exists
            if self.lstm_model is not None:
                units = params.get('units', 64)
                n_layers = params.get('layers', 2)
                self.lstm_model.lstm_units = [units // (2 ** i) for i in range(n_layers)]
                self.lstm_model.dropout = params.get('dropout', self.lstm_model.dropout)
                self.lstm_model.learning_rate = params.get('learning_rate', self.lstm_model.learning_rate)

            logger.info(f"Loaded optimized LSTM params: {params}")

        # Store config metadata
        self._optimized_params_config = config
        self._optimized_params_loaded = True

        logger.info(f"Loaded optimized parameters for {len(loaded_params)} models from {config_path}")
        return loaded_params

    def get_model_kwargs_from_optimized_params(
        self,
        config_path: str = "config/optimized_params.json"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get model kwargs from optimized params for use in train().

        Returns kwargs that can be passed to train() method's **model_kwargs.

        Args:
            config_path: Path to the optimized parameters JSON file

        Returns:
            Dictionary ready to be passed as **model_kwargs to train()

        Example:
            ```python
            ensemble = StackedEnsemble()
            model_kwargs = ensemble.get_model_kwargs_from_optimized_params()
            ensemble.train(X, y, **model_kwargs)
            ```
        """
        import json
        from pathlib import Path

        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Optimized params file not found: {config_path}")
            return {}

        with open(config_file, 'r') as f:
            config = json.load(f)

        model_kwargs = {}

        if 'xgboost' in config:
            params = config['xgboost'].get('params', {})
            model_kwargs['xgboost'] = {
                'n_estimators': params.get('n_estimators', 100),
                'max_depth': params.get('max_depth', 6),
                'learning_rate': params.get('learning_rate', 0.1),
            }

        if 'random_forest' in config:
            params = config['random_forest'].get('params', {})
            model_kwargs['random_forest'] = {
                'n_estimators': params.get('n_estimators', 100),
                'max_depth': params.get('max_depth', 15),
                'min_samples_split': params.get('min_samples_split', 5),
                'min_samples_leaf': params.get('min_samples_leaf', 2),
            }

        if 'lstm' in config:
            params = config['lstm'].get('params', {})
            units = params.get('units', 64)
            n_layers = params.get('layers', 2)
            model_kwargs['lstm'] = {
                'lstm_units': [units // (2 ** i) for i in range(n_layers)],
                'dropout': params.get('dropout', 0.2),
                'learning_rate': params.get('learning_rate', 0.001),
            }

        return model_kwargs


# Legacy Ensemble class for backward compatibility
class Ensemble:
    """
    Simple ensemble for basic predictions (backward compatibility)

    For production use, prefer StackedEnsemble.
    """

    def __init__(self):
        """Initialize ensemble model"""
        self.weights = {
            'rrs_score': 0.30,
            'trend_alignment': 0.25,
            'volatility_score': 0.20,
            'technical_score': 0.25
        }
        logger.info("Legacy Ensemble model initialized")

    def predict(self, features: np.ndarray, signal: Dict) -> float:
        """
        Predict probability of successful trade

        Args:
            features: Feature array from FeatureEngineer
            signal: Original signal data

        Returns:
            Probability score (0-100)
        """
        rrs = signal.get('rrs', 0)
        atr = signal.get('atr', 0)
        price = signal.get('price', 1)
        direction = signal.get('direction', 'long')
        daily_strong = signal.get('daily_strong', False)
        daily_weak = signal.get('daily_weak', False)

        rrs_score = self._calculate_rrs_score(rrs)
        trend_score = self._calculate_trend_alignment(direction, daily_strong, daily_weak, rrs)
        atr_percent = (atr / price * 100) if price > 0 else 0
        volatility_score = self._calculate_volatility_score(atr_percent)
        technical_score = self._calculate_technical_score(features)

        probability = (
            self.weights['rrs_score'] * rrs_score +
            self.weights['trend_alignment'] * trend_score +
            self.weights['volatility_score'] * volatility_score +
            self.weights['technical_score'] * technical_score
        )

        return max(0, min(100, probability))

    def _calculate_rrs_score(self, rrs: float) -> float:
        abs_rrs = abs(rrs)
        if abs_rrs >= 4.0:
            return 95.0
        elif abs_rrs >= 3.5:
            return 85.0
        elif abs_rrs >= 3.0:
            return 75.0
        elif abs_rrs >= 2.5:
            return 65.0
        elif abs_rrs >= 2.0:
            return 50.0
        return 30.0

    def _calculate_trend_alignment(
        self, direction: str, daily_strong: bool, daily_weak: bool, rrs: float
    ) -> float:
        score = 50.0
        if direction == 'long':
            if daily_strong and rrs > 0:
                score = 90.0
            elif daily_strong or rrs > 0:
                score = 70.0
            elif daily_weak or rrs < 0:
                score = 30.0
        elif direction == 'short':
            if daily_weak and rrs < 0:
                score = 90.0
            elif daily_weak or rrs < 0:
                score = 70.0
            elif daily_strong or rrs > 0:
                score = 30.0
        return score

    def _calculate_volatility_score(self, atr_percent: float) -> float:
        if 1.5 <= atr_percent <= 3.0:
            return 90.0
        elif 1.0 <= atr_percent < 1.5:
            return 75.0
        elif 3.0 < atr_percent <= 4.0:
            return 70.0
        elif 0.5 <= atr_percent < 1.0:
            return 60.0
        elif 4.0 < atr_percent <= 5.0:
            return 50.0
        return 30.0

    def _calculate_technical_score(self, features: np.ndarray) -> float:
        if len(features) < 7:
            return 50.0
        trend_strength = features[6] if len(features) > 6 else 0.5
        volume_ratio = features[5] if len(features) > 5 else 1.0
        score = 50.0 + (trend_strength * 30.0)
        if volume_ratio > 1.5:
            score += 15.0
        elif volume_ratio > 1.2:
            score += 10.0
        return min(score, 100.0)
