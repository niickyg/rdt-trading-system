"""
XGBoost Classifier for Trade Success Prediction

Predicts probability of trade success (will it reach 2R within 10 days?)
Uses time-series cross-validation and tracks feature importance.
Includes distributed tracing for model training and prediction.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger

# Import distributed tracing
try:
    from tracing import trace, get_tracer, get_current_span
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    # Create no-op decorators
    def trace(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def get_tracer():
        return None
    def get_current_span():
        return None

try:
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, precision_score,
        recall_score, f1_score, confusion_matrix
    )
    import joblib
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost scikit-learn joblib")

from ml.safe_model_loader import safe_load_model, safe_save_model, ModelSecurityError


@dataclass
class XGBoostMetrics:
    """Validation metrics for XGBoost model"""
    accuracy: float = 0.0
    auc: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'accuracy': self.accuracy,
            'auc': self.auc,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            'feature_importance': self.feature_importance,
            'timestamp': self.timestamp.isoformat()
        }


class XGBoostTradeClassifier:
    """
    XGBoost classifier for predicting trade success probability

    Predicts binary outcome: Will trade reach 2R within 10 days?

    Features expected:
    - RRS value
    - ATR percentage
    - Price momentum (various periods)
    - Volume metrics
    - Technical indicators
    - Risk/Reward ratio
    - Market conditions (SPY trend, volatility)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        use_gpu: bool = False,
        random_state: int = 42
    ):
        """
        Initialize XGBoost classifier

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            use_gpu: Whether to use GPU acceleration
            random_state: Random seed for reproducibility
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost scikit-learn joblib")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.random_state = random_state

        # Model
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.is_trained = False

        # Metrics
        self.train_metrics: Optional[XGBoostMetrics] = None
        self.val_metrics: Optional[XGBoostMetrics] = None

        # Feature importance tracking
        self.feature_importance_history: List[Dict[str, float]] = []

        logger.info(f"Initialized XGBoostTradeClassifier (GPU: {use_gpu})")

    def _create_model(self) -> xgb.XGBClassifier:
        """Create XGBoost classifier with configured parameters"""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': self.random_state,
            'n_jobs': -1,
        }

        if self.use_gpu:
            try:
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = 0
                logger.info("GPU acceleration enabled")
            except Exception as e:
                logger.warning(f"GPU not available, falling back to CPU: {e}")
                params['tree_method'] = 'hist'
        else:
            params['tree_method'] = 'hist'

        return xgb.XGBClassifier(**params)

    def prepare_features(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for training/prediction

        Args:
            data: DataFrame with features
            feature_columns: Optional list of feature column names

        Returns:
            Tuple of (feature array, feature names)
        """
        if feature_columns is None:
            # Use all numeric columns except target
            feature_columns = [col for col in data.columns
                             if col not in ['target', 'success', 'symbol', 'date', 'timestamp']]

        # Handle missing values
        data_clean = data[feature_columns].fillna(0)

        return data_clean.values, feature_columns

    @trace("ml.train", attributes={"ml.model": "xgboost", "ml.operation": "train"})
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_splits: int = 5,
        validation_split: float = 0.2
    ) -> XGBoostMetrics:
        """
        Train model with time-series cross-validation

        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,) - binary 0/1
            feature_names: Optional feature names
            n_splits: Number of CV splits
            validation_split: Fraction for final validation

        Returns:
            Validation metrics
        """
        # Add tracing attributes
        span = get_current_span()
        if span:
            span.set_attribute("ml.samples", len(X))
            span.set_attribute("ml.features", X.shape[1])
            span.set_attribute("ml.n_splits", n_splits)
            span.set_attribute("ml.validation_split", validation_split)
            span.set_attribute("ml.n_estimators", self.n_estimators)
            span.set_attribute("ml.max_depth", self.max_depth)
            span.set_attribute("ml.learning_rate", self.learning_rate)

        logger.info(f"Training XGBoost with {len(X)} samples, {X.shape[1]} features")

        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Split data for final validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        logger.info(f"Running {n_splits}-fold time-series cross-validation")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

            model = self._create_model()
            model.fit(
                X_cv_train, y_cv_train,
                eval_set=[(X_cv_val, y_cv_val)],
                verbose=False
            )

            y_pred = model.predict(X_cv_val)
            y_pred_proba = model.predict_proba(X_cv_val)[:, 1]

            acc = accuracy_score(y_cv_val, y_pred)
            auc = roc_auc_score(y_cv_val, y_pred_proba)
            cv_scores.append({'fold': fold, 'accuracy': acc, 'auc': auc})

            logger.debug(f"Fold {fold}: Accuracy={acc:.4f}, AUC={auc:.4f}")

        # Log CV results
        avg_acc = np.mean([s['accuracy'] for s in cv_scores])
        avg_auc = np.mean([s['auc'] for s in cv_scores])
        logger.info(f"CV Results: Avg Accuracy={avg_acc:.4f}, Avg AUC={avg_auc:.4f}")

        # Train final model on all training data
        logger.info("Training final model on full training set")
        self.model = self._create_model()
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        self.is_trained = True

        # Calculate validation metrics
        self.val_metrics = self._calculate_metrics(X_val, y_val)

        # Update feature importance
        self._update_feature_importance()

        # Add final metrics to span
        if span:
            span.set_attribute("ml.val_accuracy", self.val_metrics.accuracy)
            span.set_attribute("ml.val_auc", self.val_metrics.auc)
            span.set_attribute("ml.val_precision", self.val_metrics.precision)
            span.set_attribute("ml.val_recall", self.val_metrics.recall)
            span.set_attribute("ml.val_f1", self.val_metrics.f1_score)

        logger.info(f"Training complete. Val Accuracy: {self.val_metrics.accuracy:.4f}, AUC: {self.val_metrics.auc:.4f}")

        return self.val_metrics

    @trace("ml.predict", attributes={"ml.model": "xgboost", "ml.operation": "predict"})
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels

        Args:
            X: Feature array

        Returns:
            Predicted labels (0 or 1)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Add tracing attributes
        span = get_current_span()
        if span:
            span.set_attribute("ml.samples", len(X))
            span.set_attribute("ml.features", X.shape[1] if len(X.shape) > 1 else 1)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Feature array

        Returns:
            Probability array (n_samples, 2) for [prob_failure, prob_success]
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict_proba(X)

    def predict_success_probability(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of trade success (reaching 2R)

        Args:
            X: Feature array

        Returns:
            Success probabilities (n_samples,)
        """
        probas = self.predict_proba(X)
        return probas[:, 1]  # Probability of class 1 (success)

    def _calculate_metrics(
        self,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> XGBoostMetrics:
        """Calculate comprehensive metrics"""
        if self.model is None:
            return XGBoostMetrics()

        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        return XGBoostMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            auc=roc_auc_score(y_true, y_pred_proba),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1_score=f1_score(y_true, y_pred, zero_division=0),
            confusion_matrix=confusion_matrix(y_true, y_pred),
            feature_importance=self._get_feature_importance()
        )

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model is None:
            return {}

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance.tolist()))

    def _update_feature_importance(self):
        """Update feature importance history"""
        importance = self._get_feature_importance()
        self.feature_importance_history.append({
            'timestamp': datetime.now().isoformat(),
            'importance': importance
        })

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most important features

        Args:
            n: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if not self.is_trained:
            return []

        importance = self._get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    def save(self, path: str):
        """
        Save model to disk

        Args:
            path: Path to save model file
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Nothing to save.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'val_metrics': self.val_metrics,
            'feature_importance_history': self.feature_importance_history,
            'config': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'use_gpu': self.use_gpu,
                'random_state': self.random_state
            }
        }

        safe_save_model(model_data, str(save_path))
        logger.info(f"Model saved to {save_path}")

    def load(self, path: str):
        """
        Load model from disk

        Args:
            path: Path to model file
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        model_data = safe_load_model(str(load_path), allow_unverified=False)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        self.val_metrics = model_data.get('val_metrics')
        self.feature_importance_history = model_data.get('feature_importance_history', [])

        config = model_data.get('config', {})
        self.n_estimators = config.get('n_estimators', self.n_estimators)
        self.max_depth = config.get('max_depth', self.max_depth)
        self.learning_rate = config.get('learning_rate', self.learning_rate)
        self.use_gpu = config.get('use_gpu', self.use_gpu)
        self.random_state = config.get('random_state', self.random_state)

        logger.info(f"Model loaded from {load_path}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of model metrics"""
        if self.val_metrics is None:
            return {}

        return {
            'validation': self.val_metrics.to_dict(),
            'top_features': self.get_top_features(10),
            'is_trained': self.is_trained,
            'n_features': len(self.feature_names)
        }
