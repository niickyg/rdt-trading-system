"""
Random Forest Classifier for Trade Success Prediction

Provides ensemble diversity by using a different algorithm than XGBoost.
Predicts probability of trade success (will it reach 2R within 10 days?)
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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, precision_score,
        recall_score, f1_score, confusion_matrix
    )
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn joblib")


from ml.safe_model_loader import safe_load_model, safe_save_model, ModelSecurityError

@dataclass
class RandomForestMetrics:
    """Validation metrics for Random Forest model"""
    accuracy: float = 0.0
    auc: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    oob_score: Optional[float] = None
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
            'oob_score': self.oob_score,
            'timestamp': self.timestamp.isoformat()
        }


class RandomForestTradeClassifier:
    """
    Random Forest classifier for predicting trade success probability

    Uses bagging ensemble for robustness and diversity from XGBoost.
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
        max_depth: Optional[int] = 15,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = 'sqrt',
        oob_score: bool = True,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize Random Forest classifier

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum tree depth (None for unlimited)
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            max_features: Number of features for best split
            oob_score: Whether to use out-of-bag samples for validation
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available. Install with: pip install scikit-learn joblib")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Model
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = []
        self.is_trained = False

        # Metrics
        self.train_metrics: Optional[RandomForestMetrics] = None
        self.val_metrics: Optional[RandomForestMetrics] = None

        # Feature importance tracking
        self.feature_importance_history: List[Dict[str, float]] = []

        logger.info(f"Initialized RandomForestTradeClassifier with {n_estimators} trees")

    def _create_model(self) -> RandomForestClassifier:
        """Create Random Forest classifier with configured parameters"""
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            oob_score=self.oob_score,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0,
            class_weight='balanced'  # Handle class imbalance
        )

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

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_splits: int = 5,
        validation_split: float = 0.2
    ) -> RandomForestMetrics:
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
        logger.info(f"Training Random Forest with {len(X)} samples, {X.shape[1]} features")

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
            model.fit(X_cv_train, y_cv_train)

            y_pred = model.predict(X_cv_val)
            y_pred_proba = model.predict_proba(X_cv_val)[:, 1]

            acc = accuracy_score(y_cv_val, y_pred)
            auc = roc_auc_score(y_cv_val, y_pred_proba)

            # OOB score if available
            oob = model.oob_score_ if self.oob_score else None

            cv_scores.append({
                'fold': fold,
                'accuracy': acc,
                'auc': auc,
                'oob_score': oob
            })

            logger.debug(f"Fold {fold}: Accuracy={acc:.4f}, AUC={auc:.4f}" +
                        (f", OOB={oob:.4f}" if oob else ""))

        # Log CV results
        avg_acc = np.mean([s['accuracy'] for s in cv_scores])
        avg_auc = np.mean([s['auc'] for s in cv_scores])
        logger.info(f"CV Results: Avg Accuracy={avg_acc:.4f}, Avg AUC={avg_auc:.4f}")

        # Train final model on all training data
        logger.info("Training final model on full training set")
        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        self.is_trained = True

        # Calculate validation metrics
        self.val_metrics = self._calculate_metrics(X_val, y_val)

        # Update feature importance
        self._update_feature_importance()

        logger.info(f"Training complete. Val Accuracy: {self.val_metrics.accuracy:.4f}, AUC: {self.val_metrics.auc:.4f}")
        if self.oob_score and self.model.oob_score_:
            logger.info(f"OOB Score: {self.model.oob_score_:.4f}")

        return self.val_metrics

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
    ) -> RandomForestMetrics:
        """Calculate comprehensive metrics"""
        if self.model is None:
            return RandomForestMetrics()

        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        oob = None
        if self.oob_score and hasattr(self.model, 'oob_score_'):
            oob = self.model.oob_score_

        return RandomForestMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            auc=roc_auc_score(y_true, y_pred_proba),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1_score=f1_score(y_true, y_pred, zero_division=0),
            confusion_matrix=confusion_matrix(y_true, y_pred),
            feature_importance=self._get_feature_importance(),
            oob_score=oob
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

    def get_tree_stats(self) -> Dict[str, Any]:
        """Get statistics about the forest trees"""
        if not self.is_trained or self.model is None:
            return {}

        depths = [tree.get_depth() for tree in self.model.estimators_]
        n_leaves = [tree.get_n_leaves() for tree in self.model.estimators_]

        return {
            'n_trees': len(self.model.estimators_),
            'avg_depth': np.mean(depths),
            'max_depth': np.max(depths),
            'min_depth': np.min(depths),
            'avg_leaves': np.mean(n_leaves),
            'max_leaves': np.max(n_leaves),
            'min_leaves': np.min(n_leaves)
        }

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
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'oob_score': self.oob_score,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs
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
        self.min_samples_split = config.get('min_samples_split', self.min_samples_split)
        self.min_samples_leaf = config.get('min_samples_leaf', self.min_samples_leaf)
        self.max_features = config.get('max_features', self.max_features)
        self.oob_score = config.get('oob_score', self.oob_score)
        self.random_state = config.get('random_state', self.random_state)
        self.n_jobs = config.get('n_jobs', self.n_jobs)

        logger.info(f"Model loaded from {load_path}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of model metrics"""
        if self.val_metrics is None:
            return {}

        summary = {
            'validation': self.val_metrics.to_dict(),
            'top_features': self.get_top_features(10),
            'is_trained': self.is_trained,
            'n_features': len(self.feature_names)
        }

        if self.is_trained:
            summary['tree_stats'] = self.get_tree_stats()

        return summary
