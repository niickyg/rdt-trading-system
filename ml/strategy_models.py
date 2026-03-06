"""
Per-Strategy ML Model Manager

Manages separate ML models for each trading strategy. Each strategy gets:
- Its own feature set
- Its own model file
- Its own training data (filtered by strategy_name)
- Its own confidence threshold
"""

import threading
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
from loguru import logger

try:
    from sklearn.ensemble import GradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from ml.safe_model_loader import safe_load_model, safe_save_model
    SAFE_LOADER_AVAILABLE = True
except ImportError:
    SAFE_LOADER_AVAILABLE = False


class StrategyModel:
    """ML model for a single strategy."""

    def __init__(
        self,
        strategy_name: str,
        model_dir: str = "models/strategy",
        confidence_threshold: float = 0.6,
    ):
        self.strategy_name = strategy_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_trained = False
        self._load()

    @property
    def model_path(self) -> Path:
        return self.model_dir / f"{self.strategy_name}_model.joblib"

    def _load(self):
        """Load model from disk if it exists."""
        if not self.model_path.exists():
            return
        try:
            if SAFE_LOADER_AVAILABLE:
                self.model = safe_load_model(str(self.model_path))
            else:
                logger.warning(
                    f"safe_model_loader unavailable — skipping load of '{self.strategy_name}' model. "
                    "Install dependencies to enable SHA-256 verified model loading."
                )
                return
            self.is_trained = True
            logger.info(f"Loaded ML model for strategy '{self.strategy_name}'")
        except Exception as e:
            logger.warning(f"Failed to load model for '{self.strategy_name}': {e}")

    def train(self, features: pd.DataFrame, labels: pd.Series) -> Dict:
        """
        Train the strategy model.

        Args:
            features: Feature DataFrame
            labels: Binary labels (1 = profitable trade, 0 = losing trade)

        Returns:
            Training metrics dict
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, cannot train strategy model")
            return {'error': 'sklearn not available'}

        if len(features) < 20:
            logger.info(f"Strategy '{self.strategy_name}': not enough data to train ({len(features)} samples)")
            return {'error': 'insufficient_data', 'samples': len(features)}

        try:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
            self.model.fit(features, labels)
            self.is_trained = True

            # Save model
            self._save()

            # Calculate training metrics
            train_preds = self.model.predict(features)
            accuracy = (train_preds == labels).mean()
            train_proba = self.model.predict_proba(features)[:, 1]

            metrics = {
                'strategy': self.strategy_name,
                'samples': len(features),
                'train_accuracy': round(accuracy, 4),
                'mean_confidence': round(train_proba.mean(), 4),
                'features_used': features.shape[1],
            }
            logger.info(f"Trained model for '{self.strategy_name}': {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Training error for '{self.strategy_name}': {e}")
            return {'error': str(e)}

    def predict(self, features: pd.DataFrame) -> float:
        """
        Predict probability of profitable trade.

        Args:
            features: Single-row or multi-row feature DataFrame

        Returns:
            Mean predicted probability (0.0 to 1.0)
        """
        if not self.is_trained or self.model is None:
            return 0.5  # No model = neutral confidence

        try:
            proba = self.model.predict_proba(features)[:, 1]
            return float(proba.mean())
        except Exception as e:
            logger.debug(f"Prediction error for '{self.strategy_name}': {e}")
            return 0.5

    def meets_threshold(self, features: pd.DataFrame) -> bool:
        """Check if prediction meets the confidence threshold."""
        return self.predict(features) >= self.confidence_threshold

    def _save(self):
        """Save model to disk."""
        if self.model is None:
            return
        try:
            if SAFE_LOADER_AVAILABLE:
                safe_save_model(self.model, str(self.model_path))
            else:
                import joblib
                joblib.dump(self.model, str(self.model_path))
            logger.debug(f"Saved model for '{self.strategy_name}'")
        except Exception as e:
            logger.warning(f"Failed to save model for '{self.strategy_name}': {e}")


class StrategyModelManager:
    """
    Manages ML models for all strategies.

    Each strategy gets its own isolated model that is trained only on
    that strategy's trade outcomes.
    """

    def __init__(self, models_dir: str = "models/strategy"):
        self.models_dir = models_dir
        self.models: Dict[str, StrategyModel] = {}

    def get_model(self, strategy_name: str) -> StrategyModel:
        """Get or create a model for a strategy."""
        if strategy_name not in self.models:
            self.models[strategy_name] = StrategyModel(
                strategy_name=strategy_name,
                model_dir=self.models_dir,
            )
        return self.models[strategy_name]

    def train(self, strategy_name: str, features: pd.DataFrame, labels: pd.Series) -> Dict:
        """Train a strategy-specific model."""
        model = self.get_model(strategy_name)
        return model.train(features, labels)

    def predict(self, strategy_name: str, features: pd.DataFrame) -> float:
        """Get prediction from a strategy-specific model."""
        model = self.get_model(strategy_name)
        return model.predict(features)

    def get_all_models(self) -> Dict[str, StrategyModel]:
        """Get all loaded models."""
        return dict(self.models)

    def get_status(self) -> Dict:
        """Get status of all models."""
        return {
            name: {
                'is_trained': model.is_trained,
                'confidence_threshold': model.confidence_threshold,
                'model_path': str(model.model_path),
            }
            for name, model in self.models.items()
        }


# Module-level singleton
_manager: Optional[StrategyModelManager] = None
_manager_lock = threading.Lock()


def get_strategy_model_manager() -> StrategyModelManager:
    """Get the global strategy model manager."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = StrategyModelManager()
    return _manager
