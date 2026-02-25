"""
LSTM Time-Series Model for Trade Direction Prediction

TensorFlow/Keras implementation for sequence prediction with GPU acceleration.
Predicts: up/down/neutral based on last 20 bars.

GPU Support:
- Automatically detects CUDA/MPS GPU availability
- Configures memory growth to prevent OOM
- Falls back gracefully to CPU if no GPU
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger
from ml.safe_model_loader import safe_load_model, safe_save_model, ModelSecurityError

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report
    )
    import joblib
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Install with: pip install tensorflow scikit-learn")

# GPU utilities
try:
    from ml.gpu_utils import (
        is_gpu_available,
        get_gpu_summary,
        configure_gpu,
        get_optimal_batch_size,
        update_gpu_metrics as update_gpu_metrics_util,
        GPU_METRICS_AVAILABLE
    )
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    GPU_METRICS_AVAILABLE = False


@dataclass
class LSTMMetrics:
    """Validation metrics for LSTM model"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    class_distribution: Dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            'class_distribution': self.class_distribution,
            'timestamp': self.timestamp.isoformat()
        }


class LSTMTradePredictor:
    """
    LSTM model for time-series prediction of trade direction

    Predicts: up (0), neutral (1), down (2)
    Uses sequences of OHLCV + indicators over last N bars

    Input shape: (batch_size, sequence_length, n_features)
    Output: 3-class softmax (up/neutral/down)
    """

    def __init__(
        self,
        sequence_length: int = 20,
        n_features: int = 10,
        lstm_units: List[int] = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        random_state: int = 42
    ):
        """
        Initialize LSTM predictor

        Args:
            sequence_length: Number of time steps (bars) in sequence
            n_features: Number of features per time step
            lstm_units: List of LSTM layer sizes
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            random_state: Random seed for reproducibility
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units or [64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.random_state = random_state

        # Set random seeds (use local RandomState to avoid affecting global state)
        self._rng = np.random.RandomState(random_state)
        tf.random.set_seed(random_state)

        # Model
        self.model: Optional[keras.Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False

        # Feature names
        self.feature_names: List[str] = []

        # Metrics
        self.train_metrics: Optional[LSTMMetrics] = None
        self.val_metrics: Optional[LSTMMetrics] = None
        self.history: Optional[Dict] = None

        # Class mapping
        self.class_names = ['up', 'neutral', 'down']

        # GPU status
        self._gpu_available = False
        self._device = 'CPU'
        if GPU_UTILS_AVAILABLE:
            self._gpu_available = is_gpu_available()
            if self._gpu_available:
                gpu_summary = get_gpu_summary()
                self._device = gpu_summary.get('device_strategy', 'GPU')

        device_info = f", device={self._device}" if self._gpu_available else ""
        logger.info(f"Initialized LSTMTradePredictor (seq_len={sequence_length}, features={n_features}{device_info})")

    def _build_model(self) -> keras.Model:
        """Build LSTM architecture"""
        model = models.Sequential(name='LSTM_Trade_Predictor')

        # Input layer
        model.add(layers.Input(shape=(self.sequence_length, self.n_features)))

        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                name=f'lstm_{i+1}'
            ))
            model.add(layers.Dropout(self.dropout, name=f'dropout_{i+1}'))

        # Dense layers
        model.add(layers.Dense(32, activation='relu', name='dense_1'))
        model.add(layers.Dropout(self.dropout, name='dropout_final'))

        # Output layer (3 classes: up, neutral, down)
        model.add(layers.Dense(3, activation='softmax', name='output'))

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def prepare_sequences(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_column: str = 'direction'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequential data for LSTM

        Args:
            data: DataFrame with time-series data (sorted by date)
            feature_columns: Feature column names
            target_column: Target column name

        Returns:
            Tuple of (X_sequences, y_targets)
            X shape: (n_samples, sequence_length, n_features)
            y shape: (n_samples,)
        """
        if feature_columns is None:
            feature_columns = [col for col in data.columns
                             if col not in ['target', 'direction', 'symbol', 'date', 'timestamp']]

        self.feature_names = feature_columns
        self.n_features = len(feature_columns)

        # Extract features and target
        features = data[feature_columns].values
        target = data[target_column].values if target_column in data.columns else None

        # Create sequences
        X_sequences = []
        y_targets = []

        for i in range(len(features) - self.sequence_length):
            X_sequences.append(features[i:i + self.sequence_length])
            if target is not None:
                y_targets.append(target[i + self.sequence_length])

        X = np.array(X_sequences)
        y = np.array(y_targets) if y_targets else None

        logger.debug(f"Created {len(X)} sequences of shape {X.shape[1:]}")

        return X, y

    def _scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Scale features using StandardScaler

        Args:
            X: Feature array (n_samples, sequence_length, n_features)
            fit: Whether to fit the scaler

        Returns:
            Scaled feature array
        """
        original_shape = X.shape
        # Reshape to 2D for scaling
        X_reshaped = X.reshape(-1, X.shape[-1])

        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X_reshaped)

        # Reshape back to 3D
        return X_scaled.reshape(original_shape)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        verbose: int = 1
    ) -> LSTMMetrics:
        """
        Train LSTM model

        Args:
            X: Sequences (n_samples, sequence_length, n_features)
            y: Targets (n_samples,) - class labels 0/1/2
            validation_split: Fraction for validation
            epochs: Training epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            verbose: Verbosity level

        Returns:
            Validation metrics
        """
        logger.info(f"Training LSTM with {len(X)} sequences, {epochs} epochs")

        # Log GPU status
        if self._gpu_available:
            logger.info(f"GPU training enabled: {self._device}")
            if GPU_UTILS_AVAILABLE and GPU_METRICS_AVAILABLE:
                try:
                    update_gpu_metrics_util()
                except Exception:
                    pass
        else:
            logger.info("Training on CPU (no GPU available)")

        # Auto-adjust batch size based on GPU memory if available
        if GPU_UTILS_AVAILABLE and self._gpu_available:
            suggested_batch_size = get_optimal_batch_size(
                model_size_mb=5.0,  # Approximate LSTM model size
                sequence_length=self.sequence_length,
                n_features=self.n_features
            )
            if batch_size > suggested_batch_size:
                logger.info(f"Reducing batch size from {batch_size} to {suggested_batch_size} based on GPU memory")
                batch_size = suggested_batch_size

        # Scale features
        X_scaled = self._scale_features(X, fit=True)

        # Split data (maintaining time-series order)
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
        logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}")

        # Build model
        self.model = self._build_model()
        logger.info(f"Model architecture:\n{self.model.summary()}")

        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )

        self.is_trained = True
        self.history = history.history

        # Calculate metrics
        self.val_metrics = self._calculate_metrics(X_val, y_val)

        logger.info(f"Training complete. Val Accuracy: {self.val_metrics.accuracy:.4f}")

        return self.val_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels

        Args:
            X: Sequences (n_samples, sequence_length, n_features)

        Returns:
            Predicted class labels (0=up, 1=neutral, 2=down)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_scaled = self._scale_features(X, fit=False)
        predictions = self.model.predict(X_scaled, verbose=0)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Sequences (n_samples, sequence_length, n_features)

        Returns:
            Probability array (n_samples, 3) for [up, neutral, down]
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_scaled = self._scale_features(X, fit=False)
        return self.model.predict(X_scaled, verbose=0)

    def predict_direction(self, X: np.ndarray) -> List[str]:
        """
        Predict direction as string labels

        Args:
            X: Sequences

        Returns:
            List of direction strings ['up', 'neutral', 'down']
        """
        predictions = self.predict(X)
        return [self.class_names[pred] for pred in predictions]

    def _calculate_metrics(
        self,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> LSTMMetrics:
        """Calculate comprehensive metrics"""
        if self.model is None:
            return LSTMMetrics()

        y_pred = self.predict(X)

        # Calculate class distribution
        class_dist = {
            name: int(count)
            for name, count in zip(self.class_names, np.bincount(y_true, minlength=3))
        }

        return LSTMMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average='weighted', zero_division=0),
            recall=recall_score(y_true, y_pred, average='weighted', zero_division=0),
            f1_score=f1_score(y_true, y_pred, average='weighted', zero_division=0),
            confusion_matrix=confusion_matrix(y_true, y_pred),
            class_distribution=class_dist
        )

    def get_classification_report(self, X: np.ndarray, y_true: np.ndarray) -> str:
        """
        Get detailed classification report

        Args:
            X: Sequences
            y_true: True labels

        Returns:
            Classification report as string
        """
        y_pred = self.predict(X)
        return classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0
        )

    def save(self, path: str):
        """
        Save model to disk

        Args:
            path: Path to save model (.h5 for Keras, .pkl for metadata)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Nothing to save.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        model_path = save_path.with_suffix('.h5')
        self.model.save(model_path)

        # Save metadata
        metadata_path = save_path.with_suffix('.pkl')
        metadata = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'val_metrics': self.val_metrics,
            'history': self.history,
            'config': {
                'sequence_length': self.sequence_length,
                'n_features': self.n_features,
                'lstm_units': self.lstm_units,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'random_state': self.random_state
            }
        }
        safe_save_model(metadata, str(metadata_path))

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")

    def load(self, path: str):
        """
        Load model from disk

        Args:
            path: Path to model file (without extension or with .h5/.pkl)
        """
        load_path = Path(path)

        # Load Keras model
        model_path = load_path.with_suffix('.h5')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = keras.models.load_model(model_path)

        # Load metadata
        metadata_path = load_path.with_suffix('.pkl')
        if metadata_path.exists():
            metadata = safe_load_model(str(metadata_path), allow_unverified=False)

            self.scaler = metadata.get('scaler')
            self.feature_names = metadata.get('feature_names', [])
            self.is_trained = metadata.get('is_trained', True)
            self.val_metrics = metadata.get('val_metrics')
            self.history = metadata.get('history')

            config = metadata.get('config', {})
            self.sequence_length = config.get('sequence_length', self.sequence_length)
            self.n_features = config.get('n_features', self.n_features)
            self.lstm_units = config.get('lstm_units', self.lstm_units)
            self.dropout = config.get('dropout', self.dropout)
            self.learning_rate = config.get('learning_rate', self.learning_rate)
            self.random_state = config.get('random_state', self.random_state)

        logger.info(f"Model loaded from {model_path}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of model metrics"""
        summary = {
            'is_trained': self.is_trained,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'architecture': {
                'lstm_units': self.lstm_units,
                'dropout': self.dropout
            },
            'gpu': {
                'available': self._gpu_available,
                'device': self._device
            }
        }

        if self.val_metrics:
            summary['validation'] = self.val_metrics.to_dict()

        if self.history:
            summary['training_history'] = {
                'final_train_loss': self.history['loss'][-1],
                'final_val_loss': self.history['val_loss'][-1],
                'final_train_acc': self.history['accuracy'][-1],
                'final_val_acc': self.history['val_accuracy'][-1],
                'epochs_trained': len(self.history['loss'])
            }

        return summary

    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information for this model.

        Returns:
            Dict with GPU availability and device information
        """
        info = {
            'gpu_available': self._gpu_available,
            'device': self._device
        }

        if GPU_UTILS_AVAILABLE and self._gpu_available:
            info['details'] = get_gpu_summary()

        return info
