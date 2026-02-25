"""
Objective Functions for Optuna Optimization

Provides objective functions for hyperparameter optimization with various
metrics including accuracy, precision, recall, F1, and profit_factor.
Supports time-series cross-validation and early stopping.
"""

from __future__ import annotations
from typing import Dict, Any, Callable, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from loguru import logger

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, log_loss
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available")


class OptimizationMetric(str, Enum):
    """Supported optimization metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUC = "auc"
    LOG_LOSS = "log_loss"
    PROFIT_FACTOR = "profit_factor"


@dataclass
class TrialResult:
    """Result of a single trial"""
    trial_number: int
    params: Dict[str, Any]
    value: float
    metric: str
    cv_scores: List[float]
    cv_std: float
    duration_seconds: float
    pruned: bool = False
    error_message: Optional[str] = None


def calculate_profit_factor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    returns: Optional[np.ndarray] = None
) -> float:
    """
    Calculate profit factor from predictions.

    Profit Factor = Sum of Winning Trades / Sum of Losing Trades

    If returns are provided, uses actual returns.
    Otherwise, estimates based on prediction accuracy.

    Args:
        y_true: True labels (0/1)
        y_pred: Predicted labels (0/1)
        y_proba: Optional prediction probabilities
        returns: Optional actual returns for each sample

    Returns:
        Profit factor (>1 is profitable)
    """
    if returns is not None:
        # Use actual returns
        correct_predictions = (y_true == y_pred)
        winning_returns = np.sum(returns[correct_predictions & (y_pred == 1)])
        losing_returns = np.abs(np.sum(returns[~correct_predictions | (y_pred == 0)]))
    else:
        # Estimate based on prediction outcomes
        # True positives (predicted win, actually won)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        # False positives (predicted win, actually lost)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        # False negatives (predicted loss, actually won - missed opportunity)
        fn = np.sum((y_pred == 0) & (y_true == 1))
        # True negatives (predicted loss, actually lost - avoided loss)
        tn = np.sum((y_pred == 0) & (y_true == 0))

        # Estimate profit: wins + avoided losses
        winning_returns = tp + tn * 0.5  # Avoided losses count as partial win

        # Estimate losses: false positives + missed opportunities
        losing_returns = fp + fn * 0.5  # Missed opportunities are partial loss

    if losing_returns == 0:
        return float('inf') if winning_returns > 0 else 1.0

    return winning_returns / losing_returns


def calculate_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    metric: str = "f1",
    returns: Optional[np.ndarray] = None
) -> float:
    """
    Calculate the specified metric.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Optional prediction probabilities
        metric: Metric to calculate
        returns: Optional returns for profit_factor calculation

    Returns:
        Metric value
    """
    metric = metric.lower()

    if metric == OptimizationMetric.ACCURACY:
        return accuracy_score(y_true, y_pred)
    elif metric == OptimizationMetric.PRECISION:
        return precision_score(y_true, y_pred, zero_division=0)
    elif metric == OptimizationMetric.RECALL:
        return recall_score(y_true, y_pred, zero_division=0)
    elif metric == OptimizationMetric.F1:
        return f1_score(y_true, y_pred, zero_division=0)
    elif metric == OptimizationMetric.AUC:
        if y_proba is not None:
            try:
                return roc_auc_score(y_true, y_proba)
            except ValueError:
                return 0.5  # Default if only one class present
        return 0.5
    elif metric == OptimizationMetric.LOG_LOSS:
        if y_proba is not None:
            try:
                # Return negative log_loss since we want to maximize
                return -log_loss(y_true, y_proba)
            except ValueError:
                return -1.0
        return -1.0
    elif metric == OptimizationMetric.PROFIT_FACTOR:
        return calculate_profit_factor(y_true, y_pred, y_proba, returns)
    else:
        raise ValueError(f"Unknown metric: {metric}. "
                        f"Supported: {[e.value for e in OptimizationMetric]}")


class EarlyStoppingCallback:
    """
    Callback for early stopping during cross-validation.

    Stops training if validation score doesn't improve for patience epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        metric: str = "f1"
    ):
        """
        Initialize early stopping callback.

        Args:
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            metric: Metric to monitor
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop

    def reset(self):
        """Reset callback state"""
        self.best_score = None
        self.counter = 0
        self.early_stop = False


class OptunaObjective:
    """
    Objective function factory for Optuna optimization.

    Creates objective functions for different model types with
    cross-validation and early stopping support.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        metric: str = "f1",
        n_cv_splits: int = 5,
        early_stopping_patience: int = 10,
        X_lstm: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None
    ):
        """
        Initialize objective function.

        Args:
            X: Feature array
            y: Target array
            model_type: Type of model ('xgboost', 'random_forest', 'lstm')
            metric: Metric to optimize
            n_cv_splits: Number of cross-validation splits
            early_stopping_patience: Patience for early stopping
            X_lstm: Optional LSTM sequences (for LSTM model)
            returns: Optional returns array for profit_factor metric
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available. Install with: pip install scikit-learn")

        self.X = X
        self.y = y
        self.model_type = model_type.lower()
        self.metric = metric.lower()
        self.n_cv_splits = n_cv_splits
        self.early_stopping_patience = early_stopping_patience
        self.X_lstm = X_lstm
        self.returns = returns

        # Import search spaces
        from ml.optimization.search_spaces import suggest_params

        self.suggest_params = suggest_params
        self.early_stopping_callback = EarlyStoppingCallback(
            patience=early_stopping_patience,
            metric=metric
        )

        logger.info(
            f"Initialized OptunaObjective for {model_type} with {metric} metric, "
            f"{n_cv_splits} CV splits"
        )

    def __call__(self, trial: Trial) -> float:
        """
        Evaluate a trial.

        Args:
            trial: Optuna trial object

        Returns:
            Objective value (higher is better)
        """
        # Suggest hyperparameters
        params = self.suggest_params(trial, self.model_type)

        # Create model with suggested parameters
        model = self._create_model(params)

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X)):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # Handle LSTM sequences
            if self.model_type == 'lstm' and self.X_lstm is not None:
                X_lstm_train = self.X_lstm[train_idx]
                X_lstm_val = self.X_lstm[val_idx]
            else:
                X_lstm_train, X_lstm_val = None, None

            # Get returns for this fold if available
            returns_val = None
            if self.returns is not None:
                returns_val = self.returns[val_idx]

            # Train and evaluate
            try:
                score = self._train_and_evaluate(
                    model, X_train, y_train, X_val, y_val,
                    X_lstm_train, X_lstm_val, returns_val, params
                )
                cv_scores.append(score)

                # Report intermediate value for pruning
                trial.report(np.mean(cv_scores), fold)

                # Check if trial should be pruned
                if trial.should_prune():
                    logger.debug(f"Trial {trial.number} pruned at fold {fold}")
                    raise optuna.TrialPruned()

            except Exception as e:
                logger.warning(f"Error in fold {fold}: {e}")
                cv_scores.append(0.0)

        # Return mean CV score
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)

        logger.debug(
            f"Trial {trial.number}: {self.metric}={mean_score:.4f} "
            f"(+/- {std_score:.4f})"
        )

        return mean_score

    def _create_model(self, params: Dict[str, Any]):
        """Create model with given parameters"""
        if self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    **params,
                    objective='binary:logistic',
                    eval_metric='logloss',
                    random_state=42,
                    n_jobs=-1,
                    tree_method='hist'
                )
            except ImportError:
                raise ImportError("XGBoost not available")

        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            # Filter out unsupported parameters
            rf_params = {k: v for k, v in params.items()
                        if k in ['n_estimators', 'max_depth', 'min_samples_split',
                                'min_samples_leaf', 'max_features', 'bootstrap',
                                'class_weight']}
            return RandomForestClassifier(
                **rf_params,
                random_state=42,
                n_jobs=-1,
                oob_score=params.get('bootstrap', True)
            )

        elif self.model_type == 'lstm':
            # For LSTM, return params dict since model creation is different
            return params

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _train_and_evaluate(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_lstm_train: Optional[np.ndarray] = None,
        X_lstm_val: Optional[np.ndarray] = None,
        returns_val: Optional[np.ndarray] = None,
        params: Optional[Dict] = None
    ) -> float:
        """Train model and evaluate on validation set"""

        if self.model_type == 'lstm':
            return self._train_and_evaluate_lstm(
                params, X_train, y_train, X_val, y_val,
                X_lstm_train, X_lstm_val, returns_val
            )

        # Train tree-based model
        if self.model_type == 'xgboost':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

        # Get predictions
        y_pred = model.predict(X_val)

        # Get probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_val)
            if proba.shape[1] == 2:
                y_proba = proba[:, 1]
            else:
                y_proba = proba

        # Calculate metric
        return calculate_metric(y_val, y_pred, y_proba, self.metric, returns_val)

    def _train_and_evaluate_lstm(
        self,
        params: Dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_lstm_train: Optional[np.ndarray],
        X_lstm_val: Optional[np.ndarray],
        returns_val: Optional[np.ndarray]
    ) -> float:
        """Train and evaluate LSTM model"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers, callbacks
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("TensorFlow not available for LSTM")

        # Use LSTM sequences if available, otherwise use regular features
        if X_lstm_train is not None:
            X_train_seq = X_lstm_train
            X_val_seq = X_lstm_val
        else:
            # Reshape regular features to sequences
            seq_len = min(20, X_train.shape[0] // 2)
            X_train_seq = self._create_sequences(X_train, seq_len)
            X_val_seq = self._create_sequences(X_val, seq_len)
            y_train = y_train[seq_len:]
            y_val = y_val[seq_len:]

        # Scale features
        scaler = StandardScaler()
        n_samples, seq_len, n_features = X_train_seq.shape
        X_train_flat = X_train_seq.reshape(-1, n_features)
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_samples, seq_len, n_features)

        X_val_flat = X_val_seq.reshape(-1, n_features)
        X_val_scaled = scaler.transform(X_val_flat).reshape(X_val_seq.shape)

        # Build model
        units = params.get('units', 64)
        n_layers = params.get('layers', 2)
        dropout = params.get('dropout', 0.2)
        learning_rate = params.get('learning_rate', 0.001)
        batch_size = params.get('batch_size', 32)

        model = keras.Sequential()
        model.add(layers.Input(shape=(seq_len, n_features)))

        for i in range(n_layers):
            return_sequences = i < n_layers - 1
            model.add(layers.LSTM(units, return_sequences=return_sequences))
            model.add(layers.Dropout(dropout))

        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1, activation='sigmoid'))

        # Compile
        optimizer_name = params.get('optimizer', 'adam')
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train with early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=0
        )

        model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=50,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )

        # Evaluate
        y_proba = model.predict(X_val_scaled, verbose=0).flatten()
        y_pred = (y_proba > 0.5).astype(int)

        return calculate_metric(y_val, y_pred, y_proba, self.metric, returns_val)

    def _create_sequences(self, X: np.ndarray, seq_len: int) -> np.ndarray:
        """Create sequences from feature array"""
        sequences = []
        for i in range(len(X) - seq_len):
            sequences.append(X[i:i + seq_len])
        return np.array(sequences)


def create_objective(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    metric: str = "f1",
    n_cv_splits: int = 5,
    **kwargs
) -> Callable[[Trial], float]:
    """
    Factory function to create an objective function.

    Args:
        X: Feature array
        y: Target array
        model_type: Type of model
        metric: Metric to optimize
        n_cv_splits: Number of CV splits
        **kwargs: Additional arguments for OptunaObjective

    Returns:
        Objective function for Optuna
    """
    objective = OptunaObjective(
        X=X,
        y=y,
        model_type=model_type,
        metric=metric,
        n_cv_splits=n_cv_splits,
        **kwargs
    )
    return objective
