"""
Optuna-based Hyperparameter Optimizer for ML Models

Provides hyperparameter optimization for XGBoost, RandomForest, and LSTM models
with support for pruning, early stopping, and comprehensive logging.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from loguru import logger

try:
    import optuna
    from optuna.trial import TrialState
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")

from ml.optimization.search_spaces import (
    get_search_space,
    suggest_params,
    get_default_params,
    validate_params,
    ModelType
)
from ml.optimization.objective import (
    create_objective,
    OptunaObjective,
    OptimizationMetric,
    TrialResult
)


@dataclass
class OptimizationConfig:
    """Configuration for optimization"""
    model_type: str
    metric: str = "f1"
    n_trials: int = 100
    timeout: Optional[int] = None  # seconds
    n_cv_splits: int = 5
    early_stopping_patience: int = 10
    n_startup_trials: int = 10
    n_jobs: int = 1
    seed: int = 42
    study_name: Optional[str] = None
    storage: Optional[str] = None  # SQLite or PostgreSQL URL
    load_if_exists: bool = True
    pruner_type: str = "median"  # 'median' or 'hyperband'


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization"""
    best_params: Dict[str, Any]
    best_value: float
    metric: str
    n_trials: int
    n_complete: int
    n_pruned: int
    duration_seconds: float
    study_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'metric': self.metric,
            'n_trials': self.n_trials,
            'n_complete': self.n_complete,
            'n_pruned': self.n_pruned,
            'duration_seconds': self.duration_seconds,
            'study_name': self.study_name,
            'timestamp': self.timestamp.isoformat(),
            'optimization_history': self.optimization_history
        }

    def save(self, path: str):
        """Save result to JSON file"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Optimization result saved to {save_path}")

    @classmethod
    def load(cls, path: str) -> 'OptimizationResult':
        """Load result from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ModelOptimizer:
    """
    Hyperparameter optimizer using Optuna.

    Supports XGBoost, RandomForest, and LSTM models with:
    - Time-series cross-validation
    - Pruning for early stopping of bad trials
    - Multiple optimization metrics
    - Study persistence for resuming
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        X_lstm: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None
    ):
        """
        Initialize optimizer.

        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,) - binary 0/1
            model_type: Type of model ('xgboost', 'random_forest', 'lstm')
            X_lstm: Optional LSTM sequences (for LSTM model)
            returns: Optional returns for profit_factor metric
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")

        self.X = X
        self.y = y
        self.model_type = model_type.lower()
        self.X_lstm = X_lstm
        self.returns = returns

        # Validate model type
        if self.model_type not in [e.value for e in ModelType]:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Supported: {[e.value for e in ModelType]}")

        # Study and results
        self.study: Optional[optuna.Study] = None
        self.result: Optional[OptimizationResult] = None
        self._optimization_history: List[Dict[str, Any]] = []

        logger.info(
            f"Initialized ModelOptimizer for {model_type} with "
            f"{X.shape[0]} samples, {X.shape[1]} features"
        )

    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        metric: str = "f1",
        n_cv_splits: int = 5,
        early_stopping_patience: int = 10,
        n_startup_trials: int = 10,
        n_jobs: int = 1,
        seed: int = 42,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        load_if_exists: bool = True,
        pruner_type: str = "median",
        callbacks: Optional[List[Callable]] = None,
        show_progress_bar: bool = True
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of trials to run
            timeout: Maximum time in seconds (None for no limit)
            metric: Metric to optimize ('accuracy', 'precision', 'recall',
                    'f1', 'auc', 'log_loss', 'profit_factor')
            n_cv_splits: Number of cross-validation splits
            early_stopping_patience: Early stopping patience
            n_startup_trials: Random trials before using sampler
            n_jobs: Number of parallel jobs
            seed: Random seed
            study_name: Name for the study (for persistence)
            storage: Database URL for study persistence
            load_if_exists: Load existing study if it exists
            pruner_type: Pruner type ('median' or 'hyperband')
            callbacks: Optional list of Optuna callbacks
            show_progress_bar: Show progress bar during optimization

        Returns:
            OptimizationResult with best parameters and history
        """
        import time
        start_time = time.time()

        # Create study name if not provided
        if study_name is None:
            study_name = f"{self.model_type}_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting optimization: {study_name}")
        logger.info(f"  Model: {self.model_type}")
        logger.info(f"  Metric: {metric}")
        logger.info(f"  Trials: {n_trials}")
        logger.info(f"  Timeout: {timeout}s" if timeout else "  Timeout: None")

        # Create pruner
        if pruner_type == "hyperband":
            pruner = HyperbandPruner(
                min_resource=1,
                max_resource=n_cv_splits,
                reduction_factor=3
            )
        else:
            pruner = MedianPruner(
                n_startup_trials=n_startup_trials,
                n_warmup_steps=2,
                interval_steps=1
            )

        # Create sampler
        sampler = TPESampler(
            seed=seed,
            n_startup_trials=n_startup_trials
        )

        # Create or load study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
            direction="maximize",  # We want to maximize metrics
            sampler=sampler,
            pruner=pruner
        )

        # Create objective function
        objective = OptunaObjective(
            X=self.X,
            y=self.y,
            model_type=self.model_type,
            metric=metric,
            n_cv_splits=n_cv_splits,
            early_stopping_patience=early_stopping_patience,
            X_lstm=self.X_lstm,
            returns=self.returns
        )

        # Add logging callback
        def logging_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            """Log trial results"""
            self._optimization_history.append({
                'trial': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'duration': trial.duration.total_seconds() if trial.duration else None
            })

            if trial.state == TrialState.COMPLETE:
                logger.info(
                    f"Trial {trial.number}: {metric}={trial.value:.4f} "
                    f"(best={study.best_value:.4f})"
                )
            elif trial.state == TrialState.PRUNED:
                logger.debug(f"Trial {trial.number} pruned")

        all_callbacks = [logging_callback]
        if callbacks:
            all_callbacks.extend(callbacks)

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            callbacks=all_callbacks,
            show_progress_bar=show_progress_bar
        )

        duration = time.time() - start_time

        # Get trial statistics
        n_complete = len([t for t in self.study.trials if t.state == TrialState.COMPLETE])
        n_pruned = len([t for t in self.study.trials if t.state == TrialState.PRUNED])

        # Create result
        self.result = OptimizationResult(
            best_params=self.study.best_params,
            best_value=self.study.best_value,
            metric=metric,
            n_trials=len(self.study.trials),
            n_complete=n_complete,
            n_pruned=n_pruned,
            duration_seconds=duration,
            study_name=study_name,
            optimization_history=self._optimization_history
        )

        logger.info(f"Optimization complete in {duration:.1f}s")
        logger.info(f"  Best {metric}: {self.study.best_value:.4f}")
        logger.info(f"  Best params: {self.study.best_params}")
        logger.info(f"  Trials: {n_complete} complete, {n_pruned} pruned")

        return self.result

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best parameters found.

        Returns:
            Dictionary of best parameters

        Raises:
            ValueError: If optimization hasn't been run yet
        """
        if self.study is None or self.result is None:
            raise ValueError("Optimization hasn't been run yet. Call optimize() first.")

        return self.result.best_params

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Get the optimization history.

        Returns:
            List of trial results with parameters and values
        """
        if self.study is None:
            return []

        history = []
        for trial in self.study.trials:
            history.append({
                'trial': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'duration': trial.duration.total_seconds() if trial.duration else None,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
            })

        return history

    def get_importance(self) -> Dict[str, float]:
        """
        Get hyperparameter importance.

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if self.study is None:
            return {}

        try:
            importance = optuna.importance.get_param_importances(self.study)
            return dict(importance)
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
            return {}

    def get_best_trial(self) -> Optional[Dict[str, Any]]:
        """
        Get the best trial details.

        Returns:
            Dictionary with best trial details
        """
        if self.study is None:
            return None

        trial = self.study.best_trial
        return {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name,
            'duration': trial.duration.total_seconds() if trial.duration else None,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
        }

    def get_contour_plot(
        self,
        param1: str,
        param2: str,
        save_path: Optional[str] = None
    ):
        """
        Generate contour plot for two parameters.

        Args:
            param1: First parameter name
            param2: Second parameter name
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        if self.study is None:
            raise ValueError("Optimization hasn't been run yet")

        try:
            from optuna.visualization import plot_contour
            fig = plot_contour(self.study, params=[param1, param2])

            if save_path:
                fig.write_html(save_path)
                logger.info(f"Contour plot saved to {save_path}")

            return fig
        except ImportError:
            logger.warning("Plotly not available for visualization")
            return None

    def get_optimization_history_plot(self, save_path: Optional[str] = None):
        """
        Generate optimization history plot.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        if self.study is None:
            raise ValueError("Optimization hasn't been run yet")

        try:
            from optuna.visualization import plot_optimization_history
            fig = plot_optimization_history(self.study)

            if save_path:
                fig.write_html(save_path)
                logger.info(f"History plot saved to {save_path}")

            return fig
        except ImportError:
            logger.warning("Plotly not available for visualization")
            return None

    def get_param_importances_plot(self, save_path: Optional[str] = None):
        """
        Generate parameter importance plot.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        if self.study is None:
            raise ValueError("Optimization hasn't been run yet")

        try:
            from optuna.visualization import plot_param_importances
            fig = plot_param_importances(self.study)

            if save_path:
                fig.write_html(save_path)
                logger.info(f"Importance plot saved to {save_path}")

            return fig
        except ImportError:
            logger.warning("Plotly not available for visualization")
            return None

    def save_best_params(self, path: str):
        """
        Save best parameters to JSON file.

        Args:
            path: Path to save the parameters
        """
        if self.result is None:
            raise ValueError("Optimization hasn't been run yet")

        self.result.save(path)

    def create_optimized_model(self):
        """
        Create a model instance with the best parameters.

        Returns:
            Model instance configured with best parameters
        """
        if self.result is None:
            raise ValueError("Optimization hasn't been run yet")

        params = self.result.best_params

        if self.model_type == 'xgboost':
            from ml.models.xgboost_model import XGBoostTradeClassifier
            return XGBoostTradeClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=42
            )

        elif self.model_type == 'random_forest':
            from ml.models.random_forest_model import RandomForestTradeClassifier
            return RandomForestTradeClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 15),
                min_samples_split=params.get('min_samples_split', 5),
                min_samples_leaf=params.get('min_samples_leaf', 2),
                random_state=42
            )

        elif self.model_type == 'lstm':
            from ml.models.lstm_model import LSTMTradePredictor
            units = params.get('units', 64)
            n_layers = params.get('layers', 2)
            lstm_units = [units // (2 ** i) for i in range(n_layers)]

            return LSTMTradePredictor(
                lstm_units=lstm_units,
                dropout=params.get('dropout', 0.2),
                learning_rate=params.get('learning_rate', 0.001),
                random_state=42
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


class OptimizationJobManager:
    """
    Manager for running optimization jobs in the background.

    Tracks job status, results, and provides an interface for
    async optimization.
    """

    def __init__(self, results_dir: str = "data/optimization"):
        """
        Initialize job manager.

        Args:
            results_dir: Directory to store optimization results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._optimizers: Dict[str, ModelOptimizer] = {}

    def create_job(
        self,
        job_id: str,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        config: OptimizationConfig
    ) -> str:
        """
        Create an optimization job.

        Args:
            job_id: Unique job identifier
            X: Feature array
            y: Target array
            model_type: Type of model
            config: Optimization configuration

        Returns:
            Job ID
        """
        self._jobs[job_id] = {
            'id': job_id,
            'model_type': model_type,
            'config': asdict(config),
            'status': 'created',
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'result': None,
            'error': None
        }

        self._optimizers[job_id] = ModelOptimizer(
            X=X,
            y=y,
            model_type=model_type
        )

        logger.info(f"Created optimization job {job_id}")
        return job_id

    def start_job(self, job_id: str) -> bool:
        """
        Start an optimization job.

        Args:
            job_id: Job ID

        Returns:
            True if job started successfully
        """
        if job_id not in self._jobs:
            logger.error(f"Job {job_id} not found")
            return False

        job = self._jobs[job_id]
        optimizer = self._optimizers.get(job_id)

        if optimizer is None:
            logger.error(f"Optimizer for job {job_id} not found")
            return False

        job['status'] = 'running'
        job['started_at'] = datetime.now().isoformat()

        try:
            config = job['config']
            result = optimizer.optimize(
                n_trials=config['n_trials'],
                timeout=config.get('timeout'),
                metric=config['metric'],
                n_cv_splits=config['n_cv_splits'],
                early_stopping_patience=config['early_stopping_patience'],
                show_progress_bar=False
            )

            job['status'] = 'completed'
            job['completed_at'] = datetime.now().isoformat()
            job['result'] = result.to_dict()

            # Save result
            result_path = self.results_dir / f"{job_id}_result.json"
            result.save(str(result_path))

            logger.info(f"Job {job_id} completed successfully")
            return True

        except Exception as e:
            job['status'] = 'failed'
            job['completed_at'] = datetime.now().isoformat()
            job['error'] = str(e)
            logger.error(f"Job {job_id} failed: {e}")
            return False

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status.

        Args:
            job_id: Job ID

        Returns:
            Job status dictionary or None if not found
        """
        return self._jobs.get(job_id)

    def get_job_result(self, job_id: str) -> Optional[OptimizationResult]:
        """
        Get job result.

        Args:
            job_id: Job ID

        Returns:
            OptimizationResult or None if not found/not complete
        """
        job = self._jobs.get(job_id)
        if job is None or job['result'] is None:
            return None

        return OptimizationResult(**job['result'])

    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all jobs.

        Returns:
            List of job status dictionaries
        """
        return list(self._jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Job ID

        Returns:
            True if job was cancelled
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False

        if job['status'] == 'running':
            # Note: Optuna doesn't support cancellation directly
            # We'd need to implement this with threading/multiprocessing
            job['status'] = 'cancelled'
            job['completed_at'] = datetime.now().isoformat()
            logger.info(f"Job {job_id} marked as cancelled")
            return True

        return False


# Global job manager instance
_job_manager: Optional[OptimizationJobManager] = None


def get_job_manager() -> OptimizationJobManager:
    """Get the global job manager instance"""
    global _job_manager
    if _job_manager is None:
        _job_manager = OptimizationJobManager()
    return _job_manager
