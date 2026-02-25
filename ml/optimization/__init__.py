"""
ML Model Optimization with Optuna

This module provides hyperparameter optimization for ML models using Optuna.
Supports XGBoost, RandomForest, and LSTM models with pruning and early stopping.
"""

from ml.optimization.optuna_optimizer import ModelOptimizer
from ml.optimization.search_spaces import (
    get_search_space,
    XGBOOST_SEARCH_SPACE,
    RANDOM_FOREST_SEARCH_SPACE,
    LSTM_SEARCH_SPACE,
)
from ml.optimization.objective import (
    create_objective,
    OptunaObjective,
    calculate_profit_factor,
)

__all__ = [
    'ModelOptimizer',
    'get_search_space',
    'XGBOOST_SEARCH_SPACE',
    'RANDOM_FOREST_SEARCH_SPACE',
    'LSTM_SEARCH_SPACE',
    'create_objective',
    'OptunaObjective',
    'calculate_profit_factor',
]
