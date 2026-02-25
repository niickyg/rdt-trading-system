"""
Hyperparameter Search Spaces for ML Models

Defines the search spaces for Optuna hyperparameter optimization.
Each model type has its own search space configuration.
"""

from typing import Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum


class ModelType(str, Enum):
    """Supported model types for optimization"""
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    LSTM = "lstm"


@dataclass
class SearchSpaceConfig:
    """Configuration for a hyperparameter search space"""
    name: str
    param_type: str  # 'int', 'float', 'categorical', 'loguniform'
    low: Any = None
    high: Any = None
    choices: list = None
    log: bool = False
    step: Any = None

    def suggest(self, trial, name: str = None):
        """Suggest a value using Optuna trial"""
        param_name = name or self.name

        if self.param_type == 'int':
            return trial.suggest_int(param_name, self.low, self.high, step=self.step or 1)
        elif self.param_type == 'float':
            if self.log:
                return trial.suggest_float(param_name, self.low, self.high, log=True)
            return trial.suggest_float(param_name, self.low, self.high, step=self.step)
        elif self.param_type == 'categorical':
            return trial.suggest_categorical(param_name, self.choices)
        elif self.param_type == 'loguniform':
            return trial.suggest_float(param_name, self.low, self.high, log=True)
        else:
            raise ValueError(f"Unknown param_type: {self.param_type}")


# XGBoost Search Space
XGBOOST_SEARCH_SPACE = {
    'learning_rate': SearchSpaceConfig(
        name='learning_rate',
        param_type='loguniform',
        low=0.001,
        high=0.3,
    ),
    'max_depth': SearchSpaceConfig(
        name='max_depth',
        param_type='int',
        low=3,
        high=12,
    ),
    'n_estimators': SearchSpaceConfig(
        name='n_estimators',
        param_type='int',
        low=50,
        high=500,
        step=50,
    ),
    'min_child_weight': SearchSpaceConfig(
        name='min_child_weight',
        param_type='int',
        low=1,
        high=10,
    ),
    'subsample': SearchSpaceConfig(
        name='subsample',
        param_type='float',
        low=0.5,
        high=1.0,
        step=0.05,
    ),
    'colsample_bytree': SearchSpaceConfig(
        name='colsample_bytree',
        param_type='float',
        low=0.5,
        high=1.0,
        step=0.05,
    ),
    'gamma': SearchSpaceConfig(
        name='gamma',
        param_type='float',
        low=0.0,
        high=5.0,
        step=0.1,
    ),
    'reg_alpha': SearchSpaceConfig(
        name='reg_alpha',
        param_type='loguniform',
        low=1e-8,
        high=10.0,
    ),
    'reg_lambda': SearchSpaceConfig(
        name='reg_lambda',
        param_type='loguniform',
        low=1e-8,
        high=10.0,
    ),
}


# Random Forest Search Space
RANDOM_FOREST_SEARCH_SPACE = {
    'n_estimators': SearchSpaceConfig(
        name='n_estimators',
        param_type='int',
        low=50,
        high=500,
        step=50,
    ),
    'max_depth': SearchSpaceConfig(
        name='max_depth',
        param_type='int',
        low=5,
        high=30,
    ),
    'min_samples_split': SearchSpaceConfig(
        name='min_samples_split',
        param_type='int',
        low=2,
        high=20,
    ),
    'min_samples_leaf': SearchSpaceConfig(
        name='min_samples_leaf',
        param_type='int',
        low=1,
        high=10,
    ),
    'max_features': SearchSpaceConfig(
        name='max_features',
        param_type='categorical',
        choices=['sqrt', 'log2', None],
    ),
    'bootstrap': SearchSpaceConfig(
        name='bootstrap',
        param_type='categorical',
        choices=[True, False],
    ),
    'class_weight': SearchSpaceConfig(
        name='class_weight',
        param_type='categorical',
        choices=['balanced', 'balanced_subsample', None],
    ),
}


# LSTM Search Space
LSTM_SEARCH_SPACE = {
    'units': SearchSpaceConfig(
        name='units',
        param_type='int',
        low=16,
        high=256,
        step=16,
    ),
    'layers': SearchSpaceConfig(
        name='layers',
        param_type='int',
        low=1,
        high=4,
    ),
    'dropout': SearchSpaceConfig(
        name='dropout',
        param_type='float',
        low=0.0,
        high=0.5,
        step=0.05,
    ),
    'learning_rate': SearchSpaceConfig(
        name='learning_rate',
        param_type='loguniform',
        low=1e-5,
        high=1e-2,
    ),
    'batch_size': SearchSpaceConfig(
        name='batch_size',
        param_type='categorical',
        choices=[16, 32, 64, 128, 256],
    ),
    'recurrent_dropout': SearchSpaceConfig(
        name='recurrent_dropout',
        param_type='float',
        low=0.0,
        high=0.3,
        step=0.05,
    ),
    'optimizer': SearchSpaceConfig(
        name='optimizer',
        param_type='categorical',
        choices=['adam', 'rmsprop', 'sgd'],
    ),
}


def get_search_space(model_type: str) -> Dict[str, SearchSpaceConfig]:
    """
    Get the search space configuration for a given model type.

    Args:
        model_type: Type of model ('xgboost', 'random_forest', 'lstm')

    Returns:
        Dictionary of search space configurations
    """
    model_type = model_type.lower()

    if model_type == ModelType.XGBOOST or model_type == 'xgboost':
        return XGBOOST_SEARCH_SPACE
    elif model_type == ModelType.RANDOM_FOREST or model_type == 'random_forest':
        return RANDOM_FOREST_SEARCH_SPACE
    elif model_type == ModelType.LSTM or model_type == 'lstm':
        return LSTM_SEARCH_SPACE
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported types: {[e.value for e in ModelType]}")


def suggest_params(trial, model_type: str, params: list = None) -> Dict[str, Any]:
    """
    Suggest hyperparameters for a trial.

    Args:
        trial: Optuna trial object
        model_type: Type of model
        params: Optional list of parameter names to suggest (None = all)

    Returns:
        Dictionary of suggested parameters
    """
    search_space = get_search_space(model_type)
    suggested = {}

    for name, config in search_space.items():
        if params is None or name in params:
            suggested[name] = config.suggest(trial)

    return suggested


def get_default_params(model_type: str) -> Dict[str, Any]:
    """
    Get default parameters for a model type.

    Args:
        model_type: Type of model

    Returns:
        Dictionary of default parameters
    """
    model_type = model_type.lower()

    if model_type == 'xgboost':
        return {
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.0,
            'reg_alpha': 1e-5,
            'reg_lambda': 1.0,
        }
    elif model_type == 'random_forest':
        return {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'class_weight': 'balanced',
        }
    elif model_type == 'lstm':
        return {
            'units': 64,
            'layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'recurrent_dropout': 0.0,
            'optimizer': 'adam',
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def validate_params(model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clip parameters to valid ranges.

    Args:
        model_type: Type of model
        params: Parameters to validate

    Returns:
        Validated parameters
    """
    search_space = get_search_space(model_type)
    validated = {}

    for name, value in params.items():
        if name in search_space:
            config = search_space[name]

            if config.param_type in ('int', 'float', 'loguniform'):
                # Clip to valid range
                if config.low is not None:
                    value = max(value, config.low)
                if config.high is not None:
                    value = min(value, config.high)

                # Convert to correct type
                if config.param_type == 'int':
                    value = int(value)
                else:
                    value = float(value)

            elif config.param_type == 'categorical':
                if value not in config.choices:
                    value = config.choices[0]  # Default to first choice

            validated[name] = value
        else:
            # Pass through unknown parameters
            validated[name] = value

    return validated
