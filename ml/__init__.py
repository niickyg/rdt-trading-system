"""
Machine Learning module for trading system
"""

# Fix curl_cffi chrome136 impersonation issue in Docker
# curl_cffi 0.13.0 maps 'chrome' to 'chrome136' which may not be supported
try:
    from curl_cffi.requests import impersonate
    impersonate.DEFAULT_CHROME = 'chrome110'
    if hasattr(impersonate, 'REAL_TARGET_MAP'):
        impersonate.REAL_TARGET_MAP['chrome'] = 'chrome110'
except ImportError:
    pass

from ml.ensemble import Ensemble, StackedEnsemble
from ml.regime_detector import MarketRegimeDetector, RegimeDetector
from ml.feature_engineering import FeatureEngineer, calculate_features_for_symbol
from ml.drift_detector import DriftDetector, DriftThresholds, DriftReport, DriftSeverity
from ml.model_monitor import ModelMonitor, ModelMonitorRegistry, get_monitor_registry
from ml.monitoring_store import MonitoringStore
from ml.dynamic_sizer import DynamicPositionSizer, SizingInput, SizingResult, get_dynamic_sizer

# Import optimization module (optional, requires optuna)
try:
    from ml.optimization import (
        ModelOptimizer,
        get_search_space,
        create_objective,
        XGBOOST_SEARCH_SPACE,
        RANDOM_FOREST_SEARCH_SPACE,
        LSTM_SEARCH_SPACE,
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Import GPU utilities (optional, requires tensorflow)
try:
    from ml.gpu_utils import (
        is_gpu_available,
        is_cuda_available,
        is_mps_available,
        get_gpu_info,
        get_gpu_count,
        configure_gpu,
        get_optimal_batch_size,
        setup_gpu_for_training,
        get_gpu_summary,
        GPUInfo,
    )
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

__all__ = [
    'Ensemble',
    'StackedEnsemble',
    'MarketRegimeDetector',
    'RegimeDetector',
    'FeatureEngineer',
    'calculate_features_for_symbol',
    'DriftDetector',
    'DriftThresholds',
    'DriftReport',
    'DriftSeverity',
    'ModelMonitor',
    'ModelMonitorRegistry',
    'get_monitor_registry',
    'MonitoringStore',
    'DynamicPositionSizer',
    'SizingInput',
    'SizingResult',
    'get_dynamic_sizer',
    'OPTIMIZATION_AVAILABLE',
    'GPU_UTILS_AVAILABLE',
]

# Add optimization exports if available
if OPTIMIZATION_AVAILABLE:
    __all__.extend([
        'ModelOptimizer',
        'get_search_space',
        'create_objective',
        'XGBOOST_SEARCH_SPACE',
        'RANDOM_FOREST_SEARCH_SPACE',
        'LSTM_SEARCH_SPACE',
    ])

# Add GPU utilities exports if available
if GPU_UTILS_AVAILABLE:
    __all__.extend([
        'is_gpu_available',
        'is_cuda_available',
        'is_mps_available',
        'get_gpu_info',
        'get_gpu_count',
        'configure_gpu',
        'get_optimal_batch_size',
        'setup_gpu_for_training',
        'get_gpu_summary',
        'GPUInfo',
    ])
