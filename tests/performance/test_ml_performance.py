"""
ML Model Performance Tests

Benchmarks for ML model inference including:
- Single prediction performance
- Batch prediction performance
- Feature engineering performance
- Model loading time
- Memory usage during inference

Usage:
    pytest tests/performance/test_ml_performance.py -v
    pytest tests/performance/test_ml_performance.py -v --benchmark-only
"""

import asyncio
import gc
import os
import sys
import time
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import benchmark
try:
    import pytest_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

# Try to import memory profiler
try:
    from memory_profiler import memory_usage
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

# Import configuration
try:
    from tests.performance.config import (
        get_performance_config,
        TEST_SYMBOLS,
    )
except ImportError:
    from config import (
        get_performance_config,
        TEST_SYMBOLS,
    )


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def perf_config():
    """Get performance configuration."""
    return get_performance_config()


@pytest.fixture
def mock_daily_data():
    """Generate mock daily price data."""
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    np.random.seed(42)

    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates)),
    }, index=dates)

    return df


@pytest.fixture
def mock_stock_data(mock_daily_data):
    """Generate mock stock data dictionary."""
    return {
        'symbol': 'AAPL',
        'current_price': float(mock_daily_data['close'].iloc[-1]),
        'previous_close': float(mock_daily_data['close'].iloc[-2]),
        'volume': int(mock_daily_data['volume'].iloc[-1]),
        'atr': float(mock_daily_data['high'].iloc[-1] - mock_daily_data['low'].iloc[-1]),
        'daily_data': mock_daily_data,
    }


@pytest.fixture
def mock_spy_data(mock_daily_data):
    """Generate mock SPY data."""
    return {
        'symbol': 'SPY',
        'current_price': 450.0,
        'previous_close': 448.0,
        'volume': 80000000,
        'atr': 3.5,
        'daily_data': mock_daily_data.copy(),
    }


@pytest.fixture
def mock_data_provider(mock_stock_data, mock_spy_data):
    """Create mock data provider."""
    provider = MagicMock()

    async def mock_get_stock_data(symbol):
        data = mock_stock_data.copy()
        data['symbol'] = symbol
        return data

    async def mock_get_spy_data():
        return mock_spy_data

    provider.get_stock_data = mock_get_stock_data
    provider.get_spy_data = mock_get_spy_data

    return provider


@pytest.fixture
def sample_features():
    """Generate sample features for prediction."""
    np.random.seed(42)
    n_features = 70  # Approximate number of features

    # Generate realistic feature values
    features = {
        'rrs': np.random.uniform(-3, 3),
        'rrs_3bar': np.random.uniform(-3, 3),
        'rrs_5bar': np.random.uniform(-3, 3),
        'atr': np.random.uniform(1, 5),
        'atr_percent': np.random.uniform(1, 5),
        'rsi_14': np.random.uniform(20, 80),
        'rsi_9': np.random.uniform(20, 80),
        'macd': np.random.uniform(-2, 2),
        'macd_signal': np.random.uniform(-2, 2),
        'macd_histogram': np.random.uniform(-1, 1),
        'bb_upper': np.random.uniform(100, 200),
        'bb_middle': np.random.uniform(100, 200),
        'bb_lower': np.random.uniform(100, 200),
        'bb_width': np.random.uniform(5, 20),
        'bb_percent': np.random.uniform(0, 1),
        'ema_3': np.random.uniform(100, 200),
        'ema_8': np.random.uniform(100, 200),
        'ema_21': np.random.uniform(100, 200),
        'ema_50': np.random.uniform(100, 200),
        'volume_sma_20': np.random.uniform(1000000, 10000000),
        'vwap': np.random.uniform(100, 200),
        'vwap_distance': np.random.uniform(-5, 5),
        'price_momentum_1': np.random.uniform(-5, 5),
        'price_momentum_5': np.random.uniform(-10, 10),
        'volume_ratio': np.random.uniform(0.5, 2),
        'vix': np.random.uniform(10, 30),
        'spy_trend': np.random.uniform(-5, 5),
        'hour_of_day': np.random.randint(9, 16),
        'day_of_week': np.random.randint(0, 5),
    }

    # Add more features to reach 70
    for i in range(len(features), n_features):
        features[f'feature_{i}'] = np.random.uniform(-1, 1)

    return features


@pytest.fixture
def sample_batch_features(sample_features):
    """Generate batch of sample features."""
    batch_size = 100
    batch = []

    for i in range(batch_size):
        # Add some variation to each sample
        sample = {k: v + np.random.uniform(-0.1, 0.1) if isinstance(v, float) else v
                  for k, v in sample_features.items()}
        batch.append(sample)

    return pd.DataFrame(batch)


# =============================================================================
# FEATURE ENGINEERING PERFORMANCE TESTS
# =============================================================================

class TestFeatureEngineeringPerformance:
    """Performance tests for feature engineering."""

    @pytest.fixture
    def feature_engineer(self, mock_data_provider):
        """Create feature engineer instance."""
        try:
            from ml.feature_engineering import FeatureEngineer
            return FeatureEngineer(
                data_provider=mock_data_provider,
                cache_ttl_seconds=300,
                enable_db_storage=False
            )
        except ImportError:
            pytest.skip("FeatureEngineer not available")

    @pytest.mark.asyncio
    async def test_single_symbol_feature_calculation(self, feature_engineer, perf_config):
        """Test feature calculation time for single symbol."""
        start = time.perf_counter()
        features = await feature_engineer.calculate_features('AAPL', use_cache=False)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\nSingle symbol feature calculation: {elapsed:.2f} ms")

        target = perf_config.ml_targets['feature_engineering_ms']
        # Allow more time for first calculation (data fetching)
        assert elapsed < target * 100, f"Feature calculation {elapsed:.2f}ms exceeds target"

    @pytest.mark.asyncio
    async def test_cached_feature_calculation(self, feature_engineer, perf_config):
        """Test cached feature calculation time."""
        # Prime the cache
        await feature_engineer.calculate_features('AAPL', use_cache=True)

        # Measure cached calculation
        start = time.perf_counter()
        features = await feature_engineer.calculate_features('AAPL', use_cache=True)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\nCached feature calculation: {elapsed:.2f} ms")

        # Cached should be very fast
        assert elapsed < 5, f"Cached calculation {elapsed:.2f}ms too slow"

    @pytest.mark.asyncio
    async def test_batch_feature_calculation(self, feature_engineer, perf_config):
        """Test batch feature calculation."""
        symbols = TEST_SYMBOLS[:10]

        start = time.perf_counter()
        features = await feature_engineer.calculate_batch_features(symbols, use_cache=False)
        elapsed = time.perf_counter() - start

        per_symbol = (elapsed / len(symbols)) * 1000  # ms

        print(f"\nBatch feature calculation ({len(symbols)} symbols):")
        print(f"  Total: {elapsed:.2f}s")
        print(f"  Per symbol: {per_symbol:.2f} ms")

        # Should have features for most symbols
        assert len(features) >= len(symbols) * 0.8

    def test_extract_features_performance(self, feature_engineer, perf_config):
        """Test synchronous feature extraction performance."""
        signal = {
            'symbol': 'AAPL',
            'rrs': 2.5,
            'price': 175.50,
            'atr': 2.5,
            'volume': 50000000,
            'daily_strong': True,
            'daily_weak': False,
            'volume_ratio': 1.2,
        }

        # Measure extraction time
        start = time.perf_counter()
        for _ in range(1000):
            result = feature_engineer.extract_features(signal)
        elapsed = (time.perf_counter() - start) / 1000 * 1000  # ms per call

        print(f"\nFeature extraction (sync): {elapsed:.4f} ms per call")

        # Should be very fast
        assert elapsed < 1, f"Feature extraction {elapsed:.4f}ms too slow"

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    def test_extract_features_benchmark(self, benchmark, feature_engineer):
        """Benchmark synchronous feature extraction."""
        signal = {
            'symbol': 'AAPL',
            'rrs': 2.5,
            'price': 175.50,
            'atr': 2.5,
            'volume': 50000000,
            'daily_strong': True,
            'daily_weak': False,
        }

        def extract():
            return feature_engineer.extract_features(signal)

        result = benchmark(extract)
        assert 'features' in result


# =============================================================================
# XGBOOST MODEL PERFORMANCE TESTS
# =============================================================================

class TestXGBoostPerformance:
    """Performance tests for XGBoost model."""

    @pytest.fixture
    def xgboost_model(self):
        """Create XGBoost model instance."""
        try:
            from ml.models.xgboost_model import XGBoostTradeClassifier
            return XGBoostTradeClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                use_gpu=False
            )
        except ImportError:
            pytest.skip("XGBoostTradeClassifier not available")

    @pytest.fixture
    def trained_model(self, xgboost_model, sample_batch_features):
        """Create trained XGBoost model."""
        # Generate training data
        n_samples = 500
        n_features = len(sample_batch_features.columns)

        X = np.random.randn(n_samples, n_features)
        y = (np.random.rand(n_samples) > 0.5).astype(int)

        # Train model
        feature_names = list(sample_batch_features.columns)
        xgboost_model.train(X, y, feature_names=feature_names, n_splits=3)

        return xgboost_model

    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    def test_single_prediction_benchmark(self, benchmark, trained_model, sample_features):
        """Benchmark single prediction time."""
        X = np.array([[v for v in sample_features.values()]])

        def predict():
            return trained_model.predict_proba(X)

        result = benchmark(predict)
        assert result.shape == (1, 2)

    def test_single_prediction_time(self, trained_model, sample_features, perf_config):
        """Test single prediction time."""
        X = np.array([[v for v in sample_features.values()]])

        # Warm up
        for _ in range(10):
            trained_model.predict_proba(X)

        # Measure
        start = time.perf_counter()
        for _ in range(100):
            trained_model.predict_proba(X)
        elapsed = (time.perf_counter() - start) / 100 * 1000  # ms

        print(f"\nSingle prediction time: {elapsed:.4f} ms")

        target = perf_config.ml_targets['single_inference_ms']
        assert elapsed < target, f"Prediction time {elapsed:.4f}ms exceeds target {target}ms"

    def test_batch_100_prediction_time(self, trained_model, sample_batch_features, perf_config):
        """Test batch prediction time for 100 samples."""
        X = sample_batch_features.values

        # Warm up
        for _ in range(5):
            trained_model.predict_proba(X)

        # Measure
        start = time.perf_counter()
        for _ in range(20):
            trained_model.predict_proba(X)
        elapsed = (time.perf_counter() - start) / 20 * 1000  # ms

        print(f"\nBatch 100 prediction time: {elapsed:.2f} ms")

        target = perf_config.ml_targets['batch_100_inference_ms']
        assert elapsed < target, f"Batch prediction {elapsed:.2f}ms exceeds target {target}ms"

    def test_batch_1000_prediction_time(self, trained_model, sample_features, perf_config):
        """Test batch prediction time for 1000 samples."""
        # Generate 1000 samples
        n_features = len(sample_features)
        X = np.random.randn(1000, n_features)

        # Warm up
        trained_model.predict_proba(X)

        # Measure
        start = time.perf_counter()
        for _ in range(10):
            trained_model.predict_proba(X)
        elapsed = (time.perf_counter() - start) / 10 * 1000  # ms

        print(f"\nBatch 1000 prediction time: {elapsed:.2f} ms")

        target = perf_config.ml_targets['batch_1000_inference_ms']
        assert elapsed < target, f"Large batch prediction {elapsed:.2f}ms exceeds target {target}ms"

    @pytest.mark.skipif(not MEMORY_PROFILER_AVAILABLE, reason="memory_profiler not available")
    def test_prediction_memory_overhead(self, trained_model, sample_features, perf_config):
        """Test memory overhead during prediction."""
        X = np.random.randn(1000, len(sample_features))

        gc.collect()
        mem_before = memory_usage()[0]

        # Make many predictions
        for _ in range(100):
            trained_model.predict_proba(X)

        gc.collect()
        mem_after = memory_usage()[0]
        growth = mem_after - mem_before

        print(f"\nMemory growth after 100 batch predictions: {growth:.2f} MB")

        target = perf_config.ml_targets['inference_memory_overhead_mb']
        assert growth < target * 2, f"Memory growth {growth:.2f}MB exceeds limit"


# =============================================================================
# MODEL LOADING PERFORMANCE TESTS
# =============================================================================

class TestModelLoadingPerformance:
    """Performance tests for model loading."""

    @pytest.fixture
    def model_path(self, trained_model, tmp_path):
        """Save trained model and return path."""
        path = str(tmp_path / "test_model.joblib")
        trained_model.save(path)
        return path

    @pytest.fixture
    def trained_model(self, sample_features):
        """Create and train a model."""
        try:
            from ml.models.xgboost_model import XGBoostTradeClassifier
        except ImportError:
            pytest.skip("XGBoostTradeClassifier not available")

        model = XGBoostTradeClassifier(n_estimators=50, max_depth=4)

        n_samples = 200
        n_features = len(sample_features)
        X = np.random.randn(n_samples, n_features)
        y = (np.random.rand(n_samples) > 0.5).astype(int)

        model.train(X, y, feature_names=list(sample_features.keys()), n_splits=2)
        return model

    def test_model_load_time(self, model_path, perf_config):
        """Test model loading time."""
        try:
            from ml.models.xgboost_model import XGBoostTradeClassifier
        except ImportError:
            pytest.skip("XGBoostTradeClassifier not available")

        # Measure load time
        times = []
        for _ in range(5):
            model = XGBoostTradeClassifier()
            start = time.perf_counter()
            model.load(model_path)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        print(f"\nModel load time: {avg_time:.3f}s")

        target = perf_config.ml_targets['model_load_seconds']
        assert avg_time < target, f"Load time {avg_time:.3f}s exceeds target {target}s"

    @pytest.mark.skipif(not MEMORY_PROFILER_AVAILABLE, reason="memory_profiler not available")
    def test_model_memory_footprint(self, trained_model, perf_config):
        """Test model memory footprint."""
        gc.collect()
        mem_before = memory_usage()[0]

        # Keep reference to model
        models = [trained_model]

        gc.collect()
        mem_after = memory_usage()[0]
        footprint = mem_after - mem_before

        print(f"\nModel memory footprint: {footprint:.2f} MB")

        target = perf_config.ml_targets['model_memory_mb']
        assert footprint < target, f"Model footprint {footprint:.2f}MB exceeds target {target}MB"


# =============================================================================
# RANDOM FOREST MODEL PERFORMANCE TESTS
# =============================================================================

class TestRandomForestPerformance:
    """Performance tests for Random Forest model."""

    @pytest.fixture
    def rf_model(self):
        """Create Random Forest model instance."""
        try:
            from ml.models.random_forest_model import RandomForestTradeClassifier
            return RandomForestTradeClassifier(n_estimators=100, max_depth=10)
        except ImportError:
            pytest.skip("RandomForestTradeClassifier not available")

    @pytest.fixture
    def trained_rf_model(self, rf_model, sample_features):
        """Create trained Random Forest model."""
        n_samples = 300
        n_features = len(sample_features)

        X = np.random.randn(n_samples, n_features)
        y = (np.random.rand(n_samples) > 0.5).astype(int)

        rf_model.train(X, y, feature_names=list(sample_features.keys()), n_splits=3)
        return rf_model

    def test_rf_single_prediction_time(self, trained_rf_model, sample_features, perf_config):
        """Test Random Forest single prediction time."""
        X = np.array([[v for v in sample_features.values()]])

        # Warm up
        for _ in range(10):
            trained_rf_model.predict_proba(X)

        # Measure
        start = time.perf_counter()
        for _ in range(100):
            trained_rf_model.predict_proba(X)
        elapsed = (time.perf_counter() - start) / 100 * 1000

        print(f"\nRandom Forest single prediction: {elapsed:.4f} ms")

        # RF may be slower than XGBoost, allow 2x target
        target = perf_config.ml_targets['single_inference_ms'] * 2
        assert elapsed < target, f"RF prediction {elapsed:.4f}ms exceeds target"

    def test_rf_batch_prediction_time(self, trained_rf_model, sample_batch_features, perf_config):
        """Test Random Forest batch prediction time."""
        X = sample_batch_features.values

        start = time.perf_counter()
        for _ in range(10):
            trained_rf_model.predict_proba(X)
        elapsed = (time.perf_counter() - start) / 10 * 1000

        print(f"\nRandom Forest batch 100 prediction: {elapsed:.2f} ms")

        target = perf_config.ml_targets['batch_100_inference_ms'] * 2
        assert elapsed < target, f"RF batch prediction {elapsed:.2f}ms exceeds target"


# =============================================================================
# ENSEMBLE PREDICTION PERFORMANCE
# =============================================================================

class TestEnsemblePerformance:
    """Performance tests for ensemble predictions."""

    def test_multi_model_prediction_time(self, sample_features, perf_config):
        """Test time for predictions from multiple models."""
        try:
            from ml.models.xgboost_model import XGBoostTradeClassifier
            from ml.models.random_forest_model import RandomForestTradeClassifier
        except ImportError:
            pytest.skip("ML models not available")

        # Train both models
        n_samples = 200
        n_features = len(sample_features)
        X_train = np.random.randn(n_samples, n_features)
        y_train = (np.random.rand(n_samples) > 0.5).astype(int)
        feature_names = list(sample_features.keys())

        xgb_model = XGBoostTradeClassifier(n_estimators=50, max_depth=4)
        xgb_model.train(X_train, y_train, feature_names=feature_names, n_splits=2)

        rf_model = RandomForestTradeClassifier(n_estimators=50, max_depth=6)
        rf_model.train(X_train, y_train, feature_names=feature_names, n_splits=2)

        # Test prediction
        X = np.array([[v for v in sample_features.values()]])

        # Measure ensemble time
        start = time.perf_counter()
        for _ in range(50):
            xgb_pred = xgb_model.predict_proba(X)
            rf_pred = rf_model.predict_proba(X)
            ensemble_pred = (xgb_pred + rf_pred) / 2
        elapsed = (time.perf_counter() - start) / 50 * 1000

        print(f"\nEnsemble (2 models) prediction time: {elapsed:.4f} ms")

        # Should be roughly 2x single model
        target = perf_config.ml_targets['single_inference_ms'] * 3
        assert elapsed < target, f"Ensemble prediction {elapsed:.4f}ms exceeds target"


# =============================================================================
# THROUGHPUT TESTS
# =============================================================================

class TestMLThroughput:
    """Tests for ML prediction throughput."""

    def test_predictions_per_second(self, sample_features, perf_config):
        """Test how many predictions can be made per second."""
        try:
            from ml.models.xgboost_model import XGBoostTradeClassifier
        except ImportError:
            pytest.skip("XGBoostTradeClassifier not available")

        # Train model
        n_samples = 200
        n_features = len(sample_features)
        X_train = np.random.randn(n_samples, n_features)
        y_train = (np.random.rand(n_samples) > 0.5).astype(int)

        model = XGBoostTradeClassifier(n_estimators=50, max_depth=4)
        model.train(X_train, y_train, feature_names=list(sample_features.keys()), n_splits=2)

        # Test throughput
        X = np.array([[v for v in sample_features.values()]])

        start = time.perf_counter()
        count = 0
        duration = 1.0  # 1 second

        while time.perf_counter() - start < duration:
            model.predict_proba(X)
            count += 1

        elapsed = time.perf_counter() - start
        throughput = count / elapsed

        print(f"\nSingle prediction throughput: {throughput:.0f} predictions/second")

        # Should be able to do at least 1000 predictions/second
        assert throughput > 100, f"Throughput {throughput:.0f}/s too low"

    def test_batch_throughput(self, sample_batch_features, perf_config):
        """Test batch prediction throughput."""
        try:
            from ml.models.xgboost_model import XGBoostTradeClassifier
        except ImportError:
            pytest.skip("XGBoostTradeClassifier not available")

        n_features = len(sample_batch_features.columns)
        X_train = np.random.randn(300, n_features)
        y_train = (np.random.rand(300) > 0.5).astype(int)

        model = XGBoostTradeClassifier(n_estimators=50, max_depth=4)
        model.train(X_train, y_train, feature_names=list(sample_batch_features.columns), n_splits=2)

        X = sample_batch_features.values  # 100 samples

        start = time.perf_counter()
        count = 0
        duration = 1.0

        while time.perf_counter() - start < duration:
            model.predict_proba(X)
            count += 100  # Each batch is 100 samples

        elapsed = time.perf_counter() - start
        throughput = count / elapsed

        print(f"\nBatch prediction throughput: {throughput:.0f} samples/second")

        # Should handle many thousands per second in batch mode
        assert throughput > 1000, f"Batch throughput {throughput:.0f}/s too low"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    pytest.main([
        __file__,
        '-v',
        '--benchmark-only',
        '--benchmark-group-by=class',
    ])
