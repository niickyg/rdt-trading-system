"""
Unit tests for Market Regime Detector

Run with: pytest tests/test_regime_detector.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.regime_detector import MarketRegimeDetector


@pytest.fixture
def sample_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)

    data = pd.DataFrame({
        'Open': 100 + np.random.randn(len(dates)).cumsum(),
        'High': 101 + np.random.randn(len(dates)).cumsum(),
        'Low': 99 + np.random.randn(len(dates)).cumsum(),
        'Close': 100 + np.random.randn(len(dates)).cumsum(),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    # Ensure High >= Close >= Low
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

    return data


@pytest.fixture
def detector():
    """Create a detector instance for testing."""
    return MarketRegimeDetector(
        n_regimes=4,
        n_iter=10,  # Reduced for testing speed
        random_state=42
    )


class TestMarketRegimeDetector:
    """Test suite for MarketRegimeDetector class."""

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.n_regimes == 4
        assert detector.n_iter == 10
        assert detector.random_state == 42
        assert detector.model is not None
        assert detector.scaler is not None

    def test_regime_mappings(self, detector):
        """Test regime mappings are defined."""
        assert len(detector.REGIMES) == 4
        assert 'bull_trending' in detector.REGIMES.values()
        assert 'bear_trending' in detector.REGIMES.values()
        assert 'high_volatility' in detector.REGIMES.values()
        assert 'low_volatility' in detector.REGIMES.values()

    def test_strategy_allocations(self, detector):
        """Test strategy allocations sum to 1.0."""
        for regime, allocation in detector.STRATEGY_ALLOCATIONS.items():
            total = sum(allocation.values())
            assert abs(total - 1.0) < 1e-6, f"{regime} allocation doesn't sum to 1.0"

            # Check all strategies are present
            assert 'momentum' in allocation
            assert 'mean_reversion' in allocation
            assert 'pairs' in allocation
            assert 'options' in allocation

    def test_feature_extraction(self, detector, sample_data):
        """Test feature extraction from data."""
        features = detector.extract_features(sample_data)

        # Check features were extracted
        assert not features.empty
        assert len(features) < len(sample_data)  # Due to rolling windows

        # Check expected columns exist
        expected_features = [
            'return_1d', 'return_5d', 'return_20d',
            'volatility_10d', 'volatility_30d',
            'volume_ratio_5d', 'volume_ratio_20d',
            'sma_10_50_cross', 'sma_50_200_cross',
            'momentum_5d', 'momentum_20d',
            'hl_ratio', 'atr_14', 'rsi_14'
        ]

        for feature in expected_features:
            assert feature in features.columns, f"Missing feature: {feature}"

        # Check for NaN values
        assert not features.isnull().any().any(), "Features contain NaN values"

    def test_feature_names_stored(self, detector, sample_data):
        """Test that feature names are stored."""
        features = detector.extract_features(sample_data)
        assert detector.feature_names is not None
        assert len(detector.feature_names) == len(features.columns)

    def test_training(self, detector, sample_data, monkeypatch):
        """Test model training."""
        # Mock fetch_data to use sample data
        def mock_fetch(symbol, period, interval):
            return sample_data

        monkeypatch.setattr(detector, 'fetch_data', mock_fetch)

        # Train model
        metrics = detector.train(symbol="SPY", period="5y")

        # Check metrics were returned
        assert 'log_likelihood' in metrics
        assert 'n_samples' in metrics
        assert 'n_features' in metrics
        assert 'regime_distribution' in metrics
        assert 'transition_matrix' in metrics

        # Check model was trained
        assert detector.regime_mapping is not None
        assert len(detector.regime_mapping) == 4

        # Check scaler was fitted
        assert detector.scaler.mean_ is not None

    def test_regime_mapping(self, detector, sample_data, monkeypatch):
        """Test regime mapping logic."""
        def mock_fetch(symbol, period, interval):
            return sample_data

        monkeypatch.setattr(detector, 'fetch_data', mock_fetch)

        detector.train(symbol="SPY", period="5y")

        # Check all regimes are mapped
        mapped_regimes = set(detector.regime_mapping.values())
        expected_regimes = {'bull_trending', 'bear_trending', 'high_volatility', 'low_volatility'}
        assert mapped_regimes == expected_regimes

    def test_prediction(self, detector, sample_data, monkeypatch):
        """Test regime prediction."""
        def mock_fetch(symbol, period, interval):
            return sample_data

        monkeypatch.setattr(detector, 'fetch_data', mock_fetch)

        # Train first
        detector.train(symbol="SPY", period="5y")

        # Predict current regime
        regime, info = detector.predict(data=sample_data, return_confidence=True)

        # Check regime is valid
        assert regime in ['bull_trending', 'bear_trending', 'high_volatility', 'low_volatility']

        # Check info contains required fields
        assert 'regime_state' in info
        assert 'regime_name' in info
        assert 'timestamp' in info
        assert 'confidence_scores' in info
        assert 'strategy_allocation' in info

        # Check confidence scores
        confidence_scores = info['confidence_scores']
        assert len(confidence_scores) == 4
        assert abs(sum(confidence_scores.values()) - 1.0) < 1e-5

    def test_sequence_prediction(self, detector, sample_data, monkeypatch):
        """Test regime sequence prediction."""
        def mock_fetch(symbol, period, interval):
            return sample_data

        monkeypatch.setattr(detector, 'fetch_data', mock_fetch)

        detector.train(symbol="SPY", period="5y")

        # Predict sequence
        results = detector.predict_sequence(sample_data)

        # Check results
        assert not results.empty
        assert 'regime_state' in results.columns
        assert 'regime_name' in results.columns

        # Check confidence columns
        for regime in ['bull_trending', 'bear_trending', 'high_volatility', 'low_volatility']:
            assert f'confidence_{regime}' in results.columns

        # Check all regimes are valid
        for regime in results['regime_name']:
            assert regime in ['bull_trending', 'bear_trending', 'high_volatility', 'low_volatility']

    def test_strategy_allocation(self, detector):
        """Test strategy allocation retrieval."""
        for regime_name in ['bull_trending', 'bear_trending', 'high_volatility', 'low_volatility']:
            allocation = detector.get_strategy_allocation(regime_name)

            # Check allocation
            assert isinstance(allocation, dict)
            assert len(allocation) == 4
            assert abs(sum(allocation.values()) - 1.0) < 1e-6

    def test_save_load_model(self, detector, sample_data, monkeypatch):
        """Test model save/load functionality."""
        def mock_fetch(symbol, period, interval):
            return sample_data

        monkeypatch.setattr(detector, 'fetch_data', mock_fetch)

        # Train model
        detector.train(symbol="SPY", period="5y")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            detector.save_model(tmp_path)

            # Load into new detector
            new_detector = MarketRegimeDetector()
            new_detector.load_model(tmp_path)

            # Check model was loaded
            assert new_detector.model is not None
            assert new_detector.scaler is not None
            assert new_detector.regime_mapping is not None
            assert new_detector.feature_names is not None

            # Check predictions match
            regime1, _ = detector.predict(data=sample_data)
            regime2, _ = new_detector.predict(data=sample_data)
            assert regime1 == regime2

        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)

    def test_regime_statistics(self, detector, sample_data, monkeypatch):
        """Test regime statistics calculation."""
        def mock_fetch(symbol, period, interval):
            return sample_data

        monkeypatch.setattr(detector, 'fetch_data', mock_fetch)

        detector.train(symbol="SPY", period="5y")

        # Get statistics
        stats = detector.get_regime_statistics(sample_data)

        # Check statistics
        assert not stats.empty
        assert 'regime' in stats.columns
        assert 'avg_return' in stats.columns
        assert 'volatility' in stats.columns
        assert 'sharpe_ratio' in stats.columns
        assert 'max_drawdown' in stats.columns

        # Check all regimes are present
        assert len(stats) <= 4

    def test_regime_transitions(self, detector, sample_data, monkeypatch):
        """Test regime transition analysis."""
        def mock_fetch(symbol, period, interval):
            return sample_data

        monkeypatch.setattr(detector, 'fetch_data', mock_fetch)

        detector.train(symbol="SPY", period="5y")

        # Get sequence
        results = detector.predict_sequence(sample_data)

        # Get transitions
        transitions = detector.get_regime_transitions(results)

        # Check transitions
        assert isinstance(transitions, pd.DataFrame)
        if not transitions.empty:
            assert 'date' in transitions.columns
            assert 'from_regime' in transitions.columns
            assert 'to_regime' in transitions.columns

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create simple price series
        prices = pd.Series([100, 110, 105, 95, 100, 105])

        # Calculate drawdown
        dd = MarketRegimeDetector._calculate_max_drawdown(prices)

        # Check result
        assert dd < 0  # Drawdown should be negative
        assert dd > -1  # Should not be less than -100%

    def test_invalid_data_handling(self, detector):
        """Test handling of invalid data."""
        # Empty DataFrame
        empty_data = pd.DataFrame()

        with pytest.raises(Exception):
            detector.extract_features(empty_data)

    def test_different_regime_counts(self, sample_data, monkeypatch):
        """Test with different numbers of regimes."""
        for n_regimes in [2, 3, 5]:
            detector = MarketRegimeDetector(
                n_regimes=n_regimes,
                n_iter=10,
                random_state=42
            )

            def mock_fetch(symbol, period, interval):
                return sample_data

            monkeypatch.setattr(detector, 'fetch_data', mock_fetch)

            metrics = detector.train(symbol="SPY", period="5y")

            # Check correct number of regimes
            assert len(set(detector.regime_mapping.keys())) == n_regimes

    def test_confidence_scores_sum_to_one(self, detector, sample_data, monkeypatch):
        """Test that confidence scores sum to 1.0."""
        def mock_fetch(symbol, period, interval):
            return sample_data

        monkeypatch.setattr(detector, 'fetch_data', mock_fetch)

        detector.train(symbol="SPY", period="5y")

        # Get prediction with confidence
        _, info = detector.predict(data=sample_data, return_confidence=True)

        # Check sum
        total_confidence = sum(info['confidence_scores'].values())
        assert abs(total_confidence - 1.0) < 1e-5

    def test_reproducibility(self, sample_data, monkeypatch):
        """Test that results are reproducible with same random seed."""
        def mock_fetch(symbol, period, interval):
            return sample_data

        # Train two models with same seed
        detector1 = MarketRegimeDetector(n_regimes=4, n_iter=10, random_state=42)
        monkeypatch.setattr(detector1, 'fetch_data', mock_fetch)
        detector1.train(symbol="SPY", period="5y")

        detector2 = MarketRegimeDetector(n_regimes=4, n_iter=10, random_state=42)
        monkeypatch.setattr(detector2, 'fetch_data', mock_fetch)
        detector2.train(symbol="SPY", period="5y")

        # Check predictions match
        regime1, _ = detector1.predict(data=sample_data)
        regime2, _ = detector2.predict(data=sample_data)
        assert regime1 == regime2


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_full_workflow(self, sample_data, monkeypatch):
        """Test complete training and prediction workflow."""
        # Initialize detector
        detector = MarketRegimeDetector(n_regimes=4, n_iter=10, random_state=42)

        def mock_fetch(symbol, period, interval):
            return sample_data

        monkeypatch.setattr(detector, 'fetch_data', mock_fetch)

        # Train
        metrics = detector.train(symbol="SPY", period="5y")
        assert metrics is not None

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            detector.save_model(tmp_path)

            # Load model
            new_detector = MarketRegimeDetector()
            new_detector.load_model(tmp_path)

            # Predict
            regime, info = new_detector.predict(data=sample_data)
            assert regime is not None

            # Get allocation
            allocation = info['strategy_allocation']
            assert abs(sum(allocation.values()) - 1.0) < 1e-6

        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
