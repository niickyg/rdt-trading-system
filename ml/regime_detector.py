"""
Market Regime Detection System using Hidden Markov Models

This module implements a sophisticated market regime detection system that uses
HMM to identify different market states and allocate strategies accordingly.
"""

# Fix curl_cffi chrome136 impersonation issue in Docker
try:
    from curl_cffi.requests import impersonate
    impersonate.DEFAULT_CHROME = 'chrome110'
    if hasattr(impersonate, 'REAL_TARGET_MAP'):
        impersonate.REAL_TARGET_MAP['chrome'] = 'chrome110'
except ImportError:
    pass

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
import pickle
import warnings
from pathlib import Path
from loguru import logger
from ml.safe_model_loader import safe_load_model, safe_save_model, ModelSecurityError
from utils.paths import get_project_root

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    StandardScaler = None
    silhouette_score = None
    davies_bouldin_score = None
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    GaussianHMM = None
    logger.warning("hmmlearn not available. Install with: pip install hmmlearn")

YF_AVAILABLE = False  # yfinance removed — using IBKR/DB cache for historical data

warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='sklearn')


class MarketRegimeDetector:
    """
    Detects market regimes using Hidden Markov Models and allocates strategies
    based on the detected regime.

    Regimes:
        0: Bull Trending - Strong upward momentum
        1: Bear Trending - Strong downward momentum
        2: High Volatility - Elevated market uncertainty
        3: Low Volatility - Calm market conditions
    """

    REGIMES = {
        0: 'bull_trending',
        1: 'bear_trending',
        2: 'high_volatility',
        3: 'low_volatility'
    }

    STRATEGY_ALLOCATIONS = {
        'bull_trending': {
            'momentum': 0.60,
            'mean_reversion': 0.10,
            'pairs': 0.20,
            'options': 0.10
        },
        'bear_trending': {
            'momentum': 0.30,
            'mean_reversion': 0.30,
            'pairs': 0.20,
            'options': 0.20
        },
        'high_volatility': {
            'momentum': 0.20,
            'mean_reversion': 0.20,
            'pairs': 0.20,
            'options': 0.40
        },
        'low_volatility': {
            'momentum': 0.40,
            'mean_reversion': 0.40,
            'pairs': 0.10,
            'options': 0.10
        }
    }

    def __init__(
        self,
        n_regimes: int = 4,
        n_iter: int = 100,
        random_state: int = 42,
        model_path: Optional[str] = None
    ):
        """
        Initialize the Market Regime Detector.

        Args:
            n_regimes: Number of market regimes to detect
            n_iter: Number of EM iterations for HMM training
            random_state: Random seed for reproducibility
            model_path: Path to save/load the trained model
        """
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state
        self.model_path = model_path or str(get_project_root() / "models" / "regime_detector.pkl")

        # Initialize HMM model (if available)
        if HMM_AVAILABLE:
            self.model = GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=n_iter,
                random_state=random_state,
                verbose=False
            )
        else:
            self.model = None
            logger.warning("HMM model not available - HMM-based detection disabled")

        # Feature scaler
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None

        # Training data storage
        self.features_train = None
        self.raw_data = None
        self.feature_names = None

        # Regime mapping (will be learned during training)
        self.regime_mapping = None

        logger.info(f"Initialized MarketRegimeDetector with {n_regimes} regimes")

    def fetch_data(
        self,
        symbol: str = "SPY",
        period: str = "5y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical market data from DB cache or IBKR.

        For short periods (3mo or less), uses the PostgreSQL daily bar cache.
        For longer periods (training), attempts IBKR direct historical request.

        Args:
            symbol: Ticker symbol to fetch
            period: Data period (e.g. '3mo', '5y')
            interval: Data interval (1d = daily)

        Returns:
            DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
        """
        try:
            from data.database.historical_cache import get_historical_cache
            cache = get_historical_cache()

            # Map period to lookback days
            period_map = {
                '1mo': 30, '3mo': 90, '6mo': 180,
                '1y': 365, '2y': 730, '5y': 1825,
            }
            lookback = period_map.get(period, 90)

            # For short lookbacks, use DB cache
            if lookback <= 365:
                df = cache.get_daily_bars(symbol, lookback_days=lookback)
                if df is not None and not df.empty:
                    # Uppercase columns for compatibility with existing code
                    df_out = df.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume',
                    })
                    logger.info(f"Fetched {len(df_out)} data points for {symbol} from DB cache")
                    return df_out

            # For longer periods or if DB cache is empty, try IBKR
            # Map period to IBKR duration string
            ibkr_duration_map = {
                '1mo': '1 M', '3mo': '3 M', '6mo': '6 M',
                '1y': '1 Y', '2y': '2 Y', '5y': '5 Y',
            }
            ibkr_duration = ibkr_duration_map.get(period, '3 M')

            # Try to get IBKR client
            try:
                from web.trading_init import initialize_all_components
                # Import won't re-initialize if already done
            except ImportError:
                pass

            # Try the broker's get_historical_bars
            try:
                from brokers.ibkr.client import IBKRClient
                # Access global broker if available
                import gc
                for obj in gc.get_referrers(IBKRClient):
                    if isinstance(obj, dict):
                        for v in obj.values():
                            if isinstance(v, IBKRClient) and v._connected:
                                df = v.get_historical_bars(symbol, ibkr_duration, '1 day')
                                if df is not None and not df.empty:
                                    df_out = df.rename(columns={
                                        'open': 'Open', 'high': 'High', 'low': 'Low',
                                        'close': 'Close', 'volume': 'Volume',
                                    })
                                    logger.info(f"Fetched {len(df_out)} data points for {symbol} from IBKR")
                                    return df_out
            except Exception:
                pass

            raise ValueError(f"No data available for {symbol} (period={period})")

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for regime classification.

        Features:
            - Returns (1d, 5d, 20d)
            - Volatility (10d, 30d rolling std)
            - Volume ratios
            - Trend indicators (SMA crossovers)

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with extracted features
        """
        logger.info("Extracting features for regime detection...")

        df = data.copy()
        features = pd.DataFrame(index=df.index)

        # Returns
        features['return_1d'] = df['Close'].pct_change(1)
        features['return_5d'] = df['Close'].pct_change(5)
        features['return_20d'] = df['Close'].pct_change(20)

        # Volatility
        features['volatility_10d'] = df['Close'].pct_change().rolling(window=10).std()
        features['volatility_30d'] = df['Close'].pct_change().rolling(window=30).std()

        # Volume analysis
        features['volume_ratio_5d'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
        features['volume_ratio_20d'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

        # Trend indicators
        features['sma_10'] = df['Close'].rolling(window=10).mean()
        features['sma_50'] = df['Close'].rolling(window=50).mean()
        features['sma_200'] = df['Close'].rolling(window=200).mean()

        # SMA crossovers (normalized)
        features['sma_10_50_cross'] = (features['sma_10'] - features['sma_50']) / df['Close']
        features['sma_50_200_cross'] = (features['sma_50'] - features['sma_200']) / df['Close']

        # Price momentum
        features['momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
        features['momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1

        # High-Low range
        features['hl_ratio'] = (df['High'] - df['Low']) / df['Close']

        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr_14'] = true_range.rolling(window=14).mean() / df['Close']

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))

        # Remove SMA columns (only keep crossovers)
        features = features.drop(['sma_10', 'sma_50', 'sma_200'], axis=1)

        # Drop NaN values
        features = features.dropna()

        logger.info(f"Extracted {len(features.columns)} features from {len(features)} samples")
        self.feature_names = features.columns.tolist()

        return features

    def train(
        self,
        symbol: str = "SPY",
        period: str = "5y",
        interval: str = "1d"
    ) -> Dict:
        """
        Train the HMM model on historical data.

        Args:
            symbol: Ticker symbol to train on
            period: Data period
            interval: Data interval

        Returns:
            Dictionary with training metrics
        """
        try:
            logger.info("Starting HMM training...")

            # Fetch and prepare data
            self.raw_data = self.fetch_data(symbol, period, interval)
            features = self.extract_features(self.raw_data)

            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            self.features_train = features_scaled

            # Train HMM
            logger.info("Training Hidden Markov Model...")
            self.model.fit(features_scaled)

            # Predict regimes for training data
            regimes = self.model.predict(features_scaled)

            # Map regimes to meaningful labels
            self.regime_mapping = self._map_regimes(features, regimes)

            # Calculate training metrics
            metrics = self._calculate_metrics(features_scaled, regimes, features)

            logger.success("HMM training completed successfully")
            logger.info(f"Training metrics: {metrics}")

            return metrics

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def _map_regimes(
        self,
        features: pd.DataFrame,
        regimes: np.ndarray
    ) -> Dict[int, str]:
        """
        Map HMM states to meaningful regime labels based on feature characteristics.

        Args:
            features: Feature DataFrame
            regimes: Predicted regime sequence

        Returns:
            Mapping from HMM state to regime name
        """
        logger.info("Mapping HMM states to regime labels...")

        regime_stats = {}

        for state in range(self.n_regimes):
            mask = regimes == state
            state_features = features[mask]

            if len(state_features) == 0:
                continue

            regime_stats[state] = {
                'avg_return_20d': state_features['return_20d'].mean(),
                'avg_volatility_30d': state_features['volatility_30d'].mean(),
                'count': mask.sum()
            }

        # Sort states by characteristics
        mapping = {}
        available_states = list(regime_stats.keys())

        if len(available_states) == 0:
            logger.warning("No regime states found with data")
            return {i: ['low_volatility', 'high_volatility', 'bull_trending', 'bear_trending'][i % 4]
                    for i in range(self.n_regimes)}

        # Find bull trending: high positive returns, moderate volatility
        bull_state = max(
            available_states,
            key=lambda x: regime_stats[x]['avg_return_20d']
        )
        mapping[bull_state] = 'bull_trending'
        available_states.remove(bull_state)

        # Find bear trending: high negative returns (if states remain)
        if available_states:
            bear_state = min(
                available_states,
                key=lambda x: regime_stats[x]['avg_return_20d']
            )
            mapping[bear_state] = 'bear_trending'
            available_states.remove(bear_state)

        # Find high volatility: highest volatility (if states remain)
        if available_states:
            high_vol_state = max(
                available_states,
                key=lambda x: regime_stats[x]['avg_volatility_30d']
            )
            mapping[high_vol_state] = 'high_volatility'
            available_states.remove(high_vol_state)

        # Remaining states are low volatility
        for state in available_states:
            mapping[state] = 'low_volatility'

        # Fill in any missing regimes (states with no samples)
        all_regimes = ['bull_trending', 'bear_trending', 'high_volatility', 'low_volatility']
        for state in range(self.n_regimes):
            if state not in mapping:
                # Assign based on state number
                mapping[state] = all_regimes[state % 4]

        logger.info(f"Regime mapping: {mapping}")
        for state, regime in mapping.items():
            if state in regime_stats:
                stats = regime_stats[state]
                logger.info(f"  {regime}: avg_return={stats['avg_return_20d']:.4f}, "
                           f"avg_vol={stats['avg_volatility_30d']:.4f}, samples={stats['count']}")
            else:
                logger.info(f"  {regime}: (no samples in this state)")

        return mapping

    def _calculate_metrics(
        self,
        features_scaled: np.ndarray,
        regimes: np.ndarray,
        features_raw: pd.DataFrame
    ) -> Dict:
        """
        Calculate validation metrics for the trained model.

        Args:
            features_scaled: Scaled features
            regimes: Predicted regimes
            features_raw: Raw features

        Returns:
            Dictionary with metrics
        """
        metrics = {
            'log_likelihood': self.model.score(features_scaled),
            'n_samples': len(features_scaled),
            'n_features': features_scaled.shape[1]
        }

        # Regime distribution
        unique, counts = np.unique(regimes, return_counts=True)
        regime_dist = {
            self.regime_mapping.get(state, f'state_{state}'): int(count)
            for state, count in zip(unique, counts)
        }
        metrics['regime_distribution'] = regime_dist

        # Clustering quality metrics
        try:
            if len(unique) > 1:
                metrics['silhouette_score'] = float(silhouette_score(features_scaled, regimes))
                metrics['davies_bouldin_score'] = float(davies_bouldin_score(features_scaled, regimes))
        except Exception as e:
            logger.warning(f"Could not calculate clustering metrics: {e}")

        # Transition matrix
        metrics['transition_matrix'] = self.model.transmat_.tolist()

        return metrics

    def predict(
        self,
        data: Optional[pd.DataFrame] = None,
        return_confidence: bool = True
    ) -> Tuple[str, Dict]:
        """
        Predict the current market regime.

        Args:
            data: Recent market data (if None, fetches latest)
            return_confidence: Whether to return confidence scores

        Returns:
            Tuple of (regime_name, additional_info)
        """
        try:
            # Fetch recent data if not provided
            if data is None:
                data = self.fetch_data(symbol="SPY", period="3mo", interval="1d")

            # Extract features
            features = self.extract_features(data)

            # Use only the last observation for current regime
            features_latest = features.iloc[[-1]]
            features_scaled = self.scaler.transform(features_latest)

            # Predict regime
            regime_state = self.model.predict(features_scaled)[0]
            regime_name = self.regime_mapping.get(regime_state, f'unknown_state_{regime_state}')

            # Calculate confidence scores
            info = {
                'regime_state': int(regime_state),
                'regime_name': regime_name,
                'timestamp': data.index[-1]
            }

            if return_confidence:
                # Get posterior probabilities
                log_prob, posteriors = self.model.score_samples(features_scaled)
                confidence_scores = {
                    self.regime_mapping.get(i, f'state_{i}'): float(posteriors[0, i])
                    for i in range(self.n_regimes)
                }
                info['confidence_scores'] = confidence_scores
                info['log_likelihood'] = float(log_prob)

            # Get strategy allocation
            info['strategy_allocation'] = self.get_strategy_allocation(regime_name)

            logger.info(f"Current regime: {regime_name} (confidence: {info.get('confidence_scores', {}).get(regime_name, 'N/A')})")

            return regime_name, info

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def predict_sequence(
        self,
        data: pd.DataFrame,
        window_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Predict regime sequence for a time series.

        Args:
            data: Historical market data
            window_size: If specified, uses rolling window prediction

        Returns:
            DataFrame with regime predictions over time
        """
        try:
            features = self.extract_features(data)
            features_scaled = self.scaler.transform(features)

            # Predict regimes
            regimes = self.model.predict(features_scaled)

            # Get posterior probabilities
            log_prob, posteriors = self.model.score_samples(features_scaled)

            # Create results DataFrame
            results = pd.DataFrame(index=features.index)
            results['regime_state'] = regimes
            results['regime_name'] = [
                self.regime_mapping.get(s, f'state_{s}') for s in regimes
            ]

            # Add confidence scores
            for i in range(self.n_regimes):
                regime_name = self.regime_mapping.get(i, f'state_{i}')
                results[f'confidence_{regime_name}'] = posteriors[:, i]

            logger.info(f"Predicted regime sequence for {len(results)} periods")

            return results

        except Exception as e:
            logger.error(f"Error predicting sequence: {e}")
            raise

    def get_strategy_allocation(self, regime_name: str) -> Dict[str, float]:
        """
        Get strategy allocation weights for a given regime.

        Args:
            regime_name: Name of the market regime

        Returns:
            Dictionary with strategy allocations
        """
        allocation = self.STRATEGY_ALLOCATIONS.get(
            regime_name,
            {'momentum': 0.25, 'mean_reversion': 0.25, 'pairs': 0.25, 'options': 0.25}
        )

        logger.debug(f"Strategy allocation for {regime_name}: {allocation}")
        return allocation

    def save_model(self, path: Optional[str] = None) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model (uses self.model_path if None)
        """
        try:
            save_path = path or self.model_path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'regime_mapping': self.regime_mapping,
                'feature_names': self.feature_names,
                'n_regimes': self.n_regimes,
                'trained_at': datetime.now()
            }

            safe_save_model(model_data, str(save_path))

            logger.success(f"Model saved to {save_path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, path: Optional[str] = None) -> None:
        """
        Load a trained model from disk.

        Args:
            path: Path to load the model from (uses self.model_path if None)
        """
        try:
            load_path = path or self.model_path

            model_data = safe_load_model(str(load_path), allow_unverified=False)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.regime_mapping = model_data['regime_mapping']
            self.feature_names = model_data['feature_names']
            self.n_regimes = model_data['n_regimes']

            logger.success(f"Model loaded from {load_path}")
            logger.info(f"Model trained at: {model_data.get('trained_at', 'Unknown')}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_regime_statistics(
        self,
        data: pd.DataFrame,
        regimes: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Calculate statistics for each regime.

        Args:
            data: Market data
            regimes: Predicted regimes (if None, predicts from data)

        Returns:
            DataFrame with regime statistics
        """
        try:
            if regimes is None:
                features = self.extract_features(data)
                features_scaled = self.scaler.transform(features)
                regimes = self.model.predict(features_scaled)
                data_aligned = data.loc[features.index]
            else:
                data_aligned = data

            stats = []

            for state in range(self.n_regimes):
                mask = regimes == state
                regime_data = data_aligned[mask]

                if len(regime_data) == 0:
                    continue

                regime_name = self.regime_mapping.get(state, f'state_{state}')
                returns = regime_data['Close'].pct_change()

                stat = {
                    'regime': regime_name,
                    'state': state,
                    'count': mask.sum(),
                    'avg_return': returns.mean(),
                    'volatility': returns.std(),
                    'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                    'max_drawdown': self._calculate_max_drawdown(regime_data['Close']),
                    'avg_volume': regime_data['Volume'].mean()
                }

                stats.append(stat)

            stats_df = pd.DataFrame(stats)
            logger.info("Regime statistics calculated")

            return stats_df

        except Exception as e:
            logger.error(f"Error calculating regime statistics: {e}")
            raise

    @staticmethod
    def _calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown from a price series."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def get_regime_transitions(
        self,
        results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze regime transitions.

        Args:
            results: DataFrame with regime predictions over time

        Returns:
            DataFrame with transition analysis
        """
        try:
            transitions = []

            for i in range(1, len(results)):
                if results['regime_name'].iloc[i] != results['regime_name'].iloc[i-1]:
                    transitions.append({
                        'date': results.index[i],
                        'from_regime': results['regime_name'].iloc[i-1],
                        'to_regime': results['regime_name'].iloc[i],
                        'from_state': results['regime_state'].iloc[i-1],
                        'to_state': results['regime_state'].iloc[i]
                    })

            transitions_df = pd.DataFrame(transitions)
            logger.info(f"Found {len(transitions)} regime transitions")

            return transitions_df

        except Exception as e:
            logger.error(f"Error analyzing regime transitions: {e}")
            raise


def main():
    """Example usage of MarketRegimeDetector."""

    # Initialize detector
    detector = MarketRegimeDetector(
        n_regimes=4,
        n_iter=100,
        random_state=42
    )

    # Train model
    logger.info("Training regime detector on SPY data...")
    metrics = detector.train(symbol="SPY", period="5y")

    logger.info("\nTraining Metrics:")
    for key, value in metrics.items():
        if key != 'transition_matrix':
            logger.info(f"  {key}: {value}")

    # Save model
    detector.save_model()

    # Predict current regime
    logger.info("\nPredicting current market regime...")
    regime, info = detector.predict(return_confidence=True)

    logger.info(f"\nCurrent Regime: {regime}")
    logger.info(f"Confidence Scores: {info['confidence_scores']}")
    logger.info(f"Strategy Allocation: {info['strategy_allocation']}")

    # Get regime statistics
    logger.info("\nCalculating regime statistics...")
    stats = detector.get_regime_statistics(detector.raw_data)
    logger.info("\n" + stats.to_string())


class RegimeDetector:
    """
    Simplified wrapper around MarketRegimeDetector for trading signal analysis.

    Provides a simpler interface for real-time regime detection without
    requiring full HMM model training for every signal.
    """

    # Regime confidence multipliers for trading
    REGIME_MULTIPLIERS = {
        'bull_trending': 1.1,      # Boost confidence in bull market
        'bear_trending': 0.9,      # Reduce confidence in bear market
        'high_volatility': 0.85,   # Reduce confidence in volatile market
        'low_volatility': 1.05     # Slight boost in calm market
    }

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize RegimeDetector wrapper.

        Args:
            model_path: Path to trained HMM model (optional)
        """
        self.detector = MarketRegimeDetector(model_path=model_path)
        self.current_regime = None
        self.current_confidence = 0.0
        self.last_update = None
        self.cache_duration = timedelta(hours=1)  # Cache regime for 1 hour

        # Try to load pre-trained model
        try:
            self.detector.load_model()
            logger.info("Loaded pre-trained regime detection model")
        except Exception as e:
            logger.warning(f"Could not load regime model: {e}. Using heuristic detection.")
            self.detector = None

    def detect_regime(self, signal: Dict) -> Tuple[str, float]:
        """
        Detect current market regime from signal data.

        Args:
            signal: Trading signal data

        Returns:
            Tuple of (regime_name, confidence)
        """
        # Check cache
        now = datetime.now()
        if (self.current_regime and self.last_update and
            now - self.last_update < self.cache_duration):
            logger.debug(f"Using cached regime: {self.current_regime} ({self.current_confidence:.2f})")
            return self.current_regime, self.current_confidence

        # If we have a trained model, use it
        if self.detector:
            try:
                regime_name, info = self.detector.predict(return_confidence=True)
                confidence_scores = info.get('confidence_scores', {})
                confidence = confidence_scores.get(regime_name, 0.5)

                # Update cache
                self.current_regime = regime_name
                self.current_confidence = confidence
                self.last_update = now

                logger.debug(f"Detected regime: {regime_name} ({confidence:.2f})")
                return regime_name, confidence

            except Exception as e:
                logger.warning(f"HMM regime detection failed: {e}. Falling back to heuristic.")

        # Fallback: Use heuristic regime detection based on signal data
        regime, confidence = self._detect_regime_heuristic(signal)

        # Update cache
        self.current_regime = regime
        self.current_confidence = confidence
        self.last_update = now

        return regime, confidence

    def _detect_regime_heuristic(self, signal: Dict) -> Tuple[str, float]:
        """
        Heuristic-based regime detection when HMM model is unavailable.

        Uses signal characteristics to estimate market regime.

        Args:
            signal: Trading signal data

        Returns:
            Tuple of (regime_name, confidence)
        """
        rrs = signal.get('rrs', 0)
        atr = signal.get('atr', 0)
        price = signal.get('price', 1)
        daily_strong = signal.get('daily_strong', False)
        daily_weak = signal.get('daily_weak', False)

        # Calculate volatility
        atr_percent = (atr / price * 100) if price > 0 else 0

        # Determine regime based on characteristics
        confidence = 0.7  # Base confidence for heuristic

        # High volatility regime
        if atr_percent > 4.0:
            return 'high_volatility', 0.75

        # Low volatility regime
        if atr_percent < 1.0:
            return 'low_volatility', 0.70

        # Bull trending regime
        if (daily_strong and rrs > 2.0) or rrs > 3.5:
            return 'bull_trending', 0.80

        # Bear trending regime
        if (daily_weak and rrs < -2.0) or rrs < -3.5:
            return 'bear_trending', 0.80

        # Default to low volatility if unclear
        return 'low_volatility', 0.60

    def get_regime_multiplier(self, regime_name: str) -> float:
        """
        Get confidence multiplier for a given regime.

        Different market regimes affect trading confidence differently.

        Args:
            regime_name: Name of the market regime

        Returns:
            Multiplier to apply to ML probability (0.0-1.2)
        """
        multiplier = self.REGIME_MULTIPLIERS.get(regime_name, 1.0)
        logger.debug(f"Regime multiplier for {regime_name}: {multiplier}")
        return multiplier

    def get_current_regime(self) -> Optional[str]:
        """Get the currently cached regime."""
        return self.current_regime

    def clear_cache(self):
        """Clear the regime cache to force new detection."""
        self.current_regime = None
        self.current_confidence = 0.0
        self.last_update = None
        logger.debug("Regime cache cleared")


if __name__ == "__main__":
    main()
