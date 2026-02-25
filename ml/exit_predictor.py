"""
ML-Based Optimal Exit Predictor

Predicts the optimal exit strategy for each trade based on market conditions
at entry time. Uses XGBoost to classify trades into MFE buckets:
  - Class 0: MFE < 1x ATR  (quick scalp or loser)
  - Class 1: MFE 1-2x ATR  (swing trade)
  - Class 2: MFE > 2x ATR  (runner)

Each class maps to a different exit strategy with tailored trailing stops
and profit targets, addressing the problem that static ATR-multiple targets
are suboptimal (1x wins 75% but leaves money on table, 3x only wins 42%).
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from ml.safe_model_loader import safe_load_model, safe_save_model, ModelSecurityError
from utils.paths import get_project_root, get_models_dir

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score
    )
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logger.warning("yfinance not available. Install with: pip install yfinance")

from shared.indicators.rrs import (
    RRSCalculator, calculate_ema, calculate_sma,
    check_daily_strength_relaxed, check_daily_weakness_relaxed
)


# ---------------------------------------------------------------------------
# Feature names (order matters -- must match training and prediction)
# ---------------------------------------------------------------------------
EXIT_FEATURE_NAMES = [
    'rrs_strength',          # RRS value at entry
    'atr_pct',               # ATR as % of price
    'volume_ratio',          # Current volume / 20-day average
    'hour_of_day',           # 0-23 (captures morning momentum vs afternoon fade)
    'day_of_week',           # 0-4 (Mon-Fri)
    'market_regime',         # 1=bull, 0=chop, -1=bear
    'dist_ema8_pct',         # Price distance from 8 EMA as %
    'dist_ema21_pct',        # Price distance from 21 EMA as %
    'dist_ema50_pct',        # Price distance from 50 EMA as %
    'ema_alignment',         # 1 if EMA3>EMA8>EMA21 (bull) or reversed (bear), else 0
    'rsi_14',                # RSI(14) at entry
    'bar_pattern_score',     # 3-bar pattern score before entry
    'daily_strength_score',  # Daily strength/weakness score (0-5)
    'price_momentum_5d',     # 5-day price momentum %
    'volume_trend',          # 5-day avg vol / 20-day avg vol
    'bb_width_pct',          # Bollinger Band width as % of price
    'price_position_range',  # Where price sits in recent range (0-1)
    'direction_is_long',     # 1 for long, 0 for short
]


@dataclass
class ExitPrediction:
    """Result of an exit strategy prediction."""
    mfe_class: int              # 0=scalp, 1=swing, 2=runner
    strategy: str               # 'quick_scalp', 'swing', 'runner'
    confidence: float           # Probability of the predicted class
    class_probabilities: Dict[str, float]  # Probabilities for all 3 classes
    recommended_target_r: float  # Target in R-multiples
    recommended_trail_r: float   # Trail amount in R-multiples
    trail_activation_r: float    # When to activate trailing stop (R)
    hold_estimate_hours: float   # Estimated hold time in trading hours


@dataclass
class ExitModelMetrics:
    """Metrics from model training/validation."""
    accuracy: float = 0.0
    f1_macro: float = 0.0
    class_report: str = ""
    confusion_matrix: Optional[np.ndarray] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_test: int = 0
    class_distribution: Dict[int, int] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            'accuracy': self.accuracy,
            'f1_macro': self.f1_macro,
            'class_report': self.class_report,
            'n_train': self.n_train,
            'n_test': self.n_test,
            'class_distribution': self.class_distribution,
            'feature_importance': self.feature_importance,
            'timestamp': self.timestamp,
        }


# ---------------------------------------------------------------------------
# Exit strategy parameters per class
# ---------------------------------------------------------------------------
EXIT_STRATEGIES: Dict[int, Dict[str, Any]] = {
    0: {  # quick_scalp
        'name': 'quick_scalp',
        'target_r': 1.0,
        'trail_r': 0.0,       # No trail -- just a fixed target
        'trail_activation_r': 0.0,
        'hold_hours': 2.0,
    },
    1: {  # swing
        'name': 'swing',
        'target_r': 2.0,
        'trail_r': 0.75,
        'trail_activation_r': 1.0,
        'hold_hours': 8.0,
    },
    2: {  # runner
        'name': 'runner',
        'target_r': 3.0,
        'trail_r': 1.0,
        'trail_activation_r': 1.5,
        'hold_hours': 16.0,
    },
}


class ExitPredictor:
    """
    XGBoost-based optimal exit predictor.

    Predicts whether a trade is likely to be a quick scalp (< 1x ATR MFE),
    a swing trade (1-2x ATR MFE), or a runner (> 2x ATR MFE), then
    recommends the appropriate exit strategy.

    Usage:
        predictor = ExitPredictor()

        # Training
        df = generate_exit_training_data(['AAPL', 'MSFT'], days=180)
        metrics = predictor.train(df)

        # Prediction
        features = predictor.extract_features(signal_data)
        prediction = predictor.predict(features)
    """

    def __init__(self, random_state: int = 42):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is required for ExitPredictor. "
                "Install with: pip install xgboost"
            )
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for ExitPredictor. "
                "Install with: pip install scikit-learn"
            )

        self.random_state = random_state
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        self.feature_names = EXIT_FEATURE_NAMES.copy()
        self.val_metrics: Optional[ExitModelMetrics] = None
        self.version = "1.0.0"

        self._rrs_calculator = RRSCalculator()

        logger.info("ExitPredictor initialized")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> ExitModelMetrics:
        """
        Train the exit predictor on historical trade data.

        Args:
            df: DataFrame with feature columns (EXIT_FEATURE_NAMES) and
                a 'target' column (0, 1, or 2).
            test_size: Fraction of data to hold out for validation.

        Returns:
            ExitModelMetrics with validation performance.
        """
        logger.info(f"Training ExitPredictor with {len(df)} samples")

        # Validate columns
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        if 'target' not in df.columns:
            raise ValueError("DataFrame must contain a 'target' column")

        X = df[self.feature_names].values.astype(np.float32)
        y = df['target'].values.astype(int)

        # Handle NaN / inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size,
            random_state=self.random_state, stratify=y
        )

        # Class weights to handle imbalance
        unique, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        sample_weights = np.ones(len(y_train))
        weight_map = {}
        for cls, cnt in zip(unique, counts):
            w = total / (len(unique) * cnt)
            weight_map[cls] = w
            sample_weights[y_train == cls] = w

        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            objective='multi:softprob',
            num_class=3,
            random_state=self.random_state,
            eval_metric='mlogloss',
            n_jobs=1,  # Single thread to avoid potential hangs
        )

        self.model.fit(X_train, y_train, sample_weight=sample_weights)

        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        report = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # Feature importance
        importances = self.model.feature_importances_
        fi_dict = {
            name: float(imp)
            for name, imp in sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1], reverse=True
            )
        }

        # Class distribution
        cls_dist = {int(c): int(cnt) for c, cnt in zip(*np.unique(y, return_counts=True))}

        self.val_metrics = ExitModelMetrics(
            accuracy=acc,
            f1_macro=f1,
            class_report=report,
            confusion_matrix=cm,
            feature_importance=fi_dict,
            n_train=len(X_train),
            n_test=len(X_test),
            class_distribution=cls_dist,
            timestamp=datetime.now().isoformat(),
        )

        logger.info(
            f"ExitPredictor training complete: accuracy={acc:.3f}, "
            f"F1(macro)={f1:.3f}, classes={cls_dist}"
        )
        logger.info(f"\n{report}")

        return self.val_metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, features: np.ndarray) -> ExitPrediction:
        """
        Predict optimal exit strategy for a single trade.

        Args:
            features: 1-D numpy array of shape (n_features,) matching
                      EXIT_FEATURE_NAMES ordering.

        Returns:
            ExitPrediction with recommended strategy.
        """
        if not self.is_trained or self.model is None or self.scaler is None:
            raise ValueError("ExitPredictor not trained. Call train() first or load() a saved model.")

        features = np.asarray(features, dtype=np.float32).reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_scaled = self.scaler.transform(features)

        proba = self.model.predict_proba(features_scaled)[0]
        pred_class = int(np.argmax(proba))

        strategy = EXIT_STRATEGIES[pred_class]

        return ExitPrediction(
            mfe_class=pred_class,
            strategy=strategy['name'],
            confidence=float(proba[pred_class]),
            class_probabilities={
                'quick_scalp': float(proba[0]),
                'swing': float(proba[1]),
                'runner': float(proba[2]),
            },
            recommended_target_r=strategy['target_r'],
            recommended_trail_r=strategy['trail_r'],
            trail_activation_r=strategy['trail_activation_r'],
            hold_estimate_hours=strategy['hold_hours'],
        )

    def predict_batch(self, features: np.ndarray) -> List[ExitPrediction]:
        """
        Predict optimal exit strategy for a batch of trades.

        Args:
            features: 2-D array of shape (n_samples, n_features).

        Returns:
            List of ExitPrediction objects.
        """
        if not self.is_trained or self.model is None or self.scaler is None:
            raise ValueError("ExitPredictor not trained.")

        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_scaled = self.scaler.transform(features)

        probas = self.model.predict_proba(features_scaled)
        pred_classes = np.argmax(probas, axis=1)

        results = []
        for i in range(len(features)):
            cls = int(pred_classes[i])
            proba = probas[i]
            strategy = EXIT_STRATEGIES[cls]
            results.append(ExitPrediction(
                mfe_class=cls,
                strategy=strategy['name'],
                confidence=float(proba[cls]),
                class_probabilities={
                    'quick_scalp': float(proba[0]),
                    'swing': float(proba[1]),
                    'runner': float(proba[2]),
                },
                recommended_target_r=strategy['target_r'],
                recommended_trail_r=strategy['trail_r'],
                trail_activation_r=strategy['trail_activation_r'],
                hold_estimate_hours=strategy['hold_hours'],
            ))
        return results

    # ------------------------------------------------------------------
    # Feature extraction (from live signal data)
    # ------------------------------------------------------------------
    def extract_features(self, signal: Dict, stock_data: Optional[Dict] = None) -> np.ndarray:
        """
        Extract features from a signal dictionary for prediction.

        This matches the feature order in EXIT_FEATURE_NAMES so the output
        can be passed directly to predict().

        Args:
            signal: Signal dictionary from scanner or signal file.
            stock_data: Optional raw stock data dict with 'daily' DataFrame.

        Returns:
            1-D numpy array of feature values.
        """
        features = np.zeros(len(self.feature_names), dtype=np.float32)

        price = signal.get('price', signal.get('entry_price', 0))
        atr = signal.get('atr', 0)
        direction = signal.get('direction', 'long')

        # rrs_strength
        features[0] = float(signal.get('rrs', 0))

        # atr_pct
        features[1] = (atr / price * 100) if price > 0 else 0

        # volume_ratio
        features[2] = float(signal.get('volume_ratio', 1.0))

        # hour_of_day
        now = datetime.now()
        features[3] = float(now.hour)

        # day_of_week
        features[4] = float(now.weekday())

        # market_regime (simplified: use SPY trend if available)
        regime = signal.get('market_regime', 0)
        if regime == 0:
            # Try to infer from daily_strong / daily_weak
            if signal.get('daily_strong', False):
                regime = 1
            elif signal.get('daily_weak', False):
                regime = -1
        # Convert string regime names to numeric values
        if isinstance(regime, str):
            regime_map = {
                'bull_trending': 1.0,
                'low_volatility': 0.5,
                'high_volatility': -0.5,
                'bear_trending': -1.0,
            }
            regime = regime_map.get(regime, 0.0)
        features[5] = float(regime)

        # EMA distances and other daily-derived features
        daily_df = None
        if stock_data and 'daily' in stock_data:
            daily_df = stock_data['daily']
        elif stock_data and 'daily_data' in stock_data:
            daily_df = stock_data['daily_data']

        if daily_df is not None and len(daily_df) >= 50:
            close = daily_df['close'] if 'close' in daily_df.columns else daily_df.iloc[:, 0]
            current_price = float(close.iloc[-1])

            ema8 = calculate_ema(close, 8)
            ema21 = calculate_ema(close, 21)
            ema50 = calculate_ema(close, 50)
            ema3 = calculate_ema(close, 3)

            # dist_ema8_pct
            if float(ema8.iloc[-1]) != 0:
                features[6] = (current_price - float(ema8.iloc[-1])) / float(ema8.iloc[-1]) * 100
            # dist_ema21_pct
            if float(ema21.iloc[-1]) != 0:
                features[7] = (current_price - float(ema21.iloc[-1])) / float(ema21.iloc[-1]) * 100
            # dist_ema50_pct
            if float(ema50.iloc[-1]) != 0:
                features[8] = (current_price - float(ema50.iloc[-1])) / float(ema50.iloc[-1]) * 100

            # ema_alignment
            e3 = float(ema3.iloc[-1])
            e8 = float(ema8.iloc[-1])
            e21 = float(ema21.iloc[-1])
            if e3 > e8 > e21 or e3 < e8 < e21:
                features[9] = 1.0

            # rsi_14
            features[10] = _calculate_rsi(close, 14)

            # bar_pattern_score (3-bar pattern)
            features[11] = _calculate_bar_pattern_score(daily_df)

            # daily_strength_score
            try:
                strength = check_daily_strength_relaxed(daily_df)
                features[12] = float(strength.get('strength_score', 0))
            except Exception:
                features[12] = 0.0

            # price_momentum_5d
            if len(close) >= 6:
                prev = float(close.iloc[-6])
                if prev > 0:
                    features[13] = (current_price - prev) / prev * 100

            # volume_trend
            if 'volume' in daily_df.columns and len(daily_df) >= 20:
                vol = daily_df['volume']
                avg5 = float(vol.tail(5).mean())
                avg20 = float(vol.tail(20).mean())
                if avg20 > 0:
                    features[14] = avg5 / avg20

            # bb_width_pct
            if len(close) >= 20:
                sma20 = calculate_sma(close, 20)
                std20 = close.rolling(20).std()
                bb_w = float(std20.iloc[-1]) * 4  # upper - lower = 4 std
                if current_price > 0:
                    features[15] = bb_w / current_price * 100

            # price_position_range
            if len(daily_df) >= 20:
                high20 = float(daily_df['high'].tail(20).max())
                low20 = float(daily_df['low'].tail(20).min())
                rng = high20 - low20
                if rng > 0:
                    features[16] = (current_price - low20) / rng
        else:
            # Fallback: fill from signal where possible
            features[10] = float(signal.get('rsi_14', 50))
            features[12] = float(signal.get('daily_strength_score', 0))
            features[14] = float(signal.get('volume_trend', 1.0))

        # direction_is_long
        features[17] = 1.0 if direction == 'long' else 0.0

        return features

    # ------------------------------------------------------------------
    # Save / Load (follows safe_model_loader pattern)
    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None):
        """
        Save model, scaler, and metadata with integrity manifest.

        Args:
            path: Directory to save to. Defaults to models/exit_predictor/.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")

        if path is None:
            path = str(get_models_dir() / 'exit_predictor')

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model + scaler together
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'version': self.version,
            'is_trained': self.is_trained,
            'val_metrics': self.val_metrics.to_dict() if self.val_metrics else None,
            'exit_strategies': EXIT_STRATEGIES,
        }

        model_path = str(save_dir / 'exit_predictor.pkl')
        safe_save_model(model_data, model_path)
        logger.info(f"ExitPredictor saved to {model_path}")

    def load(self, path: Optional[str] = None):
        """
        Load model with integrity verification.

        Args:
            path: Directory to load from. Defaults to models/exit_predictor/.
        """
        if path is None:
            path = str(get_models_dir() / 'exit_predictor')

        load_dir = Path(path)
        model_path = str(load_dir / 'exit_predictor.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Exit predictor model not found: {model_path}")

        model_data = safe_load_model(model_path, allow_unverified=False)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.version = model_data.get('version', '1.0.0')
        self.is_trained = model_data.get('is_trained', True)

        logger.info(f"ExitPredictor loaded from {model_path} (v{self.version})")

    # ------------------------------------------------------------------
    # Metrics / Info
    # ------------------------------------------------------------------
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return a summary of model performance."""
        if self.val_metrics is None:
            return {'is_trained': False}
        result = self.val_metrics.to_dict()
        result['is_trained'] = self.is_trained
        result['version'] = self.version
        result['n_features'] = len(self.feature_names)
        return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI from a price series."""
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        val = float(rsi.iloc[-1])
        if np.isnan(val) or np.isinf(val):
            return 50.0
        return val
    except Exception:
        return 50.0


def _calculate_bar_pattern_score(df: pd.DataFrame) -> float:
    """
    Score the last 3 bars for bullish/bearish pattern.

    Returns:
        Score from -3 (very bearish) to +3 (very bullish).
    """
    try:
        if len(df) < 3:
            return 0.0

        last3 = df.tail(3)
        close = last3['close'].values
        openp = last3['open'].values

        score = 0.0
        for i in range(3):
            if close[i] > openp[i]:
                score += 1.0  # green bar
            else:
                score -= 1.0  # red bar

        return score
    except Exception:
        return 0.0


def _detect_market_regime(spy_daily: pd.DataFrame) -> int:
    """
    Simple market regime detection from SPY daily data.

    Returns:
        1 = bull, 0 = chop, -1 = bear
    """
    try:
        if len(spy_daily) < 50:
            return 0

        close = spy_daily['close']
        ema21 = calculate_ema(close, 21)
        ema50 = calculate_ema(close, 50)

        e21 = float(ema21.iloc[-1])
        e50 = float(ema50.iloc[-1])
        current = float(close.iloc[-1])

        if current > e21 > e50:
            return 1  # bull
        elif current < e21 < e50:
            return -1  # bear
        else:
            return 0  # chop
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Training Data Generator
# ---------------------------------------------------------------------------
def generate_exit_training_data(
    symbols: List[str],
    days: int = 180,
    rrs_threshold: float = 2.0,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Generate training data for the ExitPredictor by scanning historical data
    for RRS signals and computing the actual MFE/MAE after each signal.

    For each historical signal:
      1. Identify entry point (day when RRS crosses threshold)
      2. Track forward 10 trading days to compute MFE and MAE
      3. Classify MFE into target buckets (0, 1, 2)
      4. Extract features at entry time

    Args:
        symbols: List of stock tickers to scan.
        days: Number of calendar days of history to use.
        rrs_threshold: Minimum absolute RRS to consider as a signal.
        atr_period: ATR calculation period.

    Returns:
        DataFrame with feature columns + 'target' + metadata columns.
    """
    if not YF_AVAILABLE:
        raise ImportError("yfinance is required for training data generation")

    rrs_calc = RRSCalculator(atr_period=atr_period)
    all_rows: List[Dict] = []

    # Download SPY data once
    logger.info(f"Downloading SPY data for {days} days...")
    try:
        spy_ticker = yf.Ticker('SPY')
        spy_daily = spy_ticker.history(period=f"{days}d", interval='1d')
        if spy_daily.empty:
            logger.error("Failed to download SPY data")
            return pd.DataFrame()
        spy_daily.columns = [c.lower() for c in spy_daily.columns]
    except Exception as e:
        logger.error(f"Error downloading SPY data: {e}")
        return pd.DataFrame()

    spy_regime = _detect_market_regime(spy_daily)

    for symbol in symbols:
        logger.info(f"Processing {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            daily = ticker.history(period=f"{days}d", interval='1d')
            if daily.empty or len(daily) < 60:
                logger.debug(f"{symbol}: insufficient data ({len(daily)} days)")
                continue

            # Normalize columns
            daily.columns = [c.lower() for c in daily.columns]

            # Calculate ATR series
            atr_series = rrs_calc.calculate_atr(daily)

            # Calculate RRS series
            rrs_df = rrs_calc.calculate_rrs(daily, spy_daily, periods=1)

            # Align lengths
            min_len = min(len(daily), len(rrs_df), len(atr_series))
            daily = daily.iloc[-min_len:]
            rrs_series = rrs_df['rrs'].iloc[-min_len:]
            atr_aligned = atr_series.iloc[-min_len:]

            # Scan for signals (RRS crossing threshold)
            # Need at least 10 forward bars for MFE/MAE
            for i in range(50, min_len - 10):
                rrs_val = float(rrs_series.iloc[i])
                if abs(rrs_val) < rrs_threshold:
                    continue

                # Skip if previous bar also above threshold (avoid double-counting)
                prev_rrs = float(rrs_series.iloc[i - 1])
                if abs(prev_rrs) >= rrs_threshold:
                    continue

                direction = 'long' if rrs_val > 0 else 'short'
                entry_price = float(daily['close'].iloc[i])
                entry_atr = float(atr_aligned.iloc[i])

                if entry_price <= 0 or entry_atr <= 0 or np.isnan(entry_atr):
                    continue

                # Compute MFE and MAE over next 10 bars
                forward = daily.iloc[i + 1: i + 11]
                if len(forward) < 5:
                    continue

                if direction == 'long':
                    mfe = float(forward['high'].max()) - entry_price
                    mae = entry_price - float(forward['low'].min())
                else:
                    mfe = entry_price - float(forward['low'].min())
                    mae = float(forward['high'].max()) - entry_price

                mfe_atr = mfe / entry_atr if entry_atr > 0 else 0
                mae_atr = mae / entry_atr if entry_atr > 0 else 0

                # Classify target
                if mfe_atr < 1.0:
                    target = 0  # quick_scalp
                elif mfe_atr < 2.0:
                    target = 1  # swing
                else:
                    target = 2  # runner

                # Extract features at entry time
                entry_date = daily.index[i]
                lookback = daily.iloc[max(0, i - 50): i + 1]

                close_lb = lookback['close']
                ema3 = calculate_ema(close_lb, 3)
                ema8 = calculate_ema(close_lb, 8)
                ema21 = calculate_ema(close_lb, 21)
                ema50 = calculate_ema(close_lb, min(50, len(close_lb)))

                e3 = float(ema3.iloc[-1])
                e8 = float(ema8.iloc[-1])
                e21 = float(ema21.iloc[-1])
                e50 = float(ema50.iloc[-1])

                dist_ema8 = (entry_price - e8) / e8 * 100 if e8 != 0 else 0
                dist_ema21 = (entry_price - e21) / e21 * 100 if e21 != 0 else 0
                dist_ema50 = (entry_price - e50) / e50 * 100 if e50 != 0 else 0

                ema_aligned = 1.0 if (e3 > e8 > e21) or (e3 < e8 < e21) else 0.0

                rsi = _calculate_rsi(close_lb, 14)
                bar_pattern = _calculate_bar_pattern_score(lookback)

                # Daily strength score
                try:
                    ds = check_daily_strength_relaxed(lookback)
                    daily_score = float(ds.get('strength_score', 0))
                except Exception:
                    daily_score = 0.0

                # Momentum
                if len(close_lb) >= 6:
                    prev5 = float(close_lb.iloc[-6])
                    mom5 = (entry_price - prev5) / prev5 * 100 if prev5 > 0 else 0
                else:
                    mom5 = 0.0

                # Volume trend
                if 'volume' in lookback.columns and len(lookback) >= 20:
                    vol = lookback['volume']
                    avg5 = float(vol.tail(5).mean())
                    avg20 = float(vol.tail(20).mean())
                    vol_trend = avg5 / avg20 if avg20 > 0 else 1.0
                    vol_ratio = float(vol.iloc[-1]) / avg20 if avg20 > 0 else 1.0
                else:
                    vol_trend = 1.0
                    vol_ratio = 1.0

                # Bollinger Band width
                if len(close_lb) >= 20:
                    std20 = float(close_lb.rolling(20).std().iloc[-1])
                    bb_width = std20 * 4 / entry_price * 100 if entry_price > 0 else 0
                else:
                    bb_width = 0.0

                # Price position in range
                if len(lookback) >= 20:
                    h20 = float(lookback['high'].tail(20).max())
                    l20 = float(lookback['low'].tail(20).min())
                    rng = h20 - l20
                    pos_range = (entry_price - l20) / rng if rng > 0 else 0.5
                else:
                    pos_range = 0.5

                # Use entry date for time features
                hour = entry_date.hour if hasattr(entry_date, 'hour') else 10
                dow = entry_date.weekday() if hasattr(entry_date, 'weekday') else 2

                row = {
                    'rrs_strength': rrs_val,
                    'atr_pct': entry_atr / entry_price * 100,
                    'volume_ratio': vol_ratio,
                    'hour_of_day': float(hour),
                    'day_of_week': float(dow),
                    'market_regime': float(spy_regime),
                    'dist_ema8_pct': dist_ema8,
                    'dist_ema21_pct': dist_ema21,
                    'dist_ema50_pct': dist_ema50,
                    'ema_alignment': ema_aligned,
                    'rsi_14': rsi,
                    'bar_pattern_score': bar_pattern,
                    'daily_strength_score': daily_score,
                    'price_momentum_5d': mom5,
                    'volume_trend': vol_trend,
                    'bb_width_pct': bb_width,
                    'price_position_range': pos_range,
                    'direction_is_long': 1.0 if direction == 'long' else 0.0,
                    # Target and metadata
                    'target': target,
                    'mfe_atr': mfe_atr,
                    'mae_atr': mae_atr,
                    'symbol': symbol,
                    'entry_date': str(entry_date),
                    'entry_price': entry_price,
                    'entry_atr': entry_atr,
                    'direction': direction,
                }
                all_rows.append(row)

            # Be polite to yfinance
            time.sleep(0.3)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue

    if not all_rows:
        logger.warning("No training samples generated")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    logger.info(
        f"Generated {len(df)} training samples from {len(symbols)} symbols. "
        f"Class distribution: {df['target'].value_counts().to_dict()}"
    )
    return df
