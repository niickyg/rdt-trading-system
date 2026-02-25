"""
ML-Based Signal Decay Predictor

Predicts how long an RRS signal will remain valid, replacing the static
30-minute TTL with per-signal dynamic TTL. Uses XGBoost regression to
estimate signal lifespan in minutes (10-480 range).

Each prediction includes:
  - estimated_valid_minutes: predicted lifespan of the signal
  - decay_rate: fast (<30 min), medium (30-120 min), slow (>120 min)
  - recommended_ttl: suggested TTL for this specific signal

Addresses the problem that a quick 10-minute RRS scalp and a multi-day
hold currently get the same 30-minute TTL. A model that predicts signal
lifespan improves timing, reduces stale signals, and extends strong ones.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
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
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    check_daily_strength_relaxed, check_daily_weakness_relaxed,
)


# ---------------------------------------------------------------------------
# Feature names (order matters -- must match training and prediction)
# ---------------------------------------------------------------------------
DECAY_FEATURE_NAMES = [
    'rrs_strength',           # RRS value at signal generation
    'rrs_rate_of_change',     # RRS acceleration (how fast RRS is growing/shrinking)
    'atr_pct',                # ATR as % of price (volatile stocks = faster decay)
    'volume_ratio',           # Current volume / 20-day average (high = slower decay)
    'market_regime',          # 1=bull, 0=chop, -1=bear
    'hour_of_day',            # 0-23 (morning signals last longer)
    'day_of_week',            # 0-4 (Mon-Fri; Monday/Friday behave differently)
    'is_earnings_related',    # 1 if near earnings date, 0 otherwise
    'sector_code',            # Numeric sector code (tech decays faster than utilities)
    'dist_vwap_pct',          # Distance from daily VWAP as % of price
    'rsi_14',                 # RSI at signal generation
    'ema_alignment',          # 1 if EMA3>EMA8>EMA21 (or reversed for short), else 0
    'bb_width_pct',           # Bollinger Band width as % of price
    'price_momentum_5d',      # 5-day price momentum %
    'daily_strength_score',   # Daily strength/weakness score (0-5)
    'direction_is_long',      # 1 for long, 0 for short
]


# ---------------------------------------------------------------------------
# Sector mapping (numeric codes for model)
# ---------------------------------------------------------------------------
SECTOR_CODES: Dict[str, int] = {
    'Technology': 0,
    'Communication Services': 1,
    'Consumer Cyclical': 2,
    'Consumer Defensive': 3,
    'Financial Services': 4,
    'Healthcare': 5,
    'Industrials': 6,
    'Energy': 7,
    'Basic Materials': 8,
    'Utilities': 9,
    'Real Estate': 10,
    'Unknown': 11,
}

# Quick symbol-to-sector lookup for the default watchlist
# Avoids slow yfinance info calls during prediction
SYMBOL_SECTOR_MAP: Dict[str, str] = {
    # Mega-cap Tech
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Communication Services',
    'AMZN': 'Consumer Cyclical', 'NVDA': 'Technology', 'META': 'Communication Services',
    'TSLA': 'Consumer Cyclical',
    # Semiconductors
    'AMD': 'Technology', 'AVGO': 'Technology', 'QCOM': 'Technology',
    'INTC': 'Technology', 'MU': 'Technology',
    # Financials
    'JPM': 'Financial Services', 'BAC': 'Financial Services',
    'GS': 'Financial Services', 'MS': 'Financial Services',
    # Consumer
    'WMT': 'Consumer Defensive', 'HD': 'Consumer Cyclical',
    'COST': 'Consumer Defensive', 'MCD': 'Consumer Cyclical',
    'SBUX': 'Consumer Cyclical',
    # Healthcare
    'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'PFE': 'Healthcare',
    'ABBV': 'Healthcare',
    # Energy/Materials
    'XOM': 'Energy', 'CVX': 'Energy', 'FCX': 'Basic Materials',
    'NEM': 'Basic Materials',
    # Consumer Goods
    'PG': 'Consumer Defensive', 'KO': 'Consumer Defensive',
    'PEP': 'Consumer Defensive', 'GIS': 'Consumer Defensive',
}


@dataclass
class DecayPrediction:
    """Result of a signal decay prediction."""
    estimated_valid_minutes: float    # How long the signal will remain valid (10-480)
    decay_rate: str                   # 'fast', 'medium', 'slow'
    recommended_ttl: int              # Suggested TTL in minutes for this signal
    confidence_interval: tuple        # (lower, upper) bound estimate in minutes
    features_used: Dict[str, float]   # Feature values used for this prediction


@dataclass
class DecayModelMetrics:
    """Metrics from model training/validation."""
    mae: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_test: int = 0
    bucket_distribution: Dict[str, int] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            'mae': self.mae,
            'rmse': self.rmse,
            'r2': self.r2,
            'n_train': self.n_train,
            'n_test': self.n_test,
            'feature_importance': self.feature_importance,
            'bucket_distribution': self.bucket_distribution,
            'timestamp': self.timestamp,
        }


class SignalDecayPredictor:
    """
    XGBoost-based signal decay predictor.

    Predicts how long an RRS signal will remain actionable, then
    recommends a per-signal TTL to replace the static 30-minute default.

    Usage:
        predictor = SignalDecayPredictor()

        # Training
        df = generate_decay_training_data(['AAPL', 'MSFT'], days=60)
        metrics = predictor.train(df)

        # Prediction
        features = predictor.extract_features(signal_data)
        prediction = predictor.predict(features)
    """

    # Limits for predicted minutes (clamped to this range)
    MIN_VALID_MINUTES = 10
    MAX_VALID_MINUTES = 480

    # Default TTL when model is not trained (fallback to existing behavior)
    DEFAULT_TTL_MINUTES = 30

    def __init__(self, random_state: int = 42):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is required for SignalDecayPredictor. "
                "Install with: pip install xgboost"
            )
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for SignalDecayPredictor. "
                "Install with: pip install scikit-learn"
            )

        self.random_state = random_state
        self.model: Optional[xgb.XGBRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        self.feature_names = DECAY_FEATURE_NAMES.copy()
        self.val_metrics: Optional[DecayModelMetrics] = None
        self.version = "1.0.0"

        self._rrs_calculator = RRSCalculator()

        logger.info("SignalDecayPredictor initialized")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> DecayModelMetrics:
        """
        Train the decay predictor on historical signal data.

        Args:
            df: DataFrame with feature columns (DECAY_FEATURE_NAMES) and
                a 'target_minutes' column (actual valid duration in minutes).
            test_size: Fraction of data to hold out for validation.

        Returns:
            DecayModelMetrics with validation performance.
        """
        logger.info(f"Training SignalDecayPredictor with {len(df)} samples")

        # Validate columns
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        if 'target_minutes' not in df.columns:
            raise ValueError("DataFrame must contain a 'target_minutes' column")

        X = df[self.feature_names].values.astype(np.float32)
        y = df['target_minutes'].values.astype(np.float32)

        # Clamp target to valid range
        y = np.clip(y, self.MIN_VALID_MINUTES, self.MAX_VALID_MINUTES)

        # Handle NaN / inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=self.DEFAULT_TTL_MINUTES)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size,
            random_state=self.random_state,
        )

        # Train XGBoost regressor
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='reg:squarederror',
            random_state=self.random_state,
            n_jobs=1,  # Single thread to avoid potential hangs
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred = np.clip(y_pred, self.MIN_VALID_MINUTES, self.MAX_VALID_MINUTES)

        mae_val = mean_absolute_error(y_test, y_pred)
        rmse_val = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_val = r2_score(y_test, y_pred)

        # Feature importance
        importances = self.model.feature_importances_
        fi_dict = {
            name: float(imp)
            for name, imp in sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1], reverse=True,
            )
        }

        # Bucket distribution
        buckets = _bucket_minutes(y)
        bucket_dist = {}
        for b in ['short', 'medium', 'long']:
            bucket_dist[b] = int(np.sum(buckets == b))

        self.val_metrics = DecayModelMetrics(
            mae=float(mae_val),
            rmse=rmse_val,
            r2=float(r2_val),
            feature_importance=fi_dict,
            n_train=len(X_train),
            n_test=len(X_test),
            bucket_distribution=bucket_dist,
            timestamp=datetime.now().isoformat(),
        )

        logger.info(
            f"SignalDecayPredictor training complete: MAE={mae_val:.1f} min, "
            f"RMSE={rmse_val:.1f} min, R2={r2_val:.3f}"
        )
        logger.info(f"Bucket distribution: {bucket_dist}")

        return self.val_metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, features: np.ndarray) -> DecayPrediction:
        """
        Predict signal decay for a single signal.

        Args:
            features: 1-D numpy array of shape (n_features,) matching
                      DECAY_FEATURE_NAMES ordering.

        Returns:
            DecayPrediction with estimated valid minutes and recommended TTL.
        """
        if not self.is_trained or self.model is None or self.scaler is None:
            raise ValueError(
                "SignalDecayPredictor not trained. Call train() first or load() a saved model."
            )

        features = np.asarray(features, dtype=np.float32).reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_scaled = self.scaler.transform(features)

        predicted_minutes = float(self.model.predict(features_scaled)[0])
        predicted_minutes = np.clip(
            predicted_minutes, self.MIN_VALID_MINUTES, self.MAX_VALID_MINUTES
        )

        # Determine decay rate bucket
        if predicted_minutes < 30:
            decay_rate = 'fast'
        elif predicted_minutes <= 120:
            decay_rate = 'medium'
        else:
            decay_rate = 'slow'

        # Recommended TTL: round to nearest 5 minutes, add a small buffer
        buffer_factor = 1.15  # 15% buffer beyond predicted expiry
        recommended_ttl = int(round(predicted_minutes * buffer_factor / 5) * 5)
        recommended_ttl = max(self.MIN_VALID_MINUTES, min(recommended_ttl, self.MAX_VALID_MINUTES))

        # Simple confidence interval estimate (+/- MAE from validation)
        mae = self.val_metrics.mae if self.val_metrics else 15.0
        ci_lower = max(self.MIN_VALID_MINUTES, predicted_minutes - mae)
        ci_upper = min(self.MAX_VALID_MINUTES, predicted_minutes + mae)

        # Build feature dict for transparency
        features_flat = features.flatten()
        features_dict = {
            name: float(features_flat[i])
            for i, name in enumerate(self.feature_names)
        }

        return DecayPrediction(
            estimated_valid_minutes=round(predicted_minutes, 1),
            decay_rate=decay_rate,
            recommended_ttl=recommended_ttl,
            confidence_interval=(round(ci_lower, 1), round(ci_upper, 1)),
            features_used=features_dict,
        )

    def predict_from_signal(self, signal: Dict, stock_data: Optional[Dict] = None) -> DecayPrediction:
        """
        Convenience method: extract features from a signal dict and predict.

        Args:
            signal: Signal dictionary from scanner or signal file.
            stock_data: Optional raw stock data dict with 'daily' DataFrame.

        Returns:
            DecayPrediction with estimated TTL.
        """
        features = self.extract_features(signal, stock_data)
        return self.predict(features)

    # ------------------------------------------------------------------
    # Feature extraction (from live signal data)
    # ------------------------------------------------------------------
    def extract_features(self, signal: Dict, stock_data: Optional[Dict] = None) -> np.ndarray:
        """
        Extract features from a signal dictionary for prediction.

        Args:
            signal: Signal dictionary from scanner or signal file.
            stock_data: Optional raw stock data dict with 'daily' DataFrame.

        Returns:
            1-D numpy array of feature values matching DECAY_FEATURE_NAMES.
        """
        features = np.zeros(len(self.feature_names), dtype=np.float32)

        price = signal.get('price', signal.get('entry_price', 0))
        atr = signal.get('atr', 0)
        direction = signal.get('direction', 'long')
        symbol = signal.get('symbol', '')

        # 0: rrs_strength
        features[0] = float(signal.get('rrs', 0))

        # 1: rrs_rate_of_change (acceleration)
        # If the signal carries previous RRS, compute rate of change
        rrs_prev = signal.get('rrs_previous', None)
        rrs_current = float(signal.get('rrs', 0))
        if rrs_prev is not None:
            features[1] = rrs_current - float(rrs_prev)
        else:
            # Estimate from RRS magnitude (stronger signals tend to accelerate)
            features[1] = 0.0

        # 2: atr_pct
        features[2] = (atr / price * 100) if price > 0 else 0

        # 3: volume_ratio
        features[3] = float(signal.get('volume_ratio', 1.0))

        # 4: market_regime
        regime = signal.get('market_regime', 0)
        if regime == 0:
            if signal.get('daily_strong', False):
                regime = 1
            elif signal.get('daily_weak', False):
                regime = -1
        features[4] = float(regime)

        # 5: hour_of_day
        now = datetime.now()
        features[5] = float(now.hour)

        # 6: day_of_week
        features[6] = float(now.weekday())

        # 7: is_earnings_related
        features[7] = float(signal.get('is_earnings_related', 0))

        # 8: sector_code
        sector = SYMBOL_SECTOR_MAP.get(symbol, 'Unknown')
        features[8] = float(SECTOR_CODES.get(sector, SECTOR_CODES['Unknown']))

        # 9: dist_vwap_pct (distance from VWAP)
        vwap_dist = signal.get('dist_vwap_pct', None)
        if vwap_dist is not None:
            features[9] = float(vwap_dist)
        elif stock_data is not None:
            features[9] = _calculate_vwap_distance(stock_data, price)

        # Features that need daily data
        daily_df = None
        if stock_data and 'daily' in stock_data:
            daily_df = stock_data['daily']
        elif stock_data and 'daily_data' in stock_data:
            daily_df = stock_data['daily_data']

        if daily_df is not None and len(daily_df) >= 50:
            close = daily_df['close'] if 'close' in daily_df.columns else daily_df.iloc[:, 0]
            current_price = float(close.iloc[-1])

            # 10: rsi_14
            features[10] = _calculate_rsi(close, 14)

            # 11: ema_alignment
            ema3 = calculate_ema(close, 3)
            ema8 = calculate_ema(close, 8)
            ema21 = calculate_ema(close, 21)
            e3 = float(ema3.iloc[-1])
            e8 = float(ema8.iloc[-1])
            e21 = float(ema21.iloc[-1])
            if e3 > e8 > e21 or e3 < e8 < e21:
                features[11] = 1.0

            # 12: bb_width_pct
            if len(close) >= 20:
                std20 = float(close.rolling(20).std().iloc[-1])
                if current_price > 0:
                    features[12] = std20 * 4 / current_price * 100

            # 13: price_momentum_5d
            if len(close) >= 6:
                prev = float(close.iloc[-6])
                if prev > 0:
                    features[13] = (current_price - prev) / prev * 100

            # 14: daily_strength_score
            try:
                strength = check_daily_strength_relaxed(daily_df)
                features[14] = float(strength.get('strength_score', 0))
            except Exception:
                features[14] = 0.0
        else:
            # Fallback from signal
            features[10] = float(signal.get('rsi_14', 50))
            features[14] = float(signal.get('daily_strength_score', 0))

        # 15: direction_is_long
        features[15] = 1.0 if direction == 'long' else 0.0

        return features

    # ------------------------------------------------------------------
    # Save / Load (follows safe_model_loader pattern)
    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None):
        """
        Save model, scaler, and metadata with integrity manifest.

        Args:
            path: Directory to save to. Defaults to models/signal_decay/.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")

        if path is None:
            path = str(get_models_dir() / 'signal_decay')

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'version': self.version,
            'is_trained': self.is_trained,
            'val_metrics': self.val_metrics.to_dict() if self.val_metrics else None,
        }

        model_path = str(save_dir / 'signal_decay.pkl')
        safe_save_model(model_data, model_path)
        logger.info(f"SignalDecayPredictor saved to {model_path}")

    def load(self, path: Optional[str] = None):
        """
        Load model with integrity verification.

        Args:
            path: Directory to load from. Defaults to models/signal_decay/.
        """
        if path is None:
            path = str(get_models_dir() / 'signal_decay')

        load_dir = Path(path)
        model_path = str(load_dir / 'signal_decay.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Signal decay model not found: {model_path}")

        model_data = safe_load_model(model_path, allow_unverified=False)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.version = model_data.get('version', '1.0.0')
        self.is_trained = model_data.get('is_trained', True)

        metrics_dict = model_data.get('val_metrics')
        if metrics_dict:
            self.val_metrics = DecayModelMetrics(**{
                k: v for k, v in metrics_dict.items()
                if k in DecayModelMetrics.__dataclass_fields__
            })

        logger.info(f"SignalDecayPredictor loaded from {model_path} (v{self.version})")

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


def _calculate_vwap_distance(stock_data: Dict, current_price: float) -> float:
    """
    Calculate distance from VWAP as % of price.

    Uses intraday 5m data if available, otherwise returns 0.
    """
    try:
        df_5m = stock_data.get('5m')
        if df_5m is None or df_5m.empty:
            return 0.0

        # Ensure lowercase columns
        cols = [c.lower() for c in df_5m.columns]
        if 'close' not in cols or 'volume' not in cols:
            return 0.0

        close = df_5m['close'] if 'close' in df_5m.columns else df_5m[df_5m.columns[0]]
        volume = df_5m['volume'] if 'volume' in df_5m.columns else df_5m[df_5m.columns[-1]]
        high = df_5m['high'] if 'high' in df_5m.columns else close
        low = df_5m['low'] if 'low' in df_5m.columns else close

        # VWAP = cumsum(typical_price * volume) / cumsum(volume)
        typical_price = (high + low + close) / 3
        cum_vol = volume.cumsum()
        cum_tp_vol = (typical_price * volume).cumsum()

        # Avoid division by zero
        if float(cum_vol.iloc[-1]) == 0:
            return 0.0

        vwap = float(cum_tp_vol.iloc[-1] / cum_vol.iloc[-1])

        if vwap > 0 and current_price > 0:
            return (current_price - vwap) / vwap * 100
        return 0.0
    except Exception:
        return 0.0


def _bucket_minutes(minutes_array: np.ndarray) -> np.ndarray:
    """Bucket minutes into short/medium/long categories."""
    buckets = np.empty(len(minutes_array), dtype=object)
    buckets[minutes_array < 30] = 'short'
    buckets[(minutes_array >= 30) & (minutes_array <= 120)] = 'medium'
    buckets[minutes_array > 120] = 'long'
    return buckets


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


def _is_near_earnings(symbol: str, check_date: datetime, window_days: int = 5) -> bool:
    """
    Check if a date is near an earnings announcement for a symbol.

    Uses yfinance calendar data. Returns False on any error (graceful degradation).
    """
    if not YF_AVAILABLE:
        return False
    try:
        ticker = yf.Ticker(symbol)
        calendar = ticker.calendar
        if calendar is None or calendar.empty:
            return False
        # calendar may have 'Earnings Date' row(s)
        if hasattr(calendar, 'loc') and 'Earnings Date' in calendar.index:
            earnings_dates = calendar.loc['Earnings Date']
            for ed in earnings_dates:
                if isinstance(ed, (datetime, pd.Timestamp)):
                    diff = abs((ed - check_date).days)
                    if diff <= window_days:
                        return True
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Training Data Generator
# ---------------------------------------------------------------------------
def generate_decay_training_data(
    symbols: List[str],
    days: int = 60,
    rrs_threshold: float = 2.0,
    atr_period: int = 14,
    rrs_check_interval_bars: int = 1,
) -> pd.DataFrame:
    """
    Generate training data for the SignalDecayPredictor.

    For each historical RRS signal:
      1. Identify entry point (when RRS crosses threshold on intraday data)
      2. Track RRS value every bar (5-min) after signal
      3. Signal "expires" when RRS drops below threshold (2.0 for strong)
      4. Record actual valid duration in minutes
      5. Extract features at signal generation time

    Since intraday data is limited (~60 days on yfinance), we use daily
    data to simulate with bar-count as a proxy for time, then scale to
    approximate intraday durations.

    Args:
        symbols: List of stock tickers to scan.
        days: Number of calendar days of history to use.
        rrs_threshold: Minimum absolute RRS to consider as a signal.
        atr_period: ATR calculation period.
        rrs_check_interval_bars: How many bars between RRS checks.

    Returns:
        DataFrame with feature columns + 'target_minutes' + metadata.
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
        logger.info(f"Processing {symbol} for decay training data...")
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

            # Determine sector
            sector = SYMBOL_SECTOR_MAP.get(symbol, 'Unknown')
            sector_code = float(SECTOR_CODES.get(sector, SECTOR_CODES['Unknown']))

            # Check earnings proximity (once for the symbol)
            mid_date = daily.index[len(daily) // 2]
            if hasattr(mid_date, 'to_pydatetime'):
                mid_date = mid_date.to_pydatetime()
            is_earnings = _is_near_earnings(symbol, mid_date)

            # Scan for signals (RRS crossing threshold)
            # Need forward bars to measure decay
            for i in range(50, min_len - 20):
                rrs_val = float(rrs_series.iloc[i])
                if abs(rrs_val) < rrs_threshold:
                    continue

                # Skip if previous bar also above threshold (crossing detection)
                prev_rrs = float(rrs_series.iloc[i - 1])
                if abs(prev_rrs) >= rrs_threshold:
                    continue

                direction = 'long' if rrs_val > 0 else 'short'
                entry_price = float(daily['close'].iloc[i])
                entry_atr = float(atr_aligned.iloc[i])

                if entry_price <= 0 or entry_atr <= 0 or np.isnan(entry_atr):
                    continue

                # Track RRS forward to find when signal expires
                # Signal expires when |RRS| drops below threshold
                valid_bars = 0
                for j in range(i + 1, min(i + 20, min_len)):
                    forward_rrs = float(rrs_series.iloc[j])
                    if direction == 'long' and forward_rrs < rrs_threshold:
                        break
                    elif direction == 'short' and forward_rrs > -rrs_threshold:
                        break
                    valid_bars += 1

                # Convert daily bars to approximate minutes
                # 1 daily bar ~ 390 minutes of market time
                # But most signals don't last full days, so we scale:
                # 1 bar = ~30 min for a rough intraday proxy
                # This is a simplification; with intraday data, this would be exact.
                minutes_per_bar = 390.0 / 13.0  # ~30 min (13 half-hour periods per day)
                target_minutes = valid_bars * minutes_per_bar
                target_minutes = max(10.0, min(target_minutes, 480.0))

                # RRS rate of change at signal time
                if i >= 2:
                    rrs_prev = float(rrs_series.iloc[i - 1])
                    rrs_roc = rrs_val - rrs_prev
                else:
                    rrs_roc = 0.0

                # Extract features at entry time
                lookback = daily.iloc[max(0, i - 50): i + 1]
                close_lb = lookback['close']

                ema3 = calculate_ema(close_lb, 3)
                ema8 = calculate_ema(close_lb, 8)
                ema21 = calculate_ema(close_lb, 21)

                e3 = float(ema3.iloc[-1])
                e8 = float(ema8.iloc[-1])
                e21 = float(ema21.iloc[-1])
                ema_aligned = 1.0 if (e3 > e8 > e21) or (e3 < e8 < e21) else 0.0

                rsi = _calculate_rsi(close_lb, 14)

                # Bollinger Band width
                if len(close_lb) >= 20:
                    std20 = float(close_lb.rolling(20).std().iloc[-1])
                    bb_width = std20 * 4 / entry_price * 100 if entry_price > 0 else 0
                else:
                    bb_width = 0.0

                # Price momentum 5d
                if len(close_lb) >= 6:
                    prev5 = float(close_lb.iloc[-6])
                    mom5 = (entry_price - prev5) / prev5 * 100 if prev5 > 0 else 0
                else:
                    mom5 = 0.0

                # Daily strength score
                try:
                    ds = check_daily_strength_relaxed(lookback)
                    daily_score = float(ds.get('strength_score', 0))
                except Exception:
                    daily_score = 0.0

                # Volume ratio
                if 'volume' in lookback.columns and len(lookback) >= 20:
                    vol = lookback['volume']
                    avg20 = float(vol.tail(20).mean())
                    vol_ratio = float(vol.iloc[-1]) / avg20 if avg20 > 0 else 1.0
                else:
                    vol_ratio = 1.0

                # Time features from entry date
                entry_date = daily.index[i]
                hour = entry_date.hour if hasattr(entry_date, 'hour') else 10
                dow = entry_date.weekday() if hasattr(entry_date, 'weekday') else 2

                row = {
                    # Features (must match DECAY_FEATURE_NAMES order)
                    'rrs_strength': rrs_val,
                    'rrs_rate_of_change': rrs_roc,
                    'atr_pct': entry_atr / entry_price * 100,
                    'volume_ratio': vol_ratio,
                    'market_regime': float(spy_regime),
                    'hour_of_day': float(hour),
                    'day_of_week': float(dow),
                    'is_earnings_related': 1.0 if is_earnings else 0.0,
                    'sector_code': sector_code,
                    'dist_vwap_pct': 0.0,  # VWAP not available from daily data
                    'rsi_14': rsi,
                    'ema_alignment': ema_aligned,
                    'bb_width_pct': bb_width,
                    'price_momentum_5d': mom5,
                    'daily_strength_score': daily_score,
                    'direction_is_long': 1.0 if direction == 'long' else 0.0,
                    # Target
                    'target_minutes': target_minutes,
                    # Metadata (not used as features)
                    'symbol': symbol,
                    'entry_date': str(entry_date),
                    'entry_price': entry_price,
                    'entry_atr': entry_atr,
                    'direction': direction,
                    'valid_bars': valid_bars,
                    'rrs_at_signal': rrs_val,
                }
                all_rows.append(row)

            # Be polite to yfinance
            time.sleep(0.3)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue

    if not all_rows:
        logger.warning("No training samples generated for decay predictor")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Log bucket distribution
    buckets = _bucket_minutes(df['target_minutes'].values)
    bucket_counts = {}
    for b in ['short', 'medium', 'long']:
        bucket_counts[b] = int(np.sum(buckets == b))

    logger.info(
        f"Generated {len(df)} decay training samples from {len(symbols)} symbols. "
        f"Bucket distribution: {bucket_counts}"
    )
    return df
