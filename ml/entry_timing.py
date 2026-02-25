"""
ML-Based Entry Timing Model

Predicts the optimal entry timing after a signal fires. The scanner finds
WHAT to trade, but this model predicts WHEN to enter:

  - 'enter_now':         Price is likely to continue immediately -- market order.
  - 'wait_for_pullback': Price is extended -- wait for a pullback via limit order.
  - 'skip':              Price is likely to reverse > 1% -- skip the trade.

Features are extracted from 5-minute bars around the signal time.  The model
is an XGBoost 3-class classifier trained on historical 5m data.

Usage:
    model = EntryTimingModel()

    # Training
    df = generate_entry_timing_data(['AAPL', 'MSFT', 'GOOGL'], days=60)
    metrics = model.train(df)
    model.save()

    # Prediction (live)
    model.load()
    features = model.extract_features(bars_5m, signal)
    prediction = model.predict(features)
    # prediction.entry_action in ('enter_now', 'wait_for_pullback', 'skip')
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


# ---------------------------------------------------------------------------
# Feature names (order matters -- must match training and prediction)
# ---------------------------------------------------------------------------
ENTRY_TIMING_FEATURE_NAMES = [
    'price_position_in_range',    # Current price position in last 15-bar range (0=low, 1=high)
    'last_3_bar_pattern',         # All green=1, mixed=0, all red=-1
    'volume_ratio_last_bar',      # Volume of last bar vs 15-bar average
    'rsi_14_5m',                  # RSI(14) on 5m bars
    'vwap_distance_pct',          # Distance from approximated VWAP (%)
    'time_since_open_minutes',    # Minutes since market open (9:30 ET)
    'bar_size_vs_avg',            # Current bar range / avg 15-bar range (extension indicator)
    'round_number_proximity',     # Whether price just broke above/below a round number
    'momentum_5bar',              # 5-bar momentum (% change over last 5 bars)
    'direction_is_long',          # 1 for long signals, 0 for short
]

# Target class mapping
ENTRY_ACTION_MAP = {
    0: 'enter_now',
    1: 'wait_for_pullback',
    2: 'skip',
}

ENTRY_ACTION_REVERSE = {v: k for k, v in ENTRY_ACTION_MAP.items()}


@dataclass
class EntryTimingPrediction:
    """Result of an entry timing prediction."""
    entry_action: str               # 'enter_now', 'wait_for_pullback', 'skip'
    expected_pullback_pct: float    # Expected pullback % to wait for (0.2-1.5 typical)
    confidence: float               # Probability of the predicted class (0-1)
    class_probabilities: Dict[str, float]  # Probabilities for all 3 classes
    action_class: int               # Raw class (0, 1, 2)


@dataclass
class EntryTimingMetrics:
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
# Pullback estimation table (maps confidence to expected pullback %)
# Higher confidence in wait_for_pullback -> larger pullback expected.
# ---------------------------------------------------------------------------
def _estimate_pullback_pct(
    action_class: int,
    wait_prob: float,
    bar_size_vs_avg: float,
    price_position: float,
) -> float:
    """
    Estimate the pullback percentage to wait for.

    Combines the model's wait_for_pullback probability with technical context
    to produce a limit-order offset.  Returns 0.0 for enter_now / skip.
    """
    if action_class != 1:  # Only meaningful for wait_for_pullback
        return 0.0

    # Base pullback scales with how extended the move is
    base = 0.2 + (price_position * 0.5)  # 0.2% at range-low, 0.7% at range-high

    # Extension premium: if the bar is much larger than average, expect more pullback
    extension_factor = min(bar_size_vs_avg, 3.0) / 2.0  # 0.5 - 1.5x
    base *= extension_factor

    # Confidence scaling: higher confidence -> wait for more
    base *= (0.5 + wait_prob)  # 0.5x .. 1.5x

    # Clamp to reasonable range
    return max(0.2, min(base, 1.5))


class EntryTimingModel:
    """
    XGBoost-based entry timing predictor.

    Predicts whether to enter immediately, wait for a pullback, or skip
    after a trading signal fires.

    Usage:
        model = EntryTimingModel()

        # Training
        df = generate_entry_timing_data(['AAPL', 'MSFT'], days=60)
        metrics = model.train(df)
        model.save()

        # Prediction
        features = model.extract_features(bars_5m, signal)
        prediction = model.predict(features)
    """

    def __init__(self, random_state: int = 42):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is required for EntryTimingModel. "
                "Install with: pip install xgboost"
            )
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for EntryTimingModel. "
                "Install with: pip install scikit-learn"
            )

        self.random_state = random_state
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        self.feature_names = ENTRY_TIMING_FEATURE_NAMES.copy()
        self.val_metrics: Optional[EntryTimingMetrics] = None
        self.version = "1.0.0"

        logger.info("EntryTimingModel initialized")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> EntryTimingMetrics:
        """
        Train the entry timing model on historical data.

        Args:
            df: DataFrame with feature columns (ENTRY_TIMING_FEATURE_NAMES)
                and a 'target' column (0=enter_now, 1=wait_for_pullback, 2=skip).
            test_size: Fraction of data to hold out for validation.

        Returns:
            EntryTimingMetrics with validation performance.
        """
        logger.info(f"Training EntryTimingModel with {len(df)} samples")

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
        for cls, cnt in zip(unique, counts):
            w = total / (len(unique) * cnt)
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
            n_jobs=1,
        )

        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)

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

        cls_dist = {int(c): int(cnt) for c, cnt in zip(*np.unique(y, return_counts=True))}

        self.val_metrics = EntryTimingMetrics(
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
            f"EntryTimingModel training complete: accuracy={acc:.3f}, "
            f"F1(macro)={f1:.3f}, classes={cls_dist}"
        )
        logger.info(f"\n{report}")

        return self.val_metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, features: np.ndarray) -> EntryTimingPrediction:
        """
        Predict entry timing for a single signal.

        Args:
            features: 1-D numpy array of shape (n_features,) matching
                      ENTRY_TIMING_FEATURE_NAMES ordering.

        Returns:
            EntryTimingPrediction with recommended action.
        """
        if not self.is_trained or self.model is None or self.scaler is None:
            raise ValueError(
                "EntryTimingModel not trained. Call train() first or load() a saved model."
            )

        features = np.asarray(features, dtype=np.float32).reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_scaled = self.scaler.transform(features)

        proba = self.model.predict_proba(features_scaled)[0]
        pred_class = int(np.argmax(proba))

        # Estimate pullback percentage using feature context
        price_position = float(features[0, 0])    # price_position_in_range
        bar_size_vs_avg = float(features[0, 6])    # bar_size_vs_avg

        pullback_pct = _estimate_pullback_pct(
            action_class=pred_class,
            wait_prob=float(proba[1]),
            bar_size_vs_avg=bar_size_vs_avg,
            price_position=price_position,
        )

        return EntryTimingPrediction(
            entry_action=ENTRY_ACTION_MAP[pred_class],
            expected_pullback_pct=pullback_pct,
            confidence=float(proba[pred_class]),
            class_probabilities={
                'enter_now': float(proba[0]),
                'wait_for_pullback': float(proba[1]),
                'skip': float(proba[2]),
            },
            action_class=pred_class,
        )

    # ------------------------------------------------------------------
    # Feature extraction (from live 5-minute bars + signal)
    # ------------------------------------------------------------------
    def extract_features(
        self,
        bars_5m: pd.DataFrame,
        signal: Dict,
    ) -> np.ndarray:
        """
        Extract features from recent 5-minute bars and a signal dictionary.

        The DataFrame ``bars_5m`` should have at least 15 rows of 5-minute
        OHLCV data (columns: open, high, low, close, volume) sorted by time
        ascending.  Column names are normalised to lowercase internally.

        Args:
            bars_5m: DataFrame of 5-minute bars (>= 15 bars preferred).
            signal: Signal dictionary with at least 'direction'.

        Returns:
            1-D numpy array matching ENTRY_TIMING_FEATURE_NAMES.
        """
        features = np.zeros(len(self.feature_names), dtype=np.float32)

        # Normalise column names
        bars = bars_5m.copy()
        bars.columns = [c.lower() for c in bars.columns]

        # Ensure we have enough data
        n = len(bars)
        lookback = min(n, 15)
        recent = bars.tail(lookback)

        if lookback < 3:
            logger.warning("extract_features: fewer than 3 bars, returning zeros")
            return features

        close = recent['close']
        high = recent['high']
        low = recent['low']
        volume = recent['volume'] if 'volume' in recent.columns else pd.Series([0] * lookback)
        openp = recent['open']

        current_close = float(close.iloc[-1])

        # 0: price_position_in_range
        range_high = float(high.max())
        range_low = float(low.min())
        rng = range_high - range_low
        if rng > 0:
            features[0] = (current_close - range_low) / rng
        else:
            features[0] = 0.5

        # 1: last_3_bar_pattern (all green=1, mixed=0, all red=-1)
        last3_close = close.tail(3).values
        last3_open = openp.tail(3).values
        greens = sum(1 for c, o in zip(last3_close, last3_open) if c > o)
        reds = 3 - greens
        if greens == 3:
            features[1] = 1.0
        elif reds == 3:
            features[1] = -1.0
        else:
            features[1] = 0.0

        # 2: volume_ratio_last_bar (last bar vol / avg 15-bar vol)
        avg_vol = float(volume.mean())
        if avg_vol > 0:
            features[2] = float(volume.iloc[-1]) / avg_vol
        else:
            features[2] = 1.0

        # 3: rsi_14_5m
        features[3] = _calculate_rsi_from_series(close, 14)

        # 4: vwap_distance_pct (approximate VWAP from available bars)
        typical_price = (high + low + close) / 3.0
        cumvol = volume.cumsum()
        cum_tp_vol = (typical_price * volume).cumsum()
        last_cumvol = float(cumvol.iloc[-1])
        if last_cumvol > 0:
            vwap = float(cum_tp_vol.iloc[-1]) / last_cumvol
            if vwap > 0:
                features[4] = (current_close - vwap) / vwap * 100
        # else stays 0

        # 5: time_since_open_minutes
        # Try to derive from the bar timestamps, otherwise use current time
        try:
            last_ts = bars.index[-1]
            if hasattr(last_ts, 'hour'):
                market_open_minutes = 9 * 60 + 30
                bar_minutes = last_ts.hour * 60 + last_ts.minute
                features[5] = max(0.0, float(bar_minutes - market_open_minutes))
            else:
                now = datetime.now()
                features[5] = max(0.0, float((now.hour * 60 + now.minute) - (9 * 60 + 30)))
        except Exception:
            now = datetime.now()
            features[5] = max(0.0, float((now.hour * 60 + now.minute) - (9 * 60 + 30)))

        # 6: bar_size_vs_avg (current bar range / average bar range)
        bar_ranges = (high - low).values
        avg_range = float(np.mean(bar_ranges))
        current_range = float(bar_ranges[-1])
        if avg_range > 0:
            features[6] = current_range / avg_range
        else:
            features[6] = 1.0

        # 7: round_number_proximity
        # Check if price just crossed a round number (integer dollar for <$50,
        # $5 multiples for $50-$200, $10 multiples for >$200)
        if current_close < 50:
            round_level = round(current_close)
        elif current_close < 200:
            round_level = round(current_close / 5) * 5
        else:
            round_level = round(current_close / 10) * 10

        if lookback >= 2:
            prev_close = float(close.iloc[-2])
            crossed = (prev_close < round_level <= current_close) or \
                      (prev_close > round_level >= current_close)
            near = abs(current_close - round_level) / max(current_close, 1) < 0.003
            features[7] = 1.0 if crossed else (0.5 if near else 0.0)
        else:
            features[7] = 0.0

        # 8: momentum_5bar (% change over last 5 bars)
        bars_for_mom = min(5, lookback)
        if bars_for_mom >= 2:
            price_ago = float(close.iloc[-bars_for_mom])
            if price_ago > 0:
                features[8] = (current_close - price_ago) / price_ago * 100

        # 9: direction_is_long
        direction = signal.get('direction', 'long')
        features[9] = 1.0 if direction == 'long' else 0.0

        return features

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None):
        """Save model, scaler, and metadata with integrity manifest."""
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")

        if path is None:
            path = str(get_models_dir() / 'entry_timing')

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

        model_path = str(save_dir / 'entry_timing.pkl')
        safe_save_model(model_data, model_path)
        logger.info(f"EntryTimingModel saved to {model_path}")

    def load(self, path: Optional[str] = None):
        """Load model with integrity verification."""
        if path is None:
            path = str(get_models_dir() / 'entry_timing')

        load_dir = Path(path)
        model_path = str(load_dir / 'entry_timing.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Entry timing model not found: {model_path}")

        model_data = safe_load_model(model_path, allow_unverified=False)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.version = model_data.get('version', '1.0.0')
        self.is_trained = model_data.get('is_trained', True)

        logger.info(f"EntryTimingModel loaded from {model_path} (v{self.version})")

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
def _calculate_rsi_from_series(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI from a price series."""
    try:
        if len(prices) < period + 1:
            return 50.0
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


# ---------------------------------------------------------------------------
# Training Data Generator
# ---------------------------------------------------------------------------
def generate_entry_timing_data(
    symbols: List[str],
    days: int = 60,
    rrs_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Generate training data for EntryTimingModel using historical 5-minute bars.

    For each symbol:
      1. Download daily data to identify RRS signal days (RRS crosses threshold).
      2. For each signal day, download 5-minute bars.
      3. At the signal bar (approximate via daily direction), extract features
         from the preceding 15 bars.
      4. Look forward 30 minutes (6 bars of 5m) to compute the target label:
         - If price pulls back > 0.3% then recovers -> 'wait_for_pullback' (1)
         - If price immediately continues in signal direction -> 'enter_now' (0)
         - If price reverses > 1% -> 'skip' (2)

    Args:
        symbols: List of stock tickers.
        days: Calendar days of history.
        rrs_threshold: Minimum absolute RRS for a signal.

    Returns:
        DataFrame with feature columns + 'target' + metadata.
    """
    if not YF_AVAILABLE:
        raise ImportError("yfinance is required for training data generation")

    from shared.indicators.rrs import RRSCalculator
    rrs_calc = RRSCalculator()
    all_rows: List[Dict] = []

    # Download SPY daily data
    logger.info(f"Downloading SPY daily data for {days} days...")
    try:
        spy_ticker = yf.Ticker('SPY')
        spy_daily = spy_ticker.history(period=f"{days}d", interval='1d')
        if spy_daily.empty:
            logger.error("Failed to download SPY daily data")
            return pd.DataFrame()
        spy_daily.columns = [c.lower() for c in spy_daily.columns]
    except Exception as e:
        logger.error(f"Error downloading SPY data: {e}")
        return pd.DataFrame()

    for symbol in symbols:
        logger.info(f"Processing {symbol}...")
        try:
            # --- Daily data for RRS signal detection ---
            ticker = yf.Ticker(symbol)
            daily = ticker.history(period=f"{days}d", interval='1d')
            if daily.empty or len(daily) < 30:
                logger.debug(f"{symbol}: insufficient daily data ({len(daily)} days)")
                continue
            daily.columns = [c.lower() for c in daily.columns]

            # Calculate RRS
            rrs_df = rrs_calc.calculate_rrs(daily, spy_daily, periods=1)
            min_len = min(len(daily), len(rrs_df))
            daily = daily.iloc[-min_len:]
            rrs_series = rrs_df['rrs'].iloc[-min_len:]

            # Find signal days (RRS crosses threshold for the first time)
            signal_dates = []
            for i in range(1, min_len - 1):
                rrs_val = float(rrs_series.iloc[i])
                prev_rrs = float(rrs_series.iloc[i - 1])
                if abs(rrs_val) >= rrs_threshold and abs(prev_rrs) < rrs_threshold:
                    direction = 'long' if rrs_val > 0 else 'short'
                    signal_dates.append((daily.index[i], direction, rrs_val))

            if not signal_dates:
                logger.debug(f"{symbol}: no RRS signals found")
                continue

            # --- Download 5-minute data for signal days ---
            # yfinance allows max 60 days of 5m data
            start_date = daily.index[0]
            end_date = daily.index[-1] + timedelta(days=1)

            try:
                bars_5m_all = ticker.history(
                    start=start_date, end=end_date,
                    interval='5m'
                )
                if bars_5m_all.empty:
                    logger.debug(f"{symbol}: no 5m data available")
                    continue
                bars_5m_all.columns = [c.lower() for c in bars_5m_all.columns]
            except Exception as e:
                logger.debug(f"{symbol}: failed to download 5m data: {e}")
                continue

            # Process each signal day
            for signal_date, direction, rrs_val in signal_dates:
                try:
                    # Get 5m bars for this day
                    day_str = signal_date.strftime('%Y-%m-%d')
                    day_mask = bars_5m_all.index.strftime('%Y-%m-%d') == day_str
                    day_bars = bars_5m_all[day_mask]

                    if len(day_bars) < 25:
                        # Need at least 15 lookback + 6 forward + a few buffer
                        continue

                    # Find a reasonable "signal bar" -- use bars around 10:00-11:00
                    # (typical signal time after open strength is confirmed)
                    signal_mask = (day_bars.index.hour >= 10) & (day_bars.index.hour <= 11)
                    signal_candidates = day_bars[signal_mask]
                    if len(signal_candidates) < 1:
                        signal_candidates = day_bars.iloc[15:20]  # fallback
                    if len(signal_candidates) < 1:
                        continue

                    # For each candidate bar, generate a training sample
                    for bar_idx_in_day in range(len(day_bars)):
                        bar_ts = day_bars.index[bar_idx_in_day]
                        # Only use bars where we have 15 lookback and 6 forward
                        if bar_idx_in_day < 15 or bar_idx_in_day + 6 >= len(day_bars):
                            continue
                        # Limit to market hours 10:00-15:00
                        if bar_ts.hour < 10 or bar_ts.hour >= 15:
                            continue

                        lookback = day_bars.iloc[bar_idx_in_day - 15: bar_idx_in_day + 1]
                        forward = day_bars.iloc[bar_idx_in_day + 1: bar_idx_in_day + 7]

                        if len(lookback) < 15 or len(forward) < 6:
                            continue

                        entry_price = float(lookback['close'].iloc[-1])
                        if entry_price <= 0:
                            continue

                        # --- Compute target label from forward bars ---
                        fwd_high = float(forward['high'].max())
                        fwd_low = float(forward['low'].min())

                        if direction == 'long':
                            # Max adverse excursion (downside) and favorable (upside)
                            pullback_pct = (entry_price - fwd_low) / entry_price * 100
                            continuation_pct = (fwd_high - entry_price) / entry_price * 100
                            reversal_pct = pullback_pct
                        else:
                            pullback_pct = (fwd_high - entry_price) / entry_price * 100
                            continuation_pct = (entry_price - fwd_low) / entry_price * 100
                            reversal_pct = pullback_pct

                        # Classify
                        if reversal_pct > 1.0 and continuation_pct < 0.3:
                            target = 2  # skip
                        elif pullback_pct > 0.3 and continuation_pct > pullback_pct * 0.5:
                            target = 1  # wait_for_pullback
                        else:
                            target = 0  # enter_now

                        # --- Extract features ---
                        close = lookback['close']
                        high_lb = lookback['high']
                        low_lb = lookback['low']
                        volume_lb = lookback['volume'] if 'volume' in lookback.columns else pd.Series([0] * len(lookback))
                        openp = lookback['open']

                        current_close = float(close.iloc[-1])

                        # price_position_in_range
                        rng_h = float(high_lb.max())
                        rng_l = float(low_lb.min())
                        rng = rng_h - rng_l
                        price_pos = (current_close - rng_l) / rng if rng > 0 else 0.5

                        # last_3_bar_pattern
                        l3c = close.tail(3).values
                        l3o = openp.tail(3).values
                        greens = sum(1 for c, o in zip(l3c, l3o) if c > o)
                        if greens == 3:
                            bar_pattern = 1.0
                        elif greens == 0:
                            bar_pattern = -1.0
                        else:
                            bar_pattern = 0.0

                        # volume_ratio_last_bar
                        avg_v = float(volume_lb.mean())
                        vol_ratio = float(volume_lb.iloc[-1]) / avg_v if avg_v > 0 else 1.0

                        # rsi_14_5m
                        rsi_val = _calculate_rsi_from_series(close, 14)

                        # vwap_distance_pct
                        tp = (high_lb + low_lb + close) / 3.0
                        cumvol = volume_lb.cumsum()
                        cum_tp_vol = (tp * volume_lb).cumsum()
                        last_cv = float(cumvol.iloc[-1])
                        vwap_dist = 0.0
                        if last_cv > 0:
                            vwap_val = float(cum_tp_vol.iloc[-1]) / last_cv
                            if vwap_val > 0:
                                vwap_dist = (current_close - vwap_val) / vwap_val * 100

                        # time_since_open_minutes
                        ts_open_min = max(0.0, float(
                            bar_ts.hour * 60 + bar_ts.minute - (9 * 60 + 30)
                        ))

                        # bar_size_vs_avg
                        bar_ranges = (high_lb - low_lb).values
                        avg_br = float(np.mean(bar_ranges))
                        cur_br = float(bar_ranges[-1])
                        bsva = cur_br / avg_br if avg_br > 0 else 1.0

                        # round_number_proximity
                        if current_close < 50:
                            rl = round(current_close)
                        elif current_close < 200:
                            rl = round(current_close / 5) * 5
                        else:
                            rl = round(current_close / 10) * 10
                        prev_c = float(close.iloc[-2]) if len(close) >= 2 else current_close
                        crossed = (prev_c < rl <= current_close) or (prev_c > rl >= current_close)
                        near = abs(current_close - rl) / max(current_close, 1) < 0.003
                        round_prox = 1.0 if crossed else (0.5 if near else 0.0)

                        # momentum_5bar
                        bars_for_mom = min(5, len(close))
                        p_ago = float(close.iloc[-bars_for_mom])
                        mom5 = (current_close - p_ago) / p_ago * 100 if p_ago > 0 else 0.0

                        # direction_is_long
                        dir_long = 1.0 if direction == 'long' else 0.0

                        row = {
                            'price_position_in_range': price_pos,
                            'last_3_bar_pattern': bar_pattern,
                            'volume_ratio_last_bar': vol_ratio,
                            'rsi_14_5m': rsi_val,
                            'vwap_distance_pct': vwap_dist,
                            'time_since_open_minutes': ts_open_min,
                            'bar_size_vs_avg': bsva,
                            'round_number_proximity': round_prox,
                            'momentum_5bar': mom5,
                            'direction_is_long': dir_long,
                            # Target + metadata
                            'target': target,
                            'symbol': symbol,
                            'signal_date': day_str,
                            'bar_time': str(bar_ts),
                            'entry_price': entry_price,
                            'direction': direction,
                            'rrs': rrs_val,
                            'pullback_pct': pullback_pct,
                            'continuation_pct': continuation_pct,
                        }
                        all_rows.append(row)

                except Exception as e:
                    logger.debug(f"Error processing signal day {signal_date} for {symbol}: {e}")
                    continue

            # Be polite to yfinance
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue

    if not all_rows:
        logger.warning("No training samples generated")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    logger.info(
        f"Generated {len(df)} entry timing samples from {len(symbols)} symbols. "
        f"Class distribution: {df['target'].value_counts().to_dict()}"
    )
    return df
