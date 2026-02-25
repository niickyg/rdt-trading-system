"""
Mean Reversion Scanner

Identifies oversold/overbought reversal setups as a complement to the
momentum-based RealTimeScanner.  Produces signals compatible with the
existing signal format so they can be merged into `active_signals.json`.

Strategies:
  - RSI Reversal: RSI(14) crosses back from oversold/overbought territory
  - VWAP Reversion: Price extended from VWAP with volume confirmation
  - Bollinger Band Squeeze: Price touches outer BB with RSI confirmation
  - Composite Mean Reversion Score (MRS): 0-10 score combining all factors

Risk management filters:
  - Skip if ATR% > 5% (too volatile)
  - Skip if ADX > 35 (strong trend = bad for mean reversion)
  - Stop: swing low/high or 1.5x ATR fallback
  - Target: VWAP or 20-EMA, minimum 1.5R
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from loguru import logger

from utils.timezone import format_timestamp, get_eastern_time


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using exponential (Wilder) smoothing."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR from a DataFrame with columns: high, low, close."""
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period, adjust=False).mean()


def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ADX from a DataFrame with columns: high, low, close."""
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100.0
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    return adx


def _calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute cumulative VWAP for the session.

    Expects columns: high, low, close, volume.
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    cum_tp_vol = (typical_price * df['volume']).cumsum()
    cum_vol = df['volume'].cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_vol


def _calculate_bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    """Return (middle, upper, lower) Bollinger Bands."""
    middle = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def _calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Mean Reversion Scanner
# ---------------------------------------------------------------------------

class MeanReversionScanner:
    """
    Scans for mean-reversion setups: oversold bounces and overbought fades.

    Uses the same config dict / symbol list conventions as RealTimeScanner
    so it can be called alongside the momentum scanner.
    """

    # Default thresholds (overridable via config)
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    RSI_CONFIRM_LONG = 30   # RSI must cross back above this for long entry
    RSI_CONFIRM_SHORT = 70  # RSI must cross back below this for short entry
    BB_RSI_LONG_THRESHOLD = 35
    BB_RSI_SHORT_THRESHOLD = 65
    VWAP_ATR_DISTANCE = 2.0
    VWAP_VOLUME_MULTIPLIER = 1.5
    MAX_ATR_PCT = 5.0        # Skip if ATR% > 5%
    MAX_ADX = 35.0            # Skip if ADX > 35
    MIN_RR = 1.5             # Minimum risk-reward
    ATR_STOP_MULTIPLIER = 1.5

    def __init__(self, config: Dict):
        self.config = config
        # Allow overrides from config
        self.rsi_oversold = config.get('mr_rsi_oversold', self.RSI_OVERSOLD)
        self.rsi_overbought = config.get('mr_rsi_overbought', self.RSI_OVERBOUGHT)
        self.max_atr_pct = config.get('mr_max_atr_pct', self.MAX_ATR_PCT)
        self.max_adx = config.get('mr_max_adx', self.MAX_ADX)
        self.min_rr = config.get('mr_min_rr', self.MIN_RR)
        self.atr_stop_mult = config.get('mr_atr_stop_multiplier', self.ATR_STOP_MULTIPLIER)
        self.vwap_atr_distance = config.get('mr_vwap_atr_distance', self.VWAP_ATR_DISTANCE)
        self.vwap_vol_mult = config.get('mr_vwap_volume_multiplier', self.VWAP_VOLUME_MULTIPLIER)
        logger.info("MeanReversionScanner initialized")

    # ------------------------------------------------------------------
    # Pre-flight filters
    # ------------------------------------------------------------------

    def _passes_filters(self, daily: pd.DataFrame, atr: float, price: float) -> bool:
        """Return True if the symbol passes volatility and trend filters."""
        # ATR% filter
        if price <= 0:
            return False
        atr_pct = (atr / price) * 100.0
        if atr_pct > self.max_atr_pct:
            logger.debug(f"Skipped: ATR%={atr_pct:.1f}% > {self.max_atr_pct}%")
            return False

        # ADX filter — need enough daily bars
        if len(daily) >= 28:
            adx = _calculate_adx(daily)
            if adx.iloc[-1] > self.max_adx:
                logger.debug(f"Skipped: ADX={adx.iloc[-1]:.1f} > {self.max_adx}")
                return False

        return True

    # ------------------------------------------------------------------
    # Individual pattern detectors
    # ------------------------------------------------------------------

    def _detect_rsi_reversal(self, daily: pd.DataFrame, atr: float) -> Optional[Dict]:
        """
        RSI Reversal:
          Long  — RSI was below 30 within last 3 bars, now crosses back above 30.
          Short — RSI was above 70 within last 3 bars, now crosses back below 70.
        """
        if len(daily) < 20:
            return None

        rsi = _calculate_rsi(daily['close'])
        if rsi.isna().iloc[-1]:
            return None

        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        recent_rsi = rsi.iloc[-4:-1]  # last 3 bars before current

        # Long: RSI was below oversold, now crossing back above
        if (recent_rsi < self.rsi_oversold).any() and current_rsi > self.rsi_oversold and prev_rsi <= self.rsi_oversold:
            return {
                'direction': 'long',
                'sub_type': 'rsi_reversal',
                'rsi': round(float(current_rsi), 2),
                'confidence_component': min(1.0, (self.rsi_oversold - recent_rsi.min()) / 10.0),
            }

        # Short: RSI was above overbought, now crossing back below
        if (recent_rsi > self.rsi_overbought).any() and current_rsi < self.rsi_overbought and prev_rsi >= self.rsi_overbought:
            return {
                'direction': 'short',
                'sub_type': 'rsi_reversal',
                'rsi': round(float(current_rsi), 2),
                'confidence_component': min(1.0, (recent_rsi.max() - self.rsi_overbought) / 10.0),
            }

        return None

    def _detect_vwap_reversion(self, intraday: pd.DataFrame, atr: float) -> Optional[Dict]:
        """
        VWAP Reversion:
          Long  — Price > 2 ATR below VWAP with 1.5x average volume.
          Short — Price > 2 ATR above VWAP with 1.5x average volume.
        """
        if len(intraday) < 10:
            return None

        vwap = _calculate_vwap(intraday)
        if vwap.isna().iloc[-1] or atr <= 0:
            return None

        current_price = intraday['close'].iloc[-1]
        current_vwap = vwap.iloc[-1]
        distance_from_vwap = current_price - current_vwap

        # Volume confirmation
        avg_volume = intraday['volume'].rolling(window=20, min_periods=5).mean()
        if avg_volume.isna().iloc[-1] or avg_volume.iloc[-1] <= 0:
            return None
        volume_ratio = intraday['volume'].iloc[-1] / avg_volume.iloc[-1]

        # Long: price well below VWAP with volume spike
        if distance_from_vwap < -self.vwap_atr_distance * atr and volume_ratio >= self.vwap_vol_mult:
            return {
                'direction': 'long',
                'sub_type': 'vwap_reversion',
                'vwap': round(float(current_vwap), 2),
                'distance_atr': round(float(abs(distance_from_vwap) / atr), 2),
                'volume_ratio': round(float(volume_ratio), 2),
                'confidence_component': min(1.0, volume_ratio / 3.0),
            }

        # Short: price well above VWAP with volume spike
        if distance_from_vwap > self.vwap_atr_distance * atr and volume_ratio >= self.vwap_vol_mult:
            return {
                'direction': 'short',
                'sub_type': 'vwap_reversion',
                'vwap': round(float(current_vwap), 2),
                'distance_atr': round(float(abs(distance_from_vwap) / atr), 2),
                'volume_ratio': round(float(volume_ratio), 2),
                'confidence_component': min(1.0, volume_ratio / 3.0),
            }

        return None

    def _detect_bb_squeeze(self, daily: pd.DataFrame, atr: float) -> Optional[Dict]:
        """
        Bollinger Band Squeeze:
          Long  — Price touches lower BB(20,2), RSI < 35, then green candle.
          Short — Price touches upper BB(20,2), RSI > 65, then red candle.
        """
        if len(daily) < 22:
            return None

        close = daily['close']
        _middle, upper, lower = _calculate_bollinger(close, period=20, num_std=2.0)
        rsi = _calculate_rsi(close)

        if lower.isna().iloc[-1] or rsi.isna().iloc[-1]:
            return None

        prev_close = close.iloc[-2]
        prev_low = daily['low'].iloc[-2]
        current_close = close.iloc[-1]
        current_open = daily['open'].iloc[-1]
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        # Long: previous bar touched lower BB, RSI < 35, current bar is green
        if prev_low <= lower.iloc[-2] and prev_rsi < self.BB_RSI_LONG_THRESHOLD and current_close > current_open:
            return {
                'direction': 'long',
                'sub_type': 'bb_squeeze',
                'rsi': round(float(current_rsi), 2),
                'bb_lower': round(float(lower.iloc[-1]), 2),
                'bb_upper': round(float(upper.iloc[-1]), 2),
                'confidence_component': min(1.0, (self.BB_RSI_LONG_THRESHOLD - prev_rsi) / 15.0),
            }

        # Short: previous bar touched upper BB, RSI > 65, current bar is red
        prev_high = daily['high'].iloc[-2]
        if prev_high >= upper.iloc[-2] and prev_rsi > self.BB_RSI_SHORT_THRESHOLD and current_close < current_open:
            return {
                'direction': 'short',
                'sub_type': 'bb_squeeze',
                'rsi': round(float(current_rsi), 2),
                'bb_lower': round(float(lower.iloc[-1]), 2),
                'bb_upper': round(float(upper.iloc[-1]), 2),
                'confidence_component': min(1.0, (prev_rsi - self.BB_RSI_SHORT_THRESHOLD) / 15.0),
            }

        return None

    # ------------------------------------------------------------------
    # Composite Mean Reversion Score (MRS): 0 – 10
    # ------------------------------------------------------------------

    def _calculate_mrs(self, daily: pd.DataFrame, intraday: Optional[pd.DataFrame], atr: float) -> float:
        """
        Composite Mean Reversion Score.

        Components (each 0-2.5, total 0-10):
          1. RSI distance from 50  (further = higher score)
          2. Distance from VWAP in ATR units  (further = higher)
          3. Bollinger Band position (closer to outer band = higher)
          4. Volume confirmation  (higher volume = higher)
        """
        score = 0.0
        close = daily['close']
        current_price = close.iloc[-1]

        # 1. RSI distance from 50 (0 - 2.5)
        rsi = _calculate_rsi(close)
        if not rsi.isna().iloc[-1]:
            rsi_dist = abs(rsi.iloc[-1] - 50.0)
            score += min(2.5, rsi_dist / 20.0 * 2.5)

        # 2. VWAP distance (0 - 2.5)
        if intraday is not None and len(intraday) >= 10 and atr > 0:
            vwap = _calculate_vwap(intraday)
            if not vwap.isna().iloc[-1]:
                vwap_dist = abs(current_price - vwap.iloc[-1]) / atr
                score += min(2.5, vwap_dist / 3.0 * 2.5)

        # 3. Bollinger Band position (0 - 2.5)
        if len(daily) >= 22:
            _middle, upper, lower = _calculate_bollinger(close, period=20, num_std=2.0)
            if not lower.isna().iloc[-1] and not upper.isna().iloc[-1]:
                bb_range = upper.iloc[-1] - lower.iloc[-1]
                if bb_range > 0:
                    # 0 = at middle, 1 = at outer band
                    bb_pos = abs(current_price - (_middle := (upper.iloc[-1] + lower.iloc[-1]) / 2)) / (bb_range / 2)
                    score += min(2.5, bb_pos * 2.5)

        # 4. Volume confirmation (0 - 2.5)
        vol = daily['volume']
        avg_vol = vol.rolling(window=20, min_periods=5).mean()
        if not avg_vol.isna().iloc[-1] and avg_vol.iloc[-1] > 0:
            vol_ratio = vol.iloc[-1] / avg_vol.iloc[-1]
            score += min(2.5, (vol_ratio - 0.5) / 2.0 * 2.5)

        return round(max(0.0, min(10.0, score)), 2)

    # ------------------------------------------------------------------
    # Risk management: stop / target
    # ------------------------------------------------------------------

    def _compute_stop_target(
        self,
        direction: str,
        entry_price: float,
        atr: float,
        daily: pd.DataFrame,
        intraday: Optional[pd.DataFrame],
    ) -> Optional[Dict]:
        """
        Compute stop_price and target_price.

        Stop: swing low/high over last 5 bars, or 1.5x ATR fallback.
        Target: VWAP or 20-EMA (mean reversion target), minimum 1.5R.

        Returns dict with stop_price, target_price, risk, reward, rr
        or None if R:R is below minimum.
        """
        if atr <= 0 or entry_price <= 0:
            return None

        lookback = min(5, len(daily) - 1)
        recent = daily.iloc[-(lookback + 1):-1]

        if direction == 'long':
            # Stop below swing low or ATR fallback
            swing_low = recent['low'].min() if len(recent) > 0 else entry_price
            atr_stop = entry_price - self.atr_stop_mult * atr
            stop_price = min(swing_low, atr_stop)
            # Safety: stop must be below entry
            if stop_price >= entry_price:
                stop_price = entry_price - self.atr_stop_mult * atr

            risk = entry_price - stop_price
            if risk <= 0:
                return None

            # Target: VWAP or 20-EMA, whichever is further
            ema20 = _calculate_ema(daily['close'], 20)
            target_ema = ema20.iloc[-1] if not ema20.isna().iloc[-1] else entry_price + 2 * atr

            target_vwap = entry_price + 2 * atr  # fallback
            if intraday is not None and len(intraday) >= 10:
                vwap = _calculate_vwap(intraday)
                if not vwap.isna().iloc[-1] and vwap.iloc[-1] > entry_price:
                    target_vwap = vwap.iloc[-1]

            target_price = max(target_ema, target_vwap)

            # Enforce minimum R:R
            reward = target_price - entry_price
            if reward / risk < self.min_rr:
                target_price = entry_price + self.min_rr * risk
                reward = target_price - entry_price

        else:  # short
            swing_high = recent['high'].max() if len(recent) > 0 else entry_price
            atr_stop = entry_price + self.atr_stop_mult * atr
            stop_price = max(swing_high, atr_stop)
            if stop_price <= entry_price:
                stop_price = entry_price + self.atr_stop_mult * atr

            risk = stop_price - entry_price
            if risk <= 0:
                return None

            ema20 = _calculate_ema(daily['close'], 20)
            target_ema = ema20.iloc[-1] if not ema20.isna().iloc[-1] else entry_price - 2 * atr

            target_vwap = entry_price - 2 * atr
            if intraday is not None and len(intraday) >= 10:
                vwap = _calculate_vwap(intraday)
                if not vwap.isna().iloc[-1] and vwap.iloc[-1] < entry_price:
                    target_vwap = vwap.iloc[-1]

            target_price = min(target_ema, target_vwap)

            reward = entry_price - target_price
            if reward / risk < self.min_rr:
                target_price = entry_price - self.min_rr * risk
                reward = entry_price - target_price

        rr = reward / risk if risk > 0 else 0
        return {
            'stop_price': round(float(stop_price), 2),
            'target_price': round(float(target_price), 2),
            'risk': round(float(risk), 2),
            'reward': round(float(reward), 2),
            'rr': round(float(rr), 2),
        }

    # ------------------------------------------------------------------
    # Main scan for a single symbol
    # ------------------------------------------------------------------

    def scan_symbol(
        self,
        symbol: str,
        daily: pd.DataFrame,
        intraday: Optional[pd.DataFrame],
        atr: float,
        current_price: float,
    ) -> List[Dict]:
        """
        Scan a single symbol for mean-reversion setups.

        Args:
            symbol: Ticker symbol.
            daily: Daily OHLCV DataFrame (lowercase columns).
            intraday: Intraday (e.g. 5m) OHLCV DataFrame or None.
            atr: Current ATR value.
            current_price: Latest price.

        Returns:
            List of signal dicts (may be empty).
        """
        if not self._passes_filters(daily, atr, current_price):
            return []

        signals: List[Dict] = []
        detections = []

        # Run all pattern detectors
        rsi_det = self._detect_rsi_reversal(daily, atr)
        if rsi_det:
            detections.append(rsi_det)

        vwap_det = self._detect_vwap_reversion(intraday, atr) if intraday is not None else None
        if vwap_det:
            detections.append(vwap_det)

        bb_det = self._detect_bb_squeeze(daily, atr)
        if bb_det:
            detections.append(bb_det)

        # Compute MRS for context
        mrs = self._calculate_mrs(daily, intraday, atr)

        for det in detections:
            direction = det['direction']
            rt = self._compute_stop_target(direction, current_price, atr, daily, intraday)
            if rt is None:
                continue

            # Confidence: combine pattern-specific component with MRS
            pattern_conf = det.get('confidence_component', 0.5)
            confidence = round(min(1.0, 0.4 * pattern_conf + 0.6 * (mrs / 10.0)), 2)

            signal = {
                'symbol': str(symbol),
                'direction': direction,
                'strength': 'strong' if mrs >= 7 else ('moderate' if mrs >= 4 else 'weak'),
                'rrs': round(float(mrs), 2),  # Use MRS in the rrs field for sorting compatibility
                'entry_price': round(float(current_price), 2),
                'stop_price': rt['stop_price'],
                'target_price': rt['target_price'],
                'atr': round(float(atr), 2),
                'confidence': confidence,
                'generated_at': format_timestamp(),
                'strategy': 'mean_reversion',
                'strategy_sub_type': det['sub_type'],
                'mean_reversion_score': mrs,
                # Extra detail fields
                'risk_reward': rt['rr'],
                # Preserve standard fields expected by save_signals / dashboard
                'stock_change_pct': 0.0,
                'spy_change_pct': 0.0,
                'daily_strong': False,
                'daily_weak': False,
                'extended_hours': False,
                'session': 'regular',
                'timeframe_alignment': False,
                'trend_by_timeframe': {},
                'entry_timing_score': 50,
                'alignment_direction': 'unknown',
                'key_levels': {},
            }

            # Attach pattern-specific detail
            for key in ('rsi', 'vwap', 'distance_atr', 'volume_ratio',
                        'bb_lower', 'bb_upper'):
                if key in det:
                    signal[f'mr_{key}'] = det[key]

            signals.append(signal)

        return signals


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_mean_reversion_scan(
    symbols: List[str],
    config: Dict,
    batch_5m: Optional[pd.DataFrame] = None,
    batch_daily: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    """
    Run a mean-reversion scan over *symbols* and return a list of signal dicts
    compatible with the existing signal format.

    This function is designed to be called from ``RealTimeScanner.scan_once()``
    after the batch data has already been fetched, so callers should pass
    ``batch_5m`` and ``batch_daily`` to avoid redundant API calls.

    Args:
        symbols: List of ticker symbols to scan.
        config: Scanner configuration dictionary.
        batch_5m: Pre-fetched batch intraday data (multi-level columns).
        batch_daily: Pre-fetched batch daily data (multi-level columns).

    Returns:
        List of signal dicts, possibly empty.
    """
    if not config.get('mean_reversion_enabled', True):
        return []

    scanner = MeanReversionScanner(config)
    all_signals: List[Dict] = []

    for symbol in symbols:
        try:
            daily = _extract_symbol_df(symbol, batch_daily)
            intraday = _extract_symbol_df(symbol, batch_5m)

            if daily is None or len(daily) < 22:
                continue

            # Compute ATR from daily data
            atr_series = _calculate_atr(daily)
            if atr_series.isna().iloc[-1]:
                continue
            atr = float(atr_series.iloc[-1])

            current_price = float(daily['close'].iloc[-1])
            if current_price <= 0:
                continue

            sigs = scanner.scan_symbol(symbol, daily, intraday, atr, current_price)
            all_signals.extend(sigs)

        except Exception as e:
            logger.debug(f"Mean reversion scan error for {symbol}: {e}")
            continue

    logger.info(f"Mean reversion scan complete: {len(all_signals)} signals from {len(symbols)} symbols")
    return all_signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_symbol_df(
    symbol: str, batch: Optional[pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """
    Extract a single symbol's DataFrame from a multi-level batch DataFrame.

    Returns None if the symbol is not found or the data is empty.
    """
    if batch is None or batch.empty:
        return None

    try:
        if isinstance(batch.columns, pd.MultiIndex):
            level_values = batch.columns.get_level_values(0)
            if symbol in level_values:
                df = batch[symbol].dropna(how='all')
            else:
                return None
        else:
            # Single-symbol DataFrame — assume it IS the symbol
            df = batch.dropna(how='all')

        if df.empty:
            return None

        # Normalize column names to lowercase
        df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
        return df

    except Exception:
        return None
