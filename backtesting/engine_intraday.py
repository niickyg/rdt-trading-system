"""
Intraday Backtesting Engine
Processes 1-minute bar data for realistic intraday stop/target/VWAP simulation.

Unlike engine_enhanced.py (daily bars only), this engine:
  - Iterates through every 1-min bar within market hours (9:30-16:00 ET)
  - Checks stops/targets on each bar (realistic intraday stop-out behavior)
  - Calculates VWAP incrementally and applies the VWAP gate
  - Applies first-hour filter (blocks signals before configurable time)
  - Uses exact stop/target prices for exits (not bar close)

Input: Dict[str, pd.DataFrame] of 1-min parquet data (UTC timestamps).
Output: Same EnhancedBacktestResult / EnhancedTrade for compatibility.
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

from loguru import logger

from backtesting.engine_enhanced import EnhancedTrade, EnhancedBacktestResult
from risk.models import RiskLimits
from risk.position_sizer import PositionSizer


# ============================================================================
# Constants
# ============================================================================

# US Eastern timezone offset from UTC (standard = -5h, DST = -4h)
# We use pytz-free approach: detect DST from the data timestamps
_MARKET_OPEN_ET = time(9, 30)
_MARKET_CLOSE_ET = time(16, 0)


# ============================================================================
# Timezone helpers (no pytz dependency)
# ============================================================================

def _is_dst(dt_date: date) -> bool:
    """Check if a date falls within US Eastern Daylight Time (EDT).

    EDT: second Sunday in March at 2:00 AM through first Sunday in November
    at 2:00 AM.
    """
    year = dt_date.year

    # Second Sunday in March
    march_1 = date(year, 3, 1)
    # Day of week: Monday=0, Sunday=6
    days_until_sunday = (6 - march_1.weekday()) % 7
    first_sunday_march = march_1 + timedelta(days=days_until_sunday)
    second_sunday_march = first_sunday_march + timedelta(days=7)

    # First Sunday in November
    nov_1 = date(year, 11, 1)
    days_until_sunday_nov = (6 - nov_1.weekday()) % 7
    first_sunday_nov = nov_1 + timedelta(days=days_until_sunday_nov)

    return second_sunday_march <= dt_date < first_sunday_nov


def _et_offset(dt_date: date) -> timedelta:
    """Return UTC offset for US Eastern time on the given date."""
    if _is_dst(dt_date):
        return timedelta(hours=-4)
    return timedelta(hours=-5)


def _utc_to_et(utc_dt: datetime, dt_date: date) -> time:
    """Convert a UTC datetime to Eastern time (time only)."""
    offset = _et_offset(dt_date)
    et_dt = utc_dt + offset
    return et_dt.time()


def _market_open_utc(dt_date: date) -> pd.Timestamp:
    """Return market open time in UTC for the given date (tz-aware)."""
    offset = _et_offset(dt_date)
    et_open = datetime.combine(dt_date, _MARKET_OPEN_ET)
    utc_naive = et_open - offset
    return pd.Timestamp(utc_naive, tz='UTC')


def _market_close_utc(dt_date: date) -> pd.Timestamp:
    """Return market close time in UTC for the given date (tz-aware)."""
    offset = _et_offset(dt_date)
    et_close = datetime.combine(dt_date, _MARKET_CLOSE_ET)
    utc_naive = et_close - offset
    return pd.Timestamp(utc_naive, tz='UTC')


# ============================================================================
# VIX / Sector / Regime / SPY Gate / SMA Gate — self-contained copies
# (mirrors run_walkforward_v2.py logic; engine is self-contained)
# ============================================================================

def _get_vix_regime(vix_value: float) -> Dict:
    """VIX regime classification (mirrors scanner/vix_filter.py)."""
    if vix_value < 15.0:
        return {"level": "low", "pos_mult": 1.10, "rrs_adj": -0.25, "allow_longs": True}
    elif vix_value < 20.0:
        return {"level": "normal", "pos_mult": 1.00, "rrs_adj": 0.00, "allow_longs": True}
    elif vix_value < 25.0:
        return {"level": "elevated", "pos_mult": 0.75, "rrs_adj": 0.50, "allow_longs": True}
    elif vix_value < 35.0:
        return {"level": "high", "pos_mult": 0.50, "rrs_adj": 1.00, "allow_longs": True}
    else:
        return {"level": "extreme", "pos_mult": 0.00, "rrs_adj": 2.00, "allow_longs": False}


def _detect_regime(spy_daily: pd.DataFrame, current_date: date) -> Dict:
    """Detect market regime from SPY daily bars (mirrors regime_params.py)."""
    close_col = _close_col(spy_daily)
    high_col = _high_col(spy_daily)
    low_col = _low_col(spy_daily)

    spy_up_to = spy_daily[spy_daily.index.date <= current_date]
    if len(spy_up_to) < 200:
        return {"regime": "default", "rrs_threshold": 2.0, "risk_per_trade": 0.015, "wider_stops": False}

    close_s = spy_up_to[close_col].astype(float)
    current_price = float(close_s.iloc[-1])
    ema_50 = float(close_s.ewm(span=50, adjust=False).mean().iloc[-1])
    ema_200 = float(close_s.ewm(span=200, adjust=False).mean().iloc[-1])

    high_s = spy_up_to[high_col].astype(float)
    low_s = spy_up_to[low_col].astype(float)
    tr = pd.concat([
        high_s - low_s,
        (high_s - close_s.shift(1)).abs(),
        (low_s - close_s.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_14 = float(tr.rolling(14).mean().iloc[-1])
    atr_pct = (atr_14 / current_price) * 100

    if atr_pct > 2.0:
        return {"regime": "high_volatility", "rrs_threshold": 2.25, "risk_per_trade": 0.01, "wider_stops": True}
    if current_price > ema_50 and ema_50 > ema_200:
        return {"regime": "bull_trending", "rrs_threshold": 1.5, "risk_per_trade": 0.02, "wider_stops": False}
    if current_price < ema_50 and ema_50 < ema_200:
        return {"regime": "bear_trending", "rrs_threshold": 2.5, "risk_per_trade": 0.01, "wider_stops": False}
    return {"regime": "default", "rrs_threshold": 2.0, "risk_per_trade": 0.015, "wider_stops": False}


def _get_spy_gate(spy_daily: pd.DataFrame, current_date: date) -> Dict:
    """SPY hard gate — block longs in bearish, shorts in bullish."""
    close_col = _close_col(spy_daily)
    spy_up_to = spy_daily[spy_daily.index.date <= current_date]
    if len(spy_up_to) < 200:
        return {"allow_longs": True, "allow_shorts": True, "spy_trend": "unknown"}

    close_s = spy_up_to[close_col].astype(float)
    current_price = float(close_s.iloc[-1])
    ema_50 = float(close_s.ewm(span=50, adjust=False).mean().iloc[-1])
    ema_200 = float(close_s.ewm(span=200, adjust=False).mean().iloc[-1])

    above_50 = current_price > ema_50
    above_200 = current_price > ema_200

    if above_50 and above_200:
        return {"allow_longs": True, "allow_shorts": False, "spy_trend": "bullish"}
    elif not above_50 and not above_200:
        return {"allow_longs": False, "allow_shorts": True, "spy_trend": "bearish"}
    return {"allow_longs": True, "allow_shorts": True, "spy_trend": "mixed"}


def _check_stock_sma_gate(stock_daily: pd.DataFrame, current_date: date) -> Dict:
    """Stock 50/200 SMA gate for longs/shorts filtering."""
    close_col = _close_col(stock_daily)
    data_up_to = stock_daily[stock_daily.index.date <= current_date]
    if len(data_up_to) < 50:
        return {"above_sma50": None, "above_sma200": None}

    close_s = data_up_to[close_col].astype(float)
    current_price = float(close_s.iloc[-1])
    sma_50 = float(close_s.rolling(50).mean().iloc[-1])
    above_50 = current_price > sma_50

    above_200 = None
    if len(data_up_to) >= 200:
        sma_200 = float(close_s.rolling(200).mean().iloc[-1])
        above_200 = current_price > sma_200

    return {"above_sma50": above_50, "above_sma200": above_200}


def _compute_sector_rs(
    sector_etf_data: Dict[str, pd.DataFrame],
    spy_daily: pd.DataFrame,
    current_date: date,
    lookback: int = 5,
) -> Dict[str, float]:
    """Compute sector relative strength vs SPY."""
    # Import SECTOR_ETF_MAP lazily to avoid circular imports at module level
    try:
        from scanner.sector_filter import SECTOR_ETF_MAP
    except ImportError:
        return {}

    close_col = _close_col(spy_daily)
    spy_up_to = spy_daily[spy_daily.index.date <= current_date]
    if len(spy_up_to) < lookback + 1:
        return {}

    spy_current = float(spy_up_to[close_col].iloc[-1])
    spy_past = float(spy_up_to[close_col].iloc[-(lookback + 1)])
    spy_pct = ((spy_current / spy_past) - 1) * 100

    result = {}
    for sector_name, etf_ticker in SECTOR_ETF_MAP.items():
        etf_df = sector_etf_data.get(etf_ticker)
        if etf_df is None:
            continue
        etf_col = _close_col(etf_df)
        etf_up_to = etf_df[etf_df.index.date <= current_date]
        if len(etf_up_to) < lookback + 1:
            continue
        etf_current = float(etf_up_to[etf_col].iloc[-1])
        etf_past = float(etf_up_to[etf_col].iloc[-(lookback + 1)])
        etf_pct = ((etf_current / etf_past) - 1) * 100
        sector_rs = etf_pct - spy_pct
        if np.isfinite(sector_rs):
            result[sector_name] = sector_rs
    return result


def _get_sector_boost(symbol: str, direction: str, sector_rs_map: Dict[str, float]) -> float:
    """Get RRS boost/penalty from sector relative strength."""
    try:
        from risk.risk_manager import SECTOR_MAP
    except ImportError:
        return 0.0

    sector = SECTOR_MAP.get(symbol, "other")
    if sector == "other" or sector not in sector_rs_map:
        return 0.0
    rs = sector_rs_map[sector]
    if direction == "long":
        if rs > 0:
            return 0.25
        elif rs < -1.0:
            return -0.50
    elif direction == "short":
        if rs < -1.0:
            return 0.25
        elif rs > 0:
            return -0.50
    return 0.0


def _get_intermarket_adjustment(
    intermarket_data: Optional[Dict[str, pd.DataFrame]],
    current_date: date,
    lookback: int = 20,
) -> Tuple[float, float]:
    """
    Compute intermarket RRS adjustment and position size multiplier.

    Uses the same Murphy framework as scanner/intermarket_analyzer.py but
    operates on pre-loaded daily DataFrames instead of live yfinance fetches.

    Returns: (rrs_adjustment, position_multiplier)
    """
    if intermarket_data is None:
        return 0.0, 1.0

    def _get_trend(symbol_key: str) -> Optional[float]:
        df = intermarket_data.get(symbol_key)
        if df is None:
            return None
        col = _close_col(df)
        up_to = df[df.index.date <= current_date]
        if len(up_to) < lookback + 1:
            return None
        current = float(up_to[col].iloc[-1])
        past = float(up_to[col].iloc[-(lookback + 1)])
        if past == 0:
            return None
        return (current / past) - 1.0

    bonds = _get_trend("TLT")
    spy = _get_trend("SPY")
    dollar = _get_trend("UUP")
    gold = _get_trend("GLD")
    smallcap = _get_trend("IWM")

    signals = []
    weights = {"bonds_stocks": 0.35, "risk_on_off": 0.30, "dollar": 0.20, "gold": 0.15}

    # Bonds vs stocks divergence
    if bonds is not None and spy is not None:
        if bonds > 0 and spy < 0:
            signals.append(("bonds_stocks", -1.0))  # risk off
        elif bonds < 0 and spy > 0:
            signals.append(("bonds_stocks", 1.0))   # risk on
        else:
            signals.append(("bonds_stocks", 0.0))
    # Dollar trend (rising = headwind)
    if dollar is not None:
        signals.append(("dollar", np.clip(-dollar * 10, -1.0, 1.0)))
    # Gold signal (rising gold + falling stocks = risk off)
    if gold is not None and spy is not None:
        if gold > 0 and spy < 0:
            signals.append(("gold", -1.0))
        elif gold < 0 and spy > 0:
            signals.append(("gold", 1.0))
        else:
            signals.append(("gold", 0.0))
    # Risk on/off ratio (IWM/SPY)
    if smallcap is not None and spy is not None:
        ratio_change = smallcap - spy
        signals.append(("risk_on_off", np.clip(ratio_change * 10, -1.0, 1.0)))

    if not signals:
        return 0.0, 1.0

    composite = sum(weights.get(k, 0) * v for k, v in signals)
    total_weight = sum(weights.get(k, 0) for k, _ in signals)
    if total_weight > 0:
        composite /= total_weight
        composite *= total_weight  # scale back by actual coverage
    composite = float(np.clip(composite, -1.0, 1.0))

    # Regime thresholds
    if composite > 0.3:
        return -0.25, 1.10  # risk_on
    elif composite < -0.3:
        return 0.50, 0.75   # risk_off
    return 0.0, 1.0          # neutral


# ============================================================================
# Column-name helpers (handle uppercase/lowercase)
# ============================================================================

def _close_col(df: pd.DataFrame) -> str:
    return "Close" if "Close" in df.columns else "close"

def _open_col(df: pd.DataFrame) -> str:
    return "Open" if "Open" in df.columns else "open"

def _high_col(df: pd.DataFrame) -> str:
    return "High" if "High" in df.columns else "high"

def _low_col(df: pd.DataFrame) -> str:
    return "Low" if "Low" in df.columns else "low"

def _volume_col(df: pd.DataFrame) -> str:
    return "Volume" if "Volume" in df.columns else "volume"


# ============================================================================
# Daily chart confirmation (3/8 EMA crossover — mirrors rrs.py relaxed)
# ============================================================================

def _check_daily_strength(daily_df: pd.DataFrame) -> bool:
    """Relaxed daily strength check (any 3 of 5 criteria)."""
    if len(daily_df) < 21:
        return False
    c = daily_df["close"].astype(float)
    o = daily_df["open"].astype(float)
    lo = daily_df["low"].astype(float)

    ema3 = c.ewm(span=3, adjust=False).mean()
    ema8 = c.ewm(span=8, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()

    last3_c = c.iloc[-3:]
    last3_o = o.iloc[-3:]

    score = sum([
        float(ema3.iloc[-1]) > float(ema8.iloc[-1]),
        float(ema8.iloc[-1]) > float(ema21.iloc[-1]),
        float(c.iloc[-1]) > float(ema8.iloc[-1]),
        float(lo.iloc[-1]) > float(lo.iloc[-5]) if len(lo) >= 5 else False,
        int((last3_c > last3_o).sum()) >= 2,
    ])
    return score >= 3


def _check_daily_weakness(daily_df: pd.DataFrame) -> bool:
    """Relaxed daily weakness check (any 3 of 5 criteria)."""
    if len(daily_df) < 21:
        return False
    c = daily_df["close"].astype(float)
    o = daily_df["open"].astype(float)
    hi = daily_df["high"].astype(float)

    ema3 = c.ewm(span=3, adjust=False).mean()
    ema8 = c.ewm(span=8, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()

    last3_c = c.iloc[-3:]
    last3_o = o.iloc[-3:]

    score = sum([
        float(ema8.iloc[-1]) > float(ema3.iloc[-1]),
        float(ema21.iloc[-1]) > float(ema8.iloc[-1]),
        float(c.iloc[-1]) < float(ema8.iloc[-1]),
        float(hi.iloc[-1]) < float(hi.iloc[-5]) if len(hi) >= 5 else False,
        int((last3_c < last3_o).sum()) >= 2,
    ])
    return score >= 3


# ============================================================================
# ATR calculation (self-contained, mirrors shared/indicators/rrs.py)
# ============================================================================

def _calculate_atr(daily_df: pd.DataFrame, period: int = 14) -> float:
    """Calculate the latest ATR value from daily OHLC data."""
    if len(daily_df) < period + 1:
        return 0.0
    h = daily_df["high"].astype(float)
    lo = daily_df["low"].astype(float)
    c = daily_df["close"].astype(float)

    tr = pd.concat([
        h - lo,
        (h - c.shift(1)).abs(),
        (lo - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    val = atr.iloc[-1]
    return float(val) if np.isfinite(val) else 0.0


# ============================================================================
# IntradayBacktestEngine
# ============================================================================

class IntradayBacktestEngine:
    """
    Intraday backtesting engine that processes 1-minute bar data.

    For each trading day:
      1. Iterate through 1-min bars within 9:30 AM - 4:00 PM ET
      2. On every bar: check stops / targets / trailing / scaled exits
      3. At scan_time (default 10:00 AM): generate and filter signals
      4. Calculate VWAP incrementally for the VWAP gate
    """

    def __init__(
        self,
        initial_capital: float = 25000.0,
        risk_limits: Optional[RiskLimits] = None,
        rrs_threshold: float = 2.0,
        max_positions: int = 8,

        # Stop / Target
        stop_atr_multiplier: float = 1.5,
        target_atr_multiplier: float = 2.0,

        # Trailing stop
        use_trailing_stop: bool = True,
        breakeven_trigger_r: float = 1.0,
        trailing_atr_multiplier: float = 1.0,

        # Scaled exits
        use_scaled_exits: bool = True,
        scale_1_target_r: float = 1.0,
        scale_1_percent: float = 0.5,
        scale_2_target_r: float = 1.5,
        scale_2_percent: float = 0.25,

        # Time stop
        use_time_stop: bool = True,
        max_holding_days: int = 12,
        stale_trade_days: int = 6,

        # Intraday-specific
        scan_time_minutes_after_open: int = 30,  # 10:00 AM = 30 min after 9:30
        first_hour_block: bool = True,
        vwap_gate_enabled: bool = True,
        spy_gate_enabled: bool = True,
        sma_gate_enabled: bool = True,

        # External filter data (daily bars)
        vix_data: Optional[pd.DataFrame] = None,
        sector_etf_data: Optional[Dict[str, pd.DataFrame]] = None,
        intermarket_data: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        self.initial_capital = initial_capital
        self.risk_limits = risk_limits or RiskLimits()

        self.rrs_threshold = rrs_threshold
        self.max_positions = max_positions

        self.stop_atr_multiplier = stop_atr_multiplier
        self.target_atr_multiplier = target_atr_multiplier

        self.use_trailing_stop = use_trailing_stop
        self.breakeven_trigger_r = breakeven_trigger_r
        self.trailing_atr_multiplier = trailing_atr_multiplier

        self.use_scaled_exits = use_scaled_exits
        self.scale_1_target_r = scale_1_target_r
        self.scale_1_percent = scale_1_percent
        self.scale_2_target_r = scale_2_target_r
        self.scale_2_percent = scale_2_percent

        self.use_time_stop = use_time_stop
        self.max_holding_days = max_holding_days
        self.stale_trade_days = stale_trade_days

        self.scan_time_minutes_after_open = scan_time_minutes_after_open
        self.first_hour_block = first_hour_block
        self.vwap_gate_enabled = vwap_gate_enabled
        self.spy_gate_enabled = spy_gate_enabled
        self.sma_gate_enabled = sma_gate_enabled

        self.vix_data = vix_data
        self.sector_etf_data = sector_etf_data or {}
        self.intermarket_data = intermarket_data

        self.position_sizer = PositionSizer(self.risk_limits)

        # ---- State (reset each run) ----
        self.capital: float = initial_capital
        self.positions: Dict[str, EnhancedTrade] = {}
        self.trades: List[EnhancedTrade] = []
        self.equity_curve: List[Dict] = []
        self.peak_capital: float = initial_capital

        # Tracking counters
        self.breakeven_activations: int = 0
        self.scale_1_exits: int = 0
        self.scale_2_exits: int = 0

        self.signals_generated: int = 0
        self.signals_filtered: int = 0
        self.spy_gate_blocks: int = 0
        self.sma_gate_blocks: int = 0
        self.vwap_gate_blocks: int = 0
        self.first_hour_blocks: int = 0
        self.stops_hit_intraday: int = 0
        self.targets_hit_intraday: int = 0

    # ====================================================================
    # Public API
    # ====================================================================

    def run(
        self,
        intraday_data: Dict[str, pd.DataFrame],
        start_date: date,
        end_date: date,
    ) -> EnhancedBacktestResult:
        """
        Run the intraday backtest.

        Args:
            intraday_data: Dict mapping symbol -> 1-min bar DataFrame.
                           Must include 'SPY'.  Columns: open, high, low,
                           close, volume (lowercase).  DatetimeIndex in UTC.
            start_date: First trading day to simulate.
            end_date: Last trading day to simulate.

        Returns:
            EnhancedBacktestResult compatible with existing comparison code.
        """
        self._reset()

        if "SPY" not in intraday_data:
            logger.warning("SPY not found in intraday_data — required for RRS")
            return self._build_result(start_date, end_date)

        # Ensure all DataFrames have lowercase columns (do once upfront)
        for symbol in list(intraday_data.keys()):
            df = intraday_data[symbol]
            if df is not None and "close" not in df.columns:
                df.columns = [c.lower() for c in df.columns]
                intraday_data[symbol] = df

        # Pre-compute daily bars from 1-min data for all symbols
        daily_data = self._build_daily_bars(intraday_data)

        # Determine the set of trading dates from SPY's daily bars
        spy_daily = daily_data.get("SPY")
        if spy_daily is None or len(spy_daily) == 0:
            logger.warning("No daily bars derived for SPY")
            return self._build_result(start_date, end_date)

        all_dates = sorted(set(spy_daily.index.date))
        all_dates = [d for d in all_dates if start_date <= d <= end_date]

        if len(all_dates) == 0:
            logger.warning("No trading dates in range")
            return self._build_result(start_date, end_date)

        for current_date in all_dates:
            self._process_day(current_date, intraday_data, daily_data)

        # Close remaining positions at last available close
        last_date = all_dates[-1]
        for symbol in list(self.positions.keys()):
            sym_daily = daily_data.get(symbol)
            if sym_daily is not None:
                day_data = sym_daily[sym_daily.index.date == last_date]
                if len(day_data) > 0:
                    final_price = float(day_data["close"].iloc[-1])
                    self._close_position(symbol, final_price, last_date, "backtest_end")

        return self._build_result(all_dates[0], all_dates[-1])

    # ====================================================================
    # Daily bars from 1-min data
    # ====================================================================

    def _build_daily_bars(
        self, intraday_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Build daily OHLCV bars from 1-min data for indicator calculations.

        We only keep columns needed: open, high, low, close, volume.
        Index is a DatetimeIndex with one entry per trading day.
        """
        daily_data: Dict[str, pd.DataFrame] = {}
        for symbol, df in intraday_data.items():
            if df is None or len(df) == 0:
                continue
            # Ensure lowercase columns
            df_lower = df
            if "close" not in df.columns:
                df_lower = df.copy()
                df_lower.columns = [c.lower() for c in df_lower.columns]

            # Use pandas resample for vectorized aggregation
            daily = df_lower.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            }).dropna()

            if len(daily) > 0:
                daily_data[symbol] = daily
        return daily_data

    # ====================================================================
    # Day processing
    # ====================================================================

    def _process_day(
        self,
        current_date: date,
        intraday_data: Dict[str, pd.DataFrame],
        daily_data: Dict[str, pd.DataFrame],
    ):
        """Process all 1-min bars for a single trading day."""
        # Compute the UTC boundaries for this day's market hours
        market_open = _market_open_utc(current_date)
        market_close = _market_close_utc(current_date)
        scan_time_utc = market_open + timedelta(minutes=self.scan_time_minutes_after_open)

        # First-hour end: 30 min after open (RDT "avoid first 30 min" rule)
        first_hour_end_utc = market_open + timedelta(minutes=30)

        # ---- Pre-compute day-level context ----
        # VIX regime
        day_vix_regime = {"level": "normal", "pos_mult": 1.0, "rrs_adj": 0.0, "allow_longs": True}
        if self.vix_data is not None:
            vix_col = _close_col(self.vix_data)
            vix_up_to = self.vix_data[self.vix_data.index.date <= current_date]
            if len(vix_up_to) > 0:
                day_vix_regime = _get_vix_regime(float(vix_up_to[vix_col].iloc[-1]))

        # SPY daily data for gates
        spy_daily = daily_data.get("SPY")
        day_spy_gate = {"allow_longs": True, "allow_shorts": True, "spy_trend": "unknown"}
        day_regime = {"regime": "default", "rrs_threshold": 2.0, "risk_per_trade": 0.015, "wider_stops": False}
        day_sector_rs: Dict[str, float] = {}
        day_intermarket_rrs_adj = 0.0
        day_intermarket_pos_mult = 1.0

        if spy_daily is not None:
            if self.spy_gate_enabled:
                day_spy_gate = _get_spy_gate(spy_daily, current_date)
            day_regime = _detect_regime(spy_daily, current_date)
            day_sector_rs = _compute_sector_rs(self.sector_etf_data, spy_daily, current_date)
            day_intermarket_rrs_adj, day_intermarket_pos_mult = _get_intermarket_adjustment(
                self.intermarket_data, current_date
            )

        # Build a per-day context dict to avoid recalculating
        day_ctx = {
            "vix_regime": day_vix_regime,
            "spy_gate": day_spy_gate,
            "regime": day_regime,
            "sector_rs": day_sector_rs,
            "intermarket_rrs_adj": day_intermarket_rrs_adj,
            "intermarket_pos_mult": day_intermarket_pos_mult,
        }

        # Scan at 3 fixed times during the day: 30, 120, 210 min after open
        # (10:00 AM, 11:30 AM, 1:00 PM ET)
        scan_times = [
            market_open + timedelta(minutes=30),
            market_open + timedelta(minutes=120),
            market_open + timedelta(minutes=210),
        ]
        next_scan_idx = 0

        # Get SPY intraday for this day (needed for bar-by-bar)
        spy_intraday = intraday_data.get("SPY")
        if spy_intraday is None:
            return

        # Ensure lowercase columns on SPY
        if "close" not in spy_intraday.columns:
            return

        # Filter SPY bars to today's market hours
        spy_today_mask = (spy_intraday.index >= market_open) & (spy_intraday.index < market_close)
        spy_today = spy_intraday[spy_today_mask]

        if len(spy_today) == 0:
            return

        # ---- Pre-slice today's data for ALL symbols (avoids repeated index lookups) ----
        today_data: Dict[str, pd.DataFrame] = {}
        for symbol, df in intraday_data.items():
            if df is None or len(df) == 0:
                continue
            if "close" not in df.columns:
                continue
            mask = (df.index >= market_open) & (df.index < market_close)
            sym_today = df[mask]
            if len(sym_today) > 0:
                today_data[symbol] = sym_today

        # ---- Pre-compute VWAP for all symbols vectorized ----
        # For each symbol, compute running VWAP as a Series indexed by timestamp.
        # VWAP = cumsum(TP * V) / cumsum(V), where TP = (H + L + C) / 3
        day_vwaps: Dict[str, pd.Series] = {}
        for symbol, sym_today in today_data.items():
            h = sym_today["high"].values
            lo = sym_today["low"].values
            c = sym_today["close"].values
            v = sym_today["volume"].values
            tp = (h + lo + c) / 3.0
            cum_tpv = np.cumsum(tp * v)
            cum_vol = np.cumsum(v)
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                vwap_vals = np.where(cum_vol > 0, cum_tpv / cum_vol, 0.0)
            day_vwaps[symbol] = pd.Series(vwap_vals, index=sym_today.index)

        # Build a set of timestamps for each symbol's today data for O(1) membership test
        today_ts_sets: Dict[str, set] = {sym: set(df.index) for sym, df in today_data.items()}

        # ---- Build intraday_vwap dict that will be updated as we iterate ----
        # At each scan time, we need the VWAP up to that bar.
        # We pre-computed the full running VWAP, so we just look up by timestamp.
        intraday_vwap: Dict[str, float] = {}

        # Collect all unique timestamps across all symbols for this day
        # But iterate only through SPY's timestamps (all symbols should align)
        for bar_ts in spy_today.index:
            bar_time_utc = bar_ts.to_pydatetime() if hasattr(bar_ts, 'to_pydatetime') else bar_ts

            # ---- Check positions on every bar (using pre-sliced data) ----
            self._check_positions_on_bar_fast(today_data, today_ts_sets, bar_ts, current_date)

            # ---- Scan for signals at fixed times (3x/day) ----
            if next_scan_idx < len(scan_times) and bar_time_utc >= scan_times[next_scan_idx]:
                next_scan_idx += 1

                # Snapshot VWAP values at this scan time from pre-computed series
                for symbol, vwap_series in day_vwaps.items():
                    if bar_ts in today_ts_sets.get(symbol, set()):
                        # Use .loc for exact timestamp lookup from pre-computed series
                        intraday_vwap[symbol] = float(vwap_series.loc[bar_ts])

                # Check first-hour block (first 30 min after open)
                is_first_hour = bar_time_utc < first_hour_end_utc
                self._scan_for_signals(
                    current_date, intraday_data, daily_data,
                    intraday_vwap, bar_ts, day_ctx, is_first_hour
                )

        # ---- Time stop check at end of day ----
        if self.use_time_stop:
            self._check_time_stops(current_date, intraday_data, daily_data)

        # ---- Record equity at end of day ----
        position_value = 0.0
        for p in self.positions.values():
            sym_daily = daily_data.get(p.symbol)
            if sym_daily is not None:
                day_row = sym_daily[sym_daily.index.date == current_date]
                if len(day_row) > 0:
                    position_value += p.remaining_shares * float(day_row["close"].iloc[-1])

        total_equity = self.capital + position_value
        self.equity_curve.append({
            "date": current_date,
            "equity": total_equity,
            "positions": len(self.positions),
        })
        if total_equity > self.peak_capital:
            self.peak_capital = total_equity

    # ====================================================================
    # Position checking on each 1-min bar
    # ====================================================================

    def _check_positions_on_bar_fast(
        self,
        today_data: Dict[str, pd.DataFrame],
        today_ts_sets: Dict[str, set],
        bar_ts: pd.Timestamp,
        current_date: date,
    ):
        """Check stops, targets, trailing stops, and scaled exits on a single bar (optimized)."""
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            if bar_ts not in today_ts_sets.get(symbol, set()):
                continue

            bar = today_data[symbol].loc[bar_ts]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[0]

            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])

            if high == 0 or low == 0 or close == 0:
                continue

            stop_distance = abs(position.entry_price - position.original_stop)
            if stop_distance == 0:
                continue

            # Current profit for MFE/MAE tracking
            if position.direction == "long":
                current_profit = close - position.entry_price
                current_profit_r = current_profit / stop_distance
            else:
                current_profit = position.entry_price - close
                current_profit_r = current_profit / stop_distance

            # Update MFE / MAE
            if current_profit > position.max_favorable_excursion:
                position.max_favorable_excursion = current_profit
            if current_profit < -position.max_adverse_excursion:
                position.max_adverse_excursion = abs(current_profit)

            # === STOP LOSS CHECK ===
            if position.direction == "long" and low <= position.stop_price:
                self._close_position(symbol, position.stop_price, current_date, "stop_loss")
                self.stops_hit_intraday += 1
                continue
            elif position.direction == "short" and high >= position.stop_price:
                self._close_position(symbol, position.stop_price, current_date, "stop_loss")
                self.stops_hit_intraday += 1
                continue

            # === SCALED EXIT 1 ===
            if self.use_scaled_exits and not position.scale_1_hit:
                scale_1_price = self._target_at_r(position, self.scale_1_target_r)
                if position.direction == "long" and high >= scale_1_price:
                    self._scale_out(position, self.scale_1_percent, scale_1_price, current_date, "scale_1")
                    position.scale_1_hit = True
                    self.scale_1_exits += 1
                elif position.direction == "short" and low <= scale_1_price:
                    self._scale_out(position, self.scale_1_percent, scale_1_price, current_date, "scale_1")
                    position.scale_1_hit = True
                    self.scale_1_exits += 1

            # === BREAKEVEN STOP ===
            if self.use_trailing_stop and not position.breakeven_activated:
                if current_profit_r >= self.breakeven_trigger_r:
                    position.stop_price = position.entry_price
                    position.breakeven_activated = True
                    self.breakeven_activations += 1

            # === TRAILING STOP UPDATE ===
            if self.use_trailing_stop and position.breakeven_activated:
                trail_distance = position.atr_at_entry * self.trailing_atr_multiplier
                if position.direction == "long":
                    new_trail = high - trail_distance
                    if new_trail > position.stop_price:
                        position.stop_price = new_trail
                        position.trailing_stop_price = new_trail
                else:
                    new_trail = low + trail_distance
                    if new_trail < position.stop_price:
                        position.stop_price = new_trail
                        position.trailing_stop_price = new_trail

            # === SCALED EXIT 2 ===
            if self.use_scaled_exits and position.scale_1_hit and not position.scale_2_hit:
                scale_2_price = self._target_at_r(position, self.scale_2_target_r)
                if position.direction == "long" and high >= scale_2_price:
                    self._scale_out(position, self.scale_2_percent, scale_2_price, current_date, "scale_2")
                    position.scale_2_hit = True
                    self.scale_2_exits += 1
                elif position.direction == "short" and low <= scale_2_price:
                    self._scale_out(position, self.scale_2_percent, scale_2_price, current_date, "scale_2")
                    position.scale_2_hit = True
                    self.scale_2_exits += 1

            # === FULL TARGET HIT ===
            if position.direction == "long" and high >= position.target_price:
                self._close_position(symbol, position.target_price, current_date, "take_profit")
                self.targets_hit_intraday += 1
                continue
            elif position.direction == "short" and low <= position.target_price:
                self._close_position(symbol, position.target_price, current_date, "take_profit")
                self.targets_hit_intraday += 1
                continue

    def _check_positions_on_bar(
        self,
        intraday_data: Dict[str, pd.DataFrame],
        bar_ts: pd.Timestamp,
        current_date: date,
    ):
        """Check stops, targets, trailing stops, and scaled exits on a single bar."""
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            df = intraday_data.get(symbol)
            if df is None or bar_ts not in df.index:
                continue

            bar = df.loc[bar_ts]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[0]

            high = float(bar.get("high", bar.get("High", 0)))
            low = float(bar.get("low", bar.get("Low", 0)))
            close = float(bar.get("close", bar.get("Close", 0)))

            if high == 0 or low == 0 or close == 0:
                continue

            stop_distance = abs(position.entry_price - position.original_stop)
            if stop_distance == 0:
                continue

            # Current profit for MFE/MAE tracking
            if position.direction == "long":
                current_profit = close - position.entry_price
                current_profit_r = current_profit / stop_distance
            else:
                current_profit = position.entry_price - close
                current_profit_r = current_profit / stop_distance

            # Update MFE / MAE
            if current_profit > position.max_favorable_excursion:
                position.max_favorable_excursion = current_profit
            if current_profit < -position.max_adverse_excursion:
                position.max_adverse_excursion = abs(current_profit)

            # === STOP LOSS CHECK ===
            if position.direction == "long" and low <= position.stop_price:
                self._close_position(symbol, position.stop_price, current_date, "stop_loss")
                self.stops_hit_intraday += 1
                continue
            elif position.direction == "short" and high >= position.stop_price:
                self._close_position(symbol, position.stop_price, current_date, "stop_loss")
                self.stops_hit_intraday += 1
                continue

            # === SCALED EXIT 1 ===
            if self.use_scaled_exits and not position.scale_1_hit:
                scale_1_price = self._target_at_r(position, self.scale_1_target_r)
                if position.direction == "long" and high >= scale_1_price:
                    self._scale_out(position, self.scale_1_percent, scale_1_price, current_date, "scale_1")
                    position.scale_1_hit = True
                    self.scale_1_exits += 1
                elif position.direction == "short" and low <= scale_1_price:
                    self._scale_out(position, self.scale_1_percent, scale_1_price, current_date, "scale_1")
                    position.scale_1_hit = True
                    self.scale_1_exits += 1

            # === BREAKEVEN STOP ===
            if self.use_trailing_stop and not position.breakeven_activated:
                if current_profit_r >= self.breakeven_trigger_r:
                    position.stop_price = position.entry_price
                    position.breakeven_activated = True
                    self.breakeven_activations += 1

            # === TRAILING STOP UPDATE ===
            if self.use_trailing_stop and position.breakeven_activated:
                trail_distance = position.atr_at_entry * self.trailing_atr_multiplier
                if position.direction == "long":
                    new_trail = high - trail_distance
                    if new_trail > position.stop_price:
                        position.stop_price = new_trail
                        position.trailing_stop_price = new_trail
                else:
                    new_trail = low + trail_distance
                    if new_trail < position.stop_price:
                        position.stop_price = new_trail
                        position.trailing_stop_price = new_trail

            # === SCALED EXIT 2 ===
            if self.use_scaled_exits and position.scale_1_hit and not position.scale_2_hit:
                scale_2_price = self._target_at_r(position, self.scale_2_target_r)
                if position.direction == "long" and high >= scale_2_price:
                    self._scale_out(position, self.scale_2_percent, scale_2_price, current_date, "scale_2")
                    position.scale_2_hit = True
                    self.scale_2_exits += 1
                elif position.direction == "short" and low <= scale_2_price:
                    self._scale_out(position, self.scale_2_percent, scale_2_price, current_date, "scale_2")
                    position.scale_2_hit = True
                    self.scale_2_exits += 1

            # === FULL TARGET HIT ===
            if position.direction == "long" and high >= position.target_price:
                self._close_position(symbol, position.target_price, current_date, "take_profit")
                self.targets_hit_intraday += 1
                continue
            elif position.direction == "short" and low <= position.target_price:
                self._close_position(symbol, position.target_price, current_date, "take_profit")
                self.targets_hit_intraday += 1
                continue

    # ====================================================================
    # Time stops (checked once at end of day)
    # ====================================================================

    def _check_time_stops(
        self,
        current_date: date,
        intraday_data: Dict[str, pd.DataFrame],
        daily_data: Dict[str, pd.DataFrame],
    ):
        """Apply time-based stop (max holding days, stale trade)."""
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            entry_d = position.entry_date
            if isinstance(entry_d, datetime):
                entry_d = entry_d.date()
            holding_days = (current_date - entry_d).days

            # Get closing price for today
            close_price = None
            sym_daily = daily_data.get(symbol)
            if sym_daily is not None:
                day_row = sym_daily[sym_daily.index.date == current_date]
                if len(day_row) > 0:
                    close_price = float(day_row["close"].iloc[-1])

            if close_price is None:
                continue

            # Max holding days
            if holding_days >= self.max_holding_days:
                self._close_position(symbol, close_price, current_date, "time_stop_max_days")
                continue

            # Stale trade (not profitable after X days)
            stop_distance = abs(position.entry_price - position.original_stop)
            if stop_distance > 0 and holding_days >= self.stale_trade_days:
                if position.direction == "long":
                    profit_r = (close_price - position.entry_price) / stop_distance
                else:
                    profit_r = (position.entry_price - close_price) / stop_distance
                if profit_r < 0.5:
                    self._close_position(symbol, close_price, current_date, "time_stop_stale")
                    continue

    # ====================================================================
    # Signal scanning (once per day at scan_time)
    # ====================================================================

    def _scan_for_signals(
        self,
        current_date: date,
        intraday_data: Dict[str, pd.DataFrame],
        daily_data: Dict[str, pd.DataFrame],
        intraday_vwap: Dict[str, float],
        scan_bar_ts: pd.Timestamp,
        day_ctx: Dict,
        is_first_hour: bool,
    ):
        """Generate and filter signals at scan time."""
        spy_daily = daily_data.get("SPY")
        if spy_daily is None:
            return

        spy_up_to = spy_daily[spy_daily.index.date <= current_date]
        if len(spy_up_to) < 20:
            return

        spy_close = float(spy_up_to["close"].iloc[-1])
        spy_prev_close = float(spy_up_to["close"].iloc[-2])

        # Effective RRS threshold
        vix_regime = day_ctx["vix_regime"]
        regime = day_ctx["regime"]
        base_rrs_threshold = regime.get("rrs_threshold", self.rrs_threshold)
        vix_rrs_adj = vix_regime.get("rrs_adj", 0.0)
        intermarket_rrs_adj = day_ctx["intermarket_rrs_adj"]
        effective_rrs_threshold = base_rrs_threshold + vix_rrs_adj + intermarket_rrs_adj

        for symbol in intraday_data.keys():
            if symbol == "SPY":
                continue
            if symbol in self.positions:
                continue
            if len(self.positions) >= self.max_positions:
                break

            sym_daily = daily_data.get(symbol)
            if sym_daily is None:
                continue

            sym_up_to = sym_daily[sym_daily.index.date <= current_date]
            if len(sym_up_to) < 20:
                continue

            try:
                # Get current price from the scan bar
                sym_intraday = intraday_data[symbol]
                if scan_bar_ts not in sym_intraday.index:
                    continue
                scan_bar = sym_intraday.loc[scan_bar_ts]
                if isinstance(scan_bar, pd.DataFrame):
                    scan_bar = scan_bar.iloc[0]
                current_price = float(scan_bar["close"]) if "close" in scan_bar.index else float(scan_bar.get("Close", 0))
                if current_price <= 0:
                    continue

                # Previous day close
                stock_prev_close = float(sym_up_to["close"].iloc[-1])
                # If we have today's close already in daily data, use previous day
                if sym_up_to.index.date[-1] == current_date and len(sym_up_to) >= 2:
                    stock_prev_close = float(sym_up_to["close"].iloc[-2])
                elif sym_up_to.index.date[-1] != current_date:
                    stock_prev_close = float(sym_up_to["close"].iloc[-1])

                # ATR from daily data
                atr = _calculate_atr(sym_up_to)
                if atr <= 0:
                    continue

                # RRS calculation
                atr_pct = (atr / current_price) * 100
                if atr_pct <= 0:
                    continue
                stock_pc = ((current_price / stock_prev_close) - 1) * 100
                spy_pc = ((spy_close / spy_prev_close) - 1) * 100
                rrs = (stock_pc - spy_pc) / atr_pct

                if not np.isfinite(rrs):
                    continue

                # Determine direction
                direction = None
                if rrs > 0:
                    direction = "long"
                elif rrs < 0:
                    direction = "short"
                else:
                    continue

                self.signals_generated += 1

                # ---- GATE 1: First-Hour Gate ----
                if self.first_hour_block and is_first_hour:
                    self.first_hour_blocks += 1
                    self.signals_filtered += 1
                    continue

                # ---- GATE 2: SPY Hard Gate ----
                if self.spy_gate_enabled:
                    spy_gate = day_ctx["spy_gate"]
                    if direction == "long" and not spy_gate.get("allow_longs", True):
                        self.spy_gate_blocks += 1
                        self.signals_filtered += 1
                        continue
                    if direction == "short" and not spy_gate.get("allow_shorts", True):
                        self.spy_gate_blocks += 1
                        self.signals_filtered += 1
                        continue

                # ---- GATE 3: Stock 50/200 SMA Gate ----
                if self.sma_gate_enabled:
                    sma_info = _check_stock_sma_gate(sym_daily, current_date)
                    if direction == "long" and sma_info["above_sma50"] is False:
                        self.sma_gate_blocks += 1
                        self.signals_filtered += 1
                        continue
                    if direction == "short" and sma_info["above_sma50"] is True:
                        self.sma_gate_blocks += 1
                        self.signals_filtered += 1
                        continue

                # ---- GATE 4: VWAP Gate ----
                if self.vwap_gate_enabled:
                    sym_vwap = intraday_vwap.get(symbol)
                    if sym_vwap is not None and sym_vwap > 0:
                        if direction == "long" and current_price < sym_vwap:
                            self.vwap_gate_blocks += 1
                            self.signals_filtered += 1
                            continue
                        if direction == "short" and current_price > sym_vwap:
                            self.vwap_gate_blocks += 1
                            self.signals_filtered += 1
                            continue

                # ---- Sector boost ----
                sector_boost = _get_sector_boost(symbol, direction, day_ctx["sector_rs"])
                adjusted_rrs = abs(rrs) + sector_boost

                # ---- RRS threshold check ----
                if adjusted_rrs < effective_rrs_threshold:
                    self.signals_filtered += 1
                    continue

                # ---- VIX extreme block ----
                if direction == "long" and not vix_regime.get("allow_longs", True):
                    self.signals_filtered += 1
                    continue

                # ---- Daily chart confirmation ----
                confirmed = False
                if direction == "long" and _check_daily_strength(sym_up_to):
                    confirmed = True
                elif direction == "short" and _check_daily_weakness(sym_up_to):
                    confirmed = True

                if not confirmed:
                    self.signals_filtered += 1
                    continue

                # ---- Enter position ----
                regime_risk = regime.get("risk_per_trade", 0.015)
                position_mult = vix_regime.get("pos_mult", 1.0)
                position_mult *= day_ctx["intermarket_pos_mult"]

                stop_mult = self.stop_atr_multiplier
                if regime.get("wider_stops", False):
                    stop_mult = max(stop_mult, 2.0)

                self._enter_position(
                    symbol=symbol,
                    direction=direction,
                    entry_price=current_price,
                    atr=atr,
                    entry_date=current_date,
                    rrs=rrs,
                    position_mult=position_mult,
                    risk_per_trade=regime_risk,
                    stop_multiplier=stop_mult,
                )

            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
                continue

    # ====================================================================
    # Position entry / exit / scaling
    # ====================================================================

    def _enter_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        atr: float,
        entry_date: date,
        rrs: float,
        position_mult: float = 1.0,
        risk_per_trade: float = 0.015,
        stop_multiplier: float = 1.5,
    ):
        """Enter a new position with regime-adjusted sizing."""
        temp_limits = RiskLimits(
            max_risk_per_trade=risk_per_trade,
            max_open_positions=self.max_positions,
        )
        temp_sizer = PositionSizer(temp_limits)

        sizing = temp_sizer.calculate_position_size(
            account_size=self.capital,
            entry_price=entry_price,
            atr=atr,
            direction=direction,
            stop_multiplier=stop_multiplier,
            target_multiplier=self.target_atr_multiplier,
        )

        if sizing.shares == 0:
            return

        adjusted_shares = max(1, int(sizing.shares * position_mult))
        required = adjusted_shares * entry_price
        if required > self.capital:
            return

        trade = EnhancedTrade(
            symbol=symbol,
            direction=direction,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=adjusted_shares,
            remaining_shares=adjusted_shares,
            stop_price=sizing.stop_price,
            original_stop=sizing.stop_price,
            target_price=sizing.target_price,
            rrs_at_entry=rrs,
            atr_at_entry=atr,
        )

        self.positions[symbol] = trade
        self.capital -= required

    def _close_position(
        self, symbol: str, exit_price: float, exit_date: date, reason: str
    ):
        """Close a position completely."""
        if symbol not in self.positions:
            return

        trade = self.positions[symbol]
        trade.exit_date = exit_date
        trade.exit_price = exit_price
        trade.exit_reason = reason

        if trade.direction == "long":
            remaining_pnl = (exit_price - trade.entry_price) * trade.remaining_shares
        else:
            remaining_pnl = (trade.entry_price - exit_price) * trade.remaining_shares

        trade.pnl += remaining_pnl
        trade.pnl_percent = (trade.pnl / (trade.entry_price * trade.shares)) * 100

        if isinstance(trade.entry_date, datetime):
            trade.holding_days = (exit_date - trade.entry_date.date()).days
        else:
            trade.holding_days = (exit_date - trade.entry_date).days

        self.capital += (trade.entry_price * trade.remaining_shares) + remaining_pnl

        self.trades.append(trade)
        del self.positions[symbol]

    def _scale_out(
        self,
        position: EnhancedTrade,
        scale_percent: float,
        exit_price: float,
        exit_date: date,
        reason: str,
    ):
        """Scale out of a portion of the position."""
        shares_to_sell = int(position.remaining_shares * scale_percent)
        if shares_to_sell == 0:
            shares_to_sell = 1
        if shares_to_sell >= position.remaining_shares:
            return

        if position.direction == "long":
            pnl = (exit_price - position.entry_price) * shares_to_sell
        else:
            pnl = (position.entry_price - exit_price) * shares_to_sell

        self.capital += (position.entry_price * shares_to_sell) + pnl
        position.remaining_shares -= shares_to_sell
        position.pnl += pnl

    def _target_at_r(self, position: EnhancedTrade, r_multiple: float) -> float:
        """Calculate target price for a given R multiple."""
        stop_dist = abs(position.entry_price - position.original_stop)
        target_dist = stop_dist * r_multiple
        if position.direction == "long":
            return position.entry_price + target_dist
        return position.entry_price - target_dist

    # ====================================================================
    # VWAP (used in scan, calculated incrementally in _process_day)
    # ====================================================================

    @staticmethod
    def _calculate_intraday_vwap(bars: pd.DataFrame) -> float:
        """
        Calculate VWAP from a DataFrame of 1-min bars (all bars so far today).

        Returns current VWAP value.
        """
        if len(bars) == 0:
            return 0.0

        h = bars.get("high", bars.get("High", pd.Series(dtype=float)))
        lo = bars.get("low", bars.get("Low", pd.Series(dtype=float)))
        c = bars.get("close", bars.get("Close", pd.Series(dtype=float)))
        v = bars.get("volume", bars.get("Volume", pd.Series(dtype=float)))

        tp = (h + lo + c) / 3.0
        cum_tpv = (tp * v).sum()
        cum_vol = v.sum()
        if cum_vol > 0:
            return float(cum_tpv / cum_vol)
        return 0.0

    # ====================================================================
    # State management
    # ====================================================================

    def _reset(self):
        """Reset all state for a fresh run."""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.peak_capital = self.initial_capital
        self.breakeven_activations = 0
        self.scale_1_exits = 0
        self.scale_2_exits = 0
        self.signals_generated = 0
        self.signals_filtered = 0
        self.spy_gate_blocks = 0
        self.sma_gate_blocks = 0
        self.vwap_gate_blocks = 0
        self.first_hour_blocks = 0
        self.stops_hit_intraday = 0
        self.targets_hit_intraday = 0

    # ====================================================================
    # Result construction
    # ====================================================================

    def _build_result(self, start_date: date, end_date: date) -> EnhancedBacktestResult:
        """Build EnhancedBacktestResult from completed trades."""
        total_trades = len(self.trades)

        if total_trades == 0:
            return EnhancedBacktestResult(
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
                final_capital=self.capital,
                total_return=0,
                total_return_pct=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                avg_holding_days=0,
                trades=self.trades,
                equity_curve=self.equity_curve,
            )

        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]

        total_wins = sum(t.pnl for t in winners)
        total_losses = abs(sum(t.pnl for t in losers))

        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        # Exit-reason counts
        trades_stopped_out = len([t for t in self.trades if "stop_loss" in t.exit_reason])
        trades_target_hit = len([t for t in self.trades if "take_profit" in t.exit_reason])
        trades_trailing_stopped = len([
            t for t in self.trades
            if t.trailing_stop_price > 0 and "stop" in t.exit_reason
        ])
        trades_time_stopped = len([t for t in self.trades if "time_stop" in t.exit_reason])

        # MFE / MAE
        avg_mfe = sum(t.max_favorable_excursion for t in self.trades) / total_trades
        avg_mae = sum(t.max_adverse_excursion for t in self.trades) / total_trades

        # Max drawdown
        max_dd = 0.0
        peak = self.initial_capital
        for point in self.equity_curve:
            if point["equity"] > peak:
                peak = point["equity"]
            dd = peak - point["equity"]
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio
        returns = [t.pnl_percent for t in self.trades]
        avg_return = sum(returns) / len(returns) if returns else 0
        std_return = (
            (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            if len(returns) > 1 else 0
        )
        sharpe = (avg_return / std_return) if std_return > 0 else 0

        return EnhancedBacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            total_return=self.capital - self.initial_capital,
            total_return_pct=((self.capital - self.initial_capital) / self.initial_capital) * 100,
            total_trades=total_trades,
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=len(winners) / total_trades if total_trades > 0 else 0,
            avg_win=total_wins / len(winners) if winners else 0,
            avg_loss=total_losses / len(losers) if losers else 0,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            max_drawdown_pct=(max_dd / self.peak_capital) * 100 if self.peak_capital > 0 else 0,
            sharpe_ratio=sharpe,
            avg_holding_days=sum(t.holding_days for t in self.trades) / total_trades,
            trades_stopped_out=trades_stopped_out,
            trades_target_hit=trades_target_hit,
            trades_trailing_stopped=trades_trailing_stopped,
            trades_time_stopped=trades_time_stopped,
            breakeven_activations=self.breakeven_activations,
            scale_1_exits=self.scale_1_exits,
            scale_2_exits=self.scale_2_exits,
            avg_mfe=avg_mfe,
            avg_mae=avg_mae,
            trades=self.trades,
            equity_curve=self.equity_curve,
        )
