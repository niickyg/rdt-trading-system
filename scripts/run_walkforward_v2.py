#!/usr/bin/env python3
"""
Walk-Forward Backtest V2: Baseline vs Old Filters vs RDT-Aligned Filters (2 Years)

Three configurations compared:
  A) Baseline     — Config C engine, static RRS 2.0, no filters
  B) Old Filters  — VIX regime + sector RS + regime-adaptive params
  C) RDT Filters  — Old Filters + SPY hard gate + 50/200 SMA gate + lightweight MTF

Uses 2 years of daily data with 6 walk-forward windows (~4 months each).

Note: VWAP filter cannot be simulated with daily bars (intraday only).
      First-hour filter also cannot be simulated with daily bars.

Usage:
    cd /home/user0/rdt-trading-system
    python scripts/run_walkforward_v2.py
"""

import sys
import os
import time
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np
import pandas as pd

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

# Configure logging - WARNING level for clean output
logger.remove()
logger.add(sys.stderr, level="WARNING")

from backtesting.engine_enhanced import EnhancedBacktestEngine, EnhancedBacktestResult
from backtesting.data_loader import DataLoader
from risk.models import RiskLimits
from risk.risk_manager import SECTOR_MAP
from scanner.sector_filter import SECTOR_ETF_MAP


# ============================================================================
# Configuration
# ============================================================================

INITIAL_CAPITAL = 25000.0
DATA_DAYS = 730  # 2 years

WATCHLIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    'JPM', 'V', 'JNJ', 'UNH', 'HD', 'PG', 'MA', 'DIS',
    'PYPL', 'ADBE', 'CRM', 'NFLX', 'INTC', 'AMD', 'CSCO',
    'PEP', 'KO', 'MRK', 'ABT', 'TMO', 'COST', 'AVGO', 'TXN'
]

SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLC', 'XLU']

# 6 walk-forward windows across 2 years (~4 months each, skip first 90 days for warmup)
WALK_FORWARD_WINDOWS = [
    {"name": "Window 1 (Q2 2024)", "start_day": 91,  "end_day": 175},
    {"name": "Window 2 (Q3 2024)", "start_day": 176, "end_day": 260},
    {"name": "Window 3 (Q4 2024)", "start_day": 261, "end_day": 345},
    {"name": "Window 4 (Q1 2025)", "start_day": 346, "end_day": 430},
    {"name": "Window 5 (Q2 2025)", "start_day": 431, "end_day": 505},
    {"name": "Window 6 (Recent)",  "start_day": 506, "end_day": 600},
]

# Enhanced engine base config (Config C: scaled exits, 1.5x stop, 2.0x target)
BASE_ENGINE_CONFIG = {
    "rrs_threshold": 2.0,
    "stop_multiplier": 1.5,
    "target_multiplier": 2.0,
    "max_risk_per_trade": 0.015,
    "max_positions": 8,
    "use_trailing_stop": True,
    "breakeven_trigger_r": 1.0,
    "trailing_atr_multiplier": 1.0,
    "use_scaled_exits": True,
    "scale_1_target_r": 1.0,
    "scale_1_percent": 0.5,
    "scale_2_target_r": 1.5,
    "scale_2_percent": 0.25,
    "use_time_stop": True,
    "max_holding_days": 12,
    "stale_trade_days": 6,
}


# ============================================================================
# VIX Regime Logic (mirrors scanner/vix_filter.py)
# ============================================================================

def get_vix_regime_for_value(vix_value: float) -> Dict:
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


# ============================================================================
# Sector RS Calculation (mirrors scanner/sector_filter.py)
# ============================================================================

def compute_sector_rs_on_date(
    sector_etf_data: Dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
    current_date: date,
    lookback: int = 5,
) -> Dict[str, float]:
    close_col = 'Close' if 'Close' in spy_data.columns else 'close'
    spy_up_to = spy_data[spy_data.index.date <= current_date]
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
        etf_col = 'Close' if 'Close' in etf_df.columns else 'close'
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


def get_sector_boost(symbol: str, direction: str, sector_rs_map: Dict[str, float]) -> float:
    sector = SECTOR_MAP.get(symbol, 'other')
    if sector == 'other' or sector not in sector_rs_map:
        return 0.0
    rs = sector_rs_map[sector]
    if direction == 'long':
        if rs > 0:
            return 0.25
        elif rs < -1.0:
            return -0.50
    elif direction == 'short':
        if rs < -1.0:
            return 0.25
        elif rs > 0:
            return -0.50
    return 0.0


# ============================================================================
# Regime Detection (mirrors scanner/regime_params.py)
# ============================================================================

def detect_regime_on_date(spy_data: pd.DataFrame, current_date: date) -> Dict:
    close_col = 'Close' if 'Close' in spy_data.columns else 'close'
    high_col = 'High' if 'High' in spy_data.columns else 'high'
    low_col = 'Low' if 'Low' in spy_data.columns else 'low'

    spy_up_to = spy_data[spy_data.index.date <= current_date]
    if len(spy_up_to) < 200:
        return {"regime": "default", "rrs_threshold": 2.0, "risk_per_trade": 0.015, "wider_stops": False}

    close_series = spy_up_to[close_col]
    current_price = float(close_series.iloc[-1])

    ema_50 = float(close_series.ewm(span=50, adjust=False).mean().iloc[-1])
    ema_200 = float(close_series.ewm(span=200, adjust=False).mean().iloc[-1])

    high_s = spy_up_to[high_col].astype(float)
    low_s = spy_up_to[low_col].astype(float)
    close_s = close_series.astype(float)
    tr = pd.concat([
        high_s - low_s,
        (high_s - close_s.shift(1)).abs(),
        (low_s - close_s.shift(1)).abs()
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


# ============================================================================
# NEW: SPY Hard Gate (mirrors scanner logic)
# ============================================================================

def get_spy_gate_decision(spy_data: pd.DataFrame, current_date: date) -> Dict:
    """
    SPY Hard Gate: determine if longs/shorts are allowed based on SPY 50/200 EMA.
    Returns {allow_longs: bool, allow_shorts: bool, spy_trend: str}
    """
    close_col = 'Close' if 'Close' in spy_data.columns else 'close'
    spy_up_to = spy_data[spy_data.index.date <= current_date]
    if len(spy_up_to) < 200:
        return {"allow_longs": True, "allow_shorts": True, "spy_trend": "unknown"}

    close_series = spy_up_to[close_col]
    current_price = float(close_series.iloc[-1])
    ema_50 = float(close_series.ewm(span=50, adjust=False).mean().iloc[-1])
    ema_200 = float(close_series.ewm(span=200, adjust=False).mean().iloc[-1])

    above_50 = current_price > ema_50
    above_200 = current_price > ema_200

    if above_50 and above_200:
        # Bullish: allow longs, block shorts
        return {"allow_longs": True, "allow_shorts": False, "spy_trend": "bullish"}
    elif not above_50 and not above_200:
        # Bearish: block longs, allow shorts
        return {"allow_longs": False, "allow_shorts": True, "spy_trend": "bearish"}
    else:
        # Mixed: allow both with caution
        return {"allow_longs": True, "allow_shorts": True, "spy_trend": "mixed"}


# ============================================================================
# NEW: Stock 50/200 SMA Gate (mirrors scanner logic)
# ============================================================================

def check_stock_sma(stock_data: pd.DataFrame, current_date: date) -> Dict:
    """
    Check if stock price is above/below 50 SMA and 200 SMA.
    For longs: must be above 50 SMA. For shorts: must be below 50 SMA.
    """
    close_col = 'Close' if 'Close' in stock_data.columns else 'close'
    data_up_to = stock_data[stock_data.index.date <= current_date]

    if len(data_up_to) < 50:
        return {"above_sma50": None, "above_sma200": None, "filter": "unknown"}

    close_series = data_up_to[close_col].astype(float)
    current_price = float(close_series.iloc[-1])
    sma_50 = float(close_series.rolling(50).mean().iloc[-1])

    above_50 = current_price > sma_50

    above_200 = None
    if len(data_up_to) >= 200:
        sma_200 = float(close_series.rolling(200).mean().iloc[-1])
        above_200 = current_price > sma_200

    if above_50 and (above_200 is None or above_200):
        return {"above_sma50": True, "above_sma200": above_200, "filter": "strong"}
    elif above_50:
        return {"above_sma50": True, "above_sma200": False, "filter": "moderate"}
    elif above_200 is not None and not above_200:
        return {"above_sma50": False, "above_sma200": False, "filter": "counter_trend"}
    else:
        return {"above_sma50": False, "above_sma200": above_200, "filter": "weak"}


# ============================================================================
# NEW: Lightweight MTF Alignment (from daily data — weekly + monthly proxy)
# ============================================================================

def check_mtf_alignment_daily(stock_data: pd.DataFrame, current_date: date, direction: str) -> Dict:
    """
    Multi-timeframe alignment from daily bars.
    Since we can't resample 5m data in backtests, we check:
      1. Daily: price vs 8 EMA
      2. Weekly (resampled): price vs 8 EMA
      3. Monthly (resampled): price vs 8 EMA
    Returns alignment score (0-3) and classification.
    """
    close_col = 'Close' if 'Close' in stock_data.columns else 'close'
    high_col = 'High' if 'High' in stock_data.columns else 'high'
    low_col = 'Low' if 'Low' in stock_data.columns else 'low'
    open_col = 'Open' if 'Open' in stock_data.columns else 'open'

    data_up_to = stock_data[stock_data.index.date <= current_date].copy()
    if len(data_up_to) < 30:
        return {"mtf_score": 0, "mtf_alignment": "unknown", "details": {}}

    aligned_count = 0
    details = {}

    # 1. Daily: price vs 8 EMA
    daily_close = data_up_to[close_col].astype(float)
    daily_ema8 = daily_close.ewm(span=8, adjust=False).mean()
    price = float(daily_close.iloc[-1])
    ema_val = float(daily_ema8.iloc[-1])
    if direction == "long":
        daily_aligned = price > ema_val
    else:
        daily_aligned = price < ema_val
    if daily_aligned:
        aligned_count += 1
    details["daily"] = {"aligned": daily_aligned, "price": price, "ema8": round(ema_val, 2)}

    # 2. Weekly (resample daily to weekly)
    try:
        weekly = data_up_to.resample('W').agg({
            open_col: 'first', high_col: 'max', low_col: 'min', close_col: 'last'
        }).dropna()
        if len(weekly) >= 8:
            weekly_close = weekly[close_col].astype(float)
            weekly_ema8 = weekly_close.ewm(span=8, adjust=False).mean()
            w_price = float(weekly_close.iloc[-1])
            w_ema = float(weekly_ema8.iloc[-1])
            if direction == "long":
                w_aligned = w_price > w_ema
            else:
                w_aligned = w_price < w_ema
            if w_aligned:
                aligned_count += 1
            details["weekly"] = {"aligned": w_aligned, "price": w_price, "ema8": round(w_ema, 2)}
        else:
            details["weekly"] = {"aligned": None, "reason": "insufficient_data"}
    except Exception:
        details["weekly"] = {"aligned": None, "reason": "error"}

    # 3. Monthly (resample daily to monthly)
    try:
        monthly = data_up_to.resample('ME').agg({
            open_col: 'first', high_col: 'max', low_col: 'min', close_col: 'last'
        }).dropna()
        if len(monthly) >= 8:
            monthly_close = monthly[close_col].astype(float)
            monthly_ema8 = monthly_close.ewm(span=8, adjust=False).mean()
            m_price = float(monthly_close.iloc[-1])
            m_ema = float(monthly_ema8.iloc[-1])
            if direction == "long":
                m_aligned = m_price > m_ema
            else:
                m_aligned = m_price < m_ema
            if m_aligned:
                aligned_count += 1
            details["monthly"] = {"aligned": m_aligned, "price": m_price, "ema8": round(m_ema, 2)}
        else:
            details["monthly"] = {"aligned": None, "reason": "insufficient_data"}
    except Exception:
        details["monthly"] = {"aligned": None, "reason": "error"}

    # Classification
    if aligned_count >= 3:
        alignment = "strong"
    elif aligned_count >= 2:
        alignment = "moderate"
    else:
        alignment = "weak"

    return {"mtf_score": aligned_count, "mtf_alignment": alignment, "details": details}


# ============================================================================
# Old Filtered Engine (VIX + Sector + Regime only — same as V1)
# ============================================================================

class OldFilteredEngine(EnhancedBacktestEngine):
    """VIX regime + sector RS + regime-adaptive params (no SPY gate, no SMA, no MTF)."""

    def __init__(self, *args, **kwargs):
        self.vix_data: Optional[pd.DataFrame] = kwargs.pop('vix_data', None)
        self.sector_etf_data: Dict[str, pd.DataFrame] = kwargs.pop('sector_etf_data', {})
        super().__init__(*args, **kwargs)
        self.signals_generated = 0
        self.signals_filtered_out = 0
        self.vix_blocks = 0
        self.sector_adjustments = 0
        self.regime_adjustments = 0
        self.high_vix_days: List[date] = []
        self._day_vix_regime: Dict = {}
        self._day_sector_rs: Dict[str, float] = {}
        self._day_market_regime: Dict = {}
        self._day_position_mult: float = 1.0

    def _process_day(self, current_date, stock_data, spy_data):
        if self.vix_data is not None:
            vix_col = 'Close' if 'Close' in self.vix_data.columns else 'close'
            vix_up_to = self.vix_data[self.vix_data.index.date <= current_date]
            if len(vix_up_to) > 0:
                vix_val = float(vix_up_to[vix_col].iloc[-1])
                self._day_vix_regime = get_vix_regime_for_value(vix_val)
                if vix_val >= 25:
                    self.high_vix_days.append(current_date)
            else:
                self._day_vix_regime = {"level": "normal", "pos_mult": 1.0, "rrs_adj": 0.0, "allow_longs": True}
        else:
            self._day_vix_regime = {"level": "normal", "pos_mult": 1.0, "rrs_adj": 0.0, "allow_longs": True}

        self._day_sector_rs = compute_sector_rs_on_date(self.sector_etf_data, spy_data, current_date)
        self._day_market_regime = detect_regime_on_date(spy_data, current_date)
        self._day_position_mult = self._day_vix_regime.get("pos_mult", 1.0)
        super()._process_day(current_date, stock_data, spy_data)

    def _scan_for_signals(self, current_date, stock_data, spy_data):
        from shared.indicators.rrs import RRSCalculator, check_daily_strength_relaxed, check_daily_weakness_relaxed

        spy_current = spy_data[spy_data.index.date <= current_date]
        if len(spy_current) < 20:
            return

        close_col = 'Close' if 'Close' in spy_current.columns else 'close'
        spy_close = spy_current[close_col].iloc[-1]
        spy_prev_close = spy_current[close_col].iloc[-2]

        base_rrs_threshold = self._day_market_regime.get("rrs_threshold", self.rrs_threshold)
        vix_rrs_adj = self._day_vix_regime.get("rrs_adj", 0.0)
        effective_rrs_threshold = base_rrs_threshold + vix_rrs_adj

        for symbol, data in stock_data.items():
            if symbol in self.positions:
                continue
            if len(self.positions) >= self.max_positions:
                break

            current_data = data[data.index.date <= current_date]
            if len(current_data) < 20:
                continue

            try:
                current_data_lower = current_data.copy()
                current_data_lower.columns = [c.lower() for c in current_data_lower.columns]

                atr = self.rrs_calculator.calculate_atr(current_data_lower).iloc[-1]
                stock_close = current_data_lower['close'].iloc[-1]
                stock_prev_close = current_data_lower['close'].iloc[-2]

                rrs_result = self.rrs_calculator.calculate_rrs_current(
                    stock_data={"current_price": stock_close, "previous_close": stock_prev_close},
                    spy_data={"current_price": spy_close, "previous_close": spy_prev_close},
                    stock_atr=atr
                )
                rrs = rrs_result["rrs"]

                direction = None
                if rrs > 0:
                    direction = "long"
                elif rrs < 0:
                    direction = "short"
                else:
                    continue

                self.signals_generated += 1

                sector_boost = get_sector_boost(symbol, direction, self._day_sector_rs)
                if sector_boost != 0:
                    self.sector_adjustments += 1
                adjusted_rrs = abs(rrs) + sector_boost

                if adjusted_rrs < effective_rrs_threshold:
                    self.signals_filtered_out += 1
                    continue

                if direction == "long" and not self._day_vix_regime.get("allow_longs", True):
                    self.vix_blocks += 1
                    self.signals_filtered_out += 1
                    continue

                daily_strength = check_daily_strength_relaxed(current_data_lower)
                daily_weakness = check_daily_weakness_relaxed(current_data_lower)

                confirmed = False
                if direction == "long" and rrs > 0 and daily_strength["is_strong"]:
                    confirmed = True
                elif direction == "short" and rrs < 0 and daily_weakness["is_weak"]:
                    confirmed = True

                if not confirmed:
                    self.signals_filtered_out += 1
                    continue

                regime_risk = self._day_market_regime.get("risk_per_trade", 0.015)
                if regime_risk != self.risk_limits.max_risk_per_trade:
                    self.regime_adjustments += 1

                self._enter_position_filtered(
                    symbol=symbol, direction=direction, entry_price=stock_close,
                    atr=atr, entry_date=current_date, rrs=rrs,
                    position_mult=self._day_position_mult, risk_per_trade=regime_risk,
                )
            except Exception:
                continue

    def _enter_position_filtered(self, symbol, direction, entry_price, atr, entry_date, rrs, position_mult, risk_per_trade):
        from backtesting.engine_enhanced import EnhancedTrade
        temp_limits = RiskLimits(max_risk_per_trade=risk_per_trade, max_open_positions=self.max_positions)
        from risk import PositionSizer
        temp_sizer = PositionSizer(temp_limits)

        stop_mult = self.stop_atr_multiplier
        if self._day_market_regime.get("wider_stops", False):
            stop_mult = max(stop_mult, 2.0)

        sizing = temp_sizer.calculate_position_size(
            account_size=self.capital, entry_price=entry_price, atr=atr,
            direction=direction, stop_multiplier=stop_mult, target_multiplier=self.target_atr_multiplier,
        )
        if sizing.shares == 0:
            return

        adjusted_shares = max(1, int(sizing.shares * position_mult))
        required = adjusted_shares * entry_price
        if required > self.capital:
            return

        trade = EnhancedTrade(
            symbol=symbol, direction=direction, entry_date=entry_date,
            entry_price=entry_price, shares=adjusted_shares, remaining_shares=adjusted_shares,
            stop_price=sizing.stop_price, original_stop=sizing.stop_price,
            target_price=sizing.target_price, rrs_at_entry=rrs, atr_at_entry=atr,
        )
        self.positions[symbol] = trade
        self.capital -= required


# ============================================================================
# NEW: RDT-Aligned Filtered Engine (Old + SPY Gate + SMA + MTF)
# ============================================================================

class RDTFilteredEngine(OldFilteredEngine):
    """
    Full RDT-aligned filters: everything from OldFilteredEngine PLUS:
      - SPY Hard Gate (block longs in bearish market, shorts in bullish)
      - Stock 50/200 SMA gate (longs must be above 50 SMA)
      - Lightweight MTF alignment (daily + weekly + monthly must agree)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spy_gate_blocks = 0
        self.sma_gate_blocks = 0
        self.mtf_gate_blocks = 0
        self._day_spy_gate: Dict = {}

    def _process_day(self, current_date, stock_data, spy_data):
        # Compute SPY gate for the day
        self._day_spy_gate = get_spy_gate_decision(spy_data, current_date)
        super()._process_day(current_date, stock_data, spy_data)

    def _scan_for_signals(self, current_date, stock_data, spy_data):
        from shared.indicators.rrs import RRSCalculator, check_daily_strength_relaxed, check_daily_weakness_relaxed

        spy_current = spy_data[spy_data.index.date <= current_date]
        if len(spy_current) < 20:
            return

        close_col = 'Close' if 'Close' in spy_current.columns else 'close'
        spy_close = spy_current[close_col].iloc[-1]
        spy_prev_close = spy_current[close_col].iloc[-2]

        base_rrs_threshold = self._day_market_regime.get("rrs_threshold", self.rrs_threshold)
        vix_rrs_adj = self._day_vix_regime.get("rrs_adj", 0.0)
        effective_rrs_threshold = base_rrs_threshold + vix_rrs_adj

        for symbol, data in stock_data.items():
            if symbol in self.positions:
                continue
            if len(self.positions) >= self.max_positions:
                break

            current_data = data[data.index.date <= current_date]
            if len(current_data) < 20:
                continue

            try:
                current_data_lower = current_data.copy()
                current_data_lower.columns = [c.lower() for c in current_data_lower.columns]

                atr = self.rrs_calculator.calculate_atr(current_data_lower).iloc[-1]
                stock_close = current_data_lower['close'].iloc[-1]
                stock_prev_close = current_data_lower['close'].iloc[-2]

                rrs_result = self.rrs_calculator.calculate_rrs_current(
                    stock_data={"current_price": stock_close, "previous_close": stock_prev_close},
                    spy_data={"current_price": spy_close, "previous_close": spy_prev_close},
                    stock_atr=atr
                )
                rrs = rrs_result["rrs"]

                direction = None
                if rrs > 0:
                    direction = "long"
                elif rrs < 0:
                    direction = "short"
                else:
                    continue

                self.signals_generated += 1

                # --- SPY HARD GATE ---
                if direction == "long" and not self._day_spy_gate.get("allow_longs", True):
                    self.spy_gate_blocks += 1
                    self.signals_filtered_out += 1
                    continue
                if direction == "short" and not self._day_spy_gate.get("allow_shorts", True):
                    self.spy_gate_blocks += 1
                    self.signals_filtered_out += 1
                    continue

                # --- STOCK 50/200 SMA GATE ---
                sma_info = check_stock_sma(current_data, current_date)
                if direction == "long" and sma_info["above_sma50"] is False:
                    self.sma_gate_blocks += 1
                    self.signals_filtered_out += 1
                    continue
                if direction == "short" and sma_info["above_sma50"] is True:
                    self.sma_gate_blocks += 1
                    self.signals_filtered_out += 1
                    continue

                # --- LIGHTWEIGHT MTF GATE ---
                mtf_info = check_mtf_alignment_daily(current_data, current_date, direction)
                if mtf_info["mtf_alignment"] == "weak":
                    self.mtf_gate_blocks += 1
                    self.signals_filtered_out += 1
                    continue

                # --- Old filters: Sector boost + VIX + Regime ---
                sector_boost = get_sector_boost(symbol, direction, self._day_sector_rs)
                if sector_boost != 0:
                    self.sector_adjustments += 1
                adjusted_rrs = abs(rrs) + sector_boost

                if adjusted_rrs < effective_rrs_threshold:
                    self.signals_filtered_out += 1
                    continue

                if direction == "long" and not self._day_vix_regime.get("allow_longs", True):
                    self.vix_blocks += 1
                    self.signals_filtered_out += 1
                    continue

                # Daily chart confirmation
                daily_strength = check_daily_strength_relaxed(current_data_lower)
                daily_weakness = check_daily_weakness_relaxed(current_data_lower)

                confirmed = False
                if direction == "long" and rrs > 0 and daily_strength["is_strong"]:
                    confirmed = True
                elif direction == "short" and rrs < 0 and daily_weakness["is_weak"]:
                    confirmed = True

                if not confirmed:
                    self.signals_filtered_out += 1
                    continue

                regime_risk = self._day_market_regime.get("risk_per_trade", 0.015)
                if regime_risk != self.risk_limits.max_risk_per_trade:
                    self.regime_adjustments += 1

                self._enter_position_filtered(
                    symbol=symbol, direction=direction, entry_price=stock_close,
                    atr=atr, entry_date=current_date, rrs=rrs,
                    position_mult=self._day_position_mult, risk_per_trade=regime_risk,
                )
            except Exception:
                continue


# ============================================================================
# Data Loading
# ============================================================================

def load_all_data() -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame], Dict[str, pd.DataFrame]]:
    cache_dir = str(PROJECT_ROOT / "data" / "backtest_cache")
    loader = DataLoader(cache_dir=cache_dir)

    end_date = date.today()
    start_date = end_date - timedelta(days=DATA_DAYS + 250)  # extra 250 days for 200 SMA warmup

    print(f"Loading data for {len(WATCHLIST)} stocks + SPY + {len(SECTOR_ETFS)} sector ETFs + VIX...")
    print(f"  Date range: {start_date} to {end_date} ({DATA_DAYS} days + 250-day warmup)")
    print()

    # SPY
    print("  Downloading SPY...")
    spy_data = loader.load_spy_data(start_date, end_date, use_cache=True)
    print(f"  SPY: {len(spy_data)} trading days")

    # VIX
    print("  Downloading ^VIX...")
    vix_data = None
    try:
        vix_data = loader._load_symbol("^VIX", start_date, end_date, use_cache=True)
        if vix_data is not None:
            print(f"  ^VIX: {len(vix_data)} trading days")
        else:
            print("  WARNING: ^VIX returned no data")
    except Exception as e:
        print(f"  WARNING: Failed to load ^VIX: {e}")

    # Sector ETFs
    print(f"  Downloading {len(SECTOR_ETFS)} sector ETFs...")
    sector_etf_data = {}
    for etf in SECTOR_ETFS:
        try:
            df = loader._load_symbol(etf, start_date, end_date, use_cache=True)
            if df is not None and len(df) > 20:
                sector_etf_data[etf] = df
        except Exception:
            pass
        time.sleep(0.3)
    print(f"  Sector ETFs loaded: {len(sector_etf_data)}/{len(SECTOR_ETFS)}")

    # Stocks
    print(f"  Downloading {len(WATCHLIST)} stocks...")
    stock_data = {}
    for i, symbol in enumerate(WATCHLIST):
        try:
            df = loader._load_symbol(symbol, start_date, end_date, use_cache=True)
            if df is not None and len(df) > 20:
                stock_data[symbol] = df
            if (i + 1) % 10 == 0:
                print(f"    {i + 1}/{len(WATCHLIST)} loaded...")
        except Exception:
            pass
        if i > 0 and i % 5 == 0:
            time.sleep(0.5)

    print(f"  Stocks loaded: {len(stock_data)}/{len(WATCHLIST)}")
    print()

    return stock_data, spy_data, vix_data, sector_etf_data


# ============================================================================
# Engine Runners
# ============================================================================

@dataclass
class WindowResult:
    window_name: str
    start_date: date
    end_date: date
    result: EnhancedBacktestResult
    signals_generated: int = 0
    signals_filtered: int = 0
    high_vix_days: int = 0
    filter_detail: str = ""


def make_engine_kwargs(config: Dict) -> Dict:
    """Build common kwargs for all engine types."""
    risk_limits = RiskLimits(
        max_risk_per_trade=config["max_risk_per_trade"],
        max_open_positions=config["max_positions"],
    )
    return {
        "initial_capital": INITIAL_CAPITAL,
        "risk_limits": risk_limits,
        "rrs_threshold": config["rrs_threshold"],
        "max_positions": config["max_positions"],
        "use_relaxed_criteria": True,
        "stop_atr_multiplier": config["stop_multiplier"],
        "target_atr_multiplier": config["target_multiplier"],
        "use_trailing_stop": config["use_trailing_stop"],
        "breakeven_trigger_r": config["breakeven_trigger_r"],
        "trailing_atr_multiplier": config["trailing_atr_multiplier"],
        "use_scaled_exits": config["use_scaled_exits"],
        "scale_1_target_r": config["scale_1_target_r"],
        "scale_1_percent": config["scale_1_percent"],
        "scale_2_target_r": config["scale_2_target_r"],
        "scale_2_percent": config["scale_2_percent"],
        "use_time_stop": config["use_time_stop"],
        "max_holding_days": config["max_holding_days"],
        "stale_trade_days": config["stale_trade_days"],
    }


def run_baseline(stock_data, spy_data, start_date, end_date, window_name) -> WindowResult:
    config = deepcopy(BASE_ENGINE_CONFIG)
    engine = EnhancedBacktestEngine(**make_engine_kwargs(config))
    result = engine.run(stock_data, spy_data, start_date=start_date, end_date=end_date)
    return WindowResult(window_name=window_name, start_date=result.start_date, end_date=result.end_date, result=result)


def run_old_filtered(stock_data, spy_data, vix_data, sector_etf_data, start_date, end_date, window_name) -> WindowResult:
    config = deepcopy(BASE_ENGINE_CONFIG)
    kwargs = make_engine_kwargs(config)
    kwargs["vix_data"] = vix_data
    kwargs["sector_etf_data"] = sector_etf_data
    engine = OldFilteredEngine(**kwargs)
    result = engine.run(stock_data, spy_data, start_date=start_date, end_date=end_date)
    return WindowResult(
        window_name=window_name, start_date=result.start_date, end_date=result.end_date, result=result,
        signals_generated=engine.signals_generated, signals_filtered=engine.signals_filtered_out,
        high_vix_days=len(engine.high_vix_days),
        filter_detail=f"VIX blocks: {engine.vix_blocks}, Sector adj: {engine.sector_adjustments}, Regime adj: {engine.regime_adjustments}",
    )


def run_rdt_filtered(stock_data, spy_data, vix_data, sector_etf_data, start_date, end_date, window_name) -> WindowResult:
    config = deepcopy(BASE_ENGINE_CONFIG)
    kwargs = make_engine_kwargs(config)
    kwargs["vix_data"] = vix_data
    kwargs["sector_etf_data"] = sector_etf_data
    engine = RDTFilteredEngine(**kwargs)
    result = engine.run(stock_data, spy_data, start_date=start_date, end_date=end_date)
    return WindowResult(
        window_name=window_name, start_date=result.start_date, end_date=result.end_date, result=result,
        signals_generated=engine.signals_generated, signals_filtered=engine.signals_filtered_out,
        high_vix_days=len(engine.high_vix_days),
        filter_detail=(
            f"SPY gate: {engine.spy_gate_blocks}, SMA gate: {engine.sma_gate_blocks}, "
            f"MTF gate: {engine.mtf_gate_blocks}, VIX blocks: {engine.vix_blocks}, "
            f"Sector adj: {engine.sector_adjustments}, Regime adj: {engine.regime_adjustments}"
        ),
    )


# ============================================================================
# Output
# ============================================================================

def get_largest_daily_loss(result: EnhancedBacktestResult) -> float:
    if len(result.equity_curve) < 2:
        return 0.0
    worst = 0.0
    for i in range(1, len(result.equity_curve)):
        daily_change = result.equity_curve[i]["equity"] - result.equity_curve[i - 1]["equity"]
        if daily_change < worst:
            worst = daily_change
    return worst


def print_results(
    baseline_results: List[WindowResult],
    old_filtered_results: List[WindowResult],
    rdt_filtered_results: List[WindowResult],
):
    w = 120

    print()
    print("=" * w)
    print("WALK-FORWARD BACKTEST V2: 3-WAY COMPARISON (2 YEARS)".center(w))
    print(f"Capital: ${INITIAL_CAPITAL:,.0f} | Engine: Config C (Scaled Exits, 1.5x Stop, 2.0x Target)".center(w))
    print("=" * w)
    print()
    print("  A) Baseline     — No filters, static RRS 2.0")
    print("  B) Old Filters  — VIX regime + Sector RS + Regime-adaptive params")
    print("  C) RDT Filters  — Old Filters + SPY Hard Gate + 50/200 SMA + MTF Alignment")
    print()
    print("  NOTE: VWAP + First-Hour filters cannot be simulated with daily bars. Omitted.")
    print()

    # Per-window
    for i in range(len(baseline_results)):
        bl = baseline_results[i]
        ol = old_filtered_results[i]
        rd = rdt_filtered_results[i]

        print("-" * w)
        print(f"  {bl.window_name}: {bl.start_date} to {bl.end_date}".center(w))
        print("-" * w)

        header = f"  {'Metric':<28} {'A) Baseline':>18} {'B) Old Filter':>18} {'C) RDT Filter':>18} {'B-A Delta':>14} {'C-A Delta':>14}"
        print(header)
        print("  " + "-" * (w - 4))

        def row(label, bv, ov, rv, fmt="dollar"):
            d_ba = ov - bv
            d_ca = rv - bv
            if fmt == "dollar":
                return f"  {label:<28} {'${:>14,.2f}'.format(bv)} {'${:>14,.2f}'.format(ov)} {'${:>14,.2f}'.format(rv)} {'${:>+10,.2f}'.format(d_ba)} {'${:>+10,.2f}'.format(d_ca)}"
            elif fmt == "pct":
                return f"  {label:<28} {'{:>14.2f}%'.format(bv)} {'{:>14.2f}%'.format(ov)} {'{:>14.2f}%'.format(rv)} {'{:>+10.2f}%'.format(d_ba)} {'{:>+10.2f}%'.format(d_ca)}"
            elif fmt == "float":
                return f"  {label:<28} {'{:>15.2f}'.format(bv)} {'{:>15.2f}'.format(ov)} {'{:>15.2f}'.format(rv)} {'{:>+11.2f}'.format(d_ba)} {'{:>+11.2f}'.format(d_ca)}"
            elif fmt == "int":
                return f"  {label:<28} {'{:>15d}'.format(int(bv))} {'{:>15d}'.format(int(ov))} {'{:>15d}'.format(int(rv))} {'{:>+11d}'.format(int(d_ba))} {'{:>+11d}'.format(int(d_ca))}"

        br, orr, rr = bl.result, ol.result, rd.result
        print(row("Total Return ($)", br.total_return, orr.total_return, rr.total_return, "dollar"))
        print(row("Total Return (%)", br.total_return_pct, orr.total_return_pct, rr.total_return_pct, "pct"))
        print(row("Win Rate", br.win_rate * 100, orr.win_rate * 100, rr.win_rate * 100, "pct"))
        print(row("Profit Factor", br.profit_factor, orr.profit_factor, rr.profit_factor, "float"))
        print(row("Sharpe Ratio", br.sharpe_ratio, orr.sharpe_ratio, rr.sharpe_ratio, "float"))
        print(row("Max Drawdown ($)", br.max_drawdown, orr.max_drawdown, rr.max_drawdown, "dollar"))
        print(row("Total Trades", br.total_trades, orr.total_trades, rr.total_trades, "int"))
        bl_wd = get_largest_daily_loss(br)
        ol_wd = get_largest_daily_loss(orr)
        rd_wd = get_largest_daily_loss(rr)
        print(row("Worst Single Day", bl_wd, ol_wd, rd_wd, "dollar"))
        print()

        # Filter details
        if ol.signals_generated > 0:
            pct = (ol.signals_filtered / ol.signals_generated) * 100
            print(f"  B) Old Filters: {ol.signals_generated} signals, {ol.signals_filtered} filtered ({pct:.1f}%) | {ol.filter_detail}")
        if rd.signals_generated > 0:
            pct = (rd.signals_filtered / rd.signals_generated) * 100
            print(f"  C) RDT Filters: {rd.signals_generated} signals, {rd.signals_filtered} filtered ({pct:.1f}%) | {rd.filter_detail}")
        print()

    # Aggregate
    print("=" * w)
    print("AGGREGATE RESULTS (All Windows Combined)".center(w))
    print("=" * w)
    print()

    def agg(results):
        total_ret = sum(r.result.total_return for r in results)
        total_trades = sum(r.result.total_trades for r in results)
        all_trades = []
        for r in results:
            all_trades.extend(r.result.trades)
        winners = sum(1 for t in all_trades if t.pnl > 0)
        wr = (winners / total_trades * 100) if total_trades > 0 else 0
        gross_wins = sum(t.pnl for t in all_trades if t.pnl > 0)
        gross_losses = abs(sum(t.pnl for t in all_trades if t.pnl <= 0))
        pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')
        max_dd = max((r.result.max_drawdown for r in results), default=0)
        worst_day = min((get_largest_daily_loss(r.result) for r in results), default=0)
        trading_days = sum(len(r.result.equity_curve) for r in results)
        return {
            "total_return": total_ret,
            "total_return_pct": total_ret / INITIAL_CAPITAL * 100,
            "total_trades": total_trades,
            "win_rate": wr,
            "profit_factor": pf,
            "max_drawdown": max_dd,
            "worst_day": worst_day,
            "trading_days": trading_days,
        }

    ba = agg(baseline_results)
    oa = agg(old_filtered_results)
    ra = agg(rdt_filtered_results)

    header = f"  {'Metric':<28} {'A) Baseline':>18} {'B) Old Filter':>18} {'C) RDT Filter':>18} {'B-A Delta':>14} {'C-A Delta':>14}"
    print(header)
    print("  " + "-" * (w - 4))

    def agg_row(label, bv, ov, rv, fmt="dollar"):
        d_ba = ov - bv
        d_ca = rv - bv
        if fmt == "dollar":
            return f"  {label:<28} {'${:>14,.2f}'.format(bv)} {'${:>14,.2f}'.format(ov)} {'${:>14,.2f}'.format(rv)} {'${:>+10,.2f}'.format(d_ba)} {'${:>+10,.2f}'.format(d_ca)}"
        elif fmt == "pct":
            return f"  {label:<28} {'{:>14.2f}%'.format(bv)} {'{:>14.2f}%'.format(ov)} {'{:>14.2f}%'.format(rv)} {'{:>+10.2f}%'.format(d_ba)} {'{:>+10.2f}%'.format(d_ca)}"
        elif fmt == "float":
            return f"  {label:<28} {'{:>15.2f}'.format(bv)} {'{:>15.2f}'.format(ov)} {'{:>15.2f}'.format(rv)} {'{:>+11.2f}'.format(d_ba)} {'{:>+11.2f}'.format(d_ca)}"
        elif fmt == "int":
            return f"  {label:<28} {'{:>15d}'.format(int(bv))} {'{:>15d}'.format(int(ov))} {'{:>15d}'.format(int(rv))} {'{:>+11d}'.format(int(d_ba))} {'{:>+11d}'.format(int(d_ca))}"

    print(agg_row("Total Return ($)", ba["total_return"], oa["total_return"], ra["total_return"], "dollar"))
    print(agg_row("Total Return (%)", ba["total_return_pct"], oa["total_return_pct"], ra["total_return_pct"], "pct"))
    print(agg_row("Total Trades", ba["total_trades"], oa["total_trades"], ra["total_trades"], "int"))
    print(agg_row("Win Rate", ba["win_rate"], oa["win_rate"], ra["win_rate"], "pct"))
    print(agg_row("Profit Factor", ba["profit_factor"], oa["profit_factor"], ra["profit_factor"], "float"))
    print(agg_row("Max Drawdown ($)", ba["max_drawdown"], oa["max_drawdown"], ra["max_drawdown"], "dollar"))
    print(agg_row("Worst Day Loss ($)", ba["worst_day"], oa["worst_day"], ra["worst_day"], "dollar"))
    print()

    # Signal filtering summary
    total_signals_old = sum(r.signals_generated for r in old_filtered_results)
    total_filtered_old = sum(r.signals_filtered for r in old_filtered_results)
    total_signals_rdt = sum(r.signals_generated for r in rdt_filtered_results)
    total_filtered_rdt = sum(r.signals_filtered for r in rdt_filtered_results)

    if total_signals_old > 0:
        pct = (total_filtered_old / total_signals_old) * 100
        print(f"  B) Old Filters: {total_signals_old} signals → {total_filtered_old} filtered ({pct:.1f}%)")
    if total_signals_rdt > 0:
        pct = (total_filtered_rdt / total_signals_rdt) * 100
        print(f"  C) RDT Filters: {total_signals_rdt} signals → {total_filtered_rdt} filtered ({pct:.1f}%)")
    print()

    # Annualized estimates
    print("-" * w)
    print("ANNUALIZED RETURN ESTIMATES".center(w))
    print("-" * w)
    print()

    for label, a in [("A) Baseline", ba), ("B) Old Filters", oa), ("C) RDT Filters", ra)]:
        if a["trading_days"] > 0:
            daily = a["total_return"] / INITIAL_CAPITAL / a["trading_days"]
            annual = daily * 252 * 100
            print(f"  {label:<20} {annual:>8.1f}% annualized  (from {a['trading_days']} trading days)")
    print()

    # Final verdict
    print("=" * w)
    print("VERDICT".center(w))
    print("=" * w)
    print()

    best_label = "A) Baseline"
    best_return = ba["total_return"]
    if oa["total_return"] > best_return:
        best_label = "B) Old Filters"
        best_return = oa["total_return"]
    if ra["total_return"] > best_return:
        best_label = "C) RDT Filters"
        best_return = ra["total_return"]

    print(f"  Highest Total Return: {best_label} (${best_return:,.2f})")
    print()

    # Risk-adjusted (return / max drawdown)
    def risk_adj(a):
        if a["max_drawdown"] == 0:
            return float('inf')
        return a["total_return"] / abs(a["max_drawdown"])

    ba_ra = risk_adj(ba)
    oa_ra = risk_adj(oa)
    ra_ra = risk_adj(ra)

    best_ra_label = "A) Baseline"
    best_ra = ba_ra
    if oa_ra > best_ra:
        best_ra_label = "B) Old Filters"
        best_ra = oa_ra
    if ra_ra > best_ra:
        best_ra_label = "C) RDT Filters"
        best_ra = ra_ra

    print(f"  Risk-Adjusted (Return/MaxDD): A={ba_ra:.2f}, B={oa_ra:.2f}, C={ra_ra:.2f}")
    print(f"  Best Risk-Adjusted: {best_ra_label}")
    print()

    # Improvement from B→C
    if oa["total_return"] != 0:
        improvement = ra["total_return"] - oa["total_return"]
        pct_imp = improvement / abs(oa["total_return"]) * 100
        print(f"  RDT filters (C) vs Old filters (B): ${improvement:+,.2f} ({pct_imp:+.1f}%)")
    improvement_vs_base = ra["total_return"] - ba["total_return"]
    print(f"  RDT filters (C) vs Baseline (A):    ${improvement_vs_base:+,.2f}")
    print()
    print("=" * w)
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    print()
    print("=" * 120)
    print("RDT TRADING SYSTEM - WALK-FORWARD BACKTEST V2 (2 YEARS, 3-WAY)".center(120))
    print("=" * 120)
    print()

    t0 = time.time()
    stock_data, spy_data, vix_data, sector_etf_data = load_all_data()
    load_time = time.time() - t0
    print(f"Data loading took {load_time:.1f}s")
    print()

    if len(stock_data) < 5:
        print("ERROR: Too few symbols loaded. Check yfinance connectivity.")
        sys.exit(1)

    all_trading_dates = sorted(spy_data.index.date)
    print(f"Total trading dates available: {len(all_trading_dates)}")
    print(f"Date range: {all_trading_dates[0]} to {all_trading_dates[-1]}")
    print()

    baseline_results = []
    old_filtered_results = []
    rdt_filtered_results = []

    for window in WALK_FORWARD_WINDOWS:
        start_idx = min(window["start_day"], len(all_trading_dates) - 1)
        end_idx = min(window["end_day"], len(all_trading_dates) - 1)
        w_start = all_trading_dates[start_idx]
        w_end = all_trading_dates[end_idx]

        print(f"--- {window['name']}: days {window['start_day']}-{window['end_day']} ({w_start} to {w_end}) ---")

        # A) Baseline
        print(f"  Running A) Baseline...")
        t1 = time.time()
        bl = run_baseline(stock_data, spy_data, w_start, w_end, window["name"])
        print(f"    {bl.result.total_trades} trades, {bl.result.win_rate*100:.1f}% WR, "
              f"{bl.result.total_return_pct:.2f}% return ({time.time()-t1:.1f}s)")

        # B) Old Filters
        print(f"  Running B) Old Filters...")
        t1 = time.time()
        ol = run_old_filtered(stock_data, spy_data, vix_data, sector_etf_data, w_start, w_end, window["name"])
        print(f"    {ol.result.total_trades} trades, {ol.result.win_rate*100:.1f}% WR, "
              f"{ol.result.total_return_pct:.2f}% return ({time.time()-t1:.1f}s)")

        # C) RDT Filters
        print(f"  Running C) RDT Filters...")
        t1 = time.time()
        rd = run_rdt_filtered(stock_data, spy_data, vix_data, sector_etf_data, w_start, w_end, window["name"])
        print(f"    {rd.result.total_trades} trades, {rd.result.win_rate*100:.1f}% WR, "
              f"{rd.result.total_return_pct:.2f}% return ({time.time()-t1:.1f}s)")
        print()

        baseline_results.append(bl)
        old_filtered_results.append(ol)
        rdt_filtered_results.append(rd)

    print_results(baseline_results, old_filtered_results, rdt_filtered_results)


if __name__ == "__main__":
    main()
