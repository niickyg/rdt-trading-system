#!/usr/bin/env python3
"""
Walk-Forward Backtest: Baseline vs Filtered (VIX + Sector RS + Regime-Adaptive)

Measures the actual P&L impact of the new filters by running walk-forward
windows with and without them applied to the enhanced backtest engine (Config C).

Usage:
    cd /home/user0/rdt-trading-system
    python scripts/run_walkforward.py
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
DATA_DAYS = 365  # Download 365 days of data

WATCHLIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    'JPM', 'V', 'JNJ', 'UNH', 'HD', 'PG', 'MA', 'DIS',
    'PYPL', 'ADBE', 'CRM', 'NFLX', 'INTC', 'AMD', 'CSCO',
    'PEP', 'KO', 'MRK', 'ABT', 'TMO', 'COST', 'AVGO', 'TXN'
]

SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLC', 'XLU']

# Walk-forward windows (day indices into the trading date list)
WALK_FORWARD_WINDOWS = [
    {"name": "Window 1", "start_day": 91,  "end_day": 180},
    {"name": "Window 2", "start_day": 181, "end_day": 270},
    {"name": "Window 3", "start_day": 271, "end_day": 365},
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
# VIX Regime Logic (mirrors scanner/vix_filter.py for daily bar simulation)
# ============================================================================

def get_vix_regime_for_value(vix_value: float) -> Dict:
    """
    Determine VIX regime from a closing VIX value.
    Returns position_size_multiplier, rrs_threshold_adjustment, allow_longs.
    """
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
# Sector RS Calculation (from daily data, mirrors scanner/sector_filter.py)
# ============================================================================

def compute_sector_rs_on_date(
    sector_etf_data: Dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
    current_date: date,
    lookback: int = 5,
) -> Dict[str, float]:
    """
    Compute sector RS vs SPY for each sector on a given date.
    Returns {sector_name: sector_rs_value}.
    """
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
    """
    Calculate RRS boost/penalty for a symbol based on sector alignment.
    """
    sector = SECTOR_MAP.get(symbol, 'other')
    if sector == 'other' or sector not in sector_rs_map:
        return 0.0

    rs = sector_rs_map[sector]

    if direction == 'long':
        if rs > 0:
            return 0.25   # strong sector, long = wind at back
        elif rs < -1.0:
            return -0.50  # weak sector, long = fighting sector
    elif direction == 'short':
        if rs < -1.0:
            return 0.25   # weak sector, short = confirmed
        elif rs > 0:
            return -0.50  # strong sector, short = against sector

    return 0.0


# ============================================================================
# Regime Detection (from SPY daily data, mirrors scanner/regime_params.py)
# ============================================================================

def detect_regime_on_date(spy_data: pd.DataFrame, current_date: date) -> Dict:
    """
    Detect market regime from SPY daily data.
    Returns rrs_threshold and risk_per_trade adjustments.
    """
    close_col = 'Close' if 'Close' in spy_data.columns else 'close'
    high_col = 'High' if 'High' in spy_data.columns else 'high'
    low_col = 'Low' if 'Low' in spy_data.columns else 'low'

    spy_up_to = spy_data[spy_data.index.date <= current_date]
    if len(spy_up_to) < 200:
        # Not enough data for 200 EMA, return default
        return {"regime": "default", "rrs_threshold": 2.0, "risk_per_trade": 0.015, "wider_stops": False}

    close_series = spy_up_to[close_col]
    current_price = float(close_series.iloc[-1])

    ema_50 = float(close_series.ewm(span=50, adjust=False).mean().iloc[-1])
    ema_200 = float(close_series.ewm(span=200, adjust=False).mean().iloc[-1])

    # ATR% calculation (14-day ATR as % of price)
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

    # High volatility check first (overrides trend)
    if atr_pct > 2.0:
        return {"regime": "high_volatility", "rrs_threshold": 2.25, "risk_per_trade": 0.01, "wider_stops": True}

    # Bull trending
    if current_price > ema_50 and ema_50 > ema_200:
        return {"regime": "bull_trending", "rrs_threshold": 1.5, "risk_per_trade": 0.02, "wider_stops": False}

    # Bear trending
    if current_price < ema_50 and ema_50 < ema_200:
        return {"regime": "bear_trending", "rrs_threshold": 2.5, "risk_per_trade": 0.01, "wider_stops": False}

    # Default/mixed
    return {"regime": "default", "rrs_threshold": 2.0, "risk_per_trade": 0.015, "wider_stops": False}


# ============================================================================
# Filtered Engine Subclass
# ============================================================================

class FilteredBacktestEngine(EnhancedBacktestEngine):
    """
    Subclass of EnhancedBacktestEngine that applies VIX, sector RS,
    and regime-adaptive filters during signal scanning.

    Does NOT modify the parent engine code -- just overrides _scan_for_signals
    and _enter_position to apply filters.
    """

    def __init__(self, *args, **kwargs):
        # Extract filter-specific data before passing to parent
        self.vix_data: Optional[pd.DataFrame] = kwargs.pop('vix_data', None)
        self.sector_etf_data: Dict[str, pd.DataFrame] = kwargs.pop('sector_etf_data', {})
        super().__init__(*args, **kwargs)

        # Tracking
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
        """Override to compute daily filter context before scanning."""
        # Compute VIX regime for the day
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

        # Compute sector RS for the day
        self._day_sector_rs = compute_sector_rs_on_date(
            self.sector_etf_data, spy_data, current_date
        )

        # Detect market regime
        self._day_market_regime = detect_regime_on_date(spy_data, current_date)

        # Position size multiplier from VIX
        self._day_position_mult = self._day_vix_regime.get("pos_mult", 1.0)

        # Now proceed with normal day processing
        super()._process_day(current_date, stock_data, spy_data)

    def _scan_for_signals(self, current_date, stock_data, spy_data):
        """Override signal scanning to apply filters."""
        from shared.indicators.rrs import (
            RRSCalculator,
            check_daily_strength_relaxed,
            check_daily_weakness_relaxed
        )

        spy_current = spy_data[spy_data.index.date <= current_date]
        if len(spy_current) < 20:
            return

        close_col = 'Close' if 'Close' in spy_current.columns else 'close'
        spy_close = spy_current[close_col].iloc[-1]
        spy_prev_close = spy_current[close_col].iloc[-2]

        # Compute effective RRS threshold from regime + VIX
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

                # Determine preliminary direction
                direction = None
                if rrs > 0:
                    direction = "long"
                elif rrs < 0:
                    direction = "short"
                else:
                    continue

                self.signals_generated += 1

                # Apply sector boost/penalty to RRS
                sector_boost = get_sector_boost(symbol, direction, self._day_sector_rs)
                if sector_boost != 0:
                    self.sector_adjustments += 1
                adjusted_rrs = abs(rrs) + sector_boost

                # Check against effective threshold (regime + VIX adjusted)
                if adjusted_rrs < effective_rrs_threshold:
                    self.signals_filtered_out += 1
                    continue

                # VIX block check (extreme regime blocks all longs)
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

                # Adjust risk per trade from regime
                regime_risk = self._day_market_regime.get("risk_per_trade", 0.015)
                if regime_risk != self.risk_limits.max_risk_per_trade:
                    self.regime_adjustments += 1

                # Enter position (with VIX position sizing applied)
                self._enter_position_filtered(
                    symbol=symbol,
                    direction=direction,
                    entry_price=stock_close,
                    atr=atr,
                    entry_date=current_date,
                    rrs=rrs,
                    position_mult=self._day_position_mult,
                    risk_per_trade=regime_risk,
                )

            except Exception:
                continue

    def _enter_position_filtered(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        atr: float,
        entry_date: date,
        rrs: float,
        position_mult: float,
        risk_per_trade: float,
    ):
        """Enter position with VIX position size multiplier and regime risk."""
        from backtesting.engine_enhanced import EnhancedTrade

        # Use regime-specific risk for sizing
        temp_limits = RiskLimits(
            max_risk_per_trade=risk_per_trade,
            max_open_positions=self.max_positions,
        )
        from risk import PositionSizer
        temp_sizer = PositionSizer(temp_limits)

        # Wider stops in high-vol regime
        stop_mult = self.stop_atr_multiplier
        if self._day_market_regime.get("wider_stops", False):
            stop_mult = max(stop_mult, 2.0)

        sizing = temp_sizer.calculate_position_size(
            account_size=self.capital,
            entry_price=entry_price,
            atr=atr,
            direction=direction,
            stop_multiplier=stop_mult,
            target_multiplier=self.target_atr_multiplier,
        )

        if sizing.shares == 0:
            return

        # Apply VIX position size multiplier
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


# ============================================================================
# Data Loading
# ============================================================================

def load_all_data() -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Download 365 days of data for watchlist + SPY + sector ETFs + VIX.
    Returns (stock_data, spy_data, vix_data, sector_etf_data).
    """
    cache_dir = str(PROJECT_ROOT / "data" / "backtest_cache")
    loader = DataLoader(cache_dir=cache_dir)

    end_date = date.today()
    start_date = end_date - timedelta(days=DATA_DAYS + 60)  # extra for EMA warmup

    print(f"Loading data for {len(WATCHLIST)} stocks + SPY + {len(SECTOR_ETFS)} sector ETFs + VIX...")
    print(f"  Date range: {start_date} to {end_date}")
    print()

    # Load SPY
    print("  Downloading SPY...")
    spy_data = loader.load_spy_data(start_date, end_date, use_cache=True)
    print(f"  SPY: {len(spy_data)} trading days")

    # Load VIX
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

    # Load sector ETFs
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

    # Load stocks
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
    """Result for a single walk-forward window."""
    window_name: str
    start_date: date
    end_date: date
    result: EnhancedBacktestResult
    signals_generated: int = 0
    signals_filtered: int = 0
    high_vix_days: int = 0
    regime_info: str = ""


def run_baseline(
    stock_data: Dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
    start_date: date,
    end_date: date,
    window_name: str,
) -> WindowResult:
    """Run baseline (no filters, static RRS 2.0)."""
    config = deepcopy(BASE_ENGINE_CONFIG)

    risk_limits = RiskLimits(
        max_risk_per_trade=config["max_risk_per_trade"],
        max_open_positions=config["max_positions"],
    )

    engine = EnhancedBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        risk_limits=risk_limits,
        rrs_threshold=config["rrs_threshold"],
        max_positions=config["max_positions"],
        use_relaxed_criteria=True,
        stop_atr_multiplier=config["stop_multiplier"],
        target_atr_multiplier=config["target_multiplier"],
        use_trailing_stop=config["use_trailing_stop"],
        breakeven_trigger_r=config["breakeven_trigger_r"],
        trailing_atr_multiplier=config["trailing_atr_multiplier"],
        use_scaled_exits=config["use_scaled_exits"],
        scale_1_target_r=config["scale_1_target_r"],
        scale_1_percent=config["scale_1_percent"],
        scale_2_target_r=config["scale_2_target_r"],
        scale_2_percent=config["scale_2_percent"],
        use_time_stop=config["use_time_stop"],
        max_holding_days=config["max_holding_days"],
        stale_trade_days=config["stale_trade_days"],
    )

    result = engine.run(stock_data, spy_data, start_date=start_date, end_date=end_date)

    return WindowResult(
        window_name=window_name,
        start_date=result.start_date,
        end_date=result.end_date,
        result=result,
    )


def run_filtered(
    stock_data: Dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
    vix_data: Optional[pd.DataFrame],
    sector_etf_data: Dict[str, pd.DataFrame],
    start_date: date,
    end_date: date,
    window_name: str,
) -> WindowResult:
    """Run filtered (VIX + sector RS + regime-adaptive)."""
    config = deepcopy(BASE_ENGINE_CONFIG)

    risk_limits = RiskLimits(
        max_risk_per_trade=config["max_risk_per_trade"],
        max_open_positions=config["max_positions"],
    )

    engine = FilteredBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        risk_limits=risk_limits,
        rrs_threshold=config["rrs_threshold"],
        max_positions=config["max_positions"],
        use_relaxed_criteria=True,
        stop_atr_multiplier=config["stop_multiplier"],
        target_atr_multiplier=config["target_multiplier"],
        use_trailing_stop=config["use_trailing_stop"],
        breakeven_trigger_r=config["breakeven_trigger_r"],
        trailing_atr_multiplier=config["trailing_atr_multiplier"],
        use_scaled_exits=config["use_scaled_exits"],
        scale_1_target_r=config["scale_1_target_r"],
        scale_1_percent=config["scale_1_percent"],
        scale_2_target_r=config["scale_2_target_r"],
        scale_2_percent=config["scale_2_percent"],
        use_time_stop=config["use_time_stop"],
        max_holding_days=config["max_holding_days"],
        stale_trade_days=config["stale_trade_days"],
        # Filter-specific data
        vix_data=vix_data,
        sector_etf_data=sector_etf_data,
    )

    result = engine.run(stock_data, spy_data, start_date=start_date, end_date=end_date)

    return WindowResult(
        window_name=window_name,
        start_date=result.start_date,
        end_date=result.end_date,
        result=result,
        signals_generated=engine.signals_generated,
        signals_filtered=engine.signals_filtered_out,
        high_vix_days=len(engine.high_vix_days),
        regime_info=f"VIX blocks: {engine.vix_blocks}, Sector adj: {engine.sector_adjustments}, Regime adj: {engine.regime_adjustments}",
    )


# ============================================================================
# Output Formatting
# ============================================================================

def get_largest_daily_loss(result: EnhancedBacktestResult) -> float:
    """Get largest single-day equity loss from equity curve."""
    if len(result.equity_curve) < 2:
        return 0.0
    worst = 0.0
    for i in range(1, len(result.equity_curve)):
        daily_change = result.equity_curve[i]["equity"] - result.equity_curve[i - 1]["equity"]
        if daily_change < worst:
            worst = daily_change
    return worst


def print_comparison_table(
    baseline_results: List[WindowResult],
    filtered_results: List[WindowResult],
):
    """Print the walk-forward comparison table."""
    w = 100

    print()
    print("=" * w)
    print("WALK-FORWARD BACKTEST: BASELINE vs FILTERED".center(w))
    print(f"Capital: ${INITIAL_CAPITAL:,.0f} | Engine: Config C (Scaled Exits, 1.5x Stop, 2.0x Target)".center(w))
    print("=" * w)
    print()

    # Note about first-hour skip
    print("  NOTE: First-hour skip cannot be simulated with daily bars. Omitted from this test.")
    print("        In live trading, the first-hour filter avoids choppy opening-range signals.")
    print()

    # Per-window comparison
    for i, (bl, fl) in enumerate(zip(baseline_results, filtered_results)):
        br = bl.result
        fr = fl.result

        print("-" * w)
        print(f"  {bl.window_name}: {bl.start_date} to {bl.end_date}".center(w))
        print("-" * w)

        header = f"  {'Metric':<30} {'Baseline':>16} {'Filtered':>16} {'Delta':>16}"
        print(header)
        print("  " + "-" * (w - 4))

        def row(label, bv, fv, fmt="dollar"):
            delta = fv - bv
            if fmt == "dollar":
                return f"  {label:<30} {'${:>12,.2f}'.format(bv)} {'${:>12,.2f}'.format(fv)} {'${:>+12,.2f}'.format(delta)}"
            elif fmt == "pct":
                return f"  {label:<30} {'{:>12.2f}%'.format(bv)} {'{:>12.2f}%'.format(fv)} {'{:>+12.2f}%'.format(delta)}"
            elif fmt == "float":
                return f"  {label:<30} {'{:>13.2f}'.format(bv)} {'{:>13.2f}'.format(fv)} {'{:>+13.2f}'.format(delta)}"
            elif fmt == "int":
                return f"  {label:<30} {'{:>13d}'.format(int(bv))} {'{:>13d}'.format(int(fv))} {'{:>+13d}'.format(int(delta))}"

        print(row("Total Return ($)", br.total_return, fr.total_return, "dollar"))
        print(row("Total Return (%)", br.total_return_pct, fr.total_return_pct, "pct"))
        print(row("Win Rate", br.win_rate * 100, fr.win_rate * 100, "pct"))
        print(row("Profit Factor", br.profit_factor, fr.profit_factor, "float"))
        print(row("Sharpe Ratio", br.sharpe_ratio, fr.sharpe_ratio, "float"))
        print(row("Max Drawdown ($)", br.max_drawdown, fr.max_drawdown, "dollar"))
        print(row("Max Drawdown (%)", br.max_drawdown_pct, fr.max_drawdown_pct, "pct"))
        print(row("Total Trades", br.total_trades, fr.total_trades, "int"))

        bl_worst_day = get_largest_daily_loss(br)
        fl_worst_day = get_largest_daily_loss(fr)
        print(row("Worst Single Day", bl_worst_day, fl_worst_day, "dollar"))

        print()
        # Filter stats
        if fl.signals_generated > 0:
            pct_filtered = (fl.signals_filtered / fl.signals_generated) * 100
            print(f"  Filter Stats: {fl.signals_generated} signals generated, "
                  f"{fl.signals_filtered} filtered out ({pct_filtered:.1f}%)")
        print(f"  High VIX days (>=25): {fl.high_vix_days}")
        print(f"  {fl.regime_info}")
        print()

    # Aggregate results
    print("=" * w)
    print("AGGREGATE RESULTS (All Windows Combined)".center(w))
    print("=" * w)
    print()

    # Sum up returns
    bl_total_return = sum(r.result.total_return for r in baseline_results)
    fl_total_return = sum(r.result.total_return for r in filtered_results)
    bl_total_trades = sum(r.result.total_trades for r in baseline_results)
    fl_total_trades = sum(r.result.total_trades for r in filtered_results)

    bl_all_trades = []
    fl_all_trades = []
    for r in baseline_results:
        bl_all_trades.extend(r.result.trades)
    for r in filtered_results:
        fl_all_trades.extend(r.result.trades)

    bl_winners = sum(1 for t in bl_all_trades if t.pnl > 0)
    fl_winners = sum(1 for t in fl_all_trades if t.pnl > 0)
    bl_wr = (bl_winners / bl_total_trades * 100) if bl_total_trades > 0 else 0
    fl_wr = (fl_winners / fl_total_trades * 100) if fl_total_trades > 0 else 0

    bl_gross_wins = sum(t.pnl for t in bl_all_trades if t.pnl > 0)
    bl_gross_losses = abs(sum(t.pnl for t in bl_all_trades if t.pnl <= 0))
    fl_gross_wins = sum(t.pnl for t in fl_all_trades if t.pnl > 0)
    fl_gross_losses = abs(sum(t.pnl for t in fl_all_trades if t.pnl <= 0))
    bl_pf = bl_gross_wins / bl_gross_losses if bl_gross_losses > 0 else float('inf')
    fl_pf = fl_gross_wins / fl_gross_losses if fl_gross_losses > 0 else float('inf')

    bl_max_dd = max((r.result.max_drawdown for r in baseline_results), default=0)
    fl_max_dd = max((r.result.max_drawdown for r in filtered_results), default=0)

    bl_worst_day = min((get_largest_daily_loss(r.result) for r in baseline_results), default=0)
    fl_worst_day = min((get_largest_daily_loss(r.result) for r in filtered_results), default=0)

    total_signals = sum(r.signals_generated for r in filtered_results)
    total_filtered = sum(r.signals_filtered for r in filtered_results)
    total_high_vix = sum(r.high_vix_days for r in filtered_results)

    header = f"  {'Metric':<30} {'Baseline':>16} {'Filtered':>16} {'Delta':>16}"
    print(header)
    print("  " + "-" * (w - 4))
    print(f"  {'Total Return ($)':<30} {'${:>12,.2f}'.format(bl_total_return)} {'${:>12,.2f}'.format(fl_total_return)} {'${:>+12,.2f}'.format(fl_total_return - bl_total_return)}")
    print(f"  {'Total Return (%)':<30} {'{:>12.2f}%'.format(bl_total_return / INITIAL_CAPITAL * 100)} {'{:>12.2f}%'.format(fl_total_return / INITIAL_CAPITAL * 100)} {'{:>+12.2f}%'.format((fl_total_return - bl_total_return) / INITIAL_CAPITAL * 100)}")
    print(f"  {'Total Trades':<30} {bl_total_trades:>13d} {fl_total_trades:>13d} {fl_total_trades - bl_total_trades:>+13d}")
    print(f"  {'Win Rate':<30} {bl_wr:>12.2f}% {fl_wr:>12.2f}% {fl_wr - bl_wr:>+12.2f}%")
    print(f"  {'Profit Factor':<30} {bl_pf:>13.2f} {fl_pf:>13.2f} {fl_pf - bl_pf:>+13.2f}")
    print(f"  {'Max Single-Window Drawdown':<30} {'${:>12,.2f}'.format(bl_max_dd)} {'${:>12,.2f}'.format(fl_max_dd)} {'${:>+12,.2f}'.format(fl_max_dd - bl_max_dd)}")
    print(f"  {'Worst Single Day Loss':<30} {'${:>12,.2f}'.format(bl_worst_day)} {'${:>12,.2f}'.format(fl_worst_day)} {'${:>+12,.2f}'.format(fl_worst_day - bl_worst_day)}")
    print()

    # Signal filtering summary
    if total_signals > 0:
        pct_filtered = (total_filtered / total_signals) * 100
        print(f"  Signals: {total_signals} generated, {total_filtered} filtered out ({pct_filtered:.1f}%)")
    print(f"  Total high VIX days (>=25): {total_high_vix}")
    print()

    # VIX high-regime analysis
    print("-" * w)
    print("VIX HIGH-REGIME ANALYSIS".center(w))
    print("-" * w)
    if total_high_vix > 0:
        print(f"  {total_high_vix} trading days had VIX >= 25 across all windows.")
        # Check if filtering helped during those periods
        print("  During high VIX periods, the filter reduces position sizes (0.5x) and raises")
        print("  RRS thresholds (+1.0), blocking marginal signals. VIX > 35 blocks all longs.")
        dd_improvement = bl_max_dd - fl_max_dd
        if dd_improvement > 0:
            print(f"  Drawdown improvement from filtering: ${dd_improvement:,.2f}")
        else:
            print(f"  Max drawdown was similar or worse with filtering (delta: ${dd_improvement:,.2f})")
    else:
        print("  No trading days had VIX >= 25. VIX high-regime filter had minimal impact.")
        print("  The filter still applied VIX-based adjustments (low/normal/elevated levels).")
    print()

    # Final verdict
    print("=" * w)
    print("SUMMARY".center(w))
    print("=" * w)
    print()

    return_delta = fl_total_return - bl_total_return
    if return_delta > 0:
        print(f"  FILTERS HELPED: +${return_delta:,.2f} improvement over baseline")
    elif return_delta < 0:
        print(f"  FILTERS HURT: ${return_delta:,.2f} worse than baseline")
    else:
        print(f"  FILTERS NEUTRAL: No change from baseline")

    delta_pct = (return_delta / INITIAL_CAPITAL) * 100
    print(f"  Return improvement: {delta_pct:+.2f}% of capital")
    print()

    # Annualized return estimate (from filtered)
    # Each window covers ~90 days (3 months), 3 windows = 9 months of test data
    trading_days_tested = sum(
        len(r.result.equity_curve) for r in filtered_results
    )
    if trading_days_tested > 0:
        daily_return = fl_total_return / INITIAL_CAPITAL / trading_days_tested
        annual_return = daily_return * 252 * 100  # 252 trading days/year
        print(f"  Filtered system annualized return estimate: {annual_return:.1f}%")
        print(f"  (Based on {trading_days_tested} trading days tested)")
    print()

    bl_daily_return = bl_total_return / INITIAL_CAPITAL / max(trading_days_tested, 1)
    bl_annual = bl_daily_return * 252 * 100
    print(f"  Baseline annualized return estimate: {bl_annual:.1f}%")
    print()

    print(f"  Largest single-day loss (Baseline):  ${bl_worst_day:,.2f}")
    print(f"  Largest single-day loss (Filtered):  ${fl_worst_day:,.2f}")

    if fl_worst_day > bl_worst_day:  # less negative = better
        print(f"  Filters reduced worst daily loss by ${fl_worst_day - bl_worst_day:,.2f}")
    else:
        print(f"  Filters did not improve worst daily loss")

    print()
    print("=" * w)
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    print()
    print("=" * 100)
    print("RDT TRADING SYSTEM - WALK-FORWARD BACKTEST".center(100))
    print("Baseline (no filters) vs Filtered (VIX + Sector RS + Regime-Adaptive)".center(100))
    print("=" * 100)
    print()

    # Step 1: Load all data
    t0 = time.time()
    stock_data, spy_data, vix_data, sector_etf_data = load_all_data()
    load_time = time.time() - t0
    print(f"Data loading took {load_time:.1f}s")
    print()

    if len(stock_data) < 5:
        print("ERROR: Too few symbols loaded. Check yfinance connectivity.")
        sys.exit(1)

    # Get trading dates from SPY
    all_trading_dates = sorted(spy_data.index.date)
    print(f"Total trading dates available: {len(all_trading_dates)}")
    print(f"Date range: {all_trading_dates[0]} to {all_trading_dates[-1]}")
    print()

    # Step 2: Run walk-forward windows
    baseline_results = []
    filtered_results = []

    for window in WALK_FORWARD_WINDOWS:
        # Map day indices to actual trading dates
        start_idx = min(window["start_day"], len(all_trading_dates) - 1)
        end_idx = min(window["end_day"], len(all_trading_dates) - 1)
        w_start = all_trading_dates[start_idx]
        w_end = all_trading_dates[end_idx]

        print(f"--- {window['name']}: days {window['start_day']}-{window['end_day']} "
              f"({w_start} to {w_end}) ---")

        # Baseline
        print(f"  Running Baseline...")
        t1 = time.time()
        bl = run_baseline(stock_data, spy_data, w_start, w_end, window["name"])
        print(f"    {bl.result.total_trades} trades, {bl.result.win_rate*100:.1f}% WR, "
              f"{bl.result.total_return_pct:.2f}% return ({time.time()-t1:.1f}s)")

        # Filtered
        print(f"  Running Filtered...")
        t1 = time.time()
        fl = run_filtered(stock_data, spy_data, vix_data, sector_etf_data, w_start, w_end, window["name"])
        print(f"    {fl.result.total_trades} trades, {fl.result.win_rate*100:.1f}% WR, "
              f"{fl.result.total_return_pct:.2f}% return ({time.time()-t1:.1f}s)")
        print()

        baseline_results.append(bl)
        filtered_results.append(fl)

    # Step 3: Print comparison
    print_comparison_table(baseline_results, filtered_results)


if __name__ == "__main__":
    main()
