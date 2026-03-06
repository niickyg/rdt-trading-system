"""
Intraday Exit Manager

Monitors open positions on 5-minute bars and generates exit signals when
intraday relative strength deteriorates. Modeled after options/exit_manager.py.

Exit triggers (by priority):
1. RS Loss       — Intraday RRS < threshold AND trend falling
2. VWAP Breakdown — Price below VWAP for N consecutive 5m bars
3. Time Stop     — Position unprofitable after N minutes
4. Breakeven Stop — Position reached +0.5R, tighten stop to entry price
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from shared.indicators.rrs import RRSCalculator, calculate_vwap
from utils.timezone import is_market_open


def _filter_today_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Filter a 5m bar DataFrame to only include today's bars.

    VWAP must reset daily, so we strip yesterday's data before computing it.
    """
    if df is None or df.empty:
        return df
    try:
        today_str = datetime.now().strftime('%Y-%m-%d')
        mask = df.index.strftime('%Y-%m-%d') == today_str
        today_df = df[mask]
        return today_df if not today_df.empty else None
    except Exception:
        return df


@dataclass
class IntradayExitSignal:
    """Represents an intraday exit trigger with reason and priority."""
    symbol: str
    reason: str
    priority: int        # 1=RS loss, 2=VWAP break, 3=time stop, 4=breakeven
    action: str          # "close" or "tighten_stop"
    new_stop_price: Optional[float] = None

    def __repr__(self):
        return (
            f"IntradayExitSignal({self.symbol}, {self.reason}, "
            f"priority={self.priority}, action={self.action})"
        )


class IntradayExitManager:
    """
    Monitors open positions for intraday exit conditions.

    Call check_exits() periodically (every bar_refresh_interval seconds,
    typically 5 minutes) with current positions and streaming prices.
    """

    def __init__(
        self,
        intraday_data,
        rs_loss_threshold: float = -0.5,
        vwap_confirm_bars: int = 2,
        time_stop_minutes: int = 60,
        breakeven_r_threshold: float = 0.5,
    ):
        """
        Args:
            intraday_data: IntradayDataService instance for fetching 5m bars.
            rs_loss_threshold: Intraday RRS below this triggers RS loss exit.
            vwap_confirm_bars: Consecutive bars below VWAP to confirm breakdown.
            time_stop_minutes: Minutes before unprofitable positions are closed.
            breakeven_r_threshold: R-multiple at which stop moves to breakeven.
        """
        self._intraday_data = intraday_data
        self._rrs_calc = RRSCalculator()

        self._rs_loss_threshold = rs_loss_threshold
        self._vwap_confirm_bars = vwap_confirm_bars
        self._time_stop_minutes = time_stop_minutes
        self._breakeven_r_threshold = breakeven_r_threshold

        # Tracked positions: symbol -> registration metadata
        self._tracked: Dict[str, Dict] = {}

        # VWAP breakdown streak counter: symbol -> consecutive bars below VWAP
        self._vwap_below_count: Dict[str, int] = {}

        # Track whether breakeven stop has already been applied
        self._breakeven_applied: Dict[str, bool] = {}

        # Track last bar timestamp used for VWAP check to avoid double-counting
        self._vwap_last_bar_ts: Dict[str, object] = {}

        logger.info(
            f"IntradayExitManager initialized "
            f"(rs_loss={rs_loss_threshold}, vwap_bars={vwap_confirm_bars}, "
            f"time_stop={time_stop_minutes}min, breakeven_r={breakeven_r_threshold})"
        )

    def register_position(self, symbol: str):
        """Register a newly opened position for intraday monitoring."""
        self._tracked[symbol] = {'registered_at': datetime.now()}
        self._vwap_below_count[symbol] = 0
        self._breakeven_applied[symbol] = False
        logger.debug(f"IntradayExitManager: registered {symbol}")

    def unregister_position(self, symbol: str):
        """Remove a closed position from intraday monitoring."""
        self._tracked.pop(symbol, None)
        self._vwap_below_count.pop(symbol, None)
        self._breakeven_applied.pop(symbol, None)
        self._vwap_last_bar_ts.pop(symbol, None)
        logger.debug(f"IntradayExitManager: unregistered {symbol}")

    async def check_exits(
        self,
        positions: Dict[str, Dict],
        price_updates: Dict[str, float],
    ) -> List[IntradayExitSignal]:
        """
        Check all tracked positions for intraday exit triggers.

        Args:
            positions: symbol -> position_risk dict from executor, containing:
                entry_price, stop_price, atr, direction, shares, entry_time (optional)
            price_updates: symbol -> current price from streaming quotes.

        Returns:
            List of IntradayExitSignal objects sorted by priority (lowest first).
        """
        # Market hours check: only run during market hours (ET)
        if not is_market_open():
            return []

        signals: List[IntradayExitSignal] = []

        # Get SPY 5m bars (shared, cached)
        spy_bars = await self._intraday_data.get_spy_5m_bars()

        for symbol in list(self._tracked.keys()):
            pos = positions.get(symbol)
            if pos is None:
                continue

            current_price = price_updates.get(symbol)
            if current_price is None:
                continue

            try:
                pos_signals = await self._check_position(
                    symbol, pos, current_price, spy_bars
                )
                signals.extend(pos_signals)
            except Exception as e:
                logger.error(f"IntradayExitManager: check failed for {symbol}: {e}")

        # Sort by priority (lower = higher priority)
        signals.sort(key=lambda s: s.priority)
        return signals

    async def _check_position(
        self,
        symbol: str,
        pos: Dict,
        current_price: float,
        spy_bars,
    ) -> List[IntradayExitSignal]:
        """Check a single position for all intraday exit triggers."""
        signals: List[IntradayExitSignal] = []

        entry_price = pos.get('entry_price', 0)
        stop_price = pos.get('stop_price', 0)
        direction = pos.get('direction', 'long')

        if entry_price <= 0:
            return signals

        # Fetch 5m bars for this symbol
        stock_bars = await self._intraday_data.get_5m_bars(symbol)

        # 1. RS Loss check
        if stock_bars is not None and spy_bars is not None:
            rs_signal = self._check_rs_loss(symbol, stock_bars, spy_bars, direction)
            if rs_signal:
                signals.append(rs_signal)

        # 2. VWAP Breakdown check
        if stock_bars is not None:
            vwap_signal = self._check_vwap_breakdown(
                symbol, stock_bars, current_price, direction
            )
            if vwap_signal:
                signals.append(vwap_signal)

        # 3. Time Stop check
        time_signal = self._check_time_stop(
            symbol, pos, current_price, entry_price, direction
        )
        if time_signal:
            signals.append(time_signal)

        # 4. Breakeven Stop check
        be_signal = self._check_breakeven_stop(
            symbol, pos, current_price, entry_price, stop_price, direction
        )
        if be_signal:
            signals.append(be_signal)

        return signals

    def _check_rs_loss(
        self,
        symbol: str,
        stock_bars,
        spy_bars,
        direction: str,
    ) -> Optional[IntradayExitSignal]:
        """Check if intraday RRS has deteriorated below threshold."""
        result = self._rrs_calc.calculate_intraday_rrs(stock_bars, spy_bars)
        if result is None:
            return None

        rrs = result['intraday_rrs']
        trend = result['rrs_trend']

        # For longs: exit if RRS is negative and falling
        if direction == 'long' and rrs < self._rs_loss_threshold and trend == 'falling':
            return IntradayExitSignal(
                symbol=symbol,
                reason=(
                    f"Intraday RS loss: RRS={rrs:.2f} < {self._rs_loss_threshold}, "
                    f"trend={trend}"
                ),
                priority=1,
                action="close",
            )

        # For shorts: exit if RRS is positive and rising
        if direction == 'short' and rrs > abs(self._rs_loss_threshold) and trend == 'rising':
            return IntradayExitSignal(
                symbol=symbol,
                reason=(
                    f"Intraday RS loss (short): RRS={rrs:.2f} > "
                    f"{abs(self._rs_loss_threshold)}, trend={trend}"
                ),
                priority=1,
                action="close",
            )

        return None

    def _check_vwap_breakdown(
        self,
        symbol: str,
        stock_bars,
        current_price: float,
        direction: str,
    ) -> Optional[IntradayExitSignal]:
        """Check if price is below VWAP for N consecutive 5m bars."""
        try:
            # Filter to today's bars only — VWAP must reset daily
            today_bars = _filter_today_bars(stock_bars)
            if today_bars is None or len(today_bars) < 3:
                return None

            # Only update count once per unique bar to avoid inflation
            # when check_exits is called multiple times within the same bar
            last_bar_ts = today_bars.index[-1]
            prev_ts = self._vwap_last_bar_ts.get(symbol)
            if prev_ts is not None and last_bar_ts == prev_ts:
                # Same bar as last check — no update needed
                return None

            self._vwap_last_bar_ts[symbol] = last_bar_ts

            vwap_series = calculate_vwap(today_bars)
            if vwap_series is None or len(vwap_series) < 2:
                return None

            current_vwap = float(vwap_series.iloc[-1])
            if current_vwap <= 0:
                return None

            if direction == 'long':
                adverse = current_price < current_vwap
            else:
                # For shorts, "breakdown" means price above VWAP
                adverse = current_price > current_vwap

            if adverse:
                self._vwap_below_count[symbol] = self._vwap_below_count.get(symbol, 0) + 1
            else:
                self._vwap_below_count[symbol] = 0

            if self._vwap_below_count.get(symbol, 0) >= self._vwap_confirm_bars:
                side_label = "below" if direction == 'long' else "above"
                bars_count = self._vwap_below_count[symbol]
                # Reset counter after firing to prevent duplicate signals on next bar
                self._vwap_below_count[symbol] = 0
                return IntradayExitSignal(
                    symbol=symbol,
                    reason=(
                        f"VWAP breakdown: price {side_label} VWAP for "
                        f"{bars_count} bars "
                        f"(price=${current_price:.2f}, VWAP=${current_vwap:.2f})"
                    ),
                    priority=2,
                    action="close",
                )
        except Exception as e:
            logger.debug(f"VWAP check failed for {symbol}: {e}")

        return None

    def _check_time_stop(
        self,
        symbol: str,
        pos: Dict,
        current_price: float,
        entry_price: float,
        direction: str,
    ) -> Optional[IntradayExitSignal]:
        """Check if position has been unprofitable past the time limit."""
        entry_time = pos.get('entry_time') or self._tracked.get(symbol, {}).get('registered_at')
        if entry_time is None:
            return None

        if isinstance(entry_time, str):
            try:
                entry_time = datetime.fromisoformat(entry_time)
            except (ValueError, TypeError):
                return None

        # Use timezone-aware comparison to avoid naive/aware mismatch
        if entry_time.tzinfo is not None:
            now = datetime.now(tz=entry_time.tzinfo)
        else:
            now = datetime.now()

        elapsed_minutes = (now - entry_time).total_seconds() / 60
        if elapsed_minutes < self._time_stop_minutes:
            return None

        # Only trigger if position is unprofitable
        if direction == 'long':
            profitable = current_price > entry_price
        else:
            profitable = current_price < entry_price

        if profitable:
            return None

        return IntradayExitSignal(
            symbol=symbol,
            reason=(
                f"Time stop: unprofitable after {elapsed_minutes:.0f} min "
                f"(entry=${entry_price:.2f}, current=${current_price:.2f})"
            ),
            priority=3,
            action="close",
        )

    def _check_breakeven_stop(
        self,
        symbol: str,
        pos: Dict,
        current_price: float,
        entry_price: float,
        stop_price: float,
        direction: str,
    ) -> Optional[IntradayExitSignal]:
        """Check if position has reached +0.5R and should move stop to breakeven."""
        if self._breakeven_applied.get(symbol, False):
            return None

        if entry_price <= 0 or stop_price <= 0:
            return None

        initial_risk = abs(entry_price - stop_price)
        if initial_risk <= 0:
            return None

        if direction == 'long':
            unrealized_r = (current_price - entry_price) / initial_risk
        else:
            unrealized_r = (entry_price - current_price) / initial_risk

        if unrealized_r >= self._breakeven_r_threshold:
            self._breakeven_applied[symbol] = True
            return IntradayExitSignal(
                symbol=symbol,
                reason=(
                    f"Breakeven stop: reached +{unrealized_r:.2f}R "
                    f"(threshold={self._breakeven_r_threshold}R), "
                    f"moving stop to entry ${entry_price:.2f}"
                ),
                priority=4,
                action="tighten_stop",
                new_stop_price=entry_price,
            )

        return None
