"""
Enhanced Backtesting Engine
Adds trailing stops, scaled exits, and time-based exits for improved performance
"""

import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger

from risk import PositionSizer, RiskLimits
from shared.indicators.rrs import (
    RRSCalculator,
    check_daily_strength_relaxed,
    check_daily_weakness_relaxed
)


@dataclass
class EnhancedTrade:
    """Enhanced trade record with scaling and trailing stop tracking"""
    symbol: str
    direction: str
    entry_date: date
    entry_price: float
    shares: int = 0
    remaining_shares: int = 0  # For scaled exits
    stop_price: float = 0
    original_stop: float = 0  # Keep track of initial stop
    target_price: float = 0
    trailing_stop_price: float = 0  # Current trailing stop level
    breakeven_activated: bool = False
    scale_1_hit: bool = False
    scale_2_hit: bool = False

    # Exit tracking
    exit_date: Optional[date] = None
    exit_price: Optional[float] = None
    pnl: float = 0
    pnl_percent: float = 0
    exit_reason: str = ""

    # Metrics
    rrs_at_entry: float = 0
    atr_at_entry: float = 0
    holding_days: int = 0
    max_favorable_excursion: float = 0  # Max profit during trade
    max_adverse_excursion: float = 0  # Max drawdown during trade

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class EnhancedBacktestResult:
    """Enhanced results with detailed trade analytics"""
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_holding_days: float

    # Enhanced metrics
    trades_stopped_out: int = 0
    trades_target_hit: int = 0
    trades_trailing_stopped: int = 0
    trades_time_stopped: int = 0
    breakeven_activations: int = 0
    scale_1_exits: int = 0
    scale_2_exits: int = 0
    avg_mfe: float = 0  # Average max favorable excursion
    avg_mae: float = 0  # Average max adverse excursion

    trades: List[EnhancedTrade] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "avg_holding_days": self.avg_holding_days,
            "trades_stopped_out": self.trades_stopped_out,
            "trades_target_hit": self.trades_target_hit,
            "trades_trailing_stopped": self.trades_trailing_stopped,
            "breakeven_activations": self.breakeven_activations,
            "avg_mfe": self.avg_mfe,
            "avg_mae": self.avg_mae
        }


class EnhancedBacktestEngine:
    """
    Enhanced backtesting engine with advanced exit management

    Features:
    - Trailing stops (move stop to breakeven after 1R, then trail)
    - Scaled exits (take partial profits at multiple targets)
    - Time-based exits (close positions not moving after X days)
    - Maximum adverse/favorable excursion tracking
    """

    def __init__(
        self,
        initial_capital: float = 25000,
        risk_limits: Optional[RiskLimits] = None,

        # Entry parameters
        rrs_threshold: float = 2.0,
        max_positions: int = 5,
        use_relaxed_criteria: bool = True,

        # Stop/Target parameters
        stop_atr_multiplier: float = 0.75,
        target_atr_multiplier: float = 2.0,

        # Trailing stop parameters
        use_trailing_stop: bool = True,
        breakeven_trigger_r: float = 1.0,  # Move to BE after 1R profit
        trailing_atr_multiplier: float = 1.0,  # Trail by 1x ATR

        # Scaled exit parameters
        use_scaled_exits: bool = True,
        scale_1_target_r: float = 1.0,  # First scale at 1R
        scale_1_percent: float = 0.5,  # Take 50% off
        scale_2_target_r: float = 2.0,  # Second scale at 2R
        scale_2_percent: float = 0.25,  # Take another 25%

        # Time stop parameters
        use_time_stop: bool = True,
        max_holding_days: int = 10,  # Close if not profitable after X days
        stale_trade_days: int = 5,  # Close if price hasn't moved after X days
    ):
        self.initial_capital = initial_capital
        self.risk_limits = risk_limits or RiskLimits()

        # Entry params
        self.rrs_threshold = rrs_threshold
        self.max_positions = max_positions
        self.use_relaxed_criteria = use_relaxed_criteria

        # Exit params
        self.stop_atr_multiplier = stop_atr_multiplier
        self.target_atr_multiplier = target_atr_multiplier

        # Trailing stop
        self.use_trailing_stop = use_trailing_stop
        self.breakeven_trigger_r = breakeven_trigger_r
        self.trailing_atr_multiplier = trailing_atr_multiplier

        # Scaled exits
        self.use_scaled_exits = use_scaled_exits
        self.scale_1_target_r = scale_1_target_r
        self.scale_1_percent = scale_1_percent
        self.scale_2_target_r = scale_2_target_r
        self.scale_2_percent = scale_2_percent

        # Time stop
        self.use_time_stop = use_time_stop
        self.max_holding_days = max_holding_days
        self.stale_trade_days = stale_trade_days

        self.rrs_calculator = RRSCalculator()
        self.position_sizer = PositionSizer(self.risk_limits)

        # State
        self.capital = initial_capital
        self.positions: Dict[str, EnhancedTrade] = {}
        self.trades: List[EnhancedTrade] = []
        self.equity_curve: List[Dict] = []
        self.peak_capital = initial_capital

        # Metrics tracking
        self.breakeven_activations = 0
        self.scale_1_exits = 0
        self.scale_2_exits = 0

    def run(
        self,
        stock_data: Dict[str, pd.DataFrame],
        spy_data: pd.DataFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> EnhancedBacktestResult:
        """Run enhanced backtest"""
        # Reset state
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.peak_capital = self.initial_capital
        self.breakeven_activations = 0
        self.scale_1_exits = 0
        self.scale_2_exits = 0

        # Get date range
        all_dates = list(spy_data.index.date)
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        logger.info(f"Running enhanced backtest: {all_dates[0]} to {all_dates[-1]}")

        for current_date in all_dates:
            self._process_day(current_date, stock_data, spy_data)

        # Close remaining positions
        for symbol in list(self.positions.keys()):
            final_price = self._get_price(stock_data[symbol], all_dates[-1], "close")
            if final_price:
                self._close_position(symbol, final_price, all_dates[-1], "backtest_end")

        result = self._calculate_results(all_dates[0], all_dates[-1])
        logger.info(f"Enhanced backtest complete: {result.total_trades} trades, {result.win_rate:.1%} win rate, {result.total_return_pct:.2f}% return")

        return result

    def _process_day(
        self,
        current_date: date,
        stock_data: Dict[str, pd.DataFrame],
        spy_data: pd.DataFrame
    ):
        """Process a single trading day with enhanced exit logic"""
        # Update existing positions
        self._update_positions_enhanced(current_date, stock_data)

        # Look for new signals
        if len(self.positions) < self.max_positions:
            self._scan_for_signals(current_date, stock_data, spy_data)

        # Record equity
        position_value = sum(
            p.remaining_shares * self._get_price(stock_data[p.symbol], current_date, "close")
            for p in self.positions.values()
            if self._get_price(stock_data[p.symbol], current_date, "close")
        )
        total_equity = self.capital + position_value

        self.equity_curve.append({
            "date": current_date,
            "equity": total_equity,
            "positions": len(self.positions)
        })

        if total_equity > self.peak_capital:
            self.peak_capital = total_equity

    def _update_positions_enhanced(self, current_date: date, stock_data: Dict[str, pd.DataFrame]):
        """Update positions with trailing stops, scaled exits, and time stops"""
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            data = stock_data.get(symbol)
            if data is None:
                continue

            high = self._get_price(data, current_date, "high")
            low = self._get_price(data, current_date, "low")
            close = self._get_price(data, current_date, "close")

            if high is None or low is None or close is None:
                continue

            # Calculate current profit in R terms
            stop_distance = abs(position.entry_price - position.original_stop)
            if stop_distance == 0:
                continue

            if position.direction == "long":
                current_profit_r = (close - position.entry_price) / stop_distance
                current_profit = close - position.entry_price
            else:
                current_profit_r = (position.entry_price - close) / stop_distance
                current_profit = position.entry_price - close

            # Update MFE/MAE
            if current_profit > position.max_favorable_excursion:
                position.max_favorable_excursion = current_profit
            if current_profit < -position.max_adverse_excursion:
                position.max_adverse_excursion = abs(current_profit)

            # === STOP LOSS CHECK ===
            if position.direction == "long":
                if low <= position.stop_price:
                    self._close_position(symbol, position.stop_price, current_date, "stop_loss")
                    continue
            else:
                if high >= position.stop_price:
                    self._close_position(symbol, position.stop_price, current_date, "stop_loss")
                    continue

            # === SCALED EXIT 1 ===
            if self.use_scaled_exits and not position.scale_1_hit:
                scale_1_price = self._calculate_target_price(position, self.scale_1_target_r)
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
                    # Move stop to breakeven (entry price)
                    if position.direction == "long":
                        position.stop_price = position.entry_price
                    else:
                        position.stop_price = position.entry_price
                    position.breakeven_activated = True
                    self.breakeven_activations += 1
                    logger.debug(f"{symbol}: Breakeven stop activated at ${position.stop_price:.2f}")

            # === TRAILING STOP UPDATE ===
            if self.use_trailing_stop and position.breakeven_activated:
                atr = position.atr_at_entry
                trail_distance = atr * self.trailing_atr_multiplier

                if position.direction == "long":
                    new_trailing_stop = high - trail_distance
                    if new_trailing_stop > position.stop_price:
                        position.stop_price = new_trailing_stop
                        position.trailing_stop_price = new_trailing_stop
                else:
                    new_trailing_stop = low + trail_distance
                    if new_trailing_stop < position.stop_price:
                        position.stop_price = new_trailing_stop
                        position.trailing_stop_price = new_trailing_stop

            # === SCALED EXIT 2 ===
            if self.use_scaled_exits and position.scale_1_hit and not position.scale_2_hit:
                scale_2_price = self._calculate_target_price(position, self.scale_2_target_r)
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
                continue
            elif position.direction == "short" and low <= position.target_price:
                self._close_position(symbol, position.target_price, current_date, "take_profit")
                continue

            # === TIME STOP ===
            if self.use_time_stop:
                holding_days = (current_date - position.entry_date).days

                # Max holding days
                if holding_days >= self.max_holding_days:
                    self._close_position(symbol, close, current_date, "time_stop_max_days")
                    continue

                # Stale trade (not profitable after X days)
                if holding_days >= self.stale_trade_days and current_profit_r < 0.5:
                    self._close_position(symbol, close, current_date, "time_stop_stale")
                    continue

    def _calculate_target_price(self, position: EnhancedTrade, r_multiple: float) -> float:
        """Calculate target price for a given R multiple"""
        stop_distance = abs(position.entry_price - position.original_stop)
        target_distance = stop_distance * r_multiple

        if position.direction == "long":
            return position.entry_price + target_distance
        else:
            return position.entry_price - target_distance

    def _scale_out(
        self,
        position: EnhancedTrade,
        scale_percent: float,
        exit_price: float,
        exit_date: date,
        reason: str
    ):
        """Scale out of a portion of the position"""
        shares_to_sell = int(position.remaining_shares * scale_percent)
        if shares_to_sell == 0:
            shares_to_sell = 1

        if shares_to_sell >= position.remaining_shares:
            # Would close entire position
            return

        # Calculate P&L for scaled portion
        if position.direction == "long":
            pnl = (exit_price - position.entry_price) * shares_to_sell
        else:
            pnl = (position.entry_price - exit_price) * shares_to_sell

        # Return capital
        self.capital += (position.entry_price * shares_to_sell) + pnl
        position.remaining_shares -= shares_to_sell
        position.pnl += pnl

        logger.debug(f"{position.symbol}: Scaled out {shares_to_sell} shares at ${exit_price:.2f} ({reason}), P&L: ${pnl:.2f}")

    def _scan_for_signals(
        self,
        current_date: date,
        stock_data: Dict[str, pd.DataFrame],
        spy_data: pd.DataFrame
    ):
        """Scan for entry signals"""
        spy_current = spy_data[spy_data.index.date <= current_date]
        if len(spy_current) < 20:
            return

        close_col = 'Close' if 'Close' in spy_current.columns else 'close'
        spy_close = spy_current[close_col].iloc[-1]
        spy_prev_close = spy_current[close_col].iloc[-2]

        for symbol, data in stock_data.items():
            if symbol in self.positions:
                continue

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

                if abs(rrs) < self.rrs_threshold:
                    continue

                if self.use_relaxed_criteria:
                    daily_strength = check_daily_strength_relaxed(current_data_lower)
                    daily_weakness = check_daily_weakness_relaxed(current_data_lower)
                else:
                    from shared.indicators.rrs import check_daily_strength, check_daily_weakness
                    daily_strength = check_daily_strength(current_data_lower)
                    daily_weakness = check_daily_weakness(current_data_lower)

                direction = None
                if rrs > self.rrs_threshold and daily_strength["is_strong"]:
                    direction = "long"
                elif rrs < -self.rrs_threshold and daily_weakness["is_weak"]:
                    direction = "short"

                if direction:
                    self._enter_position(
                        symbol=symbol,
                        direction=direction,
                        entry_price=stock_close,
                        atr=atr,
                        entry_date=current_date,
                        rrs=rrs
                    )

            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
                continue

    def _enter_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        atr: float,
        entry_date: date,
        rrs: float
    ):
        """Enter a new position"""
        sizing = self.position_sizer.calculate_position_size(
            account_size=self.capital,
            entry_price=entry_price,
            atr=atr,
            direction=direction,
            stop_multiplier=self.stop_atr_multiplier,
            target_multiplier=self.target_atr_multiplier
        )

        if sizing.shares == 0:
            return

        required = sizing.shares * entry_price
        if required > self.capital:
            return

        trade = EnhancedTrade(
            symbol=symbol,
            direction=direction,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=sizing.shares,
            remaining_shares=sizing.shares,
            stop_price=sizing.stop_price,
            original_stop=sizing.stop_price,
            target_price=sizing.target_price,
            rrs_at_entry=rrs,
            atr_at_entry=atr
        )

        self.positions[symbol] = trade
        self.capital -= required

        logger.debug(f"Entered {direction} {symbol} @ ${entry_price:.2f}, Stop: ${sizing.stop_price:.2f}, Target: ${sizing.target_price:.2f}")

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_date: date,
        reason: str
    ):
        """Close a position completely"""
        if symbol not in self.positions:
            return

        trade = self.positions[symbol]
        trade.exit_date = exit_date
        trade.exit_price = exit_price
        trade.exit_reason = reason

        # Calculate remaining P&L
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

        # Return capital
        self.capital += (trade.entry_price * trade.remaining_shares) + remaining_pnl

        self.trades.append(trade)
        del self.positions[symbol]

        logger.debug(f"Closed {symbol} @ ${exit_price:.2f} ({reason}), Total P&L: ${trade.pnl:.2f}")

    def _get_price(self, data: pd.DataFrame, target_date: date, column: str) -> Optional[float]:
        """Get price for a specific date"""
        try:
            day_data = data[data.index.date == target_date]
            if len(day_data) > 0:
                col = column.capitalize() if column.lower() in ['open', 'high', 'low', 'close'] else column
                return float(day_data[col].iloc[0])
        except:
            pass
        return None

    def _calculate_results(self, start_date: date, end_date: date) -> EnhancedBacktestResult:
        """Calculate enhanced backtest results"""
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
                equity_curve=self.equity_curve
            )

        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]

        total_wins = sum(t.pnl for t in winners)
        total_losses = abs(sum(t.pnl for t in losers))

        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Exit reason counts
        trades_stopped_out = len([t for t in self.trades if "stop_loss" in t.exit_reason])
        trades_target_hit = len([t for t in self.trades if "take_profit" in t.exit_reason])
        trades_trailing_stopped = len([t for t in self.trades if t.trailing_stop_price > 0 and "stop" in t.exit_reason])
        trades_time_stopped = len([t for t in self.trades if "time_stop" in t.exit_reason])

        # MFE/MAE
        avg_mfe = sum(t.max_favorable_excursion for t in self.trades) / total_trades
        avg_mae = sum(t.max_adverse_excursion for t in self.trades) / total_trades

        # Max drawdown
        max_dd = 0
        peak = self.initial_capital
        for point in self.equity_curve:
            if point["equity"] > peak:
                peak = point["equity"]
            dd = peak - point["equity"]
            if dd > max_dd:
                max_dd = dd

        # Sharpe
        returns = [t.pnl_percent for t in self.trades]
        avg_return = sum(returns) / len(returns) if returns else 0
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5 if len(returns) > 1 else 0
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
            equity_curve=self.equity_curve
        )


def compare_engines(
    stock_data: Dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
    initial_capital: float = 25000
) -> Dict:
    """Compare standard vs enhanced engine performance"""
    from backtesting.engine import BacktestEngine

    # Standard engine
    standard_engine = BacktestEngine(
        initial_capital=initial_capital,
        rrs_threshold=2.0,
        use_relaxed_criteria=True,
        stop_atr_multiplier=0.75,
        target_atr_multiplier=1.5
    )
    standard_result = standard_engine.run(stock_data, spy_data)

    # Enhanced engine
    enhanced_engine = EnhancedBacktestEngine(
        initial_capital=initial_capital,
        rrs_threshold=2.0,
        use_relaxed_criteria=True,
        stop_atr_multiplier=0.75,
        target_atr_multiplier=2.0,
        use_trailing_stop=True,
        use_scaled_exits=True,
        use_time_stop=True
    )
    enhanced_result = enhanced_engine.run(stock_data, spy_data)

    comparison = {
        "standard": {
            "return_pct": standard_result.total_return_pct,
            "win_rate": standard_result.win_rate,
            "profit_factor": standard_result.profit_factor,
            "max_drawdown_pct": standard_result.max_drawdown_pct,
            "total_trades": standard_result.total_trades
        },
        "enhanced": {
            "return_pct": enhanced_result.total_return_pct,
            "win_rate": enhanced_result.win_rate,
            "profit_factor": enhanced_result.profit_factor,
            "max_drawdown_pct": enhanced_result.max_drawdown_pct,
            "total_trades": enhanced_result.total_trades,
            "breakeven_activations": enhanced_result.breakeven_activations,
            "scale_1_exits": enhanced_result.scale_1_exits
        },
        "improvement": {
            "return_delta": enhanced_result.total_return_pct - standard_result.total_return_pct,
            "win_rate_delta": enhanced_result.win_rate - standard_result.win_rate,
            "profit_factor_delta": enhanced_result.profit_factor - standard_result.profit_factor
        }
    }

    return comparison
