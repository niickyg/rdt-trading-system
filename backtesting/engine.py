"""
Backtesting Engine
Run historical simulations of trading strategies
"""

import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from loguru import logger

from risk import RiskManager, PositionSizer, RiskLimits
from shared.indicators.rrs import (
    RRSCalculator,
    check_daily_strength,
    check_daily_weakness,
    check_daily_strength_relaxed,
    check_daily_weakness_relaxed
)


@dataclass
class BacktestTrade:
    """Record of a backtested trade"""
    symbol: str
    direction: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    shares: int = 0
    stop_price: float = 0
    target_price: float = 0
    pnl: float = 0
    pnl_percent: float = 0
    exit_reason: str = ""
    rrs_at_entry: float = 0
    holding_days: int = 0

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestResult:
    """Results of a backtest run"""
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
    trades: List[BacktestTrade] = field(default_factory=list)
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
            "avg_holding_days": self.avg_holding_days
        }


class BacktestEngine:
    """
    Backtesting engine for RDT trading strategy

    Runs historical simulations to evaluate strategy performance.
    """

    def __init__(
        self,
        initial_capital: float = 25000,
        risk_limits: Optional[RiskLimits] = None,
        rrs_threshold: float = 1.5,  # Lowered from 2.0 based on Hari's methodology
        max_positions: int = 5,
        use_relaxed_criteria: bool = True,  # Use relaxed daily chart checks
        stop_atr_multiplier: float = 1.0,  # Tighter stops per RDT methodology
        target_atr_multiplier: float = 2.0  # 2:1 R/R minimum
    ):
        self.initial_capital = initial_capital
        self.risk_limits = risk_limits or RiskLimits()
        self.rrs_threshold = rrs_threshold
        self.max_positions = max_positions
        self.use_relaxed_criteria = use_relaxed_criteria
        self.stop_atr_multiplier = stop_atr_multiplier
        self.target_atr_multiplier = target_atr_multiplier

        self.rrs_calculator = RRSCalculator()
        self.position_sizer = PositionSizer(self.risk_limits)

        # State during backtest
        self.capital = initial_capital
        self.positions: Dict[str, BacktestTrade] = {}
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict] = []
        self.peak_capital = initial_capital

    def run(
        self,
        stock_data: Dict[str, pd.DataFrame],
        spy_data: pd.DataFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> BacktestResult:
        """
        Run backtest on historical data

        Args:
            stock_data: Dict of symbol -> OHLCV DataFrame
            spy_data: SPY OHLCV DataFrame
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            BacktestResult with performance metrics
        """
        # Reset state
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.peak_capital = self.initial_capital

        # Get date range
        all_dates = spy_data.index.date
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        logger.info(f"Running backtest: {all_dates[0]} to {all_dates[-1]}")

        # Iterate through each day
        for current_date in all_dates:
            self._process_day(current_date, stock_data, spy_data)

        # Close any remaining positions at end
        for symbol in list(self.positions.keys()):
            final_price = self._get_price(stock_data[symbol], all_dates[-1], "close")
            if final_price:
                self._close_position(symbol, final_price, all_dates[-1], "backtest_end")

        # Calculate results
        result = self._calculate_results(all_dates[0], all_dates[-1])
        logger.info(f"Backtest complete: {result.total_trades} trades, {result.win_rate:.1%} win rate")

        return result

    def _process_day(
        self,
        current_date: date,
        stock_data: Dict[str, pd.DataFrame],
        spy_data: pd.DataFrame
    ):
        """Process a single trading day"""
        # Update existing positions
        self._update_positions(current_date, stock_data)

        # Look for new signals
        if len(self.positions) < self.max_positions:
            self._scan_for_signals(current_date, stock_data, spy_data)

        # Record equity
        position_value = sum(
            p.shares * self._get_price(stock_data[p.symbol], current_date, "close")
            for p in self.positions.values()
            if self._get_price(stock_data[p.symbol], current_date, "close")
        )
        total_equity = self.capital + position_value

        self.equity_curve.append({
            "date": current_date,
            "equity": total_equity,
            "positions": len(self.positions)
        })

        # Track peak for drawdown
        if total_equity > self.peak_capital:
            self.peak_capital = total_equity

    def _update_positions(self, current_date: date, stock_data: Dict[str, pd.DataFrame]):
        """Check stops and targets for open positions"""
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            data = stock_data.get(symbol)
            if data is None:
                continue

            high = self._get_price(data, current_date, "high")
            low = self._get_price(data, current_date, "low")
            close = self._get_price(data, current_date, "close")

            if high is None or low is None:
                continue

            # Check stop hit
            if position.direction == "long":
                if low <= position.stop_price:
                    self._close_position(symbol, position.stop_price, current_date, "stop_loss")
                    continue
                if high >= position.target_price:
                    self._close_position(symbol, position.target_price, current_date, "take_profit")
                    continue
            else:  # short
                if high >= position.stop_price:
                    self._close_position(symbol, position.stop_price, current_date, "stop_loss")
                    continue
                if low <= position.target_price:
                    self._close_position(symbol, position.target_price, current_date, "take_profit")
                    continue

    def _scan_for_signals(
        self,
        current_date: date,
        stock_data: Dict[str, pd.DataFrame],
        spy_data: pd.DataFrame
    ):
        """Scan for entry signals"""
        # Get SPY data up to current date
        spy_current = spy_data[spy_data.index.date <= current_date]
        if len(spy_current) < 20:
            return

        # Handle both uppercase and lowercase column names
        close_col = 'Close' if 'Close' in spy_current.columns else 'close'
        spy_close = spy_current[close_col].iloc[-1]
        spy_prev_close = spy_current[close_col].iloc[-2]

        for symbol, data in stock_data.items():
            if symbol in self.positions:
                continue

            # Get data up to current date
            current_data = data[data.index.date <= current_date]
            if len(current_data) < 20:
                continue

            try:
                # Normalize column names to lowercase for indicator functions
                current_data_lower = current_data.copy()
                current_data_lower.columns = [c.lower() for c in current_data_lower.columns]

                # Calculate ATR
                atr = self.rrs_calculator.calculate_atr(current_data_lower).iloc[-1]

                # Calculate RRS
                stock_close = current_data_lower['close'].iloc[-1]
                stock_prev_close = current_data_lower['close'].iloc[-2]

                rrs_result = self.rrs_calculator.calculate_rrs_current(
                    stock_data={"current_price": stock_close, "previous_close": stock_prev_close},
                    spy_data={"current_price": spy_close, "previous_close": spy_prev_close},
                    stock_atr=atr
                )

                rrs = rrs_result["rrs"]

                # Check threshold
                if abs(rrs) < self.rrs_threshold:
                    continue

                # Check daily chart (use lowercase column data)
                # Use relaxed or strict criteria based on configuration
                if self.use_relaxed_criteria:
                    daily_strength = check_daily_strength_relaxed(current_data_lower)
                    daily_weakness = check_daily_weakness_relaxed(current_data_lower)
                else:
                    daily_strength = check_daily_strength(current_data_lower)
                    daily_weakness = check_daily_weakness(current_data_lower)

                # Determine direction
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
        # Calculate position size with configured stop/target multipliers
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

        # Check if we have enough capital
        required = sizing.shares * entry_price
        if required > self.capital:
            return

        # Create trade
        trade = BacktestTrade(
            symbol=symbol,
            direction=direction,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=sizing.shares,
            stop_price=sizing.stop_price,
            target_price=sizing.target_price,
            rrs_at_entry=rrs
        )

        self.positions[symbol] = trade
        self.capital -= required

        logger.debug(f"Entered {direction} {symbol} @ ${entry_price:.2f}")

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_date: date,
        reason: str
    ):
        """Close a position"""
        if symbol not in self.positions:
            return

        trade = self.positions[symbol]
        trade.exit_date = exit_date
        trade.exit_price = exit_price
        trade.exit_reason = reason

        # Calculate P&L
        if trade.direction == "long":
            trade.pnl = (exit_price - trade.entry_price) * trade.shares
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.shares

        trade.pnl_percent = (trade.pnl / (trade.entry_price * trade.shares)) * 100
        trade.holding_days = (exit_date - trade.entry_date.date()).days if isinstance(trade.entry_date, datetime) else (exit_date - trade.entry_date).days

        # Return capital
        self.capital += (trade.entry_price * trade.shares) + trade.pnl

        # Move to completed trades
        self.trades.append(trade)
        del self.positions[symbol]

        logger.debug(f"Closed {symbol} @ ${exit_price:.2f} P&L: ${trade.pnl:.2f}")

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

    def _calculate_results(self, start_date: date, end_date: date) -> BacktestResult:
        """Calculate backtest results"""
        total_trades = len(self.trades)

        if total_trades == 0:
            return BacktestResult(
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

        # Max drawdown
        max_dd = 0
        peak = self.initial_capital
        for point in self.equity_curve:
            if point["equity"] > peak:
                peak = point["equity"]
            dd = peak - point["equity"]
            if dd > max_dd:
                max_dd = dd

        # Simple Sharpe (would need daily returns for proper calculation)
        returns = [t.pnl_percent for t in self.trades]
        avg_return = sum(returns) / len(returns) if returns else 0
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5 if len(returns) > 1 else 0
        sharpe = (avg_return / std_return) if std_return > 0 else 0

        return BacktestResult(
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
            max_drawdown_pct=(max_dd / self.peak_capital) * 100,
            sharpe_ratio=sharpe,
            avg_holding_days=sum(t.holding_days for t in self.trades) / total_trades,
            trades=self.trades,
            equity_curve=self.equity_curve
        )
