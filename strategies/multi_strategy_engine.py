"""
Multi-Strategy Trading Engine

Orchestrates multiple trading strategies to maximize returns
through diversification and capital efficiency.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Type
import pandas as pd
from loguru import logger

from strategies.base_strategy import (
    BaseStrategy,
    StrategySignal,
    StrategyResult,
    SignalDirection,
    MomentumStrategy
)
from strategies.leveraged_etf import LeveragedETFStrategy, SectorRotationStrategy
from strategies.kelly_sizer import KellyCriterionSizer, VolatilityAdjustedSizer
from risk.models import RiskLimits


@dataclass
class PortfolioState:
    """Current state of the portfolio across all strategies"""
    total_capital: float
    cash: float
    total_exposure: float
    positions: Dict[str, Dict] = field(default_factory=dict)
    daily_pnl: float = 0
    weekly_pnl: float = 0
    open_risk: float = 0

    @property
    def exposure_pct(self) -> float:
        return self.total_exposure / self.total_capital if self.total_capital > 0 else 0


@dataclass
class MultiStrategyResult:
    """Combined results from all strategies"""
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float

    # Strategy breakdown
    strategy_results: Dict[str, StrategyResult] = field(default_factory=dict)

    # Aggregate metrics
    total_trades: int = 0
    overall_win_rate: float = 0
    overall_profit_factor: float = 0
    max_drawdown: float = 0
    max_drawdown_pct: float = 0
    sharpe_ratio: float = 0

    # Contribution analysis
    strategy_contributions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'start_date': str(self.start_date),
            'end_date': str(self.end_date),
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'total_return_pct': self.total_return_pct,
            'total_trades': self.total_trades,
            'overall_win_rate': self.overall_win_rate,
            'max_drawdown_pct': self.max_drawdown_pct,
            'strategy_contributions': self.strategy_contributions
        }


class MultiStrategyEngine:
    """
    Engine that runs multiple trading strategies simultaneously

    Handles:
    - Capital allocation across strategies
    - Signal aggregation and conflict resolution
    - Portfolio-level risk management
    - Performance tracking per strategy
    """

    def __init__(
        self,
        initial_capital: float = 25000,
        risk_limits: Optional[RiskLimits] = None,
        use_kelly: bool = True,
        use_volatility_adjustment: bool = True
    ):
        self.initial_capital = initial_capital
        self.risk_limits = risk_limits or RiskLimits(
            max_risk_per_trade=0.03,
            max_daily_loss=0.06,
            max_open_positions=20
        )

        # Position sizing
        self.use_kelly = use_kelly
        self.kelly_sizer = KellyCriterionSizer() if use_kelly else None
        self.vol_sizer = VolatilityAdjustedSizer() if use_volatility_adjustment else None

        # Strategies
        self.strategies: Dict[str, BaseStrategy] = {}

        # State
        self.portfolio = PortfolioState(
            total_capital=initial_capital,
            cash=initial_capital,
            total_exposure=0
        )

        # Tracking
        self.all_trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.peak_capital = initial_capital

    def add_strategy(self, strategy: BaseStrategy):
        """Add a strategy to the engine"""
        self.strategies[strategy.name] = strategy
        logger.info(
            f"Added strategy: {strategy.name} "
            f"(allocation={strategy.capital_allocation*100}%)"
        )

    def remove_strategy(self, name: str):
        """Remove a strategy"""
        if name in self.strategies:
            del self.strategies[name]
            logger.info(f"Removed strategy: {name}")

    def setup_default_strategies(self):
        """Setup recommended strategy combination for 100% return goal"""

        # Strategy 1: Core RRS Momentum (40% allocation)
        momentum = MomentumStrategy(
            name="RRS_Momentum",
            capital_allocation=0.40,
            max_positions=10,
            risk_per_trade=0.03,
            rrs_threshold=1.75,
            stop_atr_mult=0.75,
            target_atr_mult=1.5
        )
        self.add_strategy(momentum)

        # Strategy 2: Leveraged ETFs (25% allocation)
        leveraged = LeveragedETFStrategy(
            name="Leveraged_ETF",
            capital_allocation=0.25,
            max_positions=4,
            risk_per_trade=0.02,
            rrs_threshold=1.5,
            use_inverse=True
        )
        self.add_strategy(leveraged)

        # Strategy 3: Sector Rotation (20% allocation)
        sector = SectorRotationStrategy(
            name="Sector_Rotation",
            capital_allocation=0.20,
            max_positions=4,
            risk_per_trade=0.02,
            top_n=2
        )
        self.add_strategy(sector)

        # Reserve 15% as cash buffer for opportunities/drawdowns

        logger.info(
            f"Setup {len(self.strategies)} strategies with "
            f"{sum(s.capital_allocation for s in self.strategies.values())*100}% allocation"
        )

    def run_backtest(
        self,
        stock_data: Dict[str, pd.DataFrame],
        spy_data: pd.DataFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> MultiStrategyResult:
        """
        Run backtest across all strategies

        Args:
            stock_data: Historical OHLCV data per symbol
            spy_data: SPY historical data
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            MultiStrategyResult with combined performance
        """
        # Reset state
        self._reset_state()

        # Get date range
        all_dates = sorted(list(set(spy_data.index.date)))
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        if not all_dates:
            logger.error("No dates in backtest range")
            return self._empty_result(start_date or date.today(), end_date or date.today())

        logger.info(f"Running multi-strategy backtest: {all_dates[0]} to {all_dates[-1]}")
        logger.info(f"Strategies: {list(self.strategies.keys())}")

        # Process each day
        for current_date in all_dates:
            self._process_day(current_date, stock_data, spy_data)

        # Close remaining positions at end
        for symbol in list(self.portfolio.positions.keys()):
            final_price = self._get_price(stock_data, symbol, all_dates[-1], 'close')
            if final_price:
                self._close_position(symbol, final_price, all_dates[-1], 'backtest_end')

        # Calculate results
        result = self._calculate_results(all_dates[0], all_dates[-1])

        logger.info(
            f"Backtest complete: {result.total_trades} trades, "
            f"{result.total_return_pct:.2f}% return, "
            f"{result.max_drawdown_pct:.1f}% max drawdown"
        )

        return result

    def _reset_state(self):
        """Reset engine state for new backtest"""
        self.portfolio = PortfolioState(
            total_capital=self.initial_capital,
            cash=self.initial_capital,
            total_exposure=0
        )
        self.all_trades = []
        self.equity_curve = []
        self.peak_capital = self.initial_capital

        for strategy in self.strategies.values():
            strategy.reset()

    def _process_day(
        self,
        current_date: date,
        stock_data: Dict[str, pd.DataFrame],
        spy_data: pd.DataFrame
    ):
        """Process a single trading day"""

        # Update existing positions
        self._update_positions(current_date, stock_data)

        # Scan for new signals from all strategies
        all_signals = []
        for strategy in self.strategies.values():
            if not strategy.is_active:
                continue

            try:
                signals = strategy.scan(stock_data, spy_data, current_date)
                all_signals.extend(signals)
            except Exception as e:
                logger.debug(f"Strategy {strategy.name} scan error: {e}")

        # Filter and prioritize signals
        filtered_signals = self._filter_signals(all_signals)

        # Execute trades for top signals
        for signal in filtered_signals[:5]:  # Max 5 new trades per day
            self._execute_signal(signal, stock_data, current_date)

        # Record equity
        self._record_equity(current_date, stock_data)

    def _update_positions(
        self,
        current_date: date,
        stock_data: Dict[str, pd.DataFrame]
    ):
        """Check all positions for exits"""
        for symbol in list(self.portfolio.positions.keys()):
            position = self.portfolio.positions[symbol]
            strategy_name = position.get('strategy', '')
            strategy = self.strategies.get(strategy_name)

            if not strategy:
                continue

            if symbol not in stock_data:
                continue

            current_data = stock_data[symbol]

            # Check strategy-specific exit
            exit_reason = strategy.should_exit(symbol, position, current_data, current_date)

            if exit_reason:
                exit_price = self._get_price(stock_data, symbol, current_date, 'close')
                if exit_price:
                    self._close_position(symbol, exit_price, current_date, exit_reason)

    def _filter_signals(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """Filter and prioritize signals"""
        # Remove duplicates (same symbol from different strategies)
        seen_symbols = set(self.portfolio.positions.keys())
        unique_signals = []

        for signal in signals:
            if signal.symbol in seen_symbols:
                continue
            if not signal.is_valid:
                continue
            seen_symbols.add(signal.symbol)
            unique_signals.append(signal)

        # Sort by strength and risk/reward
        unique_signals.sort(
            key=lambda s: (s.strength.value, s.risk_reward_ratio),
            reverse=True
        )

        return unique_signals

    def _execute_signal(
        self,
        signal: StrategySignal,
        stock_data: Dict[str, pd.DataFrame],
        current_date: date
    ):
        """Execute a trading signal"""
        strategy = self.strategies.get(signal.strategy_name)
        if not strategy:
            return

        # Check if strategy can take new position
        if not strategy.can_take_new_position():
            return

        # Check portfolio limits
        if len(self.portfolio.positions) >= self.risk_limits.max_open_positions:
            return

        # Check daily loss limit
        if self.portfolio.daily_pnl <= -(self.portfolio.total_capital * self.risk_limits.max_daily_loss):
            logger.warning("Daily loss limit reached, no new trades")
            return

        # Calculate position size
        allocated_capital = strategy.get_allocation(self.portfolio.total_capital)
        shares = strategy.calculate_position_size(signal, allocated_capital)

        if shares == 0:
            return

        # Calculate required capital
        required_capital = shares * signal.entry_price

        if required_capital > self.portfolio.cash:
            # Reduce size to fit available cash
            shares = int(self.portfolio.cash / signal.entry_price)
            if shares == 0:
                return
            required_capital = shares * signal.entry_price

        # Execute trade
        position = {
            'symbol': signal.symbol,
            'direction': signal.direction.value,
            'entry_date': current_date,
            'entry_price': signal.entry_price,
            'shares': shares,
            'stop_price': signal.stop_price,
            'target_price': signal.target_price,
            'strategy': signal.strategy_name,
            'rrs': signal.rrs_value,
            'position_value': required_capital
        }

        # Update portfolio
        self.portfolio.positions[signal.symbol] = position
        self.portfolio.cash -= required_capital
        self.portfolio.total_exposure += required_capital
        self.portfolio.open_risk += shares * signal.risk_per_share

        # Update strategy tracking
        strategy.add_position(signal.symbol, position)

        logger.debug(
            f"Opened {signal.direction.value} {signal.symbol} "
            f"@ ${signal.entry_price:.2f} x {shares} shares "
            f"({signal.strategy_name})"
        )

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_date: date,
        exit_reason: str
    ):
        """Close a position"""
        if symbol not in self.portfolio.positions:
            return

        position = self.portfolio.positions[symbol]
        entry_price = position['entry_price']
        shares = position['shares']
        direction = position['direction']

        # Calculate P&L
        if direction == 'long':
            pnl = (exit_price - entry_price) * shares
        else:
            pnl = (entry_price - exit_price) * shares

        pnl_pct = (pnl / position['position_value']) * 100 if position['position_value'] > 0 else 0

        # Record trade
        trade = {
            'symbol': symbol,
            'direction': direction,
            'entry_date': position['entry_date'],
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'shares': shares,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'strategy': position['strategy'],
            'holding_days': (exit_date - position['entry_date']).days
        }
        self.all_trades.append(trade)

        # Update portfolio
        self.portfolio.cash += position['position_value'] + pnl
        self.portfolio.total_exposure -= position['position_value']
        self.portfolio.daily_pnl += pnl

        # Update strategy
        strategy = self.strategies.get(position['strategy'])
        if strategy:
            strategy.remove_position(symbol)

        # Remove position
        del self.portfolio.positions[symbol]

        logger.debug(
            f"Closed {symbol} @ ${exit_price:.2f} "
            f"P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%) "
            f"Reason: {exit_reason}"
        )

    def _record_equity(self, current_date: date, stock_data: Dict[str, pd.DataFrame]):
        """Record equity curve point"""
        # Calculate current position values
        position_value = 0
        for symbol, position in self.portfolio.positions.items():
            current_price = self._get_price(stock_data, symbol, current_date, 'close')
            if current_price:
                if position['direction'] == 'long':
                    position_value += current_price * position['shares']
                else:
                    # Short position
                    position_value += (
                        position['entry_price'] * 2 - current_price
                    ) * position['shares']

        total_equity = self.portfolio.cash + position_value

        self.equity_curve.append({
            'date': current_date,
            'equity': total_equity,
            'cash': self.portfolio.cash,
            'positions': len(self.portfolio.positions),
            'exposure_pct': (position_value / total_equity * 100) if total_equity > 0 else 0
        })

        # Track peak
        if total_equity > self.peak_capital:
            self.peak_capital = total_equity

        # Update total capital
        self.portfolio.total_capital = total_equity

    def _get_price(
        self,
        stock_data: Dict[str, pd.DataFrame],
        symbol: str,
        target_date: date,
        price_type: str = 'close'
    ) -> Optional[float]:
        """Get price for a symbol on a date"""
        if symbol not in stock_data:
            return None

        data = stock_data[symbol]
        try:
            day_data = data[data.index.date == target_date]
            if len(day_data) > 0:
                col = price_type.capitalize() if price_type.lower() in ['open', 'high', 'low', 'close'] else price_type
                return float(day_data[col].iloc[0])
        except Exception:
            pass
        return None

    def _calculate_results(self, start_date: date, end_date: date) -> MultiStrategyResult:
        """Calculate comprehensive results"""
        if not self.equity_curve:
            return self._empty_result(start_date, end_date)

        final_equity = self.equity_curve[-1]['equity']
        total_return = final_equity - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        # Calculate drawdown
        max_dd = 0
        peak = self.initial_capital
        for point in self.equity_curve:
            if point['equity'] > peak:
                peak = point['equity']
            dd = peak - point['equity']
            if dd > max_dd:
                max_dd = dd

        # Trade statistics
        total_trades = len(self.all_trades)
        winners = [t for t in self.all_trades if t['pnl'] > 0]
        losers = [t for t in self.all_trades if t['pnl'] <= 0]

        win_rate = len(winners) / total_trades if total_trades > 0 else 0
        total_wins = sum(t['pnl'] for t in winners)
        total_losses = abs(sum(t['pnl'] for t in losers))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Calculate Sharpe
        returns = [t['pnl_pct'] for t in self.all_trades]
        if len(returns) > 1:
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe = (avg_return / std_return) if std_return > 0 else 0
        else:
            sharpe = 0

        # Strategy contributions
        strategy_contributions = {}
        for name in self.strategies.keys():
            strategy_trades = [t for t in self.all_trades if t['strategy'] == name]
            contribution = sum(t['pnl'] for t in strategy_trades)
            strategy_contributions[name] = contribution

        return MultiStrategyResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            overall_win_rate=win_rate,
            overall_profit_factor=profit_factor,
            max_drawdown=max_dd,
            max_drawdown_pct=(max_dd / self.peak_capital * 100) if self.peak_capital > 0 else 0,
            sharpe_ratio=sharpe,
            strategy_contributions=strategy_contributions
        )

    def _empty_result(self, start_date: date, end_date: date) -> MultiStrategyResult:
        """Return empty result"""
        return MultiStrategyResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_return=0,
            total_return_pct=0
        )

    def print_summary(self, result: MultiStrategyResult):
        """Print comprehensive summary"""
        print("\n" + "=" * 70)
        print("MULTI-STRATEGY BACKTEST RESULTS")
        print("=" * 70)
        print(f"Period: {result.start_date} to {result.end_date}")
        print(f"Strategies: {', '.join(self.strategies.keys())}")
        print("-" * 70)
        print(f"Initial Capital: ${result.initial_capital:,.2f}")
        print(f"Final Capital:   ${result.final_capital:,.2f}")
        print(f"Total Return:    ${result.total_return:,.2f} ({result.total_return_pct:+.2f}%)")
        print("-" * 70)
        print(f"Total Trades: {result.total_trades}")
        print(f"Win Rate: {result.overall_win_rate:.1%}")
        print(f"Profit Factor: {result.overall_profit_factor:.2f}")
        print(f"Max Drawdown: {result.max_drawdown_pct:.1f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print("-" * 70)
        print("STRATEGY CONTRIBUTIONS:")
        for name, contribution in result.strategy_contributions.items():
            pct = (contribution / result.initial_capital * 100) if result.initial_capital > 0 else 0
            print(f"  {name}: ${contribution:+,.2f} ({pct:+.2f}%)")
        print("=" * 70)


def run_multi_strategy_backtest(
    watchlist: List[str] = None,
    days: int = 365,
    initial_capital: float = 25000
) -> MultiStrategyResult:
    """
    Convenience function to run multi-strategy backtest

    Args:
        watchlist: List of symbols to trade
        days: Number of days to backtest
        initial_capital: Starting capital

    Returns:
        MultiStrategyResult
    """
    from backtesting.data_loader import DataLoader
    from config.watchlists import get_full_watchlist, LEVERAGED_ETFS, SECTOR_ETFS

    # Build comprehensive watchlist
    if watchlist is None:
        watchlist = get_full_watchlist()

    # Add leveraged ETFs and sector ETFs
    from strategies.leveraged_etf import LEVERAGED_ETFS as LEV_ETFS, UNDERLYING_ETFS
    all_symbols = set(watchlist)
    all_symbols.update(LEV_ETFS.keys())
    all_symbols.update(UNDERLYING_ETFS)
    all_symbols.update(SectorRotationStrategy.SECTOR_ETFS.keys())
    watchlist = list(all_symbols)

    # Load data
    loader = DataLoader(cache_dir="data/historical")
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    logger.info(f"Loading data for {len(watchlist)} symbols...")
    stock_data = loader.load_stock_data(watchlist, start_date, end_date)
    spy_data = loader.load_spy_data(start_date, end_date)

    # Setup engine
    engine = MultiStrategyEngine(initial_capital=initial_capital)
    engine.setup_default_strategies()

    # Run backtest
    result = engine.run_backtest(stock_data, spy_data, start_date, end_date)

    # Print results
    engine.print_summary(result)

    return result


if __name__ == "__main__":
    result = run_multi_strategy_backtest(days=365)
