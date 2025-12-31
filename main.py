#!/usr/bin/env python3
"""
RDT Trading System - Main Entry Point

Run the autonomous trading system with various modes:
- scanner: Semi-automated signal scanning
- bot: Fully automated trading (paper or live)
- backtest: Run historical backtest
- dashboard: Monitor the system
"""

import os
import sys
import asyncio
import argparse
from datetime import date, timedelta
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_settings
from brokers import get_broker
from risk import RiskManager, RiskLimits
from portfolio import PositionManager
from shared.data_provider import DataProvider
from agents import Orchestrator, run_trading_system
from monitoring import TradingDashboard


def setup_logging(level: str = "INFO"):
    """Configure logging"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"
    )
    logger.add(
        "data/logs/trading_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG"
    )


def get_default_watchlist(size: str = "full") -> list:
    """
    Default watchlist of liquid stocks

    Args:
        size: 'core' (50 stocks), 'full' (150+ stocks), or sector name

    Returns:
        List of ticker symbols
    """
    try:
        from config.watchlists import get_watchlist_by_name
        return get_watchlist_by_name(size)
    except ImportError:
        # Fallback if watchlists module not available
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            'JPM', 'V', 'JNJ', 'UNH', 'HD', 'PG', 'MA', 'DIS',
            'PYPL', 'ADBE', 'CRM', 'NFLX', 'AMD', 'INTC', 'CSCO',
            'PEP', 'KO', 'MRK', 'ABT', 'TMO', 'COST', 'AVGO', 'TXN',
            'BA', 'CAT', 'GE', 'IBM', 'WMT', 'CVX', 'XOM', 'QCOM',
            'GOOG', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'AXP',
            'LLY', 'ABBV', 'BMY', 'GILD', 'AMGN', 'MDT', 'SYK', 'ISRG',
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'VLO', 'PSX',
            'HON', 'UPS', 'RTX', 'LMT', 'DE', 'MMM', 'UNP', 'FDX'
        ]


async def run_scanner_mode(settings, watchlist: list):
    """Run in scanner mode (semi-automated)"""
    logger.info("Starting Scanner Mode (Semi-Automated)")
    logger.info("Signals will be displayed, you execute trades manually")
    logger.info("-" * 50)

    from scanner.realtime_scanner import RealTimeScanner

    config = {
        'atr_period': settings.rrs.atr_period,
        'rrs_strong_threshold': settings.rrs.strong_threshold,
        'scan_interval_seconds': settings.scanner.scan_interval_seconds,
        'min_volume': settings.scanner.min_volume,
        'min_price': settings.scanner.min_price,
        'alert_method': settings.alert.method
    }

    scanner = RealTimeScanner(config)
    scanner.watchlist = watchlist
    scanner.run_continuous()


async def run_bot_mode(settings, watchlist: list, auto_trade: bool = False):
    """Run in full autonomous mode"""
    mode = "LIVE" if not settings.trading.paper_trading else "PAPER"
    auto = "AUTO-TRADE" if auto_trade else "MANUAL-EXECUTE"

    logger.warning(f"Starting Bot Mode: {mode} / {auto}")
    if not settings.trading.paper_trading:
        logger.warning("*** LIVE TRADING MODE - REAL MONEY AT RISK ***")

    # Create components
    if settings.trading.paper_trading:
        broker = get_broker("paper", initial_balance=settings.trading.account_size)
    else:
        broker = get_broker(
            "schwab",
            app_key=settings.broker.schwab_app_key,
            app_secret=settings.broker.schwab_app_secret,
            callback_url=settings.broker.schwab_callback_url
        )
        broker.connect()

    risk_limits = RiskLimits(
        max_risk_per_trade=settings.trading.max_risk_per_trade,
        max_daily_loss=settings.trading.max_daily_loss,
        max_position_size=settings.trading.max_position_size
    )

    risk_manager = RiskManager(
        account_size=settings.trading.account_size,
        risk_limits=risk_limits
    )

    data_provider = DataProvider()

    config = {
        'scan_interval': settings.scanner.interval_seconds,
        'rrs_threshold': settings.rrs.strong_threshold
    }

    # Run the system
    await run_trading_system(
        broker=broker,
        risk_manager=risk_manager,
        data_provider=data_provider,
        watchlist=watchlist,
        config=config,
        auto_trade=auto_trade
    )


async def run_backtest_mode(settings, watchlist: list, days: int = 365, use_optimized: bool = True):
    """Run backtest on historical data"""
    logger.info(f"Running Backtest for last {days} days")
    if use_optimized:
        logger.info("Using OPTIMIZED strategy parameters (ResearchAgent)")

    from backtesting import BacktestEngine, DataLoader

    # Load data
    loader = DataLoader(cache_dir="data/historical")
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    logger.info(f"Loading data: {start_date} to {end_date}")
    stock_data = loader.load_stock_data(watchlist, start_date, end_date)
    spy_data = loader.load_spy_data(start_date, end_date)

    # Run backtest
    risk_limits = RiskLimits(
        max_risk_per_trade=settings.trading.max_risk_per_trade,
        max_daily_loss=settings.trading.max_daily_loss
    )

    # Use optimized parameters from ResearchAgent
    # After testing: Higher RRS threshold for quality over quantity
    if use_optimized:
        rrs_threshold = 2.0  # Selective entries - only strong RS/RW
        use_relaxed = True   # Use relaxed daily chart criteria
        stop_multiplier = 0.75  # Tighter stops (0.75x ATR) - cut losers quickly
        target_multiplier = 1.5  # 1.5:1 R/R - take profits earlier
    else:
        rrs_threshold = settings.rrs.strong_threshold
        use_relaxed = False
        stop_multiplier = 1.5
        target_multiplier = 3.0

    engine = BacktestEngine(
        initial_capital=settings.trading.account_size,
        risk_limits=risk_limits,
        rrs_threshold=rrs_threshold,
        use_relaxed_criteria=use_relaxed,
        stop_atr_multiplier=stop_multiplier,
        target_atr_multiplier=target_multiplier
    )

    result = engine.run(stock_data, spy_data, start_date, end_date)

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period: {result.start_date} to {result.end_date}")
    print(f"Initial Capital: ${result.initial_capital:,.2f}")
    print(f"Final Capital: ${result.final_capital:,.2f}")
    print(f"Total Return: ${result.total_return:,.2f} ({result.total_return_pct:.2f}%)")
    print("-" * 40)
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Avg Win: ${result.avg_win:,.2f}")
    print(f"Avg Loss: ${result.avg_loss:,.2f}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print("-" * 40)
    print(f"Max Drawdown: ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.1f}%)")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Avg Holding Days: {result.avg_holding_days:.1f}")
    print("=" * 60)

    return result


async def run_dashboard_mode():
    """Run dashboard monitoring"""
    logger.info("Starting Dashboard Mode")

    dashboard = TradingDashboard()
    await dashboard.run_console_loop(refresh_seconds=5)


async def run_optimize_mode(settings, watchlist: list, days: int = 365):
    """Run parameter optimization to find best strategy configuration"""
    logger.info(f"Running Parameter Optimization for last {days} days")
    logger.info(f"Testing across {len(watchlist)} stocks")

    from backtesting.parameter_optimizer import ParameterOptimizer, ParameterSet

    optimizer = ParameterOptimizer(
        initial_capital=settings.trading.account_size,
        data_dir="data/historical",
        results_dir="data/optimization"
    )

    # Load data
    optimizer.load_data(watchlist, days=days)

    # Generate parameter grid
    param_sets = optimizer.generate_parameter_grid(
        rrs_thresholds=[1.5, 1.75, 2.0, 2.25, 2.5],
        stop_multipliers=[0.5, 0.75, 1.0],
        target_multipliers=[1.5, 2.0, 2.5, 3.0],
        max_positions_list=[5, 7, 10],
        use_relaxed_list=[True]
    )

    logger.info(f"Testing {len(param_sets)} parameter combinations...")

    # Run optimization
    results = optimizer.run_optimization(param_sets)

    # Print summary
    optimizer.print_summary(results, top_n=15)

    # Save results
    filepath = optimizer.save_results(results)
    logger.info(f"Results saved to {filepath}")

    # Get recommended parameters
    recommended = optimizer.get_recommended_parameters(results)
    if recommended:
        print("\n" + "=" * 60)
        print("RECOMMENDED CONFIGURATION")
        print("=" * 60)
        print(f"RRS Threshold: {recommended.rrs_threshold}")
        print(f"Stop Multiplier: {recommended.stop_atr_multiplier}x ATR")
        print(f"Target Multiplier: {recommended.target_atr_multiplier}x ATR")
        print(f"Max Positions: {recommended.max_positions}")
        print(f"Use Relaxed Criteria: {recommended.use_relaxed_criteria}")
        print("=" * 60)

    return results


async def run_enhanced_backtest_mode(settings, watchlist: list, days: int = 365):
    """Run enhanced backtest with trailing stops and scaled exits"""
    logger.info(f"Running ENHANCED Backtest for last {days} days")
    logger.info("Features: Trailing stops, Scaled exits, Time stops")

    from backtesting.engine_enhanced import EnhancedBacktestEngine
    from backtesting import DataLoader

    # Load data
    loader = DataLoader(cache_dir="data/historical")
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    logger.info(f"Loading data: {start_date} to {end_date}")
    stock_data = loader.load_stock_data(watchlist, start_date, end_date)
    spy_data = loader.load_spy_data(start_date, end_date)

    risk_limits = RiskLimits(
        max_risk_per_trade=settings.trading.max_risk_per_trade,
        max_daily_loss=settings.trading.max_daily_loss
    )

    engine = EnhancedBacktestEngine(
        initial_capital=settings.trading.account_size,
        risk_limits=risk_limits,
        rrs_threshold=1.75,  # OPTIMIZED: was 2.0
        max_positions=10,  # AGGRESSIVE: was 5
        use_relaxed_criteria=True,
        stop_atr_multiplier=0.75,
        target_atr_multiplier=1.5,  # OPTIMIZED: was 2.0
        use_trailing_stop=True,
        breakeven_trigger_r=1.0,
        trailing_atr_multiplier=1.0,
        use_scaled_exits=True,
        scale_1_target_r=1.0,
        scale_1_percent=0.5,
        use_time_stop=True,
        max_holding_days=10
    )

    result = engine.run(stock_data, spy_data, start_date, end_date)

    # Print enhanced results
    print("\n" + "=" * 60)
    print("ENHANCED BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period: {result.start_date} to {result.end_date}")
    print(f"Initial Capital: ${result.initial_capital:,.2f}")
    print(f"Final Capital: ${result.final_capital:,.2f}")
    print(f"Total Return: ${result.total_return:,.2f} ({result.total_return_pct:.2f}%)")
    print("-" * 40)
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Avg Win: ${result.avg_win:,.2f}")
    print(f"Avg Loss: ${result.avg_loss:,.2f}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print("-" * 40)
    print("EXIT ANALYSIS:")
    print(f"  Stop Loss Exits: {result.trades_stopped_out}")
    print(f"  Take Profit Exits: {result.trades_target_hit}")
    print(f"  Trailing Stop Exits: {result.trades_trailing_stopped}")
    print(f"  Time Stop Exits: {result.trades_time_stopped}")
    print(f"  Breakeven Activations: {result.breakeven_activations}")
    print(f"  Scale 1 Exits: {result.scale_1_exits}")
    print("-" * 40)
    print(f"Max Drawdown: ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.1f}%)")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Avg Holding Days: {result.avg_holding_days:.1f}")
    print(f"Avg MFE: ${result.avg_mfe:.2f}")
    print(f"Avg MAE: ${result.avg_mae:.2f}")
    print("=" * 60)

    return result


async def run_multi_strategy_mode(settings, watchlist: list, days: int = 365):
    """Run multi-strategy backtest combining all revenue-generating strategies"""
    logger.info(f"Running MULTI-STRATEGY Backtest for last {days} days")
    logger.info("Strategies: RRS Momentum, Leveraged ETFs, Sector Rotation")

    from strategies.multi_strategy_engine import MultiStrategyEngine
    from backtesting import DataLoader

    # Load data
    loader = DataLoader(cache_dir="data/historical")
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    # Add leveraged ETFs and sector ETFs to watchlist
    from config.watchlists import LEVERAGED_ETFS, SECTOR_ETFS
    all_symbols = set(watchlist)
    all_symbols.update(LEVERAGED_ETFS)
    all_symbols.update(SECTOR_ETFS)
    extended_watchlist = list(all_symbols)

    logger.info(f"Loading data for {len(extended_watchlist)} symbols...")
    stock_data = loader.load_stock_data(extended_watchlist, start_date, end_date)
    spy_data = loader.load_spy_data(start_date, end_date)

    # Setup multi-strategy engine
    engine = MultiStrategyEngine(
        initial_capital=settings.trading.account_size,
        use_kelly=True,
        use_volatility_adjustment=True
    )
    engine.setup_default_strategies()

    # Run backtest
    result = engine.run_backtest(stock_data, spy_data, start_date, end_date)

    # Print results
    engine.print_summary(result)

    return result


async def run_web_server():
    """Run the web application for signal service"""
    logger.info("Starting Web Server for Signal Service")
    logger.info("Access at: http://localhost:8080")

    import subprocess
    subprocess.run(["python", "-m", "web.app"], cwd="/home/user0/rdt-trading-system")


def main():
    parser = argparse.ArgumentParser(
        description="RDT Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py scanner              # Run signal scanner
  python main.py bot                  # Run bot (manual execution)
  python main.py bot --auto           # Run bot (auto execution)
  python main.py backtest             # Run 1-year backtest
  python main.py backtest --days 180  # Run 180-day backtest
  python main.py backtest --enhanced  # Run with trailing stops & scaled exits
  python main.py backtest --multi     # Run multi-strategy backtest
  python main.py optimize             # Find optimal parameters
  python main.py optimize --days 730  # 2-year optimization
  python main.py dashboard            # Run monitoring dashboard
  python main.py web                  # Run signal service web app

Watchlist Options:
  --watchlist core       # Top 50 liquid stocks
  --watchlist full       # 150+ stocks (default)
  --watchlist technology # Tech sector only
  --watchlist aggressive # High volatility + leveraged ETFs
        """
    )

    parser.add_argument(
        "mode",
        choices=["scanner", "bot", "backtest", "optimize", "dashboard", "web"],
        help="Operating mode"
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help="Enable automatic trade execution (bot mode only)"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days for backtest (default: 365)"
    )

    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (overrides default)"
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict (original) strategy instead of optimized parameters"
    )

    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Use enhanced backtest with trailing stops and scaled exits"
    )

    parser.add_argument(
        "--multi",
        action="store_true",
        help="Use multi-strategy backtest (RRS + Leveraged ETFs + Sector Rotation)"
    )

    parser.add_argument(
        "--watchlist",
        type=str,
        default="full",
        help="Watchlist to use: core, full, technology, financials, aggressive, etc."
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    settings = get_settings()

    # Get watchlist
    if args.symbols:
        watchlist = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        watchlist = get_default_watchlist(args.watchlist)

    logger.info(f"Watchlist: {len(watchlist)} symbols ({args.watchlist})")

    # Run selected mode
    try:
        if args.mode == "scanner":
            asyncio.run(run_scanner_mode(settings, watchlist))

        elif args.mode == "bot":
            asyncio.run(run_bot_mode(settings, watchlist, auto_trade=args.auto))

        elif args.mode == "backtest":
            if args.multi:
                asyncio.run(run_multi_strategy_mode(settings, watchlist, days=args.days))
            elif args.enhanced:
                asyncio.run(run_enhanced_backtest_mode(settings, watchlist, days=args.days))
            else:
                use_optimized = not args.strict
                asyncio.run(run_backtest_mode(settings, watchlist, days=args.days, use_optimized=use_optimized))

        elif args.mode == "optimize":
            asyncio.run(run_optimize_mode(settings, watchlist, days=args.days))

        elif args.mode == "dashboard":
            asyncio.run(run_dashboard_mode())

        elif args.mode == "web":
            asyncio.run(run_web_server())

    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
