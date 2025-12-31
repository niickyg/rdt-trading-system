#!/usr/bin/env python3
"""
Test different risk profiles to find optimal risk/reward balance
"""

import asyncio
from datetime import date, timedelta
from loguru import logger
import sys

from backtesting.engine_enhanced import EnhancedBacktestEngine
from backtesting import DataLoader
from risk import RiskLimits
from config.watchlists import get_watchlist_by_name


async def test_risk_profile(
    name: str,
    max_risk_per_trade: float,
    max_daily_loss: float,
    max_positions: int,
    watchlist: list,
    days: int = 365
):
    """Test a specific risk profile"""

    logger.info(f"\n{'='*70}")
    logger.info(f"Testing {name.upper()} Risk Profile")
    logger.info(f"  Risk per trade: {max_risk_per_trade*100:.1f}%")
    logger.info(f"  Max daily loss: {max_daily_loss*100:.1f}%")
    logger.info(f"  Max positions: {max_positions}")
    logger.info(f"{'='*70}")

    # Load data
    loader = DataLoader(cache_dir="data/historical")
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    stock_data = loader.load_stock_data(watchlist, start_date, end_date)
    spy_data = loader.load_spy_data(start_date, end_date)

    # Create risk limits
    risk_limits = RiskLimits(
        max_risk_per_trade=max_risk_per_trade,
        max_daily_loss=max_daily_loss
    )

    # Run enhanced backtest with optimized parameters
    engine = EnhancedBacktestEngine(
        initial_capital=25000,
        risk_limits=risk_limits,
        rrs_threshold=1.75,  # Optimized from earlier tests
        max_positions=max_positions,
        use_relaxed_criteria=True,
        stop_atr_multiplier=0.75,
        target_atr_multiplier=1.5,
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

    return result


async def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    # Load watchlist
    watchlist = get_watchlist_by_name("core")
    print(f"Testing on {len(watchlist)} stocks (core watchlist)")
    print(f"Backtest period: 365 days\n")

    # Define risk profiles to test
    risk_profiles = [
        {
            "name": "Conservative (Current)",
            "max_risk_per_trade": 0.01,  # 1%
            "max_daily_loss": 0.02,       # 2%
            "max_positions": 5
        },
        {
            "name": "Moderate",
            "max_risk_per_trade": 0.02,  # 2%
            "max_daily_loss": 0.04,       # 4%
            "max_positions": 7
        },
        {
            "name": "Aggressive",
            "max_risk_per_trade": 0.03,  # 3%
            "max_daily_loss": 0.06,       # 6%
            "max_positions": 10
        },
        {
            "name": "Very Aggressive",
            "max_risk_per_trade": 0.04,  # 4%
            "max_daily_loss": 0.08,       # 8%
            "max_positions": 10
        }
    ]

    results = []

    for profile in risk_profiles:
        result = await test_risk_profile(
            name=profile["name"],
            max_risk_per_trade=profile["max_risk_per_trade"],
            max_daily_loss=profile["max_daily_loss"],
            max_positions=profile["max_positions"],
            watchlist=watchlist,
            days=365
        )
        results.append({
            "profile": profile,
            "result": result
        })

    # Print comparison table
    print("\n" + "=" * 120)
    print("RISK PROFILE COMPARISON")
    print("=" * 120)
    print(f"{'Profile':<20} {'Return':<12} {'Profit':<12} {'Win Rate':<12} {'Drawdown':<12} {'Sharpe':<10} {'Trades':<10}")
    print("-" * 120)

    for r in results:
        profile = r["profile"]
        result = r["result"]
        print(f"{profile['name']:<20} "
              f"${result.total_return:>8,.0f} ({result.total_return_pct:>5.2f}%)  "
              f"${result.final_capital-25000:>8,.0f}  "
              f"{result.win_rate:>10.1%}  "
              f"{result.max_drawdown_pct:>10.1f}%  "
              f"{result.sharpe_ratio:>8.2f}  "
              f"{result.total_trades:>8}")

    print("=" * 120)

    # Detailed breakdown
    print("\n" + "=" * 120)
    print("DETAILED RESULTS")
    print("=" * 120)

    for r in results:
        profile = r["profile"]
        result = r["result"]

        print(f"\n{profile['name'].upper()}")
        print("-" * 60)
        print(f"Risk Settings:")
        print(f"  Risk per trade: {profile['max_risk_per_trade']*100:.1f}%")
        print(f"  Max daily loss: {profile['max_daily_loss']*100:.1f}%")
        print(f"  Max positions: {profile['max_positions']}")
        print()
        print(f"Performance:")
        print(f"  Return: ${result.total_return:,.2f} ({result.total_return_pct:.2f}%)")
        print(f"  Final Capital: ${result.final_capital:,.2f}")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Win Rate: {result.win_rate:.1%}")
        print(f"  Avg Win: ${result.avg_win:.2f}")
        print(f"  Avg Loss: ${result.avg_loss:.2f}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown_pct:.1f}%")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print()
        print(f"Exit Analysis:")
        print(f"  Stop Loss: {result.trades_stopped_out}")
        print(f"  Take Profit: {result.trades_target_hit}")
        print(f"  Trailing Stop: {result.trades_trailing_stopped}")
        print(f"  Time Stop: {result.trades_time_stopped}")
        print(f"  Breakeven: {result.breakeven_activations}")
        print(f"  Scaled Exits: {result.scale_1_exits}")

    print("\n" + "=" * 120)

    # Find best profile
    best = max(results, key=lambda x: x["result"].total_return_pct)

    print("\nRECOMMENDATION")
    print("=" * 60)
    print(f"Best Profile: {best['profile']['name']}")
    print(f"Expected Annual Return: {best['result'].total_return_pct:.2f}%")
    print(f"Max Drawdown: {best['result'].max_drawdown_pct:.1f}%")
    print(f"Sharpe Ratio: {best['result'].sharpe_ratio:.2f}")
    print()
    print(f"Configuration:")
    print(f"  MAX_RISK_PER_TRADE={best['profile']['max_risk_per_trade']}")
    print(f"  MAX_DAILY_LOSS={best['profile']['max_daily_loss']}")
    print(f"  RRS_STRONG_THRESHOLD=1.75")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
