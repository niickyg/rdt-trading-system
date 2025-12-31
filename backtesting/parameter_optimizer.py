"""
Parameter Optimization Engine
Systematic grid search and optimization for RDT trading strategy parameters

This module enables rapid iteration to find optimal strategy configurations
that maximize risk-adjusted returns.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from loguru import logger

from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.data_loader import DataLoader
from risk.models import RiskLimits


@dataclass
class ParameterSet:
    """A set of strategy parameters to test"""
    rrs_threshold: float = 2.0
    stop_atr_multiplier: float = 0.75
    target_atr_multiplier: float = 1.5
    use_relaxed_criteria: bool = True
    max_positions: int = 5
    max_risk_per_trade: float = 0.01
    daily_strength_min_score: int = 3  # Minimum score out of 5

    def to_dict(self) -> Dict:
        return {
            "rrs_threshold": self.rrs_threshold,
            "stop_atr_multiplier": self.stop_atr_multiplier,
            "target_atr_multiplier": self.target_atr_multiplier,
            "use_relaxed_criteria": self.use_relaxed_criteria,
            "max_positions": self.max_positions,
            "max_risk_per_trade": self.max_risk_per_trade,
            "daily_strength_min_score": self.daily_strength_min_score
        }

    def __hash__(self):
        return hash(tuple(sorted(self.to_dict().items())))


@dataclass
class OptimizationResult:
    """Results from a single parameter optimization run"""
    parameters: ParameterSet
    backtest_result: BacktestResult
    score: float  # Composite optimization score
    rank: int = 0

    def to_dict(self) -> Dict:
        return {
            "parameters": self.parameters.to_dict(),
            "metrics": {
                "total_return_pct": self.backtest_result.total_return_pct,
                "win_rate": self.backtest_result.win_rate,
                "profit_factor": self.backtest_result.profit_factor,
                "max_drawdown_pct": self.backtest_result.max_drawdown_pct,
                "sharpe_ratio": self.backtest_result.sharpe_ratio,
                "total_trades": self.backtest_result.total_trades,
                "avg_holding_days": self.backtest_result.avg_holding_days
            },
            "score": self.score,
            "rank": self.rank
        }


class ScoringFunction:
    """Composite scoring function for strategy evaluation"""

    def __init__(
        self,
        return_weight: float = 0.30,
        profit_factor_weight: float = 0.25,
        sharpe_weight: float = 0.20,
        drawdown_weight: float = 0.15,
        win_rate_weight: float = 0.10
    ):
        self.return_weight = return_weight
        self.profit_factor_weight = profit_factor_weight
        self.sharpe_weight = sharpe_weight
        self.drawdown_weight = drawdown_weight
        self.win_rate_weight = win_rate_weight

    def calculate(self, result: BacktestResult) -> float:
        """
        Calculate composite score for a backtest result

        Higher is better. Score is normalized to roughly 0-100 range.
        """
        if result.total_trades < 10:
            return 0.0  # Not enough trades to evaluate

        # Normalize each component
        # Return: 0-50% annual return maps to 0-100
        return_score = min(result.total_return_pct * 2, 100)

        # Profit Factor: 1.0-3.0 maps to 0-100
        pf_score = max(0, min((result.profit_factor - 1.0) * 50, 100))

        # Sharpe: 0-3.0 maps to 0-100
        sharpe_score = max(0, min(result.sharpe_ratio * 33.33, 100))

        # Drawdown: 0% is 100, 20% is 0 (inverted - lower is better)
        dd_score = max(0, 100 - result.max_drawdown_pct * 5)

        # Win Rate: 20%-60% maps to 0-100
        wr_score = max(0, min((result.win_rate * 100 - 20) * 2.5, 100))

        # Weighted composite
        score = (
            return_score * self.return_weight +
            pf_score * self.profit_factor_weight +
            sharpe_score * self.sharpe_weight +
            dd_score * self.drawdown_weight +
            wr_score * self.win_rate_weight
        )

        # Bonus for high trade count (more statistically significant)
        if result.total_trades >= 50:
            score *= 1.1
        elif result.total_trades >= 100:
            score *= 1.2

        return round(score, 2)


class ParameterOptimizer:
    """
    Grid search optimizer for trading strategy parameters

    Systematically tests parameter combinations to find optimal
    configurations that maximize risk-adjusted returns.
    """

    def __init__(
        self,
        initial_capital: float = 25000,
        data_dir: str = "data/historical",
        results_dir: str = "data/optimization",
        scoring_function: Optional[ScoringFunction] = None
    ):
        self.initial_capital = initial_capital
        self.data_loader = DataLoader(cache_dir=data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = scoring_function or ScoringFunction()

        # Cache for historical data
        self._stock_data: Optional[Dict[str, pd.DataFrame]] = None
        self._spy_data: Optional[pd.DataFrame] = None

    def load_data(
        self,
        watchlist: List[str],
        days: int = 365,
        end_date: Optional[date] = None
    ):
        """Load historical data for backtesting"""
        end_date = end_date or date.today()
        start_date = end_date - timedelta(days=days)

        logger.info(f"Loading data: {start_date} to {end_date}")
        self._stock_data = self.data_loader.load_stock_data(watchlist, start_date, end_date)
        self._spy_data = self.data_loader.load_spy_data(start_date, end_date)

        logger.info(f"Loaded {len(self._stock_data)} stocks")
        return self

    def generate_parameter_grid(
        self,
        rrs_thresholds: List[float] = [1.5, 1.75, 2.0, 2.25, 2.5],
        stop_multipliers: List[float] = [0.5, 0.75, 1.0, 1.25],
        target_multipliers: List[float] = [1.25, 1.5, 2.0, 2.5, 3.0],
        max_positions_list: List[int] = [3, 5, 7],
        use_relaxed_list: List[bool] = [True, False]
    ) -> List[ParameterSet]:
        """Generate all parameter combinations for grid search"""
        combinations = list(product(
            rrs_thresholds,
            stop_multipliers,
            target_multipliers,
            max_positions_list,
            use_relaxed_list
        ))

        parameter_sets = []
        for rrs, stop, target, max_pos, relaxed in combinations:
            # Filter out invalid combinations
            if target <= stop:  # Target must be greater than stop for positive R:R
                continue

            param_set = ParameterSet(
                rrs_threshold=rrs,
                stop_atr_multiplier=stop,
                target_atr_multiplier=target,
                max_positions=max_pos,
                use_relaxed_criteria=relaxed
            )
            parameter_sets.append(param_set)

        logger.info(f"Generated {len(parameter_sets)} parameter combinations")
        return parameter_sets

    def run_single_backtest(
        self,
        params: ParameterSet,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> OptimizationResult:
        """Run a single backtest with given parameters"""
        if self._stock_data is None or self._spy_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        risk_limits = RiskLimits(
            max_risk_per_trade=params.max_risk_per_trade,
            max_open_positions=params.max_positions
        )

        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            risk_limits=risk_limits,
            rrs_threshold=params.rrs_threshold,
            max_positions=params.max_positions,
            use_relaxed_criteria=params.use_relaxed_criteria,
            stop_atr_multiplier=params.stop_atr_multiplier,
            target_atr_multiplier=params.target_atr_multiplier
        )

        result = engine.run(
            self._stock_data,
            self._spy_data,
            start_date,
            end_date
        )

        score = self.scorer.calculate(result)

        return OptimizationResult(
            parameters=params,
            backtest_result=result,
            score=score
        )

    def run_optimization(
        self,
        parameter_sets: Optional[List[ParameterSet]] = None,
        parallel: bool = False,
        max_workers: int = 4
    ) -> List[OptimizationResult]:
        """
        Run optimization across all parameter sets

        Args:
            parameter_sets: List of parameter sets to test. If None, uses default grid.
            parallel: Whether to run backtests in parallel
            max_workers: Number of parallel workers

        Returns:
            List of OptimizationResult, sorted by score (best first)
        """
        if parameter_sets is None:
            parameter_sets = self.generate_parameter_grid()

        results = []
        total = len(parameter_sets)

        logger.info(f"Running {total} backtests...")

        if parallel:
            # Parallel execution (note: may have issues with shared data)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.run_single_backtest, params): params
                    for params in parameter_sets
                }

                for i, future in enumerate(as_completed(futures)):
                    try:
                        result = future.result()
                        results.append(result)
                        if (i + 1) % 10 == 0:
                            logger.info(f"Completed {i + 1}/{total} backtests")
                    except Exception as e:
                        logger.error(f"Backtest failed: {e}")
        else:
            # Sequential execution
            for i, params in enumerate(parameter_sets):
                try:
                    result = self.run_single_backtest(params)
                    results.append(result)
                    if (i + 1) % 10 == 0:
                        logger.info(f"Completed {i + 1}/{total} backtests")
                except Exception as e:
                    logger.error(f"Backtest failed for {params}: {e}")

        # Sort by score and assign ranks
        results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1

        logger.info(f"Optimization complete. Best score: {results[0].score if results else 'N/A'}")

        return results

    def walk_forward_optimization(
        self,
        parameter_sets: List[ParameterSet],
        in_sample_days: int = 180,
        out_sample_days: int = 60,
        num_periods: int = 4
    ) -> Dict[str, Any]:
        """
        Walk-forward optimization to prevent overfitting

        Trains on in-sample period, validates on out-of-sample period,
        then walks forward in time.
        """
        if self._spy_data is None:
            raise ValueError("Data not loaded")

        all_dates = list(self._spy_data.index.date)
        total_days = in_sample_days + out_sample_days

        wf_results = []

        for period in range(num_periods):
            # Calculate date ranges
            period_end_idx = len(all_dates) - (period * out_sample_days)
            period_start_idx = period_end_idx - total_days

            if period_start_idx < 0:
                break

            in_sample_start = all_dates[period_start_idx]
            in_sample_end = all_dates[period_start_idx + in_sample_days]
            out_sample_start = in_sample_end
            out_sample_end = all_dates[period_end_idx - 1]

            logger.info(f"Period {period + 1}: IS {in_sample_start} to {in_sample_end}, OS {out_sample_start} to {out_sample_end}")

            # Run in-sample optimization
            is_results = []
            for params in parameter_sets:
                try:
                    result = self.run_single_backtest(params, in_sample_start, in_sample_end)
                    is_results.append(result)
                except Exception:
                    continue

            if not is_results:
                continue

            # Find best in-sample parameters
            is_results.sort(key=lambda x: x.score, reverse=True)
            best_is_params = is_results[0].parameters

            # Run out-of-sample test with best parameters
            try:
                os_result = self.run_single_backtest(best_is_params, out_sample_start, out_sample_end)

                wf_results.append({
                    "period": period + 1,
                    "in_sample": {
                        "start": str(in_sample_start),
                        "end": str(in_sample_end),
                        "score": is_results[0].score,
                        "return_pct": is_results[0].backtest_result.total_return_pct
                    },
                    "out_sample": {
                        "start": str(out_sample_start),
                        "end": str(out_sample_end),
                        "score": os_result.score,
                        "return_pct": os_result.backtest_result.total_return_pct
                    },
                    "best_params": best_is_params.to_dict()
                })
            except Exception as e:
                logger.error(f"Out-of-sample test failed: {e}")

        # Calculate overall walk-forward efficiency
        if wf_results:
            is_scores = [r["in_sample"]["score"] for r in wf_results]
            os_scores = [r["out_sample"]["score"] for r in wf_results]
            wf_efficiency = sum(os_scores) / sum(is_scores) if sum(is_scores) > 0 else 0
        else:
            wf_efficiency = 0

        return {
            "periods": wf_results,
            "walk_forward_efficiency": wf_efficiency,
            "interpretation": self._interpret_wf_efficiency(wf_efficiency)
        }

    def _interpret_wf_efficiency(self, efficiency: float) -> str:
        """Interpret walk-forward efficiency ratio"""
        if efficiency >= 0.8:
            return "Excellent - Strategy is robust and likely to perform well live"
        elif efficiency >= 0.6:
            return "Good - Strategy shows reasonable out-of-sample performance"
        elif efficiency >= 0.4:
            return "Fair - Some overfitting detected, use caution"
        else:
            return "Poor - Significant overfitting, strategy needs revision"

    def save_results(
        self,
        results: List[OptimizationResult],
        filename: Optional[str] = None
    ) -> Path:
        """Save optimization results to JSON file"""
        if filename is None:
            filename = f"optimization_{date.today().isoformat()}.json"

        filepath = self.results_dir / filename

        data = {
            "timestamp": date.today().isoformat(),
            "initial_capital": self.initial_capital,
            "total_combinations_tested": len(results),
            "top_10_results": [r.to_dict() for r in results[:10]],
            "all_results": [r.to_dict() for r in results]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {filepath}")
        return filepath

    def get_recommended_parameters(
        self,
        results: List[OptimizationResult],
        min_trades: int = 50,
        max_drawdown_pct: float = 10.0
    ) -> Optional[ParameterSet]:
        """
        Get recommended parameters based on optimization results

        Filters for practical constraints and returns best scoring parameters.
        """
        filtered = [
            r for r in results
            if r.backtest_result.total_trades >= min_trades
            and r.backtest_result.max_drawdown_pct <= max_drawdown_pct
        ]

        if not filtered:
            logger.warning("No results meet the criteria. Returning best overall.")
            return results[0].parameters if results else None

        return filtered[0].parameters

    def print_summary(self, results: List[OptimizationResult], top_n: int = 10):
        """Print optimization summary"""
        print("\n" + "=" * 80)
        print("PARAMETER OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"Total combinations tested: {len(results)}")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print("-" * 80)

        print(f"\nTOP {top_n} PARAMETER CONFIGURATIONS:\n")

        header = f"{'Rank':<5} {'RRS':>5} {'Stop':>6} {'Target':>7} {'MaxPos':>7} {'Relaxed':>8} | {'Return':>8} {'WinRate':>8} {'PF':>6} {'DD':>6} {'Score':>7}"
        print(header)
        print("-" * len(header))

        for i, result in enumerate(results[:top_n]):
            p = result.parameters
            r = result.backtest_result

            print(f"{result.rank:<5} "
                  f"{p.rrs_threshold:>5.2f} "
                  f"{p.stop_atr_multiplier:>6.2f} "
                  f"{p.target_atr_multiplier:>7.2f} "
                  f"{p.max_positions:>7} "
                  f"{'Yes' if p.use_relaxed_criteria else 'No':>8} | "
                  f"{r.total_return_pct:>7.2f}% "
                  f"{r.win_rate*100:>7.1f}% "
                  f"{r.profit_factor:>6.2f} "
                  f"{r.max_drawdown_pct:>5.1f}% "
                  f"{result.score:>7.1f}")

        print("\n" + "=" * 80)

        # Best result details
        best = results[0]
        print("\nBEST CONFIGURATION DETAILS:")
        print(f"  RRS Threshold: {best.parameters.rrs_threshold}")
        print(f"  Stop Multiplier: {best.parameters.stop_atr_multiplier}x ATR")
        print(f"  Target Multiplier: {best.parameters.target_atr_multiplier}x ATR")
        print(f"  Max Positions: {best.parameters.max_positions}")
        print(f"  Use Relaxed Daily Criteria: {best.parameters.use_relaxed_criteria}")
        print(f"\n  Expected Annual Return: {best.backtest_result.total_return_pct:.2f}%")
        print(f"  Expected $ Return on ${self.initial_capital:,}: ${best.backtest_result.total_return:,.2f}")
        print("=" * 80)


def run_quick_optimization(
    watchlist: List[str] = None,
    days: int = 365,
    initial_capital: float = 25000
) -> Tuple[List[OptimizationResult], ParameterSet]:
    """
    Quick optimization with sensible defaults

    Returns:
        Tuple of (all results, recommended parameters)
    """
    if watchlist is None:
        watchlist = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            'JPM', 'V', 'JNJ', 'UNH', 'HD', 'PG', 'MA', 'DIS',
            'PYPL', 'ADBE', 'CRM', 'NFLX', 'AMD', 'INTC', 'CSCO',
            'PEP', 'KO', 'MRK', 'ABT', 'TMO', 'COST', 'AVGO', 'TXN'
        ]

    optimizer = ParameterOptimizer(initial_capital=initial_capital)
    optimizer.load_data(watchlist, days=days)

    # Focused parameter grid (fewer combinations for speed)
    param_sets = optimizer.generate_parameter_grid(
        rrs_thresholds=[1.5, 2.0, 2.5],
        stop_multipliers=[0.5, 0.75, 1.0],
        target_multipliers=[1.5, 2.0, 2.5],
        max_positions_list=[5, 7],
        use_relaxed_list=[True]
    )

    results = optimizer.run_optimization(param_sets)
    optimizer.print_summary(results)
    optimizer.save_results(results)

    recommended = optimizer.get_recommended_parameters(results)

    return results, recommended


if __name__ == "__main__":
    # Run optimization
    results, best_params = run_quick_optimization()

    if best_params:
        print(f"\nRecommended parameters: {best_params.to_dict()}")
