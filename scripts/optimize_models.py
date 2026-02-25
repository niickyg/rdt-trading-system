#!/usr/bin/env python3
"""
ML Model Hyperparameter Optimization Script

CLI script for running Optuna-based hyperparameter optimization on ML models.

Usage:
    python scripts/optimize_models.py --model-type xgboost --n-trials 100 --metric f1
    python scripts/optimize_models.py --model-type random_forest --timeout 3600
    python scripts/optimize_models.py --model-type lstm --n-trials 50 --metric auc

Arguments:
    --model-type: Model type to optimize (xgboost, random_forest, lstm)
    --n-trials: Number of optimization trials (default: 100)
    --timeout: Maximum optimization time in seconds (optional)
    --metric: Optimization metric (accuracy, precision, recall, f1, auc, profit_factor)
    --cv-splits: Number of cross-validation splits (default: 5)
    --output: Output path for best parameters (default: config/optimized_params.json)
    --data-file: Path to training data CSV file (optional)
    --generate-report: Generate HTML optimization report
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)


def load_training_data(data_file: str = None) -> tuple:
    """
    Load training data for optimization.

    Args:
        data_file: Optional path to CSV data file

    Returns:
        Tuple of (X, y, feature_names)
    """
    if data_file and os.path.exists(data_file):
        logger.info(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)

        # Assume 'target' column contains labels
        if 'target' in df.columns:
            y = df['target'].values
            X = df.drop(columns=['target', 'symbol', 'date', 'timestamp'],
                       errors='ignore').values
            feature_names = [col for col in df.columns
                           if col not in ['target', 'symbol', 'date', 'timestamp']]
        else:
            raise ValueError("Data file must contain a 'target' column")

    else:
        # Generate synthetic training data for demonstration
        logger.info("Generating synthetic training data")
        np.random.seed(42)

        n_samples = 2000
        n_features = 20

        # Generate features
        X = np.random.randn(n_samples, n_features)

        # Generate labels with some correlation to features
        weights = np.random.randn(n_features)
        linear_combo = X @ weights
        probs = 1 / (1 + np.exp(-linear_combo + np.random.randn(n_samples) * 2))
        y = (probs > 0.5).astype(int)

        feature_names = [f'feature_{i}' for i in range(n_features)]

        logger.info(f"Generated {n_samples} samples with {n_features} features")
        logger.info(f"Class distribution: {np.bincount(y)}")

    return X, y, feature_names


def generate_optimization_report(
    result,
    optimizer,
    output_dir: str,
    model_type: str
):
    """
    Generate HTML optimization report.

    Args:
        result: OptimizationResult object
        optimizer: ModelOptimizer instance
        output_dir: Output directory for report
        model_type: Model type
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_path = output_path / f"optimization_report_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    logger.info(f"Generating optimization report: {report_path}")

    # Generate plots
    try:
        history_fig = optimizer.get_optimization_history_plot()
        importance_fig = optimizer.get_param_importances_plot()

        history_html = history_fig.to_html(full_html=False) if history_fig else ""
        importance_html = importance_fig.to_html(full_html=False) if importance_fig else ""
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")
        history_html = ""
        importance_html = ""

    # Get parameter importance
    param_importance = optimizer.get_importance()

    # Build HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Hyperparameter Optimization Report - {model_type}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            background: #e8f5e9;
            padding: 10px 20px;
            border-radius: 5px;
            margin: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2e7d32;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f5f5f5;
        }}
        .code {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            font-family: monospace;
            overflow-x: auto;
        }}
        h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Hyperparameter Optimization Report</h1>
        <p>Model: {model_type.upper()} | Metric: {result.metric} | Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="card">
        <h2>Summary</h2>
        <div class="metric">
            <div class="metric-value">{result.best_value:.4f}</div>
            <div class="metric-label">Best {result.metric}</div>
        </div>
        <div class="metric">
            <div class="metric-value">{result.n_complete}</div>
            <div class="metric-label">Completed Trials</div>
        </div>
        <div class="metric">
            <div class="metric-value">{result.n_pruned}</div>
            <div class="metric-label">Pruned Trials</div>
        </div>
        <div class="metric">
            <div class="metric-value">{result.duration_seconds:.1f}s</div>
            <div class="metric-label">Duration</div>
        </div>
    </div>

    <div class="card">
        <h2>Best Parameters</h2>
        <div class="code">
{json.dumps(result.best_params, indent=2)}
        </div>
    </div>

    {"<div class='card'><h2>Parameter Importance</h2><table><tr><th>Parameter</th><th>Importance</th></tr>" +
     "".join([f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in sorted(param_importance.items(), key=lambda x: -x[1])]) +
     "</table></div>" if param_importance else ""}

    {"<div class='card'><h2>Optimization History</h2>" + history_html + "</div>" if history_html else ""}

    {"<div class='card'><h2>Parameter Importances</h2>" + importance_html + "</div>" if importance_html else ""}

    <div class="card">
        <h2>Trial History</h2>
        <table>
            <tr>
                <th>Trial</th>
                <th>Value</th>
                <th>State</th>
                <th>Duration (s)</th>
            </tr>
            {"".join([f"<tr><td>{t['trial']}</td><td>{t['value']:.4f if t['value'] else 'N/A'}</td><td>{t['state']}</td><td>{t['duration']:.2f if t['duration'] else 'N/A'}</td></tr>" for t in result.optimization_history[:50]])}
        </table>
        {f"<p><i>Showing first 50 of {len(result.optimization_history)} trials</i></p>" if len(result.optimization_history) > 50 else ""}
    </div>

    <div class="card">
        <h2>Usage Example</h2>
        <div class="code">
from ml.models.{model_type}_model import {'XGBoostTradeClassifier' if model_type == 'xgboost' else 'RandomForestTradeClassifier' if model_type == 'random_forest' else 'LSTMTradePredictor'}

# Load optimized parameters
params = {json.dumps(result.best_params, indent=4)}

# Create model with optimized parameters
model = {'XGBoostTradeClassifier' if model_type == 'xgboost' else 'RandomForestTradeClassifier' if model_type == 'random_forest' else 'LSTMTradePredictor'}(**params)

# Train model
model.train(X_train, y_train)
        </div>
    </div>
</body>
</html>
"""

    with open(report_path, 'w') as f:
        f.write(html_content)

    logger.info(f"Report saved to {report_path}")
    return str(report_path)


def save_config(result, output_path: str, model_type: str):
    """
    Save best parameters to config file.

    Args:
        result: OptimizationResult
        output_path: Output path for config
        model_type: Model type
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config if it exists
    if output_path.exists():
        with open(output_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    # Update with new parameters
    config[model_type] = {
        'params': result.best_params,
        'metric': result.metric,
        'best_value': result.best_value,
        'n_trials': result.n_trials,
        'optimized_at': result.timestamp.isoformat()
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved optimized parameters to {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for ML models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-type', '-m',
        type=str,
        required=True,
        choices=['xgboost', 'random_forest', 'lstm'],
        help='Model type to optimize'
    )

    parser.add_argument(
        '--n-trials', '-n',
        type=int,
        default=100,
        help='Number of optimization trials'
    )

    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=None,
        help='Maximum optimization time in seconds'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='f1',
        choices=['accuracy', 'precision', 'recall', 'f1', 'auc', 'log_loss', 'profit_factor'],
        help='Optimization metric'
    )

    parser.add_argument(
        '--cv-splits',
        type=int,
        default=5,
        help='Number of cross-validation splits'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='config/optimized_params.json',
        help='Output path for best parameters'
    )

    parser.add_argument(
        '--data-file', '-d',
        type=str,
        default=None,
        help='Path to training data CSV file'
    )

    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate HTML optimization report'
    )

    parser.add_argument(
        '--report-dir',
        type=str,
        default='reports/optimization',
        help='Directory for optimization reports'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel optimization jobs'
    )

    parser.add_argument(
        '--pruner',
        type=str,
        default='median',
        choices=['median', 'hyperband'],
        help='Pruner type for early stopping'
    )

    parser.add_argument(
        '--storage',
        type=str,
        default=None,
        help='Database URL for study persistence (e.g., sqlite:///optuna.db)'
    )

    parser.add_argument(
        '--study-name',
        type=str,
        default=None,
        help='Study name for persistence'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level="DEBUG"
        )

    # Print banner
    print("\n" + "=" * 60)
    print("       ML Model Hyperparameter Optimization")
    print("=" * 60)
    print(f"  Model:   {args.model_type}")
    print(f"  Metric:  {args.metric}")
    print(f"  Trials:  {args.n_trials}")
    print(f"  Timeout: {args.timeout}s" if args.timeout else "  Timeout: None")
    print("=" * 60 + "\n")

    # Load data
    X, y, feature_names = load_training_data(args.data_file)

    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Class distribution: {np.bincount(y)}")

    # Import optimizer
    try:
        from ml.optimization.optuna_optimizer import ModelOptimizer
    except ImportError as e:
        logger.error(f"Failed to import optimizer: {e}")
        logger.error("Make sure Optuna is installed: pip install optuna")
        sys.exit(1)

    # Create optimizer
    optimizer = ModelOptimizer(
        X=X,
        y=y,
        model_type=args.model_type
    )

    # Run optimization
    try:
        result = optimizer.optimize(
            n_trials=args.n_trials,
            timeout=args.timeout,
            metric=args.metric,
            n_cv_splits=args.cv_splits,
            seed=args.seed,
            n_jobs=args.n_jobs,
            pruner_type=args.pruner,
            storage=args.storage,
            study_name=args.study_name,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        result = optimizer.result
        if result is None:
            sys.exit(1)

    # Print results
    print("\n" + "=" * 60)
    print("                 OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"  Best {args.metric}: {result.best_value:.4f}")
    print(f"  Completed trials: {result.n_complete}")
    print(f"  Pruned trials: {result.n_pruned}")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    print("\n  Best Parameters:")
    for param, value in result.best_params.items():
        if isinstance(value, float):
            print(f"    {param}: {value:.6f}")
        else:
            print(f"    {param}: {value}")
    print("=" * 60 + "\n")

    # Save configuration
    save_config(result, args.output, args.model_type)

    # Generate report if requested
    if args.generate_report:
        report_path = generate_optimization_report(
            result, optimizer, args.report_dir, args.model_type
        )
        print(f"Report generated: {report_path}")

    # Print parameter importance
    importance = optimizer.get_importance()
    if importance:
        print("\nParameter Importance:")
        for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"  {param}: {imp:.4f}")

    print(f"\nBest parameters saved to: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
