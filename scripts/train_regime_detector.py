#!/usr/bin/env python3
"""
Training Script for Market Regime Detector

This script trains a Hidden Markov Model on historical SPY data to detect
market regimes and evaluates the model's performance.

Usage:
    python scripts/train_regime_detector.py [options]

Options:
    --symbol SYMBOL        Ticker symbol to train on (default: SPY)
    --period PERIOD        Data period (default: 5y)
    --n-regimes N          Number of regimes (default: 4)
    --n-iter N             HMM iterations (default: 100)
    --output PATH          Model output path
    --visualize            Generate visualization plots
    --evaluate             Run detailed evaluation
"""

# Fix curl_cffi chrome136 impersonation issue in Docker
# curl_cffi 0.13.0 maps 'chrome' to 'chrome136' which may not be supported
from curl_cffi.requests import impersonate
impersonate.DEFAULT_CHROME = 'chrome110'
if hasattr(impersonate, 'REAL_TARGET_MAP'):
    impersonate.REAL_TARGET_MAP['chrome'] = 'chrome110'

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

# Import the regime detector
from ml.regime_detector import MarketRegimeDetector

# Import GPU utilities for accelerated training
try:
    from ml.gpu_utils import (
        is_gpu_available,
        get_gpu_summary,
        setup_gpu_for_training,
        update_gpu_metrics,
        get_optimal_batch_size,
        GPU_METRICS_AVAILABLE
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPU_METRICS_AVAILABLE = False
    logger.debug("GPU utilities not available")

# Import settings for GPU configuration
try:
    from config.settings import get_settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    logger.warning("matplotlib/seaborn not available - visualization disabled")
    HAS_PLOTTING = False


def setup_logging(verbose: bool = False):
    """Configure logging settings."""
    logger.remove()
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, format=log_format, level=level, colorize=True)

    # Also log to file
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / f"regime_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_file, format=log_format, level="DEBUG", rotation="10 MB")

    logger.info(f"Logging to {log_file}")


def setup_gpu(use_gpu: str = "auto", memory_limit: float = None, device_id: int = None) -> dict:
    """
    Configure GPU for training based on settings and command-line arguments.

    Args:
        use_gpu: GPU usage mode ('auto', 'true', 'false')
        memory_limit: GPU memory limit in MB or fraction (0-1)
        device_id: Specific GPU device ID to use

    Returns:
        Dict with GPU configuration status
    """
    gpu_config = {
        'gpu_available': False,
        'gpu_configured': False,
        'device': 'cpu',
        'error': None
    }

    if not GPU_AVAILABLE:
        logger.info("GPU utilities not available, using CPU")
        return gpu_config

    # Get settings from config if available
    if SETTINGS_AVAILABLE:
        try:
            settings = get_settings()
            if use_gpu == "auto":
                use_gpu = settings.gpu.use_gpu
            if memory_limit is None:
                memory_limit = settings.gpu.gpu_memory_limit
            if device_id is None:
                device_id = settings.gpu.gpu_device_id
        except Exception as e:
            logger.debug(f"Could not load GPU settings: {e}")

    # Log GPU availability
    gpu_summary = get_gpu_summary()
    gpu_config['gpu_available'] = gpu_summary.get('gpu_available', False)

    if gpu_config['gpu_available']:
        logger.info("=" * 60)
        logger.info("GPU CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"GPU Count: {gpu_summary.get('gpu_count', 0)}")
        logger.info(f"CUDA Available: {gpu_summary.get('cuda_available', False)}")
        logger.info(f"MPS Available (Apple): {gpu_summary.get('mps_available', False)}")
        logger.info(f"Device Strategy: {gpu_summary.get('device_strategy', 'cpu')}")

        for gpu_info in gpu_summary.get('gpus', []):
            logger.info(
                f"  GPU {gpu_info.get('device_id', 0)}: {gpu_info.get('name', 'Unknown')} "
                f"({gpu_info.get('memory_total_mb', 0):.0f}MB total)"
            )

        logger.info("=" * 60)

    # Configure GPU
    try:
        result = setup_gpu_for_training(
            use_gpu=use_gpu,
            memory_limit=memory_limit,
            device_id=device_id,
            enable_mixed_prec=False  # Disable mixed precision for HMM (not TensorFlow-based)
        )

        gpu_config.update(result)

        if result.get('gpu_configured'):
            logger.success(f"GPU configured successfully: {result.get('device', 'GPU')}")
        else:
            logger.info("Using CPU for training")

    except Exception as e:
        gpu_config['error'] = str(e)
        logger.warning(f"Error configuring GPU: {e}. Using CPU.")

    return gpu_config


def log_gpu_usage():
    """Log current GPU usage during training."""
    if not GPU_AVAILABLE or not GPU_METRICS_AVAILABLE:
        return

    try:
        update_gpu_metrics()
    except Exception as e:
        logger.debug(f"Could not update GPU metrics: {e}")


def visualize_regimes(detector: MarketRegimeDetector, data: pd.DataFrame, results: pd.DataFrame):
    """
    Generate visualization plots for regime detection.

    Args:
        detector: Trained MarketRegimeDetector
        data: Original market data
        results: Regime prediction results
    """
    if not HAS_PLOTTING:
        logger.warning("Plotting libraries not available - skipping visualization")
        return

    logger.info("Generating visualization plots...")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "ml" / "data" / "visualizations"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # 1. Price chart with regime overlays
    fig, ax = plt.subplots(figsize=(15, 8))

    # Align data with results
    aligned_data = data.loc[results.index]

    # Plot price
    ax.plot(aligned_data.index, aligned_data['Close'], color='black',
            linewidth=1, label='SPY Close Price', alpha=0.7)

    # Color regions by regime
    regime_colors = {
        'bull_trending': 'green',
        'bear_trending': 'red',
        'high_volatility': 'orange',
        'low_volatility': 'blue'
    }

    current_regime = None
    start_idx = 0

    for i in range(len(results)):
        if results['regime_name'].iloc[i] != current_regime:
            if current_regime is not None:
                # Fill the previous regime
                ax.axvspan(
                    results.index[start_idx],
                    results.index[i-1],
                    alpha=0.2,
                    color=regime_colors.get(current_regime, 'gray'),
                    label=current_regime if current_regime not in ax.get_legend_handles_labels()[1] else ""
                )
            current_regime = results['regime_name'].iloc[i]
            start_idx = i

    # Fill the last regime
    if current_regime is not None:
        ax.axvspan(
            results.index[start_idx],
            results.index[-1],
            alpha=0.2,
            color=regime_colors.get(current_regime, 'gray'),
            label=current_regime if current_regime not in ax.get_legend_handles_labels()[1] else ""
        )

    ax.set_title('Market Regimes Detected by HMM', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'regime_overlay.png', dpi=300, bbox_inches='tight')
    logger.success(f"Saved regime overlay plot to {output_dir / 'regime_overlay.png'}")
    plt.close()

    # 2. Regime distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Pie chart
    regime_counts = results['regime_name'].value_counts()
    colors_pie = [regime_colors.get(regime, 'gray') for regime in regime_counts.index]
    ax1.pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%',
            colors=colors_pie, startangle=90)
    ax1.set_title('Regime Distribution', fontsize=14, fontweight='bold')

    # Bar chart
    ax2.bar(regime_counts.index, regime_counts.values, color=colors_pie, alpha=0.7)
    ax2.set_title('Regime Frequency', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Regime', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'regime_distribution.png', dpi=300, bbox_inches='tight')
    logger.success(f"Saved regime distribution plot to {output_dir / 'regime_distribution.png'}")
    plt.close()

    # 3. Confidence scores over time
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, regime in enumerate(['bull_trending', 'bear_trending', 'high_volatility', 'low_volatility']):
        col_name = f'confidence_{regime}'
        if col_name in results.columns:
            ax = axes[idx]
            ax.plot(results.index, results[col_name],
                   color=regime_colors.get(regime, 'gray'), linewidth=1)
            ax.fill_between(results.index, 0, results[col_name],
                           color=regime_colors.get(regime, 'gray'), alpha=0.3)
            ax.set_title(f'{regime.replace("_", " ").title()} Confidence',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel('Confidence Score', fontsize=10)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_scores.png', dpi=300, bbox_inches='tight')
    logger.success(f"Saved confidence scores plot to {output_dir / 'confidence_scores.png'}")
    plt.close()

    # 4. Transition matrix heatmap
    if detector.model.transmat_ is not None:
        fig, ax = plt.subplots(figsize=(10, 8))

        regime_labels = [detector.regime_mapping.get(i, f'State {i}')
                        for i in range(detector.n_regimes)]

        sns.heatmap(detector.model.transmat_, annot=True, fmt='.3f',
                   cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Transition Probability'},
                   xticklabels=regime_labels, yticklabels=regime_labels)

        ax.set_title('Regime Transition Matrix', fontsize=16, fontweight='bold')
        ax.set_xlabel('To Regime', fontsize=12)
        ax.set_ylabel('From Regime', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_dir / 'transition_matrix.png', dpi=300, bbox_inches='tight')
        logger.success(f"Saved transition matrix plot to {output_dir / 'transition_matrix.png'}")
        plt.close()

    # 5. Regime statistics comparison
    stats = detector.get_regime_statistics(aligned_data, results['regime_state'].values)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Average returns
    axes[0, 0].bar(stats['regime'], stats['avg_return'],
                   color=[regime_colors.get(r, 'gray') for r in stats['regime']], alpha=0.7)
    axes[0, 0].set_title('Average Daily Returns by Regime', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Return', fontsize=10)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    # Volatility
    axes[0, 1].bar(stats['regime'], stats['volatility'],
                   color=[regime_colors.get(r, 'gray') for r in stats['regime']], alpha=0.7)
    axes[0, 1].set_title('Volatility by Regime', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Volatility', fontsize=10)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Sharpe ratio
    axes[1, 0].bar(stats['regime'], stats['sharpe_ratio'],
                   color=[regime_colors.get(r, 'gray') for r in stats['regime']], alpha=0.7)
    axes[1, 0].set_title('Sharpe Ratio by Regime', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Sharpe Ratio', fontsize=10)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    # Max drawdown
    axes[1, 1].bar(stats['regime'], stats['max_drawdown'],
                   color=[regime_colors.get(r, 'gray') for r in stats['regime']], alpha=0.7)
    axes[1, 1].set_title('Max Drawdown by Regime', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Max Drawdown', fontsize=10)
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'regime_statistics.png', dpi=300, bbox_inches='tight')
    logger.success(f"Saved regime statistics plot to {output_dir / 'regime_statistics.png'}")
    plt.close()

    logger.success(f"All visualizations saved to {output_dir}")


def evaluate_model(detector: MarketRegimeDetector, test_symbol: str = "QQQ"):
    """
    Evaluate model performance on test data.

    Args:
        detector: Trained MarketRegimeDetector
        test_symbol: Symbol to test on (different from training)
    """
    logger.info(f"Evaluating model on {test_symbol}...")

    try:
        # Fetch test data
        test_data = detector.fetch_data(symbol=test_symbol, period="2y")

        # Predict regimes
        test_results = detector.predict_sequence(test_data)

        # Calculate statistics
        test_stats = detector.get_regime_statistics(test_data, test_results['regime_state'].values)

        logger.info("\nTest Set Performance:")
        logger.info(f"Symbol: {test_symbol}")
        logger.info(f"Samples: {len(test_results)}")
        logger.info("\nRegime Statistics:")
        logger.info("\n" + test_stats.to_string())

        # Analyze transitions
        transitions = detector.get_regime_transitions(test_results)
        logger.info(f"\nNumber of regime transitions: {len(transitions)}")

        if len(transitions) > 0:
            logger.info("\nMost common transitions:")
            transition_counts = transitions.groupby(['from_regime', 'to_regime']).size()
            top_transitions = transition_counts.nlargest(5)
            for (from_r, to_r), count in top_transitions.items():
                logger.info(f"  {from_r} -> {to_r}: {count} times")

        # Calculate out-of-sample metrics
        returns = test_data['Close'].pct_change().loc[test_results.index]

        metrics = {
            'test_symbol': test_symbol,
            'n_samples': len(test_results),
            'n_transitions': len(transitions),
            'regime_distribution': test_results['regime_name'].value_counts().to_dict(),
            'avg_return': float(returns.mean()),
            'volatility': float(returns.std()),
            'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        }

        return metrics, test_results

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


def save_metrics(metrics: dict, output_path: Path):
    """Save training metrics to JSON file."""
    metrics_file = output_path.parent / "training_metrics.json"

    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    metrics_clean = convert_types(metrics)

    with open(metrics_file, 'w') as f:
        json.dump(metrics_clean, f, indent=2, default=str)

    logger.success(f"Saved training metrics to {metrics_file}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Market Regime Detector using Hidden Markov Models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='SPY',
        help='Ticker symbol to train on (default: SPY)'
    )
    parser.add_argument(
        '--period',
        type=str,
        default='5y',
        help='Data period (default: 5y)'
    )
    parser.add_argument(
        '--n-regimes',
        type=int,
        default=4,
        help='Number of market regimes (default: 4)'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=100,
        help='Number of HMM EM iterations (default: 100)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Model output path (default: models/regime_detector.pkl)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run detailed evaluation on test data'
    )
    parser.add_argument(
        '--test-symbol',
        type=str,
        default='QQQ',
        help='Symbol for test evaluation (default: QQQ)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    # GPU options
    parser.add_argument(
        '--use-gpu',
        type=str,
        default='auto',
        choices=['auto', 'true', 'false', 'yes', 'no'],
        help='GPU usage: auto (detect), true (force GPU), false (force CPU)'
    )
    parser.add_argument(
        '--gpu-memory-limit',
        type=float,
        default=None,
        help='GPU memory limit in MB or fraction (0-1)'
    )
    parser.add_argument(
        '--gpu-device',
        type=int,
        default=None,
        help='Specific GPU device ID to use'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    logger.info("="*80)
    logger.info("Market Regime Detector Training")
    logger.info("="*80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Period: {args.period}")
    logger.info(f"Regimes: {args.n_regimes}")
    logger.info(f"Iterations: {args.n_iter}")
    logger.info("="*80)

    # Setup GPU (for any TensorFlow-based models that might be used)
    gpu_config = setup_gpu(
        use_gpu=args.use_gpu,
        memory_limit=args.gpu_memory_limit,
        device_id=args.gpu_device
    )

    if gpu_config.get('gpu_configured'):
        logger.info(f"Training device: {gpu_config.get('device', 'GPU')}")
    else:
        logger.info("Training device: CPU")

    # Initialize detector
    output_path = args.output or str(
        Path(__file__).parent.parent / "models" / "regime_detector.pkl"
    )

    detector = MarketRegimeDetector(
        n_regimes=args.n_regimes,
        n_iter=args.n_iter,
        random_state=42,
        model_path=output_path
    )

    # Train model
    logger.info("\nStep 1: Training HMM model...")
    training_start_time = time.time()

    try:
        # Log GPU usage at start of training
        log_gpu_usage()

        metrics = detector.train(
            symbol=args.symbol,
            period=args.period,
            interval='1d'
        )

        training_duration = time.time() - training_start_time

        # Log GPU usage at end of training
        log_gpu_usage()

        logger.success("\nTraining completed successfully!")
        logger.info(f"Training duration: {training_duration:.2f} seconds")

        # Add GPU info to metrics
        metrics['training_duration_seconds'] = training_duration
        metrics['gpu_config'] = {
            'gpu_available': gpu_config.get('gpu_available', False),
            'gpu_configured': gpu_config.get('gpu_configured', False),
            'device': gpu_config.get('device', 'cpu')
        }
        logger.info("\nTraining Metrics:")
        logger.info(f"  Log Likelihood: {metrics['log_likelihood']:.2f}")
        logger.info(f"  Samples: {metrics['n_samples']}")
        logger.info(f"  Features: {metrics['n_features']}")

        if 'silhouette_score' in metrics:
            logger.info(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        if 'davies_bouldin_score' in metrics:
            logger.info(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")

        logger.info("\n  Regime Distribution:")
        for regime, count in metrics['regime_distribution'].items():
            pct = count / metrics['n_samples'] * 100
            logger.info(f"    {regime}: {count} ({pct:.1f}%)")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

    # Save model
    logger.info("\nStep 2: Saving model...")
    try:
        detector.save_model(output_path)
        logger.success(f"Model saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return 1

    # Save metrics
    try:
        save_metrics(metrics, Path(output_path))
    except Exception as e:
        logger.warning(f"Failed to save metrics: {e}")

    # Predict current regime
    logger.info("\nStep 3: Testing current regime prediction...")
    try:
        current_regime, info = detector.predict(return_confidence=True)

        logger.info(f"\nCurrent Market Regime: {current_regime.upper()}")
        logger.info(f"Timestamp: {info['timestamp']}")
        logger.info("\nConfidence Scores:")
        for regime, score in info['confidence_scores'].items():
            bar = '█' * int(score * 50)
            logger.info(f"  {regime:20s}: {score:.4f} {bar}")

        logger.info("\nRecommended Strategy Allocation:")
        for strategy, weight in info['strategy_allocation'].items():
            logger.info(f"  {strategy:20s}: {weight*100:5.1f}%")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")

    # Generate visualizations
    if args.visualize:
        logger.info("\nStep 4: Generating visualizations...")
        try:
            results = detector.predict_sequence(detector.raw_data)
            visualize_regimes(detector, detector.raw_data, results)
        except Exception as e:
            logger.error(f"Visualization failed: {e}")

    # Run evaluation
    if args.evaluate:
        logger.info("\nStep 5: Running model evaluation...")
        try:
            eval_metrics, test_results = evaluate_model(detector, args.test_symbol)

            # Combine with training metrics
            all_metrics = {
                'training': metrics,
                'evaluation': eval_metrics,
                'timestamp': datetime.now().isoformat()
            }

            # Save combined metrics
            save_metrics(all_metrics, Path(output_path))

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

    # Get regime statistics
    logger.info("\nRegime Performance Statistics:")
    try:
        stats = detector.get_regime_statistics(detector.raw_data)
        logger.info("\n" + stats.to_string())
    except Exception as e:
        logger.warning(f"Failed to calculate statistics: {e}")

    logger.info("\n" + "="*80)
    logger.success("Training pipeline completed successfully!")
    logger.info("="*80)

    logger.info(f"\nModel saved to: {output_path}")
    logger.info("\nTo use the trained model:")
    logger.info("  from ml.regime_detector import MarketRegimeDetector")
    logger.info("  detector = MarketRegimeDetector()")
    logger.info(f"  detector.load_model('{output_path}')")
    logger.info("  regime, info = detector.predict()")

    return 0


if __name__ == "__main__":
    sys.exit(main())
