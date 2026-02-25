#!/usr/bin/env python3
"""
Training Script for XGBoost Trade Classifier

This script trains an XGBoost classifier to predict trade success probability.
Uses time-series cross-validation to prevent look-ahead bias.

Usage:
    python scripts/train_xgboost.py [options]

Options:
    --data PATH           Training data path (default: data/training/signals_labeled.csv)
    --output PATH         Model output path (default: models/xgboost_trade_classifier.pkl)
    --n-estimators INT    Number of trees (default: 200)
    --max-depth INT       Max tree depth (default: 6)
    --learning-rate FLOAT Learning rate (default: 0.05)
    --cv-splits INT       Number of CV splits (default: 5)
    --use-gpu             Use GPU acceleration
    --verbose             Enable verbose logging
"""

# Fix curl_cffi chrome136 impersonation issue
from curl_cffi.requests import impersonate
impersonate.DEFAULT_CHROME = 'chrome110'
if hasattr(impersonate, 'REAL_TARGET_MAP'):
    impersonate.REAL_TARGET_MAP['chrome'] = 'chrome110'

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

try:
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, precision_score,
        recall_score, f1_score, confusion_matrix, classification_report
    )
    import joblib
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    logger.error("Required dependencies not available. Install: pip install xgboost scikit-learn joblib")


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
    log_file = log_dir / f"train_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_file, format=log_format, level="DEBUG", rotation="10 MB")


def load_training_data(data_path: str) -> pd.DataFrame:
    """Load and validate training data."""
    logger.info(f"Loading training data from {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")

    # Validate required columns
    required = ['success']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature column names from dataframe."""
    exclude = [
        'symbol', 'date', 'direction', 'success', 'hit_stop', 'exit_day',
        'entry_price', 'exit_price', 'stop_loss', 'target', 'atr',
        'pnl_r', 'max_favorable_r', 'max_adverse_r'
    ]
    features = [c for c in df.columns if c not in exclude]
    return features


def prepare_data(df: pd.DataFrame, feature_columns: list):
    """Prepare feature matrix and target vector."""
    # Handle missing values
    X = df[feature_columns].fillna(0).values
    y = df['success'].values

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Class distribution: {np.bincount(y.astype(int))}")

    return X, y, feature_columns


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    n_splits: int = 5,
    use_gpu: bool = False
) -> tuple:
    """
    Train XGBoost classifier with time-series cross-validation.

    Returns:
        Tuple of (model, metrics, feature_importance)
    """
    logger.info("=" * 60)
    logger.info("Training XGBoost Classifier")
    logger.info("=" * 60)
    logger.info(f"  n_estimators: {n_estimators}")
    logger.info(f"  max_depth: {max_depth}")
    logger.info(f"  learning_rate: {learning_rate}")
    logger.info(f"  cv_splits: {n_splits}")
    logger.info(f"  use_gpu: {use_gpu}")
    logger.info("=" * 60)

    # Model parameters
    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1,
    }

    if use_gpu:
        try:
            params['tree_method'] = 'gpu_hist'
            params['gpu_id'] = 0
            logger.info("GPU acceleration enabled")
        except Exception:
            logger.warning("GPU not available, using CPU")
            params['tree_method'] = 'hist'
    else:
        params['tree_method'] = 'hist'

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_metrics = {
        'accuracy': [],
        'auc': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    logger.info("\nRunning time-series cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        logger.info(f"\nFold {fold + 1}/{n_splits}")
        logger.info(f"  Train size: {len(train_idx)}, Val size: {len(val_idx)}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Predict
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # Calculate metrics
        cv_metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        cv_metrics['auc'].append(roc_auc_score(y_val, y_prob))
        cv_metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        cv_metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        cv_metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))

        logger.info(f"  Accuracy: {cv_metrics['accuracy'][-1]:.4f}")
        logger.info(f"  AUC: {cv_metrics['auc'][-1]:.4f}")
        logger.info(f"  F1: {cv_metrics['f1'][-1]:.4f}")

    # Print CV summary
    logger.info("\n" + "=" * 60)
    logger.info("Cross-Validation Summary")
    logger.info("=" * 60)
    for metric, values in cv_metrics.items():
        logger.info(f"  {metric}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

    # Train final model on all data
    logger.info("\nTraining final model on all data...")
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y, verbose=False)

    # Feature importance
    importance = dict(zip(feature_names, final_model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    logger.info("\nTop 15 Feature Importances:")
    for i, (feat, imp) in enumerate(list(importance.items())[:15]):
        bar = '█' * int(imp * 50)
        logger.info(f"  {i+1:2d}. {feat:30s}: {imp:.4f} {bar}")

    # Final metrics
    y_pred_final = final_model.predict(X)
    y_prob_final = final_model.predict_proba(X)[:, 1]

    final_metrics = {
        'cv_accuracy_mean': float(np.mean(cv_metrics['accuracy'])),
        'cv_accuracy_std': float(np.std(cv_metrics['accuracy'])),
        'cv_auc_mean': float(np.mean(cv_metrics['auc'])),
        'cv_auc_std': float(np.std(cv_metrics['auc'])),
        'cv_precision_mean': float(np.mean(cv_metrics['precision'])),
        'cv_recall_mean': float(np.mean(cv_metrics['recall'])),
        'cv_f1_mean': float(np.mean(cv_metrics['f1'])),
        'train_accuracy': float(accuracy_score(y, y_pred_final)),
        'train_auc': float(roc_auc_score(y, y_prob_final)),
        'n_samples': len(y),
        'n_features': X.shape[1],
        'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
        'timestamp': datetime.now().isoformat()
    }

    return final_model, final_metrics, importance


def save_model(model, output_path: str, metrics: dict, importance: dict, feature_names: list):
    """Save trained model and metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, output_path)
    logger.success(f"Model saved to {output_path}")

    # Save metrics
    metrics_path = output_path.parent / f"{output_path.stem}_metrics.json"
    # Convert numpy types
    metrics_clean = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer, np.floating)):
            metrics_clean[k] = float(v)
        elif isinstance(v, dict):
            metrics_clean[k] = {str(kk): float(vv) if isinstance(vv, (np.integer, np.floating)) else vv
                               for kk, vv in v.items()}
        else:
            metrics_clean[k] = v

    with open(metrics_path, 'w') as f:
        json.dump(metrics_clean, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Save feature importance
    importance_path = output_path.parent / f"{output_path.stem}_importance.json"
    importance_clean = {k: float(v) for k, v in importance.items()}
    with open(importance_path, 'w') as f:
        json.dump(importance_clean, f, indent=2)
    logger.info(f"Feature importance saved to {importance_path}")

    # Save feature names
    features_path = output_path.parent / f"{output_path.stem}_features.json"
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"Feature names saved to {features_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost trade classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Training data path (default: data/training/signals_labeled.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Model output path (default: models/xgboost_trade_classifier.pkl)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=200,
        help='Number of trees (default: 200)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='Max tree depth (default: 6)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.05,
        help='Learning rate (default: 0.05)'
    )
    parser.add_argument(
        '--cv-splits',
        type=int,
        default=5,
        help='Number of CV splits (default: 5)'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU acceleration'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if not DEPS_AVAILABLE:
        logger.error("Required dependencies not available")
        return 1

    setup_logging(args.verbose)

    logger.info("=" * 80)
    logger.info("XGBoost Trade Classifier Training")
    logger.info("=" * 80)

    # Set paths
    data_path = args.data or str(
        Path(__file__).parent.parent / "data" / "training" / "signals_labeled.csv"
    )
    output_path = args.output or str(
        Path(__file__).parent.parent / "models" / "xgboost_trade_classifier.pkl"
    )

    # Load data
    try:
        df = load_training_data(data_path)
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return 1

    # Prepare data
    feature_columns = get_feature_columns(df)
    X, y, feature_names = prepare_data(df, feature_columns)

    # Train model
    start_time = time.time()
    model, metrics, importance = train_xgboost(
        X, y, feature_names,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_splits=args.cv_splits,
        use_gpu=args.use_gpu
    )
    training_time = time.time() - start_time

    metrics['training_time_seconds'] = training_time
    logger.info(f"\nTraining completed in {training_time:.2f} seconds")

    # Save model
    save_model(model, output_path, metrics, importance, feature_names)

    logger.info("\n" + "=" * 80)
    logger.success("XGBoost training completed successfully!")
    logger.info("=" * 80)

    logger.info(f"\nModel saved to: {output_path}")
    logger.info("\nTo use the trained model:")
    logger.info("  import joblib")
    logger.info(f"  model = joblib.load('{output_path}')")
    logger.info("  proba = model.predict_proba(features)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
