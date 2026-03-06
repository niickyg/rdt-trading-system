#!/usr/bin/env python3
"""
Training Script for Stacked Ensemble Model

This script trains a stacked ensemble that combines XGBoost and Random Forest
classifiers using a Logistic Regression meta-learner.

Usage:
    python scripts/train_ensemble.py [options]

Options:
    --data PATH           Training data path (default: data/training/signals_labeled.csv)
    --output PATH         Model output directory (default: models/ensemble/)
    --xgb-estimators INT  XGBoost trees (default: 200)
    --rf-estimators INT   Random Forest trees (default: 200)
    --cv-splits INT       Number of CV splits (default: 5)
    --use-gpu             Use GPU acceleration for XGBoost
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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, precision_score,
        recall_score, f1_score, confusion_matrix, log_loss
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
    log_file = log_dir / f"train_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_file, format=log_format, level="DEBUG", rotation="10 MB")


def load_training_data(data_path: str) -> pd.DataFrame:
    """Load and validate training data."""
    logger.info(f"Loading training data from {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")

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
    X = df[feature_columns].fillna(0).values
    y = df['success'].values

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Class distribution: {np.bincount(y.astype(int))}")

    return X, y, feature_columns


class StackedEnsembleTrainer:
    """
    Trainer for stacked ensemble model.

    Level 0: XGBoost, Random Forest
    Level 1: Logistic Regression meta-learner
    """

    def __init__(
        self,
        xgb_n_estimators: int = 200,
        xgb_max_depth: int = 6,
        xgb_learning_rate: float = 0.05,
        rf_n_estimators: int = 200,
        rf_max_depth: int = 10,
        use_gpu: bool = False,
        random_state: int = 42
    ):
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.use_gpu = use_gpu
        self.random_state = random_state

        # Models
        self.xgboost_model = None
        self.rf_model = None
        self.meta_learner = None

        # Feature names
        self.feature_names = []

        # Metrics
        self.metrics = {}

    def _create_xgboost(self) -> xgb.XGBClassifier:
        """Create XGBoost classifier."""
        params = {
            'n_estimators': self.xgb_n_estimators,
            'max_depth': self.xgb_max_depth,
            'learning_rate': self.xgb_learning_rate,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': self.random_state,
            'n_jobs': -1,
        }

        if self.use_gpu:
            try:
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = 0
            except Exception:
                params['tree_method'] = 'hist'
        else:
            params['tree_method'] = 'hist'

        return xgb.XGBClassifier(**params)

    def _create_random_forest(self) -> RandomForestClassifier:
        """Create Random Forest classifier."""
        return RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )

    def _create_meta_learner(self) -> LogisticRegression:
        """Create Logistic Regression meta-learner."""
        return LogisticRegression(
            C=1.0,
            random_state=self.random_state,
            max_iter=1000,
            solver='lbfgs'
        )

    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from base models for meta-learner."""
        xgb_proba = self.xgboost_model.predict_proba(X)[:, 1]
        rf_proba = self.rf_model.predict_proba(X)[:, 1]

        # Stack predictions as features for meta-learner
        return np.column_stack([xgb_proba, rf_proba])

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        n_splits: int = 5
    ) -> dict:
        """
        Train stacked ensemble with out-of-fold predictions.

        To prevent overfitting, we use out-of-fold predictions
        to train the meta-learner.
        """
        logger.info("=" * 60)
        logger.info("Training Stacked Ensemble")
        logger.info("=" * 60)
        logger.info(f"  XGBoost estimators: {self.xgb_n_estimators}")
        logger.info(f"  XGBoost max_depth: {self.xgb_max_depth}")
        logger.info(f"  RF estimators: {self.rf_n_estimators}")
        logger.info(f"  RF max_depth: {self.rf_max_depth}")
        logger.info(f"  CV splits: {n_splits}")
        logger.info(f"  GPU: {self.use_gpu}")
        logger.info("=" * 60)

        self.feature_names = feature_names

        # Time-series cross-validation for out-of-fold predictions
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Arrays to collect OOF predictions
        oof_xgb = np.zeros(len(X))
        oof_rf = np.zeros(len(X))
        oof_indices = []

        # CV metrics
        cv_metrics = {
            'xgb_auc': [],
            'rf_auc': [],
            'ensemble_auc': []
        }

        logger.info("\nPhase 1: Generating out-of-fold predictions...")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"\nFold {fold + 1}/{n_splits}")
            logger.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train XGBoost
            xgb_model = self._create_xgboost()
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
            oof_xgb[val_idx] = xgb_proba

            # Train Random Forest
            rf_model = self._create_random_forest()
            rf_model.fit(X_train, y_train)
            rf_proba = rf_model.predict_proba(X_val)[:, 1]
            oof_rf[val_idx] = rf_proba

            # Track indices and metrics
            oof_indices.extend(val_idx)

            cv_metrics['xgb_auc'].append(roc_auc_score(y_val, xgb_proba))
            cv_metrics['rf_auc'].append(roc_auc_score(y_val, rf_proba))

            # Simple average for ensemble
            ensemble_proba = (xgb_proba + rf_proba) / 2
            cv_metrics['ensemble_auc'].append(roc_auc_score(y_val, ensemble_proba))

            logger.info(f"  XGB AUC: {cv_metrics['xgb_auc'][-1]:.4f}")
            logger.info(f"  RF AUC: {cv_metrics['rf_auc'][-1]:.4f}")
            logger.info(f"  Ensemble AUC: {cv_metrics['ensemble_auc'][-1]:.4f}")

        # Print CV summary
        logger.info("\n" + "=" * 60)
        logger.info("Out-of-Fold CV Summary")
        logger.info("=" * 60)
        logger.info(f"  XGBoost AUC: {np.mean(cv_metrics['xgb_auc']):.4f} (+/- {np.std(cv_metrics['xgb_auc']):.4f})")
        logger.info(f"  RF AUC: {np.mean(cv_metrics['rf_auc']):.4f} (+/- {np.std(cv_metrics['rf_auc']):.4f})")
        logger.info(f"  Ensemble AUC: {np.mean(cv_metrics['ensemble_auc']):.4f} (+/- {np.std(cv_metrics['ensemble_auc']):.4f})")

        # Train meta-learner on OOF predictions
        logger.info("\nPhase 2: Training meta-learner on OOF predictions...")

        # Only use indices that were in validation sets
        oof_indices = np.array(oof_indices)
        meta_X = np.column_stack([oof_xgb[oof_indices], oof_rf[oof_indices]])
        meta_y = y[oof_indices]

        self.meta_learner = self._create_meta_learner()
        self.meta_learner.fit(meta_X, meta_y)

        # Get meta-learner weights
        meta_weights = self.meta_learner.coef_[0]
        logger.info(f"  Meta-learner weights: XGBoost={meta_weights[0]:.4f}, RF={meta_weights[1]:.4f}")

        # Train final base models on all data
        logger.info("\nPhase 3: Training final base models on all data...")

        self.xgboost_model = self._create_xgboost()
        self.xgboost_model.fit(X, y, verbose=False)
        logger.info("  XGBoost trained")

        self.rf_model = self._create_random_forest()
        self.rf_model.fit(X, y)
        logger.info("  Random Forest trained")

        # Final evaluation
        logger.info("\nPhase 4: Final evaluation...")

        final_proba = self.predict_proba(X)
        final_pred = (final_proba >= 0.5).astype(int)

        self.metrics = {
            'cv_xgb_auc_mean': float(np.mean(cv_metrics['xgb_auc'])),
            'cv_xgb_auc_std': float(np.std(cv_metrics['xgb_auc'])),
            'cv_rf_auc_mean': float(np.mean(cv_metrics['rf_auc'])),
            'cv_rf_auc_std': float(np.std(cv_metrics['rf_auc'])),
            'cv_ensemble_auc_mean': float(np.mean(cv_metrics['ensemble_auc'])),
            'cv_ensemble_auc_std': float(np.std(cv_metrics['ensemble_auc'])),
            'train_accuracy': float(accuracy_score(y, final_pred)),
            'train_auc': float(roc_auc_score(y, final_proba)),
            'train_precision': float(precision_score(y, final_pred, zero_division=0)),
            'train_recall': float(recall_score(y, final_pred, zero_division=0)),
            'train_f1': float(f1_score(y, final_pred, zero_division=0)),
            'train_log_loss': float(log_loss(y, final_proba)),
            'meta_weights': {
                'xgboost': float(meta_weights[0]),
                'random_forest': float(meta_weights[1])
            },
            'n_samples': len(y),
            'n_features': X.shape[1],
            'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"  Final Train AUC: {self.metrics['train_auc']:.4f}")
        logger.info(f"  Final Train Accuracy: {self.metrics['train_accuracy']:.4f}")
        logger.info(f"  Final Train F1: {self.metrics['train_f1']:.4f}")

        return self.metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability using the ensemble."""
        if self.xgboost_model is None or self.rf_model is None or self.meta_learner is None:
            raise ValueError("Model not trained")

        base_preds = self._get_base_predictions(X)
        return self.meta_learner.predict_proba(base_preds)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class using the ensemble."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def save(self, output_dir: str):
        """Save ensemble models and metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save base models
        joblib.dump(self.xgboost_model, output_dir / 'xgboost_model.pkl')
        joblib.dump(self.rf_model, output_dir / 'random_forest_model.pkl')
        joblib.dump(self.meta_learner, output_dir / 'meta_learner.pkl')

        logger.success(f"Models saved to {output_dir}")

        # Save metrics
        metrics_clean = {}
        for k, v in self.metrics.items():
            if isinstance(v, (np.integer, np.floating)):
                metrics_clean[k] = float(v)
            elif isinstance(v, dict):
                metrics_clean[k] = {str(kk): float(vv) if isinstance(vv, (np.integer, np.floating)) else vv
                                   for kk, vv in v.items()}
            else:
                metrics_clean[k] = v

        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_clean, f, indent=2)

        # Save feature names
        with open(output_dir / 'feature_names.json', 'w') as f:
            json.dump(self.feature_names, f, indent=2)

        # Save feature importance (from XGBoost)
        xgb_importance = dict(zip(self.feature_names, self.xgboost_model.feature_importances_))
        xgb_importance = dict(sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True))
        with open(output_dir / 'xgboost_importance.json', 'w') as f:
            json.dump({k: float(v) for k, v in xgb_importance.items()}, f, indent=2)

        # Save config for loading
        config = {
            'xgb_n_estimators': self.xgb_n_estimators,
            'xgb_max_depth': self.xgb_max_depth,
            'xgb_learning_rate': self.xgb_learning_rate,
            'rf_n_estimators': self.rf_n_estimators,
            'rf_max_depth': self.rf_max_depth,
            'use_gpu': self.use_gpu,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'version': '1.0.0'
        }
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Metrics and config saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: str) -> 'StackedEnsembleTrainer':
        """Load ensemble from directory."""
        model_dir = Path(model_dir)

        # Load config
        with open(model_dir / 'config.json') as f:
            config = json.load(f)

        # Create instance
        trainer = cls(
            xgb_n_estimators=config.get('xgb_n_estimators', 200),
            xgb_max_depth=config.get('xgb_max_depth', 6),
            xgb_learning_rate=config.get('xgb_learning_rate', 0.05),
            rf_n_estimators=config.get('rf_n_estimators', 200),
            rf_max_depth=config.get('rf_max_depth', 10),
            use_gpu=config.get('use_gpu', False),
            random_state=config.get('random_state', 42)
        )

        # Load models with integrity verification
        from ml.safe_model_loader import safe_load_model
        trainer.xgboost_model = safe_load_model(str(model_dir / 'xgboost_model.pkl'))
        trainer.rf_model = safe_load_model(str(model_dir / 'random_forest_model.pkl'))
        trainer.meta_learner = safe_load_model(str(model_dir / 'meta_learner.pkl'))
        trainer.feature_names = config.get('feature_names', [])

        # Load metrics
        with open(model_dir / 'metrics.json') as f:
            trainer.metrics = json.load(f)

        return trainer


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train stacked ensemble model",
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
        help='Model output directory (default: models/ensemble/)'
    )
    parser.add_argument(
        '--xgb-estimators',
        type=int,
        default=200,
        help='XGBoost trees (default: 200)'
    )
    parser.add_argument(
        '--xgb-depth',
        type=int,
        default=6,
        help='XGBoost max depth (default: 6)'
    )
    parser.add_argument(
        '--rf-estimators',
        type=int,
        default=200,
        help='Random Forest trees (default: 200)'
    )
    parser.add_argument(
        '--rf-depth',
        type=int,
        default=10,
        help='Random Forest max depth (default: 10)'
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
        help='Use GPU acceleration for XGBoost'
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
    logger.info("Stacked Ensemble Training")
    logger.info("=" * 80)

    # Set paths
    data_path = args.data or str(
        Path(__file__).parent.parent / "data" / "training" / "signals_labeled.csv"
    )
    output_dir = args.output or str(
        Path(__file__).parent.parent / "models" / "ensemble"
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

    # Train ensemble
    start_time = time.time()

    trainer = StackedEnsembleTrainer(
        xgb_n_estimators=args.xgb_estimators,
        xgb_max_depth=args.xgb_depth,
        rf_n_estimators=args.rf_estimators,
        rf_max_depth=args.rf_depth,
        use_gpu=args.use_gpu
    )

    metrics = trainer.train(X, y, feature_names, n_splits=args.cv_splits)

    training_time = time.time() - start_time
    logger.info(f"\nTraining completed in {training_time:.2f} seconds")

    # Save ensemble
    trainer.save(output_dir)

    logger.info("\n" + "=" * 80)
    logger.success("Stacked Ensemble training completed successfully!")
    logger.info("=" * 80)

    logger.info(f"\nModels saved to: {output_dir}")
    logger.info("\nTo use the trained ensemble:")
    logger.info("  from scripts.train_ensemble import StackedEnsembleTrainer")
    logger.info(f"  ensemble = StackedEnsembleTrainer.load('{output_dir}')")
    logger.info("  proba = ensemble.predict_proba(features)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
