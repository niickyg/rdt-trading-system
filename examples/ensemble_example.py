"""
Example: Training and Using the ML Ensemble System

Demonstrates:
1. Data preparation for ensemble training
2. Training individual models (XGBoost, Random Forest, LSTM)
3. Training the stacked ensemble with meta-learner
4. Making predictions
5. Analyzing model performance and feature importance
6. Saving and loading models
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from ml.ensemble import StackedEnsemble
from ml.models.xgboost_model import XGBoostTradeClassifier
from ml.models.random_forest_model import RandomForestTradeClassifier
from ml.models.lstm_model import LSTMTradePredictor


def generate_synthetic_training_data(
    n_samples: int = 1000,
    n_features: int = 15,
    sequence_length: int = 20
) -> tuple:
    """
    Generate synthetic training data for demonstration

    In production, this would come from historical trade data with labels:
    - Features: RRS, ATR%, momentum, volume, technical indicators, etc.
    - Target: Binary (1 if trade reached 2R within 10 days, 0 otherwise)
    - Sequences: OHLCV data for LSTM

    Args:
        n_samples: Number of samples
        n_features: Number of features per sample
        sequence_length: Sequence length for LSTM

    Returns:
        Tuple of (X, y, X_lstm, feature_names)
    """
    logger.info(f"Generating {n_samples} synthetic samples...")

    # Random seed for reproducibility
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Feature names
    feature_names = [
        'rrs', 'atr_percent', 'price_momentum_5d', 'price_momentum_10d',
        'volume_ratio', 'trend_strength', 'volatility_percentile',
        'rsi', 'macd', 'bollinger_position', 'spy_trend', 'sector_strength',
        'market_volatility', 'distance_to_sma20', 'distance_to_sma50'
    ]

    # Generate target (binary: will reach 2R within 10 days?)
    # Make it correlated with some features for realistic learning
    signal = (
        0.3 * X[:, 0] +  # RRS
        0.2 * X[:, 5] +  # Trend strength
        0.1 * X[:, 7] +  # RSI
        0.1 * np.random.randn(n_samples)
    )
    y = (signal > 0.2).astype(int)

    # Generate sequential data for LSTM
    X_lstm = np.random.randn(n_samples, sequence_length, 10)
    # Make LSTM sequences correlated with target
    for i in range(n_samples):
        if y[i] == 1:
            X_lstm[i] += 0.2  # Upward trend for successful trades

    logger.info(f"Generated data: X shape={X.shape}, y distribution={np.bincount(y)}")

    return X, y, X_lstm, feature_names


def example_1_train_individual_models():
    """Example 1: Train individual models separately"""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 1: Training Individual Models")
    logger.info("=" * 70)

    # Generate data
    X, y, X_lstm, feature_names = generate_synthetic_training_data(
        n_samples=1000,
        n_features=15
    )

    # 1. Train XGBoost
    logger.info("\n--- Training XGBoost ---")
    xgb_model = XGBoostTradeClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_gpu=False  # Set to True if GPU available
    )
    xgb_metrics = xgb_model.train(X, y, feature_names=feature_names)

    logger.info(f"XGBoost Results:")
    logger.info(f"  Accuracy: {xgb_metrics.accuracy:.4f}")
    logger.info(f"  AUC: {xgb_metrics.auc:.4f}")
    logger.info(f"  Precision: {xgb_metrics.precision:.4f}")
    logger.info(f"  Recall: {xgb_metrics.recall:.4f}")

    # Top features
    top_features = xgb_model.get_top_features(5)
    logger.info(f"  Top 5 Features:")
    for feat, importance in top_features:
        logger.info(f"    {feat}: {importance:.4f}")

    # 2. Train Random Forest
    logger.info("\n--- Training Random Forest ---")
    rf_model = RandomForestTradeClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5
    )
    rf_metrics = rf_model.train(X, y, feature_names=feature_names)

    logger.info(f"Random Forest Results:")
    logger.info(f"  Accuracy: {rf_metrics.accuracy:.4f}")
    logger.info(f"  AUC: {rf_metrics.auc:.4f}")
    logger.info(f"  OOB Score: {rf_metrics.oob_score:.4f}")

    # 3. Train LSTM
    logger.info("\n--- Training LSTM ---")
    lstm_model = LSTMTradePredictor(
        sequence_length=20,
        n_features=10,
        lstm_units=[64, 32],
        dropout=0.2
    )
    # Convert binary to 3-class for LSTM (0=up, 1=neutral, 2=down)
    y_lstm = y.copy()  # For demo, use same labels
    lstm_metrics = lstm_model.train(X_lstm, y_lstm, epochs=30, verbose=0)

    logger.info(f"LSTM Results:")
    logger.info(f"  Accuracy: {lstm_metrics.accuracy:.4f}")
    logger.info(f"  Precision: {lstm_metrics.precision:.4f}")

    return xgb_model, rf_model, lstm_model


def example_2_train_stacked_ensemble():
    """Example 2: Train stacked ensemble with meta-learner"""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 2: Training Stacked Ensemble with Meta-Learner")
    logger.info("=" * 70)

    # Generate data
    X, y, X_lstm, feature_names = generate_synthetic_training_data(
        n_samples=1000,
        n_features=15
    )

    # Create ensemble
    ensemble = StackedEnsemble(
        use_xgboost=True,
        use_random_forest=True,
        use_lstm=False,  # Disable LSTM for faster demo
        meta_learner_C=1.0,
        random_state=42
    )

    # Train ensemble
    logger.info("\nTraining ensemble (this may take a few minutes)...")
    metrics = ensemble.train(
        X=X,
        y=y,
        X_lstm=None,  # Not using LSTM in this example
        feature_names=feature_names,
        validation_split=0.2,
        cv_folds=5
    )

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("ENSEMBLE RESULTS")
    logger.info("=" * 70)

    logger.info(f"\nEnsemble Metrics:")
    logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
    logger.info(f"  AUC: {metrics.auc:.4f}")
    logger.info(f"  Precision: {metrics.precision:.4f}")
    logger.info(f"  Recall: {metrics.recall:.4f}")
    logger.info(f"  F1 Score: {metrics.f1_score:.4f}")
    logger.info(f"  Log Loss: {metrics.log_loss:.4f}")

    logger.info(f"\nModel Weights (learned by meta-learner):")
    for model_name, weight in metrics.model_weights.items():
        logger.info(f"  {model_name}: {weight:.4f}")

    logger.info(f"\nBase Model Performance:")
    for model_name, perf in metrics.base_model_performance.items():
        logger.info(f"  {model_name}:")
        logger.info(f"    Accuracy: {perf['accuracy']:.4f}")
        logger.info(f"    AUC: {perf['auc']:.4f}")

    return ensemble


def example_3_make_predictions(ensemble: StackedEnsemble):
    """Example 3: Making predictions with trained ensemble"""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 3: Making Predictions")
    logger.info("=" * 70)

    # Generate new test data
    X_test = np.random.randn(10, 15)

    # Predict probabilities
    probas = ensemble.predict_success_probability(X_test)

    logger.info("\nPredictions for 10 new samples:")
    logger.info(f"{'Sample':<10} {'Success Prob':<15} {'Prediction':<15}")
    logger.info("-" * 40)

    for i, prob in enumerate(probas):
        prediction = "SUCCESS" if prob > 0.5 else "FAILURE"
        logger.info(f"Sample {i+1:<3}  {prob:>6.2%}          {prediction:<15}")

    # Get individual model predictions
    base_predictions = ensemble.get_base_model_predictions(X_test)

    logger.info("\nBase Model Predictions (first 3 samples):")
    for model_name, preds in base_predictions.items():
        logger.info(f"  {model_name}: {preds[:3]}")

    return probas


def example_4_feature_importance(ensemble: StackedEnsemble):
    """Example 4: Analyzing feature importance"""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 4: Feature Importance Analysis")
    logger.info("=" * 70)

    importance = ensemble.get_feature_importance()

    for model_name, features in importance.items():
        logger.info(f"\n{model_name.upper()} - Top 10 Features:")
        for i, (feat, imp) in enumerate(features, 1):
            logger.info(f"  {i:2d}. {feat:<25} {imp:.6f}")


def example_5_save_and_load():
    """Example 5: Saving and loading models"""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 5: Saving and Loading Models")
    logger.info("=" * 70)

    # Generate data and train
    X, y, X_lstm, feature_names = generate_synthetic_training_data(n_samples=500)

    logger.info("\nTraining ensemble...")
    ensemble = StackedEnsemble(
        use_xgboost=True,
        use_random_forest=True,
        use_lstm=False
    )
    ensemble.train(X, y, feature_names=feature_names)

    # Save
    save_path = Path(__file__).parent.parent / "ml" / "data" / "ensemble_demo"
    logger.info(f"\nSaving ensemble to {save_path}...")
    ensemble.save(str(save_path))

    # Load
    logger.info(f"\nLoading ensemble from {save_path}...")
    new_ensemble = StackedEnsemble()
    new_ensemble.load(str(save_path))

    # Verify
    X_test = np.random.randn(5, 15)
    original_preds = ensemble.predict_success_probability(X_test)
    loaded_preds = new_ensemble.predict_success_probability(X_test)

    logger.info(f"\nVerification (predictions should match):")
    logger.info(f"  Original: {original_preds}")
    logger.info(f"  Loaded:   {loaded_preds}")
    logger.info(f"  Match: {np.allclose(original_preds, loaded_preds)}")

    return new_ensemble


def example_6_comprehensive_metrics():
    """Example 6: Getting comprehensive metrics summary"""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 6: Comprehensive Metrics Summary")
    logger.info("=" * 70)

    # Train ensemble
    X, y, _, feature_names = generate_synthetic_training_data(n_samples=500)
    ensemble = StackedEnsemble(use_xgboost=True, use_random_forest=True)
    ensemble.train(X, y, feature_names=feature_names)

    # Get metrics summary
    summary = ensemble.get_metrics_summary()

    logger.info("\nEnsemble Summary:")
    logger.info(f"  Trained: {summary['is_trained']}")
    logger.info(f"  Base Models: {summary['n_base_models']}")
    logger.info(f"  Features: {summary['n_features']}")

    logger.info("\nModel Weights:")
    for model, weight in summary['model_weights'].items():
        logger.info(f"  {model}: {weight:.4f}")

    if 'xgboost' in summary:
        xgb_summary = summary['xgboost']
        logger.info("\nXGBoost Summary:")
        logger.info(f"  Accuracy: {xgb_summary['validation']['accuracy']:.4f}")
        logger.info(f"  AUC: {xgb_summary['validation']['auc']:.4f}")


def main():
    """Run all examples"""
    logger.info("=" * 70)
    logger.info("ML ENSEMBLE SYSTEM - COMPREHENSIVE EXAMPLES")
    logger.info("=" * 70)

    try:
        # Example 1: Train individual models
        example_1_train_individual_models()

        # Example 2: Train stacked ensemble
        ensemble = example_2_train_stacked_ensemble()

        # Example 3: Make predictions
        example_3_make_predictions(ensemble)

        # Example 4: Feature importance
        example_4_feature_importance(ensemble)

        # Example 5: Save and load
        example_5_save_and_load()

        # Example 6: Comprehensive metrics
        example_6_comprehensive_metrics()

        logger.info("\n" + "=" * 70)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
