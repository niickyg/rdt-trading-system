"""
Test and demonstration script for the Feature Engineering Pipeline

This script demonstrates how to use the comprehensive feature engineering pipeline
to calculate and analyze 60+ features for trading signals.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger
from ml.feature_engineering import FeatureEngineer, calculate_features_for_symbol


async def demo_basic_feature_calculation():
    """Demonstrate basic feature calculation for a single symbol."""
    logger.info("=== Demo 1: Basic Feature Calculation ===")

    # Initialize feature engineer
    engineer = FeatureEngineer(cache_ttl_seconds=300)

    # Calculate features for AAPL
    logger.info("Calculating features for AAPL...")
    features_df = await engineer.calculate_features("AAPL")

    if features_df is not None:
        logger.info(f"Successfully calculated {len(features_df.columns)} features")

        # Display some key features
        logger.info("\n--- Technical Features ---")
        tech_cols = ['rrs', 'rrs_3bar', 'rrs_5bar', 'rsi_14', 'macd', 'atr_percent']
        for col in tech_cols:
            if col in features_df.columns:
                logger.info(f"{col}: {features_df[col].iloc[0]:.4f}")

        logger.info("\n--- Microstructure Features ---")
        micro_cols = ['vwap_distance_percent', 'volume_ratio', 'price_momentum_5']
        for col in micro_cols:
            if col in features_df.columns:
                logger.info(f"{col}: {features_df[col].iloc[0]:.4f}")

        logger.info("\n--- Regime Features ---")
        regime_cols = ['vix', 'spy_trend', 'correlation_with_spy']
        for col in regime_cols:
            if col in features_df.columns:
                logger.info(f"{col}: {features_df[col].iloc[0]:.4f}")

        logger.info("\n--- Derived Features ---")
        derived_cols = ['trend_strength_composite', 'breakout_probability', 'daily_alignment_score']
        for col in derived_cols:
            if col in features_df.columns:
                logger.info(f"{col}: {features_df[col].iloc[0]:.4f}")

        return features_df
    else:
        logger.error("Failed to calculate features")
        return None


async def demo_batch_feature_calculation():
    """Demonstrate batch feature calculation for multiple symbols."""
    logger.info("\n=== Demo 2: Batch Feature Calculation ===")

    engineer = FeatureEngineer()

    # Calculate features for multiple symbols in parallel
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    logger.info(f"Calculating features for {len(symbols)} symbols in parallel...")

    features_dict = await engineer.calculate_batch_features(symbols)

    logger.info(f"Successfully calculated features for {len(features_dict)} symbols")

    # Compare RRS across symbols
    logger.info("\n--- RRS Comparison ---")
    for symbol, df in features_dict.items():
        if df is not None and 'rrs' in df.columns:
            rrs = df['rrs'].iloc[0]
            logger.info(f"{symbol}: RRS = {rrs:.4f}")


async def demo_feature_categories():
    """Demonstrate accessing features by category."""
    logger.info("\n=== Demo 3: Feature Categories ===")

    engineer = FeatureEngineer()

    # Get feature names by category
    categories = ['technical', 'microstructure', 'regime', 'temporal', 'derived']

    for category in categories:
        features = engineer.get_feature_names(category)
        logger.info(f"\n{category.upper()} features ({len(features)}):")
        logger.info(", ".join(features))


async def demo_feature_caching():
    """Demonstrate feature caching for performance."""
    logger.info("\n=== Demo 4: Feature Caching ===")

    engineer = FeatureEngineer(cache_ttl_seconds=60)

    import time

    # First calculation (no cache)
    logger.info("First calculation (no cache)...")
    start = time.time()
    features1 = await engineer.calculate_features("AAPL", use_cache=False)
    time1 = time.time() - start
    logger.info(f"Time: {time1:.2f}s")

    # Second calculation (with cache)
    logger.info("\nSecond calculation (with cache)...")
    start = time.time()
    features2 = await engineer.calculate_features("AAPL", use_cache=True)
    time2 = time.time() - start
    logger.info(f"Time: {time2:.2f}s")

    logger.info(f"\nSpeedup: {time1/time2:.1f}x faster with cache")


async def demo_convenience_function():
    """Demonstrate the convenience function."""
    logger.info("\n=== Demo 5: Convenience Function ===")

    # Use the convenience function for quick calculations
    features = await calculate_features_for_symbol("TSLA")

    if features is not None:
        logger.info("Features calculated using convenience function")
        logger.info(f"Shape: {features.shape}")


def demo_feature_analysis(features_df):
    """Analyze and display feature statistics."""
    logger.info("\n=== Feature Analysis ===")

    if features_df is None:
        logger.warning("No features to analyze")
        return

    # Remove non-numeric columns
    numeric_cols = features_df.select_dtypes(include=[float, int]).columns

    logger.info(f"\nTotal numeric features: {len(numeric_cols)}")

    # Feature statistics
    logger.info("\n--- Feature Statistics ---")
    stats = features_df[numeric_cols].describe().T
    logger.info(f"\n{stats[['mean', 'std', 'min', 'max']].head(20)}")

    # Identify strong signals
    logger.info("\n--- Strong Signal Detection ---")

    if 'rrs' in features_df.columns:
        rrs = features_df['rrs'].iloc[0]
        if abs(rrs) > 2:
            logger.info(f"STRONG RRS SIGNAL: {rrs:.4f}")

    if 'rsi_14' in features_df.columns:
        rsi = features_df['rsi_14'].iloc[0]
        if rsi > 70:
            logger.info(f"OVERBOUGHT RSI: {rsi:.2f}")
        elif rsi < 30:
            logger.info(f"OVERSOLD RSI: {rsi:.2f}")

    if 'volume_ratio' in features_df.columns:
        vol_ratio = features_df['volume_ratio'].iloc[0]
        if vol_ratio > 2.0:
            logger.info(f"HIGH VOLUME: {vol_ratio:.2f}x average")

    if 'breakout_probability' in features_df.columns:
        breakout = features_df['breakout_probability'].iloc[0]
        if breakout > 0.7:
            logger.info(f"HIGH BREAKOUT PROBABILITY: {breakout:.2f}")


async def demo_database_storage():
    """Demonstrate database storage (optional)."""
    logger.info("\n=== Demo 6: Database Storage (Optional) ===")

    # Note: This requires a PostgreSQL/TimescaleDB connection
    db_url = "postgresql://user:password@localhost:5432/trading_db"

    logger.info("Database storage is OPTIONAL and requires configuration")
    logger.info(f"Example connection string: {db_url}")
    logger.info("\nTo enable database storage:")
    logger.info("1. Set up PostgreSQL/TimescaleDB")
    logger.info("2. Create a database")
    logger.info("3. Initialize FeatureEngineer with db_url and enable_db_storage=True")
    logger.info("\nExample:")
    logger.info("  engineer = FeatureEngineer(")
    logger.info("      db_url='postgresql://user:pass@localhost:5432/db',")
    logger.info("      enable_db_storage=True")
    logger.info("  )")


def demo_cache_persistence():
    """Demonstrate saving and loading cache."""
    logger.info("\n=== Demo 7: Cache Persistence ===")

    # Example of cache save/load
    cache_file = "/tmp/feature_cache.pkl"

    logger.info(f"Cache can be saved to disk: {cache_file}")
    logger.info("\nUsage:")
    logger.info("  # Save cache")
    logger.info("  engineer.save_cache('/tmp/feature_cache.pkl')")
    logger.info("\n  # Load cache (in new session)")
    logger.info("  engineer.load_cache('/tmp/feature_cache.pkl')")


async def main():
    """Run all demonstrations."""
    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline - Demonstration")
    logger.info("=" * 60)

    try:
        # Demo 1: Basic calculation
        features = await demo_basic_feature_calculation()

        # Demo 2: Batch calculation
        await demo_batch_feature_calculation()

        # Demo 3: Feature categories
        await demo_feature_categories()

        # Demo 4: Caching
        await demo_feature_caching()

        # Demo 5: Convenience function
        await demo_convenience_function()

        # Feature analysis
        if features is not None:
            demo_feature_analysis(features)

        # Demo 6: Database storage (informational)
        await demo_database_storage()

        # Demo 7: Cache persistence (informational)
        demo_cache_persistence()

        logger.info("\n" + "=" * 60)
        logger.info("All demonstrations completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error in demonstration: {e}", exc_info=True)


if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Run demonstrations
    asyncio.run(main())
