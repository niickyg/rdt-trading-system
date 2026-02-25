"""
A/B Testing Framework for ML Model Comparison.

This module provides a complete framework for running A/B tests to compare
different ML model versions in the RDT Trading System.

Features:
- Experiment configuration and management
- Fixed traffic split and multi-armed bandit (Thompson sampling)
- Statistical analysis (chi-square, t-test, z-test)
- Database persistence for experiments and results
- Prometheus metrics integration

Usage:
    from ml.ab_testing import (
        Experiment,
        ExperimentManager,
        get_experiment_manager,
        analyze_experiment,
        get_winner,
    )

    # Create an experiment
    manager = get_experiment_manager()
    experiment = manager.create_experiment(
        name="xgboost_v2_vs_v1",
        model_a_id="xgboost_v1",
        model_b_id="xgboost_v2",
        traffic_split=0.5,
        use_thompson_sampling=True,
    )

    # Start the experiment
    manager.start_experiment("xgboost_v2_vs_v1")

    # Get model for a request
    model_id, variant = experiment.get_model_for_request(request_id="req_123")

    # Record prediction
    request_id = experiment.record_prediction(
        symbol="AAPL",
        direction="long",
        variant=variant,
        model_id=model_id,
        prediction_probability=0.75,
        prediction_class=1,
        confidence=0.85,
    )

    # Record outcome
    experiment.record_outcome(
        request_id=request_id,
        outcome=OutcomeType.WIN,
        pnl=150.0,
        pnl_percent=2.5,
    )

    # Analyze results
    result = analyze_experiment(experiment)
    print(f"Winner: {result.winner}, Confidence: {result.confidence}")
"""

from ml.ab_testing.experiment import (
    Experiment,
    PredictionRecord,
    OutcomeRecord,
    VariantStats,
)

from ml.ab_testing.experiment_manager import (
    ExperimentManager,
    get_experiment_manager,
    reset_experiment_manager,
)

from ml.ab_testing.analysis import (
    ABTestAnalyzer,
    AnalysisResult,
    TestResult,
    Winner,
    analyze_experiment,
    get_winner,
    calculate_confidence,
)

from ml.ab_testing.models import (
    ABExperiment,
    ABPrediction,
    ABOutcome,
    ABExperimentStats,
    ExperimentStatus,
    ModelVariant,
    OutcomeType,
)

__all__ = [
    # Experiment classes
    "Experiment",
    "PredictionRecord",
    "OutcomeRecord",
    "VariantStats",
    # Manager
    "ExperimentManager",
    "get_experiment_manager",
    "reset_experiment_manager",
    # Analysis
    "ABTestAnalyzer",
    "AnalysisResult",
    "TestResult",
    "Winner",
    "analyze_experiment",
    "get_winner",
    "calculate_confidence",
    # Database models
    "ABExperiment",
    "ABPrediction",
    "ABOutcome",
    "ABExperimentStats",
    # Enums
    "ExperimentStatus",
    "ModelVariant",
    "OutcomeType",
]
