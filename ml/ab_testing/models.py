"""
Database models for A/B Testing Framework.

Models for storing experiment configurations, predictions, and outcomes.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import (
    String, Integer, Float, DateTime, Boolean, Text,
    Enum as SQLEnum, Index, CheckConstraint, Numeric, BigInteger,
    ForeignKey, JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Import Base from existing models
from data.database.models import Base


class ExperimentStatus(str, Enum):
    """Status of an A/B experiment."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ModelVariant(str, Enum):
    """Model variant in experiment."""
    CONTROL = "control"  # Model A (baseline)
    TREATMENT = "treatment"  # Model B (challenger)


class OutcomeType(str, Enum):
    """Type of trade outcome."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    PENDING = "pending"


class ABExperiment(Base):
    """
    A/B Experiment configuration model.

    Stores the configuration for comparing two model versions.
    """
    __tablename__ = "ab_experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Model configuration
    model_a_id: Mapped[str] = mapped_column(String(128), nullable=False)  # Control model identifier
    model_a_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    model_b_id: Mapped[str] = mapped_column(String(128), nullable=False)  # Treatment model identifier
    model_b_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Traffic split configuration
    traffic_split: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False, default=0.5)  # % to model B

    # Multi-armed bandit configuration
    use_thompson_sampling: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    alpha_a: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False, default=1.0)  # Beta dist param for A
    beta_a: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False, default=1.0)
    alpha_b: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False, default=1.0)  # Beta dist param for B
    beta_b: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False, default=1.0)

    # Time configuration
    start_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Status
    status: Mapped[str] = mapped_column(
        SQLEnum(ExperimentStatus, native_enum=False),
        default=ExperimentStatus.DRAFT,
        nullable=False
    )

    # Metrics thresholds for auto-stop
    min_samples_per_variant: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    confidence_threshold: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False, default=0.95)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, onupdate=datetime.utcnow)
    created_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Configuration as JSON for flexibility
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_ab_experiments_name", "name"),
        Index("ix_ab_experiments_status", "status"),
        Index("ix_ab_experiments_start_date", "start_date"),
        Index("ix_ab_experiments_status_start_date", "status", "start_date"),
        CheckConstraint("traffic_split >= 0 AND traffic_split <= 1", name="ck_ab_experiments_traffic_split_range"),
        CheckConstraint("min_samples_per_variant > 0", name="ck_ab_experiments_min_samples_positive"),
        CheckConstraint("confidence_threshold > 0 AND confidence_threshold < 1", name="ck_ab_experiments_confidence_range"),
    )


class ABPrediction(Base):
    """
    Individual prediction record for A/B testing.

    Records each prediction made during an experiment, including which model
    was used and the prediction details.
    """
    __tablename__ = "ab_predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, ForeignKey("ab_experiments.id"), nullable=False)

    # Request identification
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    # Symbol and signal info
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # long/short

    # Which model was used
    variant: Mapped[str] = mapped_column(
        SQLEnum(ModelVariant, native_enum=False),
        nullable=False
    )
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)

    # Prediction details
    prediction_probability: Mapped[float] = mapped_column(Numeric(8, 6), nullable=False)
    prediction_class: Mapped[int] = mapped_column(Integer, nullable=False)  # 0 or 1
    confidence: Mapped[float] = mapped_column(Numeric(8, 6), nullable=False)

    # Signal details at prediction time
    entry_price: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    stop_price: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    target_price: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    rrs_at_entry: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)

    # Feature values at prediction time (for debugging/analysis)
    features: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Timestamps
    prediction_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_ab_predictions_experiment_id", "experiment_id"),
        Index("ix_ab_predictions_request_id", "request_id"),
        Index("ix_ab_predictions_variant", "variant"),
        Index("ix_ab_predictions_symbol", "symbol"),
        Index("ix_ab_predictions_prediction_time", "prediction_time"),
        Index("ix_ab_predictions_experiment_variant", "experiment_id", "variant"),
        Index("ix_ab_predictions_experiment_time", "experiment_id", "prediction_time"),
    )


class ABOutcome(Base):
    """
    Outcome record for A/B test predictions.

    Records the actual outcome of trades that were predicted during an experiment.
    """
    __tablename__ = "ab_outcomes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("ab_predictions.id"), nullable=False)
    experiment_id: Mapped[int] = mapped_column(Integer, ForeignKey("ab_experiments.id"), nullable=False)

    # Outcome type
    outcome: Mapped[str] = mapped_column(
        SQLEnum(OutcomeType, native_enum=False),
        nullable=False
    )

    # P&L details
    pnl: Mapped[Optional[float]] = mapped_column(Numeric(12, 2), nullable=True)
    pnl_percent: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)

    # Exit details
    exit_price: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    exit_reason: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # Holding period
    holding_period_hours: Mapped[Optional[float]] = mapped_column(Numeric(10, 2), nullable=True)

    # Whether prediction was correct (outcome matches prediction)
    prediction_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Timestamps
    outcome_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_ab_outcomes_prediction_id", "prediction_id"),
        Index("ix_ab_outcomes_experiment_id", "experiment_id"),
        Index("ix_ab_outcomes_outcome", "outcome"),
        Index("ix_ab_outcomes_outcome_time", "outcome_time"),
        Index("ix_ab_outcomes_experiment_outcome", "experiment_id", "outcome"),
    )


class ABExperimentStats(Base):
    """
    Aggregated statistics for A/B experiments.

    Stores computed statistics for each variant to avoid recomputation.
    Updated periodically or on demand.
    """
    __tablename__ = "ab_experiment_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, ForeignKey("ab_experiments.id"), nullable=False)
    variant: Mapped[str] = mapped_column(
        SQLEnum(ModelVariant, native_enum=False),
        nullable=False
    )

    # Sample counts
    total_predictions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_outcomes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Conversion/success metrics
    wins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    losses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    breakeven: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Financial metrics
    total_pnl: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False, default=0.0)
    avg_pnl: Mapped[Optional[float]] = mapped_column(Numeric(12, 2), nullable=True)
    avg_pnl_percent: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)

    # Prediction accuracy
    correct_predictions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    accuracy: Mapped[Optional[float]] = mapped_column(Numeric(5, 4), nullable=True)

    # Confidence metrics
    avg_confidence: Mapped[Optional[float]] = mapped_column(Numeric(8, 6), nullable=True)
    avg_probability: Mapped[Optional[float]] = mapped_column(Numeric(8, 6), nullable=True)

    # Win rate
    win_rate: Mapped[Optional[float]] = mapped_column(Numeric(5, 4), nullable=True)

    # Timestamps
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_ab_experiment_stats_experiment_id", "experiment_id"),
        Index("ix_ab_experiment_stats_variant", "variant"),
        Index("ix_ab_experiment_stats_experiment_variant", "experiment_id", "variant", unique=True),
    )
