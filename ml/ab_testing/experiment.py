"""
A/B Testing Experiment Class.

Manages individual experiments for comparing ML model versions.
Supports both fixed traffic split and multi-armed bandit (Thompson sampling).
"""

import uuid
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from loguru import logger

from ml.ab_testing.models import (
    ExperimentStatus,
    ModelVariant,
    OutcomeType,
)


@dataclass
class PredictionRecord:
    """Record of a prediction made during an experiment."""
    request_id: str
    symbol: str
    direction: str
    variant: ModelVariant
    model_id: str
    prediction_probability: float
    prediction_class: int
    confidence: float
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    rrs_at_entry: Optional[float] = None
    features: Optional[Dict] = None
    prediction_time: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OutcomeRecord:
    """Record of an outcome for a prediction."""
    prediction_id: str
    outcome: OutcomeType
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    holding_period_hours: Optional[float] = None
    prediction_correct: Optional[bool] = None
    outcome_time: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VariantStats:
    """Statistics for a single variant."""
    variant: ModelVariant
    total_predictions: int = 0
    total_outcomes: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    total_pnl: float = 0.0
    correct_predictions: int = 0
    sum_confidence: float = 0.0
    sum_probability: float = 0.0

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.wins + self.losses + self.breakeven
        if total == 0:
            return 0.0
        return self.wins / total

    @property
    def avg_pnl(self) -> float:
        """Calculate average P&L."""
        if self.total_outcomes == 0:
            return 0.0
        return self.total_pnl / self.total_outcomes

    @property
    def accuracy(self) -> float:
        """Calculate prediction accuracy."""
        if self.total_outcomes == 0:
            return 0.0
        return self.correct_predictions / self.total_outcomes

    @property
    def avg_confidence(self) -> float:
        """Calculate average confidence."""
        if self.total_predictions == 0:
            return 0.0
        return self.sum_confidence / self.total_predictions

    @property
    def avg_probability(self) -> float:
        """Calculate average probability."""
        if self.total_predictions == 0:
            return 0.0
        return self.sum_probability / self.total_predictions

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "variant": self.variant.value,
            "total_predictions": self.total_predictions,
            "total_outcomes": self.total_outcomes,
            "wins": self.wins,
            "losses": self.losses,
            "breakeven": self.breakeven,
            "total_pnl": self.total_pnl,
            "correct_predictions": self.correct_predictions,
            "win_rate": round(self.win_rate, 4),
            "avg_pnl": round(self.avg_pnl, 2),
            "accuracy": round(self.accuracy, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_probability": round(self.avg_probability, 4),
        }


class Experiment:
    """
    A/B Test Experiment for comparing ML model versions.

    Supports:
    - Fixed traffic split (e.g., 50/50, 80/20)
    - Multi-armed bandit with Thompson sampling
    - Recording predictions and outcomes
    - Real-time statistics computation
    """

    def __init__(
        self,
        name: str,
        model_a_id: str,
        model_b_id: str,
        traffic_split: float = 0.5,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_thompson_sampling: bool = False,
        model_a_version: Optional[str] = None,
        model_b_version: Optional[str] = None,
        min_samples_per_variant: int = 100,
        confidence_threshold: float = 0.95,
        experiment_id: Optional[int] = None,
        description: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize an A/B experiment.

        Args:
            name: Experiment name
            model_a_id: Control model identifier
            model_b_id: Treatment model identifier
            traffic_split: Fraction of traffic to send to model B (0.0 to 1.0)
            start_date: When the experiment starts
            end_date: When the experiment ends
            use_thompson_sampling: Whether to use Thompson sampling instead of fixed split
            model_a_version: Version string for model A
            model_b_version: Version string for model B
            min_samples_per_variant: Minimum samples needed before analysis
            confidence_threshold: Statistical significance threshold
            experiment_id: Database ID if persisted
            description: Experiment description
            config: Additional configuration
        """
        self.name = name
        self.model_a_id = model_a_id
        self.model_b_id = model_b_id
        self.model_a_version = model_a_version
        self.model_b_version = model_b_version
        self.traffic_split = traffic_split
        self.start_date = start_date or datetime.utcnow()
        self.end_date = end_date
        self.use_thompson_sampling = use_thompson_sampling
        self.min_samples_per_variant = min_samples_per_variant
        self.confidence_threshold = confidence_threshold
        self.experiment_id = experiment_id
        self.description = description
        self.config = config or {}

        # Status
        self.status = ExperimentStatus.DRAFT

        # Thompson sampling parameters (Beta distribution)
        # Prior: Beta(1, 1) = Uniform
        self.alpha_a = 1.0  # Successes for A + 1
        self.beta_a = 1.0   # Failures for A + 1
        self.alpha_b = 1.0  # Successes for B + 1
        self.beta_b = 1.0   # Failures for B + 1

        # In-memory tracking (for fast access)
        self._predictions: Dict[str, PredictionRecord] = {}
        self._outcomes: Dict[str, OutcomeRecord] = {}
        self._stats_a = VariantStats(variant=ModelVariant.CONTROL)
        self._stats_b = VariantStats(variant=ModelVariant.TREATMENT)

        logger.info(
            f"Initialized experiment '{name}': {model_a_id} vs {model_b_id}, "
            f"split={traffic_split}, thompson_sampling={use_thompson_sampling}"
        )

    @property
    def is_active(self) -> bool:
        """Check if experiment is currently active."""
        if self.status != ExperimentStatus.ACTIVE:
            return False

        now = datetime.utcnow()
        if self.start_date and now < self.start_date:
            return False
        if self.end_date and now > self.end_date:
            return False

        return True

    def start(self) -> None:
        """Start the experiment."""
        if self.status not in (ExperimentStatus.DRAFT, ExperimentStatus.PAUSED):
            logger.warning(f"Cannot start experiment in status {self.status}")
            return

        self.status = ExperimentStatus.ACTIVE
        if not self.start_date:
            self.start_date = datetime.utcnow()

        logger.info(f"Started experiment '{self.name}'")

    def pause(self) -> None:
        """Pause the experiment."""
        if self.status != ExperimentStatus.ACTIVE:
            logger.warning(f"Cannot pause experiment in status {self.status}")
            return

        self.status = ExperimentStatus.PAUSED
        logger.info(f"Paused experiment '{self.name}'")

    def stop(self) -> None:
        """Stop/complete the experiment."""
        if self.status not in (ExperimentStatus.ACTIVE, ExperimentStatus.PAUSED):
            logger.warning(f"Cannot stop experiment in status {self.status}")
            return

        self.status = ExperimentStatus.COMPLETED
        self.end_date = datetime.utcnow()
        logger.info(f"Stopped experiment '{self.name}'")

    def archive(self) -> None:
        """Archive the experiment."""
        self.status = ExperimentStatus.ARCHIVED
        logger.info(f"Archived experiment '{self.name}'")

    def get_model_for_request(self, request_id: Optional[str] = None) -> Tuple[str, ModelVariant]:
        """
        Determine which model to use for a request.

        Args:
            request_id: Optional request ID for deterministic assignment

        Returns:
            Tuple of (model_id, variant)
        """
        if not self.is_active:
            # Default to control when not active
            return self.model_a_id, ModelVariant.CONTROL

        if self.use_thompson_sampling:
            variant = self._thompson_sample()
        else:
            variant = self._fixed_split_sample(request_id)

        model_id = self.model_a_id if variant == ModelVariant.CONTROL else self.model_b_id

        logger.debug(f"Experiment '{self.name}' assigned variant {variant.value} for request {request_id}")

        return model_id, variant

    def _fixed_split_sample(self, request_id: Optional[str] = None) -> ModelVariant:
        """
        Sample variant using fixed traffic split.

        Uses request_id for deterministic assignment if provided.
        """
        if request_id:
            # Deterministic based on request_id hash
            hash_val = hash(request_id) % 10000
            threshold = int(self.traffic_split * 10000)
            return ModelVariant.TREATMENT if hash_val < threshold else ModelVariant.CONTROL
        else:
            # Random sampling
            return ModelVariant.TREATMENT if random.random() < self.traffic_split else ModelVariant.CONTROL

    def _thompson_sample(self) -> ModelVariant:
        """
        Sample variant using Thompson sampling (multi-armed bandit).

        Draws from Beta distributions and selects the variant with the higher sample.
        """
        # Sample from posterior distributions
        sample_a = np.random.beta(self.alpha_a, self.beta_a)
        sample_b = np.random.beta(self.alpha_b, self.beta_b)

        # Select the variant with higher sampled value
        if sample_b > sample_a:
            return ModelVariant.TREATMENT
        else:
            return ModelVariant.CONTROL

    def record_prediction(
        self,
        symbol: str,
        direction: str,
        variant: ModelVariant,
        model_id: str,
        prediction_probability: float,
        prediction_class: int,
        confidence: float,
        request_id: Optional[str] = None,
        entry_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        target_price: Optional[float] = None,
        rrs_at_entry: Optional[float] = None,
        features: Optional[Dict] = None,
    ) -> str:
        """
        Record a prediction made during the experiment.

        Args:
            symbol: Trading symbol
            direction: Trade direction (long/short)
            variant: Which variant was used
            model_id: Model identifier
            prediction_probability: Predicted success probability
            prediction_class: Predicted class (0 or 1)
            confidence: Prediction confidence
            request_id: Optional request ID (generated if not provided)
            entry_price: Entry price at prediction time
            stop_price: Stop loss price
            target_price: Target price
            rrs_at_entry: RRS value at entry
            features: Feature values used for prediction

        Returns:
            Request ID for tracking
        """
        if not request_id:
            request_id = str(uuid.uuid4())

        record = PredictionRecord(
            request_id=request_id,
            symbol=symbol,
            direction=direction,
            variant=variant,
            model_id=model_id,
            prediction_probability=prediction_probability,
            prediction_class=prediction_class,
            confidence=confidence,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            rrs_at_entry=rrs_at_entry,
            features=features,
            prediction_time=datetime.utcnow(),
        )

        self._predictions[request_id] = record

        # Update stats
        stats = self._stats_a if variant == ModelVariant.CONTROL else self._stats_b
        stats.total_predictions += 1
        stats.sum_confidence += confidence
        stats.sum_probability += prediction_probability

        logger.debug(
            f"Recorded prediction for {symbol}: variant={variant.value}, "
            f"prob={prediction_probability:.4f}, class={prediction_class}"
        )

        return request_id

    def record_outcome(
        self,
        request_id: str,
        outcome: OutcomeType,
        pnl: Optional[float] = None,
        pnl_percent: Optional[float] = None,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
        holding_period_hours: Optional[float] = None,
    ) -> bool:
        """
        Record the outcome for a prediction.

        Args:
            request_id: The prediction's request ID
            outcome: Trade outcome (win/loss/breakeven)
            pnl: Profit/loss in dollars
            pnl_percent: Profit/loss as percentage
            exit_price: Exit price
            exit_reason: Reason for exit
            holding_period_hours: How long the trade was held

        Returns:
            True if outcome was recorded successfully
        """
        if request_id not in self._predictions:
            logger.warning(f"No prediction found for request_id {request_id}")
            return False

        prediction = self._predictions[request_id]

        # Determine if prediction was correct
        prediction_correct = None
        if outcome != OutcomeType.PENDING:
            if outcome == OutcomeType.WIN:
                prediction_correct = prediction.prediction_class == 1
            elif outcome == OutcomeType.LOSS:
                prediction_correct = prediction.prediction_class == 0
            elif outcome == OutcomeType.BREAKEVEN:
                prediction_correct = True  # Breakeven counts as correct

        outcome_record = OutcomeRecord(
            prediction_id=request_id,
            outcome=outcome,
            pnl=pnl,
            pnl_percent=pnl_percent,
            exit_price=exit_price,
            exit_reason=exit_reason,
            holding_period_hours=holding_period_hours,
            prediction_correct=prediction_correct,
            outcome_time=datetime.utcnow(),
        )

        self._outcomes[request_id] = outcome_record

        # Update stats
        stats = self._stats_a if prediction.variant == ModelVariant.CONTROL else self._stats_b
        stats.total_outcomes += 1

        if outcome == OutcomeType.WIN:
            stats.wins += 1
        elif outcome == OutcomeType.LOSS:
            stats.losses += 1
        elif outcome == OutcomeType.BREAKEVEN:
            stats.breakeven += 1

        if pnl is not None:
            stats.total_pnl += pnl

        if prediction_correct:
            stats.correct_predictions += 1

        # Update Thompson sampling parameters
        if self.use_thompson_sampling and outcome != OutcomeType.PENDING:
            self._update_thompson_params(prediction.variant, outcome)

        logger.debug(
            f"Recorded outcome for {prediction.symbol}: variant={prediction.variant.value}, "
            f"outcome={outcome.value}, pnl={pnl}"
        )

        return True

    def _update_thompson_params(self, variant: ModelVariant, outcome: OutcomeType) -> None:
        """
        Update Thompson sampling parameters based on outcome.

        Args:
            variant: Which variant was used
            outcome: The outcome
        """
        # Win = success, Loss = failure, Breakeven = partial success (0.5)
        if variant == ModelVariant.CONTROL:
            if outcome == OutcomeType.WIN:
                self.alpha_a += 1.0
            elif outcome == OutcomeType.LOSS:
                self.beta_a += 1.0
            elif outcome == OutcomeType.BREAKEVEN:
                self.alpha_a += 0.5
                self.beta_a += 0.5
        else:
            if outcome == OutcomeType.WIN:
                self.alpha_b += 1.0
            elif outcome == OutcomeType.LOSS:
                self.beta_b += 1.0
            elif outcome == OutcomeType.BREAKEVEN:
                self.alpha_b += 0.5
                self.beta_b += 0.5

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current experiment statistics.

        Returns:
            Dictionary with experiment stats
        """
        return {
            "experiment_name": self.name,
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "is_active": self.is_active,
            "model_a": {
                "id": self.model_a_id,
                "version": self.model_a_version,
            },
            "model_b": {
                "id": self.model_b_id,
                "version": self.model_b_version,
            },
            "traffic_split": self.traffic_split,
            "use_thompson_sampling": self.use_thompson_sampling,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "control": self._stats_a.to_dict(),
            "treatment": self._stats_b.to_dict(),
            "thompson_params": {
                "alpha_a": self.alpha_a,
                "beta_a": self.beta_a,
                "alpha_b": self.alpha_b,
                "beta_b": self.beta_b,
            } if self.use_thompson_sampling else None,
        }

    def get_variant_stats(self, variant: ModelVariant) -> VariantStats:
        """Get stats for a specific variant."""
        return self._stats_a if variant == ModelVariant.CONTROL else self._stats_b

    def has_sufficient_data(self) -> bool:
        """Check if there's enough data for statistical analysis."""
        return (
            self._stats_a.total_outcomes >= self.min_samples_per_variant and
            self._stats_b.total_outcomes >= self.min_samples_per_variant
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary for serialization."""
        return {
            "name": self.name,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "model_a_id": self.model_a_id,
            "model_a_version": self.model_a_version,
            "model_b_id": self.model_b_id,
            "model_b_version": self.model_b_version,
            "traffic_split": self.traffic_split,
            "use_thompson_sampling": self.use_thompson_sampling,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "status": self.status.value,
            "min_samples_per_variant": self.min_samples_per_variant,
            "confidence_threshold": self.confidence_threshold,
            "alpha_a": self.alpha_a,
            "beta_a": self.beta_a,
            "alpha_b": self.alpha_b,
            "beta_b": self.beta_b,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Create experiment from dictionary."""
        experiment = cls(
            name=data["name"],
            model_a_id=data["model_a_id"],
            model_b_id=data["model_b_id"],
            traffic_split=data.get("traffic_split", 0.5),
            start_date=datetime.fromisoformat(data["start_date"]) if data.get("start_date") else None,
            end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            use_thompson_sampling=data.get("use_thompson_sampling", False),
            model_a_version=data.get("model_a_version"),
            model_b_version=data.get("model_b_version"),
            min_samples_per_variant=data.get("min_samples_per_variant", 100),
            confidence_threshold=data.get("confidence_threshold", 0.95),
            experiment_id=data.get("experiment_id"),
            description=data.get("description"),
            config=data.get("config"),
        )

        # Restore status
        if "status" in data:
            experiment.status = ExperimentStatus(data["status"])

        # Restore Thompson parameters
        if data.get("alpha_a") is not None:
            experiment.alpha_a = data["alpha_a"]
            experiment.beta_a = data["beta_a"]
            experiment.alpha_b = data["alpha_b"]
            experiment.beta_b = data["beta_b"]

        return experiment
