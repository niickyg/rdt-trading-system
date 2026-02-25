"""
A/B Testing Experiment Manager.

Manages the lifecycle of A/B experiments including creation, persistence,
and retrieval.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager

from sqlalchemy import select, update, and_, or_
from sqlalchemy.orm import Session
from loguru import logger

from ml.ab_testing.experiment import Experiment
from ml.ab_testing.models import (
    ABExperiment,
    ABPrediction,
    ABOutcome,
    ABExperimentStats,
    ExperimentStatus,
    ModelVariant,
    OutcomeType,
)

# Try to import database manager
try:
    from data.database.connection import get_db_manager, DatabaseManager
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logger.warning("Database connection not available for A/B testing")


class ExperimentManager:
    """
    Manager for A/B testing experiments.

    Handles:
    - Creating, starting, stopping, and archiving experiments
    - Persisting experiments to database
    - Loading experiments from database
    - Managing active experiments
    """

    def __init__(self, db_manager: Optional["DatabaseManager"] = None):
        """
        Initialize the experiment manager.

        Args:
            db_manager: Optional database manager. Uses global if not provided.
        """
        self._experiments: Dict[str, Experiment] = {}
        self._active_experiment: Optional[Experiment] = None
        self._db_manager = db_manager

        if DATABASE_AVAILABLE and not self._db_manager:
            try:
                self._db_manager = get_db_manager()
            except Exception as e:
                logger.warning(f"Could not get database manager: {e}")

        logger.info("ExperimentManager initialized")

    @contextmanager
    def _get_session(self):
        """Get a database session."""
        if not self._db_manager:
            raise RuntimeError("Database manager not available")

        with self._db_manager.get_session() as session:
            yield session

    def create_experiment(
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
        description: Optional[str] = None,
        config: Optional[Dict] = None,
        created_by: Optional[str] = None,
        persist: bool = True,
    ) -> Experiment:
        """
        Create a new A/B experiment.

        Args:
            name: Unique experiment name
            model_a_id: Control model identifier
            model_b_id: Treatment model identifier
            traffic_split: Fraction of traffic to model B
            start_date: When to start the experiment
            end_date: When to end the experiment
            use_thompson_sampling: Use multi-armed bandit
            model_a_version: Version string for model A
            model_b_version: Version string for model B
            min_samples_per_variant: Minimum samples for analysis
            confidence_threshold: Statistical significance threshold
            description: Experiment description
            config: Additional configuration
            created_by: User who created the experiment
            persist: Whether to save to database

        Returns:
            Created Experiment instance
        """
        # Check for existing experiment with same name
        if name in self._experiments:
            raise ValueError(f"Experiment with name '{name}' already exists")

        # Create experiment instance
        experiment = Experiment(
            name=name,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            traffic_split=traffic_split,
            start_date=start_date,
            end_date=end_date,
            use_thompson_sampling=use_thompson_sampling,
            model_a_version=model_a_version,
            model_b_version=model_b_version,
            min_samples_per_variant=min_samples_per_variant,
            confidence_threshold=confidence_threshold,
            description=description,
            config=config,
        )

        # Persist to database if available
        if persist and self._db_manager:
            try:
                db_experiment = self._persist_experiment(experiment, created_by)
                experiment.experiment_id = db_experiment.id
            except Exception as e:
                logger.error(f"Failed to persist experiment: {e}")
                raise

        # Cache in memory
        self._experiments[name] = experiment

        logger.info(f"Created experiment '{name}' (id={experiment.experiment_id})")

        return experiment

    def _persist_experiment(self, experiment: Experiment, created_by: Optional[str] = None) -> ABExperiment:
        """Persist experiment to database."""
        with self._get_session() as session:
            db_experiment = ABExperiment(
                name=experiment.name,
                description=experiment.description,
                model_a_id=experiment.model_a_id,
                model_a_version=experiment.model_a_version,
                model_b_id=experiment.model_b_id,
                model_b_version=experiment.model_b_version,
                traffic_split=experiment.traffic_split,
                use_thompson_sampling=experiment.use_thompson_sampling,
                alpha_a=experiment.alpha_a,
                beta_a=experiment.beta_a,
                alpha_b=experiment.alpha_b,
                beta_b=experiment.beta_b,
                start_date=experiment.start_date,
                end_date=experiment.end_date,
                status=experiment.status,
                min_samples_per_variant=experiment.min_samples_per_variant,
                confidence_threshold=experiment.confidence_threshold,
                config=experiment.config,
                created_by=created_by,
            )
            session.add(db_experiment)
            session.flush()  # Get the ID

            return db_experiment

    def get_experiment(self, name: str) -> Optional[Experiment]:
        """
        Get an experiment by name.

        Args:
            name: Experiment name

        Returns:
            Experiment if found, None otherwise
        """
        # Check cache first
        if name in self._experiments:
            return self._experiments[name]

        # Try to load from database
        if self._db_manager:
            try:
                experiment = self._load_experiment_by_name(name)
                if experiment:
                    self._experiments[name] = experiment
                    return experiment
            except Exception as e:
                logger.error(f"Failed to load experiment '{name}': {e}")

        return None

    def get_experiment_by_id(self, experiment_id: int) -> Optional[Experiment]:
        """
        Get an experiment by database ID.

        Args:
            experiment_id: Database ID

        Returns:
            Experiment if found, None otherwise
        """
        # Check cache
        for exp in self._experiments.values():
            if exp.experiment_id == experiment_id:
                return exp

        # Load from database
        if self._db_manager:
            try:
                experiment = self._load_experiment_by_id(experiment_id)
                if experiment:
                    self._experiments[experiment.name] = experiment
                    return experiment
            except Exception as e:
                logger.error(f"Failed to load experiment {experiment_id}: {e}")

        return None

    def _load_experiment_by_name(self, name: str) -> Optional[Experiment]:
        """Load experiment from database by name."""
        with self._get_session() as session:
            stmt = select(ABExperiment).where(ABExperiment.name == name)
            result = session.execute(stmt).scalar_one_or_none()

            if result:
                return self._db_to_experiment(result)

        return None

    def _load_experiment_by_id(self, experiment_id: int) -> Optional[Experiment]:
        """Load experiment from database by ID."""
        with self._get_session() as session:
            stmt = select(ABExperiment).where(ABExperiment.id == experiment_id)
            result = session.execute(stmt).scalar_one_or_none()

            if result:
                return self._db_to_experiment(result)

        return None

    def _db_to_experiment(self, db_exp: ABExperiment) -> Experiment:
        """Convert database model to Experiment instance."""
        experiment = Experiment(
            name=db_exp.name,
            model_a_id=db_exp.model_a_id,
            model_b_id=db_exp.model_b_id,
            traffic_split=float(db_exp.traffic_split),
            start_date=db_exp.start_date,
            end_date=db_exp.end_date,
            use_thompson_sampling=db_exp.use_thompson_sampling,
            model_a_version=db_exp.model_a_version,
            model_b_version=db_exp.model_b_version,
            min_samples_per_variant=db_exp.min_samples_per_variant,
            confidence_threshold=float(db_exp.confidence_threshold),
            experiment_id=db_exp.id,
            description=db_exp.description,
            config=db_exp.config,
        )

        experiment.status = ExperimentStatus(db_exp.status)
        experiment.alpha_a = float(db_exp.alpha_a)
        experiment.beta_a = float(db_exp.beta_a)
        experiment.alpha_b = float(db_exp.alpha_b)
        experiment.beta_b = float(db_exp.beta_b)

        return experiment

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        include_archived: bool = False,
    ) -> List[Experiment]:
        """
        List all experiments.

        Args:
            status: Filter by status
            include_archived: Include archived experiments

        Returns:
            List of experiments
        """
        experiments = []

        if self._db_manager:
            try:
                with self._get_session() as session:
                    stmt = select(ABExperiment)

                    conditions = []
                    if status:
                        conditions.append(ABExperiment.status == status)
                    if not include_archived:
                        conditions.append(ABExperiment.status != ExperimentStatus.ARCHIVED)

                    if conditions:
                        stmt = stmt.where(and_(*conditions))

                    stmt = stmt.order_by(ABExperiment.created_at.desc())
                    results = session.execute(stmt).scalars().all()

                    for db_exp in results:
                        experiment = self._db_to_experiment(db_exp)
                        self._experiments[experiment.name] = experiment
                        experiments.append(experiment)
            except Exception as e:
                logger.error(f"Failed to list experiments: {e}")
        else:
            # Return from cache only
            experiments = list(self._experiments.values())
            if status:
                experiments = [e for e in experiments if e.status == status]
            if not include_archived:
                experiments = [e for e in experiments if e.status != ExperimentStatus.ARCHIVED]

        return experiments

    def get_active_experiments(self) -> List[Experiment]:
        """Get all currently active experiments."""
        return self.list_experiments(status=ExperimentStatus.ACTIVE)

    def get_active_experiment(self) -> Optional[Experiment]:
        """
        Get the primary active experiment.

        If multiple experiments are active, returns the most recently started one.
        """
        if self._active_experiment and self._active_experiment.is_active:
            return self._active_experiment

        active = self.get_active_experiments()
        if active:
            # Sort by start date, most recent first
            active.sort(key=lambda e: e.start_date or datetime.min, reverse=True)
            self._active_experiment = active[0]
            return self._active_experiment

        return None

    def start_experiment(self, name: str) -> bool:
        """
        Start an experiment.

        Args:
            name: Experiment name

        Returns:
            True if started successfully
        """
        experiment = self.get_experiment(name)
        if not experiment:
            logger.error(f"Experiment '{name}' not found")
            return False

        experiment.start()

        # Update database
        if self._db_manager and experiment.experiment_id:
            try:
                self._update_experiment_status(experiment)
            except Exception as e:
                logger.error(f"Failed to update experiment status in database: {e}")

        # Set as active if no other active experiment
        if not self._active_experiment or not self._active_experiment.is_active:
            self._active_experiment = experiment

        return True

    def stop_experiment(self, name: str) -> bool:
        """
        Stop an experiment.

        Args:
            name: Experiment name

        Returns:
            True if stopped successfully
        """
        experiment = self.get_experiment(name)
        if not experiment:
            logger.error(f"Experiment '{name}' not found")
            return False

        experiment.stop()

        # Update database
        if self._db_manager and experiment.experiment_id:
            try:
                self._update_experiment_status(experiment)
            except Exception as e:
                logger.error(f"Failed to update experiment status in database: {e}")

        # Clear active experiment if it was stopped
        if self._active_experiment and self._active_experiment.name == name:
            self._active_experiment = None

        return True

    def pause_experiment(self, name: str) -> bool:
        """
        Pause an experiment.

        Args:
            name: Experiment name

        Returns:
            True if paused successfully
        """
        experiment = self.get_experiment(name)
        if not experiment:
            logger.error(f"Experiment '{name}' not found")
            return False

        experiment.pause()

        # Update database
        if self._db_manager and experiment.experiment_id:
            try:
                self._update_experiment_status(experiment)
            except Exception as e:
                logger.error(f"Failed to update experiment status in database: {e}")

        return True

    def archive_experiment(self, name: str) -> bool:
        """
        Archive an experiment.

        Args:
            name: Experiment name

        Returns:
            True if archived successfully
        """
        experiment = self.get_experiment(name)
        if not experiment:
            logger.error(f"Experiment '{name}' not found")
            return False

        experiment.archive()

        # Update database
        if self._db_manager and experiment.experiment_id:
            try:
                self._update_experiment_status(experiment)
            except Exception as e:
                logger.error(f"Failed to update experiment status in database: {e}")

        # Remove from active
        if self._active_experiment and self._active_experiment.name == name:
            self._active_experiment = None

        return True

    def _update_experiment_status(self, experiment: Experiment) -> None:
        """Update experiment status in database."""
        with self._get_session() as session:
            stmt = (
                update(ABExperiment)
                .where(ABExperiment.id == experiment.experiment_id)
                .values(
                    status=experiment.status,
                    end_date=experiment.end_date,
                    alpha_a=experiment.alpha_a,
                    beta_a=experiment.beta_a,
                    alpha_b=experiment.alpha_b,
                    beta_b=experiment.beta_b,
                    updated_at=datetime.utcnow(),
                )
            )
            session.execute(stmt)

    def record_prediction(
        self,
        experiment_name: str,
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
    ) -> Optional[str]:
        """
        Record a prediction in an experiment.

        Args:
            experiment_name: Name of the experiment
            ... (other args passed to Experiment.record_prediction)

        Returns:
            Request ID if successful, None otherwise
        """
        experiment = self.get_experiment(experiment_name)
        if not experiment:
            logger.error(f"Experiment '{experiment_name}' not found")
            return None

        # Record in experiment
        request_id = experiment.record_prediction(
            symbol=symbol,
            direction=direction,
            variant=variant,
            model_id=model_id,
            prediction_probability=prediction_probability,
            prediction_class=prediction_class,
            confidence=confidence,
            request_id=request_id,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            rrs_at_entry=rrs_at_entry,
            features=features,
        )

        # Persist to database
        if self._db_manager and experiment.experiment_id:
            try:
                self._persist_prediction(
                    experiment_id=experiment.experiment_id,
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
                )
            except Exception as e:
                logger.error(f"Failed to persist prediction: {e}")

        return request_id

    def _persist_prediction(
        self,
        experiment_id: int,
        request_id: str,
        symbol: str,
        direction: str,
        variant: ModelVariant,
        model_id: str,
        prediction_probability: float,
        prediction_class: int,
        confidence: float,
        entry_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        target_price: Optional[float] = None,
        rrs_at_entry: Optional[float] = None,
        features: Optional[Dict] = None,
    ) -> None:
        """Persist prediction to database."""
        with self._get_session() as session:
            db_prediction = ABPrediction(
                experiment_id=experiment_id,
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
            )
            session.add(db_prediction)

    def record_outcome(
        self,
        experiment_name: str,
        request_id: str,
        outcome: OutcomeType,
        pnl: Optional[float] = None,
        pnl_percent: Optional[float] = None,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
        holding_period_hours: Optional[float] = None,
    ) -> bool:
        """
        Record an outcome for a prediction.

        Args:
            experiment_name: Name of the experiment
            request_id: Prediction's request ID
            ... (other args passed to Experiment.record_outcome)

        Returns:
            True if successful
        """
        experiment = self.get_experiment(experiment_name)
        if not experiment:
            logger.error(f"Experiment '{experiment_name}' not found")
            return False

        # Record in experiment
        success = experiment.record_outcome(
            request_id=request_id,
            outcome=outcome,
            pnl=pnl,
            pnl_percent=pnl_percent,
            exit_price=exit_price,
            exit_reason=exit_reason,
            holding_period_hours=holding_period_hours,
        )

        if not success:
            return False

        # Persist to database
        if self._db_manager and experiment.experiment_id:
            try:
                self._persist_outcome(
                    experiment_id=experiment.experiment_id,
                    request_id=request_id,
                    outcome=outcome,
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    holding_period_hours=holding_period_hours,
                    prediction=experiment._predictions.get(request_id),
                )

                # Update Thompson parameters in database
                self._update_experiment_status(experiment)
            except Exception as e:
                logger.error(f"Failed to persist outcome: {e}")

        return True

    def _persist_outcome(
        self,
        experiment_id: int,
        request_id: str,
        outcome: OutcomeType,
        pnl: Optional[float],
        pnl_percent: Optional[float],
        exit_price: Optional[float],
        exit_reason: Optional[str],
        holding_period_hours: Optional[float],
        prediction: Optional[Any],
    ) -> None:
        """Persist outcome to database."""
        with self._get_session() as session:
            # Find the prediction
            stmt = select(ABPrediction).where(ABPrediction.request_id == request_id)
            db_prediction = session.execute(stmt).scalar_one_or_none()

            if not db_prediction:
                logger.warning(f"Prediction {request_id} not found in database")
                return

            # Determine if prediction was correct
            prediction_correct = None
            if outcome != OutcomeType.PENDING:
                if outcome == OutcomeType.WIN:
                    prediction_correct = db_prediction.prediction_class == 1
                elif outcome == OutcomeType.LOSS:
                    prediction_correct = db_prediction.prediction_class == 0
                elif outcome == OutcomeType.BREAKEVEN:
                    prediction_correct = True

            db_outcome = ABOutcome(
                prediction_id=db_prediction.id,
                experiment_id=experiment_id,
                outcome=outcome,
                pnl=pnl,
                pnl_percent=pnl_percent,
                exit_price=exit_price,
                exit_reason=exit_reason,
                holding_period_hours=holding_period_hours,
                prediction_correct=prediction_correct,
            )
            session.add(db_outcome)

    def update_stats(self, experiment_name: str) -> bool:
        """
        Update aggregated stats for an experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            True if successful
        """
        experiment = self.get_experiment(experiment_name)
        if not experiment:
            return False

        if not self._db_manager or not experiment.experiment_id:
            return False

        try:
            self._compute_and_persist_stats(experiment)
            return True
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")
            return False

    def _compute_and_persist_stats(self, experiment: Experiment) -> None:
        """Compute and persist experiment stats."""
        with self._get_session() as session:
            for variant in [ModelVariant.CONTROL, ModelVariant.TREATMENT]:
                # Get predictions for variant
                pred_stmt = (
                    select(ABPrediction)
                    .where(
                        and_(
                            ABPrediction.experiment_id == experiment.experiment_id,
                            ABPrediction.variant == variant,
                        )
                    )
                )
                predictions = session.execute(pred_stmt).scalars().all()

                # Get outcomes for variant
                outcome_stmt = (
                    select(ABOutcome, ABPrediction)
                    .join(ABPrediction, ABOutcome.prediction_id == ABPrediction.id)
                    .where(
                        and_(
                            ABOutcome.experiment_id == experiment.experiment_id,
                            ABPrediction.variant == variant,
                        )
                    )
                )
                outcomes = session.execute(outcome_stmt).all()

                # Compute stats
                total_predictions = len(predictions)
                total_outcomes = len(outcomes)
                wins = sum(1 for o, _ in outcomes if o.outcome == OutcomeType.WIN)
                losses = sum(1 for o, _ in outcomes if o.outcome == OutcomeType.LOSS)
                breakeven = sum(1 for o, _ in outcomes if o.outcome == OutcomeType.BREAKEVEN)
                total_pnl = sum(float(o.pnl or 0) for o, _ in outcomes)
                correct_predictions = sum(1 for o, _ in outcomes if o.prediction_correct)

                sum_confidence = sum(float(p.confidence) for p in predictions)
                sum_probability = sum(float(p.prediction_probability) for p in predictions)

                # Calculate derived metrics
                win_rate = wins / (wins + losses + breakeven) if (wins + losses + breakeven) > 0 else None
                avg_pnl = total_pnl / total_outcomes if total_outcomes > 0 else None
                accuracy = correct_predictions / total_outcomes if total_outcomes > 0 else None
                avg_confidence = sum_confidence / total_predictions if total_predictions > 0 else None
                avg_probability = sum_probability / total_predictions if total_predictions > 0 else None

                # Upsert stats
                stmt = select(ABExperimentStats).where(
                    and_(
                        ABExperimentStats.experiment_id == experiment.experiment_id,
                        ABExperimentStats.variant == variant,
                    )
                )
                existing = session.execute(stmt).scalar_one_or_none()

                if existing:
                    existing.total_predictions = total_predictions
                    existing.total_outcomes = total_outcomes
                    existing.wins = wins
                    existing.losses = losses
                    existing.breakeven = breakeven
                    existing.total_pnl = total_pnl
                    existing.avg_pnl = avg_pnl
                    existing.correct_predictions = correct_predictions
                    existing.accuracy = accuracy
                    existing.avg_confidence = avg_confidence
                    existing.avg_probability = avg_probability
                    existing.win_rate = win_rate
                    existing.computed_at = datetime.utcnow()
                else:
                    stats = ABExperimentStats(
                        experiment_id=experiment.experiment_id,
                        variant=variant,
                        total_predictions=total_predictions,
                        total_outcomes=total_outcomes,
                        wins=wins,
                        losses=losses,
                        breakeven=breakeven,
                        total_pnl=total_pnl,
                        avg_pnl=avg_pnl,
                        correct_predictions=correct_predictions,
                        accuracy=accuracy,
                        avg_confidence=avg_confidence,
                        avg_probability=avg_probability,
                        win_rate=win_rate,
                    )
                    session.add(stats)


# Global experiment manager instance
_experiment_manager: Optional[ExperimentManager] = None


def get_experiment_manager() -> ExperimentManager:
    """Get or create the global experiment manager."""
    global _experiment_manager
    if _experiment_manager is None:
        _experiment_manager = ExperimentManager()
    return _experiment_manager


def reset_experiment_manager() -> None:
    """Reset the global experiment manager."""
    global _experiment_manager
    _experiment_manager = None
