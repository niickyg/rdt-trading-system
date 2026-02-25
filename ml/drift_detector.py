"""
ML Model Drift Detection for RDT Trading System

Provides comprehensive drift detection using:
- Population Stability Index (PSI) for feature drift
- Kolmogorov-Smirnov test for distribution changes
- Performance degradation detection (accuracy, precision, recall)

Configurable thresholds with automatic alerting.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
from scipy import stats
from loguru import logger

from ml.monitoring_store import MonitoringStore


class DriftSeverity(Enum):
    """Severity levels for drift detection"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftThresholds:
    """Configurable thresholds for drift detection"""
    # PSI thresholds (Population Stability Index)
    psi_low: float = 0.1  # No significant change
    psi_medium: float = 0.2  # Moderate change, investigate
    psi_high: float = 0.25  # Significant change, action required

    # KS test thresholds (p-value based)
    ks_significance_level: float = 0.05  # Alpha for significance
    ks_critical_pvalue: float = 0.01  # Critical p-value

    # Performance degradation thresholds (relative change)
    accuracy_degradation_warning: float = 0.05  # 5% drop
    accuracy_degradation_critical: float = 0.10  # 10% drop

    precision_degradation_warning: float = 0.08  # 8% drop
    precision_degradation_critical: float = 0.15  # 15% drop

    recall_degradation_warning: float = 0.08  # 8% drop
    recall_degradation_critical: float = 0.15  # 15% drop

    # Minimum samples required for reliable detection
    min_samples_feature_drift: int = 100
    min_samples_prediction_drift: int = 50
    min_samples_performance: int = 30


@dataclass
class FeatureDriftResult:
    """Result of feature drift detection for a single feature"""
    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_pvalue: float
    baseline_mean: float
    current_mean: float
    baseline_std: float
    current_std: float
    severity: DriftSeverity
    drift_detected: bool


@dataclass
class PredictionDriftResult:
    """Result of prediction distribution drift detection"""
    psi_score: float
    ks_statistic: float
    ks_pvalue: float
    baseline_distribution: Dict[int, float]
    current_distribution: Dict[int, float]
    severity: DriftSeverity
    drift_detected: bool


@dataclass
class PerformanceDriftResult:
    """Result of performance degradation detection"""
    metric_name: str
    baseline_value: float
    current_value: float
    absolute_change: float
    relative_change: float
    severity: DriftSeverity
    degradation_detected: bool


@dataclass
class DriftReport:
    """Comprehensive drift detection report"""
    timestamp: datetime
    feature_drift_results: List[FeatureDriftResult]
    prediction_drift_result: Optional[PredictionDriftResult]
    performance_drift_results: List[PerformanceDriftResult]

    # Summary
    total_features_checked: int
    features_with_drift: int
    prediction_drift_detected: bool
    performance_degradation_detected: bool

    # Overall assessment
    overall_severity: DriftSeverity
    requires_action: bool
    recommended_action: str

    # Sample counts
    baseline_samples: int
    current_samples: int

    def to_dict(self) -> Dict:
        """Convert report to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'feature_drift_results': [
                {
                    'feature_name': r.feature_name,
                    'psi_score': r.psi_score,
                    'ks_statistic': r.ks_statistic,
                    'ks_pvalue': r.ks_pvalue,
                    'baseline_mean': r.baseline_mean,
                    'current_mean': r.current_mean,
                    'severity': r.severity.value,
                    'drift_detected': r.drift_detected
                }
                for r in self.feature_drift_results
            ],
            'prediction_drift_result': {
                'psi_score': self.prediction_drift_result.psi_score,
                'ks_statistic': self.prediction_drift_result.ks_statistic,
                'ks_pvalue': self.prediction_drift_result.ks_pvalue,
                'severity': self.prediction_drift_result.severity.value,
                'drift_detected': self.prediction_drift_result.drift_detected
            } if self.prediction_drift_result else None,
            'performance_drift_results': [
                {
                    'metric_name': r.metric_name,
                    'baseline_value': r.baseline_value,
                    'current_value': r.current_value,
                    'relative_change': r.relative_change,
                    'severity': r.severity.value,
                    'degradation_detected': r.degradation_detected
                }
                for r in self.performance_drift_results
            ],
            'summary': {
                'total_features_checked': self.total_features_checked,
                'features_with_drift': self.features_with_drift,
                'prediction_drift_detected': self.prediction_drift_detected,
                'performance_degradation_detected': self.performance_degradation_detected,
                'overall_severity': self.overall_severity.value,
                'requires_action': self.requires_action,
                'recommended_action': self.recommended_action
            },
            'samples': {
                'baseline': self.baseline_samples,
                'current': self.current_samples
            }
        }


class DriftDetector:
    """
    Drift detection for ML models in the RDT Trading System.

    Monitors:
    - Feature drift using PSI and KS tests
    - Prediction distribution drift
    - Performance degradation (accuracy, precision, recall)

    Usage:
        detector = DriftDetector(monitoring_store, thresholds)
        report = detector.run_full_drift_check()

        if report.requires_action:
            # Trigger retraining or alerts
            pass
    """

    def __init__(
        self,
        monitoring_store: MonitoringStore,
        thresholds: Optional[DriftThresholds] = None,
        baseline_performance: Optional[Dict[str, float]] = None
    ):
        """
        Initialize drift detector.

        Args:
            monitoring_store: MonitoringStore instance with historical data
            thresholds: Custom drift detection thresholds
            baseline_performance: Baseline performance metrics from training
        """
        self.monitoring_store = monitoring_store
        self.thresholds = thresholds or DriftThresholds()
        self.baseline_performance = baseline_performance or {}

        # Cache for efficiency
        self._last_check_time: Optional[datetime] = None
        self._last_report: Optional[DriftReport] = None

        logger.info("DriftDetector initialized")

    def set_baseline_performance(self, metrics: Dict[str, float]):
        """
        Set baseline performance metrics.

        Args:
            metrics: Dictionary with 'accuracy', 'precision', 'recall', 'f1_score'
        """
        self.baseline_performance = metrics
        logger.info(f"Set baseline performance: accuracy={metrics.get('accuracy', 0):.4f}")

    def check_feature_drift(
        self,
        feature_name: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[FeatureDriftResult]:
        """
        Check for feature drift using PSI and KS tests.

        Args:
            feature_name: Specific feature to check (or all if None)
            since: Only use data since this timestamp

        Returns:
            List of FeatureDriftResult for each feature checked
        """
        results = []

        # Get baseline distributions
        baseline_features, _ = self.monitoring_store.get_baseline_distributions()

        if baseline_features is None:
            logger.warning("No baseline feature distributions available")
            return results

        # Get current distributions
        current_features = self.monitoring_store.get_feature_distributions(since=since)

        if current_features is None:
            logger.warning("No current feature data available")
            return results

        # Get recent predictions for KS test
        predictions = self.monitoring_store.get_recent_predictions(since=since)
        if len(predictions) < self.thresholds.min_samples_feature_drift:
            logger.warning(
                f"Insufficient samples for feature drift detection "
                f"({len(predictions)} < {self.thresholds.min_samples_feature_drift})"
            )
            return results

        features_to_check = [feature_name] if feature_name else list(baseline_features.keys())

        for fname in features_to_check:
            if fname not in baseline_features or fname not in current_features:
                continue

            baseline = baseline_features[fname]
            current = current_features[fname]

            # Calculate PSI
            psi_score = self._calculate_psi_from_stats(
                baseline_mean=baseline['mean'],
                baseline_std=baseline['std'],
                current_mean=current['mean'],
                current_std=current['std']
            )

            # Get feature index for KS test
            feature_names = self.monitoring_store._feature_names
            if feature_names and fname in feature_names:
                feature_idx = feature_names.index(fname)
                current_values = np.array([p.features[feature_idx] for p in predictions])

                # Simulate baseline values for KS test (using normal distribution approximation)
                baseline_values = np.random.normal(
                    baseline['mean'],
                    max(baseline['std'], 0.001),  # Avoid zero std
                    len(current_values)
                )

                ks_stat, ks_pvalue = stats.ks_2samp(baseline_values, current_values)
            else:
                ks_stat, ks_pvalue = 0.0, 1.0

            # Determine severity
            severity = self._get_feature_drift_severity(psi_score, ks_pvalue)
            drift_detected = severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL]

            results.append(FeatureDriftResult(
                feature_name=fname,
                psi_score=psi_score,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pvalue,
                baseline_mean=baseline['mean'],
                current_mean=current['mean'],
                baseline_std=baseline['std'],
                current_std=current['std'],
                severity=severity,
                drift_detected=drift_detected
            ))

        return results

    def check_prediction_drift(
        self,
        since: Optional[datetime] = None
    ) -> Optional[PredictionDriftResult]:
        """
        Check for drift in prediction distribution.

        Args:
            since: Only use data since this timestamp

        Returns:
            PredictionDriftResult or None if insufficient data
        """
        # Get baseline distribution
        _, baseline_pred = self.monitoring_store.get_baseline_distributions()

        if baseline_pred is None:
            logger.warning("No baseline prediction distribution available")
            return None

        # Get current distribution
        current_pred = self.monitoring_store.get_prediction_distribution(since=since)

        if current_pred is None:
            logger.warning("No current prediction data available")
            return None

        # Check sample count
        predictions = self.monitoring_store.get_recent_predictions(since=since)
        if len(predictions) < self.thresholds.min_samples_prediction_drift:
            logger.warning(
                f"Insufficient samples for prediction drift detection "
                f"({len(predictions)} < {self.thresholds.min_samples_prediction_drift})"
            )
            return None

        # Calculate PSI for prediction distribution
        psi_score = self._calculate_psi_distributions(baseline_pred, current_pred)

        # KS test on prediction probabilities
        baseline_probs = np.array([p for p in baseline_pred.values()])
        current_probs = np.array([p for p in current_pred.values()])

        # Expand probabilities for KS test
        baseline_samples = []
        current_samples = []
        for class_label, prob in baseline_pred.items():
            baseline_samples.extend([class_label] * int(prob * 1000))
        for class_label, prob in current_pred.items():
            current_samples.extend([class_label] * int(prob * 1000))

        if baseline_samples and current_samples:
            ks_stat, ks_pvalue = stats.ks_2samp(baseline_samples, current_samples)
        else:
            ks_stat, ks_pvalue = 0.0, 1.0

        # Determine severity
        severity = self._get_prediction_drift_severity(psi_score, ks_pvalue)
        drift_detected = severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL]

        return PredictionDriftResult(
            psi_score=psi_score,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            baseline_distribution=baseline_pred,
            current_distribution=current_pred,
            severity=severity,
            drift_detected=drift_detected
        )

    def check_performance_drift(
        self,
        since: Optional[datetime] = None,
        n_recent: Optional[int] = None
    ) -> List[PerformanceDriftResult]:
        """
        Check for performance degradation.

        Args:
            since: Only use data since this timestamp
            n_recent: Only use n most recent outcomes

        Returns:
            List of PerformanceDriftResult for each metric
        """
        results = []

        if not self.baseline_performance:
            logger.warning("No baseline performance metrics set")
            return results

        # Get current performance
        current_metrics = self.monitoring_store.get_performance_metrics(
            since=since, n_recent=n_recent
        )

        if current_metrics['sample_count'] < self.thresholds.min_samples_performance:
            logger.warning(
                f"Insufficient samples for performance drift detection "
                f"({current_metrics['sample_count']} < {self.thresholds.min_samples_performance})"
            )
            return results

        # Check each metric
        metrics_to_check = ['accuracy', 'precision', 'recall', 'f1_score']

        for metric in metrics_to_check:
            baseline_value = self.baseline_performance.get(metric, 0.0)
            current_value = current_metrics.get(metric, 0.0)

            if baseline_value > 0:
                absolute_change = current_value - baseline_value
                relative_change = absolute_change / baseline_value
            else:
                absolute_change = current_value
                relative_change = 0.0 if current_value == 0 else float('inf')

            severity = self._get_performance_drift_severity(metric, relative_change)
            degradation_detected = relative_change < -self._get_warning_threshold(metric)

            results.append(PerformanceDriftResult(
                metric_name=metric,
                baseline_value=baseline_value,
                current_value=current_value,
                absolute_change=absolute_change,
                relative_change=relative_change,
                severity=severity,
                degradation_detected=degradation_detected
            ))

        return results

    def run_full_drift_check(
        self,
        since: Optional[datetime] = None
    ) -> DriftReport:
        """
        Run comprehensive drift check.

        Args:
            since: Only use data since this timestamp (default: last 24 hours)

        Returns:
            Complete DriftReport
        """
        if since is None:
            since = datetime.now() - timedelta(hours=24)

        # Run all checks
        feature_results = self.check_feature_drift(since=since)
        prediction_result = self.check_prediction_drift(since=since)
        performance_results = self.check_performance_drift(since=since)

        # Count drifts
        features_with_drift = sum(1 for r in feature_results if r.drift_detected)
        prediction_drift = prediction_result.drift_detected if prediction_result else False
        performance_degradation = any(r.degradation_detected for r in performance_results)

        # Determine overall severity
        all_severities = [r.severity for r in feature_results]
        if prediction_result:
            all_severities.append(prediction_result.severity)
        all_severities.extend([r.severity for r in performance_results])

        overall_severity = max(all_severities, key=lambda s: list(DriftSeverity).index(s)) if all_severities else DriftSeverity.NONE

        # Determine if action is required
        requires_action = overall_severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]

        # Recommend action
        recommended_action = self._get_recommended_action(
            overall_severity,
            features_with_drift,
            prediction_drift,
            performance_degradation
        )

        # Get sample counts
        predictions = self.monitoring_store.get_recent_predictions(since=since)
        baseline_features, _ = self.monitoring_store.get_baseline_distributions()
        baseline_samples = 0
        if baseline_features:
            first_feature = next(iter(baseline_features.values()), {})
            baseline_samples = first_feature.get('count', 0) if isinstance(first_feature, dict) else 0

        report = DriftReport(
            timestamp=datetime.now(),
            feature_drift_results=feature_results,
            prediction_drift_result=prediction_result,
            performance_drift_results=performance_results,
            total_features_checked=len(feature_results),
            features_with_drift=features_with_drift,
            prediction_drift_detected=prediction_drift,
            performance_degradation_detected=performance_degradation,
            overall_severity=overall_severity,
            requires_action=requires_action,
            recommended_action=recommended_action,
            baseline_samples=baseline_samples,
            current_samples=len(predictions)
        )

        self._last_check_time = datetime.now()
        self._last_report = report

        logger.info(
            f"Drift check complete: severity={overall_severity.value}, "
            f"features_with_drift={features_with_drift}/{len(feature_results)}, "
            f"requires_action={requires_action}"
        )

        return report

    def get_last_report(self) -> Optional[DriftReport]:
        """Get the most recent drift report."""
        return self._last_report

    def _calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI = sum((current_pct - baseline_pct) * ln(current_pct / baseline_pct))

        Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Moderate change
        - PSI >= 0.2: Significant change

        Args:
            baseline: Baseline distribution values
            current: Current distribution values
            n_bins: Number of bins for histogram

        Returns:
            PSI score
        """
        # Create bins from baseline
        _, bin_edges = np.histogram(baseline, bins=n_bins)

        # Calculate histograms
        baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
        current_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to percentages
        baseline_pct = baseline_counts / len(baseline)
        current_pct = current_counts / len(current)

        # Avoid division by zero
        baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
        current_pct = np.where(current_pct == 0, 0.0001, current_pct)

        # Calculate PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

        return float(psi)

    def _calculate_psi_from_stats(
        self,
        baseline_mean: float,
        baseline_std: float,
        current_mean: float,
        current_std: float,
        n_samples: int = 1000
    ) -> float:
        """
        Calculate PSI using distribution statistics.

        Args:
            baseline_mean: Baseline mean
            baseline_std: Baseline standard deviation
            current_mean: Current mean
            current_std: Current standard deviation
            n_samples: Number of samples to generate for PSI calculation

        Returns:
            PSI score
        """
        # Generate synthetic samples based on normal distribution assumption
        baseline_samples = np.random.normal(baseline_mean, max(baseline_std, 0.001), n_samples)
        current_samples = np.random.normal(current_mean, max(current_std, 0.001), n_samples)

        return self._calculate_psi(baseline_samples, current_samples)

    def _calculate_psi_distributions(
        self,
        baseline_dist: Dict[int, float],
        current_dist: Dict[int, float]
    ) -> float:
        """
        Calculate PSI for categorical distributions.

        Args:
            baseline_dist: Baseline class distribution
            current_dist: Current class distribution

        Returns:
            PSI score
        """
        all_classes = set(baseline_dist.keys()) | set(current_dist.keys())

        psi = 0.0
        for cls in all_classes:
            baseline_pct = baseline_dist.get(cls, 0.0001)
            current_pct = current_dist.get(cls, 0.0001)

            # Avoid log(0)
            baseline_pct = max(baseline_pct, 0.0001)
            current_pct = max(current_pct, 0.0001)

            psi += (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)

        return float(psi)

    def _get_feature_drift_severity(
        self,
        psi_score: float,
        ks_pvalue: float
    ) -> DriftSeverity:
        """Determine severity of feature drift."""
        if psi_score >= self.thresholds.psi_high:
            return DriftSeverity.CRITICAL
        elif psi_score >= self.thresholds.psi_medium:
            if ks_pvalue < self.thresholds.ks_critical_pvalue:
                return DriftSeverity.HIGH
            return DriftSeverity.MEDIUM
        elif psi_score >= self.thresholds.psi_low:
            return DriftSeverity.LOW
        return DriftSeverity.NONE

    def _get_prediction_drift_severity(
        self,
        psi_score: float,
        ks_pvalue: float
    ) -> DriftSeverity:
        """Determine severity of prediction drift."""
        if psi_score >= self.thresholds.psi_high * 1.5:  # Higher threshold for predictions
            return DriftSeverity.CRITICAL
        elif psi_score >= self.thresholds.psi_medium:
            return DriftSeverity.HIGH
        elif psi_score >= self.thresholds.psi_low:
            return DriftSeverity.MEDIUM
        return DriftSeverity.NONE

    def _get_performance_drift_severity(
        self,
        metric: str,
        relative_change: float
    ) -> DriftSeverity:
        """Determine severity of performance degradation."""
        if metric == 'accuracy':
            warning = self.thresholds.accuracy_degradation_warning
            critical = self.thresholds.accuracy_degradation_critical
        elif metric == 'precision':
            warning = self.thresholds.precision_degradation_warning
            critical = self.thresholds.precision_degradation_critical
        elif metric == 'recall':
            warning = self.thresholds.recall_degradation_warning
            critical = self.thresholds.recall_degradation_critical
        else:
            warning = 0.05
            critical = 0.10

        if relative_change < -critical:
            return DriftSeverity.CRITICAL
        elif relative_change < -warning:
            return DriftSeverity.HIGH
        elif relative_change < -warning / 2:
            return DriftSeverity.MEDIUM
        elif relative_change < 0:
            return DriftSeverity.LOW
        return DriftSeverity.NONE

    def _get_warning_threshold(self, metric: str) -> float:
        """Get warning threshold for a metric."""
        if metric == 'accuracy':
            return self.thresholds.accuracy_degradation_warning
        elif metric == 'precision':
            return self.thresholds.precision_degradation_warning
        elif metric == 'recall':
            return self.thresholds.recall_degradation_warning
        return 0.05

    def _get_recommended_action(
        self,
        severity: DriftSeverity,
        features_with_drift: int,
        prediction_drift: bool,
        performance_degradation: bool
    ) -> str:
        """Get recommended action based on drift results."""
        if severity == DriftSeverity.CRITICAL:
            if performance_degradation:
                return "IMMEDIATE: Trigger model retraining. Performance has degraded significantly."
            return "URGENT: Investigate data pipeline. Multiple features show significant drift."

        elif severity == DriftSeverity.HIGH:
            if prediction_drift:
                return "Schedule retraining within 24 hours. Prediction distribution has shifted."
            return f"Monitor closely. {features_with_drift} features show drift."

        elif severity == DriftSeverity.MEDIUM:
            return "Continue monitoring. Some drift detected but within acceptable limits."

        elif severity == DriftSeverity.LOW:
            return "No action required. Minor variations within normal range."

        return "No action required. Model performance is stable."

    def get_drift_metrics_for_prometheus(self) -> Dict[str, float]:
        """
        Get drift metrics formatted for Prometheus.

        Returns:
            Dictionary of metric name to value
        """
        metrics = {}

        if not self._last_report:
            return metrics

        report = self._last_report

        # Feature PSI scores
        for result in report.feature_drift_results:
            safe_name = result.feature_name.replace('.', '_').replace('-', '_')
            metrics[f'psi_{safe_name}'] = result.psi_score

        # Prediction drift
        if report.prediction_drift_result:
            metrics['prediction_psi'] = report.prediction_drift_result.psi_score

        # Performance metrics
        for result in report.performance_drift_results:
            metrics[f'current_{result.metric_name}'] = result.current_value
            metrics[f'baseline_{result.metric_name}'] = result.baseline_value

        # Summary metrics
        severity_map = {
            DriftSeverity.NONE: 0,
            DriftSeverity.LOW: 1,
            DriftSeverity.MEDIUM: 2,
            DriftSeverity.HIGH: 3,
            DriftSeverity.CRITICAL: 4
        }
        metrics['overall_drift_severity'] = severity_map[report.overall_severity]
        metrics['features_with_drift'] = report.features_with_drift
        metrics['requires_action'] = 1.0 if report.requires_action else 0.0

        return metrics
