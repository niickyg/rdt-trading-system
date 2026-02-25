"""
Statistical Analysis for A/B Testing.

Provides statistical tests and analysis methods for evaluating
A/B experiment results.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
from scipy import stats
from loguru import logger

from ml.ab_testing.experiment import Experiment, VariantStats
from ml.ab_testing.models import ModelVariant, OutcomeType


class Winner(str, Enum):
    """Experiment winner determination."""
    CONTROL = "control"
    TREATMENT = "treatment"
    NO_WINNER = "no_winner"
    INCONCLUSIVE = "inconclusive"


@dataclass
class TestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    details: Optional[Dict] = None


@dataclass
class AnalysisResult:
    """Complete analysis result for an experiment."""
    experiment_name: str
    experiment_id: Optional[int]
    analysis_time: str
    has_sufficient_data: bool
    winner: Winner
    confidence: float
    summary: str

    # Per-variant stats
    control_stats: Dict[str, Any]
    treatment_stats: Dict[str, Any]

    # Test results
    conversion_test: Optional[TestResult] = None
    pnl_test: Optional[TestResult] = None
    accuracy_test: Optional[TestResult] = None

    # Lift calculations
    conversion_lift: Optional[float] = None
    pnl_lift: Optional[float] = None
    accuracy_lift: Optional[float] = None

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "analysis_time": self.analysis_time,
            "has_sufficient_data": self.has_sufficient_data,
            "winner": self.winner.value,
            "confidence": round(self.confidence, 4),
            "summary": self.summary,
            "control": self.control_stats,
            "treatment": self.treatment_stats,
            "lift": {
                "conversion": round(self.conversion_lift, 4) if self.conversion_lift else None,
                "pnl": round(self.pnl_lift, 4) if self.pnl_lift else None,
                "accuracy": round(self.accuracy_lift, 4) if self.accuracy_lift else None,
            },
            "recommendations": self.recommendations,
        }

        if self.conversion_test:
            result["conversion_test"] = {
                "test_name": self.conversion_test.test_name,
                "statistic": round(self.conversion_test.statistic, 4),
                "p_value": round(self.conversion_test.p_value, 6),
                "is_significant": self.conversion_test.is_significant,
                "effect_size": round(self.conversion_test.effect_size, 4) if self.conversion_test.effect_size else None,
            }

        if self.pnl_test:
            result["pnl_test"] = {
                "test_name": self.pnl_test.test_name,
                "statistic": round(self.pnl_test.statistic, 4),
                "p_value": round(self.pnl_test.p_value, 6),
                "is_significant": self.pnl_test.is_significant,
                "effect_size": round(self.pnl_test.effect_size, 4) if self.pnl_test.effect_size else None,
            }

        if self.accuracy_test:
            result["accuracy_test"] = {
                "test_name": self.accuracy_test.test_name,
                "statistic": round(self.accuracy_test.statistic, 4),
                "p_value": round(self.accuracy_test.p_value, 6),
                "is_significant": self.accuracy_test.is_significant,
            }

        return result


class ABTestAnalyzer:
    """
    Statistical analyzer for A/B test experiments.

    Provides:
    - Chi-square test for conversion/win rates
    - T-test for continuous metrics (P&L)
    - Z-test for proportions
    - Confidence interval calculations
    - Effect size calculations
    """

    def __init__(self, confidence_threshold: float = 0.95):
        """
        Initialize the analyzer.

        Args:
            confidence_threshold: Required confidence level (e.g., 0.95 for 95%)
        """
        self.confidence_threshold = confidence_threshold
        self.alpha = 1 - confidence_threshold

    def analyze_experiment(self, experiment: Experiment) -> AnalysisResult:
        """
        Perform complete statistical analysis of an experiment.

        Args:
            experiment: The experiment to analyze

        Returns:
            AnalysisResult with all test results and recommendations
        """
        from datetime import datetime

        stats_a = experiment.get_variant_stats(ModelVariant.CONTROL)
        stats_b = experiment.get_variant_stats(ModelVariant.TREATMENT)

        has_sufficient_data = experiment.has_sufficient_data()

        # Initialize result
        result = AnalysisResult(
            experiment_name=experiment.name,
            experiment_id=experiment.experiment_id,
            analysis_time=datetime.utcnow().isoformat(),
            has_sufficient_data=has_sufficient_data,
            winner=Winner.INCONCLUSIVE,
            confidence=0.0,
            summary="",
            control_stats=stats_a.to_dict(),
            treatment_stats=stats_b.to_dict(),
            recommendations=[],
        )

        if not has_sufficient_data:
            result.summary = (
                f"Insufficient data for statistical analysis. "
                f"Need {experiment.min_samples_per_variant} samples per variant. "
                f"Control: {stats_a.total_outcomes}, Treatment: {stats_b.total_outcomes}"
            )
            result.recommendations.append(
                "Continue running the experiment until sufficient data is collected."
            )
            return result

        # Run statistical tests
        result.conversion_test = self._test_conversion_rates(stats_a, stats_b)
        result.pnl_test = self._test_pnl(experiment)
        result.accuracy_test = self._test_accuracy(stats_a, stats_b)

        # Calculate lift
        result.conversion_lift = self._calculate_lift(stats_a.win_rate, stats_b.win_rate)
        result.pnl_lift = self._calculate_lift(stats_a.avg_pnl, stats_b.avg_pnl)
        result.accuracy_lift = self._calculate_lift(stats_a.accuracy, stats_b.accuracy)

        # Determine winner
        result.winner, result.confidence = self._determine_winner(result)

        # Generate summary and recommendations
        result.summary = self._generate_summary(result)
        result.recommendations = self._generate_recommendations(result)

        logger.info(f"Analysis complete for '{experiment.name}': {result.winner.value}")

        return result

    def _test_conversion_rates(
        self,
        stats_a: VariantStats,
        stats_b: VariantStats,
    ) -> TestResult:
        """
        Test for significant difference in conversion/win rates using chi-square test.

        Args:
            stats_a: Control variant stats
            stats_b: Treatment variant stats

        Returns:
            TestResult with chi-square test results
        """
        # Build contingency table
        # [wins, non-wins] for each variant
        a_wins = stats_a.wins
        a_non_wins = stats_a.losses + stats_a.breakeven
        b_wins = stats_b.wins
        b_non_wins = stats_b.losses + stats_b.breakeven

        contingency_table = np.array([
            [a_wins, a_non_wins],
            [b_wins, b_non_wins]
        ])

        # Perform chi-square test
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        except ValueError:
            # Not enough variation
            return TestResult(
                test_name="chi_square",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_level=self.confidence_threshold,
                details={"error": "Insufficient variation for chi-square test"},
            )

        is_significant = p_value < self.alpha

        # Calculate effect size (Cramer's V)
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = math.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0.0

        return TestResult(
            test_name="chi_square",
            statistic=chi2,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_threshold,
            effect_size=cramers_v,
            details={
                "degrees_of_freedom": dof,
                "expected": expected.tolist(),
                "observed": contingency_table.tolist(),
            },
        )

    def _test_pnl(self, experiment: Experiment) -> Optional[TestResult]:
        """
        Test for significant difference in P&L using Welch's t-test.

        Args:
            experiment: The experiment

        Returns:
            TestResult with t-test results, or None if insufficient data
        """
        # Collect P&L values for each variant
        pnl_a = []
        pnl_b = []

        for request_id, outcome in experiment._outcomes.items():
            if outcome.pnl is not None:
                prediction = experiment._predictions.get(request_id)
                if prediction:
                    if prediction.variant == ModelVariant.CONTROL:
                        pnl_a.append(outcome.pnl)
                    else:
                        pnl_b.append(outcome.pnl)

        if len(pnl_a) < 2 or len(pnl_b) < 2:
            return None

        pnl_a = np.array(pnl_a)
        pnl_b = np.array(pnl_b)

        # Perform Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(pnl_a, pnl_b, equal_var=False)

        is_significant = p_value < self.alpha

        # Calculate Cohen's d (effect size)
        pooled_std = math.sqrt(
            (pnl_a.var() * (len(pnl_a) - 1) + pnl_b.var() * (len(pnl_b) - 1)) /
            (len(pnl_a) + len(pnl_b) - 2)
        )
        cohens_d = (pnl_b.mean() - pnl_a.mean()) / pooled_std if pooled_std > 0 else 0.0

        # Calculate confidence interval for the difference
        mean_diff = pnl_b.mean() - pnl_a.mean()
        se_diff = math.sqrt(pnl_a.var() / len(pnl_a) + pnl_b.var() / len(pnl_b))
        t_critical = stats.t.ppf(1 - self.alpha / 2, len(pnl_a) + len(pnl_b) - 2)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff

        return TestResult(
            test_name="welch_ttest",
            statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_threshold,
            effect_size=cohens_d,
            confidence_interval=(ci_lower, ci_upper),
            details={
                "control_mean": float(pnl_a.mean()),
                "control_std": float(pnl_a.std()),
                "control_n": len(pnl_a),
                "treatment_mean": float(pnl_b.mean()),
                "treatment_std": float(pnl_b.std()),
                "treatment_n": len(pnl_b),
                "mean_difference": mean_diff,
            },
        )

    def _test_accuracy(
        self,
        stats_a: VariantStats,
        stats_b: VariantStats,
    ) -> TestResult:
        """
        Test for significant difference in prediction accuracy using z-test for proportions.

        Args:
            stats_a: Control variant stats
            stats_b: Treatment variant stats

        Returns:
            TestResult with z-test results
        """
        n_a = stats_a.total_outcomes
        n_b = stats_b.total_outcomes

        if n_a == 0 or n_b == 0:
            return TestResult(
                test_name="z_test_proportions",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_level=self.confidence_threshold,
            )

        p_a = stats_a.accuracy
        p_b = stats_b.accuracy

        # Pooled proportion
        p_pooled = (stats_a.correct_predictions + stats_b.correct_predictions) / (n_a + n_b)

        # Standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n_a + 1 / n_b))

        if se == 0:
            return TestResult(
                test_name="z_test_proportions",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_level=self.confidence_threshold,
            )

        # Z-statistic
        z = (p_b - p_a) / se

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        is_significant = p_value < self.alpha

        return TestResult(
            test_name="z_test_proportions",
            statistic=z,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_threshold,
            details={
                "control_accuracy": p_a,
                "treatment_accuracy": p_b,
                "pooled_proportion": p_pooled,
            },
        )

    def _calculate_lift(
        self,
        baseline: float,
        treatment: float,
    ) -> Optional[float]:
        """
        Calculate percentage lift from baseline to treatment.

        Args:
            baseline: Baseline (control) value
            treatment: Treatment value

        Returns:
            Lift as a decimal (e.g., 0.1 for 10% lift)
        """
        if baseline is None or treatment is None:
            return None
        if baseline == 0:
            return None
        return (treatment - baseline) / baseline

    def _determine_winner(
        self,
        result: AnalysisResult,
    ) -> Tuple[Winner, float]:
        """
        Determine the experiment winner based on test results.

        Args:
            result: Analysis result with test results

        Returns:
            Tuple of (winner, confidence)
        """
        # Primary metric: conversion/win rate
        if result.conversion_test and result.conversion_test.is_significant:
            if result.conversion_lift and result.conversion_lift > 0:
                return Winner.TREATMENT, 1 - result.conversion_test.p_value
            elif result.conversion_lift and result.conversion_lift < 0:
                return Winner.CONTROL, 1 - result.conversion_test.p_value

        # Secondary: P&L
        if result.pnl_test and result.pnl_test.is_significant:
            if result.pnl_lift and result.pnl_lift > 0:
                return Winner.TREATMENT, 1 - result.pnl_test.p_value
            elif result.pnl_lift and result.pnl_lift < 0:
                return Winner.CONTROL, 1 - result.pnl_test.p_value

        # No significant difference
        return Winner.NO_WINNER, 0.0

    def _generate_summary(self, result: AnalysisResult) -> str:
        """Generate a human-readable summary."""
        parts = []

        if result.winner == Winner.TREATMENT:
            parts.append(
                f"Treatment model (B) is the winner with {result.confidence * 100:.1f}% confidence."
            )
        elif result.winner == Winner.CONTROL:
            parts.append(
                f"Control model (A) is the winner with {result.confidence * 100:.1f}% confidence."
            )
        elif result.winner == Winner.NO_WINNER:
            parts.append("No statistically significant winner. Results are inconclusive.")
        else:
            parts.append("Analysis is inconclusive due to insufficient data or no significant difference.")

        # Add lift information
        if result.conversion_lift:
            direction = "higher" if result.conversion_lift > 0 else "lower"
            parts.append(
                f"Treatment win rate is {abs(result.conversion_lift) * 100:.1f}% {direction} than control."
            )

        if result.pnl_lift:
            direction = "higher" if result.pnl_lift > 0 else "lower"
            parts.append(
                f"Treatment avg P&L is {abs(result.pnl_lift) * 100:.1f}% {direction} than control."
            )

        return " ".join(parts)

    def _generate_recommendations(self, result: AnalysisResult) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if result.winner == Winner.TREATMENT:
            recommendations.append("Consider deploying the treatment model (B) to production.")
            recommendations.append("Monitor performance closely after deployment.")

        elif result.winner == Winner.CONTROL:
            recommendations.append("Keep the control model (A) in production.")
            recommendations.append("Investigate why treatment model underperformed.")

        elif result.winner == Winner.NO_WINNER:
            # Check if close to significance
            if result.conversion_test and result.conversion_test.p_value < 0.1:
                recommendations.append(
                    "Results are trending toward significance. Consider extending the experiment."
                )
            else:
                recommendations.append("Continue running the experiment to collect more data.")

            # Check sample size
            ctrl_n = result.control_stats.get("total_outcomes", 0)
            treat_n = result.treatment_stats.get("total_outcomes", 0)
            min_n = min(ctrl_n, treat_n)

            if min_n < 500:
                recommendations.append(
                    f"Current sample size ({min_n}) is relatively small. "
                    "Consider running until at least 500 outcomes per variant."
                )

        # Effect size recommendations
        if result.conversion_test and result.conversion_test.effect_size:
            effect = result.conversion_test.effect_size
            if effect < 0.1:
                recommendations.append(
                    "Effect size is very small. Consider whether the difference is practically significant."
                )
            elif effect < 0.3:
                recommendations.append("Effect size is small but potentially meaningful.")
            else:
                recommendations.append("Effect size is substantial.")

        return recommendations

    def calculate_required_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        power: float = 0.8,
    ) -> int:
        """
        Calculate required sample size per variant.

        Args:
            baseline_rate: Expected baseline conversion rate
            minimum_detectable_effect: Minimum relative effect to detect
            power: Statistical power (default 0.8)

        Returns:
            Required sample size per variant
        """
        # Treatment rate
        treatment_rate = baseline_rate * (1 + minimum_detectable_effect)

        # Z-scores
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(power)

        # Pooled rate
        p_pooled = (baseline_rate + treatment_rate) / 2

        # Sample size formula
        n = (
            2 * p_pooled * (1 - p_pooled) * ((z_alpha + z_beta) ** 2) /
            ((treatment_rate - baseline_rate) ** 2)
        )

        return int(math.ceil(n))

    def get_winner(self, experiment: Experiment) -> Tuple[Winner, float]:
        """
        Quick method to get the winner without full analysis.

        Args:
            experiment: The experiment

        Returns:
            Tuple of (winner, confidence)
        """
        result = self.analyze_experiment(experiment)
        return result.winner, result.confidence

    def calculate_confidence(self, experiment: Experiment) -> float:
        """
        Calculate current confidence level.

        Args:
            experiment: The experiment

        Returns:
            Confidence level (0.0 to 1.0)
        """
        result = self.analyze_experiment(experiment)
        return result.confidence


# Convenience functions
def analyze_experiment(experiment: Experiment, confidence_threshold: float = 0.95) -> AnalysisResult:
    """
    Analyze an experiment.

    Args:
        experiment: The experiment to analyze
        confidence_threshold: Required confidence level

    Returns:
        AnalysisResult
    """
    analyzer = ABTestAnalyzer(confidence_threshold=confidence_threshold)
    return analyzer.analyze_experiment(experiment)


def get_winner(experiment: Experiment, confidence_threshold: float = 0.95) -> Tuple[Winner, float]:
    """
    Get the winner of an experiment.

    Args:
        experiment: The experiment
        confidence_threshold: Required confidence level

    Returns:
        Tuple of (winner, confidence)
    """
    analyzer = ABTestAnalyzer(confidence_threshold=confidence_threshold)
    return analyzer.get_winner(experiment)


def calculate_confidence(experiment: Experiment, confidence_threshold: float = 0.95) -> float:
    """
    Calculate confidence level for an experiment.

    Args:
        experiment: The experiment
        confidence_threshold: Required confidence level

    Returns:
        Confidence level
    """
    analyzer = ABTestAnalyzer(confidence_threshold=confidence_threshold)
    return analyzer.calculate_confidence(experiment)
