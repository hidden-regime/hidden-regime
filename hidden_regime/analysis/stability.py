"""
Regime stability metrics for trading quality validation.

Provides convenient wrapper for validating regime detection quality before deploying
trading strategies. Essential for Sharpe 10+ strategies where regime quality directly
impacts win rate and drawdown.

Key Metrics:
- Mean duration: 20-90 days ideal for daily trading
- Persistence: >0.6 indicates autocorrelated regimes (not random)
- Stability score: >0.6 indicates model is not overfitting
- Switching frequency: <0.2 indicates stable regimes

Usage:
    from hidden_regime.analysis.stability import RegimeStabilityMetrics

    # After running regime detection
    metrics = RegimeStabilityMetrics(analysis_results)
    quality = metrics.get_metrics()

    # Validate before trading
    if quality['mean_duration'] < 10:
        print("⚠️ Regimes too noisy, increase n_states or add smoothing")
    if quality['persistence'] < 0.5:
        print("⚠️ No autocorrelation, regimes are random")
    if quality['stability_score'] < 0.4:
        print("⚠️ Model unstable, don't trade this")
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from .performance import RegimePerformanceAnalyzer
from ..utils.exceptions import ValidationError


class RegimeStabilityMetrics:
    """
    Convenient wrapper for regime stability validation.

    Provides easy access to stability metrics from RegimePerformanceAnalyzer
    in a format optimized for trading quality checks.

    Attributes:
        results: DataFrame with regime analysis results
        analyzer: RegimePerformanceAnalyzer instance
        _cached_metrics: Cached metrics to avoid recomputation
    """

    def __init__(self, analysis_results: pd.DataFrame):
        """
        Initialize stability metrics calculator.

        Args:
            analysis_results: DataFrame with regime detection results.
                Must contain 'predicted_state' and 'confidence' columns.

        Raises:
            ValidationError: If required columns are missing
        """
        if analysis_results is None or analysis_results.empty:
            raise ValidationError("analysis_results cannot be empty")

        required_cols = ['predicted_state', 'confidence']
        missing_cols = [col for col in required_cols if col not in analysis_results.columns]
        if missing_cols:
            raise ValidationError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {list(analysis_results.columns)}"
            )

        self.results = analysis_results
        self.analyzer = RegimePerformanceAnalyzer()
        self._cached_metrics: Optional[Dict[str, Any]] = None

    def get_metrics(self) -> Dict[str, float]:
        """
        Get stability metrics in simple format.

        Returns comprehensive stability metrics optimized for trading validation:
        - mean_duration: Average regime duration in observations
        - median_duration: Median regime duration (robust to outliers)
        - persistence: Autocorrelation measure (0-1, higher = more persistent)
        - stability_score: Overall stability score (0-1, higher = more stable)
        - switching_frequency: Fraction of time regimes switch (0-1, lower = better)
        - consistency: Confidence consistency score (0-1, higher = better)

        Returns:
            Dict with stability metrics and trading quality indicators
        """
        if self._cached_metrics is not None:
            return self._cached_metrics

        # Get comprehensive performance analysis
        perf = self.analyzer.analyze_regime_performance(self.results)

        # Extract stability metrics
        stability = perf.get('stability_metrics', {})
        duration = perf.get('duration_analysis', {})

        # Handle case where analysis failed
        if 'error' in stability or 'error' in duration:
            return {
                'mean_duration': 0.0,
                'median_duration': 0.0,
                'persistence': 0.0,
                'stability_score': 0.0,
                'switching_frequency': 1.0,
                'consistency': 0.0,
                'quality_rating': 'Poor',
                'error': stability.get('error') or duration.get('error')
            }

        # Extract duration statistics
        overall_duration = duration.get('overall_duration_stats', {})
        mean_duration = overall_duration.get('mean', 0.0)
        median_duration = overall_duration.get('median', 0.0)

        # Calculate persistence (1 - switching_frequency)
        switching_freq = stability.get('regime_switching_frequency', 0.5)
        persistence = 1.0 - switching_freq

        # Get consistency score
        consistency = stability.get('consistency_score', 0.0)

        # Compute overall stability score
        stability_score = self._compute_stability_score(
            mean_duration=mean_duration,
            persistence=persistence,
            consistency=consistency,
            switching_freq=switching_freq
        )

        # Get quality rating
        quality_rating = self._get_quality_rating(
            mean_duration=mean_duration,
            persistence=persistence,
            stability_score=stability_score
        )

        metrics = {
            'mean_duration': float(mean_duration),
            'median_duration': float(median_duration),
            'persistence': float(persistence),
            'stability_score': float(stability_score),
            'switching_frequency': float(switching_freq),
            'consistency': float(consistency),
            'quality_rating': quality_rating,
            'is_tradeable': self._is_tradeable(
                mean_duration, persistence, stability_score
            )
        }

        self._cached_metrics = metrics
        return metrics

    def _compute_stability_score(
        self,
        mean_duration: float,
        persistence: float,
        consistency: float,
        switching_freq: float
    ) -> float:
        """
        Compute 0-1 stability score from component metrics.

        Scoring algorithm:
        - Duration score (30%): Reward 20-90 day durations
        - Persistence score (30%): Reward high persistence (>0.6)
        - Consistency score (20%): Reward consistent confidence
        - Switching penalty (20%): Penalize high switching (>0.2)

        Args:
            mean_duration: Average regime duration
            persistence: Persistence score (0-1)
            consistency: Consistency score (0-1)
            switching_freq: Switching frequency (0-1)

        Returns:
            Stability score (0-1)
        """
        # Duration score: ideal is 20-90 days, score degrades outside
        if 20 <= mean_duration <= 90:
            duration_score = 1.0
        elif mean_duration < 20:
            # Too short: penalize linearly below 20
            duration_score = max(0.0, mean_duration / 20.0)
        else:
            # Too long: penalize above 90 (but less severely)
            duration_score = max(0.0, 1.0 - (mean_duration - 90) / 200.0)

        # Persistence score: target is >0.6
        persistence_score = min(1.0, persistence / 0.6)

        # Consistency score: already 0-1
        consistency_score = consistency

        # Switching penalty: target is <0.2
        switching_score = 1.0 - min(1.0, switching_freq / 0.2)

        # Weighted combination
        score = (
            0.30 * duration_score +
            0.30 * persistence_score +
            0.20 * consistency_score +
            0.20 * switching_score
        )

        return max(0.0, min(1.0, score))

    def _get_quality_rating(
        self,
        mean_duration: float,
        persistence: float,
        stability_score: float
    ) -> str:
        """
        Generate quality rating for trading readiness.

        Ratings:
        - Excellent: Ready for Sharpe 10+ strategies
        - Good: Ready for Sharpe 5-7 strategies
        - Fair: Usable for simple strategies
        - Poor: Do not trade - regime quality too low

        Args:
            mean_duration: Average regime duration
            persistence: Persistence score
            stability_score: Overall stability score

        Returns:
            Quality rating string
        """
        # Check if meets Sharpe 10+ criteria
        if (stability_score >= 0.8 and
            persistence >= 0.7 and
            20 <= mean_duration <= 90):
            return "Excellent"

        # Check if meets Sharpe 5-7 criteria
        if (stability_score >= 0.6 and
            persistence >= 0.6 and
            10 <= mean_duration <= 120):
            return "Good"

        # Check if minimally usable
        if (stability_score >= 0.4 and
            persistence >= 0.4 and
            mean_duration >= 5):
            return "Fair"

        return "Poor"

    def _is_tradeable(
        self,
        mean_duration: float,
        persistence: float,
        stability_score: float
    ) -> bool:
        """
        Determine if regime quality is sufficient for trading.

        Minimum criteria:
        - Mean duration >= 10 days
        - Persistence >= 0.5
        - Stability score >= 0.4

        Args:
            mean_duration: Average regime duration
            persistence: Persistence score
            stability_score: Overall stability score

        Returns:
            True if tradeable, False otherwise
        """
        return (
            mean_duration >= 10 and
            persistence >= 0.5 and
            stability_score >= 0.4
        )

    def get_detailed_report(self) -> str:
        """
        Generate detailed text report of stability metrics.

        Returns:
            Formatted text report with recommendations
        """
        metrics = self.get_metrics()

        report = []
        report.append("=" * 60)
        report.append("REGIME STABILITY REPORT")
        report.append("=" * 60)
        report.append("")

        # Overall rating
        rating = metrics['quality_rating']
        tradeable = metrics['is_tradeable']
        report.append(f"Overall Quality: {rating}")
        report.append(f"Trading Ready:   {'✅ YES' if tradeable else '❌ NO'}")
        report.append("")

        # Key metrics
        report.append("Key Metrics:")
        report.append(f"  Mean Duration:        {metrics['mean_duration']:.1f} days")
        report.append(f"  Median Duration:      {metrics['median_duration']:.1f} days")
        report.append(f"  Persistence:          {metrics['persistence']:.3f} ({metrics['persistence']:.1%})")
        report.append(f"  Stability Score:      {metrics['stability_score']:.3f}")
        report.append(f"  Switching Frequency:  {metrics['switching_frequency']:.3f}")
        report.append(f"  Consistency:          {metrics['consistency']:.3f}")
        report.append("")

        # Recommendations
        report.append("Recommendations:")
        report.extend(self._generate_recommendations(metrics))
        report.append("")

        # Trading strategy suitability
        report.append("Strategy Suitability:")
        if rating == "Excellent":
            report.append("  ✅ Sharpe 10+ strategies (high-conviction, selective)")
            report.append("  ✅ Multi-timeframe confirmation strategies")
            report.append("  ✅ Regime transition options strategies")
        elif rating == "Good":
            report.append("  ✅ Sharpe 5-7 strategies (moderate frequency)")
            report.append("  ✅ Single-timeframe regime switching")
            report.append("  ⚠️ May struggle with Sharpe 10+ (needs improvement)")
        elif rating == "Fair":
            report.append("  ⚠️ Simple regime switching only")
            report.append("  ❌ Not suitable for high-Sharpe strategies")
            report.append("  ❌ Requires significant improvement")
        else:  # Poor
            report.append("  ❌ Not suitable for trading")
            report.append("  ❌ Regimes likely random or overfitted")
            report.append("  ❌ Revisit model parameters")

        report.append("=" * 60)

        return "\n".join(report)

    def _generate_recommendations(self, metrics: Dict[str, float]) -> list:
        """Generate actionable recommendations based on metrics."""
        recommendations = []

        # Duration recommendations
        mean_dur = metrics['mean_duration']
        if mean_dur < 10:
            recommendations.append("  ⚠️ Regimes too noisy (duration < 10 days)")
            recommendations.append("     → Increase n_states or lookback_days")
            recommendations.append("     → Add regime smoothing filter")
        elif mean_dur > 120:
            recommendations.append("  ⚠️ Regimes too stable (duration > 120 days)")
            recommendations.append("     → Decrease lookback_days for faster adaptation")
            recommendations.append("     → Consider using multi-timeframe model")

        # Persistence recommendations
        persistence = metrics['persistence']
        if persistence < 0.5:
            recommendations.append("  ⚠️ Low persistence - regimes may be random")
            recommendations.append("     → Check regime separation (mean returns by regime)")
            recommendations.append("     → Validate with out-of-sample testing")
            recommendations.append("     → Consider different feature engineering")

        # Stability recommendations
        stability = metrics['stability_score']
        if stability < 0.4:
            recommendations.append("  ⚠️ Low stability - model may be overfitting")
            recommendations.append("     → Run walk-forward validation")
            recommendations.append("     → Check regime consistency over time")
            recommendations.append("     → Simplify model (reduce n_states)")

        # Switching frequency recommendations
        switching = metrics['switching_frequency']
        if switching > 0.3:
            recommendations.append("  ⚠️ High switching frequency")
            recommendations.append("     → Add minimum regime duration filter")
            recommendations.append("     → Increase confidence threshold")
            recommendations.append("     → Use multi-timeframe confirmation")

        if not recommendations:
            recommendations.append("  ✅ All metrics within acceptable ranges")
            recommendations.append("  ✅ Regime quality is good for trading")

        return recommendations

    def validate_for_trading(
        self,
        min_duration: float = 10.0,
        min_persistence: float = 0.5,
        min_stability: float = 0.4
    ) -> tuple[bool, list]:
        """
        Validate regime quality against specific trading criteria.

        Args:
            min_duration: Minimum acceptable mean duration (default: 10 days)
            min_persistence: Minimum acceptable persistence (default: 0.5)
            min_stability: Minimum acceptable stability score (default: 0.4)

        Returns:
            Tuple of (is_valid, list_of_failures)
        """
        metrics = self.get_metrics()
        failures = []

        if metrics['mean_duration'] < min_duration:
            failures.append(
                f"Mean duration {metrics['mean_duration']:.1f} < {min_duration}"
            )

        if metrics['persistence'] < min_persistence:
            failures.append(
                f"Persistence {metrics['persistence']:.3f} < {min_persistence}"
            )

        if metrics['stability_score'] < min_stability:
            failures.append(
                f"Stability score {metrics['stability_score']:.3f} < {min_stability}"
            )

        is_valid = len(failures) == 0

        return is_valid, failures

    def clear_cache(self) -> None:
        """Clear cached metrics to force recomputation."""
        self._cached_metrics = None
