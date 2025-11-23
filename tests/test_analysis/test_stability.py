"""
Tests for RegimeStabilityMetrics wrapper class.

Tests cover:
- Initialization and validation
- Metric computation
- Quality ratings
- Tradeable validation
- Detailed reporting
"""

import pytest
import pandas as pd
import numpy as np

from hidden_regime.analysis.stability import RegimeStabilityMetrics
from hidden_regime.utils.exceptions import ValidationError


class TestRegimeStabilityMetricsInitialization:
    """Test initialization and validation."""

    def test_initialization_valid(self):
        """Test successful initialization with valid data."""
        data = pd.DataFrame({
            'predicted_state': [0, 0, 1, 1, 1, 2, 2, 0],
            'confidence': [0.8, 0.9, 0.7, 0.8, 0.75, 0.85, 0.9, 0.8]
        })

        metrics = RegimeStabilityMetrics(data)
        assert metrics is not None
        assert metrics.results is not None
        assert len(metrics.results) == 8

    def test_initialization_empty_data(self):
        """Test that empty data raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            RegimeStabilityMetrics(pd.DataFrame())

    def test_initialization_none_data(self):
        """Test that None data raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            RegimeStabilityMetrics(None)

    def test_initialization_missing_columns(self):
        """Test that missing required columns raises ValidationError."""
        # Missing confidence column
        data = pd.DataFrame({
            'predicted_state': [0, 1, 2]
        })

        with pytest.raises(ValidationError, match="Missing required columns"):
            RegimeStabilityMetrics(data)

        # Missing predicted_state column
        data = pd.DataFrame({
            'confidence': [0.8, 0.9, 0.7]
        })

        with pytest.raises(ValidationError, match="Missing required columns"):
            RegimeStabilityMetrics(data)


class TestRegimeStabilityMetricsComputation:
    """Test metric computation."""

    def test_get_metrics_basic(self):
        """Test basic metrics computation."""
        # Create data with stable regimes
        data = pd.DataFrame({
            'predicted_state': [0]*20 + [1]*30 + [2]*25,  # Stable regimes
            'confidence': [0.8 + np.random.randn() * 0.05 for _ in range(75)]
        })

        metrics = RegimeStabilityMetrics(data)
        result = metrics.get_metrics()

        # Check all required keys present
        assert 'mean_duration' in result
        assert 'median_duration' in result
        assert 'persistence' in result
        assert 'stability_score' in result
        assert 'switching_frequency' in result
        assert 'consistency' in result
        assert 'quality_rating' in result
        assert 'is_tradeable' in result

    def test_get_metrics_stable_regimes(self):
        """Test metrics with very stable regimes."""
        # Long duration regimes
        data = pd.DataFrame({
            'predicted_state': [0]*50 + [1]*50,
            'confidence': [0.85] * 100
        })

        metrics = RegimeStabilityMetrics(data)
        result = metrics.get_metrics()

        # Should have high persistence (only 1 switch)
        assert result['persistence'] > 0.95
        # Low switching frequency
        assert result['switching_frequency'] < 0.05
        # Good consistency (constant confidence)
        assert result['consistency'] > 0.8
        # High stability score
        assert result['stability_score'] > 0.6

    def test_get_metrics_noisy_regimes(self):
        """Test metrics with noisy, unstable regimes."""
        # Rapid switching
        data = pd.DataFrame({
            'predicted_state': [i % 3 for i in range(100)],  # Switches every step
            'confidence': [0.5 + np.random.randn() * 0.2 for _ in range(100)]
        })

        metrics = RegimeStabilityMetrics(data)
        result = metrics.get_metrics()

        # Should have low persistence
        assert result['persistence'] < 0.2
        # High switching frequency
        assert result['switching_frequency'] > 0.8
        # Low stability score
        assert result['stability_score'] < 0.4
        # Should not be tradeable
        assert result['is_tradeable'] is False

    def test_get_metrics_caching(self):
        """Test that metrics are cached."""
        data = pd.DataFrame({
            'predicted_state': [0]*20 + [1]*20,
            'confidence': [0.8] * 40
        })

        metrics = RegimeStabilityMetrics(data)

        # First call
        result1 = metrics.get_metrics()

        # Second call should return cached result
        result2 = metrics.get_metrics()

        assert result1 is result2  # Same object

        # Clear cache and get new result
        metrics.clear_cache()
        result3 = metrics.get_metrics()

        # Should be equal but not same object
        assert result3 == result1
        assert result3 is not result1


class TestRegimeStabilityQualityRatings:
    """Test quality rating generation."""

    def test_excellent_rating(self):
        """Test that excellent regimes get 'Excellent' rating."""
        # Perfect regimes: 30 days duration, high persistence, high stability
        data = pd.DataFrame({
            'predicted_state': [0]*30 + [1]*35 + [2]*30,
            'confidence': [0.9] * 95
        })

        metrics = RegimeStabilityMetrics(data)
        result = metrics.get_metrics()

        assert result['quality_rating'] == 'Excellent'
        assert result['is_tradeable'] is True
        assert result['stability_score'] >= 0.8
        assert result['persistence'] >= 0.7

    def test_good_rating(self):
        """Test that good regimes get 'Good' rating."""
        # Good regimes: 15 days duration, decent persistence
        data = pd.DataFrame({
            'predicted_state': [0]*15 + [1]*20 + [2]*18 + [0]*15,
            'confidence': [0.75 + np.random.randn() * 0.05 for _ in range(68)]
        })

        metrics = RegimeStabilityMetrics(data)
        result = metrics.get_metrics()

        # Should be at least 'Good' or better
        assert result['quality_rating'] in ['Good', 'Excellent']
        assert result['is_tradeable'] is True

    def test_fair_rating(self):
        """Test that marginal regimes get 'Fair' rating."""
        # Marginal regimes: short duration but some persistence
        data = pd.DataFrame({
            'predicted_state': [0]*8 + [1]*10 + [2]*8 + [0]*6,
            'confidence': [0.65 + np.random.randn() * 0.1 for _ in range(32)]
        })

        metrics = RegimeStabilityMetrics(data)
        result = metrics.get_metrics()

        # Should be Fair or possibly Good
        assert result['quality_rating'] in ['Fair', 'Good']

    def test_poor_rating(self):
        """Test that bad regimes get 'Poor' rating."""
        # Bad regimes: very short duration, high noise
        data = pd.DataFrame({
            'predicted_state': [i % 3 for i in range(50)],  # Switches constantly
            'confidence': [0.4 + np.random.randn() * 0.2 for _ in range(50)]
        })

        metrics = RegimeStabilityMetrics(data)
        result = metrics.get_metrics()

        assert result['quality_rating'] == 'Poor'
        assert result['is_tradeable'] is False


class TestRegimeStabilityValidation:
    """Test trading validation functionality."""

    def test_validate_for_trading_pass(self):
        """Test validation passes for good regimes."""
        data = pd.DataFrame({
            'predicted_state': [0]*25 + [1]*30 + [2]*25,
            'confidence': [0.85] * 80
        })

        metrics = RegimeStabilityMetrics(data)
        is_valid, failures = metrics.validate_for_trading()

        assert is_valid is True
        assert len(failures) == 0

    def test_validate_for_trading_fail_duration(self):
        """Test validation fails for short duration."""
        data = pd.DataFrame({
            'predicted_state': [i % 3 for i in range(30)],  # 1 day duration
            'confidence': [0.8] * 30
        })

        metrics = RegimeStabilityMetrics(data)
        is_valid, failures = metrics.validate_for_trading(min_duration=10)

        assert is_valid is False
        assert any('Mean duration' in f for f in failures)

    def test_validate_for_trading_fail_persistence(self):
        """Test validation fails for low persistence."""
        data = pd.DataFrame({
            'predicted_state': [i % 2 for i in range(100)],  # High switching
            'confidence': [0.8] * 100
        })

        metrics = RegimeStabilityMetrics(data)
        is_valid, failures = metrics.validate_for_trading(min_persistence=0.6)

        assert is_valid is False
        assert any('Persistence' in f for f in failures)

    def test_validate_for_trading_custom_thresholds(self):
        """Test validation with custom thresholds."""
        data = pd.DataFrame({
            'predicted_state': [0]*15 + [1]*15,
            'confidence': [0.75] * 30
        })

        metrics = RegimeStabilityMetrics(data)

        # Should pass with lenient thresholds
        is_valid, _ = metrics.validate_for_trading(
            min_duration=10,
            min_persistence=0.5,
            min_stability=0.3
        )
        assert is_valid is True

        # Should fail with strict thresholds
        is_valid, failures = metrics.validate_for_trading(
            min_duration=50,
            min_persistence=0.9,
            min_stability=0.9
        )
        assert is_valid is False
        assert len(failures) > 0


class TestRegimeStabilityReporting:
    """Test detailed report generation."""

    def test_get_detailed_report_structure(self):
        """Test that detailed report has correct structure."""
        data = pd.DataFrame({
            'predicted_state': [0]*30 + [1]*30,
            'confidence': [0.8] * 60
        })

        metrics = RegimeStabilityMetrics(data)
        report = metrics.get_detailed_report()

        # Check report contains key sections
        assert "REGIME STABILITY REPORT" in report
        assert "Overall Quality:" in report
        assert "Trading Ready:" in report
        assert "Key Metrics:" in report
        assert "Recommendations:" in report
        assert "Strategy Suitability:" in report

    def test_get_detailed_report_excellent(self):
        """Test report for excellent regime quality."""
        data = pd.DataFrame({
            'predicted_state': [0]*40 + [1]*45 + [2]*40,
            'confidence': [0.9] * 125
        })

        metrics = RegimeStabilityMetrics(data)
        report = metrics.get_detailed_report()

        assert "Excellent" in report
        assert "✅ YES" in report  # Trading ready
        assert "Sharpe 10+" in report  # Suitable for high-Sharpe strategies

    def test_get_detailed_report_poor(self):
        """Test report for poor regime quality."""
        data = pd.DataFrame({
            'predicted_state': [i % 3 for i in range(50)],
            'confidence': [0.5 + np.random.randn() * 0.2 for _ in range(50)]
        })

        metrics = RegimeStabilityMetrics(data)
        report = metrics.get_detailed_report()

        assert "Poor" in report
        assert "❌ NO" in report  # Not trading ready
        assert "Not suitable for trading" in report

    def test_get_detailed_report_recommendations(self):
        """Test that report includes actionable recommendations."""
        # Create regime with specific issues
        data = pd.DataFrame({
            'predicted_state': [i % 3 for i in range(60)],  # Too noisy
            'confidence': [0.8] * 60
        })

        metrics = RegimeStabilityMetrics(data)
        report = metrics.get_detailed_report()

        # Should include recommendations for noise
        assert "Recommendations:" in report
        # Should suggest increasing parameters or smoothing
        assert ("increase" in report.lower() or
                "smooth" in report.lower() or
                "filter" in report.lower())


class TestRegimeStabilityEdgeCases:
    """Test edge cases and error handling."""

    def test_single_regime(self):
        """Test with only one regime (no transitions)."""
        data = pd.DataFrame({
            'predicted_state': [0] * 100,
            'confidence': [0.8] * 100
        })

        metrics = RegimeStabilityMetrics(data)
        result = metrics.get_metrics()

        # Should have perfect persistence (no switches)
        assert result['persistence'] == 1.0
        assert result['switching_frequency'] == 0.0
        # Very long duration
        assert result['mean_duration'] == 100.0

    def test_minimum_data(self):
        """Test with minimum viable data."""
        data = pd.DataFrame({
            'predicted_state': [0, 0, 1],
            'confidence': [0.8, 0.8, 0.8]
        })

        metrics = RegimeStabilityMetrics(data)
        result = metrics.get_metrics()

        # Should compute without errors
        assert 'mean_duration' in result
        assert result['mean_duration'] > 0

    def test_nan_handling(self):
        """Test handling of NaN values in confidence."""
        data = pd.DataFrame({
            'predicted_state': [0, 0, 1, 1, 2, 2],
            'confidence': [0.8, np.nan, 0.9, 0.8, np.nan, 0.7]
        })

        metrics = RegimeStabilityMetrics(data)
        result = metrics.get_metrics()

        # Should handle NaNs gracefully
        assert 'mean_duration' in result
        assert not np.isnan(result['stability_score'])


class TestRegimeStabilityRealWorldScenarios:
    """Test with realistic regime detection scenarios."""

    def test_spy_like_regimes(self):
        """Test with SPY-like regime characteristics (target for Sharpe 10+)."""
        # Simulate realistic SPY regimes:
        # - 3 regimes
        # - ~30 day average duration
        # - High confidence in bull/bear, moderate in sideways
        np.random.seed(42)

        regimes = []
        confidences = []

        # Bull regime (40 days, high confidence)
        regimes.extend([0] * 40)
        confidences.extend([0.85 + np.random.randn() * 0.05 for _ in range(40)])

        # Transition to sideways (25 days, medium confidence)
        regimes.extend([1] * 25)
        confidences.extend([0.70 + np.random.randn() * 0.08 for _ in range(25)])

        # Bear regime (20 days, high confidence)
        regimes.extend([2] * 20)
        confidences.extend([0.88 + np.random.randn() * 0.04 for _ in range(20)])

        # Back to bull (35 days)
        regimes.extend([0] * 35)
        confidences.extend([0.82 + np.random.randn() * 0.06 for _ in range(35)])

        data = pd.DataFrame({
            'predicted_state': regimes,
            'confidence': confidences
        })

        metrics = RegimeStabilityMetrics(data)
        result = metrics.get_metrics()

        # Should be tradeable
        assert result['is_tradeable'] is True
        # Duration should be reasonable
        assert 20 <= result['mean_duration'] <= 50
        # Should have good persistence (only 3 transitions)
        assert result['persistence'] > 0.9
        # Quality should be Good or Excellent
        assert result['quality_rating'] in ['Good', 'Excellent']

    def test_crypto_like_regimes(self):
        """Test with crypto-like fast-switching regimes."""
        np.random.seed(42)

        # Crypto: shorter regimes, more volatility
        regimes = []
        for _ in range(20):  # 20 regime episodes
            regime = np.random.randint(0, 3)
            duration = np.random.randint(3, 12)  # 3-12 days
            regimes.extend([regime] * duration)

        data = pd.DataFrame({
            'predicted_state': regimes,
            'confidence': [0.70 + np.random.randn() * 0.15 for _ in range(len(regimes))]
        })

        metrics = RegimeStabilityMetrics(data)
        result = metrics.get_metrics()

        # Crypto regimes are noisier
        assert result['mean_duration'] < 20
        # Lower persistence due to more switching
        assert result['persistence'] < 0.9
        # May or may not be tradeable depending on exact parameters
        # Quality likely Fair or Poor for high-frequency crypto


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
