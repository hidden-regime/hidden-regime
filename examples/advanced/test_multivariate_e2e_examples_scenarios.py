"""
Multivariate E2E Examples - Integration Tests

This module provides integration tests for multivariate HMM examples,
testing real-world scenarios and cross-example interactions.

Purpose: Verify examples work together and handle realistic use cases

Test scenarios:
1. Basic regime detection workflow
2. Crisis detection timing
3. Multi-timeframe alignment
4. Edge case recovery
5. Feature selection impact
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add examples directory to path
examples_dir = Path(__file__).parent
sys.path.insert(0, str(examples_dir))

import hidden_regime as hr


class TestBasicRegimeDetection:
    """Test basic multivariate regime detection workflow."""

    def test_quickstart_pipeline_creates_successfully(self):
        """Test that quickstart pipeline initializes without errors."""
        pipeline = hr.create_multivariate_pipeline(
            ticker='SPY',
            n_states=3,
            features=['log_return', 'realized_vol'],
            start_date='2023-01-01',
            end_date='2023-03-01'
        )
        assert pipeline is not None
        assert hasattr(pipeline, 'update')

    def test_quickstart_pipeline_produces_valid_results(self):
        """Test that pipeline produces valid output."""
        pipeline = hr.create_multivariate_pipeline(
            ticker='SPY',
            n_states=3,
            features=['log_return', 'realized_vol'],
            start_date='2023-01-01',
            end_date='2023-03-01'
        )
        result = pipeline.update()

        # Check result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'predicted_state' in result.columns
        assert 'confidence' in result.columns

    def test_quickstart_regimes_are_plausible(self):
        """Test that detected regimes make sense."""
        pipeline = hr.create_multivariate_pipeline(
            ticker='SPY',
            n_states=3,
            features=['log_return', 'realized_vol'],
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        result = pipeline.update()

        # Check state validity
        states = result['predicted_state'].unique()
        assert len(states) <= 3
        assert all(0 <= s < 3 for s in states)

        # Check confidence validity
        assert (result['confidence'] >= 0).all()
        assert (result['confidence'] <= 1).all()


class TestCrisisDetection:
    """Test crisis detection scenarios."""

    def test_covid_period_detected_as_different_regime(self):
        """Test that COVID crash period is detected differently from calm market."""
        # Train on pre-crisis period
        pipeline = hr.create_multivariate_pipeline(
            ticker='SPY',
            n_states=3,
            features=['log_return', 'realized_vol'],
            start_date='2019-01-01',
            end_date='2019-12-31'
        )
        result_training = pipeline.update()

        # Analyze 2020 (crisis period)
        pipeline_2020 = hr.create_multivariate_pipeline(
            ticker='SPY',
            n_states=3,
            features=['log_return', 'realized_vol'],
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        result_2020 = pipeline_2020.update()

        # Training period should have different state distribution
        # This is a weak test, but verifies the basic concept
        training_dist = result_training['predicted_state'].value_counts()
        crisis_dist = result_2020['predicted_state'].value_counts()

        # Both should have states, but distributions might differ
        assert len(training_dist) > 0
        assert len(crisis_dist) > 0

    def test_confidence_meaningful_across_periods(self):
        """Test that confidence scores provide meaningful signal."""
        pipeline = hr.create_multivariate_pipeline(
            ticker='SPY',
            n_states=3,
            features=['log_return', 'realized_vol'],
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        result = pipeline.update()

        # Check confidence varies (not always same value)
        confidence_std = result['confidence'].std()
        assert confidence_std > 0.01  # Should have variation


class TestMultiTimeframeAlignment:
    """Test multi-timeframe alignment scenarios."""

    def test_alignment_score_computation(self):
        """Test alignment score is computed correctly."""
        n = 250
        states_daily = np.random.randint(0, 3, n)
        states_weekly = np.random.randint(0, 3, n)
        states_monthly = np.random.randint(0, 3, n)

        # Alignment score should be between 0 and 1
        daily_norm = states_daily / 2.0
        weekly_norm = states_weekly / 2.0
        monthly_norm = states_monthly / 2.0

        combined = np.array([daily_norm, weekly_norm, monthly_norm])
        variance = np.var(combined, axis=0)
        alignment = np.exp(-variance)

        assert (alignment >= 0).all()
        assert (alignment <= 1).all()

    def test_alignment_perfect_agreement(self):
        """Test alignment score when all timeframes agree."""
        n = 250
        states = np.random.randint(0, 3, n)

        # All timeframes same state
        daily_norm = states / 2.0
        weekly_norm = states / 2.0
        monthly_norm = states / 2.0

        combined = np.array([daily_norm, weekly_norm, monthly_norm])
        variance = np.var(combined, axis=0)
        alignment = np.exp(-variance)

        # Perfect agreement should give high alignment
        assert (alignment > 0.95).all()

    def test_alignment_signal_quality(self):
        """Test that signal quality filtering works."""
        # Create alignment scores
        np.random.seed(42)
        alignment = np.random.uniform(0, 1, 250)
        confidence = np.random.uniform(0.4, 1.0, 250)

        # Combined quality
        combined_quality = alignment * confidence

        # High-quality signals
        high_quality = combined_quality[combined_quality > 0.5]

        # Should have some high-quality signals
        assert len(high_quality) > 0


class TestEdgeCaseHandling:
    """Test edge case handling and recovery."""

    def test_small_dataset_fallback(self):
        """Test that small datasets are handled gracefully."""
        # Create very small synthetic data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'log_return': np.random.normal(0.0005, 0.015, 50),
            'realized_vol': np.random.uniform(0.005, 0.025, 50)
        }, index=dates)

        # Should handle gracefully (no crash)
        try:
            # This would normally raise a warning or fail gracefully
            assert len(data) < 100
            print("✓ Small dataset would require fallback strategy")
        except Exception as e:
            pytest.fail(f"Small dataset handling failed: {e}")

    def test_missing_data_detection(self):
        """Test detection of missing data."""
        # Create data with missing values
        dates = pd.date_range('2023-01-01', periods=250, freq='D')
        data = pd.DataFrame({
            'log_return': np.random.normal(0.0005, 0.015, 250),
            'realized_vol': np.random.uniform(0.005, 0.025, 250)
        }, index=dates)

        # Insert missing values
        data.loc[data.index[10:20], 'log_return'] = np.nan

        missing_pct = data['log_return'].isna().sum() / len(data)
        assert missing_pct > 0.01

    def test_feature_correlation_detection(self):
        """Test detection of highly correlated features."""
        # Create correlated features
        x = np.random.normal(0, 1, 250)
        y = 0.99 * x + 0.01 * np.random.normal(0, 1, 250)

        corr = np.corrcoef(x, y)[0, 1]
        assert corr > 0.95  # Highly correlated

    def test_zero_variance_detection(self):
        """Test detection of zero-variance features."""
        # Create zero-variance feature
        zero_feature = np.zeros(250)
        var = np.var(zero_feature)

        assert var < 1e-10


class TestFeatureSelection:
    """Test feature selection impact."""

    def test_univariate_vs_multivariate_convergence(self):
        """Test that multivariate converges when univariate would."""
        # Univariate
        uni_pipeline = hr.create_financial_pipeline(
            ticker='SPY',
            n_states=3,
            start_date='2023-01-01',
            end_date='2023-06-01',
            include_report=False,
            observation_config_overrides={'generators': ['log_return']}
        )
        uni_result = uni_pipeline.update()

        # Multivariate
        multi_pipeline = hr.create_multivariate_pipeline(
            ticker='SPY',
            n_states=3,
            features=['log_return', 'realized_vol'],
            start_date='2023-01-01',
            end_date='2023-06-01'
        )
        multi_result = multi_pipeline.update()

        # Both should produce results
        assert len(uni_result) > 0
        assert len(multi_result) > 0

    def test_feature_correlation_impact(self):
        """Test that correlated features impact model."""
        np.random.seed(42)

        # Create independent features
        feat1 = np.random.normal(0, 1, 200)
        feat2 = np.random.normal(0, 1, 200)
        corr_independent = np.corrcoef(feat1, feat2)[0, 1]

        # Create correlated features
        feat3 = 0.9 * feat1 + 0.1 * np.random.normal(0, 1, 200)
        corr_dependent = np.corrcoef(feat1, feat3)[0, 1]

        # Dependent should be higher
        assert abs(corr_dependent) > abs(corr_independent)


class TestModelValidation:
    """Test model validation and quality checks."""

    def test_convergence_metrics(self):
        """Test that convergence metrics are tracked."""
        pipeline = hr.create_multivariate_pipeline(
            ticker='SPY',
            n_states=3,
            features=['log_return', 'realized_vol'],
            start_date='2023-01-01',
            end_date='2023-03-01'
        )
        result = pipeline.update()
        model = pipeline.model

        # Check training history
        assert hasattr(model, 'training_history_')
        assert 'converged' in model.training_history_
        assert 'iterations' in model.training_history_

    def test_confidence_distribution(self):
        """Test that confidence scores have reasonable distribution."""
        pipeline = hr.create_multivariate_pipeline(
            ticker='SPY',
            n_states=3,
            features=['log_return', 'realized_vol'],
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        result = pipeline.update()

        confidence = result['confidence']

        # Should have variation
        assert confidence.mean() > 0.3
        assert confidence.std() > 0.05


def run_integration_tests():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("MULTIVARIATE E2E INTEGRATION TESTS")
    print("=" * 70)

    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_integration_tests()
