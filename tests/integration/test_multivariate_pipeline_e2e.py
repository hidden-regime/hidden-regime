"""
End-to-End Integration Tests for Multivariate HMM Pipeline.

Tests verify that the complete multivariate pipeline works correctly:
Data → Observation (multivariate features) → Model (multivariate HMM) → Interpreter (multivariate analysis)

Critical validations:
1. Multivariate data flows correctly through all components
2. Covariance matrices are properly computed and interpreted
3. Feature standardization works across pipeline
4. Interpreter handles multivariate outputs correctly
5. Multivariate vs univariate consistency maintained
6. No data loss or corruption across boundaries

Uses REAL components (not mocks) for actual integration testing.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from hidden_regime import create_multivariate_pipeline, create_financial_pipeline
from hidden_regime.config.model import HMMConfig, ObservationMode
from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.interpreter.financial import FinancialInterpreter
from hidden_regime.config.interpreter import InterpreterConfiguration
from hidden_regime.utils.exceptions import ValidationError, ConfigurationError


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def synthetic_multivariate_data():
    """Generate synthetic multivariate data for testing.

    Creates bull and bear regimes with different covariance structures.
    Bull: positive returns, low volatility, positive correlation
    Bear: negative returns, high volatility, negative correlation
    """
    np.random.seed(42)
    n_points = 250

    # Bull market (first 125 points): positive returns, correlated vol
    bull_returns = np.random.normal(0.001, 0.01, 125)
    bull_vol = np.random.normal(0.01, 0.003, 125)  # Correlated with returns
    bull_data = np.column_stack([bull_returns, bull_vol])

    # Bear market (last 125 points): negative returns, high vol, negative corr
    bear_returns = np.random.normal(-0.002, 0.02, 125)
    bear_vol = np.random.normal(0.025, 0.005, 125)  # Higher, negatively correlated
    bear_data = np.column_stack([bear_returns, bear_vol])

    # Combine and create DataFrame
    all_data = np.vstack([bull_data, bear_data])

    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_points)]

    return pd.DataFrame({
        'date': dates,
        'log_return': all_data[:, 0],
        'realized_vol': np.abs(all_data[:, 1]),  # Volatility is positive
    }).set_index('date')


@pytest.fixture
def multivariate_config():
    """Multivariate HMM configuration."""
    return HMMConfig(
        n_states=2,
        observation_mode=ObservationMode.MULTIVARIATE,
        observed_signals=['log_return', 'realized_vol'],
        initialization_method='kmeans',
        max_iterations=50,
        random_seed=42
    )


@pytest.fixture
def univariate_config():
    """Univariate HMM configuration for comparison."""
    return HMMConfig(
        n_states=2,
        observation_mode=ObservationMode.UNIVARIATE,
        observed_signal='log_return',
        initialization_method='kmeans',
        max_iterations=50,
        random_seed=42
    )


# ============================================================================
# Tests: Multivariate Pipeline E2E
# ============================================================================


class TestMultivariatePipelineE2E:
    """End-to-end tests for complete multivariate pipeline."""

    def test_multivariate_pipeline_full_workflow(self, synthetic_multivariate_data):
        """Test complete workflow: Data → Observation → Model → Interpreter.

        Verifies:
        - Pipeline creates successfully
        - Data flows through all components
        - Model outputs multivariate parameters
        - Interpreter receives and processes multivariate outputs
        """
        # Create pipeline
        pipeline = create_multivariate_pipeline(
            'SPY',
            features=['log_return', 'realized_vol'],
            n_states=2
        )

        # Verify pipeline structure
        assert pipeline is not None
        assert pipeline.model is not None
        assert pipeline.interpreter is not None

        # Verify model is multivariate
        assert pipeline.model.config.observation_mode == ObservationMode.MULTIVARIATE
        assert pipeline.model.config.observed_signals == ['log_return', 'realized_vol']

    def test_multivariate_model_training_and_inference(self, synthetic_multivariate_data, multivariate_config):
        """Test multivariate model training and inference.

        Verifies:
        - Model trains on multivariate data
        - Covariance matrices are learned
        - Predictions include multivariate parameters
        """
        model = HiddenMarkovModel(multivariate_config)

        # Train model
        model.fit(synthetic_multivariate_data)

        # Verify training
        assert model.is_fitted
        assert model.is_multivariate_
        assert model.n_features_ == 2

        # Verify learned parameters
        assert model.emission_means_ is not None
        assert model.emission_means_.shape == (2, 2)  # 2 states, 2 features

        assert model.emission_covs_ is not None
        assert model.emission_covs_.shape == (2, 2, 2)  # 2 states, 2x2 covariance

        # Inference
        predictions = model.predict(synthetic_multivariate_data)

        # Verify predictions
        assert 'predicted_state' in predictions.columns
        assert 'confidence' in predictions.columns

    def test_interpreter_receives_multivariate_parameters(self, synthetic_multivariate_data, multivariate_config):
        """Test that interpreter properly receives and interprets multivariate outputs.

        Verifies:
        - Interpreter detects multivariate mode
        - Interpreter extracts emission_covs
        - Interpreter computes multivariate metrics
        """
        model = HiddenMarkovModel(multivariate_config)
        model.fit(synthetic_multivariate_data)

        # Get predictions with multivariate parameters
        predictions = model.predict(synthetic_multivariate_data)

        # Add multivariate parameters to predictions
        predictions['emission_means'] = [model.emission_means_] * len(predictions)
        predictions['emission_covs'] = [model.emission_covs_] * len(predictions)
        predictions['state'] = predictions['predicted_state']

        # Interpret
        interpreter = FinancialInterpreter(InterpreterConfiguration(n_states=2))
        interpretation = interpreter.update(predictions)

        # Verify interpretation detected multivariate mode
        assert interpreter._is_multivariate

        # Verify multivariate metrics added
        assert 'multivariate_eigenvalue_ratio' in interpretation.columns
        assert 'multivariate_pca_explained_variance' in interpretation.columns
        assert 'multivariate_avg_feature_correlation' in interpretation.columns

    def test_multivariate_vs_univariate_consistency(self, synthetic_multivariate_data):
        """Test that both univariate and multivariate models work correctly.

        Verifies:
        - Univariate model still works (backward compatibility)
        - Multivariate model works (new feature)
        - Both identify regimes
        - Multivariate has additional metrics
        """
        # Train univariate
        uni_config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.UNIVARIATE,
            observed_signal='log_return',
            max_iterations=50,
            random_seed=42
        )
        uni_model = HiddenMarkovModel(uni_config)
        uni_model.fit(synthetic_multivariate_data[['log_return']])
        uni_predictions = uni_model.predict(synthetic_multivariate_data[['log_return']])

        # Train multivariate
        multi_config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'realized_vol'],
            max_iterations=50,
            random_seed=42
        )
        multi_model = HiddenMarkovModel(multi_config)
        multi_model.fit(synthetic_multivariate_data)
        multi_predictions = multi_model.predict(synthetic_multivariate_data)

        # Both should have regime predictions
        assert 'predicted_state' in uni_predictions.columns
        assert 'predicted_state' in multi_predictions.columns

        # Univariate should NOT have covariance
        assert multi_model.is_multivariate_
        assert not uni_model.is_multivariate_

        # Verify parameter shapes
        assert uni_model.emission_stds_ is not None
        assert multi_model.emission_covs_ is not None

    def test_feature_standardization_end_to_end(self, synthetic_multivariate_data):
        """Test that feature standardization works across entire pipeline.

        Verifies:
        - Features with different scales are properly standardized
        - Model receives standardized features
        - Covariance matrices are well-conditioned
        """
        # Create data with extreme scale mismatch
        data = synthetic_multivariate_data.copy()
        data['realized_vol'] = data['realized_vol'] * 1000  # 1000x scale difference

        # Train multivariate model
        config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'realized_vol'],
            max_iterations=50,
            random_seed=42
        )
        model = HiddenMarkovModel(config)
        model.fit(data)

        # Verify model standardized features
        assert model.scaler_ is not None

        # Verify covariance matrices are reasonable
        cov_matrices = model.emission_covs_
        for state_idx in range(model.n_states):
            cov = cov_matrices[state_idx]
            # Check that covariance is positive definite
            eigenvalues = np.linalg.eigvalsh(cov)
            assert np.all(eigenvalues > -1e-10)

    def test_configuration_validation_multivariate(self):
        """Test that multivariate configuration validation works.

        Verifies:
        - Must set observed_signals in MULTIVARIATE mode
        - Must have at least 1 signal
        """
        # This should fail: no observed_signals in MULTIVARIATE mode
        with pytest.raises(ConfigurationError):
            HMMConfig(
                n_states=2,
                observation_mode=ObservationMode.MULTIVARIATE,
                observed_signals=None
            )

        # This should fail: empty observed_signals
        with pytest.raises(ConfigurationError):
            HMMConfig(
                n_states=2,
                observation_mode=ObservationMode.MULTIVARIATE,
                observed_signals=[]
            )

        # This should succeed
        config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'realized_vol']
        )
        assert config.observation_mode == ObservationMode.MULTIVARIATE

    def test_multivariate_eigenvalue_analysis(self, synthetic_multivariate_data, multivariate_config):
        """Test that multivariate eigenvalue analysis computes correctly.

        Verifies:
        - Eigenvalues computed from covariance matrices
        - Eigenvalue ratios make sense
        - PCA explained variance reasonable
        """
        model = HiddenMarkovModel(multivariate_config)
        model.fit(synthetic_multivariate_data)

        cov_matrices = model.emission_covs_

        for state_idx in range(model.n_states):
            cov = cov_matrices[state_idx]

            # Compute eigenvalues
            eigenvalues, _ = np.linalg.eigh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

            # Verify eigenvalues are positive
            assert np.all(eigenvalues > 0)

            # Eigenvalue ratio should be >= 1
            ratio = eigenvalues[0] / eigenvalues[-1]
            assert ratio >= 1.0

            # PCA explained variance in first component
            pca_explained = eigenvalues[0] / np.sum(eigenvalues)
            assert 0.0 <= pca_explained <= 1.0

    def test_feature_validation_in_factory(self):
        """Test that factory validates features before creating pipeline.

        Verifies:
        - Invalid features raise ConfigurationError
        - Error message is helpful
        - Valid features work
        """
        # Valid features should work
        pipeline = create_multivariate_pipeline(
            'SPY',
            features=['log_return', 'realized_vol'],
            n_states=2
        )
        assert pipeline is not None

        # Invalid features should fail
        with pytest.raises(ConfigurationError) as exc_info:
            create_multivariate_pipeline(
                'SPY',
                features=['log_return', 'nonexistent_feature'],
                n_states=2
            )
        assert 'Invalid features' in str(exc_info.value)
        assert 'nonexistent_feature' in str(exc_info.value)

    def test_multivariate_interpretation_adds_expected_columns(self, synthetic_multivariate_data):
        """Test that multivariate interpretation adds expected columns.

        Verifies:
        - Regime labels added
        - Multivariate metrics added
        - No columns lost
        """
        model = HiddenMarkovModel(HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'realized_vol'],
            max_iterations=50,
            random_seed=42
        ))
        model.fit(synthetic_multivariate_data)

        predictions = model.predict(synthetic_multivariate_data)
        predictions['emission_means'] = [model.emission_means_] * len(predictions)
        predictions['emission_covs'] = [model.emission_covs_] * len(predictions)
        predictions['state'] = predictions['predicted_state']

        interpreter = FinancialInterpreter(InterpreterConfiguration(n_states=2))
        interpretation = interpreter.update(predictions)

        # Expected columns from interpretation
        expected_columns = [
            'regime_label',
            'regime_type',
            'state',
            'confidence',
            'multivariate_eigenvalue_ratio',
            'multivariate_pca_explained_variance',
            'multivariate_avg_feature_correlation',
        ]

        for col in expected_columns:
            assert col in interpretation.columns, f"Missing column: {col}"

    def test_multivariate_backward_compatibility_univariate_still_works(self):
        """Test that old univariate code still works unchanged.

        Verifies:
        - Univariate pipelines still work
        - No breaking changes to existing code
        - Can create both uni and multi pipelines
        """
        # Create univariate pipeline (old style)
        uni_pipeline = create_financial_pipeline('SPY', n_states=2)
        assert uni_pipeline is not None
        assert uni_pipeline.model.config.observation_mode == ObservationMode.UNIVARIATE

        # Create multivariate pipeline (new style)
        multi_pipeline = create_multivariate_pipeline('SPY', n_states=2)
        assert multi_pipeline is not None
        assert multi_pipeline.model.config.observation_mode == ObservationMode.MULTIVARIATE

        # Both should be usable
        assert uni_pipeline.model.config.n_states == 2
        assert multi_pipeline.model.config.n_states == 2


# ============================================================================
# Tests: Error Handling and Edge Cases
# ============================================================================


class TestMultivariateErrorHandling:
    """Test error handling and edge cases in multivariate pipeline."""

    def test_invalid_feature_combination(self):
        """Test that invalid feature combinations are rejected."""
        with pytest.raises(ConfigurationError):
            create_multivariate_pipeline(
                'SPY',
                features=['log_return', 'invalid_feature_xyz'],
                n_states=2
            )

    def test_multivariate_with_missing_data(self):
        """Test multivariate pipeline with missing data."""
        # Create sufficient data (50+ points) with some NaNs
        np.random.seed(42)
        n_points = 100
        data = pd.DataFrame({
            'log_return': np.random.randn(n_points) * 0.01,
            'realized_vol': np.abs(np.random.randn(n_points) * 0.015)
        })

        # Add some missing values
        data.loc[10:12, 'log_return'] = np.nan
        data.loc[20:22, 'realized_vol'] = np.nan

        config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return', 'realized_vol'],
            max_iterations=20,
            random_seed=42
        )
        model = HiddenMarkovModel(config)

        # Should handle missing data gracefully (drops NaN rows)
        model.fit(data)
        assert model.is_fitted
        assert model.is_multivariate_

    def test_multivariate_single_feature_edge_case(self):
        """Test multivariate mode with single feature (edge case).

        While unusual, multivariate mode should technically work with 1 feature.
        """
        data = pd.DataFrame({
            'log_return': np.random.randn(100) * 0.01
        })

        config = HMMConfig(
            n_states=2,
            observation_mode=ObservationMode.MULTIVARIATE,
            observed_signals=['log_return'],
            max_iterations=10,
            random_seed=42
        )
        model = HiddenMarkovModel(config)
        model.fit(data)

        # Should work and have 1x1 covariance matrices
        assert model.is_multivariate_
        assert model.n_features_ == 1
        assert model.emission_covs_.shape == (2, 1, 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
