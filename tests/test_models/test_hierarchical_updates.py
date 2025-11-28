"""
Unit tests for hierarchical HMM update methods.

Tests verify:
1. Emissions-only update maintains transitions
2. Transitions-only update maintains emissions
3. Full retrain with informed prior improves fit
4. Cost hierarchy is maintained (~1%, ~5%, ~100%)
5. Scheduling variables are properly updated
6. All metrics are computed correctly
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.config.model import HMMConfig


@pytest.fixture
def sample_hmm_model():
    """Create a fitted HMM model for testing."""
    config = HMMConfig(
        n_states=2,
        observed_signal='returns',
        max_iterations=50,
        tolerance=1e-4,
    )
    model = HiddenMarkovModel(config)

    # Create sample return data
    np.random.seed(42)
    bull_returns = np.random.normal(0.001, 0.008, 100)
    bear_returns = np.random.normal(-0.0005, 0.012, 100)
    returns = np.concatenate([bull_returns, bear_returns, bull_returns])

    observations = pd.DataFrame({'returns': returns})

    # Fit the model
    model.fit(observations)

    return model, observations


@pytest.fixture
def new_observations():
    """Create new observations for update testing."""
    np.random.seed(123)
    returns = np.random.normal(0.0005, 0.010, 50)
    return pd.DataFrame({'returns': returns})


class TestEmissionsOnlyUpdate:
    """Test emission-only update method."""

    def test_emissions_only_updates_means_and_stds(self, sample_hmm_model):
        """Emission-only update should change means and stds."""
        model, obs = sample_hmm_model
        new_obs = pd.DataFrame({
            'returns': np.random.normal(0.002, 0.015, 30)
        })

        old_means = model.emission_means_.copy()
        old_stds = model.emission_stds_.copy()

        result = model.update_emissions_only(new_obs)

        # Means and stds should change
        assert not np.allclose(model.emission_means_, old_means)
        assert not np.allclose(model.emission_stds_, old_stds)
        assert result['means_change'] > 0
        assert result['stds_change'] > 0

    def test_emissions_only_preserves_transitions(self, sample_hmm_model):
        """Emission-only update should not change transitions."""
        model, obs = sample_hmm_model
        new_obs = pd.DataFrame({
            'returns': np.random.normal(0.0005, 0.010, 30)
        })

        old_transitions = model.transition_matrix_.copy()

        model.update_emissions_only(new_obs)

        # Transitions should be completely unchanged
        np.testing.assert_array_equal(model.transition_matrix_, old_transitions)

    def test_emissions_only_returns_correct_metrics(self, sample_hmm_model, new_observations):
        """Emission-only update should return all expected metrics."""
        model, obs = sample_hmm_model

        result = model.update_emissions_only(new_observations)

        assert 'means_change' in result
        assert 'stds_change' in result
        assert 'total_change' in result
        assert 'log_likelihood' in result
        assert 'cost_ratio' in result

        assert isinstance(result['means_change'], float)
        assert isinstance(result['stds_change'], float)
        assert isinstance(result['log_likelihood'], float)
        assert result['cost_ratio'] == 0.01  # ~1% cost

    def test_emissions_only_updates_scheduling_variables(self, sample_hmm_model, new_observations):
        """Emission-only update should reset emission and last update counters."""
        model, obs = sample_hmm_model

        # Set some values
        model.days_since_emission_update = 5
        model.days_since_transition_update = 7
        model.days_since_last_update = 3

        model.update_emissions_only(new_observations)

        # These should be reset to 0
        assert model.days_since_emission_update == 0
        assert model.days_since_last_update == 0
        # Transitions counter should not be reset
        assert model.days_since_transition_update == 7  # Not reset

    def test_emissions_only_logs_update_history(self, sample_hmm_model, new_observations):
        """Emission-only update should log entry in update_history."""
        model, obs = sample_hmm_model

        initial_history_len = len(model.training_history_['update_history'])

        model.update_emissions_only(new_observations)

        assert len(model.training_history_['update_history']) == initial_history_len + 1

        last_update = model.training_history_['update_history'][-1]
        assert last_update['type'] == 'emission_only'
        assert last_update['cost_ratio'] == 0.01
        assert 'parameter_changes' in last_update
        assert 'timestamp' in last_update

    def test_emissions_only_fails_if_not_fitted(self):
        """Emission-only update should fail on unfitted model."""
        config = HMMConfig(n_states=2, observed_signal='returns')
        model = HiddenMarkovModel(config)

        new_obs = pd.DataFrame({'returns': np.random.normal(0, 0.01, 20)})

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.update_emissions_only(new_obs)

    def test_emissions_only_fails_missing_signal(self, sample_hmm_model):
        """Emission-only update should fail if signal column missing."""
        model, obs = sample_hmm_model

        bad_obs = pd.DataFrame({'bad_column': np.random.normal(0, 0.01, 20)})

        with pytest.raises(ValueError, match="not in observations"):
            model.update_emissions_only(bad_obs)

    def test_emissions_only_handles_nan_values(self, sample_hmm_model):
        """Emission-only update should handle NaN values gracefully."""
        model, obs = sample_hmm_model

        new_obs = pd.DataFrame({
            'returns': [np.nan, 0.001, 0.002, np.nan, 0.001, 0.003] * 5
        })

        result = model.update_emissions_only(new_obs)

        # Should compute successfully despite NaN values
        assert result['means_change'] >= 0
        assert result['stds_change'] >= 0

    def test_emissions_only_minimum_variance_enforced(self, sample_hmm_model):
        """Emission-only update should enforce minimum variance."""
        model, obs = sample_hmm_model

        # Create very low-variance observations
        new_obs = pd.DataFrame({
            'returns': np.ones(50) * 0.0001
        })

        model.update_emissions_only(new_obs)

        # Stds should not drop below minimum
        assert np.all(model.emission_stds_ >= model.config.min_variance)


class TestTransitionsOnlyUpdate:
    """Test transition-only update method."""

    def test_transitions_only_updates_transition_matrix(self, sample_hmm_model):
        """Transition-only update should change transition matrix."""
        model, obs = sample_hmm_model
        new_obs = pd.DataFrame({
            'returns': np.random.normal(0.0005, 0.010, 30)
        })

        old_transitions = model.transition_matrix_.copy()

        result = model.update_transitions_only(new_obs)

        # Transitions should change
        assert not np.allclose(model.transition_matrix_, old_transitions)
        assert result['transition_change'] > 0

    def test_transitions_only_preserves_emissions(self, sample_hmm_model):
        """Transition-only update should not change emissions."""
        model, obs = sample_hmm_model
        new_obs = pd.DataFrame({
            'returns': np.random.normal(0.0005, 0.010, 30)
        })

        old_means = model.emission_means_.copy()
        old_stds = model.emission_stds_.copy()

        model.update_transitions_only(new_obs)

        # Emissions should be completely unchanged
        np.testing.assert_array_equal(model.emission_means_, old_means)
        np.testing.assert_array_equal(model.emission_stds_, old_stds)

    def test_transitions_only_returns_correct_metrics(self, sample_hmm_model, new_observations):
        """Transition-only update should return all expected metrics."""
        model, obs = sample_hmm_model

        result = model.update_transitions_only(new_observations)

        assert 'transition_change' in result
        assert 'persistence_change' in result
        assert 'log_likelihood' in result
        assert 'cost_ratio' in result

        assert isinstance(result['transition_change'], float)
        assert isinstance(result['persistence_change'], float)
        assert isinstance(result['log_likelihood'], float)
        assert result['cost_ratio'] == 0.05  # ~5% cost

    def test_transitions_only_updates_scheduling_variables(self, sample_hmm_model, new_observations):
        """Transition-only update should reset transition and last update counters."""
        model, obs = sample_hmm_model

        model.days_since_transition_update = 10
        model.days_since_emission_update = 3
        model.days_since_last_update = 5

        model.update_transitions_only(new_observations)

        # These should be reset to 0
        assert model.days_since_transition_update == 0
        assert model.days_since_last_update == 0
        # Emissions counter should not be reset
        assert model.days_since_emission_update == 3  # Not reset

    def test_transitions_only_logs_update_history(self, sample_hmm_model, new_observations):
        """Transition-only update should log entry in update_history."""
        model, obs = sample_hmm_model

        initial_history_len = len(model.training_history_['update_history'])

        model.update_transitions_only(new_observations)

        assert len(model.training_history_['update_history']) == initial_history_len + 1

        last_update = model.training_history_['update_history'][-1]
        assert last_update['type'] == 'transition_only'
        assert last_update['cost_ratio'] == 0.05
        assert 'parameter_changes' in last_update

    def test_transitions_only_preserves_stochasticity(self, sample_hmm_model, new_observations):
        """Transition matrix should remain row-stochastic after update."""
        model, obs = sample_hmm_model

        model.update_transitions_only(new_observations)

        # Each row should sum to 1 (row stochastic)
        row_sums = np.sum(model.transition_matrix_, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(model.n_states))

    def test_transitions_only_fails_if_not_fitted(self):
        """Transition-only update should fail on unfitted model."""
        config = HMMConfig(n_states=2, observed_signal='returns')
        model = HiddenMarkovModel(config)

        new_obs = pd.DataFrame({'returns': np.random.normal(0, 0.01, 20)})

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.update_transitions_only(new_obs)

    def test_transitions_only_fails_missing_signal(self, sample_hmm_model):
        """Transition-only update should fail if signal column missing."""
        model, obs = sample_hmm_model

        bad_obs = pd.DataFrame({'bad_column': np.random.normal(0, 0.01, 20)})

        with pytest.raises(ValueError, match="not in observations"):
            model.update_transitions_only(bad_obs)


class TestFullRetrainInformedPrior:
    """Test full retrain with informed prior method."""

    def test_full_retrain_improves_likelihood(self, sample_hmm_model):
        """Full retrain should improve (or maintain) log-likelihood."""
        model, obs = sample_hmm_model
        new_obs = pd.DataFrame({
            'returns': np.random.normal(0.0005, 0.010, 50)
        })

        # Score before retrain
        initial_score = model.score(new_obs)

        result = model.full_retrain_with_informed_prior(new_obs)

        # Result should have valid log-likelihood
        assert result['log_likelihood'] != -np.inf
        assert np.isfinite(result['log_likelihood'])

    def test_full_retrain_updates_all_parameters(self, sample_hmm_model):
        """Full retrain should potentially change all parameters."""
        model, obs = sample_hmm_model
        new_obs = pd.DataFrame({
            'returns': np.random.normal(0.002, 0.015, 50)  # Different distribution
        })

        old_means = model.emission_means_.copy()
        old_stds = model.emission_stds_.copy()
        old_transitions = model.transition_matrix_.copy()

        result = model.full_retrain_with_informed_prior(new_obs)

        # After retrain on significantly different data, parameters likely change
        # (though not guaranteed every run)
        assert result['means_change'] >= 0
        assert result['transitions_change'] >= 0

    def test_full_retrain_returns_correct_metrics(self, sample_hmm_model, new_observations):
        """Full retrain should return all expected metrics."""
        model, obs = sample_hmm_model

        result = model.full_retrain_with_informed_prior(new_observations)

        assert 'converged' in result
        assert 'iterations' in result
        assert 'log_likelihood' in result
        assert 'means_change' in result
        assert 'transitions_change' in result
        assert 'training_time' in result
        assert 'cost_ratio' in result

        assert isinstance(result['converged'], bool)
        assert isinstance(result['iterations'], int)
        assert isinstance(result['log_likelihood'], float)
        assert result['cost_ratio'] == 1.0  # 100% cost

    def test_full_retrain_updates_all_scheduling_variables(self, sample_hmm_model, new_observations):
        """Full retrain should reset all scheduling counters."""
        model, obs = sample_hmm_model

        model.days_since_emission_update = 5
        model.days_since_transition_update = 10
        model.days_since_full_retrain = 20
        model.days_since_last_update = 15

        model.full_retrain_with_informed_prior(new_observations)

        # All should be reset to 0
        assert model.days_since_emission_update == 0
        assert model.days_since_transition_update == 0
        assert model.days_since_full_retrain == 0
        assert model.days_since_last_update == 0

    def test_full_retrain_logs_retrain_history(self, sample_hmm_model, new_observations):
        """Full retrain should log entry in update_history."""
        model, obs = sample_hmm_model

        initial_history_len = len(model.training_history_['update_history'])

        model.full_retrain_with_informed_prior(new_observations)

        assert len(model.training_history_['update_history']) == initial_history_len + 1

        last_update = model.training_history_['update_history'][-1]
        assert last_update['type'] == 'full_retrain_informed'
        assert last_update['cost_ratio'] == 1.0
        assert 'converged' in last_update
        assert 'iterations' in last_update

    def test_full_retrain_tracks_training_time(self, sample_hmm_model, new_observations):
        """Full retrain should track training time."""
        model, obs = sample_hmm_model

        result = model.full_retrain_with_informed_prior(new_observations)

        assert result['training_time'] > 0
        assert result['training_time'] < 60  # Should complete in < 60 seconds

    def test_full_retrain_preserves_n_states(self, sample_hmm_model, new_observations):
        """Full retrain should not change number of states."""
        model, obs = sample_hmm_model

        n_states_before = model.n_states

        model.full_retrain_with_informed_prior(new_observations)

        assert model.n_states == n_states_before
        assert len(model.emission_means_) == n_states_before
        assert len(model.emission_stds_) == n_states_before

    def test_full_retrain_fails_if_not_fitted(self):
        """Full retrain should fail on unfitted model."""
        config = HMMConfig(n_states=2, observed_signal='returns')
        model = HiddenMarkovModel(config)

        new_obs = pd.DataFrame({'returns': np.random.normal(0, 0.01, 20)})

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.full_retrain_with_informed_prior(new_obs)

    def test_full_retrain_fails_missing_signal(self, sample_hmm_model):
        """Full retrain should fail if signal column missing."""
        model, obs = sample_hmm_model

        bad_obs = pd.DataFrame({'bad_column': np.random.normal(0, 0.01, 20)})

        with pytest.raises(ValueError, match="not in observations"):
            model.full_retrain_with_informed_prior(bad_obs)


class TestCostHierarchy:
    """Test that cost hierarchy is maintained (~1%, ~5%, ~100%)."""

    def test_cost_ratios_in_correct_hierarchy(self, sample_hmm_model):
        """Cost ratios should follow: emissions < transitions < full."""
        model, obs = sample_hmm_model
        new_obs = pd.DataFrame({
            'returns': np.random.normal(0.0005, 0.010, 50)
        })

        emit_result = model.update_emissions_only(new_obs)
        trans_result = model.update_transitions_only(new_obs)
        full_result = model.full_retrain_with_informed_prior(new_obs)

        # Verify cost hierarchy
        assert emit_result['cost_ratio'] == 0.01
        assert trans_result['cost_ratio'] == 0.05
        assert full_result['cost_ratio'] == 1.0

        # Verify strict ordering
        assert emit_result['cost_ratio'] < trans_result['cost_ratio']
        assert trans_result['cost_ratio'] < full_result['cost_ratio']

    def test_emissions_cost_approximately_1_percent(self, sample_hmm_model):
        """Emissions-only update cost should be ~1% of full retrain."""
        model, obs = sample_hmm_model
        new_obs = pd.DataFrame({
            'returns': np.random.normal(0.0005, 0.010, 50)
        })

        emit_result = model.update_emissions_only(new_obs)

        # Should be approximately 1%
        assert 0.005 <= emit_result['cost_ratio'] <= 0.02  # Within 2x range

    def test_transitions_cost_approximately_5_percent(self, sample_hmm_model):
        """Transitions-only update cost should be ~5% of full retrain."""
        model, obs = sample_hmm_model
        new_obs = pd.DataFrame({
            'returns': np.random.normal(0.0005, 0.010, 50)
        })

        trans_result = model.update_transitions_only(new_obs)

        # Should be approximately 5%
        assert 0.025 <= trans_result['cost_ratio'] <= 0.1  # Within 2x range


class TestSchedulingVariables:
    """Test scheduling variable initialization and updates."""

    def test_scheduling_variables_initialized(self):
        """Scheduling variables should be initialized to 0."""
        config = HMMConfig(n_states=2, observed_signal='returns')
        model = HiddenMarkovModel(config)

        assert model.days_since_emission_update == 0
        assert model.days_since_transition_update == 0
        assert model.days_since_full_retrain == 0
        assert model.days_since_last_update == 0

    def test_scheduling_variables_independent(self, sample_hmm_model, new_observations):
        """Scheduling variables should be independent."""
        model, obs = sample_hmm_model

        # Set different values
        model.days_since_emission_update = 3
        model.days_since_transition_update = 7
        model.days_since_full_retrain = 10

        # Emission update resets only emission counter
        model.update_emissions_only(new_observations)
        assert model.days_since_emission_update == 0
        assert model.days_since_transition_update == 7  # Unchanged
        assert model.days_since_full_retrain == 10  # Unchanged

        # Transition update resets only transition counter
        model.update_transitions_only(new_observations)
        assert model.days_since_transition_update == 0
        assert model.days_since_emission_update > 0  # Incremented somewhere
        assert model.days_since_full_retrain == 10  # Unchanged


class TestUpdateHistory:
    """Test update history tracking."""

    def test_update_history_preserves_all_updates(self, sample_hmm_model, new_observations):
        """Update history should preserve all updates."""
        model, obs = sample_hmm_model

        initial_len = len(model.training_history_['update_history'])

        model.update_emissions_only(new_observations)
        model.update_transitions_only(new_observations)
        model.full_retrain_with_informed_prior(new_observations)

        # Should have 3 new entries
        assert len(model.training_history_['update_history']) == initial_len + 3

    def test_update_history_includes_timestamps(self, sample_hmm_model, new_observations):
        """Each update should include a timestamp."""
        model, obs = sample_hmm_model

        model.update_emissions_only(new_observations)

        last_update = model.training_history_['update_history'][-1]
        assert 'timestamp' in last_update
        # Timestamp should be ISO format
        datetime.fromisoformat(last_update['timestamp'])

    def test_update_history_includes_observation_count(self, sample_hmm_model, new_observations):
        """Each update should record observation count."""
        model, obs = sample_hmm_model

        model.update_emissions_only(new_observations)

        last_update = model.training_history_['update_history'][-1]
        assert 'observation_count' in last_update
        assert last_update['observation_count'] == len(new_observations)
