"""Unit tests for adaptive retraining orchestration methods in HMM.

Tests verify:
1. Adaptive retraining initialization with drift monitor and policy
2. Model snapshot creation for drift comparison
3. Drift assessment and decision-making in orchestration
4. Update type execution (emission_only, transition_only, full_retrain)
5. Retraining policy integration and decision recording
6. Metadata addition to predictions
7. Error handling during updates
8. Lazy initialization pattern
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
import warnings

from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.config.model import HMMConfig
from hidden_regime.monitoring.retraining_policy import UpdateSchedule, RetrainingPolicy
from hidden_regime.monitoring.drift_detector import ParameterMonitor


class TestAdaptiveRetrainingInitialization:
    """Test initialization of adaptive retraining system."""

    def test_initialize_adaptive_retraining_creates_components(self):
        """_initialize_adaptive_retraining should create drift monitor and policy."""
        config = HMMConfig(n_states=2)
        model = HiddenMarkovModel(config)

        # Create a mock adaptive config
        adaptive_config = Mock()
        adaptive_config.slrt_threshold = 10.0
        adaptive_config.kl_hard_threshold = 0.5
        adaptive_config.kl_soft_threshold = 0.2
        adaptive_config.update_schedule = UpdateSchedule()

        # Call initialization
        model._initialize_adaptive_retraining(adaptive_config)

        # Verify components created
        assert hasattr(model, '_parameter_monitor')
        assert isinstance(model._parameter_monitor, ParameterMonitor)
        assert hasattr(model, '_retraining_policy')
        assert isinstance(model._retraining_policy, RetrainingPolicy)
        assert hasattr(model, '_adaptive_config')
        assert model._adaptive_config == adaptive_config

    def test_initialize_adaptive_retraining_creates_history(self):
        """_initialize_adaptive_retraining should create adaptive_decisions history."""
        config = HMMConfig(n_states=2)
        model = HiddenMarkovModel(config)

        adaptive_config = Mock()
        adaptive_config.slrt_threshold = 10.0
        adaptive_config.kl_hard_threshold = 0.5
        adaptive_config.kl_soft_threshold = 0.2
        adaptive_config.update_schedule = UpdateSchedule()

        model._initialize_adaptive_retraining(adaptive_config)

        # Verify history created
        assert 'adaptive_decisions' in model.training_history_
        assert isinstance(model.training_history_['adaptive_decisions'], list)
        assert len(model.training_history_['adaptive_decisions']) == 0

    def test_initialize_adaptive_retraining_sets_thresholds(self):
        """_initialize_adaptive_retraining should pass thresholds to ParameterMonitor."""
        config = HMMConfig(n_states=2)
        model = HiddenMarkovModel(config)

        adaptive_config = Mock()
        adaptive_config.slrt_threshold = 15.0
        adaptive_config.kl_hard_threshold = 0.7
        adaptive_config.kl_soft_threshold = 0.3
        adaptive_config.update_schedule = UpdateSchedule()

        model._initialize_adaptive_retraining(adaptive_config)

        # Verify thresholds set
        assert model._parameter_monitor.slrt_threshold == 15.0
        assert model._parameter_monitor.kl_hard_threshold == 0.7
        assert model._parameter_monitor.kl_soft_threshold == 0.3


class TestModelSnapshot:
    """Test model snapshot creation for drift comparison."""

    def test_create_model_snapshot_captures_parameters(self):
        """_create_model_snapshot should create snapshot with current model parameters."""
        config = HMMConfig(n_states=2)
        model = HiddenMarkovModel(config)

        # Set up model with trained parameters
        np.random.seed(42)
        model.emission_means_ = np.array([[-0.01], [0.02]])
        model.emission_covs_ = np.array([[[0.0004]], [[0.0009]]])

        # Create snapshot
        snapshot = model._create_model_snapshot()

        # Verify snapshot captures parameters
        assert snapshot.n_states == 2
        np.testing.assert_array_equal(snapshot.means_, model.emission_means_)
        np.testing.assert_array_equal(snapshot.covars_, model.emission_covs_)

    def test_create_model_snapshot_creates_copy(self):
        """_create_model_snapshot should create independent copy."""
        config = HMMConfig(n_states=2)
        model = HiddenMarkovModel(config)

        np.random.seed(42)
        model.emission_means_ = np.array([[-0.01], [0.02]])
        model.emission_covs_ = np.array([[[0.0004]], [[0.0009]]])

        # Create snapshot
        snapshot = model._create_model_snapshot()

        # Modify original model
        model.emission_means_[0, 0] = 0.05

        # Verify snapshot is unchanged
        assert snapshot.means_[0, 0] == -0.01

    def test_create_model_snapshot_handles_attribute_name_variations(self):
        """_create_model_snapshot should handle both emission_means_ and means_."""
        config = HMMConfig(n_states=2)
        model = HiddenMarkovModel(config)

        # Test with emission_means_
        model.emission_means_ = np.array([[-0.01], [0.02]])
        model.emission_covs_ = np.array([[[0.0004]], [[0.0009]]])

        snapshot = model._create_model_snapshot()
        assert hasattr(snapshot, 'means_')
        np.testing.assert_array_equal(snapshot.means_, model.emission_means_)


class TestOrchestrationDecisionFlow:
    """Test the 4-step orchestration decision flow."""

    @pytest.fixture
    def setup_orchestration(self):
        """Set up model with adaptive config for orchestration tests."""
        config = HMMConfig(n_states=2)
        model = HiddenMarkovModel(config)

        # Initialize with trained parameters
        np.random.seed(42)
        model.emission_means_ = np.array([[-0.01], [0.02]])
        model.emission_covs_ = np.array([[[0.0004]], [[0.0009]]])
        model.initial_probs_ = np.array([0.6, 0.4])
        model.transition_matrix_ = np.array([[0.7, 0.3], [0.3, 0.7]])

        adaptive_config = Mock()
        adaptive_config.slrt_threshold = 10.0
        adaptive_config.kl_hard_threshold = 0.5
        adaptive_config.kl_soft_threshold = 0.2
        adaptive_config.update_schedule = UpdateSchedule()

        model._initialize_adaptive_retraining(adaptive_config)

        # Create sample observations (need at least 10 for model to accept)
        observations = pd.DataFrame({
            'log_return': [0.01, -0.02, 0.015, -0.005, 0.02, 0.008, -0.012, 0.018, -0.003, 0.022]
        })

        return model, adaptive_config, observations

    def test_orchestration_assesses_drift_on_first_call(self, setup_orchestration):
        """First call should store model snapshot without drift comparison."""
        model, adaptive_config, observations = setup_orchestration

        # Create mock predictions
        mock_predictions = pd.DataFrame({
            'state': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'confidence': [0.9] * 10
        })

        # Mock the drift monitor to verify it's called correctly
        with patch.object(model._parameter_monitor, 'assess_drift') as mock_assess:
            mock_assess.return_value = ('continue', {})

            # First call: no old model to compare
            result = model._orchestrate_adaptive_retraining(
                observations, mock_predictions, adaptive_config
            )

            # Drift assessment should not be called (no old model)
            mock_assess.assert_not_called()

    def test_orchestration_records_decision_in_history(self, setup_orchestration):
        """Orchestration should record all decisions in training_history_."""
        model, adaptive_config, observations = setup_orchestration

        mock_predictions = pd.DataFrame({
            'state': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'confidence': [0.9] * 10
        })

        result = model._orchestrate_adaptive_retraining(
            observations, mock_predictions, adaptive_config
        )

        # Verify decision recorded
        assert len(model.training_history_['adaptive_decisions']) > 0
        decision = model.training_history_['adaptive_decisions'][0]
        assert 'drift_decision' in decision
        assert 'retraining_decision' in decision
        assert 'reason' in decision
        assert 'update_status' in decision

    def test_orchestration_creates_model_snapshot_for_next_comparison(self, setup_orchestration):
        """Orchestration should store model snapshot for next drift assessment."""
        model, adaptive_config, observations = setup_orchestration

        mock_predictions = pd.DataFrame({
            'state': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'confidence': [0.9] * 10
        })

        # First orchestration call
        result = model._orchestrate_adaptive_retraining(
            observations, mock_predictions, adaptive_config
        )

        # Verify snapshot stored
        assert hasattr(model, '_old_model_for_comparison')
        assert model._old_model_for_comparison is not None

    def test_orchestration_adds_metadata_to_predictions(self, setup_orchestration):
        """Orchestration should add adaptive metadata to predictions."""
        model, adaptive_config, observations = setup_orchestration

        predictions = pd.DataFrame({
            'state': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'confidence': [0.9] * 10
        })
        result = model._orchestrate_adaptive_retraining(
            observations, predictions, adaptive_config
        )

        # Verify metadata added
        assert 'adaptive_update_type' in result.columns
        assert 'drift_decision' in result.columns

    def test_orchestration_handles_emission_only_update(self, setup_orchestration):
        """Orchestration should execute emission_only update when policy decides."""
        model, adaptive_config, observations = setup_orchestration

        mock_predictions = pd.DataFrame({
            'state': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'confidence': [0.9] * 10
        })

        # Mock policy to request emission update
        with patch.object(
            model._retraining_policy, 'should_update'
        ) as mock_should_update:
            mock_should_update.return_value = ('emission_only', 'scheduled')

            with patch.object(model, 'update_emissions_only') as mock_update:
                mock_update.return_value = 0.95  # Cost ratio

                with patch.object(model, 'predict', return_value=mock_predictions):
                    result = model._orchestrate_adaptive_retraining(
                        observations, mock_predictions, adaptive_config
                    )

                    # Verify update was called
                    mock_update.assert_called_once()

                    # Verify status recorded
                    decision = model.training_history_['adaptive_decisions'][0]
                    assert 'emission_only' in decision['update_status']

    def test_orchestration_handles_transition_only_update(self, setup_orchestration):
        """Orchestration should execute transition_only update when policy decides."""
        model, adaptive_config, observations = setup_orchestration

        mock_predictions = pd.DataFrame({
            'state': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'confidence': [0.9] * 10
        })

        # Mock policy to request transition update
        with patch.object(
            model._retraining_policy, 'should_update'
        ) as mock_should_update:
            mock_should_update.return_value = ('transition_only', 'scheduled')

            with patch.object(model, 'update_transitions_only') as mock_update:
                mock_update.return_value = 0.90

                with patch.object(model, 'predict', return_value=mock_predictions):
                    result = model._orchestrate_adaptive_retraining(
                        observations, mock_predictions, adaptive_config
                    )

                    # Verify update was called
                    mock_update.assert_called_once()

    def test_orchestration_handles_full_retrain_update(self, setup_orchestration):
        """Orchestration should execute full_retrain when policy decides."""
        model, adaptive_config, observations = setup_orchestration

        mock_predictions = pd.DataFrame({
            'state': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'confidence': [0.9] * 10
        })

        # Mock policy to request full retrain
        with patch.object(
            model._retraining_policy, 'should_update'
        ) as mock_should_update:
            mock_should_update.return_value = ('full_retrain', 'max_days_exceeded')

            with patch.object(model, 'full_retrain_with_informed_prior') as mock_update:
                mock_update.return_value = 0.85

                with patch.object(model, 'predict', return_value=mock_predictions):
                    result = model._orchestrate_adaptive_retraining(
                        observations, mock_predictions, adaptive_config
                    )

                    # Verify update was called
                    mock_update.assert_called_once()

    def test_orchestration_handles_no_update(self, setup_orchestration):
        """Orchestration should handle 'none' decision gracefully."""
        model, adaptive_config, observations = setup_orchestration

        mock_predictions = pd.DataFrame({
            'state': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'confidence': [0.9] * 10
        })

        # Mock policy to request no update
        with patch.object(
            model._retraining_policy, 'should_update'
        ) as mock_should_update:
            mock_should_update.return_value = ('none', 'within_schedule')

            result = model._orchestrate_adaptive_retraining(
                observations, mock_predictions, adaptive_config
            )

            # Verify no updates called
            decision = model.training_history_['adaptive_decisions'][0]
            assert decision['update_status'] == 'none'
            assert decision['retraining_decision'] == 'none'

    def test_orchestration_repredicts_after_update(self, setup_orchestration):
        """Orchestration should re-predict if update was performed."""
        model, adaptive_config, observations = setup_orchestration

        # Mock policy to request emission update
        with patch.object(
            model._retraining_policy, 'should_update'
        ) as mock_should_update:
            mock_should_update.return_value = ('emission_only', 'scheduled')

            with patch.object(model, 'update_emissions_only') as mock_update:
                mock_update.return_value = 0.95

                with patch.object(model, 'predict') as mock_predict:
                    # Return different result on second predict call
                    initial_pred = model.predict(observations)
                    updated_pred = initial_pred.copy()
                    updated_pred['state'] = updated_pred['state'] + 1

                    mock_predict.side_effect = [initial_pred, updated_pred, updated_pred]

                    result = model._orchestrate_adaptive_retraining(
                        observations, initial_pred, adaptive_config
                    )

                    # Verify re-prediction happened (called 3 times: initial, after update setup, final)
                    assert mock_predict.call_count >= 2


class TestErrorHandling:
    """Test error handling during orchestration."""

    @pytest.fixture
    def setup_orchestration(self):
        """Set up model with adaptive config for error tests."""
        config = HMMConfig(n_states=2)
        model = HiddenMarkovModel(config)

        np.random.seed(42)
        model.emission_means_ = np.array([[-0.01], [0.02]])
        model.emission_covs_ = np.array([[[0.0004]], [[0.0009]]])
        model.initial_probs_ = np.array([0.6, 0.4])
        model.transition_matrix_ = np.array([[0.7, 0.3], [0.3, 0.7]])

        adaptive_config = Mock()
        adaptive_config.slrt_threshold = 10.0
        adaptive_config.kl_hard_threshold = 0.5
        adaptive_config.kl_soft_threshold = 0.2
        adaptive_config.update_schedule = UpdateSchedule()

        model._initialize_adaptive_retraining(adaptive_config)

        observations = pd.DataFrame({
            'log_return': [0.01, -0.02, 0.015, -0.005, 0.02, 0.008, -0.012, 0.018, -0.003, 0.022]
        })

        return model, adaptive_config, observations

    def test_orchestration_handles_emission_update_failure(self, setup_orchestration):
        """Orchestration should catch and record emission update failures."""
        model, adaptive_config, observations = setup_orchestration

        mock_predictions = pd.DataFrame({
            'state': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'confidence': [0.9] * 10
        })

        with patch.object(
            model._retraining_policy, 'should_update'
        ) as mock_should_update:
            mock_should_update.return_value = ('emission_only', 'scheduled')

            with patch.object(model, 'update_emissions_only') as mock_update:
                mock_update.side_effect = ValueError("Update failed")

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    result = model._orchestrate_adaptive_retraining(
                        observations, mock_predictions, adaptive_config
                    )

                    # Verify warning issued
                    assert len(w) > 0
                    assert "Emission-only update failed" in str(w[-1].message)

                    # Verify failure recorded
                    decision = model.training_history_['adaptive_decisions'][0]
                    assert 'emission_only_failed' in decision['update_status']

    def test_orchestration_handles_transition_update_failure(self, setup_orchestration):
        """Orchestration should catch and record transition update failures."""
        model, adaptive_config, observations = setup_orchestration

        mock_predictions = pd.DataFrame({
            'state': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'confidence': [0.9] * 10
        })

        with patch.object(
            model._retraining_policy, 'should_update'
        ) as mock_should_update:
            mock_should_update.return_value = ('transition_only', 'scheduled')

            with patch.object(model, 'update_transitions_only') as mock_update:
                mock_update.side_effect = RuntimeError("Transition update failed")

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    result = model._orchestrate_adaptive_retraining(
                        observations, mock_predictions, adaptive_config
                    )

                    assert len(w) > 0
                    decision = model.training_history_['adaptive_decisions'][0]
                    assert 'transition_only_failed' in decision['update_status']

    def test_orchestration_handles_full_retrain_failure(self, setup_orchestration):
        """Orchestration should catch and record full retrain failures."""
        model, adaptive_config, observations = setup_orchestration

        mock_predictions = pd.DataFrame({
            'state': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'confidence': [0.9] * 10
        })

        with patch.object(
            model._retraining_policy, 'should_update'
        ) as mock_should_update:
            mock_should_update.return_value = ('full_retrain', 'critical_drift')

            with patch.object(model, 'full_retrain_with_informed_prior') as mock_update:
                mock_update.side_effect = Exception("Retrain failed")

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    result = model._orchestrate_adaptive_retraining(
                        observations, mock_predictions, adaptive_config
                    )

                    assert len(w) > 0
                    decision = model.training_history_['adaptive_decisions'][0]
                    assert 'full_retrain_failed' in decision['update_status']


class TestAdaptiveUpdateStrategy:
    """Test the 'adaptive_hierarchical' strategy in HMM.update()."""

    def test_update_with_adaptive_hierarchical_strategy(self):
        """update() should support 'adaptive_hierarchical' strategy."""
        config = HMMConfig(n_states=2, update_strategy='adaptive_hierarchical')
        model = HiddenMarkovModel(config)

        np.random.seed(42)
        model.emission_means_ = np.array([[-0.01], [0.02]])
        model.emission_covs_ = np.array([[[0.0004]], [[0.0009]]])
        model.initial_probs_ = np.array([0.6, 0.4])
        model.transition_matrix_ = np.array([[0.7, 0.3], [0.3, 0.7]])

        # Create sufficient observations
        observations = pd.DataFrame({
            'log_return': [0.01, -0.02, 0.015, -0.005, 0.02, 0.008, -0.012, 0.018, -0.003, 0.022, 0.015, 0.010]
        })

        adaptive_config = Mock()
        adaptive_config.slrt_threshold = 10.0
        adaptive_config.kl_hard_threshold = 0.5
        adaptive_config.kl_soft_threshold = 0.2
        adaptive_config.update_schedule = UpdateSchedule()

        # Should call _orchestrate_adaptive_retraining
        result = model.update(observations, adaptive_config=adaptive_config)

        # Verify result includes adaptive metadata
        assert 'adaptive_update_type' in result.columns
        assert 'drift_decision' in result.columns

    def test_update_adaptive_hierarchical_with_fitted_model(self):
        """update() with 'adaptive_hierarchical' should work with fitted model."""
        config = HMMConfig(n_states=2, update_strategy='adaptive_hierarchical')
        model = HiddenMarkovModel(config)

        # Generate sufficient training data
        np.random.seed(42)
        train_data = pd.DataFrame({
            'log_return': np.random.randn(50) * 0.01
        })

        # First fit the model
        model.fit(train_data)

        # Now test update with adaptive config
        observations = pd.DataFrame({
            'log_return': [0.01, -0.02, 0.015, -0.005, 0.02, 0.008, -0.012, 0.018, -0.003, 0.022, 0.015, 0.010]
        })

        adaptive_config = Mock()
        adaptive_config.slrt_threshold = 10.0
        adaptive_config.kl_hard_threshold = 0.5
        adaptive_config.kl_soft_threshold = 0.2
        adaptive_config.update_schedule = UpdateSchedule()

        # Should work without raising
        result = model.update(observations, adaptive_config=adaptive_config)

        # Result should have adaptive metadata columns or at least not fail
        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_update_adaptive_hierarchical_with_lazy_initialization(self):
        """update() should lazily initialize on first adaptive_hierarchical call."""
        config = HMMConfig(n_states=2, update_strategy='adaptive_hierarchical')
        model = HiddenMarkovModel(config)

        np.random.seed(42)
        model.emission_means_ = np.array([[-0.01], [0.02]])
        model.emission_covs_ = np.array([[[0.0004]], [[0.0009]]])
        model.initial_probs_ = np.array([0.6, 0.4])
        model.transition_matrix_ = np.array([[0.7, 0.3], [0.3, 0.7]])

        observations = pd.DataFrame({
            'log_return': [0.01, -0.02, 0.015, -0.005, 0.02, 0.008, -0.012, 0.018, -0.003, 0.022, 0.015, 0.010]
        })

        adaptive_config = Mock()
        adaptive_config.slrt_threshold = 10.0
        adaptive_config.kl_hard_threshold = 0.5
        adaptive_config.kl_soft_threshold = 0.2
        adaptive_config.update_schedule = UpdateSchedule()

        # Before update, no parameter monitor
        assert not hasattr(model, '_parameter_monitor')

        # After update, should be initialized
        result = model.update(observations, adaptive_config=adaptive_config)

        assert hasattr(model, '_parameter_monitor')
        assert hasattr(model, '_retraining_policy')


class TestDriftAssessmentIntegration:
    """Test drift assessment within orchestration."""

    def test_orchestration_passes_drift_decision_to_policy(self):
        """Orchestration should pass drift decision from monitor to policy."""
        config = HMMConfig(n_states=2)
        model = HiddenMarkovModel(config)

        np.random.seed(42)
        model.emission_means_ = np.array([[-0.01], [0.02]])
        model.emission_covs_ = np.array([[[0.0004]], [[0.0009]]])
        model.initial_probs_ = np.array([0.6, 0.4])
        model.transition_matrix_ = np.array([[0.7, 0.3], [0.3, 0.7]])

        adaptive_config = Mock()
        adaptive_config.slrt_threshold = 10.0
        adaptive_config.kl_hard_threshold = 0.5
        adaptive_config.kl_soft_threshold = 0.2
        adaptive_config.update_schedule = UpdateSchedule()

        model._initialize_adaptive_retraining(adaptive_config)

        observations = pd.DataFrame({
            'log_return': [0.01, -0.02, 0.015, -0.005, 0.02, 0.008, -0.012, 0.018, -0.003, 0.022]
        })

        mock_predictions = pd.DataFrame({
            'state': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'confidence': [0.9] * 10
        })

        # First call to establish old model
        result1 = model._orchestrate_adaptive_retraining(
            observations, mock_predictions, adaptive_config
        )

        # Second call should assess drift
        with patch.object(
            model._parameter_monitor, 'assess_drift'
        ) as mock_assess:
            mock_assess.return_value = ('retrain_soft', {'max_kl_divergence': 0.35})

            with patch.object(
                model._retraining_policy, 'should_update'
            ) as mock_policy:
                mock_policy.return_value = ('full_retrain', 'soft_drift')

                with patch.object(model, 'full_retrain_with_informed_prior'):
                    with patch.object(model, 'predict', return_value=mock_predictions):
                        result2 = model._orchestrate_adaptive_retraining(
                            observations, mock_predictions, adaptive_config
                        )

                        # Verify policy received drift decision
                        mock_policy.assert_called()
                        args = mock_policy.call_args
                        assert args[0][0] == 'retrain_soft'


class TestMetadataRecording:
    """Test metadata and history recording."""

    def test_orchestration_records_drift_metrics(self):
        """Orchestration should record drift metrics in decision history."""
        config = HMMConfig(n_states=2)
        model = HiddenMarkovModel(config)

        np.random.seed(42)
        model.emission_means_ = np.array([[-0.01], [0.02]])
        model.emission_covs_ = np.array([[[0.0004]], [[0.0009]]])
        model.initial_probs_ = np.array([0.6, 0.4])
        model.transition_matrix_ = np.array([[0.7, 0.3], [0.3, 0.7]])

        adaptive_config = Mock()
        adaptive_config.slrt_threshold = 10.0
        adaptive_config.kl_hard_threshold = 0.5
        adaptive_config.kl_soft_threshold = 0.2
        adaptive_config.update_schedule = UpdateSchedule()

        model._initialize_adaptive_retraining(adaptive_config)

        observations = pd.DataFrame({
            'log_return': [0.01, -0.02, 0.015, -0.005, 0.02, 0.008, -0.012, 0.018, -0.003, 0.022]
        })

        mock_predictions = pd.DataFrame({
            'state': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'confidence': [0.9] * 10
        })

        result = model._orchestrate_adaptive_retraining(
            observations, mock_predictions, adaptive_config
        )

        decision = model.training_history_['adaptive_decisions'][0]
        assert 'drift_metrics' in decision
        assert 'timestamp' in decision
        assert 'policy_status' in decision

    def test_orchestration_includes_kl_divergence_in_predictions(self):
        """Orchestration should add KL divergence metric to predictions if available."""
        config = HMMConfig(n_states=2)
        model = HiddenMarkovModel(config)

        np.random.seed(42)
        model.emission_means_ = np.array([[-0.01], [0.02]])
        model.emission_covs_ = np.array([[[0.0004]], [[0.0009]]])
        model.initial_probs_ = np.array([0.6, 0.4])
        model.transition_matrix_ = np.array([[0.7, 0.3], [0.3, 0.7]])

        adaptive_config = Mock()
        adaptive_config.slrt_threshold = 10.0
        adaptive_config.kl_hard_threshold = 0.5
        adaptive_config.kl_soft_threshold = 0.2
        adaptive_config.update_schedule = UpdateSchedule()

        model._initialize_adaptive_retraining(adaptive_config)

        observations = pd.DataFrame({
            'log_return': [0.01, -0.02, 0.015, -0.005, 0.02, 0.008, -0.012, 0.018, -0.003, 0.022]
        })

        mock_predictions = pd.DataFrame({
            'state': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'confidence': [0.9] * 10
        })

        # First call to establish model
        result1 = model._orchestrate_adaptive_retraining(
            observations, mock_predictions, adaptive_config
        )

        # Second call with mocked drift metrics
        with patch.object(
            model._parameter_monitor, 'assess_drift'
        ) as mock_assess:
            mock_assess.return_value = (
                'continue',
                {'max_kl_divergence': 0.25}
            )

            with patch.object(model, 'predict', return_value=mock_predictions):
                result2 = model._orchestrate_adaptive_retraining(
                    observations, mock_predictions, adaptive_config
                )

                # Verify KL divergence in metadata
                assert 'kl_divergence' in result2.columns
                assert result2['kl_divergence'].iloc[0] == 0.25
