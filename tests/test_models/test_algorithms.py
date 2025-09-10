"""
Unit tests for HMM algorithms.

Tests Forward-Backward, Viterbi, and Baum-Welch algorithms
with focus on numerical stability and mathematical correctness.
"""

import numpy as np
import pytest
from scipy.stats import norm

from hidden_regime.models.algorithms import HMMAlgorithms


class TestHMMAlgorithms:
    """Test cases for HMM algorithms."""

    @pytest.fixture
    def simple_hmm_params(self):
        """Simple 2-state HMM parameters for testing."""
        initial_probs = np.array([0.6, 0.4])
        transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        emission_params = np.array(
            [
                [-0.02, 0.05],  # Bear regime: negative mean, higher volatility
                [0.01, 0.04],  # Bull regime: positive mean, higher volatility
            ]
        )
        return initial_probs, transition_matrix, emission_params

    @pytest.fixture
    def sample_observations(self):
        """Sample observations for testing."""
        np.random.seed(42)
        return np.array([0.01, -0.005, 0.02, -0.01, 0.005, -0.02, 0.015])

    def test_log_emission_probability(self, sample_observations, simple_hmm_params):
        """Test log emission probability calculation."""
        _, _, emission_params = simple_hmm_params
        means = emission_params[:, 0]
        stds = emission_params[:, 1]

        log_probs = HMMAlgorithms.log_emission_probability(
            sample_observations, means, stds
        )

        # Check shape
        assert log_probs.shape == (len(sample_observations), 2)

        # Check that all probabilities are finite
        assert np.all(np.isfinite(log_probs))

        # Verify against scipy.stats.norm
        for t in range(len(sample_observations)):
            for s in range(2):
                expected = norm.logpdf(sample_observations[t], means[s], stds[s])
                assert np.isclose(log_probs[t, s], expected, atol=1e-10)

    def test_forward_algorithm(self, sample_observations, simple_hmm_params):
        """Test forward algorithm implementation."""
        initial_probs, transition_matrix, emission_params = simple_hmm_params

        forward_probs, log_likelihood = HMMAlgorithms.forward_algorithm(
            sample_observations, initial_probs, transition_matrix, emission_params
        )

        # Check shape
        assert forward_probs.shape == (len(sample_observations), 2)

        # Check that all forward probabilities are finite
        assert np.all(np.isfinite(forward_probs))

        # Check that log likelihood is finite
        assert np.isfinite(log_likelihood)
        # Note: log likelihood can be positive for small sample sizes with good fit

        # Check normalization property (sum of exp(forward_probs[-1]) should equal exp(log_likelihood))
        final_probs_sum = np.exp(np.logaddexp.reduce(forward_probs[-1]))
        assert np.isclose(final_probs_sum, np.exp(log_likelihood), rtol=1e-10)

    def test_backward_algorithm(self, sample_observations, simple_hmm_params):
        """Test backward algorithm implementation."""
        _, transition_matrix, emission_params = simple_hmm_params

        backward_probs = HMMAlgorithms.backward_algorithm(
            sample_observations, transition_matrix, emission_params
        )

        # Check shape
        assert backward_probs.shape == (len(sample_observations), 2)

        # Check that all backward probabilities are finite
        assert np.all(np.isfinite(backward_probs))

        # Check that final backward probabilities are zero (log(1) = 0)
        assert np.allclose(backward_probs[-1], 0.0, atol=1e-10)

    def test_viterbi_algorithm(self, sample_observations, simple_hmm_params):
        """Test Viterbi algorithm implementation."""
        initial_probs, transition_matrix, emission_params = simple_hmm_params

        best_path, best_prob = HMMAlgorithms.viterbi_algorithm(
            sample_observations, initial_probs, transition_matrix, emission_params
        )

        # Check path shape and values
        assert len(best_path) == len(sample_observations)
        assert np.all(best_path >= 0)
        assert np.all(best_path < 2)
        assert best_path.dtype == int

        # Check probability
        assert np.isfinite(best_prob)
        # Note: log probability can be positive for small samples with good fit

    def test_forward_backward_algorithm(self, sample_observations, simple_hmm_params):
        """Test combined forward-backward algorithm."""
        initial_probs, transition_matrix, emission_params = simple_hmm_params

        gamma, xi, log_likelihood = HMMAlgorithms.forward_backward_algorithm(
            sample_observations, initial_probs, transition_matrix, emission_params
        )

        # Check gamma (state probabilities)
        assert gamma.shape == (len(sample_observations), 2)
        assert np.all(gamma >= 0)
        assert np.all(gamma <= 1)
        # Each time step should sum to 1
        assert np.allclose(np.sum(gamma, axis=1), 1.0, atol=1e-10)

        # Check xi (transition probabilities)
        assert xi.shape == (len(sample_observations) - 1, 2, 2)
        assert np.all(xi >= 0)
        assert np.all(xi <= 1)
        # Each time step should sum to 1
        for t in range(len(sample_observations) - 1):
            assert np.isclose(np.sum(xi[t]), 1.0, atol=1e-10)

        # Check log likelihood
        assert np.isfinite(log_likelihood)
        # Note: log likelihood can be positive for small samples

    def test_baum_welch_update(self, sample_observations, simple_hmm_params):
        """Test Baum-Welch parameter update."""
        initial_probs, transition_matrix, emission_params = simple_hmm_params

        # Get gamma and xi from forward-backward
        gamma, xi, _ = HMMAlgorithms.forward_backward_algorithm(
            sample_observations, initial_probs, transition_matrix, emission_params
        )

        # Update parameters
        new_initial_probs, new_transition_matrix, new_emission_params = (
            HMMAlgorithms.baum_welch_update(
                sample_observations, gamma, xi, regularization=1e-6
            )
        )

        # Check initial probabilities
        assert len(new_initial_probs) == 2
        assert np.all(new_initial_probs >= 0)
        assert np.isclose(np.sum(new_initial_probs), 1.0, atol=1e-10)

        # Check transition matrix
        assert new_transition_matrix.shape == (2, 2)
        assert np.all(new_transition_matrix >= 0)
        assert np.allclose(np.sum(new_transition_matrix, axis=1), 1.0, atol=1e-10)

        # Check emission parameters
        assert new_emission_params.shape == (2, 2)
        assert np.all(new_emission_params[:, 1] > 0)  # Positive standard deviations
        assert np.all(np.isfinite(new_emission_params))

    def test_compute_likelihood(self, sample_observations, simple_hmm_params):
        """Test likelihood computation."""
        initial_probs, transition_matrix, emission_params = simple_hmm_params

        log_likelihood = HMMAlgorithms.compute_likelihood(
            sample_observations, initial_probs, transition_matrix, emission_params
        )

        assert np.isfinite(log_likelihood)
        # Note: log likelihood can be positive for small samples

        # Should match forward algorithm result
        _, forward_likelihood = HMMAlgorithms.forward_algorithm(
            sample_observations, initial_probs, transition_matrix, emission_params
        )
        assert np.isclose(log_likelihood, forward_likelihood, rtol=1e-10)

    def test_predict_next_state_probs(self, simple_hmm_params):
        """Test next state probability prediction."""
        _, transition_matrix, _ = simple_hmm_params

        current_probs = np.array([0.7, 0.3])
        next_probs = HMMAlgorithms.predict_next_state_probs(
            current_probs, transition_matrix
        )

        assert len(next_probs) == 2
        assert np.all(next_probs >= 0)
        assert np.isclose(np.sum(next_probs), 1.0, atol=1e-10)

        # Manual calculation
        expected = current_probs @ transition_matrix
        assert np.allclose(next_probs, expected, rtol=1e-10)

    def test_decode_states_online(self, simple_hmm_params):
        """Test online state decoding."""
        _, transition_matrix, emission_params = simple_hmm_params

        prev_probs = np.array([0.6, 0.4])
        new_observation = 0.01

        updated_probs = HMMAlgorithms.decode_states_online(
            new_observation, prev_probs, transition_matrix, emission_params
        )

        assert len(updated_probs) == 2
        assert np.all(updated_probs >= 0)
        assert np.isclose(np.sum(updated_probs), 1.0, atol=1e-10)
        assert np.all(np.isfinite(updated_probs))

    def test_decode_states_online_zero_emissions(self, simple_hmm_params):
        """Test online decoding when all emission probabilities are zero."""
        _, transition_matrix, emission_params = simple_hmm_params

        prev_probs = np.array([0.6, 0.4])
        # Extreme observation that gives near-zero probabilities
        extreme_observation = 10.0  # Very unlikely under normal distributions

        updated_probs = HMMAlgorithms.decode_states_online(
            extreme_observation, prev_probs, transition_matrix, emission_params
        )

        # Should fall back to uniform distribution
        assert len(updated_probs) == 2
        assert np.all(updated_probs >= 0)
        assert np.isclose(np.sum(updated_probs), 1.0, atol=1e-10)

    def test_numerical_stability_underflow(self):
        """Test numerical stability with underflow conditions."""
        # Create scenario that might cause underflow
        observations = np.array([-5.0, 5.0, -5.0, 5.0] * 10)  # Extreme values
        initial_probs = np.array([0.5, 0.5])
        transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        emission_params = np.array(
            [
                [0.0, 0.1],  # Narrow distribution around 0
                [0.0, 0.1],  # Narrow distribution around 0
            ]
        )

        # Forward algorithm should handle underflow gracefully
        forward_probs, log_likelihood = HMMAlgorithms.forward_algorithm(
            observations, initial_probs, transition_matrix, emission_params
        )

        assert np.all(np.isfinite(forward_probs))
        assert np.isfinite(log_likelihood)

        # Backward algorithm should also be stable
        backward_probs = HMMAlgorithms.backward_algorithm(
            observations, transition_matrix, emission_params
        )

        assert np.all(np.isfinite(backward_probs))

    def test_consistency_forward_backward_viterbi(
        self, sample_observations, simple_hmm_params
    ):
        """Test consistency between forward-backward and Viterbi algorithms."""
        initial_probs, transition_matrix, emission_params = simple_hmm_params

        # Forward-backward
        gamma, _, fb_log_likelihood = HMMAlgorithms.forward_backward_algorithm(
            sample_observations, initial_probs, transition_matrix, emission_params
        )

        # Viterbi
        best_path, viterbi_log_prob = HMMAlgorithms.viterbi_algorithm(
            sample_observations, initial_probs, transition_matrix, emission_params
        )

        # Viterbi probability should be less than or equal to forward-backward likelihood
        # (since Viterbi finds the single best path, while FB marginalizes over all paths)
        assert (
            viterbi_log_prob <= fb_log_likelihood + 1e-10
        )  # Small tolerance for numerical precision

        # Most likely states from gamma should often match Viterbi path
        fb_states = np.argmax(gamma, axis=1)
        agreement = np.mean(fb_states == best_path)
        # Should agree on at least some states (exact agreement not guaranteed)
        assert agreement > 0.3

    def test_empty_observations(self, simple_hmm_params):
        """Test handling of empty observation sequences."""
        initial_probs, transition_matrix, emission_params = simple_hmm_params
        empty_obs = np.array([])

        # Should handle empty sequences gracefully or raise appropriate error
        with pytest.raises((IndexError, ValueError)):
            HMMAlgorithms.forward_algorithm(
                empty_obs, initial_probs, transition_matrix, emission_params
            )

    def test_single_observation(self, simple_hmm_params):
        """Test algorithms with single observation."""
        initial_probs, transition_matrix, emission_params = simple_hmm_params
        single_obs = np.array([0.01])

        # Forward algorithm
        forward_probs, log_likelihood = HMMAlgorithms.forward_algorithm(
            single_obs, initial_probs, transition_matrix, emission_params
        )

        assert forward_probs.shape == (1, 2)
        assert np.all(np.isfinite(forward_probs))
        assert np.isfinite(log_likelihood)

        # Backward algorithm
        backward_probs = HMMAlgorithms.backward_algorithm(
            single_obs, transition_matrix, emission_params
        )

        assert backward_probs.shape == (1, 2)
        assert np.allclose(backward_probs[0], 0.0)  # Should be log(1) = 0

        # Viterbi algorithm
        best_path, best_prob = HMMAlgorithms.viterbi_algorithm(
            single_obs, initial_probs, transition_matrix, emission_params
        )

        assert len(best_path) == 1
        assert 0 <= best_path[0] < 2
        assert np.isfinite(best_prob)

    def test_parameter_edge_cases(self, sample_observations):
        """Test algorithms with edge case parameters."""
        # Very small transition probabilities
        initial_probs = np.array([1.0, 0.0])  # Deterministic start
        transition_matrix = np.array([[0.999999, 0.000001], [0.000001, 0.999999]])
        emission_params = np.array([[0.0, 0.01], [0.0, 0.01]])

        # Should handle extreme transition probabilities
        forward_probs, log_likelihood = HMMAlgorithms.forward_algorithm(
            sample_observations, initial_probs, transition_matrix, emission_params
        )

        assert np.all(np.isfinite(forward_probs))
        assert np.isfinite(log_likelihood)
