"""
Unit tests for HMM core algorithms (Forward-Backward, Viterbi, Baum-Welch).

These tests use simple 2-state HMMs with hand-calculated expected values
to verify the mathematical correctness of the algorithms.
"""

import pytest
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

from hidden_regime.models.algorithms import HMMAlgorithms


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_2state_hmm():
    """Simple 2-state HMM for testing with clear separation."""
    return {
        'initial_probs': np.array([0.6, 0.4]),
        'transition_matrix': np.array([
            [0.7, 0.3],  # State 0 -> State 0/1
            [0.4, 0.6]   # State 1 -> State 0/1
        ]),
        'emission_params': np.array([
            [-0.01, 0.02],  # State 0: mean=-1%, std=2%
            [0.02, 0.03]    # State 1: mean=2%, std=3%
        ])
    }


@pytest.fixture
def simple_observations():
    """Simple observation sequence for testing."""
    return np.array([-0.015, 0.025, 0.018, -0.012, 0.030])


@pytest.fixture
def known_path_hmm():
    """HMM designed to have a known Viterbi path."""
    return {
        'initial_probs': np.array([0.9, 0.1]),  # Start in state 0
        'transition_matrix': np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ]),
        'emission_params': np.array([
            [-0.05, 0.01],  # State 0: strong negative mean, low std
            [0.05, 0.01]    # State 1: strong positive mean, low std
        ])
    }


# ============================================================================
# Test log_emission_probability
# ============================================================================


@pytest.mark.unit
class TestLogEmissionProbability:
    """Test log emission probability calculations."""

    def test_shape(self):
        """Test output shape is (T, n_states)."""
        observations = np.array([0.01, -0.02, 0.03])
        means = np.array([-0.01, 0.02])
        stds = np.array([0.02, 0.03])

        log_probs = HMMAlgorithms.log_emission_probability(observations, means, stds)

        assert log_probs.shape == (3, 2), "Shape should be (T, n_states)"

    def test_gaussian_pdf_correctness(self):
        """Test that log emission matches scipy norm.logpdf."""
        observations = np.array([0.0, 0.01])
        means = np.array([0.0])
        stds = np.array([0.02])

        log_probs = HMMAlgorithms.log_emission_probability(observations, means, stds)

        # Verify against scipy
        expected_0 = norm.logpdf(0.0, loc=0.0, scale=0.02)
        expected_1 = norm.logpdf(0.01, loc=0.0, scale=0.02)

        np.testing.assert_almost_equal(log_probs[0, 0], expected_0)
        np.testing.assert_almost_equal(log_probs[1, 0], expected_1)

    def test_higher_probability_for_closer_observations(self):
        """Test that observations closer to mean have higher probability."""
        observations = np.array([0.0, 0.1])  # 0.0 is closer to mean
        means = np.array([0.0])
        stds = np.array([0.02])

        log_probs = HMMAlgorithms.log_emission_probability(observations, means, stds)

        # Observation at mean should have higher log probability
        assert log_probs[0, 0] > log_probs[1, 0]

    def test_multiple_states(self):
        """Test emission probabilities for multiple states."""
        observations = np.array([0.01])
        means = np.array([-0.02, 0.02])  # Two states
        stds = np.array([0.01, 0.01])

        log_probs = HMMAlgorithms.log_emission_probability(observations, means, stds)

        # Should be higher probability for state 1 (mean=0.02)
        assert log_probs[0, 1] > log_probs[0, 0]

    def test_all_finite(self):
        """Test that all log probabilities are finite."""
        observations = np.array([0.01, -0.02, 0.03])
        means = np.array([-0.01, 0.02])
        stds = np.array([0.02, 0.03])

        log_probs = HMMAlgorithms.log_emission_probability(observations, means, stds)

        assert np.all(np.isfinite(log_probs)), "Log probabilities should be finite"


# ============================================================================
# Test forward_algorithm
# ============================================================================


@pytest.mark.unit
class TestForwardAlgorithm:
    """Test forward algorithm implementation."""

    def test_shape(self, simple_2state_hmm, simple_observations):
        """Test output shape is (T, n_states)."""
        forward_probs, log_likelihood = HMMAlgorithms.forward_algorithm(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        assert forward_probs.shape == (5, 2), "Shape should be (T, n_states)"

    def test_log_likelihood_is_scalar(self, simple_2state_hmm, simple_observations):
        """Test that log likelihood is a scalar."""
        _, log_likelihood = HMMAlgorithms.forward_algorithm(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        assert isinstance(log_likelihood, (float, np.floating))

    def test_log_likelihood_is_finite(self, simple_2state_hmm, simple_observations):
        """Test that log likelihood is finite."""
        _, log_likelihood = HMMAlgorithms.forward_algorithm(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        assert np.isfinite(log_likelihood), "Log likelihood should be finite"

    def test_initialization_step(self, simple_2state_hmm):
        """Test first timestep matches hand calculation."""
        observations = np.array([0.0])

        forward_probs, _ = HMMAlgorithms.forward_algorithm(
            observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        # At t=0: forward[0, s] = log(initial[s]) + log(emission[0, s])
        log_emissions = HMMAlgorithms.log_emission_probability(
            observations,
            simple_2state_hmm['emission_params'][:, 0],
            simple_2state_hmm['emission_params'][:, 1]
        )
        expected_forward_0 = np.log(simple_2state_hmm['initial_probs'] + 1e-10) + log_emissions[0]

        np.testing.assert_array_almost_equal(forward_probs[0], expected_forward_0)

    def test_single_state(self):
        """Test forward algorithm with single state HMM."""
        observations = np.array([0.01, 0.02])
        initial_probs = np.array([1.0])
        transition_matrix = np.array([[1.0]])
        emission_params = np.array([[0.0, 0.02]])

        forward_probs, log_likelihood = HMMAlgorithms.forward_algorithm(
            observations, initial_probs, transition_matrix, emission_params
        )

        assert forward_probs.shape == (2, 1)
        assert np.isfinite(log_likelihood)

    def test_deterministic_path(self):
        """Test forward algorithm with deterministic transitions."""
        observations = np.array([0.0, 0.0, 0.0])
        initial_probs = np.array([1.0, 0.0])  # Start in state 0
        transition_matrix = np.array([
            [1.0, 0.0],  # Always stay in state 0
            [0.0, 1.0]
        ])
        emission_params = np.array([[0.0, 0.01], [0.0, 0.01]])

        forward_probs, _ = HMMAlgorithms.forward_algorithm(
            observations, initial_probs, transition_matrix, emission_params
        )

        # Since we start in state 0 and always stay there, state 0 should dominate
        # Convert from log space for interpretation
        forward_probs_prob = np.exp(forward_probs)

        # Normalize each timestep to see relative probabilities
        for t in range(len(observations)):
            prob_sum = np.exp(logsumexp(forward_probs[t]))
            normalized = forward_probs_prob[t] / prob_sum
            # State 0 should have much higher probability
            assert normalized[0] > 0.9


# ============================================================================
# Test backward_algorithm
# ============================================================================


@pytest.mark.unit
class TestBackwardAlgorithm:
    """Test backward algorithm implementation."""

    def test_shape(self, simple_2state_hmm, simple_observations):
        """Test output shape is (T, n_states)."""
        backward_probs = HMMAlgorithms.backward_algorithm(
            simple_observations,
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        assert backward_probs.shape == (5, 2), "Shape should be (T, n_states)"

    def test_final_timestep_initialization(self, simple_2state_hmm, simple_observations):
        """Test that final timestep is initialized to log(1) = 0."""
        backward_probs = HMMAlgorithms.backward_algorithm(
            simple_observations,
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        # Final timestep should be 0 (log(1)) for all states
        np.testing.assert_array_almost_equal(backward_probs[-1], [0.0, 0.0])

    def test_single_state(self):
        """Test backward algorithm with single state HMM."""
        observations = np.array([0.01, 0.02])
        transition_matrix = np.array([[1.0]])
        emission_params = np.array([[0.0, 0.02]])

        backward_probs = HMMAlgorithms.backward_algorithm(
            observations, transition_matrix, emission_params
        )

        assert backward_probs.shape == (2, 1)
        assert np.all(np.isfinite(backward_probs))

    def test_backward_probabilities_are_log_values(self, simple_2state_hmm, simple_observations):
        """Test that backward probabilities are in log space (can be positive or negative)."""
        backward_probs = HMMAlgorithms.backward_algorithm(
            simple_observations,
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        # All values should be finite (no inf or nan)
        assert np.all(np.isfinite(backward_probs))


# ============================================================================
# Test viterbi_algorithm
# ============================================================================


@pytest.mark.unit
class TestViterbiAlgorithm:
    """Test Viterbi algorithm implementation."""

    def test_path_length(self, simple_2state_hmm, simple_observations):
        """Test that path length matches observation length."""
        best_path, _ = HMMAlgorithms.viterbi_algorithm(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        assert len(best_path) == len(simple_observations)

    def test_path_contains_valid_states(self, simple_2state_hmm, simple_observations):
        """Test that path only contains valid state indices."""
        best_path, _ = HMMAlgorithms.viterbi_algorithm(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        assert np.all(best_path >= 0)
        assert np.all(best_path < 2)  # 2 states

    def test_best_prob_is_finite(self, simple_2state_hmm, simple_observations):
        """Test that best probability is finite."""
        _, best_prob = HMMAlgorithms.viterbi_algorithm(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        assert isinstance(best_prob, (float, np.floating))
        assert np.isfinite(best_prob), "Log probability should be finite"

    def test_deterministic_path(self, known_path_hmm):
        """Test Viterbi with observations that strongly suggest a known path."""
        # Observations strongly aligned with state means
        observations = np.array([-0.05, -0.04, 0.05, 0.04, 0.05])

        best_path, _ = HMMAlgorithms.viterbi_algorithm(
            observations,
            known_path_hmm['initial_probs'],
            known_path_hmm['transition_matrix'],
            known_path_hmm['emission_params']
        )

        # Should start in state 0 (negative observations) then transition to state 1
        assert best_path[0] == 0, "Should start in state 0"
        assert best_path[1] == 0, "Should stay in state 0"
        assert best_path[2] == 1, "Should transition to state 1"
        assert best_path[3] == 1, "Should stay in state 1"
        assert best_path[4] == 1, "Should stay in state 1"

    def test_single_state(self):
        """Test Viterbi with single state HMM."""
        observations = np.array([0.01, 0.02, 0.03])
        initial_probs = np.array([1.0])
        transition_matrix = np.array([[1.0]])
        emission_params = np.array([[0.0, 0.02]])

        best_path, best_prob = HMMAlgorithms.viterbi_algorithm(
            observations, initial_probs, transition_matrix, emission_params
        )

        assert np.all(best_path == 0), "All states should be 0"
        assert np.isfinite(best_prob)

    def test_path_is_integer_dtype(self, simple_2state_hmm, simple_observations):
        """Test that returned path has integer dtype."""
        best_path, _ = HMMAlgorithms.viterbi_algorithm(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        assert best_path.dtype == np.int_ or best_path.dtype == np.int32 or best_path.dtype == np.int64


# ============================================================================
# Test forward_backward_algorithm
# ============================================================================


@pytest.mark.unit
class TestForwardBackwardAlgorithm:
    """Test combined forward-backward algorithm."""

    def test_gamma_shape(self, simple_2state_hmm, simple_observations):
        """Test gamma shape is (T, n_states)."""
        gamma, xi, _ = HMMAlgorithms.forward_backward_algorithm(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        assert gamma.shape == (5, 2)

    def test_xi_shape(self, simple_2state_hmm, simple_observations):
        """Test xi shape is (T-1, n_states, n_states)."""
        gamma, xi, _ = HMMAlgorithms.forward_backward_algorithm(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        assert xi.shape == (4, 2, 2), "Xi should be (T-1, n_states, n_states)"

    def test_gamma_probabilities_sum_to_one(self, simple_2state_hmm, simple_observations):
        """Test that gamma probabilities sum to 1 at each timestep."""
        gamma, _, _ = HMMAlgorithms.forward_backward_algorithm(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        # Gamma should sum to 1 across states for each timestep
        row_sums = np.sum(gamma, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(5), decimal=5)

    def test_xi_probabilities_sum_to_one(self, simple_2state_hmm, simple_observations):
        """Test that xi probabilities sum to 1 at each timestep."""
        _, xi, _ = HMMAlgorithms.forward_backward_algorithm(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        # Xi should sum to 1 across all state pairs for each timestep
        for t in range(len(xi)):
            xi_sum = np.sum(xi[t])
            np.testing.assert_almost_equal(xi_sum, 1.0, decimal=5)

    def test_gamma_values_are_probabilities(self, simple_2state_hmm, simple_observations):
        """Test that gamma values are valid probabilities [0, 1]."""
        gamma, _, _ = HMMAlgorithms.forward_backward_algorithm(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        assert np.all(gamma >= 0)
        assert np.all(gamma <= 1)

    def test_xi_values_are_probabilities(self, simple_2state_hmm, simple_observations):
        """Test that xi values are valid probabilities [0, 1]."""
        _, xi, _ = HMMAlgorithms.forward_backward_algorithm(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        assert np.all(xi >= 0)
        assert np.all(xi <= 1)

    def test_log_likelihood_matches_forward(self, simple_2state_hmm, simple_observations):
        """Test that log likelihood from forward-backward matches forward algorithm."""
        _, _, ll_fb = HMMAlgorithms.forward_backward_algorithm(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        _, ll_forward = HMMAlgorithms.forward_algorithm(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        np.testing.assert_almost_equal(ll_fb, ll_forward, decimal=5)


# ============================================================================
# Test baum_welch_update
# ============================================================================


@pytest.mark.unit
class TestBaumWelchUpdate:
    """Test Baum-Welch parameter update step."""

    def test_initial_probs_sum_to_one(self, simple_observations):
        """Test that updated initial probabilities sum to 1."""
        T = len(simple_observations)
        n_states = 2

        # Create dummy gamma and xi
        gamma = np.random.rand(T, n_states)
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        xi = np.random.rand(T - 1, n_states, n_states)
        for t in range(T - 1):
            xi[t] = xi[t] / xi[t].sum()

        new_initial_probs, _, _ = HMMAlgorithms.baum_welch_update(
            simple_observations, gamma, xi
        )

        np.testing.assert_almost_equal(np.sum(new_initial_probs), 1.0)

    def test_transition_matrix_rows_sum_to_one(self, simple_observations):
        """Test that transition matrix rows sum to 1."""
        T = len(simple_observations)
        n_states = 2

        gamma = np.random.rand(T, n_states)
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        xi = np.random.rand(T - 1, n_states, n_states)
        for t in range(T - 1):
            xi[t] = xi[t] / xi[t].sum()

        _, new_transition_matrix, _ = HMMAlgorithms.baum_welch_update(
            simple_observations, gamma, xi
        )

        row_sums = np.sum(new_transition_matrix, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(n_states))

    def test_emission_params_shape(self, simple_observations):
        """Test that emission parameters have correct shape."""
        T = len(simple_observations)
        n_states = 2

        gamma = np.random.rand(T, n_states)
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        xi = np.random.rand(T - 1, n_states, n_states)
        for t in range(T - 1):
            xi[t] = xi[t] / xi[t].sum()

        _, _, new_emission_params = HMMAlgorithms.baum_welch_update(
            simple_observations, gamma, xi
        )

        assert new_emission_params.shape == (n_states, 2)

    def test_emission_std_is_positive(self, simple_observations):
        """Test that emission standard deviations are positive."""
        T = len(simple_observations)
        n_states = 2

        gamma = np.random.rand(T, n_states)
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        xi = np.random.rand(T - 1, n_states, n_states)
        for t in range(T - 1):
            xi[t] = xi[t] / xi[t].sum()

        _, _, new_emission_params = HMMAlgorithms.baum_welch_update(
            simple_observations, gamma, xi
        )

        assert np.all(new_emission_params[:, 1] > 0), "Standard deviations must be positive"

    def test_regularization_prevents_zeros(self):
        """Test that regularization prevents zero probabilities."""
        observations = np.array([0.01, 0.02])
        n_states = 2

        # Create gamma with one state having very low probability
        gamma = np.array([[0.99, 0.01], [0.99, 0.01]])
        xi = np.array([[[0.98, 0.01], [0.01, 0.0]]])

        new_initial_probs, new_transition_matrix, _ = HMMAlgorithms.baum_welch_update(
            observations, gamma, xi, regularization=1e-6
        )

        # No probabilities should be exactly zero due to regularization
        assert np.all(new_initial_probs > 0)
        assert np.all(new_transition_matrix > 0)

    def test_weighted_mean_calculation(self):
        """Test that emission means are correctly weighted by gamma."""
        observations = np.array([1.0, 2.0, 3.0])
        n_states = 2

        # State 0 has high probability for observation 1
        # State 1 has high probability for observations 2 and 3
        gamma = np.array([
            [0.9, 0.1],
            [0.1, 0.9],
            [0.1, 0.9]
        ])

        xi = np.random.rand(2, n_states, n_states)
        for t in range(2):
            xi[t] = xi[t] / xi[t].sum()

        _, _, new_emission_params = HMMAlgorithms.baum_welch_update(
            observations, gamma, xi
        )

        # State 0 mean should be close to 1.0 (weighted heavily towards first obs)
        # State 1 mean should be close to 2.5 (weighted towards last two obs)
        assert new_emission_params[0, 0] < new_emission_params[1, 0], \
            "State 0 should have lower mean than state 1"


# ============================================================================
# Test compute_likelihood
# ============================================================================


@pytest.mark.unit
class TestComputeLikelihood:
    """Test likelihood computation."""

    def test_returns_scalar(self, simple_2state_hmm, simple_observations):
        """Test that likelihood is a scalar value."""
        log_likelihood = HMMAlgorithms.compute_likelihood(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        assert isinstance(log_likelihood, (float, np.floating))

    def test_log_likelihood_is_finite(self, simple_2state_hmm, simple_observations):
        """Test that log likelihood is finite."""
        log_likelihood = HMMAlgorithms.compute_likelihood(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        assert np.isfinite(log_likelihood)

    def test_matches_forward_algorithm(self, simple_2state_hmm, simple_observations):
        """Test that compute_likelihood matches forward algorithm."""
        ll_compute = HMMAlgorithms.compute_likelihood(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        _, ll_forward = HMMAlgorithms.forward_algorithm(
            simple_observations,
            simple_2state_hmm['initial_probs'],
            simple_2state_hmm['transition_matrix'],
            simple_2state_hmm['emission_params']
        )

        np.testing.assert_almost_equal(ll_compute, ll_forward)


# ============================================================================
# Test predict_next_state_probs
# ============================================================================


@pytest.mark.unit
class TestPredictNextStateProbs:
    """Test next state probability prediction."""

    def test_output_shape(self):
        """Test output has correct shape."""
        current_probs = np.array([0.7, 0.3])
        transition_matrix = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])

        next_probs = HMMAlgorithms.predict_next_state_probs(
            current_probs, transition_matrix
        )

        assert next_probs.shape == (2,)

    def test_probabilities_sum_to_one(self):
        """Test that predicted probabilities sum to 1."""
        current_probs = np.array([0.6, 0.4])
        transition_matrix = np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ])

        next_probs = HMMAlgorithms.predict_next_state_probs(
            current_probs, transition_matrix
        )

        np.testing.assert_almost_equal(np.sum(next_probs), 1.0)

    def test_deterministic_state(self):
        """Test with deterministic current state."""
        current_probs = np.array([1.0, 0.0])  # Definitely in state 0
        transition_matrix = np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ])

        next_probs = HMMAlgorithms.predict_next_state_probs(
            current_probs, transition_matrix
        )

        # Should match first row of transition matrix
        np.testing.assert_array_almost_equal(next_probs, [0.7, 0.3])

    def test_steady_state(self):
        """Test with steady state distribution."""
        # For symmetric transition matrix [0.5, 0.5; 0.5, 0.5]
        # uniform distribution is steady state
        current_probs = np.array([0.5, 0.5])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])

        next_probs = HMMAlgorithms.predict_next_state_probs(
            current_probs, transition_matrix
        )

        # Should remain at steady state
        np.testing.assert_array_almost_equal(next_probs, [0.5, 0.5])


# ============================================================================
# Test decode_states_online
# ============================================================================


@pytest.mark.unit
class TestDecodeStatesOnline:
    """Test online state decoding."""

    def test_output_shape(self):
        """Test output has correct shape."""
        new_observation = 0.02
        prev_state_probs = np.array([0.6, 0.4])
        transition_matrix = np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ])
        emission_params = np.array([
            [-0.01, 0.02],
            [0.02, 0.03]
        ])

        updated_probs = HMMAlgorithms.decode_states_online(
            new_observation, prev_state_probs, transition_matrix, emission_params
        )

        assert updated_probs.shape == (2,)

    def test_probabilities_sum_to_one(self):
        """Test that updated probabilities sum to 1."""
        new_observation = 0.02
        prev_state_probs = np.array([0.6, 0.4])
        transition_matrix = np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ])
        emission_params = np.array([
            [-0.01, 0.02],
            [0.02, 0.03]
        ])

        updated_probs = HMMAlgorithms.decode_states_online(
            new_observation, prev_state_probs, transition_matrix, emission_params
        )

        np.testing.assert_almost_equal(np.sum(updated_probs), 1.0)

    def test_observation_matches_state_mean(self):
        """Test that observation close to state mean increases that state's probability."""
        new_observation = 0.05  # Very close to state 1 mean
        prev_state_probs = np.array([0.5, 0.5])  # Start uniform
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        emission_params = np.array([
            [-0.05, 0.01],  # State 0: negative mean
            [0.05, 0.01]    # State 1: positive mean, matches observation
        ])

        updated_probs = HMMAlgorithms.decode_states_online(
            new_observation, prev_state_probs, transition_matrix, emission_params
        )

        # State 1 should have much higher probability
        assert updated_probs[1] > updated_probs[0]
        assert updated_probs[1] > 0.8  # Should be strongly confident

    def test_fallback_to_uniform_on_zero_probabilities(self):
        """Test fallback to uniform distribution when all probabilities are zero."""
        new_observation = 100.0  # Extremely unlikely observation
        prev_state_probs = np.array([0.5, 0.5])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        emission_params = np.array([
            [0.0, 0.001],  # Very small std makes 100 extremely unlikely
            [0.0, 0.001]
        ])

        updated_probs = HMMAlgorithms.decode_states_online(
            new_observation, prev_state_probs, transition_matrix, emission_params
        )

        # Should still sum to 1 and be valid probabilities
        np.testing.assert_almost_equal(np.sum(updated_probs), 1.0)
        assert np.all(updated_probs >= 0)
        assert np.all(updated_probs <= 1)

    def test_values_are_probabilities(self):
        """Test that output values are valid probabilities [0, 1]."""
        new_observation = 0.01
        prev_state_probs = np.array([0.7, 0.3])
        transition_matrix = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])
        emission_params = np.array([
            [-0.01, 0.02],
            [0.02, 0.03]
        ])

        updated_probs = HMMAlgorithms.decode_states_online(
            new_observation, prev_state_probs, transition_matrix, emission_params
        )

        assert np.all(updated_probs >= 0)
        assert np.all(updated_probs <= 1)
