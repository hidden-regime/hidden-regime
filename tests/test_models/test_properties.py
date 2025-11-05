"""
Property-based tests for HMM algorithms using Hypothesis.

These tests verify mathematical invariants that should hold for all valid
HMM parameters and observations using generative property-based testing.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

from hidden_regime.models.algorithms import HMMAlgorithms
from hidden_regime.models.utils import (
    normalize_probabilities,
    log_normalize,
    validate_hmm_parameters,
    initialize_parameters_random,
)


# ============================================================================
# Hypothesis Strategies for HMM Parameters
# ============================================================================


@st.composite
def valid_probabilities(draw, size):
    """Generate valid probability arrays that sum to 1."""
    # Generate random positive values
    values = draw(arrays(
        dtype=np.float64,
        shape=(size,),
        elements=st.floats(min_value=0.01, max_value=10.0)
    ))
    # Normalize to sum to 1
    return values / np.sum(values)


@st.composite
def valid_transition_matrix(draw, n_states):
    """Generate valid transition matrix with rows summing to 1."""
    matrix = np.zeros((n_states, n_states))
    for i in range(n_states):
        matrix[i] = draw(valid_probabilities(n_states))
    return matrix


@st.composite
def valid_emission_params(draw, n_states):
    """Generate valid emission parameters (means and stds)."""
    means = draw(arrays(
        dtype=np.float64,
        shape=(n_states,),
        elements=st.floats(min_value=-0.1, max_value=0.1)
    ))
    means = np.sort(means)  # Sort for interpretability

    stds = draw(arrays(
        dtype=np.float64,
        shape=(n_states,),
        elements=st.floats(min_value=0.005, max_value=0.1)
    ))

    return np.column_stack([means, stds])


@st.composite
def valid_observations(draw, min_size=5, max_size=20):
    """Generate valid observation sequences (log returns)."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    observations = draw(arrays(
        dtype=np.float64,
        shape=(size,),
        elements=st.floats(min_value=-0.15, max_value=0.15, allow_nan=False, allow_infinity=False)
    ))
    return observations


@st.composite
def valid_hmm_system(draw, n_states=2):
    """Generate a complete valid HMM system."""
    return {
        'initial_probs': draw(valid_probabilities(n_states)),
        'transition_matrix': draw(valid_transition_matrix(n_states)),
        'emission_params': draw(valid_emission_params(n_states)),
        'observations': draw(valid_observations())
    }


# ============================================================================
# Property Tests for Probability Normalization
# ============================================================================


@pytest.mark.unit
@pytest.mark.property
class TestProbabilityProperties:
    """Test mathematical properties of probability operations."""

    @given(st.lists(st.floats(min_value=0.0, max_value=100.0), min_size=2, max_size=10))
    @settings(max_examples=100, deadline=1000)
    def test_normalized_probabilities_sum_to_one(self, values):
        """Property: Normalized probabilities always sum to 1."""
        arr = np.array(values)
        assume(np.all(np.isfinite(arr)))
        assume(np.sum(arr) > 0)

        normalized = normalize_probabilities(arr)

        assert np.isfinite(np.sum(normalized))
        np.testing.assert_almost_equal(np.sum(normalized), 1.0, decimal=6)

    @given(st.lists(st.floats(min_value=-1000, max_value=0), min_size=2, max_size=10))
    @settings(max_examples=100, deadline=1000)
    def test_log_normalized_probs_exp_to_one(self, log_values):
        """Property: Exp of log-normalized values sum to 1."""
        log_arr = np.array(log_values)
        assume(np.all(np.isfinite(log_arr)))

        normalized_log = log_normalize(log_arr)
        probs = np.exp(normalized_log)

        assert np.all(np.isfinite(probs))
        np.testing.assert_almost_equal(np.sum(probs), 1.0, decimal=6)

    @given(valid_probabilities(5))
    @settings(max_examples=50, deadline=1000)
    def test_normalizing_valid_probs_is_idempotent(self, probs):
        """Property: Normalizing already-normalized probs doesn't change them."""
        result = normalize_probabilities(probs)

        np.testing.assert_array_almost_equal(probs, result, decimal=6)

    @given(valid_transition_matrix(3))
    @settings(max_examples=50, deadline=1000)
    def test_transition_matrix_rows_sum_to_one(self, trans_matrix):
        """Property: Every row of transition matrix sums to 1."""
        row_sums = np.sum(trans_matrix, axis=1)

        for row_sum in row_sums:
            np.testing.assert_almost_equal(row_sum, 1.0, decimal=6)


# ============================================================================
# Property Tests for HMM Algorithms
# ============================================================================


@pytest.mark.unit
@pytest.mark.property
class TestHMMAlgorithmProperties:
    """Test mathematical properties of HMM algorithms."""

    @given(valid_hmm_system(n_states=2))
    @settings(max_examples=50, deadline=2000)
    def test_forward_algorithm_log_likelihood_is_finite(self, hmm):
        """Property: Log likelihood is always finite."""
        _, log_likelihood = HMMAlgorithms.forward_algorithm(
            hmm['observations'],
            hmm['initial_probs'],
            hmm['transition_matrix'],
            hmm['emission_params']
        )

        assert np.isfinite(log_likelihood), f"Log likelihood should be finite, got {log_likelihood}"

    @given(valid_hmm_system(n_states=2))
    @settings(max_examples=50, deadline=2000)
    def test_forward_backward_gamma_sums_to_one(self, hmm):
        """Property: Gamma (state probs) sum to 1 at each timestep."""
        gamma, xi, _ = HMMAlgorithms.forward_backward_algorithm(
            hmm['observations'],
            hmm['initial_probs'],
            hmm['transition_matrix'],
            hmm['emission_params']
        )

        # Check gamma at each timestep
        for t in range(len(hmm['observations'])):
            gamma_sum = np.sum(gamma[t])
            np.testing.assert_almost_equal(
                gamma_sum, 1.0, decimal=5,
                err_msg=f"Gamma at timestep {t} should sum to 1, got {gamma_sum}"
            )

    @given(valid_hmm_system(n_states=2))
    @settings(max_examples=50, deadline=2000)
    def test_forward_backward_xi_sums_to_one(self, hmm):
        """Property: Xi (pairwise probs) sum to 1 at each timestep."""
        _, xi, _ = HMMAlgorithms.forward_backward_algorithm(
            hmm['observations'],
            hmm['initial_probs'],
            hmm['transition_matrix'],
            hmm['emission_params']
        )

        # Check xi at each timestep
        for t in range(len(xi)):
            xi_sum = np.sum(xi[t])
            np.testing.assert_almost_equal(
                xi_sum, 1.0, decimal=5,
                err_msg=f"Xi at timestep {t} should sum to 1, got {xi_sum}"
            )

    @given(valid_hmm_system(n_states=2))
    @settings(max_examples=50, deadline=2000)
    def test_viterbi_path_length_matches_observations(self, hmm):
        """Property: Viterbi path length equals observation length."""
        best_path, _ = HMMAlgorithms.viterbi_algorithm(
            hmm['observations'],
            hmm['initial_probs'],
            hmm['transition_matrix'],
            hmm['emission_params']
        )

        assert len(best_path) == len(hmm['observations'])

    @given(valid_hmm_system(n_states=2))
    @settings(max_examples=50, deadline=2000)
    def test_viterbi_path_contains_valid_states(self, hmm):
        """Property: Viterbi path only contains valid state indices."""
        n_states = len(hmm['initial_probs'])

        best_path, _ = HMMAlgorithms.viterbi_algorithm(
            hmm['observations'],
            hmm['initial_probs'],
            hmm['transition_matrix'],
            hmm['emission_params']
        )

        assert np.all(best_path >= 0)
        assert np.all(best_path < n_states)

    @given(valid_hmm_system(n_states=2))
    @settings(max_examples=50, deadline=2000)
    def test_viterbi_log_prob_is_finite(self, hmm):
        """Property: Viterbi log probability is always finite."""
        _, best_prob = HMMAlgorithms.viterbi_algorithm(
            hmm['observations'],
            hmm['initial_probs'],
            hmm['transition_matrix'],
            hmm['emission_params']
        )

        assert np.isfinite(best_prob)

    @given(valid_hmm_system(n_states=2))
    @settings(max_examples=30, deadline=2000)
    def test_forward_backward_log_likelihood_matches_forward(self, hmm):
        """Property: Forward-backward likelihood matches forward algorithm."""
        _, _, ll_fb = HMMAlgorithms.forward_backward_algorithm(
            hmm['observations'],
            hmm['initial_probs'],
            hmm['transition_matrix'],
            hmm['emission_params']
        )

        _, ll_forward = HMMAlgorithms.forward_algorithm(
            hmm['observations'],
            hmm['initial_probs'],
            hmm['transition_matrix'],
            hmm['emission_params']
        )

        np.testing.assert_almost_equal(ll_fb, ll_forward, decimal=5)

    @given(valid_hmm_system(n_states=2))
    @settings(max_examples=30, deadline=2000)
    def test_compute_likelihood_matches_forward(self, hmm):
        """Property: compute_likelihood matches forward algorithm."""
        ll_compute = HMMAlgorithms.compute_likelihood(
            hmm['observations'],
            hmm['initial_probs'],
            hmm['transition_matrix'],
            hmm['emission_params']
        )

        _, ll_forward = HMMAlgorithms.forward_algorithm(
            hmm['observations'],
            hmm['initial_probs'],
            hmm['transition_matrix'],
            hmm['emission_params']
        )

        np.testing.assert_almost_equal(ll_compute, ll_forward, decimal=5)


# ============================================================================
# Property Tests for Baum-Welch Updates
# ============================================================================


@pytest.mark.unit
@pytest.mark.property
class TestBaumWelchProperties:
    """Test mathematical properties of Baum-Welch parameter updates."""

    @given(
        valid_observations(min_size=10, max_size=30),
        st.integers(min_value=2, max_value=3)
    )
    @settings(max_examples=30, deadline=3000)
    def test_baum_welch_maintains_probability_constraints(self, observations, n_states):
        """Property: Baum-Welch updates maintain probability constraints."""
        # Create valid gamma and xi
        T = len(observations)

        # Generate valid gamma (sum to 1 at each timestep)
        gamma = np.random.rand(T, n_states)
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        # Generate valid xi (sum to 1 at each timestep)
        xi = np.random.rand(T - 1, n_states, n_states)
        for t in range(T - 1):
            xi[t] = xi[t] / xi[t].sum()

        new_initial, new_trans, new_emission = HMMAlgorithms.baum_welch_update(
            observations, gamma, xi
        )

        # Check probability constraints
        np.testing.assert_almost_equal(np.sum(new_initial), 1.0, decimal=5)

        for i in range(n_states):
            row_sum = np.sum(new_trans[i])
            np.testing.assert_almost_equal(row_sum, 1.0, decimal=5)

        assert np.all(new_emission[:, 1] > 0), "Stds must be positive"

    @given(
        valid_observations(min_size=10, max_size=30),
        st.integers(min_value=2, max_value=3)
    )
    @settings(max_examples=30, deadline=3000)
    def test_baum_welch_updated_params_are_valid(self, observations, n_states):
        """Property: Baum-Welch produces valid HMM parameters."""
        T = len(observations)

        gamma = np.random.rand(T, n_states)
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        xi = np.random.rand(T - 1, n_states, n_states)
        for t in range(T - 1):
            xi[t] = xi[t] / xi[t].sum()

        new_initial, new_trans, new_emission = HMMAlgorithms.baum_welch_update(
            observations, gamma, xi
        )

        # Should pass validation
        try:
            validate_hmm_parameters(new_initial, new_trans, new_emission)
        except ValueError as e:
            pytest.fail(f"Baum-Welch produced invalid parameters: {e}")


# ============================================================================
# Property Tests for Parameter Initialization
# ============================================================================


@pytest.mark.unit
@pytest.mark.property
class TestInitializationProperties:
    """Test properties of parameter initialization."""

    @given(
        arrays(dtype=np.float64, shape=st.integers(50, 200),
               elements=st.floats(min_value=-0.1, max_value=0.1)),
        st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=30, deadline=2000)
    def test_random_initialization_produces_valid_params(self, returns, n_states):
        """Property: Random initialization always produces valid parameters."""
        # Filter out edge case where all returns are identical (std=0)
        assume(np.std(returns) > 0.001)

        initial, trans, emission = initialize_parameters_random(
            n_states, returns, random_seed=42
        )

        # Should pass validation
        try:
            validate_hmm_parameters(initial, trans, emission)
        except ValueError as e:
            pytest.fail(f"Random initialization produced invalid parameters: {e}")

    @given(
        arrays(dtype=np.float64, shape=st.integers(50, 200),
               elements=st.floats(min_value=-0.1, max_value=0.1)),
        st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=30, deadline=2000)
    def test_random_initialization_emission_means_sorted(self, returns, n_states):
        """Property: Random initialization produces sorted emission means."""
        _, _, emission = initialize_parameters_random(
            n_states, returns, random_seed=42
        )

        means = emission[:, 0]
        assert np.all(means[:-1] <= means[1:]), "Means should be sorted"


# ============================================================================
# Property Tests for Online Decoding
# ============================================================================


@pytest.mark.unit
@pytest.mark.property
class TestOnlineDecodingProperties:
    """Test properties of online state decoding."""

    @given(
        st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
        valid_probabilities(2),
        valid_transition_matrix(2),
        valid_emission_params(2)
    )
    @settings(max_examples=50, deadline=1000)
    def test_online_decode_probabilities_sum_to_one(
        self, observation, prev_probs, trans_matrix, emission_params
    ):
        """Property: Online decoded probabilities sum to 1."""
        updated_probs = HMMAlgorithms.decode_states_online(
            observation, prev_probs, trans_matrix, emission_params
        )

        np.testing.assert_almost_equal(np.sum(updated_probs), 1.0, decimal=6)

    @given(
        st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
        valid_probabilities(2),
        valid_transition_matrix(2),
        valid_emission_params(2)
    )
    @settings(max_examples=50, deadline=1000)
    def test_online_decode_produces_valid_probabilities(
        self, observation, prev_probs, trans_matrix, emission_params
    ):
        """Property: Online decoded probabilities are in [0, 1]."""
        updated_probs = HMMAlgorithms.decode_states_online(
            observation, prev_probs, trans_matrix, emission_params
        )

        assert np.all(updated_probs >= 0)
        assert np.all(updated_probs <= 1)


# ============================================================================
# Property Tests for State Prediction
# ============================================================================


@pytest.mark.unit
@pytest.mark.property
class TestStatePredictionProperties:
    """Test properties of state probability prediction."""

    @given(valid_probabilities(3), valid_transition_matrix(3))
    @settings(max_examples=50, deadline=1000)
    def test_predict_next_state_sums_to_one(self, current_probs, trans_matrix):
        """Property: Predicted state probabilities sum to 1."""
        next_probs = HMMAlgorithms.predict_next_state_probs(
            current_probs, trans_matrix
        )

        np.testing.assert_almost_equal(np.sum(next_probs), 1.0, decimal=6)

    @given(valid_probabilities(3), valid_transition_matrix(3))
    @settings(max_examples=50, deadline=1000)
    def test_predict_next_state_produces_valid_probabilities(
        self, current_probs, trans_matrix
    ):
        """Property: Predicted probabilities are in [0, 1]."""
        next_probs = HMMAlgorithms.predict_next_state_probs(
            current_probs, trans_matrix
        )

        assert np.all(next_probs >= 0)
        assert np.all(next_probs <= 1)

    @given(st.integers(min_value=0, max_value=2), valid_transition_matrix(3))
    @settings(max_examples=30, deadline=1000)
    def test_predict_from_deterministic_state_matches_transition_row(
        self, state_idx, trans_matrix
    ):
        """Property: Prediction from deterministic state equals transition row."""
        # Start in deterministic state
        current_probs = np.zeros(3)
        current_probs[state_idx] = 1.0

        next_probs = HMMAlgorithms.predict_next_state_probs(
            current_probs, trans_matrix
        )

        # Should match the transition matrix row
        np.testing.assert_array_almost_equal(
            next_probs, trans_matrix[state_idx], decimal=6
        )


# ============================================================================
# Property Tests for Emission Probabilities
# ============================================================================


@pytest.mark.unit
@pytest.mark.property
class TestEmissionProbabilityProperties:
    """Test properties of emission probability calculations."""

    @given(
        arrays(dtype=np.float64, shape=st.integers(5, 20),
               elements=st.floats(min_value=-0.15, max_value=0.15)),
        valid_emission_params(2)
    )
    @settings(max_examples=50, deadline=1000)
    def test_log_emission_probs_are_finite(self, observations, emission_params):
        """Property: Log emission probabilities are always finite."""
        log_probs = HMMAlgorithms.log_emission_probability(
            observations, emission_params[:, 0], emission_params[:, 1]
        )

        assert np.all(np.isfinite(log_probs))

    @given(
        arrays(dtype=np.float64, shape=st.integers(5, 20),
               elements=st.floats(min_value=-0.15, max_value=0.15)),
        valid_emission_params(3)
    )
    @settings(max_examples=50, deadline=1000)
    def test_emission_prob_shape(self, observations, emission_params):
        """Property: Emission probability shape is (T, n_states)."""
        log_probs = HMMAlgorithms.log_emission_probability(
            observations, emission_params[:, 0], emission_params[:, 1]
        )

        assert log_probs.shape == (len(observations), len(emission_params))
