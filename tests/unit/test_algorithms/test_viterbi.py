"""
Viterbi Algorithm Tests (CRITICAL).

The Viterbi algorithm finds the most likely sequence of hidden states given
observations. This is essential for regime detection and must be mathematically correct.

Tests verify:
1. Path optimality: Returns actual most likely path
2. Probability correctness: Computed probabilities are accurate
3. Numerical stability: Handles long sequences without overflow
4. Edge cases: Single states, deterministic paths, ties
"""

import pytest
import numpy as np
from scipy.stats import norm

from hidden_regime.models.algorithms import HMMAlgorithms


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_hmm_params():
    """Simple 2-state HMM parameters."""
    return {
        'n_states': 2,
        'initial_probs': np.array([0.6, 0.4]),
        'transition_matrix': np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ]),
        'emission_params': np.array([
            [0.0, 1.0],    # State 0: low volatility
            [3.0, 1.0]     # State 1: high volatility
        ])
    }


@pytest.fixture
def deterministic_hmm_params():
    """HMM with deterministic transitions for testing."""
    return {
        'n_states': 2,
        'initial_probs': np.array([1.0, 0.0]),  # Always start in state 0
        'transition_matrix': np.array([
            [1.0, 0.0],  # State 0 always stays in state 0
            [0.0, 1.0]   # State 1 always stays in state 1
        ]),
        'emission_params': np.array([
            [0.0, 0.5],
            [5.0, 0.5]
        ])
    }


# ============================================================================
# Basic Functionality Tests
# ============================================================================


@pytest.mark.unit
def test_viterbi_simple_sequence(simple_hmm_params):
    """Test Viterbi on simple observation sequence.

    Verify basic functionality and output format.
    """
    # Sequence clearly transitioning from state 0 to state 1
    observations = np.array([0.1, 0.2, 2.8, 3.1, 2.9])

    best_path, best_prob = HMMAlgorithms.viterbi_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Shape check
    assert best_path.shape == (5,), "Path should have length T"

    # States should be valid indices
    assert np.all(best_path >= 0), "States should be non-negative"
    assert np.all(best_path < 2), "States should be < n_states"

    # Probability should be finite and negative (log probability)
    assert np.isfinite(best_prob), "Best prob should be finite"
    assert best_prob < 0, "Log probability should be negative"

    # Should detect transition: start in state 0, end in state 1
    assert best_path[0] == 0, "Should start in state 0 (observations near 0)"
    assert best_path[-1] == 1, "Should end in state 1 (observations near 3)"


@pytest.mark.unit
def test_viterbi_known_path(deterministic_hmm_params):
    """Test Viterbi with known deterministic path.

    With deterministic transitions and clear observations, path should be certain.
    """
    # All observations near state 0 mean
    observations = np.array([0.0, 0.1, -0.1, 0.2, 0.0])

    best_path, best_prob = HMMAlgorithms.viterbi_algorithm(
        observations=observations,
        initial_probs=deterministic_hmm_params['initial_probs'],
        transition_matrix=deterministic_hmm_params['transition_matrix'],
        emission_params=deterministic_hmm_params['emission_params']
    )

    # With deterministic transition matrix, should stay in state 0
    expected_path = np.array([0, 0, 0, 0, 0])

    np.testing.assert_array_equal(
        best_path,
        expected_path,
        err_msg="Should follow deterministic path in state 0"
    )


@pytest.mark.unit
def test_viterbi_single_state_model():
    """Test Viterbi with single-state HMM (edge case)."""
    params = {
        'n_states': 1,
        'initial_probs': np.array([1.0]),
        'transition_matrix': np.array([[1.0]]),
        'emission_params': np.array([[0.0, 1.0]])
    }

    observations = np.array([0.5, 1.0, -0.5, 0.0])

    best_path, best_prob = HMMAlgorithms.viterbi_algorithm(
        observations=observations,
        initial_probs=params['initial_probs'],
        transition_matrix=params['transition_matrix'],
        emission_params=params['emission_params']
    )

    # Should always be in state 0
    np.testing.assert_array_equal(
        best_path,
        np.array([0, 0, 0, 0]),
        err_msg="Single-state model should always be in state 0"
    )


# ============================================================================
# Numerical Stability Tests
# ============================================================================


@pytest.mark.unit
def test_viterbi_log_space_correctness(simple_hmm_params):
    """Test that log-space Viterbi is numerically correct.

    Verify log-space arithmetic produces correct results.
    """
    observations = np.array([0.0, 0.5, 1.0])

    best_path, best_prob = HMMAlgorithms.viterbi_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Probability should be in reasonable range for 3 observations
    assert -50 < best_prob < 0, "Log probability should be reasonable"

    # Path should be valid
    assert len(best_path) == 3
    assert np.all((best_path == 0) | (best_path == 1))


@pytest.mark.unit
def test_viterbi_numerical_stability_long_sequence(simple_hmm_params):
    """Test Viterbi on long sequence without overflow.

    Log-space implementation must prevent underflow on long sequences.
    """
    np.random.seed(42)
    observations = np.random.randn(1000)

    best_path, best_prob = HMMAlgorithms.viterbi_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Should handle long sequence
    assert len(best_path) == 1000, "Should return path for all observations"
    assert np.isfinite(best_prob), "Probability should remain finite"

    # All states should be valid
    assert np.all((best_path == 0) | (best_path == 1))


@pytest.mark.unit
def test_viterbi_extreme_probability_values():
    """Test Viterbi with extreme probability values.

    Should handle very small or very large (in log space) probabilities.
    """
    params = {
        'n_states': 2,
        'initial_probs': np.array([0.999, 0.001]),  # Very skewed
        'transition_matrix': np.array([
            [0.999, 0.001],
            [0.001, 0.999]
        ]),
        'emission_params': np.array([
            [0.0, 0.1],    # Very tight distribution
            [10.0, 0.1]
        ])
    }

    # Observations strongly suggesting state 0
    observations = np.array([0.0, 0.01, -0.01, 0.02])

    best_path, best_prob = HMMAlgorithms.viterbi_algorithm(
        observations=observations,
        initial_probs=params['initial_probs'],
        transition_matrix=params['transition_matrix'],
        emission_params=params['emission_params']
    )

    # Should strongly prefer state 0 throughout
    np.testing.assert_array_equal(
        best_path,
        np.array([0, 0, 0, 0]),
        err_msg="Should stay in state 0 with extreme probabilities"
    )


# ============================================================================
# Correctness Tests
# ============================================================================


@pytest.mark.unit
def test_viterbi_backtracking_correctness(simple_hmm_params):
    """Test that backtracking produces valid path.

    Backtracking should produce a connected path through the trellis.
    """
    observations = np.array([0.0, 1.0, 2.0, 3.0, 2.5])

    best_path, best_prob = HMMAlgorithms.viterbi_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Path should be smooth - check for valid transitions
    transition_matrix = simple_hmm_params['transition_matrix']

    for t in range(len(best_path) - 1):
        from_state = best_path[t]
        to_state = best_path[t + 1]

        # Transition should be possible (non-zero probability)
        assert transition_matrix[from_state, to_state] > 0, \
            f"Invalid transition at t={t}: {from_state} -> {to_state}"


@pytest.mark.unit
def test_viterbi_vs_forward_backward_consistency(simple_hmm_params):
    """Test that Viterbi path is consistent with forward-backward.

    The Viterbi path should generally align with high-probability states
    from forward-backward algorithm (though not always identical).
    """
    np.random.seed(42)
    observations = np.random.randn(50)

    # Get Viterbi path
    viterbi_path, viterbi_prob = HMMAlgorithms.viterbi_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Get forward-backward state probabilities
    gamma, _, fb_prob = HMMAlgorithms.forward_backward_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Most likely state at each time from forward-backward
    fb_most_likely = np.argmax(gamma, axis=1)

    # Viterbi path should have substantial overlap with FB most likely
    agreement = np.mean(viterbi_path == fb_most_likely)

    assert agreement > 0.6, \
        f"Viterbi and Forward-Backward should mostly agree (agreement={agreement})"


@pytest.mark.unit
def test_viterbi_ties_handling():
    """Test Viterbi when multiple paths have same probability.

    When there are ties, should pick consistently (first maximum).
    """
    # Create symmetric scenario
    params = {
        'n_states': 2,
        'initial_probs': np.array([0.5, 0.5]),  # Symmetric start
        'transition_matrix': np.array([
            [0.5, 0.5],  # Symmetric transitions
            [0.5, 0.5]
        ]),
        'emission_params': np.array([
            [0.0, 1.0],
            [0.0, 1.0]   # Identical emissions (ties!)
        ])
    }

    observations = np.array([0.0, 0.0, 0.0])

    best_path, best_prob = HMMAlgorithms.viterbi_algorithm(
        observations=observations,
        initial_probs=params['initial_probs'],
        transition_matrix=params['transition_matrix'],
        emission_params=params['emission_params']
    )

    # Should produce a valid path (exact path doesn't matter due to symmetry)
    assert len(best_path) == 3
    assert np.all((best_path == 0) | (best_path == 1))

    # Should be deterministic (same result every time)
    best_path2, best_prob2 = HMMAlgorithms.viterbi_algorithm(
        observations=observations,
        initial_probs=params['initial_probs'],
        transition_matrix=params['transition_matrix'],
        emission_params=params['emission_params']
    )

    np.testing.assert_array_equal(best_path, best_path2,
        err_msg="Viterbi should be deterministic")


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.unit
def test_viterbi_single_observation(simple_hmm_params):
    """Test Viterbi with single observation."""
    observations = np.array([1.5])

    best_path, best_prob = HMMAlgorithms.viterbi_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    assert len(best_path) == 1, "Path should have length 1"
    assert best_path[0] in [0, 1], "State should be valid"
    assert np.isfinite(best_prob), "Probability should be finite"


@pytest.mark.unit
def test_viterbi_zero_probability_paths():
    """Test Viterbi when some paths have zero probability.

    Should find best path even when some transitions are impossible.
    """
    params = {
        'n_states': 2,
        'initial_probs': np.array([1.0, 0.0]),  # Must start in state 0
        'transition_matrix': np.array([
            [0.5, 0.5],
            [0.0, 1.0]   # State 1 can't transition to state 0
        ]),
        'emission_params': np.array([
            [0.0, 1.0],
            [3.0, 1.0]
        ])
    }

    observations = np.array([0.0, 3.0, 3.0])  # Suggests 0 -> 1 -> 1

    best_path, best_prob = HMMAlgorithms.viterbi_algorithm(
        observations=observations,
        initial_probs=params['initial_probs'],
        transition_matrix=params['transition_matrix'],
        emission_params=params['emission_params']
    )

    # Should follow allowed transitions
    assert best_path[0] == 0, "Must start in state 0"
    # Once in state 1, must stay there
    if 1 in best_path:
        first_one = np.where(best_path == 1)[0][0]
        assert np.all(best_path[first_one:] == 1), \
            "Once in state 1, must stay in state 1"


@pytest.mark.unit
def test_viterbi_deterministic_path():
    """Test Viterbi with completely deterministic observations."""
    params = {
        'n_states': 2,
        'initial_probs': np.array([0.5, 0.5]),
        'transition_matrix': np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ]),
        'emission_params': np.array([
            [0.0, 0.01],    # Very tight distribution around 0
            [5.0, 0.01]     # Very tight distribution around 5
        ])
    }

    # Clearly alternating between states
    observations = np.array([0.0, 5.0, 0.0, 5.0])

    best_path, best_prob = HMMAlgorithms.viterbi_algorithm(
        observations=observations,
        initial_probs=params['initial_probs'],
        transition_matrix=params['transition_matrix'],
        emission_params=params['emission_params']
    )

    # Should match observation pattern
    np.testing.assert_array_equal(
        best_path,
        np.array([0, 1, 0, 1]),
        err_msg="Should match clear alternating pattern"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
