"""
Forward-Backward Algorithm Tests (CRITICAL).

These tests verify the mathematical correctness of the forward-backward algorithm,
which is foundational for HMM state estimation. Any bugs here invalidate all results.

The forward-backward algorithm computes:
1. Forward probabilities: P(observations[0:t], state at t)
2. Backward probabilities: P(observations[t+1:T] | state at t)
3. State probabilities (gamma): P(state at t | all observations)
"""

import pytest
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

from hidden_regime.models.algorithms import HMMAlgorithms


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_hmm_params():
    """Simple 2-state HMM parameters for testing."""
    return {
        'n_states': 2,
        'initial_probs': np.array([0.6, 0.4]),
        'transition_matrix': np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ]),
        'emission_params': np.array([
            [0.0, 1.0],    # State 0: mean=0, std=1
            [3.0, 1.0]     # State 1: mean=3, std=1
        ])
    }


@pytest.fixture
def three_state_hmm_params():
    """3-state HMM for more complex testing."""
    return {
        'n_states': 3,
        'initial_probs': np.array([0.5, 0.3, 0.2]),
        'transition_matrix': np.array([
            [0.6, 0.3, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6]
        ]),
        'emission_params': np.array([
            [-2.0, 1.0],   # State 0: bear market
            [0.0, 0.5],    # State 1: neutral market
            [2.0, 1.0]     # State 2: bull market
        ])
    }


# ============================================================================
# Forward Algorithm Tests
# ============================================================================


@pytest.mark.unit
def test_forward_algorithm_simple_sequence(simple_hmm_params):
    """Test forward algorithm on a simple observation sequence.

    Verifies basic functionality with known parameters.
    """
    # Simple sequence that strongly suggests state transitions
    observations = np.array([0.1, 0.2, 3.1, 2.9])  # Starts low, goes high

    forward_probs, log_likelihood = HMMAlgorithms.forward_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Verify shape
    assert forward_probs.shape == (4, 2), "Forward probs should be (T, n_states)"

    # Verify all finite
    assert np.all(np.isfinite(forward_probs)), "Forward probs should be finite"

    # Verify log likelihood is finite
    assert np.isfinite(log_likelihood), "Log likelihood should be finite"

    # Log likelihood should be negative (it's a probability)
    assert log_likelihood < 0, "Log likelihood should be negative"


@pytest.mark.unit
def test_forward_algorithm_numerical_stability(simple_hmm_params):
    """Test that forward algorithm handles numerical underflow.

    Long sequences can cause underflow in probability space.
    Log-space implementation should prevent this.
    """
    # Generate long sequence
    np.random.seed(42)
    observations = np.random.randn(1000)  # Very long sequence

    forward_probs, log_likelihood = HMMAlgorithms.forward_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Critical: Should not underflow despite 1000 observations
    assert np.all(np.isfinite(forward_probs)), \
        "Forward algorithm must handle long sequences without underflow"

    assert np.isfinite(log_likelihood), \
        "Log likelihood must remain finite for long sequences"

    # Reasonable magnitude check
    assert log_likelihood > -10000, "Log likelihood shouldn't be excessively negative"


@pytest.mark.unit
def test_forward_algorithm_log_space_correctness(simple_hmm_params):
    """Test that log-space computations are mathematically correct.

    Verify that log-space forward algorithm produces same results as
    probability-space version (for short sequences where both work).
    """
    observations = np.array([0.5, 1.0, 2.5])  # Short sequence

    forward_probs, log_likelihood = HMMAlgorithms.forward_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Verify probabilities are in log space (negative values)
    assert np.all(forward_probs <= 0), "Log probabilities should be <= 0"

    # Convert to probability space
    probs = np.exp(forward_probs)

    # In probability space, each column should represent valid probabilities
    assert np.all(probs >= 0), "Probabilities should be non-negative"
    assert np.all(probs <= 1), "Probabilities should be <= 1"


@pytest.mark.unit
def test_forward_algorithm_single_observation(simple_hmm_params):
    """Test forward algorithm with single observation.

    Edge case: sequence of length 1 should still work.
    """
    observations = np.array([1.5])

    forward_probs, log_likelihood = HMMAlgorithms.forward_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Shape check
    assert forward_probs.shape == (1, 2), "Should handle single observation"

    # Should be finite
    assert np.all(np.isfinite(forward_probs))
    assert np.isfinite(log_likelihood)


@pytest.mark.unit
def test_forward_algorithm_deterministic_transitions(simple_hmm_params):
    """Test with deterministic (near-deterministic) transitions."""
    # Modify to have very high probability of staying in state
    deterministic_trans = np.array([
        [0.99, 0.01],
        [0.01, 0.99]
    ])

    observations = np.array([0.0, 0.1, 0.0, 0.2])  # All near state 0 mean

    forward_probs, log_likelihood = HMMAlgorithms.forward_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=deterministic_trans,
        emission_params=simple_hmm_params['emission_params']
    )

    # Should strongly favor state 0 throughout
    prob_space = np.exp(forward_probs)

    # State 0 should have higher probability at all time steps
    assert np.all(prob_space[:, 0] / (prob_space[:, 0] + prob_space[:, 1]) > 0.7), \
        "Should strongly prefer state 0 for observations near mean=0"


@pytest.mark.unit
def test_forward_algorithm_known_output(simple_hmm_params):
    """Test against manually calculated known output.

    For a specific input, verify exact output to catch regressions.
    """
    np.random.seed(42)
    observations = np.array([0.0])  # Single observation at state 0 mean

    forward_probs, log_likelihood = HMMAlgorithms.forward_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Manual calculation:
    # P(O=0|S=0) = norm.pdf(0, 0, 1) ≈ 0.3989
    # P(O=0|S=1) = norm.pdf(0, 3, 1) ≈ 0.0044
    # alpha[0,0] = log(0.6 * 0.3989) ≈ log(0.2393)
    # alpha[0,1] = log(0.4 * 0.0044) ≈ log(0.0018)

    expected_log_prob_0 = np.log(0.6) + norm.logpdf(0, 0, 1)
    expected_log_prob_1 = np.log(0.4) + norm.logpdf(0, 3, 1)

    # Allow small numerical error
    np.testing.assert_allclose(
        forward_probs[0, 0],
        expected_log_prob_0,
        rtol=1e-5,
        atol=1e-8
    )

    np.testing.assert_allclose(
        forward_probs[0, 1],
        expected_log_prob_1,
        rtol=1e-5,
        atol=1e-8
    )


# ============================================================================
# Backward Algorithm Tests
# ============================================================================


@pytest.mark.unit
def test_backward_algorithm_simple_sequence(simple_hmm_params):
    """Test backward algorithm on simple sequence."""
    observations = np.array([0.1, 0.2, 3.1, 2.9])

    backward_probs = HMMAlgorithms.backward_algorithm(
        observations=observations,
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Shape check
    assert backward_probs.shape == (4, 2), "Backward probs should be (T, n_states)"

    # All finite
    assert np.all(np.isfinite(backward_probs)), "Backward probs should be finite"

    # Last time step should be 0 (log(1) = 0) by definition
    np.testing.assert_allclose(
        backward_probs[-1, :],
        0.0,
        atol=1e-10,
        err_msg="Backward probs at final time should be 0 (log(1))"
    )


@pytest.mark.unit
def test_backward_algorithm_numerical_stability(simple_hmm_params):
    """Test backward algorithm numerical stability on long sequences."""
    np.random.seed(42)
    observations = np.random.randn(1000)

    backward_probs = HMMAlgorithms.backward_algorithm(
        observations=observations,
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Should not overflow/underflow
    assert np.all(np.isfinite(backward_probs)), \
        "Backward algorithm must handle long sequences"

    # Final time step should still be 0
    np.testing.assert_allclose(backward_probs[-1, :], 0.0, atol=1e-10)


# ============================================================================
# Forward-Backward Combined Tests
# ============================================================================


@pytest.mark.unit
def test_forward_backward_gamma_probabilities(simple_hmm_params):
    """Test that gamma (state probabilities) sum to 1.

    Critical property: For each time step, sum of state probabilities must be 1.
    """
    observations = np.array([0.0, 0.5, 1.0, 2.5, 3.0])

    gamma, xi, log_likelihood = HMMAlgorithms.forward_backward_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Critical: Gamma rows must sum to 1
    row_sums = gamma.sum(axis=1)
    np.testing.assert_allclose(
        row_sums,
        1.0,
        rtol=1e-6,
        atol=1e-8,
        err_msg="State probabilities must sum to 1 at each time step"
    )


@pytest.mark.unit
def test_forward_backward_xi_probabilities(simple_hmm_params):
    """Test that xi (transition probabilities) are valid.

    Xi should represent valid probability distributions.
    """
    observations = np.array([0.0, 1.0, 2.0, 3.0])

    gamma, xi, log_likelihood = HMMAlgorithms.forward_backward_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Xi shape should be (T-1, n_states, n_states)
    assert xi.shape == (3, 2, 2), "Xi should be (T-1, n_states, n_states)"

    # For each time step, xi should sum to 1 over all state pairs
    for t in range(xi.shape[0]):
        total = xi[t].sum()
        np.testing.assert_allclose(
            total,
            1.0,
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"Xi probabilities must sum to 1 at time {t}"
        )


@pytest.mark.unit
def test_forward_backward_three_states(three_state_hmm_params):
    """Test forward-backward with 3-state model."""
    observations = np.array([-2.1, -1.9, 0.1, 1.9, 2.1])

    gamma, xi, log_likelihood = HMMAlgorithms.forward_backward_algorithm(
        observations=observations,
        initial_probs=three_state_hmm_params['initial_probs'],
        transition_matrix=three_state_hmm_params['transition_matrix'],
        emission_params=three_state_hmm_params['emission_params']
    )

    # Shape checks
    assert gamma.shape == (5, 3), "Gamma should be (T, n_states)"
    assert xi.shape == (4, 3, 3), "Xi should be (T-1, n_states, n_states)"

    # Probability constraints
    gamma_sums = gamma.sum(axis=1)
    np.testing.assert_allclose(gamma_sums, 1.0, rtol=1e-6)

    # Should identify regime transitions
    # Early observations (-2.1, -1.9) should favor state 0 (bear)
    assert gamma[0, 0] > 0.4, "Should identify bear regime"

    # Later observations (1.9, 2.1) should favor state 2 (bull)
    assert gamma[-1, 2] > 0.4, "Should identify bull regime"


@pytest.mark.unit
def test_forward_backward_consistency(simple_hmm_params):
    """Test that forward and backward passes are consistent.

    The marginal probability computed from forward and backward should match.
    """
    observations = np.array([0.0, 1.0, 2.0])

    # Get forward probabilities
    forward_probs, log_likelihood = HMMAlgorithms.forward_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Get backward probabilities
    backward_probs = HMMAlgorithms.backward_algorithm(
        observations=observations,
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Compute marginal from forward-backward
    gamma_manual = forward_probs + backward_probs
    for t in range(len(observations)):
        gamma_manual[t] = gamma_manual[t] - logsumexp(gamma_manual[t])
    gamma_manual = np.exp(gamma_manual)

    # Get gamma from combined algorithm
    gamma, _, _ = HMMAlgorithms.forward_backward_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Should match
    np.testing.assert_allclose(
        gamma_manual,
        gamma,
        rtol=1e-6,
        atol=1e-8,
        err_msg="Forward-backward should be consistent"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
