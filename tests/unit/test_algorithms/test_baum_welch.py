"""
Baum-Welch Algorithm Tests (CRITICAL).

The Baum-Welch algorithm trains HMM parameters via Expectation-Maximization.
This is the core learning algorithm and must be mathematically correct for
the model to learn meaningful regimes.

Tests verify:
1. Parameter convergence: Algorithm converges to stable parameters
2. Likelihood improvement: Likelihood increases monotonically
3. Parameter validity: Updated parameters remain valid (probabilities, etc.)
4. Numerical stability: Handles various data distributions
5. Known model recovery: Can recover parameters from synthetic data
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
    """Simple 2-state HMM for testing."""
    return {
        'n_states': 2,
        'initial_probs': np.array([0.6, 0.4]),
        'transition_matrix': np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ]),
        'emission_params': np.array([
            [0.0, 1.0],
            [3.0, 1.0]
        ])
    }


@pytest.fixture
def synthetic_data(simple_hmm_params):
    """Generate synthetic data from known HMM for recovery tests."""
    np.random.seed(42)
    n_samples = 500

    # Generate hidden state sequence
    states = []
    current_state = np.random.choice(2, p=simple_hmm_params['initial_probs'])
    states.append(current_state)

    for _ in range(n_samples - 1):
        current_state = np.random.choice(
            2,
            p=simple_hmm_params['transition_matrix'][current_state]
        )
        states.append(current_state)

    states = np.array(states)

    # Generate observations from states
    observations = np.zeros(n_samples)
    for i, state in enumerate(states):
        mean = simple_hmm_params['emission_params'][state, 0]
        std = simple_hmm_params['emission_params'][state, 1]
        observations[i] = np.random.normal(mean, std)

    return observations, states


# ============================================================================
# Parameter Update Tests
# ============================================================================


@pytest.mark.unit
def test_baum_welch_update_probability_constraints(simple_hmm_params, synthetic_data):
    """Test that Baum-Welch updates produce valid probability distributions.

    Critical properties:
    - Initial probs sum to 1
    - Transition matrix rows sum to 1
    - All probabilities in [0, 1]
    - Standard deviations are positive
    """
    observations, _ = synthetic_data

    # Get gamma and xi from forward-backward
    gamma, xi, _ = HMMAlgorithms.forward_backward_algorithm(
        observations=observations,
        initial_probs=simple_hmm_params['initial_probs'],
        transition_matrix=simple_hmm_params['transition_matrix'],
        emission_params=simple_hmm_params['emission_params']
    )

    # Perform Baum-Welch update
    new_initial, new_trans, new_emission = HMMAlgorithms.baum_welch_update(
        observations=observations,
        gamma=gamma,
        xi=xi
    )

    # Test initial probabilities
    np.testing.assert_allclose(
        np.sum(new_initial),
        1.0,
        rtol=1e-6,
        err_msg="Initial probabilities must sum to 1"
    )
    assert np.all(new_initial >= 0) and np.all(new_initial <= 1), \
        "Initial probabilities must be in [0, 1]"

    # Test transition matrix
    row_sums = np.sum(new_trans, axis=1)
    np.testing.assert_allclose(
        row_sums,
        1.0,
        rtol=1e-6,
        err_msg="Transition matrix rows must sum to 1"
    )
    assert np.all(new_trans >= 0) and np.all(new_trans <= 1), \
        "Transition probabilities must be in [0, 1]"

    # Test emission parameters
    assert np.all(new_emission[:, 1] > 0), \
        "Standard deviations must be positive"


@pytest.mark.unit
def test_baum_welch_likelihood_monotonic_increase(simple_hmm_params, synthetic_data):
    """Test that likelihood increases monotonically during EM iterations.

    CRITICAL: Baum-Welch is guaranteed to never decrease likelihood.
    This is a fundamental property of EM algorithms.
    """
    observations, _ = synthetic_data

    # Start with initial parameters
    current_initial = simple_hmm_params['initial_probs'].copy()
    current_trans = simple_hmm_params['transition_matrix'].copy()
    current_emission = simple_hmm_params['emission_params'].copy()

    likelihoods = []

    # Run 10 Baum-Welch iterations
    for iteration in range(10):
        # Compute current likelihood
        likelihood = HMMAlgorithms.compute_likelihood(
            observations, current_initial, current_trans, current_emission
        )
        likelihoods.append(likelihood)

        # E-step: Forward-backward
        gamma, xi, _ = HMMAlgorithms.forward_backward_algorithm(
            observations, current_initial, current_trans, current_emission
        )

        # M-step: Parameter update
        current_initial, current_trans, current_emission = \
            HMMAlgorithms.baum_welch_update(observations, gamma, xi)

    # Verify monotonic increase
    for i in range(1, len(likelihoods)):
        assert likelihoods[i] >= likelihoods[i-1] - 1e-6, \
            f"Likelihood decreased at iteration {i}: {likelihoods[i-1]} -> {likelihoods[i]}"


@pytest.mark.unit
def test_baum_welch_parameter_updates_converge(simple_hmm_params, synthetic_data):
    """Test that parameters converge to stable values.

    After enough iterations, parameter changes should become small.
    """
    observations, _ = synthetic_data

    current_initial = simple_hmm_params['initial_probs'].copy()
    current_trans = simple_hmm_params['transition_matrix'].copy()
    current_emission = simple_hmm_params['emission_params'].copy()

    # Run many iterations
    for _ in range(50):
        gamma, xi, _ = HMMAlgorithms.forward_backward_algorithm(
            observations, current_initial, current_trans, current_emission
        )

        new_initial, new_trans, new_emission = \
            HMMAlgorithms.baum_welch_update(observations, gamma, xi)

        current_initial = new_initial
        current_trans = new_trans
        current_emission = new_emission

    # Run one more iteration to check stability
    gamma, xi, _ = HMMAlgorithms.forward_backward_algorithm(
        observations, current_initial, current_trans, current_emission
    )

    final_initial, final_trans, final_emission = \
        HMMAlgorithms.baum_welch_update(observations, gamma, xi)

    # Parameters should be nearly identical after convergence
    np.testing.assert_allclose(
        final_initial,
        current_initial,
        rtol=1e-3,
        err_msg="Initial probs should stabilize"
    )

    np.testing.assert_allclose(
        final_trans,
        current_trans,
        rtol=1e-3,
        err_msg="Transition matrix should stabilize"
    )

    np.testing.assert_allclose(
        final_emission,
        current_emission,
        rtol=1e-2,
        err_msg="Emission params should stabilize"
    )


@pytest.mark.unit
def test_baum_welch_known_model_recovery(simple_hmm_params, synthetic_data):
    """Test that Baum-Welch can recover known parameters from synthetic data.

    When given data generated from known parameters, algorithm should
    recover approximately the same parameters.
    """
    observations, _ = synthetic_data

    # Start with random initialization
    np.random.seed(123)  # Different seed than data generation
    current_initial = np.random.dirichlet([1, 1])
    current_trans = np.array([
        np.random.dirichlet([1, 1]),
        np.random.dirichlet([1, 1])
    ])
    current_emission = np.array([
        [np.random.randn(), np.random.rand() + 0.5],
        [np.random.randn() + 2, np.random.rand() + 0.5]
    ])

    # Run Baum-Welch
    for _ in range(100):
        gamma, xi, _ = HMMAlgorithms.forward_backward_algorithm(
            observations, current_initial, current_trans, current_emission
        )

        current_initial, current_trans, current_emission = \
            HMMAlgorithms.baum_welch_update(observations, gamma, xi)

    # Should approximately recover true parameters
    # (order of states may be swapped, so check both orderings)

    true_means = simple_hmm_params['emission_params'][:, 0]
    learned_means = current_emission[:, 0]

    # Check if learned in same order or reversed
    if abs(learned_means[0] - true_means[0]) < abs(learned_means[0] - true_means[1]):
        # Same order
        expected_emission = simple_hmm_params['emission_params']
    else:
        # Reversed order
        expected_emission = simple_hmm_params['emission_params'][[1, 0], :]

    # Means should be close (within 0.5)
    np.testing.assert_allclose(
        sorted(current_emission[:, 0]),
        sorted(expected_emission[:, 0]),
        atol=0.5,
        err_msg="Should recover emission means approximately"
    )


@pytest.mark.unit
def test_baum_welch_random_initialization(synthetic_data):
    """Test that Baum-Welch works with different random initializations.

    Should converge to reasonable parameters regardless of initialization.
    """
    observations, _ = synthetic_data

    for seed in [1, 2, 3]:
        np.random.seed(seed)

        # Random initialization
        initial_probs = np.random.dirichlet([1, 1])
        transition_matrix = np.array([
            np.random.dirichlet([1, 1]),
            np.random.dirichlet([1, 1])
        ])
        emission_params = np.array([
            [np.random.randn() * 2, np.random.rand() + 0.5],
            [np.random.randn() * 2, np.random.rand() + 0.5]
        ])

        # Run Baum-Welch
        for _ in range(50):
            gamma, xi, _ = HMMAlgorithms.forward_backward_algorithm(
                observations, initial_probs, transition_matrix, emission_params
            )

            initial_probs, transition_matrix, emission_params = \
                HMMAlgorithms.baum_welch_update(observations, gamma, xi)

        # Should produce valid parameters
        assert np.allclose(np.sum(initial_probs), 1.0)
        assert np.all(initial_probs >= 0) and np.all(initial_probs <= 1)
        assert np.all(emission_params[:, 1] > 0)  # Positive std devs


# ============================================================================
# Numerical Stability Tests
# ============================================================================


@pytest.mark.unit
def test_baum_welch_numerical_stability_long_sequence():
    """Test Baum-Welch on long sequences without numerical issues."""
    np.random.seed(42)

    # Very long sequence
    observations = np.concatenate([
        np.random.randn(500),
        np.random.randn(500) + 3
    ])

    # Initialize
    initial_probs = np.array([0.5, 0.5])
    transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
    emission_params = np.array([
        [0.0, 1.0],
        [3.0, 1.0]
    ])

    # Run several iterations
    for _ in range(20):
        gamma, xi, likelihood = HMMAlgorithms.forward_backward_algorithm(
            observations, initial_probs, transition_matrix, emission_params
        )

        initial_probs, transition_matrix, emission_params = \
            HMMAlgorithms.baum_welch_update(observations, gamma, xi)

        # All values should be finite
        assert np.all(np.isfinite(initial_probs))
        assert np.all(np.isfinite(transition_matrix))
        assert np.all(np.isfinite(emission_params))
        assert np.isfinite(likelihood)


@pytest.mark.unit
def test_baum_welch_zero_probability_prevention():
    """Test that regularization prevents zero probabilities."""
    np.random.seed(42)

    # Highly imbalanced data (strongly favors one state)
    observations = np.random.randn(100) * 0.1  # All near 0

    initial_probs = np.array([0.99, 0.01])
    transition_matrix = np.array([[0.99, 0.01], [0.5, 0.5]])
    emission_params = np.array([[0.0, 0.1], [10.0, 1.0]])

    # Run Baum-Welch with regularization
    for _ in range(10):
        gamma, xi, _ = HMMAlgorithms.forward_backward_algorithm(
            observations, initial_probs, transition_matrix, emission_params
        )

        initial_probs, transition_matrix, emission_params = \
            HMMAlgorithms.baum_welch_update(
                observations, gamma, xi, regularization=1e-6
            )

    # Should not have exact zeros (regularization prevents it)
    assert np.all(initial_probs > 0), "Regularization should prevent zero probs"
    assert np.all(transition_matrix > 0), "Regularization should prevent zero transitions"


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.unit
def test_baum_welch_deterministic_seed():
    """Test that Baum-Welch is deterministic with fixed inputs."""
    np.random.seed(42)
    observations = np.random.randn(100)

    initial_probs = np.array([0.5, 0.5])
    transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
    emission_params = np.array([[0.0, 1.0], [3.0, 1.0]])

    # Run twice with same inputs
    results1 = []
    for _ in range(5):
        gamma, xi, _ = HMMAlgorithms.forward_backward_algorithm(
            observations, initial_probs, transition_matrix, emission_params
        )
        initial_probs, transition_matrix, emission_params = \
            HMMAlgorithms.baum_welch_update(observations, gamma, xi)
        results1.append((initial_probs.copy(), transition_matrix.copy()))

    # Reset and run again
    initial_probs = np.array([0.5, 0.5])
    transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
    emission_params = np.array([[0.0, 1.0], [3.0, 1.0]])

    results2 = []
    for _ in range(5):
        gamma, xi, _ = HMMAlgorithms.forward_backward_algorithm(
            observations, initial_probs, transition_matrix, emission_params
        )
        initial_probs, transition_matrix, emission_params = \
            HMMAlgorithms.baum_welch_update(observations, gamma, xi)
        results2.append((initial_probs.copy(), transition_matrix.copy()))

    # Should be identical
    for i in range(5):
        np.testing.assert_array_equal(results1[i][0], results2[i][0])
        np.testing.assert_array_equal(results1[i][1], results2[i][1])


@pytest.mark.unit
def test_baum_welch_transition_matrix_validity():
    """Test that transition matrix remains valid after updates."""
    np.random.seed(42)
    observations = np.random.randn(200)

    initial_probs = np.array([0.6, 0.4])
    transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
    emission_params = np.array([[0.0, 1.0], [2.0, 1.0]])

    for iteration in range(20):
        gamma, xi, _ = HMMAlgorithms.forward_backward_algorithm(
            observations, initial_probs, transition_matrix, emission_params
        )

        initial_probs, transition_matrix, emission_params = \
            HMMAlgorithms.baum_welch_update(observations, gamma, xi)

        # Validate at each iteration
        assert transition_matrix.shape == (2, 2), \
            f"Iteration {iteration}: Shape should be (2,2)"

        row_sums = np.sum(transition_matrix, axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, rtol=1e-6,
            err_msg=f"Iteration {iteration}: Rows must sum to 1"
        )

        assert np.all(transition_matrix >= 0) and np.all(transition_matrix <= 1), \
            f"Iteration {iteration}: All values must be in [0, 1]"


@pytest.mark.unit
def test_baum_welch_emission_matrix_validity():
    """Test that emission parameters remain valid after updates."""
    np.random.seed(42)
    observations = np.random.randn(200)

    initial_probs = np.array([0.5, 0.5])
    transition_matrix = np.array([[0.8, 0.2], [0.2, 0.8]])
    emission_params = np.array([[0.0, 1.0], [3.0, 1.0]])

    for iteration in range(20):
        gamma, xi, _ = HMMAlgorithms.forward_backward_algorithm(
            observations, initial_probs, transition_matrix, emission_params
        )

        initial_probs, transition_matrix, emission_params = \
            HMMAlgorithms.baum_welch_update(observations, gamma, xi)

        # Validate emission parameters
        assert emission_params.shape == (2, 2), \
            f"Iteration {iteration}: Emission shape should be (2, 2)"

        # Standard deviations must be positive
        assert np.all(emission_params[:, 1] > 0), \
            f"Iteration {iteration}: Standard deviations must be positive"

        # Means should be finite
        assert np.all(np.isfinite(emission_params[:, 0])), \
            f"Iteration {iteration}: Means must be finite"


@pytest.mark.unit
def test_baum_welch_state_probability_validity():
    """Test that state probabilities (initial probs) remain valid."""
    np.random.seed(42)
    observations = np.random.randn(150)

    initial_probs = np.array([0.7, 0.3])
    transition_matrix = np.array([[0.6, 0.4], [0.4, 0.6]])
    emission_params = np.array([[-1.0, 1.0], [1.0, 1.0]])

    for iteration in range(15):
        gamma, xi, _ = HMMAlgorithms.forward_backward_algorithm(
            observations, initial_probs, transition_matrix, emission_params
        )

        initial_probs, transition_matrix, emission_params = \
            HMMAlgorithms.baum_welch_update(observations, gamma, xi)

        # Validate initial probabilities
        assert len(initial_probs) == 2, \
            f"Iteration {iteration}: Should have 2 initial probs"

        np.testing.assert_allclose(
            np.sum(initial_probs), 1.0, rtol=1e-6,
            err_msg=f"Iteration {iteration}: Initial probs must sum to 1"
        )

        assert np.all(initial_probs >= 0) and np.all(initial_probs <= 1), \
            f"Iteration {iteration}: Initial probs must be in [0, 1]"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
