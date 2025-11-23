"""
Unit tests for HMM utility functions.

Tests parameter validation, initialization, convergence checking,
probability normalization, and regime interpretation functions.
"""

import pytest
import numpy as np
import pandas as pd
import warnings

from hidden_regime.models.utils import (
    validate_returns_data,
    initialize_parameters_random,
    initialize_parameters_kmeans,
    check_convergence,
    normalize_probabilities,
    log_normalize,
    validate_hmm_parameters,
    validate_state_quality,
    analyze_state_transitions,
    calculate_state_statistics,
)


# ============================================================================
# Test validate_returns_data
# ============================================================================


@pytest.mark.unit
class TestValidateReturnsData:
    """Test returns data validation."""

    def test_accepts_numpy_array(self):
        """Test that function accepts numpy arrays."""
        returns = np.array([0.01, -0.02, 0.03])
        result = validate_returns_data(returns)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, returns)

    def test_converts_pandas_series(self):
        """Test that pandas Series are converted to numpy arrays."""
        returns = pd.Series([0.01, -0.02, 0.03])
        result = validate_returns_data(returns)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, returns.values)

    def test_converts_list(self):
        """Test that lists are converted to numpy arrays."""
        returns = [0.01, -0.02, 0.03]
        result = validate_returns_data(returns)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(returns))

    def test_raises_on_empty_array(self):
        """Test that empty array raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_returns_data(np.array([]))

    def test_raises_on_multidimensional_array(self):
        """Test that multidimensional arrays raise ValueError."""
        returns = np.array([[0.01, 0.02], [0.03, 0.04]])
        with pytest.raises(ValueError, match="must be a 1D array"):
            validate_returns_data(returns)

    def test_removes_nan_values(self):
        """Test that NaN values are removed with warning."""
        returns = np.array([0.01, np.nan, 0.03, np.nan, 0.05])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_returns_data(returns)

            assert len(w) == 1
            assert "non-finite" in str(w[0].message).lower()

        expected = np.array([0.01, 0.03, 0.05])
        np.testing.assert_array_equal(result, expected)

    def test_removes_inf_values(self):
        """Test that infinite values are removed."""
        returns = np.array([0.01, np.inf, 0.03, -np.inf, 0.05])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = validate_returns_data(returns)

        expected = np.array([0.01, 0.03, 0.05])
        np.testing.assert_array_equal(result, expected)

    def test_raises_when_all_values_invalid(self):
        """Test that ValueError is raised when all values are removed."""
        returns = np.array([np.nan, np.inf, -np.inf])

        with pytest.raises(ValueError, match="No valid returns"):
            validate_returns_data(returns)

    def test_output_is_float64(self):
        """Test that output is converted to float64."""
        returns = np.array([1, 2, 3], dtype=np.int32)
        result = validate_returns_data(returns)

        assert result.dtype == np.float64


# ============================================================================
# Test initialize_parameters_random
# ============================================================================


@pytest.mark.unit
class TestInitializeParametersRandom:
    """Test random parameter initialization."""

    def test_returns_three_arrays(self):
        """Test that function returns three arrays."""
        returns = np.random.randn(100) * 0.02
        result = initialize_parameters_random(2, returns, random_seed=42)

        assert len(result) == 3
        initial_probs, transition_matrix, emission_params = result
        assert isinstance(initial_probs, np.ndarray)
        assert isinstance(transition_matrix, np.ndarray)
        assert isinstance(emission_params, np.ndarray)

    def test_initial_probs_shape(self):
        """Test initial probabilities have correct shape."""
        returns = np.random.randn(100) * 0.02
        initial_probs, _, _ = initialize_parameters_random(3, returns, random_seed=42)

        assert initial_probs.shape == (3,)

    def test_initial_probs_sum_to_one(self):
        """Test initial probabilities sum to 1."""
        returns = np.random.randn(100) * 0.02
        initial_probs, _, _ = initialize_parameters_random(2, returns, random_seed=42)

        np.testing.assert_almost_equal(np.sum(initial_probs), 1.0)

    def test_transition_matrix_shape(self):
        """Test transition matrix has correct shape."""
        returns = np.random.randn(100) * 0.02
        _, transition_matrix, _ = initialize_parameters_random(3, returns, random_seed=42)

        assert transition_matrix.shape == (3, 3)

    def test_transition_matrix_rows_sum_to_one(self):
        """Test transition matrix rows sum to 1."""
        returns = np.random.randn(100) * 0.02
        _, transition_matrix, _ = initialize_parameters_random(2, returns, random_seed=42)

        row_sums = np.sum(transition_matrix, axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0])

    def test_emission_params_shape(self):
        """Test emission parameters have correct shape."""
        returns = np.random.randn(100) * 0.02
        _, _, emission_params = initialize_parameters_random(3, returns, random_seed=42)

        assert emission_params.shape == (3, 2), "Should be (n_states, 2) for [mean, std]"

    def test_emission_stds_are_positive(self):
        """Test that emission standard deviations are positive."""
        returns = np.random.randn(100) * 0.02
        _, _, emission_params = initialize_parameters_random(2, returns, random_seed=42)

        assert np.all(emission_params[:, 1] > 0), "Standard deviations must be positive"

    def test_emission_means_are_sorted(self):
        """Test that emission means are sorted (for interpretability)."""
        returns = np.random.randn(100) * 0.02
        _, _, emission_params = initialize_parameters_random(3, returns, random_seed=42)

        means = emission_params[:, 0]
        assert np.all(means[:-1] <= means[1:]), "Means should be sorted"

    def test_random_seed_reproducibility(self):
        """Test that random seed produces reproducible results."""
        returns = np.random.randn(100) * 0.02

        result1 = initialize_parameters_random(2, returns, random_seed=42)
        result2 = initialize_parameters_random(2, returns, random_seed=42)

        for arr1, arr2 in zip(result1, result2):
            np.testing.assert_array_equal(arr1, arr2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        returns = np.random.randn(100) * 0.02

        result1 = initialize_parameters_random(2, returns, random_seed=42)
        result2 = initialize_parameters_random(2, returns, random_seed=123)

        # At least one parameter should be different
        assert not np.array_equal(result1[0], result2[0]) or \
               not np.array_equal(result1[1], result2[1]) or \
               not np.array_equal(result1[2], result2[2])

    def test_diagonal_bias_in_transition_matrix(self):
        """Test that transition matrix has diagonal bias (favor staying in state)."""
        returns = np.random.randn(100) * 0.02
        _, transition_matrix, _ = initialize_parameters_random(2, returns, random_seed=42)

        # Diagonal elements should generally be larger than off-diagonal
        # (though not guaranteed in every random case)
        diagonal_avg = np.mean(np.diag(transition_matrix))
        off_diagonal_avg = np.mean(transition_matrix[~np.eye(2, dtype=bool)])

        assert diagonal_avg >= off_diagonal_avg


# ============================================================================
# Test initialize_parameters_kmeans
# ============================================================================


@pytest.mark.unit
class TestInitializeParametersKMeans:
    """Test K-means parameter initialization."""

    def test_returns_four_items(self):
        """Test that function returns four items (params + diagnostics)."""
        returns = np.random.randn(200) * 0.02
        result = initialize_parameters_kmeans(2, returns, random_seed=42)

        assert len(result) == 4
        initial_probs, transition_matrix, emission_params, diagnostics = result
        assert isinstance(diagnostics, dict)

    def test_initial_probs_sum_to_one(self):
        """Test initial probabilities sum to 1."""
        returns = np.random.randn(200) * 0.02
        initial_probs, _, _, _ = initialize_parameters_kmeans(2, returns, random_seed=42)

        np.testing.assert_almost_equal(np.sum(initial_probs), 1.0)

    def test_transition_matrix_rows_sum_to_one(self):
        """Test transition matrix rows sum to 1."""
        returns = np.random.randn(200) * 0.02
        _, transition_matrix, _, _ = initialize_parameters_kmeans(2, returns, random_seed=42)

        row_sums = np.sum(transition_matrix, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(2))

    def test_emission_params_shape(self):
        """Test emission parameters have correct shape."""
        returns = np.random.randn(200) * 0.02
        _, _, emission_params, _ = initialize_parameters_kmeans(3, returns, random_seed=42)

        assert emission_params.shape == (3, 2)

    def test_diagnostics_contains_expected_keys(self):
        """Test that diagnostics dictionary contains expected keys."""
        returns = np.random.randn(200) * 0.02
        _, _, _, diagnostics = initialize_parameters_kmeans(2, returns, random_seed=42)

        expected_keys = ['method', 'n_states', 'n_observations', 'warnings']
        for key in expected_keys:
            assert key in diagnostics

    def test_fallback_to_random_with_insufficient_data(self):
        """Test fallback to random initialization with insufficient data."""
        returns = np.array([0.01, 0.02])  # Only 2 points
        _, _, _, diagnostics = initialize_parameters_kmeans(3, returns, random_seed=42)

        assert diagnostics['method'] == 'random_fallback'
        assert diagnostics['reason'] == 'insufficient_data'

    def test_emission_means_are_sorted(self):
        """Test that emission means are sorted (bear, sideways, bull)."""
        returns = np.random.randn(200) * 0.02
        _, _, emission_params, _ = initialize_parameters_kmeans(3, returns, random_seed=42)

        means = emission_params[:, 0]
        assert np.all(means[:-1] <= means[1:]), "Means should be sorted"

    def test_emission_stds_are_positive(self):
        """Test that emission standard deviations are positive."""
        returns = np.random.randn(200) * 0.02
        _, _, emission_params, _ = initialize_parameters_kmeans(2, returns, random_seed=42)

        assert np.all(emission_params[:, 1] > 0)

    def test_reproducibility_with_seed(self):
        """Test that random seed produces reproducible results."""
        returns = np.random.randn(200) * 0.02

        result1 = initialize_parameters_kmeans(2, returns, random_seed=42)
        result2 = initialize_parameters_kmeans(2, returns, random_seed=42)

        for arr1, arr2 in zip(result1[:3], result2[:3]):
            np.testing.assert_array_almost_equal(arr1, arr2)

    def test_financial_constraints_applied(self):
        """Test that financial constraints are applied to means."""
        # Create data with extreme outliers that should be constrained
        returns = np.concatenate([
            np.random.randn(80) * 0.01 - 0.20,  # Extreme bear: -20%
            np.random.randn(80) * 0.01 + 0.20,  # Extreme bull: +20%
        ])

        _, _, emission_params, diagnostics = initialize_parameters_kmeans(2, returns, random_seed=42)

        means_pct = np.exp(emission_params[:, 0]) - 1

        # Means should be constrained within reasonable bounds
        # Bear: -8% to -0.1%, Bull: 0.1% to 5%
        assert means_pct[0] >= -0.08, "Bear regime should not be more negative than -8%"
        assert means_pct[1] <= 0.05 + 1e-10, "Bull regime should not be more positive than 5%"  # Allow small floating point error

    def test_transition_matrix_no_zeros(self):
        """Test that transition matrix has no zero probabilities (due to regularization)."""
        returns = np.random.randn(200) * 0.02
        _, transition_matrix, _, _ = initialize_parameters_kmeans(2, returns, random_seed=42)

        assert np.all(transition_matrix > 0), "Regularization should prevent zeros"

    def test_warnings_in_diagnostics(self):
        """Test that diagnostics includes warnings list."""
        returns = np.random.randn(200) * 0.02
        _, _, _, diagnostics = initialize_parameters_kmeans(2, returns, random_seed=42)

        assert 'warnings' in diagnostics
        assert isinstance(diagnostics['warnings'], list)


# ============================================================================
# Test check_convergence
# ============================================================================


@pytest.mark.unit
class TestCheckConvergence:
    """Test convergence checking."""

    def test_not_converged_before_min_iterations(self):
        """Test that convergence returns False before min iterations."""
        log_likelihoods = [-100.0, -95.0, -92.0, -90.0, -89.5]
        result = check_convergence(log_likelihoods, tolerance=0.1, min_iterations=10)

        assert result is False

    def test_converged_when_improvement_small(self):
        """Test convergence when improvement is below tolerance."""
        # Need at least 5 small improvements (function checks last 5)
        log_likelihoods = [-100.0] * 5 + [-99.9, -99.89, -99.885, -99.882, -99.881, -99.8805, -99.8803]
        result = check_convergence(log_likelihoods, tolerance=0.01, min_iterations=10)

        assert result is True

    def test_not_converged_when_still_improving(self):
        """Test no convergence when still improving significantly."""
        log_likelihoods = [-100.0, -95.0, -90.0, -85.0, -80.0, -75.0, -70.0, -65.0, -60.0, -55.0, -50.0, -45.0]
        result = check_convergence(log_likelihoods, tolerance=0.1, min_iterations=10)

        assert result is False

    def test_with_exact_min_iterations(self):
        """Test with exactly minimum iterations."""
        # 11 values = 10 improvements
        log_likelihoods = [-100.0, -99.5, -99.4, -99.35, -99.33, -99.32, -99.315, -99.313, -99.312, -99.311, -99.310]
        result = check_convergence(log_likelihoods, tolerance=0.01, min_iterations=10)

        # Should check convergence at this point
        assert isinstance(result, bool)

    def test_handles_non_monotonic_improvement(self):
        """Test handling of non-monotonic likelihood (shouldn't happen but test robustness)."""
        log_likelihoods = [-100.0, -95.0, -96.0, -94.0, -93.0, -93.1, -92.9, -92.85, -92.84, -92.83, -92.82]
        result = check_convergence(log_likelihoods, tolerance=0.1, min_iterations=10)

        # Should handle absolute improvements
        assert isinstance(result, bool)


# ============================================================================
# Test normalize_probabilities
# ============================================================================


@pytest.mark.unit
class TestNormalizeProbabilities:
    """Test probability normalization."""

    def test_normalizes_1d_array(self):
        """Test normalization of 1D array."""
        probs = np.array([0.3, 0.5, 0.2])
        result = normalize_probabilities(probs)

        np.testing.assert_almost_equal(np.sum(result), 1.0)
        np.testing.assert_array_almost_equal(result, probs)  # Already normalized

    def test_normalizes_unnormalized_array(self):
        """Test normalization of unnormalized array."""
        probs = np.array([1.0, 2.0, 3.0])
        result = normalize_probabilities(probs)

        np.testing.assert_almost_equal(np.sum(result), 1.0)
        expected = np.array([1/6, 2/6, 3/6])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalizes_matrix_rows(self):
        """Test normalization along rows (axis=1)."""
        probs = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        result = normalize_probabilities(probs, axis=1)

        # Each row should sum to 1
        row_sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0])

    def test_handles_zero_probabilities(self):
        """Test handling of zero probabilities with epsilon."""
        probs = np.array([0.0, 0.0, 1.0])
        result = normalize_probabilities(probs)

        # Should add epsilon to avoid division by zero
        assert np.all(result > 0)
        np.testing.assert_almost_equal(np.sum(result), 1.0)

    def test_handles_all_zeros(self):
        """Test handling of all-zero array."""
        probs = np.array([0.0, 0.0, 0.0])
        result = normalize_probabilities(probs)

        # Should become uniform due to epsilon
        np.testing.assert_almost_equal(np.sum(result), 1.0)
        np.testing.assert_array_almost_equal(result, [1/3, 1/3, 1/3], decimal=5)


# ============================================================================
# Test log_normalize
# ============================================================================


@pytest.mark.unit
class TestLogNormalize:
    """Test log probability normalization."""

    def test_log_probabilities_sum_to_one_in_prob_space(self):
        """Test that normalized log probs sum to 1 in probability space."""
        log_probs = np.array([-1.0, -2.0, -3.0])
        result = log_normalize(log_probs)

        # Convert to probability space and check sum
        probs = np.exp(result)
        np.testing.assert_almost_equal(np.sum(probs), 1.0)

    def test_handles_large_negative_values(self):
        """Test numerical stability with large negative values."""
        log_probs = np.array([-1000.0, -1001.0, -1002.0])
        result = log_normalize(log_probs)

        # Should not overflow or underflow
        assert np.all(np.isfinite(result))
        probs = np.exp(result)
        np.testing.assert_almost_equal(np.sum(probs), 1.0)

    def test_normalizes_matrix_rows(self):
        """Test normalization along rows in log space."""
        log_probs = np.array([
            [-1.0, -2.0, -3.0],
            [-2.0, -3.0, -4.0]
        ])
        result = log_normalize(log_probs, axis=1)

        # Each row should sum to 1 in probability space
        for row in result:
            probs = np.exp(row)
            np.testing.assert_almost_equal(np.sum(probs), 1.0)

    def test_preserves_relative_ordering(self):
        """Test that normalization preserves relative ordering."""
        log_probs = np.array([-1.0, -2.0, -3.0])
        result = log_normalize(log_probs)

        # Order should be preserved
        assert result[0] > result[1] > result[2]


# ============================================================================
# Test validate_hmm_parameters
# ============================================================================


@pytest.mark.unit
class TestValidateHMMParameters:
    """Test HMM parameter validation."""

    def test_accepts_valid_parameters(self):
        """Test that valid parameters pass validation."""
        initial_probs = np.array([0.6, 0.4])
        transition_matrix = np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ])
        emission_params = np.array([
            [-0.01, 0.02],
            [0.02, 0.03]
        ])

        # Should not raise
        validate_hmm_parameters(initial_probs, transition_matrix, emission_params)

    def test_raises_on_initial_probs_not_sum_to_one(self):
        """Test that invalid initial probs raise error."""
        initial_probs = np.array([0.6, 0.5])  # Sums to 1.1
        transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        emission_params = np.array([[-0.01, 0.02], [0.02, 0.03]])

        with pytest.raises(ValueError, match="must sum to 1"):
            validate_hmm_parameters(initial_probs, transition_matrix, emission_params)

    def test_raises_on_negative_initial_probs(self):
        """Test that negative initial probs raise error."""
        initial_probs = np.array([1.2, -0.2])
        transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        emission_params = np.array([[-0.01, 0.02], [0.02, 0.03]])

        with pytest.raises(ValueError, match="must be non-negative"):
            validate_hmm_parameters(initial_probs, transition_matrix, emission_params)

    def test_raises_on_wrong_transition_matrix_shape(self):
        """Test that wrong transition matrix shape raises error."""
        initial_probs = np.array([0.6, 0.4])
        transition_matrix = np.array([[0.7, 0.3]])  # Wrong shape
        emission_params = np.array([[-0.01, 0.02], [0.02, 0.03]])

        with pytest.raises(ValueError, match="must be 2x2"):
            validate_hmm_parameters(initial_probs, transition_matrix, emission_params)

    def test_raises_on_transition_rows_not_sum_to_one(self):
        """Test that invalid transition matrix raises error."""
        initial_probs = np.array([0.6, 0.4])
        transition_matrix = np.array([
            [0.7, 0.4],  # Sums to 1.1
            [0.4, 0.6]
        ])
        emission_params = np.array([[-0.01, 0.02], [0.02, 0.03]])

        with pytest.raises(ValueError, match="rows must sum to 1"):
            validate_hmm_parameters(initial_probs, transition_matrix, emission_params)

    def test_raises_on_negative_emission_std(self):
        """Test that negative emission std raises error."""
        initial_probs = np.array([0.6, 0.4])
        transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        emission_params = np.array([
            [-0.01, 0.02],
            [0.02, -0.01]  # Negative std
        ])

        with pytest.raises(ValueError, match="must be positive"):
            validate_hmm_parameters(initial_probs, transition_matrix, emission_params)

    def test_raises_on_wrong_emission_params_shape(self):
        """Test that wrong emission params shape raises error."""
        initial_probs = np.array([0.6, 0.4])
        transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        emission_params = np.array([[-0.01], [0.02]])  # Missing std column

        with pytest.raises(ValueError, match="must be 2x2"):
            validate_hmm_parameters(initial_probs, transition_matrix, emission_params)


# ============================================================================
# Test validate_state_quality
# ============================================================================


@pytest.mark.unit
class TestValidateRegimeEconomics:
    """Test regime economic validation."""

    def test_valid_regimes_pass(self):
        """Test that economically valid regimes pass validation."""
        emission_params = np.array([
            [-0.02, 0.025],  # Bear
            [0.001, 0.015],  # Sideways
            [0.025, 0.02]    # Bull
        ])

        is_valid, details = validate_state_quality(emission_params, '3_state')

        assert is_valid is True
        assert details['mean_ordering_correct'] is True

    def test_detects_unordered_means(self):
        """Test that unordered means are detected."""
        emission_params = np.array([
            [0.02, 0.02],   # Should be lowest but isn't
            [-0.02, 0.025],
            [0.01, 0.015]
        ])

        is_valid, details = validate_state_quality(emission_params)

        assert is_valid is False
        assert details['mean_ordering_correct'] is False
        assert any("not properly ordered" in v for v in details['violations'])

    def test_detects_poor_separation(self):
        """Test that poorly separated regimes are detected."""
        emission_params = np.array([
            [0.0, 0.02],
            [0.0001, 0.02],  # Very close to state 0
            [0.0002, 0.02]
        ])

        is_valid, details = validate_state_quality(emission_params)

        # Should detect poor separation
        assert len(details['violations']) > 0

    def test_detects_unrealistic_volatility(self):
        """Test that unrealistic volatility is detected."""
        emission_params = np.array([
            [-0.02, 0.0001],  # Unrealistically low volatility
            [0.0, 0.02],
            [0.02, 0.02]
        ])

        is_valid, details = validate_state_quality(emission_params)

        assert details['volatility_reasonable'] is False

    def test_checks_for_bear_and_bull_regimes(self):
        """Test that validation checks for presence of bear and bull regimes."""
        # All slightly positive (no clear bear)
        emission_params = np.array([
            [0.005, 0.02],
            [0.01, 0.02],
            [0.02, 0.02]
        ])

        is_valid, details = validate_state_quality(emission_params, '3_state')

        # Should warn about no clear bear regime
        if not is_valid:
            assert any("bear" in v.lower() for v in details['violations'])

    def test_returns_regime_separation_details(self):
        """Test that validation returns separation details."""
        emission_params = np.array([
            [-0.02, 0.02],
            [0.0, 0.02],
            [0.02, 0.02]
        ])

        _, details = validate_state_quality(emission_params)

        assert 'regime_separation' in details
        assert len(details['regime_separation']) > 0


# ============================================================================
# Test analyze_state_transitions
# ============================================================================


@pytest.mark.unit
class TestAnalyzeRegimeTransitions:
    """Test regime transition analysis."""

    def test_returns_expected_keys(self):
        """Test that analysis returns expected dictionary keys."""
        states = np.array([0, 0, 1, 1, 1, 0, 0])
        transition_matrix = np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ])

        result = analyze_state_transitions(states, transition_matrix)

        expected_keys = ['persistence_analysis', 'transition_patterns',
                        'stability_metrics', 'empirical_transition_matrix']
        for key in expected_keys:
            assert key in result

    def test_calculates_empirical_transitions(self):
        """Test that empirical transitions are calculated correctly."""
        states = np.array([0, 0, 1, 1, 0])  # 0->0, 0->1, 1->1, 1->0
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])

        result = analyze_state_transitions(states, transition_matrix)
        empirical = np.array(result['empirical_transition_matrix'])

        # Expected: 0->0 (1 time), 0->1 (1 time), 1->1 (1 time), 1->0 (1 time)
        # From state 0: 1 to 0, 1 to 1 = [0.5, 0.5]
        # From state 1: 1 to 0, 1 to 1 = [0.5, 0.5]
        np.testing.assert_array_almost_equal(empirical, [[0.5, 0.5], [0.5, 0.5]], decimal=1)

    def test_calculates_expected_duration(self):
        """Test that expected duration is calculated correctly."""
        states = np.array([0, 0, 1, 1, 1])
        transition_matrix = np.array([
            [0.8, 0.2],  # Expected duration: 1/(1-0.8) = 5
            [0.3, 0.7]   # Expected duration: 1/(1-0.7) = 3.33
        ])

        result = analyze_state_transitions(states, transition_matrix)
        persistence = result['persistence_analysis']

        assert 'Regime 0' in persistence
        assert 'Regime 1' in persistence

        np.testing.assert_almost_equal(
            persistence['Regime 0']['expected_duration'], 5.0, decimal=1
        )
        np.testing.assert_almost_equal(
            persistence['Regime 1']['expected_duration'], 3.33, decimal=2
        )

    def test_identifies_most_stable_regime(self):
        """Test that most stable regime is identified."""
        states = np.array([0, 0, 0, 1, 1, 0])
        transition_matrix = np.array([
            [0.9, 0.1],  # More stable
            [0.4, 0.6]   # Less stable
        ])

        result = analyze_state_transitions(states, transition_matrix)
        stability = result['stability_metrics']

        assert stability['most_stable_regime'] == 'Regime 0'
        assert stability['least_stable_regime'] == 'Regime 1'


# ============================================================================
# Test calculate_state_statistics
# ============================================================================


@pytest.mark.unit
class TestCalculateRegimeStatistics:
    """Test regime statistics calculation."""

    def test_returns_stats_for_each_regime(self):
        """Test that statistics are returned for each regime."""
        states = np.array([0, 0, 1, 1, 1, 0])
        returns = np.array([0.01, 0.02, -0.01, -0.02, -0.01, 0.01])

        stats = calculate_state_statistics(states, returns)

        assert 'regime_stats' in stats
        assert 0 in stats['regime_stats']
        assert 1 in stats['regime_stats']

    def test_calculates_frequency(self):
        """Test that regime frequency is calculated correctly."""
        states = np.array([0, 0, 1, 1, 1, 0])  # 3 of state 0, 3 of state 1
        returns = np.array([0.01, 0.02, -0.01, -0.02, -0.01, 0.01])

        stats = calculate_state_statistics(states, returns)

        assert stats['regime_stats'][0]['frequency'] == 0.5
        assert stats['regime_stats'][1]['frequency'] == 0.5

    def test_calculates_mean_return(self):
        """Test that mean return is calculated correctly."""
        states = np.array([0, 0, 1, 1])
        returns = np.array([0.01, 0.02, 0.03, 0.04])

        stats = calculate_state_statistics(states, returns)

        np.testing.assert_almost_equal(
            stats['regime_stats'][0]['mean_return'], 0.015  # (0.01 + 0.02) / 2
        )
        np.testing.assert_almost_equal(
            stats['regime_stats'][1]['mean_return'], 0.035  # (0.03 + 0.04) / 2
        )

    def test_calculates_duration_statistics(self):
        """Test that duration statistics are calculated."""
        states = np.array([0, 0, 0, 1, 1, 0, 0])  # State 0: [3, 2], State 1: [2]
        returns = np.zeros(7)

        stats = calculate_state_statistics(states, returns)

        # State 0: 2 episodes with durations [3, 2]
        assert stats['regime_stats'][0]['n_episodes'] == 2
        np.testing.assert_almost_equal(stats['regime_stats'][0]['avg_duration'], 2.5)
        assert stats['regime_stats'][0]['min_duration'] == 2
        assert stats['regime_stats'][0]['max_duration'] == 3

        # State 1: 1 episode with duration [2]
        assert stats['regime_stats'][1]['n_episodes'] == 1
        assert stats['regime_stats'][1]['avg_duration'] == 2

    def test_handles_single_observation_regime(self):
        """Test handling of regime with single observation."""
        states = np.array([0, 1, 0, 0])
        returns = np.array([0.01, 0.02, 0.03, 0.04])

        stats = calculate_state_statistics(states, returns)

        # State 1 appears only once
        assert stats['regime_stats'][1]['total_periods'] == 1
        assert stats['regime_stats'][1]['avg_duration'] == 1

    def test_includes_min_max_returns(self):
        """Test that min/max returns are included."""
        states = np.array([0, 0, 0])
        returns = np.array([0.01, 0.05, 0.02])

        stats = calculate_state_statistics(states, returns)

        assert stats['regime_stats'][0]['min_return'] == 0.01
        assert stats['regime_stats'][0]['max_return'] == 0.05
