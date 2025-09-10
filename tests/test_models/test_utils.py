"""
Unit tests for HMM utility functions.

Tests parameter initialization, validation, and helper functions.
"""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from hidden_regime.models.utils import (
    calculate_regime_statistics,
    check_convergence,
    get_regime_interpretation,
    initialize_parameters_kmeans,
    initialize_parameters_random,
    log_normalize,
    normalize_probabilities,
    validate_hmm_parameters,
    validate_returns_data,
)


class TestHMMUtils:
    """Test cases for HMM utility functions."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for testing."""
        np.random.seed(42)
        return np.random.normal(0, 0.02, 100)

    def test_validate_returns_data_numpy_array(self, sample_returns):
        """Test returns data validation with numpy array."""
        result = validate_returns_data(sample_returns)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) == len(sample_returns)
        assert np.array_equal(result, sample_returns)

    def test_validate_returns_data_pandas_series(self, sample_returns):
        """Test returns data validation with pandas Series."""
        series = pd.Series(sample_returns)
        result = validate_returns_data(series)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) == len(sample_returns)
        assert np.array_equal(result, sample_returns)

    def test_validate_returns_data_with_nans(self):
        """Test returns data validation with NaN values."""
        data_with_nans = np.array([1.0, 2.0, np.nan, 3.0, np.inf, -np.inf, 4.0])

        with warnings.catch_warnings(record=True) as w:
            result = validate_returns_data(data_with_nans)

            # Should warn about removing non-finite values
            assert len(w) == 1
            assert "non-finite values" in str(w[0].message)

        # Should return only finite values
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.array_equal(result, expected)

    def test_validate_returns_data_invalid_input(self):
        """Test returns data validation with invalid input."""
        # Empty array
        with pytest.raises(ValueError, match="Returns array cannot be empty"):
            validate_returns_data([])

        # All NaN values
        with pytest.raises(ValueError, match="No valid returns after removing"):
            validate_returns_data([np.nan, np.inf, -np.inf])

        # Multi-dimensional array
        with pytest.raises(ValueError, match="Returns must be a 1D array"):
            validate_returns_data(np.array([[1, 2], [3, 4]]))

    def test_initialize_parameters_random(self, sample_returns):
        """Test random parameter initialization."""
        n_states = 3

        initial_probs, transition_matrix, emission_params = (
            initialize_parameters_random(n_states, sample_returns, random_seed=42)
        )

        # Check shapes
        assert initial_probs.shape == (n_states,)
        assert transition_matrix.shape == (n_states, n_states)
        assert emission_params.shape == (n_states, 2)

        # Check constraints
        assert np.allclose(np.sum(initial_probs), 1.0)
        assert np.allclose(np.sum(transition_matrix, axis=1), 1.0)
        assert np.all(emission_params[:, 1] > 0)  # Positive standard deviations

        # Check that means are sorted (for interpretability)
        assert np.all(emission_params[:-1, 0] <= emission_params[1:, 0])

        # Test reproducibility
        initial_probs2, transition_matrix2, emission_params2 = (
            initialize_parameters_random(n_states, sample_returns, random_seed=42)
        )

        assert np.array_equal(initial_probs, initial_probs2)
        assert np.array_equal(transition_matrix, transition_matrix2)
        assert np.array_equal(emission_params, emission_params2)

    def test_initialize_parameters_kmeans_success(self, sample_returns):
        """Test K-means parameter initialization (successful case)."""
        n_states = 2

        initial_probs, transition_matrix, emission_params = (
            initialize_parameters_kmeans(n_states, sample_returns, random_seed=42)
        )

        # Check shapes
        assert initial_probs.shape == (n_states,)
        assert transition_matrix.shape == (n_states, n_states)
        assert emission_params.shape == (n_states, 2)

        # Check constraints
        assert np.allclose(np.sum(initial_probs), 1.0, atol=1e-10)
        assert np.allclose(np.sum(transition_matrix, axis=1), 1.0, atol=1e-10)
        assert np.all(emission_params[:, 1] > 0)

    def test_initialize_parameters_kmeans_insufficient_data(self):
        """Test K-means initialization with insufficient data."""
        small_data = np.array([1.0, 2.0])  # Only 2 points for 3 states
        n_states = 3

        with warnings.catch_warnings(record=True) as w:
            initial_probs, transition_matrix, emission_params = (
                initialize_parameters_kmeans(n_states, small_data, random_seed=42)
            )

            # Should warn and fall back to random initialization
            assert len(w) == 1
            assert "Insufficient data for K-means" in str(w[0].message)

        # Should still return valid parameters
        assert initial_probs.shape == (n_states,)
        assert transition_matrix.shape == (n_states, n_states)
        assert emission_params.shape == (n_states, 2)

    @patch("hidden_regime.models.utils.KMeans")
    def test_initialize_parameters_kmeans_failure(self, mock_kmeans, sample_returns):
        """Test K-means initialization when clustering fails."""
        # Mock KMeans to raise exception
        mock_kmeans.return_value.fit_predict.side_effect = Exception(
            "Clustering failed"
        )

        n_states = 3

        with warnings.catch_warnings(record=True) as w:
            initial_probs, transition_matrix, emission_params = (
                initialize_parameters_kmeans(n_states, sample_returns, random_seed=42)
            )

            # Should warn and fall back to random initialization
            assert len(w) == 1
            assert "K-means clustering failed" in str(w[0].message)

        # Should still return valid parameters
        assert initial_probs.shape == (n_states,)
        assert np.allclose(np.sum(initial_probs), 1.0)

    def test_check_convergence_not_enough_data(self):
        """Test convergence checking with insufficient data."""
        short_history = [-100.0, -95.0, -90.0]

        result = check_convergence(short_history, tolerance=1e-6, min_iterations=10)
        assert result is False

    def test_check_convergence_converged(self):
        """Test convergence detection when converged."""
        # Create history that shows convergence
        converged_history = [
            -100.0,
            -90.0,
            -85.0,
            -82.0,
            -81.0,
            -80.99,
            -80.98,
            -80.97,
            -80.96,
            -80.959,
            -80.958,
            -80.957,
        ]

        result = check_convergence(converged_history, tolerance=1e-2, min_iterations=10)
        assert result is True

    def test_check_convergence_not_converged(self):
        """Test convergence detection when not converged."""
        # Create history that shows continued improvement
        not_converged_history = [
            -100.0,
            -90.0,
            -80.0,
            -70.0,
            -60.0,
            -50.0,
            -40.0,
            -30.0,
            -20.0,
            -10.0,
            -5.0,
            -2.0,
        ]

        result = check_convergence(
            not_converged_history, tolerance=1e-6, min_iterations=10
        )
        assert result is False

    def test_normalize_probabilities_1d(self):
        """Test probability normalization for 1D array."""
        probs = np.array([2.0, 3.0, 5.0])
        normalized = normalize_probabilities(probs)

        assert np.allclose(np.sum(normalized), 1.0)
        assert np.allclose(normalized, [0.2, 0.3, 0.5])

    def test_normalize_probabilities_2d(self):
        """Test probability normalization for 2D array."""
        probs = np.array([[2.0, 3.0, 5.0], [1.0, 1.0, 2.0]])

        # Normalize rows
        normalized = normalize_probabilities(probs, axis=1)

        assert normalized.shape == probs.shape
        assert np.allclose(np.sum(normalized, axis=1), 1.0)
        assert np.allclose(normalized[0], [0.2, 0.3, 0.5])
        assert np.allclose(normalized[1], [0.25, 0.25, 0.5])

    def test_normalize_probabilities_zero_handling(self):
        """Test probability normalization with zeros."""
        probs = np.array([0.0, 0.0, 0.0])
        normalized = normalize_probabilities(probs)

        # Should add epsilon and normalize
        assert np.allclose(np.sum(normalized), 1.0)
        assert np.all(normalized > 0)

    def test_log_normalize_1d(self):
        """Test log probability normalization for 1D array."""
        log_probs = np.array([-1.0, -2.0, -3.0])
        normalized = log_normalize(log_probs)

        # Check that exp(normalized) sums to 1
        assert np.isclose(np.sum(np.exp(normalized)), 1.0)

        # Check numerical stability
        assert np.all(np.isfinite(normalized))

    def test_log_normalize_2d(self):
        """Test log probability normalization for 2D array."""
        log_probs = np.array([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]])

        normalized = log_normalize(log_probs, axis=1)

        assert normalized.shape == log_probs.shape
        for i in range(2):
            assert np.isclose(np.sum(np.exp(normalized[i])), 1.0)

    def test_log_normalize_extreme_values(self):
        """Test log normalization with extreme values."""
        # Very large differences in log probabilities
        log_probs = np.array([-1000.0, -1.0, -1001.0])
        normalized = log_normalize(log_probs)

        # Should handle extreme values without overflow/underflow
        assert np.all(np.isfinite(normalized))
        assert np.isclose(np.sum(np.exp(normalized)), 1.0)

    def test_validate_hmm_parameters_valid(self):
        """Test HMM parameter validation with valid parameters."""
        initial_probs = np.array([0.3, 0.4, 0.3])
        transition_matrix = np.array(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]]
        )
        emission_params = np.array([[-0.02, 0.03], [0.00, 0.02], [0.01, 0.025]])

        # Should not raise any exceptions
        validate_hmm_parameters(initial_probs, transition_matrix, emission_params)

    def test_validate_hmm_parameters_invalid_initial(self):
        """Test parameter validation with invalid initial probabilities."""
        # Initial probabilities don't sum to 1
        initial_probs = np.array([0.3, 0.4, 0.4])  # Sum = 1.1
        transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        emission_params = np.array([[0.0, 0.1], [0.0, 0.1]])

        with pytest.raises(ValueError, match="Initial probabilities must sum to 1"):
            validate_hmm_parameters(initial_probs, transition_matrix, emission_params)

        # Negative initial probabilities
        initial_probs = np.array([0.5, -0.1, 0.6])

        with pytest.raises(
            ValueError, match="Initial probabilities must be non-negative"
        ):
            validate_hmm_parameters(initial_probs, transition_matrix, emission_params)

    def test_validate_hmm_parameters_invalid_transition(self):
        """Test parameter validation with invalid transition matrix."""
        initial_probs = np.array([0.5, 0.5])
        emission_params = np.array([[0.0, 0.1], [0.0, 0.1]])

        # Wrong shape
        transition_matrix = np.array([[0.7, 0.3]])  # Should be 2x2

        with pytest.raises(ValueError, match="Transition matrix must be"):
            validate_hmm_parameters(initial_probs, transition_matrix, emission_params)

        # Rows don't sum to 1
        transition_matrix = np.array([[0.7, 0.2], [0.4, 0.6]])  # Sum = 0.9

        with pytest.raises(ValueError, match="Transition matrix rows must sum to 1"):
            validate_hmm_parameters(initial_probs, transition_matrix, emission_params)

        # Negative probabilities
        transition_matrix = np.array(
            [[1.1, -0.1], [0.4, 0.6]]  # Sums to 1.0 but has negative value
        )

        with pytest.raises(
            ValueError, match="Transition probabilities must be non-negative"
        ):
            validate_hmm_parameters(initial_probs, transition_matrix, emission_params)

    def test_validate_hmm_parameters_invalid_emission(self):
        """Test parameter validation with invalid emission parameters."""
        initial_probs = np.array([0.5, 0.5])
        transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])

        # Wrong shape
        emission_params = np.array([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2]])  # Should be Nx2

        with pytest.raises(ValueError, match="Emission parameters must be"):
            validate_hmm_parameters(initial_probs, transition_matrix, emission_params)

        # Non-positive standard deviations
        emission_params = np.array([[0.0, 0.0], [0.0, 0.1]])  # First std = 0

        with pytest.raises(
            ValueError, match="Emission standard deviations must be positive"
        ):
            validate_hmm_parameters(initial_probs, transition_matrix, emission_params)

    def test_get_regime_interpretation(self):
        """Test regime interpretation generation."""
        emission_params = np.array(
            [
                [-0.01, 0.03],  # Bear regime
                [0.001, 0.015],  # Sideways regime
                [0.008, 0.025],  # Bull regime
            ]
        )

        interpretations = [
            get_regime_interpretation(i, emission_params) for i in range(3)
        ]

        assert "Bear" in interpretations[0]
        assert "Sideways" in interpretations[1]
        assert "Bull" in interpretations[2]

        # Check volatility descriptions
        assert "Moderate Vol" in interpretations[0]  # 0.03 < 0.035 (High Vol threshold)
        assert "Low Vol" in interpretations[1]  # 0.015 <= 0.015 (Low Vol threshold)
        assert "Moderate Vol" in interpretations[2]  # 0.025 between 0.015 and 0.035

    def test_calculate_regime_statistics(self):
        """Test regime statistics calculation."""
        # Create test data
        states = np.array([0, 0, 1, 1, 1, 2, 2, 0, 0])
        returns = np.array(
            [-0.02, -0.01, 0.005, 0.001, 0.002, 0.01, 0.012, -0.015, -0.008]
        )

        stats = calculate_regime_statistics(states, returns)

        # Check overall structure
        assert "n_observations" in stats
        assert "n_states" in stats
        assert "regime_stats" in stats

        assert stats["n_observations"] == 9
        assert stats["n_states"] == 3

        # Check regime-specific stats
        for regime in range(3):
            regime_stats = stats["regime_stats"][regime]

            assert "frequency" in regime_stats
            assert "mean_return" in regime_stats
            assert "std_return" in regime_stats
            assert "min_return" in regime_stats
            assert "max_return" in regime_stats
            assert "total_periods" in regime_stats
            assert "avg_duration" in regime_stats
            assert "min_duration" in regime_stats
            assert "max_duration" in regime_stats
            assert "n_episodes" in regime_stats

        # Verify specific calculations for regime 0 (appears at positions 0,1,7,8)
        regime_0_stats = stats["regime_stats"][0]
        assert regime_0_stats["total_periods"] == 4
        assert regime_0_stats["n_episodes"] == 2  # Two separate episodes
        assert regime_0_stats["frequency"] == 4 / 9

        # Check duration calculations (regime 0: episodes of length 2, 2)
        assert regime_0_stats["avg_duration"] == 2.0
        assert regime_0_stats["min_duration"] == 2
        assert regime_0_stats["max_duration"] == 2

    def test_calculate_regime_statistics_with_dates(self):
        """Test regime statistics calculation with dates."""
        states = np.array([0, 1, 1, 0])
        returns = np.array([0.01, -0.01, 0.005, 0.002])
        dates = pd.date_range("2023-01-01", periods=4, freq="D")

        stats = calculate_regime_statistics(states, returns, dates)

        # Should work the same as without dates (dates not used in current implementation)
        assert stats["n_observations"] == 4
        assert len(stats["regime_stats"]) == 2

    def test_calculate_regime_statistics_single_regime(self):
        """Test statistics calculation with only one regime."""
        states = np.array([0, 0, 0, 0])
        returns = np.array([0.01, -0.01, 0.005, 0.002])

        stats = calculate_regime_statistics(states, returns)

        assert stats["n_states"] == 1
        assert 0 in stats["regime_stats"]

        regime_stats = stats["regime_stats"][0]
        assert regime_stats["frequency"] == 1.0
        assert regime_stats["total_periods"] == 4
        assert regime_stats["n_episodes"] == 1
        assert regime_stats["avg_duration"] == 4.0
