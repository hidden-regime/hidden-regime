"""
Shared pytest fixtures and configuration for the entire test suite.

This module provides common fixtures available to all tests,
including sample data, mock objects, and configuration helpers.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock


# ============================================================================
# Sample Data Fixtures
# ============================================================================


@pytest.fixture
def sample_dates():
    """Generate sample date range for testing."""
    start = datetime(2020, 1, 1)
    return pd.date_range(start=start, periods=100, freq='D')


@pytest.fixture
def sample_returns(sample_dates):
    """Generate sample log returns for testing."""
    np.random.seed(42)
    returns = np.random.randn(len(sample_dates)) * 0.02
    return pd.Series(returns, index=sample_dates, name='returns')


@pytest.fixture
def sample_prices(sample_dates):
    """Generate sample price series for testing."""
    np.random.seed(42)
    # Start at 100 and do random walk
    returns = np.random.randn(len(sample_dates)) * 0.02
    log_returns = np.log(1 + returns)
    prices = 100 * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, index=sample_dates, name='close')


@pytest.fixture
def sample_ohlc_data(sample_dates):
    """Generate sample OHLC data for testing."""
    np.random.seed(42)
    n = len(sample_dates)

    # Generate close prices
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_price = np.roll(close, 1)
    open_price[0] = close[0]

    volume = np.random.randint(1000000, 10000000, n)

    return pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=sample_dates)


@pytest.fixture
def sample_regime_sequence():
    """Generate sample regime sequence (states) for testing."""
    np.random.seed(42)
    # Create regime sequence with some persistence
    states = []
    current_state = 0
    for _ in range(100):
        if np.random.rand() < 0.1:  # 10% chance to switch
            current_state = (current_state + 1) % 3
        states.append(current_state)
    return np.array(states)


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def simple_hmm_config():
    """Provide simple HMM configuration for testing."""
    return {
        'n_states': 3,
        'initialization_method': 'random',
        'random_seed': 42,
        'max_iterations': 100,
        'convergence_threshold': 0.01
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide temporary directory for output files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# ============================================================================
# Mock Object Fixtures
# ============================================================================


@pytest.fixture
def mock_yfinance_ticker():
    """Provide mock yfinance Ticker object."""
    mock_ticker = MagicMock()

    # Mock history method
    def mock_history(period=None, start=None, end=None, **kwargs):
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        close = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))

        return pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.01,
            'Low': close * 0.98,
            'Close': close,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

    mock_ticker.history = Mock(side_effect=mock_history)
    return mock_ticker


@pytest.fixture
def mock_hmm_model():
    """Provide mock HMM model for testing."""
    mock_model = MagicMock()
    mock_model.n_states = 3
    mock_model.is_fitted = True
    mock_model.initial_probs = np.array([0.33, 0.34, 0.33])
    mock_model.transition_matrix = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.1, 0.7]
    ])
    mock_model.emission_params = np.array([
        [-0.02, 0.025],  # Bear regime
        [0.0, 0.015],    # Sideways
        [0.03, 0.02]     # Bull regime
    ])

    def mock_predict(observations):
        np.random.seed(42)
        return np.random.randint(0, 3, len(observations))

    mock_model.predict = Mock(side_effect=mock_predict)
    return mock_model


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Markers are defined in pyproject.toml
    pass


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test characteristics."""
    for item in items:
        # Add 'fast' marker to unit tests automatically
        if 'unit' in item.keywords:
            item.add_marker(pytest.mark.fast)

        # Add 'slow' marker to e2e tests automatically
        if 'e2e' in item.keywords and 'slow' not in item.keywords:
            item.add_marker(pytest.mark.slow)


# ============================================================================
# Helpers
# ============================================================================


@pytest.fixture
def assert_array_shape():
    """Provide helper to assert array shapes."""
    def _assert_shape(array, expected_shape):
        assert array.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {array.shape}"
    return _assert_shape


@pytest.fixture
def assert_probabilities_valid():
    """Provide helper to validate probability arrays."""
    def _assert_valid(probs, axis=None):
        # Check all values are in [0, 1]
        assert np.all(probs >= 0), "Probabilities must be non-negative"
        assert np.all(probs <= 1), "Probabilities must be <= 1"

        # Check sums to 1
        if axis is None:
            np.testing.assert_almost_equal(np.sum(probs), 1.0, decimal=5)
        else:
            sums = np.sum(probs, axis=axis)
            np.testing.assert_array_almost_equal(sums, np.ones(sums.shape), decimal=5)

    return _assert_valid
