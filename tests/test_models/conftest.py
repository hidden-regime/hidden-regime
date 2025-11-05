"""
Fixtures specific to HMM model testing.

Provides fixtures for model parameters, initialization, and validation.
"""

import pytest
import numpy as np


@pytest.fixture
def simple_2state_hmm_params():
    """Provide simple 2-state HMM parameters for testing."""
    return {
        'initial_probs': np.array([0.6, 0.4]),
        'transition_matrix': np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ]),
        'emission_params': np.array([
            [-0.01, 0.02],  # State 0: mean=-1%, std=2%
            [0.02, 0.03]    # State 1: mean=2%, std=3%
        ])
    }


@pytest.fixture
def simple_3state_hmm_params():
    """Provide simple 3-state HMM parameters for testing."""
    return {
        'initial_probs': np.array([0.33, 0.34, 0.33]),
        'transition_matrix': np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7]
        ]),
        'emission_params': np.array([
            [-0.02, 0.025],  # Bear regime
            [0.0, 0.015],    # Sideways
            [0.03, 0.02]     # Bull regime
        ])
    }


@pytest.fixture
def deterministic_hmm_params():
    """Provide HMM with deterministic transitions for testing."""
    return {
        'initial_probs': np.array([1.0, 0.0]),
        'transition_matrix': np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ]),
        'emission_params': np.array([
            [0.0, 0.01],
            [0.0, 0.01]
        ])
    }


@pytest.fixture
def sample_log_returns():
    """Provide sample log returns for model training."""
    np.random.seed(42)
    # Create returns with distinct regimes
    returns = np.concatenate([
        np.random.randn(30) * 0.01 - 0.02,  # Bear
        np.random.randn(40) * 0.01,          # Sideways
        np.random.randn(30) * 0.01 + 0.03,  # Bull
    ])
    return returns


@pytest.fixture
def sample_observations_short():
    """Provide short observation sequence for testing."""
    return np.array([0.01, -0.02, 0.03, -0.01, 0.02])


@pytest.fixture
def sample_observations_long():
    """Provide longer observation sequence for testing."""
    np.random.seed(42)
    return np.random.randn(200) * 0.02


@pytest.fixture
def known_path_hmm():
    """Provide HMM designed to have a predictable Viterbi path."""
    return {
        'initial_probs': np.array([0.9, 0.1]),  # Start in state 0
        'transition_matrix': np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ]),
        'emission_params': np.array([
            [-0.05, 0.01],  # Strong negative mean, low std
            [0.05, 0.01]    # Strong positive mean, low std
        ])
    }


@pytest.fixture
def model_training_data():
    """Provide realistic data for model training tests."""
    np.random.seed(42)
    n = 200

    # Create 3-regime sequence
    states = []
    current = 0
    for _ in range(n):
        if np.random.rand() < 0.05:  # 5% chance to switch
            current = (current + 1) % 3
        states.append(current)

    # Generate observations from states
    regime_params = [
        (-0.02, 0.02),   # Bear
        (0.0, 0.01),     # Sideways
        (0.03, 0.025)    # Bull
    ]

    observations = []
    for state in states:
        mean, std = regime_params[state]
        obs = np.random.randn() * std + mean
        observations.append(obs)

    return {
        'observations': np.array(observations),
        'true_states': np.array(states),
        'n_states': 3
    }


@pytest.fixture
def valid_returns_for_kmeans():
    """Provide returns suitable for K-means initialization."""
    np.random.seed(42)
    # Need enough variety for clustering
    returns = np.concatenate([
        np.random.randn(70) * 0.015 - 0.02,  # Bear cluster
        np.random.randn(80) * 0.01,           # Sideways cluster
        np.random.randn(50) * 0.02 + 0.025,  # Bull cluster
    ])
    np.random.shuffle(returns)
    return returns


@pytest.fixture
def edge_case_returns():
    """Provide edge case return data for testing robustness."""
    return {
        'all_zeros': np.zeros(100),
        'all_same': np.ones(100) * 0.01,
        'with_nans': np.array([0.01, np.nan, 0.02, np.nan, 0.03]),
        'with_inf': np.array([0.01, np.inf, 0.02, -np.inf, 0.03]),
        'very_small': np.ones(100) * 1e-10,
        'very_large': np.ones(100) * 10.0
    }
