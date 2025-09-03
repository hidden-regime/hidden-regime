"""
Utility functions for Hidden Markov Model implementation.

Provides helper functions for parameter validation, initialization,
convergence checking, and numerical stability.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import warnings

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. K-means initialization will fall back to random.")


def validate_returns_data(returns: np.ndarray) -> np.ndarray:
    """
    Validate and prepare returns data for HMM training.
    
    Args:
        returns: Array of log returns
        
    Returns:
        Validated and cleaned returns array
        
    Raises:
        ValueError: If data is invalid for HMM training
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    returns = np.asarray(returns, dtype=np.float64)
    
    if returns.ndim != 1:
        raise ValueError("Returns must be a 1D array")
    
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")
    
    # Remove NaN values
    finite_mask = np.isfinite(returns)
    if not finite_mask.all():
        n_removed = (~finite_mask).sum()
        warnings.warn(f"Removed {n_removed} non-finite values from returns data")
        returns = returns[finite_mask]
    
    if len(returns) == 0:
        raise ValueError("No valid returns after removing non-finite values")
    
    return returns


def initialize_parameters_random(
    n_states: int, 
    returns: np.ndarray, 
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize HMM parameters randomly.
    
    Args:
        n_states: Number of hidden states
        returns: Returns data for parameter scaling
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (initial_probs, transition_matrix, emission_params)
        emission_params is array of shape (n_states, 2) with [mean, std]
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Initial state probabilities (uniform)
    initial_probs = np.ones(n_states) / n_states
    
    # Transition matrix (slightly favor staying in same state)
    transition_matrix = np.random.rand(n_states, n_states)
    # Add diagonal bias
    np.fill_diagonal(transition_matrix, transition_matrix.diagonal() + 0.5)
    # Normalize rows
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    
    # Emission parameters
    returns_mean = np.mean(returns)
    returns_std = np.std(returns)
    
    # Generate means around overall mean with some spread
    means = np.random.normal(returns_mean, returns_std * 0.5, n_states)
    means = np.sort(means)  # Sort for interpretability (bear, sideways, bull)
    
    # Generate standard deviations
    stds = np.random.uniform(returns_std * 0.5, returns_std * 1.5, n_states)
    
    emission_params = np.column_stack([means, stds])
    
    return initial_probs, transition_matrix, emission_params


def initialize_parameters_kmeans(
    n_states: int,
    returns: np.ndarray,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize HMM parameters using K-means clustering.
    
    Args:
        n_states: Number of hidden states
        returns: Returns data
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (initial_probs, transition_matrix, emission_params)
    """
    if not SKLEARN_AVAILABLE:
        warnings.warn("scikit-learn not available, falling back to random initialization")
        return initialize_parameters_random(n_states, returns, random_seed)
        
    if len(returns) < n_states:
        warnings.warn("Insufficient data for K-means initialization, falling back to random")
        return initialize_parameters_random(n_states, returns, random_seed)
    
    # Prepare data for clustering (use returns and lagged returns for context)
    features = []
    for i in range(len(returns)):
        feature_vec = [returns[i]]
        # Add lagged returns if available
        for lag in range(1, min(4, i + 1)):
            feature_vec.append(returns[i - lag])
        features.append(feature_vec)
    
    # Pad shorter feature vectors
    max_len = max(len(f) for f in features)
    for i, f in enumerate(features):
        while len(f) < max_len:
            f.append(0.0)
    
    features = np.array(features)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_states, random_state=random_seed, n_init=10)
    try:
        cluster_labels = kmeans.fit_predict(features)
    except Exception:
        warnings.warn("K-means clustering failed, falling back to random initialization")
        return initialize_parameters_random(n_states, returns, random_seed)
    
    # Initial probabilities from cluster frequencies
    unique, counts = np.unique(cluster_labels, return_counts=True)
    initial_probs = np.zeros(n_states)
    for i, count in zip(unique, counts):
        initial_probs[i] = count / len(cluster_labels)
    
    # Transition matrix from cluster sequence
    transition_matrix = np.zeros((n_states, n_states))
    for i in range(len(cluster_labels) - 1):
        transition_matrix[cluster_labels[i], cluster_labels[i + 1]] += 1
    
    # Add small regularization to avoid zeros
    transition_matrix += 0.01
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    
    # Emission parameters from cluster statistics
    emission_params = np.zeros((n_states, 2))
    for state in range(n_states):
        state_returns = returns[cluster_labels == state]
        if len(state_returns) > 0:
            emission_params[state, 0] = np.mean(state_returns)
            emission_params[state, 1] = max(np.std(state_returns), 1e-6)
        else:
            # Fallback for empty clusters
            emission_params[state, 0] = np.mean(returns)
            emission_params[state, 1] = np.std(returns)
    
    return initial_probs, transition_matrix, emission_params


def check_convergence(
    log_likelihoods: list, 
    tolerance: float, 
    min_iterations: int = 10
) -> bool:
    """
    Check if training has converged based on log-likelihood improvement.
    
    Args:
        log_likelihoods: List of log-likelihood values
        tolerance: Convergence tolerance
        min_iterations: Minimum iterations before checking convergence
        
    Returns:
        True if converged, False otherwise
    """
    if len(log_likelihoods) < min_iterations + 1:
        return False
    
    # Check improvement over last few iterations
    recent_improvements = []
    for i in range(min(5, len(log_likelihoods) - 1)):
        improvement = log_likelihoods[-(i+1)] - log_likelihoods[-(i+2)]
        recent_improvements.append(abs(improvement))
    
    # Converged if all recent improvements are small (with small epsilon for floating point precision)
    return all(imp <= tolerance + 1e-12 for imp in recent_improvements)


def normalize_probabilities(probs: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Normalize probabilities ensuring they sum to 1.
    
    Args:
        probs: Probability array
        axis: Axis along which to normalize (None for entire array)
        
    Returns:
        Normalized probabilities
    """
    probs = np.asarray(probs)
    
    # Add small epsilon to avoid division by zero
    probs = np.maximum(probs, 1e-10)
    
    if axis is None:
        return probs / np.sum(probs)
    else:
        return probs / np.sum(probs, axis=axis, keepdims=True)


def log_normalize(log_probs: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Normalize log probabilities using log-sum-exp trick for numerical stability.
    
    Args:
        log_probs: Log probability array
        axis: Axis along which to normalize
        
    Returns:
        Normalized log probabilities
    """
    if axis is None:
        max_log_prob = np.max(log_probs)
        log_sum = max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))
        return log_probs - log_sum
    else:
        max_log_prob = np.max(log_probs, axis=axis, keepdims=True)
        log_sum = max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob), 
                                             axis=axis, keepdims=True))
        return log_probs - log_sum


def validate_hmm_parameters(
    initial_probs: np.ndarray,
    transition_matrix: np.ndarray,
    emission_params: np.ndarray
) -> None:
    """
    Validate HMM parameters for mathematical consistency.
    
    Args:
        initial_probs: Initial state probabilities
        transition_matrix: State transition probabilities
        emission_params: Emission parameters [means, stds]
        
    Raises:
        ValueError: If parameters are invalid
    """
    n_states = len(initial_probs)
    
    # Check initial probabilities
    if not np.allclose(np.sum(initial_probs), 1.0, atol=1e-6):
        raise ValueError("Initial probabilities must sum to 1")
    
    if np.any(initial_probs < 0):
        raise ValueError("Initial probabilities must be non-negative")
    
    # Check transition matrix
    if transition_matrix.shape != (n_states, n_states):
        raise ValueError(f"Transition matrix must be {n_states}x{n_states}")
    
    if not np.allclose(np.sum(transition_matrix, axis=1), 1.0, atol=1e-6):
        raise ValueError("Transition matrix rows must sum to 1")
    
    if np.any(transition_matrix < 0):
        raise ValueError("Transition probabilities must be non-negative")
    
    # Check emission parameters
    if emission_params.shape != (n_states, 2):
        raise ValueError(f"Emission parameters must be {n_states}x2 (means, stds)")
    
    if np.any(emission_params[:, 1] <= 0):
        raise ValueError("Emission standard deviations must be positive")


def get_regime_interpretation(state_idx: int, emission_params: np.ndarray) -> str:
    """
    Get human-readable interpretation of a regime state.
    
    Args:
        state_idx: State index
        emission_params: Emission parameters [means, stds]
        
    Returns:
        String interpretation of the regime
    """
    mean_return = emission_params[state_idx, 0]
    volatility = emission_params[state_idx, 1]
    
    # Classify based on mean return
    if mean_return < -0.005:  # Less than -0.5% daily
        regime_type = "Bear"
    elif mean_return > 0.005:  # Greater than 0.5% daily
        regime_type = "Bull"
    else:
        regime_type = "Sideways"
    
    # Add volatility characterization
    if volatility >= 0.03:  # >=3% daily volatility
        vol_desc = "High Vol"
    elif volatility <= 0.02:  # <=2% daily volatility
        vol_desc = "Low Vol"
    else:
        vol_desc = "Moderate Vol"
    
    return f"{regime_type} ({vol_desc})"


def calculate_regime_statistics(
    states: np.ndarray,
    returns: np.ndarray,
    dates: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for regime sequences.
    
    Args:
        states: State sequence
        returns: Corresponding returns
        dates: Optional dates for temporal analysis
        
    Returns:
        Dictionary with regime statistics
    """
    n_states = len(np.unique(states))
    stats = {
        'n_observations': len(states),
        'n_states': n_states,
        'regime_stats': {}
    }
    
    for state in range(n_states):
        state_mask = states == state
        state_returns = returns[state_mask]
        
        if len(state_returns) > 0:
            regime_stats = {
                'frequency': np.sum(state_mask) / len(states),
                'mean_return': np.mean(state_returns),
                'std_return': np.std(state_returns),
                'min_return': np.min(state_returns),
                'max_return': np.max(state_returns),
                'total_periods': np.sum(state_mask)
            }
            
            # Calculate regime duration statistics
            durations = []
            current_duration = 0
            for i, s in enumerate(states):
                if s == state:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                        current_duration = 0
            
            # Don't forget the last duration if it ends with this state
            if current_duration > 0:
                durations.append(current_duration)
            
            if durations:
                regime_stats.update({
                    'avg_duration': np.mean(durations),
                    'min_duration': np.min(durations),
                    'max_duration': np.max(durations),
                    'n_episodes': len(durations)
                })
            
            stats['regime_stats'][state] = regime_stats
    
    return stats