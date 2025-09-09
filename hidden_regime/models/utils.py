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
    warnings.warn(
        "scikit-learn not available. K-means initialization will fall back to random."
    )


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
    n_states: int, returns: np.ndarray, random_seed: Optional[int] = None
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
    n_states: int, returns: np.ndarray, random_seed: Optional[int] = None
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
        warnings.warn(
            "scikit-learn not available, falling back to random initialization"
        )
        return initialize_parameters_random(n_states, returns, random_seed)

    if len(returns) < n_states:
        warnings.warn(
            "Insufficient data for K-means initialization, falling back to random"
        )
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
        warnings.warn(
            "K-means clustering failed, falling back to random initialization"
        )
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
    log_likelihoods: list, tolerance: float, min_iterations: int = 10
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
        improvement = log_likelihoods[-(i + 1)] - log_likelihoods[-(i + 2)]
        recent_improvements.append(abs(improvement))

    # Converged if all recent improvements are small (with small epsilon for floating point precision)
    return all(imp <= tolerance + 1e-12 for imp in recent_improvements)


def normalize_probabilities(
    probs: np.ndarray, axis: Optional[int] = None
) -> np.ndarray:
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
        log_sum = max_log_prob + np.log(
            np.sum(np.exp(log_probs - max_log_prob), axis=axis, keepdims=True)
        )
        return log_probs - log_sum


def validate_hmm_parameters(
    initial_probs: np.ndarray,
    transition_matrix: np.ndarray,
    emission_params: np.ndarray,
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

    # Enhanced regime classification with more nuanced thresholds
    if mean_return < -0.015:  # Less than -1.5% daily (severe bear)
        if volatility >= 0.04:  # >=4% daily volatility
            regime_type = "Crisis"
        else:
            regime_type = "Bear"
    elif mean_return < -0.002:  # -0.2% to -1.5% daily (mild bear)
        regime_type = "Bear"
    elif mean_return > 0.015:  # Greater than 1.5% daily (strong bull)
        if volatility >= 0.04:  # High volatility bull (euphoric)
            regime_type = "Euphoric Bull"
        else:
            regime_type = "Bull"
    elif mean_return > 0.002:  # 0.2% to 1.5% daily (mild bull)
        regime_type = "Bull"
    else:  # -0.2% to 0.2% daily
        regime_type = "Sideways"

    # Add volatility characterization for context
    if volatility >= 0.035:  # >=3.5% daily volatility
        vol_desc = "High Vol"
    elif volatility <= 0.015:  # <=1.5% daily volatility
        vol_desc = "Low Vol"
    else:
        vol_desc = "Moderate Vol"

    return f"{regime_type} ({vol_desc})"


def get_standardized_regime_name(
    state_idx: int, emission_params: np.ndarray, regime_type: str = "3_state"
) -> str:
    """
    Get standardized regime name based on configuration type.

    Args:
        state_idx: State index
        emission_params: Emission parameters [means, stds]
        regime_type: Type of regime configuration ('3_state', '4_state', '5_state')

    Returns:
        Standardized regime name
    """
    from .state_standardizer import StateStandardizer

    standardizer = StateStandardizer()
    config = standardizer.get_config(regime_type)

    if config is None:
        return get_regime_interpretation(state_idx, emission_params)

    mean_return = emission_params[state_idx, 0]
    volatility = emission_params[state_idx, 1]

    # Find best matching regime based on thresholds
    best_match = None
    best_score = float("inf")

    for i, state_name in enumerate(config.state_names):
        if state_name in config.state_thresholds:
            thresholds = config.state_thresholds[state_name]
            score = 0

            # Score based on mean return criteria
            if "mean_return" in thresholds:
                score += abs(mean_return - thresholds["mean_return"])
            if "min_return" in thresholds:
                if mean_return < thresholds["min_return"]:
                    score += (
                        thresholds["min_return"] - mean_return
                    ) * 2  # Penalty for violation
            if "max_return" in thresholds:
                if mean_return > thresholds["max_return"]:
                    score += (
                        mean_return - thresholds["max_return"]
                    ) * 2  # Penalty for violation

            # Score based on volatility criteria
            if "min_volatility" in thresholds:
                if volatility < thresholds["min_volatility"]:
                    score += (thresholds["min_volatility"] - volatility) * 0.5
            if "max_volatility" in thresholds:
                if volatility > thresholds["max_volatility"]:
                    score += (volatility - thresholds["max_volatility"]) * 0.5

            if score < best_score:
                best_score = score
                best_match = state_name

    return (
        best_match
        if best_match
        else get_regime_interpretation(state_idx, emission_params)
    )


def validate_regime_economics(
    emission_params: np.ndarray, regime_type: str = "3_state"
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that detected regimes make economic sense.

    Args:
        emission_params: Emission parameters [means, stds]
        regime_type: Type of regime configuration

    Returns:
        Tuple of (is_valid, validation_details)
    """
    n_states = len(emission_params)
    means = emission_params[:, 0]
    stds = emission_params[:, 1]

    validation_results = {
        "is_economically_valid": True,
        "violations": [],
        "regime_separation": {},
        "mean_ordering_correct": False,
        "volatility_reasonable": True,
    }

    # Check if means are properly ordered (should generally increase)
    sorted_indices = np.argsort(means)
    expected_order = list(range(n_states))
    validation_results["mean_ordering_correct"] = list(sorted_indices) == expected_order

    if not validation_results["mean_ordering_correct"]:
        validation_results["violations"].append("Mean returns not properly ordered")
        validation_results["is_economically_valid"] = False

    # Check regime separation (Cohen's d between adjacent regimes)
    for i in range(n_states - 1):
        idx1, idx2 = sorted_indices[i], sorted_indices[i + 1]
        mean1, std1 = means[idx1], stds[idx1]
        mean2, std2 = means[idx2], stds[idx2]

        # Calculate Cohen's d for separation
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        cohens_d = abs(mean2 - mean1) / pooled_std if pooled_std > 0 else 0

        validation_results["regime_separation"][f"regime_{idx1}_vs_{idx2}"] = {
            "cohens_d": cohens_d,
            "well_separated": cohens_d
            >= 0.5,  # Cohen's d >= 0.5 indicates medium effect
        }

        if cohens_d < 0.3:  # Small effect size threshold
            validation_results["violations"].append(
                f"Poor separation between regime {idx1} and {idx2} (Cohen's d = {cohens_d:.3f})"
            )
            validation_results["is_economically_valid"] = False

    # Check volatility reasonableness
    min_vol, max_vol = np.min(stds), np.max(stds)
    if min_vol <= 0.005:  # Less than 0.5% daily
        validation_results["violations"].append(
            "Unrealistically low volatility detected"
        )
        validation_results["volatility_reasonable"] = False
        validation_results["is_economically_valid"] = False

    if max_vol >= 0.08:  # Greater than 8% daily
        validation_results["violations"].append("Extremely high volatility detected")
        validation_results["volatility_reasonable"] = False

    # Specific checks based on regime type
    if regime_type in ["3_state", "4_state", "5_state"]:
        bear_indices = [i for i in range(n_states) if means[i] < -0.001]
        bull_indices = [i for i in range(n_states) if means[i] > 0.001]

        if not bear_indices:
            validation_results["violations"].append("No clear bear regime detected")
            validation_results["is_economically_valid"] = False

        if not bull_indices:
            validation_results["violations"].append("No clear bull regime detected")
            validation_results["is_economically_valid"] = False

    return validation_results["is_economically_valid"], validation_results


def analyze_regime_transitions(
    states: np.ndarray,
    transition_matrix: np.ndarray,
    regime_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    Analyze regime transition patterns and persistence.

    Args:
        states: State sequence
        transition_matrix: Transition probability matrix
        regime_names: Optional mapping of state indices to regime names

    Returns:
        Dictionary with transition analysis results
    """
    n_states = len(np.unique(states))

    if regime_names is None:
        regime_names = {i: f"Regime {i}" for i in range(n_states)}

    # Calculate empirical transition frequencies
    empirical_transitions = np.zeros((n_states, n_states))
    for i in range(len(states) - 1):
        empirical_transitions[states[i], states[i + 1]] += 1

    # Normalize to probabilities
    empirical_transition_probs = empirical_transitions / (
        empirical_transitions.sum(axis=1, keepdims=True) + 1e-10
    )

    # Calculate persistence (diagonal elements)
    persistence = {
        regime_names[i]: {
            "theoretical_persistence": transition_matrix[i, i],
            "empirical_persistence": empirical_transition_probs[i, i],
            "expected_duration": (
                1 / (1 - transition_matrix[i, i])
                if transition_matrix[i, i] < 1
                else float("inf")
            ),
        }
        for i in range(n_states)
    }

    # Find most likely transitions
    transition_patterns = {}
    for i in range(n_states):
        for j in range(n_states):
            if i != j and transition_matrix[i, j] > 0.1:  # Only significant transitions
                pattern_name = f"{regime_names[i]} → {regime_names[j]}"
                transition_patterns[pattern_name] = {
                    "probability": transition_matrix[i, j],
                    "empirical_frequency": empirical_transition_probs[i, j],
                    "count": int(empirical_transitions[i, j]),
                }

    # Calculate overall stability metrics
    stability_metrics = {
        "average_persistence": np.mean(np.diag(transition_matrix)),
        "regime_switching_rate": 1 - np.mean(np.diag(transition_matrix)),
        "most_stable_regime": regime_names[np.argmax(np.diag(transition_matrix))],
        "least_stable_regime": regime_names[np.argmin(np.diag(transition_matrix))],
    }

    return {
        "persistence_analysis": persistence,
        "transition_patterns": transition_patterns,
        "stability_metrics": stability_metrics,
        "empirical_transition_matrix": empirical_transition_probs.tolist(),
        "theoretical_transition_matrix": transition_matrix.tolist(),
    }


def calculate_regime_statistics(
    states: np.ndarray, returns: np.ndarray, dates: Optional[np.ndarray] = None
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
    stats = {"n_observations": len(states), "n_states": n_states, "regime_stats": {}}

    for state in range(n_states):
        state_mask = states == state
        state_returns = returns[state_mask]

        if len(state_returns) > 0:
            regime_stats = {
                "frequency": np.sum(state_mask) / len(states),
                "mean_return": np.mean(state_returns),
                "std_return": np.std(state_returns),
                "min_return": np.min(state_returns),
                "max_return": np.max(state_returns),
                "total_periods": np.sum(state_mask),
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
                regime_stats.update(
                    {
                        "avg_duration": np.mean(durations),
                        "min_duration": np.min(durations),
                        "max_duration": np.max(durations),
                        "n_episodes": len(durations),
                    }
                )

            stats["regime_stats"][state] = regime_stats

    return stats
