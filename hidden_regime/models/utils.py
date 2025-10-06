"""
Utility functions for Hidden Markov Model implementation.

Provides helper functions for parameter validation, initialization,
convergence checking, and numerical stability.
"""

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Initialize HMM parameters using financial domain-constrained K-means clustering.

    This function applies financial domain knowledge to prevent unrealistic
    regime centers while preserving the statistical benefits of data-driven initialization.

    Args:
        n_states: Number of hidden states
        returns: Log returns data
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (initial_probs, transition_matrix, emission_params, diagnostics)
        where diagnostics is a dict containing KMeans quality metrics and warnings
    """

    if len(returns) < n_states:
        warnings.warn(
            "Insufficient data for K-means initialization, falling back to random"
        )
        init_probs, trans_mat, emission_params = initialize_parameters_random(
            n_states, returns, random_seed
        )
        fallback_diagnostics = {
            'method': 'random_fallback',
            'reason': 'insufficient_data',
            'n_states': n_states,
            'n_observations': len(returns),
            'warnings': ['Insufficient data for KMeans, used random initialization']
        }
        return init_probs, trans_mat, emission_params, fallback_diagnostics

    # STEP 1: Filter extreme outliers in percentage space (easier financial intuition)
    returns_pct = np.exp(returns) - 1  # Convert log returns to percentages

    # Filter outliers: daily moves >15% are likely noise/earnings/splits
    outlier_threshold_pct = 0.15  # 15% daily move threshold
    outlier_mask = np.abs(returns_pct) <= outlier_threshold_pct

    if outlier_mask.sum() < n_states * 5:  # Need minimum data per state
        warnings.warn("Too many outliers removed, using less aggressive filtering")
        outlier_threshold_pct = 0.25  # 25% threshold as fallback
        outlier_mask = np.abs(returns_pct) <= outlier_threshold_pct

    # Filter data
    filtered_returns_pct = returns_pct[outlier_mask]
    filtered_returns_log = np.log(filtered_returns_pct + 1)  # Back to log space

    # STEP 2: Enhanced feature engineering for clustering
    features = []
    for i in range(len(filtered_returns_log)):
        feature_vec = [filtered_returns_log[i]]
        # Add lagged returns for temporal context (helps identify regimes)
        for lag in range(1, min(4, i + 1)):
            feature_vec.append(filtered_returns_log[i - lag])
        features.append(feature_vec)

    # Pad shorter feature vectors
    max_len = max(len(f) for f in features)
    for i, f in enumerate(features):
        while len(f) < max_len:
            f.append(0.0)

    features = np.array(features)

    # STEP 3: K-means clustering on filtered data
    kmeans = KMeans(n_clusters=n_states, random_state=random_seed, n_init=10)
    try:
        cluster_labels = kmeans.fit_predict(features)
    except Exception as e:
        warnings.warn(
            f"K-means clustering failed ({str(e)}), falling back to random initialization"
        )
        init_probs, trans_mat, emission_params = initialize_parameters_random(
            n_states, returns, random_seed
        )
        fallback_diagnostics = {
            'method': 'random_fallback',
            'reason': 'kmeans_failed',
            'error': str(e),
            'n_states': n_states,
            'n_observations': len(returns),
            'warnings': [f'KMeans clustering failed: {str(e)}']
        }
        return init_probs, trans_mat, emission_params, fallback_diagnostics

    # Collect KMeans diagnostics
    diagnostics = _collect_kmeans_diagnostics(
        kmeans, features, cluster_labels, n_states, filtered_returns_log
    )

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

    # STEP 4: Calculate raw emission parameters from clusters
    raw_emission_params = np.zeros((n_states, 2))
    cluster_means = []

    for state in range(n_states):
        # Use original filtered returns for state statistics
        state_mask = cluster_labels == state
        state_returns = filtered_returns_log[state_mask]

        if len(state_returns) > 0:
            mean_return = np.mean(state_returns)
            std_return = max(np.std(state_returns), 1e-6)
        else:
            # Fallback for empty clusters
            mean_return = np.mean(filtered_returns_log)
            std_return = np.std(filtered_returns_log)

        raw_emission_params[state, 0] = mean_return
        raw_emission_params[state, 1] = std_return
        cluster_means.append((state, mean_return))

    # STEP 5: Apply financial domain constraints to emission means
    # Convert means to percentage space for constraint application
    raw_means_pct = np.exp([mean for _, mean in cluster_means]) - 1
    sorted_cluster_info = sorted(
        cluster_means, key=lambda x: x[1]
    )  # Sort by log return

    # Define financial constraints in percentage space
    constrained_means_pct = []
    constraint_applied = []
    for i, (original_state, raw_mean_log) in enumerate(sorted_cluster_info):
        raw_mean_pct = np.exp(raw_mean_log) - 1

        # Apply progressive constraints based on sorted position
        if i == 0:  # Lowest return state
            # Bear regime: -8% to -0.1% daily reasonable range
            constrained_pct = np.clip(raw_mean_pct, -0.08, -0.001)
        elif i == len(sorted_cluster_info) - 1:  # Highest return state
            # Bull regime: 0.1% to 5% daily reasonable range
            constrained_pct = np.clip(raw_mean_pct, 0.001, 0.05)
        else:  # Middle states
            # Sideways-type regimes: -0.5% to 1.5% daily reasonable range
            constrained_pct = np.clip(raw_mean_pct, -0.005, 0.015)

        constrained_means_pct.append(constrained_pct)
        constraint_applied.append(abs(constrained_pct - raw_mean_pct) > 1e-6)

    # Convert constrained means back to log space
    constrained_means_log = np.log(np.array(constrained_means_pct) + 1)

    # Add constraint diagnostics
    diagnostics['constraint_distortion'] = _compute_constraint_distortion(
        raw_means_pct, constrained_means_pct, constraint_applied, sorted_cluster_info
    )

    # STEP 6: Build final parameters with constrained means and consistent ordering
    emission_params = np.zeros((n_states, 2))
    state_reordering = {}

    for new_state, (old_state, _) in enumerate(sorted_cluster_info):
        # Use constrained mean, original std
        emission_params[new_state, 0] = constrained_means_log[new_state]
        emission_params[new_state, 1] = raw_emission_params[old_state, 1]
        state_reordering[old_state] = new_state

    # Reorder initial probabilities and transition matrix
    reordered_initial_probs = np.zeros_like(initial_probs)
    for old_state, new_state in state_reordering.items():
        reordered_initial_probs[new_state] = initial_probs[old_state]

    reordered_transition_matrix = np.zeros_like(transition_matrix)
    for old_i, new_i in state_reordering.items():
        for old_j, new_j in state_reordering.items():
            reordered_transition_matrix[new_i, new_j] = transition_matrix[old_i, old_j]

    # STEP 7: Validate final parameters make financial sense
    _validate_financial_constraints(emission_params, n_states)

    # Add warnings to diagnostics
    diagnostics['warnings'] = _generate_initialization_warnings(diagnostics)

    return reordered_initial_probs, reordered_transition_matrix, emission_params, diagnostics


def _collect_kmeans_diagnostics(
    kmeans, features: np.ndarray, cluster_labels: np.ndarray,
    n_states: int, returns: np.ndarray
) -> Dict[str, Any]:
    """
    Collect diagnostic information about KMeans clustering quality.

    Args:
        kmeans: Fitted KMeans object
        features: Feature array used for clustering
        cluster_labels: Cluster assignments
        n_states: Number of states
        returns: Original returns data

    Returns:
        Dictionary with clustering diagnostics
    """
    diagnostics = {
        'method': 'kmeans',
        'n_states': n_states,
        'n_observations': len(returns),
        'n_features': features.shape[1] if len(features.shape) > 1 else 1,
    }

    # Basic KMeans metrics
    diagnostics['kmeans_inertia'] = kmeans.inertia_
    diagnostics['kmeans_iterations'] = kmeans.n_iter_

    # Cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = {int(state): int(count) for state, count in zip(unique, counts)}
    diagnostics['cluster_sizes'] = cluster_sizes

    # Check for imbalanced clusters
    min_size = min(counts)
    max_size = max(counts)
    diagnostics['cluster_balance_ratio'] = float(min_size / max_size) if max_size > 0 else 0.0

    # Quality metrics (only if we have enough data points)
    if len(returns) >= n_states * 2:
        try:
            # Silhouette score: measures how similar an object is to its own cluster
            # Range: [-1, 1], higher is better
            # Note: Financial data typically produces 0.10-0.20 (not poor, just overlapping regimes)
            score = float(silhouette_score(features, cluster_labels))
            diagnostics['silhouette_score'] = score
            diagnostics['silhouette_interpretation'] = _interpret_silhouette_for_financial_data(score)
        except Exception as e:
            diagnostics['silhouette_score'] = None
            diagnostics['silhouette_error'] = str(e)
            diagnostics['silhouette_interpretation'] = 'Unable to compute'

        try:
            # Calinski-Harabasz score: ratio of between-cluster to within-cluster variance
            # Higher is better, no fixed range
            diagnostics['calinski_harabasz_score'] = float(
                calinski_harabasz_score(features, cluster_labels)
            )
        except Exception as e:
            diagnostics['calinski_harabasz_score'] = None
            diagnostics['calinski_harabasz_error'] = str(e)

    return diagnostics


def _compute_constraint_distortion(
    raw_means_pct: list,
    constrained_means_pct: list,
    constraint_applied: list,
    sorted_cluster_info: list
) -> Dict[str, Any]:
    """
    Compute metrics about how much financial constraints distorted KMeans results.

    Args:
        raw_means_pct: Raw cluster means in percentage space
        constrained_means_pct: Constrained means in percentage space
        constraint_applied: Boolean list of whether constraint was applied
        sorted_cluster_info: List of (state, mean) tuples sorted by mean

    Returns:
        Dictionary with constraint distortion metrics
    """
    distortion = {
        'n_states_constrained': sum(constraint_applied),
        'states_constrained': [i for i, applied in enumerate(constraint_applied) if applied],
    }

    # Calculate distortion magnitude for each state
    state_distortions = []
    for i, (raw_pct, const_pct) in enumerate(zip(raw_means_pct, constrained_means_pct)):
        original_state = sorted_cluster_info[i][0]
        distortion_pct = abs(const_pct - raw_pct)
        state_distortions.append({
            'state': int(original_state),
            'raw_mean_pct': float(raw_pct),
            'constrained_mean_pct': float(const_pct),
            'distortion_pct': float(distortion_pct),
            'distortion_relative': float(distortion_pct / (abs(raw_pct) + 1e-10)),
        })

    distortion['state_distortions'] = state_distortions

    # Overall distortion metrics
    distortions_abs = [sd['distortion_pct'] for sd in state_distortions]
    distortion['max_distortion_pct'] = float(max(distortions_abs))
    distortion['mean_distortion_pct'] = float(np.mean(distortions_abs))

    return distortion


def _interpret_silhouette_for_financial_data(score: float) -> str:
    """
    Provide domain-specific interpretation of silhouette scores for financial data.

    Financial returns typically produce lower silhouette scores (0.10-0.20) than
    other domains due to inherently overlapping regimes. Bull/bear/sideways states
    have fuzzy boundaries and gradual transitions, not sharp cluster separation.

    Args:
        score: Silhouette coefficient [-1, 1]

    Returns:
        Human-readable interpretation string
    """
    if score > 0.25:
        return "Excellent separation for financial data (regimes are well-defined)"
    elif score > 0.15:
        return "Good separation for financial regimes (typical for real markets)"
    elif score > 0.05:
        return "Typical overlapping market regimes (expected for financial returns)"
    elif score > 0.0:
        return "Weak cluster separation (regimes may be poorly defined)"
    else:
        return "Very poor clustering (cluster assignments may be incorrect)"


def _generate_initialization_warnings(diagnostics: Dict[str, Any]) -> list:
    """
    Generate warnings based on initialization diagnostics.

    Args:
        diagnostics: Diagnostics dictionary from initialization

    Returns:
        List of warning strings
    """
    warnings_list = []

    # Check clustering quality
    silhouette = diagnostics.get('silhouette_score')
    if silhouette is not None:
        # Note: Financial returns typically produce scores of 0.10-0.20 due to overlapping regimes
        # These thresholds are adjusted for financial time series data
        if silhouette < 0.05:
            warnings_list.append(
                f"Very low cluster separation (silhouette={silhouette:.3f}). "
                f"Note: Financial returns often produce scores of 0.10-0.20 due to overlapping regimes. "
                "Consider checking regime persistence and transition probabilities as better quality metrics."
            )
        elif silhouette < 0.15:
            # Informational only - typical for financial data
            warnings_list.append(
                f"Cluster separation (silhouette={silhouette:.3f}) is typical for financial data. "
                "Regimes are inherently overlapping. Validate using regime stability and likelihood."
            )

    # Check cluster balance
    balance = diagnostics.get('cluster_balance_ratio', 1.0)
    if balance < 0.2:
        warnings_list.append(
            f"Highly imbalanced clusters (ratio={balance:.3f}). "
            f"Cluster sizes: {diagnostics.get('cluster_sizes')}. "
            "Some regimes may be underrepresented."
        )

    # Check constraint distortion
    constraint_info = diagnostics.get('constraint_distortion', {})
    n_constrained = constraint_info.get('n_states_constrained', 0)
    max_distortion = constraint_info.get('max_distortion_pct', 0)

    if n_constrained > 0:
        if max_distortion > 0.5:  # >50% distortion
            warnings_list.append(
                f"Severe constraint distortion detected: {n_constrained} states constrained, "
                f"max distortion = {max_distortion:.1%}. "
                "Financial constraints may be inappropriate for this data. "
                "Consider initialization_method='random' for non-financial data."
            )
        elif max_distortion > 0.2:  # >20% distortion
            warnings_list.append(
                f"Moderate constraint distortion: {n_constrained} states constrained, "
                f"max distortion = {max_distortion:.1%}. "
                "Verify that data represents financial log returns."
            )

    return warnings_list


def _validate_financial_constraints(emission_params: np.ndarray, n_states: int) -> None:
    """Validate that emission parameters satisfy financial constraints."""
    means_log = emission_params[:, 0]
    means_pct = np.exp(means_log) - 1

    for i, mean_pct in enumerate(means_pct):
        # Warn about extreme regimes
        if abs(mean_pct) > 0.10:  # >10% daily is extreme
            warnings.warn(
                f"State {i} has extreme daily return: {mean_pct:.2%}. "
                f"Consider using more states or different initialization."
            )

        # Check gaps between adjacent regimes
        if i > 0:
            gap_pct = means_pct[i] - means_pct[i - 1]
            if gap_pct > 0.03:  # >3% gap between regimes is large
                warnings.warn(
                    f"Large gap between regimes {i-1} and {i}: {gap_pct:.2%} daily return difference"
                )


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
