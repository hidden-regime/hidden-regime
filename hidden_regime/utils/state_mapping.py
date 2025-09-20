"""
State mapping utilities for consistent financial regime interpretation.

Provides functions to map HMM states to financial regimes based on actual
emission parameters rather than arbitrary state numbering.
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Any


def map_states_to_financial_regimes(
    emission_means: np.ndarray,
    n_states: int,
    validate: bool = True
) -> Dict[int, str]:
    """
    Map HMM states to financial regimes based on actual emission characteristics.

    Uses threshold-based classification to assign regime names based on actual
    return values rather than forcing arbitrary state orderings.

    Args:
        emission_means: Array of emission means for each state (in log return space)
        n_states: Number of states in the model
        validate: Whether to validate that mappings make financial sense

    Returns:
        Dictionary mapping state indices to regime names

    Raises:
        ValueError: If emission_means length doesn't match n_states

    Example:
        >>> emission_means = np.array([-0.005, 0.001, 0.134])  # Bear, Sideways, Explosive
        >>> mapping = map_states_to_financial_regimes(emission_means, 3)
        >>> print(mapping)  # {0: 'Bear', 1: 'Sideways', 2: 'Euphoric'}
    """
    if len(emission_means) != n_states:
        raise ValueError(f"emission_means length ({len(emission_means)}) must match n_states ({n_states})")

    # Convert log returns to percentages for threshold-based classification
    means_pct = np.exp(emission_means) - 1

    # Classify each state independently based on actual return characteristics
    regime_classifications = []
    for i, mean_pct in enumerate(means_pct):
        regime_name = _classify_regime_by_return_threshold(mean_pct)
        regime_classifications.append((i, regime_name, mean_pct))

    # Handle duplicate regime names by adding qualifiers
    final_mapping = _resolve_duplicate_regime_names(regime_classifications)

    # Validate the mapping makes financial sense
    if validate:
        validation_warnings = _validate_threshold_based_mapping(final_mapping, means_pct)
        for warning in validation_warnings:
            warnings.warn(warning, UserWarning)

    return final_mapping


def _classify_regime_by_return_threshold(mean_return_pct: float) -> str:
    """
    Classify a regime based on its daily return threshold.

    Uses financial domain knowledge to assign meaningful regime names
    based on actual return characteristics.

    Args:
        mean_return_pct: Daily return in percentage form

    Returns:
        Regime name based on return threshold
    """
    if mean_return_pct < -0.03:      # Less than -3% daily
        return "Crisis"
    elif mean_return_pct < -0.005:   # -3% to -0.5% daily
        return "Bear"
    elif mean_return_pct < 0.01:     # -0.5% to +1% daily
        return "Sideways"
    elif mean_return_pct < 0.05:     # +1% to +5% daily
        return "Bull"
    else:                            # Greater than +5% daily
        return "Euphoric"


def _resolve_duplicate_regime_names(
    regime_classifications: List[Tuple[int, str, float]]
) -> Dict[int, str]:
    """
    Resolve cases where multiple states get the same regime name.

    Adds qualifiers like "Weak", "Moderate", "Strong" to distinguish
    between multiple states in the same regime category.

    Args:
        regime_classifications: List of (state_index, regime_name, return_pct) tuples

    Returns:
        Final mapping from state indices to unique regime names
    """
    # Group states by regime name
    regime_groups = {}
    for state_idx, regime_name, return_pct in regime_classifications:
        if regime_name not in regime_groups:
            regime_groups[regime_name] = []
        regime_groups[regime_name].append((state_idx, return_pct))

    final_mapping = {}

    for regime_name, state_list in regime_groups.items():
        if len(state_list) == 1:
            # Single state with this regime name
            state_idx, _ = state_list[0]
            final_mapping[state_idx] = regime_name
        else:
            # Multiple states - add qualifiers based on relative strength
            sorted_states = sorted(state_list, key=lambda x: x[1])  # Sort by return

            for i, (state_idx, return_pct) in enumerate(sorted_states):
                if len(sorted_states) == 2:
                    # Two states: Weak/Strong
                    qualifier = "Weak" if i == 0 else "Strong"
                elif len(sorted_states) == 3:
                    # Three states: Weak/Moderate/Strong
                    qualifiers = ["Weak", "Moderate", "Strong"]
                    qualifier = qualifiers[i]
                else:
                    # More than three: use numbers
                    qualifier = f"Level {i+1}"

                final_mapping[state_idx] = f"{qualifier} {regime_name}"

    return final_mapping


def _validate_threshold_based_mapping(
    mapping: Dict[int, str],
    means_pct: np.ndarray
) -> List[str]:
    """
    Validate that threshold-based mapping makes financial sense.

    Args:
        mapping: State index to regime name mapping
        means_pct: Emission means in percentage space

    Returns:
        List of validation warning messages
    """
    warnings_list = []

    # Check for extreme regimes
    for state_idx, regime_name in mapping.items():
        mean_pct = means_pct[state_idx]

        if abs(mean_pct) > 0.10:  # >10% daily is extreme
            warnings_list.append(
                f"Extreme regime detected: {regime_name} (state {state_idx}) "
                f"has {mean_pct:.2%} daily return. Consider using more states."
            )

        # Check regime-specific thresholds
        if "Crisis" in regime_name and mean_pct > -0.01:
            warnings_list.append(
                f"Crisis regime (state {state_idx}) has insufficient negative return: {mean_pct:.2%}"
            )

        if "Bear" in regime_name and mean_pct > 0:
            warnings_list.append(
                f"Bear regime (state {state_idx}) has positive return: {mean_pct:.2%}"
            )

        if "Bull" in regime_name and mean_pct < 0:
            warnings_list.append(
                f"Bull regime (state {state_idx}) has negative return: {mean_pct:.2%}"
            )

        if "Euphoric" in regime_name and mean_pct < 0.02:
            warnings_list.append(
                f"Euphoric regime (state {state_idx}) has insufficient positive return: {mean_pct:.2%}"
            )

    # Check for large gaps between regimes
    sorted_returns = sorted(means_pct)
    for i in range(1, len(sorted_returns)):
        gap = sorted_returns[i] - sorted_returns[i-1]
        if gap > 0.05:  # >5% gap between adjacent regimes
            warnings_list.append(
                f"Large gap between regimes: {gap:.2%} daily return difference. "
                f"Consider adjusting number of states."
            )

    return warnings_list


def validate_financial_mapping(
    mapping: Dict[int, str],
    sorted_means: np.ndarray,
    sorted_indices: np.ndarray
) -> List[str]:
    """
    Validate that the state mapping makes financial sense.

    Args:
        mapping: State index to regime name mapping
        sorted_means: Emission means sorted in ascending order
        sorted_indices: Original state indices in sorted order

    Returns:
        List of validation warning messages
    """
    warnings_list = []

    # Check that Bear regimes have negative expected returns
    for state_idx, regime_name in mapping.items():
        mean_return = sorted_means[np.where(sorted_indices == state_idx)[0][0]]

        if regime_name == "Bear" and mean_return > 0:
            warnings_list.append(
                f"Bear regime (state {state_idx}) has positive expected return {mean_return:.4f}. "
                "This may indicate insufficient data or model misspecification."
            )

        elif regime_name == "Bull" and mean_return < 0:
            warnings_list.append(
                f"Bull regime (state {state_idx}) has negative expected return {mean_return:.4f}. "
                "This may indicate insufficient data or model misspecification."
            )

        elif regime_name == "Crisis" and mean_return > -0.001:
            warnings_list.append(
                f"Crisis regime (state {state_idx}) has insufficiently negative expected return {mean_return:.4f}. "
                "Crisis regimes should represent significant market stress."
            )

        elif regime_name == "Euphoric" and mean_return < 0.002:
            warnings_list.append(
                f"Euphoric regime (state {state_idx}) has insufficiently positive expected return {mean_return:.4f}. "
                "Euphoric regimes should represent exceptional market performance."
            )

    # Check for reasonable spread between regimes
    if len(sorted_means) >= 2:
        spread = sorted_means[-1] - sorted_means[0]
        if spread < 0.001:  # Less than 0.1% daily return difference
            warnings_list.append(
                f"Very small spread between regime returns ({spread:.4f}). "
                "Consider reducing the number of states or using more data."
            )

    return warnings_list


def get_regime_characteristics(regime_name: str) -> Dict[str, float]:
    """
    Get expected characteristics for a financial regime.

    Args:
        regime_name: Name of the financial regime

    Returns:
        Dictionary with expected characteristics (return, volatility, duration)
    """
    characteristics = {
        "Crisis": {
            "expected_return": -0.005,  # -0.5% daily
            "expected_volatility": 0.040,  # 4.0% daily volatility
            "expected_duration": 5.0  # 5 days average
        },
        "Bear": {
            "expected_return": -0.002,  # -0.2% daily
            "expected_volatility": 0.025,  # 2.5% daily volatility
            "expected_duration": 8.0  # 8 days average
        },
        "Sideways": {
            "expected_return": 0.0001,  # 0.01% daily
            "expected_volatility": 0.012,  # 1.2% daily volatility
            "expected_duration": 15.0  # 15 days average
        },
        "Bull": {
            "expected_return": 0.001,  # 0.1% daily
            "expected_volatility": 0.018,  # 1.8% daily volatility
            "expected_duration": 12.0  # 12 days average
        },
        "Euphoric": {
            "expected_return": 0.003,  # 0.3% daily
            "expected_volatility": 0.022,  # 2.2% daily volatility
            "expected_duration": 6.0  # 6 days average (unsustainable)
        }
    }

    return characteristics.get(regime_name, {
        "expected_return": 0.0,
        "expected_volatility": 0.015,
        "expected_duration": 10.0
    })


def create_consistent_regime_labels(n_states: int) -> List[str]:
    """
    Create consistent regime labels for a given number of states.

    Args:
        n_states: Number of states in the model

    Returns:
        List of regime labels in order from most negative to most positive
    """
    if n_states == 2:
        return ["Bear", "Bull"]
    elif n_states == 3:
        return ["Bear", "Sideways", "Bull"]
    elif n_states == 4:
        return ["Crisis", "Bear", "Sideways", "Bull"]
    elif n_states == 5:
        return ["Crisis", "Bear", "Sideways", "Bull", "Euphoric"]
    else:
        # For unusual numbers of states
        labels = ["Crisis"] if n_states > 3 else []
        labels.append("Bear")

        # Add intermediate states
        for i in range(n_states - 3 if n_states > 3 else n_states - 2):
            if i == 0 and n_states == 3:
                labels.append("Sideways")
            else:
                labels.append(f"Regime_{i+2}")

        if n_states > 2:
            labels.append("Bull")
        if n_states > 4:
            labels.append("Euphoric")

        return labels[:n_states]  # Ensure we don't exceed n_states


def percent_change_to_log_return(pct: float) -> float:
    """
    Convert percentage change to log return.

    Args:
        pct: Percentage change (e.g., 0.05 for 5%)

    Returns:
        Log return equivalent

    Example:
        >>> percent_change_to_log_return(0.05)  # 5% increase
        0.04879016416943204
    """
    return np.log(pct + 1.0)


def log_return_to_percent_change(log_return: float) -> float:
    """
    Convert log return to percentage change.

    Args:
        log_return: Log return value

    Returns:
        Percentage change equivalent

    Example:
        >>> log_return_to_percent_change(0.04879016416943204)
        0.05000000000000001
    """
    return np.exp(log_return) - 1.0


def apply_regime_mapping_to_analysis(
    analysis: "pd.DataFrame",
    emission_means: np.ndarray,
    n_states: int
) -> "pd.DataFrame":
    """
    Apply flexible threshold-based regime mapping to analysis results.

    Args:
        analysis: Analysis DataFrame with predicted_state column
        emission_means: Emission means from fitted HMM model
        n_states: Number of states in the model

    Returns:
        DataFrame with corrected regime_name and regime_type columns
    """
    import pandas as pd

    # Get the flexible threshold-based state mapping
    state_mapping = map_states_to_financial_regimes(emission_means, n_states)

    # Apply mapping to create regime names
    analysis = analysis.copy()
    analysis['regime_name'] = analysis['predicted_state'].map(state_mapping)
    analysis['regime_type'] = analysis['regime_name']  # They're the same now

    # Add expected characteristics based on actual regime characteristics
    means_pct = np.exp(emission_means) - 1  # Convert to percentage space

    for state_idx in range(n_states):
        state_mask = analysis['predicted_state'] == state_idx
        if state_mask.any():
            regime_name = state_mapping[state_idx]
            actual_return_pct = means_pct[state_idx]

            # Use actual characteristics instead of generic templates
            characteristics = _get_actual_regime_characteristics(
                regime_name, actual_return_pct, emission_means[state_idx]
            )

            # Add characteristics as columns for this state
            for char_name, char_value in characteristics.items():
                analysis.loc[state_mask, char_name] = char_value

    return analysis


def _get_actual_regime_characteristics(
    regime_name: str,
    actual_return_pct: float,
    actual_return_log: float
) -> Dict[str, float]:
    """
    Get regime characteristics based on actual observed parameters.

    Args:
        regime_name: Name of the regime
        actual_return_pct: Actual daily return in percentage
        actual_return_log: Actual daily return in log space

    Returns:
        Dictionary with regime characteristics
    """
    # Use actual observed returns rather than generic templates
    characteristics = {
        "expected_return": actual_return_log,  # For model compatibility (log space)
        "expected_return_pct": actual_return_pct,  # For user display (percentage)
        "expected_volatility": _estimate_volatility_from_regime_type(regime_name, actual_return_pct),
        "expected_duration": _estimate_duration_from_regime_type(regime_name),
        "regime_strength": _classify_regime_strength(actual_return_pct)
    }

    return characteristics


def _estimate_volatility_from_regime_type(regime_name: str, return_pct: float) -> float:
    """Estimate typical volatility for regime type."""
    if "Crisis" in regime_name:
        return 0.045  # 4.5% daily volatility for crisis
    elif "Bear" in regime_name:
        return 0.025  # 2.5% daily volatility for bear
    elif "Sideways" in regime_name:
        return 0.012  # 1.2% daily volatility for sideways
    elif "Bull" in regime_name:
        base_vol = 0.018  # 1.8% daily volatility for bull
        # Stronger bull markets tend to be more volatile
        if abs(return_pct) > 0.02:  # >2% daily return
            base_vol += 0.005
        return base_vol
    elif "Euphoric" in regime_name:
        return 0.035  # 3.5% daily volatility for euphoric (high risk)
    else:
        return 0.015  # Default volatility


def _estimate_duration_from_regime_type(regime_name: str) -> float:
    """Estimate typical duration for regime type."""
    if "Crisis" in regime_name:
        return 4.0   # Crisis regimes are brief
    elif "Bear" in regime_name:
        return 8.0   # Bear regimes moderate duration
    elif "Sideways" in regime_name:
        return 15.0  # Sideways regimes tend to persist
    elif "Bull" in regime_name:
        return 12.0  # Bull regimes moderate duration
    elif "Euphoric" in regime_name:
        return 5.0   # Euphoric regimes are unsustainable
    else:
        return 10.0  # Default duration


def _classify_regime_strength(return_pct: float) -> str:
    """Classify the strength of a regime based on its return."""
    abs_return = abs(return_pct)

    if abs_return < 0.005:
        return "Weak"
    elif abs_return < 0.02:
        return "Moderate"
    elif abs_return < 0.05:
        return "Strong"
    else:
        return "Extreme"