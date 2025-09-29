"""
Unified regime mapping utilities for consistent regime interpretation across the system.

Provides centralized functions to map RegimeProfile objects to visualization-ready
labels and colors, ensuring consistency between financial characterization and
visualization systems.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from ..financial.regime_characterizer import RegimeProfile, RegimeType


# Base color mapping for regime types
REGIME_TYPE_COLORS = {
    RegimeType.BULLISH: '#2E8B57',      # Sea Green
    RegimeType.BEARISH: '#DC143C',      # Crimson Red
    RegimeType.SIDEWAYS: '#DAA520',     # Golden Rod
    RegimeType.CRISIS: '#8B0000',       # Dark Red
    RegimeType.MIXED: '#9370DB'         # Medium Purple
}

# Enhanced color variations for multiple regimes of same type
REGIME_COLOR_VARIATIONS = {
    RegimeType.BULLISH: [
        '#228B22',  # Forest Green (strong)
        '#32CD32',  # Lime Green (moderate)
        '#90EE90',  # Light Green (weak)
        '#006400',  # Dark Green (very strong)
        '#9ACD32'   # Yellow Green (emerging)
    ],
    RegimeType.BEARISH: [
        '#B22222',  # Fire Brick (strong)
        '#CD5C5C',  # Indian Red (moderate)
        '#F08080',  # Light Coral (weak)
        '#8B0000',  # Dark Red (very strong)
        '#A0522D'   # Sienna (emerging)
    ],
    RegimeType.SIDEWAYS: [
        '#B8860B',  # Dark Golden Rod
        '#DAA520',  # Golden Rod
        '#F0E68C',  # Khaki
        '#D2691E',  # Chocolate
        '#BDB76B'   # Dark Khaki
    ],
    RegimeType.CRISIS: [
        '#8B0000',  # Dark Red
        '#800000',  # Maroon
        '#DC143C',  # Crimson
        '#B22222',  # Fire Brick
        '#CD5C5C'   # Indian Red
    ],
    RegimeType.MIXED: [
        '#9370DB',  # Medium Purple
        '#8A2BE2',  # Blue Violet
        '#9932CC',  # Dark Orchid
        '#BA55D3',  # Medium Orchid
        '#DDA0DD'   # Plum
    ]
}


def create_regime_mapping(regime_profiles: Dict[int, RegimeProfile]) -> Dict[str, any]:
    """
    Create comprehensive regime mapping from RegimeProfile objects.

    Args:
        regime_profiles: Dictionary mapping state IDs to RegimeProfile objects

    Returns:
        Dictionary containing labels, colors, and metadata for visualization
    """
    if not regime_profiles:
        return {
            'labels': [],
            'colors': [],
            'state_to_label': {},
            'state_to_color': {},
            'regime_types': [],
            'metadata': {}
        }

    # Group regimes by type for intelligent labeling
    regimes_by_type = {}
    for state_id, profile in regime_profiles.items():
        regime_type = profile.regime_type
        if regime_type not in regimes_by_type:
            regimes_by_type[regime_type] = []
        regimes_by_type[regime_type].append((state_id, profile))

    # Generate labels and colors
    state_to_label = {}
    state_to_color = {}
    labels = []
    colors = []
    regime_types = []

    for regime_type, regimes in regimes_by_type.items():
        # Sort regimes of same type by return (descending)
        regimes.sort(key=lambda x: x[1].annualized_return, reverse=True)

        # Generate distinct labels for multiple regimes of same type
        if len(regimes) == 1:
            state_id, profile = regimes[0]
            label = _create_single_regime_label(profile)
            color = REGIME_TYPE_COLORS[regime_type]
        else:
            # Multiple regimes of same type - create descriptive labels
            for i, (state_id, profile) in enumerate(regimes):
                label = _create_multi_regime_label(profile, i, len(regimes))
                color = _get_regime_color_variation(regime_type, i)

                state_to_label[state_id] = label
                state_to_color[state_id] = color
                labels.append(label)
                colors.append(color)
                regime_types.append(regime_type)
            continue

        state_to_label[state_id] = label
        state_to_color[state_id] = color
        labels.append(label)
        colors.append(color)
        regime_types.append(regime_type)

    # Create ordered lists based on state IDs
    ordered_labels = []
    ordered_colors = []
    ordered_types = []

    for state_id in sorted(regime_profiles.keys()):
        ordered_labels.append(state_to_label[state_id])
        ordered_colors.append(state_to_color[state_id])
        ordered_types.append(regime_profiles[state_id].regime_type)

    return {
        'labels': ordered_labels,
        'colors': ordered_colors,
        'state_to_label': state_to_label,
        'state_to_color': state_to_color,
        'regime_types': ordered_types,
        'metadata': {
            'n_states': len(regime_profiles),
            'regimes_by_type': {str(k): len(v) for k, v in regimes_by_type.items()},
            'has_multiple_same_type': any(len(v) > 1 for v in regimes_by_type.values())
        }
    }


def _create_single_regime_label(profile: RegimeProfile) -> str:
    """Create label for a single regime of its type."""
    base_name = profile.regime_type.value.title()

    # Add confidence indicator if low confidence
    if profile.confidence_score < 0.5:
        return f"{base_name}*"  # Asterisk indicates low confidence

    return base_name


def _create_multi_regime_label(profile: RegimeProfile, index: int, total: int) -> str:
    """Create descriptive label for multiple regimes of same type."""
    base_name = profile.regime_type.value.title()

    # Create descriptive modifiers based on relative performance
    if profile.regime_type in [RegimeType.BULLISH, RegimeType.BEARISH]:
        return_pct = profile.annualized_return * 100

        if index == 0:  # Highest return
            modifier = "Strong"
        elif index == total - 1:  # Lowest return
            modifier = "Weak"
        else:
            modifier = "Moderate"

        # Include return percentage for clarity
        return f"{modifier} {base_name} ({return_pct:+.0f}%)"

    elif profile.regime_type == RegimeType.SIDEWAYS:
        vol_pct = profile.annualized_volatility * 100

        if index == 0:
            modifier = "Stable"
        elif index == total - 1:
            modifier = "Volatile"
        else:
            modifier = "Typical"

        return f"{modifier} {base_name} ({vol_pct:.0f}% vol)"

    else:  # CRISIS or MIXED
        if total > 1:
            return f"{base_name} {index + 1}"
        return base_name


def _get_regime_color_variation(regime_type: RegimeType, index: int) -> str:
    """Get color variation for multiple regimes of same type."""
    variations = REGIME_COLOR_VARIATIONS.get(regime_type, [REGIME_TYPE_COLORS[regime_type]])

    if index < len(variations):
        return variations[index]
    else:
        # Fallback to base color if we run out of variations
        return REGIME_TYPE_COLORS[regime_type]


def get_regime_labels_from_profiles(regime_profiles: Dict[int, RegimeProfile]) -> List[str]:
    """
    Get ordered regime labels from RegimeProfile objects.

    Args:
        regime_profiles: Dictionary mapping state IDs to RegimeProfile objects

    Returns:
        List of regime labels ordered by state ID
    """
    mapping = create_regime_mapping(regime_profiles)
    return mapping['labels']


def get_regime_colors_from_profiles(regime_profiles: Dict[int, RegimeProfile]) -> List[str]:
    """
    Get ordered regime colors from RegimeProfile objects.

    Args:
        regime_profiles: Dictionary mapping state IDs to RegimeProfile objects

    Returns:
        List of regime colors ordered by state ID
    """
    mapping = create_regime_mapping(regime_profiles)
    return mapping['colors']


def get_regime_mapping_for_visualization(regime_profiles: Optional[Dict[int, RegimeProfile]],
                                       n_states: int) -> Tuple[List[str], List[str]]:
    """
    Get regime labels and colors suitable for visualization functions.

    This function provides a fallback to generic labels if regime profiles are not available,
    ensuring backward compatibility with existing visualization functions.

    Args:
        regime_profiles: Optional dictionary mapping state IDs to RegimeProfile objects
        n_states: Number of states (used for fallback)

    Returns:
        Tuple of (labels, colors) lists
    """
    if regime_profiles and len(regime_profiles) == n_states:
        # Use financial characterization
        mapping = create_regime_mapping(regime_profiles)
        return mapping['labels'], mapping['colors']
    else:
        # Fallback to generic labels (backward compatibility)
        from ..visualization.plotting import get_regime_names, get_regime_colors
        labels = get_regime_names(n_states)
        colors = get_regime_colors(n_states)
        return labels, colors


def format_regime_summary(regime_profiles: Dict[int, RegimeProfile]) -> str:
    """
    Create a human-readable summary of regime characteristics.

    Args:
        regime_profiles: Dictionary mapping state IDs to RegimeProfile objects

    Returns:
        Formatted string summarizing regime types and characteristics
    """
    if not regime_profiles:
        return "No regime profiles available"

    mapping = create_regime_mapping(regime_profiles)

    # Count regimes by type
    type_counts = {}
    for regime_type in mapping['regime_types']:
        type_name = regime_type.value.title()
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    # Format summary
    summary_parts = []
    for regime_type, count in type_counts.items():
        if count == 1:
            summary_parts.append(f"{count} {regime_type}")
        else:
            summary_parts.append(f"{count} {regime_type} variants")

    if len(summary_parts) == 1:
        return summary_parts[0]
    elif len(summary_parts) == 2:
        return f"{summary_parts[0]} and {summary_parts[1]}"
    else:
        return f"{', '.join(summary_parts[:-1])}, and {summary_parts[-1]}"