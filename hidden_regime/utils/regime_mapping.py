"""
Unified regime mapping utilities for consistent regime interpretation across the system.

Provides centralized functions to map RegimeProfile objects to visualization-ready
labels and colors, ensuring consistency between financial characterization and
visualization systems.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..interpreter.regime_types import RegimeProfile, RegimeType

# Base color mapping for regime types (colorblind-safe)
# Source: ColorBrewer2 diverging scheme (Red-Yellow-Blue)
REGIME_TYPE_COLORS = {
    RegimeType.BULLISH: "#4575b4",    # Blue (colorblind safe)
    RegimeType.BEARISH: "#d73027",    # Red (colorblind safe)
    RegimeType.SIDEWAYS: "#fee08b",   # Yellow (colorblind safe)
    RegimeType.CRISIS: "#a50026",     # Dark Red (colorblind safe)
    RegimeType.MIXED: "#9970ab",      # Purple (colorblind safe)
}

# Enhanced color variations for multiple regimes of same type
REGIME_COLOR_VARIATIONS = {
    RegimeType.BULLISH: [
        "#228B22",  # Forest Green (strong)
        "#32CD32",  # Lime Green (moderate)
        "#90EE90",  # Light Green (weak)
        "#006400",  # Dark Green (very strong)
        "#9ACD32",  # Yellow Green (emerging)
    ],
    RegimeType.BEARISH: [
        "#B22222",  # Fire Brick (strong)
        "#CD5C5C",  # Indian Red (moderate)
        "#F08080",  # Light Coral (weak)
        "#8B0000",  # Dark Red (very strong)
        "#A0522D",  # Sienna (emerging)
    ],
    RegimeType.SIDEWAYS: [
        "#B8860B",  # Dark Golden Rod
        "#DAA520",  # Golden Rod
        "#F0E68C",  # Khaki
        "#D2691E",  # Chocolate
        "#BDB76B",  # Dark Khaki
    ],
    RegimeType.CRISIS: [
        "#8B0000",  # Dark Red
        "#800000",  # Maroon
        "#DC143C",  # Crimson
        "#B22222",  # Fire Brick
        "#CD5C5C",  # Indian Red
    ],
    RegimeType.MIXED: [
        "#9370DB",  # Medium Purple
        "#8A2BE2",  # Blue Violet
        "#9932CC",  # Dark Orchid
        "#BA55D3",  # Medium Orchid
        "#DDA0DD",  # Plum
    ],
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
            "labels": [],
            "colors": [],
            "state_to_label": {},
            "state_to_color": {},
            "regime_types": [],
            "metadata": {},
        }

    # Group regimes by label for intelligent labeling
    # Use data-driven labels, not enum
    regimes_by_label = {}
    for state_id, profile in regime_profiles.items():
        label = profile.get_display_name()
        if label not in regimes_by_label:
            regimes_by_label[label] = []
        regimes_by_label[label].append((state_id, profile))

    # Generate labels and colors
    state_to_label = {}
    state_to_color = {}
    labels = []
    colors = []
    regime_types = []

    # Process each unique label
    for label, regimes in regimes_by_label.items():
        for state_id, profile in regimes:
            # Use the data-driven label directly
            display_label = profile.get_display_name()

            # Get color from profile (already assigned in data-driven classification)
            # Fallback to pattern-based color if not set
            if hasattr(profile, 'color') and profile.color:
                color = profile.color
            else:
                color = _get_color_from_label(display_label)

            state_to_label[state_id] = display_label
            state_to_color[state_id] = color
            labels.append(display_label)
            colors.append(color)
            # Store the data-driven label instead of enum
            regime_types.append(display_label)

    # Create ordered lists based on state IDs
    ordered_labels = []
    ordered_colors = []
    ordered_types = []

    for state_id in sorted(regime_profiles.keys()):
        ordered_labels.append(state_to_label[state_id])
        ordered_colors.append(state_to_color[state_id])
        # Use data-driven label instead of enum
        ordered_types.append(regime_profiles[state_id].get_display_name())

    return {
        "labels": ordered_labels,
        "colors": ordered_colors,
        "state_to_label": state_to_label,
        "state_to_color": state_to_color,
        "regime_types": ordered_types,
        "metadata": {
            "n_states": len(regime_profiles),
            "regimes_by_label": {str(k): len(v) for k, v in regimes_by_label.items()},
            "has_multiple_same_label": any(len(v) > 1 for v in regimes_by_label.values()),
        },
    }


def _get_color_from_label(label: str) -> str:
    """
    Get color based on data-driven label string (pattern matching).

    This function supports ANY label, not just predefined ones.
    Colors are assigned based on keyword patterns in the label.
    """
    label_lower = label.lower()

    # Bullish patterns - shades of blue/green
    if 'strong_bullish' in label_lower or 'very_bullish' in label_lower:
        return "#006400"  # Dark Green (very strong)
    elif 'weak_bullish' in label_lower:
        return "#90EE90"  # Light Green (weak)
    elif 'bullish' in label_lower:
        return "#4575b4"  # Blue (standard bullish)

    # Bearish patterns - shades of red
    elif 'strong_bearish' in label_lower or 'very_bearish' in label_lower:
        return "#8B0000"  # Dark Red (very strong)
    elif 'weak_bearish' in label_lower:
        return "#F08080"  # Light Coral (weak)
    elif 'bearish' in label_lower:
        return "#d73027"  # Red (standard bearish)

    # Sideways/neutral patterns - shades of yellow/gold
    elif 'sideways' in label_lower or 'neutral' in label_lower:
        return "#fee08b"  # Yellow

    # Crisis patterns - dark red
    elif 'crisis' in label_lower:
        return "#a50026"  # Dark Red (crisis)

    # Default for any other label
    else:
        return "#9970ab"  # Purple (mixed/unknown)


def get_regime_labels_from_profiles(
    regime_profiles: Dict[int, RegimeProfile],
) -> List[str]:
    """
    Get ordered regime labels from RegimeProfile objects.

    Args:
        regime_profiles: Dictionary mapping state IDs to RegimeProfile objects

    Returns:
        List of regime labels ordered by state ID
    """
    mapping = create_regime_mapping(regime_profiles)
    return mapping["labels"]


def get_regime_colors_from_profiles(
    regime_profiles: Dict[int, RegimeProfile],
) -> List[str]:
    """
    Get ordered regime colors from RegimeProfile objects.

    Args:
        regime_profiles: Dictionary mapping state IDs to RegimeProfile objects

    Returns:
        List of regime colors ordered by state ID
    """
    mapping = create_regime_mapping(regime_profiles)
    return mapping["colors"]


def get_regime_mapping_for_visualization(
    regime_profiles: Optional[Dict[int, RegimeProfile]], n_states: int
) -> Tuple[List[str], List[str]]:
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
        return mapping["labels"], mapping["colors"]
    else:
        # Fallback to generic labels (backward compatibility)
        from ..visualization.plotting import get_regime_colors, get_regime_names

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

    # Count regimes by label (data-driven)
    label_counts = {}
    for label in mapping["regime_types"]:
        # label is now a string, not an enum
        label_counts[label] = label_counts.get(label, 0) + 1

    # Format summary
    summary_parts = []
    for label, count in label_counts.items():
        if count == 1:
            summary_parts.append(f"{count} {label}")
        else:
            summary_parts.append(f"{count} {label} variants")

    if len(summary_parts) == 1:
        return summary_parts[0]
    elif len(summary_parts) == 2:
        return f"{summary_parts[0]} and {summary_parts[1]}"
    else:
        return f"{', '.join(summary_parts[:-1])}, and {summary_parts[-1]}"
