"""
Analysis components for hidden-regime pipeline.

Provides analysis components that implement the AnalysisComponent interface
for interpreting model outputs with domain knowledge and generating insights.
"""

from .indicator_comparison import IndicatorPerformanceComparator
from .market_event_study import MarketEventStudy
from .performance import RegimePerformanceAnalyzer
from .stability import RegimeStabilityMetrics


def get_regime_color_map(regime_profiles=None):
    """
    Get consistent colorblind-safe color mapping for regime types.

    Args:
        regime_profiles: Optional Dict[int, RegimeProfile] to extract colors from.
                        If provided, colors are pulled from profiles for consistency.

    Returns:
        Dictionary mapping regime_name -> color for visualizations

    Note: This uses the colorblind-safe color scheme for accessibility.
    """
    # If regime_profiles provided, extract colors from them (single source of truth)
    if regime_profiles:
        color_map = {}
        for profile in regime_profiles.values():
            regime_name = profile.get_display_name()
            color = (
                profile.color if hasattr(profile, "color") else "#808080"
            )  # Fallback to gray
            color_map[regime_name] = color
            # Legacy support (capitalized)
            color_map[regime_name.capitalize()] = color
        return color_map

    # Fallback to default colorblind-safe colors if no profiles provided
    return {
        "bearish": "#d73027",  # Red (colorblind safe)
        "crisis": "#a50026",  # Dark red (colorblind safe)
        "sideways": "#fee08b",  # Yellow (colorblind safe)
        "bullish": "#4575b4",  # Blue (colorblind safe)
        "mixed": "#9970ab",  # Purple (colorblind safe)
        # Legacy support for old naming
        "Bear": "#d73027",
        "Crisis": "#a50026",
        "Sideways": "#fee08b",
        "Bull": "#4575b4",
    }


__all__ = [
    "RegimePerformanceAnalyzer",
    "RegimeStabilityMetrics",
    "IndicatorPerformanceComparator",
    "MarketEventStudy",
    "get_regime_color_map",
]
