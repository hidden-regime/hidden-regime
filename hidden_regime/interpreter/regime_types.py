"""
Regime type definitions and profiles for financial interpretation.

This module contains ALL financial domain knowledge about regime types:
- RegimeType enum (bull, bear, sideways, crisis, mixed)
- RegimeProfile dataclass (comprehensive financial characteristics)
- Color mappings for visualization

This enforces Principle #3: ALL financial domain knowledge in the interpreter.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RegimeType(Enum):
    """Financial regime types based on actual market behavior.

    These represent the semantic interpretation of HMM states.
    The model only knows states 0, 1, 2... The interpreter adds this meaning.
    """

    BULLISH = "bullish"  # Strong positive returns, moderate volatility
    BEARISH = "bearish"  # Negative returns, often high volatility
    SIDEWAYS = "sideways"  # Low returns, typically low volatility
    CRISIS = "crisis"  # Extreme volatility, negative returns
    MIXED = "mixed"  # Unclear financial characteristics


# Colorblind-safe color mapping for regime types
# Source: ColorBrewer2 diverging scheme (Red-Yellow-Blue)
REGIME_TYPE_COLORS = {
    RegimeType.BULLISH: "#4575b4",    # Blue (colorblind safe)
    RegimeType.BEARISH: "#d73027",    # Red (colorblind safe)
    RegimeType.SIDEWAYS: "#fee08b",   # Yellow (colorblind safe)
    RegimeType.CRISIS: "#a50026",     # Dark Red (colorblind safe)
    RegimeType.MIXED: "#9970ab",      # Purple (colorblind safe)
}


@dataclass
class RegimeProfile:
    """
    Complete financial profile of a detected regime.

    Contains ALL financial characteristics computed by the interpreter:
    - Returns and volatility (annualized)
    - Trading statistics (win rate, drawdown)
    - Regime behavior (persistence, transitions)
    - Visualization (color-blind safe colors)

    This is the OUTPUT of the interpreter - it contains all financial domain knowledge
    derived from the mathematical HMM states.
    """

    # State identification
    state_id: int  # HMM state index (0, 1, 2...)
    regime_type: RegimeType  # Enum classification (for backward compatibility)
    color: str  # Colorblind-safe hex color for visualization

    # Return characteristics
    mean_daily_return: float  # Average daily return
    daily_volatility: float  # Daily standard deviation
    annualized_return: float  # Annualized return
    annualized_volatility: float  # Annualized volatility

    # Regime behavior
    persistence_days: float  # How persistent this regime is
    regime_strength: float  # How distinct this regime is from others
    confidence_score: float  # Statistical confidence in classification

    # Trading characteristics
    win_rate: float  # Percentage of positive return days
    max_drawdown: float  # Maximum drawdown during this regime
    return_skewness: float  # Distribution shape
    return_kurtosis: float  # Tail risk

    # Transition behavior
    avg_duration: float  # Average time spent in this regime
    transition_volatility: float  # Volatility around regime changes

    # Data-driven label (replaces heuristic enum-based labeling)
    regime_type_str: Optional[str] = None

    def get_display_name(self) -> str:
        """
        Get the display name for this regime.

        Returns data-driven regime_type_str if available,
        otherwise falls back to enum value for backward compatibility.
        """
        if self.regime_type_str is not None:
            return self.regime_type_str
        return self.regime_type.value
