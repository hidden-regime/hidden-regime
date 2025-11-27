"""
Regime type definitions for financial interpretation.

This module contains the SINGLE SOURCE OF TRUTH for regime types:
- RegimeLabel: Complete encapsulation of a regime's financial and trading semantics
- RegimeCharacteristics: Financial metrics (returns, volatility, drawdown, etc.)
- TradingSemantics: How this regime should be traded (bias, position sign, thresholds)
- Color mappings for visualization

This enforces the Architecture Principle: ALL financial domain knowledge in one place.
RegimeLabel is what the Interpreter PRODUCES and what Strategy objects CONSUME.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Literal, Optional


class RegimeType(Enum):
    """Financial regime types based on actual market behavior.

    These represent the semantic interpretation of HMM states.
    The model only knows states 0, 1, 2... The interpreter adds this meaning.

    Kept for backward compatibility and visualization lookups.
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


@dataclass(frozen=True)
class RegimeCharacteristics:
    """
    Financial metrics characterizing a regime type.

    These are statistics computed from actual market data within this regime.
    Immutable to prevent accidental modification.
    """

    # Return characteristics
    mean_daily_return: float
    annualized_return: float
    daily_volatility: float
    annualized_volatility: float

    # Trading characteristics
    win_rate: float  # Percentage of positive return days
    max_drawdown: float  # Maximum drawdown during this regime
    return_skewness: float  # Distribution shape
    return_kurtosis: float  # Tail risk
    sharpe_ratio: float  # Risk-adjusted return

    # Regime behavior
    persistence_days: float  # How persistent this regime is (avg duration)
    regime_strength: float  # How distinct from others (0-1)
    transition_volatility: float  # Volatility around regime changes

    # Transition probabilities to other regimes
    transition_probs: Dict[str, float] = field(default_factory=dict)

    # Optional state-level info
    state_id: Optional[int] = None


@dataclass(frozen=True)
class TradingSemantics:
    """
    How this regime should be fundamentally traded.

    These are strategy-agnostic fundamentals that ANY trading strategy should respect.
    Immutable to prevent accidental modification.
    """

    # Fundamental bias
    bias: Literal["positive", "negative", "neutral"]

    # Typical position direction (strategy-independent)
    typical_position_sign: int  # +1 for long, -1 for short, 0 for flat/neutral

    # When is this regime actionable?
    confidence_threshold: float  # Minimum confidence to act on this regime

    # Position sizing hints
    position_sizing_bias: float = 1.0  # 1.0 = normal, 0.5 = half, 2.0 = double


@dataclass(frozen=True)
class RegimeLabel:
    """
    SINGLE SOURCE OF TRUTH for what a regime means financially and tradingly.

    This is what the Interpreter PRODUCES.
    This is what Strategy objects CONSUME.

    By encapsulating regime semantics here, we prevent interpretation drift:
    - Change RegimeLabel definition once
    - All strategies automatically use the new definition
    - QuantConnect reads the same RegimeLabel, no re-interpretation needed

    Immutable to prevent accidental modification and enable safe sharing.
    """

    # Identity
    name: str  # "BULLISH", "BEARISH", "SIDEWAYS", "CRISIS", "MIXED"
    regime_type: RegimeType  # Enum for color lookup and backward compat
    color: str  # Colorblind-safe hex color for visualization

    # Financial characteristics (what this regime looks like in data)
    characteristics: RegimeCharacteristics

    # Trading semantics (how this regime should be traded)
    trading_semantics: TradingSemantics

    # Confidence in this classification
    regime_strength: float  # 0-1, how distinct this regime is

    # Optional metadata (e.g., timeframe_alignment for multi-timeframe strategies)
    metadata: dict = field(default_factory=dict)

    def get_display_name(self) -> str:
        """Get the display name for this regime."""
        return self.name

    def __hash__(self):
        """Make RegimeLabel hashable for use in dicts."""
        return hash(self.name)
