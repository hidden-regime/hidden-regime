"""
Trading strategy module.

This package contains the strategy system that interprets RegimeLabel objects
and generates TradingSignal objects.

Key components:
- Strategy: Abstract base class for all trading strategies
- RiskManagementConfig: Risk parameters bundled with strategies
- Concrete strategy implementations (RegimeFollowing, Contrarian, etc.)
- Decorator/wrapper strategies for composability (ConfidenceWeighted, etc.)
"""

from hidden_regime.strategies.base import (
    RiskManagementConfig,
    Strategy,
)
from hidden_regime.strategies.implementations import (
    ConfidenceWeightedStrategy,
    ContrarianStrategy,
    MultiTimeframeAlignmentStrategy,
    RegimeFollowingStrategy,
    VolatilityAdjustedStrategy,
)

__all__ = [
    "Strategy",
    "RiskManagementConfig",
    "RegimeFollowingStrategy",
    "ContrarianStrategy",
    "ConfidenceWeightedStrategy",
    "VolatilityAdjustedStrategy",
    "MultiTimeframeAlignmentStrategy",
]
