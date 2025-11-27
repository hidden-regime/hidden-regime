"""
Base strategy class and risk management configuration.

This module defines the Strategy abstraction that translates RegimeLabel objects
into TradingSignal objects. Strategies can be:
- Core strategies (leaf nodes): RegimeFollowing, Contrarian
- Wrapper strategies (decorators): ConfidenceWeighted, VolatilityAdjusted, MultiTimeframeAlignment

The decorator pattern allows composable strategies:
  ConfidenceWeighted(MultiTimeframeAlignment(RegimeFollowing()))
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from hidden_regime.interpreter.regime_types import RegimeLabel
from hidden_regime.signals.types import TradingSignal


@dataclass
class RiskManagementConfig:
    """
    Risk management constraints applied to all strategies.

    These are hard limits and directives that apply regardless of strategy type:
    - Position sizing limits (max_position_size, kelly_criterion)
    - Restriction rules (prevent_shorts, prevent_longs)
    - Loss limits (max_daily_loss)

    Bundled with each Strategy to ensure explicit, auditable risk parameters.
    """

    # Position sizing
    max_position_size: float = 1.0  # Maximum position size (0-1, can exceed for leverage)
    kelly_criterion: Optional[bool] = None  # Use Kelly criterion for sizing? None = default strategy behavior
    kelly_fraction: float = 0.25  # Fraction of full Kelly to use (conservative)

    # Direction restrictions
    prevent_shorts: bool = False  # If True, convert shorts to neutral
    prevent_longs: bool = False  # If True, convert longs to neutral

    # Loss limits
    max_daily_loss: Optional[float] = None  # Max loss in one day (e.g., 0.02 = 2%)

    # Position sizing hints
    position_sizing_bias: float = 1.0  # 1.0 = normal, 0.5 = half, 2.0 = double


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Strategies consume RegimeLabel objects and produce TradingSignal objects.
    All strategies are immutable after creation to prevent accidental modification.

    Key responsibilities:
    1. Map RegimeLabel → TradingSignal (direction, position size, confidence)
    2. Respect RiskManagementConfig constraints
    3. Provide strategy name for audit trail
    4. Support composition via wrapping (decorator pattern)

    Example usage:
        strategy = RegimeFollowingStrategy(
            long_confidence_threshold=0.65,
            short_confidence_threshold=0.65,
            risk_management=RiskManagementConfig(prevent_shorts=True)
        )
        signal = strategy.get_signal_for_regime(regime_label)
    """

    def __init__(self, name: str, description: str, risk_management: Optional[RiskManagementConfig] = None):
        """Initialize strategy with configuration.

        Args:
            name: Human-readable name of this strategy
            description: Description of what this strategy does
            risk_management: RiskManagementConfig with hard constraints
        """
        self.name = name
        self.description = description
        self.risk_management = risk_management or RiskManagementConfig()

    @abstractmethod
    def get_signal_for_regime(self, regime: RegimeLabel) -> TradingSignal:
        """
        Generate a trading signal from a regime label.

        This is the core method that all strategies must implement.
        It translates "BULLISH regime" into "Long signal with 0.8 position size".

        Args:
            regime: RegimeLabel object from the Interpreter

        Returns:
            TradingSignal with direction, position_size, confidence, etc.
        """
        pass
