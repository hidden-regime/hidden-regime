"""Configuration for Signal Generation component.

Defines all parameters needed to configure trading signal generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SignalGenerationConfiguration:
    """Configuration for the Signal Generator component.

    The Signal Generator creates trading signals from Interpreter outputs.
    This configuration controls signal generation strategies and position sizing.

    Attributes:
        strategy_type: Type of signal generation strategy
            - "regime_following": Long bull regimes, short bear regimes
            - "regime_contrarian": Opposite of regime (fade signals)
            - "confidence_weighted": Position size scales with regime confidence
            - "multi_timeframe": Use timeframe alignment as filter
        confidence_threshold: Minimum regime confidence to generate signal (0-1)
        position_size_range: (min, max) position size scaling
        enable_regime_change_exits: Exit immediately on regime change
        lookback_days: Window for calculating signal statistics
    """

    strategy_type: str = "regime_following"
    confidence_threshold: float = 0.70
    position_size_range: tuple = field(default_factory=lambda: (0.0, 1.0))
    enable_regime_change_exits: bool = True
    lookback_days: int = 20

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate strategy type
        valid_strategies = [
            "regime_following",
            "regime_contrarian",
            "confidence_weighted",
            "multi_timeframe",
        ]
        if self.strategy_type not in valid_strategies:
            raise ValueError(
                f"strategy_type must be one of {valid_strategies}, "
                f"got {self.strategy_type}"
            )

        # Validate confidence threshold
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.confidence_threshold}"
            )

        # Validate position size range
        min_size, max_size = self.position_size_range
        if not (0.0 <= min_size <= max_size <= 10.0):
            raise ValueError(
                f"position_size_range must be (min, max) with 0 <= min <= max <= 10, "
                f"got {self.position_size_range}"
            )

        # Validate lookback
        if self.lookback_days < 1:
            raise ValueError(f"lookback_days must be >= 1, got {self.lookback_days}")

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "strategy_type": self.strategy_type,
            "confidence_threshold": self.confidence_threshold,
            "position_size_range": self.position_size_range,
            "enable_regime_change_exits": self.enable_regime_change_exits,
            "lookback_days": self.lookback_days,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "SignalGenerationConfiguration":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            New SignalGenerationConfiguration instance
        """
        return cls(**config_dict)
