"""Configuration for Strategy components.

Defines all parameters needed to configure trading strategies and their composition.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StrategyConfiguration:
    """Base configuration for Strategy components.

    Attributes:
        strategy_type: Type of strategy to create
            - "regime_following": Follow regime direction
            - "contrarian": Fade regime direction
            - "confidence_weighted": Wrap another strategy, scale by regime confidence
            - "volatility_adjusted": Wrap another strategy, reduce in high vol
            - "multi_timeframe_alignment": Wrap another strategy, filter by alignment
        base_strategy: Name of base strategy to wrap (for decorator strategies)
        long_confidence_threshold: Minimum confidence for long signals (0-1)
        short_confidence_threshold: Minimum confidence for short signals (0-1)
    """

    strategy_type: str = "regime_following"
    base_strategy: Optional[str] = None
    long_confidence_threshold: float = 0.65
    short_confidence_threshold: float = 0.65

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        valid_types = [
            "regime_following",
            "contrarian",
            "confidence_weighted",
            "volatility_adjusted",
            "multi_timeframe_alignment",
        ]
        if self.strategy_type not in valid_types:
            raise ValueError(
                f"strategy_type must be one of {valid_types}, "
                f"got {self.strategy_type}"
            )

        if not (0.0 <= self.long_confidence_threshold <= 1.0):
            raise ValueError(
                f"long_confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.long_confidence_threshold}"
            )

        if not (0.0 <= self.short_confidence_threshold <= 1.0):
            raise ValueError(
                f"short_confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.short_confidence_threshold}"
            )

        if self.strategy_type in [
            "confidence_weighted",
            "volatility_adjusted",
            "multi_timeframe_alignment",
        ] and not self.base_strategy:
            raise ValueError(
                f"Decorator strategy '{self.strategy_type}' requires base_strategy"
            )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "strategy_type": self.strategy_type,
            "base_strategy": self.base_strategy,
            "long_confidence_threshold": self.long_confidence_threshold,
            "short_confidence_threshold": self.short_confidence_threshold,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "StrategyConfiguration":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            New StrategyConfiguration instance
        """
        return cls(**config_dict)


@dataclass
class ComposedStrategyConfiguration:
    """Configuration for composing multiple strategies.

    Enables creating complex strategy chains for A/B testing and research.

    Attributes:
        base_strategy_config: Configuration for the base strategy
        decorator_layers: List of decorator strategy configs to apply in order
            Applied from first to last, so last is outermost
            Example: [confidence_weighted, volatility_adjusted]
            Creates: VolatilityAdjusted(ConfidenceWeighted(RegimeFollowing))
    """

    base_strategy_config: StrategyConfiguration
    decorator_layers: List[StrategyConfiguration] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.base_strategy_config.strategy_type in [
            "confidence_weighted",
            "volatility_adjusted",
            "multi_timeframe_alignment",
        ]:
            raise ValueError(
                "base_strategy_config cannot be a decorator strategy"
            )

        for i, decorator_config in enumerate(self.decorator_layers):
            if decorator_config.strategy_type not in [
                "confidence_weighted",
                "volatility_adjusted",
                "multi_timeframe_alignment",
            ]:
                raise ValueError(
                    f"decorator_layers[{i}] must be a decorator strategy, "
                    f"got {decorator_config.strategy_type}"
                )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "base_strategy_config": self.base_strategy_config.to_dict(),
            "decorator_layers": [
                config.to_dict() for config in self.decorator_layers
            ],
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ComposedStrategyConfiguration":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration

        Returns:
            New ComposedStrategyConfiguration instance
        """
        base_config = StrategyConfiguration.from_dict(
            config_dict["base_strategy_config"]
        )
        decorator_configs = [
            StrategyConfiguration.from_dict(d) for d in config_dict.get("decorator_layers", [])
        ]
        return cls(
            base_strategy_config=base_config,
            decorator_layers=decorator_configs,
        )
