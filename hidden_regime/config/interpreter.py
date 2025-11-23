"""Configuration for Interpreter component.

Defines all parameters needed to configure regime interpretation,
including state labeling, regime characteristics, and visualization properties.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class InterpreterConfiguration:
    """Configuration for the Interpreter component.

    The Interpreter component adds domain knowledge to model outputs.
    This configuration controls how regime states are interpreted and labeled.

    Attributes:
        n_states: Number of HMM states (must match model)
        interpretation_method: How to assign regime labels
            - "data_driven": Use training data characteristics (default)
            - "threshold": Use fixed return thresholds
            - "manual": User-provided labels via force_regime_labels
        min_regime_days: Minimum duration (days) for regime spell to be valid
        regime_colors: Dict mapping regime labels to hex colors for visualization
        force_regime_labels: Optional list of manual regime label overrides
            If provided, must have exactly n_states labels
        acknowledge_override: Must be True if using force_regime_labels
            Prevents accidental overrides without explicit acknowledgment
    """

    n_states: int
    interpretation_method: str = "data_driven"
    min_regime_days: int = 2

    # Visualization colors
    regime_colors: Dict[str, str] = field(
        default_factory=lambda: {
            "bullish": "#2E7D32",  # Green
            "bearish": "#C62828",  # Red
            "sideways": "#F57F17",  # Orange
            "crisis": "#6A1B9A",  # Purple
            "euphoric": "#1565C0",  # Blue
        }
    )

    # Manual label override
    force_regime_labels: Optional[List[str]] = None
    acknowledge_override: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate interpretation method
        valid_methods = ["data_driven", "threshold", "manual"]
        if self.interpretation_method not in valid_methods:
            raise ValueError(
                f"interpretation_method must be one of {valid_methods}, "
                f"got {self.interpretation_method}"
            )

        # Validate n_states
        if self.n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {self.n_states}")
        if self.n_states > 10:
            raise ValueError(f"n_states must be <= 10, got {self.n_states}")

        # Validate manual labels if provided
        if self.force_regime_labels is not None:
            if not self.acknowledge_override:
                raise ValueError(
                    "Using force_regime_labels requires acknowledge_override=True. "
                    "Please set acknowledge_override=True to confirm you want to override "
                    "automatic regime labeling."
                )
            if len(self.force_regime_labels) != self.n_states:
                raise ValueError(
                    f"force_regime_labels must have exactly {self.n_states} labels, "
                    f"got {len(self.force_regime_labels)}"
                )

        # Validate min_regime_days
        if self.min_regime_days < 1:
            raise ValueError(f"min_regime_days must be >= 1, got {self.min_regime_days}")

    def get_regime_color(self, regime_label: str) -> str:
        """Get color for a specific regime label.

        Args:
            regime_label: Regime label (e.g., "bullish", "bearish")

        Returns:
            Hex color code for visualization

        Raises:
            KeyError: If regime_label not found in colors
        """
        if regime_label not in self.regime_colors:
            # Fallback to default if not found
            return "#808080"  # Gray
        return self.regime_colors[regime_label]

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "n_states": self.n_states,
            "interpretation_method": self.interpretation_method,
            "min_regime_days": self.min_regime_days,
            "regime_colors": self.regime_colors,
            "force_regime_labels": self.force_regime_labels,
            "acknowledge_override": self.acknowledge_override,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "InterpreterConfiguration":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            New InterpreterConfiguration instance
        """
        return cls(**config_dict)
