"""Base Interpreter implementation.

Provides the foundation for all interpreter components.
Handles regime labeling, color assignment, and characteristic calculation.
"""

from abc import abstractmethod
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hidden_regime.config.interpreter import InterpreterConfiguration
from hidden_regime.pipeline.interfaces import InterpreterComponent


class BaseInterpreter(InterpreterComponent):
    """Base interpreter for regime interpretation.

    Provides common functionality for all interpreter implementations.
    Subclasses should implement specific regime labeling strategies.

    This component takes model output (state indices + confidence) and adds:
    - Regime labels (e.g., "Bear", "Bull", "Sideways")
    - Regime characteristics (returns, volatility, duration)
    - Regime colors for visualization
    - Regime strength/confidence scores
    """

    def __init__(self, config: InterpreterConfiguration):
        """Initialize interpreter.

        Args:
            config: InterpreterConfiguration object
        """
        self.config = config
        self._regime_profiles: Optional[Dict[int, Dict]] = None
        self._state_labels: Optional[Dict[int, str]] = None

    @abstractmethod
    def _assign_regime_labels(
        self, model_output: pd.DataFrame
    ) -> Dict[int, str]:
        """Assign regime labels to states.

        Must be implemented by subclasses.

        Args:
            model_output: Model output with state indices and parameters

        Returns:
            Dictionary mapping state index to regime label
        """
        pass

    def update(self, model_output: pd.DataFrame) -> pd.DataFrame:
        """Interpret model output and add regime information.

        Args:
            model_output: Raw model predictions with columns:
                - timestamp: datetime
                - state: int (0, 1, 2, ...)
                - confidence: float (max probability)
                - state_probabilities: array (all state probabilities)
                - [optional] emission_means: array
                - [optional] emission_stds: array

        Returns:
            DataFrame with added columns:
                - regime_label: str (regime name)
                - regime_type: str (regime type category)
                - regime_color: str (hex color)
                - regime_strength: float (0-1 strength)
        """
        # Make a copy to avoid modifying input
        output = model_output.copy()

        # Assign state labels (once per component lifecycle)
        if self._state_labels is None:
            self._state_labels = self._assign_regime_labels(model_output)

        # Add regime labels based on state mapping
        output["regime_label"] = output["state"].map(self._state_labels)

        # Add regime colors
        output["regime_color"] = output["regime_label"].apply(
            self.config.get_regime_color
        )

        # Add regime type (category)
        # If using forced labels, use them as-is for regime_type
        if self.config.force_regime_labels is not None:
            output["regime_type"] = output["regime_label"]
        else:
            output["regime_type"] = output["regime_label"].apply(
                self._get_regime_type
            )

        # Add regime strength (based on confidence)
        output["regime_strength"] = output["confidence"]
        # Also add regime_confidence as an alias for clarity
        output["regime_confidence"] = output["confidence"]

        # Calculate regime characteristics if available
        if "emission_means" in model_output.columns and "emission_stds" in model_output.columns:
            output["regime_return"] = output.apply(
                lambda row: self._get_regime_return(row["state"]),
                axis=1
            )
            output["regime_volatility"] = output.apply(
                lambda row: self._get_regime_volatility(row["state"]),
                axis=1
            )

        return output

    def _get_regime_type(self, regime_label: str) -> str:
        """Get regime type category from label.

        Args:
            regime_label: Regime label (e.g., "Bull", "Bear")

        Returns:
            Regime type category (e.g., "bullish", "bearish")
        """
        if regime_label is None:
            return "unknown"

        label_lower = regime_label.lower()

        # "Crisis" labels indicate extreme market conditions (priority over direction)
        # "Crash" labels indicate direction with high volatility (direction takes priority)
        if label_lower.startswith("crisis"):
            return "crisis"
        elif "bear" in label_lower or "downtrend" in label_lower:
            return "bearish"
        elif "bull" in label_lower or "uptrend" in label_lower:
            return "bullish"
        elif "crash" in label_lower:  # Standalone crash without direction
            return "crisis"
        elif "sideways" in label_lower or "range" in label_lower:
            return "sideways"
        else:
            return "neutral"

    def _get_regime_return(self, state: int) -> Optional[float]:
        """Get expected return for a regime state.

        Args:
            state: State index

        Returns:
            Expected annualized return or None
        """
        if self._regime_profiles is None or state not in self._regime_profiles:
            return None
        return self._regime_profiles[state].get("expected_return")

    def _get_regime_volatility(self, state: int) -> Optional[float]:
        """Get expected volatility for a regime state.

        Args:
            state: State index

        Returns:
            Expected annualized volatility or None
        """
        if self._regime_profiles is None or state not in self._regime_profiles:
            return None
        return self._regime_profiles[state].get("expected_volatility")

    def plot(self, **kwargs) -> plt.Figure:
        """Generate visualization for interpreter.

        Creates a simple informational plot about regime labels.

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if self._state_labels is None:
            ax.text(0.5, 0.5, "No regime labels assigned yet", ha="center", va="center")
        else:
            # Create a simple legend of regime labels and colors
            y_pos = 0.9
            ax.text(0.5, y_pos, "Regime Interpretation", ha="center", fontsize=14, fontweight="bold")
            y_pos -= 0.1

            for state, label in sorted(self._state_labels.items()):
                color = self.config.get_regime_color(label)
                ax.add_patch(
                    plt.Rectangle((0.1, y_pos - 0.03), 0.05, 0.05, facecolor=color, edgecolor="black")
                )
                ax.text(0.2, y_pos, f"State {state}: {label}", va="center", fontsize=11)
                y_pos -= 0.1

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        return fig
