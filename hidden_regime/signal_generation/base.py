"""Base Signal Generator implementation.

Provides the foundation for all signal generation components.
Handles position signal generation, position sizing, and trading logic.
"""

from abc import abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hidden_regime.config.signal_generation import SignalGenerationConfiguration
from hidden_regime.pipeline.interfaces import SignalGeneratorComponent


class BaseSignalGenerator(SignalGeneratorComponent):
    """Base signal generator for creating trading signals.

    Takes Interpreter output (regime labels + characteristics) and generates
    trading signals (position direction and size).

    Subclasses implement specific trading strategies.
    """

    def __init__(self, config: SignalGenerationConfiguration):
        """Initialize signal generator.

        Args:
            config: SignalGenerationConfiguration object
        """
        self.config = config
        self._previous_regime: Optional[int] = None
        self._regime_change_count = 0

    @abstractmethod
    def _calculate_base_signal(self, row: pd.Series) -> float:
        """Calculate base signal before position sizing.

        Must be implemented by subclasses.

        Args:
            row: Row of interpreter output with regime information

        Returns:
            Base signal (-1.0 to 1.0, where -1=short, 0=neutral, 1=long)
        """
        pass

    def update(self, interpreter_output: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from regime interpretation.

        Args:
            interpreter_output: Interpreter output with columns:
                - state: int
                - regime_label: str
                - regime_type: str
                - regime_strength: float
                - [optional] regime_return: float
                - [optional] regime_volatility: float

        Returns:
            DataFrame with added columns:
                - base_signal: float (-1.0 to 1.0)
                - signal_strength: float (confidence in signal)
                - position_size: float (sized position)
                - signal_valid: bool (whether to trade)
        """
        # Make a copy to avoid modifying input
        output = interpreter_output.copy()

        # Filter by confidence threshold (preserve existing False values)
        confidence_filter = output["regime_strength"] >= self.config.confidence_threshold
        if "signal_valid" in output.columns:
            # AND with existing signal_valid to preserve False values
            output["signal_valid"] = output["signal_valid"] & confidence_filter
        else:
            output["signal_valid"] = confidence_filter

        # Calculate base signal for each row
        output["base_signal"] = output.apply(self._calculate_base_signal, axis=1)

        # Calculate signal strength (based on confidence)
        output["signal_strength"] = output["regime_strength"] * output["signal_valid"].astype(float)

        # Calculate position size
        output["position_size"] = output.apply(
            lambda row: self._calculate_position_size(row),
            axis=1
        )

        # Detect regime changes for exit signals
        if self.config.enable_regime_change_exits:
            state_diff = output["state"].diff()
            # Only mark as changed if diff is non-zero AND not NaN (first row)
            output["regime_changed"] = (state_diff != 0) & (state_diff.notna())
            output.loc[output["regime_changed"] == True, "position_size"] = 0.0
        else:
            output["regime_changed"] = False

        # Track regime transitions
        self._track_regime_transitions(output)

        return output

    def _calculate_position_size(self, row: pd.Series) -> float:
        """Calculate position size based on strategy and confidence.

        Args:
            row: Row with signal information

        Returns:
            Position size (0 = no trade, positive = long, negative = short)
        """
        if not row["signal_valid"]:
            return 0.0

        min_size, max_size = self.config.position_size_range
        strength = row["signal_strength"]

        # Scale position size by signal strength
        base_position = row["base_signal"]
        sized_position = base_position * (min_size + (max_size - min_size) * strength)

        return np.clip(sized_position, -max_size, max_size)

    def _track_regime_transitions(self, output: pd.DataFrame) -> None:
        """Track regime transitions for monitoring.

        Args:
            output: Signal output DataFrame
        """
        if len(output) > 0:
            current_regime = output.iloc[-1]["state"]
            if self._previous_regime is not None and current_regime != self._previous_regime:
                self._regime_change_count += 1
            self._previous_regime = current_regime

    def plot(self, **kwargs) -> plt.Figure:
        """Generate visualization for signal generator.

        Creates a simple informational plot about signal configuration.

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Title
        ax.text(
            0.5,
            0.95,
            f"Signal Generator: {self.config.strategy_type.title()}",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )

        # Configuration information
        y_pos = 0.85
        config_items = [
            f"Strategy: {self.config.strategy_type}",
            f"Confidence Threshold: {self.config.confidence_threshold:.1%}",
            f"Position Size Range: {self.config.position_size_range}",
            f"Regime Change Exits: {self.config.enable_regime_change_exits}",
            f"Lookback Days: {self.config.lookback_days}",
            f"Total Regime Changes: {self._regime_change_count}",
        ]

        for item in config_items:
            ax.text(0.1, y_pos, item, va="top", fontsize=11, family="monospace")
            y_pos -= 0.12

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        return fig
