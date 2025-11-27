"""Pipeline component wrapper for the new SignalGenerator.

Adapts the new SignalGenerator (which returns List[TradingSignal]) to the
SignalGeneratorComponent interface (which returns DataFrame) for pipeline compatibility.

This enables the pipeline architecture to work seamlessly with the new strategy-based
signal generation while maintaining backward compatibility.
"""

from typing import List

import pandas as pd

from hidden_regime.pipeline.interfaces import SignalGeneratorComponent
from hidden_regime.signals.types import TradingSignal
from hidden_regime.strategies import Strategy

from .signal_generator import SignalGenerator


class StrategyBasedSignalGeneratorComponent(SignalGeneratorComponent):
    """Pipeline component wrapper for strategy-based signal generation.

    This component wraps the new SignalGenerator and provides:
    - SignalGeneratorComponent interface compliance
    - DataFrame input/output for pipeline compatibility
    - List[TradingSignal] internal representation
    - Backward compatibility with existing pipeline code

    Example usage:
        from hidden_regime.strategies import RegimeFollowingStrategy
        from hidden_regime.signal_generation import StrategyBasedSignalGeneratorComponent

        strategy = RegimeFollowingStrategy()
        signal_component = StrategyBasedSignalGeneratorComponent(strategy)

        # Pipeline will call update() which returns DataFrame
        signals_df = signal_component.update(interpreter_output)
    """

    def __init__(self, strategy: Strategy):
        """Initialize the signal generator component.

        Args:
            strategy: Strategy object that generates signals from regimes
        """
        self._generator = SignalGenerator(strategy)
        self.name = f"SignalGenerator({strategy.__class__.__name__})"

    def update(self, interpreter_output: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from interpreter output.

        This method implements the SignalGeneratorComponent interface.
        It returns a DataFrame for pipeline compatibility.

        Args:
            interpreter_output: DataFrame with 'regime_label' column containing
                RegimeLabel objects from the Interpreter component.

        Returns:
            DataFrame with trading signals and supporting columns:
            - direction: 'long', 'short', or 'neutral'
            - position_size: Float 0.0 to 1.0 (or higher for leveraged)
            - confidence: Float 0.0 to 1.0
            - strategy_name: Name of the strategy that generated the signal
            - risk_management_applied: Dict of constraints applied
            - valid: Boolean indicating if signal is actionable
            - regime_name: Name of the regime (from RegimeLabel)
            - regime_type: Type of regime (BULLISH, BEARISH, etc.)

        Raises:
            ValueError: If 'regime_label' column is missing or contains None values
        """
        if interpreter_output.empty:
            return pd.DataFrame()

        # Generate signals using the internal SignalGenerator
        signals: List[TradingSignal] = self._generator.update(interpreter_output)

        # Convert to DataFrame for pipeline compatibility
        signals_df = self._generator.to_dataframe(signals)

        return signals_df

    def get_signals_as_objects(
        self, interpreter_output: pd.DataFrame
    ) -> List[TradingSignal]:
        """
        Generate trading signals and return as TradingSignal objects.

        This method provides access to the strongly-typed signal objects
        for code that wants to work with TradingSignal directly.

        Args:
            interpreter_output: DataFrame with 'regime_label' column

        Returns:
            List of TradingSignal objects with full audit trail and metadata
        """
        return self._generator.update(interpreter_output)

    def plot(self, **kwargs) -> "matplotlib.pyplot.Figure":
        """
        Generate visualization for trading signals.

        Creates a simple summary visualization of signal statistics.

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # Add text with signal generator information
        text_content = (
            f"Signal Generator Component\n"
            f"Strategy: {self._generator.strategy.__class__.__name__}\n"
            f"Component: {self.name}\n\n"
            f"Use pipeline.update() to generate signals\n"
            f"then access results via pipeline.component_outputs['signals']"
        )

        ax.text(
            0.5,
            0.5,
            text_content,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax.axis("off")
        fig.suptitle("Trading Signal Generation", fontsize=14, fontweight="bold")

        return fig

    def __repr__(self) -> str:
        return f"StrategyBasedSignalGeneratorComponent({self._generator.strategy.name})"
