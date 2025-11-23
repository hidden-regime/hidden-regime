"""
Base interfaces for all pipeline components.

Defines the common interface that all pipeline components must implement,
ensuring consistency across Data, Observation, Model, Analysis, and Report components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


class PipelineComponent(ABC):
    """
    Base interface for all pipeline components.

    All components (Data, Observation, Model, Analysis, Report) must implement this interface
    to ensure they can be used interchangeably in the pipeline framework.
    """

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """
        Process input and return output.

        This is the core method that processes data through the component.
        Each component type will have different input/output signatures.
        """
        pass

    @abstractmethod
    def plot(self, **kwargs) -> plt.Figure:
        """
        Generate visualization for this component.

        Returns:
            matplotlib Figure object with component-specific visualization
        """
        pass

    def __getstate__(self) -> Dict[str, Any]:
        """Support for pickle serialization."""
        return self.__dict__.copy()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Support for pickle deserialization."""
        self.__dict__.update(state)


class DataComponent(PipelineComponent):
    """Interface for data loading and management components."""

    @abstractmethod
    def get_all_data(self) -> pd.DataFrame:
        """
        Get complete dataset.

        Returns:
            DataFrame with complete data including timestamps
        """
        pass

    @abstractmethod
    def update(self, current_date: Optional[str] = None) -> pd.DataFrame:
        """
        Update data, optionally fetching new data up to current_date.

        Args:
            current_date: Optional date to update data up to

        Returns:
            Updated DataFrame with any new data
        """
        pass


class ObservationComponent(PipelineComponent):
    """Interface for observation generation components."""

    @abstractmethod
    def update(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate observations from raw data.

        Args:
            data: Raw data DataFrame

        Returns:
            DataFrame with generated observations
        """
        pass


class ModelComponent(PipelineComponent):
    """Interface for model training and prediction components."""

    @abstractmethod
    def fit(self, observations: pd.DataFrame) -> None:
        """
        Train the model on observations.

        Args:
            observations: Training data
        """
        pass

    @abstractmethod
    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for observations.

        Args:
            observations: Input observations

        Returns:
            DataFrame with predictions (e.g., predicted_state, confidence)
        """
        pass

    @abstractmethod
    def update(self, observations: pd.DataFrame) -> pd.DataFrame:
        """
        Process observations (fit if first time, then predict).

        Args:
            observations: Input observations

        Returns:
            DataFrame with predictions
        """
        pass


class InterpreterComponent(PipelineComponent):
    """Interface for regime interpretation components.

    The Interpreter component adds domain knowledge to model outputs.
    It maps model state indices to semantic regime labels and characteristics.

    Key Principle: ALL financial domain knowledge belongs here.
    - State indices (0,1,2) → regime labels (Bear, Bull, Sideways)
    - Regime characteristics (returns, volatility, etc.)
    - Consistent colors and naming for visualization

    The Model provides ONLY pure math outputs. The Interpreter provides ONLY domain interpretation.
    """

    @abstractmethod
    def update(self, model_output: pd.DataFrame) -> pd.DataFrame:
        """
        Interpret model output and add regime labels/characteristics.

        Args:
            model_output: Raw model predictions (state indices, confidence, etc.)

        Returns:
            DataFrame with regime labels, types, colors, and characteristics added
        """
        pass


# AnalysisComponent has been REMOVED in version 2.0.0
# Use InterpreterComponent instead


class SignalGeneratorComponent(PipelineComponent):
    """Interface for signal generation components.

    The Signal Generator component creates trading signals from Interpreter outputs.
    It implements trading logic and position sizing strategies.

    Key Principle: ALL trading logic belongs here.
    - Position signals (long/short/neutral)
    - Strategy implementation
    - Position sizing
    - Entry/exit rules

    The Interpreter provides regime interpretation. The Signal Generator provides trading logic.
    These must remain separate.
    """

    @abstractmethod
    def update(self, interpreter_output: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from regime interpretation.

        Args:
            interpreter_output: Interpreter output with regime labels and characteristics

        Returns:
            DataFrame with trading signals, position sizes, and strategy information added
        """
        pass


class ReportComponent(PipelineComponent):
    """Interface for report generation components."""

    @abstractmethod
    def update(self, **kwargs) -> str:
        """
        Generate report from pipeline results.

        Returns:
            Generated report (e.g., markdown string)
        """
        pass

    def plot(self, **kwargs) -> plt.Figure:
        """Reports may not have meaningful plots."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "Report Component\n(See update() for content)",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return fig
