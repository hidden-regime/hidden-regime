"""
Core Pipeline implementation for hidden-regime.

Provides the main Pipeline class that orchestrates the Data → Observation → Model → Analysis → Report flow.
The Pipeline serves as the primary user interface and coordinates all components through their standardized interfaces.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Type

import matplotlib.pyplot as plt
import pandas as pd

from .interfaces import (
    DataComponent,
    InterpreterComponent,
    ModelComponent,
    ObservationComponent,
    ReportComponent,
    SignalGeneratorComponent,
)


class Pipeline:
    """
    Main Pipeline orchestrator for hidden-regime analysis.

    Implements the ROADMAP architecture:
    Data → Observations → Model → Interpreter → Signal Generator → Report

    The Pipeline orchestrates this flow and serves as the primary user interface.
    All components follow standardized interfaces ensuring consistent behavior
    and extensibility.

    Component Responsibilities:
    - Data: Load and validate price data
    - Observations: Generate features from raw data
    - Model: Pure math (HMM algorithms, NO domain knowledge)
    - Interpreter: Add domain knowledge (regime labels, characteristics)
    - Signal Generator: Trading logic (position sizing, entry/exit)
    - Report: Metrics, visualizations, summaries

    Example usage (new architecture):
        pipeline = Pipeline(
            data=data_component,
            observation=observation_component,
            model=model_component,
            interpreter=interpreter_component,
            signal_generator=signal_generator_component,
            report=report_component
        )

    Backward compatibility (old architecture):
        pipeline = Pipeline(
            data=data_component,
            observation=observation_component,
            model=model_component,
            analysis=analysis_component,  # Still supported
            report=report_component
        )
    """

    def __init__(
        self,
        data: DataComponent,
        observation: ObservationComponent,
        model: ModelComponent,
        interpreter: InterpreterComponent,
        signal_generator: Optional[SignalGeneratorComponent] = None,
        report: Optional[ReportComponent] = None,
    ):
        """
        Initialize pipeline with components.

        Args:
            data: Data loading and management component
            observation: Observation generation component
            model: Model training and prediction component
            interpreter: Interpreter component for regime interpretation
            signal_generator: Optional signal generator component
            report: Optional report generation component
        """
        self.data = data
        self.observation = observation
        self.model = model
        self.interpreter = interpreter
        self.signal_generator = signal_generator
        self.report = report

        # Pipeline state tracking
        self.last_update = None
        self.update_count = 0
        self.component_outputs = {}

        # Configure logging
        self.logger = logging.getLogger(f"Pipeline-{id(self)}")

    def update(self, current_date: Optional[str] = None) -> str:
        """
        Execute complete pipeline flow:
        Data → Observations → Model → Interpreter → Signal Generator → Report

        Args:
            current_date: Optional date for data updates (YYYY-MM-DD format)

        Returns:
            Report output (typically markdown string) or interpreter output if no report component
        """
        try:
            self.logger.info(f"Starting pipeline update {self.update_count + 1}")
            update_start = datetime.now()

            # Step 1: Update data
            self.logger.debug("Step 1: Updating data component")
            data_output = self.data.update(current_date=current_date)
            self.component_outputs["data"] = data_output

            if len(data_output) == 0:
                raise ValueError("No data available from data component")

            # Step 2: Generate observations
            self.logger.debug("Step 2: Generating observations")
            observation_output = self.observation.update(data_output)
            self.component_outputs["observations"] = observation_output

            if len(observation_output) == 0:
                raise ValueError("No observations generated from data")

            # Step 3: Model processing (fit if first time, then predict)
            self.logger.debug("Step 3: Processing model")
            model_output = self.model.update(observation_output)
            self.component_outputs["model"] = model_output

            if len(model_output) == 0:
                raise ValueError("No predictions generated from model")

            # Step 4: Interpretation (add domain knowledge to model output)
            self.logger.debug("Step 4: Interpreting regimes")

            # Pass raw data for performance calculations (win_rate, max_drawdown, etc.)
            interpreter_output = self.interpreter.update(
                model_output, raw_data=data_output
            )
            self.component_outputs["interpreter"] = interpreter_output

            # Step 5: Signal generation (if signal generator provided)
            signal_output = None
            if self.signal_generator is not None:
                self.logger.debug("Step 5: Generating trading signals")
                signal_output = self.signal_generator.update(interpreter_output)
                self.component_outputs["signals"] = signal_output

            # Step 6: Generate report (if report component provided)
            if self.report is not None:
                self.logger.debug("Step 6: Generating report")
                report_output = self.report.update(
                    data=data_output,
                    observations=observation_output,
                    model_output=model_output,
                    interpreter_output=interpreter_output,
                    signals=signal_output,
                    raw_data=data_output,  # For backward compatibility
                )
                self.component_outputs["report"] = report_output
                result = report_output
            else:
                # Return interpreter output if no report component
                result = self._format_interpreter_output(interpreter_output)

            # Update pipeline state
            self.last_update = update_start
            self.update_count += 1

            update_duration = (datetime.now() - update_start).total_seconds()
            self.logger.info(f"Pipeline update completed in {update_duration:.2f}s")

            return result

        except Exception as e:
            self.logger.error(f"Pipeline update failed: {str(e)}")
            raise

    def _format_interpreter_output(self, interpreter_output: pd.DataFrame) -> str:
        """Format interpreter output as simple markdown when no report component."""
        if interpreter_output.empty:
            return "# Regime Analysis Results\n\nNo results available."

        # Get current state info (last row)
        current = interpreter_output.iloc[-1]

        report_lines = [
            "# Regime Analysis Results",
            "",
        ]

        # Add regime label and type
        if "regime_label" in current:
            regime_label = current.get("regime_label", "Unknown")
            report_lines.append(f"**Current Regime**: {regime_label}")

        if "regime_type" in current:
            regime_type = current.get("regime_type", "unknown")
            report_lines.append(f"**Regime Type**: {regime_type.title()}")

        # Add confidence
        if "regime_strength" in current:
            strength = current.get("regime_strength", 0.5)
            # Handle both string and numeric types
            if isinstance(strength, str):
                strength = float(strength)
            report_lines.append(f"**Regime Confidence**: {strength:.1%}")

        # Add regime characteristics if available
        # Support both old (regime_return) and new (expected_return) column names
        if "expected_return" in current.index:
            expected_return = current.get("expected_return")
            if expected_return is not None:
                # Handle both string and numeric types
                if isinstance(expected_return, str):
                    expected_return = float(expected_return)
                report_lines.append(
                    f"**Expected Annual Return**: {expected_return:.2%}"
                )
        elif "regime_return" in current.index:
            expected_return = current.get("regime_return")
            if expected_return is not None:
                # Handle both string and numeric types
                if isinstance(expected_return, str):
                    expected_return = float(expected_return)
                report_lines.append(
                    f"**Expected Annual Return**: {expected_return:.2%}"
                )

        # Support both old (regime_volatility) and new (expected_volatility) column names
        if "expected_volatility" in current.index:
            volatility = current.get("expected_volatility")
            if volatility is not None:
                # Handle both string and numeric types
                if isinstance(volatility, str):
                    volatility = float(volatility)
                report_lines.append(f"**Expected Annual Volatility**: {volatility:.2%}")
        elif "regime_volatility" in current.index:
            volatility = current.get("regime_volatility")
            if volatility is not None:
                # Handle both string and numeric types
                if isinstance(volatility, str):
                    volatility = float(volatility)
                report_lines.append(f"**Expected Annual Volatility**: {volatility:.2%}")

        # Add current date if available
        if hasattr(current, "name"):
            report_lines.append(f"\n**As of**: {current.name}")

        return "\n".join(filter(None, report_lines))

    def get_component_output(self, component_name: str) -> Optional[Any]:
        """
        Get the last output from a specific component.

        Args:
            component_name: Name of component ('data', 'observations', 'model', 'analysis', 'report')

        Returns:
            Last output from the specified component, or None if not available
        """
        return self.component_outputs.get(component_name)

    @property
    def data_output(self) -> pd.DataFrame:
        """
        Easily access the data output as a DataFrame
        """
        return self.get_component_output("data")

    @property
    def model_output(self) -> pd.DataFrame:
        """
        Easily access the model output as a DataFrame
        """
        return self.get_component_output("model")

    @property
    def interpreter_output(self) -> pd.DataFrame:
        """
        Easily access the interpreter output as a DataFrame.
        This is the regime-labeled output with domain knowledge added.
        """
        return self.get_component_output("interpreter")

    @property
    def signals_output(self) -> Optional[pd.DataFrame]:
        """
        Easily access the signal generator output as a DataFrame.
        Returns None if no signal generator component is used.
        """
        return self.get_component_output("signals")

    @property
    def observations_output(self) -> pd.DataFrame:
        """
        Easily access the observation output as a DataFrame
        """
        return self.get_component_output("observations")

    def plot(self, components: Optional[list] = None, **kwargs) -> plt.Figure:
        """
        Generate visualization showing outputs from all or specified components.

        Args:
            components: List of component names to plot. If None, plot all components.
            **kwargs: Additional plotting arguments passed to component plot methods

        Returns:
            matplotlib Figure with subplots for each component
        """
        if components is None:
            components = ["data", "observations", "model", "analysis"]

        # Filter components to only those with plot methods and outputs
        available_components = []
        for component_name in components:
            component = getattr(self, component_name, None)
            if component is not None and hasattr(component, "plot"):
                available_components.append(component_name)

        n_components = len(available_components)
        if n_components == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                "No components available to plot",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return fig

        # Create subplots
        fig, axes = plt.subplots(n_components, 1, figsize=(14, 4 * n_components))

        # Handle single subplot case
        if n_components == 1:
            axes = [axes]

        plot_idx = 0
        for component_name in available_components:
            ax = axes[plot_idx]
            component = getattr(self, component_name, None)

            if component_name == "data" and hasattr(self.data, "plot"):
                # Pass regime data for overlay if analysis is available
                plot_kwargs = kwargs.copy()
                if "analysis" in self.component_outputs:
                    plot_kwargs["regime_data"] = self.component_outputs["analysis"]
                self.data.plot(ax=ax, **plot_kwargs)

            elif component_name == "observations" and hasattr(self.observation, "plot"):
                self.observation.plot(ax=ax, **kwargs)

            elif component_name == "model" and hasattr(self.model, "plot"):
                self.model.plot(ax=ax, **kwargs)

            # The title is set by the component's compact plot method
            plot_idx += 1

        plt.tight_layout()
        return fig

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about pipeline state and performance.

        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            "update_count": self.update_count,
            "last_update": self.last_update,
            "components": {
                "data": type(self.data).__name__,
                "observation": type(self.observation).__name__,
                "model": type(self.model).__name__,
                "interpreter": type(self.interpreter).__name__,
                "signal_generator": (
                    type(self.signal_generator).__name__
                    if self.signal_generator
                    else None
                ),
                "report": type(self.report).__name__ if self.report else None,
            },
        }

        # Add data shape if available
        if "data" in self.component_outputs:
            data_output = self.component_outputs["data"]
            if hasattr(data_output, "shape"):
                stats["data_shape"] = data_output.shape

        # Add model info if available
        if "model" in self.component_outputs:
            model_output = self.component_outputs["model"]
            if hasattr(model_output, "shape"):
                stats["model_output_shape"] = model_output.shape

        # Add interpreter info if available
        if "interpreter" in self.component_outputs:
            interpreter_output = self.component_outputs["interpreter"]
            if hasattr(interpreter_output, "shape"):
                stats["interpreter_output_shape"] = interpreter_output.shape

        # Add signals info if available
        if "signals" in self.component_outputs:
            signals_output = self.component_outputs["signals"]
            if hasattr(signals_output, "shape"):
                stats["signals_output_shape"] = signals_output.shape

        return stats

    def __getstate__(self) -> Dict[str, Any]:
        """Support for pickle serialization."""
        return self.__dict__.copy()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Support for pickle deserialization."""
        self.__dict__.update(state)
        # Recreate logger after unpickling
        self.logger = logging.getLogger(f"Pipeline-{id(self)}")

    def __repr__(self) -> str:
        """String representation of pipeline."""
        components = []
        components.append(f"Data: {type(self.data).__name__}")
        components.append(f"Observation: {type(self.observation).__name__}")
        components.append(f"Model: {type(self.model).__name__}")
        components.append(f"Interpreter: {type(self.interpreter).__name__}")
        if self.signal_generator:
            components.append(
                f"SignalGenerator: {type(self.signal_generator).__name__}"
            )
        if self.report:
            components.append(f"Report: {type(self.report).__name__}")

        return f"Pipeline({', '.join(components)}, updates={self.update_count})"
