"""Pipeline configuration master object.

Composes all component configurations into a single configuration object
for pipeline creation and management.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.config.model import HMMConfig
from hidden_regime.config.interpreter import InterpreterConfiguration
from hidden_regime.config.observation import FinancialObservationConfig
from hidden_regime.config.report import ReportConfiguration
from hidden_regime.config.signal_generation import SignalGenerationConfiguration


@dataclass
class MultiTimeframeConfig:
    """Configuration for multi-timeframe regime detection.

    Attributes:
        enabled: Whether to enable multi-timeframe analysis (default: False)
        timeframes: List of timeframes to analyze (default: ['daily', 'weekly', 'monthly'])
        alignment_threshold: Minimum alignment score to generate signals (default: 0.7)
        use_alignment_for_position_sizing: Whether to scale positions by alignment (default: True)
    """

    enabled: bool = False
    timeframes: List[str] = field(default_factory=lambda: ["daily", "weekly", "monthly"])
    alignment_threshold: float = 0.7
    use_alignment_for_position_sizing: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "timeframes": self.timeframes,
            "alignment_threshold": self.alignment_threshold,
            "use_alignment_for_position_sizing": self.use_alignment_for_position_sizing,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "MultiTimeframeConfig":
        """Create from dictionary."""
        return cls(
            enabled=config_dict.get("enabled", False),
            timeframes=config_dict.get("timeframes", ["daily", "weekly", "monthly"]),
            alignment_threshold=config_dict.get("alignment_threshold", 0.7),
            use_alignment_for_position_sizing=config_dict.get(
                "use_alignment_for_position_sizing", True
            ),
        )


@dataclass
class PipelineConfiguration:
    """Master configuration for the entire pipeline.

    Composes all component configurations into a single object.
    Provides validation of cross-component compatibility.

    Attributes:
        data_config: Configuration for Data component (yfinance, data loading)
        observation_config: Configuration for Observation component (feature generation)
        model_config: Configuration for Model component (HMM parameters)
        interpreter_config: Configuration for Interpreter component (regime labeling)
        signal_generator_config: Optional configuration for Signal Generator component
        report_config: Optional configuration for Report component
        multitimeframe_config: Optional configuration for Multi-Timeframe analysis
    """

    data_config: FinancialDataConfig = field(default_factory=FinancialDataConfig)
    observation_config: FinancialObservationConfig = field(
        default_factory=FinancialObservationConfig
    )
    model_config: HMMConfig = field(default_factory=HMMConfig)
    interpreter_config: InterpreterConfiguration = field(
        default_factory=lambda: InterpreterConfiguration(n_states=3)
    )
    signal_generator_config: Optional[SignalGenerationConfiguration] = None
    report_config: Optional[ReportConfiguration] = None
    multitimeframe_config: Optional[MultiTimeframeConfig] = None

    def __post_init__(self) -> None:
        """Validate cross-component compatibility."""
        self.validate()

    def validate(self) -> None:
        """Validate that component configurations are compatible.

        Raises:
            ValueError: If configuration is invalid
        """
        # Check that interpreter n_states matches model n_states
        if self.interpreter_config.n_states != self.model_config.n_states:
            raise ValueError(
                f"Interpreter n_states ({self.interpreter_config.n_states}) "
                f"must match Model n_states ({self.model_config.n_states})"
            )

    def to_dict(self) -> dict:
        """Convert entire configuration to dictionary.

        Returns:
            Nested dictionary with all component configurations
        """
        return {
            "data_config": self.data_config.to_dict(),
            "observation_config": self.observation_config.to_dict(),
            "model_config": self.model_config.to_dict(),
            "interpreter_config": self.interpreter_config.to_dict(),
            "signal_generator_config": (
                self.signal_generator_config.to_dict()
                if self.signal_generator_config
                else None
            ),
            "report_config": (
                self.report_config.to_dict() if self.report_config else None
            ),
            "multitimeframe_config": (
                self.multitimeframe_config.to_dict()
                if self.multitimeframe_config
                else None
            ),
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PipelineConfiguration":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            New PipelineConfiguration instance
        """
        # Convert nested dicts back to config objects
        return cls(
            data_config=FinancialDataConfig.from_dict(config_dict.get("data_config", {})),
            observation_config=FinancialObservationConfig.from_dict(
                config_dict.get("observation_config", {})
            ),
            model_config=HMMConfig.from_dict(config_dict.get("model_config", {})),
            interpreter_config=InterpreterConfiguration.from_dict(
                config_dict.get("interpreter_config", {})
            ),
            signal_generator_config=(
                SignalGenerationConfiguration.from_dict(
                    config_dict.get("signal_generator_config", {})
                )
                if config_dict.get("signal_generator_config")
                else None
            ),
            report_config=(
                ReportConfiguration.from_dict(config_dict.get("report_config", {}))
                if config_dict.get("report_config")
                else None
            ),
            multitimeframe_config=(
                MultiTimeframeConfig.from_dict(
                    config_dict.get("multitimeframe_config", {})
                )
                if config_dict.get("multitimeframe_config")
                else None
            ),
        )

    def get_summary(self) -> str:
        """Get human-readable summary of configuration.

        Returns:
            Formatted string with all configuration settings
        """
        lines = ["Pipeline Configuration Summary", "=" * 40]

        lines.append("\nData Loading:")
        lines.append(f"  Ticker: {self.data_config.ticker}")
        lines.append(
            f"  Date Range: {self.data_config.start_date} to {self.data_config.end_date}"
        )

        lines.append("\nObservation Generation:")
        lines.append(f"  Generators: {self.observation_config.generators}")

        lines.append("\nModel (HMM):")
        lines.append(f"  States: {self.model_config.n_states}")
        lines.append(f"  Max Iterations: {self.model_config.max_iterations}")
        lines.append(f"  Update Strategy: {self.model_config.update_strategy}")

        lines.append("\nInterpreter:")
        lines.append(f"  Method: {self.interpreter_config.interpretation_method}")

        if self.multitimeframe_config and self.multitimeframe_config.enabled:
            lines.append("\nMulti-Timeframe Analysis:")
            lines.append(f"  Enabled: {self.multitimeframe_config.enabled}")
            lines.append(f"  Timeframes: {', '.join(self.multitimeframe_config.timeframes)}")
            lines.append(f"  Alignment Threshold: {self.multitimeframe_config.alignment_threshold:.0%}")
            lines.append(
                f"  Position Sizing by Alignment: {self.multitimeframe_config.use_alignment_for_position_sizing}"
            )

        if self.signal_generator_config:
            lines.append("\nSignal Generator:")
            lines.append(f"  Strategy: {self.signal_generator_config.strategy_type}")
            lines.append(
                f"  Confidence Threshold: {self.signal_generator_config.confidence_threshold:.0%}"
            )

        if self.report_config:
            lines.append("\nReport:")
            lines.append(f"  Format: {self.report_config.output_format}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PipelineConfiguration("
            f"data={self.data_config.ticker}, "
            f"states={self.model_config.n_states}, "
            f"signals={self.signal_generator_config is not None})"
        )
