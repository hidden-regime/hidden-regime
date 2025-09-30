"""
Case study configuration for temporal market regime analysis.

Provides configuration for running comprehensive case studies that analyze
regime evolution over time periods with proper temporal isolation.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from ..utils.exceptions import ConfigurationError


@dataclass
class CaseStudyConfig:
    """
    Configuration for market regime case studies.

    Defines parameters for temporal analysis including training periods,
    evaluation periods, and output requirements for comprehensive studies.
    """

    # Core case study parameters (from case_study.md specification)
    ticker: str = "NVDA"
    start_date: str = "2024-04-02"  # First evaluation date
    end_date: str = "2025-09-26"  # Last evaluation date
    n_training: int = 252  # Training days before start_date
    frequency: Literal["business_days", "daily", "hourly"] = "business_days"
    n_states: int = 4

    # Analysis configuration
    include_technical_indicators: bool = True
    indicators_to_compare: List[str] = (
        None  # Will default to ['rsi', 'macd', 'bollinger_bands', 'moving_average']
    )
    calculate_buy_hold_comparison: bool = True
    include_risk_metrics: bool = True

    # Regime labeling configuration (advanced users only)
    force_regime_labels: Optional[List[str]] = (
        None  # Override regime labels (e.g., ['Custom_Low', 'Custom_Mid', 'Custom_High'])
    )
    acknowledge_override: bool = False  # Must be True when using force_regime_labels

    # Trading simulation configuration
    enable_simulation: bool = True  # Enable trading simulation engine
    simulation_initial_capital: Optional[float] = (
        None  # Starting capital (auto-calculated if None)
    )
    simulation_transaction_cost: float = 0.0  # Transaction cost per trade
    simulation_stop_loss_pct: float = 0.05  # Default 5% stop-loss
    simulation_max_position_pct: float = 0.1  # Maximum 10% position size
    simulation_include_technical_indicators: bool = (
        True  # Include TA indicators in simulation
    )

    # Visualization configuration
    create_animations: bool = True
    save_individual_frames: bool = False
    animation_fps: int = 2
    plot_style: Literal["professional", "academic", "presentation"] = "professional"
    color_scheme: Literal[
        "classic",
        "professional",
        "academic",
        "pastel",
        "colorblind_safe",
        "high_contrast",
        "blue_red",
        "viridis",
        "okabe_ito",
    ] = "colorblind_safe"

    # Output configuration
    output_directory: Optional[str] = None  # Will default to ./output/case_studies
    save_intermediate_results: bool = True
    generate_comprehensive_report: bool = True

    # Model configuration overrides
    model_config_overrides: Optional[Dict[str, Any]] = None
    analysis_config_overrides: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Post-initialization validation and default setting."""

        # Set default indicators if not provided
        if self.indicators_to_compare is None:
            self.indicators_to_compare = [
                "rsi",
                "macd",
                "bollinger_bands",
                "moving_average",
            ]

        # Set default output directory if not provided
        if self.output_directory is None:
            self.output_directory = (
                f"./output/case_studies/{self.ticker}_{self.start_date}_{self.end_date}"
            )

    def validate(self) -> None:
        """Validate case study configuration parameters."""
        super().validate()

        # Validate ticker
        if not self.ticker or len(self.ticker.strip()) == 0:
            raise ConfigurationError("ticker cannot be empty")

        # Validate dates
        try:
            start_dt = pd.to_datetime(self.start_date)
            end_dt = pd.to_datetime(self.end_date)
        except ValueError as e:
            raise ConfigurationError(f"Invalid date format: {e}")

        if start_dt >= end_dt:
            raise ConfigurationError(
                f"start_date {self.start_date} must be before end_date {self.end_date}"
            )

        # Validate training period
        if self.n_training <= 0:
            raise ConfigurationError(
                f"n_training must be positive, got {self.n_training}"
            )

        if self.n_training < 30:
            raise ConfigurationError(
                f"n_training should be at least 30 days for meaningful analysis, got {self.n_training}"
            )

        # Validate n_states
        if self.n_states < 2 or self.n_states > 10:
            raise ConfigurationError(
                f"n_states must be between 2 and 10, got {self.n_states}"
            )

        # Validate frequency
        valid_frequencies = ["business_days", "daily", "hourly"]
        if self.frequency not in valid_frequencies:
            raise ConfigurationError(
                f"frequency must be one of {valid_frequencies}, got {self.frequency}"
            )

        # Validate indicators
        valid_indicators = [
            "rsi",
            "macd",
            "bollinger_bands",
            "moving_average",
            "stochastic",
            "williams_r",
            "atr",
        ]
        for indicator in self.indicators_to_compare:
            if indicator not in valid_indicators:
                raise ConfigurationError(
                    f"Unknown indicator: {indicator}. Valid indicators: {valid_indicators}"
                )

        # Validate animation parameters
        if self.animation_fps <= 0 or self.animation_fps > 30:
            raise ConfigurationError(
                f"animation_fps must be between 1 and 30, got {self.animation_fps}"
            )

        # Validate regime labeling override
        if self.force_regime_labels is not None:
            # Must acknowledge override
            if not self.acknowledge_override:
                raise ConfigurationError(
                    "force_regime_labels requires acknowledge_override=True. "
                    "This acknowledges that you understand you are overriding data-driven regime detection."
                )

            # Must match n_states
            if len(self.force_regime_labels) != self.n_states:
                raise ConfigurationError(
                    f"force_regime_labels length ({len(self.force_regime_labels)}) must match n_states ({self.n_states})"
                )

            # Must be unique
            if len(set(self.force_regime_labels)) != len(self.force_regime_labels):
                raise ConfigurationError("force_regime_labels must be unique")

            # Warn about risks
            import warnings

            warnings.warn(
                "WARNING: You are overriding data-driven regime detection with custom labels. "
                "This may result in misleading interpretations if labels don't match actual market behavior. "
                "Use only when you understand the implications.",
                UserWarning,
            )

    def get_training_date_range(self) -> tuple[str, str]:
        """
        Calculate the training date range based on start_date and n_training.

        Returns:
            Tuple of (training_start_date, training_end_date) as strings
        """
        start_dt = pd.to_datetime(self.start_date)

        if self.frequency == "business_days":
            training_start = start_dt - pd.Timedelta(
                days=int(self.n_training * 1.4)
            )  # Buffer for weekends
            training_start = pd.bdate_range(end=start_dt, periods=self.n_training + 1)[
                0
            ]
        else:
            training_start = start_dt - pd.Timedelta(days=self.n_training)

        # Training ends the day before evaluation starts
        training_end = start_dt - pd.Timedelta(days=1)

        return training_start.strftime("%Y-%m-%d"), training_end.strftime("%Y-%m-%d")

    def get_evaluation_dates(self) -> List[str]:
        """
        Get list of evaluation dates for stepping through the case study.

        Returns:
            List of date strings for evaluation period
        """
        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)

        if self.frequency == "business_days":
            date_range = pd.bdate_range(start=start_dt, end=end_dt)
        elif self.frequency == "daily":
            date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")
        elif self.frequency == "hourly":
            date_range = pd.date_range(start=start_dt, end=end_dt, freq="H")
        else:
            raise ConfigurationError(f"Unsupported frequency: {self.frequency}")

        return [dt.strftime("%Y-%m-%d") for dt in date_range]

    def get_total_analysis_period(self) -> int:
        """
        Get total number of periods in the analysis.

        Returns:
            Number of time periods in evaluation range
        """
        return len(self.get_evaluation_dates())

    def create_output_structure(self) -> Dict[str, str]:
        """
        Create output directory structure for case study results.

        Returns:
            Dictionary mapping output types to directory paths
        """
        import os

        base_dir = self.output_directory
        structure = {
            "base": base_dir,
            "reports": os.path.join(base_dir, "reports"),
            "plots": os.path.join(base_dir, "plots"),
            "animations": os.path.join(base_dir, "animations"),
            "data": os.path.join(base_dir, "data"),
            "frames": (
                os.path.join(base_dir, "frames")
                if self.save_individual_frames
                else None
            ),
        }

        # Create directories
        for dir_path in structure.values():
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

        return structure

    def get_model_config_with_overrides(self) -> Dict[str, Any]:
        """
        Get model configuration with case study specific overrides.

        Returns:
            Dictionary with model configuration parameters
        """
        base_config = {
            "n_states": self.n_states,
            "initialization_method": "kmeans",
            "random_seed": 42,  # For reproducible results
            "tolerance": 1e-6,
            "max_iterations": 1000,
        }

        if self.model_config_overrides:
            base_config.update(self.model_config_overrides)

        return base_config

    def get_analysis_config_with_overrides(self) -> Dict[str, Any]:
        """
        Get analysis configuration with case study specific overrides.

        Returns:
            Dictionary with analysis configuration parameters
        """
        base_config = {
            "n_states": self.n_states,
            "calculate_regime_statistics": True,
            "include_duration_analysis": True,
            "include_return_analysis": True,
            "include_volatility_analysis": True,
            "include_trading_signals": True,
            "indicator_comparisons": (
                self.indicators_to_compare
                if self.include_technical_indicators
                else None
            ),
            "include_indicator_performance": self.include_technical_indicators,
        }

        if self.analysis_config_overrides:
            base_config.update(self.analysis_config_overrides)

        return base_config

    def get_summary_info(self) -> Dict[str, Any]:
        """
        Get summary information about the case study configuration.

        Returns:
            Dictionary with case study summary information
        """
        training_start, training_end = self.get_training_date_range()
        evaluation_dates = self.get_evaluation_dates()

        return {
            "ticker": self.ticker,
            "analysis_period": {
                "start": self.start_date,
                "end": self.end_date,
                "total_periods": len(evaluation_dates),
            },
            "training_period": {
                "start": training_start,
                "end": training_end,
                "n_training_days": self.n_training,
            },
            "configuration": {
                "n_states": self.n_states,
                "frequency": self.frequency,
                "indicators": self.indicators_to_compare,
                "include_animations": self.create_animations,
                "include_buy_hold": self.calculate_buy_hold_comparison,
            },
            "output_directory": self.output_directory,
        }

    @classmethod
    def create_quick_study(
        cls, ticker: str = "SPY", days_back: int = 90, n_states: int = 3
    ) -> "CaseStudyConfig":
        """
        Create configuration for a quick case study (last N days).

        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to analyze
            n_states: Number of regime states

        Returns:
            CaseStudyConfig for quick analysis
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        return cls(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            n_training=max(
                60, days_back // 2
            ),  # Use half the period for training, minimum 60 days
            n_states=n_states,
            create_animations=False,  # Quick studies skip animations
            include_technical_indicators=False,  # Quick studies skip indicators
        )

    @classmethod
    def create_comprehensive_study(
        cls, ticker: str, start_date: str, end_date: str, n_states: int = 4
    ) -> "CaseStudyConfig":
        """
        Create configuration for comprehensive case study with all features.

        Args:
            ticker: Stock ticker symbol
            start_date: Analysis start date (YYYY-MM-DD)
            end_date: Analysis end date (YYYY-MM-DD)
            n_states: Number of regime states

        Returns:
            CaseStudyConfig for comprehensive analysis
        """
        return cls(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            n_training=252,  # One year of training
            n_states=n_states,
            include_technical_indicators=True,
            create_animations=True,
            calculate_buy_hold_comparison=True,
            include_risk_metrics=True,
            generate_comprehensive_report=True,
            save_intermediate_results=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CaseStudyConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "CaseStudyConfig":
        """Create configuration from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, filepath: str) -> "CaseStudyConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            json_str = f.read()
        return cls.from_json(json_str)

    def copy(self, **kwargs) -> "CaseStudyConfig":
        """Create a copy of configuration with optional parameter updates."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.__class__.from_dict(config_dict)
