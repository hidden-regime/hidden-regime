"""
Analysis configuration classes for pipeline analysis components.

Provides configuration for interpreting model outputs, applying domain knowledge,
and generating meaningful insights from regime predictions including financial
analysis with indicators and performance metrics.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import numpy as np

from ..utils.exceptions import ConfigurationError
from .base import BaseConfig


@dataclass(frozen=True)
class AnalysisConfig(BaseConfig):
    """
    Base configuration for analysis components.
    """

    n_states: int = 3
    observed_signal: str = "log_return"

    def validate(self) -> None:
        """Validate analysis configuration."""
        super().validate()

        if self.n_states < 2:
            raise ConfigurationError(
                f"n_states must be at least 2, got {self.n_states}"
            )

        if not self.observed_signal:
            raise ConfigurationError("observed_signal cannot be empty")

    def create_component(self) -> Any:
        """Create analysis component - to be implemented by specific configs."""
        raise NotImplementedError("Subclasses must implement create_component")


@dataclass(frozen=True)
class FinancialAnalysisConfig(AnalysisConfig):
    """
    Configuration for financial analysis components.

    Provides financial interpretation of regime states including comparisons
    with technical indicators, performance metrics, and trading insights.
    """

    # Financial interpretation
    regime_labels: Optional[List[str]] = None

    # User override parameters (advanced users only)
    force_regime_labels: Optional[List[str]] = None  # Override regime labels
    acknowledge_override: bool = False  # Must be True when using force_regime_labels
    price_column: str = "close"

    # Indicator comparisons
    indicator_comparisons: Optional[List[str]] = field(
        default_factory=lambda: ["rsi", "macd"]
    )
    include_indicator_performance: bool = True

    # Performance analysis
    calculate_regime_statistics: bool = True
    include_duration_analysis: bool = True
    include_return_analysis: bool = True
    include_volatility_analysis: bool = True

    # Trading analysis
    include_trading_signals: bool = True
    position_sizing_method: str = "regime_confidence"
    risk_adjustment: bool = True

    # Regime interpretation parameters
    volatility_window: int = 20
    return_window: int = 20
    minimum_regime_duration: int = 2

    def validate(self) -> None:
        """Validate financial analysis configuration."""
        super().validate()

        # Validate regime labels if provided
        if self.regime_labels is not None:
            if len(self.regime_labels) != self.n_states:
                raise ConfigurationError(
                    f"regime_labels length {len(self.regime_labels)} must match n_states {self.n_states}"
                )

            # Check for unique labels
            if len(set(self.regime_labels)) != len(self.regime_labels):
                raise ConfigurationError("regime_labels must be unique")

        # Validate user override parameters
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

        # Validate indicator names
        if self.indicator_comparisons is not None:
            valid_indicators = {
                "rsi",
                "macd",
                "bollinger_bands",
                "moving_average",
                "stochastic",
                "cci",
                "williams_r",
                "momentum",
                "price_rate_of_change",
                "trix",
                "ultimate_oscillator",
            }

            for indicator in self.indicator_comparisons:
                if indicator not in valid_indicators:
                    raise ConfigurationError(f"Unknown indicator: {indicator}")

        # Validate position sizing method
        valid_methods = [
            "equal_weight",
            "regime_confidence",
            "volatility_adjusted",
            "risk_parity",
        ]
        if self.position_sizing_method not in valid_methods:
            raise ConfigurationError(
                f"position_sizing_method must be one of {valid_methods}"
            )

        # Validate window parameters
        if self.volatility_window < 5:
            raise ConfigurationError(
                f"volatility_window must be at least 5, got {self.volatility_window}"
            )

        if self.return_window < 5:
            raise ConfigurationError(
                f"return_window must be at least 5, got {self.return_window}"
            )

        if self.minimum_regime_duration < 1:
            raise ConfigurationError(
                f"minimum_regime_duration must be at least 1, got {self.minimum_regime_duration}"
            )

        # Validate price column
        if not self.price_column:
            raise ConfigurationError("price_column cannot be empty")

    def create_component(self) -> Any:
        """Create financial analysis component."""
        from ..analysis.financial import FinancialAnalysis

        return FinancialAnalysis(self)

    def get_default_regime_labels(self) -> List[str]:
        """Get default regime labels based on number of states."""
        if self.regime_labels is not None:
            return self.regime_labels

        if self.n_states == 2:
            return ["Bear", "Bull"]
        elif self.n_states == 3:
            return ["Bear", "Sideways", "Bull"]
        elif self.n_states == 4:
            return ["Crisis", "Bear", "Sideways", "Bull"]
        elif self.n_states == 5:
            return ["Crisis", "Bear", "Sideways", "Bull", "Euphoric"]
        else:
            return [f"Regime_{i+1}" for i in range(self.n_states)]

    @classmethod
    def create_minimal_financial(cls) -> "FinancialAnalysisConfig":
        """Create minimal configuration for basic financial analysis."""
        return cls(
            n_states=3,
            observed_signal="log_return",
            indicator_comparisons=None,
            include_indicator_performance=False,
            calculate_regime_statistics=True,
            include_trading_signals=False,
        )

    @classmethod
    def create_comprehensive_financial(cls) -> "FinancialAnalysisConfig":
        """Create comprehensive configuration with full analysis."""
        return cls(
            n_states=3,
            observed_signal="log_return",
            indicator_comparisons=["rsi", "macd", "bollinger_bands", "moving_average"],
            include_indicator_performance=True,
            calculate_regime_statistics=True,
            include_duration_analysis=True,
            include_return_analysis=True,
            include_volatility_analysis=True,
            include_trading_signals=True,
            position_sizing_method="regime_confidence",
            risk_adjustment=True,
        )

    @classmethod
    def create_trading_focused(cls) -> "FinancialAnalysisConfig":
        """Create configuration focused on trading analysis."""
        return cls(
            n_states=4,
            observed_signal="log_return",
            regime_labels=["Crisis", "Bear", "Sideways", "Bull"],
            indicator_comparisons=["rsi", "macd"],
            include_indicator_performance=True,
            include_trading_signals=True,
            position_sizing_method="volatility_adjusted",
            risk_adjustment=True,
            minimum_regime_duration=3,
        )
