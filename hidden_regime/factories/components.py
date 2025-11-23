"""
Component factory for creating pipeline components from configurations.

Provides centralized factory methods for instantiating Data, Observation, Model,
Analysis, and Report components from their respective configuration objects.
"""

import logging
from typing import Any, Dict, Type, Union

from ..config import (
    DataConfig,
    FinancialDataConfig,
    FinancialObservationConfig,
    HMMConfig,
    ModelConfig,
    ObservationConfig,
    ReportConfig,
)
from ..config.interpreter import InterpreterConfiguration
from ..config.signal_generation import SignalGenerationConfiguration
from ..utils.exceptions import ConfigurationError


class ComponentFactory:
    """
    Factory for creating pipeline components from configuration objects.

    Provides centralized component creation with extensible registration
    system for adding new component types.
    """

    def __init__(self):
        """Initialize component factory with default registrations."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Component registries
        self._data_registry = {}
        self._observation_registry = {}
        self._model_registry = {}
        self._interpreter_registry = {}
        self._signal_generator_registry = {}
        self._report_registry = {}

        # Register default components
        self._register_default_components()

    def _register_default_components(self):
        """Register default component implementations."""
        # Data components
        self.register_data_component("FinancialDataConfig", self._create_financial_data)

        # Observation components
        self.register_observation_component(
            "FinancialObservationConfig", self._create_financial_observation
        )

        # Model components
        self.register_model_component("HMMConfig", self._create_hmm_model)

        # Interpreter components (new architecture)
        self.register_interpreter_component(
            "InterpreterConfiguration", self._create_financial_interpreter
        )

        # Signal Generator components
        self.register_signal_generator_component(
            "SignalGenerationConfiguration", self._create_financial_signal_generator
        )

        # Report components
        self.register_report_component("ReportConfig", self._create_report)

    # Registration methods
    def register_data_component(self, config_type: str, creator_func: callable):
        """Register a data component creator function."""
        self._data_registry[config_type] = creator_func

    def register_observation_component(self, config_type: str, creator_func: callable):
        """Register an observation component creator function."""
        self._observation_registry[config_type] = creator_func

    def register_model_component(self, config_type: str, creator_func: callable):
        """Register a model component creator function."""
        self._model_registry[config_type] = creator_func

    def register_interpreter_component(self, config_type: str, creator_func: callable):
        """Register an interpreter component creator function."""
        self._interpreter_registry[config_type] = creator_func

    def register_signal_generator_component(self, config_type: str, creator_func: callable):
        """Register a signal generator component creator function."""
        self._signal_generator_registry[config_type] = creator_func

    def register_report_component(self, config_type: str, creator_func: callable):
        """Register a report component creator function."""
        self._report_registry[config_type] = creator_func

    # Component creation methods
    def create_data_component(self, config: DataConfig) -> Any:
        """
        Create data component from configuration.

        Args:
            config: Data configuration object

        Returns:
            Initialized data component
        """
        config_type = config.__class__.__name__

        if config_type not in self._data_registry:
            raise ConfigurationError(f"Unknown data configuration type: {config_type}")

        try:
            component = self._data_registry[config_type](config)
            self.logger.debug(f"Created data component: {type(component).__name__}")
            return component
        except Exception as e:
            raise ConfigurationError(f"Failed to create data component: {str(e)}")

    def create_observation_component(self, config: ObservationConfig) -> Any:
        """
        Create observation component from configuration.

        Args:
            config: Observation configuration object

        Returns:
            Initialized observation component
        """
        config_type = config.__class__.__name__

        if config_type not in self._observation_registry:
            raise ConfigurationError(
                f"Unknown observation configuration type: {config_type}"
            )

        try:
            component = self._observation_registry[config_type](config)
            self.logger.debug(
                f"Created observation component: {type(component).__name__}"
            )
            return component
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create observation component: {str(e)}"
            )

    def create_model_component(self, config: ModelConfig) -> Any:
        """
        Create model component from configuration.

        Args:
            config: Model configuration object

        Returns:
            Initialized model component
        """
        config_type = config.__class__.__name__

        if config_type not in self._model_registry:
            raise ConfigurationError(f"Unknown model configuration type: {config_type}")

        try:
            component = self._model_registry[config_type](config)
            self.logger.debug(f"Created model component: {type(component).__name__}")
            return component
        except Exception as e:
            raise ConfigurationError(f"Failed to create model component: {str(e)}")

    def create_interpreter_component(self, config: InterpreterConfiguration) -> Any:
        """
        Create interpreter component from configuration.

        Args:
            config: Interpreter configuration object

        Returns:
            Initialized interpreter component
        """
        config_type = config.__class__.__name__

        if config_type not in self._interpreter_registry:
            raise ConfigurationError(
                f"Unknown interpreter configuration type: {config_type}"
            )

        try:
            component = self._interpreter_registry[config_type](config)
            self.logger.debug(f"Created interpreter component: {type(component).__name__}")
            return component
        except Exception as e:
            raise ConfigurationError(f"Failed to create interpreter component: {str(e)}")

    def create_signal_generator_component(self, config: SignalGenerationConfiguration) -> Any:
        """
        Create signal generator component from configuration.

        Args:
            config: Signal generation configuration object

        Returns:
            Initialized signal generator component
        """
        config_type = config.__class__.__name__

        if config_type not in self._signal_generator_registry:
            raise ConfigurationError(
                f"Unknown signal generator configuration type: {config_type}"
            )

        try:
            component = self._signal_generator_registry[config_type](config)
            self.logger.debug(f"Created signal generator component: {type(component).__name__}")
            return component
        except Exception as e:
            raise ConfigurationError(f"Failed to create signal generator component: {str(e)}")

    def create_report_component(self, config: ReportConfig) -> Any:
        """
        Create report component from configuration.

        Args:
            config: Report configuration object

        Returns:
            Initialized report component
        """
        config_type = config.__class__.__name__

        if config_type not in self._report_registry:
            raise ConfigurationError(
                f"Unknown report configuration type: {config_type}"
            )

        try:
            component = self._report_registry[config_type](config)
            self.logger.debug(f"Created report component: {type(component).__name__}")
            return component
        except Exception as e:
            raise ConfigurationError(f"Failed to create report component: {str(e)}")

    # Default component creator functions
    def _create_financial_data(self, config: FinancialDataConfig) -> Any:
        """Create financial data component."""
        from ..data.financial import FinancialDataLoader

        return FinancialDataLoader(config)

    def _create_financial_observation(self, config: FinancialObservationConfig) -> Any:
        """Create financial observation component."""
        from ..observations.financial import FinancialObservationGenerator

        return FinancialObservationGenerator(config)

    def _create_hmm_model(self, config: HMMConfig) -> Any:
        """Create HMM model component."""
        from ..models.hmm import HiddenMarkovModel

        return HiddenMarkovModel(config)

    def _create_financial_interpreter(self, config: InterpreterConfiguration) -> Any:
        """Create financial interpreter component (new architecture)."""
        from ..interpreter.financial import FinancialInterpreter

        return FinancialInterpreter(config)

    def _create_financial_signal_generator(self, config: SignalGenerationConfiguration) -> Any:
        """Create financial signal generator component."""
        from ..signal_generation.financial import FinancialSignalGenerator

        return FinancialSignalGenerator(config)

    def _create_report(self, config: ReportConfig) -> Any:
        """Create report component."""
        from ..reports.markdown import MarkdownReportGenerator

        return MarkdownReportGenerator(config)

    def get_registered_components(self) -> Dict[str, Dict[str, callable]]:
        """
        Get all registered component creators.

        Returns:
            Dictionary mapping component types to their registries
        """
        return {
            "data": dict(self._data_registry),
            "observation": dict(self._observation_registry),
            "model": dict(self._model_registry),
            "interpreter": dict(self._interpreter_registry),
            "signal_generator": dict(self._signal_generator_registry),
            "report": dict(self._report_registry),
        }


# Global factory instance
component_factory = ComponentFactory()
