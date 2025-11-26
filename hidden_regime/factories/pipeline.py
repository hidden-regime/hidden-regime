"""
Pipeline factory for creating complete pipelines from configurations.

Provides high-level factory methods for creating entire pipeline systems
from configuration objects, with support for financial use cases and
extensible patterns for future domains.
"""

import logging
from typing import Any, Dict, Optional

from ..config import (
    DataConfig,
    FinancialDataConfig,
    FinancialObservationConfig,
    HMMConfig,
    ModelConfig,
    ObservationConfig,
    ReportConfig,
)
from ..pipeline.core import Pipeline
from ..utils.exceptions import ConfigurationError
from .components import component_factory


class PipelineFactory:
    """
    Factory for creating complete pipeline systems.

    Provides convenient methods for creating pipelines with different
    configurations and use cases, particularly financial analysis workflows.
    """

    def __init__(self):
        """Initialize pipeline factory."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.component_factory = component_factory

    def _validate_features(self, features: list) -> None:
        """
        Validate that requested features are available.

        Args:
            features: List of feature names to validate

        Raises:
            ConfigurationError: If any feature is not available
        """
        from ..utils.exceptions import ConfigurationError

        # Valid feature names that can be generated
        valid_features = {
            "log_return", "return_ratio", "average_price", "price_change",
            "volatility", "realized_vol",
            "momentum_strength", "trend_persistence", "volatility_context", "directional_consistency",
            "rsi", "macd", "bollinger_bands", "moving_average",
            "volume_sma", "volume_ratio", "volume_change", "price_volume_trend",
        }

        invalid_features = [f for f in features if f not in valid_features]

        if invalid_features:
            raise ConfigurationError(
                f"Invalid features requested: {invalid_features}\n"
                f"Valid features are: {sorted(valid_features)}\n"
                f"Common feature combinations:\n"
                f"  - ['log_return', 'realized_vol'] (recommended for multivariate)\n"
                f"  - ['log_return', 'momentum_strength']\n"
                f"  - ['log_return', 'rsi']\n"
                f"See FinancialObservationGenerator._get_builtin_generator() for all available features."
            )

    def create_pipeline(
        self,
        data_config: DataConfig,
        observation_config: ObservationConfig,
        model_config: ModelConfig,
        interpreter_config: Optional["InterpreterConfiguration"] = None,
        signal_generator_config: Optional["SignalGenerationConfiguration"] = None,
        report_config: Optional[ReportConfig] = None,
    ) -> Pipeline:
        """
        Create pipeline from configuration objects.

        Args:
            data_config: Data component configuration
            observation_config: Observation component configuration
            model_config: Model component configuration
            interpreter_config: Interpreter component configuration
            signal_generator_config: Signal generator component configuration (optional)
            report_config: Optional report component configuration

        Returns:
            Configured Pipeline instance
        """
        try:
            # Create components using factory
            data_component = self.component_factory.create_data_component(data_config)
            observation_component = self.component_factory.create_observation_component(
                observation_config
            )
            model_component = self.component_factory.create_model_component(
                model_config
            )

            # Create interpreter and optional signal generator
            interpreter_component = None
            signal_generator_component = None

            if interpreter_config is not None:
                interpreter_component = (
                    self.component_factory.create_interpreter_component(
                        interpreter_config
                    )
                )
                if signal_generator_config is not None:
                    signal_generator_component = (
                        self.component_factory.create_signal_generator_component(
                            signal_generator_config
                        )
                    )
            else:
                raise ConfigurationError("Must provide 'interpreter_config'")

            report_component = None
            if report_config is not None:
                report_component = self.component_factory.create_report_component(
                    report_config
                )

            # Create pipeline
            pipeline = Pipeline(
                data=data_component,
                observation=observation_component,
                model=model_component,
                interpreter=interpreter_component,
                signal_generator=signal_generator_component,
                report=report_component,
            )

            self.logger.info(
                "Successfully created pipeline with Interpreter + SignalGenerator architecture"
            )
            return pipeline

        except Exception as e:
            raise ConfigurationError(f"Failed to create pipeline: {str(e)}")

    def create_financial_pipeline(
        self,
        ticker: str = "SPY",
        n_states: int = 3,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_report: bool = True,
        # Common model parameters
        tolerance: Optional[float] = None,
        max_iterations: Optional[int] = None,
        forgetting_factor: Optional[float] = None,
        random_seed: Optional[int] = None,
        **kwargs,
    ) -> Pipeline:
        """
        Create pipeline optimized for financial analysis.

        Args:
            ticker: Stock ticker symbol
            n_states: Number of regime states for HMM
            start_date: Data start date (YYYY-MM-DD)
            end_date: Data end date (YYYY-MM-DD)
            include_report: Whether to include report generation
            **kwargs: Additional configuration overrides

        Returns:
            Configured financial Pipeline instance
        """
        # Create financial data configuration
        data_config_params = {
            "source": "yfinance",
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
        }

        # Add any additional data config overrides
        data_config_params.update(kwargs.get("data_config_overrides", {}))

        data_config = FinancialDataConfig(**data_config_params)

        # Create financial observation configuration
        if "observations_config" in kwargs:
            observation_config = kwargs["observations_config"]
        else:
            if "observation_config_overrides" in kwargs:
                # Use copy() method for frozen dataclasses
                observation_config = FinancialObservationConfig.create_default_financial().copy(
                    **kwargs["observation_config_overrides"]
                )
            else:
                observation_config = FinancialObservationConfig.create_default_financial()

        # Create HMM model configuration
        model_config_params = {"n_states": n_states}

        # Add explicit common parameters if provided
        if tolerance is not None:
            model_config_params["tolerance"] = tolerance
        if max_iterations is not None:
            model_config_params["max_iterations"] = max_iterations
        if forgetting_factor is not None:
            model_config_params["forgetting_factor"] = forgetting_factor
        if random_seed is not None:
            model_config_params["random_seed"] = random_seed

        # Add any additional model config overrides
        if "model_config_overrides" in kwargs:
            model_config_params.update(kwargs["model_config_overrides"])

        model_config = HMMConfig.create_balanced().copy(**model_config_params)

        # Create interpreter configuration (new architecture)
        interpreter_config_params = {"n_states": n_states}
        if "interpreter_config_overrides" in kwargs:
            interpreter_config_params.update(kwargs["interpreter_config_overrides"])
        from hidden_regime.config.interpreter import InterpreterConfiguration

        interpreter_config = InterpreterConfiguration(**interpreter_config_params)

        # Create signal generator configuration (new architecture)
        signal_config_params = {}
        if "signal_config_overrides" in kwargs:
            signal_config_params.update(kwargs["signal_config_overrides"])
        from hidden_regime.config.signal_generation import SignalGenerationConfiguration

        signal_generator_config = SignalGenerationConfiguration(**signal_config_params)

        # Create report configuration if requested
        report_config = None
        if include_report:
            if "report_config_overrides" in kwargs:
                # Use copy() method for frozen dataclasses
                report_config = ReportConfig.create_comprehensive().copy(
                    **kwargs["report_config_overrides"]
                )
            else:
                report_config = ReportConfig.create_comprehensive()

        return self.create_pipeline(
            data_config=data_config,
            observation_config=observation_config,
            model_config=model_config,
            interpreter_config=interpreter_config,
            signal_generator_config=signal_generator_config,
            report_config=report_config,
        )

    def create_simple_regime_pipeline(
        self, ticker: str = "SPY", n_states: int = 3
    ) -> Pipeline:
        """
        Create simple regime detection pipeline with minimal configuration.

        Args:
            ticker: Stock ticker symbol
            n_states: Number of regime states

        Returns:
            Simple regime detection Pipeline
        """
        return self.create_financial_pipeline(
            ticker=ticker,
            n_states=n_states,
            include_report=False,
            observation_config_overrides={
                "generators": ["log_return"],
                "normalize_features": False,
            },
            analysis_config_overrides={
                "indicator_comparisons": None,
                "include_indicator_performance": False,
                "include_trading_signals": False,
            },
        )

    def create_trading_pipeline(
        self, ticker: str = "SPY", n_states: int = 4, risk_adjustment: bool = True
    ) -> Pipeline:
        """
        Create pipeline optimized for trading analysis.

        Args:
            ticker: Stock ticker symbol
            n_states: Number of regime states
            risk_adjustment: Whether to include risk adjustment

        Returns:
            Trading-focused Pipeline
        """
        return self.create_financial_pipeline(
            ticker=ticker,
            n_states=n_states,
            include_report=True,
            observation_config_overrides={
                "generators": ["log_return", "volatility", "rsi"],
                "normalize_features": True,
            },
            model_config_overrides={
                "initialization_method": "quantile",
                "enable_change_detection": True,
            },
            analysis_config_overrides={
                "regime_labels": (
                    ["Crisis", "Bear", "Sideways", "Bull"] if n_states == 4 else None
                ),
                "include_trading_signals": True,
                "position_sizing_method": (
                    "volatility_adjusted" if risk_adjustment else "regime_confidence"
                ),
                "risk_adjustment": risk_adjustment,
            },
        )

    def create_research_pipeline(
        self,
        ticker: str = "SPY",
        n_states: int = 3,
        comprehensive_analysis: bool = True,
    ) -> Pipeline:
        """
        Create pipeline optimized for research and analysis.

        Args:
            ticker: Stock ticker symbol
            n_states: Number of regime states
            comprehensive_analysis: Whether to include comprehensive indicators

        Returns:
            Research-focused Pipeline
        """
        indicators = (
            ["rsi", "macd", "bollinger_bands", "moving_average"]
            if comprehensive_analysis
            else ["rsi"]
        )

        return self.create_financial_pipeline(
            ticker=ticker,
            n_states=n_states,
            include_report=True,
            observation_config_overrides={
                "generators": ["log_return", "volatility"] + indicators,
                "normalize_features": True,
                "include_volume_features": True,
            },
            analysis_config_overrides={
                "indicator_comparisons": indicators,
                "include_indicator_performance": True,
                "calculate_regime_statistics": True,
                "include_duration_analysis": True,
                "include_return_analysis": True,
                "include_volatility_analysis": True,
            },
            report_config_overrides={
                "template_style": "academic",
                "include_performance_metrics": True,
                "include_risk_analysis": True,
                "save_plots": True,
            },
        )

    def create_multivariate_pipeline(
        self,
        ticker: str = "SPY",
        n_states: int = 3,
        features: list = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs,
    ) -> Pipeline:
        """
        Create pipeline for multivariate regime detection.

        Args:
            ticker: Stock ticker symbol
            n_states: Number of regime states
            features: List of feature names (default: ['log_return', 'realized_vol'])
            start_date: Data start date
            end_date: Data end date
            **kwargs: Additional configuration overrides

        Returns:
            Pipeline with multivariate HMM configuration

        Raises:
            ConfigurationError: If requested features are not available

        Example:
            >>> factory = PipelineFactory()
            >>> pipeline = factory.create_multivariate_pipeline(
            ...     'SPY',
            ...     features=['log_return', 'realized_vol'],
            ...     n_states=3
            ... )
        """
        # Default features: returns + realized volatility
        if features is None:
            features = ['log_return', 'realized_vol']

        # Validate that all requested features are available
        self._validate_features(features)

        # Build HMM config with multivariate features
        from hidden_regime.config.model import ObservationMode

        hmm_config_params = {
            'n_states': n_states,
            'observation_mode': ObservationMode.MULTIVARIATE,
            'observed_signals': features,  # Multivariate mode
            'initialization_method': 'kmeans',
            'max_iterations': kwargs.get('max_iterations', 100),
            'tolerance': kwargs.get('tolerance', 1e-6),
            'random_seed': kwargs.get('random_seed', None),
        }
        hmm_config_params.update(kwargs.get('model_config_overrides', {}))
        hmm_config = HMMConfig(**hmm_config_params)

        # Build data config
        data_config_params = {
            'source': 'yfinance',
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
        }
        data_config_params.update(kwargs.get('data_config_overrides', {}))
        data_config = FinancialDataConfig(**data_config_params)

        # Build observation config with multivariate features
        observation_config_params = {
            'generators': features,
            'normalize_features': False,  # HMM will handle standardization
            'window_size': 20,  # For realized_vol calculation
        }
        observation_config_params.update(kwargs.get('observation_config_overrides', {}))
        observation_config = FinancialObservationConfig(**observation_config_params)

        # Create components using component factory
        data_component = self.component_factory.create_data_component(data_config)
        observation_component = self.component_factory.create_observation_component(
            observation_config
        )
        model_component = self.component_factory.create_model_component(hmm_config)

        # Create interpreter configuration (required for pipeline)
        from hidden_regime.config.interpreter import InterpreterConfiguration
        interpreter_config = InterpreterConfiguration(n_states=n_states)
        interpreter_config = kwargs.get('interpreter_config_overrides', interpreter_config)

        interpreter_component = self.component_factory.create_interpreter_component(
            interpreter_config
        )

        # Create signal generator if requested
        signal_generator_component = None
        if kwargs.get('include_signals', False):
            from hidden_regime.config.signal_generation import SignalGenerationConfiguration
            signal_config = SignalGenerationConfiguration()
            signal_generator_component = self.component_factory.create_signal_generator_component(
                signal_config
            )

        # Build pipeline with all required components
        pipeline = Pipeline(
            data=data_component,
            observation=observation_component,
            model=model_component,
            interpreter=interpreter_component,
            signal_generator=signal_generator_component,
            report=None  # No report for multivariate pipelines by default
        )

        self.logger.info(
            f"Successfully created multivariate pipeline for {ticker} with features {features}"
        )

        return pipeline

    def create_case_study_pipeline(
        self,
        ticker: str = "SPY",
        n_states: int = 3,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        enable_temporal_isolation: bool = True,
        include_indicators: bool = True,
        **kwargs,
    ) -> Pipeline:
        """
        Create pipeline optimized for case study analysis with temporal isolation.

        This pipeline is specifically designed for use with TemporalController
        to ensure proper temporal isolation during case study evolution.

        Args:
            ticker: Stock ticker symbol
            n_states: Number of regime states for HMM
            start_date: Analysis start date (for data config)
            end_date: Analysis end date (for data config)
            enable_temporal_isolation: Whether to configure for temporal isolation
            include_indicators: Whether to include technical indicators
            **kwargs: Additional configuration overrides

        Returns:
            Pipeline optimized for case study analysis
        """
        # Create data configuration for case study
        data_config_params = {
            "source": "yfinance",
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
        }
        data_config_params.update(kwargs.get("data_config_overrides", {}))
        data_config = FinancialDataConfig(**data_config_params)

        # Create observation configuration
        observation_generators = ["log_return"]
        if include_indicators:
            observation_generators.extend(["rsi", "macd", "volatility"])

        observation_config = FinancialObservationConfig.create_default_financial()
        observation_config.generators = observation_generators
        observation_config.normalize_features = True

        # Apply observation overrides
        if "observation_config_overrides" in kwargs:
            for key, value in kwargs["observation_config_overrides"].items():
                setattr(observation_config, key, value)

        # Create model configuration optimized for case studies
        model_config_params = {
            "n_states": n_states,
            "random_seed": 42,  # Reproducible results
            "tolerance": 1e-6,
            "max_iterations": 1000,
        }

        # Apply model overrides
        if "model_config_overrides" in kwargs:
            model_config_params.update(kwargs["model_config_overrides"])

        model_config = HMMConfig.create_balanced().copy(**model_config_params)

        # Create analysis configuration for case studies
        analysis_config_params = {
            "n_states": n_states,
            "calculate_regime_statistics": True,
            "include_duration_analysis": False,  # Disable to prevent issues during evolution
            "include_return_analysis": True,
            "include_volatility_analysis": True,
            "include_trading_signals": True,
        }

        # Include indicator comparisons if requested
        if include_indicators:
            analysis_config_params.update(
                {
                    "indicator_comparisons": ["rsi", "macd"],
                    "include_indicator_performance": True,
                }
            )

        # Apply analysis overrides
        if "analysis_config_overrides" in kwargs:
            analysis_config_params.update(kwargs["analysis_config_overrides"])

        analysis_config = FinancialAnalysisConfig.create_comprehensive_financial().copy(
            **analysis_config_params
        )

        # Create pipeline without report component (case studies generate their own reports)
        pipeline = self.create_pipeline(
            data_config=data_config,
            observation_config=observation_config,
            model_config=model_config,
            analysis_config=analysis_config,
            report_config=None,  # No standard reports for case studies
        )

        self.logger.info(
            f"Created case study pipeline for {ticker} with {n_states} states"
        )

        return pipeline

    def get_factory_info(self) -> Dict[str, Any]:
        """
        Get information about factory capabilities.

        Returns:
            Dictionary with factory information
        """
        registered_components = self.component_factory.get_registered_components()

        return {
            "factory_type": "PipelineFactory",
            "registered_components": registered_components,
            "available_methods": [
                "create_pipeline",
                "create_financial_pipeline",
                "create_simple_regime_pipeline",
                "create_trading_pipeline",
                "create_research_pipeline",
                "create_case_study_pipeline",
            ],
            "supported_domains": ["financial"],
            "extensible": True,
        }


# Global pipeline factory instance
pipeline_factory = PipelineFactory()
