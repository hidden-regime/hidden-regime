"""
Hidden Regime - Market Regime Detection using Pipeline-Based Hidden Markov Models

A comprehensive Python package for market regime detection and analysis using
sophisticated Hidden Markov Models with Pipeline architecture for extensible
and rigorous V&V (Verification & Validation) backtesting.

Key Features:
- Pipeline-based architecture: Data → Observation → Model → Analysis → Report
- Temporal data isolation for rigorous backtesting V&V
- Configuration-driven component design for extensibility
- Hidden Markov Models for regime detection and classification
- Financial analysis with technical indicator integration
- Complete audit trails for regulatory compliance

Quick Start:
    >>> import hidden_regime as hr

    # Simple regime detection
    >>> pipeline = hr.create_simple_regime_pipeline('AAPL')
    >>> result = pipeline.update()

    # Financial analysis with trading insights
    >>> pipeline = hr.create_trading_pipeline('SPY', n_states=4)
    >>> result = pipeline.update()

For documentation and examples, visit: https://hiddenregime.com
"""

# Import version information from setuptools-scm
from ._version import __version__

# Package metadata
__title__ = "hidden-regime"
__author__ = "aoaustin"
__email__ = "contact@hiddenregime.com"
__description__ = "Market regime detection using Pipeline-based Hidden Markov Models with V&V backtesting"
__url__ = "https://hiddenregime.com"
__license__ = "MIT"
__copyright__ = "Copyright 2025 Hidden Regime"

# Configuration system
from .config import (
    AnalysisConfig,
    BaseConfig,
    DataConfig,
    FinancialAnalysisConfig,
    FinancialDataConfig,
    FinancialObservationConfig,
    HMMConfig,
    ModelConfig,
    ObservationConfig,
    ReportConfig,
)

# Factory patterns
from .factories import (
    ComponentFactory,
    PipelineFactory,
    component_factory,
    pipeline_factory,
)

# Core pipeline architecture
from .pipeline import (
    AnalysisComponent,
    DataComponent,
    ModelComponent,
    ObservationComponent,
    Pipeline,
    PipelineComponent,
    ReportComponent,
    TemporalController,
    TemporalDataStub,
)

# Exception handling
from .utils.exceptions import (
    ConfigurationError,
    DataLoadError,
    DataQualityError,
    HiddenRegimeError,
    HMMInferenceError,
    HMMTrainingError,
    ValidationError,
)

# Financial utilities
from .utils.state_mapping import (
    get_regime_characteristics,
    log_return_to_percent_change,
    map_states_to_financial_regimes,
    percent_change_to_log_return,
)


# Convenience functions using pipeline factory
def create_pipeline(
    data_config, observation_config, model_config, analysis_config, report_config=None
):
    """
    Create pipeline from configuration objects.

    Args:
        data_config: Data component configuration
        observation_config: Observation component configuration
        model_config: Model component configuration
        analysis_config: Analysis component configuration
        report_config: Optional report component configuration

    Returns:
        Configured Pipeline instance

    Example:
        >>> import hidden_regime as hr
        >>> data_cfg = hr.FinancialDataConfig(ticker='AAPL')
        >>> obs_cfg = hr.FinancialObservationConfig.create_default_financial()
        >>> model_cfg = hr.HMMConfig.create_balanced()
        >>> analysis_cfg = hr.FinancialAnalysisConfig.create_comprehensive_financial()
        >>> pipeline = hr.create_pipeline(data_cfg, obs_cfg, model_cfg, analysis_cfg)
    """
    return pipeline_factory.create_pipeline(
        data_config=data_config,
        observation_config=observation_config,
        model_config=model_config,
        analysis_config=analysis_config,
        report_config=report_config,
    )


def create_financial_pipeline(
    ticker="SPY",
    n_states=3,
    start_date=None,
    end_date=None,
    include_report=True,
    # Common model parameters
    tolerance=None,
    max_iterations=None,
    forgetting_factor=None,
    random_seed=None,
    **kwargs,
):
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

    Example:
        >>> import hidden_regime as hr
        >>> pipeline = hr.create_financial_pipeline('AAPL', n_states=4)
        >>> result = pipeline.update()
    """
    return pipeline_factory.create_financial_pipeline(
        ticker=ticker,
        n_states=n_states,
        start_date=start_date,
        end_date=end_date,
        include_report=include_report,
        tolerance=tolerance,
        max_iterations=max_iterations,
        forgetting_factor=forgetting_factor,
        random_seed=random_seed,
        **kwargs,
    )


def create_simple_regime_pipeline(ticker="SPY", n_states=3):
    """
    Create simple regime detection pipeline with minimal configuration.

    Args:
        ticker: Stock ticker symbol
        n_states: Number of regime states

    Returns:
        Simple regime detection Pipeline

    Example:
        >>> import hidden_regime as hr
        >>> pipeline = hr.create_simple_regime_pipeline('NVDA')
        >>> result = pipeline.update()
        >>> print(result)  # Shows current regime analysis
    """
    return pipeline_factory.create_simple_regime_pipeline(
        ticker=ticker, n_states=n_states
    )


def create_trading_pipeline(ticker="SPY", n_states=4, risk_adjustment=True):
    """
    Create pipeline optimized for trading analysis.

    Args:
        ticker: Stock ticker symbol
        n_states: Number of regime states
        risk_adjustment: Whether to include risk adjustment

    Returns:
        Trading-focused Pipeline

    Example:
        >>> import hidden_regime as hr
        >>> pipeline = hr.create_trading_pipeline('QQQ', n_states=4)
        >>> result = pipeline.update()
    """
    return pipeline_factory.create_trading_pipeline(
        ticker=ticker, n_states=n_states, risk_adjustment=risk_adjustment
    )


def create_research_pipeline(ticker="SPY", n_states=3, comprehensive_analysis=True):
    """
    Create pipeline optimized for research and analysis.

    Args:
        ticker: Stock ticker symbol
        n_states: Number of regime states
        comprehensive_analysis: Whether to include comprehensive indicators

    Returns:
        Research-focused Pipeline

    Example:
        >>> import hidden_regime as hr
        >>> pipeline = hr.create_research_pipeline('BTC-USD', comprehensive_analysis=True)
        >>> result = pipeline.update()
    """
    return pipeline_factory.create_research_pipeline(
        ticker=ticker, n_states=n_states, comprehensive_analysis=comprehensive_analysis
    )


# Temporal V&V functions
def create_temporal_controller(pipeline, full_dataset):
    """
    Create temporal controller for backtesting with V&V isolation.

    Args:
        pipeline: Pipeline instance to control
        full_dataset: Complete dataset for temporal slicing

    Returns:
        TemporalController instance

    Example:
        >>> import hidden_regime as hr
        >>> pipeline = hr.create_simple_regime_pipeline('AAPL')
        >>> data = pipeline.data.get_all_data()
        >>> controller = hr.create_temporal_controller(pipeline, data)
        >>> results = controller.step_through_time('2023-01-01', '2023-12-31')
    """
    return TemporalController(pipeline, full_dataset)


# Legacy compatibility functions (simplified)
def detect_regimes(ticker, n_states=3, start_date=None, end_date=None):
    """
    Legacy compatibility function for simple regime detection.

    Args:
        ticker: Stock ticker symbol
        n_states: Number of regimes to detect
        start_date: Start date for analysis
        end_date: End date for analysis

    Returns:
        Analysis result string

    Example:
        >>> import hidden_regime as hr
        >>> result = hr.detect_regimes('AAPL', n_states=3)
        >>> print(result)
    """
    pipeline = create_simple_regime_pipeline(ticker, n_states)

    # Update data config if dates provided
    if start_date or end_date:
        data_config = pipeline.data.config
        if start_date:
            data_config.start_date = start_date
        if end_date:
            data_config.end_date = end_date

    return pipeline.update()


# Main API exports
__all__ = [
    # Core pipeline architecture
    "Pipeline",
    "TemporalController",
    "TemporalDataStub",
    # Component interfaces
    "PipelineComponent",
    "DataComponent",
    "ObservationComponent",
    "ModelComponent",
    "AnalysisComponent",
    "ReportComponent",
    # Configuration system
    "BaseConfig",
    "DataConfig",
    "FinancialDataConfig",
    "ObservationConfig",
    "FinancialObservationConfig",
    "ModelConfig",
    "HMMConfig",
    "AnalysisConfig",
    "FinancialAnalysisConfig",
    "ReportConfig",
    # Factory patterns
    "PipelineFactory",
    "ComponentFactory",
    "pipeline_factory",
    "component_factory",
    # Exception handling
    "HiddenRegimeError",
    "DataLoadError",
    "DataQualityError",
    "ValidationError",
    "ConfigurationError",
    "HMMTrainingError",
    "HMMInferenceError",
    # Financial utilities
    "percent_change_to_log_return",
    "log_return_to_percent_change",
    "map_states_to_financial_regimes",
    "get_regime_characteristics",
    # Pipeline creation functions
    "create_pipeline",
    "create_financial_pipeline",
    "create_simple_regime_pipeline",
    "create_trading_pipeline",
    "create_research_pipeline",
    # Temporal V&V functions
    "create_temporal_controller",
    # Legacy compatibility
    "detect_regimes",
]
