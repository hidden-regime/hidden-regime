"""
Configuration system for hidden-regime pipeline components.

Provides configuration classes for all pipeline components with validation,
serialization, and factory pattern support for creating configured components.
"""

from .base import BaseConfig
from .data import DataConfig, FinancialDataConfig
from .interpreter import InterpreterConfiguration
from .model import HMMConfig, ModelConfig
from .observation import FinancialObservationConfig, ObservationConfig
from .pipeline import PipelineConfiguration
from .report import ReportConfig, ReportConfiguration
from .signal_generation import SignalGenerationConfiguration

__all__ = [
    # Base configuration
    "BaseConfig",
    # Data configurations
    "DataConfig",
    "FinancialDataConfig",
    # Observation configurations
    "ObservationConfig",
    "FinancialObservationConfig",
    # Model configurations
    "ModelConfig",
    "HMMConfig",
    # Interpreter configurations
    "InterpreterConfiguration",
    # Signal generation configurations
    "SignalGenerationConfiguration",
    # Report configurations
    "ReportConfig",
    "ReportConfiguration",
    # Master pipeline configuration
    "PipelineConfiguration",
]
