"""
Configuration system for hidden-regime pipeline components.

Provides configuration classes for all pipeline components with validation,
serialization, and factory pattern support for creating configured components.
"""

from .analysis import AnalysisConfig, FinancialAnalysisConfig
from .base import BaseConfig
from .data import DataConfig, FinancialDataConfig
from .model import HMMConfig, ModelConfig
from .observation import FinancialObservationConfig, ObservationConfig
from .report import ReportConfig

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
    # Analysis configurations
    "AnalysisConfig",
    "FinancialAnalysisConfig",
    # Report configurations
    "ReportConfig",
]
