"""
Configuration system for hidden-regime pipeline components.

Provides configuration classes for all pipeline components with validation,
serialization, and factory pattern support for creating configured components.
"""

from .base import BaseConfig
from .data import DataConfig, FinancialDataConfig
from .observation import ObservationConfig, FinancialObservationConfig
from .model import ModelConfig, HMMConfig
from .analysis import AnalysisConfig, FinancialAnalysisConfig
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