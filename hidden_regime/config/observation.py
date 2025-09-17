"""
Observation configuration classes for pipeline observation components.

Provides configuration for generating observations from raw data including
financial indicators, transformations, and feature engineering parameters.
"""

from dataclasses import dataclass, field
from typing import List, Callable, Optional, Any, Union
import numpy as np

from .base import BaseConfig
from ..utils.exceptions import ConfigurationError


@dataclass
class ObservationConfig(BaseConfig):
    """
    Base configuration for observation generation components.
    """
    
    generators: List[Union[str, Callable]] = field(default_factory=list)
    
    def validate(self) -> None:
        """Validate observation configuration."""
        super().validate()
        
        if not self.generators:
            raise ConfigurationError("At least one observation generator must be specified")
        
        # Validate generator types
        for i, generator in enumerate(self.generators):
            if not (isinstance(generator, str) or callable(generator)):
                raise ConfigurationError(f"Generator {i} must be string or callable, got {type(generator)}")
    
    def create_component(self) -> Any:
        """Create observation component - to be implemented by specific configs."""
        raise NotImplementedError("Subclasses must implement create_component")


@dataclass 
class FinancialObservationConfig(ObservationConfig):
    """
    Configuration for financial observation generation.
    
    Provides common financial transformations and technical indicators
    for generating observations from OHLCV data.
    """
    
    # Use string identifiers for built-in financial generators
    generators: List[str] = field(default_factory=lambda: ["log_return"])
    
    # Financial-specific parameters
    price_column: str = "close"
    volume_column: str = "volume"
    include_volume_features: bool = False
    normalize_features: bool = True
    
    def validate(self) -> None:
        """Validate financial observation configuration."""
        super().validate()
        
        # Validate built-in generator names
        valid_generators = {
            "log_return", "return_ratio", "average_price", "price_change",
            "volatility", "rsi", "macd", "bollinger_bands", "moving_average",
            "volume_sma", "volume_ratio", "price_volume_trend"
        }
        
        for generator in self.generators:
            if isinstance(generator, str) and generator not in valid_generators:
                raise ConfigurationError(f"Unknown financial generator: {generator}")
        
        # Validate column names
        if not self.price_column:
            raise ConfigurationError("price_column cannot be empty")
        
        if self.include_volume_features and not self.volume_column:
            raise ConfigurationError("volume_column required when include_volume_features=True")
    
    def create_component(self) -> Any:
        """Create financial observation component."""
        from ..observations.financial import FinancialObservationGenerator
        return FinancialObservationGenerator(self)
    
    @classmethod
    def create_default_financial(cls) -> 'FinancialObservationConfig':
        """Create default configuration for financial observations."""
        return cls(
            generators=["log_return", "volatility", "rsi"],
            price_column="close",
            include_volume_features=False,
            normalize_features=True
        )
    
    @classmethod
    def create_comprehensive_financial(cls) -> 'FinancialObservationConfig':
        """Create comprehensive configuration with many financial indicators."""
        return cls(
            generators=[
                "log_return", "volatility", "rsi", "macd", 
                "bollinger_bands", "moving_average", "volume_ratio"
            ],
            price_column="close",
            volume_column="volume",
            include_volume_features=True,
            normalize_features=True
        )
    
    @classmethod
    def create_minimal_financial(cls) -> 'FinancialObservationConfig':
        """Create minimal configuration with just returns."""
        return cls(
            generators=["log_return"],
            price_column="close", 
            include_volume_features=False,
            normalize_features=False
        )