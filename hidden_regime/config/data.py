"""
Data configuration classes for pipeline data components.

Provides configuration for data loading including financial data sources,
date ranges, frequency, and data quality parameters.
"""

from dataclasses import dataclass
from typing import Optional, Literal, Any
from datetime import datetime, date
import pandas as pd

from .base import BaseConfig
from abc import ABC
import json
from dataclasses import asdict
from ..utils.exceptions import ConfigurationError


@dataclass
class MutableBaseConfig(ABC):
    """
    Mutable base configuration class for configs that need to be modified after creation.
    Similar to BaseConfig but allows modifications (used for DataConfig).
    """

    def validate(self) -> None:
        """Validate configuration parameters."""
        pass

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MutableBaseConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            Configuration instance
        """
        return cls(**config_dict)

    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'MutableBaseConfig':
        """
        Create configuration from JSON string.

        Args:
            json_str: JSON string with configuration

        Returns:
            Configuration instance
        """
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    def save(self, filepath: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            filepath: Path to save configuration file
        """
        with open(filepath, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, filepath: str) -> 'MutableBaseConfig':
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to configuration file

        Returns:
            Configuration instance
        """
        with open(filepath, 'r') as f:
            json_str = f.read()
        return cls.from_json(json_str)

    def copy(self, **kwargs) -> 'MutableBaseConfig':
        """
        Create a copy of configuration with optional parameter updates.

        Args:
            **kwargs: Parameters to update in the copy

        Returns:
            New configuration instance with updated parameters
        """
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.__class__.from_dict(config_dict)

    def __post_init__(self):
        """Called after dataclass initialization to run validation."""
        self.validate()

    def __eq__(self, other) -> bool:
        """Check equality with another configuration."""
        if not isinstance(other, self.__class__):
            return False
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        """Default hash implementation."""
        # This should be overridden by subclasses for specific hashing needs
        items = sorted(self.to_dict().items())
        return hash(str(items))


@dataclass
class DataConfig(MutableBaseConfig):
    """
    Base configuration for data loading components.
    """
    
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    num_samples: Optional[int] = None
    frequency: str = "days"
    
    def validate(self) -> None:
        """Validate data configuration parameters."""
        super().validate()
        
        # Validate date formats if provided
        if self.start_date is not None:
            try:
                pd.to_datetime(self.start_date)
            except ValueError:
                raise ConfigurationError(f"Invalid start_date format: {self.start_date}")
        
        if self.end_date is not None:
            try:
                pd.to_datetime(self.end_date)
            except ValueError:
                raise ConfigurationError(f"Invalid end_date format: {self.end_date}")
        
        # Validate date ordering
        if self.start_date and self.end_date:
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            if start >= end:
                raise ConfigurationError(f"start_date {self.start_date} must be before end_date {self.end_date}")
        
        # Validate num_samples
        if self.num_samples is not None and self.num_samples <= 0:
            raise ConfigurationError(f"num_samples must be positive, got {self.num_samples}")
        
        # Validate frequency
        valid_frequencies = ["days", "hours", "minutes", "seconds"]
        if self.frequency not in valid_frequencies:
            raise ConfigurationError(f"frequency must be one of {valid_frequencies}, got {self.frequency}")
    
    def create_component(self) -> Any:
        """Create data component - to be implemented by specific data configs."""
        raise NotImplementedError("Subclasses must implement create_component")

    def __hash__(self) -> int:
        """Custom hash implementation for DataConfig."""
        # Hash based on immutable core attributes, excluding dates which may change
        return hash((
            self.source,
            self.num_samples,
            self.frequency
        ))

    def __eq__(self, other) -> bool:
        """Custom equality comparison for DataConfig."""
        if not isinstance(other, DataConfig):
            return False
        return (
            self.source == other.source and
            self.start_date == other.start_date and
            self.end_date == other.end_date and
            self.num_samples == other.num_samples and
            self.frequency == other.frequency
        )


@dataclass
class FinancialDataConfig(DataConfig):
    """
    Configuration for financial data sources like yfinance.
    """
    
    source: str = "yfinance"
    ticker: str = "SPY"
    
    def validate(self) -> None:
        """Validate financial data configuration."""
        super().validate()
        
        # Validate ticker format
        if not self.ticker or len(self.ticker.strip()) == 0:
            raise ConfigurationError("ticker cannot be empty")
        
        # Basic ticker validation (alphanumeric and common symbols)
        ticker_clean = self.ticker.replace("^", "").replace("-", "").replace(".", "")
        if not ticker_clean.isalnum():
            raise ConfigurationError(f"Invalid ticker format: {self.ticker}")
        
        # Validate source
        valid_sources = ["yfinance", "alpha_vantage", "quandl", "csv", "manual"]
        if self.source not in valid_sources:
            raise ConfigurationError(f"source must be one of {valid_sources}, got {self.source}")
    
    def create_component(self) -> Any:
        """Create financial data component."""
        from ..data.financial import FinancialDataLoader
        return FinancialDataLoader(self)
    
    def get_cache_key(self) -> str:
        """Generate cache key for this data configuration."""
        return f"{self.source}_{self.ticker}_{self.start_date}_{self.end_date}_{self.frequency}"

    def __hash__(self) -> int:
        """Custom hash implementation for FinancialDataConfig."""
        # Hash based on immutable core attributes, excluding dates which may change
        return hash((
            self.source,
            self.ticker,
            self.num_samples,
            self.frequency
        ))

    def __eq__(self, other) -> bool:
        """Custom equality comparison for FinancialDataConfig."""
        if not isinstance(other, FinancialDataConfig):
            return False
        return (
            self.source == other.source and
            self.ticker == other.ticker and
            self.start_date == other.start_date and
            self.end_date == other.end_date and
            self.num_samples == other.num_samples and
            self.frequency == other.frequency
        )