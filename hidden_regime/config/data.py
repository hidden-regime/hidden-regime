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
from ..utils.exceptions import ConfigurationError


@dataclass
class DataConfig(BaseConfig):
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