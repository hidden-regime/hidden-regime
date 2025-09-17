"""
Financial data loading component for pipeline architecture.

Provides FinancialDataLoader that implements DataComponent interface for loading
stock market data from various sources with robust error handling and validation.
"""

import time
import warnings
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..pipeline.interfaces import DataComponent
from ..config.data import FinancialDataConfig
from ..utils.exceptions import DataLoadError, ValidationError

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance not available. Install with: pip install yfinance")


class FinancialDataLoader(DataComponent):
    """
    Financial data loader component for pipeline architecture.
    
    Implements DataComponent interface to provide stock market data loading
    with robust error handling, caching, and data quality validation.
    """
    
    def __init__(self, config: FinancialDataConfig):
        """
        Initialize financial data loader with configuration.
        
        Args:
            config: FinancialDataConfig with data loading parameters
        """
        self.config = config
        self._cache = {}  # Simple in-memory cache
        self._last_data = None
        
        # Validate yfinance availability if needed
        if not YFINANCE_AVAILABLE and self.config.source == "yfinance":
            raise DataLoadError(
                "yfinance is not installed but set as data source. "
                "Install with: pip install yfinance"
            )
    
    def get_all_data(self) -> pd.DataFrame:
        """
        Get complete dataset.
        
        Returns:
            DataFrame with complete data including timestamps
        """
        if self._last_data is None:
            # Load data if not already loaded
            self._last_data = self._load_data()
        
        return self._last_data.copy()
    
    def update(self, current_date: Optional[str] = None) -> pd.DataFrame:
        """
        Update data, optionally fetching new data up to current_date.
        
        Args:
            current_date: Optional date to update data up to
            
        Returns:
            Updated DataFrame with any new data
        """
        # For now, we'll reload all data each time
        # In future, this could be optimized to only fetch new data
        self._last_data = self._load_data(end_date_override=current_date)
        return self._last_data.copy()
    
    def _load_data(self, end_date_override: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from configured source.
        
        Args:
            end_date_override: Optional override for end date
            
        Returns:
            DataFrame with loaded and processed data
        """
        # Determine date range
        start_date = self.config.start_date
        end_date = end_date_override or self.config.end_date
        
        # Apply num_samples limit if specified
        if self.config.num_samples is not None and start_date is None and end_date is None:
            # Calculate approximate start date based on frequency and num_samples
            if self.config.frequency == "days":
                days_back = self.config.num_samples
            else:
                days_back = self.config.num_samples  # Fallback
            
            end_dt = pd.Timestamp.now() if end_date is None else pd.to_datetime(end_date)
            start_dt = end_dt - pd.Timedelta(days=days_back)
            start_date = start_dt.strftime('%Y-%m-%d')
            if end_date is None:
                end_date = end_dt.strftime('%Y-%m-%d')
        
        # Validate inputs
        self._validate_inputs(self.config.ticker, start_date, end_date)
        
        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date) if start_date else None
        end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
        
        # Check cache first
        cache_key = f"{self.config.ticker}_{start_dt}_{end_dt}_{self.config.source}"
        if cache_key in self._cache:
            cached_data, cache_time = self._cache[cache_key]
            # Simple cache expiry (24 hours)
            if (datetime.now() - cache_time).total_seconds() < 86400:
                return cached_data.copy()
        
        # Load data based on source
        if self.config.source == "yfinance":
            raw_data = self._load_from_yfinance(self.config.ticker, start_dt, end_dt)
        else:
            raise DataLoadError(f"Unsupported data source: {self.config.source}")
        
        # Process raw data into standardized format
        processed_data = self._process_raw_data(raw_data)
        
        # Apply num_samples limit after loading if specified
        if self.config.num_samples is not None and len(processed_data) > self.config.num_samples:
            processed_data = processed_data.tail(self.config.num_samples).reset_index(drop=True)
        
        # Validate data quality
        self._validate_data_quality(processed_data, self.config.ticker)
        
        # Cache the result
        self._cache[cache_key] = (processed_data.copy(), datetime.now())
        
        return processed_data
    
    def _load_from_yfinance(
        self, 
        ticker: str, 
        start_date: Optional[pd.Timestamp], 
        end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Load data from yfinance with retry logic."""
        if not YFINANCE_AVAILABLE:
            raise DataLoadError("yfinance not available")
        
        # Default to 2 years of data if no start date
        if start_date is None:
            start_date = end_date - pd.Timedelta(days=730)
        
        for attempt in range(3):  # Simple retry logic
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(
                    start=start_date, 
                    end=end_date, 
                    auto_adjust=True, 
                    prepost=False
                )
                
                if data.empty:
                    raise DataLoadError(f"No data found for ticker {ticker}")
                
                return data
                
            except Exception as e:
                if attempt < 2:
                    time.sleep(1.0 * (2**attempt))  # Exponential backoff
                    continue
                else:
                    raise DataLoadError(f"Failed to load data for {ticker}: {e}")
    
    def _process_raw_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Process raw yfinance data into standardized format."""
        result = pd.DataFrame()
        
        # Reset index to make date a column if needed
        if isinstance(raw_data.index, pd.DatetimeIndex):
            data = raw_data.reset_index()
        else:
            data = raw_data.copy()
        
        # Ensure we have the date column
        if "Date" in data.columns:
            result.index = pd.to_datetime(data["Date"])
        elif "index" in data.columns:
            result.index = pd.to_datetime(data["index"])
        else:
            result.index = raw_data.index
        
        # Add standard OHLCV columns
        if "Open" in data.columns:
            result["open"] = data["Open"]
        if "High" in data.columns:
            result["high"] = data["High"]
        if "Low" in data.columns:
            result["low"] = data["Low"]
        if "Close" in data.columns:
            result["close"] = data["Close"]
        if "Volume" in data.columns:
            result["volume"] = data["Volume"]
        
        # Calculate price based on preference (for backward compatibility)
        if all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
            # Use OHLC average or close based on configuration (defaulting to close)
            result["price"] = data["Close"]  # Default to close for simplicity
        else:
            result["price"] = data["Close"]
        
        # Remove any rows with missing essential data
        result = result.dropna(subset=["price"]).reset_index(drop=False)
        
        return result
    
    def _validate_inputs(
        self,
        ticker: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> None:
        """Validate input parameters."""
        if not ticker or not isinstance(ticker, str):
            raise ValidationError("Ticker must be a non-empty string")
        
        if start_date and end_date:
            try:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
            except Exception as e:
                raise ValidationError(f"Invalid date format: {e}")
            
            if start_dt >= end_dt:
                raise ValidationError("Start date must be before end date")
            
            if end_dt > datetime.now():
                raise ValidationError("End date cannot be in the future")
            
            # Check minimum time period
            if (end_dt - start_dt).days < 7:
                raise ValidationError("Minimum 7-day time period required")
    
    def _validate_data_quality(self, data: pd.DataFrame, ticker: str) -> None:
        """Validate loaded data quality."""
        if data.empty:
            raise DataLoadError(f"No data loaded for {ticker}")
        
        if len(data) < 10:  # Minimum observations
            raise DataLoadError(
                f"Insufficient data for {ticker}: {len(data)} < 10"
            )
        
        # Check for reasonable price values
        if "price" in data.columns and (data["price"] <= 0).any():
            raise DataLoadError(f"Invalid price values found for {ticker}")
        
        if "close" in data.columns and (data["close"] <= 0).any():
            raise DataLoadError(f"Invalid close price values found for {ticker}")
    
    def plot(self, **kwargs) -> plt.Figure:
        """
        Generate visualization of financial data.
        
        Returns:
            matplotlib Figure with financial data plots
        """
        if self._last_data is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data loaded yet', ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        data = self._last_data
        
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Price data
        ax1 = axes[0]
        if "close" in data.columns:
            ax1.plot(data.index, data["close"], label="Close Price", linewidth=1.5)
        elif "price" in data.columns:
            ax1.plot(data.index, data["price"], label="Price", linewidth=1.5)
        
        ax1.set_title(f'Price Data for {self.config.ticker}')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Volume (if available)
        ax2 = axes[1]
        if "volume" in data.columns:
            ax2.bar(data.index, data["volume"], alpha=0.6, label="Volume")
            ax2.set_title('Trading Volume')
            ax2.set_ylabel('Volume')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Volume data not available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Trading Volume')
        
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_entries": len(self._cache),
            "ticker": self.config.ticker,
            "source": self.config.source,
            "last_data_shape": self._last_data.shape if self._last_data is not None else None
        }
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()