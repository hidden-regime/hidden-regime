"""
Hidden Regime - Market Regime Detection using Hidden Markov Models

A comprehensive Python package for market regime detection and analysis using
sophisticated Hidden Markov Models with Bayesian uncertainty quantification.

Features:
- Multi-source market data loading with robust error handling
- Comprehensive data preprocessing and validation
- Advanced outlier detection and data quality assessment
- Extensible architecture for HMM regime detection (coming soon)

For documentation and examples, visit: https://hiddenregime.com
"""

__version__ = "0.1.0"
__author__ = "aoaustin"
__email__ = "contact@hiddenregime.com"

# Import main data pipeline classes
from .data import DataLoader, DataPreprocessor, DataValidator
from .config import DataConfig, ValidationConfig, PreprocessingConfig
from .utils.exceptions import (
    HiddenRegimeError,
    DataLoadError, 
    DataQualityError,
    ValidationError
)

# Main API exports
__all__ = [
    # Data pipeline
    'DataLoader',
    'DataPreprocessor', 
    'DataValidator',
    
    # Configuration
    'DataConfig',
    'ValidationConfig',
    'PreprocessingConfig',
    
    # Exceptions
    'HiddenRegimeError',
    'DataLoadError',
    'DataQualityError', 
    'ValidationError',
    
    # Convenience functions
    'load_stock_data',
    'validate_data',
]

# Convenience functions for common operations
def load_stock_data(ticker, start_date, end_date, **kwargs):
    """
    Convenience function to load stock data.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'SPY')
        start_date: Start date as string 'YYYY-MM-DD' or datetime
        end_date: End date as string 'YYYY-MM-DD' or datetime
        **kwargs: Additional arguments passed to DataLoader
        
    Returns:
        DataFrame with columns: date, price, log_return, volume (optional)
        
    Example:
        >>> import hidden_regime as hr
        >>> data = hr.load_stock_data('AAPL', '2023-01-01', '2023-12-31')
        >>> print(data.head())
    """
    loader = DataLoader(**kwargs)
    return loader.load_stock_data(ticker, start_date, end_date)

def validate_data(data, ticker=None, **kwargs):
    """
    Convenience function to validate data quality.
    
    Args:
        data: DataFrame to validate
        ticker: Optional ticker symbol for context
        **kwargs: Additional arguments passed to DataValidator
        
    Returns:
        ValidationResult with detailed findings and recommendations
        
    Example:
        >>> import hidden_regime as hr
        >>> data = hr.load_stock_data('AAPL', '2023-01-01', '2023-12-31')
        >>> result = hr.validate_data(data, 'AAPL')
        >>> print(f"Data quality score: {result.quality_score:.2f}")
    """
    validator = DataValidator(**kwargs)
    return validator.validate_data(data, ticker)