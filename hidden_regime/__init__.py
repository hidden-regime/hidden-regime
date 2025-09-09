"""
Hidden Regime - Market Regime Detection using Hidden Markov Models

A comprehensive Python package for market regime detection and analysis using
sophisticated Hidden Markov Models with Bayesian uncertainty quantification.

Features:
- Multi-source market data loading with robust error handling
- Comprehensive data preprocessing and validation
- Advanced outlier detection and data quality assessment
- Production-ready Hidden Markov Models for regime detection
- Real-time regime inference and analysis

For documentation and examples, visit: https://hiddenregime.com
"""

# Import version information from setuptools-scm
from ._version import __version__

# Package metadata
__title__ = "hidden-regime"
__author__ = "aoaustin"
__email__ = "contact@hiddenregime.com"
__description__ = "Market regime detection using Hidden Markov Models with Bayesian uncertainty quantification"
__url__ = "https://hiddenregime.com"
__license__ = "MIT"
__copyright__ = "Copyright 2024 aoaustin"

# Import main data pipeline classes
from .data import DataLoader, DataPreprocessor, DataValidator
from .config import DataConfig, ValidationConfig, PreprocessingConfig
from .utils.exceptions import (
    HiddenRegimeError,
    DataLoadError, 
    DataQualityError,
    ValidationError,
    HMMTrainingError,
    HMMInferenceError
)

# Import HMM classes
from .models import (
    HiddenMarkovModel, HMMConfig,
    OnlineHMM, OnlineHMMConfig,
    StreamingProcessor, StreamingConfig
)

# Main API exports
__all__ = [
    # Data pipeline
    'DataLoader',
    'DataPreprocessor', 
    'DataValidator',
    
    # HMM regime detection
    'HiddenMarkovModel',
    'HMMConfig',
    'OnlineHMM',
    'OnlineHMMConfig',
    'StreamingProcessor',
    'StreamingConfig',
    
    # Configuration
    'DataConfig',
    'ValidationConfig',
    'PreprocessingConfig',
    
    # Exceptions
    'HiddenRegimeError',
    'DataLoadError',
    'DataQualityError', 
    'ValidationError',
    'HMMTrainingError',
    'HMMInferenceError',
    
    # Convenience functions
    'load_stock_data',
    'validate_data',
    'detect_regimes',
    'analyze_regime_transitions',
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


def detect_regimes(
    returns, 
    n_states=3, 
    config=None, 
    return_model=False, 
    **kwargs
):
    """
    Convenience function to detect market regimes using HMM.
    
    Args:
        returns: Log returns time series (array-like)
        n_states: Number of regimes to detect (default: 3)
        config: HMMConfig object (optional)
        return_model: If True, return (states, model), else just states
        **kwargs: Additional arguments passed to HMM.fit()
        
    Returns:
        Most likely state sequence, or (states, model) if return_model=True
        
    Example:
        >>> import hidden_regime as hr
        >>> data = hr.load_stock_data('AAPL', '2023-01-01', '2023-12-31')
        >>> states = hr.detect_regimes(data['log_return'])
        >>> print(f"Detected {len(set(states))} distinct regimes")
    """
    hmm = HiddenMarkovModel(n_states=n_states, config=config)
    hmm.fit(returns, **kwargs)
    states = hmm.predict(returns)
    
    if return_model:
        return states, hmm
    else:
        return states


def analyze_regime_transitions(
    returns, 
    dates=None, 
    n_states=3, 
    config=None, 
    **kwargs
):
    """
    Convenience function for comprehensive regime transition analysis.
    
    Args:
        returns: Log returns time series 
        dates: Optional dates corresponding to returns
        n_states: Number of regimes to detect (default: 3)
        config: HMMConfig object (optional)
        **kwargs: Additional arguments passed to HMM.fit()
        
    Returns:
        Dictionary with comprehensive regime analysis results
        
    Example:
        >>> import hidden_regime as hr
        >>> data = hr.load_stock_data('AAPL', '2023-01-01', '2023-12-31')
        >>> analysis = hr.analyze_regime_transitions(
        ...     data['log_return'], 
        ...     data['date']
        ... )
        >>> print(f"Average bull regime duration: {analysis['regime_statistics']['regime_stats'][2]['avg_duration']:.1f} days")
    """
    hmm = HiddenMarkovModel(n_states=n_states, config=config)
    hmm.fit(returns, **kwargs)
    return hmm.analyze_regimes(returns, dates)