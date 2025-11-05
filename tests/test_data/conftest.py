"""
Fixtures specific to data loading and processing tests.

Provides fixtures for mock data sources, data validation, and processing.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture
def sample_ticker_data():
    """Provide sample ticker OHLCV data."""
    dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
    np.random.seed(42)

    close = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.02))
    high = close * (1 + np.abs(np.random.randn(252) * 0.01))
    low = close * (1 - np.abs(np.random.randn(252) * 0.01))
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    volume = np.random.randint(1000000, 10000000, 252)

    return pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)


@pytest.fixture
def mock_yfinance_download():
    """Provide mock for yfinance.download function."""
    def _mock_download(ticker, start=None, end=None, **kwargs):
        dates = pd.date_range(start=start or '2020-01-01', periods=100, freq='D')
        np.random.seed(42)

        close = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))

        return pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.01,
            'Low': close * 0.98,
            'Close': close,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

    return _mock_download


@pytest.fixture
def mock_empty_yfinance_data():
    """Provide mock that returns empty DataFrame (ticker not found)."""
    def _mock_empty(*args, **kwargs):
        return pd.DataFrame()

    return _mock_empty


@pytest.fixture
def mock_insufficient_yfinance_data():
    """Provide mock that returns insufficient data."""
    def _mock_insufficient(*args, **kwargs):
        dates = pd.date_range(start='2020-01-01', periods=5, freq='D')
        return pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [101, 102, 103, 104, 105],
            'Low': [99, 100, 101, 102, 103],
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000000] * 5
        }, index=dates)

    return _mock_insufficient


@pytest.fixture
def data_with_missing_values():
    """Provide data with missing values for validation testing."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    np.random.seed(42)

    close = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))

    # Introduce missing values
    close[10:15] = np.nan  # Gap of 5 days
    close[50] = np.nan     # Single missing value

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.01,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)


@pytest.fixture
def data_with_invalid_values():
    """Provide data with invalid values (inf, negative prices)."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    np.random.seed(42)

    close = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))

    # Introduce invalid values
    close[20] = np.inf
    close[30] = -10.0  # Negative price
    close[40] = 0.0    # Zero price

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.01,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)


@pytest.fixture
def multi_ticker_data():
    """Provide data for multiple tickers."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    np.random.seed(42)

    data = {}
    for ticker in ['AAPL', 'MSFT', 'GOOGL']:
        close = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
        data[ticker] = pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.01,
            'Low': close * 0.98,
            'Close': close,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

    return data


@pytest.fixture
def data_with_timezone():
    """Provide data with timezone-aware datetime index."""
    dates = pd.date_range(
        start='2020-01-01',
        periods=100,
        freq='D',
        tz='America/New_York'
    )
    np.random.seed(42)

    close = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.01,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)


@pytest.fixture
def data_loader_config():
    """Provide standard data loader configuration."""
    return {
        'ticker': 'TEST',
        'start_date': '2020-01-01',
        'end_date': '2021-01-01',
        'price_column': 'Close',
        'validate_data': True,
        'min_periods': 50
    }


@pytest.fixture
def sample_cache_data():
    """Provide sample cache data structure."""
    return {
        'ticker': 'TEST',
        'data': pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2020-01-01', periods=3, freq='D')),
        'metadata': {
            'source': 'yfinance',
            'cached_at': datetime.now().isoformat(),
            'version': '1.0'
        }
    }


@pytest.fixture
def processor_config():
    """Provide data processor configuration."""
    return {
        'method': 'log_returns',
        'window': None,
        'fill_method': 'ffill',
        'max_gap': 3
    }
