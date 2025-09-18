"""
Unit tests for FinancialDataLoader component.

Tests the financial data loading functionality including yfinance integration,
data validation, caching, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from hidden_regime.data import FinancialDataLoader
from hidden_regime.config import FinancialDataConfig
from hidden_regime.utils.exceptions import ValidationError


class TestFinancialDataLoader:
    """Test cases for FinancialDataLoader."""
    
    def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        config = FinancialDataConfig()
        loader = FinancialDataLoader(config)
        
        assert loader.config is config
        assert loader.config.ticker == "SPY"
        assert loader.config.source == "yfinance"
    
    def test_initialization_custom_config(self):
        """Test initialization with custom configuration."""
        config = FinancialDataConfig(
            ticker="AAPL",
            start_date="2023-01-01", 
            end_date="2023-12-31",
            source="yfinance"
        )
        
        loader = FinancialDataLoader(config)
        
        assert loader.config.ticker == "AAPL"
        assert loader.config.start_date == "2023-01-01"
        assert loader.config.end_date == "2023-12-31"
    
    @patch('yfinance.Ticker')
    def test_successful_data_loading(self, mock_ticker_class):
        """Test successful data loading from yfinance."""
        # Mock yfinance response
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        # Create sample OHLCV data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 30),
            'High': np.random.uniform(110, 120, 30),
            'Low': np.random.uniform(90, 100, 30),
            'Close': np.random.uniform(100, 110, 30),
            'Volume': np.random.randint(1000000, 5000000, 30)
        }, index=dates)
        
        mock_ticker.history.return_value = sample_data
        
        config = FinancialDataConfig(
            ticker="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-30"
        )
        loader = FinancialDataLoader(config)
        
        result = loader.update()
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == 30
        
        # Check for standardized column names (lowercase)
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            assert col in result.columns
        
        # Verify yfinance was called with correct parameters
        mock_ticker_class.assert_called_once_with("AAPL")
        mock_ticker.history.assert_called_once()
    
    @patch('yfinance.Ticker')
    def test_get_all_data(self, mock_ticker_class):
        """Test get_all_data method."""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 20),
            'High': np.random.uniform(110, 120, 20),
            'Low': np.random.uniform(90, 100, 20),
            'Close': np.random.uniform(100, 110, 20),
            'Volume': np.random.randint(1000000, 5000000, 20)
        }, index=dates)
        
        mock_ticker.history.return_value = sample_data
        
        config = FinancialDataConfig(ticker="TSLA")
        loader = FinancialDataLoader(config)
        
        result = loader.get_all_data()
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == 20
    
    @patch('yfinance.Ticker')
    def test_empty_data_handling(self, mock_ticker_class):
        """Test handling of empty data from yfinance."""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()  # Empty DataFrame
        
        config = FinancialDataConfig(ticker="INVALID")
        loader = FinancialDataLoader(config)
        
        result = loader.update()
        
        # Should return empty DataFrame rather than crash
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    @patch('yfinance.Ticker')
    def test_yfinance_error_handling(self, mock_ticker_class):
        """Test handling of yfinance errors."""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.side_effect = Exception("Network error")
        
        config = FinancialDataConfig(ticker="AAPL")
        loader = FinancialDataLoader(config)
        
        # Should handle the exception gracefully
        result = loader.update()
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    @patch('yfinance.Ticker')
    def test_data_column_standardization(self, mock_ticker_class):
        """Test that column names are standardized to lowercase."""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        # Create data with uppercase column names (as yfinance returns)
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        sample_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'Low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            'Close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'Volume': [1000000] * 10,
            'Adj Close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111]  # Extra column
        }, index=dates)
        
        mock_ticker.history.return_value = sample_data
        
        config = FinancialDataConfig(ticker="AAPL")
        loader = FinancialDataLoader(config)
        
        result = loader.update()
        
        # Check standardized column names
        expected_columns = {'open', 'high', 'low', 'close', 'volume'}
        result_columns = set(result.columns)
        
        assert expected_columns.issubset(result_columns)
        
        # Should not have uppercase columns
        uppercase_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        assert not uppercase_columns.intersection(result_columns)
    
    @patch('yfinance.Ticker') 
    def test_current_date_parameter(self, mock_ticker_class):
        """Test update method with current_date parameter."""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        dates = pd.date_range('2023-01-01', periods=15, freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 15),
            'High': np.random.uniform(110, 120, 15),
            'Low': np.random.uniform(90, 100, 15),
            'Close': np.random.uniform(100, 110, 15),
            'Volume': np.random.randint(1000000, 5000000, 15)
        }, index=dates)
        
        mock_ticker.history.return_value = sample_data
        
        config = FinancialDataConfig(
            ticker="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        loader = FinancialDataLoader(config)
        
        # Test with current_date
        result = loader.update(current_date="2023-01-15")
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        
        # Verify yfinance history was called
        mock_ticker.history.assert_called_once()
    
    def test_plot_method(self):
        """Test plot method generates matplotlib figure."""
        config = FinancialDataConfig(ticker="AAPL")
        loader = FinancialDataLoader(config)
        
        # Test plot with no data
        fig = loader.plot()
        
        assert fig is not None
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        
        # Clean up
        plt.close(fig)
    
    @patch('yfinance.Ticker')
    def test_plot_method_with_data(self, mock_ticker_class):
        """Test plot method with actual data."""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        sample_data = pd.DataFrame({
            'Open': range(100, 110),
            'High': range(105, 115),
            'Low': range(95, 105),
            'Close': range(102, 112),
            'Volume': [1000000] * 10
        }, index=dates)
        
        mock_ticker.history.return_value = sample_data
        
        config = FinancialDataConfig(ticker="AAPL")
        loader = FinancialDataLoader(config)
        
        # Load data first
        loader.update()
        
        # Test plot with data
        fig = loader.plot()
        
        assert fig is not None
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        
        # Should have at least one axes with data
        axes = fig.get_axes()
        assert len(axes) > 0
        
        # Clean up
        plt.close(fig)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should work
        valid_config = FinancialDataConfig(
            ticker="AAPL",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        loader = FinancialDataLoader(valid_config)
        assert loader.config.ticker == "AAPL"
        
        # Invalid ticker should be caught by config validation
        with pytest.raises(ValidationError):
            invalid_config = FinancialDataConfig(ticker="")
            invalid_config.validate()
    
    @patch('yfinance.Ticker')
    def test_data_quality_basic_checks(self, mock_ticker_class):
        """Test basic data quality checks."""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        # Create data with some quality issues
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        sample_data = pd.DataFrame({
            'Open': [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109],
            'High': [105, 106, 107, np.inf, 109, 110, 111, 112, 113, 114],
            'Low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            'Close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'Volume': [1000000] * 10
        }, index=dates)
        
        mock_ticker.history.return_value = sample_data
        
        config = FinancialDataConfig(ticker="TEST")
        loader = FinancialDataLoader(config)
        
        result = loader.update()
        
        # Should still return data (basic loader doesn't filter quality issues)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == 10
    
    @patch('yfinance.Ticker')
    def test_multiple_updates(self, mock_ticker_class):
        """Test multiple update calls."""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        sample_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000000] * 5
        }, index=dates)
        
        mock_ticker.history.return_value = sample_data
        
        config = FinancialDataConfig(ticker="AAPL")
        loader = FinancialDataLoader(config)
        
        # First update
        result1 = loader.update()
        assert not result1.empty
        
        # Second update
        result2 = loader.update()
        assert not result2.empty
        
        # Should be called twice
        assert mock_ticker.history.call_count == 2
    
    def test_serialization_support(self):
        """Test that FinancialDataLoader can be serialized."""
        import pickle
        
        config = FinancialDataConfig(ticker="AAPL")
        loader = FinancialDataLoader(config)
        
        # Serialize and deserialize
        serialized = pickle.dumps(loader)
        restored_loader = pickle.loads(serialized)
        
        assert restored_loader.config.ticker == "AAPL"
        assert isinstance(restored_loader, FinancialDataLoader)
    
    def test_string_representation(self):
        """Test string representation of FinancialDataLoader."""
        config = FinancialDataConfig(
            ticker="AAPL", 
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        loader = FinancialDataLoader(config)
        
        repr_str = repr(loader)
        assert "FinancialDataLoader" in repr_str
        assert "AAPL" in repr_str


class TestFinancialDataLoaderEdgeCases:
    """Test edge cases and error conditions for FinancialDataLoader."""
    
    @patch('yfinance.Ticker')
    def test_malformed_ticker_response(self, mock_ticker_class):
        """Test handling of malformed ticker response."""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        # Return data with wrong structure
        mock_ticker.history.return_value = pd.Series([1, 2, 3])  # Wrong type
        
        config = FinancialDataConfig(ticker="MALFORMED")
        loader = FinancialDataLoader(config)
        
        result = loader.update()
        
        # Should handle gracefully
        assert isinstance(result, pd.DataFrame)
        # May be empty or have processed the data somehow
    
    @patch('yfinance.Ticker')
    def test_partial_data_columns(self, mock_ticker_class):
        """Test handling of partial data columns."""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        # Return data with only some columns
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        partial_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'Close': [102, 103, 104, 105, 106],
            # Missing High, Low, Volume
        }, index=dates)
        
        mock_ticker.history.return_value = partial_data
        
        config = FinancialDataConfig(ticker="PARTIAL")
        loader = FinancialDataLoader(config)
        
        result = loader.update()
        
        assert isinstance(result, pd.DataFrame)
        assert 'open' in result.columns
        assert 'close' in result.columns
        # May or may not have other columns depending on implementation
    
    @patch('yfinance.Ticker')
    def test_extremely_large_dataset(self, mock_ticker_class):
        """Test handling of very large datasets."""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        # Create large dataset
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')  # ~3 years daily
        large_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 1000),
            'High': np.random.uniform(110, 120, 1000),
            'Low': np.random.uniform(90, 100, 1000),
            'Close': np.random.uniform(100, 110, 1000),
            'Volume': np.random.randint(1000000, 5000000, 1000)
        }, index=dates)
        
        mock_ticker.history.return_value = large_data
        
        config = FinancialDataConfig(ticker="LARGE")
        loader = FinancialDataLoader(config)
        
        result = loader.update()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1000
        assert not result.empty
    
    @patch('yfinance.Ticker')
    def test_data_with_timezone_index(self, mock_ticker_class):
        """Test handling of data with timezone-aware index."""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        # Create timezone-aware data
        dates = pd.date_range('2023-01-01', periods=5, freq='D', tz='UTC')
        tz_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000000] * 5
        }, index=dates)
        
        mock_ticker.history.return_value = tz_data
        
        config = FinancialDataConfig(ticker="TZ")
        loader = FinancialDataLoader(config)
        
        result = loader.update()
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        # Should preserve or handle timezone appropriately
    
    def test_config_edge_cases(self):
        """Test configuration edge cases."""
        # Very old dates
        old_config = FinancialDataConfig(
            ticker="AAPL",
            start_date="1900-01-01",
            end_date="1900-12-31"
        )
        loader = FinancialDataLoader(old_config)
        assert loader.config.start_date == "1900-01-01"
        
        # Future dates
        future_config = FinancialDataConfig(
            ticker="AAPL",
            start_date="2030-01-01",
            end_date="2030-12-31"
        )
        loader = FinancialDataLoader(future_config)
        assert loader.config.start_date == "2030-01-01"
    
    @patch('yfinance.Ticker')
    def test_concurrent_access(self, mock_ticker_class):
        """Test concurrent access to the same loader."""
        import threading
        
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        sample_data = pd.DataFrame({
            'Open': range(100, 110),
            'High': range(105, 115),
            'Low': range(95, 105),
            'Close': range(102, 112),
            'Volume': [1000000] * 10
        }, index=dates)
        
        mock_ticker.history.return_value = sample_data
        
        config = FinancialDataConfig(ticker="CONCURRENT")
        loader = FinancialDataLoader(config)
        
        results = []
        exceptions = []
        
        def update_data():
            try:
                result = loader.update()
                results.append(result)
            except Exception as e:
                exceptions.append(e)
        
        # Create multiple threads
        threads = [threading.Thread(target=update_data) for _ in range(3)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access without crashing
        assert len(exceptions) == 0  # No exceptions
        assert len(results) == 3     # All threads completed
        
        # All results should be valid DataFrames
        for result in results:
            assert isinstance(result, pd.DataFrame)
            assert not result.empty


if __name__ == "__main__":
    pytest.main([__file__])