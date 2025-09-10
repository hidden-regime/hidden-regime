"""
Unit tests for DataLoader class.

Tests data loading functionality with various scenarios including
success cases, error handling, and edge cases.
"""

import warnings
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hidden_regime.config.settings import DataConfig
from hidden_regime.data.loader import DataLoader
from hidden_regime.utils.exceptions import DataLoadError, ValidationError
from tests.fixtures.sample_data import MockYFinanceTicker, create_sample_stock_data


class TestDataLoader:
    """Test suite for DataLoader class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()
        self.config = DataConfig()
        self.sample_ticker = "AAPL"
        self.start_date = "2023-01-01"
        self.end_date = "2023-12-31"

    def test_init_default_config(self):
        """Test DataLoader initialization with default configuration."""
        loader = DataLoader()
        assert loader.config.default_source == "yfinance"
        assert loader.config.use_ohlc_average == True
        assert loader.config.cache_enabled == True
        assert loader._cache == {}

    def test_init_custom_config(self):
        """Test DataLoader initialization with custom configuration."""
        custom_config = DataConfig(
            default_source="yfinance",
            use_ohlc_average=False,
            cache_enabled=False,
            max_missing_data_pct=0.1,
        )
        loader = DataLoader(config=custom_config)
        assert loader.config.use_ohlc_average == False
        assert loader.config.cache_enabled == False
        assert loader.config.max_missing_data_pct == 0.1

    def test_validate_inputs_valid(self):
        """Test input validation with valid parameters."""
        # Should not raise any exceptions
        self.loader._validate_inputs("AAPL", "2023-01-01", "2023-12-31")

    def test_validate_inputs_invalid_ticker(self):
        """Test input validation with invalid ticker."""
        with pytest.raises(ValidationError, match="Ticker must be a non-empty string"):
            self.loader._validate_inputs("", self.start_date, self.end_date)

        with pytest.raises(ValidationError, match="Ticker must be a non-empty string"):
            self.loader._validate_inputs(None, self.start_date, self.end_date)

    def test_validate_inputs_invalid_dates(self):
        """Test input validation with invalid dates."""
        with pytest.raises(ValidationError, match="Invalid date format"):
            self.loader._validate_inputs(
                self.sample_ticker, "invalid-date", self.end_date
            )

        with pytest.raises(ValidationError, match="Start date must be before end date"):
            self.loader._validate_inputs(self.sample_ticker, "2023-12-31", "2023-01-01")

        future_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        with pytest.raises(ValidationError, match="End date cannot be in the future"):
            self.loader._validate_inputs(
                self.sample_ticker, self.start_date, future_date
            )

    def test_validate_inputs_short_period(self):
        """Test input validation with very short time period."""
        with pytest.raises(ValidationError, match="Minimum 7-day time period required"):
            self.loader._validate_inputs(self.sample_ticker, "2023-01-01", "2023-01-03")

    @patch("yfinance.Ticker")
    def test_load_from_yfinance_success(self, mock_ticker_class):
        """Test successful data loading from yfinance."""
        # Set up mock
        mock_ticker = MockYFinanceTicker("AAPL")
        mock_ticker_class.return_value = mock_ticker

        # Load data
        result = self.loader._load_from_yfinance(
            "AAPL", datetime(2023, 1, 1), datetime(2023, 1, 31)
        )

        # Verify result
        assert not result.empty
        assert "Close" in result.columns
        assert "Volume" in result.columns
        mock_ticker_class.assert_called_once_with("AAPL")

    @patch("yfinance.Ticker")
    def test_load_from_yfinance_empty_data(self, mock_ticker_class):
        """Test handling of empty data from yfinance."""
        # Set up mock to return empty DataFrame
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        with pytest.raises(DataLoadError, match="No data found for ticker"):
            self.loader._load_from_yfinance(
                "INVALID", datetime(2023, 1, 1), datetime(2023, 1, 31)
            )

    @patch("yfinance.Ticker")
    def test_load_from_yfinance_api_failure(self, mock_ticker_class):
        """Test handling of API failures."""
        # Set up mock to raise exception
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("API Error")
        mock_ticker_class.return_value = mock_ticker

        with pytest.raises(DataLoadError, match="Failed to load data for"):
            self.loader._load_from_yfinance(
                "AAPL", datetime(2023, 1, 1), datetime(2023, 1, 31)
            )

    def test_process_raw_data_ohlc_average(self):
        """Test processing raw data with OHLC average."""
        # Create mock yfinance data
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        raw_data = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [102, 103, 104, 105, 106],
                "Low": [99, 100, 101, 102, 103],
                "Close": [101, 102, 103, 104, 105],
                "Volume": [1000000] * 5,
            },
            index=dates,
        )
        raw_data.index.name = "Date"  # This makes reset_index() create 'Date' column

        result = self.loader._process_raw_data(raw_data, use_ohlc_avg=True)

        # Verify OHLC average calculation
        # Note: First row is dropped due to log return calculation, so index 0 = second row of raw data
        expected_price_0 = (101 + 103 + 100 + 102) / 4  # Second day = 101.5
        actual_price_0 = result["price"].iloc[0]
        assert np.isclose(
            actual_price_0, expected_price_0, rtol=1e-10
        ), f"Expected {expected_price_0}, got {actual_price_0}"

        # Verify other columns
        assert "date" in result.columns
        assert "log_return" in result.columns
        assert "volume" in result.columns
        assert len(result) == 4  # One row lost to log return calculation

    def test_process_raw_data_close_price(self):
        """Test processing raw data with close price only."""
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        raw_data = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "Close": [101, 102, 103, 104, 105],
                "Volume": [1000000] * 5,
            },
            index=dates,
        )
        raw_data.index.name = "Date"  # This makes reset_index() create 'Date' column

        result = self.loader._process_raw_data(raw_data, use_ohlc_avg=False)

        # Verify close price used
        # Note: First row is dropped due to log return calculation, so index 0 = second row of raw data
        actual_price_0 = result["price"].iloc[0]
        expected_price_0 = 102  # Second day close price
        assert np.isclose(
            actual_price_0, expected_price_0, rtol=1e-10
        ), f"Expected {expected_price_0}, got {actual_price_0}"
        assert "log_return" in result.columns
        assert len(result) == 4  # One row lost to log return calculation

    def test_validate_data_quality_success(self):
        """Test data quality validation with good data."""
        good_data = create_sample_stock_data(n_days=100, add_volume=True)

        # Should not raise any exceptions
        self.loader._validate_data_quality(good_data, "AAPL")

    def test_validate_data_quality_empty(self):
        """Test data quality validation with empty data."""
        empty_data = pd.DataFrame()

        with pytest.raises(DataLoadError, match="No data loaded"):
            self.loader._validate_data_quality(empty_data, "AAPL")

    def test_validate_data_quality_insufficient(self):
        """Test data quality validation with insufficient data."""
        small_data = create_sample_stock_data(n_days=10)  # Less than default minimum

        with pytest.raises(DataLoadError, match="Insufficient data"):
            self.loader._validate_data_quality(small_data, "AAPL")

    def test_validate_data_quality_excessive_missing(self):
        """Test data quality validation with excessive missing data."""
        data_with_missing = create_sample_stock_data(n_days=100)
        # Add lots of missing values
        data_with_missing.loc[:20, ["price", "log_return"]] = np.nan

        with pytest.raises(DataLoadError, match="Excessive missing data"):
            self.loader._validate_data_quality(data_with_missing, "AAPL")

    def test_validate_data_quality_invalid_prices(self):
        """Test data quality validation with invalid prices."""
        data_with_invalid = create_sample_stock_data(n_days=50)
        data_with_invalid.loc[10, "price"] = -5  # Negative price
        data_with_invalid.loc[20, "price"] = 0  # Zero price

        with pytest.raises(DataLoadError, match="Invalid price values"):
            self.loader._validate_data_quality(data_with_invalid, "AAPL")

    @patch("hidden_regime.data.loader.yf.Ticker")
    def test_load_stock_data_success(self, mock_ticker_class):
        """Test successful end-to-end data loading."""
        # Set up mock
        mock_ticker = MockYFinanceTicker("AAPL")
        mock_ticker_class.return_value = mock_ticker

        result = self.loader.load_stock_data("AAPL", "2023-01-01", "2023-03-31")

        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "date" in result.columns
        assert "price" in result.columns
        assert "log_return" in result.columns
        assert "volume" in result.columns

    @patch("hidden_regime.data.loader.yf.Ticker")
    def test_load_stock_data_caching(self, mock_ticker_class):
        """Test caching functionality."""
        # Set up mock
        mock_ticker = MockYFinanceTicker("AAPL")
        mock_ticker_class.return_value = mock_ticker

        # Load data twice
        result1 = self.loader.load_stock_data("AAPL", "2023-01-01", "2023-03-31")
        result2 = self.loader.load_stock_data("AAPL", "2023-01-01", "2023-03-31")

        # Mock should only be called once due to caching
        assert mock_ticker_class.call_count == 1

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_load_multiple_stocks_success(self):
        """Test loading multiple stocks."""
        with patch("hidden_regime.data.loader.yf.Ticker") as mock_ticker_class:
            # Set up mocks for different tickers
            def ticker_side_effect(ticker):
                return MockYFinanceTicker(ticker)

            mock_ticker_class.side_effect = ticker_side_effect

            tickers = ["AAPL", "GOOGL", "MSFT"]
            results = self.loader.load_multiple_stocks(
                tickers, "2023-01-01", "2023-03-31"
            )

            assert len(results) == 3
            for ticker in tickers:
                assert ticker in results
                assert isinstance(results[ticker], pd.DataFrame)
                assert not results[ticker].empty

    def test_load_multiple_stocks_partial_failure(self):
        """Test loading multiple stocks with some failures."""
        with patch("hidden_regime.data.loader.yf.Ticker") as mock_ticker_class:

            def ticker_side_effect(ticker):
                if ticker == "INVALID":
                    return MockYFinanceTicker(ticker, should_fail=True)
                return MockYFinanceTicker(ticker)

            mock_ticker_class.side_effect = ticker_side_effect

            tickers = ["AAPL", "INVALID", "GOOGL"]

            with warnings.catch_warnings(record=True) as w:
                results = self.loader.load_multiple_stocks(
                    tickers, "2023-01-01", "2023-03-31"
                )

            # Should have 2 successful results and 1 warning
            assert len(results) == 2
            assert "AAPL" in results
            assert "GOOGL" in results
            assert "INVALID" not in results
            assert len(w) >= 1  # At least one warning for the failed ticker

    def test_cache_management(self):
        """Test cache management methods."""
        # Add some dummy data to cache
        self.loader._cache["test_key"] = (pd.DataFrame(), datetime.now())

        # Test cache stats
        stats = self.loader.get_cache_stats()
        assert stats["cache_entries"] == 1
        assert stats["cache_enabled"] == True

        # Test cache clearing
        self.loader.clear_cache()
        stats_after_clear = self.loader.get_cache_stats()
        assert stats_after_clear["cache_entries"] == 0

    def test_unsupported_data_source(self):
        """Test handling of unsupported data sources."""
        with pytest.raises(DataLoadError, match="Unsupported data source"):
            self.loader.load_stock_data(
                "AAPL", "2023-01-01", "2023-03-31", source="unsupported_source"
            )

    def test_datetime_input_conversion(self):
        """Test that datetime inputs are properly handled."""
        start_dt = datetime(2023, 1, 1)
        end_dt = datetime(2023, 3, 31)

        with patch("hidden_regime.data.loader.yf.Ticker") as mock_ticker_class:
            mock_ticker = MockYFinanceTicker("AAPL")
            mock_ticker_class.return_value = mock_ticker

            result = self.loader.load_stock_data("AAPL", start_dt, end_dt)

            # Should work without errors
            assert isinstance(result, pd.DataFrame)
            assert not result.empty

    def test_config_override_parameters(self):
        """Test that method parameters override config defaults."""
        # Create loader with OHLC average disabled
        config = DataConfig(use_ohlc_average=False)
        loader = DataLoader(config)

        with patch("hidden_regime.data.loader.yf.Ticker") as mock_ticker_class:
            mock_ticker = MockYFinanceTicker("AAPL")
            mock_ticker_class.return_value = mock_ticker

            # Override to use OHLC average
            result = loader.load_stock_data(
                "AAPL", "2023-01-01", "2023-03-31", use_ohlc_avg=True
            )

            # Should work and use OHLC average despite config setting
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
