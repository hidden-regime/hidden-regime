"""
Test DatetimeIndex consistency and timezone handling in FinancialDataLoader.

This test suite verifies:
1. num_samples and start_date configs produce same index type (DatetimeIndex)
2. Timezone-aware DatetimeIndex works correctly
3. Timezone-naive DatetimeIndex works correctly
4. No conflicts between timezone-aware and timezone-naive operations
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.data.financial import FinancialDataLoader


class TestDatetimeIndexConsistency:
    """Test that FinancialDataLoader preserves DatetimeIndex consistently."""

    def create_mock_yfinance_data(
        self, n_days: int = 250, timezone_aware: bool = True
    ) -> pd.DataFrame:
        """Create realistic mock yfinance data with timezone handling."""
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")

        # yfinance returns timezone-aware data by default (America/New_York)
        if timezone_aware:
            dates = dates.tz_localize("America/New_York")

        data = pd.DataFrame(
            {
                "Open": np.random.uniform(90, 110, n_days),
                "High": np.random.uniform(95, 115, n_days),
                "Low": np.random.uniform(85, 105, n_days),
                "Close": np.random.uniform(90, 110, n_days),
                "Volume": np.random.randint(1000000, 10000000, n_days),
            },
            index=dates,
        )

        # Ensure OHLC relationships are valid
        data["High"] = data[["High", "Open", "Close"]].max(axis=1)
        data["Low"] = data[["Low", "Open", "Close"]].min(axis=1)

        return data

    @patch("hidden_regime.data.financial.yf.Ticker")
    def test_num_samples_preserves_datetime_index(self, mock_ticker_class):
        """Test that num_samples config preserves DatetimeIndex (not RangeIndex)."""
        # Setup mock
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_data = self.create_mock_yfinance_data(n_days=250, timezone_aware=True)
        mock_ticker.history.return_value = mock_data

        # Test with num_samples
        config = FinancialDataConfig(ticker="SPY", num_samples=100)
        loader = FinancialDataLoader(config)
        result = loader.load_data()

        # Verify DatetimeIndex is preserved
        assert isinstance(
            result.index, pd.DatetimeIndex
        ), f"Expected DatetimeIndex, got {type(result.index)}"
        assert len(result) == 100
        # Verify timezone is preserved
        assert result.index.tz is not None, "Timezone should be preserved"

    @patch("hidden_regime.data.financial.yf.Ticker")
    def test_start_date_preserves_datetime_index(self, mock_ticker_class):
        """Test that start_date config preserves DatetimeIndex."""
        # Setup mock
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_data = self.create_mock_yfinance_data(n_days=250, timezone_aware=True)
        mock_ticker.history.return_value = mock_data

        # Test with start_date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        config = FinancialDataConfig(
            ticker="SPY", start_date=start_date.strftime("%Y-%m-%d")
        )
        loader = FinancialDataLoader(config)
        result = loader.load_data()

        # Verify DatetimeIndex is preserved
        assert isinstance(
            result.index, pd.DatetimeIndex
        ), f"Expected DatetimeIndex, got {type(result.index)}"
        # Verify timezone is preserved
        assert result.index.tz is not None, "Timezone should be preserved"

    @patch("hidden_regime.data.financial.yf.Ticker")
    def test_index_type_consistency_between_configs(self, mock_ticker_class):
        """Test that num_samples and start_date produce the same index type."""
        # Setup mock
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_data = self.create_mock_yfinance_data(n_days=250, timezone_aware=True)
        mock_ticker.history.return_value = mock_data

        # Config 1: Using num_samples
        config_num = FinancialDataConfig(ticker="SPY", num_samples=100)
        loader_num = FinancialDataLoader(config_num)
        result_num = loader_num.load_data()

        # Config 2: Using start_date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        config_date = FinancialDataConfig(
            ticker="SPY", start_date=start_date.strftime("%Y-%m-%d")
        )
        loader_date = FinancialDataLoader(config_date)
        result_date = loader_date.load_data()

        # Both should have DatetimeIndex
        assert type(result_num.index) == type(result_date.index), (
            f"Index types should match: "
            f"num_samples={type(result_num.index)}, "
            f"start_date={type(result_date.index)}"
        )
        assert isinstance(result_num.index, pd.DatetimeIndex)
        assert isinstance(result_date.index, pd.DatetimeIndex)

        # Both should preserve timezone
        assert result_num.index.tz == result_date.index.tz


class TestTimezoneHandling:
    """Test timezone-aware and timezone-naive data handling."""

    def create_mock_yfinance_data(
        self, n_days: int = 250, timezone_aware: bool = True
    ) -> pd.DataFrame:
        """Create realistic mock yfinance data with timezone handling."""
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")

        if timezone_aware:
            dates = dates.tz_localize("America/New_York")

        data = pd.DataFrame(
            {
                "Open": np.random.uniform(90, 110, n_days),
                "High": np.random.uniform(95, 115, n_days),
                "Low": np.random.uniform(85, 105, n_days),
                "Close": np.random.uniform(90, 110, n_days),
                "Volume": np.random.randint(1000000, 10000000, n_days),
            },
            index=dates,
        )

        data["High"] = data[["High", "Open", "Close"]].max(axis=1)
        data["Low"] = data[["Low", "Open", "Close"]].min(axis=1)

        return data

    @patch("hidden_regime.data.financial.yf.Ticker")
    def test_timezone_aware_data_handling(self, mock_ticker_class):
        """Test that timezone-aware data is handled correctly."""
        # Setup mock with timezone-aware data
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_data = self.create_mock_yfinance_data(n_days=250, timezone_aware=True)
        mock_ticker.history.return_value = mock_data

        config = FinancialDataConfig(ticker="SPY", num_samples=100)
        loader = FinancialDataLoader(config)
        result = loader.load_data()

        # Verify timezone is preserved
        assert result.index.tz is not None, "Timezone should be preserved"
        assert str(result.index.tz) == "America/New_York"

        # Verify data is valid
        assert len(result) == 100
        assert "price" in result.columns
        assert "log_return" in result.columns
        assert not result["price"].isna().any()

    @patch("hidden_regime.data.financial.yf.Ticker")
    def test_timezone_naive_data_handling(self, mock_ticker_class):
        """Test that timezone-naive data is handled correctly."""
        # Setup mock with timezone-naive data
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_data = self.create_mock_yfinance_data(n_days=250, timezone_aware=False)
        mock_ticker.history.return_value = mock_data

        config = FinancialDataConfig(ticker="SPY", num_samples=100)
        loader = FinancialDataLoader(config)
        result = loader.load_data()

        # Verify timezone is not present
        assert result.index.tz is None, "Timezone-naive should remain naive"

        # Verify data is valid
        assert len(result) == 100
        assert "price" in result.columns
        assert "log_return" in result.columns
        assert not result["price"].isna().any()

    @patch("hidden_regime.data.financial.yf.Ticker")
    def test_datetime_operations_work_with_both_types(self, mock_ticker_class):
        """Test that datetime operations work with both timezone types."""
        # Setup mock
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        # Test timezone-aware
        mock_data_aware = self.create_mock_yfinance_data(
            n_days=250, timezone_aware=True
        )
        mock_ticker.history.return_value = mock_data_aware

        config = FinancialDataConfig(ticker="SPY", num_samples=100)
        loader = FinancialDataLoader(config)
        result_aware = loader.load_data()

        # Datetime operations should work
        first_date = result_aware.index[0]
        last_date = result_aware.index[-1]
        time_range = last_date - first_date
        assert time_range.days >= 0

        # Test timezone-naive
        mock_data_naive = self.create_mock_yfinance_data(
            n_days=250, timezone_aware=False
        )
        mock_ticker.history.return_value = mock_data_naive

        config2 = FinancialDataConfig(ticker="SPY", num_samples=100)
        loader2 = FinancialDataLoader(config2)
        result_naive = loader2.load_data()

        # Datetime operations should work
        first_date = result_naive.index[0]
        last_date = result_naive.index[-1]
        time_range = last_date - first_date
        assert time_range.days >= 0


class TestDatetimeIndexIntegration:
    """Integration tests for DatetimeIndex with other components."""

    def create_mock_yfinance_data(
        self, n_days: int = 250, timezone_aware: bool = True
    ) -> pd.DataFrame:
        """Create realistic mock yfinance data."""
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")

        if timezone_aware:
            dates = dates.tz_localize("America/New_York")

        data = pd.DataFrame(
            {
                "Open": np.random.uniform(90, 110, n_days),
                "High": np.random.uniform(95, 115, n_days),
                "Low": np.random.uniform(85, 105, n_days),
                "Close": np.random.uniform(90, 110, n_days),
                "Volume": np.random.randint(1000000, 10000000, n_days),
            },
            index=dates,
        )

        data["High"] = data[["High", "Open", "Close"]].max(axis=1)
        data["Low"] = data[["Low", "Open", "Close"]].min(axis=1)

        return data

    @patch("hidden_regime.data.financial.yf.Ticker")
    def test_slicing_with_datetime_index(self, mock_ticker_class):
        """Test that slicing operations work with DatetimeIndex."""
        # Setup mock
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_data = self.create_mock_yfinance_data(n_days=250, timezone_aware=True)
        mock_ticker.history.return_value = mock_data

        config = FinancialDataConfig(ticker="SPY", num_samples=100)
        loader = FinancialDataLoader(config)
        result = loader.load_data()

        # Test date-based slicing
        mid_date = result.index[len(result) // 2]
        subset = result.loc[:mid_date]

        assert len(subset) > 0
        assert len(subset) <= len(result)
        assert isinstance(subset.index, pd.DatetimeIndex)

    @patch("hidden_regime.data.financial.yf.Ticker")
    def test_merging_with_datetime_index(self, mock_ticker_class):
        """Test that merging operations work with DatetimeIndex."""
        # Setup mock
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_data = self.create_mock_yfinance_data(n_days=250, timezone_aware=True)
        mock_ticker.history.return_value = mock_data

        config = FinancialDataConfig(ticker="SPY", num_samples=100)
        loader = FinancialDataLoader(config)
        result = loader.load_data()

        # Create another dataframe with overlapping dates
        new_data = pd.DataFrame(
            {"new_column": np.random.random(len(result))}, index=result.index
        )

        # Merge should work
        merged = pd.merge(
            result, new_data, left_index=True, right_index=True, how="inner"
        )

        assert "new_column" in merged.columns
        assert len(merged) == len(result)
        assert isinstance(merged.index, pd.DatetimeIndex)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
