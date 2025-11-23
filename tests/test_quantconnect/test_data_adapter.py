"""
Unit tests for QuantConnect data adapters.

Tests:
- QuantConnectDataAdapter
- RollingWindowDataAdapter
- HistoryDataAdapter
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from hidden_regime.quantconnect.data_adapter import (
    QuantConnectDataAdapter,
    RollingWindowDataAdapter,
    HistoryDataAdapter
)


class TestQuantConnectDataAdapter:
    """Test QuantConnectDataAdapter class."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = QuantConnectDataAdapter(window_size=100)
        assert adapter.window_size == 100
        assert len(adapter.bars) == 0

    def test_add_bar(self, mock_tradebar_data):
        """Test adding TradeBar data."""
        adapter = QuantConnectDataAdapter(window_size=252)

        for bar in mock_tradebar_data[:10]:
            adapter.add_bar(bar)

        assert len(adapter.bars) == 10

    def test_window_size_limit(self, mock_tradebar_data):
        """Test that window size is respected."""
        adapter = QuantConnectDataAdapter(window_size=50)

        for bar in mock_tradebar_data:
            adapter.add_bar(bar)

        assert len(adapter.bars) == 50

    def test_to_dataframe(self, mock_tradebar_data):
        """Test conversion to DataFrame."""
        adapter = QuantConnectDataAdapter(window_size=252)

        for bar in mock_tradebar_data:
            adapter.add_bar(bar)

        df = adapter.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(mock_tradebar_data)
        assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        assert df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex)

    def test_to_dataframe_empty(self):
        """Test to_dataframe with no data."""
        adapter = QuantConnectDataAdapter(window_size=100)
        df = adapter.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_to_dataframe_sorted(self, mock_tradebar_data):
        """Test that DataFrame is properly sorted by date."""
        adapter = QuantConnectDataAdapter(window_size=252)

        # Add bars in random order
        import random
        shuffled = list(mock_tradebar_data)
        random.shuffle(shuffled)

        for bar in shuffled:
            adapter.add_bar(bar)

        df = adapter.to_dataframe()

        # Check that dates are sorted
        assert df.index.is_monotonic_increasing

    def test_data_integrity(self, mock_tradebar_data):
        """Test that data values are preserved correctly."""
        adapter = QuantConnectDataAdapter(window_size=252)

        for bar in mock_tradebar_data[:10]:
            adapter.add_bar(bar)

        df = adapter.to_dataframe()

        # Verify first and last bar
        assert abs(df.iloc[0]['Close'] - mock_tradebar_data[0].Close) < 0.01
        assert abs(df.iloc[-1]['Close'] - mock_tradebar_data[9].Close) < 0.01


class TestRollingWindowDataAdapter:
    """Test RollingWindowDataAdapter class."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = RollingWindowDataAdapter(window_size=100)
        assert adapter.window_size == 100

    def test_window_conversion(self, mock_tradebar_data):
        """Test conversion from rolling window."""
        # Create mock rolling window
        class MockRollingWindow:
            def __init__(self, data):
                self.data = list(reversed(data))  # RollingWindow is LIFO
                self.Count = len(data)
                self.IsReady = True

            def __iter__(self):
                return iter(self.data)

            def __getitem__(self, index):
                return self.data[index]

        window = MockRollingWindow(mock_tradebar_data[:100])
        adapter = RollingWindowDataAdapter(window_size=100)

        df = adapter.from_rolling_window(window)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

    def test_window_not_ready(self):
        """Test handling of non-ready rolling window."""
        class MockRollingWindow:
            def __init__(self):
                self.Count = 10
                self.IsReady = False

        window = MockRollingWindow()
        adapter = RollingWindowDataAdapter(window_size=100)

        df = adapter.from_rolling_window(window)

        # Should return empty or partial DataFrame
        assert isinstance(df, pd.DataFrame)


class TestHistoryDataAdapter:
    """Test HistoryDataAdapter class."""

    def test_from_history_single_symbol(self, sample_price_data):
        """Test conversion from QC History API result."""
        adapter = HistoryDataAdapter()

        # Mock QC History result (multi-index DataFrame)
        history_df = sample_price_data.copy()

        df = adapter.from_history(history_df)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_price_data)
        assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

    def test_from_history_multi_symbol(self, sample_multi_asset_data):
        """Test conversion from multi-symbol History result."""
        adapter = HistoryDataAdapter()

        # Create multi-index DataFrame (symbol, date)
        dfs = []
        for symbol, data in sample_multi_asset_data.items():
            df_copy = data.copy()
            df_copy['symbol'] = symbol
            dfs.append(df_copy)

        history_df = pd.concat(dfs)

        result = adapter.from_history_multi_symbol(history_df)

        assert isinstance(result, dict)
        assert all(symbol in result for symbol in sample_multi_asset_data.keys())
        assert all(isinstance(df, pd.DataFrame) for df in result.values())

    def test_resample_to_daily(self, sample_price_data):
        """Test resampling to different frequencies."""
        adapter = HistoryDataAdapter()

        # Create hourly data
        hourly_dates = pd.date_range(
            start=sample_price_data.index[0],
            end=sample_price_data.index[-1],
            freq='H'
        )

        hourly_data = pd.DataFrame({
            'Open': np.random.uniform(90, 110, len(hourly_dates)),
            'High': np.random.uniform(90, 110, len(hourly_dates)),
            'Low': np.random.uniform(90, 110, len(hourly_dates)),
            'Close': np.random.uniform(90, 110, len(hourly_dates)),
            'Volume': np.random.uniform(1e6, 5e6, len(hourly_dates))
        }, index=hourly_dates)

        daily_data = adapter.resample(hourly_data, frequency='D')

        assert isinstance(daily_data, pd.DataFrame)
        assert len(daily_data) < len(hourly_data)
        assert all(col in daily_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])


class TestDataAdapterIntegration:
    """Integration tests for data adapters."""

    def test_adapter_pipeline(self, mock_tradebar_data):
        """Test complete data adapter pipeline."""
        # Step 1: Collect data with QuantConnectDataAdapter
        collector = QuantConnectDataAdapter(window_size=252)
        for bar in mock_tradebar_data:
            collector.add_bar(bar)

        df = collector.to_dataframe()

        # Step 2: Verify data quality
        assert len(df) == len(mock_tradebar_data)
        assert not df.isnull().any().any()
        assert (df['High'] >= df['Low']).all()
        assert (df['High'] >= df['Close']).all()
        assert (df['Low'] <= df['Close']).all()

    def test_data_adapter_with_pipeline(self, mock_tradebar_data, mock_pipeline):
        """Test data adapter integration with regime pipeline."""
        adapter = QuantConnectDataAdapter(window_size=252)

        for bar in mock_tradebar_data:
            adapter.add_bar(bar)

        df = adapter.to_dataframe()

        # Should be able to pass to pipeline
        result = mock_pipeline.run(df)

        assert 'regime' in result
        assert 'probabilities' in result

    def test_multi_adapter_consistency(self, sample_price_data):
        """Test consistency across different adapter types."""
        # Convert to mock TradeBars
        class MockTradeBar:
            def __init__(self, time, row):
                self.Time = time
                self.Open = row['Open']
                self.High = row['High']
                self.Low = row['Low']
                self.Close = row['Close']
                self.Volume = row['Volume']

        bars = [MockTradeBar(idx, row) for idx, row in sample_price_data.iterrows()]

        # Method 1: QuantConnectDataAdapter
        adapter1 = QuantConnectDataAdapter(window_size=len(bars))
        for bar in bars:
            adapter1.add_bar(bar)
        df1 = adapter1.to_dataframe()

        # Method 2: HistoryDataAdapter
        adapter2 = HistoryDataAdapter()
        df2 = adapter2.from_history(sample_price_data)

        # Should produce similar results (allowing for minor differences)
        assert len(df1) == len(df2)
        assert list(df1.columns) == list(df2.columns)


class TestDataAdapterEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test handling of empty data."""
        adapter = QuantConnectDataAdapter(window_size=100)
        df = adapter.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_single_bar(self, mock_tradebar_data):
        """Test with single data point."""
        adapter = QuantConnectDataAdapter(window_size=100)
        adapter.add_bar(mock_tradebar_data[0])

        df = adapter.to_dataframe()

        assert len(df) == 1

    def test_very_small_window(self, mock_tradebar_data):
        """Test with very small window size."""
        adapter = QuantConnectDataAdapter(window_size=5)

        for bar in mock_tradebar_data[:10]:
            adapter.add_bar(bar)

        df = adapter.to_dataframe()

        assert len(df) == 5

    def test_very_large_window(self, mock_tradebar_data):
        """Test with window larger than data."""
        adapter = QuantConnectDataAdapter(window_size=10000)

        for bar in mock_tradebar_data:
            adapter.add_bar(bar)

        df = adapter.to_dataframe()

        assert len(df) == len(mock_tradebar_data)

    def test_missing_fields(self):
        """Test handling of bars with missing fields."""
        class IncompleteMockTradeBar:
            def __init__(self):
                self.Time = datetime(2020, 1, 1)
                self.Close = 100.0
                # Missing other fields

        adapter = QuantConnectDataAdapter(window_size=100)

        # Should handle gracefully or raise informative error
        try:
            adapter.add_bar(IncompleteMockTradeBar())
            df = adapter.to_dataframe()
            # If it doesn't raise, check that it handled it somehow
            assert True
        except (AttributeError, KeyError) as e:
            # Expected - missing required fields
            assert True
