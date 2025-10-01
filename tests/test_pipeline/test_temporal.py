"""
Unit tests for TemporalController and TemporalDataStub.

Tests the temporal analysis functionality that enables V&V backtesting
with proper data isolation to prevent temporal data leakage.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from hidden_regime.pipeline import Pipeline
from hidden_regime.pipeline.temporal import TemporalController, TemporalDataStub
from hidden_regime.utils.exceptions import ValidationError


class TestTemporalDataStub:
    """Test cases for TemporalDataStub."""

    def create_sample_data(self, n_periods=100):
        """Create sample financial data for testing."""
        dates = pd.date_range("2024-01-01", periods=n_periods, freq="D")
        np.random.seed(42)

        close_prices = 100 + np.cumsum(np.random.normal(0, 1, n_periods))

        return pd.DataFrame(
            {
                "open": close_prices + np.random.normal(0, 0.5, n_periods),
                "high": close_prices + np.abs(np.random.normal(0, 1, n_periods)),
                "low": close_prices - np.abs(np.random.normal(0, 1, n_periods)),
                "close": close_prices,
                "volume": np.random.randint(1000000, 10000000, n_periods),
                "price": close_prices,  # Add price column for plotting
                "log_return": pd.Series(
                    np.log(close_prices)
                ).diff(),  # Add log_return for plotting
            },
            index=dates,
        )

    def test_temporal_data_stub_initialization(self):
        """Test TemporalDataStub initialization."""
        data = self.create_sample_data(50)
        stub = TemporalDataStub(data)

        assert stub.filtered_data is not None
        assert len(stub.filtered_data) == 50
        assert stub.creation_time is not None

        # Should be a copy, not the same object
        assert stub.filtered_data is not data

    def test_get_all_data_returns_copy(self):
        """Test that get_all_data returns a copy of filtered data."""
        data = self.create_sample_data(30)
        stub = TemporalDataStub(data)

        result1 = stub.get_all_data()
        result2 = stub.get_all_data()

        # Should be equal but different objects
        pd.testing.assert_frame_equal(result1, result2)
        assert result1 is not result2
        assert result1 is not stub.filtered_data

    def test_update_ignores_current_date(self):
        """Test that update method ignores current_date parameter."""
        data = self.create_sample_data(40)
        stub = TemporalDataStub(data)

        # Should return same data regardless of current_date
        result1 = stub.update()
        result2 = stub.update(current_date="2024-12-31")
        result3 = stub.update(current_date="2025-01-01")

        pd.testing.assert_frame_equal(result1, result2)
        pd.testing.assert_frame_equal(result2, result3)

    def test_temporal_data_isolation(self):
        """Test that data stub prevents access to future data."""
        # Create full dataset
        full_data = self.create_sample_data(100)

        # Create filtered data (only first 50 days)
        filtered_data = full_data.iloc[:50]
        stub = TemporalDataStub(filtered_data)

        # Should only have access to filtered portion
        result = stub.get_all_data()
        assert len(result) == 50
        assert result.index.max() == filtered_data.index.max()

        # Should not have access to future data
        future_dates = full_data.iloc[50:].index
        assert not any(date in result.index for date in future_dates)

    def test_plot_functionality(self):
        """Test plotting functionality."""
        data = self.create_sample_data(60)
        stub = TemporalDataStub(data)

        fig = stub.plot()
        assert fig is not None
        assert len(fig.axes) >= 1


class TestTemporalController:
    """Test cases for TemporalController."""

    def create_mock_pipeline(self):
        """Create a mock pipeline for testing."""
        pipeline = Mock()
        pipeline.update = Mock(return_value="Analysis Result")
        pipeline.data = Mock()
        return pipeline

    def create_sample_dataset(self, n_periods=200):
        """Create sample dataset for temporal analysis."""
        dates = pd.date_range("2024-01-01", periods=n_periods, freq="D")
        np.random.seed(42)

        close_prices = 100 + np.cumsum(np.random.normal(0, 1, n_periods))

        return pd.DataFrame(
            {
                "open": close_prices + np.random.normal(0, 0.5, n_periods),
                "high": close_prices + np.abs(np.random.normal(0, 1, n_periods)),
                "low": close_prices - np.abs(np.random.normal(0, 1, n_periods)),
                "close": close_prices,
                "volume": np.random.randint(1000000, 10000000, n_periods),
                "price": close_prices,
                "log_return": pd.Series(np.log(close_prices)).diff(),
            },
            index=dates,
        )

    def test_temporal_controller_initialization(self):
        """Test TemporalController initialization."""
        pipeline = self.create_mock_pipeline()
        dataset = self.create_sample_dataset(100)

        controller = TemporalController(pipeline, dataset)

        assert controller.pipeline is pipeline
        assert controller.full_dataset is not None
        assert len(controller.full_dataset) == 100
        assert controller.access_log == []
        assert controller.original_data is None

    def test_temporal_controller_invalid_index(self):
        """Test TemporalController with invalid index."""
        pipeline = self.create_mock_pipeline()

        # Dataset without DatetimeIndex
        invalid_dataset = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104]}
        )  # No DatetimeIndex

        with pytest.raises(ValueError, match="Dataset must have DatetimeIndex"):
            TemporalController(pipeline, invalid_dataset)

    def test_update_as_of_date(self):
        """Test updating as of a specific date."""
        pipeline = self.create_mock_pipeline()
        dataset = self.create_sample_dataset(100)

        controller = TemporalController(pipeline, dataset)

        # Update as of a date in the middle of the dataset
        target_date = "2024-02-15"
        result = controller.update_as_of(target_date)

        # Should have called pipeline.update()
        pipeline.update.assert_called_once()

        # Should have logged the access
        assert len(controller.access_log) == 1
        log_entry = controller.access_log[0]
        assert log_entry["as_of_date"] == target_date
        assert log_entry["data_end"] <= pd.to_datetime(target_date)

        # Should return the pipeline result
        assert result == "Analysis Result"

    def test_update_as_of_date_no_data(self):
        """Test updating as of date with no available data."""
        pipeline = self.create_mock_pipeline()
        dataset = self.create_sample_dataset(100)

        controller = TemporalController(pipeline, dataset)

        # Try to update as of date before dataset starts
        early_date = "2023-01-01"

        with pytest.raises(ValueError, match="No data available up to"):
            controller.update_as_of(early_date)

    def test_step_through_time_daily(self):
        """Test stepping through time with daily frequency."""
        pipeline = self.create_mock_pipeline()
        dataset = self.create_sample_dataset(30)

        controller = TemporalController(pipeline, dataset)

        start_date = "2024-01-05"
        end_date = "2024-01-15"

        results = controller.step_through_time(
            start_date=start_date, end_date=end_date, freq="D"
        )

        assert isinstance(results, list)
        assert len(results) > 0

        # Each result should be a tuple of (date, report)
        for date, report in results:
            assert isinstance(date, str)
            assert report == "Analysis Result"

        # Should have multiple access log entries
        assert len(controller.access_log) == len(results)

    def test_step_through_time_weekly(self):
        """Test stepping through time with weekly frequency."""
        pipeline = self.create_mock_pipeline()
        dataset = self.create_sample_dataset(60)

        controller = TemporalController(pipeline, dataset)

        start_date = "2024-01-01"
        end_date = "2024-01-31"

        results = controller.step_through_time(
            start_date=start_date, end_date=end_date, freq="W"  # Weekly frequency
        )

        assert isinstance(results, list)

        # Weekly should have fewer results than daily for same period
        daily_results = controller.step_through_time(
            start_date=start_date, end_date=end_date, freq="D"
        )

        assert len(results) <= len(daily_results)

    def test_temporal_isolation_enforcement(self):
        """Test that temporal isolation prevents data leakage."""
        pipeline = self.create_mock_pipeline()
        dataset = self.create_sample_dataset(100)

        controller = TemporalController(pipeline, dataset)

        # Store original data reference
        original_data = pipeline.data

        # Update as of early date
        early_date = "2024-01-15"
        controller.update_as_of(early_date)

        # Pipeline's data should have been restored after update
        assert pipeline.data is original_data

        # Access log should show temporal boundary was enforced
        log_entry = controller.access_log[0]
        assert log_entry["data_end"] <= pd.to_datetime(early_date)
        assert log_entry["num_observations"] < len(dataset)

    def test_data_component_restoration(self):
        """Test that original data component is restored after updates."""
        pipeline = self.create_mock_pipeline()
        dataset = self.create_sample_dataset(50)

        controller = TemporalController(pipeline, dataset)
        original_data = pipeline.data

        # Update as of date
        controller.update_as_of("2024-01-20")

        # Data component should be restored
        assert pipeline.data is original_data
        assert controller.original_data is not None

    def test_audit_trail_completeness(self):
        """Test that complete audit trail is maintained."""
        pipeline = self.create_mock_pipeline()
        dataset = self.create_sample_dataset(40)

        controller = TemporalController(pipeline, dataset)

        # Perform multiple updates
        dates = ["2024-01-10", "2024-01-15", "2024-01-20"]
        for date in dates:
            controller.update_as_of(date)

        # Should have logged all accesses
        assert len(controller.access_log) == len(dates)

        # Each log entry should have required fields
        for i, log_entry in enumerate(controller.access_log):
            assert "timestamp" in log_entry
            assert "as_of_date" in log_entry
            assert "data_start" in log_entry
            assert "data_end" in log_entry
            assert "num_observations" in log_entry
            assert "total_dataset_size" in log_entry
            assert "data_coverage" in log_entry

            assert log_entry["as_of_date"] == dates[i]
            assert log_entry["total_dataset_size"] == len(dataset)

    def test_exception_handling_with_restoration(self):
        """Test that data component is restored even if pipeline update fails."""
        pipeline = self.create_mock_pipeline()
        dataset = self.create_sample_dataset(50)

        # Make pipeline.update raise an exception
        pipeline.update.side_effect = Exception("Pipeline failed")

        controller = TemporalController(pipeline, dataset)
        original_data = pipeline.data

        # Update should fail but still restore data component
        with pytest.raises(Exception, match="Pipeline failed"):
            controller.update_as_of("2024-01-15")

        # Data component should still be restored
        assert pipeline.data is original_data

    def test_chronological_ordering(self):
        """Test that dataset is sorted chronologically."""
        pipeline = self.create_mock_pipeline()

        # Create unordered dataset
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        shuffled_dates = dates.to_series().sample(frac=1).index  # Shuffle dates

        dataset = pd.DataFrame({"close": np.random.randn(20)}, index=shuffled_dates)

        controller = TemporalController(pipeline, dataset)

        # Dataset should be sorted in controller
        assert controller.full_dataset.index.is_monotonic_increasing

    def test_performance_with_large_dataset(self):
        """Test performance with larger dataset."""
        import time

        pipeline = self.create_mock_pipeline()
        dataset = self.create_sample_dataset(365)  # One year of data

        controller = TemporalController(pipeline, dataset)

        start_time = time.time()

        results = controller.step_through_time(
            start_date="2024-01-01",
            end_date="2024-01-31",
            freq="W",  # Weekly to keep test fast
        )

        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 10.0  # Less than 10 seconds
        assert len(results) > 0

    def test_data_coverage_calculation(self):
        """Test data coverage calculation in audit log."""
        pipeline = self.create_mock_pipeline()
        dataset = self.create_sample_dataset(100)

        controller = TemporalController(pipeline, dataset)

        # Update as of middle date
        controller.update_as_of("2024-02-15")  # Roughly day 46

        log_entry = controller.access_log[0]

        # Coverage should be less than 1.0 since we're not using all data
        assert 0 < log_entry["data_coverage"] < 1.0

        # Coverage should equal filtered data size / total size
        expected_coverage = (
            log_entry["num_observations"] / log_entry["total_dataset_size"]
        )
        assert abs(log_entry["data_coverage"] - expected_coverage) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__])
