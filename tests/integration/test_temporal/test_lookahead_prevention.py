"""
Temporal isolation tests - Look-ahead bias prevention (CRITICAL).

These tests are ESSENTIAL for backtest validity. If future data leaks into
model training or predictions, all backtest results are invalid.

Tests verify that:
1. Features never use future data
2. Model training only uses past data
3. Predictions only use past data
4. Date boundaries are strictly enforced
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from hidden_regime.pipeline.temporal import TemporalController, TemporalDataStub
from hidden_regime.pipeline.core import Pipeline


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def full_dataset():
    """Generate 2 years of daily OHLC data for temporal testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
    n = len(dates)

    # Generate realistic price data with trend
    returns = np.random.randn(n) * 0.02 + 0.0002  # Slight upward drift
    close = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'Open': close * (1 + np.random.randn(n) * 0.005),
        'High': close * (1 + np.abs(np.random.randn(n) * 0.01)),
        'Low': close * (1 - np.abs(np.random.randn(n) * 0.01)),
        'Close': close,
        'Volume': np.random.randint(1_000_000, 10_000_000, n),
        'price': close,  # For TemporalDataStub compatibility
    }, index=dates)

    # Add log returns
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df.iloc[0, df.columns.get_loc('log_return')] = 0

    return df


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing."""
    pipeline = Mock(spec=Pipeline)
    pipeline.data = Mock()
    pipeline.update = Mock(return_value="Mock Analysis Report")
    return pipeline


# ============================================================================
# TemporalDataStub Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.unit
def test_temporal_data_stub_only_returns_filtered_data(full_dataset):
    """Test that TemporalDataStub only provides pre-filtered data.

    The stub should NEVER provide access to data beyond the filter boundary,
    regardless of what parameters are passed to its methods.
    """
    # Filter to first 100 days
    filtered = full_dataset.iloc[:100]
    stub = TemporalDataStub(filtered)

    # Verify get_all_data returns only filtered data
    data = stub.get_all_data()
    assert len(data) == 100, "Stub should return exactly 100 days"
    assert data.index.max() == filtered.index.max(), "Max date should be filter boundary"
    assert data.index.min() == filtered.index.min(), "Min date should be unchanged"


@pytest.mark.integration
@pytest.mark.unit
def test_temporal_data_stub_update_ignores_current_date(full_dataset):
    """Test that update() ignores current_date parameter.

    This prevents any attempt to request future data by passing a later date.
    """
    filtered = full_dataset.iloc[:100]
    stub = TemporalDataStub(filtered)

    # Try to trick stub with future date
    future_date = full_dataset.index[200].strftime('%Y-%m-%d')
    data = stub.update(current_date=future_date)

    # Should still return only filtered data
    assert len(data) == 100, "Stub should ignore future date request"
    assert data.index.max() == filtered.index.max(), "Should not provide future data"


@pytest.mark.integration
@pytest.mark.unit
def test_temporal_data_stub_returns_copy_not_reference(full_dataset):
    """Test that stub returns copies, not references.

    This prevents external code from modifying the filtered dataset.
    """
    filtered = full_dataset.iloc[:100]
    stub = TemporalDataStub(filtered)

    # Get data and modify it
    data1 = stub.get_all_data()
    data1['Close'] = 999.0

    # Get data again
    data2 = stub.get_all_data()

    # Modifications should not propagate
    assert not (data2['Close'] == 999.0).all(), "Should return independent copies"


# ============================================================================
# TemporalController - Boundary Enforcement Tests
# ============================================================================


@pytest.mark.integration
def test_temporal_controller_strict_date_boundary(full_dataset, mock_pipeline):
    """Test that TemporalController strictly enforces date boundaries.

    CRITICAL: Model must never see data after as_of_date.
    """
    controller = TemporalController(
        pipeline=mock_pipeline,
        full_dataset=full_dataset,
        enable_data_collection=False
    )

    # Update as of mid-2020
    as_of_date = '2020-06-30'
    controller.update_as_of(as_of_date)

    # Verify pipeline received filtered data
    assert mock_pipeline.data is not None
    filtered_data = mock_pipeline.data.get_all_data()

    # Critical assertions
    max_date = pd.to_datetime(as_of_date)
    assert filtered_data.index.max() <= max_date, \
        f"Data contains future dates! Max: {filtered_data.index.max()}, Boundary: {max_date}"
    assert len(filtered_data) < len(full_dataset), \
        "Filtered data should be smaller than full dataset"


@pytest.mark.integration
def test_temporal_controller_no_future_data_in_features(full_dataset, mock_pipeline):
    """Test that feature calculations never use future data.

    When analyzing at time T, features must only use data before T.
    """
    controller = TemporalController(
        pipeline=mock_pipeline,
        full_dataset=full_dataset,
        enable_data_collection=False
    )

    # Analyze at specific date
    analysis_date = '2020-09-15'
    controller.update_as_of(analysis_date)

    # Get the data that was provided to pipeline
    provided_data = mock_pipeline.data.get_all_data()

    # Verify no future data
    assert provided_data.index.max() <= pd.to_datetime(analysis_date), \
        "Features must not use future data"

    # Verify data is substantial enough for analysis
    assert len(provided_data) > 30, "Should have enough historical data"


@pytest.mark.integration
def test_temporal_controller_preserves_chronological_order(full_dataset, mock_pipeline):
    """Test that temporal filtering preserves chronological order.

    Data must remain in time-sorted order for correct feature calculation.
    """
    controller = TemporalController(
        pipeline=mock_pipeline,
        full_dataset=full_dataset,
        enable_data_collection=False
    )

    controller.update_as_of('2020-12-31')
    data = mock_pipeline.data.get_all_data()

    # Verify chronological order
    assert data.index.is_monotonic_increasing, "Data must be chronologically sorted"

    # Verify no duplicate dates
    assert not data.index.has_duplicates, "Should not have duplicate timestamps"


@pytest.mark.integration
def test_temporal_controller_audit_log_created(full_dataset, mock_pipeline):
    """Test that controller maintains complete audit trail.

    Every data access should be logged for V&V compliance.
    """
    controller = TemporalController(
        pipeline=mock_pipeline,
        full_dataset=full_dataset,
        enable_data_collection=False
    )

    # Perform multiple updates
    dates = ['2020-03-31', '2020-06-30', '2020-09-30']
    for date in dates:
        controller.update_as_of(date)

    # Verify audit log
    assert len(controller.access_log) == 3, "Should log each access"

    for i, log_entry in enumerate(controller.access_log):
        assert 'as_of_date' in log_entry
        assert 'data_end' in log_entry
        assert 'num_observations' in log_entry
        assert log_entry['as_of_date'] == dates[i]


@pytest.mark.integration
def test_temporal_controller_raises_on_empty_data(full_dataset, mock_pipeline):
    """Test that controller raises error when no data available.

    If as_of_date is before dataset start, should fail gracefully.
    """
    controller = TemporalController(
        pipeline=mock_pipeline,
        full_dataset=full_dataset,
        enable_data_collection=False
    )

    # Try to access data before dataset starts
    too_early_date = '2019-01-01'

    with pytest.raises(ValueError, match="No data available"):
        controller.update_as_of(too_early_date)


@pytest.mark.integration
def test_temporal_controller_restores_original_data(full_dataset, mock_pipeline):
    """Test that controller restores original data component.

    After temporal analysis, pipeline should be restored to original state.
    """
    controller = TemporalController(
        pipeline=mock_pipeline,
        full_dataset=full_dataset,
        enable_data_collection=False
    )

    # Store original data component
    original_data = mock_pipeline.data

    # Run temporal analysis
    controller.update_as_of('2020-06-30')

    # Verify data was restored
    assert mock_pipeline.data == original_data, \
        "Original data component should be restored"


# ============================================================================
# Rolling Window Tests
# ============================================================================


@pytest.mark.integration
def test_step_through_time_no_lookahead(full_dataset, mock_pipeline):
    """Test that stepping through time never uses future data.

    CRITICAL for backtesting - each step must only see past data.
    """
    controller = TemporalController(
        pipeline=mock_pipeline,
        full_dataset=full_dataset,
        enable_data_collection=False
    )

    # Step through Q2 2020 monthly
    results = controller.step_through_time(
        start_date='2020-04-01',
        end_date='2020-06-30',
        freq='M'
    )

    # Verify each step maintains temporal isolation
    for date_str, _ in results:
        analysis_date = pd.to_datetime(date_str)

        # Check audit log for this date
        matching_logs = [
            log for log in controller.access_log
            if log['as_of_date'] == date_str
        ]

        assert len(matching_logs) > 0, f"Should have log entry for {date_str}"
        log = matching_logs[0]

        # Critical: data_end must be <= analysis date
        data_end = pd.to_datetime(log['data_end'])
        assert data_end <= analysis_date, \
            f"Data end {data_end} exceeds analysis date {analysis_date}"


@pytest.mark.integration
def test_step_through_time_increasing_data_size(full_dataset, mock_pipeline):
    """Test that data size increases monotonically when stepping forward.

    Each subsequent step should have more historical data available.
    """
    controller = TemporalController(
        pipeline=mock_pipeline,
        full_dataset=full_dataset,
        enable_data_collection=False
    )

    results = controller.step_through_time(
        start_date='2020-01-31',
        end_date='2020-12-31',
        freq='M'
    )

    # Extract data sizes from audit log
    data_sizes = [log['num_observations'] for log in controller.access_log]

    # Verify monotonic increase
    for i in range(1, len(data_sizes)):
        assert data_sizes[i] >= data_sizes[i-1], \
            f"Data size should increase: {data_sizes[i-1]} -> {data_sizes[i]}"


@pytest.mark.integration
def test_rolling_window_no_data_contamination(full_dataset, mock_pipeline):
    """Test that rolling windows never contaminate each other.

    Each window should be independent - no state should leak between windows.
    """
    controller = TemporalController(
        pipeline=mock_pipeline,
        full_dataset=full_dataset,
        enable_data_collection=False
    )

    # Create two controllers with different dates
    controller.update_as_of('2020-06-30')
    data_june = mock_pipeline.data.get_all_data()

    controller.update_as_of('2020-03-31')
    data_march = mock_pipeline.data.get_all_data()

    # Verify March window doesn't have June data
    assert len(data_march) < len(data_june), "Earlier window should have less data"
    assert data_march.index.max() < data_june.index.max(), \
        "Earlier window should have earlier max date"


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.integration
def test_temporal_controller_handles_timezone_aware_index(mock_pipeline):
    """Test that controller handles timezone-aware DatetimeIndex."""
    # Create timezone-aware dataset
    dates = pd.date_range(
        start='2020-01-01',
        end='2020-12-31',
        freq='D',
        tz='America/New_York'
    )

    df = pd.DataFrame({
        'price': 100 + np.random.randn(len(dates)),
        'log_return': np.random.randn(len(dates)) * 0.02
    }, index=dates)

    controller = TemporalController(
        pipeline=mock_pipeline,
        full_dataset=df,
        enable_data_collection=False
    )

    # Should handle timezone-aware filtering
    controller.update_as_of('2020-06-30')
    data = mock_pipeline.data.get_all_data()

    assert len(data) > 0, "Should handle timezone-aware data"
    assert data.index.tz is not None, "Should preserve timezone info"


@pytest.mark.integration
def test_temporal_controller_requires_datetime_index(mock_pipeline):
    """Test that controller validates DatetimeIndex requirement."""
    # Create dataset with non-datetime index
    df = pd.DataFrame({
        'price': [100, 101, 102],
        'log_return': [0.0, 0.01, 0.01]
    }, index=[0, 1, 2])  # Integer index

    with pytest.raises(ValueError, match="must have DatetimeIndex"):
        TemporalController(
            pipeline=mock_pipeline,
            full_dataset=df,
            enable_data_collection=False
        )


@pytest.mark.integration
def test_temporal_controller_handles_gaps_in_data(full_dataset, mock_pipeline):
    """Test that controller handles missing dates in dataset.

    Not all calendar dates may have data (weekends, holidays).
    """
    # Remove some random dates to create gaps
    sample_indices = np.random.choice(
        len(full_dataset),
        size=int(len(full_dataset) * 0.8),
        replace=False
    )
    gapped_dataset = full_dataset.iloc[sorted(sample_indices)]

    controller = TemporalController(
        pipeline=mock_pipeline,
        full_dataset=gapped_dataset,
        enable_data_collection=False
    )

    # Should handle stepping through dates with gaps
    results = controller.step_through_time(
        start_date='2020-01-01',
        end_date='2020-12-31',
        freq='M'
    )

    # Should only process dates that exist in dataset
    assert len(results) <= 12, "Should only process available dates"
    assert all(r[1] for r in results), "All results should be non-empty"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
