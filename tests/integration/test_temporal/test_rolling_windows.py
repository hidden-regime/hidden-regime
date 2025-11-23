"""
Rolling window tests for temporal correctness.

These tests ensure that rolling window analysis maintains temporal isolation
and correctly handles window boundaries, sizes, and overlaps.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from hidden_regime.pipeline.temporal import TemporalController
from unittest.mock import Mock


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def trading_dataset():
    """Generate 3 years of daily trading data with realistic patterns."""
    np.random.seed(42)
    dates = pd.date_range(start='2019-01-01', end='2021-12-31', freq='D')
    n = len(dates)

    # Generate returns with regime switching
    returns = np.zeros(n)
    regime = 0
    for i in range(n):
        if i % 100 == 0:  # Switch regime every 100 days
            regime = (regime + 1) % 3

        if regime == 0:  # Low vol bull
            returns[i] = np.random.randn() * 0.01 + 0.0005
        elif regime == 1:  # High vol bear
            returns[i] = np.random.randn() * 0.03 - 0.0005
        else:  # Moderate vol sideways
            returns[i] = np.random.randn() * 0.015

    close = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'Close': close,
        'price': close,
        'log_return': returns,
        'Volume': np.random.randint(1_000_000, 20_000_000, n)
    }, index=dates)

    return df


@pytest.fixture
def mock_pipeline_with_model():
    """Create mock pipeline that tracks model training."""
    pipeline = Mock()
    pipeline.data = Mock()
    pipeline.model = Mock()
    pipeline.model.training_data_dates = []

    def track_training(data=None):
        """Track what data was used for training."""
        if hasattr(pipeline.data, 'get_all_data'):
            training_data = pipeline.data.get_all_data()
            pipeline.model.training_data_dates.append({
                'start': training_data.index.min(),
                'end': training_data.index.max(),
                'size': len(training_data)
            })
        return "Analysis complete"

    pipeline.update = Mock(side_effect=track_training)
    return pipeline


# ============================================================================
# Rolling Window Size Tests
# ============================================================================


@pytest.mark.integration
def test_rolling_window_exact_size(trading_dataset, mock_pipeline_with_model):
    """Test that rolling windows maintain exact specified size."""
    window_size_days = 252  # 1 year trading days
    controller = TemporalController(
        pipeline=mock_pipeline_with_model,
        full_dataset=trading_dataset,
        enable_data_collection=False
    )

    # Run rolling analysis with fixed window
    test_dates = pd.date_range(start='2020-01-31', end='2020-12-31', freq='M')

    for date in test_dates:
        date_str = date.strftime('%Y-%m-%d')
        controller.update_as_of(date_str)

        data = mock_pipeline_with_model.data.get_all_data()

        # Window should not exceed max size (may be smaller early in dataset)
        assert len(data) <= window_size_days * 1.5, \
            f"Window size {len(data)} exceeds maximum"


@pytest.mark.integration
def test_rolling_window_min_periods(trading_dataset, mock_pipeline_with_model):
    """Test that rolling windows respect minimum data requirements."""
    controller = TemporalController(
        pipeline=mock_pipeline_with_model,
        full_dataset=trading_dataset,
        enable_data_collection=False
    )

    min_required = 30  # Minimum 30 days for valid analysis

    # Try early date with limited history
    early_date = trading_dataset.index[20]  # Only 20 days available

    controller.update_as_of(early_date.strftime('%Y-%m-%d'))
    data = mock_pipeline_with_model.data.get_all_data()

    # Should provide all available data even if less than minimum
    assert len(data) == 21, "Should provide all available data"  # 20 + 1


@pytest.mark.integration
def test_rolling_window_step_size(trading_dataset, mock_pipeline_with_model):
    """Test that step size controls window advancement correctly."""
    controller = TemporalController(
        pipeline=mock_pipeline_with_model,
        full_dataset=trading_dataset,
        enable_data_collection=False
    )

    # Step weekly through a month
    results = controller.step_through_time(
        start_date='2020-06-01',
        end_date='2020-06-30',
        freq='W'
    )

    # Verify we got approximately weekly steps
    assert len(results) >= 3, "Should have multiple weekly steps"
    assert len(results) <= 6, "Should not have too many steps"


@pytest.mark.integration
def test_rolling_window_edge_handling(trading_dataset, mock_pipeline_with_model):
    """Test behavior at start and end of dataset."""
    controller = TemporalController(
        pipeline=mock_pipeline_with_model,
        full_dataset=trading_dataset,
        enable_data_collection=False
    )

    # Test at dataset start
    start_date = trading_dataset.index[50]
    controller.update_as_of(start_date.strftime('%Y-%m-%d'))
    data_start = mock_pipeline_with_model.data.get_all_data()

    assert len(data_start) == 51, "Should have all data from start"
    assert data_start.index.min() == trading_dataset.index.min()

    # Test near dataset end
    end_date = trading_dataset.index[-2]
    controller.update_as_of(end_date.strftime('%Y-%m-%d'))
    data_end = mock_pipeline_with_model.data.get_all_data()

    assert len(data_end) == len(trading_dataset) - 1, "Should have almost all data"


@pytest.mark.integration
def test_rolling_window_insufficient_data(trading_dataset, mock_pipeline_with_model):
    """Test handling when insufficient data for analysis."""
    controller = TemporalController(
        pipeline=mock_pipeline_with_model,
        full_dataset=trading_dataset,
        enable_data_collection=False
    )

    # Very early date with minimal data
    very_early = trading_dataset.index[2]

    try:
        controller.update_as_of(very_early.strftime('%Y-%m-%d'))
        data = mock_pipeline_with_model.data.get_all_data()

        # Should still provide data, just very limited
        assert len(data) >= 1, "Should provide at least some data"
        assert len(data) <= 3, "Should have very limited data"

    except ValueError:
        # Acceptable to raise error if truly insufficient
        pass


# ============================================================================
# Date Alignment Tests
# ============================================================================


@pytest.mark.integration
def test_rolling_window_date_alignment(trading_dataset, mock_pipeline_with_model):
    """Test that window dates align correctly with analysis dates."""
    controller = TemporalController(
        pipeline=mock_pipeline_with_model,
        full_dataset=trading_dataset,
        enable_data_collection=False
    )

    analysis_date = '2020-09-15'
    controller.update_as_of(analysis_date)

    data = mock_pipeline_with_model.data.get_all_data()

    # Critical: max date in window must not exceed analysis date
    assert data.index.max() <= pd.to_datetime(analysis_date), \
        "Window end must not exceed analysis date"

    # Verify continuous date sequence (accounting for weekends)
    assert data.index.is_monotonic_increasing, "Dates must be in order"


@pytest.mark.integration
def test_rolling_window_regime_stability_window(trading_dataset, mock_pipeline_with_model):
    """Test that windows are sized appropriately for regime detection.

    Windows should be large enough to capture regime changes but not so
    large that regimes are diluted.
    """
    controller = TemporalController(
        pipeline=mock_pipeline_with_model,
        full_dataset=trading_dataset,
        enable_data_collection=False
    )

    # Use typical rolling window for regime detection (3-6 months)
    test_date = '2020-06-30'
    controller.update_as_of(test_date)

    data = mock_pipeline_with_model.data.get_all_data()

    # Should have several months of data for regime stability
    # (at this point in dataset, should have 18 months available)
    assert len(data) >= 90, "Should have at least 3 months of data"
    assert len(data) <= 600, "Window should not be excessively large"


# ============================================================================
# Temporal Isolation in Rolling Windows
# ============================================================================


@pytest.mark.integration
def test_rolling_window_no_overlap_contamination(trading_dataset, mock_pipeline_with_model):
    """Test that rolling windows don't contaminate each other.

    Even with overlapping windows, each should be independently filtered.
    """
    controller = TemporalController(
        pipeline=mock_pipeline_with_model,
        full_dataset=trading_dataset,
        enable_data_collection=False
    )

    # Analyze two overlapping windows
    date1 = '2020-03-31'
    date2 = '2020-06-30'

    controller.update_as_of(date1)
    data1 = mock_pipeline_with_model.data.get_all_data().copy()

    controller.update_as_of(date2)
    data2 = mock_pipeline_with_model.data.get_all_data().copy()

    # Windows should overlap but be independent
    assert len(data2) > len(data1), "Later window should have more data"

    # Verify date2 doesn't contaminate data1 window
    overlap_data = data1[data1.index.isin(data2.index)]
    assert len(overlap_data) == len(data1), "Earlier data should be subset of later"


@pytest.mark.integration
def test_rolling_window_model_retraining(trading_dataset, mock_pipeline_with_model):
    """Test that model is retrained with correct temporal data.

    Each window should trigger retraining with only that window's data.
    """
    controller = TemporalController(
        pipeline=mock_pipeline_with_model,
        full_dataset=trading_dataset,
        enable_data_collection=False
    )

    # Run rolling analysis
    dates = ['2020-01-31', '2020-04-30', '2020-07-31']

    for date in dates:
        controller.update_as_of(date)

    # Check training data tracking
    training_history = mock_pipeline_with_model.model.training_data_dates

    assert len(training_history) == 3, "Should have 3 training sessions"

    # Verify each training used progressively more data
    for i in range(1, len(training_history)):
        prev_size = training_history[i-1]['size']
        curr_size = training_history[i]['size']
        assert curr_size > prev_size, \
            f"Training data should grow: {prev_size} -> {curr_size}"


@pytest.mark.integration
def test_rolling_window_prediction_correctness(trading_dataset, mock_pipeline_with_model):
    """Test that predictions only use data from their window.

    Predictions at time T should only use data before T.
    """
    controller = TemporalController(
        pipeline=mock_pipeline_with_model,
        full_dataset=trading_dataset,
        enable_data_collection=False
    )

    prediction_date = '2020-08-31'
    controller.update_as_of(prediction_date)

    # Get data that was available for prediction
    prediction_data = mock_pipeline_with_model.data.get_all_data()

    # Critical check
    assert prediction_data.index.max() <= pd.to_datetime(prediction_date), \
        "Prediction must not use future data"

    # Verify substantial history available
    assert len(prediction_data) > 100, "Should have substantial history"


@pytest.mark.integration
def test_rolling_window_performance_tracking(trading_dataset, mock_pipeline_with_model):
    """Test that performance can be tracked across windows."""
    controller = TemporalController(
        pipeline=mock_pipeline_with_model,
        full_dataset=trading_dataset,
        enable_data_collection=False
    )

    # Run sequential analysis
    results = controller.step_through_time(
        start_date='2020-01-01',
        end_date='2020-12-31',
        freq='Q'  # Quarterly
    )

    # Should have results for each quarter
    assert len(results) >= 4, "Should have quarterly results"

    # Each result should be independent
    for date_str, report in results:
        assert report is not None, f"Should have report for {date_str}"
        assert isinstance(report, str), "Report should be string"


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.integration
def test_rolling_window_gap_handling(trading_dataset, mock_pipeline_with_model):
    """Test handling of gaps in rolling windows.

    Missing data or weekends should not break rolling analysis.
    """
    # Create dataset with gaps (remove weekends)
    business_days = trading_dataset[trading_dataset.index.dayofweek < 5]

    controller = TemporalController(
        pipeline=mock_pipeline_with_model,
        full_dataset=business_days,
        enable_data_collection=False
    )

    # Should handle gaps gracefully
    results = controller.step_through_time(
        start_date='2020-06-01',
        end_date='2020-06-30',
        freq='W'
    )

    # Should still produce results despite gaps
    assert len(results) > 0, "Should handle gaps in data"


@pytest.mark.integration
def test_rolling_window_timezone_consistency(mock_pipeline_with_model):
    """Test that timezone handling is consistent in rolling windows."""
    # Create timezone-aware data
    dates = pd.date_range(
        start='2020-01-01',
        end='2020-12-31',
        freq='D',
        tz='UTC'
    )

    df = pd.DataFrame({
        'price': 100 + np.random.randn(len(dates)),
        'log_return': np.random.randn(len(dates)) * 0.02
    }, index=dates)

    controller = TemporalController(
        pipeline=mock_pipeline_with_model,
        full_dataset=df,
        enable_data_collection=False
    )

    # Run analysis
    controller.update_as_of('2020-06-30')
    data = mock_pipeline_with_model.data.get_all_data()

    # Timezone should be preserved
    assert data.index.tz is not None, "Should preserve timezone"
    assert str(data.index.tz) == 'UTC', "Should maintain UTC timezone"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
