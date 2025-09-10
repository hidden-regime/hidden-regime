"""
Tests for Streaming Data Processing functionality.

This module tests the streaming data interfaces, processors, and
data sources for real-time market regime detection.

Author: aoaustin
Created: 2025-09-03
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from hidden_regime.models.config import HMMConfig
from hidden_regime.models.online_hmm import OnlineHMM, OnlineHMMConfig
from hidden_regime.models.streaming import (
    SimulatedDataSource,
    StreamingConfig,
    StreamingDataSource,
    StreamingMode,
    StreamingObservation,
    StreamingProcessor,
    StreamingResult,
)


class TestStreamingConfig:
    """Tests for StreamingConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StreamingConfig()
        assert config.mode == StreamingMode.REAL_TIME
        assert config.max_updates_per_second > 0
        assert config.max_observations_per_second > 0
        assert config.input_buffer_size > 0

    def test_config_modes(self):
        """Test different streaming modes."""
        real_time_config = StreamingConfig(mode=StreamingMode.REAL_TIME)
        micro_batch_config = StreamingConfig(
            mode=StreamingMode.MICRO_BATCH, batch_size=10
        )
        time_window_config = StreamingConfig(
            mode=StreamingMode.TIME_WINDOW, time_window_seconds=30.0
        )

        assert real_time_config.mode == StreamingMode.REAL_TIME
        assert micro_batch_config.mode == StreamingMode.MICRO_BATCH
        assert micro_batch_config.batch_size == 10
        assert time_window_config.mode == StreamingMode.TIME_WINDOW
        assert time_window_config.time_window_seconds == 30.0


class TestStreamingObservation:
    """Tests for StreamingObservation class."""

    def test_valid_observation(self):
        """Test creation of valid streaming observation."""
        obs = StreamingObservation(
            timestamp=pd.Timestamp("2024-01-01 10:30:00"),
            symbol="AAPL",
            price=150.0,
            log_return=0.01,
            volume=1000000,
        )

        assert obs.symbol == "AAPL"
        assert obs.price == 150.0
        assert obs.log_return == 0.01
        assert obs.volume == 1000000

    def test_invalid_price(self):
        """Test validation of invalid prices."""
        with pytest.raises(ValueError, match="Invalid price"):
            StreamingObservation(
                timestamp=pd.Timestamp("2024-01-01"),
                symbol="AAPL",
                price=-10.0,  # Negative price
                log_return=0.01,
            )

        with pytest.raises(ValueError, match="Invalid price"):
            StreamingObservation(
                timestamp=pd.Timestamp("2024-01-01"),
                symbol="AAPL",
                price=np.inf,  # Infinite price
                log_return=0.01,
            )

    def test_invalid_log_return(self):
        """Test validation of invalid log returns."""
        with pytest.raises(ValueError, match="Invalid log_return"):
            StreamingObservation(
                timestamp=pd.Timestamp("2024-01-01"),
                symbol="AAPL",
                price=150.0,
                log_return=np.nan,  # NaN log return
            )


class TestStreamingResult:
    """Tests for StreamingResult class."""

    def test_streaming_result_creation(self):
        """Test creation of streaming result."""
        result = StreamingResult(
            timestamp=pd.Timestamp("2024-01-01 10:30:00"),
            symbol="AAPL",
            regime=1,
            regime_probabilities=[0.1, 0.7, 0.2],
            confidence=0.7,
            regime_interpretation="Bull Market",
            regime_characteristics={"mean_return": 0.01, "volatility": 0.02},
            diagnostics={"change_detected": False},
            processing_time_ms=5.2,
        )

        assert result.symbol == "AAPL"
        assert result.regime == 1
        assert result.confidence == 0.7
        assert result.processing_time_ms == 5.2


class TestSimulatedDataSource:
    """Tests for SimulatedDataSource class."""

    @pytest.fixture
    def sample_historical_data(self):
        """Create sample historical data."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
        log_returns = np.diff(np.log(prices))
        log_returns = np.concatenate([[0], log_returns])  # Add first return as 0

        return pd.DataFrame(
            {
                "date": dates,
                "symbol": "TEST",
                "price": prices,
                "log_return": log_returns,
                "volume": np.random.randint(1000000, 5000000, 100),
            }
        )

    def test_initialization(self, sample_historical_data):
        """Test initialization of simulated data source."""
        source = SimulatedDataSource(
            historical_data=sample_historical_data,
            speed_multiplier=10.0,
            add_noise=True,
            noise_std=0.001,
        )

        assert source.speed_multiplier == 10.0
        assert source.add_noise is True
        assert source.noise_std == 0.001
        assert not source.is_connected()

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, sample_historical_data):
        """Test connection management."""
        source = SimulatedDataSource(sample_historical_data)

        # Initially not connected
        assert not source.is_connected()

        # Connect
        connected = await source.connect()
        assert connected is True
        assert source.is_connected()

        # Disconnect
        disconnected = await source.disconnect()
        assert disconnected is True
        assert not source.is_connected()

    @pytest.mark.asyncio
    async def test_stream_observations(self, sample_historical_data):
        """Test streaming observations from historical data."""
        source = SimulatedDataSource(
            historical_data=sample_historical_data,
            speed_multiplier=100.0,  # Very fast for testing
        )

        await source.connect()

        observations = []
        async for obs in source.stream_observations(["TEST"]):
            observations.append(obs)
            if len(observations) >= 5:  # Just get first 5 observations
                break

        assert len(observations) == 5

        # Check first observation
        first_obs = observations[0]
        assert first_obs.symbol == "TEST"
        assert first_obs.price > 0
        assert first_obs.log_return is not None
        assert isinstance(first_obs.timestamp, pd.Timestamp)

    @pytest.mark.asyncio
    async def test_stream_with_noise(self, sample_historical_data):
        """Test streaming with added noise."""
        source = SimulatedDataSource(
            historical_data=sample_historical_data,
            speed_multiplier=100.0,
            add_noise=True,
            noise_std=0.001,
        )

        await source.connect()

        # Get original and noisy observations
        original_returns = sample_historical_data["log_return"].values[:5]

        observations = []
        async for obs in source.stream_observations(["TEST"]):
            observations.append(obs)
            if len(observations) >= 5:
                break

        noisy_returns = [obs.log_return for obs in observations]

        # Returns should be different due to noise (with high probability)
        assert not np.array_equal(original_returns, noisy_returns)


class TestStreamingProcessor:
    """Tests for StreamingProcessor class."""

    @pytest.fixture
    def fitted_online_hmm(self):
        """Create fitted OnlineHMM for testing."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 200)

        online_config = OnlineHMMConfig(forgetting_factor=0.95, adaptation_rate=0.1)

        hmm = OnlineHMM(n_states=3, online_config=online_config)
        hmm.fit(returns[:100])

        return hmm

    @pytest.fixture
    def sample_streaming_data(self):
        """Create sample streaming data."""
        dates = pd.date_range("2024-01-01", periods=50, freq="min")
        np.random.seed(42)

        prices = 100 * np.cumprod(1 + np.random.normal(0.0001, 0.002, 50))
        log_returns = np.diff(np.log(prices))
        log_returns = np.concatenate([[0], log_returns])

        return pd.DataFrame(
            {
                "date": dates,
                "symbol": "TEST",
                "price": prices,
                "log_return": log_returns,
            }
        )

    def test_initialization(self, fitted_online_hmm, sample_streaming_data):
        """Test StreamingProcessor initialization."""
        data_source = SimulatedDataSource(sample_streaming_data, speed_multiplier=100.0)
        config = StreamingConfig(mode=StreamingMode.REAL_TIME)

        processor = StreamingProcessor(
            online_hmm=fitted_online_hmm, data_source=data_source, config=config
        )

        assert processor.online_hmm == fitted_online_hmm
        assert processor.data_source == data_source
        assert processor.config == config
        assert not processor._running

    @pytest.mark.asyncio
    async def test_real_time_processing(self, fitted_online_hmm, sample_streaming_data):
        """Test real-time streaming processing."""
        data_source = SimulatedDataSource(
            sample_streaming_data, speed_multiplier=1000.0
        )
        config = StreamingConfig(
            mode=StreamingMode.REAL_TIME,
            max_observations_per_second=1000.0,  # High rate for testing
        )

        processor = StreamingProcessor(
            online_hmm=fitted_online_hmm, data_source=data_source, config=config
        )

        # Collect results
        results = []

        async def result_callback(result: StreamingResult):
            results.append(result)
            if len(results) >= 5:  # Stop after 5 results
                await processor.stop()

        # Start processing (will run until stopped by callback)
        try:
            await processor.start(["TEST"], result_callback)
        except Exception:
            pass  # Expected when we stop the processor

        # Check results
        assert len(results) >= 3  # Should have some results

        first_result = results[0]
        assert first_result.symbol == "TEST"
        assert 0 <= first_result.regime < 3
        assert 0 <= first_result.confidence <= 1
        assert first_result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_micro_batch_processing(
        self, fitted_online_hmm, sample_streaming_data
    ):
        """Test micro-batch processing mode."""
        data_source = SimulatedDataSource(
            sample_streaming_data, speed_multiplier=1000.0
        )
        config = StreamingConfig(mode=StreamingMode.MICRO_BATCH, batch_size=3)

        processor = StreamingProcessor(
            online_hmm=fitted_online_hmm, data_source=data_source, config=config
        )

        results = []

        async def result_callback(result: StreamingResult):
            results.append(result)
            if len(results) >= 6:  # Get two batches worth
                await processor.stop()

        try:
            await processor.start(["TEST"], result_callback)
        except Exception:
            pass

        assert len(results) >= 3  # At least one batch processed

    @pytest.mark.asyncio
    async def test_observation_validation(
        self, fitted_online_hmm, sample_streaming_data
    ):
        """Test observation validation and filtering."""
        # Add some outliers to the data
        outlier_data = sample_streaming_data.copy()
        outlier_data.loc[5, "log_return"] = 0.5  # Extreme outlier
        outlier_data.loc[10, "log_return"] = -0.3  # Another outlier

        data_source = SimulatedDataSource(outlier_data, speed_multiplier=1000.0)
        config = StreamingConfig(
            mode=StreamingMode.REAL_TIME,
            validate_observations=True,
            outlier_threshold=3.0,  # Should filter extreme values
        )

        processor = StreamingProcessor(
            online_hmm=fitted_online_hmm, data_source=data_source, config=config
        )

        results = []

        async def result_callback(result: StreamingResult):
            results.append(result)
            if len(results) >= 5:
                await processor.stop()

        try:
            await processor.start(["TEST"], result_callback)
        except Exception:
            pass

        # Should have fewer results due to outlier filtering
        # (Exact number depends on timing and which observations were processed)
        assert len(results) >= 1

        # Check that processed returns are reasonable
        for result in results:
            # Regime characteristics should be reasonable (no extreme values)
            mean_return = result.regime_characteristics.get("mean_return", 0)
            assert abs(mean_return) < 0.1  # Reasonable bound

    @pytest.mark.asyncio
    async def test_rate_limiting(self, fitted_online_hmm, sample_streaming_data):
        """Test rate limiting functionality."""
        data_source = SimulatedDataSource(
            sample_streaming_data, speed_multiplier=1000.0
        )
        config = StreamingConfig(
            mode=StreamingMode.REAL_TIME,
            max_observations_per_second=5.0,  # Low rate limit
            max_updates_per_second=2.0,
        )

        processor = StreamingProcessor(
            online_hmm=fitted_online_hmm, data_source=data_source, config=config
        )

        start_time = time.time()
        results = []

        async def result_callback(result: StreamingResult):
            results.append(result)
            if len(results) >= 3:
                await processor.stop()

        try:
            await processor.start(["TEST"], result_callback)
        except Exception:
            pass

        end_time = time.time()
        duration = end_time - start_time

        # With rate limiting, should take reasonable time
        # (At least some minimum time due to rate limits)
        if len(results) >= 3:
            expected_min_time = (
                2.0 / config.max_observations_per_second
            )  # Time for 3 observations
            # Allow some tolerance for processing overhead
            assert duration >= expected_min_time * 0.5

    def test_performance_stats(self, fitted_online_hmm, sample_streaming_data):
        """Test performance statistics tracking."""
        data_source = SimulatedDataSource(sample_streaming_data)
        processor = StreamingProcessor(
            online_hmm=fitted_online_hmm, data_source=data_source
        )

        # Initially, no stats
        stats = processor.get_performance_stats()
        assert stats == {} or stats["observations_processed"] == 0

        # Simulate some processing by directly updating internal counters
        processor._observations_processed = 10
        processor._processing_times.extend([1.0, 2.0, 1.5, 3.0, 2.5])

        stats = processor.get_performance_stats()
        assert stats["observations_processed"] == 10
        assert "avg_processing_time_ms" in stats
        assert "max_processing_time_ms" in stats

    def test_recent_results_buffer(self, fitted_online_hmm, sample_streaming_data):
        """Test recent results buffer functionality."""
        data_source = SimulatedDataSource(sample_streaming_data)
        processor = StreamingProcessor(
            online_hmm=fitted_online_hmm, data_source=data_source
        )

        # Initially empty
        recent = processor.get_recent_results(5)
        assert len(recent) == 0

        # Add some mock results to buffer
        for i in range(10):
            mock_result = StreamingResult(
                timestamp=pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i),
                symbol="TEST",
                regime=i % 3,
                regime_probabilities=[0.3, 0.4, 0.3],
                confidence=0.7,
                regime_interpretation="Test",
                regime_characteristics={"mean_return": 0.01},
                diagnostics={},
                processing_time_ms=1.0,
            )
            processor._result_buffer.append(mock_result)

        # Get recent results
        recent_5 = processor.get_recent_results(5)
        recent_all = processor.get_recent_results(20)  # More than available

        assert len(recent_5) == 5
        assert len(recent_all) == 10  # Should return all available

        # Should be most recent results
        assert recent_5[-1].regime == 9 % 3  # Last added result


class TestStreamingIntegration:
    """Integration tests for complete streaming pipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_streaming(self):
        """Test complete end-to-end streaming pipeline."""
        # Generate realistic market data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="min")

        # Create regime-switching data
        regime_lengths = [30, 40, 30]  # Bull, bear, sideways
        regimes = []
        for i, length in enumerate(regime_lengths):
            regimes.extend([i] * length)

        means = [0.002, -0.003, 0.0]  # Bull, bear, sideways
        stds = [0.015, 0.025, 0.01]

        returns = []
        prices = [100.0]  # Starting price

        for regime in regimes:
            ret = np.random.normal(means[regime], stds[regime])
            returns.append(ret)
            prices.append(prices[-1] * np.exp(ret))

        historical_data = pd.DataFrame(
            {
                "date": dates,
                "symbol": "MARKET",
                "price": prices[:100],
                "log_return": returns[:100],
            }
        )

        # Create and fit OnlineHMM
        online_config = OnlineHMMConfig(
            forgetting_factor=0.98, adaptation_rate=0.05, enable_change_detection=True
        )

        hmm = OnlineHMM(n_states=3, online_config=online_config)
        hmm.fit(historical_data["log_return"][:50])  # Train on first half

        # Create streaming components
        data_source = SimulatedDataSource(
            historical_data.iloc[50:], speed_multiplier=100.0  # Stream second half
        )

        config = StreamingConfig(
            mode=StreamingMode.REAL_TIME, validate_observations=True
        )

        processor = StreamingProcessor(hmm, data_source, config)

        # Collect results
        results = []
        regime_changes = 0
        last_regime = None

        async def result_callback(result: StreamingResult):
            nonlocal last_regime, regime_changes

            results.append(result)

            if last_regime is not None and result.regime != last_regime:
                regime_changes += 1

            last_regime = result.regime

            if len(results) >= 30:  # Process enough to see regime changes
                await processor.stop()

        # Run streaming processing
        try:
            await processor.start(["MARKET"], result_callback)
        except Exception:
            pass  # Expected when stopping

        # Validate results
        assert len(results) >= 20  # Should process significant number
        assert regime_changes >= 1  # Should detect at least one regime change

        # Check that results have expected structure
        for result in results:
            assert result.symbol == "MARKET"
            assert 0 <= result.regime < 3
            assert 0 <= result.confidence <= 1
            assert len(result.regime_probabilities) == 3
            assert sum(result.regime_probabilities) == pytest.approx(1.0, rel=1e-5)

        # Check performance
        stats = processor.get_performance_stats()
        assert stats["observations_processed"] >= len(results)
        if "avg_processing_time_ms" in stats:
            assert stats["avg_processing_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_change_point_detection_streaming(self):
        """Test change point detection in streaming context."""
        # Create data with clear structural break
        np.random.seed(42)

        # Stable period
        stable_returns = np.random.normal(0.001, 0.01, 50)

        # Volatile period (crisis)
        volatile_returns = np.random.normal(-0.01, 0.04, 50)

        all_returns = np.concatenate([stable_returns, volatile_returns])
        dates = pd.date_range("2024-01-01", periods=100, freq="min")
        prices = 100 * np.cumprod(1 + all_returns)

        historical_data = pd.DataFrame(
            {
                "date": dates,
                "symbol": "CRISIS",
                "price": prices,
                "log_return": all_returns,
            }
        )

        # Create OnlineHMM with sensitive change detection
        online_config = OnlineHMMConfig(
            enable_change_detection=True,
            change_detection_threshold=2.0,  # More sensitive
            change_detection_window=20,
        )

        hmm = OnlineHMM(n_states=3, online_config=online_config)
        hmm.fit(stable_returns[:30])  # Train on stable period

        # Stream the crisis period
        data_source = SimulatedDataSource(
            historical_data.iloc[30:], speed_multiplier=100.0
        )

        processor = StreamingProcessor(hmm, data_source, StreamingConfig())

        change_points_detected = 0
        results = []

        async def result_callback(result: StreamingResult):
            nonlocal change_points_detected

            results.append(result)

            if result.diagnostics.get("change_detected", False):
                change_points_detected += 1

            if len(results) >= 50:
                await processor.stop()

        try:
            await processor.start(["CRISIS"], result_callback)
        except Exception:
            pass

        # Should detect the structural break
        assert change_points_detected >= 1
        assert len(hmm.change_points) >= 1
