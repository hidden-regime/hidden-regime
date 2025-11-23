"""
Integration tests for QuantConnect components.

Tests the full pipeline integration:
- Data adapters → Pipeline → Signal adapters → Trading
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from hidden_regime.quantconnect.data_adapter import QuantConnectDataAdapter
from hidden_regime.quantconnect.signal_adapter import (
    RegimeSignalAdapter,
    MultiAssetSignalAdapter
)
from hidden_regime.quantconnect.performance.caching import CachedRegimeDetector
from hidden_regime.quantconnect.performance.batch_updates import BatchRegimeUpdater


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""

    def test_single_asset_pipeline(self, mock_tradebar_data, mock_pipeline):
        """Test single asset from data collection to trading signal."""
        # Step 1: Collect data
        data_adapter = QuantConnectDataAdapter(window_size=252)
        for bar in mock_tradebar_data:
            data_adapter.add_bar(bar)

        df = data_adapter.to_dataframe()

        # Step 2: Detect regime
        result = mock_pipeline.run(df)
        current_regime = result['regime']
        confidence = result['confidence']

        # Step 3: Generate trading signal
        signal_adapter = RegimeSignalAdapter(
            regime_allocations={'Bull': 1.0, 'Bear': 0.0, 'Sideways': 0.5}
        )
        signal = signal_adapter.regime_to_signal(current_regime, confidence)

        # Verify complete pipeline
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert current_regime is not None
        assert signal.allocation >= 0.0
        assert signal.allocation <= 1.0

    def test_multi_asset_pipeline(self, sample_multi_asset_data, mock_pipeline):
        """Test multi-asset rotation pipeline."""
        assets = list(sample_multi_asset_data.keys())

        # Step 1: Detect regimes for all assets
        regime_signals = {}

        for asset, data in sample_multi_asset_data.items():
            result = mock_pipeline.run(data)

            signal_adapter = RegimeSignalAdapter(
                regime_allocations={'Bull': 1.0, 'Bear': 0.0, 'Sideways': 0.5}
            )

            regime_signals[asset] = signal_adapter.regime_to_signal(
                result['regime'],
                result['confidence']
            )

        # Step 2: Calculate portfolio allocations
        multi_adapter = MultiAssetSignalAdapter(
            assets=assets,
            allocation_method='confidence_weighted'
        )

        allocations = multi_adapter.calculate_allocations(regime_signals)

        # Verify multi-asset pipeline
        assert len(regime_signals) == len(assets)
        assert len(allocations) == len(assets)
        assert abs(sum(allocations.values()) - 1.0) < 0.1  # Approximately 1.0

    def test_pipeline_with_caching(self, sample_price_data, mock_pipeline):
        """Test pipeline with caching optimization."""
        detector = CachedRegimeDetector(
            max_cache_size=10,
            retrain_frequency='weekly'
        )

        # First run - should cache
        result1 = detector.detect_regime(
            ticker='SPY',
            n_states=3,
            data=sample_price_data,
            pipeline=mock_pipeline
        )

        # Second run - should hit cache
        result2 = detector.detect_regime(
            ticker='SPY',
            n_states=3,
            data=sample_price_data,
            pipeline=mock_pipeline
        )

        # Results should be consistent
        assert result1 == result2
        assert detector.model_cache.hit_count > 0

    def test_pipeline_with_batch_updates(self, sample_multi_asset_data, mock_pipeline):
        """Test pipeline with batch updates."""
        assets = list(sample_multi_asset_data.keys())

        updater = BatchRegimeUpdater(max_workers=4, use_parallel=True)

        def update_regime(asset, data, pipeline):
            result = pipeline.run(data)
            return {
                'regime': result['regime'],
                'confidence': result['confidence']
            }

        pipeline_dict = {asset: mock_pipeline for asset in assets}

        # Run batch update
        results = updater.batch_update(
            assets,
            sample_multi_asset_data,
            pipeline_dict,
            update_regime
        )

        # Verify all assets updated
        assert len(results) == len(assets)
        assert all('regime' in r for r in results.values())


class TestRealisticTradingScenarios:
    """Test realistic trading scenarios."""

    def test_regime_change_trading(self, mock_tradebar_data):
        """Test trading logic during regime changes."""
        data_adapter = QuantConnectDataAdapter(window_size=252)
        signal_adapter = RegimeSignalAdapter(
            regime_allocations={'Bull': 1.0, 'Bear': 0.0, 'Sideways': 0.5}
        )

        # Simulate regime changing from Bull to Bear
        mock_algo = Mock()
        mock_algo.holdings = {}

        # Phase 1: Bull regime
        for bar in mock_tradebar_data[:100]:
            data_adapter.add_bar(bar)

        signal_bull = signal_adapter.regime_to_signal('Bull', confidence=0.8)
        mock_algo.SetHoldings('SPY', signal_bull.allocation)

        assert mock_algo.SetHoldings.called
        assert mock_algo.SetHoldings.call_args[0][1] == 1.0  # Full allocation

        # Phase 2: Regime changes to Bear
        signal_bear = signal_adapter.regime_to_signal('Bear', confidence=0.85)
        mock_algo.Liquidate('SPY')

        assert mock_algo.Liquidate.called

    def test_multi_asset_rebalancing(self, sample_multi_asset_data):
        """Test portfolio rebalancing across multiple assets."""
        assets = list(sample_multi_asset_data.keys())

        multi_adapter = MultiAssetSignalAdapter(
            assets=assets,
            allocation_method='equal_weight',
            rebalance_threshold=0.05  # 5% threshold
        )

        # Initial allocation
        current_allocation = {'SPY': 0.25, 'QQQ': 0.25, 'TLT': 0.25, 'GLD': 0.25}

        # Regime signals suggest different allocation
        from hidden_regime.quantconnect.signal_adapter import TradingSignal

        regime_signals = {
            'SPY': TradingSignal('long', allocation=1.0, confidence=0.9),
            'QQQ': TradingSignal('long', allocation=1.0, confidence=0.8),
            'TLT': TradingSignal('cash', allocation=0.0, confidence=0.7),
            'GLD': TradingSignal('long', allocation=1.0, confidence=0.6)
        }

        new_allocation = multi_adapter.calculate_allocations(regime_signals)

        # Check if rebalance needed
        should_rebalance = multi_adapter.should_rebalance(
            current_allocation,
            new_allocation
        )

        # Should rebalance due to TLT going to 0
        assert should_rebalance

    def test_crisis_defensive_positioning(self):
        """Test defensive positioning during crisis."""
        # Crisis regime signals
        signal_adapter = RegimeSignalAdapter(
            regime_allocations={
                'Bull': 1.0,
                'Bear': 0.0,
                'Sideways': 0.5,
                'Crisis': -0.5  # Short exposure or defensive
            }
        )

        signal = signal_adapter.regime_to_signal('Crisis', confidence=0.9)

        # Should be defensive
        assert signal.direction in ['short', 'cash']

    def test_confidence_based_position_sizing(self):
        """Test position sizing based on regime confidence."""
        signal_adapter = RegimeSignalAdapter(
            regime_allocations={'Bull': 1.0},
            use_dynamic_sizing=True
        )

        # High confidence - full position
        signal_high = signal_adapter.regime_to_signal('Bull', confidence=0.95)

        # Low confidence - reduced position
        signal_low = signal_adapter.regime_to_signal('Bull', confidence=0.55)

        # High confidence should have larger allocation
        assert signal_high.allocation >= signal_low.allocation


class TestDataFlowIntegration:
    """Test data flow through the system."""

    def test_data_continuity(self, mock_tradebar_data):
        """Test that data maintains continuity through adapters."""
        adapter = QuantConnectDataAdapter(window_size=252)

        # Add bars
        for bar in mock_tradebar_data:
            adapter.add_bar(bar)

        df = adapter.to_dataframe()

        # Verify no gaps in data
        assert df.index.is_monotonic_increasing
        assert not df.isnull().any().any()

        # Verify OHLC relationships
        assert (df['High'] >= df['Low']).all()
        assert (df['High'] >= df['Open']).all()
        assert (df['High'] >= df['Close']).all()
        assert (df['Low'] <= df['Open']).all()
        assert (df['Low'] <= df['Close']).all()

    def test_timezone_handling(self):
        """Test proper timezone handling in data flow."""
        class MockTradeBar:
            def __init__(self, time):
                self.Time = time
                self.Open = 100.0
                self.High = 101.0
                self.Low = 99.0
                self.Close = 100.5
                self.Volume = 1e6

        # Create bars with different timezones
        adapter = QuantConnectDataAdapter(window_size=100)

        dates = pd.date_range('2020-01-01', periods=10, freq='D', tz='UTC')
        for date in dates:
            adapter.add_bar(MockTradeBar(date))

        df = adapter.to_dataframe()

        # Should handle timezone properly
        assert len(df) == 10

    def test_missing_data_handling(self, mock_tradebar_data):
        """Test handling of gaps in data."""
        adapter = QuantConnectDataAdapter(window_size=252)

        # Add bars with gaps (skip every 5th bar)
        for i, bar in enumerate(mock_tradebar_data):
            if i % 5 != 0:
                adapter.add_bar(bar)

        df = adapter.to_dataframe()

        # Should still produce valid DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestErrorHandlingIntegration:
    """Test error handling across components."""

    def test_pipeline_error_recovery(self, sample_price_data):
        """Test recovery from pipeline errors."""
        detector = CachedRegimeDetector(max_cache_size=10)

        # Mock pipeline that fails
        failing_pipeline = Mock()
        failing_pipeline.run.side_effect = Exception("Pipeline error")

        # Should handle gracefully
        try:
            detector.detect_regime(
                ticker='SPY',
                n_states=3,
                data=sample_price_data,
                pipeline=failing_pipeline
            )
        except Exception as e:
            # Expected - should propagate or handle gracefully
            assert str(e) == "Pipeline error"

    def test_partial_asset_failure(self, sample_multi_asset_data):
        """Test handling when some assets fail to update."""
        assets = list(sample_multi_asset_data.keys())

        updater = BatchRegimeUpdater(max_workers=4)

        def update_with_failures(asset, data, pipeline):
            if asset == 'QQQ':
                raise ValueError("QQQ update failed")
            return {'regime': 'Bull', 'confidence': 0.8}

        pipeline_dict = {asset: Mock() for asset in assets}

        # Should continue with other assets
        results = updater.batch_update(
            assets,
            sample_multi_asset_data,
            pipeline_dict,
            update_with_failures,
            continue_on_error=True
        )

        # Should have results for non-failing assets
        assert 'SPY' in results
        assert 'TLT' in results


class TestPerformanceIntegration:
    """Test performance characteristics of integrated system."""

    def test_caching_effectiveness(self, performance_test_data, mock_pipeline):
        """Test that caching improves performance."""
        import time

        detector = CachedRegimeDetector(
            max_cache_size=50,
            retrain_frequency='monthly'  # Rarely retrain
        )

        # First run - no cache
        start = time.time()
        for _ in range(10):
            detector.detect_regime(
                ticker='SPY',
                n_states=3,
                data=performance_test_data,
                pipeline=mock_pipeline
            )
        first_run_time = time.time() - start

        # Second run - with cache
        start = time.time()
        for _ in range(10):
            detector.detect_regime(
                ticker='SPY',
                n_states=3,
                data=performance_test_data,
                pipeline=mock_pipeline
            )
        cached_run_time = time.time() - start

        # Cached should be significantly faster
        assert cached_run_time < first_run_time * 0.5  # At least 2x faster

    def test_batch_processing_speedup(self, sample_multi_asset_data):
        """Test that batch processing is faster than sequential."""
        import time

        assets = list(sample_multi_asset_data.keys())

        def slow_update(asset, data, pipeline):
            time.sleep(0.05)  # Simulate slow operation
            return {'regime': 'Bull'}

        pipeline_dict = {asset: Mock() for asset in assets}

        # Sequential
        start = time.time()
        sequential_results = {}
        for asset in assets:
            sequential_results[asset] = slow_update(
                asset,
                sample_multi_asset_data[asset],
                pipeline_dict[asset]
            )
        sequential_time = time.time() - start

        # Parallel
        updater = BatchRegimeUpdater(max_workers=4, use_parallel=True)
        start = time.time()
        parallel_results = updater.batch_update(
            assets,
            sample_multi_asset_data,
            pipeline_dict,
            slow_update
        )
        parallel_time = time.time() - start

        # Parallel should be faster
        speedup = sequential_time / parallel_time
        assert speedup >= 1.5  # At least 1.5x speedup


class TestConfigurationIntegration:
    """Test configuration across components."""

    def test_config_propagation(self, qc_config):
        """Test that configuration propagates correctly."""
        # Create components with config
        detector = CachedRegimeDetector(
            max_cache_size=qc_config.cache_size,
            retrain_frequency=qc_config.retrain_frequency
        )

        updater = BatchRegimeUpdater(
            max_workers=qc_config.max_workers,
            use_parallel=qc_config.batch_updates
        )

        # Verify config applied
        assert detector.model_cache.max_cache_size == qc_config.cache_size
        assert detector.retrain_frequency == qc_config.retrain_frequency
        assert updater.max_workers == qc_config.max_workers

    def test_dynamic_config_updates(self):
        """Test updating configuration at runtime."""
        detector = CachedRegimeDetector(max_cache_size=10)

        # Update config
        detector.model_cache.max_cache_size = 50
        detector.retrain_frequency = 'daily'

        assert detector.model_cache.max_cache_size == 50
        assert detector.retrain_frequency == 'daily'
