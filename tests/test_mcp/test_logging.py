"""
Tests for MCP logging functionality.

Verifies that appropriate log messages are generated during:
1. Cache operations (hit/miss)
2. Data loading operations
3. Model training operations
4. Result caching
"""

import logging
import pytest
from unittest.mock import Mock, patch, MagicMock

from hidden_regime_mcp.tools import (
    detect_regime,
    get_regime_statistics,
    get_transition_probabilities,
)


@pytest.fixture
def mock_logger():
    """Mock logger to capture log messages"""
    with patch('hidden_regime_mcp.tools.logger') as mock_log:
        yield mock_log


@pytest.fixture
def mock_pipeline():
    """Mock pipeline that returns valid data"""
    pipeline = Mock()

    # Mock data_output
    import pandas as pd
    import numpy as np
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    pipeline.data_output = pd.DataFrame({
        'close': np.random.randn(100) + 100
    }, index=dates)

    # Mock analysis_output
    pipeline.analysis_output = pd.DataFrame({
        'regime_name': ['bull'] * 100,
        'confidence': [0.85] * 100,
        'expected_return': [0.001] * 100,
        'expected_volatility': [0.015] * 100,
        'days_in_regime': list(range(1, 101)),
        'expected_duration': [50] * 100,
        'win_rate': [0.65] * 100,
        'regime_episode': [1] * 100,
        'predicted_state': [0] * 100,
    }, index=dates)

    # Mock model
    pipeline.model = Mock()
    pipeline.model.transition_matrix_ = np.array([
        [0.9, 0.1, 0.0],
        [0.1, 0.8, 0.1],
        [0.0, 0.1, 0.9]
    ])

    pipeline.update = Mock(return_value="Pipeline updated")

    return pipeline


class TestCacheLogging:
    """Test logging for cache operations"""

    @pytest.mark.asyncio
    async def test_detect_regime_cache_hit_logs(self, mock_logger):
        """Test that cache hit is logged"""
        with patch('hidden_regime_mcp.tools.get_cache') as mock_cache_func:
            mock_cache = Mock()
            mock_cache.get.return_value = {"ticker": "SPY", "current_regime": "bull"}
            mock_cache_func.return_value = mock_cache

            await detect_regime("SPY")

            # Verify cache hit was logged
            mock_logger.info.assert_called_with(
                "Cache hit for SPY (regime detection) - returning cached result"
            )

    @pytest.mark.asyncio
    async def test_detect_regime_cache_miss_logs(self, mock_logger, mock_pipeline):
        """Test that cache miss is logged"""
        with patch('hidden_regime_mcp.tools.get_cache') as mock_cache_func, \
             patch('hidden_regime_mcp.tools.create_financial_pipeline', return_value=mock_pipeline):

            mock_cache = Mock()
            mock_cache.get.return_value = None
            mock_cache.set = Mock()
            mock_cache_func.return_value = mock_cache

            await detect_regime("SPY")

            # Verify cache miss was logged
            assert any(
                "Cache miss for SPY - running fresh regime analysis" in str(call)
                for call in mock_logger.info.call_args_list
            )

    @pytest.mark.asyncio
    async def test_regime_statistics_cache_hit_logs(self, mock_logger):
        """Test that statistics cache hit is logged"""
        with patch('hidden_regime_mcp.tools.get_cache') as mock_cache_func:
            mock_cache = Mock()
            mock_cache.get.return_value = {"ticker": "SPY", "regimes": {}}
            mock_cache_func.return_value = mock_cache

            await get_regime_statistics("SPY")

            # Verify cache hit was logged
            mock_logger.info.assert_called_with(
                "Cache hit for SPY (regime statistics) - returning cached result"
            )

    @pytest.mark.asyncio
    async def test_transition_probabilities_cache_hit_logs(self, mock_logger):
        """Test that transition probabilities cache hit is logged"""
        with patch('hidden_regime_mcp.tools.get_cache') as mock_cache_func:
            mock_cache = Mock()
            mock_cache.get.return_value = {"ticker": "SPY", "transition_matrix": {}}
            mock_cache_func.return_value = mock_cache

            await get_transition_probabilities("SPY")

            # Verify cache hit was logged
            mock_logger.info.assert_called_with(
                "Cache hit for SPY (transition probabilities) - returning cached result"
            )


class TestDataLoadingLogging:
    """Test logging for data loading operations"""

    @pytest.mark.asyncio
    async def test_detect_regime_logs_data_fetch(self, mock_logger, mock_pipeline):
        """Test that data fetching is logged"""
        with patch('hidden_regime_mcp.tools.get_cache') as mock_cache_func, \
             patch('hidden_regime_mcp.tools.create_financial_pipeline', return_value=mock_pipeline):

            mock_cache = Mock()
            mock_cache.get.return_value = None
            mock_cache.set = Mock()
            mock_cache_func.return_value = mock_cache

            await detect_regime("AAPL")

            # Verify data fetching was logged
            assert any(
                "Fetching data for AAPL from Yahoo Finance" in str(call)
                for call in mock_logger.info.call_args_list
            )

    @pytest.mark.asyncio
    async def test_detect_regime_logs_data_downloaded(self, mock_logger, mock_pipeline):
        """Test that data download completion is logged with count"""
        with patch('hidden_regime_mcp.tools.get_cache') as mock_cache_func, \
             patch('hidden_regime_mcp.tools.create_financial_pipeline', return_value=mock_pipeline):

            mock_cache = Mock()
            mock_cache.get.return_value = None
            mock_cache.set = Mock()
            mock_cache_func.return_value = mock_cache

            await detect_regime("NVDA")

            # Verify data download completion was logged with observation count
            assert any(
                "Downloaded 100 observations for NVDA" in str(call)
                for call in mock_logger.info.call_args_list
            )


class TestModelTrainingLogging:
    """Test logging for model training operations"""

    @pytest.mark.asyncio
    async def test_detect_regime_logs_model_training(self, mock_logger, mock_pipeline):
        """Test that model training is logged"""
        with patch('hidden_regime_mcp.tools.get_cache') as mock_cache_func, \
             patch('hidden_regime_mcp.tools.create_financial_pipeline', return_value=mock_pipeline):

            mock_cache = Mock()
            mock_cache.get.return_value = None
            mock_cache.set = Mock()
            mock_cache_func.return_value = mock_cache

            await detect_regime("SPY", n_states=3)

            # Verify model training was logged
            assert any(
                "Training 3-state HMM model for SPY" in str(call)
                for call in mock_logger.info.call_args_list
            )

    @pytest.mark.asyncio
    async def test_detect_regime_logs_training_completion(self, mock_logger, mock_pipeline):
        """Test that model training completion is logged"""
        with patch('hidden_regime_mcp.tools.get_cache') as mock_cache_func, \
             patch('hidden_regime_mcp.tools.create_financial_pipeline', return_value=mock_pipeline):

            mock_cache = Mock()
            mock_cache.get.return_value = None
            mock_cache.set = Mock()
            mock_cache_func.return_value = mock_cache

            await detect_regime("QQQ")

            # Verify training completion was logged
            assert any(
                "Model training completed for QQQ" in str(call)
                for call in mock_logger.info.call_args_list
            )

    @pytest.mark.asyncio
    async def test_regime_statistics_logs_model_training(self, mock_logger, mock_pipeline):
        """Test that regime statistics logs model training"""
        with patch('hidden_regime_mcp.tools.get_cache') as mock_cache_func, \
             patch('hidden_regime_mcp.tools.create_financial_pipeline', return_value=mock_pipeline):

            mock_cache = Mock()
            mock_cache.get.return_value = None
            mock_cache.set = Mock()
            mock_cache_func.return_value = mock_cache

            await get_regime_statistics("SPY", n_states=4)

            # Verify model training was logged
            assert any(
                "Training 4-state HMM model for SPY" in str(call)
                for call in mock_logger.info.call_args_list
            )


class TestResultCachingLogging:
    """Test logging for result caching operations"""

    @pytest.mark.asyncio
    async def test_detect_regime_logs_result_cached(self, mock_logger, mock_pipeline):
        """Test that result caching is logged"""
        with patch('hidden_regime_mcp.tools.get_cache') as mock_cache_func, \
             patch('hidden_regime_mcp.tools.create_financial_pipeline', return_value=mock_pipeline):

            mock_cache = Mock()
            mock_cache.get.return_value = None
            mock_cache.set = Mock()
            mock_cache_func.return_value = mock_cache

            await detect_regime("TSLA")

            # Verify result caching was logged
            assert any(
                "Regime detection completed for TSLA - result cached" in str(call)
                for call in mock_logger.info.call_args_list
            )

    @pytest.mark.asyncio
    async def test_regime_statistics_logs_result_cached(self, mock_logger, mock_pipeline):
        """Test that statistics result caching is logged"""
        with patch('hidden_regime_mcp.tools.get_cache') as mock_cache_func, \
             patch('hidden_regime_mcp.tools.create_financial_pipeline', return_value=mock_pipeline):

            mock_cache = Mock()
            mock_cache.get.return_value = None
            mock_cache.set = Mock()
            mock_cache_func.return_value = mock_cache

            await get_regime_statistics("AAPL")

            # Verify result caching was logged
            assert any(
                "Regime statistics computed for AAPL - result cached" in str(call)
                for call in mock_logger.info.call_args_list
            )

    @pytest.mark.asyncio
    async def test_transition_probabilities_logs_result_cached(self, mock_logger, mock_pipeline):
        """Test that transition probabilities result caching is logged"""
        with patch('hidden_regime_mcp.tools.get_cache') as mock_cache_func, \
             patch('hidden_regime_mcp.tools.create_financial_pipeline', return_value=mock_pipeline):

            mock_cache = Mock()
            mock_cache.get.return_value = None
            mock_cache.set = Mock()
            mock_cache_func.return_value = mock_cache

            await get_transition_probabilities("GLD")

            # Verify result caching was logged
            assert any(
                "Transition probabilities computed for GLD - result cached" in str(call)
                for call in mock_logger.info.call_args_list
            )


class TestLoggingWorkflow:
    """Test complete logging workflow"""

    @pytest.mark.asyncio
    async def test_detect_regime_full_logging_sequence(self, mock_logger, mock_pipeline):
        """Test that all expected log messages appear in correct order"""
        with patch('hidden_regime_mcp.tools.get_cache') as mock_cache_func, \
             patch('hidden_regime_mcp.tools.create_financial_pipeline', return_value=mock_pipeline):

            mock_cache = Mock()
            mock_cache.get.return_value = None
            mock_cache.set = Mock()
            mock_cache_func.return_value = mock_cache

            await detect_regime("SPY", n_states=3)

            # Get all info log calls
            log_calls = [str(call) for call in mock_logger.info.call_args_list]

            # Verify expected sequence
            expected_messages = [
                "Cache miss for SPY",
                "Fetching data for SPY",
                "Training 3-state HMM model for SPY",
                "Downloaded",
                "Model training completed",
                "result cached"
            ]

            # Check that all expected messages appear
            for expected in expected_messages:
                assert any(expected in log for log in log_calls), \
                    f"Expected log message containing '{expected}' not found"

    @pytest.mark.asyncio
    async def test_cache_hit_skips_processing_logs(self, mock_logger):
        """Test that cache hit prevents data/training logs"""
        with patch('hidden_regime_mcp.tools.get_cache') as mock_cache_func:
            mock_cache = Mock()
            mock_cache.get.return_value = {"ticker": "SPY", "current_regime": "bull"}
            mock_cache_func.return_value = mock_cache

            await detect_regime("SPY")

            # Get all info log calls
            log_calls = [str(call) for call in mock_logger.info.call_args_list]

            # Verify only cache hit is logged (no fetching/training)
            assert len(log_calls) == 1
            assert "Cache hit" in log_calls[0]
            assert "Fetching" not in str(log_calls)
            assert "Training" not in str(log_calls)


class TestLoggingLevels:
    """Test that appropriate logging levels are used"""

    @pytest.mark.asyncio
    async def test_info_level_for_normal_operations(self, mock_logger, mock_pipeline):
        """Test that INFO level is used for normal operations"""
        with patch('hidden_regime_mcp.tools.get_cache') as mock_cache_func, \
             patch('hidden_regime_mcp.tools.create_financial_pipeline', return_value=mock_pipeline):

            mock_cache = Mock()
            mock_cache.get.return_value = None
            mock_cache.set = Mock()
            mock_cache_func.return_value = mock_cache

            await detect_regime("SPY")

            # All normal operation logs should be at INFO level
            assert mock_logger.info.call_count > 0
            # No warnings or errors for successful operation
            assert mock_logger.warning.call_count == 0
            assert mock_logger.error.call_count == 0
