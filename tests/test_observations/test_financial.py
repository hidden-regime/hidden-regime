"""
Unit tests for FinancialObservationGenerator.

Tests the financial-specific observation generation functionality that creates
technical indicators and market-specific features from OHLCV data.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from hidden_regime.observations.financial import FinancialObservationGenerator
from hidden_regime.utils.exceptions import ValidationError


class TestFinancialObservationGenerator:
    """Test cases for FinancialObservationGenerator."""
    
    def create_sample_ohlcv_data(self, n_periods=50):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2024-01-01', periods=n_periods, freq='D')
        
        # Generate realistic OHLCV data
        np.random.seed(42)
        close_prices = 100 + np.cumsum(np.random.normal(0, 1, n_periods))
        
        data = pd.DataFrame({
            'open': close_prices + np.random.normal(0, 0.5, n_periods),
            'high': close_prices + np.abs(np.random.normal(0, 1, n_periods)),
            'low': close_prices - np.abs(np.random.normal(0, 1, n_periods)),
            'close': close_prices,
            'volume': np.random.randint(1000000, 10000000, n_periods)
        }, index=dates)
        
        # Ensure OHLC relationships
        data['high'] = np.maximum.reduce([data['open'], data['high'], data['low'], data['close']])
        data['low'] = np.minimum.reduce([data['open'], data['high'], data['low'], data['close']])
        
        return data
    
    def test_financial_observation_generator_initialization(self):
        """Test FinancialObservationGenerator initialization."""
        obs_gen = FinancialObservationGenerator(
            rsi_period=14,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            bb_period=20,
            bb_std=2.0
        )
        
        assert obs_gen.rsi_period == 14
        assert obs_gen.macd_fast == 12
        assert obs_gen.macd_slow == 26
        assert obs_gen.macd_signal == 9
        assert obs_gen.bb_period == 20
        assert obs_gen.bb_std == 2.0
    
    def test_financial_observation_generator_defaults(self):
        """Test default parameters."""
        obs_gen = FinancialObservationGenerator()
        
        assert obs_gen.rsi_period == 14
        assert obs_gen.macd_fast == 12
        assert obs_gen.macd_slow == 26
        assert obs_gen.macd_signal == 9
        assert obs_gen.bb_period == 20
        assert obs_gen.bb_std == 2.0
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid RSI period
        with pytest.raises(ValueError, match="RSI period must be positive"):
            FinancialObservationGenerator(rsi_period=0)
        
        # Invalid MACD parameters
        with pytest.raises(ValueError, match="MACD fast period must be positive"):
            FinancialObservationGenerator(macd_fast=0)
        
        with pytest.raises(ValueError, match="MACD slow period must be greater than fast"):
            FinancialObservationGenerator(macd_fast=26, macd_slow=12)
        
        # Invalid Bollinger Band parameters
        with pytest.raises(ValueError, match="Bollinger Band period must be positive"):
            FinancialObservationGenerator(bb_period=0)
        
        with pytest.raises(ValueError, match="Bollinger Band standard deviations must be positive"):
            FinancialObservationGenerator(bb_std=0)
    
    def test_update_with_complete_features(self):
        """Test update with all financial features enabled."""
        obs_gen = FinancialObservationGenerator()
        data = self.create_sample_ohlcv_data(60)  # Enough data for all indicators
        
        result = obs_gen.update(data)
        
        assert isinstance(result, pd.DataFrame)
        
        # Check basic features
        assert 'log_return' in result.columns
        assert 'volatility' in result.columns
        
        # Check technical indicators
        assert 'rsi' in result.columns
        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_histogram' in result.columns
        assert 'bb_upper' in result.columns
        assert 'bb_middle' in result.columns
        assert 'bb_lower' in result.columns
        assert 'bb_width' in result.columns
        assert 'bb_position' in result.columns
        
        # Check price-based features
        assert 'price_momentum' in result.columns
        assert 'volume_momentum' in result.columns
        
        # Verify no all-NaN columns
        for col in result.columns:
            assert not result[col].isna().all(), f"Column {col} is all NaN"
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        obs_gen = FinancialObservationGenerator(rsi_period=14)
        data = self.create_sample_ohlcv_data(30)
        
        rsi = obs_gen._calculate_rsi(data['close'])
        
        assert isinstance(rsi, pd.Series)
        assert rsi.name == 'rsi'
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
        
        # First 14 values should be NaN due to RSI period
        assert rsi.iloc[:13].isna().all()
        assert not rsi.iloc[14:].isna().all()
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        obs_gen = FinancialObservationGenerator(macd_fast=12, macd_slow=26, macd_signal=9)
        data = self.create_sample_ohlcv_data(50)
        
        macd_data = obs_gen._calculate_macd(data['close'])
        
        assert isinstance(macd_data, dict)
        assert 'macd' in macd_data
        assert 'macd_signal' in macd_data
        assert 'macd_histogram' in macd_data
        
        # Check that all series have same length
        assert len(macd_data['macd']) == len(data)
        assert len(macd_data['macd_signal']) == len(data)
        assert len(macd_data['macd_histogram']) == len(data)
        
        # Histogram should be MACD - Signal
        macd_line = macd_data['macd'].dropna()
        signal_line = macd_data['macd_signal'].dropna()
        histogram = macd_data['macd_histogram'].dropna()
        
        common_index = macd_line.index.intersection(signal_line.index).intersection(histogram.index)
        if len(common_index) > 0:
            np.testing.assert_array_almost_equal(
                histogram.loc[common_index].values,
                (macd_line.loc[common_index] - signal_line.loc[common_index]).values
            )
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        obs_gen = FinancialObservationGenerator(bb_period=20, bb_std=2.0)
        data = self.create_sample_ohlcv_data(40)
        
        bb_data = obs_gen._calculate_bollinger_bands(data['close'])
        
        assert isinstance(bb_data, dict)
        assert 'bb_upper' in bb_data
        assert 'bb_middle' in bb_data
        assert 'bb_lower' in bb_data
        assert 'bb_width' in bb_data
        assert 'bb_position' in bb_data
        
        # Check relationships
        upper = bb_data['bb_upper'].dropna()
        middle = bb_data['bb_middle'].dropna()
        lower = bb_data['bb_lower'].dropna()
        
        common_index = upper.index.intersection(middle.index).intersection(lower.index)
        if len(common_index) > 0:
            # Upper should be >= Middle >= Lower
            assert (upper.loc[common_index] >= middle.loc[common_index]).all()
            assert (middle.loc[common_index] >= lower.loc[common_index]).all()
        
        # Position should be between 0 and 1 when price is between bands
        position = bb_data['bb_position'].dropna()
        # Most positions should be reasonable (allowing some outliers)
        reasonable_positions = position[(position >= -0.5) & (position <= 1.5)]
        assert len(reasonable_positions) > len(position) * 0.8
    
    def test_price_momentum_calculation(self):
        """Test price momentum calculation."""
        obs_gen = FinancialObservationGenerator()
        
        # Create data with clear trend
        prices = pd.Series([100, 101, 102, 103, 104, 105], 
                          index=pd.date_range('2024-01-01', periods=6, freq='D'))
        
        momentum = obs_gen._calculate_price_momentum(prices, window=3)
        
        assert isinstance(momentum, pd.Series)
        assert momentum.name == 'price_momentum'
        
        # Should have NaN for first few values
        assert momentum.iloc[:2].isna().all()
        
        # Later values should show positive momentum for uptrend
        assert momentum.iloc[-1] > 0
    
    def test_volume_momentum_calculation(self):
        """Test volume momentum calculation."""
        obs_gen = FinancialObservationGenerator()
        
        # Create volume data
        volume = pd.Series([1000000, 1200000, 1100000, 1300000, 1400000], 
                          index=pd.date_range('2024-01-01', periods=5, freq='D'))
        
        vol_momentum = obs_gen._calculate_volume_momentum(volume, window=3)
        
        assert isinstance(vol_momentum, pd.Series)
        assert vol_momentum.name == 'volume_momentum'
        
        # Should have NaN for first few values
        assert vol_momentum.iloc[:1].isna().all()
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        obs_gen = FinancialObservationGenerator()
        
        # Create data with insufficient length for MACD slow period (26)
        data = self.create_sample_ohlcv_data(20)
        
        with pytest.raises(ValidationError, match="Insufficient data"):
            obs_gen.update(data)
    
    def test_missing_ohlcv_columns(self):
        """Test handling of missing OHLCV columns."""
        obs_gen = FinancialObservationGenerator()
        
        # Missing volume column
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D'))
        
        with pytest.raises(ValidationError, match="Required columns missing"):
            obs_gen.update(data)
    
    def test_selective_feature_generation(self):
        """Test selective feature generation."""
        obs_gen = FinancialObservationGenerator()
        data = self.create_sample_ohlcv_data(60)
        
        # Generate only RSI
        result = obs_gen.update(data, features=['rsi'])
        
        assert 'rsi' in result.columns
        assert 'log_return' in result.columns  # Always included
        
        # MACD features should not be present
        assert 'macd' not in result.columns
        assert 'bb_upper' not in result.columns
    
    def test_feature_normalization(self):
        """Test feature normalization options."""
        obs_gen = FinancialObservationGenerator()
        data = self.create_sample_ohlcv_data(60)
        
        # Test with normalization
        result_normalized = obs_gen.update(data, normalize_features=True)
        
        # Test without normalization  
        result_raw = obs_gen.update(data, normalize_features=False)
        
        # Normalized features should have different scale
        assert not np.allclose(
            result_normalized['rsi'].dropna().values,
            result_raw['rsi'].dropna().values
        )
        
        # RSI normalized should be between -1 and 1 (approximately)
        normalized_rsi = result_normalized['rsi'].dropna()
        assert normalized_rsi.min() >= -3  # Allow some outliers
        assert normalized_rsi.max() <= 3
    
    def test_outlier_handling(self):
        """Test handling of price outliers."""
        obs_gen = FinancialObservationGenerator()
        data = self.create_sample_ohlcv_data(40)
        
        # Introduce extreme outlier
        data.loc[data.index[20], 'close'] = data.loc[data.index[20], 'close'] * 10
        
        # Should handle outliers gracefully
        result = obs_gen.update(data, handle_outliers=True)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.isna().all().any()  # No completely NaN columns
    
    def test_incremental_updates(self):
        """Test incremental data updates."""
        obs_gen = FinancialObservationGenerator()
        
        # Initial data
        initial_data = self.create_sample_ohlcv_data(40)
        result1 = obs_gen.update(initial_data)
        
        # Extended data
        extended_data = self.create_sample_ohlcv_data(45)
        result2 = obs_gen.update(extended_data)
        
        # Results should be consistent for overlapping period
        common_index = result1.index.intersection(result2.index)
        if len(common_index) > 10:  # Only test if significant overlap
            # Allow small numerical differences
            pd.testing.assert_frame_equal(
                result1.loc[common_index],
                result2.loc[common_index],
                rtol=1e-10, atol=1e-10
            )
    
    def test_plot_functionality(self):
        """Test plotting functionality."""
        obs_gen = FinancialObservationGenerator()
        data = self.create_sample_ohlcv_data(60)
        
        observations = obs_gen.update(data)
        
        # Test basic plotting
        fig = obs_gen.plot(observations=observations)
        assert fig is not None
        
        # Test plotting specific indicators
        fig = obs_gen.plot(observations=observations, indicators=['rsi', 'macd'])
        assert fig is not None
        
        # Should have multiple subplots for different indicators
        assert len(fig.axes) >= 2
    
    def test_performance_with_large_dataset(self):
        """Test performance with larger dataset."""
        import time
        
        obs_gen = FinancialObservationGenerator()
        data = self.create_sample_ohlcv_data(1000)  # Large dataset
        
        start_time = time.time()
        result = obs_gen.update(data)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 5.0  # Less than 5 seconds
        
        # Should produce valid results
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 900  # Most rows should be valid
        assert len(result.columns) > 10  # Should have many features
    
    def test_caching_mechanism(self):
        """Test caching of expensive calculations."""
        obs_gen = FinancialObservationGenerator()
        data = self.create_sample_ohlcv_data(60)
        
        # First calculation
        start_time = time.time()
        result1 = obs_gen.update(data)
        first_time = time.time() - start_time
        
        # Second calculation (should be faster due to caching)
        start_time = time.time()
        result2 = obs_gen.update(data)
        second_time = time.time() - start_time
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
        
        # Second call should be faster (allowing for some variation)
        assert second_time <= first_time * 1.5


if __name__ == "__main__":
    pytest.main([__file__])