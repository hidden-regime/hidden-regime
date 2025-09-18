"""
Unit tests for base observation components.

Tests the observation generation functionality that transforms raw financial
data into features suitable for hidden Markov model training and inference.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from hidden_regime.observations.base import BaseObservationGenerator
from hidden_regime.utils.exceptions import ValidationError


class TestBaseObservationGenerator:
    """Test cases for BaseObservationGenerator."""
    
    def test_base_observation_generator_initialization(self):
        """Test BaseObservationGenerator initialization."""
        obs_gen = BaseObservationGenerator(window_size=20)
        
        assert obs_gen.window_size == 20
        assert obs_gen.features == []
        assert obs_gen._cache == {}
    
    def test_base_observation_generator_default_window_size(self):
        """Test default window size."""
        obs_gen = BaseObservationGenerator()
        assert obs_gen.window_size == 10
    
    def test_invalid_window_size(self):
        """Test invalid window size handling."""
        with pytest.raises(ValueError, match="Window size must be positive"):
            BaseObservationGenerator(window_size=0)
        
        with pytest.raises(ValueError, match="Window size must be positive"):
            BaseObservationGenerator(window_size=-5)
    
    def test_update_with_insufficient_data(self):
        """Test update with insufficient data for window size."""
        obs_gen = BaseObservationGenerator(window_size=20)
        
        # Create data with fewer rows than window size
        data = pd.DataFrame({
            'close': [100, 101, 99, 102, 98]
        }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
        
        with pytest.raises(ValidationError, match="Insufficient data"):
            obs_gen.update(data)
    
    def test_update_with_valid_data(self):
        """Test update with valid data."""
        obs_gen = BaseObservationGenerator(window_size=5)
        
        # Create sufficient data
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'open': np.random.uniform(95, 105, 10),
            'high': np.random.uniform(100, 110, 10),
            'low': np.random.uniform(90, 100, 10),
            'close': np.random.uniform(95, 105, 10),
            'volume': np.random.randint(1000000, 5000000, 10)
        }, index=dates)
        
        result = obs_gen.update(data)
        
        # Should return basic log returns
        assert isinstance(result, pd.DataFrame)
        assert 'log_return' in result.columns
        assert len(result) == len(data) - 1  # One less due to return calculation
        
        # Verify log return calculation
        expected_returns = np.log(data['close'] / data['close'].shift(1)).dropna()
        np.testing.assert_array_almost_equal(
            result['log_return'].values,
            expected_returns.values
        )
    
    def test_calculate_log_returns(self):
        """Test log return calculation."""
        obs_gen = BaseObservationGenerator()
        
        prices = pd.Series([100, 101, 99, 102, 98], name='close')
        log_returns = obs_gen._calculate_log_returns(prices)
        
        expected = np.log(prices / prices.shift(1)).dropna()
        np.testing.assert_array_almost_equal(log_returns.values, expected.values)
        assert log_returns.name == 'log_return'
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        obs_gen = BaseObservationGenerator(window_size=5)
        
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005, 0.02])
        volatility = obs_gen._calculate_volatility(returns, window=3)
        
        assert isinstance(volatility, pd.Series)
        assert volatility.name == 'volatility'
        assert len(volatility) == len(returns)
        
        # First few values should be NaN due to window
        assert pd.isna(volatility.iloc[0])
        assert pd.isna(volatility.iloc[1])
        
        # Later values should be calculated
        assert not pd.isna(volatility.iloc[-1])
    
    def test_add_feature_validation(self):
        """Test feature addition validation."""
        obs_gen = BaseObservationGenerator()
        
        # Valid feature
        obs_gen.add_feature('volatility', window=10)
        assert 'volatility' in obs_gen.features
        
        # Duplicate feature
        with pytest.raises(ValueError, match="Feature 'volatility' already exists"):
            obs_gen.add_feature('volatility', window=5)
        
        # Invalid feature name
        with pytest.raises(ValueError, match="Unknown feature type"):
            obs_gen.add_feature('invalid_feature')
    
    def test_remove_feature(self):
        """Test feature removal."""
        obs_gen = BaseObservationGenerator()
        
        obs_gen.add_feature('volatility', window=10)
        assert 'volatility' in obs_gen.features
        
        obs_gen.remove_feature('volatility')
        assert 'volatility' not in obs_gen.features
        
        # Removing non-existent feature should not raise error
        obs_gen.remove_feature('non_existent')
    
    def test_generate_features_with_multiple_features(self):
        """Test feature generation with multiple features."""
        obs_gen = BaseObservationGenerator(window_size=10)
        obs_gen.add_feature('volatility', window=5)
        obs_gen.add_feature('momentum', window=3)
        
        # Create test data
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.normal(0, 1, 20))
        }, index=dates)
        
        result = obs_gen.update(data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'log_return' in result.columns
        assert 'volatility' in result.columns
        assert 'momentum' in result.columns
        
        # Check that all features have reasonable values
        assert not result['log_return'].isna().all()
        assert not result['volatility'].isna().all()
        assert not result['momentum'].isna().all()
    
    def test_caching_behavior(self):
        """Test caching of intermediate calculations."""
        obs_gen = BaseObservationGenerator(window_size=5)
        
        # Create test data
        data = pd.DataFrame({
            'close': [100, 101, 99, 102, 98, 105, 103]
        }, index=pd.date_range('2024-01-01', periods=7, freq='D'))
        
        # First call should populate cache
        result1 = obs_gen.update(data)
        assert len(obs_gen._cache) > 0
        
        # Second call with same data should use cache
        result2 = obs_gen.update(data)
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_update_incremental_data(self):
        """Test incremental data updates."""
        obs_gen = BaseObservationGenerator(window_size=5)
        
        # Initial data
        initial_data = pd.DataFrame({
            'close': [100, 101, 99, 102, 98, 105]
        }, index=pd.date_range('2024-01-01', periods=6, freq='D'))
        
        result1 = obs_gen.update(initial_data)
        
        # Add new data point
        new_data = pd.DataFrame({
            'close': [100, 101, 99, 102, 98, 105, 103]
        }, index=pd.date_range('2024-01-01', periods=7, freq='D'))
        
        result2 = obs_gen.update(new_data)
        
        # Results should be consistent for overlapping period
        common_index = result1.index.intersection(result2.index)
        pd.testing.assert_frame_equal(
            result1.loc[common_index],
            result2.loc[common_index]
        )
        
        # New result should have one more row
        assert len(result2) == len(result1) + 1
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        obs_gen = BaseObservationGenerator(window_size=5)
        
        # Create data with missing values
        data = pd.DataFrame({
            'close': [100, np.nan, 99, 102, np.nan, 105]
        }, index=pd.date_range('2024-01-01', periods=6, freq='D'))
        
        with pytest.raises(ValidationError, match="Data contains missing values"):
            obs_gen.update(data)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        obs_gen = BaseObservationGenerator()
        
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValidationError, match="Data is empty"):
            obs_gen.update(empty_data)
    
    def test_plot_functionality(self):
        """Test plotting functionality."""
        obs_gen = BaseObservationGenerator(window_size=5)
        obs_gen.add_feature('volatility', window=3)
        
        # Create and process data
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.normal(0, 1, 10))
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
        
        observations = obs_gen.update(data)
        
        # Test plotting
        fig = obs_gen.plot(observations=observations)
        
        assert fig is not None
        assert len(fig.axes) >= 1  # Should have at least one subplot
        
        # Test with specific features
        fig = obs_gen.plot(observations=observations, features=['log_return'])
        assert fig is not None
        
        # Test with invalid features
        with pytest.raises(ValueError, match="Feature 'invalid' not found"):
            obs_gen.plot(observations=observations, features=['invalid'])
    
    def test_serialization_support(self):
        """Test pickle serialization support."""
        import pickle
        
        obs_gen = BaseObservationGenerator(window_size=15)
        obs_gen.add_feature('volatility', window=5)
        
        # Serialize and deserialize
        serialized = pickle.dumps(obs_gen)
        deserialized = pickle.loads(serialized)
        
        assert deserialized.window_size == obs_gen.window_size
        assert deserialized.features == obs_gen.features
        
        # Test functionality after deserialization
        data = pd.DataFrame({
            'close': [100, 101, 99, 102, 98, 105, 103, 101, 99, 104, 
                     102, 100, 98, 105, 107, 106]
        }, index=pd.date_range('2024-01-01', periods=16, freq='D'))
        
        result = deserialized.update(data)
        assert isinstance(result, pd.DataFrame)
        assert 'log_return' in result.columns
        assert 'volatility' in result.columns


if __name__ == "__main__":
    pytest.main([__file__])