"""
Tests for Online Hidden Markov Model implementation.

This module contains comprehensive tests for the OnlineHMM class,
including parameter stability, temporal consistency, streaming
data processing, and change point detection.

Author: aoaustin
Created: 2025-09-03
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch
import warnings

from hidden_regime.models.online_hmm import OnlineHMM, OnlineHMMConfig, SufficientStatistics
from hidden_regime.models.config import HMMConfig
from hidden_regime.utils.exceptions import HMMTrainingError, HMMInferenceError


class TestOnlineHMMConfig:
    """Tests for OnlineHMMConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OnlineHMMConfig()
        assert 0.9 <= config.forgetting_factor <= 0.999
        assert 0.001 <= config.adaptation_rate <= 0.2
        assert config.parameter_smoothing is True
        assert config.rolling_window_size >= 100
        
    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Invalid forgetting factor
        with pytest.raises(ValueError, match="forgetting_factor must be"):
            OnlineHMMConfig(forgetting_factor=0.5)
            
        # Invalid adaptation rate  
        with pytest.raises(ValueError, match="adaptation_rate must be"):
            OnlineHMMConfig(adaptation_rate=0.5)
            
        # Invalid smoothing weight
        with pytest.raises(ValueError, match="smoothing_weight must be"):
            OnlineHMMConfig(smoothing_weight=1.5)
            
        # Invalid rolling window size
        with pytest.raises(ValueError, match="rolling_window_size must be"):
            OnlineHMMConfig(rolling_window_size=50)


class TestSufficientStatistics:
    """Tests for SufficientStatistics class."""
    
    def test_initialization(self):
        """Test sufficient statistics initialization."""
        stats = SufficientStatistics()
        stats.initialize(n_states=3)
        
        assert len(stats.gamma_sum) == 3
        assert len(stats.gamma_sum_t1) == 3
        assert stats.xi_sum.shape == (3, 3)
        assert len(stats.obs_sum) == 3
        assert len(stats.obs_sq_sum) == 3
        assert stats.total_weight == 0.0
        assert stats.n_observations == 0
        
    def test_decay(self):
        """Test exponential decay of statistics."""
        stats = SufficientStatistics()
        stats.initialize(n_states=2)
        
        # Set some initial values
        stats.gamma_sum = np.array([1.0, 2.0])
        stats.xi_sum = np.array([[0.5, 0.5], [0.3, 0.7]])
        stats.total_weight = 10.0
        
        # Apply decay
        decay_factor = 0.9
        stats.decay(decay_factor)
        
        assert np.allclose(stats.gamma_sum, [0.9, 1.8])
        assert np.allclose(stats.xi_sum[0], [0.45, 0.45])
        assert stats.total_weight == 9.0


class TestOnlineHMM:
    """Tests for OnlineHMM class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample return data for testing."""
        np.random.seed(42)
        n_obs = 200
        
        # Generate regime-switching returns
        true_states = np.random.choice([0, 1, 2], size=n_obs, p=[0.3, 0.4, 0.3])
        means = [-0.01, 0.0, 0.01]  # Bear, sideways, bull
        stds = [0.02, 0.015, 0.018]
        
        returns = np.zeros(n_obs)
        for i, state in enumerate(true_states):
            returns[i] = np.random.normal(means[state], stds[state])
            
        return returns, true_states
    
    @pytest.fixture
    def fitted_online_hmm(self, sample_data):
        """Create fitted OnlineHMM for testing."""
        returns, _ = sample_data
        
        # Create and fit online HMM
        online_config = OnlineHMMConfig(
            forgetting_factor=0.95,
            adaptation_rate=0.1,
            min_observations_for_update=5
        )
        
        hmm = OnlineHMM(n_states=3, online_config=online_config)
        hmm.fit(returns[:100])  # Fit on first half of data
        
        return hmm
    
    def test_initialization(self):
        """Test OnlineHMM initialization."""
        online_config = OnlineHMMConfig()
        hmm = OnlineHMM(n_states=3, online_config=online_config)
        
        assert hmm.n_states == 3
        assert hmm.online_config == online_config
        assert not hmm.is_fitted
        assert len(hmm.observation_buffer) == 0
        assert len(hmm.change_points) == 0
        
    def test_fit_initialization(self, sample_data):
        """Test fitting and online component initialization."""
        returns, _ = sample_data
        
        hmm = OnlineHMM(n_states=3)
        hmm.fit(returns[:100], verbose=False)
        
        assert hmm.is_fitted
        assert hmm.sufficient_stats.n_observations > 0
        assert len(hmm.observation_buffer) > 0
        assert len(hmm.parameter_history) > 0
        
    def test_add_observation_basic(self, fitted_online_hmm):
        """Test basic observation processing."""
        hmm = fitted_online_hmm
        
        # Add a single observation
        result = hmm.add_observation(0.01, update_parameters=False)
        
        assert 'regime' in result
        assert 'regime_probabilities' in result
        assert 'confidence' in result
        assert 'diagnostics' in result
        assert 0 <= result['regime'] < hmm.n_states
        assert 0 <= result['confidence'] <= 1
        
    def test_add_observation_with_updates(self, fitted_online_hmm, sample_data):
        """Test observation processing with parameter updates."""
        hmm = fitted_online_hmm
        returns, _ = sample_data
        
        # Store initial parameters
        initial_transition = hmm.transition_matrix_.copy()
        
        # Add several observations to trigger updates
        for i in range(10):
            result = hmm.add_observation(returns[100 + i], update_parameters=True)
            
        # Check that parameters have been updated
        assert not np.array_equal(initial_transition, hmm.transition_matrix_)
        assert hmm._total_observations_processed == 10
        
    def test_parameter_smoothing(self, fitted_online_hmm):
        """Test parameter smoothing mechanism."""
        hmm = fitted_online_hmm
        
        # Store initial parameters
        initial_means = [params[0] for params in hmm.emission_params_]
        
        # Add observations with large changes to test smoothing
        extreme_return = 0.1  # Very large return
        for _ in range(20):
            hmm.add_observation(extreme_return, update_parameters=True)
            
        # Parameters should change but be smoothed
        new_means = [params[0] for params in hmm.emission_params_]
        changes = [abs(new - old) for new, old in zip(new_means, initial_means)]
        
        # Changes should be moderate (smoothed), not extreme
        assert all(change < 0.05 for change in changes)  # Reasonable bounds
        
    def test_change_point_detection(self, fitted_online_hmm):
        """Test structural break detection."""
        hmm = fitted_online_hmm
        
        # First, add normal observations
        for _ in range(20):
            hmm.add_observation(np.random.normal(0.001, 0.015))
            
        initial_change_points = len(hmm.change_points)
        
        # Add observations from very different regime
        for _ in range(30):
            hmm.add_observation(np.random.normal(-0.05, 0.03))  # Crisis-like regime
            
        # Should detect change point
        assert len(hmm.change_points) > initial_change_points
        
    def test_memory_management(self, fitted_online_hmm):
        """Test memory management with rolling buffers."""
        hmm = fitted_online_hmm
        max_buffer_size = hmm.online_config.rolling_window_size
        
        # Add more observations than buffer size
        for i in range(max_buffer_size + 100):
            hmm.add_observation(np.random.normal(0, 0.02))
            
        # Buffer should be at maximum size
        assert len(hmm.observation_buffer) == max_buffer_size
        assert len(hmm.recent_returns) <= hmm.online_config.change_detection_window
        
    def test_parameter_evolution_tracking(self, fitted_online_hmm, sample_data):
        """Test tracking of parameter evolution over time."""
        hmm = fitted_online_hmm
        returns, _ = sample_data
        
        initial_history_length = len(hmm.parameter_history)
        
        # Add observations to trigger parameter updates
        for i in range(20):
            hmm.add_observation(returns[100 + i], update_parameters=True)
            
        # Should have more parameter snapshots
        assert len(hmm.parameter_history) > initial_history_length
        
        # Get parameter evolution DataFrame
        evolution = hmm.get_parameter_evolution()
        assert not evolution.empty
        assert 'observation_count' in evolution.columns
        
        # Check that we have columns for each state
        for k in range(hmm.n_states):
            assert f'state_{k}_mean' in evolution.columns
            assert f'state_{k}_std' in evolution.columns
            
    def test_temporal_consistency(self, fitted_online_hmm, sample_data):
        """Test that historical regime labels remain stable."""
        hmm = fitted_online_hmm
        returns, _ = sample_data
        
        # Get initial predictions for historical data
        historical_returns = returns[:50]
        initial_states = hmm.predict(historical_returns)
        
        # Add new observations
        for i in range(50):
            hmm.add_observation(returns[100 + i], update_parameters=True)
            
        # Re-predict historical data
        updated_states = hmm.predict(historical_returns)
        
        # Most predictions should remain stable (allow some changes due to learning)
        stability_rate = np.mean(initial_states == updated_states)
        assert stability_rate >= 0.7  # At least 70% should remain stable
        
    def test_reset_adaptation(self, fitted_online_hmm):
        """Test resetting online learning components."""
        hmm = fitted_online_hmm
        
        # Add some observations to populate buffers
        for _ in range(20):
            hmm.add_observation(np.random.normal(0, 0.02))
            
        # Verify buffers have data
        assert len(hmm.observation_buffer) > 0
        assert hmm._total_observations_processed > 0
        
        # Reset adaptation
        hmm.reset_adaptation()
        
        # Buffers should be empty
        assert len(hmm.observation_buffer) == 0
        assert len(hmm.recent_returns) == 0
        assert len(hmm.parameter_history) == 0
        assert hmm._total_observations_processed == 0
        
        # Model should still be fitted
        assert hmm.is_fitted
        
    def test_error_handling(self, fitted_online_hmm):
        """Test error handling for invalid observations."""
        hmm = fitted_online_hmm
        
        # Test invalid observations
        with pytest.raises(HMMInferenceError):
            hmm.add_observation(np.inf)
            
        with pytest.raises(HMMInferenceError):
            hmm.add_observation(np.nan)
            
    def test_regime_interpretation(self, fitted_online_hmm):
        """Test regime interpretation functionality."""
        hmm = fitted_online_hmm
        
        # Test different types of returns
        bull_result = hmm.add_observation(0.02)  # Strong positive return
        bear_result = hmm.add_observation(-0.03)  # Strong negative return
        sideways_result = hmm.add_observation(0.001)  # Small return
        
        # All should have valid interpretations
        assert bull_result['regime_interpretation'] is not None
        assert bear_result['regime_interpretation'] is not None
        assert sideways_result['regime_interpretation'] is not None
        
    def test_performance_under_load(self, fitted_online_hmm):
        """Test performance with many observations."""
        hmm = fitted_online_hmm
        
        import time
        
        # Time processing of many observations
        n_obs = 1000
        start_time = time.time()
        
        for i in range(n_obs):
            hmm.add_observation(np.random.normal(0, 0.02))
            
        processing_time = time.time() - start_time
        avg_time_per_obs = processing_time / n_obs
        
        # Should process observations reasonably quickly (< 10ms per observation)
        # Note: This is more lenient as we're doing more complex online learning
        assert avg_time_per_obs < 0.01
        
    def test_repr(self, fitted_online_hmm):
        """Test string representation."""
        hmm = fitted_online_hmm
        
        # Add some observations
        for _ in range(5):
            hmm.add_observation(np.random.normal(0, 0.02))
            
        repr_str = repr(hmm)
        assert "OnlineHMM" in repr_str
        assert "fitted" in repr_str
        assert f"n_states={hmm.n_states}" in repr_str


class TestOnlineHMMIntegration:
    """Integration tests for OnlineHMM with real market data patterns."""
    
    def test_bull_bear_transition(self):
        """Test detection of bull to bear market transition."""
        np.random.seed(42)
        
        # Generate bull market data
        bull_returns = np.random.normal(0.008, 0.015, 100)  # 0.8% daily return
        
        # Generate bear market data  
        bear_returns = np.random.normal(-0.012, 0.025, 100)  # -1.2% daily return
        
        # Combine data
        all_returns = np.concatenate([bull_returns, bear_returns])
        
        # Train online HMM
        hmm = OnlineHMM(n_states=3)
        hmm.fit(bull_returns)  # Initial training on bull market
        
        # Process bear market data online
        regime_sequence = []
        for return_val in bear_returns:
            result = hmm.add_observation(return_val, update_parameters=True)
            regime_sequence.append(result['regime'])
            
        # Should eventually detect regime change
        # (Later observations should be in different regime than earlier ones)
        early_regimes = set(regime_sequence[:20])
        late_regimes = set(regime_sequence[-20:])
        
        # Should see regime diversity (detecting change)
        assert len(early_regimes.union(late_regimes)) >= 2
        
    def test_volatility_clustering(self):
        """Test detection of volatility clustering."""
        np.random.seed(42)
        
        # Generate data with volatility clustering
        returns = []
        volatilities = []
        
        # Low volatility period
        for _ in range(100):
            vol = 0.01
            returns.append(np.random.normal(0.001, vol))
            volatilities.append(vol)
            
        # High volatility period
        for _ in range(100):
            vol = 0.04
            returns.append(np.random.normal(0.001, vol))
            volatilities.append(vol)
            
        returns = np.array(returns)
        
        # Train online HMM
        hmm = OnlineHMM(n_states=3)
        hmm.fit(returns[:50])
        
        # Process remaining data
        detected_volatilities = []
        for return_val in returns[50:]:
            result = hmm.add_observation(return_val, update_parameters=True)
            # Get volatility of current regime
            regime_volatility = result['regime_characteristics']['volatility']
            detected_volatilities.append(regime_volatility)
            
        # Should detect volatility increase in second half
        first_half_vol = np.mean(detected_volatilities[:25])
        second_half_vol = np.mean(detected_volatilities[-25:])
        
        assert second_half_vol > first_half_vol * 1.5  # Significant increase
        
    def test_multiple_regime_changes(self):
        """Test handling of multiple regime changes."""
        np.random.seed(42)
        
        # Generate data with multiple regime changes
        regimes_data = [
            (np.random.normal(-0.01, 0.02, 50), "bear"),    # Bear market
            (np.random.normal(0.012, 0.018, 50), "bull"),   # Bull market  
            (np.random.normal(0.001, 0.012, 50), "sideways"), # Sideways
            (np.random.normal(-0.015, 0.03, 50), "crisis")   # Crisis
        ]
        
        all_returns = np.concatenate([data for data, _ in regimes_data])
        
        # Train online HMM
        hmm = OnlineHMM(n_states=4, online_config=OnlineHMMConfig(
            enable_change_detection=True,
            change_detection_threshold=2.0
        ))
        hmm.fit(all_returns[:50])  # Initial training with more data
        
        # Process remaining data
        change_points_detected = 0
        for i, return_val in enumerate(all_returns[50:]):
            result = hmm.add_observation(return_val, update_parameters=True)
            if result['diagnostics']['change_detected']:
                change_points_detected += 1
                
        # Should detect multiple change points
        assert change_points_detected >= 2
        assert len(hmm.change_points) >= 2