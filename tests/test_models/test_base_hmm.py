"""
Unit tests for HiddenMarkovModel class.

Tests core functionality including initialization, training, inference,
and persistence operations.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from hidden_regime.models import HiddenMarkovModel, HMMConfig
from hidden_regime.models.base_hmm import HMMTrainingError, HMMConvergenceError, HMMInferenceError


class TestHiddenMarkovModel:
    """Test cases for HiddenMarkovModel class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns data for testing."""
        np.random.seed(42)
        # Create regime-like data
        regime1 = np.random.normal(-0.02, 0.03, 30)  # Bear regime
        regime2 = np.random.normal(0.00, 0.015, 40)  # Sideways regime  
        regime3 = np.random.normal(0.015, 0.025, 30) # Bull regime
        
        returns = np.concatenate([regime1, regime2, regime3])
        return returns
    
    @pytest.fixture
    def sample_returns_series(self, sample_returns):
        """Generate sample returns as pandas Series."""
        return pd.Series(sample_returns, name='log_return')
    
    @pytest.fixture
    def config(self):
        """Generate test configuration."""
        return HMMConfig(n_states=3, max_iterations=50, tolerance=1e-4, random_seed=42)
    
    def test_initialization_default(self):
        """Test default model initialization."""
        hmm = HiddenMarkovModel()
        
        assert hmm.n_states == 3
        assert hmm.config.n_states == 3
        assert hmm.config.max_iterations == 100
        assert not hmm.is_fitted
        assert hmm.initial_probs_ is None
        assert hmm.transition_matrix_ is None
        assert hmm.emission_params_ is None
        assert hmm._current_state_probs is None
        assert hmm._last_observation is None
    
    def test_initialization_custom(self, config):
        """Test custom model initialization."""
        hmm = HiddenMarkovModel(n_states=4, config=config)
        
        # n_states from constructor should take precedence
        assert hmm.n_states == 4
        assert hmm.config.n_states == 4  # Config should be updated
        assert hmm.config.max_iterations == 50
        assert hmm.config.tolerance == 1e-4
        assert hmm.config.random_seed == 42
    
    def test_fit_basic(self, sample_returns, config):
        """Test basic model fitting."""
        hmm = HiddenMarkovModel(n_states=3, config=config)
        
        # Fit should return self
        result = hmm.fit(sample_returns, verbose=False)
        assert result is hmm
        
        # Check that model is fitted
        assert hmm.is_fitted
        assert hmm.initial_probs_ is not None
        assert hmm.transition_matrix_ is not None
        assert hmm.emission_params_ is not None
        
        # Check parameter shapes
        assert hmm.initial_probs_.shape == (3,)
        assert hmm.transition_matrix_.shape == (3, 3)
        assert hmm.emission_params_.shape == (3, 2)
        
        # Check parameter constraints
        assert np.allclose(np.sum(hmm.initial_probs_), 1.0)
        assert np.allclose(np.sum(hmm.transition_matrix_, axis=1), 1.0)
        assert np.all(hmm.emission_params_[:, 1] > 0)  # Positive standard deviations
        
        # Check training history
        assert len(hmm.training_history_['log_likelihoods']) > 0
        assert hmm.training_history_['iterations'] > 0
        assert hmm.training_history_['training_time'] > 0
    
    def test_fit_pandas_series(self, sample_returns_series, config):
        """Test fitting with pandas Series input."""
        hmm = HiddenMarkovModel(n_states=3, config=config)
        hmm.fit(sample_returns_series, verbose=False)
        
        assert hmm.is_fitted
        assert hmm.initial_probs_ is not None
    
    def test_fit_parameter_override(self, sample_returns):
        """Test parameter override during fitting."""
        hmm = HiddenMarkovModel(n_states=3)
        
        hmm.fit(
            sample_returns, 
            max_iterations=25,
            tolerance=1e-3,
            verbose=False
        )
        
        assert hmm.is_fitted
        # Training should have used override values, check training history
        assert hmm.training_history_['iterations'] <= 25
    
    def test_fit_verbose(self, sample_returns, config, capsys):
        """Test verbose training output."""
        hmm = HiddenMarkovModel(n_states=3, config=config)
        hmm.fit(sample_returns, verbose=True)
        
        captured = capsys.readouterr()
        assert "Training 3-state HMM" in captured.out
        assert "Training completed" in captured.out
    
    def test_fit_invalid_data(self):
        """Test fitting with invalid data."""
        hmm = HiddenMarkovModel(n_states=3)
        
        # Empty data
        with pytest.raises(HMMTrainingError, match="Data validation failed"):
            hmm.fit([])
        
        # Data with NaN values - provide enough data points after NaN removal
        bad_data = np.random.normal(0.001, 0.02, 35)  # 35 points to ensure >= 30 after removal
        bad_data[5] = np.nan
        bad_data[10] = np.nan
        bad_data[20] = np.nan
        hmm.fit(bad_data, verbose=False)  # Should work after NaN removal
        
        # Insufficient data
        with pytest.raises(HMMTrainingError, match="Data validation failed"):
            hmm.fit([0.1, 0.2])  # Too few observations
    
    def test_predict_unfitted(self, sample_returns):
        """Test prediction on unfitted model."""
        hmm = HiddenMarkovModel(n_states=3)
        
        with pytest.raises(HMMTrainingError, match="Model must be fitted"):
            hmm.predict(sample_returns)
    
    def test_predict_basic(self, sample_returns, config):
        """Test basic state prediction."""
        hmm = HiddenMarkovModel(n_states=3, config=config)
        hmm.fit(sample_returns, verbose=False)
        
        states = hmm.predict(sample_returns)
        
        assert len(states) == len(sample_returns)
        assert np.all(states >= 0)
        assert np.all(states < 3)
        assert states.dtype == int
    
    def test_predict_proba_basic(self, sample_returns, config):
        """Test state probability prediction."""
        hmm = HiddenMarkovModel(n_states=3, config=config)
        hmm.fit(sample_returns, verbose=False)
        
        probabilities = hmm.predict_proba(sample_returns)
        
        assert probabilities.shape == (len(sample_returns), 3)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.allclose(np.sum(probabilities, axis=1), 1.0)
    
    def test_score_basic(self, sample_returns, config):
        """Test model scoring."""
        hmm = HiddenMarkovModel(n_states=3, config=config)
        hmm.fit(sample_returns, verbose=False)
        
        score = hmm.score(sample_returns)
        
        assert isinstance(score, float)
        assert np.isfinite(score)
        # Note: Log-likelihood can be positive for small datasets with good model fit
        # This is because probability density functions can have values > 1
    
    def test_update_with_observation_unfitted(self):
        """Test online update on unfitted model."""
        hmm = HiddenMarkovModel(n_states=3)
        
        with pytest.raises(HMMTrainingError, match="Model must be fitted"):
            hmm.update_with_observation(0.01)
    
    def test_update_with_observation_basic(self, sample_returns, config):
        """Test basic online observation update."""
        hmm = HiddenMarkovModel(n_states=3, config=config)
        hmm.fit(sample_returns, verbose=False)
        
        # First update
        result = hmm.update_with_observation(0.01)
        
        assert isinstance(result, dict)
        assert 'most_likely_regime' in result
        assert 'regime_probabilities' in result
        assert 'confidence' in result
        assert 'regime_interpretation' in result
        assert 'expected_return' in result
        assert 'expected_volatility' in result
        assert 'last_observation' in result
        
        assert 0 <= result['most_likely_regime'] < 3
        assert 0 <= result['confidence'] <= 1
        assert result['last_observation'] == 0.01
        assert len(result['regime_probabilities']) == 3
        
        # Second update
        result2 = hmm.update_with_observation(-0.005)
        assert result2['last_observation'] == -0.005
        # State probabilities should have changed
        assert result2['regime_probabilities'] != result['regime_probabilities']
    
    def test_update_with_invalid_observation(self, sample_returns, config):
        """Test online update with invalid observation."""
        hmm = HiddenMarkovModel(n_states=3, config=config)
        hmm.fit(sample_returns, verbose=False)
        
        with pytest.raises(HMMInferenceError, match="Online update failed"):
            hmm.update_with_observation(np.nan)
        
        with pytest.raises(HMMInferenceError, match="Online update failed"):
            hmm.update_with_observation(np.inf)
    
    def test_get_current_regime_info_no_state(self, sample_returns, config):
        """Test getting regime info without current state."""
        hmm = HiddenMarkovModel(n_states=3, config=config)
        hmm.fit(sample_returns, verbose=False)
        
        with pytest.raises(HMMInferenceError, match="No current state available"):
            hmm.get_current_regime_info()
    
    def test_analyze_regimes_basic(self, sample_returns, config):
        """Test comprehensive regime analysis."""
        hmm = HiddenMarkovModel(n_states=3, config=config)
        hmm.fit(sample_returns, verbose=False)
        
        analysis = hmm.analyze_regimes(sample_returns)
        
        # Check structure
        assert 'model_info' in analysis
        assert 'regime_parameters' in analysis
        assert 'regime_interpretations' in analysis
        assert 'regime_statistics' in analysis
        assert 'state_sequence' in analysis
        assert 'state_probabilities' in analysis
        
        # Check model info
        assert analysis['model_info']['n_states'] == 3
        assert analysis['model_info']['n_observations'] == len(sample_returns)
        assert 'log_likelihood' in analysis['model_info']
        
        # Check regime parameters
        assert len(analysis['regime_parameters']['initial_probabilities']) == 3
        assert len(analysis['regime_parameters']['transition_matrix']) == 3
        assert len(analysis['regime_parameters']['emission_parameters']) == 3
        
        # Check interpretations
        assert len(analysis['regime_interpretations']) == 3
        
        # Check state sequence and probabilities
        assert len(analysis['state_sequence']) == len(sample_returns)
        assert len(analysis['state_probabilities']) == len(sample_returns)
    
    def test_save_unfitted_model(self):
        """Test saving unfitted model."""
        hmm = HiddenMarkovModel(n_states=3)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            with pytest.raises(HMMTrainingError, match="Cannot save unfitted model"):
                hmm.save(f.name)
    
    def test_save_load_pickle(self, sample_returns, config):
        """Test save/load with pickle format."""
        hmm = HiddenMarkovModel(n_states=3, config=config)
        hmm.fit(sample_returns, verbose=False)
        hmm.update_with_observation(0.01)  # Set current state
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = Path(f.name)
        
        try:
            # Save model
            hmm.save(filepath)
            assert filepath.exists()
            
            # Load model
            loaded_hmm = HiddenMarkovModel.load(filepath)
            
            # Check that model was loaded correctly
            assert loaded_hmm.n_states == hmm.n_states
            assert loaded_hmm.is_fitted == hmm.is_fitted
            assert np.allclose(loaded_hmm.initial_probs_, hmm.initial_probs_)
            assert np.allclose(loaded_hmm.transition_matrix_, hmm.transition_matrix_)
            assert np.allclose(loaded_hmm.emission_params_, hmm.emission_params_)
            assert loaded_hmm.training_history_ == hmm.training_history_
            assert np.allclose(loaded_hmm._current_state_probs, hmm._current_state_probs)
            assert loaded_hmm._last_observation == hmm._last_observation
            
        finally:
            if filepath.exists():
                filepath.unlink()
    
    def test_save_load_json(self, sample_returns, config):
        """Test save/load with JSON format."""
        hmm = HiddenMarkovModel(n_states=3, config=config)
        hmm.fit(sample_returns, verbose=False)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = Path(f.name)
        
        try:
            # Save model
            hmm.save(filepath)
            assert filepath.exists()
            
            # Verify JSON format
            with open(filepath, 'r') as f:
                data = json.load(f)
                assert 'n_states' in data
                assert 'config' in data
                assert 'initial_probs' in data
            
            # Load model
            loaded_hmm = HiddenMarkovModel.load(filepath)
            
            # Check that model was loaded correctly
            assert loaded_hmm.n_states == hmm.n_states
            assert loaded_hmm.is_fitted == hmm.is_fitted
            assert np.allclose(loaded_hmm.initial_probs_, hmm.initial_probs_)
            
        finally:
            if filepath.exists():
                filepath.unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(HMMTrainingError, match="Model file not found"):
            HiddenMarkovModel.load("nonexistent_file.pkl")
    
    def test_reset_state(self, sample_returns, config):
        """Test resetting real-time state."""
        hmm = HiddenMarkovModel(n_states=3, config=config)
        hmm.fit(sample_returns, verbose=False)
        hmm.update_with_observation(0.01)
        
        # Verify state is set
        assert hmm._current_state_probs is not None
        assert hmm._last_observation is not None
        
        # Reset state
        hmm.reset_state()
        
        # Verify state is reset
        assert hmm._current_state_probs is None
        assert hmm._last_observation is None
    
    def test_repr_unfitted(self):
        """Test string representation of unfitted model."""
        hmm = HiddenMarkovModel(n_states=3)
        repr_str = repr(hmm)
        
        assert "HiddenMarkovModel" in repr_str
        assert "n_states=3" in repr_str
        assert "fitted=False" in repr_str
    
    def test_repr_fitted(self, sample_returns, config):
        """Test string representation of fitted model."""
        hmm = HiddenMarkovModel(n_states=3, config=config)
        hmm.fit(sample_returns, verbose=False)
        
        repr_str = repr(hmm)
        
        assert "HiddenMarkovModel" in repr_str
        assert "n_states=3" in repr_str
        assert "fitted=True" in repr_str
        assert "log_likelihood=" in repr_str
    
    def test_training_convergence(self):
        """Test training convergence behavior."""
        # Create data that should converge quickly
        np.random.seed(42)
        easy_data = np.concatenate([
            np.random.normal(-0.05, 0.01, 50),  # Clear bear regime
            np.random.normal(0.05, 0.01, 50)    # Clear bull regime
        ])
        
        config = HMMConfig(n_states=2, max_iterations=100, tolerance=1e-8, random_seed=42)
        hmm = HiddenMarkovModel(n_states=2, config=config)
        
        hmm.fit(easy_data, verbose=False)
        
        # Should converge in reasonable number of iterations
        assert hmm.training_history_['iterations'] < 50
        assert len(hmm.training_history_['log_likelihoods']) > 5
    
    def test_numerical_stability(self, config):
        """Test numerical stability with extreme data."""
        # Create data with extreme values
        extreme_returns = np.array([0.5, -0.5, 0.3, -0.3] * 25)  # 100 observations
        
        hmm = HiddenMarkovModel(n_states=2, config=config)
        
        # Should handle extreme data without numerical issues
        hmm.fit(extreme_returns, verbose=False)
        
        assert hmm.is_fitted
        assert np.all(np.isfinite(hmm.initial_probs_))
        assert np.all(np.isfinite(hmm.transition_matrix_))
        assert np.all(np.isfinite(hmm.emission_params_))
    
    @patch('hidden_regime.models.algorithms.HMMAlgorithms.forward_backward_algorithm')
    def test_training_failure_handling(self, mock_fb, sample_returns, config):
        """Test handling of training failures."""
        # Mock forward_backward_algorithm to raise exception
        mock_fb.side_effect = Exception("Simulated training failure")
        
        hmm = HiddenMarkovModel(n_states=3, config=config)
        
        with pytest.raises(HMMTrainingError, match="Training failed at iteration"):
            hmm.fit(sample_returns, verbose=False)
        
        assert not hmm.is_fitted