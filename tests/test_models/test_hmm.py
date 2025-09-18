"""
Unit tests for HiddenMarkovModel.

Tests the core hidden Markov model implementation including parameter estimation,
state inference, model persistence, and integration with the pipeline framework.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import pickle
import tempfile
import os

from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.config.model import HMMConfig
from hidden_regime.utils.exceptions import ValidationError, HMMTrainingError, HMMInferenceError


class TestHiddenMarkovModel:
    """Test cases for HiddenMarkovModel."""
    
    def create_sample_observations(self, n_samples=100, n_features=2):
        """Create sample observations for testing."""
        np.random.seed(42)
        
        # Generate regime-switching data
        regime_lengths = [30, 40, 30]  # Three regimes
        regimes = []
        for i, length in enumerate(regime_lengths):
            regimes.extend([i] * length)
        
        observations = []
        for regime in regimes:
            if regime == 0:  # Bear regime
                obs = np.random.multivariate_normal([-0.02, 0.03], 
                                                   [[0.001, 0.0002], [0.0002, 0.0015]], 
                                                   size=1)[0]
            elif regime == 1:  # Sideways regime
                obs = np.random.multivariate_normal([0.001, 0.015], 
                                                   [[0.0005, 0.0001], [0.0001, 0.001]], 
                                                   size=1)[0]
            else:  # Bull regime
                obs = np.random.multivariate_normal([0.015, 0.02], 
                                                   [[0.0008, 0.0003], [0.0003, 0.002]], 
                                                   size=1)[0]
            observations.append(obs)
        
        return pd.DataFrame(
            observations, 
            columns=['log_return', 'volatility'],
            index=pd.date_range('2024-01-01', periods=len(observations), freq='D')
        )
    
    def test_hmm_initialization_default(self):
        """Test HMM initialization with default parameters."""
        config = HMMConfig(n_states=3)
        model = HiddenMarkovModel(config)
        
        assert model.config.n_states == 3
        assert model.config.max_iterations == 100
        assert not model.is_fitted
        
        # Internal state should be initialized
        assert model.transition_matrix is None
        assert model.means is None
        assert model.covariances is None
        assert model.initial_probs is None
        assert model.converged is False
    
    def test_hmm_initialization_custom(self):
        """Test HMM initialization with custom parameters."""
        model = HiddenMarkovModel(
            n_states=4,
            max_iter=200,
            tol=1e-8,
            random_state=123,
            covariance_type='diag',
            min_covar=1e-5
        )
        
        assert model.n_states == 4
        assert model.max_iter == 200
        assert model.tol == 1e-8
        assert model.random_state == 123
        assert model.covariance_type == 'diag'
        assert model.min_covar == 1e-5
    
    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Invalid n_states
        with pytest.raises(ValueError, match="Number of states must be at least 2"):
            HiddenMarkovModel(n_states=1)
        
        # Invalid max_iter
        with pytest.raises(ValueError, match="Maximum iterations must be positive"):
            HiddenMarkovModel(max_iter=0)
        
        # Invalid tolerance
        with pytest.raises(ValueError, match="Tolerance must be positive"):
            HiddenMarkovModel(tol=0)
        
        # Invalid covariance type
        with pytest.raises(ValueError, match="Invalid covariance type"):
            HiddenMarkovModel(covariance_type='invalid')
    
    def test_fit_basic_functionality(self):
        """Test basic fitting functionality."""
        model = HiddenMarkovModel(n_states=3, random_state=42)
        observations = self.create_sample_observations(100)
        
        # Fit the model
        model.fit(observations)
        
        assert model.is_fitted
        assert model.transition_matrix is not None
        assert model.means is not None
        assert model.covariances is not None
        assert model.initial_probs is not None
        
        # Check parameter shapes
        assert model.transition_matrix.shape == (3, 3)
        assert model.means.shape == (3, 2)  # 3 states, 2 features
        assert model.initial_probs.shape == (3,)
        
        # Transition matrix should be row-stochastic
        np.testing.assert_array_almost_equal(
            model.transition_matrix.sum(axis=1), 
            np.ones(3)
        )
        
        # Initial probabilities should sum to 1
        np.testing.assert_almost_equal(model.initial_probs.sum(), 1.0)
    
    def test_fit_with_insufficient_data(self):
        """Test fitting with insufficient data."""
        model = HiddenMarkovModel(n_states=3)
        
        # Too few observations
        observations = self.create_sample_observations(5)
        
        with pytest.raises(ValidationError, match="Insufficient data"):
            model.fit(observations)
    
    def test_fit_with_missing_data(self):
        """Test fitting with missing data."""
        model = HiddenMarkovModel(n_states=3)
        observations = self.create_sample_observations(50)
        
        # Introduce missing values
        observations.iloc[10, 0] = np.nan
        observations.iloc[20, 1] = np.nan
        
        with pytest.raises(ValidationError, match="Data contains missing values"):
            model.fit(observations)
    
    def test_fit_convergence(self):
        """Test convergence behavior."""
        # Test successful convergence
        model = HiddenMarkovModel(n_states=3, max_iter=100, tol=1e-4)
        observations = self.create_sample_observations(200)
        
        model.fit(observations)
        assert model.converged
        assert model.n_iter > 0
        assert model.n_iter <= 100
        
        # Test non-convergence
        model_no_converge = HiddenMarkovModel(n_states=5, max_iter=5, tol=1e-10)
        
        with pytest.warns(RuntimeWarning, match="Model did not converge"):
            model_no_converge.fit(observations)
        
        assert not model_no_converge.converged
        assert model_no_converge.n_iter == 5
    
    def test_predict_functionality(self):
        """Test prediction functionality."""
        model = HiddenMarkovModel(n_states=3, random_state=42)
        observations = self.create_sample_observations(100)
        
        # Fit first
        model.fit(observations)
        
        # Predict on same data
        predictions = model.predict(observations)
        
        assert isinstance(predictions, pd.DataFrame)
        assert 'predicted_state' in predictions.columns
        assert 'confidence' in predictions.columns
        assert len(predictions) == len(observations)
        
        # States should be in valid range
        assert (predictions['predicted_state'] >= 0).all()
        assert (predictions['predicted_state'] < 3).all()
        
        # Confidence should be between 0 and 1
        assert (predictions['confidence'] >= 0).all()
        assert (predictions['confidence'] <= 1).all()
    
    def test_predict_without_fitting(self):
        """Test prediction without fitting."""
        model = HiddenMarkovModel()
        observations = self.create_sample_observations(50)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(observations)
    
    def test_predict_proba_functionality(self):
        """Test probability prediction functionality."""
        model = HiddenMarkovModel(n_states=3, random_state=42)
        observations = self.create_sample_observations(100)
        
        model.fit(observations)
        probabilities = model.predict_proba(observations)
        
        assert isinstance(probabilities, pd.DataFrame)
        assert len(probabilities) == len(observations)
        assert probabilities.shape[1] == 3  # 3 states
        
        # Probabilities should sum to 1 across states
        prob_sums = probabilities.sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sums.values, np.ones(len(observations)))
        
        # All probabilities should be non-negative
        assert (probabilities >= 0).all().all()
    
    def test_update_functionality(self):
        """Test update method for pipeline interface."""
        model = HiddenMarkovModel(n_states=3, random_state=42)
        observations = self.create_sample_observations(100)
        
        # First update (should fit)
        result = model.update(observations)
        
        assert model.is_fitted
        assert isinstance(result, pd.DataFrame)
        assert 'predicted_state' in result.columns
        assert 'confidence' in result.columns
        
        # Second update (should predict only)
        new_observations = self.create_sample_observations(50)
        result2 = model.update(new_observations)
        
        assert isinstance(result2, pd.DataFrame)
        assert len(result2) == len(new_observations)
    
    def test_score_functionality(self):
        """Test model scoring functionality."""
        model = HiddenMarkovModel(n_states=3, random_state=42)
        observations = self.create_sample_observations(100)
        
        model.fit(observations)
        score = model.score(observations)
        
        assert isinstance(score, float)
        assert score <= 0  # Log-likelihood is negative
        
        # Score on new data should be different
        new_observations = self.create_sample_observations(50)
        new_score = model.score(new_observations)
        
        assert isinstance(new_score, float)
        assert new_score != score
    
    def test_different_covariance_types(self):
        """Test different covariance types."""
        observations = self.create_sample_observations(150)
        
        for cov_type in ['full', 'diag', 'spherical']:
            model = HiddenMarkovModel(n_states=3, covariance_type=cov_type, random_state=42)
            model.fit(observations)
            
            assert model.is_fitted
            predictions = model.predict(observations)
            assert len(predictions) == len(observations)
            
            # Check covariance structure
            if cov_type == 'full':
                # Full covariance matrices
                assert model.covariances.shape == (3, 2, 2)
            elif cov_type == 'diag':
                # Diagonal covariance matrices
                assert model.covariances.shape == (3, 2)
            elif cov_type == 'spherical':
                # Spherical covariance (single variance per state)
                assert model.covariances.shape == (3,)
    
    def test_state_decoding_viterbi(self):
        """Test Viterbi state decoding."""
        model = HiddenMarkovModel(n_states=3, random_state=42)
        observations = self.create_sample_observations(100)
        
        model.fit(observations)
        states = model.decode_states(observations, method='viterbi')
        
        assert isinstance(states, np.ndarray)
        assert len(states) == len(observations)
        assert (states >= 0).all()
        assert (states < 3).all()
    
    def test_state_decoding_posterior(self):
        """Test posterior state decoding."""
        model = HiddenMarkovModel(n_states=3, random_state=42)
        observations = self.create_sample_observations(100)
        
        model.fit(observations)
        states = model.decode_states(observations, method='posterior')
        
        assert isinstance(states, np.ndarray)
        assert len(states) == len(observations)
        assert (states >= 0).all()
        assert (states < 3).all()
    
    def test_regime_analysis(self):
        """Test regime analysis functionality."""
        model = HiddenMarkovModel(n_states=3, random_state=42)
        observations = self.create_sample_observations(200)
        
        model.fit(observations)
        analysis = model.get_regime_analysis(observations)
        
        assert isinstance(analysis, dict)
        
        # Should contain regime statistics
        assert 'regime_stats' in analysis
        assert len(analysis['regime_stats']) == 3
        
        for state in analysis['regime_stats']:
            regime_stats = analysis['regime_stats'][state]
            assert 'mean_duration' in regime_stats
            assert 'mean_return' in regime_stats
            assert 'volatility' in regime_stats
            assert 'frequency' in regime_stats
        
        # Should contain transition analysis
        assert 'transition_analysis' in analysis
        assert 'most_persistent' in analysis['transition_analysis']
        assert 'most_volatile' in analysis['transition_analysis']
    
    def test_model_persistence_pickle(self):
        """Test model serialization with pickle."""
        model = HiddenMarkovModel(n_states=3, random_state=42)
        observations = self.create_sample_observations(100)
        
        model.fit(observations)
        original_predictions = model.predict(observations)
        
        # Serialize and deserialize
        with tempfile.NamedTemporaryFile(delete=False) as f:
            pickle.dump(model, f)
            f.flush()
            
            with open(f.name, 'rb') as f2:
                loaded_model = pickle.load(f2)
        
        os.unlink(f.name)
        
        # Test loaded model
        assert loaded_model.is_fitted
        assert loaded_model.n_states == model.n_states
        
        loaded_predictions = loaded_model.predict(observations)
        pd.testing.assert_frame_equal(original_predictions, loaded_predictions)
    
    def test_model_persistence_custom(self):
        """Test custom model save/load functionality."""
        model = HiddenMarkovModel(n_states=3, random_state=42)
        observations = self.create_sample_observations(100)
        
        model.fit(observations)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model.save_model(f.name)
            
            loaded_model = HiddenMarkovModel.load_model(f.name)
        
        os.unlink(f.name)
        
        # Test loaded model
        assert loaded_model.is_fitted
        assert loaded_model.n_states == model.n_states
        np.testing.assert_array_equal(
            loaded_model.transition_matrix, 
            model.transition_matrix
        )
    
    def test_parameter_initialization_strategies(self):
        """Test different parameter initialization strategies."""
        observations = self.create_sample_observations(150)
        
        # Test different initialization methods
        for init_method in ['random', 'kmeans', 'manual']:
            if init_method == 'manual':
                # Provide manual initialization
                initial_params = {
                    'means': np.random.normal(0, 0.01, (3, 2)),
                    'covariances': np.array([np.eye(2) * 0.001 for _ in range(3)]),
                    'initial_probs': np.ones(3) / 3,
                    'transition_matrix': np.ones((3, 3)) / 3
                }
                model = HiddenMarkovModel(n_states=3, random_state=42)
                model.fit(observations, **initial_params)
            else:
                model = HiddenMarkovModel(
                    n_states=3, 
                    random_state=42,
                    init_method=init_method
                )
                model.fit(observations)
            
            assert model.is_fitted
            predictions = model.predict(observations)
            assert len(predictions) == len(observations)
    
    def test_online_learning_capabilities(self):
        """Test incremental/online learning capabilities."""
        model = HiddenMarkovModel(n_states=3, random_state=42)
        
        # Initial training
        initial_data = self.create_sample_observations(100)
        model.fit(initial_data)
        
        initial_params = {
            'transition_matrix': model.transition_matrix.copy(),
            'means': model.means.copy(),
            'covariances': model.covariances.copy()
        }
        
        # Incremental update
        new_data = self.create_sample_observations(50)
        model.partial_fit(new_data, learning_rate=0.1)
        
        # Parameters should have changed
        assert not np.array_equal(model.transition_matrix, initial_params['transition_matrix'])
        assert not np.array_equal(model.means, initial_params['means'])
        
        # Model should still be fitted
        assert model.is_fitted
        predictions = model.predict(new_data)
        assert len(predictions) == len(new_data)
    
    def test_model_selection_criteria(self):
        """Test model selection criteria (AIC, BIC)."""
        observations = self.create_sample_observations(200)
        
        model = HiddenMarkovModel(n_states=3, random_state=42)
        model.fit(observations)
        
        # Test AIC calculation
        aic = model.aic(observations)
        assert isinstance(aic, float)
        assert aic > 0  # AIC should be positive
        
        # Test BIC calculation
        bic = model.bic(observations)
        assert isinstance(bic, float)
        assert bic > 0  # BIC should be positive
        assert bic > aic  # BIC should be larger than AIC (penalty for complexity)
    
    def test_cross_validation_score(self):
        """Test cross-validation scoring."""
        model = HiddenMarkovModel(n_states=3, random_state=42)
        observations = self.create_sample_observations(200)
        
        cv_scores = model.cross_validate(observations, cv_folds=3)
        
        assert isinstance(cv_scores, dict)
        assert 'scores' in cv_scores
        assert 'mean_score' in cv_scores
        assert 'std_score' in cv_scores
        
        assert len(cv_scores['scores']) == 3
        assert isinstance(cv_scores['mean_score'], float)
        assert isinstance(cv_scores['std_score'], float)
    
    def test_plot_functionality(self):
        """Test plotting functionality."""
        model = HiddenMarkovModel(n_states=3, random_state=42)
        observations = self.create_sample_observations(100)
        
        model.fit(observations)
        predictions = model.predict(observations)
        
        # Test state sequence plotting
        fig = model.plot(observations=observations, plot_type='states')
        assert fig is not None
        
        # Test regime analysis plotting
        fig = model.plot(observations=observations, plot_type='regimes')
        assert fig is not None
        
        # Test transition matrix plotting
        fig = model.plot(plot_type='transitions')
        assert fig is not None
    
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities."""
        model = HiddenMarkovModel(n_states=3, random_state=42)
        observations = self.create_sample_observations(200)
        
        model.fit(observations)
        
        # Get performance metrics
        metrics = model.get_performance_metrics(observations)
        
        assert isinstance(metrics, dict)
        assert 'log_likelihood' in metrics
        assert 'aic' in metrics
        assert 'bic' in metrics
        assert 'n_parameters' in metrics
        
        # Test monitoring over time
        time_series_metrics = model.monitor_performance(observations, window_size=50)
        
        assert isinstance(time_series_metrics, pd.DataFrame)
        assert 'log_likelihood' in time_series_metrics.columns
        assert len(time_series_metrics) > 0


if __name__ == "__main__":
    pytest.main([__file__])