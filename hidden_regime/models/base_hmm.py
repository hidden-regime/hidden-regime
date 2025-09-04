"""
Base Hidden Markov Model implementation for market regime detection.

Provides the main HiddenMarkovModel class with training, inference,
and real-time regime detection capabilities.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple, List
import warnings
from datetime import datetime

from .config import HMMConfig
from .algorithms import HMMAlgorithms
from .utils import (
    validate_returns_data,
    initialize_parameters_random,
    initialize_parameters_kmeans,
    check_convergence,
    validate_hmm_parameters,
    get_regime_interpretation,
    calculate_regime_statistics
)
from ..utils.exceptions import HiddenRegimeError


class HMMTrainingError(HiddenRegimeError):
    """Raised when HMM training fails."""
    pass


class HMMConvergenceError(HiddenRegimeError):
    """Raised when HMM fails to converge."""
    pass


class HMMInferenceError(HiddenRegimeError):
    """Raised when HMM inference fails."""
    pass


class HiddenMarkovModel:
    """
    Hidden Markov Model for market regime detection.
    
    This class implements a discrete-time HMM with Gaussian emissions
    optimized for financial time series and regime detection.
    
    Attributes:
        n_states: Number of hidden states (regimes)
        config: HMM configuration object
        is_fitted: Whether the model has been trained
        initial_probs_: Learned initial state probabilities
        transition_matrix_: Learned state transition probabilities
        emission_params_: Learned emission parameters [means, stds]
        training_history_: Training convergence history
    """
    
    def __init__(self, 
                 n_states: Optional[int] = None, 
                 config: Optional[HMMConfig] = None):
        """
        Initialize Hidden Markov Model.
        
        Args:
            n_states: Number of hidden states (regimes)
            config: HMMConfig object, uses defaults if None
        """
        # Validate configuration
        if n_states is None:
            if config is None:
                raise RuntimeError('Must specify one of: n_states, config')
            else:
                self.n_states = config.n_states
                self.config = config
        else:
            if config is None:
                self.n_states = n_states
                self.config = HMMConfig(n_states=n_states)
            else:
                if config.n_states == n_states:
                    self.n_states = n_states
                    self.config = config
                else:
                    warnings.warn(
                        f"Config n_states ({config.n_states}) doesn't match "
                        f"constructor n_states ({n_states}). Using constructor value."
                    )
                    raise ValueError(f'Number of states do not match - config is {config.n_states} and n_states is {n_states}')

        # Model parameters (set after training)
        self.initial_probs_: Optional[np.ndarray] = None
        self.transition_matrix_: Optional[np.ndarray] = None
        self.emission_params_: Optional[np.ndarray] = None
        
        # Training metadata
        self.is_fitted = False
        self.training_history_: Dict[str, List] = {
            'log_likelihoods': [],
            'iterations': 0,
            'converged': False,
            'training_time': 0.0
        }
        
        # Real-time inference state
        self._current_state_probs: Optional[np.ndarray] = None
        self._last_observation: Optional[float] = None
    
    def fit(
        self, 
        returns: Union[np.ndarray, pd.Series], 
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
        verbose: bool = False
    ) -> 'HiddenMarkovModel':
        """
        Train the HMM on returns data using Baum-Welch EM algorithm.
        
        Args:
            returns: Log returns time series
            max_iterations: Maximum training iterations (overrides config)
            tolerance: Convergence tolerance (overrides config)
            verbose: Print training progress
            
        Returns:
            Self for method chaining
            
        Raises:
            HMMTrainingError: If training fails
            HMMConvergenceError: If model fails to converge
        """
        start_time = datetime.now()
        
        # Validate and prepare data
        try:
            returns_array = validate_returns_data(returns)
            self.config.validate_for_data(len(returns_array))
        except Exception as e:
            raise HMMTrainingError(f"Data validation failed: {e}")
        
        # Override config parameters if provided
        max_iter = max_iterations or self.config.max_iterations
        tol = tolerance or self.config.tolerance
        
        if verbose:
            print(f"Training {self.n_states}-state HMM on {len(returns_array)} observations")
            print(f"Max iterations: {max_iter}, Tolerance: {tol}")
        
        # Initialize parameters
        try:
            if self.config.initialization_method == 'random':
                initial_probs, transition_matrix, emission_params = initialize_parameters_random(
                    self.n_states, returns_array, self.config.random_seed
                )
            elif self.config.initialization_method == 'kmeans':
                initial_probs, transition_matrix, emission_params = initialize_parameters_kmeans(
                    self.n_states, returns_array, self.config.random_seed
                )
            else:
                raise ValueError(f"Unknown initialization method: {self.config.initialization_method}")
        except Exception as e:
            raise HMMTrainingError(f"Parameter initialization failed: {e}")
        
        # Training loop
        log_likelihoods = []
        prev_log_likelihood = float('-inf')
        
        for iteration in range(max_iter):
            try:
                # E-step: compute state probabilities
                gamma, xi, log_likelihood = HMMAlgorithms.forward_backward_algorithm(
                    returns_array, initial_probs, transition_matrix, emission_params
                )
                
                log_likelihoods.append(log_likelihood)
                
                # Check for numerical issues
                if not np.isfinite(log_likelihood):
                    raise HMMTrainingError(f"Non-finite log-likelihood at iteration {iteration}")
                
                if log_likelihood < self.config.log_likelihood_threshold:
                    raise HMMTrainingError(f"Extreme log-likelihood at iteration {iteration}: {log_likelihood}")
                
                # Check convergence
                improvement = log_likelihood - prev_log_likelihood
                if verbose and iteration % 10 == 0:
                    print(f"Iteration {iteration}: Log-likelihood = {log_likelihood:.4f}, "
                          f"Improvement = {improvement:.6f}")
                
                if (iteration > 10 and 
                    self.config.early_stopping and 
                    iteration % self.config.check_convergence_every == 0 and
                    check_convergence(log_likelihoods, tol)):
                    
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    break
                
                # M-step: update parameters
                initial_probs, transition_matrix, emission_params = HMMAlgorithms.baum_welch_update(
                    returns_array, gamma, xi, self.config.regularization
                )
                
                # Validate updated parameters
                validate_hmm_parameters(initial_probs, transition_matrix, emission_params)
                
                prev_log_likelihood = log_likelihood
                
            except Exception as e:
                raise HMMTrainingError(f"Training failed at iteration {iteration}: {e}")
        
        else:
            # Loop completed without break (no convergence)
            if self.config.early_stopping and len(log_likelihoods) >= 2:
                final_improvement = log_likelihoods[-1] - log_likelihoods[-2]
                if final_improvement > tol:
                    warnings.warn(
                        f"HMM did not converge after {max_iter} iterations. "
                        f"Final improvement: {final_improvement:.6f}"
                    )
        
        # Store results
        self.initial_probs_ = initial_probs
        self.transition_matrix_ = transition_matrix
        self.emission_params_ = emission_params
        self.is_fitted = True
        
        # Store training history
        training_time = (datetime.now() - start_time).total_seconds()
        self.training_history_ = {
            'log_likelihoods': log_likelihoods,
            'iterations': len(log_likelihoods),
            'converged': iteration < max_iter - 1,
            'training_time': training_time,
            'final_log_likelihood': log_likelihoods[-1] if log_likelihoods else float('-inf')
        }
        
        if verbose:
            print(f"Training completed in {training_time:.2f}s")
            print(f"Final log-likelihood: {log_likelihoods[-1]:.4f}")
        
        return self
    
    def predict(self, returns: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Predict most likely state sequence using Viterbi algorithm.
        
        Args:
            returns: Returns time series
            
        Returns:
            Most likely state sequence
            
        Raises:
            HMMInferenceError: If prediction fails
        """
        self._check_fitted()
        
        try:
            returns_array = validate_returns_data(returns)
            states, _ = HMMAlgorithms.viterbi_algorithm(
                returns_array,
                self.initial_probs_,
                self.transition_matrix_,
                self.emission_params_
            )
            return states
        except Exception as e:
            raise HMMInferenceError(f"State prediction failed: {e}")
    
    def predict_proba(self, returns: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Predict state probabilities using forward-backward algorithm.
        
        Args:
            returns: Returns time series
            
        Returns:
            State probabilities (T, n_states)
            
        Raises:
            HMMInferenceError: If probability prediction fails
        """
        self._check_fitted()
        
        try:
            returns_array = validate_returns_data(returns)
            gamma, _, _ = HMMAlgorithms.forward_backward_algorithm(
                returns_array,
                self.initial_probs_,
                self.transition_matrix_,
                self.emission_params_
            )
            return gamma
        except Exception as e:
            raise HMMInferenceError(f"Probability prediction failed: {e}")
    
    def score(self, returns: Union[np.ndarray, pd.Series]) -> float:
        """
        Compute log-likelihood of returns under the fitted model.
        
        Args:
            returns: Returns time series
            
        Returns:
            Log-likelihood score
            
        Raises:
            HMMInferenceError: If scoring fails
        """
        self._check_fitted()
        
        try:
            returns_array = validate_returns_data(returns)
            return HMMAlgorithms.compute_likelihood(
                returns_array,
                self.initial_probs_,
                self.transition_matrix_,
                self.emission_params_
            )
        except Exception as e:
            raise HMMInferenceError(f"Scoring failed: {e}")
    
    def update_with_observation(self, new_return: float) -> Dict[str, Any]:
        """
        Update model state with new observation for real-time inference.
        
        Args:
            new_return: New log return observation
            
        Returns:
            Dictionary with current regime information
            
        Raises:
            HMMInferenceError: If update fails
        """
        self._check_fitted()
        
        try:
            if not np.isfinite(new_return):
                raise ValueError("New return must be finite")
            
            # Initialize state probabilities if first observation
            if self._current_state_probs is None:
                self._current_state_probs = self.initial_probs_.copy()
            
            # Update state probabilities with new observation
            self._current_state_probs = HMMAlgorithms.decode_states_online(
                new_return,
                self._current_state_probs,
                self.transition_matrix_,
                self.emission_params_
            )
            
            self._last_observation = new_return
            
            return self.get_current_regime_info()
            
        except Exception as e:
            raise HMMInferenceError(f"Online update failed: {e}")
    
    def get_current_regime_info(self) -> Dict[str, Any]:
        """
        Get information about current regime state.
        
        Returns:
            Dictionary with regime information
            
        Raises:
            HMMInferenceError: If model not fitted or no current state
        """
        self._check_fitted()
        
        if self._current_state_probs is None:
            raise HMMInferenceError("No current state available. Call update_with_observation() first.")
        
        most_likely_state = np.argmax(self._current_state_probs)
        confidence = self._current_state_probs[most_likely_state]
        
        regime_info = {
            'most_likely_regime': int(most_likely_state),
            'regime_probabilities': self._current_state_probs.tolist(),
            'confidence': float(confidence),
            'regime_interpretation': get_regime_interpretation(most_likely_state, self.emission_params_),
            'expected_return': float(self.emission_params_[most_likely_state, 0]),
            'expected_volatility': float(self.emission_params_[most_likely_state, 1]),
            'last_observation': self._last_observation
        }
        
        # Add transition predictions
        if self.transition_matrix_ is not None:
            next_state_probs = HMMAlgorithms.predict_next_state_probs(
                self._current_state_probs, self.transition_matrix_
            )
            regime_info['next_period_probabilities'] = next_state_probs.tolist()
        
        return regime_info
    
    def analyze_regimes(
        self, 
        returns: Union[np.ndarray, pd.Series],
        dates: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive regime analysis of returns data.
        
        Args:
            returns: Returns time series
            dates: Optional dates corresponding to returns
            
        Returns:
            Comprehensive regime analysis results
        """
        self._check_fitted()
        
        returns_array = validate_returns_data(returns)
        
        # Get state sequence and probabilities
        states = self.predict(returns_array)
        probabilities = self.predict_proba(returns_array)
        
        # Calculate regime statistics
        regime_stats = calculate_regime_statistics(states, returns_array, dates)
        
        # Add model parameters and interpretations
        analysis = {
            'model_info': {
                'n_states': self.n_states,
                'n_observations': len(returns_array),
                'log_likelihood': self.score(returns_array),
                'training_iterations': self.training_history_['iterations'],
                'converged': self.training_history_['converged']
            },
            'regime_parameters': {
                'initial_probabilities': self.initial_probs_.tolist(),
                'transition_matrix': self.transition_matrix_.tolist(),
                'emission_parameters': self.emission_params_.tolist()
            },
            'regime_interpretations': {
                str(i): get_regime_interpretation(i, self.emission_params_)
                for i in range(self.n_states)
            },
            'regime_statistics': regime_stats,
            'state_sequence': states.tolist(),
            'state_probabilities': probabilities.tolist()
        }
        
        return analysis
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save fitted model to file.
        
        Args:
            filepath: Path to save file
            
        Raises:
            HMMTrainingError: If model not fitted or save fails
        """
        if not self.is_fitted:
            raise HMMTrainingError("Cannot save unfitted model")
        
        filepath = Path(filepath)
        
        try:
            model_data = {
                'n_states': self.n_states,
                'config': self.config.__dict__,
                'initial_probs': self.initial_probs_.tolist(),
                'transition_matrix': self.transition_matrix_.tolist(),
                'emission_params': self.emission_params_.tolist(),
                'training_history': self.training_history_,
                'is_fitted': self.is_fitted,
                'current_state_probs': (self._current_state_probs.tolist() 
                                      if self._current_state_probs is not None else None),
                'last_observation': self._last_observation,
                'save_timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            if filepath.suffix.lower() == '.json':
                with open(filepath, 'w') as f:
                    json.dump(model_data, f, indent=2)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f)
                    
        except Exception as e:
            raise HMMTrainingError(f"Failed to save model: {e}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'HiddenMarkovModel':
        """
        Load fitted model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded HiddenMarkovModel instance
            
        Raises:
            HMMTrainingError: If loading fails
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise HMMTrainingError(f"Model file not found: {filepath}")
        
        try:
            if filepath.suffix.lower() == '.json':
                with open(filepath, 'r') as f:
                    model_data = json.load(f)
            else:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
            
            # Create model instance
            config = HMMConfig(**model_data['config'])
            model = cls(n_states=model_data['n_states'], config=config)
            
            # Restore fitted parameters
            model.initial_probs_ = np.array(model_data['initial_probs'])
            model.transition_matrix_ = np.array(model_data['transition_matrix'])
            model.emission_params_ = np.array(model_data['emission_params'])
            model.is_fitted = model_data['is_fitted']
            model.training_history_ = model_data['training_history']
            
            # Restore real-time state if available
            if model_data.get('current_state_probs') is not None:
                model._current_state_probs = np.array(model_data['current_state_probs'])
            model._last_observation = model_data.get('last_observation')
            
            return model
            
        except Exception as e:
            raise HMMTrainingError(f"Failed to load model: {e}")
    
    def reset_state(self) -> None:
        """Reset real-time inference state."""
        self._current_state_probs = None
        self._last_observation = None
    
    def _check_fitted(self) -> None:
        """Check if model has been fitted."""
        if not self.is_fitted:
            raise HMMTrainingError("Model must be fitted before use")
    
    def __repr__(self) -> str:
        """String representation of the model."""
        if self.is_fitted:
            return (f"HiddenMarkovModel(n_states={self.n_states}, "
                   f"fitted=True, log_likelihood={self.training_history_['final_log_likelihood']:.2f})")
        else:
            return f"HiddenMarkovModel(n_states={self.n_states}, fitted=False)"