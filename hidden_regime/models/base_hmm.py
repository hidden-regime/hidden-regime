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
from .state_standardizer import StateStandardizer
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
                # Handle regime_type auto-adjustment for n_states
                if config.regime_type != 'auto':
                    expected_states = int(config.regime_type.split('_')[0])
                    if config.n_states != expected_states:
                        self.n_states = expected_states
                        # Create updated config with correct n_states
                        config_dict = config.__dict__.copy()
                        config_dict['n_states'] = expected_states
                        self.config = HMMConfig(**config_dict)
                    else:
                        self.n_states = config.n_states
                        self.config = config
                else:
                    self.n_states = config.n_states  # Will be updated by auto-selection
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
        
        # State standardization
        self._state_standardizer: Optional[StateStandardizer] = None
        self._state_mapping: Optional[Dict[int, str]] = None
        self._standardization_confidence: Optional[float] = None
    
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
        
        # Handle automatic state selection
        if self.config.auto_select_states:
            try:
                self._state_standardizer = StateStandardizer('auto')
                best_regime_type, validation_score, selection_details = self._state_standardizer.select_optimal_configuration(
                    returns_array, 
                    validation_threshold=self.config.state_validation_threshold,
                    max_iterations=max_iterations
                )
                
                # Update configuration with selected regime type
                self.n_states = int(best_regime_type.split('_')[0])
                config_dict = self.config.__dict__.copy()
                config_dict['n_states'] = self.n_states
                config_dict['regime_type'] = best_regime_type
                self.config = HMMConfig(**config_dict)
                
                if verbose:
                    print(f"Auto-selected {best_regime_type} configuration (validation score: {validation_score:.3f})")
                    
            except Exception as e:
                raise HMMTrainingError(f"Automatic state selection failed: {e}")
        else:
            # Initialize state standardizer with specified regime type
            if self.config.regime_type != 'auto':
                self._state_standardizer = StateStandardizer(self.config.regime_type)
        
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
        
        # Apply state standardization if configured
        if self.config.force_state_ordering and self._state_standardizer is not None:
            try:
                # Reorder states by mean return for consistency
                emission_params, transition_matrix, initial_probs = self._state_standardizer.reorder_states_by_returns(
                    emission_params, transition_matrix, initial_probs
                )
                
                if verbose:
                    print("Applied state standardization and ordering")
                    
            except Exception as e:
                warnings.warn(f"State standardization failed: {e}")
        
        # Store results
        self.initial_probs_ = initial_probs
        self.transition_matrix_ = transition_matrix
        self.emission_params_ = emission_params
        self.is_fitted = True
        
        # Create state mapping and validate if standardizer available
        if self._state_standardizer is not None:
            try:
                self._state_mapping = self._state_standardizer.standardize_states(emission_params)
                
                if self.config.validate_regime_economics:
                    states = self.predict(returns_array)
                    self._standardization_confidence = self._state_standardizer.validate_interpretation(
                        states, returns_array, emission_params
                    )
                    
                    if verbose:
                        print(f"Regime validation confidence: {self._standardization_confidence:.3f}")
                        if self._standardization_confidence < self.config.state_validation_threshold:
                            warnings.warn(f"Low regime validation confidence: {self._standardization_confidence:.3f}")
                
            except Exception as e:
                warnings.warn(f"State mapping validation failed: {e}")
        
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
        
        # Use standardized regime interpretation if available
        if self._state_standardizer is not None and self._state_mapping is not None:
            regime_details = self._state_standardizer.get_regime_interpretation_enhanced(
                most_likely_state, self.emission_params_
            )
            regime_interpretation = regime_details['interpretation']
            regime_name = regime_details['regime_name']
        else:
            regime_interpretation = get_regime_interpretation(most_likely_state, self.emission_params_)
            regime_name = f"State_{most_likely_state}"
        
        regime_info = {
            'most_likely_regime': int(most_likely_state),
            'regime_name': regime_name,
            'regime_probabilities': self._current_state_probs.tolist(),
            'confidence': float(confidence),
            'regime_interpretation': regime_interpretation,
            'expected_return': float(self.emission_params_[most_likely_state, 0]),
            'expected_volatility': float(self.emission_params_[most_likely_state, 1]),
            'last_observation': self._last_observation
        }
        
        # Add standardization information if available
        if self._standardization_confidence is not None:
            regime_info['validation_confidence'] = float(self._standardization_confidence)
            regime_info['regime_type'] = self.config.regime_type
        
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
            Comprehensive regime analysis results with standardized regime names
        """
        self._check_fitted()
        
        returns_array = validate_returns_data(returns)
        
        # Get state sequence and probabilities
        states = self.predict(returns_array)
        probabilities = self.predict_proba(returns_array)
        
        # Calculate regime statistics
        regime_stats = calculate_regime_statistics(states, returns_array, dates)
        
        # Use standardized regime interpretations if available
        if hasattr(self, '_state_standardizer') and self._state_standardizer is not None:
            regime_interpretations = {}
            for i in range(self.n_states):
                if hasattr(self, '_state_mapping') and self._state_mapping is not None:
                    # Use standardized names
                    standardized_state = self._state_mapping.get(i, i)
                    if isinstance(standardized_state, int):
                        # Get standardized regime name
                        config = self._state_standardizer.get_config(self.config.regime_type)
                        if config and standardized_state < len(config.state_names):
                            regime_interpretations[str(i)] = config.state_names[standardized_state]
                        else:
                            regime_interpretations[str(i)] = get_regime_interpretation(i, self.emission_params_)
                    else:
                        regime_interpretations[str(i)] = standardized_state
                else:
                    regime_interpretations[str(i)] = get_regime_interpretation(i, self.emission_params_)
        else:
            # Fallback to original interpretation method
            regime_interpretations = {
                str(i): get_regime_interpretation(i, self.emission_params_)
                for i in range(self.n_states)
            }
        
        # Add model parameters and interpretations
        analysis = {
            'model_info': {
                'n_states': self.n_states,
                'n_observations': len(returns_array),
                'log_likelihood': self.score(returns_array),
                'training_iterations': self.training_history_['iterations'],
                'converged': self.training_history_['converged'],
                'regime_type': getattr(self.config, 'regime_type', 'custom'),
                'standardization_applied': hasattr(self, '_state_standardizer') and self._state_standardizer is not None
            },
            'regime_parameters': {
                'initial_probabilities': self.initial_probs_.tolist(),
                'transition_matrix': self.transition_matrix_.tolist(),
                'emission_parameters': self.emission_params_.tolist()
            },
            'regime_interpretations': regime_interpretations,
            'regime_statistics': regime_stats,
            'state_sequence': states.tolist(),
            'state_probabilities': probabilities.tolist()
        }
        
        # Add standardization confidence if available
        if hasattr(self, '_standardization_confidence') and self._standardization_confidence is not None:
            analysis['standardization_confidence'] = self._standardization_confidence
            
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
            
            # Add standardization attributes if present
            if hasattr(self, '_state_mapping') and self._state_mapping is not None:
                model_data['state_mapping'] = self._state_mapping
            if hasattr(self, '_standardization_confidence') and self._standardization_confidence is not None:
                model_data['standardization_confidence'] = self._standardization_confidence
            
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
            
            # Restore standardization attributes if available
            if model_data.get('state_mapping') is not None:
                model._state_mapping = model_data['state_mapping']
            if model_data.get('standardization_confidence') is not None:
                model._standardization_confidence = model_data['standardization_confidence']
            
            # Recreate state standardizer if needed
            if (hasattr(model.config, 'regime_type') and 
                getattr(model.config, 'regime_type', 'custom') != 'custom'):
                from .state_standardizer import StateStandardizer
                model._state_standardizer = StateStandardizer()
            
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
    
    def plot(self, 
             returns: Union[np.ndarray, pd.Series],
             dates: Optional[Union[np.ndarray, pd.Series]] = None,
             plot_type: str = 'all',
             figsize: Tuple[int, int] = (16, 12),
             save_path: Optional[str] = None) -> 'matplotlib.Figure':
        """
        Create comprehensive visualizations of HMM regime detection results.
        
        Args:
            returns: Returns time series used for regime detection
            dates: Optional dates corresponding to returns
            plot_type: Type of plot ('all', 'regimes', 'probabilities', 'transitions', 
                      'statistics', 'convergence', 'duration')
            figsize: Figure size as (width, height)
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
            
        Raises:
            ImportError: If matplotlib/seaborn not available
            HMMTrainingError: If model not fitted
        """
        self._check_fitted()
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from ..visualization.plotting import (
                setup_financial_plot_style, format_financial_axis, save_plot,
                plot_returns_with_regimes, plot_regime_heatmap, get_regime_colors
            )
        except ImportError:
            raise ImportError(
                "Plotting requires matplotlib and seaborn. Install with: pip install matplotlib seaborn"
            )
        
        # Setup styling
        setup_financial_plot_style()
        
        # Validate and prepare data
        returns_array = validate_returns_data(returns)
        
        if dates is not None:
            dates_array = np.array(dates)
            if len(dates_array) != len(returns_array):
                raise ValueError("Length of dates must match length of returns")
            dates_array = pd.to_datetime(dates_array)
        else:
            dates_array = None
        
        # Get model predictions
        predicted_states = self.predict(returns_array)
        state_probabilities = self.predict_proba(returns_array)
        
        # Get regime names and colors
        regime_names = self._get_regime_names()
        colors = get_regime_colors(list(regime_names.values()))
        
        # Determine subplot configuration
        if plot_type == 'all':
            subplot_configs = [
                ('Regime Classification', 'regimes'),
                ('State Probabilities', 'probabilities'),
                ('Transition Matrix', 'transitions'),
                ('Regime Statistics', 'statistics'),
                ('Training Convergence', 'convergence')
            ]
            if self._has_duration_analysis():
                subplot_configs.append(('Regime Durations', 'duration'))
        elif plot_type in ['regimes', 'probabilities', 'transitions', 'statistics', 'convergence', 'duration']:
            subplot_configs = [(plot_type.title(), plot_type)]
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}")
        
        # Create subplots with custom layout for 'all'
        if plot_type == 'all':
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1], width_ratios=[2, 1])
            axes = [
                fig.add_subplot(gs[0, :]),  # regimes (full width)
                fig.add_subplot(gs[1, :]),  # probabilities (full width) 
                fig.add_subplot(gs[2, 0]),  # transitions
                fig.add_subplot(gs[2, 1])   # statistics/convergence
            ]
        else:
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            axes = [axes] if not isinstance(axes, list) else axes
        
        # Create each subplot
        for i, (title, plot_subtype) in enumerate(subplot_configs):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if plot_subtype == 'regimes':
                # Returns colored by predicted regime
                plot_returns_with_regimes(
                    returns_array, predicted_states, dates_array, 
                    regime_names, ax, title="Returns Colored by Detected Regime"
                )
                
                # Add regime change markers
                regime_changes = np.where(np.diff(predicted_states) != 0)[0] + 1
                if len(regime_changes) > 0 and dates_array is not None:
                    for change_idx in regime_changes:
                        ax.axvline(x=dates_array[change_idx], color='black', 
                                  linestyle=':', alpha=0.5, linewidth=1)
                
            elif plot_subtype == 'probabilities':
                # State probabilities heatmap over time
                if dates_array is not None:
                    x_data = dates_array
                    ax.set_xlabel('Date')
                else:
                    x_data = np.arange(len(returns_array))
                    ax.set_xlabel('Time')
                
                # Create probability heatmap
                im = ax.imshow(state_probabilities.T, cmap='viridis', aspect='auto', 
                              extent=[0, len(returns_array), -0.5, self.n_states-0.5])
                
                # Format axes
                ax.set_title('State Probabilities Over Time', fontweight='bold')
                ax.set_ylabel('State')
                ax.set_yticks(range(self.n_states))
                ax.set_yticklabels([regime_names.get(i, f"State {i}") for i in range(self.n_states)])
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Probability')
                
                if dates_array is not None:
                    format_financial_axis(ax, dates_array)
                
            elif plot_subtype == 'transitions':
                # Transition matrix heatmap
                transition_df = pd.DataFrame(
                    self.transition_matrix_,
                    index=[regime_names.get(i, f"State {i}") for i in range(self.n_states)],
                    columns=[regime_names.get(i, f"State {i}") for i in range(self.n_states)]
                )
                
                sns.heatmap(transition_df, ax=ax, annot=True, fmt='.3f', 
                           cmap='Blues', cbar_kws={'label': 'Probability'})
                ax.set_title('Regime Transition Matrix', fontweight='bold')
                
            elif plot_subtype == 'statistics':
                # Regime statistics comparison
                regime_stats = self._calculate_regime_stats(returns_array, predicted_states)
                
                # Create bar plot of key metrics
                metrics = ['mean_return', 'std_return', 'frequency']
                x = np.arange(self.n_states)
                width = 0.25
                
                for j, metric in enumerate(metrics):
                    values = [regime_stats[i][metric] for i in range(self.n_states)]
                    bars = ax.bar(x + j*width, values, width, 
                                 label=metric.replace('_', ' ').title(), alpha=0.8)
                    
                    # Color bars by regime
                    for bar, state in zip(bars, range(self.n_states)):
                        regime_name = regime_names.get(state, f"State {state}")
                        bar.set_color(colors.get(regime_name, '#7f7f7f'))
                
                ax.set_xlabel('Regime')
                ax.set_ylabel('Value')
                ax.set_title('Regime Characteristics', fontweight='bold')
                ax.set_xticks(x + width)
                ax.set_xticklabels([regime_names.get(i, f"State {i}") for i in range(self.n_states)])
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            elif plot_subtype == 'convergence':
                # Training convergence plot
                if 'log_likelihood_history' in self.training_history_:
                    ll_history = self.training_history_['log_likelihood_history']
                    ax.plot(ll_history, color='blue', linewidth=2)
                    ax.set_title('Training Convergence', fontweight='bold')
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Log-Likelihood')
                    ax.grid(True, alpha=0.3)
                    
                    # Mark convergence point if available
                    if self.training_history_['converged']:
                        final_iter = len(ll_history) - 1
                        ax.axvline(x=final_iter, color='green', linestyle='--', 
                                  alpha=0.7, label=f'Converged at iter {final_iter}')
                        ax.legend()
                else:
                    ax.text(0.5, 0.5, 'Convergence history not available', 
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
                    ax.set_title('Training Convergence', fontweight='bold')
                
            elif plot_subtype == 'duration':
                # Regime duration analysis
                durations = self._calculate_regime_durations(predicted_states)
                
                if durations:
                    # Create duration histogram by regime
                    for state in range(self.n_states):
                        state_durations = [d for d in durations if d['regime'] == state]
                        if state_durations:
                            regime_name = regime_names.get(state, f"State {state}")
                            duration_values = [d['duration'] for d in state_durations]
                            color = colors.get(regime_name, '#7f7f7f')
                            
                            ax.hist(duration_values, bins=20, alpha=0.6, 
                                   label=regime_name, color=color)
                    
                    ax.set_title('Regime Duration Distribution', fontweight='bold')
                    ax.set_xlabel('Duration (periods)')
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Duration analysis not available', 
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        # Add model summary text for 'all' plot
        if plot_type == 'all':
            summary_text = self._create_model_summary_text(returns_array, predicted_states)
            fig.text(0.02, 0.02, summary_text, fontsize=9, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            save_plot(fig, save_path)
        
        return fig
    
    def _get_regime_names(self) -> Dict[int, str]:
        """Get regime names from standardizer or use defaults."""
        regime_names = {}
        
        # Try to get standardized names
        if hasattr(self, '_state_standardizer') and self._state_standardizer is not None:
            config_obj = self._state_standardizer.current_config
            if config_obj and hasattr(self, '_state_mapping') and self._state_mapping:
                for i in range(self.n_states):
                    mapped_value = self._state_mapping.get(i, self._state_mapping.get(np.int64(i), i))
                    if isinstance(mapped_value, str):
                        regime_names[i] = mapped_value
                    elif isinstance(mapped_value, (int, np.integer)) and int(mapped_value) < len(config_obj.state_names):
                        regime_names[i] = config_obj.state_names[int(mapped_value)]
                    else:
                        regime_names[i] = get_regime_interpretation(i, self.emission_params_)
        
        # Fallback to interpretation or simple names
        for i in range(self.n_states):
            if i not in regime_names:
                if hasattr(self, 'emission_params_'):
                    regime_names[i] = get_regime_interpretation(i, self.emission_params_)
                else:
                    regime_names[i] = f"State {i}"
        
        return regime_names
    
    def _calculate_regime_stats(self, returns: np.ndarray, states: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Calculate statistics for each regime."""
        stats = {}
        
        for state in range(self.n_states):
            state_mask = states == state
            state_returns = returns[state_mask]
            
            if len(state_returns) > 0:
                stats[state] = {
                    'mean_return': np.mean(state_returns),
                    'std_return': np.std(state_returns),
                    'frequency': np.sum(state_mask) / len(states),
                    'min_return': np.min(state_returns),
                    'max_return': np.max(state_returns),
                    'total_periods': np.sum(state_mask)
                }
            else:
                stats[state] = {
                    'mean_return': 0.0,
                    'std_return': 0.0, 
                    'frequency': 0.0,
                    'min_return': 0.0,
                    'max_return': 0.0,
                    'total_periods': 0
                }
        
        return stats
    
    def _calculate_regime_durations(self, states: np.ndarray) -> List[Dict[str, Any]]:
        """Calculate regime duration statistics."""
        durations = []
        current_regime = states[0]
        duration = 1
        start_idx = 0
        
        for i in range(1, len(states)):
            if states[i] == current_regime:
                duration += 1
            else:
                durations.append({
                    'regime': current_regime,
                    'duration': duration,
                    'start_idx': start_idx,
                    'end_idx': i - 1
                })
                current_regime = states[i]
                duration = 1
                start_idx = i
        
        # Don't forget the last duration
        durations.append({
            'regime': current_regime,
            'duration': duration,
            'start_idx': start_idx,
            'end_idx': len(states) - 1
        })
        
        return durations
    
    def _has_duration_analysis(self) -> bool:
        """Check if duration analysis is meaningful."""
        return self.is_fitted and self.n_states >= 2
    
    def _create_model_summary_text(self, returns: np.ndarray, states: np.ndarray) -> str:
        """Create model summary text."""
        regime_stats = self._calculate_regime_stats(returns, states)
        regime_names = self._get_regime_names()
        
        summary = [
            f"HMM Model Summary",
            f"States: {self.n_states} ({', '.join(regime_names.values())})",
            f"Observations: {len(returns)}",
            f"Log-likelihood: {self.training_history_['final_log_likelihood']:.2f}",
            f"Converged: {'Yes' if self.training_history_['converged'] else 'No'}",
            f"Iterations: {self.training_history_['iterations']}"
        ]
        
        if hasattr(self, '_standardization_confidence') and self._standardization_confidence:
            summary.append(f"Standardization confidence: {self._standardization_confidence:.3f}")
        
        # Add regime frequencies
        summary.append("")
        summary.append("Regime Frequencies:")
        for state in range(self.n_states):
            name = regime_names.get(state, f"State {state}")
            freq = regime_stats[state]['frequency']
            summary.append(f"  {name}: {freq:.1%}")
        
        return '\n'.join(summary)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        if self.is_fitted:
            return (f"HiddenMarkovModel(n_states={self.n_states}, "
                   f"fitted=True, log_likelihood={self.training_history_['final_log_likelihood']:.2f})")
        else:
            return f"HiddenMarkovModel(n_states={self.n_states}, fitted=False)"