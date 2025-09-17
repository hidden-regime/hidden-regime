"""
Hidden Markov Model component for pipeline architecture.

Provides HiddenMarkovModel that implements ModelComponent interface for
regime detection using sophisticated HMM algorithms with online learning.
"""

import warnings
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from ..pipeline.interfaces import ModelComponent
from ..config.model import HMMConfig
from ..utils.exceptions import HMMTrainingError, HMMInferenceError

# Try to import existing HMM utilities
try:
    from .algorithms import HMMAlgorithms
    from .utils import (
        validate_returns_data,
        initialize_parameters_kmeans,
        initialize_parameters_random,
        validate_hmm_parameters,
        check_convergence
    )
    HMM_UTILS_AVAILABLE = True
except ImportError:
    HMM_UTILS_AVAILABLE = False
    warnings.warn("HMM utilities not available - using simplified implementation")


class HiddenMarkovModel(ModelComponent):
    """
    Hidden Markov Model component for regime detection in pipeline architecture.
    
    Implements ModelComponent interface to provide HMM-based regime detection
    with training, prediction, and online learning capabilities.
    """
    
    def __init__(self, config: HMMConfig):
        """
        Initialize HMM model with configuration.
        
        Args:
            config: HMMConfig with model parameters
        """
        self.config = config
        self.n_states = config.n_states
        
        # Model parameters (set after training)
        self.initial_probs_: Optional[np.ndarray] = None
        self.transition_matrix_: Optional[np.ndarray] = None
        self.emission_means_: Optional[np.ndarray] = None
        self.emission_stds_: Optional[np.ndarray] = None
        
        # Training state
        self.is_fitted = False
        self.training_history_ = {
            "log_likelihoods": [],
            "iterations": 0,
            "converged": False,
            "training_time": 0.0,
        }
        
        # Online learning state
        self._current_state_probs: Optional[np.ndarray] = None
        self._last_observation: Optional[float] = None
        self._sufficient_stats = None
        
        # Algorithm implementation
        if HMM_UTILS_AVAILABLE:
            self._algorithms = HMMAlgorithms()
        else:
            self._algorithms = None
    
    def fit(self, observations: pd.DataFrame) -> None:
        """
        Train the model on observations.
        
        Args:
            observations: Training data DataFrame with observation columns
        """
        # Extract the observed signal from observations
        if self.config.observed_signal not in observations.columns:
            raise ValueError(f"Observed signal '{self.config.observed_signal}' not found in observations")
        
        returns = observations[self.config.observed_signal].values
        
        # Validate returns data
        if HMM_UTILS_AVAILABLE:
            validate_returns_data(returns)
        else:
            self._validate_returns_simple(returns)
        
        # Set random seed if specified
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        # Initialize parameters
        self._initialize_parameters(returns)
        
        # Train using Baum-Welch algorithm
        self._train_baum_welch(returns)
        
        self.is_fitted = True
    
    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for observations.
        
        Args:
            observations: Input observations DataFrame
            
        Returns:
            DataFrame with predictions (predicted_state, confidence)
        """
        if not self.is_fitted:
            raise HMMInferenceError("Model must be fitted before prediction")
        
        # Extract the observed signal
        if self.config.observed_signal not in observations.columns:
            raise ValueError(f"Observed signal '{self.config.observed_signal}' not found in observations")
        
        returns = observations[self.config.observed_signal].values
        
        # Get most likely state sequence using Viterbi algorithm
        states = self._viterbi_decode(returns)
        
        # Calculate state probabilities for confidence
        state_probs = self._forward_backward(returns)
        confidence = np.max(state_probs, axis=1)
        
        # Create results DataFrame
        results = pd.DataFrame(index=observations.index)
        results['predicted_state'] = states
        results['confidence'] = confidence
        
        # Add individual state probabilities
        for i in range(self.n_states):
            results[f'state_{i}_prob'] = state_probs[:, i]
        
        return results
    
    def update(self, observations: pd.DataFrame) -> pd.DataFrame:
        """
        Process observations (fit if first time, then predict).
        
        Args:
            observations: Input observations DataFrame
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            # First time - fit the model
            self.fit(observations)
        
        # Always predict on the given observations
        predictions = self.predict(observations)
        
        # If online learning is enabled, update parameters
        if hasattr(self.config, 'adaptation_rate') and self.config.adaptation_rate > 0:
            self._online_update(observations)
        
        return predictions
    
    def plot(self, **kwargs) -> plt.Figure:
        """
        Generate visualization for this component.
        
        Returns:
            matplotlib Figure object with HMM visualization
        """
        if not self.is_fitted:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Model not fitted yet', ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        # Create subplots for model visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Transition matrix heatmap
        ax1 = axes[0, 0]
        im1 = ax1.imshow(self.transition_matrix_, cmap='Blues', aspect='auto')
        ax1.set_title('Transition Matrix')
        ax1.set_xlabel('To State')
        ax1.set_ylabel('From State')
        
        # Add text annotations
        for i in range(self.n_states):
            for j in range(self.n_states):
                text = ax1.text(j, i, f'{self.transition_matrix_[i, j]:.2f}',
                               ha="center", va="center", color="black")
        
        plt.colorbar(im1, ax=ax1)
        
        # Plot 2: Emission parameters
        ax2 = axes[0, 1]
        states = range(self.n_states)
        width = 0.35
        x = np.arange(len(states))
        
        bars1 = ax2.bar(x - width/2, self.emission_means_, width, label='Mean', alpha=0.8)
        bars2 = ax2.bar(x + width/2, self.emission_stds_, width, label='Std Dev', alpha=0.8)
        
        ax2.set_title('Emission Parameters by State')
        ax2.set_xlabel('State')
        ax2.set_ylabel('Value')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'State {i}' for i in states])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Training convergence
        ax3 = axes[1, 0]
        if self.training_history_['log_likelihoods']:
            ax3.plot(self.training_history_['log_likelihoods'], 'b-', linewidth=2)
            ax3.set_title('Training Convergence')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Log Likelihood')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No training history', ha='center', va='center')
            ax3.set_title('Training Convergence')
        
        # Plot 4: Model summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        summary_text = [
            f"States: {self.n_states}",
            f"Iterations: {self.training_history_['iterations']}",
            f"Converged: {self.training_history_['converged']}",
            f"Training Time: {self.training_history_['training_time']:.2f}s",
            "",
            "State Characteristics:",
        ]
        
        for i in range(self.n_states):
            if self.emission_means_ is not None and self.emission_stds_ is not None:
                summary_text.append(f"  State {i}: μ={self.emission_means_[i]:.4f}, σ={self.emission_stds_[i]:.4f}")
        
        ax4.text(0.05, 0.95, '\n'.join(summary_text), transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Model Summary')
        
        plt.tight_layout()
        return fig
    
    def _initialize_parameters(self, returns: np.ndarray) -> None:
        """Initialize HMM parameters."""
        try:
            if HMM_UTILS_AVAILABLE and self.config.initialization_method == "kmeans":
                # Use sophisticated initialization
                initial_probs, transition_matrix, means, stds = initialize_parameters_kmeans(
                    self.n_states, returns, self.config.random_seed
                )
            else:
                # Simple random initialization
                initial_probs, transition_matrix, means, stds = self._initialize_parameters_simple(returns)
        except Exception as e:
            # Fallback to simple initialization on any error
            print(f"Warning: Sophisticated initialization failed ({e}), using simple initialization")
            initial_probs, transition_matrix, means, stds = self._initialize_parameters_simple(returns)
        
        self.initial_probs_ = initial_probs
        self.transition_matrix_ = transition_matrix
        self.emission_means_ = means
        self.emission_stds_ = stds
    
    def _initialize_parameters_simple(self, returns: np.ndarray) -> tuple:
        """Simple parameter initialization."""
        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        # Random initial probabilities
        initial_probs = np.random.dirichlet(np.ones(self.n_states))
        
        # Random transition matrix
        transition_matrix = np.random.dirichlet(np.ones(self.n_states), self.n_states)
        
        # Initialize emission parameters based on data quantiles
        try:
            means = np.percentile(returns, np.linspace(10, 90, self.n_states))
            returns_std = np.std(returns)
            if returns_std == 0 or np.isnan(returns_std) or np.isinf(returns_std):
                returns_std = 0.01  # Default volatility
            stds = np.full(self.n_states, max(returns_std, self.config.min_variance))
        except Exception:
            # Fallback if percentile calculation fails
            means = np.linspace(-0.01, 0.01, self.n_states)  # Default range
            stds = np.full(self.n_states, 0.01)  # Default volatility
        
        return initial_probs, transition_matrix, means, stds
    
    def _train_baum_welch(self, returns: np.ndarray) -> None:
        """Train using Baum-Welch algorithm."""
        start_time = datetime.now()
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.config.max_iterations):
            # E-step: Forward-backward algorithm
            log_likelihood, alpha, beta = self._forward_backward_with_scaling(returns)
            
            # Check convergence
            if iteration > 0:
                improvement = log_likelihood - prev_log_likelihood
                if improvement < self.config.tolerance:
                    self.training_history_['converged'] = True
                    break
            
            # M-step: Update parameters
            self._update_parameters(returns, alpha, beta)
            
            # Store training history
            self.training_history_['log_likelihoods'].append(log_likelihood)
            prev_log_likelihood = log_likelihood
        
        self.training_history_['iterations'] = iteration + 1
        self.training_history_['training_time'] = (datetime.now() - start_time).total_seconds()
    
    def _forward_backward(self, returns: np.ndarray) -> np.ndarray:
        """Simplified forward-backward algorithm returning state probabilities."""
        T = len(returns)
        
        # Forward pass
        alpha = np.zeros((T, self.n_states))
        alpha[0] = self.initial_probs_ * self._emission_probability(returns[0])
        
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_matrix_[:, j]) * self._emission_probability(returns[t])[j]
        
        # Backward pass
        beta = np.ones((T, self.n_states))
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.transition_matrix_[i] * self._emission_probability(returns[t+1]) * beta[t+1])
        
        # Normalize to get state probabilities
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        
        return gamma
    
    def _forward_backward_with_scaling(self, returns: np.ndarray) -> tuple:
        """Forward-backward with scaling to prevent underflow."""
        # Simplified implementation
        gamma = self._forward_backward(returns)
        log_likelihood = np.sum(np.log(np.sum(gamma, axis=1)))
        
        return log_likelihood, gamma, gamma  # Using gamma for both alpha and beta for simplicity
    
    def _update_parameters(self, returns: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> None:
        """Update model parameters in M-step."""
        T = len(returns)
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        
        # Update initial probabilities
        self.initial_probs_ = gamma[0] / np.sum(gamma[0])
        
        # Update transition matrix
        xi = np.zeros((T-1, self.n_states, self.n_states))
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (gamma[t, i] * self.transition_matrix_[i, j] * 
                                  self._emission_probability(returns[t+1])[j] * beta[t+1, j])
        
        xi = xi / np.sum(xi, axis=(1, 2), keepdims=True)
        self.transition_matrix_ = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0, keepdims=True)
        
        # Update emission parameters
        state_weights = np.sum(gamma, axis=0)
        for k in range(self.n_states):
            if state_weights[k] > 0:
                self.emission_means_[k] = np.sum(gamma[:, k] * returns) / state_weights[k]
                self.emission_stds_[k] = np.sqrt(
                    np.sum(gamma[:, k] * (returns - self.emission_means_[k])**2) / state_weights[k]
                )
                # Ensure minimum variance
                self.emission_stds_[k] = max(self.emission_stds_[k], self.config.min_variance)
    
    def _emission_probability(self, observation: float) -> np.ndarray:
        """Calculate emission probabilities for all states."""
        probs = np.zeros(self.n_states)
        for k in range(self.n_states):
            # Gaussian emission probability
            diff = observation - self.emission_means_[k]
            probs[k] = (1 / (self.emission_stds_[k] * np.sqrt(2 * np.pi))) * \
                      np.exp(-0.5 * (diff / self.emission_stds_[k])**2)
        return probs
    
    def _viterbi_decode(self, returns: np.ndarray) -> np.ndarray:
        """Viterbi algorithm for most likely state sequence."""
        T = len(returns)
        
        # Initialize
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        delta[0] = self.initial_probs_ * self._emission_probability(returns[0])
        
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                trans_scores = delta[t-1] * self.transition_matrix_[:, j]
                psi[t, j] = np.argmax(trans_scores)
                delta[t, j] = np.max(trans_scores) * self._emission_probability(returns[t])[j]
        
        # Backward pass
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1])
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states
    
    def _online_update(self, observations: pd.DataFrame) -> None:
        """Placeholder for online learning updates."""
        # This would implement online parameter updates
        # For now, just update the last observation
        if self.config.observed_signal in observations.columns:
            self._last_observation = observations[self.config.observed_signal].iloc[-1]
    
    def _validate_returns_simple(self, returns: np.ndarray) -> None:
        """Simple validation of returns data."""
        if len(returns) < 10:
            raise ValueError("Insufficient data for training (minimum 10 observations)")
        
        if np.isnan(returns).any():
            raise ValueError("Returns contain NaN values")
        
        if np.isinf(returns).any():
            raise ValueError("Returns contain infinite values")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted model."""
        return {
            'n_states': self.n_states,
            'is_fitted': self.is_fitted,
            'config': self.config.to_dict(),
            'training_history': self.training_history_,
            'emission_means': self.emission_means_.tolist() if self.emission_means_ is not None else None,
            'emission_stds': self.emission_stds_.tolist() if self.emission_stds_ is not None else None,
        }