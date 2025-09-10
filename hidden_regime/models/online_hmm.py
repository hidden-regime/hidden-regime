"""
Online Hidden Markov Model for Real-Time Regime Detection

This module implements an online learning version of the Hidden Markov Model
that can adapt to new observations incrementally without requiring complete
retraining on the entire dataset.

Key Features:
- Incremental parameter updates with exponential forgetting
- Temporal consistency to prevent excessive historical revision
- Memory-efficient sufficient statistics tracking
- Real-time regime detection and transition prediction
- Robust change point detection for structural breaks

Author: aoaustin
Created: 2025-09-03
"""

import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from ..utils.exceptions import HMMInferenceError, HMMTrainingError
from .algorithms import HMMAlgorithms
from .base_hmm import HiddenMarkovModel
from .config import HMMConfig
from .utils import get_regime_interpretation, validate_returns_data


@dataclass
class OnlineHMMConfig:
    """Configuration for Online HMM learning parameters."""

    # Exponential forgetting parameters
    forgetting_factor: float = 0.98  # Memory decay rate (0.95-0.99)
    adaptation_rate: float = 0.05  # Learning speed (0.01-0.1)

    # Stability mechanisms
    min_observations_for_update: int = 10  # Minimum obs before parameter updates
    parameter_smoothing: bool = True  # Enable parameter smoothing
    smoothing_weight: float = 0.8  # Weight for old parameters (0.5-0.9)

    # Memory management
    rolling_window_size: int = 1000  # Max observations to keep in memory
    sufficient_stats_decay: float = 0.99  # Decay rate for sufficient statistics

    # Change detection
    enable_change_detection: bool = True  # Monitor for structural breaks
    change_detection_threshold: float = 3.0  # Std devs for change detection
    change_detection_window: int = 50  # Window size for change detection

    # Convergence monitoring
    convergence_tolerance: float = 1e-4  # Tolerance for parameter convergence
    max_adaptation_iterations: int = 5  # Max iterations per observation

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.9 <= self.forgetting_factor <= 0.999:
            raise ValueError("forgetting_factor must be between 0.9 and 0.999")
        if not 0.001 <= self.adaptation_rate <= 0.2:
            raise ValueError("adaptation_rate must be between 0.001 and 0.2")
        if not 0.1 <= self.smoothing_weight <= 0.99:
            raise ValueError("smoothing_weight must be between 0.1 and 0.99")
        if self.rolling_window_size < 100:
            raise ValueError("rolling_window_size must be at least 100")


@dataclass
class SufficientStatistics:
    """Sufficient statistics for online HMM parameter estimation."""

    # State occupation statistics
    gamma_sum: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Sum of state probabilities
    gamma_sum_t1: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Sum excluding first timestep

    # Transition statistics
    xi_sum: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Sum of transition probabilities

    # Emission statistics
    obs_sum: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Sum of observations * gamma
    obs_sq_sum: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Sum of observations^2 * gamma

    # Metadata
    total_weight: float = 0.0  # Total weight of observations
    n_observations: int = 0  # Number of observations processed

    def initialize(self, n_states: int):
        """Initialize arrays for given number of states."""
        self.gamma_sum = np.zeros(n_states)
        self.gamma_sum_t1 = np.zeros(n_states)
        self.xi_sum = np.zeros((n_states, n_states))
        self.obs_sum = np.zeros(n_states)
        self.obs_sq_sum = np.zeros(n_states)
        self.total_weight = 0.0
        self.n_observations = 0

    def decay(self, decay_factor: float):
        """Apply exponential decay to all statistics."""
        self.gamma_sum *= decay_factor
        self.gamma_sum_t1 *= decay_factor
        self.xi_sum *= decay_factor
        self.obs_sum *= decay_factor
        self.obs_sq_sum *= decay_factor
        self.total_weight *= decay_factor
        # Note: n_observations is not decayed as it tracks total processed


class OnlineHMM(HiddenMarkovModel):
    """
    Online Hidden Markov Model with incremental learning capabilities.

    This class extends the base HMM to support real-time parameter updates
    without requiring complete retraining on historical data. It uses
    exponential forgetting and sufficient statistics for memory efficiency.

    Attributes:
        online_config: Online learning configuration
        sufficient_stats: Running sufficient statistics
        observation_buffer: Fixed-size buffer of recent observations
        parameter_history: History of parameter evolution
        change_points: Detected structural break points
    """

    def __init__(
        self,
        n_states: Optional[int] = None,
        config: Optional[HMMConfig] = None,
        online_config: Optional[OnlineHMMConfig] = None,
    ):
        """
        Initialize Online HMM.

        Args:
            n_states: Number of hidden states
            config: Base HMM configuration
            online_config: Online learning configuration
        """
        super().__init__(n_states=n_states, config=config)

        # Online learning configuration
        self.online_config = online_config or OnlineHMMConfig()

        # Sufficient statistics for incremental updates
        self.sufficient_stats = SufficientStatistics()

        # Memory management
        self.observation_buffer = deque(maxlen=self.online_config.rolling_window_size)
        self.recent_returns = deque(maxlen=self.online_config.change_detection_window)

        # Parameter tracking
        self.parameter_history: List[Dict[str, Any]] = []
        self.change_points: List[int] = []

        # State tracking for real-time inference
        self._last_state_probs: Optional[np.ndarray] = None
        self._observations_since_last_update = 0
        self._total_observations_processed = 0

        # Change detection
        self._recent_likelihoods = deque(
            maxlen=self.online_config.change_detection_window
        )
        self._baseline_likelihood: Optional[float] = None

    def fit(
        self, returns: Union[np.ndarray, pd.Series], verbose: bool = False
    ) -> "OnlineHMM":
        """
        Initial training using standard batch EM, then initialize online components.

        Args:
            returns: Historical returns for initial training
            verbose: Whether to print training progress

        Returns:
            Self for method chaining
        """
        # Perform initial batch training using parent class
        super().fit(returns, verbose=verbose)

        # Initialize online learning components
        self._initialize_online_components(returns)

        if verbose:
            print(f"Online HMM initialized with {len(returns)} historical observations")
            print(
                f"Ready for real-time updates with forgetting factor: {self.online_config.forgetting_factor}"
            )

        return self

    def _initialize_online_components(
        self, initial_returns: Union[np.ndarray, pd.Series]
    ):
        """Initialize sufficient statistics and buffers from initial training data."""
        returns_array = validate_returns_data(initial_returns)

        # Initialize sufficient statistics
        self.sufficient_stats.initialize(self.n_states)

        # Populate observation buffer with recent data
        buffer_size = min(len(returns_array), self.online_config.rolling_window_size)
        for obs in returns_array[-buffer_size:]:
            self.observation_buffer.append(obs)

        # Initialize recent returns for change detection
        change_window_size = min(
            len(returns_array), self.online_config.change_detection_window
        )
        for obs in returns_array[-change_window_size:]:
            self.recent_returns.append(obs)

        # Compute initial sufficient statistics from full forward-backward pass
        self._compute_initial_sufficient_statistics(returns_array)

        # Initialize baseline likelihood for change detection
        if self.online_config.enable_change_detection:
            self._baseline_likelihood = self.score(
                returns_array[-self.online_config.change_detection_window :]
            )

        # Store initial parameters
        self._store_parameter_snapshot(0)

    def _compute_initial_sufficient_statistics(self, returns: np.ndarray):
        """Compute sufficient statistics from initial batch training."""
        # Use a simplified approach that doesn't require compute_posteriors
        # Since this is just initialization, we can use the trained parameters
        # and run forward-backward manually

        T = len(returns)

        # Extract emission parameters
        means = np.array([params[0] for params in self.emission_params_])
        stds = np.array([params[1] for params in self.emission_params_])

        # Compute emission probabilities
        log_emissions = HMMAlgorithms.log_emission_probability(returns, means, stds)

        # Forward pass
        log_alpha = np.zeros((T, self.n_states))
        log_alpha[0] = np.log(self.initial_probs_) + log_emissions[0]

        for t in range(1, T):
            for j in range(self.n_states):
                log_alpha[t, j] = (
                    logsumexp(log_alpha[t - 1] + np.log(self.transition_matrix_[:, j]))
                    + log_emissions[t, j]
                )

        # Backward pass
        log_beta = np.zeros((T, self.n_states))
        log_beta[T - 1] = 0  # log(1) = 0

        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = logsumexp(
                    np.log(self.transition_matrix_[i])
                    + log_emissions[t + 1]
                    + log_beta[t + 1]
                )

        # Compute gamma (state probabilities)
        log_gamma = log_alpha + log_beta
        # Normalize
        log_normalizer = logsumexp(log_gamma, axis=1, keepdims=True)
        log_gamma = log_gamma - log_normalizer
        gamma = np.exp(log_gamma)

        # Compute xi (transition probabilities) - simplified version
        xi = np.zeros((T - 1, self.n_states, self.n_states))
        for t in range(T - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = np.exp(
                        log_alpha[t, i]
                        + np.log(self.transition_matrix_[i, j])
                        + log_emissions[t + 1, j]
                        + log_beta[t + 1, j]
                        - logsumexp(log_alpha[T - 1])  # Normalizing constant
                    )

        # Update sufficient statistics
        self.sufficient_stats.gamma_sum = np.sum(gamma, axis=0)
        self.sufficient_stats.gamma_sum_t1 = np.sum(
            gamma[1:], axis=0
        )  # Exclude first timestep
        self.sufficient_stats.xi_sum = np.sum(xi, axis=0)

        # Emission statistics
        for k in range(self.n_states):
            self.sufficient_stats.obs_sum[k] = np.sum(gamma[:, k] * returns)
            self.sufficient_stats.obs_sq_sum[k] = np.sum(gamma[:, k] * returns**2)

        self.sufficient_stats.total_weight = len(returns)
        self.sufficient_stats.n_observations = len(returns)

    def add_observation(
        self, new_return: float, update_parameters: bool = True
    ) -> Dict[str, Any]:
        """
        Process a new observation and optionally update model parameters.

        Args:
            new_return: New log return observation
            update_parameters: Whether to update model parameters

        Returns:
            Dictionary with current regime information and diagnostics

        Raises:
            HMMInferenceError: If observation processing fails
        """
        try:
            if not np.isfinite(new_return):
                raise ValueError("New return must be finite")

            # Add to observation buffer
            self.observation_buffer.append(new_return)
            self.recent_returns.append(new_return)

            # Update counters
            self._observations_since_last_update += 1
            self._total_observations_processed += 1

            # Compute current state probabilities (forward step)
            current_state_probs = self._forward_step(new_return)

            # Change point detection
            change_detected = False
            if self.online_config.enable_change_detection:
                change_detected = self._detect_change_point(new_return)

            # Parameter update decision
            should_update = (
                update_parameters
                and self._observations_since_last_update
                >= self.online_config.min_observations_for_update
            )

            if should_update or change_detected:
                self._update_parameters_online(
                    new_return, current_state_probs, change_detected
                )
                self._observations_since_last_update = 0

            # Store current state for next iteration
            self._last_state_probs = current_state_probs.copy()

            # Generate regime information
            regime_info = self._generate_regime_info(
                current_state_probs, change_detected
            )

            return regime_info

        except Exception as e:
            raise HMMInferenceError(f"Failed to process observation: {e}")

    def _forward_step(self, observation: float) -> np.ndarray:
        """Perform single forward step to get current state probabilities."""
        if self._last_state_probs is None:
            # First observation - use initial probabilities
            prior_probs = self.initial_probs_
        else:
            # Predict next state probabilities using transition matrix
            prior_probs = np.dot(self._last_state_probs, self.transition_matrix_)

        # Compute emission probabilities
        from scipy.stats import norm

        emission_probs = np.array(
            [
                norm.pdf(observation, loc=params[0], scale=params[1])
                for params in self.emission_params_
            ]
        )

        # Update state probabilities (unnormalized)
        unnormalized_probs = prior_probs * emission_probs

        # Normalize
        total_prob = np.sum(unnormalized_probs)
        if total_prob > 0:
            normalized_probs = unnormalized_probs / total_prob
        else:
            # Fallback to uniform distribution
            normalized_probs = np.ones(self.n_states) / self.n_states
            warnings.warn("Zero likelihood encountered, using uniform distribution")

        return normalized_probs

    def _detect_change_point(self, new_return: float) -> bool:
        """Detect structural breaks using likelihood monitoring."""
        if len(self.recent_returns) < self.online_config.change_detection_window:
            return False

        # Compute likelihood of recent window
        recent_data = np.array(list(self.recent_returns))
        current_likelihood = self.score(recent_data)
        self._recent_likelihoods.append(current_likelihood)

        if self._baseline_likelihood is None:
            self._baseline_likelihood = current_likelihood
            return False

        # Compare to baseline
        likelihood_change = current_likelihood - self._baseline_likelihood

        # Update baseline with exponential moving average
        self._baseline_likelihood = (
            0.95 * self._baseline_likelihood + 0.05 * current_likelihood
        )

        # Detect significant drops in likelihood
        if len(self._recent_likelihoods) >= 10:
            recent_std = np.std(list(self._recent_likelihoods)[-10:])
            threshold = self.online_config.change_detection_threshold * recent_std

            if likelihood_change < -threshold:
                self.change_points.append(self._total_observations_processed)
                return True

        return False

    def _update_parameters_online(
        self, observation: float, state_probs: np.ndarray, change_detected: bool
    ):
        """Update model parameters using incremental learning."""
        # Apply exponential forgetting to sufficient statistics
        if not change_detected:  # Normal adaptation
            decay_factor = self.online_config.forgetting_factor
        else:  # Faster adaptation after change point
            decay_factor = self.online_config.forgetting_factor**2

        self.sufficient_stats.decay(decay_factor)

        # Update sufficient statistics with new observation
        self._update_sufficient_statistics(observation, state_probs)

        # Estimate new parameters from sufficient statistics
        new_initial_probs, new_transition_matrix, new_emission_params = (
            self._estimate_parameters_from_stats()
        )

        # Apply parameter smoothing to prevent instability
        if self.online_config.parameter_smoothing:
            self._apply_parameter_smoothing(
                new_initial_probs, new_transition_matrix, new_emission_params
            )
        else:
            # Direct update
            self.initial_probs_ = new_initial_probs
            self.transition_matrix_ = new_transition_matrix
            self.emission_params_ = new_emission_params

        # Store parameter snapshot
        self._store_parameter_snapshot(self._total_observations_processed)

    def _update_sufficient_statistics(
        self, observation: float, state_probs: np.ndarray
    ):
        """Update sufficient statistics with new observation and state probabilities."""
        # State occupation counts
        self.sufficient_stats.gamma_sum += state_probs

        # Transition counts (approximation using current and previous state probs)
        if self._last_state_probs is not None:
            # Approximate transition probabilities
            transition_contrib = np.outer(self._last_state_probs, state_probs)
            self.sufficient_stats.xi_sum += transition_contrib

        # Emission statistics
        for k in range(self.n_states):
            weight = state_probs[k]
            self.sufficient_stats.obs_sum[k] += weight * observation
            self.sufficient_stats.obs_sq_sum[k] += weight * observation**2

        # Update metadata
        self.sufficient_stats.total_weight += 1.0
        self.sufficient_stats.n_observations += 1

    def _estimate_parameters_from_stats(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
        """Estimate HMM parameters from current sufficient statistics."""
        # Initial state probabilities (not typically updated in online setting)
        new_initial_probs = self.initial_probs_.copy()

        # Transition probabilities
        new_transition_matrix = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            row_sum = np.sum(self.sufficient_stats.xi_sum[i, :])
            if row_sum > 0:
                new_transition_matrix[i, :] = (
                    self.sufficient_stats.xi_sum[i, :] / row_sum
                )
            else:
                # Fallback to current values
                new_transition_matrix[i, :] = self.transition_matrix_[i, :]

        # Emission parameters (means and standard deviations)
        new_emission_params = []
        for k in range(self.n_states):
            if self.sufficient_stats.gamma_sum[k] > 0:
                # Mean
                mean = (
                    self.sufficient_stats.obs_sum[k]
                    / self.sufficient_stats.gamma_sum[k]
                )

                # Variance (with numerical stability)
                mean_sq = (
                    self.sufficient_stats.obs_sq_sum[k]
                    / self.sufficient_stats.gamma_sum[k]
                )
                variance = max(mean_sq - mean**2, 1e-6)  # Ensure positive variance
                std = np.sqrt(variance)

                new_emission_params.append((mean, std))
            else:
                # Fallback to current parameters
                new_emission_params.append(self.emission_params_[k])

        return new_initial_probs, new_transition_matrix, new_emission_params

    def _apply_parameter_smoothing(
        self,
        new_initial_probs: np.ndarray,
        new_transition_matrix: np.ndarray,
        new_emission_params: List[Tuple[float, float]],
    ):
        """Apply exponential smoothing to parameter updates."""
        weight_old = self.online_config.smoothing_weight
        weight_new = 1.0 - weight_old

        # Smooth initial probabilities
        self.initial_probs_ = (
            weight_old * self.initial_probs_ + weight_new * new_initial_probs
        )

        # Smooth transition matrix
        self.transition_matrix_ = (
            weight_old * self.transition_matrix_ + weight_new * new_transition_matrix
        )

        # Smooth emission parameters
        for k in range(self.n_states):
            old_mean, old_std = self.emission_params_[k]
            new_mean, new_std = new_emission_params[k]

            smoothed_mean = weight_old * old_mean + weight_new * new_mean
            smoothed_std = weight_old * old_std + weight_new * new_std

            self.emission_params_[k] = (smoothed_mean, smoothed_std)

    def _generate_regime_info(
        self, state_probs: np.ndarray, change_detected: bool
    ) -> Dict[str, Any]:
        """Generate comprehensive regime information for current state."""
        # Current regime (most likely state)
        current_regime = np.argmax(state_probs)
        regime_confidence = state_probs[current_regime]

        # Regime interpretation
        mean, std = self.emission_params_[current_regime]
        interpretation = get_regime_interpretation(
            current_regime, self.emission_params_
        )

        return {
            "regime": current_regime,
            "regime_probabilities": state_probs.tolist(),
            "confidence": float(regime_confidence),
            "regime_interpretation": interpretation,
            "regime_characteristics": {
                "mean_return": float(mean),
                "volatility": float(std),
                "regime_id": int(current_regime),
            },
            "diagnostics": {
                "change_detected": change_detected,
                "total_observations": self._total_observations_processed,
                "observations_since_update": self._observations_since_last_update,
                "buffer_size": len(self.observation_buffer),
                "n_change_points": len(self.change_points),
            },
        }

    def _store_parameter_snapshot(self, observation_count: int):
        """Store snapshot of current parameters for analysis."""
        snapshot = {
            "observation_count": observation_count,
            "initial_probs": self.initial_probs_.copy(),
            "transition_matrix": self.transition_matrix_.copy(),
            "emission_params": [(mean, std) for mean, std in self.emission_params_],
            "sufficient_stats_weight": self.sufficient_stats.total_weight,
        }

        self.parameter_history.append(snapshot)

        # Keep only recent history to manage memory
        max_history = 1000
        if len(self.parameter_history) > max_history:
            self.parameter_history = self.parameter_history[-max_history:]

    def get_parameter_evolution(self) -> pd.DataFrame:
        """Get DataFrame showing parameter evolution over time."""
        if not self.parameter_history:
            return pd.DataFrame()

        records = []
        for snapshot in self.parameter_history:
            record = {"observation_count": snapshot["observation_count"]}

            # Add emission parameters
            for k in range(self.n_states):
                mean, std = snapshot["emission_params"][k]
                record[f"state_{k}_mean"] = mean
                record[f"state_{k}_std"] = std

            # Add transition probabilities (diagonal elements for persistence)
            for k in range(self.n_states):
                record[f"state_{k}_persistence"] = snapshot["transition_matrix"][k, k]

            records.append(record)

        return pd.DataFrame(records)

    def reset_adaptation(self):
        """Reset online learning components while keeping base model."""
        self.sufficient_stats.initialize(self.n_states)
        self.observation_buffer.clear()
        self.recent_returns.clear()
        self.parameter_history.clear()
        self.change_points.clear()
        self._last_state_probs = None
        self._observations_since_last_update = 0
        self._total_observations_processed = 0
        self._recent_likelihoods.clear()
        self._baseline_likelihood = None

    def __repr__(self) -> str:
        """String representation of OnlineHMM."""
        status = "fitted" if self.is_fitted else "unfitted"
        return (
            f"OnlineHMM(n_states={self.n_states}, {status}, "
            f"total_obs={self._total_observations_processed}, "
            f"change_points={len(self.change_points)})"
        )
