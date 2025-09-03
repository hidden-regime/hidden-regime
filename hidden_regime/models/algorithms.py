"""
Core algorithms for Hidden Markov Model implementation.

Implements Forward-Backward, Viterbi, and Baum-Welch algorithms
with numerical stability enhancements for market regime detection.
"""

import numpy as np
from typing import Tuple, Optional
import warnings
from scipy.stats import norm
from scipy.special import logsumexp


class HMMAlgorithms:
    """Core algorithms for HMM training and inference."""
    
    @staticmethod
    def log_emission_probability(
        observations: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray
    ) -> np.ndarray:
        """
        Calculate log emission probabilities for Gaussian distributions.
        
        Args:
            observations: Observation sequence (T,)
            means: State means (n_states,)
            stds: State standard deviations (n_states,)
            
        Returns:
            Log probabilities (T, n_states)
        """
        T = len(observations)
        n_states = len(means)
        
        log_probs = np.zeros((T, n_states))
        
        for t in range(T):
            for state in range(n_states):
                # Use scipy.stats.norm for numerical stability
                log_probs[t, state] = norm.logpdf(
                    observations[t], 
                    loc=means[state], 
                    scale=stds[state]
                )
        
        return log_probs
    
    @staticmethod
    def forward_algorithm(
        observations: np.ndarray,
        initial_probs: np.ndarray,
        transition_matrix: np.ndarray,
        emission_params: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm for computing forward probabilities.
        
        Args:
            observations: Observation sequence (T,)
            initial_probs: Initial state probabilities (n_states,)
            transition_matrix: Transition probabilities (n_states, n_states)
            emission_params: Emission parameters (n_states, 2) [means, stds]
            
        Returns:
            Tuple of (forward_probs, log_likelihood)
            forward_probs: (T, n_states) log forward probabilities
        """
        T = len(observations)
        n_states = len(initial_probs)
        
        # Initialize forward probabilities in log space
        forward_probs = np.zeros((T, n_states))
        
        # Get emission probabilities
        log_emissions = HMMAlgorithms.log_emission_probability(
            observations, emission_params[:, 0], emission_params[:, 1]
        )
        
        # Initialize (t=0)
        forward_probs[0] = np.log(initial_probs + 1e-10) + log_emissions[0]
        
        # Forward pass
        log_trans = np.log(transition_matrix + 1e-10)  # Add small value to avoid log(0)
        
        for t in range(1, T):
            for j in range(n_states):
                # Compute log P(state_j at t | observations[0:t])
                log_sum_terms = forward_probs[t-1] + log_trans[:, j]
                forward_probs[t, j] = logsumexp(log_sum_terms) + log_emissions[t, j]
        
        # Compute total log likelihood
        log_likelihood = logsumexp(forward_probs[T-1])
        
        return forward_probs, log_likelihood
    
    @staticmethod
    def backward_algorithm(
        observations: np.ndarray,
        transition_matrix: np.ndarray,
        emission_params: np.ndarray
    ) -> np.ndarray:
        """
        Backward algorithm for computing backward probabilities.
        
        Args:
            observations: Observation sequence (T,)
            transition_matrix: Transition probabilities (n_states, n_states)
            emission_params: Emission parameters (n_states, 2)
            
        Returns:
            Backward probabilities (T, n_states) in log space
        """
        T = len(observations)
        n_states = transition_matrix.shape[0]
        
        # Initialize backward probabilities
        backward_probs = np.zeros((T, n_states))
        
        # Get emission probabilities
        log_emissions = HMMAlgorithms.log_emission_probability(
            observations, emission_params[:, 0], emission_params[:, 1]
        )
        
        # Initialize (t=T-1): log(1) = 0 for all states
        backward_probs[T-1] = 0.0
        
        # Backward pass
        log_trans = np.log(transition_matrix + 1e-10)
        
        for t in range(T-2, -1, -1):
            for i in range(n_states):
                # Compute log P(observations[t+1:T] | state_i at t)
                log_sum_terms = (log_trans[i, :] + 
                               log_emissions[t+1] + 
                               backward_probs[t+1])
                backward_probs[t, i] = logsumexp(log_sum_terms)
        
        return backward_probs
    
    @staticmethod
    def viterbi_algorithm(
        observations: np.ndarray,
        initial_probs: np.ndarray,
        transition_matrix: np.ndarray,
        emission_params: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Viterbi algorithm for finding the most likely state sequence.
        
        Args:
            observations: Observation sequence (T,)
            initial_probs: Initial state probabilities (n_states,)
            transition_matrix: Transition probabilities (n_states, n_states)
            emission_params: Emission parameters (n_states, 2)
            
        Returns:
            Tuple of (best_path, best_prob)
            best_path: Most likely state sequence (T,)
            best_prob: Log probability of best path
        """
        T = len(observations)
        n_states = len(initial_probs)
        
        # Initialize Viterbi tables
        viterbi_probs = np.zeros((T, n_states))
        path_indices = np.zeros((T, n_states), dtype=int)
        
        # Get emission probabilities
        log_emissions = HMMAlgorithms.log_emission_probability(
            observations, emission_params[:, 0], emission_params[:, 1]
        )
        
        # Initialize (t=0)
        viterbi_probs[0] = np.log(initial_probs + 1e-10) + log_emissions[0]
        
        # Forward pass
        log_trans = np.log(transition_matrix + 1e-10)
        
        for t in range(1, T):
            for j in range(n_states):
                # Find most likely previous state
                transition_probs = viterbi_probs[t-1] + log_trans[:, j]
                path_indices[t, j] = np.argmax(transition_probs)
                viterbi_probs[t, j] = np.max(transition_probs) + log_emissions[t, j]
        
        # Backtrack to find best path
        best_path = np.zeros(T, dtype=int)
        
        # Find best final state
        best_path[T-1] = np.argmax(viterbi_probs[T-1])
        best_prob = np.max(viterbi_probs[T-1])
        
        # Backtrack
        for t in range(T-2, -1, -1):
            best_path[t] = path_indices[t+1, best_path[t+1]]
        
        return best_path, best_prob
    
    @staticmethod
    def forward_backward_algorithm(
        observations: np.ndarray,
        initial_probs: np.ndarray,
        transition_matrix: np.ndarray,
        emission_params: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Combined forward-backward algorithm for state probabilities.
        
        Args:
            observations: Observation sequence (T,)
            initial_probs: Initial state probabilities (n_states,)
            transition_matrix: Transition probabilities (n_states, n_states)
            emission_params: Emission parameters (n_states, 2)
            
        Returns:
            Tuple of (gamma, xi, log_likelihood)
            gamma: State probabilities (T, n_states)
            xi: Pairwise transition probabilities (T-1, n_states, n_states)
        """
        # Forward pass
        forward_probs, log_likelihood = HMMAlgorithms.forward_algorithm(
            observations, initial_probs, transition_matrix, emission_params
        )
        
        # Backward pass
        backward_probs = HMMAlgorithms.backward_algorithm(
            observations, transition_matrix, emission_params
        )
        
        T, n_states = forward_probs.shape
        
        # Compute gamma (state probabilities)
        gamma = forward_probs + backward_probs
        # Normalize in log space
        for t in range(T):
            gamma[t] = gamma[t] - logsumexp(gamma[t])
        
        # Convert to probability space
        gamma = np.exp(gamma)
        
        # Compute xi (pairwise transition probabilities)
        xi = np.zeros((T-1, n_states, n_states))
        
        log_emissions = HMMAlgorithms.log_emission_probability(
            observations, emission_params[:, 0], emission_params[:, 1]
        )
        log_trans = np.log(transition_matrix + 1e-10)
        
        for t in range(T-1):
            for i in range(n_states):
                for j in range(n_states):
                    xi[t, i, j] = (forward_probs[t, i] + 
                                 log_trans[i, j] + 
                                 log_emissions[t+1, j] + 
                                 backward_probs[t+1, j])
            
            # Normalize xi[t] in log space
            xi_t_flat = xi[t].flatten()
            xi_t_normalized = xi_t_flat - logsumexp(xi_t_flat)
            xi[t] = np.exp(xi_t_normalized.reshape(n_states, n_states))
        
        return gamma, xi, log_likelihood
    
    @staticmethod
    def baum_welch_update(
        observations: np.ndarray,
        gamma: np.ndarray,
        xi: np.ndarray,
        regularization: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Baum-Welch parameter update step.
        
        Args:
            observations: Observation sequence (T,)
            gamma: State probabilities (T, n_states)
            xi: Pairwise transition probabilities (T-1, n_states, n_states)
            regularization: Regularization parameter
            
        Returns:
            Tuple of (new_initial_probs, new_transition_matrix, new_emission_params)
        """
        T, n_states = gamma.shape
        
        # Update initial probabilities
        new_initial_probs = gamma[0] + regularization
        new_initial_probs /= np.sum(new_initial_probs)
        
        # Update transition matrix
        new_transition_matrix = np.sum(xi, axis=0) + regularization
        # Normalize rows
        row_sums = np.sum(new_transition_matrix, axis=1, keepdims=True)
        new_transition_matrix /= row_sums
        
        # Update emission parameters (means and standard deviations)
        new_emission_params = np.zeros((n_states, 2))
        
        for state in range(n_states):
            # Weighted mean
            state_weights = gamma[:, state] + regularization
            weight_sum = np.sum(state_weights)
            
            weighted_mean = np.sum(state_weights * observations) / weight_sum
            new_emission_params[state, 0] = weighted_mean
            
            # Weighted standard deviation
            weighted_var = np.sum(state_weights * (observations - weighted_mean)**2) / weight_sum
            new_emission_params[state, 1] = np.sqrt(weighted_var + regularization)
        
        return new_initial_probs, new_transition_matrix, new_emission_params
    
    @staticmethod
    def compute_likelihood(
        observations: np.ndarray,
        initial_probs: np.ndarray,
        transition_matrix: np.ndarray,
        emission_params: np.ndarray
    ) -> float:
        """
        Compute log-likelihood of observations given model parameters.
        
        Args:
            observations: Observation sequence
            initial_probs: Initial state probabilities
            transition_matrix: Transition probabilities
            emission_params: Emission parameters
            
        Returns:
            Log-likelihood of the observations
        """
        _, log_likelihood = HMMAlgorithms.forward_algorithm(
            observations, initial_probs, transition_matrix, emission_params
        )
        return log_likelihood
    
    @staticmethod
    def predict_next_state_probs(
        current_state_probs: np.ndarray,
        transition_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Predict next state probabilities given current state distribution.
        
        Args:
            current_state_probs: Current state probabilities (n_states,)
            transition_matrix: Transition probabilities (n_states, n_states)
            
        Returns:
            Next state probabilities (n_states,)
        """
        return current_state_probs @ transition_matrix
    
    @staticmethod
    def decode_states_online(
        new_observation: float,
        prev_state_probs: np.ndarray,
        transition_matrix: np.ndarray,
        emission_params: np.ndarray
    ) -> np.ndarray:
        """
        Online state decoding for new observation.
        
        Args:
            new_observation: New observation value
            prev_state_probs: Previous state probabilities
            transition_matrix: Transition matrix
            emission_params: Emission parameters
            
        Returns:
            Updated state probabilities
        """
        n_states = len(prev_state_probs)
        
        # Predict step
        predicted_probs = HMMAlgorithms.predict_next_state_probs(
            prev_state_probs, transition_matrix
        )
        
        # Update step with new observation
        emission_probs = np.zeros(n_states)
        for state in range(n_states):
            emission_probs[state] = norm.pdf(
                new_observation,
                loc=emission_params[state, 0],
                scale=emission_params[state, 1]
            )
        
        # Combine prediction and observation
        posterior_probs = predicted_probs * emission_probs
        
        # Normalize
        if np.sum(posterior_probs) > 0:
            posterior_probs /= np.sum(posterior_probs)
        else:
            # Fallback to uniform distribution if all probabilities are zero
            posterior_probs = np.ones(n_states) / n_states
        
        return posterior_probs