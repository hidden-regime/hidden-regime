"""
State Standardization for Hidden Markov Models.

Provides standardized state configurations and data-driven selection algorithms
for economically meaningful regime detection in financial markets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Literal, Any
from dataclasses import dataclass
import warnings
from scipy import stats

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class StandardRegimeConfig:
    """Configuration for standardized market regime types."""
    
    n_states: int
    state_names: List[str]
    state_thresholds: Dict[str, Dict[str, float]]  # {state_name: {metric: threshold}}
    description: str
    typical_durations: Dict[str, float]  # Expected duration in days
    
    def __post_init__(self):
        if len(self.state_names) != self.n_states:
            raise ValueError(f"Number of state names ({len(self.state_names)}) must match n_states ({self.n_states})")


class StateStandardizer:
    """
    Standardizes HMM states for consistent regime interpretation across applications.
    
    Provides data-driven configuration selection and validates that detected states
    correspond to economically meaningful market regimes.
    """
    
    def __init__(self, regime_type: Literal['3_state', '4_state', '5_state', 'auto'] = '3_state'):
        self.regime_type = regime_type
        self.standard_configs = self._create_standard_configs()
        self.current_config = None if regime_type == 'auto' else self.standard_configs[regime_type]
    
    def _create_standard_configs(self) -> Dict[str, StandardRegimeConfig]:
        """Create standardized regime configurations."""
        
        configs = {}
        
        # 3-state configuration: Bear, Sideways, Bull
        configs['3_state'] = StandardRegimeConfig(
            n_states=3,
            state_names=['Bear', 'Sideways', 'Bull'],
            state_thresholds={
                'Bear': {'mean_return': -0.005, 'max_return': -0.002, 'min_volatility': 0.015},
                'Sideways': {'mean_return': 0.0, 'max_abs_return': 0.005, 'max_volatility': 0.025}, 
                'Bull': {'mean_return': 0.005, 'min_return': 0.002, 'min_volatility': 0.015}
            },
            description="Basic 3-regime model: Bear markets (negative trend), Sideways (range-bound), Bull markets (positive trend)",
            typical_durations={'Bear': 12.0, 'Sideways': 18.0, 'Bull': 15.0}
        )
        
        # 4-state configuration: Crisis, Bear, Sideways, Bull
        configs['4_state'] = StandardRegimeConfig(
            n_states=4,
            state_names=['Crisis', 'Bear', 'Sideways', 'Bull'],
            state_thresholds={
                'Crisis': {'mean_return': -0.02, 'max_return': -0.01, 'min_volatility': 0.03},
                'Bear': {'mean_return': -0.005, 'max_return': -0.002, 'max_volatility': 0.025},
                'Sideways': {'mean_return': 0.0, 'max_abs_return': 0.005, 'max_volatility': 0.02},
                'Bull': {'mean_return': 0.005, 'min_return': 0.002, 'max_volatility': 0.03}
            },
            description="Extended 4-regime model: Crisis (extreme negative), Bear (negative), Sideways (neutral), Bull (positive)",
            typical_durations={'Crisis': 5.0, 'Bear': 10.0, 'Sideways': 20.0, 'Bull': 15.0}
        )
        
        # 5-state configuration: Crisis, Bear, Sideways, Bull, Euphoric
        configs['5_state'] = StandardRegimeConfig(
            n_states=5,
            state_names=['Crisis', 'Bear', 'Sideways', 'Bull', 'Euphoric'],
            state_thresholds={
                'Crisis': {'mean_return': -0.02, 'max_return': -0.01, 'min_volatility': 0.03},
                'Bear': {'mean_return': -0.005, 'max_return': -0.002, 'max_volatility': 0.025},
                'Sideways': {'mean_return': 0.0, 'max_abs_return': 0.005, 'max_volatility': 0.02},
                'Bull': {'mean_return': 0.01, 'min_return': 0.005, 'max_return': 0.02},
                'Euphoric': {'mean_return': 0.02, 'min_return': 0.015, 'min_volatility': 0.025}
            },
            description="Full 5-regime model: Crisis (crash), Bear (decline), Sideways (neutral), Bull (growth), Euphoric (bubble)",
            typical_durations={'Crisis': 4.0, 'Bear': 8.0, 'Sideways': 25.0, 'Bull': 12.0, 'Euphoric': 6.0}
        )
        
        return configs
    
    def standardize_states(self, emission_params: np.ndarray) -> Dict[int, str]:
        """
        Map HMM states to standardized regime names based on emission parameters.
        
        Args:
            emission_params: HMM emission parameters [n_states, 2] with [mean, std]
            
        Returns:
            Dictionary mapping state indices to regime names
        """
        if self.current_config is None:
            raise ValueError("Cannot standardize states with regime_type='auto'. Call select_optimal_configuration first.")
        
        n_states = len(emission_params)
        if n_states != self.current_config.n_states:
            raise ValueError(f"Emission parameters have {n_states} states but config expects {self.current_config.n_states}")
        
        # Sort states by mean return (ascending)
        state_order = np.argsort(emission_params[:, 0])
        
        # Map to standardized names
        state_mapping = {}
        for i, state_idx in enumerate(state_order):
            state_mapping[state_idx] = self.current_config.state_names[i]
        
        return state_mapping
    
    def validate_interpretation(self, states: np.ndarray, returns: np.ndarray, 
                              emission_params: Optional[np.ndarray] = None) -> float:
        """
        Validate that detected states correspond to expected regime characteristics.
        
        Args:
            states: State sequence
            returns: Corresponding return sequence  
            emission_params: Optional emission parameters for additional validation
            
        Returns:
            Confidence score (0-1) indicating how well states match expected regimes
        """
        if self.current_config is None:
            raise ValueError("Cannot validate interpretation with regime_type='auto'")
        
        # Get state mapping
        if emission_params is not None:
            state_mapping = self.standardize_states(emission_params)
        else:
            # Create temporary mapping based on observed statistics
            state_stats = {}
            for state_idx in np.unique(states):
                state_returns = returns[states == state_idx]
                if len(state_returns) > 0:
                    state_stats[state_idx] = {
                        'mean': np.mean(state_returns),
                        'std': np.std(state_returns)
                    }
            
            # Sort by mean return
            sorted_states = sorted(state_stats.keys(), key=lambda x: state_stats[x]['mean'])
            state_mapping = {state: self.current_config.state_names[i] 
                           for i, state in enumerate(sorted_states)}
        
        # Calculate validation score for each state
        total_score = 0.0
        total_observations = 0
        
        for state_idx, regime_name in state_mapping.items():
            state_mask = states == state_idx
            if not state_mask.any():
                continue
                
            state_returns = returns[state_mask]
            n_obs = len(state_returns)
            total_observations += n_obs
            
            # Get expected thresholds for this regime
            thresholds = self.current_config.state_thresholds[regime_name]
            
            # Calculate individual validation scores
            scores = []
            
            # Validate mean return expectations
            mean_return = np.mean(state_returns)
            if 'mean_return' in thresholds:
                expected_mean = thresholds['mean_return']
                if regime_name == 'Sideways':
                    # For sideways, check if close to zero
                    mean_score = 1.0 - min(1.0, abs(mean_return) / 0.01)
                elif expected_mean > 0:
                    # Bull-type regimes: reward positive means
                    mean_score = 1.0 if mean_return >= expected_mean else max(0.0, mean_return / expected_mean)
                else:
                    # Bear-type regimes: reward negative means  
                    mean_score = 1.0 if mean_return <= expected_mean else max(0.0, -mean_return / -expected_mean)
                scores.append(mean_score)
            
            # Validate return bounds
            if 'max_return' in thresholds and mean_return > thresholds['max_return']:
                scores.append(0.0)  # Violates upper bound
            if 'min_return' in thresholds and mean_return < thresholds['min_return']:
                scores.append(0.0)  # Violates lower bound
                
            # Validate volatility expectations
            volatility = np.std(state_returns)
            if 'min_volatility' in thresholds:
                if volatility >= thresholds['min_volatility']:
                    scores.append(1.0)
                else:
                    scores.append(volatility / thresholds['min_volatility'])
            
            if 'max_volatility' in thresholds:
                if volatility <= thresholds['max_volatility']:
                    scores.append(1.0)
                else:
                    scores.append(thresholds['max_volatility'] / volatility)
            
            # Special validation for sideways markets
            if 'max_abs_return' in thresholds:
                abs_mean = abs(mean_return)
                if abs_mean <= thresholds['max_abs_return']:
                    scores.append(1.0)
                else:
                    scores.append(thresholds['max_abs_return'] / abs_mean)
            
            # Calculate weighted average score for this state
            if scores:
                state_score = np.mean(scores) * n_obs
                total_score += state_score
        
        # Return overall confidence score
        return total_score / total_observations if total_observations > 0 else 0.0
    
    def select_optimal_configuration(self, returns: np.ndarray, 
                                   validation_threshold: float = 0.7,
                                   max_iterations: int = 100) -> Tuple[str, float, Dict[str, Any]]:
        """
        Select optimal regime configuration based on data characteristics.
        
        Args:
            returns: Historical return data
            validation_threshold: Minimum validation score required
            max_iterations: Maximum training iterations per model
            
        Returns:
            Tuple of (best_regime_type, confidence_score, selection_details)
        """
        from .base_hmm import HiddenMarkovModel  # Avoid circular import
        from .config import HMMConfig
        
        if len(returns) < 100:
            warnings.warn("Limited data for optimal configuration selection. Results may be unreliable.")
        
        results = {}
        
        # Test each configuration
        for regime_type in ['3_state', '4_state', '5_state']:
            try:
                config = self.standard_configs[regime_type]
                n_states = config.n_states
                
                # Skip if insufficient data
                if len(returns) < n_states * 20:
                    warnings.warn(f"Insufficient data for {n_states}-state model")
                    continue
                
                # Train HMM
                hmm_config = HMMConfig(
                    n_states=n_states,
                    max_iterations=max_iterations,
                    initialization_method='kmeans',
                    early_stopping=True
                )
                
                hmm = HiddenMarkovModel(config=hmm_config)
                hmm.fit(returns, verbose=False)
                
                # Calculate model fit metrics
                log_likelihood = hmm.score(returns)
                n_params = n_states * (n_states - 1) + n_states * 2  # transitions + emissions
                aic = 2 * n_params - 2 * log_likelihood
                bic = np.log(len(returns)) * n_params - 2 * log_likelihood
                
                # Get state assignments and validate interpretation
                states = hmm.predict(returns)
                
                # Temporarily set current config for validation
                old_config = self.current_config
                self.current_config = config
                
                try:
                    validation_score = self.validate_interpretation(states, returns, hmm.emission_params_)
                finally:
                    self.current_config = old_config
                
                # Calculate regime persistence
                state_changes = np.sum(states[1:] != states[:-1])
                avg_duration = len(states) / (state_changes + 1)
                
                # Calculate regime separation (how distinct are the regimes)
                separation_score = self._calculate_regime_separation(states, returns)
                
                # Overall score combining multiple criteria
                # Prioritize: validation > BIC > separation > persistence
                combined_score = (
                    validation_score * 0.4 +
                    (1.0 / (1.0 + bic / 1000)) * 0.3 +  # Normalize BIC
                    separation_score * 0.2 +
                    min(1.0, avg_duration / 5.0) * 0.1  # Normalize duration
                )
                
                results[regime_type] = {
                    'n_states': n_states,
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'bic': bic,
                    'validation_score': validation_score,
                    'separation_score': separation_score,
                    'avg_duration': avg_duration,
                    'combined_score': combined_score,
                    'meets_threshold': validation_score >= validation_threshold
                }
                
            except Exception as e:
                warnings.warn(f"Failed to evaluate {regime_type} configuration: {e}")
                continue
        
        if not results:
            raise ValueError("No valid configurations could be evaluated")
        
        # Select best configuration
        # First filter by validation threshold, then by combined score
        valid_configs = {k: v for k, v in results.items() if v['meets_threshold']}
        
        if valid_configs:
            best_type = max(valid_configs.keys(), key=lambda x: valid_configs[x]['combined_score'])
            best_results = valid_configs[best_type]
        else:
            # If none meet threshold, take best validation score
            best_type = max(results.keys(), key=lambda x: results[x]['validation_score'])
            best_results = results[best_type]
            warnings.warn(f"No configuration meets validation threshold {validation_threshold}. "
                         f"Selected {best_type} with score {best_results['validation_score']:.3f}")
        
        # Update current configuration
        self.current_config = self.standard_configs[best_type]
        self.regime_type = best_type
        
        return best_type, best_results['validation_score'], results
    
    def _calculate_regime_separation(self, states: np.ndarray, returns: np.ndarray) -> float:
        """
        Calculate how well-separated the regimes are statistically.
        
        Args:
            states: State sequence
            returns: Return sequence
            
        Returns:
            Separation score (0-1)
        """
        unique_states = np.unique(states)
        if len(unique_states) < 2:
            return 0.0
        
        # Calculate pairwise Cohen's d between all regime pairs
        cohen_d_values = []
        
        for i in range(len(unique_states)):
            for j in range(i + 1, len(unique_states)):
                state1_returns = returns[states == unique_states[i]]
                state2_returns = returns[states == unique_states[j]]
                
                if len(state1_returns) > 1 and len(state2_returns) > 1:
                    # Cohen's d calculation
                    mean1, mean2 = np.mean(state1_returns), np.mean(state2_returns)
                    std1, std2 = np.std(state1_returns), np.std(state2_returns)
                    n1, n2 = len(state1_returns), len(state2_returns)
                    
                    # Pooled standard deviation
                    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                    
                    if pooled_std > 0:
                        cohen_d = abs(mean1 - mean2) / pooled_std
                        cohen_d_values.append(cohen_d)
        
        if not cohen_d_values:
            return 0.0
        
        # Convert average Cohen's d to 0-1 score
        avg_cohen_d = np.mean(cohen_d_values)
        return min(1.0, avg_cohen_d / 2.0)  # d=2.0 is considered very large effect
    
    def reorder_states_by_returns(self, emission_params: np.ndarray, 
                                  transition_matrix: np.ndarray,
                                  initial_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reorder HMM states to ensure consistent ordering by expected return.
        
        Args:
            emission_params: Original emission parameters [n_states, 2]
            transition_matrix: Original transition matrix [n_states, n_states]
            initial_probs: Original initial probabilities [n_states]
            
        Returns:
            Tuple of reordered (emission_params, transition_matrix, initial_probs)
        """
        # Get ordering by mean return (ascending)
        state_order = np.argsort(emission_params[:, 0])
        
        # Create reordering matrix
        n_states = len(emission_params)
        reorder_matrix = np.zeros((n_states, n_states))
        for new_idx, old_idx in enumerate(state_order):
            reorder_matrix[new_idx, old_idx] = 1.0
        
        # Reorder parameters
        new_emission_params = emission_params[state_order]
        new_transition_matrix = reorder_matrix @ transition_matrix @ reorder_matrix.T
        new_initial_probs = reorder_matrix @ initial_probs
        
        return new_emission_params, new_transition_matrix, new_initial_probs
    
    def get_regime_interpretation_enhanced(self, state_idx: int, emission_params: np.ndarray) -> Dict[str, Any]:
        """
        Get enhanced regime interpretation with standardized naming and confidence.
        
        Args:
            state_idx: State index
            emission_params: Emission parameters
            
        Returns:
            Dictionary with regime interpretation details
        """
        if self.current_config is None:
            raise ValueError("Cannot get interpretation with regime_type='auto'")
        
        state_mapping = self.standardize_states(emission_params)
        regime_name = state_mapping[state_idx]
        
        mean_return = emission_params[state_idx, 0]
        volatility = emission_params[state_idx, 1]
        
        # Get expected characteristics
        expected_thresholds = self.current_config.state_thresholds[regime_name]
        expected_duration = self.current_config.typical_durations[regime_name]
        
        return {
            'regime_name': regime_name,
            'state_index': state_idx,
            'mean_return': mean_return,
            'volatility': volatility,
            'annualized_return': mean_return * 252,
            'annualized_volatility': volatility * np.sqrt(252),
            'expected_duration_days': expected_duration,
            'expected_thresholds': expected_thresholds,
            'interpretation': f"{regime_name} (μ={mean_return:.4f}, σ={volatility:.4f})"
        }
    
    def get_available_configurations(self) -> Dict[str, str]:
        """Get description of available regime configurations."""
        return {
            regime_type: config.description 
            for regime_type, config in self.standard_configs.items()
        }