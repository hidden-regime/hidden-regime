"""
Regime Analysis Module
=====================

High-level analysis functions that wrap the Hidden Markov Model
for easy regime detection and analysis of financial time series.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Union, List, Any

from .data import DataLoader
from .models import HiddenMarkovModel, HMMConfig
from .config import DataConfig, ValidationConfig, PreprocessingConfig

class RegimeAnalyzer:
    """High-level regime analysis wrapper for the Hidden Markov Model"""
    
    def __init__(self, data_config: Optional[DataConfig] = None):
        self.data_config = data_config or DataConfig()
        self.data_loader = DataLoader(self.data_config)
    
    def analyze_stock(self, symbol: str, start_date: str, end_date: str, 
                     n_states: int = 3) -> Optional[Dict[str, Any]]:
        """
        Analyze regime patterns for a single stock
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            n_states: Number of regime states (default: 3)
            
        Returns:
            Dictionary with regime analysis results or None if failed
        """
        try:
            # Load stock data
            data = self.data_loader.load_stock_data(symbol, start_date, end_date)
            if data is None or len(data) < 50:
                return None
                
            # Calculate log returns (data already has log_return column)
            returns = data['log_return'].dropna()

            # Create HMM configuration
            hmm_config = HMMConfig.for_standardized_regimes(
                regime_type=f'{n_states}_state',
                conservative=False
            )
            
            # Fit HMM model
            model = HiddenMarkovModel(config=hmm_config)
            model.fit(returns.values, verbose=False)
            
            # Get predictions
            states = model.predict(returns.values)
            probabilities = model.predict_proba(returns.values)
            
            # Update model with latest observation for current state
            model.update_with_observation(returns.iloc[-1])
            current_regime_info = model.get_current_regime_info()
            
            # Calculate regime statistics
            regime_stats = self._calculate_regime_statistics(returns, states, n_states)
            
            # Determine current regime characteristics
            current_state = current_regime_info['most_likely_regime']
            current_confidence = current_regime_info['confidence']
            
            # Map state to regime name
            regime_mapping = self._get_regime_mapping(model.emission_params_)
            current_regime = regime_mapping.get(current_state, f'State_{current_state}')
            
            # Calculate days in current regime
            days_in_regime = self._calculate_days_in_current_regime(states)
            
            # Calculate regime change frequency
            regime_changes = np.sum(np.diff(states) != 0)
            
            return {
                'symbol': symbol,
                'current_regime': current_regime,
                'current_state': int(current_state),
                'confidence': float(current_confidence),
                'days_in_regime': days_in_regime,
                'regime_changes': int(regime_changes),
                'expected_return': float(current_regime_info['expected_return']),
                'expected_volatility': float(current_regime_info['expected_volatility']),
                'regime_probabilities': current_regime_info['regime_probabilities'],
                'regime_stats': regime_stats,
                'states': states,
                'probabilities': probabilities,
                'model': model,
                'data_length': len(returns)
            }
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
            return None
    
    def _calculate_regime_statistics(self, returns: pd.Series, states: np.ndarray, 
                                   n_states: int) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each regime"""
        stats = {}
        
        for state in range(n_states):
            state_mask = states == state
            state_returns = returns[state_mask]
            
            if len(state_returns) > 0:
                stats[f'state_{state}'] = {
                    'mean_return': float(state_returns.mean()),
                    'std_return': float(state_returns.std()),
                    'count': int(len(state_returns)),
                    'frequency': float(len(state_returns) / len(returns)),
                    'annualized_return': float(state_returns.mean() * 252),
                    'annualized_volatility': float(state_returns.std() * np.sqrt(252))
                }
            else:
                stats[f'state_{state}'] = {
                    'mean_return': 0.0,
                    'std_return': 0.0,
                    'count': 0,
                    'frequency': 0.0,
                    'annualized_return': 0.0,
                    'annualized_volatility': 0.0
                }
        
        return stats
    
    def _get_regime_mapping(self, emission_params: np.ndarray) -> Dict[int, str]:
        """Map states to regime names based on mean returns"""
        # Sort states by mean return
        state_returns = [(i, emission_params[i, 0]) for i in range(len(emission_params))]
        state_returns.sort(key=lambda x: x[1])
        
        if len(state_returns) == 3:
            # Standard 3-state mapping
            return {
                state_returns[0][0]: 'Bear',      # Lowest return
                state_returns[1][0]: 'Sideways',  # Middle return  
                state_returns[2][0]: 'Bull'       # Highest return
            }
        elif len(state_returns) == 2:
            return {
                state_returns[0][0]: 'Bear',
                state_returns[1][0]: 'Bull'
            }
        else:
            # Generic mapping for other state counts
            return {i: f'State_{i}' for i in range(len(state_returns))}
    
    def _calculate_days_in_current_regime(self, states: np.ndarray) -> int:
        """Calculate number of consecutive days in current regime"""
        if len(states) == 0:
            return 0
            
        current_state = states[-1]
        days = 1
        
        for i in range(len(states) - 2, -1, -1):
            if states[i] == current_state:
                days += 1
            else:
                break
                
        return days

class BayesianHMM:
    """Compatibility wrapper for BayesianHMM that uses HiddenMarkovModel"""
    
    def __init__(self, n_states: int = 3, n_iterations: int = 100):
        self.n_states = n_states
        self.n_iterations = n_iterations
        self.model = None
        
    def fit(self, data: np.ndarray):
        """Fit the model to data"""
        config = HMMConfig(
            n_states=self.n_states,
            max_iterations=self.n_iterations,
            regime_type='3_state' if self.n_states == 3 else 'auto',
            random_seed=42
        )
        
        self.model = HiddenMarkovModel(config=config)
        self.model.fit(data.flatten(), verbose=False)
        
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict states for data"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        return self.model.predict(data.flatten())
        
    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """Predict state probabilities"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        return self.model.predict_proba(data.flatten())