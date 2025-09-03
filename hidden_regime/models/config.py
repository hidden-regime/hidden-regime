"""
Configuration classes for Hidden Markov Model implementation.

Provides dataclass-based configuration for HMM training parameters,
initialization strategies, and inference settings.
"""

from dataclasses import dataclass
from typing import Optional, Union, List, Literal
import numpy as np


@dataclass
class HMMConfig:
    """Configuration for Hidden Markov Model training and inference."""
    
    # Model structure
    n_states: int = 3
    
    # Training parameters
    max_iterations: int = 100
    tolerance: float = 1e-6
    regularization: float = 1e-6
    
    # Initialization strategy
    initialization_method: Literal['random', 'kmeans', 'custom'] = 'random'
    random_seed: Optional[int] = None
    
    # Model constraints
    min_regime_duration: int = 2  # Minimum days in regime
    min_variance: float = 1e-8    # Minimum allowed variance
    
    # Convergence criteria
    check_convergence_every: int = 5  # Check convergence every N iterations
    early_stopping: bool = True
    
    # Numerical stability
    log_likelihood_threshold: float = -1e10  # Prevent extreme likelihoods
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_states < 2:
            raise ValueError("Number of states must be at least 2")
        
        if self.n_states > 10:
            raise ValueError("Number of states should not exceed 10 for practical use")
        
        if self.max_iterations < 1:
            raise ValueError("Maximum iterations must be positive")
        
        if not 0 < self.tolerance < 1:
            raise ValueError("Tolerance must be between 0 and 1")
        
        if self.regularization < 0:
            raise ValueError("Regularization must be non-negative")
        
        if self.min_regime_duration < 1:
            raise ValueError("Minimum regime duration must be at least 1")
        
        if self.min_variance <= 0:
            raise ValueError("Minimum variance must be positive")
        
        if self.check_convergence_every < 1:
            raise ValueError("Convergence check frequency must be positive")
    
    @classmethod
    def for_market_data(cls, conservative: bool = False) -> 'HMMConfig':
        """
        Create configuration optimized for market data.
        
        Args:
            conservative: If True, use more conservative settings for stability
            
        Returns:
            HMMConfig optimized for financial time series
        """
        if conservative:
            return cls(
                n_states=3,
                max_iterations=200,
                tolerance=1e-7,
                regularization=1e-5,
                initialization_method='kmeans',
                min_regime_duration=3,
                early_stopping=True
            )
        else:
            return cls(
                n_states=3,
                max_iterations=100,
                tolerance=1e-6,
                regularization=1e-6,
                initialization_method='random',
                min_regime_duration=2,
                early_stopping=True
            )
    
    @classmethod
    def for_high_frequency(cls) -> 'HMMConfig':
        """
        Create configuration for high-frequency data.
        
        Returns:
            HMMConfig optimized for high-frequency trading data
        """
        return cls(
            n_states=4,  # More states for intraday patterns
            max_iterations=50,   # Faster training
            tolerance=1e-5,      # Less strict convergence
            regularization=1e-5, # More regularization
            initialization_method='kmeans',
            min_regime_duration=1,  # Allow shorter regimes
            early_stopping=True
        )
    
    def validate_for_data(self, n_observations: int) -> None:
        """
        Validate configuration against data characteristics.
        
        Args:
            n_observations: Number of observations in training data
            
        Raises:
            ValueError: If configuration is inappropriate for data size
        """
        # Check minimum data requirements
        min_data_per_state = 10  # Reasonable minimum for parameter estimation
        required_observations = self.n_states * min_data_per_state
        
        if n_observations < required_observations:
            raise ValueError(
                f"Insufficient data for {self.n_states} states. "
                f"Need at least {required_observations} observations, got {n_observations}"
            )
        
        # Warn about potential overfitting
        if n_observations < self.n_states * 20:
            import warnings
            warnings.warn(
                f"Limited data ({n_observations} observations) for {self.n_states} states. "
                f"Consider reducing n_states or increasing data size for better results."
            )