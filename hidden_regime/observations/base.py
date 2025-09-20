"""
Base observation generation functionality.

Provides base classes and utilities for generating observations from raw data
that can be used by models for training and prediction.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..pipeline.interfaces import ObservationComponent
from ..config.observation import ObservationConfig
from ..utils.exceptions import ValidationError


class BaseObservationGenerator(ObservationComponent):
    """
    Base class for observation generation components.
    
    Provides common functionality for transforming raw data into observations
    including validation, generator management, and plotting capabilities.
    """
    
    def __init__(self, config: ObservationConfig):
        """
        Initialize observation generator with configuration.
        
        Args:
            config: Observation configuration object
        """
        self.config = config
        self.generators = self._parse_generators(config.generators)
        self.last_data = None
        self.last_observations = None
    
    def _parse_generators(self, generators: List[Union[str, Callable]]) -> List[Callable]:
        """
        Parse generator specifications into callable functions.
        
        Args:
            generators: List of generator specifications (strings or callables)
            
        Returns:
            List of callable generator functions
        """
        parsed_generators = []
        
        for generator in generators:
            if callable(generator):
                parsed_generators.append(generator)
            elif isinstance(generator, str):
                # Try to resolve string to built-in generator
                generator_func = self._get_builtin_generator(generator)
                if generator_func is None:
                    raise ValidationError(f"Unknown generator: {generator}")
                parsed_generators.append(generator_func)
            else:
                raise ValidationError(f"Generator must be string or callable, got {type(generator)}")
        
        return parsed_generators
    
    def _get_builtin_generator(self, name: str) -> Callable:
        """
        Get built-in generator function by name.
        
        Args:
            name: Name of built-in generator
            
        Returns:
            Generator function or None if not found
        """
        builtin_generators = {
            "log_return": self._generate_log_return,
            "return_ratio": self._generate_return_ratio,
            "price_change": self._generate_price_change,
            "volatility": self._generate_volatility,
        }
        
        return builtin_generators.get(name)
    
    def update(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate observations from input data.
        
        Args:
            data: Input data DataFrame
            
        Returns:
            DataFrame with generated observations
        """
        if data.empty:
            raise ValidationError("Input data cannot be empty")
        
        # Store reference for plotting
        self.last_data = data.copy()
        
        # Generate observations using all configured generators
        observations = data.copy()
        
        for generator in self.generators:
            try:
                new_observations = generator(observations)
                
                # Merge new observations with existing ones
                if isinstance(new_observations, pd.DataFrame):
                    observations = pd.concat([observations, new_observations], axis=1)
                elif isinstance(new_observations, pd.Series):
                    observations[new_observations.name or 'observation'] = new_observations
                else:
                    raise ValidationError(f"Generator must return DataFrame or Series, got {type(new_observations)}")
                    
            except Exception as e:
                raise ValidationError(f"Generator {generator.__name__} failed: {str(e)}")
        
        # Remove duplicate columns (keep last)
        observations = observations.loc[:, ~observations.columns.duplicated(keep='last')]
        
        # Store for plotting
        self.last_observations = observations.copy()
        
        return observations
    
    def plot(self, **kwargs) -> plt.Figure:
        """
        Generate visualization of observations.
        
        Returns:
            matplotlib Figure with observation plots
        """
        if self.last_observations is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No observations generated yet', ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        # Plot each observation column
        observation_cols = [col for col in self.last_observations.columns 
                           if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if not observation_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No observation columns to plot', ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        n_cols = min(len(observation_cols), 4)  # Max 4 subplots
        fig, axes = plt.subplots(n_cols, 1, figsize=(12, 3 * n_cols))
        
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(observation_cols[:n_cols]):
            ax = axes[i]
            data = self.last_observations[col].dropna()
            
            if len(data) > 0:
                ax.plot(data.index, data.values, label=col)
                ax.set_title(f'Observation: {col}')
                ax.set_ylabel(col)
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        plt.tight_layout()
        return fig
    
    # Built-in generator functions
    def _generate_log_return(self, data: pd.DataFrame) -> pd.Series:
        """Generate log returns from price data, preserving existing calculations."""
        # Check if log_return already exists (e.g., from financial data loader)
        if 'log_return' in data.columns:
            existing_log_return = data['log_return'].dropna()
            if len(existing_log_return) > 0:
                # Use existing log_return if available and not empty
                return pd.Series(data['log_return'], index=data.index, name='log_return')

        # Fallback: calculate log returns from price data
        price_col = self._get_price_column(data)
        prices = data[price_col]
        log_returns = np.log(prices / prices.shift(1))
        return pd.Series(log_returns, index=data.index, name='log_return')
    
    def _generate_return_ratio(self, data: pd.DataFrame) -> pd.Series:
        """Generate return ratios from price data."""
        price_col = self._get_price_column(data)
        prices = data[price_col]
        return_ratios = prices / prices.shift(1)
        return pd.Series(return_ratios, index=data.index, name='return_ratio')
    
    def _generate_price_change(self, data: pd.DataFrame) -> pd.Series:
        """Generate price changes from price data."""
        price_col = self._get_price_column(data)
        prices = data[price_col]
        price_changes = prices.diff()
        return pd.Series(price_changes, index=data.index, name='price_change')
    
    def _generate_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Generate rolling volatility from log returns."""
        if 'log_return' not in data.columns:
            log_returns = self._generate_log_return(data)
        else:
            log_returns = data['log_return']
        
        volatility = log_returns.rolling(window=window).std()
        return pd.Series(volatility, index=data.index, name='volatility')
    
    def _get_price_column(self, data: pd.DataFrame) -> str:
        """
        Determine which price column to use.
        
        Args:
            data: Input data DataFrame
            
        Returns:
            Name of price column to use
        """
        # Try common price column names
        price_columns = ['close', 'Close', 'price', 'Price', 'adj_close', 'Adj Close']
        
        for col in price_columns:
            if col in data.columns:
                return col
        
        # If no standard price column found, use first numeric column
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]
        
        raise ValidationError("No suitable price column found in data")
    
    def get_observation_info(self) -> Dict[str, Any]:
        """
        Get information about generated observations.
        
        Returns:
            Dictionary with observation metadata
        """
        if self.last_observations is None:
            return {'status': 'No observations generated'}
        
        info = {
            'num_observations': len(self.last_observations),
            'observation_columns': list(self.last_observations.columns),
            'date_range': {
                'start': str(self.last_observations.index.min()),
                'end': str(self.last_observations.index.max())
            },
            'generators_used': [gen.__name__ for gen in self.generators],
            'missing_values': self.last_observations.isnull().sum().to_dict()
        }
        
        return info