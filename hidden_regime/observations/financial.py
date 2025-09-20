"""
Financial observation generation.

Provides specialized observation generators for financial time series data
including technical indicators, price transformations, and volume analysis.
"""

from typing import Any, Dict, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .base import BaseObservationGenerator
from ..config.observation import FinancialObservationConfig
from ..utils.exceptions import ValidationError


class FinancialObservationGenerator(BaseObservationGenerator):
    """
    Financial observation generator for creating trading-relevant features.
    
    Generates observations specifically designed for financial time series analysis
    including returns, technical indicators, volatility measures, and volume features.
    """
    
    def __init__(self, config: FinancialObservationConfig):
        """
        Initialize financial observation generator.
        
        Args:
            config: Financial observation configuration
        """
        super().__init__(config)
        self.config = config  # Type hint for IDE
    
    def _get_builtin_generator(self, name: str) -> Callable:
        """
        Get built-in financial generator function by name.
        
        Args:
            name: Name of built-in financial generator
            
        Returns:
            Generator function or None if not found
        """
        financial_generators = {
            # Core financial transformations
            "log_return": self._generate_log_return,
            "return_ratio": self._generate_return_ratio,
            "average_price": self._generate_average_price,
            "price_change": self._generate_price_change,

            # Volatility measures
            "volatility": self._generate_volatility,

            # Enhanced regime-relevant features
            "momentum_strength": self._generate_momentum_strength,
            "trend_persistence": self._generate_trend_persistence,
            "volatility_context": self._generate_volatility_context,
            "directional_consistency": self._generate_directional_consistency,

            # Technical indicators
            "rsi": self._generate_rsi,
            "macd": self._generate_macd,
            "bollinger_bands": self._generate_bollinger_bands,
            "moving_average": self._generate_moving_average,

            # Volume indicators (if available)
            "volume_sma": self._generate_volume_sma,
            "volume_ratio": self._generate_volume_ratio,
            "price_volume_trend": self._generate_price_volume_trend,
        }
        
        # Check financial generators first
        if name in financial_generators:
            return financial_generators[name]

        # Fallback to base generators
        return super()._get_builtin_generator(name)
    
    def update(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate financial observations from OHLCV data.
        
        Args:
            data: Financial data with OHLCV columns
            
        Returns:
            DataFrame with financial observations
        """
        # Validate financial data structure
        self._validate_financial_data(data)
        
        # Generate observations using parent method
        observations = super().update(data)
        
        # Apply normalization if requested
        if self.config.normalize_features:
            observations = self._normalize_observations(observations)
        
        return observations
    
    def _validate_financial_data(self, data: pd.DataFrame) -> None:
        """
        Validate that data contains required financial columns.
        
        Args:
            data: Input financial data
        """
        required_price_col = self.config.price_column
        
        if required_price_col not in data.columns:
            raise ValidationError(f"Required price column '{required_price_col}' not found in data")
        
        if self.config.include_volume_features:
            volume_col = self.config.volume_column
            if volume_col not in data.columns:
                raise ValidationError(f"Required volume column '{volume_col}' not found in data")
    
    def _normalize_observations(self, observations: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize observation features.
        
        Args:
            observations: DataFrame with observations
            
        Returns:
            DataFrame with normalized observations
        """
        # Columns to skip normalization (already in good ranges)
        skip_cols = ['open', 'high', 'low', 'close', 'volume', 'log_return']
        
        normalized = observations.copy()
        
        for col in observations.columns:
            if col in skip_cols:
                continue
            
            data = observations[col].dropna()
            if len(data) > 1:
                # Use rolling z-score normalization with 252-day window
                window = min(252, len(data))
                rolling_mean = data.rolling(window=window).mean()
                rolling_std = data.rolling(window=window).std()
                
                # Avoid division by zero
                rolling_std = rolling_std.fillna(1.0)
                rolling_std = rolling_std.replace(0.0, 1.0)
                
                normalized[col] = (data - rolling_mean) / rolling_std
        
        return normalized
    
    # Financial generator functions
    def _generate_average_price(self, data: pd.DataFrame) -> pd.Series:
        """Generate average price (OHLC/4)."""
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            avg_price = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        else:
            # Fallback to close price
            avg_price = data[self.config.price_column]
        
        return pd.Series(avg_price, index=data.index, name='average_price')
    
    def _generate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Generate Relative Strength Index."""
        price_col = self.config.price_column
        prices = data[price_col]
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return pd.Series(rsi, index=data.index, name='rsi')
    
    def _generate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Generate MACD indicator."""
        price_col = self.config.price_column
        prices = data[price_col]
        
        # Calculate exponential moving averages
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = macd_line.ewm(span=signal).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd_line': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }, index=data.index)
    
    def _generate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Generate Bollinger Bands."""
        price_col = self.config.price_column
        prices = data[price_col]
        
        # Calculate moving average and standard deviation
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        # Calculate bands
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Calculate position within bands
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        
        return pd.DataFrame({
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band,
            'bb_position': bb_position
        }, index=data.index)
    
    def _generate_moving_average(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Generate simple moving average."""
        price_col = self.config.price_column
        prices = data[price_col]
        
        sma = prices.rolling(window=period).mean()
        return pd.Series(sma, index=data.index, name=f'sma_{period}')
    
    def _generate_volume_sma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Generate volume simple moving average."""
        if not self.config.include_volume_features:
            return pd.Series(dtype=float, index=data.index, name='volume_sma')
        
        volume_col = self.config.volume_column
        if volume_col not in data.columns:
            return pd.Series(dtype=float, index=data.index, name='volume_sma')
        
        volume = data[volume_col]
        volume_sma = volume.rolling(window=period).mean()
        return pd.Series(volume_sma, index=data.index, name='volume_sma')
    
    def _generate_volume_ratio(self, data: pd.DataFrame) -> pd.Series:
        """Generate volume ratio (current/average)."""
        if not self.config.include_volume_features:
            return pd.Series(dtype=float, index=data.index, name='volume_ratio')
        
        volume_col = self.config.volume_column
        if volume_col not in data.columns:
            return pd.Series(dtype=float, index=data.index, name='volume_ratio')
        
        volume = data[volume_col]
        volume_sma = volume.rolling(window=20).mean()
        volume_ratio = volume / volume_sma
        return pd.Series(volume_ratio, index=data.index, name='volume_ratio')
    
    def _generate_price_volume_trend(self, data: pd.DataFrame) -> pd.Series:
        """Generate Price Volume Trend indicator."""
        if not self.config.include_volume_features:
            return pd.Series(dtype=float, index=data.index, name='pvt')
        
        volume_col = self.config.volume_column
        price_col = self.config.price_column
        
        if volume_col not in data.columns:
            return pd.Series(dtype=float, index=data.index, name='pvt')
        
        volume = data[volume_col]
        prices = data[price_col]
        
        # Calculate price change percentage
        price_change_pct = prices.pct_change()
        
        # Calculate PVT
        pvt = (price_change_pct * volume).cumsum()
        return pd.Series(pvt, index=data.index, name='pvt')
    
    def plot(self, **kwargs) -> plt.Figure:
        """
        Generate financial-specific visualization of observations.
        
        Returns:
            matplotlib Figure with financial observation plots
        """
        if self.last_observations is None:
            return super().plot(**kwargs)
        
        # Create subplots for different types of observations
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Price and moving averages
        ax1 = plt.subplot(4, 1, 1)
        price_col = self.config.price_column
        if price_col in self.last_observations.columns:
            ax1.plot(self.last_observations.index, self.last_observations[price_col], 
                    label='Price', linewidth=1.5)
        
        # Add moving averages if available
        ma_cols = [col for col in self.last_observations.columns if 'sma' in col.lower()]
        for ma_col in ma_cols:
            ax1.plot(self.last_observations.index, self.last_observations[ma_col], 
                    label=ma_col, alpha=0.7)
        
        ax1.set_title('Price and Moving Averages')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Returns
        ax2 = plt.subplot(4, 1, 2)
        if 'log_return' in self.last_observations.columns:
            returns = self.last_observations['log_return'].dropna()
            ax2.plot(returns.index, returns, label='Log Returns', alpha=0.8)
        
        ax2.set_title('Returns')
        ax2.set_ylabel('Log Return')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Technical indicators
        ax3 = plt.subplot(4, 1, 3)
        if 'rsi' in self.last_observations.columns:
            rsi = self.last_observations['rsi'].dropna()
            ax3.plot(rsi.index, rsi, label='RSI', color='purple')
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
            ax3.set_ylim(0, 100)
        
        ax3.set_title('Technical Indicators')
        ax3.set_ylabel('RSI')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Volume (if available)
        ax4 = plt.subplot(4, 1, 4)
        volume_col = self.config.volume_column
        if (self.config.include_volume_features and 
            volume_col in self.last_observations.columns):
            
            volume = self.last_observations[volume_col]
            ax4.bar(volume.index, volume, alpha=0.6, label='Volume')
            
            if 'volume_sma' in self.last_observations.columns:
                volume_sma = self.last_observations['volume_sma']
                ax4.plot(volume_sma.index, volume_sma, color='red', 
                        label='Volume SMA', linewidth=2)
        else:
            # Plot volatility instead
            if 'volatility' in self.last_observations.columns:
                volatility = self.last_observations['volatility'].dropna()
                ax4.plot(volatility.index, volatility, label='Volatility', color='orange')
        
        ax4.set_title('Volume / Volatility')
        ax4.set_xlabel('Date')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig