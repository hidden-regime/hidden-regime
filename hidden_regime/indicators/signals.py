"""
Technical Indicator Signal Generation

Converts technical indicator values into standardized trading signals
compatible with HMM regime analysis. Provides signal combination and
composite signal generation for multi-indicator strategies.
"""

import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .calculator import IndicatorCalculator


class IndicatorSignalGenerator:
    """
    Generate standardized trading signals from technical indicators.
    
    Converts continuous indicator values into discrete signals (-1, 0, 1)
    representing bearish, neutral, and bullish conditions respectively.
    """
    
    def __init__(self, 
                 strong_signal_threshold: float = 0.7,
                 weak_signal_threshold: float = 0.3):
        """
        Initialize signal generator with threshold settings.
        
        Args:
            strong_signal_threshold: Threshold for strong buy/sell signals
            weak_signal_threshold: Threshold for weak buy/sell signals
        """
        self.strong_threshold = strong_signal_threshold
        self.weak_threshold = weak_signal_threshold
        
    def generate_rsi_signals(self, rsi_values: pd.Series, 
                           overbought: float = 70,
                           oversold: float = 30) -> pd.Series:
        """
        Generate RSI-based trading signals.
        
        Args:
            rsi_values: RSI indicator values
            overbought: RSI level considered overbought (sell signal)
            oversold: RSI level considered oversold (buy signal)
            
        Returns:
            Series with trading signals (-1, 0, 1)
        """
        signals = pd.Series(0, index=rsi_values.index, dtype=int)
        signals[rsi_values >= overbought] = -1  # Sell signal
        signals[rsi_values <= oversold] = 1     # Buy signal
        return signals
    
    def generate_macd_signals(self, macd_line: pd.Series, 
                            macd_signal: pd.Series,
                            macd_histogram: pd.Series) -> pd.Series:
        """
        Generate MACD-based trading signals.
        
        Args:
            macd_line: MACD line values
            macd_signal: MACD signal line values  
            macd_histogram: MACD histogram values
            
        Returns:
            Series with trading signals (-1, 0, 1)
        """
        signals = pd.Series(0, index=macd_line.index, dtype=int)
        
        # MACD line crosses above signal line (bullish)
        macd_cross_up = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
        # MACD line crosses below signal line (bearish)
        macd_cross_down = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))
        
        # Additional confirmation from histogram
        histogram_positive = macd_histogram > 0
        histogram_negative = macd_histogram < 0
        
        signals[macd_cross_up & histogram_positive] = 1   # Strong buy
        signals[macd_cross_down & histogram_negative] = -1 # Strong sell
        
        return signals
    
    def generate_bollinger_signals(self, price: pd.Series,
                                 bb_upper: pd.Series,
                                 bb_lower: pd.Series,
                                 bb_position: pd.Series) -> pd.Series:
        """
        Generate Bollinger Bands trading signals.
        
        Args:
            price: Price series (typically close price)
            bb_upper: Upper Bollinger Band
            bb_lower: Lower Bollinger Band
            bb_position: Bollinger Band position (0-1)
            
        Returns:
            Series with trading signals (-1, 0, 1)
        """
        signals = pd.Series(0, index=price.index, dtype=int)
        
        # Price touches or crosses bands
        touches_upper = price >= bb_upper
        touches_lower = price <= bb_lower
        
        # Reversal signals when price moves back from bands
        moves_from_upper = touches_upper.shift(1) & (price < bb_upper)
        moves_from_lower = touches_lower.shift(1) & (price > bb_lower)
        
        signals[moves_from_upper] = -1  # Sell after touching upper band
        signals[moves_from_lower] = 1   # Buy after touching lower band
        
        return signals
    
    def generate_moving_average_signals(self, price: pd.Series,
                                      short_ma: pd.Series,
                                      long_ma: pd.Series) -> pd.Series:
        """
        Generate moving average crossover signals.
        
        Args:
            price: Price series
            short_ma: Short-term moving average
            long_ma: Long-term moving average
            
        Returns:
            Series with trading signals (-1, 0, 1)
        """
        signals = pd.Series(0, index=price.index, dtype=int)
        
        # Golden cross (short MA crosses above long MA)
        golden_cross = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        # Death cross (short MA crosses below long MA)
        death_cross = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        
        # Additional confirmation from price position
        price_above_short = price > short_ma
        price_below_short = price < short_ma
        
        signals[golden_cross & price_above_short] = 1   # Buy signal
        signals[death_cross & price_below_short] = -1   # Sell signal
        
        return signals
    
    def generate_stochastic_signals(self, stoch_k: pd.Series,
                                  stoch_d: pd.Series,
                                  overbought: float = 80,
                                  oversold: float = 20) -> pd.Series:
        """
        Generate Stochastic oscillator signals.
        
        Args:
            stoch_k: %K line values
            stoch_d: %D line values
            overbought: Overbought threshold
            oversold: Oversold threshold
            
        Returns:
            Series with trading signals (-1, 0, 1)
        """
        signals = pd.Series(0, index=stoch_k.index, dtype=int)
        
        # %K crosses above %D in oversold region (bullish)
        bullish_cross = ((stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1)) & 
                        (stoch_k < oversold))
        
        # %K crosses below %D in overbought region (bearish)
        bearish_cross = ((stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1)) & 
                        (stoch_k > overbought))
        
        signals[bullish_cross] = 1   # Buy signal
        signals[bearish_cross] = -1  # Sell signal
        
        return signals
    
    def generate_volume_signals(self, price: pd.Series,
                              volume: pd.Series,
                              obv: pd.Series) -> pd.Series:
        """
        Generate volume-based trading signals.
        
        Args:
            price: Price series
            volume: Volume series
            obv: On-Balance Volume
            
        Returns:
            Series with trading signals (-1, 0, 1)
        """
        signals = pd.Series(0, index=price.index, dtype=int)
        
        # Price and OBV moving in same direction (confirmation)
        price_change = price.pct_change()
        obv_change = obv.pct_change()
        
        # Volume above average confirms moves
        volume_avg = volume.rolling(window=20).mean()
        high_volume = volume > volume_avg * 1.2
        
        # Bullish: Price up, OBV up, high volume
        bullish_volume = (price_change > 0) & (obv_change > 0) & high_volume
        # Bearish: Price down, OBV down, high volume
        bearish_volume = (price_change < 0) & (obv_change < 0) & high_volume
        
        signals[bullish_volume] = 1   # Buy signal
        signals[bearish_volume] = -1  # Sell signal
        
        return signals
    
    def generate_momentum_signals(self, roc: pd.Series,
                                williams_r: pd.Series) -> pd.Series:
        """
        Generate momentum-based signals combining ROC and Williams %R.
        
        Args:
            roc: Rate of Change values
            williams_r: Williams %R values
            
        Returns:
            Series with trading signals (-1, 0, 1)
        """
        signals = pd.Series(0, index=roc.index, dtype=int)
        
        # Strong positive momentum
        strong_momentum = (roc > 5) & (williams_r > -20)  # Williams %R near high
        # Strong negative momentum
        weak_momentum = (roc < -5) & (williams_r < -80)   # Williams %R near low
        
        signals[strong_momentum] = 1   # Buy signal
        signals[weak_momentum] = -1   # Sell signal
        
        return signals


def combine_indicator_signals(signal_dict: Dict[str, pd.Series],
                            weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    Combine multiple indicator signals into composite signal.
    
    Args:
        signal_dict: Dictionary of indicator names and their signal series
        weights: Optional weights for each indicator (defaults to equal weights)
        
    Returns:
        Combined signal series with values between -1 and 1
    """
    if not signal_dict:
        raise ValueError("signal_dict cannot be empty")
    
    # Get common index from all signals
    common_index = signal_dict[list(signal_dict.keys())[0]].index
    for signals in signal_dict.values():
        common_index = common_index.intersection(signals.index)
    
    if len(common_index) == 0:
        raise ValueError("No common index found across signals")
    
    # Align all signals to common index
    aligned_signals = {}
    for name, signals in signal_dict.items():
        aligned_signals[name] = signals.reindex(common_index).fillna(0)
    
    # Set equal weights if not provided
    if weights is None:
        weights = {name: 1.0 / len(signal_dict) for name in signal_dict.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    normalized_weights = {name: w / total_weight for name, w in weights.items()}
    
    # Calculate weighted combination
    combined = pd.Series(0.0, index=common_index)
    for name, signals in aligned_signals.items():
        combined += signals * normalized_weights.get(name, 0.0)
    
    # Ensure output is in valid range [-1, 1]
    combined = combined.clip(-1, 1)
    
    return combined


def generate_composite_signal(data: pd.DataFrame,
                            indicators: pd.DataFrame,
                            signal_config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Generate comprehensive composite signal from all available indicators.
    
    Args:
        data: Price/volume data DataFrame
        indicators: Technical indicators DataFrame
        signal_config: Configuration for signal generation parameters
        
    Returns:
        DataFrame with individual and composite signals
    """
    if signal_config is None:
        signal_config = {
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'bb_sensitivity': 0.9,
            'ma_periods': [10, 50],
            'stoch_overbought': 80,
            'stoch_oversold': 20
        }
    
    generator = IndicatorSignalGenerator()
    signals = pd.DataFrame(index=indicators.index)
    
    # Generate individual signals
    if 'rsi' in indicators.columns:
        signals['rsi_signal'] = generator.generate_rsi_signals(
            indicators['rsi'],
            overbought=signal_config.get('rsi_overbought', 70),
            oversold=signal_config.get('rsi_oversold', 30)
        )
    
    if all(col in indicators.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
        signals['macd_signal'] = generator.generate_macd_signals(
            indicators['macd'],
            indicators['macd_signal'],
            indicators['macd_histogram']
        )
    
    if all(col in indicators.columns for col in ['bb_upper', 'bb_lower', 'bb_position']):
        signals['bb_signal'] = generator.generate_bollinger_signals(
            data['close'],
            indicators['bb_upper'],
            indicators['bb_lower'],
            indicators['bb_position']
        )
    
    if all(col in indicators.columns for col in ['sma_10', 'sma_50']):
        signals['ma_signal'] = generator.generate_moving_average_signals(
            data['close'],
            indicators['sma_10'],
            indicators['sma_50']
        )
    
    if all(col in indicators.columns for col in ['stoch_k', 'stoch_d']):
        signals['stoch_signal'] = generator.generate_stochastic_signals(
            indicators['stoch_k'],
            indicators['stoch_d'],
            overbought=signal_config.get('stoch_overbought', 80),
            oversold=signal_config.get('stoch_oversold', 20)
        )
    
    if all(col in indicators.columns for col in ['obv']) and 'volume' in data.columns:
        signals['volume_signal'] = generator.generate_volume_signals(
            data['close'],
            data['volume'],
            indicators['obv']
        )
    
    if all(col in indicators.columns for col in ['roc', 'williams_r']):
        signals['momentum_signal'] = generator.generate_momentum_signals(
            indicators['roc'],
            indicators['williams_r']
        )
    
    # Generate composite signal
    individual_signals = {col: signals[col] for col in signals.columns if col.endswith('_signal')}
    
    if individual_signals:
        signals['composite_signal'] = combine_indicator_signals(individual_signals)
        
        # Add signal strength
        signals['signal_strength'] = abs(signals['composite_signal'])
        
        # Add signal consensus (percentage of indicators agreeing)
        non_zero_signals = pd.DataFrame({col: (signals[col] != 0).astype(int) 
                                       for col in individual_signals.keys()})
        signals['signal_consensus'] = non_zero_signals.sum(axis=1) / len(individual_signals)
    
    return signals