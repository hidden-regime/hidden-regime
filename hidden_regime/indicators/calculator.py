"""
Technical Indicators Calculator

Unified interface for computing technical analysis indicators using the `ta` library.
Provides methods for calculating individual indicators, groups of indicators, or
comprehensive indicator suites with proper data alignment for HMM comparison.
"""

import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import ta

from ..data import DataLoader


class IndicatorCalculator:
    """
    Technical indicators calculator using the `ta` library.
    
    Provides a unified interface for computing traditional technical analysis
    indicators with proper data handling and HMM-compatible output formats.
    """
    
    def __init__(self, 
                 fill_method: str = 'ffill',
                 min_periods_pct: float = 0.7):
        """
        Initialize calculator with data handling preferences.
        
        Args:
            fill_method: Method for handling missing values ('ffill', 'bfill', 'drop')
            min_periods_pct: Minimum percentage of periods required for indicator validity
        """
        self.fill_method = fill_method
        self.min_periods_pct = min_periods_pct
        
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare price data for indicator calculation.
        
        Ensures required columns exist and handles missing data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            Cleaned DataFrame ready for indicator calculation
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create working copy
        clean_data = data.copy()
        
        # Handle missing values
        if self.fill_method == 'drop':
            clean_data = clean_data.dropna(subset=required_cols)
        elif self.fill_method in ['ffill', 'bfill']:
            clean_data[required_cols] = clean_data[required_cols].fillna(method=self.fill_method)
        
        # Validate minimum data requirements
        min_periods = int(len(clean_data) * self.min_periods_pct)
        valid_periods = clean_data[required_cols].notna().all(axis=1).sum()
        
        if valid_periods < min_periods:
            warnings.warn(
                f"Only {valid_periods} valid periods out of {len(clean_data)} "
                f"(minimum {min_periods} required for {self.min_periods_pct:.0%} coverage)"
            )
        
        return clean_data
    
    def calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum-based technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum indicators
        """
        clean_data = self.prepare_data(data)
        indicators = pd.DataFrame(index=clean_data.index)
        
        # RSI (Relative Strength Index)
        indicators['rsi'] = ta.momentum.RSIIndicator(clean_data['close']).rsi()
        indicators['rsi_14'] = ta.momentum.RSIIndicator(clean_data['close'], window=14).rsi()
        indicators['rsi_21'] = ta.momentum.RSIIndicator(clean_data['close'], window=21).rsi()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            clean_data['high'], clean_data['low'], clean_data['close']
        )
        indicators['stoch_k'] = stoch.stoch()
        indicators['stoch_d'] = stoch.stoch_signal()
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(clean_data['close'])
        indicators['macd'] = macd.macd()
        indicators['macd_signal'] = macd.macd_signal()
        indicators['macd_histogram'] = macd.macd_diff()
        
        # Williams %R
        indicators['williams_r'] = ta.momentum.WilliamsRIndicator(
            clean_data['high'], clean_data['low'], clean_data['close']
        ).williams_r()
        
        # Rate of Change (ROC)
        indicators['roc'] = ta.momentum.ROCIndicator(clean_data['close']).roc()
        indicators['roc_12'] = ta.momentum.ROCIndicator(clean_data['close'], window=12).roc()
        
        # Awesome Oscillator
        indicators['awesome_osc'] = ta.momentum.AwesomeOscillatorIndicator(
            clean_data['high'], clean_data['low']
        ).awesome_oscillator()
        
        # Commodity Channel Index (CCI)
        indicators['cci'] = ta.trend.CCIIndicator(
            clean_data['high'], clean_data['low'], clean_data['close']
        ).cci()
        
        return indicators
    
    def calculate_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend-based technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with trend indicators
        """
        clean_data = self.prepare_data(data)
        indicators = pd.DataFrame(index=clean_data.index)
        
        # Simple Moving Averages
        indicators['sma_10'] = ta.trend.SMAIndicator(clean_data['close'], window=10).sma_indicator()
        indicators['sma_20'] = ta.trend.SMAIndicator(clean_data['close'], window=20).sma_indicator()
        indicators['sma_50'] = ta.trend.SMAIndicator(clean_data['close'], window=50).sma_indicator()
        indicators['sma_200'] = ta.trend.SMAIndicator(clean_data['close'], window=200).sma_indicator()
        
        # Exponential Moving Averages
        indicators['ema_12'] = ta.trend.EMAIndicator(clean_data['close'], window=12).ema_indicator()
        indicators['ema_26'] = ta.trend.EMAIndicator(clean_data['close'], window=26).ema_indicator()
        indicators['ema_50'] = ta.trend.EMAIndicator(clean_data['close'], window=50).ema_indicator()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(clean_data['close'])
        indicators['bb_upper'] = bb.bollinger_hband()
        indicators['bb_middle'] = bb.bollinger_mavg()
        indicators['bb_lower'] = bb.bollinger_lband()
        indicators['bb_width'] = bb.bollinger_wband()
        indicators['bb_position'] = bb.bollinger_pband()
        
        # Average Directional Index (ADX)
        adx = ta.trend.ADXIndicator(clean_data['high'], clean_data['low'], clean_data['close'])
        indicators['adx'] = adx.adx()
        indicators['adx_pos'] = adx.adx_pos()
        indicators['adx_neg'] = adx.adx_neg()
        
        # Parabolic SAR
        indicators['psar'] = ta.trend.PSARIndicator(
            clean_data['high'], clean_data['low'], clean_data['close']
        ).psar()
        
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(
            clean_data['high'], clean_data['low']
        )
        indicators['ichimoku_a'] = ichimoku.ichimoku_a()
        indicators['ichimoku_b'] = ichimoku.ichimoku_b()
        indicators['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
        indicators['ichimoku_base'] = ichimoku.ichimoku_base_line()
        
        # Aroon
        aroon = ta.trend.AroonIndicator(clean_data['high'], clean_data['low'])
        indicators['aroon_up'] = aroon.aroon_up()
        indicators['aroon_down'] = aroon.aroon_down()
        indicators['aroon_indicator'] = aroon.aroon_indicator()
        
        return indicators
        
    def calculate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility-based technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility indicators
        """
        clean_data = self.prepare_data(data)
        indicators = pd.DataFrame(index=clean_data.index)
        
        # Average True Range (ATR)
        indicators['atr'] = ta.volatility.AverageTrueRange(
            clean_data['high'], clean_data['low'], clean_data['close']
        ).average_true_range()
        
        # Bollinger Bands (additional volatility metrics)
        bb = ta.volatility.BollingerBands(clean_data['close'])
        indicators['bb_volatility'] = bb.bollinger_wband()
        
        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(
            clean_data['high'], clean_data['low'], clean_data['close']
        )
        indicators['kc_upper'] = kc.keltner_channel_hband()
        indicators['kc_lower'] = kc.keltner_channel_lband()
        indicators['kc_middle'] = kc.keltner_channel_mband()
        
        # Donchian Channel
        dc = ta.volatility.DonchianChannel(
            clean_data['high'], clean_data['low'], clean_data['close']
        )
        indicators['dc_upper'] = dc.donchian_channel_hband()
        indicators['dc_lower'] = dc.donchian_channel_lband()
        indicators['dc_middle'] = dc.donchian_channel_mband()
        
        # Ulcer Index
        indicators['ulcer_index'] = ta.volatility.UlcerIndex(clean_data['close']).ulcer_index()
        
        return indicators
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume indicators
        """
        clean_data = self.prepare_data(data)
        indicators = pd.DataFrame(index=clean_data.index)
        
        # On-Balance Volume (OBV)
        indicators['obv'] = ta.volume.OnBalanceVolumeIndicator(
            clean_data['close'], clean_data['volume']
        ).on_balance_volume()
        
        # Accumulation/Distribution Line
        indicators['ad_line'] = ta.volume.AccDistIndexIndicator(
            clean_data['high'], clean_data['low'], clean_data['close'], clean_data['volume']
        ).acc_dist_index()
        
        # Chaikin Money Flow
        indicators['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            clean_data['high'], clean_data['low'], clean_data['close'], clean_data['volume']
        ).chaikin_money_flow()
        
        # Force Index
        indicators['force_index'] = ta.volume.ForceIndexIndicator(
            clean_data['close'], clean_data['volume']
        ).force_index()
        
        # Money Flow Index (MFI)
        indicators['mfi'] = ta.volume.MFIIndicator(
            clean_data['high'], clean_data['low'], clean_data['close'], clean_data['volume']
        ).money_flow_index()
        
        # Volume Weighted Average Price (VWAP)
        indicators['vwap'] = ta.volume.VolumeSMAIndicator(
            clean_data['close'], clean_data['volume']
        ).volume_sma()
        
        # Ease of Movement
        indicators['eom'] = ta.volume.EaseOfMovementIndicator(
            clean_data['high'], clean_data['low'], clean_data['volume']
        ).ease_of_movement()
        
        # Volume Price Trend
        indicators['vpt'] = ta.volume.VolumePriceTrendIndicator(
            clean_data['close'], clean_data['volume']
        ).volume_price_trend()
        
        return indicators
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive set of technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators combined
        """
        momentum = self.calculate_momentum_indicators(data)
        trend = self.calculate_trend_indicators(data)
        volatility = self.calculate_volatility_indicators(data)
        volume = self.calculate_volume_indicators(data)
        
        # Combine all indicators
        all_indicators = pd.concat([momentum, trend, volatility, volume], axis=1)
        
        # Add basic price-derived indicators
        all_indicators['log_return'] = np.log(data['close'] / data['close'].shift(1))
        all_indicators['price_change'] = data['close'].pct_change()
        all_indicators['high_low_ratio'] = data['high'] / data['low']
        all_indicators['volume_change'] = data['volume'].pct_change()
        
        return all_indicators


# Convenience functions for quick calculations
def calculate_momentum_indicators(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Calculate momentum indicators using default settings."""
    calculator = IndicatorCalculator(**kwargs)
    return calculator.calculate_momentum_indicators(data)


def calculate_trend_indicators(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Calculate trend indicators using default settings."""
    calculator = IndicatorCalculator(**kwargs)
    return calculator.calculate_trend_indicators(data)


def calculate_volatility_indicators(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Calculate volatility indicators using default settings."""
    calculator = IndicatorCalculator(**kwargs)
    return calculator.calculate_volatility_indicators(data)


def calculate_volume_indicators(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Calculate volume indicators using default settings."""
    calculator = IndicatorCalculator(**kwargs)
    return calculator.calculate_volume_indicators(data)


def calculate_all_indicators(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Calculate all technical indicators using default settings."""
    calculator = IndicatorCalculator(**kwargs)
    return calculator.calculate_all_indicators(data)