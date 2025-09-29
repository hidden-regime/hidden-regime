"""
Comprehensive technical indicator analysis and signal generation.

Provides systematic calculation of technical indicators and generation of
buy/sell signals for comparison against HMM regime detection strategies.
"""

import numpy as np
import pandas as pd
import ta
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .performance import RegimePerformanceAnalyzer
from ..utils.exceptions import AnalysisError


@dataclass
class SignalEvent:
    """Detailed information about a technical indicator signal event."""
    timestamp: str
    indicator_name: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    signal_strength: float  # Confidence/strength of signal (0-1)

    # Signal rationale and mathematical explanation
    triggering_condition: str  # What condition triggered the signal
    mathematical_explanation: str  # Detailed mathematical rationale
    market_context: Dict[str, float]  # Relevant market values at signal time

    # Signal characteristics
    signal_confidence: float  # How confident we are in this signal
    expected_duration: Optional[int] = None  # Expected signal duration in periods
    risk_assessment: str = "MEDIUM"  # LOW, MEDIUM, HIGH

    # Performance tracking
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    realized_return: Optional[float] = None


@dataclass
class IndicatorSnapshot:
    """Snapshot of technical indicator state and signal generation."""
    timestamp: str
    indicator_name: str
    indicator_value: float
    signal_generated: int  # -1, 0, 1

    # Mathematical state
    calculation_inputs: Dict[str, float]  # Inputs used in calculation
    intermediate_values: Dict[str, float]  # Intermediate calculation values
    threshold_levels: Dict[str, float]  # Relevant thresholds

    # Signal generation context
    signal_event: Optional[SignalEvent] = None
    previous_signal: Optional[int] = None
    days_since_last_signal: Optional[int] = None


class TechnicalIndicatorAnalyzer:
    """
    Comprehensive technical indicator analyzer with signal generation.

    Calculates 15+ technical indicators and generates systematic buy/sell signals
    for performance comparison against HMM regime detection strategies.
    """

    def __init__(self):
        """Initialize technical indicator analyzer."""
        self.performance_analyzer = RegimePerformanceAnalyzer()
        self.signal_history = []  # Store detailed signal generation history

        # Define comprehensive indicator set
        self.indicator_definitions = {
            # Trend indicators
            'sma_20': {'type': 'trend', 'params': {'window': 20}},
            'sma_50': {'type': 'trend', 'params': {'window': 50}},
            'ema_12': {'type': 'trend', 'params': {'window': 12}},
            'ema_26': {'type': 'trend', 'params': {'window': 26}},
            'macd': {'type': 'momentum', 'params': {}},
            'macd_signal': {'type': 'momentum', 'params': {}},

            # Momentum indicators
            'rsi': {'type': 'momentum', 'params': {'window': 14}},
            'stoch': {'type': 'momentum', 'params': {'window': 14, 'smooth_window': 3}},
            'williams_r': {'type': 'momentum', 'params': {'lbp': 14}},
            'cci': {'type': 'momentum', 'params': {'window': 20}},
            'roc': {'type': 'momentum', 'params': {'window': 12}},

            # Volatility indicators
            'bollinger_upper': {'type': 'volatility', 'params': {'window': 20, 'window_dev': 2}},
            'bollinger_lower': {'type': 'volatility', 'params': {'window': 20, 'window_dev': 2}},
            'atr': {'type': 'volatility', 'params': {'window': 14}},
            'keltner_upper': {'type': 'volatility', 'params': {'window': 20}},
            'keltner_lower': {'type': 'volatility', 'params': {'window': 20}},

            # Volume indicators
            'volume_sma': {'type': 'volume', 'params': {'window': 20}},
            'volume_ema': {'type': 'volume', 'params': {'window': 20}},
            'vwap': {'type': 'volume', 'params': {}},

            # Others
            'adx': {'type': 'trend', 'params': {'window': 14}},
            'aroon_up': {'type': 'trend', 'params': {'window': 25}},
            'aroon_down': {'type': 'trend', 'params': {'window': 25}},
        }

    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for given price data.

        Args:
            data: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            DataFrame with all calculated indicators
        """
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            raise AnalysisError(f"Data must contain columns: {required_cols}")

        # Initialize results DataFrame
        indicators = pd.DataFrame(index=data.index)

        # Calculate trend indicators
        indicators['sma_20'] = ta.trend.sma_indicator(data['close'], window=20)
        indicators['sma_50'] = ta.trend.sma_indicator(data['close'], window=50)
        indicators['ema_12'] = ta.trend.ema_indicator(data['close'], window=12)
        indicators['ema_26'] = ta.trend.ema_indicator(data['close'], window=26)

        # MACD
        macd_line = ta.trend.macd(data['close'])
        macd_signal = ta.trend.macd_signal(data['close'])
        indicators['macd'] = macd_line
        indicators['macd_signal'] = macd_signal
        indicators['macd_histogram'] = macd_line - macd_signal

        # ADX (trend strength)
        indicators['adx'] = ta.trend.adx(data['high'], data['low'], data['close'], window=14)

        # Aroon
        indicators['aroon_up'] = ta.trend.aroon_up(data['high'], data['low'], window=25)
        indicators['aroon_down'] = ta.trend.aroon_down(data['high'], data['low'], window=25)

        # Momentum indicators
        indicators['rsi'] = ta.momentum.rsi(data['close'], window=14)
        indicators['stoch'] = ta.momentum.stoch(data['high'], data['low'], data['close'], window=14, smooth_window=3)
        indicators['williams_r'] = ta.momentum.williams_r(data['high'], data['low'], data['close'], lbp=14)
        indicators['roc'] = ta.momentum.roc(data['close'], window=12)

        # CCI is in trend module
        indicators['cci'] = ta.trend.cci(data['high'], data['low'], data['close'], window=20)

        # Volatility indicators
        bb_upper = ta.volatility.bollinger_hband(data['close'], window=20, window_dev=2)
        bb_lower = ta.volatility.bollinger_lband(data['close'], window=20, window_dev=2)
        indicators['bollinger_upper'] = bb_upper
        indicators['bollinger_lower'] = bb_lower
        indicators['bollinger_width'] = (bb_upper - bb_lower) / data['close']

        indicators['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)

        # Keltner Channels
        indicators['keltner_upper'] = ta.volatility.keltner_channel_hband(data['high'], data['low'], data['close'], window=20)
        indicators['keltner_lower'] = ta.volatility.keltner_channel_lband(data['high'], data['low'], data['close'], window=20)

        # Volume indicators (if volume data available)
        if 'volume' in data.columns and not data['volume'].isna().all():
            try:
                indicators['volume_sma'] = ta.volume.volume_sma(data['close'], data['volume'], window=20)
            except:
                # Fallback to simple volume moving average
                indicators['volume_sma'] = data['volume'].rolling(window=20).mean()

            try:
                indicators['volume_ema'] = ta.volume.volume_ema(data['close'], data['volume'], window=20)
            except:
                # Fallback to simple volume EMA
                indicators['volume_ema'] = data['volume'].ewm(span=20).mean()

            try:
                indicators['vwap'] = ta.volume.volume_weighted_average_price(
                    data['high'], data['low'], data['close'], data['volume']
                )
            except:
                # Fallback to simple VWAP calculation
                typical_price = (data['high'] + data['low'] + data['close']) / 3
                indicators['vwap'] = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()

        return indicators

    def get_signal_history(self) -> List[SignalEvent]:
        """Return complete history of signal events with rationale."""
        return self.signal_history.copy()

    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary statistics of signal generation."""
        if not self.signal_history:
            return {'total_signals': 0, 'by_indicator': {}, 'by_type': {}}

        by_indicator = {}
        by_type = {'BUY': 0, 'SELL': 0, 'HOLD': 0}

        for signal in self.signal_history:
            # Count by indicator
            if signal.indicator_name not in by_indicator:
                by_indicator[signal.indicator_name] = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            by_indicator[signal.indicator_name][signal.signal_type] += 1

            # Count by type
            by_type[signal.signal_type] += 1

        return {
            'total_signals': len(self.signal_history),
            'by_indicator': by_indicator,
            'by_type': by_type,
            'avg_signal_strength': np.mean([s.signal_strength for s in self.signal_history]),
            'avg_confidence': np.mean([s.signal_confidence for s in self.signal_history])
        }

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Generate buy/sell signals for each indicator.

        Args:
            data: Original OHLCV data
            indicators: Calculated technical indicators

        Returns:
            Dictionary mapping indicator names to signal series (-1, 0, 1)
        """
        signals = {}
        price = data['close']

        # Clear previous signal history for this analysis
        self.signal_history = []

        # Moving average signals
        signals['sma_20'] = self._generate_ma_crossover_signals_with_rationale(price, indicators['sma_20'], 'SMA_20')
        signals['sma_50'] = self._generate_ma_crossover_signals_with_rationale(price, indicators['sma_50'], 'SMA_50')
        signals['ema_12'] = self._generate_ma_crossover_signals_with_rationale(price, indicators['ema_12'], 'EMA_12')
        signals['ema_26'] = self._generate_ma_crossover_signals_with_rationale(price, indicators['ema_26'], 'EMA_26')

        # MACD signals
        signals['macd'] = self._generate_macd_signals_with_rationale(indicators['macd'], indicators['macd_signal'])

        # RSI signals
        signals['rsi'] = self._generate_rsi_signals_with_rationale(indicators['rsi'])

        # Stochastic signals
        signals['stoch'] = self._generate_stochastic_signals_with_rationale(indicators['stoch'])

        # Williams %R signals
        signals['williams_r'] = self._generate_williams_r_signals_with_rationale(indicators['williams_r'])

        # CCI signals
        signals['cci'] = self._generate_cci_signals_with_rationale(indicators['cci'])

        # Bollinger Band signals
        signals['bollinger'] = self._generate_bollinger_signals_with_rationale(price, indicators['bollinger_upper'], indicators['bollinger_lower'])

        # ADX trend strength signals
        signals['adx'] = self._generate_adx_signals_with_rationale(indicators['adx'], indicators['sma_20'], price)

        # Aroon signals
        signals['aroon'] = self._generate_aroon_signals_with_rationale(indicators['aroon_up'], indicators['aroon_down'])

        # Rate of Change signals
        signals['roc'] = self._generate_roc_signals_with_rationale(indicators['roc'])

        # Volume-based signals (if available)
        if 'vwap' in indicators.columns and not indicators['vwap'].isna().all():
            signals['vwap'] = self._generate_vwap_signals_with_rationale(price, indicators['vwap'])

        return signals

    def _generate_ma_crossover_signals_with_rationale(self, price: pd.Series, ma: pd.Series, indicator_name: str) -> pd.Series:
        """Generate MA crossover signals with detailed rationale."""
        signals = pd.Series(0, index=price.index)

        # Buy when price crosses above MA, sell when below
        price_above_ma = (price > ma).fillna(False)
        price_above_ma_prev = price_above_ma.shift(1).fillna(False)

        # Generate signals with rationale tracking
        for i in range(1, len(price)):
            current_price = price.iloc[i]
            current_ma = ma.iloc[i]
            prev_price = price.iloc[i-1]
            prev_ma = ma.iloc[i-1]

            signal = 0

            # Buy signal: price crosses above MA
            if current_price > current_ma and prev_price <= prev_ma:
                signal = 1
                signal_strength = min(1.0, (current_price - current_ma) / current_ma * 10)  # Normalize

                self.signal_history.append(SignalEvent(
                    timestamp=str(price.index[i]),
                    indicator_name=indicator_name,
                    signal_type='BUY',
                    signal_strength=signal_strength,
                    triggering_condition=f'Price ({current_price:.4f}) crossed above {indicator_name} ({current_ma:.4f})',
                    mathematical_explanation=f'Bullish crossover detected: Price/MA ratio = {current_price/current_ma:.4f} > 1.0, Previous ratio = {prev_price/prev_ma:.4f}',
                    market_context={
                        'price': current_price,
                        'ma_value': current_ma,
                        'price_ma_ratio': current_price / current_ma,
                        'crossover_magnitude': (current_price - current_ma) / current_ma
                    },
                    signal_confidence=signal_strength,
                    risk_assessment='MEDIUM',
                    entry_price=current_price
                ))

            # Sell signal: price crosses below MA
            elif current_price < current_ma and prev_price >= prev_ma:
                signal = -1
                signal_strength = min(1.0, (current_ma - current_price) / current_ma * 10)  # Normalize

                self.signal_history.append(SignalEvent(
                    timestamp=str(price.index[i]),
                    indicator_name=indicator_name,
                    signal_type='SELL',
                    signal_strength=signal_strength,
                    triggering_condition=f'Price ({current_price:.4f}) crossed below {indicator_name} ({current_ma:.4f})',
                    mathematical_explanation=f'Bearish crossover detected: Price/MA ratio = {current_price/current_ma:.4f} < 1.0, Previous ratio = {prev_price/prev_ma:.4f}',
                    market_context={
                        'price': current_price,
                        'ma_value': current_ma,
                        'price_ma_ratio': current_price / current_ma,
                        'crossover_magnitude': (current_ma - current_price) / current_ma
                    },
                    signal_confidence=signal_strength,
                    risk_assessment='MEDIUM',
                    entry_price=current_price
                ))

            signals.iloc[i] = signal

        return signals

    def _generate_macd_signals_with_rationale(self, macd: pd.Series, signal: pd.Series) -> pd.Series:
        """Generate MACD crossover signals with detailed rationale."""
        signals = pd.Series(0, index=macd.index)
        histogram = macd - signal

        for i in range(1, len(macd)):
            if pd.isna(macd.iloc[i]) or pd.isna(signal.iloc[i]):
                continue

            current_macd = macd.iloc[i]
            current_signal = signal.iloc[i]
            current_histogram = histogram.iloc[i]
            prev_histogram = histogram.iloc[i-1]

            signal_value = 0

            # Bullish crossover: MACD crosses above signal line
            if current_histogram > 0 and prev_histogram <= 0:
                signal_value = 1
                momentum_strength = abs(current_histogram) / abs(current_macd) if current_macd != 0 else 0

                self.signal_history.append(SignalEvent(
                    timestamp=str(macd.index[i]),
                    indicator_name='MACD',
                    signal_type='BUY',
                    signal_strength=min(1.0, momentum_strength * 2),
                    triggering_condition=f'MACD ({current_macd:.4f}) crossed above Signal line ({current_signal:.4f})',
                    mathematical_explanation=f'Bullish momentum: MACD - Signal = {current_histogram:.4f} > 0, Previous histogram = {prev_histogram:.4f}',
                    market_context={
                        'macd_value': current_macd,
                        'signal_line': current_signal,
                        'histogram': current_histogram,
                        'momentum_strength': momentum_strength
                    },
                    signal_confidence=min(1.0, momentum_strength * 2),
                    risk_assessment='MEDIUM'
                ))

            # Bearish crossover: MACD crosses below signal line
            elif current_histogram < 0 and prev_histogram >= 0:
                signal_value = -1
                momentum_strength = abs(current_histogram) / abs(current_macd) if current_macd != 0 else 0

                self.signal_history.append(SignalEvent(
                    timestamp=str(macd.index[i]),
                    indicator_name='MACD',
                    signal_type='SELL',
                    signal_strength=min(1.0, momentum_strength * 2),
                    triggering_condition=f'MACD ({current_macd:.4f}) crossed below Signal line ({current_signal:.4f})',
                    mathematical_explanation=f'Bearish momentum: MACD - Signal = {current_histogram:.4f} < 0, Previous histogram = {prev_histogram:.4f}',
                    market_context={
                        'macd_value': current_macd,
                        'signal_line': current_signal,
                        'histogram': current_histogram,
                        'momentum_strength': momentum_strength
                    },
                    signal_confidence=min(1.0, momentum_strength * 2),
                    risk_assessment='MEDIUM'
                ))

            signals.iloc[i] = signal_value

        return signals

    def _generate_rsi_signals_with_rationale(self, rsi: pd.Series, oversold: float = 30, overbought: float = 70) -> pd.Series:
        """Generate RSI overbought/oversold signals with detailed rationale."""
        signals = pd.Series(0, index=rsi.index)

        for i in range(1, len(rsi)):
            if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i-1]):
                continue

            current_rsi = rsi.iloc[i]
            prev_rsi = rsi.iloc[i-1]
            signal_value = 0

            # Buy signal: RSI crosses above oversold threshold
            if current_rsi > oversold and prev_rsi <= oversold:
                signal_value = 1
                oversold_magnitude = (current_rsi - oversold) / (50 - oversold)  # Normalize to 0-1
                momentum_strength = min(1.0, oversold_magnitude)

                self.signal_history.append(SignalEvent(
                    timestamp=str(rsi.index[i]),
                    indicator_name='RSI',
                    signal_type='BUY',
                    signal_strength=momentum_strength,
                    triggering_condition=f'RSI ({current_rsi:.2f}) crossed above oversold threshold ({oversold})',
                    mathematical_explanation=f'Oversold recovery: RSI increased from {prev_rsi:.2f} to {current_rsi:.2f}, suggesting price may be bottoming',
                    market_context={
                        'rsi_value': current_rsi,
                        'oversold_threshold': oversold,
                        'overbought_threshold': overbought,
                        'distance_from_neutral': abs(current_rsi - 50),
                        'momentum_change': current_rsi - prev_rsi
                    },
                    signal_confidence=momentum_strength,
                    risk_assessment='MEDIUM'
                ))

            # Sell signal: RSI crosses below overbought threshold
            elif current_rsi < overbought and prev_rsi >= overbought:
                signal_value = -1
                overbought_magnitude = (overbought - current_rsi) / (overbought - 50)  # Normalize to 0-1
                momentum_strength = min(1.0, overbought_magnitude)

                self.signal_history.append(SignalEvent(
                    timestamp=str(rsi.index[i]),
                    indicator_name='RSI',
                    signal_type='SELL',
                    signal_strength=momentum_strength,
                    triggering_condition=f'RSI ({current_rsi:.2f}) crossed below overbought threshold ({overbought})',
                    mathematical_explanation=f'Overbought correction: RSI decreased from {prev_rsi:.2f} to {current_rsi:.2f}, suggesting price may be topping',
                    market_context={
                        'rsi_value': current_rsi,
                        'oversold_threshold': oversold,
                        'overbought_threshold': overbought,
                        'distance_from_neutral': abs(current_rsi - 50),
                        'momentum_change': current_rsi - prev_rsi
                    },
                    signal_confidence=momentum_strength,
                    risk_assessment='MEDIUM'
                ))

            signals.iloc[i] = signal_value

        return signals

    def _generate_stochastic_signals_with_rationale(self, stoch: pd.Series, oversold: float = 20, overbought: float = 80) -> pd.Series:
        """Generate stochastic oscillator signals with basic rationale."""
        # For now, use original logic but could be enhanced later
        signals = pd.Series(0, index=stoch.index)
        signals[(stoch > oversold) & (stoch.shift(1) <= oversold)] = 1
        signals[(stoch < overbought) & (stoch.shift(1) >= overbought)] = -1
        return signals

    def _generate_williams_r_signals_with_rationale(self, williams_r: pd.Series, oversold: float = -80, overbought: float = -20) -> pd.Series:
        """Generate Williams %R signals with basic rationale."""
        # For now, use original logic but could be enhanced later
        signals = pd.Series(0, index=williams_r.index)
        signals[(williams_r > oversold) & (williams_r.shift(1) <= oversold)] = 1
        signals[(williams_r < overbought) & (williams_r.shift(1) >= overbought)] = -1
        return signals

    def _generate_cci_signals_with_rationale(self, cci: pd.Series, oversold: float = -100, overbought: float = 100) -> pd.Series:
        """Generate Commodity Channel Index signals with basic rationale."""
        # For now, use original logic but could be enhanced later
        signals = pd.Series(0, index=cci.index)
        signals[(cci > oversold) & (cci.shift(1) <= oversold)] = 1
        signals[(cci < overbought) & (cci.shift(1) >= overbought)] = -1
        return signals

    def _generate_bollinger_signals_with_rationale(self, price: pd.Series, upper: pd.Series, lower: pd.Series) -> pd.Series:
        """Generate Bollinger Band mean reversion signals with detailed rationale."""
        signals = pd.Series(0, index=price.index)
        middle = (upper + lower) / 2  # Moving average (middle band)
        bandwidth = (upper - lower) / middle  # Bollinger Band width

        for i in range(1, len(price)):
            if pd.isna(price.iloc[i]) or pd.isna(upper.iloc[i]) or pd.isna(lower.iloc[i]):
                continue

            current_price = price.iloc[i]
            prev_price = price.iloc[i-1]
            current_upper = upper.iloc[i]
            current_lower = lower.iloc[i]
            current_middle = middle.iloc[i]
            current_bandwidth = bandwidth.iloc[i]

            signal_value = 0

            # Buy signal: price touches or crosses below lower band
            if current_price <= current_lower and prev_price > lower.iloc[i-1]:
                signal_value = 1
                oversold_magnitude = (current_lower - current_price) / current_lower
                band_position = (current_price - current_lower) / (current_upper - current_lower)

                self.signal_history.append(SignalEvent(
                    timestamp=str(price.index[i]),
                    indicator_name='Bollinger_Bands',
                    signal_type='BUY',
                    signal_strength=min(1.0, oversold_magnitude * 5),
                    triggering_condition=f'Price ({current_price:.4f}) touched lower Bollinger Band ({current_lower:.4f})',
                    mathematical_explanation=f'Mean reversion opportunity: Price at {band_position:.2%} of band width, suggesting oversold condition',
                    market_context={
                        'price': current_price,
                        'upper_band': current_upper,
                        'lower_band': current_lower,
                        'middle_band': current_middle,
                        'band_width': current_bandwidth,
                        'band_position': band_position,
                        'distance_from_middle': (current_price - current_middle) / current_middle
                    },
                    signal_confidence=min(1.0, oversold_magnitude * 3),
                    risk_assessment='HIGH'
                ))

            # Sell signal: price touches or crosses above upper band
            elif current_price >= current_upper and prev_price < upper.iloc[i-1]:
                signal_value = -1
                overbought_magnitude = (current_price - current_upper) / current_upper
                band_position = (current_price - current_lower) / (current_upper - current_lower)

                self.signal_history.append(SignalEvent(
                    timestamp=str(price.index[i]),
                    indicator_name='Bollinger_Bands',
                    signal_type='SELL',
                    signal_strength=min(1.0, overbought_magnitude * 5),
                    triggering_condition=f'Price ({current_price:.4f}) touched upper Bollinger Band ({current_upper:.4f})',
                    mathematical_explanation=f'Mean reversion opportunity: Price at {band_position:.2%} of band width, suggesting overbought condition',
                    market_context={
                        'price': current_price,
                        'upper_band': current_upper,
                        'lower_band': current_lower,
                        'middle_band': current_middle,
                        'band_width': current_bandwidth,
                        'band_position': band_position,
                        'distance_from_middle': (current_price - current_middle) / current_middle
                    },
                    signal_confidence=min(1.0, overbought_magnitude * 3),
                    risk_assessment='HIGH'
                ))

            signals.iloc[i] = signal_value

        return signals

    def _generate_adx_signals_with_rationale(self, adx: pd.Series, ma: pd.Series, price: pd.Series, threshold: float = 25) -> pd.Series:
        """Generate ADX trend strength signals with basic rationale."""
        # For now, use original logic but could be enhanced later
        signals = pd.Series(0, index=adx.index)
        strong_trend = adx > threshold
        price_above_ma = price > ma
        signals[strong_trend & price_above_ma] = 1
        signals[strong_trend & ~price_above_ma] = -1
        return signals

    def _generate_aroon_signals_with_rationale(self, aroon_up: pd.Series, aroon_down: pd.Series) -> pd.Series:
        """Generate Aroon signals with basic rationale."""
        # For now, use original logic but could be enhanced later
        signals = pd.Series(0, index=aroon_up.index)
        up_above_down = (aroon_up > aroon_down).fillna(False)
        up_above_down_prev = up_above_down.shift(1).fillna(False)
        signals[up_above_down & ~up_above_down_prev] = 1
        signals[~up_above_down & up_above_down_prev] = -1
        return signals

    def _generate_roc_signals_with_rationale(self, roc: pd.Series, threshold: float = 2.0) -> pd.Series:
        """Generate Rate of Change signals with basic rationale."""
        # For now, use original logic but could be enhanced later
        signals = pd.Series(0, index=roc.index)
        signals[(roc > threshold) & (roc.shift(1) <= threshold)] = 1
        signals[(roc < -threshold) & (roc.shift(1) >= -threshold)] = -1
        return signals

    def _generate_vwap_signals_with_rationale(self, price: pd.Series, vwap: pd.Series) -> pd.Series:
        """Generate VWAP signals with basic rationale."""
        # For now, use original logic but could be enhanced later
        signals = pd.Series(0, index=price.index)
        price_above_vwap = (price > vwap).fillna(False)
        price_above_vwap_prev = price_above_vwap.shift(1).fillna(False)
        signals[price_above_vwap & ~price_above_vwap_prev] = 1
        signals[~price_above_vwap & price_above_vwap_prev] = -1
        return signals

    def analyze_all_indicator_strategies(
        self,
        data: pd.DataFrame,
        price_column: str = 'close'
    ) -> Dict[str, Any]:
        """
        Analyze performance of all technical indicator strategies.

        Args:
            data: OHLCV price data
            price_column: Column name for price data

        Returns:
            Dictionary with performance analysis for each indicator
        """
        # Calculate all indicators
        indicators = self.calculate_all_indicators(data)

        # Generate signals for all indicators
        signals = self.generate_signals(data, indicators)

        # Calculate returns
        returns = data[price_column].pct_change().dropna()

        # Analyze performance for each strategy
        strategy_results = {}

        for indicator_name, signal_series in signals.items():
            try:
                # Align signals with returns
                aligned_signals = signal_series.reindex(returns.index, method='ffill').fillna(0)

                # Calculate strategy returns
                strategy_returns = returns * aligned_signals
                strategy_returns = strategy_returns.dropna()

                if len(strategy_returns) > 0:
                    # Calculate performance metrics with proper trade counting
                    performance = self._calculate_strategy_performance(strategy_returns, aligned_signals)
                    performance['indicator_name'] = indicator_name
                    performance['total_signals'] = (aligned_signals != 0).sum()
                    performance['buy_signals'] = (aligned_signals == 1).sum()
                    performance['sell_signals'] = (aligned_signals == -1).sum()

                    strategy_results[indicator_name] = performance

            except Exception as e:
                print(f"Warning: Failed to analyze {indicator_name}: {e}")
                continue

        return strategy_results

    def select_best_indicators(
        self,
        strategy_results: Dict[str, Any],
        n_best: int = 5,
        ranking_metric: str = 'sharpe_ratio'
    ) -> List[Tuple[str, float]]:
        """
        Select top N performing indicators based on specified metric.

        Args:
            strategy_results: Results from analyze_all_indicator_strategies
            n_best: Number of top indicators to select
            ranking_metric: Metric to rank by ('sharpe_ratio', 'total_return', 'sortino_ratio')

        Returns:
            List of (indicator_name, metric_value) tuples sorted by performance
        """
        # Extract metric values for ranking
        indicator_scores = []

        for indicator_name, results in strategy_results.items():
            if ranking_metric in results and results[ranking_metric] is not None:
                metric_value = results[ranking_metric]
                # Handle inf/-inf values
                if not np.isfinite(metric_value):
                    metric_value = 0.0
                indicator_scores.append((indicator_name, metric_value))

        # Sort by metric (descending for most metrics)
        if ranking_metric in ['max_drawdown']:  # Metrics where lower is better
            indicator_scores.sort(key=lambda x: x[1])
        else:  # Metrics where higher is better
            indicator_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top N
        return indicator_scores[:n_best]

    def _calculate_strategy_performance(self, returns: pd.Series, positions: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics for a strategy."""
        if len(returns) == 0:
            return self._empty_performance_metrics()

        # Basic return metrics
        total_return = returns.sum()
        annualized_return = returns.mean() * 252  # Assume daily data

        # Risk metrics
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0.0

        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (returns > 0).mean()

        # Calculate proper trade count from position changes
        num_trades = self._calculate_trade_count(positions) if positions is not None else len(returns)

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'maximum_drawdown': max_drawdown,
            'win_rate': win_rate,
            'number_of_trades': num_trades
        }

    def _calculate_trade_count(self, positions: pd.Series) -> int:
        """Calculate number of trades from position changes."""
        if positions is None or len(positions) == 0:
            return 0

        # Count position changes (signal transitions)
        position_changes = (positions != positions.shift(1)).sum()

        # For technical indicators, each signal change is one trade
        # (we don't add the final exit trade like in buy-and-hold)
        return int(position_changes)

    def _empty_performance_metrics(self) -> Dict[str, float]:
        """Return empty performance metrics dictionary."""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'maximum_drawdown': 0.0,
            'win_rate': 0.0,
            'number_of_trades': 0
        }

    def collect_signal_data_for_timestep(self, timestamp: str, data: pd.DataFrame, indicators: pd.DataFrame) -> Dict[str, Any]:
        """Collect technical indicator signal data for a specific timestep."""
        price = data['close']

        # Get current timestamp index
        try:
            current_idx = data.index.get_loc(pd.to_datetime(timestamp))
        except (KeyError, ValueError):
            # Fallback to most recent data if exact timestamp not found
            current_idx = len(data) - 1
            timestamp = str(data.index[current_idx])

        current_data = data.iloc[current_idx]
        current_indicators = indicators.iloc[current_idx]

        # Calculate signals up to current timestamp
        signals = self.generate_signals(data.iloc[:current_idx+1], indicators.iloc[:current_idx+1])

        # Extract current signals
        current_signals = {name: series.iloc[-1] if len(series) > 0 else 0
                          for name, series in signals.items()}

        # Build comprehensive timestep data
        timestep_data = {
            'timestamp': timestamp,
            'price_data': {
                'open': float(current_data['open']),
                'high': float(current_data['high']),
                'low': float(current_data['low']),
                'close': float(current_data['close']),
                'volume': float(current_data.get('volume', 0))
            },
            'indicator_values': {
                name: float(value) if not pd.isna(value) else None
                for name, value in current_indicators.items()
            },
            'generated_signals': current_signals,
            'signal_events': [event.__dict__ for event in self.signal_history
                             if event.timestamp == timestamp],
            'signal_summary': self.get_signal_summary(),
            'active_indicators': list(current_signals.keys()),
            'market_context': {
                'price_change': float((current_data['close'] - data['close'].iloc[max(0, current_idx-1)]) / data['close'].iloc[max(0, current_idx-1)] * 100) if current_idx > 0 else 0.0,
                'volume_ratio': float(current_data.get('volume', 0) / data['volume'].iloc[max(0, current_idx-1)]) if current_idx > 0 and 'volume' in data.columns else 1.0,
                'volatility_estimate': float(data['close'].iloc[max(0, current_idx-19):current_idx+1].pct_change().std() * np.sqrt(252)) if current_idx >= 19 else 0.0
            }
        }

        return timestep_data

    def get_signal_attribution_analysis(self) -> Dict[str, Any]:
        """Analyze signal sources and their performance contributions."""
        if not self.signal_history:
            return {'total_signals': 0, 'attribution': {}}

        attribution = {}

        # Group signals by indicator
        for signal in self.signal_history:
            indicator = signal.indicator_name
            if indicator not in attribution:
                attribution[indicator] = {
                    'total_signals': 0,
                    'buy_signals': 0,
                    'sell_signals': 0,
                    'avg_signal_strength': 0,
                    'avg_confidence': 0,
                    'risk_distribution': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
                }

            attribution[indicator]['total_signals'] += 1
            if signal.signal_type == 'BUY':
                attribution[indicator]['buy_signals'] += 1
            elif signal.signal_type == 'SELL':
                attribution[indicator]['sell_signals'] += 1

            attribution[indicator]['risk_distribution'][signal.risk_assessment] += 1

        # Calculate averages
        for indicator in attribution:
            indicator_signals = [s for s in self.signal_history if s.indicator_name == indicator]
            attribution[indicator]['avg_signal_strength'] = np.mean([s.signal_strength for s in indicator_signals])
            attribution[indicator]['avg_confidence'] = np.mean([s.signal_confidence for s in indicator_signals])

        return {
            'total_signals': len(self.signal_history),
            'attribution': attribution,
            'top_signal_generators': sorted(attribution.keys(),
                                          key=lambda x: attribution[x]['total_signals'],
                                          reverse=True)[:5]
        }