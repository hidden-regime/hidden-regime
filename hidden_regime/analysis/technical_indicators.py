"""
Comprehensive technical indicator analysis and signal generation.

Provides systematic calculation of technical indicators and generation of
buy/sell signals for comparison against HMM regime detection strategies.
"""

import numpy as np
import pandas as pd
import ta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .performance import RegimePerformanceAnalyzer
from ..utils.exceptions import AnalysisError


class TechnicalIndicatorAnalyzer:
    """
    Comprehensive technical indicator analyzer with signal generation.

    Calculates 15+ technical indicators and generates systematic buy/sell signals
    for performance comparison against HMM regime detection strategies.
    """

    def __init__(self):
        """Initialize technical indicator analyzer."""
        self.performance_analyzer = RegimePerformanceAnalyzer()

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

        # Moving average signals
        signals['sma_20'] = self._generate_ma_crossover_signals(price, indicators['sma_20'])
        signals['sma_50'] = self._generate_ma_crossover_signals(price, indicators['sma_50'])
        signals['ema_12'] = self._generate_ma_crossover_signals(price, indicators['ema_12'])
        signals['ema_26'] = self._generate_ma_crossover_signals(price, indicators['ema_26'])

        # MACD signals
        signals['macd'] = self._generate_macd_signals(indicators['macd'], indicators['macd_signal'])

        # RSI signals
        signals['rsi'] = self._generate_rsi_signals(indicators['rsi'])

        # Stochastic signals
        signals['stoch'] = self._generate_stochastic_signals(indicators['stoch'])

        # Williams %R signals
        signals['williams_r'] = self._generate_williams_r_signals(indicators['williams_r'])

        # CCI signals
        signals['cci'] = self._generate_cci_signals(indicators['cci'])

        # Bollinger Band signals
        signals['bollinger'] = self._generate_bollinger_signals(price, indicators['bollinger_upper'], indicators['bollinger_lower'])

        # ADX trend strength signals
        signals['adx'] = self._generate_adx_signals(indicators['adx'], indicators['sma_20'], price)

        # Aroon signals
        signals['aroon'] = self._generate_aroon_signals(indicators['aroon_up'], indicators['aroon_down'])

        # Rate of Change signals
        signals['roc'] = self._generate_roc_signals(indicators['roc'])

        # Volume-based signals (if available)
        if 'vwap' in indicators.columns and not indicators['vwap'].isna().all():
            signals['vwap'] = self._generate_vwap_signals(price, indicators['vwap'])

        return signals

    def _generate_ma_crossover_signals(self, price: pd.Series, ma: pd.Series) -> pd.Series:
        """Generate signals based on price crossing moving average."""
        signals = pd.Series(0, index=price.index)

        # Buy when price crosses above MA, sell when below
        price_above_ma = (price > ma).fillna(False)
        price_above_ma_prev = price_above_ma.shift(1).fillna(False)

        signals[price_above_ma & ~price_above_ma_prev] = 1  # Buy signal
        signals[~price_above_ma & price_above_ma_prev] = -1  # Sell signal

        return signals

    def _generate_macd_signals(self, macd: pd.Series, signal: pd.Series) -> pd.Series:
        """Generate MACD crossover signals."""
        signals = pd.Series(0, index=macd.index)

        macd_above_signal = (macd > signal).fillna(False)
        macd_above_signal_prev = macd_above_signal.shift(1).fillna(False)

        signals[macd_above_signal & ~macd_above_signal_prev] = 1  # Buy
        signals[~macd_above_signal & macd_above_signal_prev] = -1  # Sell

        return signals

    def _generate_rsi_signals(self, rsi: pd.Series, oversold: float = 30, overbought: float = 70) -> pd.Series:
        """Generate RSI overbought/oversold signals."""
        signals = pd.Series(0, index=rsi.index)

        # Buy when RSI crosses above oversold level
        signals[(rsi > oversold) & (rsi.shift(1) <= oversold)] = 1

        # Sell when RSI crosses below overbought level
        signals[(rsi < overbought) & (rsi.shift(1) >= overbought)] = -1

        return signals

    def _generate_stochastic_signals(self, stoch: pd.Series, oversold: float = 20, overbought: float = 80) -> pd.Series:
        """Generate stochastic oscillator signals."""
        signals = pd.Series(0, index=stoch.index)

        signals[(stoch > oversold) & (stoch.shift(1) <= oversold)] = 1
        signals[(stoch < overbought) & (stoch.shift(1) >= overbought)] = -1

        return signals

    def _generate_williams_r_signals(self, williams_r: pd.Series, oversold: float = -80, overbought: float = -20) -> pd.Series:
        """Generate Williams %R signals."""
        signals = pd.Series(0, index=williams_r.index)

        signals[(williams_r > oversold) & (williams_r.shift(1) <= oversold)] = 1
        signals[(williams_r < overbought) & (williams_r.shift(1) >= overbought)] = -1

        return signals

    def _generate_cci_signals(self, cci: pd.Series, oversold: float = -100, overbought: float = 100) -> pd.Series:
        """Generate Commodity Channel Index signals."""
        signals = pd.Series(0, index=cci.index)

        signals[(cci > oversold) & (cci.shift(1) <= oversold)] = 1
        signals[(cci < overbought) & (cci.shift(1) >= overbought)] = -1

        return signals

    def _generate_bollinger_signals(self, price: pd.Series, upper: pd.Series, lower: pd.Series) -> pd.Series:
        """Generate Bollinger Band mean reversion signals."""
        signals = pd.Series(0, index=price.index)

        # Buy when price touches lower band (oversold)
        signals[(price <= lower) & (price.shift(1) > lower)] = 1

        # Sell when price touches upper band (overbought)
        signals[(price >= upper) & (price.shift(1) < upper)] = -1

        return signals

    def _generate_adx_signals(self, adx: pd.Series, ma: pd.Series, price: pd.Series, threshold: float = 25) -> pd.Series:
        """Generate ADX trend strength signals."""
        signals = pd.Series(0, index=adx.index)

        # Strong trend + price above MA = buy, below MA = sell
        strong_trend = adx > threshold
        price_above_ma = price > ma

        signals[strong_trend & price_above_ma] = 1
        signals[strong_trend & ~price_above_ma] = -1

        return signals

    def _generate_aroon_signals(self, aroon_up: pd.Series, aroon_down: pd.Series) -> pd.Series:
        """Generate Aroon signals."""
        signals = pd.Series(0, index=aroon_up.index)

        # Buy when Aroon Up crosses above Aroon Down
        up_above_down = (aroon_up > aroon_down).fillna(False)
        up_above_down_prev = up_above_down.shift(1).fillna(False)

        signals[up_above_down & ~up_above_down_prev] = 1
        signals[~up_above_down & up_above_down_prev] = -1

        return signals

    def _generate_roc_signals(self, roc: pd.Series, threshold: float = 2.0) -> pd.Series:
        """Generate Rate of Change signals."""
        signals = pd.Series(0, index=roc.index)

        # Buy when ROC crosses above threshold, sell when below
        signals[(roc > threshold) & (roc.shift(1) <= threshold)] = 1
        signals[(roc < -threshold) & (roc.shift(1) >= -threshold)] = -1

        return signals

    def _generate_vwap_signals(self, price: pd.Series, vwap: pd.Series) -> pd.Series:
        """Generate VWAP signals."""
        signals = pd.Series(0, index=price.index)

        # Buy when price crosses above VWAP, sell when below
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