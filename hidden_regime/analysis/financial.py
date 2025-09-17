"""
Financial analysis component for pipeline architecture.

Provides FinancialAnalysis that implements AnalysisComponent interface for
interpreting HMM regime predictions in financial context with domain knowledge.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from ..pipeline.interfaces import AnalysisComponent
from ..config.analysis import FinancialAnalysisConfig
from ..utils.exceptions import ValidationError
from .performance import RegimePerformanceAnalyzer

# Try to import technical indicators
try:
    from ..indicators import (
        IndicatorCalculator,
        calculate_all_indicators,
        compare_hmm_vs_indicators
    )
    INDICATORS_AVAILABLE = True
except ImportError:
    INDICATORS_AVAILABLE = False
    warnings.warn("Technical indicators not available - indicator comparisons disabled")


class FinancialAnalysis(AnalysisComponent):
    """
    Financial analysis component for interpreting HMM regime predictions.
    
    Implements AnalysisComponent interface to provide financial domain knowledge
    interpretation of regime states including performance metrics and trading insights.
    """
    
    def __init__(self, config: FinancialAnalysisConfig):
        """
        Initialize financial analysis with configuration.
        
        Args:
            config: FinancialAnalysisConfig with analysis parameters
        """
        self.config = config
        self._last_model_output = None
        self._last_analysis = None
        self._last_raw_data = None
        
        # Get regime labels
        self.regime_labels = config.get_default_regime_labels()
        
        # Cache for computed statistics
        self._regime_stats_cache = {}
        
        # Initialize indicator calculator if available
        if INDICATORS_AVAILABLE and self.config.indicator_comparisons:
            self.indicator_calculator = IndicatorCalculator()
            self._indicators_cache = {}
        else:
            self.indicator_calculator = None
            self._indicators_cache = {}
        
        # Initialize performance analyzer
        self.performance_analyzer = RegimePerformanceAnalyzer()
    
    def update(self, model_output: pd.DataFrame, raw_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Interpret model output and add domain knowledge.
        
        Args:
            model_output: Raw model predictions with predicted_state and confidence
            raw_data: Optional raw OHLCV data for indicator calculations
            
        Returns:
            DataFrame with interpreted analysis results
        """
        if model_output.empty:
            raise ValidationError("Model output cannot be empty")
        
        # Validate required columns
        required_cols = ['predicted_state', 'confidence']
        missing_cols = [col for col in required_cols if col not in model_output.columns]
        if missing_cols:
            raise ValidationError(f"Required columns missing from model output: {missing_cols}")
        
        # Store references for plotting and indicator calculations
        self._last_model_output = model_output.copy()
        self._last_raw_data = raw_data.copy() if raw_data is not None else None
        
        # Start with model output
        analysis = model_output.copy()
        
        # Add regime interpretations
        analysis = self._add_regime_interpretations(analysis)
        
        # Add regime statistics if requested
        if self.config.calculate_regime_statistics:
            analysis = self._add_regime_statistics(analysis)
        
        # Add duration analysis if requested
        if self.config.include_duration_analysis:
            analysis = self._add_duration_analysis(analysis)
        
        # Add return analysis if requested  
        if self.config.include_return_analysis:
            analysis = self._add_return_analysis(analysis)
        
        # Add volatility analysis if requested
        if self.config.include_volatility_analysis:
            analysis = self._add_volatility_analysis(analysis)
        
        # Add indicator comparisons if requested and available
        if (self.config.include_indicator_performance and 
            self.config.indicator_comparisons and 
            raw_data is not None):
            analysis = self._add_indicator_comparisons(analysis, raw_data)
        
        # Add trading signals if requested
        if self.config.include_trading_signals:
            analysis = self._add_trading_signals(analysis)
        
        # Store for plotting
        self._last_analysis = analysis.copy()
        
        return analysis
    
    def _add_regime_interpretations(self, analysis: pd.DataFrame) -> pd.DataFrame:
        """Add regime name interpretations."""
        # Map state numbers to regime names
        analysis['regime_name'] = analysis['predicted_state'].map(
            {i: name for i, name in enumerate(self.regime_labels)}
        )
        
        # Add regime type classification
        analysis['regime_type'] = analysis['predicted_state'].map(self._classify_regime_type)
        
        return analysis
    
    def _classify_regime_type(self, state: int) -> str:
        """Classify regime type based on state number and configuration."""
        if self.config.n_states == 2:
            return "Bear" if state == 0 else "Bull"
        elif self.config.n_states == 3:
            if state == 0:
                return "Bear"
            elif state == 1:
                return "Sideways"
            else:
                return "Bull"
        elif self.config.n_states == 4:
            types = ["Crisis", "Bear", "Sideways", "Bull"]
            return types[min(state, len(types) - 1)]
        elif self.config.n_states == 5:
            types = ["Crisis", "Bear", "Sideways", "Bull", "Euphoric"]
            return types[min(state, len(types) - 1)]
        else:
            return f"Regime_{state}"
    
    def _add_regime_statistics(self, analysis: pd.DataFrame) -> pd.DataFrame:
        """Add basic regime statistics."""
        # Calculate days in current regime
        analysis['days_in_regime'] = self._calculate_days_in_regime(analysis['predicted_state'])
        
        # Add expected regime characteristics (simplified)
        regime_characteristics = self._get_regime_characteristics()
        
        for char, values in regime_characteristics.items():
            analysis[f'expected_{char}'] = analysis['predicted_state'].map(
                {i: values[i] if i < len(values) else 0.0 for i in range(self.config.n_states)}
            )
        
        return analysis
    
    def _calculate_days_in_regime(self, states: pd.Series) -> pd.Series:
        """Calculate number of consecutive days in current regime."""
        days_in_regime = []
        current_count = 1
        
        for i in range(len(states)):
            if i == 0:
                days_in_regime.append(current_count)
            elif states.iloc[i] == states.iloc[i-1]:
                current_count += 1
                days_in_regime.append(current_count)
            else:
                current_count = 1
                days_in_regime.append(current_count)
        
        return pd.Series(days_in_regime, index=states.index)
    
    def _get_regime_characteristics(self) -> Dict[str, List[float]]:
        """Get expected regime characteristics based on financial knowledge."""
        if self.config.n_states == 3:
            return {
                'return': [-0.002, 0.0001, 0.001],  # Bear, Sideways, Bull daily returns
                'volatility': [0.025, 0.012, 0.018],  # Expected daily volatility
                'duration': [8.0, 15.0, 12.0]  # Expected regime duration in days
            }
        elif self.config.n_states == 4:
            return {
                'return': [-0.005, -0.002, 0.0001, 0.001],  # Crisis, Bear, Sideways, Bull
                'volatility': [0.040, 0.025, 0.012, 0.018],
                'duration': [5.0, 8.0, 15.0, 12.0]
            }
        else:
            # Default to neutral characteristics
            return {
                'return': [0.0] * self.config.n_states,
                'volatility': [0.015] * self.config.n_states,
                'duration': [10.0] * self.config.n_states
            }
    
    def _add_duration_analysis(self, analysis: pd.DataFrame) -> pd.DataFrame:
        """Add regime duration analysis."""
        # Calculate regime transitions
        transitions = (analysis['predicted_state'] != analysis['predicted_state'].shift(1)).cumsum()
        analysis['regime_episode'] = transitions
        
        # Calculate expected remaining duration (simplified)
        analysis['expected_remaining_duration'] = analysis.apply(
            lambda row: max(1, row['expected_duration'] - row['days_in_regime']), axis=1
        )
        
        return analysis
    
    def _add_return_analysis(self, analysis: pd.DataFrame) -> pd.DataFrame:
        """Add return-based analysis."""
        # Add rolling returns if we have sufficient data
        if len(analysis) >= self.config.return_window:
            # This is a placeholder - in real implementation would need price data
            analysis['rolling_return'] = 0.0  # Placeholder
            analysis['return_vs_expected'] = 0.0  # Placeholder
        
        return analysis
    
    def _add_volatility_analysis(self, analysis: pd.DataFrame) -> pd.DataFrame:
        """Add volatility analysis."""
        # Add rolling volatility if we have sufficient data
        if len(analysis) >= self.config.volatility_window:
            # This is a placeholder - in real implementation would need price data
            analysis['rolling_volatility'] = 0.0  # Placeholder
            analysis['volatility_vs_expected'] = 0.0  # Placeholder
        
        return analysis
    
    def _add_indicator_comparisons(self, analysis: pd.DataFrame, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator comparisons."""
        if not INDICATORS_AVAILABLE or not self.indicator_calculator:
            return analysis
        
        try:
            # Calculate indicators for the specified comparisons
            for indicator_name in self.config.indicator_comparisons:
                indicator_values = self._calculate_single_indicator(indicator_name, raw_data)
                
                if indicator_values is not None and len(indicator_values) == len(analysis):
                    # Add indicator values
                    analysis[f'{indicator_name}_value'] = indicator_values
                    
                    # Add indicator signals (simplified)
                    analysis[f'{indicator_name}_signal'] = self._get_indicator_signal(
                        indicator_name, indicator_values
                    )
                    
                    # Add regime vs indicator agreement
                    analysis[f'{indicator_name}_agreement'] = self._calculate_regime_indicator_agreement(
                        analysis['predicted_state'], analysis[f'{indicator_name}_signal']
                    )
            
            # Add overall indicator consensus if multiple indicators
            if len(self.config.indicator_comparisons) > 1:
                signal_cols = [f'{ind}_signal' for ind in self.config.indicator_comparisons 
                              if f'{ind}_signal' in analysis.columns]
                if signal_cols:
                    analysis['indicator_consensus'] = analysis[signal_cols].mean(axis=1)
                    analysis['regime_consensus_agreement'] = self._calculate_regime_consensus_agreement(
                        analysis['predicted_state'], analysis['indicator_consensus']
                    )
        
        except Exception as e:
            warnings.warn(f"Indicator comparison failed: {e}")
        
        return analysis
    
    def _calculate_single_indicator(self, indicator_name: str, raw_data: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate a single technical indicator."""
        try:
            # Simple implementations for common indicators
            if indicator_name.lower() == 'rsi':
                if 'close' in raw_data.columns:
                    return self._calculate_rsi(raw_data['close'])
            elif indicator_name.lower() == 'macd':
                if 'close' in raw_data.columns:
                    return self._calculate_macd_signal(raw_data['close'])
            elif indicator_name.lower() == 'bollinger_bands':
                if 'close' in raw_data.columns:
                    return self._calculate_bollinger_position(raw_data['close'])
            elif indicator_name.lower() == 'moving_average':
                if 'close' in raw_data.columns:
                    return self._calculate_ma_signal(raw_data['close'])
            
            # Fallback: try to use full indicators calculator if available
            if hasattr(self.indicator_calculator, 'calculate_all_indicators'):
                indicators = self.indicator_calculator.calculate_all_indicators(raw_data)
                if indicator_name in indicators.columns:
                    return indicators[indicator_name]
        
        except Exception as e:
            warnings.warn(f"Failed to calculate {indicator_name}: {e}")
        
        return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd_signal(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD signal line."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line - signal_line  # MACD histogram
    
    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
        """Calculate position within Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return (prices - lower_band) / (upper_band - lower_band)
    
    def _calculate_ma_signal(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate moving average signal."""
        ma = prices.rolling(window=period).mean()
        return (prices - ma) / ma  # Relative position to MA
    
    def _get_indicator_signal(self, indicator_name: str, values: pd.Series) -> pd.Series:
        """Convert indicator values to standardized signals (-1 to 1)."""
        if indicator_name.lower() == 'rsi':
            # RSI: >70 overbought (sell), <30 oversold (buy)
            signals = np.where(values > 70, -1, np.where(values < 30, 1, 0))
            return pd.Series(signals, index=values.index)
        elif indicator_name.lower() == 'macd':
            # MACD: positive = bullish, negative = bearish
            return np.sign(values)
        elif indicator_name.lower() == 'bollinger_bands':
            # Bollinger position: >1 above upper band, <0 below lower band
            signals = np.where(values > 1, -1, np.where(values < 0, 1, 0))
            return pd.Series(signals, index=values.index)
        elif indicator_name.lower() == 'moving_average':
            # MA signal: positive = above MA (bullish), negative = below MA (bearish)
            return np.sign(values)
        else:
            # Default: standardize to -1, 1 range
            return np.sign(values - values.median())
    
    def _calculate_regime_indicator_agreement(self, regime_states: pd.Series, indicator_signals: pd.Series) -> pd.Series:
        """Calculate agreement between regime and indicator signals."""
        # Convert regime states to signals (simplified)
        if self.config.n_states == 3:
            # 0=Bear(-1), 1=Sideways(0), 2=Bull(+1)
            regime_signals = regime_states.map({0: -1, 1: 0, 2: 1})
        elif self.config.n_states == 4:
            # 0=Crisis(-1), 1=Bear(-1), 2=Sideways(0), 3=Bull(+1)
            regime_signals = regime_states.map({0: -1, 1: -1, 2: 0, 3: 1})
        else:
            # Default mapping
            regime_signals = regime_states - (self.config.n_states // 2)
        
        # Calculate agreement as correlation
        agreement = []
        window = 20  # Rolling window for agreement calculation
        
        for i in range(len(regime_signals)):
            if i < window:
                agreement.append(0.0)  # Not enough data
            else:
                regime_window = regime_signals.iloc[i-window:i]
                indicator_window = indicator_signals.iloc[i-window:i]
                
                # Calculate correlation
                try:
                    corr = regime_window.corr(indicator_window)
                    agreement.append(corr if not pd.isna(corr) else 0.0)
                except:
                    agreement.append(0.0)
        
        return pd.Series(agreement, index=regime_signals.index)
    
    def _calculate_regime_consensus_agreement(self, regime_states: pd.Series, consensus: pd.Series) -> pd.Series:
        """Calculate agreement between regime and indicator consensus."""
        # Similar to single indicator but using consensus signal
        if self.config.n_states == 3:
            regime_signals = regime_states.map({0: -1, 1: 0, 2: 1})
        elif self.config.n_states == 4:
            regime_signals = regime_states.map({0: -1, 1: -1, 2: 0, 3: 1})
        else:
            regime_signals = regime_states - (self.config.n_states // 2)
        
        # Calculate rolling correlation
        agreement = []
        window = 20
        
        for i in range(len(regime_signals)):
            if i < window:
                agreement.append(0.0)
            else:
                regime_window = regime_signals.iloc[i-window:i]
                consensus_window = consensus.iloc[i-window:i]
                
                try:
                    corr = regime_window.corr(consensus_window)
                    agreement.append(corr if not pd.isna(corr) else 0.0)
                except:
                    agreement.append(0.0)
        
        return pd.Series(agreement, index=regime_signals.index)
    
    def _add_trading_signals(self, analysis: pd.DataFrame) -> pd.DataFrame:
        """Add trading signals based on regime analysis."""
        # Simple position sizing based on regime and confidence
        position_signals = []
        
        for _, row in analysis.iterrows():
            regime_type = row['regime_type']
            confidence = row['confidence']
            
            if regime_type == "Crisis":
                signal = -0.5 * confidence  # Defensive position
            elif regime_type == "Bear":
                signal = -0.3 * confidence  # Short position
            elif regime_type == "Sideways":
                signal = 0.1 * confidence   # Minimal position
            elif regime_type == "Bull":
                signal = 0.8 * confidence   # Long position
            elif regime_type == "Euphoric":
                signal = 0.5 * confidence   # Reduced position (risk management)
            else:
                signal = 0.0
            
            # Apply risk adjustment if enabled
            if self.config.risk_adjustment:
                expected_vol = row.get('expected_volatility', 0.015)
                risk_factor = min(1.0, 0.015 / max(expected_vol, 0.005))  # Scale by volatility
                signal *= risk_factor
            
            position_signals.append(signal)
        
        analysis['position_signal'] = position_signals
        analysis['signal_strength'] = analysis['position_signal'].abs()
        
        return analysis
    
    def plot(self, **kwargs) -> plt.Figure:
        """
        Generate visualization for analysis results.
        
        Returns:
            matplotlib Figure with analysis visualizations
        """
        if self._last_analysis is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No analysis results yet', ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        analysis = self._last_analysis
        
        # Create subplots
        n_plots = 3
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
        
        # Plot 1: Regime sequence with confidence
        ax1 = axes[0]
        
        # Color map for regimes
        regime_colors = {
            "Crisis": "red",
            "Bear": "orange", 
            "Sideways": "gray",
            "Bull": "green",
            "Euphoric": "purple"
        }
        
        # Plot regime states as colored background
        for i, (idx, row) in enumerate(analysis.iterrows()):
            regime_type = row.get('regime_type', f"Regime_{row['predicted_state']}")
            color = regime_colors.get(regime_type, "blue")
            alpha = row['confidence'] * 0.7 + 0.3  # Scale alpha by confidence
            
            ax1.bar(i, 1, color=color, alpha=alpha, width=1, edgecolor='none')
        
        ax1.set_title('Regime Sequence (colored by type, opacity by confidence)')
        ax1.set_ylabel('Regime')
        ax1.set_ylim(0, 1)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=regime)
                          for regime, color in regime_colors.items()
                          if regime in analysis['regime_type'].values]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Plot 2: Confidence and days in regime
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        
        x_vals = range(len(analysis))
        line1 = ax2.plot(x_vals, analysis['confidence'], 'b-', linewidth=2, label='Confidence')
        
        if 'days_in_regime' in analysis.columns:
            line2 = ax2_twin.plot(x_vals, analysis['days_in_regime'], 'r--', linewidth=2, label='Days in Regime')
            ax2_twin.set_ylabel('Days in Regime', color='red')
            ax2_twin.tick_params(axis='y', labelcolor='red')
        
        ax2.set_title('Confidence and Regime Duration')
        ax2.set_ylabel('Confidence', color='blue')
        ax2.set_xlabel('Time')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels() if 'days_in_regime' in analysis.columns else ([], [])
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot 3: Trading signals (if available)
        ax3 = axes[2]
        if 'position_signal' in analysis.columns:
            signals = analysis['position_signal']
            
            # Color signals by regime type
            colors = [regime_colors.get(regime_type, "blue") 
                     for regime_type in analysis['regime_type']]
            
            bars = ax3.bar(x_vals, signals, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_title('Trading Position Signals')
            ax3.set_ylabel('Position Size')
            ax3.set_xlabel('Time')
            ax3.grid(True, alpha=0.3)
            
            # Add horizontal lines for reference
            ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Strong Bull')
            ax3.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Strong Bear')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No trading signals generated', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Trading Position Signals')
        
        plt.tight_layout()
        return fig
    
    def get_current_regime_summary(self) -> Dict[str, Any]:
        """Get summary of current regime state."""
        if self._last_analysis is None or len(self._last_analysis) == 0:
            return {"status": "No analysis available"}
        
        current = self._last_analysis.iloc[-1]
        
        summary = {
            "current_state": int(current['predicted_state']),
            "regime_name": current.get('regime_name', f"State_{current['predicted_state']}"),
            "regime_type": current.get('regime_type', 'Unknown'),
            "confidence": float(current['confidence']),
            "days_in_regime": int(current.get('days_in_regime', 0)),
        }
        
        # Add expected characteristics if available
        for key in ['expected_return', 'expected_volatility', 'expected_duration']:
            if key in current:
                summary[key] = float(current[key])
        
        # Add trading signal if available
        if 'position_signal' in current:
            summary['position_signal'] = float(current['position_signal'])
            summary['signal_strength'] = float(current.get('signal_strength', 0))
        
        return summary
    
    def get_comprehensive_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance analysis using RegimePerformanceAnalyzer.
        
        Returns:
            Dictionary with detailed performance metrics and analysis
        """
        if self._last_analysis is None:
            return {"status": "No analysis available"}
        
        try:
            # Get comprehensive performance analysis
            performance_metrics = self.performance_analyzer.analyze_regime_performance(
                analysis_results=self._last_analysis,
                raw_data=self._last_raw_data
            )
            
            return performance_metrics
            
        except Exception as e:
            return {"error": f"Performance analysis failed: {str(e)}"}
    
    def get_regime_transition_matrix(self) -> Optional[Dict[str, Any]]:
        """Get regime transition matrix and statistics."""
        if self._last_analysis is None:
            return None
        
        try:
            performance_metrics = self.get_comprehensive_performance_metrics()
            return performance_metrics.get('transition_analysis', {})
        except Exception:
            return None
    
    def get_regime_duration_statistics(self) -> Optional[Dict[str, Any]]:
        """Get regime duration statistics."""
        if self._last_analysis is None:
            return None
        
        try:
            performance_metrics = self.get_comprehensive_performance_metrics()
            return performance_metrics.get('duration_analysis', {})
        except Exception:
            return None