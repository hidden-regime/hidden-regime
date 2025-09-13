"""
Technical Indicators Visualization Enhancement

Extends the core plotting utilities with advanced visualizations that overlay
technical indicators on HMM regime analysis. Enables rich comparative charts
for blog content and analysis reports.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .plotting import (
    REGIME_COLORS,
    check_plotting_available,
    format_financial_axis,
    get_regime_colors,
    setup_financial_plot_style,
)

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle
    
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_price_with_regimes_and_indicators(
    data: pd.DataFrame,
    indicators: pd.DataFrame,
    regime_states: np.ndarray,
    regime_names: Optional[Dict[int, str]] = None,
    indicator_list: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 12)
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create comprehensive price chart with regime highlighting and indicator overlays.
    
    Args:
        data: Price data with OHLCV columns
        indicators: Technical indicators DataFrame
        regime_states: Array of regime assignments
        regime_names: Optional mapping of state indices to names
        indicator_list: List of indicators to plot (defaults to key indicators)
        title: Chart title
        save_path: Optional path to save the chart
        figsize: Figure size tuple
        
    Returns:
        Tuple of (figure, list of axes)
    """
    check_plotting_available()
    setup_financial_plot_style()
    
    # Default indicators if not specified
    if indicator_list is None:
        indicator_list = ['rsi', 'macd', 'bb_upper', 'bb_middle', 'bb_lower', 'sma_20', 'sma_50']
    
    # Filter available indicators
    available_indicators = [ind for ind in indicator_list if ind in indicators.columns]
    
    # Prepare regime names and colors
    unique_states = np.unique(regime_states)
    if regime_names is None:
        regime_names = {i: f"Regime {i}" for i in unique_states}
    
    colors = get_regime_colors(list(regime_names.values()))
    
    # Create subplot layout: price + indicators
    n_indicator_plots = min(2, len(available_indicators))  # Max 2 indicator subplots
    n_plots = 1 + n_indicator_plots
    
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True,
                            gridspec_kw={'height_ratios': [3] + [1] * n_indicator_plots})
    
    if n_plots == 1:
        axes = [axes]
    
    # Align data
    common_index = data.index.intersection(indicators.index)
    if len(regime_states) != len(common_index):
        # Align regime states with common index
        regime_series = pd.Series(regime_states, index=data.index[:len(regime_states)])
        aligned_regime_states = regime_series.reindex(common_index).fillna(method='ffill').values
    else:
        aligned_regime_states = regime_states
    
    aligned_data = data.reindex(common_index)
    aligned_indicators = indicators.reindex(common_index)
    
    # Main price plot with regime backgrounds
    ax_price = axes[0]
    
    # Plot regime backgrounds
    _plot_regime_backgrounds(ax_price, aligned_data.index, aligned_regime_states, 
                            regime_names, colors)
    
    # Plot price and key moving averages
    ax_price.plot(aligned_data.index, aligned_data['close'], 'k-', linewidth=1.5, 
                 label='Close Price', alpha=0.8)
    
    # Add Bollinger Bands if available
    if all(col in aligned_indicators.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
        ax_price.plot(aligned_data.index, aligned_indicators['bb_upper'], 'b--', 
                     alpha=0.6, label='BB Upper')
        ax_price.plot(aligned_data.index, aligned_indicators['bb_middle'], 'g-', 
                     alpha=0.6, label='BB Middle')
        ax_price.plot(aligned_data.index, aligned_indicators['bb_lower'], 'b--', 
                     alpha=0.6, label='BB Lower')
        ax_price.fill_between(aligned_data.index, aligned_indicators['bb_upper'], 
                             aligned_indicators['bb_lower'], alpha=0.1, color='blue')
    
    # Add moving averages
    for ma in ['sma_10', 'sma_20', 'sma_50', 'sma_200']:
        if ma in aligned_indicators.columns:
            label = ma.replace('sma_', 'SMA ')
            ax_price.plot(aligned_data.index, aligned_indicators[ma], 
                         label=label, alpha=0.7, linewidth=1)
    
    ax_price.set_ylabel('Price ($)', fontsize=10)
    ax_price.legend(loc='upper left', fontsize=8)
    ax_price.grid(True, alpha=0.3)
    
    chart_title = title or "Price with Regimes and Technical Indicators"
    ax_price.set_title(chart_title, fontsize=14, fontweight='bold', pad=20)
    
    # RSI subplot
    if len(axes) > 1 and 'rsi' in aligned_indicators.columns:
        ax_rsi = axes[1]
        _plot_rsi_with_regimes(ax_rsi, aligned_data.index, aligned_indicators['rsi'],
                              aligned_regime_states, regime_names, colors)
    
    # MACD subplot
    if len(axes) > 2 and all(col in aligned_indicators.columns for col in ['macd', 'macd_signal']):
        ax_macd = axes[2]
        _plot_macd_with_regimes(ax_macd, aligned_data.index, aligned_indicators,
                               aligned_regime_states, regime_names, colors)
    
    # Format x-axis
    format_financial_axis(axes[-1], aligned_data.index)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {save_path}")
    
    return fig, axes


def plot_hmm_vs_indicators_comparison(
    data: pd.DataFrame,
    indicators: pd.DataFrame,
    signals: pd.DataFrame,
    regime_states: np.ndarray,
    regime_names: Optional[Dict[int, str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create side-by-side comparison of HMM regimes vs indicator signals.
    
    Args:
        data: Price data DataFrame
        indicators: Technical indicators DataFrame  
        signals: Indicator signals DataFrame
        regime_states: HMM regime assignments
        regime_names: Optional regime name mapping
        title: Chart title
        save_path: Optional save path
        figsize: Figure size
        
    Returns:
        Tuple of (figure, list of axes)
    """
    check_plotting_available()
    setup_financial_plot_style()
    
    # Prepare data alignment
    common_index = data.index.intersection(indicators.index).intersection(signals.index)
    aligned_data = data.reindex(common_index)
    aligned_indicators = indicators.reindex(common_index)
    aligned_signals = signals.reindex(common_index)
    
    # Align regime states
    if len(regime_states) != len(common_index):
        regime_series = pd.Series(regime_states, index=data.index[:len(regime_states)])
        aligned_regime_states = regime_series.reindex(common_index).fillna(method='ffill').values
    else:
        aligned_regime_states = regime_states
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=figsize, 
                            gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.3})
    
    # Prepare regime names and colors
    unique_states = np.unique(aligned_regime_states)
    if regime_names is None:
        regime_names = {i: f"Regime {i}" for i in unique_states}
    
    colors = get_regime_colors(list(regime_names.values()))
    
    # Top left: Price with HMM regimes
    ax_hmm = axes[0, 0]
    _plot_regime_backgrounds(ax_hmm, aligned_data.index, aligned_regime_states, 
                            regime_names, colors)
    ax_hmm.plot(aligned_data.index, aligned_data['close'], 'k-', linewidth=1.5, alpha=0.8)
    ax_hmm.set_title('HMM Regime Detection', fontsize=12, fontweight='bold')
    ax_hmm.set_ylabel('Price ($)')
    ax_hmm.grid(True, alpha=0.3)
    
    # Top right: Price with indicator signals
    ax_indicators = axes[0, 1]
    ax_indicators.plot(aligned_data.index, aligned_data['close'], 'k-', linewidth=1.5, alpha=0.8)
    
    # Overlay indicator signals as colored backgrounds
    if 'composite_signal' in aligned_signals.columns:
        _plot_signal_backgrounds(ax_indicators, aligned_data.index, aligned_signals['composite_signal'])
    
    ax_indicators.set_title('Technical Indicator Signals', fontsize=12, fontweight='bold')
    ax_indicators.set_ylabel('Price ($)')
    ax_indicators.grid(True, alpha=0.3)
    
    # Bottom left: Signal strength comparison
    ax_strength = axes[1, 0]
    _plot_signal_strength_comparison(ax_strength, aligned_data.index, aligned_regime_states,
                                   aligned_signals, regime_names)
    
    # Bottom right: Correlation matrix
    ax_corr = axes[1, 1]
    _plot_signal_correlation_matrix(ax_corr, aligned_regime_states, aligned_signals, regime_names)
    
    # Main title
    chart_title = title or "HMM vs Technical Indicators Comparison"
    fig.suptitle(chart_title, fontsize=16, fontweight='bold', y=0.95)
    
    # Format axes
    for ax in [ax_hmm, ax_indicators]:
        format_financial_axis(ax, aligned_data.index)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Comparison chart saved to: {save_path}")
    
    return fig, axes


def plot_indicator_performance_dashboard(
    performance_results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create comprehensive dashboard showing indicator performance metrics.
    
    Args:
        performance_results: Results from compare_hmm_vs_indicators
        save_path: Optional save path
        figsize: Figure size
        
    Returns:
        Tuple of (figure, list of axes)
    """
    check_plotting_available()
    setup_financial_plot_style()
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    performance = performance_results['performance_comparison']
    correlation = performance_results['correlation_analysis']
    
    # Performance metrics comparison
    ax_perf = axes[0, 0]
    _plot_performance_metrics(ax_perf, performance)
    
    # Signal correlations
    ax_corr = axes[0, 1] 
    _plot_signal_correlations(ax_corr, correlation['signal_correlations'])
    
    # Agreement rates
    ax_agree = axes[0, 2]
    _plot_agreement_rates(ax_agree, correlation['agreement_analysis'])
    
    # Risk-adjusted returns
    ax_risk = axes[1, 0]
    _plot_risk_adjusted_returns(ax_risk, performance)
    
    # Win rates comparison
    ax_win = axes[1, 1]
    _plot_win_rates(ax_win, performance)
    
    # Drawdown analysis
    ax_dd = axes[1, 2]
    _plot_drawdown_analysis(ax_dd, performance)
    
    fig.suptitle('HMM vs Indicators Performance Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Dashboard saved to: {save_path}")
    
    return fig, axes


def create_regime_transition_visualization(
    data: pd.DataFrame,
    regime_states: np.ndarray,
    regime_analysis: Dict,
    indicators: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Visualize regime transitions with optional indicator context.
    
    Args:
        data: Price data DataFrame
        regime_states: Regime assignments
        regime_analysis: HMM analysis results
        indicators: Optional indicators DataFrame
        save_path: Optional save path
        figsize: Figure size
        
    Returns:
        Tuple of (figure, list of axes)
    """
    check_plotting_available()
    setup_financial_plot_style()
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                            gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Extract regime information
    regime_stats = regime_analysis['regime_statistics']['regime_stats']
    transitions = regime_analysis['regime_statistics']['transition_probabilities']
    
    # Prepare regime names
    regime_names = {i: f"Regime {i}" for i in range(len(regime_stats))}
    colors = get_regime_colors(list(regime_names.values()))
    
    # Main price plot with transitions
    ax_price = axes[0]
    _plot_regime_backgrounds(ax_price, data.index, regime_states, regime_names, colors)
    ax_price.plot(data.index, data['close'], 'k-', linewidth=1.5, alpha=0.8)
    
    # Highlight transition points
    transitions_idx = np.where(regime_states[1:] != regime_states[:-1])[0] + 1
    if len(transitions_idx) > 0:
        transition_dates = data.index[transitions_idx]
        transition_prices = data['close'].iloc[transitions_idx]
        ax_price.scatter(transition_dates, transition_prices, color='red', s=50, 
                        zorder=5, alpha=0.8, label='Regime Transitions')
    
    ax_price.set_title('Regime Transitions Analysis', fontsize=14, fontweight='bold')
    ax_price.set_ylabel('Price ($)')
    ax_price.legend()
    ax_price.grid(True, alpha=0.3)
    
    # Regime duration plot
    ax_duration = axes[1]
    _plot_regime_durations(ax_duration, data.index, regime_states, regime_names, colors)
    
    # Volatility indicator
    ax_vol = axes[2]
    if indicators is not None and 'atr' in indicators.columns:
        _plot_volatility_with_regimes(ax_vol, data.index, indicators['atr'], 
                                     regime_states, regime_names, colors)
    else:
        # Calculate simple volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)
        _plot_volatility_with_regimes(ax_vol, data.index, volatility, 
                                     regime_states, regime_names, colors)
    
    format_financial_axis(axes[-1], data.index)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Transition visualization saved to: {save_path}")
    
    return fig, axes


# Helper functions for specific plot components

def _plot_regime_backgrounds(ax, dates, regime_states, regime_names, colors):
    """Plot regime periods as colored backgrounds."""
    for i in range(len(regime_states) - 1):
        current_regime = regime_states[i]
        regime_name = regime_names.get(current_regime, f"Regime {current_regime}")
        color = colors.get(regime_name, REGIME_COLORS.get(current_regime, '#7f7f7f'))
        
        # Add background color for this period
        ax.axvspan(dates[i], dates[i + 1], alpha=0.2, color=color)
    
    # Add legend for regimes
    unique_regimes = np.unique(regime_states)
    legend_elements = []
    for regime in unique_regimes:
        regime_name = regime_names.get(regime, f"Regime {regime}")
        color = colors.get(regime_name, REGIME_COLORS.get(regime, '#7f7f7f'))
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3, 
                                           label=regime_name))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


def _plot_rsi_with_regimes(ax, dates, rsi_values, regime_states, regime_names, colors):
    """Plot RSI with regime backgrounds."""
    _plot_regime_backgrounds(ax, dates, regime_states, regime_names, colors)
    
    ax.plot(dates, rsi_values, 'purple', linewidth=1.5, label='RSI')
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    
    ax.set_ylabel('RSI')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_macd_with_regimes(ax, dates, indicators, regime_states, regime_names, colors):
    """Plot MACD with regime backgrounds."""
    _plot_regime_backgrounds(ax, dates, regime_states, regime_names, colors)
    
    ax.plot(dates, indicators['macd'], 'blue', linewidth=1.5, label='MACD')
    ax.plot(dates, indicators['macd_signal'], 'red', linewidth=1.5, label='Signal')
    
    if 'macd_histogram' in indicators.columns:
        histogram = indicators['macd_histogram']
        colors_hist = ['green' if x > 0 else 'red' for x in histogram]
        ax.bar(dates, histogram, color=colors_hist, alpha=0.6, width=1, label='Histogram')
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.set_ylabel('MACD')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_signal_backgrounds(ax, dates, signal_values):
    """Plot indicator signal backgrounds (green for buy, red for sell)."""
    for i in range(len(signal_values) - 1):
        signal = signal_values.iloc[i]
        if signal > 0.5:
            color = 'green'
            alpha = min(0.3, signal * 0.3)
        elif signal < -0.5:
            color = 'red'
            alpha = min(0.3, abs(signal) * 0.3)
        else:
            continue
        
        ax.axvspan(dates[i], dates[i + 1], alpha=alpha, color=color)


def _plot_signal_strength_comparison(ax, dates, regime_states, signals, regime_names):
    """Plot comparison of signal strengths over time."""
    # Create HMM signal strength (based on regime characteristics)
    hmm_strength = np.zeros(len(regime_states))
    for i, state in enumerate(regime_states):
        # Simple mapping: assume higher numbered states are more bullish
        hmm_strength[i] = (state - 1) * 0.5  # Scale to [-1, 1] range
    
    ax.plot(dates, hmm_strength, label='HMM Signal', linewidth=2, alpha=0.8)
    
    if 'composite_signal' in signals.columns:
        ax.plot(dates, signals['composite_signal'], label='Composite Indicators', 
               linewidth=2, alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.set_ylabel('Signal Strength')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Signal Strength Comparison', fontsize=10, fontweight='bold')


def _plot_signal_correlation_matrix(ax, regime_states, signals, regime_names):
    """Plot correlation matrix between HMM and indicator signals."""
    # Convert regime states to signals
    hmm_signals = pd.Series(regime_states, name='HMM')
    
    # Prepare correlation data
    corr_data = pd.DataFrame()
    corr_data['HMM'] = hmm_signals
    
    signal_cols = [col for col in signals.columns if col.endswith('_signal')]
    for col in signal_cols[:4]:  # Limit to first 4 signals
        if col in signals.columns:
            corr_data[col.replace('_signal', '')] = signals[col]
    
    # Calculate correlation matrix
    corr_matrix = corr_data.corr()
    
    # Create heatmap
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Set labels
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.columns)
    
    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Signal Correlation Matrix', fontsize=10, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', fontsize=8)


def _plot_performance_metrics(ax, performance):
    """Plot performance metrics comparison bar chart."""
    strategies = list(performance.keys())
    metrics = ['sharpe_ratio', 'total_return', 'max_drawdown']
    
    x = np.arange(len(strategies))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [performance[strategy].get(metric, 0) for strategy in strategies]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title(), alpha=0.8)
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Value')
    ax.set_title('Performance Metrics Comparison', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in strategies], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_signal_correlations(ax, correlations):
    """Plot signal correlations as horizontal bar chart."""
    signals = list(correlations.keys())
    values = list(correlations.values())
    
    colors = ['green' if v > 0 else 'red' for v in values]
    bars = ax.barh(range(len(signals)), values, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(signals)))
    ax.set_yticklabels([s.replace('_signal', '').replace('_', ' ').title() for s in signals])
    ax.set_xlabel('Correlation with HMM')
    ax.set_title('HMM-Indicator Correlations', fontsize=10, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax.grid(True, alpha=0.3)


def _plot_agreement_rates(ax, agreement_analysis):
    """Plot agreement rates between HMM and indicators."""
    signals = list(agreement_analysis.keys())
    agreement_rates = [agreement_analysis[s]['agreement_rate'] for s in signals]
    conflict_rates = [agreement_analysis[s]['conflict_rate'] for s in signals]
    
    x = np.arange(len(signals))
    width = 0.35
    
    ax.bar(x - width/2, agreement_rates, width, label='Agreement', color='green', alpha=0.7)
    ax.bar(x + width/2, conflict_rates, width, label='Conflict', color='red', alpha=0.7)
    
    ax.set_xlabel('Indicator')
    ax.set_ylabel('Rate')
    ax.set_title('Agreement vs Conflict Rates', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_signal', '').replace('_', ' ').title() for s in signals], 
                       rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_risk_adjusted_returns(ax, performance):
    """Plot risk-adjusted returns scatter plot."""
    strategies = list(performance.keys())
    returns = [performance[s].get('annualized_return', 0) for s in strategies]
    volatilities = [performance[s].get('volatility', 1) for s in strategies]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(strategies)))
    
    for i, (strategy, ret, vol) in enumerate(zip(strategies, returns, volatilities)):
        ax.scatter(vol, ret, c=[colors[i]], s=100, alpha=0.7, 
                  label=strategy.replace('_', ' ').title())
    
    ax.set_xlabel('Volatility (Annualized)')
    ax.set_ylabel('Return (Annualized)')
    ax.set_title('Risk-Return Profile', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_win_rates(ax, performance):
    """Plot win rates comparison."""
    strategies = list(performance.keys())
    win_rates = [performance[s].get('win_rate', 0) for s in strategies]
    
    bars = ax.bar(range(len(strategies)), win_rates, alpha=0.7, 
                 color=plt.cm.viridis(np.linspace(0, 1, len(strategies))))
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Win Rate')
    ax.set_title('Win Rates Comparison', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([s.replace('_', ' ').title() for s in strategies], rotation=45)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)


def _plot_drawdown_analysis(ax, performance):
    """Plot maximum drawdown comparison."""
    strategies = list(performance.keys())
    drawdowns = [abs(performance[s].get('max_drawdown', 0)) for s in strategies]
    
    bars = ax.bar(range(len(strategies)), drawdowns, alpha=0.7, color='red')
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Max Drawdown')
    ax.set_title('Maximum Drawdown Comparison', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([s.replace('_', ' ').title() for s in strategies], rotation=45)
    ax.grid(True, alpha=0.3)


def _plot_regime_durations(ax, dates, regime_states, regime_names, colors):
    """Plot regime duration analysis."""
    # Calculate regime durations
    durations = []
    current_regime = regime_states[0]
    duration = 1
    
    for i in range(1, len(regime_states)):
        if regime_states[i] == current_regime:
            duration += 1
        else:
            durations.append((current_regime, duration, dates[i-duration:i]))
            current_regime = regime_states[i]
            duration = 1
    
    # Add final regime
    durations.append((current_regime, duration, dates[-duration:]))
    
    # Plot duration bars
    for regime, dur, period in durations:
        regime_name = regime_names.get(regime, f"Regime {regime}")
        color = colors.get(regime_name, REGIME_COLORS.get(regime, '#7f7f7f'))
        
        start_date = period[0] if len(period) > 0 else dates[0]
        end_date = period[-1] if len(period) > 0 else dates[-1]
        
        ax.barh(regime, dur, left=start_date, color=color, alpha=0.7, 
               height=0.6, align='center')
    
    ax.set_xlabel('Duration (Days)')
    ax.set_ylabel('Regime')
    ax.set_title('Regime Duration Analysis', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)


def _plot_volatility_with_regimes(ax, dates, volatility, regime_states, regime_names, colors):
    """Plot volatility indicator with regime backgrounds."""
    _plot_regime_backgrounds(ax, dates, regime_states, regime_names, colors)
    
    ax.plot(dates, volatility, 'orange', linewidth=1.5, label='Volatility')
    ax.set_ylabel('Volatility')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Volatility by Regime', fontsize=10, fontweight='bold')