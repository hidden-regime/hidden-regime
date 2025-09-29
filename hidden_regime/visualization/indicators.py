"""
Visualization functions for technical indicators and HMM comparison.

Provides specialized plotting for comparing Hidden Markov Model regime detection
with traditional technical indicators and performance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Optional, Dict, List, Any, Tuple, Union
from datetime import datetime

from .plotting import (
    get_regime_colors, get_regime_names, format_financial_axis,
    create_regime_legend, setup_financial_plot_style
)
from ..utils.formatting import format_strategy_name


def plot_price_with_regimes_and_indicators(
    data: pd.DataFrame,
    regime_data: pd.DataFrame,
    indicators: Dict[str, pd.Series],
    price_column: str = 'close',
    regime_column: str = 'predicted_state',
    title: str = "Price, Regimes, and Technical Indicators",
    color_scheme: str = "professional"
) -> plt.Figure:
    """
    Plot price with regime background and technical indicator overlays.

    Args:
        data: DataFrame with price data
        regime_data: DataFrame with regime predictions
        indicators: Dictionary of indicator name -> indicator values
        price_column: Column name for price data
        regime_column: Column name for regime predictions
        title: Plot title
        color_scheme: Color scheme for regimes

    Returns:
        matplotlib Figure with price and indicators
    """
    n_indicators = len(indicators)
    fig, axes = plt.subplots(n_indicators + 1, 1, figsize=(14, 4 + 3 * n_indicators),
                            sharex=True)

    if n_indicators == 0:
        axes = [axes]
    elif n_indicators == 1:
        axes = axes if isinstance(axes, list) else [axes]

    # Align data on dates
    aligned_data = data.join(regime_data[regime_column], how='inner')

    # Get regime info
    n_states = int(aligned_data[regime_column].max()) + 1
    regime_colors = get_regime_colors(n_states, color_scheme)
    regime_names = get_regime_names(n_states)

    # Main price chart with regime backgrounds
    ax_price = axes[0]
    ax_price.plot(aligned_data.index, aligned_data[price_column],
                 color='black', linewidth=1.5, label='Price')

    # Add regime background coloring
    for regime in range(n_states):
        regime_mask = aligned_data[regime_column] == regime
        if regime_mask.sum() > 0:
            regime_dates = aligned_data.index[regime_mask]

            # Create background spans for continuous periods
            if len(regime_dates) > 0:
                current_start = regime_dates[0]
                for i in range(1, len(regime_dates)):
                    if (regime_dates[i] - regime_dates[i-1]).days > 1:
                        ax_price.axvspan(current_start, regime_dates[i-1],
                                       color=regime_colors[regime], alpha=0.2)
                        current_start = regime_dates[i]

                ax_price.axvspan(current_start, regime_dates[-1],
                               color=regime_colors[regime], alpha=0.2)

    ax_price.set_title(title, fontweight='bold')
    ax_price.set_ylabel(f'Price ({price_column})')
    ax_price.grid(True, alpha=0.3)
    create_regime_legend(regime_colors, regime_names, ax_price)

    # Plot each indicator in separate subplot
    for i, (indicator_name, indicator_data) in enumerate(indicators.items()):
        ax = axes[i + 1]

        # Align indicator data with our timeframe
        aligned_indicator = indicator_data.reindex(aligned_data.index).fillna(method='ffill')

        # Plot indicator
        if 'rsi' in indicator_name.lower():
            ax.plot(aligned_indicator.index, aligned_indicator.values,
                   color='purple', linewidth=1.5, label=indicator_name)
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
            ax.set_ylim(0, 100)

        elif 'macd' in indicator_name.lower():
            # MACD typically has multiple components
            if isinstance(aligned_indicator.iloc[0], (list, tuple, np.ndarray)):
                ax.plot(aligned_indicator.index, [x[0] if isinstance(x, (list, tuple, np.ndarray)) else x
                       for x in aligned_indicator.values],
                       color='blue', linewidth=1.5, label='MACD Line')
                ax.plot(aligned_indicator.index, [x[1] if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 1 else 0
                       for x in aligned_indicator.values],
                       color='red', linewidth=1.5, label='Signal Line')
            else:
                ax.plot(aligned_indicator.index, aligned_indicator.values,
                       color='blue', linewidth=1.5, label=indicator_name)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        elif 'bollinger' in indicator_name.lower():
            # Bollinger Bands typically have upper, middle, lower
            if isinstance(aligned_indicator.iloc[0], (list, tuple, np.ndarray)):
                upper = [x[0] if isinstance(x, (list, tuple, np.ndarray)) else x for x in aligned_indicator.values]
                middle = [x[1] if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 1 else x for x in aligned_indicator.values]
                lower = [x[2] if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 2 else x for x in aligned_indicator.values]

                ax.plot(aligned_indicator.index, upper, color='red', linewidth=1, alpha=0.7, label='Upper Band')
                ax.plot(aligned_indicator.index, middle, color='blue', linewidth=1.5, label='Middle Band')
                ax.plot(aligned_indicator.index, lower, color='red', linewidth=1, alpha=0.7, label='Lower Band')
                ax.fill_between(aligned_indicator.index, upper, lower, alpha=0.1, color='gray')
            else:
                ax.plot(aligned_indicator.index, aligned_indicator.values,
                       color='blue', linewidth=1.5, label=indicator_name)

        else:
            # Generic indicator plotting
            ax.plot(aligned_indicator.index, aligned_indicator.values,
                   linewidth=1.5, label=indicator_name)

        # Add regime backgrounds to indicator plots too
        for regime in range(n_states):
            regime_mask = aligned_data[regime_column] == regime
            if regime_mask.sum() > 0:
                regime_dates = aligned_data.index[regime_mask]

                if len(regime_dates) > 0:
                    current_start = regime_dates[0]
                    for j in range(1, len(regime_dates)):
                        if (regime_dates[j] - regime_dates[j-1]).days > 1:
                            ax.axvspan(current_start, regime_dates[j-1],
                                     color=regime_colors[regime], alpha=0.1)
                            current_start = regime_dates[j]

                    ax.axvspan(current_start, regime_dates[-1],
                             color=regime_colors[regime], alpha=0.1)

        ax.set_ylabel(indicator_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    # Format x-axis for bottom plot
    format_financial_axis(axes[-1])
    axes[-1].set_xlabel('Date')

    plt.tight_layout()
    return fig


def plot_hmm_vs_indicators_comparison(
    data: pd.DataFrame,
    regime_data: pd.DataFrame,
    indicator_signals: Dict[str, pd.Series],
    price_column: str = 'close',
    regime_column: str = 'predicted_state',
    title: str = "HMM vs Technical Indicators Performance",
    lookback_window: int = 20,
    color_scheme: str = "professional"
) -> plt.Figure:
    """
    Compare HMM regime detection against technical indicator signals.

    Args:
        data: DataFrame with price data
        regime_data: DataFrame with regime predictions
        indicator_signals: Dictionary of indicator name -> buy/sell signals (-1, 0, 1)
        price_column: Column name for price data
        regime_column: Column name for regime predictions
        title: Plot title
        lookback_window: Days to look ahead for return calculation

    Returns:
        matplotlib Figure comparing performance
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Align all data
    aligned_data = data.join(regime_data[regime_column], how='inner')

    # Calculate forward returns for performance evaluation
    aligned_data['forward_return'] = aligned_data[price_column].pct_change(lookback_window).shift(-lookback_window)

    # Get regime info
    n_states = int(aligned_data[regime_column].max()) + 1
    regime_colors = get_regime_colors(n_states, color_scheme)
    regime_names = get_regime_names(n_states)

    # Plot 1: Regime-based returns
    regime_returns = {}
    for regime in range(n_states):
        mask = aligned_data[regime_column] == regime
        if mask.sum() > 1:
            returns = aligned_data.loc[mask, 'forward_return'].dropna()
            regime_returns[regime] = returns.mean()

    if regime_returns:
        regimes = list(regime_returns.keys())
        returns = list(regime_returns.values())
        bars = axes[0].bar([regime_names[r] for r in regimes],
                          [r * 100 for r in returns],
                          color=[regime_colors[r] for r in regimes], alpha=0.7)
        axes[0].set_title(f'HMM Regime Forward Returns ({lookback_window} days)')
        axes[0].set_ylabel('Average Return (%)')
        axes[0].grid(True, alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

    # Plot 2: Indicator-based returns
    indicator_returns = {}
    for indicator_name, signals in indicator_signals.items():
        aligned_signals = signals.reindex(aligned_data.index).fillna(0)

        # Calculate returns for each signal type
        signal_returns = {}
        for signal_value in [-1, 0, 1]:
            mask = aligned_signals == signal_value
            if mask.sum() > 1:
                returns = aligned_data.loc[mask, 'forward_return'].dropna()
                if len(returns) > 0:
                    signal_returns[signal_value] = returns.mean()

        indicator_returns[indicator_name] = signal_returns

    # Plot best performing indicator signals
    if indicator_returns:
        best_signals = {}
        for ind_name, signals in indicator_returns.items():
            if signals:
                best_signal = max(signals.keys(), key=lambda x: signals[x])
                best_signals[ind_name] = signals[best_signal]

        if best_signals:
            indicators = list(best_signals.keys())
            returns = [r * 100 for r in best_signals.values()]
            axes[1].bar(indicators, returns, alpha=0.7, color='steelblue')
            axes[1].set_title(f'Best Indicator Signals Forward Returns ({lookback_window} days)')
            axes[1].set_ylabel('Average Return (%)')
            axes[1].grid(True, alpha=0.3)
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

    # Plot 3: Cumulative performance comparison
    aligned_data['returns'] = aligned_data[price_column].pct_change()

    # HMM-based strategy: go long in bull regime, short in bear, neutral in sideways
    hmm_positions = np.zeros(len(aligned_data))
    for i, regime in enumerate(aligned_data[regime_column]):
        if regime == n_states - 1:  # Highest regime number = bull
            hmm_positions[i] = 1
        elif regime == 0:  # Lowest regime number = bear
            hmm_positions[i] = -1
        else:  # Middle regimes = neutral
            hmm_positions[i] = 0

    hmm_strategy_returns = aligned_data['returns'] * hmm_positions
    hmm_cumulative = (1 + hmm_strategy_returns).cumprod()

    axes[2].plot(aligned_data.index, hmm_cumulative, label='HMM Strategy', linewidth=2)

    # Buy and hold
    buy_hold_cumulative = (1 + aligned_data['returns']).cumprod()
    axes[2].plot(aligned_data.index, buy_hold_cumulative, label='Buy & Hold', linewidth=2, alpha=0.7)

    # Best indicator strategy
    if indicator_signals:
        best_indicator = max(best_signals.keys(), key=lambda x: best_signals[x]) if best_signals else list(indicator_signals.keys())[0]
        indicator_positions = indicator_signals[best_indicator].reindex(aligned_data.index).fillna(0)
        indicator_strategy_returns = aligned_data['returns'] * indicator_positions
        indicator_cumulative = (1 + indicator_strategy_returns).cumprod()
        axes[2].plot(aligned_data.index, indicator_cumulative,
                    label=f'{best_indicator} Strategy', linewidth=2, alpha=0.7)

    axes[2].set_title('Cumulative Performance Comparison')
    axes[2].set_ylabel('Cumulative Return')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    format_financial_axis(axes[2])

    # Plot 4: Risk-adjusted performance metrics
    strategies = {'Buy & Hold': aligned_data['returns']}

    if len(hmm_strategy_returns.dropna()) > 0:
        strategies['HMM Strategy'] = hmm_strategy_returns

    if indicator_signals and best_signals:
        strategies[f'{best_indicator}'] = indicator_strategy_returns

    metrics = {}
    for strategy_name, strategy_returns in strategies.items():
        clean_returns = strategy_returns.dropna()
        if len(clean_returns) > 0:
            annual_return = clean_returns.mean() * 252
            annual_vol = clean_returns.std() * np.sqrt(252)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0
            max_dd = calculate_max_drawdown(clean_returns)

            metrics[strategy_name] = {
                'Annual Return': annual_return,
                'Annual Volatility': annual_vol,
                'Sharpe Ratio': sharpe,
                'Max Drawdown': max_dd
            }

    if metrics:
        metric_names = ['Sharpe Ratio']  # Focus on Sharpe ratio for the bar chart
        x_pos = np.arange(len(strategies))

        for i, metric in enumerate(metric_names):
            values = [metrics[strategy].get(metric, 0) for strategy in strategies.keys()]
            axes[3].bar(x_pos, values, alpha=0.7, label=metric)

        axes[3].set_title('Risk-Adjusted Performance (Sharpe Ratio)')
        axes[3].set_ylabel('Sharpe Ratio')
        axes[3].set_xticks(x_pos)
        axes[3].set_xticklabels([format_strategy_name(s) for s in strategies.keys()], rotation=45)
        axes[3].grid(True, alpha=0.3)
        axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from returns series."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def plot_indicator_performance_dashboard(
    data: pd.DataFrame,
    indicator_signals: Dict[str, pd.Series],
    price_column: str = 'close',
    title: str = "Technical Indicators Performance Dashboard"
) -> plt.Figure:
    """
    Create comprehensive dashboard for technical indicator performance.

    Args:
        data: DataFrame with price data
        indicator_signals: Dictionary of indicator signals
        price_column: Column name for price data
        title: Dashboard title

    Returns:
        matplotlib Figure with indicator performance metrics
    """
    n_indicators = len(indicator_signals)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Calculate returns
    data = data.copy()
    data['returns'] = data[price_column].pct_change()

    # Performance metrics for each indicator
    indicator_metrics = {}
    for indicator_name, signals in indicator_signals.items():
        aligned_signals = signals.reindex(data.index).fillna(0)

        # Calculate strategy returns
        strategy_returns = data['returns'] * aligned_signals
        clean_returns = strategy_returns.dropna()

        if len(clean_returns) > 0:
            metrics = {
                'total_return': (1 + clean_returns).prod() - 1,
                'annual_return': clean_returns.mean() * 252,
                'annual_volatility': clean_returns.std() * np.sqrt(252),
                'sharpe_ratio': (clean_returns.mean() * 252) / (clean_returns.std() * np.sqrt(252)) if clean_returns.std() > 0 else 0,
                'max_drawdown': calculate_max_drawdown(clean_returns),
                'win_rate': (clean_returns > 0).mean(),
                'num_trades': (aligned_signals.diff() != 0).sum()
            }
            indicator_metrics[indicator_name] = metrics

    if not indicator_metrics:
        axes[0].text(0.5, 0.5, 'No valid indicator data', ha='center', va='center')
        return fig

    # Plot 1: Total Returns
    indicators = list(indicator_metrics.keys())
    formatted_indicators = [format_strategy_name(ind) for ind in indicators]
    total_returns = [indicator_metrics[ind]['total_return'] * 100 for ind in indicators]

    bars1 = axes[0].bar(formatted_indicators, total_returns, alpha=0.7, color='steelblue')
    axes[0].set_title('Total Returns by Indicator')
    axes[0].set_ylabel('Total Return (%)')
    axes[0].grid(True, alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

    # Add buy & hold comparison
    buy_hold_return = ((1 + data['returns']).prod() - 1) * 100
    axes[0].axhline(y=buy_hold_return, color='red', linestyle='--',
                   label=f'Buy & Hold: {buy_hold_return:.1f}%')
    axes[0].legend()

    # Plot 2: Sharpe Ratios
    sharpe_ratios = [indicator_metrics[ind]['sharpe_ratio'] for ind in indicators]

    bars2 = axes[1].bar(formatted_indicators, sharpe_ratios, alpha=0.7, color='green')
    axes[1].set_title('Sharpe Ratios by Indicator')
    axes[1].set_ylabel('Sharpe Ratio')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

    # Plot 3: Max Drawdown
    max_drawdowns = [indicator_metrics[ind]['max_drawdown'] * 100 for ind in indicators]

    bars3 = axes[2].bar(formatted_indicators, max_drawdowns, alpha=0.7, color='red')
    axes[2].set_title('Maximum Drawdown by Indicator')
    axes[2].set_ylabel('Max Drawdown (%)')
    axes[2].grid(True, alpha=0.3)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)

    # Plot 4: Win Rate vs Number of Trades
    win_rates = [indicator_metrics[ind]['win_rate'] * 100 for ind in indicators]
    num_trades = [indicator_metrics[ind]['num_trades'] for ind in indicators]

    scatter = axes[3].scatter(num_trades, win_rates, alpha=0.7, s=100, c='purple')

    # Annotate points
    for i, indicator in enumerate(indicators):
        axes[3].annotate(format_strategy_name(indicator), (num_trades[i], win_rates[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

    axes[3].set_title('Win Rate vs Number of Trades')
    axes[3].set_xlabel('Number of Trades')
    axes[3].set_ylabel('Win Rate (%)')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim(0, 100)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def create_regime_transition_visualization(
    regime_data: pd.DataFrame,
    regime_column: str = 'predicted_state',
    confidence_column: str = 'confidence',
    title: str = "Regime Transition Analysis",
    color_scheme: str = "professional"
) -> plt.Figure:
    """
    Create visualization showing regime transitions and their characteristics.

    Args:
        regime_data: DataFrame with regime predictions
        regime_column: Column name for regime predictions
        confidence_column: Column name for confidence scores
        title: Plot title

    Returns:
        matplotlib Figure with transition analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    n_states = int(regime_data[regime_column].max()) + 1
    regime_colors = get_regime_colors(n_states, color_scheme)
    regime_names = get_regime_names(n_states)

    # Plot 1: Transition Matrix Heatmap
    transition_matrix = np.zeros((n_states, n_states))

    for i in range(len(regime_data) - 1):
        current_state = int(regime_data[regime_column].iloc[i])
        next_state = int(regime_data[regime_column].iloc[i + 1])
        transition_matrix[current_state, next_state] += 1

    # Normalize by row sums to get probabilities
    row_sums = transition_matrix.sum(axis=1)
    transition_probs = transition_matrix / row_sums[:, np.newaxis]
    transition_probs = np.nan_to_num(transition_probs)  # Handle division by zero

    im1 = axes[0].imshow(transition_probs, cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title('Regime Transition Probabilities')
    axes[0].set_xlabel('To Regime')
    axes[0].set_ylabel('From Regime')
    axes[0].set_xticks(range(n_states))
    axes[0].set_xticklabels(regime_names)
    axes[0].set_yticks(range(n_states))
    axes[0].set_yticklabels(regime_names)

    # Add text annotations
    for i in range(n_states):
        for j in range(n_states):
            text = axes[0].text(j, i, f'{transition_probs[i, j]:.2f}',
                              ha="center", va="center", color="black" if transition_probs[i, j] < 0.5 else "white")

    plt.colorbar(im1, ax=axes[0])

    # Plot 2: Regime Persistence (diagonal elements)
    persistence = np.diag(transition_probs)
    bars2 = axes[1].bar(regime_names, persistence, alpha=0.7, color=regime_colors)
    axes[1].set_title('Regime Persistence')
    axes[1].set_ylabel('Probability of Staying in Regime')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

    # Plot 3: Regime Duration Distribution
    regime_durations = {}
    current_regime = regime_data[regime_column].iloc[0]
    duration = 1

    for i in range(1, len(regime_data)):
        if regime_data[regime_column].iloc[i] == current_regime:
            duration += 1
        else:
            if current_regime not in regime_durations:
                regime_durations[current_regime] = []
            regime_durations[current_regime].append(duration)
            current_regime = regime_data[regime_column].iloc[i]
            duration = 1

    # Add final duration
    if current_regime not in regime_durations:
        regime_durations[current_regime] = []
    regime_durations[current_regime].append(duration)

    # Box plot of durations
    duration_data = []
    duration_labels = []
    for regime in range(n_states):
        if regime in regime_durations and regime_durations[regime]:
            duration_data.append(regime_durations[regime])
            duration_labels.append(regime_names[regime])

    if duration_data:
        bp = axes[2].boxplot(duration_data, labels=duration_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], [regime_colors[i] for i in range(len(duration_data))]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[2].set_title('Regime Duration Distribution')
        axes[2].set_ylabel('Duration (Days)')
        axes[2].grid(True, alpha=0.3)
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)

    # Plot 4: Confidence by Regime
    if confidence_column in regime_data.columns:
        confidence_by_regime = []
        confidence_labels = []

        for regime in range(n_states):
            regime_mask = regime_data[regime_column] == regime
            if regime_mask.sum() > 0:
                confidence_vals = regime_data.loc[regime_mask, confidence_column]
                confidence_by_regime.append(confidence_vals.values)
                confidence_labels.append(regime_names[regime])

        if confidence_by_regime:
            bp2 = axes[3].boxplot(confidence_by_regime, labels=confidence_labels, patch_artist=True)
            for patch, color in zip(bp2['boxes'], [regime_colors[i] for i in range(len(confidence_by_regime))]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            axes[3].set_title('Confidence by Regime')
            axes[3].set_ylabel('Confidence Score')
            axes[3].set_ylim(0, 1)
            axes[3].grid(True, alpha=0.3)
            plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig