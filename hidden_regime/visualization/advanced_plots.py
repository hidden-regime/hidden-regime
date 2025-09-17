"""
Advanced plotting suite for comprehensive regime analysis visualization.

Provides sophisticated visualization classes for regime analysis, performance metrics,
indicator comparisons, and interactive dashboards with professional styling.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
from datetime import datetime, timedelta

from .plotting import setup_financial_plot_style, get_regime_colors, create_regime_legend
from ..config.analysis import FinancialAnalysisConfig


class RegimePlotter:
    """
    Advanced regime visualization class with comprehensive plotting capabilities.
    
    Provides sophisticated regime analysis visualizations including state sequences,
    confidence bands, transition analysis, and duration statistics.
    """
    
    def __init__(self, theme: str = 'financial', figsize: Tuple[int, int] = (16, 12)):
        """
        Initialize regime plotter with styling configuration.
        
        Args:
            theme: Plotting theme ('financial', 'dark', 'light', 'academic')
            figsize: Default figure size for plots
        """
        self.theme = theme
        self.figsize = figsize
        self.regime_colors = get_regime_colors()
        
        # Enhanced color schemes
        self.color_schemes = {
            'financial': {
                'bull': '#2E8B57',      # Sea Green
                'bear': '#DC143C',      # Crimson
                'sideways': '#708090',  # Slate Gray
                'crisis': '#8B0000',    # Dark Red
                'euphoric': '#9370DB'   # Medium Purple
            },
            'academic': {
                'bull': '#1f77b4',     # Blue
                'bear': '#ff7f0e',     # Orange
                'sideways': '#2ca02c', # Green
                'crisis': '#d62728',   # Red
                'euphoric': '#9467bd'  # Purple
            },
            'pastel': {
                'bull': '#87CEEB',     # Sky Blue
                'bear': '#F0A0A0',     # Light Red
                'sideways': '#D3D3D3', # Light Gray
                'crisis': '#FFB6C1',   # Light Pink
                'euphoric': '#DDA0DD'  # Plum
            }
        }
        
        setup_financial_plot_style()
    
    def plot_comprehensive_regime_analysis(
        self, 
        analysis_results: pd.DataFrame,
        raw_data: Optional[pd.DataFrame] = None,
        indicators: Optional[Dict[str, pd.DataFrame]] = None,
        performance_metrics: Optional[Dict[str, Any]] = None
    ) -> plt.Figure:
        """
        Create comprehensive regime analysis visualization dashboard.
        
        Args:
            analysis_results: Analysis results with regime predictions
            raw_data: Optional price/volume data
            indicators: Optional technical indicators
            performance_metrics: Optional performance analysis results
            
        Returns:
            matplotlib Figure with comprehensive dashboard
        """
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main regime sequence plot
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_regime_sequence_with_confidence(ax_main, analysis_results, raw_data)
        
        # Regime distribution pie chart
        ax_dist = fig.add_subplot(gs[1, 0])
        self._plot_regime_distribution(ax_dist, analysis_results)
        
        # Confidence analysis
        ax_conf = fig.add_subplot(gs[1, 1])
        self._plot_confidence_analysis(ax_conf, analysis_results)
        
        # Duration analysis
        ax_dur = fig.add_subplot(gs[1, 2])
        self._plot_duration_analysis(ax_dur, analysis_results)
        
        # Transition heatmap
        ax_trans = fig.add_subplot(gs[2, 0])
        if performance_metrics and 'transition_analysis' in performance_metrics:
            self._plot_transition_heatmap(ax_trans, performance_metrics['transition_analysis'])
        else:
            self._plot_empty_placeholder(ax_trans, "Transition Analysis\n(No data available)")
        
        # Performance by regime
        ax_perf = fig.add_subplot(gs[2, 1])
        if performance_metrics and 'regime_performance' in performance_metrics:
            self._plot_regime_performance_bars(ax_perf, performance_metrics['regime_performance'])
        else:
            self._plot_empty_placeholder(ax_perf, "Regime Performance\n(No data available)")
        
        # Risk metrics by regime
        ax_risk = fig.add_subplot(gs[2, 2])
        if performance_metrics and 'risk_metrics' in performance_metrics:
            self._plot_risk_metrics_radar(ax_risk, performance_metrics['risk_metrics'])
        else:
            self._plot_empty_placeholder(ax_risk, "Risk Metrics\n(No data available)")
        
        # Indicator agreement (if available)
        ax_ind = fig.add_subplot(gs[3, :])
        if indicators:
            self._plot_indicator_agreement_timeline(ax_ind, analysis_results, indicators)
        else:
            self._plot_empty_placeholder(ax_ind, "Indicator Agreement Timeline\n(No indicator data available)")
        
        fig.suptitle('Comprehensive Regime Analysis Dashboard', fontsize=20, y=0.98)
        return fig
    
    def _plot_regime_sequence_with_confidence(
        self, 
        ax: plt.Axes, 
        analysis_results: pd.DataFrame,
        raw_data: Optional[pd.DataFrame] = None
    ):
        """Plot regime sequence with confidence bands and optional price overlay."""
        dates = analysis_results.index if hasattr(analysis_results.index, 'to_pydatetime') else range(len(analysis_results))
        
        # Plot price data if available
        if raw_data is not None and 'close' in raw_data.columns:
            ax_price = ax.twinx()
            common_index = analysis_results.index.intersection(raw_data.index)
            price_data = raw_data.loc[common_index, 'close']
            ax_price.plot(common_index, price_data, 'k-', alpha=0.3, linewidth=1, label='Price')
            ax_price.set_ylabel('Price', rotation=270, labelpad=15)
        
        # Plot regime sequence as colored background
        regime_types = analysis_results.get('regime_type', analysis_results['predicted_state'].map({
            0: 'bear', 1: 'sideways', 2: 'bull'
        }))
        
        for i, (idx, row) in enumerate(analysis_results.iterrows()):
            regime = regime_types.iloc[i] if hasattr(regime_types, 'iloc') else regime_types[i]
            confidence = row['confidence']
            
            color = self.color_schemes[self.theme].get(regime, '#808080')
            alpha = 0.3 + (confidence * 0.5)  # Scale alpha by confidence
            
            x_pos = dates[i] if hasattr(dates, '__getitem__') else i
            width = 1 if isinstance(dates, range) else (dates[1] - dates[0] if len(dates) > 1 else timedelta(days=1))
            
            ax.add_patch(Rectangle((x_pos, -0.5), width, 1, 
                                 facecolor=color, alpha=alpha, edgecolor='none'))
        
        # Plot confidence line
        conf_line = ax.plot(dates, analysis_results['confidence'], 'b-', linewidth=2, label='Confidence')[0]
        ax.fill_between(dates, 0, analysis_results['confidence'], alpha=0.2, color='blue')
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('Confidence')
        ax.set_title('Regime Sequence with Confidence')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis for dates
        if hasattr(analysis_results.index, 'to_pydatetime'):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_regime_distribution(self, ax: plt.Axes, analysis_results: pd.DataFrame):
        """Plot regime distribution as pie chart."""
        if 'regime_type' in analysis_results.columns:
            regime_counts = analysis_results['regime_type'].value_counts()
        else:
            regime_counts = analysis_results['predicted_state'].value_counts()
            regime_counts.index = [f'State {i}' for i in regime_counts.index]
        
        colors = [self.color_schemes[self.theme].get(regime, '#808080') for regime in regime_counts.index]
        
        wedges, texts, autotexts = ax.pie(regime_counts.values, labels=regime_counts.index, 
                                         colors=colors, autopct='%1.1f%%', startangle=90)
        
        ax.set_title('Regime Distribution')
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _plot_confidence_analysis(self, ax: plt.Axes, analysis_results: pd.DataFrame):
        """Plot confidence distribution and statistics."""
        confidence = analysis_results['confidence']
        
        # Histogram
        ax.hist(confidence, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add statistics
        mean_conf = confidence.mean()
        median_conf = confidence.median()
        
        ax.axvline(mean_conf, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_conf:.3f}')
        ax.axvline(median_conf, color='green', linestyle='--', linewidth=2, label=f'Median: {median_conf:.3f}')
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('Confidence Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_duration_analysis(self, ax: plt.Axes, analysis_results: pd.DataFrame):
        """Plot regime duration analysis."""
        if 'days_in_regime' not in analysis_results.columns:
            self._plot_empty_placeholder(ax, "Duration Analysis\n(No duration data)")
            return
        
        # Calculate regime episode durations
        episodes = []
        current_regime = None
        episode_start = 0
        
        for i, (_, row) in enumerate(analysis_results.iterrows()):
            regime = row.get('regime_type', f"State_{row['predicted_state']}")
            
            if regime != current_regime:
                if current_regime is not None:
                    episodes.append({
                        'regime': current_regime,
                        'duration': i - episode_start,
                        'start': episode_start,
                        'end': i
                    })
                current_regime = regime
                episode_start = i
        
        # Add final episode
        if current_regime is not None:
            episodes.append({
                'regime': current_regime,
                'duration': len(analysis_results) - episode_start,
                'start': episode_start,
                'end': len(analysis_results)
            })
        
        if episodes:
            episode_df = pd.DataFrame(episodes)
            regime_durations = episode_df.groupby('regime')['duration'].agg(['mean', 'std', 'count'])
            
            x_pos = np.arange(len(regime_durations))
            colors = [self.color_schemes[self.theme].get(regime, '#808080') 
                     for regime in regime_durations.index]
            
            bars = ax.bar(x_pos, regime_durations['mean'], yerr=regime_durations['std'], 
                         color=colors, alpha=0.7, capsize=5)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(regime_durations.index, rotation=45)
            ax.set_ylabel('Average Duration (Days)')
            ax.set_title('Average Regime Duration')
            ax.grid(True, alpha=0.3)
            
            # Add count annotations
            for bar, count in zip(bars, regime_durations['count']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (regime_durations['std'].max() * 0.1),
                       f'n={count}', ha='center', va='bottom', fontsize=8)
        else:
            self._plot_empty_placeholder(ax, "Duration Analysis\n(No episodes found)")
    
    def _plot_transition_heatmap(self, ax: plt.Axes, transition_analysis: Dict[str, Any]):
        """Plot regime transition probability heatmap."""
        if 'transition_matrix' not in transition_analysis:
            self._plot_empty_placeholder(ax, "Transition Matrix\n(No data available)")
            return
        
        trans_matrix = pd.DataFrame(transition_analysis['transition_matrix'])
        
        # Create heatmap
        im = ax.imshow(trans_matrix.values, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(len(trans_matrix.index)):
            for j in range(len(trans_matrix.columns)):
                text = ax.text(j, i, f'{trans_matrix.iloc[i, j]:.2f}',
                             ha='center', va='center', color='white' if trans_matrix.iloc[i, j] > 0.5 else 'black')
        
        ax.set_xticks(range(len(trans_matrix.columns)))
        ax.set_yticks(range(len(trans_matrix.index)))
        ax.set_xticklabels(trans_matrix.columns)
        ax.set_yticklabels(trans_matrix.index)
        ax.set_xlabel('To Regime')
        ax.set_ylabel('From Regime')
        ax.set_title('Transition Probabilities')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _plot_regime_performance_bars(self, ax: plt.Axes, regime_performance: Dict[str, Any]):
        """Plot performance metrics by regime."""
        if not regime_performance or 'regime_stats' not in regime_performance:
            self._plot_empty_placeholder(ax, "Regime Performance\n(No data available)")
            return
        
        regime_stats = regime_performance['regime_stats']
        regimes = list(regime_stats.keys())
        
        # Extract metrics (using mock data if not available)
        returns = [regime_stats[r].get('mean_return', 0.0) for r in regimes]
        volatilities = [regime_stats[r].get('volatility', 0.0) for r in regimes]
        
        x = np.arange(len(regimes))
        width = 0.35
        
        colors = [self.color_schemes[self.theme].get(regime, '#808080') for regime in regimes]
        
        bars1 = ax.bar(x - width/2, returns, width, label='Return', color=colors, alpha=0.7)
        bars2 = ax.bar(x + width/2, volatilities, width, label='Volatility', color=colors, alpha=0.4)
        
        ax.set_xlabel('Regime')
        ax.set_ylabel('Value')
        ax.set_title('Return vs Volatility by Regime')
        ax.set_xticks(x)
        ax.set_xticklabels(regimes)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_metrics_radar(self, ax: plt.Axes, risk_metrics: Dict[str, Any]):
        """Plot risk metrics as radar chart."""
        # Placeholder for radar chart - would need more complex implementation
        self._plot_empty_placeholder(ax, "Risk Metrics Radar\n(Placeholder)")
    
    def _plot_indicator_agreement_timeline(
        self, 
        ax: plt.Axes, 
        analysis_results: pd.DataFrame,
        indicators: Dict[str, pd.DataFrame]
    ):
        """Plot indicator agreement over time."""
        dates = analysis_results.index if hasattr(analysis_results.index, 'to_pydatetime') else range(len(analysis_results))
        
        # Plot agreement for each indicator
        for i, (ind_name, ind_data) in enumerate(indicators.items()):
            if f'{ind_name}_agreement' in analysis_results.columns:
                agreement = analysis_results[f'{ind_name}_agreement']
                ax.plot(dates, agreement, label=f'{ind_name.upper()} Agreement', 
                       linewidth=2, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good Agreement')
        ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Poor Agreement')
        
        ax.set_ylabel('Agreement')
        ax.set_title('Indicator-Regime Agreement Timeline')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        if hasattr(analysis_results.index, 'to_pydatetime'):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_empty_placeholder(self, ax: plt.Axes, message: str):
        """Plot empty placeholder with message."""
        ax.text(0.5, 0.5, message, ha='center', va='center', 
               transform=ax.transAxes, fontsize=12, color='gray',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        ax.set_xticks([])
        ax.set_yticks([])


class PerformancePlotter:
    """
    Performance analysis visualization class.
    
    Provides comprehensive performance analysis plots including returns analysis,
    risk metrics, drawdown analysis, and comparative performance studies.
    """
    
    def __init__(self, theme: str = 'financial'):
        """Initialize performance plotter."""
        self.theme = theme
        setup_financial_plot_style()
    
    def plot_performance_dashboard(
        self, 
        performance_metrics: Dict[str, Any],
        benchmark_data: Optional[Dict[str, Any]] = None
    ) -> plt.Figure:
        """
        Create comprehensive performance dashboard.
        
        Args:
            performance_metrics: Performance analysis results
            benchmark_data: Optional benchmark comparison data
            
        Returns:
            matplotlib Figure with performance dashboard
        """
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Returns analysis
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_returns_analysis(ax1, performance_metrics)
        
        # Risk metrics
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_risk_metrics(ax2, performance_metrics)
        
        # Drawdown analysis
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_drawdown_analysis(ax3, performance_metrics)
        
        # Rolling metrics
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_rolling_metrics(ax4, performance_metrics)
        
        # Performance attribution
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_performance_attribution(ax5, performance_metrics)
        
        # Regime contribution
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_regime_contribution(ax6, performance_metrics)
        
        # Statistical summary
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_statistical_summary(ax7, performance_metrics)
        
        fig.suptitle('Performance Analysis Dashboard', fontsize=20, y=0.98)
        return fig
    
    def _plot_returns_analysis(self, ax: plt.Axes, metrics: Dict[str, Any]):
        """Plot returns analysis."""
        # Placeholder implementation
        ax.text(0.5, 0.5, 'Returns Analysis\n(Implementation needed)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Returns Analysis')
    
    def _plot_risk_metrics(self, ax: plt.Axes, metrics: Dict[str, Any]):
        """Plot risk metrics."""
        ax.text(0.5, 0.5, 'Risk Metrics\n(Implementation needed)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Risk Metrics')
    
    def _plot_drawdown_analysis(self, ax: plt.Axes, metrics: Dict[str, Any]):
        """Plot drawdown analysis."""
        ax.text(0.5, 0.5, 'Drawdown Analysis\n(Implementation needed)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Drawdown Analysis')
    
    def _plot_rolling_metrics(self, ax: plt.Axes, metrics: Dict[str, Any]):
        """Plot rolling performance metrics."""
        ax.text(0.5, 0.5, 'Rolling Metrics\n(Implementation needed)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Rolling Metrics')
    
    def _plot_performance_attribution(self, ax: plt.Axes, metrics: Dict[str, Any]):
        """Plot performance attribution analysis."""
        ax.text(0.5, 0.5, 'Performance Attribution\n(Implementation needed)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Performance Attribution')
    
    def _plot_regime_contribution(self, ax: plt.Axes, metrics: Dict[str, Any]):
        """Plot regime contribution to performance."""
        ax.text(0.5, 0.5, 'Regime Contribution\n(Implementation needed)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Regime Contribution')
    
    def _plot_statistical_summary(self, ax: plt.Axes, metrics: Dict[str, Any]):
        """Plot statistical summary table."""
        ax.text(0.5, 0.5, 'Statistical Summary\n(Implementation needed)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Statistical Summary')
        ax.set_xticks([])
        ax.set_yticks([])


class ComparisonPlotter:
    """
    Comparison visualization class for regime vs indicators and multi-asset analysis.
    """
    
    def __init__(self, theme: str = 'financial'):
        """Initialize comparison plotter."""
        self.theme = theme
        setup_financial_plot_style()
    
    def plot_regime_vs_indicators_dashboard(
        self, 
        comparison_results: Dict[str, Any]
    ) -> plt.Figure:
        """
        Create comprehensive regime vs indicators comparison dashboard.
        
        Args:
            comparison_results: Results from indicator comparison analysis
            
        Returns:
            matplotlib Figure with comparison dashboard
        """
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Overall performance comparison
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_performance_comparison_overview(ax1, comparison_results)
        
        # Individual indicator analysis
        if 'indicator_analysis' in comparison_results:
            indicators = list(comparison_results['indicator_analysis'].keys())[:6]  # Limit to 6
            
            for i, indicator in enumerate(indicators):
                row = 1 + (i // 3)
                col = i % 3
                if row < 4:  # Don't exceed grid
                    ax = fig.add_subplot(gs[row, col])
                    self._plot_single_indicator_analysis(ax, comparison_results['indicator_analysis'][indicator])
        
        fig.suptitle('Regime vs Indicators Comparison Dashboard', fontsize=20, y=0.98)
        return fig
    
    def _plot_performance_comparison_overview(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot overview comparison of all indicators."""
        if 'indicator_analysis' not in results:
            ax.text(0.5, 0.5, 'No indicator analysis available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        indicators = list(results['indicator_analysis'].keys())
        correlations = []
        accuracies = []
        composite_scores = []
        
        for indicator in indicators:
            analysis = results['indicator_analysis'][indicator]
            if 'agreement_analysis' in analysis:
                corr = analysis['agreement_analysis']['correlation']['pearson_correlation']
                acc = analysis['agreement_analysis']['classification_metrics']['accuracy']
                comp = analysis['performance_rating']['composite_score']
                
                correlations.append(abs(corr))
                accuracies.append(acc)
                composite_scores.append(comp)
        
        x = np.arange(len(indicators))
        width = 0.25
        
        ax.bar(x - width, correlations, width, label='Abs. Correlation', alpha=0.8)
        ax.bar(x, accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x + width, composite_scores, width, label='Composite Score', alpha=0.8)
        
        ax.set_xlabel('Indicators')
        ax.set_ylabel('Score')
        ax.set_title('Indicator Performance Comparison Overview')
        ax.set_xticks(x)
        ax.set_xticklabels([ind.upper() for ind in indicators])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_single_indicator_analysis(self, ax: plt.Axes, indicator_analysis: Dict[str, Any]):
        """Plot analysis for a single indicator."""
        indicator_name = indicator_analysis.get('indicator_name', 'Unknown')
        
        if 'agreement_analysis' in indicator_analysis:
            agreement = indicator_analysis['agreement_analysis']
            
            # Create simple metrics visualization
            metrics = {
                'Correlation': abs(agreement['correlation']['pearson_correlation']),
                'Accuracy': agreement['classification_metrics']['accuracy'],
                'F1 Score': agreement['classification_metrics']['f1_score'],
                'Weighted Agr.': agreement['confidence_weighted_agreement']['weighted_agreement']
            }
            
            values = list(metrics.values())
            labels = list(metrics.keys())
            
            bars = ax.bar(labels, values, alpha=0.7)
            
            # Color bars by performance
            for bar, value in zip(bars, values):
                if value >= 0.7:
                    bar.set_color('green')
                elif value >= 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            ax.set_title(f'{indicator_name.upper()} Analysis')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, f'{indicator_name}\nNo analysis data', 
                   ha='center', va='center', transform=ax.transAxes)


class InteractivePlotter:
    """
    Interactive plotting capabilities (placeholder for future plotly integration).
    """
    
    def __init__(self):
        """Initialize interactive plotter."""
        self.available = False
        
        try:
            import plotly.graph_objects as go
            import plotly.subplots as sp
            self.available = True
            self.go = go
            self.sp = sp
        except ImportError:
            warnings.warn("Plotly not available - interactive plots disabled")
    
    def create_interactive_dashboard(self, data: Dict[str, Any]) -> Optional[Any]:
        """Create interactive dashboard (placeholder)."""
        if not self.available:
            return None
        
        # Placeholder for interactive dashboard implementation
        return None


# Utility functions for enhanced plotting
def create_regime_timeline_plot(
    analysis_results: pd.DataFrame,
    raw_data: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """
    Create timeline plot showing regime changes over time.
    
    Args:
        analysis_results: Analysis results with regime predictions
        raw_data: Optional price data for overlay
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot price and regimes
    dates = analysis_results.index if hasattr(analysis_results.index, 'to_pydatetime') else range(len(analysis_results))
    
    if raw_data is not None and 'close' in raw_data.columns:
        common_index = analysis_results.index.intersection(raw_data.index)
        price_data = raw_data.loc[common_index, 'close']
        ax1.plot(common_index, price_data, 'k-', linewidth=1, label='Price')
    
    # Color background by regime
    regime_colors = get_regime_colors()
    for i, (idx, row) in enumerate(analysis_results.iterrows()):
        regime_type = row.get('regime_type', f"State_{row['predicted_state']}")
        color = regime_colors.get(regime_type, '#808080')
        
        x_pos = dates[i] if hasattr(dates, '__getitem__') else i
        ax1.axvspan(x_pos, x_pos + 1, alpha=0.3, color=color)
    
    ax1.set_ylabel('Price')
    ax1.set_title('Price with Regime Background')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot confidence
    ax2.plot(dates, analysis_results['confidence'], 'b-', linewidth=2, label='Confidence')
    ax2.fill_between(dates, 0, analysis_results['confidence'], alpha=0.3)
    ax2.set_ylabel('Confidence')
    ax2.set_xlabel('Time')
    ax2.set_title('Regime Confidence')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig


def create_multi_asset_regime_comparison(
    multi_asset_results: Dict[str, pd.DataFrame],
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Create comparison plot for multiple assets' regime analysis.
    
    Args:
        multi_asset_results: Dictionary of asset name -> analysis results
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    n_assets = len(multi_asset_results)
    fig, axes = plt.subplots(n_assets, 1, figsize=figsize, sharex=True)
    
    if n_assets == 1:
        axes = [axes]
    
    regime_colors = get_regime_colors()
    
    for ax, (asset_name, results) in zip(axes, multi_asset_results.items()):
        dates = results.index if hasattr(results.index, 'to_pydatetime') else range(len(results))
        
        # Plot regime sequence
        for i, (idx, row) in enumerate(results.iterrows()):
            regime_type = row.get('regime_type', f"State_{row['predicted_state']}")
            color = regime_colors.get(regime_type, '#808080')
            confidence = row['confidence']
            
            x_pos = dates[i] if hasattr(dates, '__getitem__') else i
            ax.add_patch(Rectangle((x_pos, 0), 1, 1, 
                                 facecolor=color, alpha=0.3 + confidence * 0.5))
        
        # Plot confidence line
        ax.plot(dates, results['confidence'], 'k-', linewidth=1, alpha=0.7)
        
        ax.set_ylabel(f'{asset_name}\nConfidence')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time')
    fig.suptitle('Multi-Asset Regime Comparison', fontsize=16)
    
    # Add shared legend
    legend_elements = create_regime_legend(regime_colors)
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    return fig