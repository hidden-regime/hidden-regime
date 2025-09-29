"""
Data Collection Visualization for Hidden Regime Analysis.

Provides comprehensive plotting capabilities for data collection results,
parameter evolution, signal timeline plots, and regime transition analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import warnings
from datetime import datetime, timedelta
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

from ..data.collectors import ModelDataCollector, TimestepSnapshot
from ..analysis.regime_evolution import RegimeEvolutionAnalyzer
from ..analysis.signal_attribution import SignalAttributionAnalyzer
from ..utils.exceptions import AnalysisError


class DataCollectionVisualizationSuite:
    """
    Comprehensive visualization suite for data collection analysis.

    Provides advanced plotting capabilities including parameter evolution,
    signal attribution, regime transitions, and comprehensive dashboards
    specifically tailored for the data collection infrastructure.
    """

    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualization suite.

        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.default_figsize = figsize
        self.color_palette = sns.color_palette("husl", 10)

        # Set plotting style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
            warnings.warn(f"Style '{style}' not available, using default")

        # Define regime colors
        self.regime_colors = {
            0: '#FF6B6B',  # Red for bear/down regime
            1: '#4ECDC4',  # Teal for sideways regime
            2: '#45B7D1',  # Blue for bull/up regime
            3: '#96CEB4',  # Green for additional regimes
            4: '#FFEAA7'   # Yellow for additional regimes
        }

    def plot_parameter_evolution_timeline(self, temporal_snapshots: List[Dict[str, Any]]) -> plt.Figure:
        """
        Plot comprehensive parameter evolution timeline from temporal snapshots.

        Args:
            temporal_snapshots: List of timestep snapshots from TemporalController

        Returns:
            Figure object with parameter evolution analysis
        """
        if not temporal_snapshots:
            return self._create_empty_plot("No temporal data available")

        # Extract parameter evolution data
        timestamps = []
        transition_matrices = []
        emission_means = []
        emission_stds = []
        training_metrics = []

        for snapshot_data in temporal_snapshots:
            snapshot = snapshot_data.get('snapshot')
            if (snapshot and hasattr(snapshot, 'hmm_state') and
                snapshot.hmm_state and snapshot.hmm_state.transition_matrix):

                timestamps.append(pd.to_datetime(snapshot_data['timestamp']))
                transition_matrices.append(snapshot.hmm_state.transition_matrix)
                emission_means.append(snapshot.hmm_state.emission_means)
                emission_stds.append(snapshot.hmm_state.emission_stds)

                # Extract training metrics if available
                training_info = snapshot.hmm_state.__dict__.get('training_info', {})
                training_metrics.append({
                    'log_likelihood': training_info.get('log_likelihood', 0),
                    'n_iterations': training_info.get('n_iterations', 0),
                    'convergence_score': training_info.get('convergence_score', 0)
                })

        if not timestamps:
            return self._create_empty_plot("No HMM parameter data available")

        # Create comprehensive figure
        fig, axes = plt.subplots(4, 1, figsize=(16, 14))

        # Plot 1: Transition matrix stability over time
        if transition_matrices:
            n_states = len(transition_matrices[0])

            # Calculate persistence (diagonal elements)
            for state in range(n_states):
                persistence_values = [tm[state][state] for tm in transition_matrices]
                axes[0].plot(timestamps, persistence_values,
                           marker='o', label=f'Regime {state} Persistence',
                           color=self.regime_colors.get(state, self.color_palette[state]),
                           linewidth=2, markersize=4)

            # Calculate off-diagonal stability
            stability_scores = []
            for i in range(1, len(transition_matrices)):
                prev_tm = np.array(transition_matrices[i-1])
                curr_tm = np.array(transition_matrices[i])
                stability = 1.0 - np.mean(np.abs(curr_tm - prev_tm))
                stability_scores.append(stability)

            if stability_scores:
                ax_twin = axes[0].twinx()
                ax_twin.plot(timestamps[1:], stability_scores, 'k--', alpha=0.7,
                           label='Overall Stability', linewidth=2)
                ax_twin.set_ylabel('Stability Score', color='black')
                ax_twin.set_ylim(0, 1)

        axes[0].set_title('Transition Matrix Evolution & Stability', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Persistence Probability')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Emission parameters evolution
        if emission_means and emission_stds:
            n_states = len(emission_means[0])

            for state in range(n_states):
                means = [em[state] for em in emission_means]
                stds = [es[state] for es in emission_stds]

                # Plot means as lines
                axes[1].plot(timestamps, means,
                           label=f'Regime {state} Mean',
                           color=self.regime_colors.get(state, self.color_palette[state]),
                           linewidth=2)

                # Plot confidence bands using standard deviations
                means_array = np.array(means)
                stds_array = np.array(stds)
                axes[1].fill_between(timestamps,
                                    means_array - stds_array,
                                    means_array + stds_array,
                                    color=self.regime_colors.get(state, self.color_palette[state]),
                                    alpha=0.2)

        axes[1].set_title('Regime Emission Parameters Evolution', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Mean Return ± Std Dev')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Plot 3: Training convergence metrics
        if training_metrics:
            log_likelihoods = [tm['log_likelihood'] for tm in training_metrics]
            iterations = [tm['n_iterations'] for tm in training_metrics]

            # Plot log likelihood evolution
            axes[2].plot(timestamps, log_likelihoods, 'purple', linewidth=2, marker='s',
                       markersize=4, label='Log Likelihood')
            axes[2].set_ylabel('Log Likelihood', color='purple')

            # Plot iterations on secondary axis
            ax2_twin = axes[2].twinx()
            ax2_twin.plot(timestamps, iterations, 'orange', linewidth=2, marker='^',
                        markersize=4, label='Iterations to Convergence')
            ax2_twin.set_ylabel('Iterations', color='orange')

        axes[2].set_title('Model Training Convergence Metrics', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper left')

        # Plot 4: Parameter drift detection
        if len(emission_means) > 10:  # Need sufficient data for drift analysis
            # Calculate rolling parameter changes
            window_size = min(10, len(emission_means) // 3)
            drift_scores = []

            for i in range(window_size, len(emission_means)):
                recent_means = emission_means[i-window_size:i]
                older_means = emission_means[max(0, i-2*window_size):i-window_size]

                if len(recent_means) > 0 and len(older_means) > 0:
                    recent_avg = np.mean(recent_means, axis=0)
                    older_avg = np.mean(older_means, axis=0)
                    drift_magnitude = np.mean(np.abs(recent_avg - older_avg))
                    drift_scores.append(drift_magnitude)

            if drift_scores:
                drift_timestamps = timestamps[window_size:len(drift_scores)+window_size]
                axes[3].plot(drift_timestamps, drift_scores, 'red', linewidth=2,
                           marker='o', markersize=4, label='Parameter Drift Magnitude')

                # Add threshold line
                drift_threshold = np.percentile(drift_scores, 75) if drift_scores else 0
                axes[3].axhline(y=drift_threshold, color='red', linestyle='--', alpha=0.7,
                              label=f'75th Percentile ({drift_threshold:.4f})')

        axes[3].set_title('Parameter Drift Detection', fontsize=14, fontweight='bold')
        axes[3].set_ylabel('Drift Magnitude')
        axes[3].set_xlabel('Date')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        # Format all x-axes
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

    def plot_signal_generation_timeline(self, temporal_snapshots: List[Dict[str, Any]],
                                       price_data: pd.DataFrame) -> plt.Figure:
        """
        Plot comprehensive signal generation timeline with attribution.

        Args:
            temporal_snapshots: List of timestep snapshots
            price_data: Price data for context

        Returns:
            Figure object with signal timeline analysis
        """
        if not temporal_snapshots or price_data.empty:
            return self._create_empty_plot("No data available for signal timeline")

        # Extract signal events with detailed context
        all_signals = []
        signal_strengths_by_indicator = {}
        signal_counts_by_date = {}

        for snapshot_data in temporal_snapshots:
            snapshot = snapshot_data.get('snapshot')
            timestamp = pd.to_datetime(snapshot_data['timestamp'])

            if snapshot and hasattr(snapshot, 'technical_indicators'):
                tech_indicators = snapshot.technical_indicators

                # Extract signal events
                if tech_indicators and 'signal_events' in tech_indicators:
                    for event in tech_indicators['signal_events']:
                        if isinstance(event, dict):
                            event_with_context = event.copy()
                            event_with_context['snapshot_timestamp'] = timestamp
                            all_signals.append(event_with_context)

                # Track signal strengths by indicator
                if tech_indicators and 'generated_signals' in tech_indicators:
                    for indicator, signal_value in tech_indicators['generated_signals'].items():
                        if indicator not in signal_strengths_by_indicator:
                            signal_strengths_by_indicator[indicator] = []
                        signal_strengths_by_indicator[indicator].append({
                            'timestamp': timestamp,
                            'signal_value': signal_value,
                            'strength': abs(signal_value)
                        })

                # Count signals by date
                date_key = timestamp.date()
                if date_key not in signal_counts_by_date:
                    signal_counts_by_date[date_key] = 0
                if tech_indicators and 'signal_events' in tech_indicators:
                    signal_counts_by_date[date_key] += len(tech_indicators['signal_events'])

        # Create comprehensive figure
        fig, axes = plt.subplots(4, 1, figsize=(16, 14))

        # Plot 1: Price with signal overlays
        price_dates = pd.to_datetime(price_data.index)
        axes[0].plot(price_dates, price_data['close'], color='black', linewidth=1, label='Close Price')

        # Add signal markers
        signal_colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'blue'}
        signal_markers = {'BUY': '^', 'SELL': 'v', 'HOLD': 'o'}

        for event in all_signals:
            signal_time = event.get('snapshot_timestamp', event.get('timestamp'))
            signal_type = event.get('signal_type', 'HOLD')
            signal_strength = event.get('signal_strength', 0.5)

            if signal_time:
                signal_time = pd.to_datetime(signal_time)
                # Find closest price
                closest_price_idx = price_dates.get_indexer([signal_time], method='nearest')[0]
                if 0 <= closest_price_idx < len(price_data):
                    price_value = price_data['close'].iloc[closest_price_idx]

                    # Size marker by signal strength
                    marker_size = max(30, signal_strength * 100)

                    axes[0].scatter(signal_time, price_value,
                                  c=signal_colors.get(signal_type, 'gray'),
                                  marker=signal_markers.get(signal_type, 'o'),
                                  s=marker_size, alpha=0.7, edgecolors='black', linewidth=0.5)

        axes[0].set_title('Price Data with Signal Events (Size = Signal Strength)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Signal strength heatmap by indicator
        if signal_strengths_by_indicator:
            # Create heatmap data
            indicators = list(signal_strengths_by_indicator.keys())[:10]  # Limit to top 10
            all_dates = sorted(set(
                pd.to_datetime(signal['timestamp']).date()
                for indicator_signals in signal_strengths_by_indicator.values()
                for signal in indicator_signals
            ))

            heatmap_data = np.zeros((len(indicators), len(all_dates)))

            for i, indicator in enumerate(indicators):
                for signal in signal_strengths_by_indicator[indicator]:
                    date = pd.to_datetime(signal['timestamp']).date()
                    if date in all_dates:
                        j = all_dates.index(date)
                        heatmap_data[i, j] = max(heatmap_data[i, j], signal['strength'])

            # Plot heatmap
            im = axes[1].imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
            axes[1].set_yticks(range(len(indicators)))
            axes[1].set_yticklabels(indicators, fontsize=10)

            # Set x-axis to show dates
            date_indices = range(0, len(all_dates), max(1, len(all_dates) // 10))
            axes[1].set_xticks(date_indices)
            axes[1].set_xticklabels([str(all_dates[i]) for i in date_indices], rotation=45)

            axes[1].set_title('Signal Strength Heatmap by Indicator', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=axes[1], label='Max Signal Strength')

        # Plot 3: Daily signal activity
        if signal_counts_by_date:
            dates = list(signal_counts_by_date.keys())
            counts = list(signal_counts_by_date.values())

            axes[2].bar(dates, counts, color='orange', alpha=0.7, edgecolor='black')
            axes[2].set_title('Daily Signal Generation Activity', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('Number of Signals')
            axes[2].grid(True, alpha=0.3)

            # Add rolling average
            if len(counts) > 7:
                window_size = min(7, len(counts) // 3)
                rolling_avg = pd.Series(counts).rolling(window=window_size, center=True).mean()
                axes[2].plot(dates, rolling_avg, color='red', linewidth=2, label=f'{window_size}-day Average')
                axes[2].legend()

        # Plot 4: Signal attribution analysis
        if all_signals:
            # Count signals by indicator
            signal_counts_by_indicator = {}
            for signal in all_signals:
                indicator = signal.get('indicator_name', 'unknown')
                if indicator not in signal_counts_by_indicator:
                    signal_counts_by_indicator[indicator] = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
                signal_type = signal.get('signal_type', 'HOLD')
                signal_counts_by_indicator[indicator][signal_type] += 1

            # Create stacked bar chart
            indicators = list(signal_counts_by_indicator.keys())[:10]  # Top 10
            buy_counts = [signal_counts_by_indicator[ind]['BUY'] for ind in indicators]
            sell_counts = [signal_counts_by_indicator[ind]['SELL'] for ind in indicators]
            hold_counts = [signal_counts_by_indicator[ind]['HOLD'] for ind in indicators]

            x_pos = np.arange(len(indicators))
            axes[3].bar(x_pos, buy_counts, color='green', alpha=0.7, label='BUY')
            axes[3].bar(x_pos, sell_counts, bottom=buy_counts, color='red', alpha=0.7, label='SELL')
            axes[3].bar(x_pos, hold_counts, bottom=np.array(buy_counts) + np.array(sell_counts),
                       color='blue', alpha=0.7, label='HOLD')

            axes[3].set_xticks(x_pos)
            axes[3].set_xticklabels(indicators, rotation=45, ha='right')
            axes[3].set_title('Signal Distribution by Indicator', fontsize=14, fontweight='bold')
            axes[3].set_ylabel('Number of Signals')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_regime_transition_analysis(self, temporal_snapshots: List[Dict[str, Any]],
                                      price_data: pd.DataFrame) -> plt.Figure:
        """
        Plot comprehensive regime transition analysis with market context.

        Args:
            temporal_snapshots: List of timestep snapshots
            price_data: Price data for context

        Returns:
            Figure object with regime transition analysis
        """
        if not temporal_snapshots or price_data.empty:
            return self._create_empty_plot("No data available for regime analysis")

        # Extract regime data with enhanced context
        regime_data = []
        regime_probabilities = []
        transition_events = []

        for i, snapshot_data in enumerate(temporal_snapshots):
            snapshot = snapshot_data.get('snapshot')
            timestamp = pd.to_datetime(snapshot_data['timestamp'])

            if (snapshot and hasattr(snapshot, 'hmm_state') and
                snapshot.hmm_state and snapshot.hmm_state.most_likely_states):

                current_regime = snapshot.hmm_state.most_likely_states[-1] if snapshot.hmm_state.most_likely_states else 0
                current_probs = snapshot.hmm_state.regime_probabilities[-1] if snapshot.hmm_state.regime_probabilities else [1.0]

                regime_data.append({
                    'timestamp': timestamp,
                    'regime': current_regime,
                    'confidence': max(current_probs) if current_probs else 0.0
                })

                regime_probabilities.append({
                    'timestamp': timestamp,
                    'probabilities': current_probs
                })

                # Detect transitions
                if i > 0 and regime_data:
                    prev_regime = regime_data[-2]['regime'] if len(regime_data) > 1 else current_regime
                    if prev_regime != current_regime:
                        transition_events.append({
                            'timestamp': timestamp,
                            'from_regime': prev_regime,
                            'to_regime': current_regime,
                            'confidence': max(current_probs) if current_probs else 0.0
                        })

        if not regime_data:
            return self._create_empty_plot("No regime data available")

        # Create comprehensive figure
        fig, axes = plt.subplots(4, 1, figsize=(16, 14))

        # Plot 1: Price with regime backgrounds and transitions
        price_dates = pd.to_datetime(price_data.index)
        axes[0].plot(price_dates, price_data['close'], color='black', linewidth=1.5, label='Close Price')

        # Add regime background coloring
        regime_df = pd.DataFrame(regime_data)
        for i in range(len(regime_df) - 1):
            current_regime = regime_df.iloc[i]['regime']
            start_time = regime_df.iloc[i]['timestamp']
            end_time = regime_df.iloc[i + 1]['timestamp']

            axes[0].axvspan(start_time, end_time,
                           color=self.regime_colors.get(current_regime, 'gray'),
                           alpha=0.2)

        # Mark transition points
        for transition in transition_events:
            transition_time = transition['timestamp']
            # Find corresponding price
            closest_price_idx = price_dates.get_indexer([transition_time], method='nearest')[0]
            if 0 <= closest_price_idx < len(price_data):
                price_value = price_data['close'].iloc[closest_price_idx]
                axes[0].axvline(x=transition_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
                axes[0].scatter(transition_time, price_value, color='red', s=100, marker='*',
                              edgecolors='black', linewidth=1, zorder=5)

        axes[0].set_title('Price Evolution with Regime Context & Transitions', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Regime sequence with confidence bands
        regime_sequence = regime_df['regime'].values
        regime_confidence = regime_df['confidence'].values
        regime_times = regime_df['timestamp'].values

        # Plot regime line
        axes[1].plot(regime_times, regime_sequence, marker='o', linewidth=2, markersize=4,
                    color='darkblue', label='Most Likely Regime')

        # Add confidence as transparency or error bars
        for i in range(len(regime_times)):
            confidence = regime_confidence[i]
            alpha = confidence * 0.8 + 0.2  # Map confidence to alpha
            axes[1].scatter(regime_times[i], regime_sequence[i], s=50,
                          color=self.regime_colors.get(regime_sequence[i], 'gray'),
                          alpha=alpha, edgecolors='black', linewidth=0.5)

        axes[1].set_title('Regime Sequence with Confidence (Opacity = Confidence)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Regime ID')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yticks(range(max(regime_sequence) + 1))

        # Plot 3: Regime probabilities over time
        if regime_probabilities and regime_probabilities[0]['probabilities']:
            n_regimes = len(regime_probabilities[0]['probabilities'])
            prob_matrix = np.array([data['probabilities'][:n_regimes] for data in regime_probabilities])

            for regime in range(n_regimes):
                if regime < prob_matrix.shape[1]:
                    axes[2].plot(regime_times, prob_matrix[:, regime],
                                label=f'Regime {regime}',
                                color=self.regime_colors.get(regime, self.color_palette[regime]),
                                linewidth=2)

        axes[2].set_title('Regime Probabilities Evolution', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Probability')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)

        # Plot 4: Transition analysis
        if transition_events:
            # Analyze transition patterns
            transition_df = pd.DataFrame(transition_events)

            # Count transitions by type
            transition_counts = {}
            for _, transition in transition_df.iterrows():
                key = f"{transition['from_regime']} � {transition['to_regime']}"
                transition_counts[key] = transition_counts.get(key, 0) + 1

            # Create transition frequency chart
            transition_types = list(transition_counts.keys())
            transition_frequencies = list(transition_counts.values())

            bars = axes[3].bar(range(len(transition_types)), transition_frequencies,
                              color='steelblue', alpha=0.7, edgecolor='black')

            axes[3].set_xticks(range(len(transition_types)))
            axes[3].set_xticklabels(transition_types, rotation=45, ha='right')
            axes[3].set_title('Regime Transition Frequency Analysis', fontsize=14, fontweight='bold')
            axes[3].set_ylabel('Number of Transitions')
            axes[3].grid(True, alpha=0.3)

            # Add transition count labels on bars
            for bar, count in zip(bars, transition_frequencies):
                axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom', fontweight='bold')
        else:
            axes[3].text(0.5, 0.5, 'No regime transitions detected', ha='center', va='center',
                        transform=axes[3].transAxes, fontsize=14)
            axes[3].set_title('Regime Transition Analysis', fontsize=14, fontweight='bold')

        # Format all x-axes
        for ax in axes[:3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

    def create_data_collection_dashboard(self, temporal_snapshots: List[Dict[str, Any]],
                                       price_data: pd.DataFrame,
                                       evolution_results: Optional[Dict[str, Any]] = None,
                                       attribution_results: Optional[Dict[str, Any]] = None) -> plt.Figure:
        """
        Create comprehensive dashboard for data collection analysis.

        Args:
            temporal_snapshots: List of timestep snapshots
            price_data: Price data for context
            evolution_results: Optional regime evolution analysis results
            attribution_results: Optional signal attribution results

        Returns:
            Figure object with comprehensive dashboard
        """
        # Create large figure with grid layout
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # Extract summary data
        total_snapshots = len(temporal_snapshots)
        date_range = (temporal_snapshots[0]['timestamp'], temporal_snapshots[-1]['timestamp']) if temporal_snapshots else ('N/A', 'N/A')

        # Count data types
        hmm_snapshots = sum(1 for s in temporal_snapshots
                           if s.get('snapshot') and hasattr(s['snapshot'], 'hmm_state') and s['snapshot'].hmm_state)
        signal_snapshots = sum(1 for s in temporal_snapshots
                              if s.get('snapshot') and hasattr(s['snapshot'], 'technical_indicators') and s['snapshot'].technical_indicators)

        # Dashboard Title and Summary (top row, spans all columns)
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')

        summary_text = f"""DATA COLLECTION ANALYSIS DASHBOARD

Analysis Period: {date_range[0]} to {date_range[1]}
Total Timesteps: {total_snapshots}
HMM Data Points: {hmm_snapshots}
Signal Data Points: {signal_snapshots}
Data Quality: {'Excellent' if total_snapshots > 100 else 'Good' if total_snapshots > 50 else 'Limited'}"""

        ax_title.text(0.5, 0.5, summary_text, transform=ax_title.transAxes, fontsize=16,
                     ha='center', va='center', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # Plot 1: Parameter evolution summary (second row, left half)
        ax1 = fig.add_subplot(gs[1, :2])
        if evolution_results and 'parameter_evolution' in evolution_results:
            param_evolution = evolution_results['parameter_evolution']

            if param_evolution.get('emission_means'):
                timestamps = [pd.to_datetime(ts) for ts in param_evolution['timestamps']]
                emission_means = param_evolution['emission_means']

                if emission_means:
                    n_states = len(emission_means[0])
                    for state in range(n_states):
                        means = [em[state] for em in emission_means]
                        ax1.plot(timestamps, means,
                               label=f'Regime {state}',
                               color=self.regime_colors.get(state, self.color_palette[state]),
                               linewidth=2)

        ax1.set_title('Parameter Evolution Summary', fontweight='bold')
        ax1.set_ylabel('Mean Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Signal attribution summary (second row, right half)
        ax2 = fig.add_subplot(gs[1, 2:])
        if attribution_results and attribution_results.get('signal_performances'):
            performances = attribution_results['signal_performances']

            # Get top 5 performers by Sharpe ratio
            sorted_performances = sorted(
                performances.items(),
                key=lambda x: x[1].get('sharpe_ratio', 0),
                reverse=True
            )[:5]

            names = [name for name, _ in sorted_performances]
            sharpe_ratios = [perf.get('sharpe_ratio', 0) for _, perf in sorted_performances]

            colors = ['green' if sr > 0 else 'red' for sr in sharpe_ratios]
            ax2.bar(range(len(names)), sharpe_ratios, color=colors, alpha=0.7)
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, rotation=45, ha='right')

        ax2.set_title('Top Signal Performers (Sharpe Ratio)', fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Data collection quality metrics (third row, left)
        ax3 = fig.add_subplot(gs[2, 0])

        # Calculate data quality metrics
        quality_metrics = {
            'Completeness': hmm_snapshots / total_snapshots if total_snapshots > 0 else 0,
            'Signal Coverage': signal_snapshots / total_snapshots if total_snapshots > 0 else 0,
            'Temporal Consistency': 1.0 if total_snapshots > 10 else total_snapshots / 10.0,
            'Analysis Depth': min(1.0, total_snapshots / 252)  # Normalized to 1 year
        }

        metrics_names = list(quality_metrics.keys())
        metrics_values = list(quality_metrics.values())

        ax3.barh(range(len(metrics_names)), metrics_values, color='skyblue', alpha=0.7)
        ax3.set_yticks(range(len(metrics_names)))
        ax3.set_yticklabels(metrics_names)
        ax3.set_xlim(0, 1)
        ax3.set_title('Data Quality Metrics', fontweight='bold')

        # Add percentage labels
        for i, value in enumerate(metrics_values):
            ax3.text(value + 0.02, i, f'{value:.1%}', va='center', fontweight='bold')

        # Plot 4: Regime distribution (third row, center)
        ax4 = fig.add_subplot(gs[2, 1])

        # Extract regime distribution
        regime_counts = {}
        for snapshot_data in temporal_snapshots:
            snapshot = snapshot_data.get('snapshot')
            if (snapshot and hasattr(snapshot, 'hmm_state') and
                snapshot.hmm_state and snapshot.hmm_state.most_likely_states):
                regime = snapshot.hmm_state.most_likely_states[-1]
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

        if regime_counts:
            regimes = list(regime_counts.keys())
            counts = list(regime_counts.values())
            colors = [self.regime_colors.get(regime, 'gray') for regime in regimes]

            ax4.pie(counts, labels=[f'Regime {r}' for r in regimes],
                   colors=colors, autopct='%1.1f%%', startangle=90)

        ax4.set_title('Regime Distribution', fontweight='bold')

        # Plot 5: Signal activity timeline (third row, right two columns)
        ax5 = fig.add_subplot(gs[2, 2:])

        # Extract daily signal counts
        daily_signals = {}
        for snapshot_data in temporal_snapshots:
            snapshot = snapshot_data.get('snapshot')
            date = pd.to_datetime(snapshot_data['timestamp']).date()

            if date not in daily_signals:
                daily_signals[date] = 0

            if snapshot and hasattr(snapshot, 'technical_indicators'):
                tech_indicators = snapshot.technical_indicators
                if tech_indicators and 'signal_events' in tech_indicators:
                    daily_signals[date] += len(tech_indicators['signal_events'])

        if daily_signals:
            dates = list(daily_signals.keys())
            counts = list(daily_signals.values())

            ax5.plot(dates, counts, color='orange', linewidth=2, marker='o', markersize=3)
            ax5.fill_between(dates, counts, alpha=0.3, color='orange')

        ax5.set_title('Signal Generation Activity Timeline', fontweight='bold')
        ax5.set_ylabel('Daily Signals')
        ax5.grid(True, alpha=0.3)

        # Plot 6: Model performance evolution (bottom row, spans all columns)
        ax6 = fig.add_subplot(gs[3, :])

        # Extract training performance metrics
        training_metrics = []
        for snapshot_data in temporal_snapshots:
            snapshot = snapshot_data.get('snapshot')
            if (snapshot and hasattr(snapshot, 'hmm_state') and
                snapshot.hmm_state):
                hmm_state = snapshot.hmm_state
                training_info = hmm_state.__dict__.get('training_info', {})

                training_metrics.append({
                    'timestamp': pd.to_datetime(snapshot_data['timestamp']),
                    'log_likelihood': training_info.get('log_likelihood', 0),
                    'iterations': training_info.get('n_iterations', 0),
                    'convergence': training_info.get('convergence_score', 0)
                })

        if training_metrics:
            timestamps = [tm['timestamp'] for tm in training_metrics]
            log_likelihoods = [tm['log_likelihood'] for tm in training_metrics]

            ax6.plot(timestamps, log_likelihoods, color='purple', linewidth=2,
                    marker='s', markersize=4, label='Log Likelihood')

            # Add secondary axis for iterations
            ax6_twin = ax6.twinx()
            iterations = [tm['iterations'] for tm in training_metrics]
            ax6_twin.plot(timestamps, iterations, color='orange', linewidth=2,
                         marker='^', markersize=4, label='Iterations')
            ax6_twin.set_ylabel('Iterations', color='orange')

        ax6.set_title('Model Training Performance Evolution', fontweight='bold')
        ax6.set_ylabel('Log Likelihood', color='purple')
        ax6.set_xlabel('Date')
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc='upper left')

        plt.suptitle('Comprehensive Data Collection Analysis Dashboard',
                    fontsize=18, fontweight='bold', y=0.98)

        return fig

    def _create_empty_plot(self, message: str) -> plt.Figure:
        """Create an empty plot with a message."""
        fig, ax = plt.subplots(figsize=self.default_figsize)
        ax.text(0.5, 0.5, message, ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig

    def save_analysis_plots(self, figures: List[plt.Figure], base_filename: str,
                          output_dir: str = "plots") -> Dict[str, str]:
        """
        Save analysis plots to files.

        Args:
            figures: List of matplotlib figures
            base_filename: Base filename without extension
            output_dir: Output directory for plots

        Returns:
            Dictionary mapping plot type to filename
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}
        plot_types = [
            'parameter_evolution',
            'signal_timeline',
            'regime_transitions',
            'dashboard'
        ]

        for i, (fig, plot_type) in enumerate(zip(figures, plot_types)):
            filename = os.path.join(output_dir, f"{base_filename}_{plot_type}.png")
            try:
                fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                saved_files[plot_type] = filename
            except Exception as e:
                warnings.warn(f"Failed to save {filename}: {e}")

        return saved_files