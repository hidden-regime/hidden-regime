"""
Signal Attribution Analysis for Hidden Regime Trading Systems.

Provides comprehensive analysis of signal sources, their performance contributions,
and attribution of returns to different signal generators across time periods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from collections import defaultdict

from ..data.collectors import ModelDataCollector, TimestepSnapshot
from ..utils.exceptions import AnalysisError


@dataclass
class SignalPerformance:
    """Performance metrics for a specific signal source."""
    signal_source: str
    total_signals: int
    buy_signals: int
    sell_signals: int

    # Return metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float

    # Signal quality metrics
    signal_accuracy: float  # Percentage of profitable signals
    avg_signal_strength: float
    avg_confidence: float

    # Risk metrics
    max_drawdown: float
    var_95: float  # 95% Value at Risk
    win_rate: float

    # Attribution metrics
    return_contribution: float  # Contribution to total portfolio return
    risk_contribution: float    # Contribution to portfolio risk


@dataclass
class AttributionBreakdown:
    """Detailed attribution breakdown across time periods."""
    time_period: str
    signal_performances: Dict[str, SignalPerformance]
    total_portfolio_return: float
    total_portfolio_risk: float
    attribution_quality_score: float  # How well we can attribute returns
    unexplained_return: float  # Returns not attributed to any signal


@dataclass
class SignalInteraction:
    """Analysis of how signals interact with each other."""
    signal_pair: Tuple[str, str]
    correlation: float
    interaction_strength: float
    combined_performance: SignalPerformance
    individual_performance_sum: float
    synergy_score: float  # How much better/worse they perform together


class SignalAttributionAnalyzer:
    """
    Comprehensive analyzer for signal attribution and performance tracking.

    Tracks the performance contribution of different signal sources,
    analyzes signal interactions, and provides detailed attribution
    of returns and risk to specific signal generators.
    """

    def __init__(self, attribution_window: int = 252, min_signals_threshold: int = 5):
        """
        Initialize signal attribution analyzer.

        Args:
            attribution_window: Number of periods for rolling attribution analysis
            min_signals_threshold: Minimum signals required for attribution analysis
        """
        self.attribution_window = attribution_window
        self.min_signals_threshold = min_signals_threshold
        self.signal_history: List[Dict[str, Any]] = []
        self.performance_cache: Dict[str, SignalPerformance] = {}
        self.attribution_results: List[AttributionBreakdown] = []

    def analyze_signal_attribution(self, temporal_snapshots: List[Dict[str, Any]],
                                 price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze signal attribution from temporal snapshots and price data.

        Args:
            temporal_snapshots: List of timestep snapshots from TemporalController
            price_data: Price data for calculating returns

        Returns:
            Comprehensive signal attribution analysis
        """
        if len(temporal_snapshots) < self.min_signals_threshold:
            return {'error': 'Insufficient data for attribution analysis',
                   'snapshots_received': len(temporal_snapshots)}

        # Extract signal events from snapshots
        signal_events = self._extract_signal_events(temporal_snapshots)

        # Calculate signal performance metrics
        signal_performances = self._calculate_signal_performances(signal_events, price_data)

        # Analyze signal interactions
        signal_interactions = self._analyze_signal_interactions(signal_events, price_data)

        # Perform time-based attribution analysis
        time_based_attribution = self._perform_time_based_attribution(
            temporal_snapshots, price_data
        )

        # Calculate portfolio-level attribution
        portfolio_attribution = self._calculate_portfolio_attribution(signal_performances)

        # Generate attribution insights
        attribution_insights = self._generate_attribution_insights(
            signal_performances, signal_interactions, portfolio_attribution
        )

        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_timesteps': len(temporal_snapshots),
            'signal_performances': signal_performances,
            'signal_interactions': signal_interactions,
            'time_based_attribution': time_based_attribution,
            'portfolio_attribution': portfolio_attribution,
            'attribution_insights': attribution_insights,
            'quality_metrics': self._calculate_attribution_quality(signal_events)
        }

    def _extract_signal_events(self, temporal_snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract signal events from temporal snapshots."""
        all_signal_events = []

        for snapshot_data in temporal_snapshots:
            snapshot = snapshot_data.get('snapshot')
            timestamp = snapshot_data.get('timestamp')

            if not snapshot or not timestamp:
                continue

            # Extract technical indicator signals
            if hasattr(snapshot, 'technical_indicators') and snapshot.technical_indicators:
                tech_indicators = snapshot.technical_indicators

                # Get signal events
                signal_events = tech_indicators.get('signal_events', [])
                for event in signal_events:
                    if isinstance(event, dict):
                        event['snapshot_timestamp'] = timestamp
                        event['signal_category'] = 'technical_indicator'
                        all_signal_events.append(event)

                # Get generated signals
                generated_signals = tech_indicators.get('generated_signals', {})
                for indicator_name, signal_value in generated_signals.items():
                    if signal_value != 0:  # Only record actual signals
                        signal_event = {
                            'timestamp': timestamp,
                            'snapshot_timestamp': timestamp,
                            'indicator_name': indicator_name,
                            'signal_type': 'BUY' if signal_value > 0 else 'SELL',
                            'signal_strength': abs(signal_value),
                            'signal_confidence': abs(signal_value),  # Use signal strength as confidence
                            'signal_category': 'technical_indicator',
                            'triggering_condition': f'{indicator_name} generated {signal_value} signal'
                        }
                        all_signal_events.append(signal_event)

            # Extract HMM regime signals (if available)
            if hasattr(snapshot, 'hmm_state') and snapshot.hmm_state:
                hmm_state = snapshot.hmm_state
                if hmm_state.regime_probabilities:
                    # Convert regime probabilities to signal events
                    current_probs = hmm_state.regime_probabilities[-1] if hmm_state.regime_probabilities else []
                    if current_probs:
                        dominant_regime = np.argmax(current_probs)
                        confidence = current_probs[dominant_regime]

                        # Create regime-based signal
                        hmm_signal = {
                            'timestamp': timestamp,
                            'snapshot_timestamp': timestamp,
                            'indicator_name': 'HMM_Regime',
                            'signal_type': self._regime_to_signal_type(dominant_regime),
                            'signal_strength': confidence,
                            'signal_confidence': confidence,
                            'signal_category': 'hmm_regime',
                            'triggering_condition': f'HMM detected regime {dominant_regime} with {confidence:.2%} confidence',
                            'regime_id': dominant_regime
                        }
                        all_signal_events.append(hmm_signal)

        return all_signal_events

    def _regime_to_signal_type(self, regime_id: int) -> str:
        """Convert regime ID to signal type based on typical regime interpretation."""
        # This is a simplified mapping - could be enhanced with regime characterization
        if regime_id == 0:
            return 'SELL'  # Assume regime 0 is bearish
        elif regime_id == 1:
            return 'HOLD'  # Assume regime 1 is neutral/sideways
        elif regime_id == 2:
            return 'BUY'   # Assume regime 2 is bullish
        else:
            return 'HOLD'  # Default for other regimes

    def _calculate_signal_performances(self, signal_events: List[Dict[str, Any]],
                                     price_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics for each signal source."""
        signal_performances = {}

        # Group signals by indicator name
        signals_by_indicator = defaultdict(list)
        for event in signal_events:
            indicator_name = event.get('indicator_name', 'unknown')
            signals_by_indicator[indicator_name].append(event)

        # Calculate performance for each indicator
        for indicator_name, signals in signals_by_indicator.items():
            if len(signals) < self.min_signals_threshold:
                continue

            performance = self._calculate_indicator_performance(signals, price_data)
            if performance:
                signal_performances[indicator_name] = performance

        return signal_performances

    def _calculate_indicator_performance(self, signals: List[Dict[str, Any]],
                                       price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculate performance metrics for a specific indicator."""
        if not signals or price_data.empty:
            return None

        try:
            # Create signal series aligned with price data
            signal_series = pd.Series(0, index=price_data.index)
            returns = price_data['close'].pct_change().dropna()

            # Map signals to price data
            for signal in signals:
                signal_timestamp = pd.to_datetime(signal['timestamp'])
                signal_value = 1 if signal['signal_type'] == 'BUY' else -1 if signal['signal_type'] == 'SELL' else 0

                # Find closest timestamp in price data
                closest_idx = returns.index.get_indexer([signal_timestamp], method='nearest')[0]
                if 0 <= closest_idx < len(returns):
                    signal_series.iloc[closest_idx] = signal_value

            # Calculate strategy returns
            strategy_returns = returns * signal_series.shift(1)  # Use previous period signal
            strategy_returns = strategy_returns.dropna()

            if len(strategy_returns) == 0:
                return None

            # Calculate performance metrics
            total_return = strategy_returns.sum()
            annualized_return = strategy_returns.mean() * 252
            volatility = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0

            # Calculate drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            # Signal quality metrics
            profitable_signals = (strategy_returns > 0).sum()
            total_signals_used = (strategy_returns != 0).sum()
            signal_accuracy = profitable_signals / total_signals_used if total_signals_used > 0 else 0.0

            # Risk metrics
            var_95 = np.percentile(strategy_returns, 5)
            win_rate = (strategy_returns > 0).mean()

            # Signal strength metrics
            signal_strengths = [s.get('signal_strength', 0) for s in signals]
            signal_confidences = [s.get('signal_confidence', 0) for s in signals]

            return {
                'total_signals': len(signals),
                'buy_signals': len([s for s in signals if s.get('signal_type') == 'BUY']),
                'sell_signals': len([s for s in signals if s.get('signal_type') == 'SELL']),
                'total_return': float(total_return),
                'annualized_return': float(annualized_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'signal_accuracy': float(signal_accuracy),
                'avg_signal_strength': float(np.mean(signal_strengths)) if signal_strengths else 0.0,
                'avg_confidence': float(np.mean(signal_confidences)) if signal_confidences else 0.0,
                'max_drawdown': float(max_drawdown),
                'var_95': float(var_95),
                'win_rate': float(win_rate),
                'signals_used_in_calculation': int(total_signals_used)
            }

        except Exception as e:
            warnings.warn(f"Error calculating performance for indicator: {e}")
            return None

    def _analyze_signal_interactions(self, signal_events: List[Dict[str, Any]],
                                   price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how different signals interact with each other."""
        interactions = {}

        # Group signals by indicator
        signals_by_indicator = defaultdict(list)
        for event in signal_events:
            indicator_name = event.get('indicator_name', 'unknown')
            signals_by_indicator[indicator_name].append(event)

        # Analyze pairwise interactions
        indicator_names = list(signals_by_indicator.keys())

        for i in range(len(indicator_names)):
            for j in range(i + 1, len(indicator_names)):
                indicator1 = indicator_names[i]
                indicator2 = indicator_names[j]

                interaction = self._calculate_signal_interaction(
                    signals_by_indicator[indicator1],
                    signals_by_indicator[indicator2],
                    price_data
                )

                if interaction:
                    pair_key = f"{indicator1}_vs_{indicator2}"
                    interactions[pair_key] = interaction

        return interactions

    def _calculate_signal_interaction(self, signals1: List[Dict[str, Any]],
                                    signals2: List[Dict[str, Any]],
                                    price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculate interaction metrics between two signal sources."""
        try:
            # Create signal series for both indicators
            signal_series1 = pd.Series(0, index=price_data.index)
            signal_series2 = pd.Series(0, index=price_data.index)

            returns = price_data['close'].pct_change().dropna()

            # Map signals to price data for both indicators
            for signals, series in [(signals1, signal_series1), (signals2, signal_series2)]:
                for signal in signals:
                    signal_timestamp = pd.to_datetime(signal['timestamp'])
                    signal_value = 1 if signal['signal_type'] == 'BUY' else -1 if signal['signal_type'] == 'SELL' else 0

                    closest_idx = returns.index.get_indexer([signal_timestamp], method='nearest')[0]
                    if 0 <= closest_idx < len(returns):
                        series.iloc[closest_idx] = signal_value

            # Calculate correlation between signals
            active_signals1 = signal_series1[signal_series1 != 0]
            active_signals2 = signal_series2[signal_series2 != 0]

            # Align the signals for correlation calculation
            common_index = active_signals1.index.intersection(active_signals2.index)
            if len(common_index) > 5:  # Need at least 5 common signals
                correlation = np.corrcoef(
                    signal_series1.loc[common_index],
                    signal_series2.loc[common_index]
                )[0, 1]
            else:
                correlation = 0.0

            # Calculate combined signal performance
            combined_signal = (signal_series1 + signal_series2) / 2  # Average the signals
            combined_returns = returns * combined_signal.shift(1)
            combined_returns = combined_returns.dropna()

            if len(combined_returns) > 0:
                combined_total_return = combined_returns.sum()
                combined_sharpe = (combined_returns.mean() / combined_returns.std() * np.sqrt(252)) if combined_returns.std() > 0 else 0.0
            else:
                combined_total_return = 0.0
                combined_sharpe = 0.0

            return {
                'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                'common_signals': len(common_index),
                'combined_return': float(combined_total_return),
                'combined_sharpe': float(combined_sharpe),
                'interaction_strength': float(abs(correlation)) if not np.isnan(correlation) else 0.0
            }

        except Exception as e:
            warnings.warn(f"Error calculating signal interaction: {e}")
            return None

    def _perform_time_based_attribution(self, temporal_snapshots: List[Dict[str, Any]],
                                       price_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform attribution analysis across different time periods."""
        time_attribution = {
            'monthly': {},
            'quarterly': {},
            'rolling_window': {}
        }

        # Group snapshots by time periods
        snapshots_by_month = defaultdict(list)
        snapshots_by_quarter = defaultdict(list)

        for snapshot_data in temporal_snapshots:
            timestamp = pd.to_datetime(snapshot_data['timestamp'])
            month_key = timestamp.strftime('%Y-%m')
            quarter_key = f"{timestamp.year}-Q{timestamp.quarter}"

            snapshots_by_month[month_key].append(snapshot_data)
            snapshots_by_quarter[quarter_key].append(snapshot_data)

        # Analyze monthly attribution
        for month, snapshots in snapshots_by_month.items():
            if len(snapshots) >= 5:  # Minimum snapshots for analysis
                month_signals = self._extract_signal_events(snapshots)
                month_performance = self._calculate_signal_performances(month_signals, price_data)
                time_attribution['monthly'][month] = month_performance

        # Analyze quarterly attribution
        for quarter, snapshots in snapshots_by_quarter.items():
            if len(snapshots) >= 10:  # Minimum snapshots for analysis
                quarter_signals = self._extract_signal_events(snapshots)
                quarter_performance = self._calculate_signal_performances(quarter_signals, price_data)
                time_attribution['quarterly'][quarter] = quarter_performance

        return time_attribution

    def _calculate_portfolio_attribution(self, signal_performances: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio-level attribution metrics."""
        if not signal_performances:
            return {'no_signals': True}

        total_return = sum(perf.get('total_return', 0) for perf in signal_performances.values())
        total_signals = sum(perf.get('total_signals', 0) for perf in signal_performances.values())

        # Calculate contribution percentages
        return_contributions = {}
        signal_contributions = {}

        for indicator, performance in signal_performances.items():
            indicator_return = performance.get('total_return', 0)
            indicator_signals = performance.get('total_signals', 0)

            return_contributions[indicator] = (indicator_return / total_return * 100) if total_return != 0 else 0
            signal_contributions[indicator] = (indicator_signals / total_signals * 100) if total_signals > 0 else 0

        # Find top contributors
        top_return_contributors = sorted(return_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_signal_contributors = sorted(signal_contributions.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'total_portfolio_return': total_return,
            'total_signals_generated': total_signals,
            'return_contributions': return_contributions,
            'signal_contributions': signal_contributions,
            'top_return_contributors': top_return_contributors,
            'top_signal_contributors': top_signal_contributors,
            'diversification_score': len(signal_performances) / max(1, total_signals) * 100  # How diversified the signals are
        }

    def _generate_attribution_insights(self, signal_performances: Dict[str, Dict[str, Any]],
                                     signal_interactions: Dict[str, Any],
                                     portfolio_attribution: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level insights from attribution analysis."""
        insights = {
            'total_signal_sources': len(signal_performances),
            'best_performing_signal': None,
            'worst_performing_signal': None,
            'most_active_signal': None,
            'signal_quality_assessment': 'unknown',
            'diversification_level': 'unknown',
            'key_findings': []
        }

        if not signal_performances:
            insights['key_findings'].append("No signal performance data available")
            return insights

        # Find best and worst performing signals
        performances_by_return = sorted(
            signal_performances.items(),
            key=lambda x: x[1].get('total_return', 0),
            reverse=True
        )

        if performances_by_return:
            insights['best_performing_signal'] = {
                'name': performances_by_return[0][0],
                'return': performances_by_return[0][1].get('total_return', 0),
                'sharpe_ratio': performances_by_return[0][1].get('sharpe_ratio', 0)
            }

            insights['worst_performing_signal'] = {
                'name': performances_by_return[-1][0],
                'return': performances_by_return[-1][1].get('total_return', 0),
                'sharpe_ratio': performances_by_return[-1][1].get('sharpe_ratio', 0)
            }

        # Find most active signal source
        performances_by_signals = sorted(
            signal_performances.items(),
            key=lambda x: x[1].get('total_signals', 0),
            reverse=True
        )

        if performances_by_signals:
            insights['most_active_signal'] = {
                'name': performances_by_signals[0][0],
                'total_signals': performances_by_signals[0][1].get('total_signals', 0)
            }

        # Assess signal quality
        avg_win_rate = np.mean([perf.get('win_rate', 0) for perf in signal_performances.values()])
        avg_sharpe = np.mean([perf.get('sharpe_ratio', 0) for perf in signal_performances.values()])

        if avg_win_rate > 0.6 and avg_sharpe > 1.0:
            insights['signal_quality_assessment'] = 'excellent'
        elif avg_win_rate > 0.5 and avg_sharpe > 0.5:
            insights['signal_quality_assessment'] = 'good'
        elif avg_win_rate > 0.4 and avg_sharpe > 0.0:
            insights['signal_quality_assessment'] = 'fair'
        else:
            insights['signal_quality_assessment'] = 'poor'

        # Assess diversification
        diversification_score = portfolio_attribution.get('diversification_score', 0)
        if diversification_score > 80:
            insights['diversification_level'] = 'highly_diversified'
        elif diversification_score > 60:
            insights['diversification_level'] = 'well_diversified'
        elif diversification_score > 40:
            insights['diversification_level'] = 'moderately_diversified'
        else:
            insights['diversification_level'] = 'concentrated'

        # Generate key findings
        findings = []
        findings.append(f"Analyzed {len(signal_performances)} signal sources")

        if insights['best_performing_signal']:
            findings.append(f"Best performer: {insights['best_performing_signal']['name']} with {insights['best_performing_signal']['return']:.3f} total return")

        findings.append(f"Average signal quality: {insights['signal_quality_assessment']}")
        findings.append(f"Portfolio diversification: {insights['diversification_level']}")

        # Check for highly correlated signals
        high_correlations = [interaction for interaction in signal_interactions.values()
                           if isinstance(interaction, dict) and abs(interaction.get('correlation', 0)) > 0.8]
        if high_correlations:
            findings.append(f"Found {len(high_correlations)} highly correlated signal pairs - consider reducing redundancy")

        insights['key_findings'] = findings

        return insights

    def _calculate_attribution_quality(self, signal_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics for attribution analysis quality."""
        if not signal_events:
            return {'no_signals': True}

        # Count signals by category
        signals_by_category = defaultdict(int)
        signals_by_source = defaultdict(int)

        for event in signal_events:
            category = event.get('signal_category', 'unknown')
            source = event.get('indicator_name', 'unknown')
            signals_by_category[category] += 1
            signals_by_source[source] += 1

        return {
            'total_signal_events': len(signal_events),
            'signals_by_category': dict(signals_by_category),
            'signals_by_source': dict(signals_by_source),
            'unique_signal_sources': len(signals_by_source),
            'coverage_assessment': 'good' if len(signals_by_source) > 5 else 'limited'
        }

    def create_attribution_report(self, attribution_results: Dict[str, Any]) -> str:
        """Create a comprehensive markdown report of signal attribution analysis."""
        report = f"""# Signal Attribution Analysis Report

**Analysis Timestamp:** {attribution_results.get('analysis_timestamp', 'Unknown')}

## Executive Summary

Total signal sources analyzed: {attribution_results.get('attribution_insights', {}).get('total_signal_sources', 0)}
Signal quality assessment: {attribution_results.get('attribution_insights', {}).get('signal_quality_assessment', 'Unknown')}
Diversification level: {attribution_results.get('attribution_insights', {}).get('diversification_level', 'Unknown')}

## Signal Performance Analysis

"""

        # Add individual signal performances
        signal_performances = attribution_results.get('signal_performances', {})
        if signal_performances:
            report += "### Individual Signal Performance\n\n"

            # Sort by total return
            sorted_performances = sorted(
                signal_performances.items(),
                key=lambda x: x[1].get('total_return', 0),
                reverse=True
            )

            for signal_name, performance in sorted_performances[:10]:  # Top 10
                report += f"**{signal_name}:**\n"
                report += f"- Total Return: {performance.get('total_return', 0):.3f}\n"
                report += f"- Sharpe Ratio: {performance.get('sharpe_ratio', 0):.3f}\n"
                report += f"- Win Rate: {performance.get('win_rate', 0):.2%}\n"
                report += f"- Total Signals: {performance.get('total_signals', 0)}\n\n"

        # Add portfolio attribution
        portfolio_attribution = attribution_results.get('portfolio_attribution', {})
        if portfolio_attribution and not portfolio_attribution.get('no_signals'):
            report += "### Portfolio Attribution\n\n"
            report += f"Total Portfolio Return: {portfolio_attribution.get('total_portfolio_return', 0):.3f}\n"
            report += f"Total Signals Generated: {portfolio_attribution.get('total_signals_generated', 0)}\n\n"

            top_contributors = portfolio_attribution.get('top_return_contributors', [])
            if top_contributors:
                report += "**Top Return Contributors:**\n"
                for i, (signal_name, contribution) in enumerate(top_contributors[:5], 1):
                    report += f"{i}. {signal_name}: {contribution:.1f}%\n"
                report += "\n"

        # Add signal interactions
        signal_interactions = attribution_results.get('signal_interactions', {})
        if signal_interactions:
            report += "### Signal Interactions\n\n"

            # Show most correlated signals
            correlations = [(name, interaction.get('correlation', 0))
                          for name, interaction in signal_interactions.items()
                          if isinstance(interaction, dict)]
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            if correlations:
                report += "**Highest Correlations:**\n"
                for name, correlation in correlations[:5]:
                    report += f"- {name.replace('_vs_', ' vs ')}: {correlation:.3f}\n"
                report += "\n"

        # Add key insights
        insights = attribution_results.get('attribution_insights', {})
        if insights.get('key_findings'):
            report += "### Key Findings\n\n"
            for finding in insights['key_findings']:
                report += f"- {finding}\n"
            report += "\n"

        report += "---\n*Generated by Hidden Regime Signal Attribution Analyzer*"

        return report