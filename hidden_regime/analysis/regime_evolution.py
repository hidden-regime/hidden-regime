"""
Regime Evolution Analysis for Hidden Regime Detection.

Provides comprehensive analysis of how market regimes evolve over time,
tracking parameter drift, stability metrics, and regime transitions
to understand the dynamic nature of market conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..data.collectors import ModelDataCollector, TimestepSnapshot, HMMStateSnapshot
from ..utils.exceptions import AnalysisError


@dataclass
class RegimeTransition:
    """Information about a regime transition event."""
    timestamp: str
    from_regime: int
    to_regime: int
    transition_probability: float
    confidence: float
    duration_in_previous_regime: int
    trigger_factors: Dict[str, float]  # What caused the transition
    market_context: Dict[str, float]


@dataclass
class ParameterDriftMetrics:
    """Metrics for tracking parameter drift over time."""
    parameter_name: str
    drift_magnitude: float  # How much the parameter has changed
    drift_direction: str    # 'increasing', 'decreasing', 'stable', 'volatile'
    stability_score: float  # 0-1, where 1 is perfectly stable
    change_rate: float      # Rate of change per time period
    volatility: float       # Volatility of parameter values
    trend_strength: float   # Strength of any trend (0-1)


@dataclass
class RegimeStabilityAnalysis:
    """Analysis of regime stability and characteristics."""
    regime_id: int
    avg_duration: float
    duration_std: float
    entry_frequency: float  # How often we enter this regime
    exit_probability: float # Probability of exiting per period
    characteristic_returns: Dict[str, float]  # mean, std, skew, kurtosis
    typical_market_conditions: Dict[str, float]
    persistence_score: float  # How persistent this regime is


class RegimeEvolutionAnalyzer:
    """
    Comprehensive analyzer for regime evolution and parameter dynamics.

    Tracks how HMM parameters evolve over time, identifies regime transitions,
    and provides insights into the stability and persistence of market regimes.
    """

    def __init__(self, lookback_window: int = 50, stability_threshold: float = 0.1):
        """
        Initialize regime evolution analyzer.

        Args:
            lookback_window: Number of periods to look back for analysis
            stability_threshold: Threshold for determining parameter stability
        """
        self.lookback_window = lookback_window
        self.stability_threshold = stability_threshold
        self.regime_transitions: List[RegimeTransition] = []
        self.parameter_history: Dict[str, List[Tuple[str, float]]] = {}
        self.regime_characteristics: Dict[int, RegimeStabilityAnalysis] = {}

    def analyze_temporal_evolution(self, temporal_snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze regime evolution from temporal snapshots.

        Args:
            temporal_snapshots: List of timestep snapshots from TemporalController

        Returns:
            Comprehensive evolution analysis
        """
        if len(temporal_snapshots) < 2:
            return {'error': 'Insufficient data for evolution analysis', 'snapshots_received': len(temporal_snapshots)}

        # Extract parameter evolution
        parameter_evolution = self._extract_parameter_evolution(temporal_snapshots)

        # Detect regime transitions
        regime_transitions = self._detect_regime_transitions(temporal_snapshots)

        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(parameter_evolution)

        # Analyze regime characteristics
        regime_characteristics = self._analyze_regime_characteristics(temporal_snapshots)

        # Detect parameter drift patterns
        drift_analysis = self._analyze_parameter_drift(parameter_evolution)

        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_timesteps': len(temporal_snapshots),
            'parameter_evolution': parameter_evolution,
            'regime_transitions': regime_transitions,
            'stability_metrics': stability_metrics,
            'regime_characteristics': regime_characteristics,
            'drift_analysis': drift_analysis,
            'summary_insights': self._generate_summary_insights(
                parameter_evolution, regime_transitions, stability_metrics
            )
        }

    def _extract_parameter_evolution(self, temporal_snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract parameter evolution from temporal snapshots."""
        evolution_data = {
            'timestamps': [],
            'transition_matrices': [],
            'emission_means': [],
            'emission_stds': [],
            'regime_probabilities': [],
            'most_likely_states': []
        }

        for snapshot_data in temporal_snapshots:
            if 'snapshot' not in snapshot_data:
                continue

            snapshot = snapshot_data['snapshot']
            timestamp = snapshot_data['timestamp']

            evolution_data['timestamps'].append(timestamp)

            # Extract HMM parameters if available
            if hasattr(snapshot, 'hmm_state') and snapshot.hmm_state:
                hmm_state = snapshot.hmm_state
                evolution_data['transition_matrices'].append(hmm_state.transition_matrix)
                evolution_data['emission_means'].append(hmm_state.emission_means)
                evolution_data['emission_stds'].append(hmm_state.emission_stds)
                evolution_data['regime_probabilities'].append(hmm_state.regime_probabilities)
                evolution_data['most_likely_states'].append(hmm_state.most_likely_states)
            else:
                # Append None values to maintain alignment
                for key in ['transition_matrices', 'emission_means', 'emission_stds',
                           'regime_probabilities', 'most_likely_states']:
                    evolution_data[key].append(None)

        return evolution_data

    def _detect_regime_transitions(self, temporal_snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect and analyze regime transitions."""
        transitions = []

        for i in range(1, len(temporal_snapshots)):
            prev_snapshot = temporal_snapshots[i-1].get('snapshot')
            curr_snapshot = temporal_snapshots[i].get('snapshot')

            if not prev_snapshot or not curr_snapshot:
                continue

            # Check for regime transitions based on most likely states
            if (hasattr(prev_snapshot, 'hmm_state') and prev_snapshot.hmm_state and
                hasattr(curr_snapshot, 'hmm_state') and curr_snapshot.hmm_state):

                prev_states = prev_snapshot.hmm_state.most_likely_states
                curr_states = curr_snapshot.hmm_state.most_likely_states

                if prev_states and curr_states:
                    prev_regime = prev_states[-1] if len(prev_states) > 0 else None
                    curr_regime = curr_states[-1] if len(curr_states) > 0 else None

                    if prev_regime is not None and curr_regime is not None and prev_regime != curr_regime:
                        # Calculate transition confidence
                        prev_probs = prev_snapshot.hmm_state.regime_probabilities
                        curr_probs = curr_snapshot.hmm_state.regime_probabilities

                        transition_confidence = 0.0
                        if prev_probs and curr_probs and len(prev_probs) > 0 and len(curr_probs) > 0:
                            prev_conf = prev_probs[-1][prev_regime] if len(prev_probs[-1]) > prev_regime else 0.0
                            curr_conf = curr_probs[-1][curr_regime] if len(curr_probs[-1]) > curr_regime else 0.0
                            transition_confidence = (prev_conf + curr_conf) / 2.0

                        transition = {
                            'timestamp': temporal_snapshots[i]['timestamp'],
                            'from_regime': int(prev_regime),
                            'to_regime': int(curr_regime),
                            'confidence': float(transition_confidence),
                            'transition_index': i
                        }
                        transitions.append(transition)

        return transitions

    def _calculate_stability_metrics(self, parameter_evolution: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate parameter stability metrics."""
        stability_metrics = {}

        # Transition matrix stability
        if parameter_evolution['transition_matrices']:
            transition_matrices = [tm for tm in parameter_evolution['transition_matrices'] if tm is not None]
            if len(transition_matrices) > 1:
                stability_metrics['transition_matrix'] = self._calculate_matrix_stability(transition_matrices)

        # Emission parameters stability
        if parameter_evolution['emission_means']:
            emission_means = [em for em in parameter_evolution['emission_means'] if em is not None]
            if len(emission_means) > 1:
                stability_metrics['emission_means'] = self._calculate_vector_stability(emission_means)

        if parameter_evolution['emission_stds']:
            emission_stds = [es for es in parameter_evolution['emission_stds'] if es is not None]
            if len(emission_stds) > 1:
                stability_metrics['emission_stds'] = self._calculate_vector_stability(emission_stds)

        return stability_metrics

    def _calculate_matrix_stability(self, matrices: List[List[List[float]]]) -> Dict[str, float]:
        """Calculate stability metrics for transition matrices."""
        if len(matrices) < 2:
            return {'insufficient_data': True}

        # Convert to numpy arrays
        matrices_array = np.array(matrices)

        # Calculate element-wise standard deviation
        element_std = np.std(matrices_array, axis=0)
        overall_stability = 1.0 - np.mean(element_std)  # Higher values = more stable

        # Calculate correlation between consecutive matrices
        correlations = []
        for i in range(1, len(matrices)):
            corr = np.corrcoef(
                np.array(matrices[i-1]).flatten(),
                np.array(matrices[i]).flatten()
            )[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        avg_correlation = np.mean(correlations) if correlations else 0.0

        return {
            'overall_stability': float(overall_stability),
            'element_wise_std': float(np.mean(element_std)),
            'consecutive_correlation': float(avg_correlation),
            'num_observations': len(matrices)
        }

    def _calculate_vector_stability(self, vectors: List[List[float]]) -> Dict[str, float]:
        """Calculate stability metrics for parameter vectors."""
        if len(vectors) < 2:
            return {'insufficient_data': True}

        # Convert to numpy array
        vectors_array = np.array(vectors)

        # Calculate element-wise standard deviation
        element_std = np.std(vectors_array, axis=0)
        overall_stability = 1.0 - np.mean(element_std)

        # Calculate trend for each element
        trends = []
        for i in range(vectors_array.shape[1]):
            if len(vectors) > 2:
                slope, _, r_value, _, _ = stats.linregress(range(len(vectors)), vectors_array[:, i])
                trends.append(abs(slope))

        avg_trend_magnitude = np.mean(trends) if trends else 0.0

        return {
            'overall_stability': float(overall_stability),
            'element_wise_std': float(np.mean(element_std)),
            'avg_trend_magnitude': float(avg_trend_magnitude),
            'num_observations': len(vectors)
        }

    def _analyze_regime_characteristics(self, temporal_snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze characteristics of each regime."""
        regime_analysis = {}

        # Extract all regime states and their durations
        regime_sequences = []
        for snapshot_data in temporal_snapshots:
            snapshot = snapshot_data.get('snapshot')
            if (snapshot and hasattr(snapshot, 'hmm_state') and snapshot.hmm_state and
                snapshot.hmm_state.most_likely_states):
                regime_sequences.extend(snapshot.hmm_state.most_likely_states)

        if not regime_sequences:
            return {'no_regime_data': True}

        # Analyze each unique regime
        unique_regimes = list(set(regime_sequences))

        for regime in unique_regimes:
            # Calculate regime durations
            durations = self._calculate_regime_durations(regime_sequences, regime)

            # Calculate regime statistics
            regime_analysis[f'regime_{regime}'] = {
                'regime_id': regime,
                'occurrence_count': regime_sequences.count(regime),
                'occurrence_frequency': regime_sequences.count(regime) / len(regime_sequences),
                'avg_duration': np.mean(durations) if durations else 0.0,
                'duration_std': np.std(durations) if durations else 0.0,
                'max_duration': max(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'persistence_score': np.mean(durations) / len(regime_sequences) if durations else 0.0
            }

        return regime_analysis

    def _calculate_regime_durations(self, regime_sequence: List[int], target_regime: int) -> List[int]:
        """Calculate durations for a specific regime."""
        durations = []
        current_duration = 0

        for regime in regime_sequence:
            if regime == target_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0

        # Don't forget the last sequence if it ends with the target regime
        if current_duration > 0:
            durations.append(current_duration)

        return durations

    def _analyze_parameter_drift(self, parameter_evolution: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze parameter drift patterns."""
        drift_analysis = {}

        # Analyze emission means drift
        if parameter_evolution['emission_means']:
            emission_means = [em for em in parameter_evolution['emission_means'] if em is not None]
            if len(emission_means) > 2:
                drift_analysis['emission_means'] = self._calculate_drift_metrics(
                    emission_means, 'emission_means'
                )

        # Analyze emission stds drift
        if parameter_evolution['emission_stds']:
            emission_stds = [es for es in parameter_evolution['emission_stds'] if es is not None]
            if len(emission_stds) > 2:
                drift_analysis['emission_stds'] = self._calculate_drift_metrics(
                    emission_stds, 'emission_stds'
                )

        return drift_analysis

    def _calculate_drift_metrics(self, parameter_series: List[List[float]], parameter_name: str) -> Dict[str, Any]:
        """Calculate drift metrics for a parameter series."""
        parameter_array = np.array(parameter_series)

        drift_metrics = {}

        # Calculate drift for each parameter dimension
        for dim in range(parameter_array.shape[1]):
            values = parameter_array[:, dim]

            # Calculate linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(values)), values)

            # Calculate volatility
            volatility = np.std(values)

            # Determine drift direction
            if abs(slope) < self.stability_threshold:
                direction = 'stable'
            elif slope > 0:
                direction = 'increasing'
            else:
                direction = 'decreasing'

            # Calculate stability score
            stability_score = max(0.0, 1.0 - volatility)

            drift_metrics[f'dimension_{dim}'] = {
                'drift_magnitude': float(abs(slope)),
                'drift_direction': direction,
                'stability_score': float(stability_score),
                'change_rate': float(slope),
                'volatility': float(volatility),
                'trend_strength': float(abs(r_value)),
                'trend_significance': float(p_value)
            }

        return drift_metrics

    def _generate_summary_insights(self, parameter_evolution: Dict[str, Any],
                                 regime_transitions: List[Dict[str, Any]],
                                 stability_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level insights from the analysis."""
        insights = {
            'total_timesteps_analyzed': len(parameter_evolution['timestamps']),
            'regime_transition_frequency': len(regime_transitions) / len(parameter_evolution['timestamps']) if parameter_evolution['timestamps'] else 0,
            'parameter_stability_assessment': 'stable',  # Default
            'key_findings': []
        }

        # Assess overall parameter stability
        stability_scores = []
        for metric_name, metrics in stability_metrics.items():
            if isinstance(metrics, dict) and 'overall_stability' in metrics:
                stability_scores.append(metrics['overall_stability'])

        if stability_scores:
            avg_stability = np.mean(stability_scores)
            if avg_stability > 0.8:
                insights['parameter_stability_assessment'] = 'very_stable'
            elif avg_stability > 0.6:
                insights['parameter_stability_assessment'] = 'moderately_stable'
            elif avg_stability > 0.4:
                insights['parameter_stability_assessment'] = 'somewhat_unstable'
            else:
                insights['parameter_stability_assessment'] = 'highly_unstable'

        # Generate key findings
        findings = []

        if len(regime_transitions) > 0:
            findings.append(f"Detected {len(regime_transitions)} regime transitions across {len(parameter_evolution['timestamps'])} timesteps")

        if stability_scores:
            findings.append(f"Average parameter stability score: {np.mean(stability_scores):.3f}")

        if insights['regime_transition_frequency'] > 0.1:
            findings.append("High regime transition frequency detected - market may be volatile")
        elif insights['regime_transition_frequency'] < 0.05:
            findings.append("Low regime transition frequency - market appears stable")

        insights['key_findings'] = findings

        return insights

    def create_evolution_report(self, analysis_results: Dict[str, Any]) -> str:
        """Create a comprehensive markdown report of regime evolution analysis."""
        report = f"""# Regime Evolution Analysis Report

**Analysis Timestamp:** {analysis_results.get('analysis_timestamp', 'Unknown')}

## Executive Summary

Total timesteps analyzed: {analysis_results.get('total_timesteps', 0)}
Parameter stability: {analysis_results.get('summary_insights', {}).get('parameter_stability_assessment', 'Unknown')}
Regime transitions detected: {len(analysis_results.get('regime_transitions', []))}

## Parameter Evolution Analysis

"""

        # Add stability metrics section
        stability_metrics = analysis_results.get('stability_metrics', {})
        if stability_metrics:
            report += "### Parameter Stability Metrics\n\n"
            for param_name, metrics in stability_metrics.items():
                if isinstance(metrics, dict) and 'overall_stability' in metrics:
                    report += f"**{param_name.replace('_', ' ').title()}:**\n"
                    report += f"- Overall Stability: {metrics['overall_stability']:.3f}\n"
                    report += f"- Observations: {metrics['num_observations']}\n\n"

        # Add regime transition analysis
        regime_transitions = analysis_results.get('regime_transitions', [])
        if regime_transitions:
            report += "### Regime Transitions\n\n"
            report += f"Total transitions detected: {len(regime_transitions)}\n\n"

            for i, transition in enumerate(regime_transitions[:5]):  # Show first 5
                report += f"**Transition {i+1}:**\n"
                report += f"- Date: {transition['timestamp']}\n"
                report += f"- From Regime {transition['from_regime']} to Regime {transition['to_regime']}\n"
                report += f"- Confidence: {transition['confidence']:.3f}\n\n"

        # Add regime characteristics
        regime_chars = analysis_results.get('regime_characteristics', {})
        if regime_chars and not regime_chars.get('no_regime_data'):
            report += "### Regime Characteristics\n\n"
            for regime_key, characteristics in regime_chars.items():
                if isinstance(characteristics, dict):
                    report += f"**{regime_key.replace('_', ' ').title()}:**\n"
                    report += f"- Occurrence Frequency: {characteristics.get('occurrence_frequency', 0):.3f}\n"
                    report += f"- Average Duration: {characteristics.get('avg_duration', 0):.1f} periods\n"
                    report += f"- Persistence Score: {characteristics.get('persistence_score', 0):.3f}\n\n"

        # Add key insights
        insights = analysis_results.get('summary_insights', {})
        if insights.get('key_findings'):
            report += "### Key Findings\n\n"
            for finding in insights['key_findings']:
                report += f"- {finding}\n"
            report += "\n"

        report += "---\n*Generated by Hidden Regime Evolution Analyzer*"

        return report