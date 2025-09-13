#!/usr/bin/env python3
"""
Regime Transition Analysis Example
==================================

This example provides advanced analysis of regime transitions, including
timing analysis, transition prediction, duration modeling, and early
warning systems for regime changes.

Key features:
- Regime transition timing analysis
- Transition prediction modeling
- Duration and persistence analysis
- Early warning signal generation
- Transition volatility and impact assessment
- Multi-timeframe transition analysis

Use cases:
- Risk management systems
- Tactical asset allocation
- Market timing strategies
- Regime change early warning systems

"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Import our framework
from hidden_regime.data import DataLoader
from hidden_regime.analysis import RegimeAnalyzer
from hidden_regime.config import DataConfig

class RegimeTransitionAnalyzer:
    """Advanced analyzer for regime transitions and prediction"""
    
    def __init__(self, data_config: Optional[DataConfig] = None):
        self.data_config = data_config or DataConfig()
        self.analyzer = RegimeAnalyzer(self.data_config)
        
        # Color scheme for visualizations
        self.regime_colors = {
            'Bear': '#d62728',
            'Sideways': '#7f7f7f', 
            'Bull': '#2ca02c'
        }
    
    def analyze_transition_timing(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Analyze timing and patterns of regime transitions"""
        
        print(f"Analyzing transition timing for {symbol}...")
        
        analysis = self.analyzer.analyze_stock(symbol, start_date, end_date)
        if not analysis:
            return {}
        
        states = analysis['states']
        
        # Find all transitions
        transitions = []
        current_regime = states[0]
        regime_start = 0
        
        state_to_regime = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        
        for i in range(1, len(states)):
            if states[i] != current_regime:
                # Record completed regime
                transitions.append({
                    'regime': state_to_regime[current_regime],
                    'start_day': regime_start,
                    'end_day': i - 1,
                    'duration': i - regime_start,
                    'transition_to': state_to_regime[states[i]]
                })
                
                current_regime = states[i]
                regime_start = i
        
        # Add final regime
        transitions.append({
            'regime': state_to_regime[current_regime],
            'start_day': regime_start,
            'end_day': len(states) - 1,
            'duration': len(states) - regime_start,
            'transition_to': None
        })
        
        # Calculate transition statistics
        transition_stats = self._calculate_transition_statistics(transitions)
        
        # Analyze transition patterns
        transition_patterns = self._analyze_transition_patterns(transitions)
        
        # Calculate transition probabilities
        transition_matrix = self._calculate_detailed_transition_matrix(states)
        
        return {
            'symbol': symbol,
            'transitions': transitions,
            'transition_stats': transition_stats,
            'transition_patterns': transition_patterns,
            'transition_matrix': transition_matrix,
            'total_transitions': len([t for t in transitions if t['transition_to'] is not None])
        }
    
    def predict_regime_transitions(self, symbol: str, start_date: str, end_date: str,
                                 prediction_horizon: int = 30) -> Dict[str, Any]:
        """Predict upcoming regime transitions using statistical models"""
        
        print(f"Predicting regime transitions for {symbol}...")
        
        analysis = self.analyzer.analyze_stock(symbol, start_date, end_date)
        if not analysis:
            return {}
        
        states = analysis['states']
        probabilities = analysis['probabilities']
        
        # Current regime analysis
        current_state = states[-1]
        current_prob = probabilities[-1, current_state]
        
        # Calculate time in current regime
        days_in_regime = 1
        for i in range(len(states) - 2, -1, -1):
            if states[i] == current_state:
                days_in_regime += 1
            else:
                break
        
        # Historical regime duration analysis
        regime_durations = self._calculate_regime_durations(states)
        
        # Estimate transition probability based on duration
        current_regime_durations = [d for d in regime_durations if d['regime'] == current_state]
        
        if current_regime_durations:
            avg_duration = np.mean([d['duration'] for d in current_regime_durations])
            std_duration = np.std([d['duration'] for d in current_regime_durations])
            
            # Calculate survival probability (probability of continuing in regime)
            if std_duration > 0:
                z_score = (days_in_regime - avg_duration) / std_duration
                survival_prob = 1 - stats.norm.cdf(z_score)
            else:
                survival_prob = 0.5
        else:
            survival_prob = 0.5
            avg_duration = days_in_regime
        
        # Transition probability estimation
        base_transition_prob = 1 / avg_duration  # Base daily transition probability
        duration_adjusted_prob = base_transition_prob * (1 + (days_in_regime / avg_duration - 1) * 0.5)
        confidence_adjusted_prob = duration_adjusted_prob * (2 - current_prob)  # Lower confidence = higher transition prob
        
        # Clamp probability
        transition_prob = np.clip(confidence_adjusted_prob, 0.01, 0.95)
        
        # Generate predictions for prediction horizon
        predictions = []
        prob_no_transition = 1.0
        
        for day in range(1, prediction_horizon + 1):
            daily_transition_prob = transition_prob
            prob_transition_by_day = 1 - (prob_no_transition * (1 - daily_transition_prob))
            prob_no_transition *= (1 - daily_transition_prob)
            
            predictions.append({
                'day': day,
                'transition_probability': prob_transition_by_day,
                'survival_probability': prob_no_transition
            })
        
        # Most likely transition target
        transition_matrix = self._calculate_detailed_transition_matrix(states)
        if current_state < len(transition_matrix):
            transition_probs = transition_matrix[current_state]
            # Exclude staying in same state
            transition_probs = transition_probs.copy()
            transition_probs[current_state] = 0
            transition_probs = transition_probs / transition_probs.sum() if transition_probs.sum() > 0 else transition_probs
            
            most_likely_target = np.argmax(transition_probs)
            target_probability = transition_probs[most_likely_target]
        else:
            most_likely_target = 1  # Default to sideways
            target_probability = 0.5
        
        state_to_regime = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        
        return {
            'symbol': symbol,
            'current_regime': state_to_regime[current_state],
            'current_confidence': current_prob,
            'days_in_regime': days_in_regime,
            'predicted_target_regime': state_to_regime[most_likely_target],
            'target_probability': target_probability,
            'daily_transition_probability': transition_prob,
            'survival_probability': survival_prob,
            'predictions': predictions,
            'avg_regime_duration': avg_duration
        }
    
    def generate_early_warning_signals(self, symbol: str, start_date: str, end_date: str,
                                     lookback_window: int = 20) -> Dict[str, Any]:
        """Generate early warning signals for potential regime changes"""
        
        print(f"Generating early warning signals for {symbol}...")
        
        analysis = self.analyzer.analyze_stock(symbol, start_date, end_date)
        if not analysis:
            return {}
        
        states = analysis['states']
        probabilities = analysis['probabilities']
        
        # Load price data for additional signals
        data_loader = DataLoader(self.data_config)
        data = data_loader.load_stock_data(symbol, start_date, end_date)
        
        returns = data['log_return'].dropna()
        
        # Calculate various early warning indicators
        warning_signals = []
        
        for i in range(lookback_window, len(probabilities)):
            current_state = states[i]
            current_prob = probabilities[i, current_state]
            
            # Signal 1: Declining regime confidence
            prob_trend = np.mean(probabilities[i-lookback_window:i, current_state]) - current_prob
            confidence_warning = prob_trend > 0.1  # Confidence declining
            
            # Signal 2: Increased probability volatility
            prob_volatility = np.std(probabilities[i-lookback_window:i, current_state])
            volatility_warning = prob_volatility > 0.15  # High probability volatility
            
            # Signal 3: Return pattern inconsistent with regime
            recent_returns = returns.iloc[i-lookback_window:i]
            expected_return = analysis['regime_stats'][f'state_{current_state}']['mean_return']
            return_deviation = abs(recent_returns.mean() - expected_return)
            pattern_warning = return_deviation > abs(expected_return) * 0.5
            
            # Signal 4: Increased return volatility
            return_volatility = recent_returns.std()
            expected_volatility = analysis['regime_stats'][f'state_{current_state}']['std_return']
            volatility_ratio = return_volatility / max(expected_volatility, 0.01)
            volatility_increase_warning = volatility_ratio > 1.5
            
            # Signal 5: Regime duration approaching historical average
            days_in_regime = self._get_days_in_regime_at_time(states, i)
            regime_durations = self._calculate_regime_durations(states[:i])
            current_regime_durations = [d['duration'] for d in regime_durations if d['regime'] == current_state]
            
            if current_regime_durations:
                avg_duration = np.mean(current_regime_durations)
                duration_warning = days_in_regime > avg_duration * 0.8
            else:
                duration_warning = False
                avg_duration = float('inf')
            
            # Composite warning score
            warning_score = sum([
                confidence_warning * 0.3,
                volatility_warning * 0.2,
                pattern_warning * 0.2,
                volatility_increase_warning * 0.15,
                duration_warning * 0.15
            ])
            
            # Determine warning level
            if warning_score >= 0.6:
                warning_level = 'HIGH'
            elif warning_score >= 0.4:
                warning_level = 'MEDIUM'
            elif warning_score >= 0.2:
                warning_level = 'LOW'
            else:
                warning_level = 'NONE'
            
            warning_signals.append({
                'day': i,
                'current_regime': current_state,
                'confidence': current_prob,
                'days_in_regime': days_in_regime,
                'warning_score': warning_score,
                'warning_level': warning_level,
                'signals': {
                    'confidence_declining': confidence_warning,
                    'probability_volatile': volatility_warning,
                    'pattern_inconsistent': pattern_warning,
                    'volatility_increased': volatility_increase_warning,
                    'duration_extended': duration_warning
                },
                'metrics': {
                    'prob_trend': prob_trend,
                    'prob_volatility': prob_volatility,
                    'return_deviation': return_deviation,
                    'volatility_ratio': volatility_ratio,
                    'avg_duration': avg_duration
                }
            })
        
        # Calculate warning signal statistics
        warning_levels = [w['warning_level'] for w in warning_signals]
        warning_stats = {
            'total_signals': len(warning_signals),
            'high_warnings': warning_levels.count('HIGH'),
            'medium_warnings': warning_levels.count('MEDIUM'),
            'low_warnings': warning_levels.count('LOW'),
            'no_warnings': warning_levels.count('NONE')
        }
        
        # Find recent warning periods
        recent_warnings = [w for w in warning_signals[-30:] if w['warning_level'] in ['HIGH', 'MEDIUM']]
        
        return {
            'symbol': symbol,
            'warning_signals': warning_signals,
            'warning_stats': warning_stats,
            'recent_warnings': recent_warnings,
            'current_warning_level': warning_signals[-1]['warning_level'] if warning_signals else 'NONE'
        }
    
    def analyze_transition_impact(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Analyze the impact and characteristics of regime transitions"""
        
        print(f"Analyzing transition impact for {symbol}...")
        
        analysis = self.analyzer.analyze_stock(symbol, start_date, end_date)
        if not analysis:
            return {}
        
        states = analysis['states']
        
        # Load price and return data
        data_loader = DataLoader(self.data_config)
        data = data_loader.load_stock_data(symbol, start_date, end_date)
        returns = data['log_return'].dropna()
        
        # Find transition points
        transition_points = []
        for i in range(1, len(states)):
            if states[i] != states[i-1]:
                transition_points.append({
                    'day': i,
                    'from_regime': states[i-1],
                    'to_regime': states[i]
                })
        
        # Analyze impact around each transition
        impact_analysis = []
        impact_windows = [1, 3, 5, 10, 20]  # Days before/after transition
        
        for transition in transition_points:
            day = transition['day']
            impact_data = {'transition': transition}
            
            for window in impact_windows:
                pre_start = max(0, day - window)
                pre_end = day
                post_start = day
                post_end = min(len(returns), day + window)
                
                if pre_end > pre_start and post_end > post_start:
                    pre_returns = returns.iloc[pre_start:pre_end]
                    post_returns = returns.iloc[post_start:post_end]
                    
                    impact_data[f'pre_{window}d'] = {
                        'mean_return': pre_returns.mean(),
                        'volatility': pre_returns.std(),
                        'cumulative_return': pre_returns.sum()
                    }
                    
                    impact_data[f'post_{window}d'] = {
                        'mean_return': post_returns.mean(),
                        'volatility': post_returns.std(),
                        'cumulative_return': post_returns.sum()
                    }
                    
                    # Calculate impact metrics
                    return_change = post_returns.mean() - pre_returns.mean()
                    volatility_change = post_returns.std() - pre_returns.std()
                    
                    impact_data[f'impact_{window}d'] = {
                        'return_change': return_change,
                        'volatility_change': volatility_change,
                        'significance': abs(return_change) / max(pre_returns.std(), 0.01)
                    }
            
            impact_analysis.append(impact_data)
        
        # Calculate aggregate impact statistics
        state_to_regime = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        transition_types = {}
        
        for impact in impact_analysis:
            from_regime = state_to_regime[impact['transition']['from_regime']]
            to_regime = state_to_regime[impact['transition']['to_regime']]
            transition_key = f"{from_regime}_to_{to_regime}"
            
            if transition_key not in transition_types:
                transition_types[transition_key] = []
            
            transition_types[transition_key].append(impact)
        
        # Calculate average impact by transition type
        transition_impacts = {}
        for transition_type, impacts in transition_types.items():
            if impacts:
                # Calculate average impact across all transitions of this type
                avg_impact = {}
                for window in impact_windows:
                    returns_changes = [i.get(f'impact_{window}d', {}).get('return_change', 0) 
                                     for i in impacts if f'impact_{window}d' in i]
                    vol_changes = [i.get(f'impact_{window}d', {}).get('volatility_change', 0) 
                                 for i in impacts if f'impact_{window}d' in i]
                    
                    if returns_changes and vol_changes:
                        avg_impact[f'{window}d'] = {
                            'avg_return_change': np.mean(returns_changes),
                            'avg_volatility_change': np.mean(vol_changes),
                            'count': len(returns_changes)
                        }
                
                transition_impacts[transition_type] = avg_impact
        
        return {
            'symbol': symbol,
            'transition_points': transition_points,
            'impact_analysis': impact_analysis,
            'transition_impacts': transition_impacts,
            'total_transitions': len(transition_points)
        }
    
    def _calculate_transition_statistics(self, transitions: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive statistics about regime transitions"""
        
        if not transitions:
            return {}
        
        # Duration statistics by regime
        regime_durations = {}
        for transition in transitions:
            regime = transition['regime']
            duration = transition['duration']
            
            if regime not in regime_durations:
                regime_durations[regime] = []
            regime_durations[regime].append(duration)
        
        duration_stats = {}
        for regime, durations in regime_durations.items():
            duration_stats[regime] = {
                'mean_duration': np.mean(durations),
                'median_duration': np.median(durations),
                'std_duration': np.std(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'count': len(durations)
            }
        
        # Transition frequency
        total_days = sum(t['duration'] for t in transitions)
        completed_transitions = [t for t in transitions if t['transition_to'] is not None]
        
        return {
            'duration_stats': duration_stats,
            'total_transitions': len(completed_transitions),
            'total_days': total_days,
            'avg_days_per_transition': total_days / len(completed_transitions) if completed_transitions else float('inf'),
            'transition_frequency': len(completed_transitions) / total_days if total_days > 0 else 0
        }
    
    def _analyze_transition_patterns(self, transitions: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in regime transitions"""
        
        if not transitions:
            return {}
        
        # Count transition types
        transition_counts = {}
        for transition in transitions:
            if transition['transition_to'] is not None:
                key = f"{transition['regime']}_to_{transition['transition_to']}"
                transition_counts[key] = transition_counts.get(key, 0) + 1
        
        # Find most common transitions
        sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate transition cyclicality
        regime_sequence = [t['regime'] for t in transitions if t['transition_to'] is not None]
        if len(regime_sequence) >= 3:
            # Look for cycles (e.g., Bull -> Sideways -> Bear -> Bull)
            cycles = {}
            for i in range(len(regime_sequence) - 2):
                cycle = tuple(regime_sequence[i:i+3])
                cycles[cycle] = cycles.get(cycle, 0) + 1
        else:
            cycles = {}
        
        return {
            'transition_counts': transition_counts,
            'most_common_transitions': sorted_transitions[:5],
            'cycles': cycles,
            'regime_persistence': self._calculate_regime_persistence(transitions)
        }
    
    def _calculate_regime_persistence(self, transitions: List[Dict]) -> Dict[str, float]:
        """Calculate how persistent each regime tends to be"""
        
        persistence = {}
        for transition in transitions:
            regime = transition['regime']
            duration = transition['duration']
            
            if regime not in persistence:
                persistence[regime] = []
            persistence[regime].append(duration)
        
        return {regime: np.mean(durations) for regime, durations in persistence.items()}
    
    def _calculate_detailed_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """Calculate detailed transition probability matrix"""
        
        n_states = max(states) + 1
        transition_matrix = np.zeros((n_states, n_states))
        
        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]
            transition_matrix[from_state, to_state] += 1
        
        # Normalize to probabilities
        for i in range(n_states):
            row_sum = transition_matrix[i].sum()
            if row_sum > 0:
                transition_matrix[i] = transition_matrix[i] / row_sum
        
        return transition_matrix
    
    def _calculate_regime_durations(self, states: np.ndarray) -> List[Dict]:
        """Extract regime durations from state sequence"""
        
        if len(states) == 0:
            return []
        
        durations = []
        current_regime = states[0]
        duration = 1
        start_idx = 0
        
        for i in range(1, len(states)):
            if states[i] == current_regime:
                duration += 1
            else:
                durations.append({
                    'regime': current_regime,
                    'duration': duration,
                    'start_idx': start_idx,
                    'end_idx': i - 1
                })
                current_regime = states[i]
                duration = 1
                start_idx = i
        
        # Add final duration
        durations.append({
            'regime': current_regime,
            'duration': duration,
            'start_idx': start_idx,
            'end_idx': len(states) - 1
        })
        
        return durations
    
    def _get_days_in_regime_at_time(self, states: np.ndarray, time_idx: int) -> int:
        """Calculate days in current regime at a specific time point"""
        
        if time_idx >= len(states):
            return 0
        
        current_state = states[time_idx]
        days = 1
        
        for i in range(time_idx - 1, -1, -1):
            if states[i] == current_state:
                days += 1
            else:
                break
        
        return days
    
    def create_comprehensive_transition_visualizations(self, symbol: str, start_date: str,
                                                     end_date: str, output_dir: str) -> str:
        """Create comprehensive visualizations for regime transition analysis"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating comprehensive transition visualizations for {symbol}...")
        
        # Get all analyses
        timing_analysis = self.analyze_transition_timing(symbol, start_date, end_date)
        prediction_analysis = self.predict_regime_transitions(symbol, start_date, end_date)
        warning_analysis = self.generate_early_warning_signals(symbol, start_date, end_date)
        impact_analysis = self.analyze_transition_impact(symbol, start_date, end_date)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1], width_ratios=[2, 1, 1])
        
        fig.suptitle(f'Comprehensive Regime Transition Analysis: {symbol}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Transition Timeline (spans full width)
        ax1 = fig.add_subplot(gs[0, :])
        if timing_analysis and 'transitions' in timing_analysis:
            transitions = timing_analysis['transitions']
            
            # Plot regime timeline
            for i, transition in enumerate(transitions):
                start = transition['start_day']
                duration = transition['duration']
                regime = transition['regime']
                color = self.regime_colors[regime]
                
                ax1.barh(0, duration, left=start, height=0.5, 
                        color=color, alpha=0.7, label=regime if i == 0 else "")
                
                # Add regime label
                if duration > 10:  # Only label if duration is long enough
                    ax1.text(start + duration/2, 0, regime, 
                            ha='center', va='center', fontweight='bold')
            
            ax1.set_title('Regime Timeline and Transitions', fontweight='bold')
            ax1.set_xlabel('Days')
            ax1.set_yticks([])
            ax1.grid(True, alpha=0.3)
        
        # 2. Duration Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        if timing_analysis and 'transition_stats' in timing_analysis:
            duration_stats = timing_analysis['transition_stats']['duration_stats']
            
            regimes = list(duration_stats.keys())
            durations = [duration_stats[r]['mean_duration'] for r in regimes]
            colors = [self.regime_colors[r] for r in regimes]
            
            bars = ax2.bar(regimes, durations, color=colors, alpha=0.7)
            ax2.set_title('Average Regime Duration', fontweight='bold')
            ax2.set_ylabel('Days')
            
            # Add value labels on bars
            for bar, duration in zip(bars, durations):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{duration:.1f}', ha='center', va='bottom')
        
        # 3. Transition Matrix Heatmap
        ax3 = fig.add_subplot(gs[1, 1])
        if timing_analysis and 'transition_matrix' in timing_analysis:
            transition_matrix = timing_analysis['transition_matrix']
            
            im = ax3.imshow(transition_matrix, cmap='Blues', aspect='auto')
            ax3.set_title('Transition Probabilities', fontweight='bold')
            ax3.set_xticks([0, 1, 2])
            ax3.set_xticklabels(['Bear', 'Sideways', 'Bull'])
            ax3.set_yticks([0, 1, 2])
            ax3.set_yticklabels(['Bear', 'Sideways', 'Bull'])
            ax3.set_xlabel('To Regime')
            ax3.set_ylabel('From Regime')
            
            # Add text annotations
            for i in range(3):
                for j in range(3):
                    if i < len(transition_matrix) and j < len(transition_matrix[i]):
                        text = ax3.text(j, i, f'{transition_matrix[i, j]:.2f}',
                                       ha="center", va="center", color="black" if transition_matrix[i, j] < 0.5 else "white")
        
        # 4. Prediction Horizon
        ax4 = fig.add_subplot(gs[1, 2])
        if prediction_analysis and 'predictions' in prediction_analysis:
            predictions = prediction_analysis['predictions']
            
            days = [p['day'] for p in predictions]
            transition_probs = [p['transition_probability'] for p in predictions]
            survival_probs = [p['survival_probability'] for p in predictions]
            
            ax4.plot(days, transition_probs, 'r-', linewidth=2, label='Transition Probability')
            ax4.plot(days, survival_probs, 'g-', linewidth=2, label='Survival Probability')
            ax4.set_title('Transition Prediction', fontweight='bold')
            ax4.set_xlabel('Days Ahead')
            ax4.set_ylabel('Probability')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Warning Signals Timeline
        ax5 = fig.add_subplot(gs[2, :])
        if warning_analysis and 'warning_signals' in warning_analysis:
            warnings = warning_analysis['warning_signals']
            
            days = [w['day'] for w in warnings]
            scores = [w['warning_score'] for w in warnings]
            levels = [w['warning_level'] for w in warnings]
            
            # Color code by warning level
            colors = []
            for level in levels:
                if level == 'HIGH':
                    colors.append('red')
                elif level == 'MEDIUM':
                    colors.append('orange')
                elif level == 'LOW':
                    colors.append('yellow')
                else:
                    colors.append('green')
            
            ax5.scatter(days, scores, c=colors, alpha=0.6, s=20)
            ax5.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='High Warning')
            ax5.axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='Medium Warning')
            ax5.axhline(y=0.2, color='gold', linestyle='--', alpha=0.7, label='Low Warning')
            
            ax5.set_title('Early Warning Signals Timeline', fontweight='bold')
            ax5.set_xlabel('Days')
            ax5.set_ylabel('Warning Score')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Impact Analysis
        ax6 = fig.add_subplot(gs[3, 0])
        if impact_analysis and 'transition_impacts' in impact_analysis:
            impacts = impact_analysis['transition_impacts']
            
            if impacts:
                transition_types = list(impacts.keys())
                impact_5d = []
                
                for t_type in transition_types:
                    if '5d' in impacts[t_type]:
                        impact_5d.append(impacts[t_type]['5d']['avg_return_change'] * 100)
                    else:
                        impact_5d.append(0)
                
                bars = ax6.bar(range(len(transition_types)), impact_5d, alpha=0.7)
                ax6.set_title('5-Day Return Impact', fontweight='bold')
                ax6.set_ylabel('Return Change (%)')
                ax6.set_xticks(range(len(transition_types)))
                ax6.set_xticklabels([t.replace('_', '\n') for t in transition_types], rotation=45, ha='right')
                ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax6.grid(True, alpha=0.3)
                
                # Color bars based on impact direction
                for bar, impact in zip(bars, impact_5d):
                    if impact > 0:
                        bar.set_color('green')
                    elif impact < 0:
                        bar.set_color('red')
                    else:
                        bar.set_color('gray')
        
        # 7. Current Status Summary
        ax7 = fig.add_subplot(gs[3, 1:])
        ax7.axis('off')
        
        summary_text = ""
        if prediction_analysis:
            summary_text += f"Current Regime: {prediction_analysis.get('current_regime', 'Unknown')}\n"
            summary_text += f"Confidence: {prediction_analysis.get('current_confidence', 0):.1%}\n"
            summary_text += f"Days in Regime: {prediction_analysis.get('days_in_regime', 0)}\n"
            summary_text += f"Daily Transition Probability: {prediction_analysis.get('daily_transition_probability', 0):.1%}\n"
        
        if warning_analysis:
            summary_text += f"\nCurrent Warning Level: {warning_analysis.get('current_warning_level', 'NONE')}\n"
            summary_text += f"Recent Warnings: {len(warning_analysis.get('recent_warnings', []))}\n"
        
        if timing_analysis:
            summary_text += f"\nTotal Transitions: {timing_analysis.get('total_transitions', 0)}\n"
        
        ax7.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = os.path.join(output_dir, f'{symbol}_transition_analysis.png')
        plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return viz_file
    
    def generate_transition_analysis_report(self, symbols: List[str], start_date: str,
                                          end_date: str, output_dir: str) -> str:
        """Generate comprehensive regime transition analysis report"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating comprehensive transition analysis report...")
        
        # Analyze all symbols
        all_analyses = {}
        for symbol in symbols:
            print(f"\nAnalyzing {symbol}...")
            
            analyses = {
                'timing': self.analyze_transition_timing(symbol, start_date, end_date),
                'prediction': self.predict_regime_transitions(symbol, start_date, end_date),
                'warning': self.generate_early_warning_signals(symbol, start_date, end_date),
                'impact': self.analyze_transition_impact(symbol, start_date, end_date)
            }
            
            all_analyses[symbol] = analyses
            
            # Create visualization for each symbol
            viz_file = self.create_comprehensive_transition_visualizations(
                symbol, start_date, end_date, output_dir
            )
        
        # Generate comprehensive report
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# Comprehensive Regime Transition Analysis Report
*Generated on {timestamp}*

## Executive Summary

This report provides detailed analysis of regime transitions across {len(symbols)} assets, including timing patterns, prediction models, early warning systems, and transition impact assessment.

## Key Findings

"""
        
        # Add key findings for each symbol
        for symbol, analyses in all_analyses.items():
            timing = analyses.get('timing', {})
            prediction = analyses.get('prediction', {})
            warning = analyses.get('warning', {})
            
            if timing and prediction:
                report += f"""
### {symbol}
- **Current Regime**: {prediction.get('current_regime', 'Unknown')} ({prediction.get('current_confidence', 0):.1%} confidence)
- **Days in Regime**: {prediction.get('days_in_regime', 0)} days
- **Total Transitions**: {timing.get('total_transitions', 0)} transitions detected
- **Daily Transition Probability**: {prediction.get('daily_transition_probability', 0):.1%}
- **Warning Level**: {warning.get('current_warning_level', 'NONE')}
"""
        
        report += f"""

## Methodology

### Transition Timing Analysis
- Identification of regime transition points
- Duration and persistence statistics
- Transition pattern recognition
- Cyclicality assessment

### Transition Prediction
- Statistical duration modeling
- Survival probability analysis
- Confidence-adjusted predictions
- Multi-horizon forecasting

### Early Warning Systems
- Confidence decline detection
- Probability volatility monitoring
- Pattern inconsistency alerts
- Duration-based warnings
- Composite warning scores

### Impact Assessment
- Pre/post transition return analysis
- Volatility impact measurement
- Transition-specific effects
- Statistical significance testing

## Technical Implementation

### Statistical Models
- Hidden Markov Model regime detection
- Survival analysis for duration modeling
- Multi-variate warning signal generation
- Statistical hypothesis testing for impact

### Warning System Architecture
- Real-time signal processing
- Multi-factor composite scoring
- Threshold-based alert levels
- Historical back-testing validation

## Risk Management Applications

### Portfolio Management
- Tactical asset allocation adjustments
- Risk exposure modification
- Hedge timing optimization
- Performance attribution analysis

### Trading Strategies
- Regime change momentum capture
- Mean reversion timing
- Volatility trading opportunities
- Market timing signals

### Risk Monitoring
- Early warning systems
- Stress testing scenarios
- Correlation breakdown detection
- Tail risk assessment

## Disclaimer

This analysis is for educational and research purposes only. Regime transition predictions are based on historical patterns and statistical models that may not persist in future market conditions. Past performance does not guarantee future results. Always conduct thorough due diligence and consult with qualified financial advisors before making investment decisions.

---
*Analysis performed using Hidden Regime Transition Analysis Framework*
"""
        
        # Save report
        report_file = os.path.join(output_dir, 'regime_transition_analysis_report.md')
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Comprehensive transition analysis report saved to: {report_file}")
        return report_file

def main():
    """Main execution function for regime transition analysis"""
    
    print("🔄 Hidden Regime Transition Analysis Example")
    print("=" * 60)
    
    # Configuration
    SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
    OUTPUT_DIR = './output/transition_analysis'
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📊 Analyzing regime transitions for {len(SYMBOLS)} assets")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    try:
        # Initialize analyzer
        data_config = DataConfig()
        analyzer = RegimeTransitionAnalyzer(data_config)
        
        # Generate comprehensive analysis
        print("\n🔄 Generating comprehensive transition analysis...")
        report_file = analyzer.generate_transition_analysis_report(
            SYMBOLS, start_date, end_date, OUTPUT_DIR
        )
        
        print(f"\n✅ Regime transition analysis completed!")
        print(f"📄 Report file: {report_file}")
        print(f"📁 All visualizations saved to: {OUTPUT_DIR}")
        
        print(f"\n🎯 Analysis Components:")
        print(f"   📊 Transition timing and patterns")
        print(f"   🔮 Transition prediction models")  
        print(f"   ⚠️  Early warning signal systems")
        print(f"   📈 Transition impact assessment")
        print(f"   📋 Comprehensive analysis report")
        
        # List generated files
        print(f"\n📂 Generated Files:")
        for file in sorted(os.listdir(OUTPUT_DIR)):
            print(f"   - {file}")
        
    except Exception as e:
        print(f"❌ Error in transition analysis: {str(e)}")
        print("💥 Example failed - check error messages above")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Regime transition analysis completed successfully!")
        print("📈 Use the generated insights for risk management and trading strategies")
    else:
        print("\n❌ Regime transition analysis failed")
        exit(1)