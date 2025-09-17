"""
Comprehensive performance metrics for regime analysis.

Provides detailed statistical analysis and performance measurement capabilities
for HMM regime detection including transition analysis, duration statistics,
and comparative performance metrics.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

from ..utils.exceptions import ValidationError


class RegimePerformanceAnalyzer:
    """
    Comprehensive performance analyzer for regime detection results.
    
    Provides detailed statistical analysis including regime characteristics,
    transition patterns, duration analysis, and performance comparisons.
    """
    
    def __init__(self):
        """Initialize the performance analyzer."""
        self.regime_stats_cache = {}
        self.transition_stats_cache = {}
    
    def analyze_regime_performance(self, 
                                 analysis_results: pd.DataFrame,
                                 raw_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Comprehensive performance analysis of regime detection results.
        
        Args:
            analysis_results: DataFrame with regime analysis results
            raw_data: Optional raw price data for return calculations
            
        Returns:
            Dictionary with comprehensive performance metrics
        """
        if analysis_results.empty:
            raise ValidationError("Analysis results cannot be empty")
        
        performance_report = {}
        
        # Basic regime statistics
        performance_report['regime_distribution'] = self._analyze_regime_distribution(analysis_results)
        
        # Transition analysis
        performance_report['transition_analysis'] = self._analyze_regime_transitions(analysis_results)
        
        # Duration analysis
        performance_report['duration_analysis'] = self._analyze_regime_durations(analysis_results)
        
        # Confidence analysis
        performance_report['confidence_analysis'] = self._analyze_confidence_metrics(analysis_results)
        
        # Performance by regime (if return data available)
        if raw_data is not None and 'close' in raw_data.columns:
            performance_report['regime_performance'] = self._analyze_regime_returns(
                analysis_results, raw_data
            )
        
        # Model stability metrics
        performance_report['stability_metrics'] = self._analyze_model_stability(analysis_results)
        
        # Summary statistics
        performance_report['summary'] = self._generate_performance_summary(performance_report)
        
        return performance_report
    
    def _analyze_regime_distribution(self, analysis_results: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the distribution of regime states."""
        regime_col = 'predicted_state' if 'predicted_state' in analysis_results.columns else None
        regime_name_col = 'regime_name' if 'regime_name' in analysis_results.columns else None
        
        if regime_col is None:
            return {'error': 'No regime state column found'}
        
        distribution = {}
        
        # State distribution
        state_counts = analysis_results[regime_col].value_counts().sort_index()
        total_periods = len(analysis_results)
        
        distribution['state_counts'] = state_counts.to_dict()
        distribution['state_percentages'] = (state_counts / total_periods * 100).to_dict()
        distribution['total_periods'] = total_periods
        distribution['unique_states'] = len(state_counts)
        
        # Regime name distribution (if available)
        if regime_name_col is not None:
            name_counts = analysis_results[regime_name_col].value_counts()
            distribution['regime_name_counts'] = name_counts.to_dict()
            distribution['regime_name_percentages'] = (name_counts / total_periods * 100).to_dict()
        
        # Dominance analysis
        max_state_pct = max(distribution['state_percentages'].values()) if distribution['state_percentages'] else 0
        distribution['regime_dominance'] = {
            'most_frequent_state': max(distribution['state_percentages'], 
                                     key=distribution['state_percentages'].get) if distribution['state_percentages'] else None,
            'dominance_percentage': max_state_pct,
            'is_dominated': max_state_pct > 60.0,  # One regime dominates >60% of time
            'balance_score': 1.0 - (max_state_pct / 100.0)  # Higher = more balanced
        }
        
        return distribution
    
    def _analyze_regime_transitions(self, analysis_results: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regime transition patterns."""
        regime_col = 'predicted_state'
        if regime_col not in analysis_results.columns:
            return {'error': 'No regime state column found'}
        
        states = analysis_results[regime_col]
        
        # Calculate transitions
        transitions = []
        transition_matrix = {}
        
        unique_states = sorted(states.unique())
        
        # Initialize transition matrix
        for from_state in unique_states:
            transition_matrix[from_state] = {to_state: 0 for to_state in unique_states}
        
        # Count transitions
        total_transitions = 0
        for i in range(1, len(states)):
            from_state = states.iloc[i-1]
            to_state = states.iloc[i]
            
            transitions.append({
                'from_state': from_state,
                'to_state': to_state,
                'timestamp': states.index[i],
                'is_change': from_state != to_state
            })
            
            transition_matrix[from_state][to_state] += 1
            if from_state != to_state:
                total_transitions += 1
        
        # Convert counts to probabilities
        transition_probabilities = {}
        for from_state in unique_states:
            total_from = sum(transition_matrix[from_state].values())
            if total_from > 0:
                transition_probabilities[from_state] = {
                    to_state: count / total_from 
                    for to_state, count in transition_matrix[from_state].items()
                }
            else:
                transition_probabilities[from_state] = {
                    to_state: 0.0 for to_state in unique_states
                }
        
        # Persistence analysis
        persistence = {}
        for state in unique_states:
            if state in transition_probabilities:
                persistence[state] = transition_probabilities[state].get(state, 0.0)
        
        return {
            'total_transitions': total_transitions,
            'transition_rate': total_transitions / len(states) if len(states) > 0 else 0,
            'transition_matrix_counts': transition_matrix,
            'transition_probabilities': transition_probabilities,
            'persistence_by_state': persistence,
            'average_persistence': np.mean(list(persistence.values())) if persistence else 0,
            'most_persistent_state': max(persistence, key=persistence.get) if persistence else None,
            'least_persistent_state': min(persistence, key=persistence.get) if persistence else None,
            'stability_score': np.mean(list(persistence.values())) if persistence else 0  # Higher = more stable
        }
    
    def _analyze_regime_durations(self, analysis_results: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regime duration patterns."""
        regime_col = 'predicted_state'
        if regime_col not in analysis_results.columns:
            return {'error': 'No regime state column found'}
        
        states = analysis_results[regime_col]
        
        # Calculate regime episodes
        episodes = []
        current_state = states.iloc[0]
        current_duration = 1
        start_idx = 0
        
        for i in range(1, len(states)):
            if states.iloc[i] == current_state:
                current_duration += 1
            else:
                # End of current episode
                episodes.append({
                    'state': current_state,
                    'duration': current_duration,
                    'start_index': start_idx,
                    'end_index': i - 1,
                    'start_date': states.index[start_idx],
                    'end_date': states.index[i - 1]
                })
                
                # Start new episode
                current_state = states.iloc[i]
                current_duration = 1
                start_idx = i
        
        # Add final episode
        episodes.append({
            'state': current_state,
            'duration': current_duration,
            'start_index': start_idx,
            'end_index': len(states) - 1,
            'start_date': states.index[start_idx],
            'end_date': states.index[-1]
        })
        
        # Analyze durations by state
        duration_stats = {}
        unique_states = sorted(states.unique())
        
        for state in unique_states:
            state_episodes = [ep for ep in episodes if ep['state'] == state]
            durations = [ep['duration'] for ep in state_episodes]
            
            if durations:
                duration_stats[state] = {
                    'count': len(durations),
                    'mean_duration': np.mean(durations),
                    'median_duration': np.median(durations),
                    'std_duration': np.std(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'total_time': sum(durations),
                    'percentage_of_time': sum(durations) / len(states) * 100
                }
            else:
                duration_stats[state] = {
                    'count': 0,
                    'mean_duration': 0,
                    'median_duration': 0,
                    'std_duration': 0,
                    'min_duration': 0,
                    'max_duration': 0,
                    'total_time': 0,
                    'percentage_of_time': 0
                }
        
        # Overall duration statistics
        all_durations = [ep['duration'] for ep in episodes]
        
        return {
            'total_episodes': len(episodes),
            'episodes': episodes,
            'duration_stats_by_state': duration_stats,
            'overall_duration_stats': {
                'mean': np.mean(all_durations) if all_durations else 0,
                'median': np.median(all_durations) if all_durations else 0,
                'std': np.std(all_durations) if all_durations else 0,
                'min': min(all_durations) if all_durations else 0,
                'max': max(all_durations) if all_durations else 0
            },
            'most_persistent_regime': max(duration_stats, key=lambda x: duration_stats[x]['mean_duration']) if duration_stats else None,
            'least_persistent_regime': min(duration_stats, key=lambda x: duration_stats[x]['mean_duration']) if duration_stats else None
        }
    
    def _analyze_confidence_metrics(self, analysis_results: pd.DataFrame) -> Dict[str, Any]:
        """Analyze model confidence metrics."""
        conf_col = 'confidence'
        if conf_col not in analysis_results.columns:
            return {'error': 'No confidence column found'}
        
        confidence = analysis_results[conf_col].dropna()
        
        if len(confidence) == 0:
            return {'error': 'No valid confidence values found'}
        
        # Overall confidence statistics
        conf_stats = {
            'mean': confidence.mean(),
            'median': confidence.median(),
            'std': confidence.std(),
            'min': confidence.min(),
            'max': confidence.max(),
            'q25': confidence.quantile(0.25),
            'q75': confidence.quantile(0.75)
        }
        
        # Confidence by regime
        regime_confidence = {}
        if 'predicted_state' in analysis_results.columns:
            for state in analysis_results['predicted_state'].unique():
                state_mask = analysis_results['predicted_state'] == state
                state_conf = analysis_results.loc[state_mask, conf_col].dropna()
                
                if len(state_conf) > 0:
                    regime_confidence[state] = {
                        'mean': state_conf.mean(),
                        'median': state_conf.median(),
                        'std': state_conf.std(),
                        'count': len(state_conf)
                    }
        
        # Confidence quality assessment
        high_conf_threshold = 0.7
        medium_conf_threshold = 0.5
        
        conf_quality = {
            'high_confidence_pct': (confidence > high_conf_threshold).mean() * 100,
            'medium_confidence_pct': ((confidence > medium_conf_threshold) & 
                                    (confidence <= high_conf_threshold)).mean() * 100,
            'low_confidence_pct': (confidence <= medium_conf_threshold).mean() * 100,
            'average_confidence_score': conf_stats['mean'],
            'confidence_stability': 1.0 - (conf_stats['std'] / conf_stats['mean']) if conf_stats['mean'] > 0 else 0
        }
        
        return {
            'overall_stats': conf_stats,
            'confidence_by_regime': regime_confidence,
            'confidence_quality': conf_quality,
            'most_confident_regime': max(regime_confidence, 
                                       key=lambda x: regime_confidence[x]['mean']) if regime_confidence else None,
            'least_confident_regime': min(regime_confidence, 
                                        key=lambda x: regime_confidence[x]['mean']) if regime_confidence else None
        }
    
    def _analyze_regime_returns(self, analysis_results: pd.DataFrame, raw_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze returns performance by regime."""
        if 'close' not in raw_data.columns:
            return {'error': 'No close price data available'}
        
        # Calculate returns
        prices = raw_data['close']
        returns = prices.pct_change().dropna()
        
        # Align with analysis results
        common_index = analysis_results.index.intersection(returns.index)
        if len(common_index) == 0:
            return {'error': 'No common dates between analysis and price data'}
        
        aligned_analysis = analysis_results.loc[common_index]
        aligned_returns = returns.loc[common_index]
        
        # Calculate performance by regime
        regime_performance = {}
        
        for state in aligned_analysis['predicted_state'].unique():
            state_mask = aligned_analysis['predicted_state'] == state
            state_returns = aligned_returns[state_mask]
            
            if len(state_returns) > 0:
                regime_performance[state] = {
                    'mean_return': state_returns.mean(),
                    'std_return': state_returns.std(),
                    'sharpe_ratio': state_returns.mean() / state_returns.std() if state_returns.std() > 0 else 0,
                    'total_return': (1 + state_returns).prod() - 1,
                    'positive_days_pct': (state_returns > 0).mean() * 100,
                    'max_daily_return': state_returns.max(),
                    'min_daily_return': state_returns.min(),
                    'volatility_annualized': state_returns.std() * np.sqrt(252),
                    'return_annualized': state_returns.mean() * 252,
                    'days_count': len(state_returns)
                }
        
        # Overall performance comparison
        overall_return = aligned_returns.mean()
        overall_vol = aligned_returns.std()
        
        performance_summary = {
            'best_performing_regime': max(regime_performance, 
                                         key=lambda x: regime_performance[x]['mean_return']) if regime_performance else None,
            'worst_performing_regime': min(regime_performance, 
                                         key=lambda x: regime_performance[x]['mean_return']) if regime_performance else None,
            'most_volatile_regime': max(regime_performance, 
                                      key=lambda x: regime_performance[x]['std_return']) if regime_performance else None,
            'least_volatile_regime': min(regime_performance, 
                                       key=lambda x: regime_performance[x]['std_return']) if regime_performance else None,
            'overall_benchmark': {
                'mean_return': overall_return,
                'std_return': overall_vol,
                'sharpe_ratio': overall_return / overall_vol if overall_vol > 0 else 0
            }
        }
        
        return {
            'regime_performance': regime_performance,
            'performance_summary': performance_summary,
            'analysis_period': {
                'start_date': str(common_index.min()),
                'end_date': str(common_index.max()),
                'total_days': len(common_index)
            }
        }
    
    def _analyze_model_stability(self, analysis_results: pd.DataFrame) -> Dict[str, Any]:
        """Analyze model stability and consistency metrics."""
        if 'confidence' not in analysis_results.columns:
            return {'error': 'No confidence data available for stability analysis'}
        
        confidence = analysis_results['confidence'].dropna()
        
        # Confidence trend analysis
        if len(confidence) > 10:
            # Calculate rolling statistics
            window = min(20, len(confidence) // 4)
            rolling_mean = confidence.rolling(window=window).mean()
            rolling_std = confidence.rolling(window=window).std()
            
            # Trend analysis
            x = np.arange(len(confidence))
            trend_coef = np.polyfit(x, confidence, 1)[0] if len(confidence) > 1 else 0
            
            stability_metrics = {
                'confidence_trend': 'improving' if trend_coef > 0.001 else 'declining' if trend_coef < -0.001 else 'stable',
                'trend_coefficient': trend_coef,
                'rolling_volatility': rolling_std.mean() if not rolling_std.empty else 0,
                'consistency_score': 1.0 - (confidence.std() / confidence.mean()) if confidence.mean() > 0 else 0
            }
        else:
            stability_metrics = {
                'confidence_trend': 'insufficient_data',
                'trend_coefficient': 0,
                'rolling_volatility': 0,
                'consistency_score': 0
            }
        
        # Regime switching frequency
        if 'predicted_state' in analysis_results.columns:
            state_changes = (analysis_results['predicted_state'] != 
                           analysis_results['predicted_state'].shift(1)).sum()
            stability_metrics['regime_switching_frequency'] = state_changes / len(analysis_results)
            stability_metrics['stability_rating'] = self._get_stability_rating(stability_metrics)
        
        return stability_metrics
    
    def _get_stability_rating(self, metrics: Dict[str, Any]) -> str:
        """Generate overall stability rating."""
        score = 0
        
        # Consistency score (0-1, higher is better)
        score += metrics.get('consistency_score', 0) * 40
        
        # Low switching frequency is good (0-1, lower is better)
        switching_freq = metrics.get('regime_switching_frequency', 0.5)
        score += (1 - min(switching_freq * 2, 1)) * 30  # Penalize high switching
        
        # Low rolling volatility is good
        rolling_vol = metrics.get('rolling_volatility', 0.5)
        score += (1 - min(rolling_vol * 4, 1)) * 30  # Penalize high volatility
        
        if score >= 80:
            return 'Excellent'
        elif score >= 60:
            return 'Good'
        elif score >= 40:
            return 'Fair'
        else:
            return 'Poor'
    
    def _generate_performance_summary(self, performance_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of performance analysis."""
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'overall_quality': 'Unknown'
        }
        
        # Extract key metrics
        if 'regime_distribution' in performance_report:
            dist = performance_report['regime_distribution']
            summary['total_periods'] = dist.get('total_periods', 0)
            summary['unique_regimes'] = dist.get('unique_states', 0)
            summary['balance_score'] = dist.get('regime_dominance', {}).get('balance_score', 0)
        
        if 'confidence_analysis' in performance_report:
            conf = performance_report['confidence_analysis']
            summary['average_confidence'] = conf.get('overall_stats', {}).get('mean', 0)
            summary['high_confidence_periods'] = conf.get('confidence_quality', {}).get('high_confidence_pct', 0)
        
        if 'stability_metrics' in performance_report:
            stability = performance_report['stability_metrics']
            summary['stability_rating'] = stability.get('stability_rating', 'Unknown')
            summary['consistency_score'] = stability.get('consistency_score', 0)
        
        # Overall quality assessment
        quality_score = 0
        if summary.get('average_confidence', 0) > 0.6:
            quality_score += 25
        if summary.get('balance_score', 0) > 0.4:
            quality_score += 25
        if summary.get('consistency_score', 0) > 0.6:
            quality_score += 25
        if summary.get('high_confidence_periods', 0) > 50:
            quality_score += 25
        
        if quality_score >= 75:
            summary['overall_quality'] = 'Excellent'
        elif quality_score >= 50:
            summary['overall_quality'] = 'Good'
        elif quality_score >= 25:
            summary['overall_quality'] = 'Fair'
        else:
            summary['overall_quality'] = 'Poor'
        
        summary['quality_score'] = quality_score
        
        return summary