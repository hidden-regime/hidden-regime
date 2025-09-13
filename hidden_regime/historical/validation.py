"""
Historical Validation Framework

Validates HMM regime detection performance against known historical market events.
Provides metrics and analysis to assess how well the algorithm identifies
different market regimes during well-documented periods.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..models import HiddenMarkovModel, HMMConfig
from .datasets import MAJOR_MARKET_EVENTS, load_historical_period


def validate_historical_detection(
    event_name: str,
    ticker: str = "SPY",
    hmm_config: Optional[HMMConfig] = None,
    verbose: bool = True
) -> Dict:
    """
    Validate HMM regime detection against a historical market event.
    
    Loads historical data for the specified event, runs HMM regime detection,
    and compares results against expected regime characteristics.
    
    Args:
        event_name: Name of historical event from MAJOR_MARKET_EVENTS
        ticker: Stock ticker to analyze (default: SPY)
        hmm_config: HMM configuration (uses conservative defaults if None)
        verbose: Print detailed results
        
    Returns:
        Dictionary with validation results and performance metrics
        
    Example:
        >>> results = validate_historical_detection('2008_financial_crisis')
        >>> print(f"Regime accuracy: {results['regime_accuracy']:.2%}")
    """
    if event_name not in MAJOR_MARKET_EVENTS:
        available = list(MAJOR_MARKET_EVENTS.keys())
        raise ValueError(f"Unknown event '{event_name}'. Available: {available}")
    
    # Load historical data
    data_dict, event_info = load_historical_period(event_name, [ticker])
    
    if ticker not in data_dict:
        raise ValueError(f"Failed to load data for {ticker} during {event_name}")
    
    data = data_dict[ticker]
    returns = data['log_return'].dropna()
    dates = data['date'][1:]  # Match returns length
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATING: {event_info['name']}")
        print(f"{'='*60}")
        print(f"Ticker: {ticker}")
        print(f"Period: {event_info['start_date']} to {event_info['end_date']}")
        print(f"Data points: {len(returns)}")
        print(f"Expected regime: {event_info['expected_regime']}")
    
    # Configure HMM for historical validation
    if hmm_config is None:
        hmm_config = HMMConfig(
            n_states=3,
            max_iterations=100,
            tolerance=1e-4,
            random_seed=42,
            initialization_method='kmeans'
        )
    
    # Train HMM and get regime detection
    hmm = HiddenMarkovModel(config=hmm_config)
    hmm.fit(returns, verbose=False)
    
    states = hmm.predict(returns)
    regime_analysis = hmm.analyze_regimes(returns, dates)
    
    # Analyze detected regimes
    regime_stats = regime_analysis['regime_statistics']['regime_stats']
    
    # Identify which detected regime matches expected characteristics
    expected_regime = event_info['expected_regime']
    validation_results = {
        'event_name': event_info['name'],
        'event_key': event_name,
        'ticker': ticker,
        'period': f"{event_info['start_date']} to {event_info['end_date']}",
        'expected_regime': expected_regime,
        'data_points': len(returns),
        'hmm_config': hmm_config,
        'regime_analysis': regime_analysis,
        'validation_metrics': {}
    }
    
    # Calculate validation metrics
    metrics = _calculate_validation_metrics(
        states, returns, dates, event_info, regime_stats, verbose
    )
    validation_results['validation_metrics'] = metrics
    
    if verbose:
        _print_validation_summary(validation_results)
    
    return validation_results


def validate_regime_accuracy(
    states: np.ndarray,
    returns: pd.Series,
    expected_regime_type: str,
    event_characteristics: Dict
) -> Dict:
    """
    Calculate accuracy metrics for regime detection against expected characteristics.
    
    Args:
        states: Detected regime sequence
        returns: Log returns time series
        expected_regime_type: Expected regime type ('bull', 'bear', 'sideways', 'crisis')
        event_characteristics: Expected characteristics from event metadata
        
    Returns:
        Dictionary with accuracy metrics and regime analysis
    """
    unique_states = np.unique(states)
    n_regimes = len(unique_states)
    
    # Calculate statistics for each detected regime
    regime_stats = {}
    for state in unique_states:
        mask = states == state
        regime_returns = returns[mask]
        
        regime_stats[state] = {
            'frequency': np.sum(mask) / len(states),
            'mean_return': regime_returns.mean(),
            'volatility': regime_returns.std(),
            'total_days': np.sum(mask),
            'avg_duration': _calculate_average_duration(states, state)
        }
    
    # Find best matching regime based on expected characteristics
    best_match_state, match_score = _find_best_regime_match(
        regime_stats, expected_regime_type, event_characteristics
    )
    
    # Calculate accuracy metrics
    accuracy_metrics = {
        'regime_match_state': best_match_state,
        'regime_match_score': match_score,
        'regime_statistics': regime_stats,
        'dominant_regime': max(regime_stats.keys(), key=lambda k: regime_stats[k]['frequency']),
        'regime_consistency': _calculate_regime_consistency(states),
        'transition_frequency': _calculate_transition_frequency(states),
    }
    
    return accuracy_metrics


def run_comprehensive_historical_validation(
    events: Optional[List[str]] = None,
    tickers: Optional[List[str]] = None,
    hmm_config: Optional[HMMConfig] = None
) -> pd.DataFrame:
    """
    Run comprehensive validation across multiple historical events.
    
    Args:
        events: List of event names to validate (uses all if None)
        tickers: List of tickers to test (defaults to ['SPY'])
        hmm_config: HMM configuration (uses defaults if None)
        
    Returns:
        DataFrame with comprehensive validation results
    """
    if events is None:
        events = list(MAJOR_MARKET_EVENTS.keys())
    
    if tickers is None:
        tickers = ['SPY']
    
    results = []
    
    print(f"Running comprehensive historical validation...")
    print(f"Events: {len(events)}")
    print(f"Tickers: {tickers}")
    print(f"Total combinations: {len(events) * len(tickers)}")
    
    for event in events:
        for ticker in tickers:
            try:
                result = validate_historical_detection(
                    event, ticker, hmm_config, verbose=False
                )
                
                # Extract key metrics for summary
                metrics = result['validation_metrics']
                summary_row = {
                    'event_name': result['event_name'],
                    'event_key': result['event_key'],
                    'ticker': ticker,
                    'expected_regime': result['expected_regime'],
                    'data_points': result['data_points'],
                    'best_match_state': metrics.get('regime_match_state', -1),
                    'match_score': metrics.get('regime_match_score', 0.0),
                    'regime_consistency': metrics.get('regime_consistency', 0.0),
                    'dominant_regime_frequency': metrics.get('dominant_regime_frequency', 0.0),
                    'validation_passed': metrics.get('validation_passed', False),
                }
                
                results.append(summary_row)
                print(f"✓ {event} ({ticker}): Score {summary_row['match_score']:.3f}")
                
            except Exception as e:
                print(f"✗ {event} ({ticker}): {str(e)}")
                # Add failed result
                results.append({
                    'event_name': event,
                    'event_key': event, 
                    'ticker': ticker,
                    'expected_regime': 'unknown',
                    'data_points': 0,
                    'best_match_state': -1,
                    'match_score': 0.0,
                    'regime_consistency': 0.0,
                    'dominant_regime_frequency': 0.0,
                    'validation_passed': False,
                })
    
    results_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    overall_score = results_df['match_score'].mean()
    pass_rate = results_df['validation_passed'].mean()
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total validations: {len(results_df)}")
    print(f"Overall match score: {overall_score:.3f}")
    print(f"Validation pass rate: {pass_rate:.1%}")
    print(f"Best performing events:")
    
    top_events = results_df.nlargest(3, 'match_score')
    for _, row in top_events.iterrows():
        print(f"  {row['event_name']} ({row['ticker']}): {row['match_score']:.3f}")
    
    return results_df


def _calculate_validation_metrics(
    states: np.ndarray,
    returns: pd.Series,
    dates: pd.Series,
    event_info: Dict,
    regime_stats: Dict,
    verbose: bool = True
) -> Dict:
    """Calculate detailed validation metrics."""
    metrics = {}
    
    # Find best matching regime
    expected_regime = event_info['expected_regime']
    expected_chars = event_info.get('characteristics', {})
    
    best_match_state, match_score = _find_best_regime_match(
        regime_stats, expected_regime, expected_chars
    )
    
    metrics['regime_match_state'] = best_match_state
    metrics['regime_match_score'] = match_score
    metrics['regime_consistency'] = _calculate_regime_consistency(states)
    metrics['transition_frequency'] = _calculate_transition_frequency(states)
    
    # Get dominant regime info
    dominant_state = max(regime_stats.keys(), key=lambda k: regime_stats[k]['frequency'])
    metrics['dominant_regime_state'] = dominant_state
    metrics['dominant_regime_frequency'] = regime_stats[dominant_state]['frequency']
    
    # Validation pass/fail based on match score
    metrics['validation_passed'] = match_score > 0.6  # 60% threshold
    
    if verbose:
        print(f"\nRegime Detection Results:")
        for state, stats in regime_stats.items():
            marker = "👑" if state == dominant_state else "  "
            match_marker = "⭐" if state == best_match_state else "  "
            print(f"{marker}{match_marker} State {state}: "
                  f"{stats['frequency']:.1%} frequency, "
                  f"{stats['mean_return']:.4f} mean return, "
                  f"{stats['volatility']:.4f} volatility")
        
        print(f"\nValidation Metrics:")
        print(f"  Best match: State {best_match_state} (score: {match_score:.3f})")
        print(f"  Regime consistency: {metrics['regime_consistency']:.3f}")
        print(f"  Validation passed: {metrics['validation_passed']}")
    
    return metrics


def _find_best_regime_match(
    regime_stats: Dict,
    expected_regime_type: str,
    expected_chars: Dict
) -> Tuple[int, float]:
    """Find which detected regime best matches expected characteristics."""
    if not expected_chars:
        # No characteristics to match against, return dominant regime
        dominant_state = max(regime_stats.keys(), key=lambda k: regime_stats[k]['frequency'])
        return dominant_state, 0.5
    
    best_state = -1
    best_score = 0.0
    
    for state, stats in regime_stats.items():
        score = 0.0
        criteria = 0
        
        # Check mean return match
        if 'mean_return' in expected_chars:
            expected_return = expected_chars['mean_return']
            actual_return = stats['mean_return']
            
            # Score based on how well signs match and magnitude similarity
            if (expected_return > 0 and actual_return > 0) or \
               (expected_return < 0 and actual_return < 0):
                sign_score = 1.0
            else:
                sign_score = 0.0
            
            # Magnitude similarity (normalized)
            if expected_return != 0:
                magnitude_score = 1.0 - min(1.0, abs(actual_return - expected_return) / abs(expected_return))
            else:
                magnitude_score = 1.0 if abs(actual_return) < 0.005 else 0.0
            
            score += 0.7 * sign_score + 0.3 * magnitude_score
            criteria += 1
        
        # Check volatility match
        if 'volatility' in expected_chars:
            expected_vol = expected_chars['volatility']
            actual_vol = stats['volatility']
            
            vol_score = 1.0 - min(1.0, abs(actual_vol - expected_vol) / expected_vol)
            score += vol_score
            criteria += 1
        
        # Check frequency/dominance for certain regime types
        if expected_regime_type in ['bear', 'bull', 'crisis']:
            # These regimes should be dominant during their periods
            frequency_score = stats['frequency'] 
            score += frequency_score
            criteria += 1
        
        # Average score across criteria
        if criteria > 0:
            final_score = score / criteria
            if final_score > best_score:
                best_score = final_score
                best_state = state
    
    return best_state, best_score


def _calculate_regime_consistency(states: np.ndarray) -> float:
    """Calculate how consistent regime detection is (fewer transitions = more consistent)."""
    transitions = np.sum(states[1:] != states[:-1])
    max_possible_transitions = len(states) - 1
    
    # Consistency is inverse of transition rate
    consistency = 1.0 - (transitions / max_possible_transitions)
    return consistency


def _calculate_transition_frequency(states: np.ndarray) -> float:
    """Calculate transition frequency (transitions per observation)."""
    transitions = np.sum(states[1:] != states[:-1])
    return transitions / (len(states) - 1)


def _calculate_average_duration(states: np.ndarray, target_state: int) -> float:
    """Calculate average duration of a specific regime."""
    durations = []
    current_duration = 0
    in_target = False
    
    for state in states:
        if state == target_state:
            current_duration += 1
            in_target = True
        else:
            if in_target and current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
            in_target = False
    
    # Handle case where sequence ends in target state
    if in_target and current_duration > 0:
        durations.append(current_duration)
    
    return np.mean(durations) if durations else 0.0


def _print_validation_summary(results: Dict) -> None:
    """Print formatted validation summary."""
    metrics = results['validation_metrics']
    
    print(f"\n{'='*40}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*40}")
    print(f"Match Score: {metrics['regime_match_score']:.3f}")
    print(f"Consistency: {metrics['regime_consistency']:.3f}")
    print(f"Result: {'✅ PASSED' if metrics['validation_passed'] else '❌ FAILED'}")
    
    if metrics['validation_passed']:
        print(f"\n🎉 HMM successfully detected {results['expected_regime']} regime characteristics!")
    else:
        print(f"\n⚠️  HMM detection needs improvement for {results['expected_regime']} regime.")
        print("Consider adjusting HMM configuration or increasing data quality.")