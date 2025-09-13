"""
Screening Criteria Definitions

Comprehensive screening criteria for identifying interesting stocks based on
regime changes, momentum patterns, volatility characteristics, and technical
indicators. Enables complex multi-factor screening strategies.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class ScreeningCriteria:
    """
    Definition of screening criteria for stock filtering.
    
    Supports multiple criteria types including regime-based, momentum-based,
    volatility-based, and custom function-based filters.
    """
    
    name: str
    description: str
    criteria_type: str  # 'regime', 'momentum', 'volatility', 'technical', 'custom'
    
    # Regime-based criteria
    target_regime: Optional[int] = None
    min_confidence: Optional[float] = None
    max_days_in_regime: Optional[int] = None
    min_days_in_regime: Optional[int] = None
    
    # Momentum criteria
    min_return_20d: Optional[float] = None
    max_return_20d: Optional[float] = None
    min_return_60d: Optional[float] = None
    max_return_60d: Optional[float] = None
    
    # Volatility criteria
    min_volatility: Optional[float] = None
    max_volatility: Optional[float] = None
    volatility_percentile_min: Optional[float] = None
    volatility_percentile_max: Optional[float] = None
    
    # Technical criteria
    price_near_high: Optional[float] = None  # Within X% of 52-week high
    price_near_low: Optional[float] = None   # Within X% of 52-week low
    volume_spike: Optional[float] = None     # Volume X times average
    
    # Model quality criteria
    min_log_likelihood: Optional[float] = None
    require_convergence: bool = True
    min_data_points: int = 100
    
    # Custom function criteria
    custom_function: Optional[Callable] = None
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Logical operators
    combine_with_and: bool = True  # True for AND, False for OR


class RegimeChangeDetector:
    """
    Specialized detector for identifying regime changes and transitions.
    
    Provides various methods for detecting regime changes including recent
    transitions, regime stability analysis, and transition pattern matching.
    """
    
    def __init__(self, 
                 lookback_days: int = 30,
                 confidence_threshold: float = 0.7):
        """
        Initialize regime change detector.
        
        Args:
            lookback_days: Days to look back for regime analysis
            confidence_threshold: Minimum confidence for regime detection
        """
        self.lookback_days = lookback_days
        self.confidence_threshold = confidence_threshold
    
    def detect_recent_regime_change(self, analysis: Dict) -> Dict:
        """
        Detect if a stock has recently changed regimes.
        
        Args:
            analysis: Stock analysis results from batch processor
            
        Returns:
            Dictionary with regime change detection results
        """
        current_regime = analysis['current_regime']
        regime_stats = analysis['regime_analysis']['regime_statistics']['regime_stats']
        
        days_in_regime = current_regime['days_in_regime']
        confidence = current_regime['confidence']
        regime_id = current_regime['regime']
        
        # Check if recent change
        is_recent_change = days_in_regime <= self.lookback_days
        
        # Check confidence
        is_confident = confidence >= self.confidence_threshold
        
        # Analyze regime characteristics
        regime_mean = regime_stats[regime_id]['mean_return']
        regime_vol = regime_stats[regime_id]['volatility']
        
        # Classify regime type
        if regime_mean > 0.005:  # > 0.5% daily
            regime_type = 'bullish'
        elif regime_mean < -0.005:  # < -0.5% daily
            regime_type = 'bearish'
        else:
            regime_type = 'neutral'
        
        return {
            'has_recent_change': is_recent_change and is_confident,
            'days_in_regime': days_in_regime,
            'confidence': confidence,
            'regime_type': regime_type,
            'regime_characteristics': {
                'mean_return': regime_mean,
                'volatility': regime_vol,
                'expected_duration': regime_stats[regime_id]['avg_duration']
            },
            'change_significance': self._calculate_change_significance(analysis)
        }
    
    def _calculate_change_significance(self, analysis: Dict) -> float:
        """Calculate significance score for regime change."""
        current_regime = analysis['current_regime']
        regime_stats = analysis['regime_analysis']['regime_statistics']['regime_stats']
        
        confidence = current_regime['confidence']
        days_in_regime = current_regime['days_in_regime']
        regime_id = current_regime['regime']
        
        # Significance based on confidence and recency
        recency_score = max(0, 1 - (days_in_regime / self.lookback_days))
        confidence_score = confidence
        
        # Bonus for dramatic regime changes
        regime_mean = regime_stats[regime_id]['mean_return']
        drama_score = min(1.0, abs(regime_mean) / 0.02)  # 2% daily return threshold
        
        significance = (recency_score * 0.4 + confidence_score * 0.4 + drama_score * 0.2)
        
        return significance
    
    def detect_regime_instability(self, analysis: Dict) -> Dict:
        """
        Detect regime instability (frequent regime changes).
        
        Args:
            analysis: Stock analysis results
            
        Returns:
            Dictionary with instability analysis
        """
        regime_analysis = analysis['regime_analysis']
        
        # Calculate transition frequency
        transition_count = regime_analysis['regime_statistics'].get('transition_count', 0)
        data_points = analysis['data_points']
        
        transition_frequency = transition_count / data_points if data_points > 0 else 0
        
        # High frequency indicates instability
        is_unstable = transition_frequency > 0.1  # More than 10% of days are transitions
        
        return {
            'is_unstable': is_unstable,
            'transition_frequency': transition_frequency,
            'transition_count': transition_count,
            'stability_score': 1 - min(1.0, transition_frequency / 0.2)  # Normalized
        }


def apply_screening_criteria(analysis: Dict, criteria: ScreeningCriteria) -> Tuple[bool, Dict]:
    """
    Apply screening criteria to a stock analysis result.
    
    Args:
        analysis: Stock analysis from batch processor
        criteria: Screening criteria to apply
        
    Returns:
        Tuple of (passes_criteria, details_dict)
    """
    details = {
        'criteria_name': criteria.name,
        'passes': False,
        'sub_criteria': {}
    }
    
    passes_list = []
    
    try:
        current_regime = analysis['current_regime']
        recent_metrics = analysis['recent_metrics']
        model_info = analysis['hmm_model_info']
        
        # Regime-based criteria
        if criteria.criteria_type in ['regime', 'all']:
            regime_passes = _check_regime_criteria(current_regime, criteria, details)
            passes_list.append(regime_passes)
        
        # Momentum criteria
        if criteria.criteria_type in ['momentum', 'all']:
            momentum_passes = _check_momentum_criteria(recent_metrics, criteria, details)
            passes_list.append(momentum_passes)
        
        # Volatility criteria
        if criteria.criteria_type in ['volatility', 'all']:
            volatility_passes = _check_volatility_criteria(recent_metrics, criteria, details)
            passes_list.append(volatility_passes)
        
        # Model quality criteria
        if criteria.criteria_type in ['quality', 'all']:
            quality_passes = _check_quality_criteria(model_info, analysis, criteria, details)
            passes_list.append(quality_passes)
        
        # Custom function criteria
        if criteria.custom_function is not None:
            custom_passes = _check_custom_criteria(analysis, criteria, details)
            passes_list.append(custom_passes)
        
        # Combine results
        if criteria.combine_with_and:
            details['passes'] = all(passes_list) if passes_list else False
        else:
            details['passes'] = any(passes_list) if passes_list else False
            
    except Exception as e:
        details['error'] = str(e)
        details['passes'] = False
    
    return details['passes'], details


def _check_regime_criteria(current_regime: Dict, criteria: ScreeningCriteria, details: Dict) -> bool:
    """Check regime-based screening criteria."""
    passes = []
    
    if criteria.target_regime is not None:
        regime_match = current_regime['regime'] == criteria.target_regime
        details['sub_criteria']['target_regime'] = regime_match
        passes.append(regime_match)
    
    if criteria.min_confidence is not None:
        confidence_check = current_regime['confidence'] >= criteria.min_confidence
        details['sub_criteria']['min_confidence'] = confidence_check
        passes.append(confidence_check)
    
    if criteria.max_days_in_regime is not None:
        days_check = current_regime['days_in_regime'] <= criteria.max_days_in_regime
        details['sub_criteria']['max_days_in_regime'] = days_check
        passes.append(days_check)
    
    if criteria.min_days_in_regime is not None:
        min_days_check = current_regime['days_in_regime'] >= criteria.min_days_in_regime
        details['sub_criteria']['min_days_in_regime'] = min_days_check
        passes.append(min_days_check)
    
    return all(passes) if passes else True


def _check_momentum_criteria(recent_metrics: Dict, criteria: ScreeningCriteria, details: Dict) -> bool:
    """Check momentum-based screening criteria."""
    passes = []
    
    # Assuming we have 20-day returns (can be expanded)
    return_20d = recent_metrics.get('return_20d_annualized', 0)
    
    if criteria.min_return_20d is not None:
        return_check = return_20d >= criteria.min_return_20d
        details['sub_criteria']['min_return_20d'] = return_check
        passes.append(return_check)
    
    if criteria.max_return_20d is not None:
        return_check = return_20d <= criteria.max_return_20d
        details['sub_criteria']['max_return_20d'] = return_check
        passes.append(return_check)
    
    return all(passes) if passes else True


def _check_volatility_criteria(recent_metrics: Dict, criteria: ScreeningCriteria, details: Dict) -> bool:
    """Check volatility-based screening criteria."""
    passes = []
    
    volatility = recent_metrics.get('volatility_20d_annualized', 0)
    
    if criteria.min_volatility is not None:
        vol_check = volatility >= criteria.min_volatility
        details['sub_criteria']['min_volatility'] = vol_check
        passes.append(vol_check)
    
    if criteria.max_volatility is not None:
        vol_check = volatility <= criteria.max_volatility
        details['sub_criteria']['max_volatility'] = vol_check
        passes.append(vol_check)
    
    return all(passes) if passes else True


def _check_quality_criteria(model_info: Dict, analysis: Dict, criteria: ScreeningCriteria, details: Dict) -> bool:
    """Check model quality criteria."""
    passes = []
    
    if criteria.require_convergence:
        converged = model_info.get('converged', False)
        details['sub_criteria']['converged'] = converged
        passes.append(converged)
    
    if criteria.min_log_likelihood is not None:
        log_likelihood = model_info.get('log_likelihood', float('-inf'))
        likelihood_check = log_likelihood >= criteria.min_log_likelihood
        details['sub_criteria']['min_log_likelihood'] = likelihood_check
        passes.append(likelihood_check)
    
    if criteria.min_data_points > 0:
        data_points = analysis.get('data_points', 0)
        data_check = data_points >= criteria.min_data_points
        details['sub_criteria']['min_data_points'] = data_check
        passes.append(data_check)
    
    return all(passes) if passes else True


def _check_custom_criteria(analysis: Dict, criteria: ScreeningCriteria, details: Dict) -> bool:
    """Check custom function criteria."""
    try:
        result = criteria.custom_function(analysis, **criteria.custom_parameters)
        details['sub_criteria']['custom_function'] = result
        return bool(result)
    except Exception as e:
        details['sub_criteria']['custom_function_error'] = str(e)
        return False


# Predefined criteria factories

def create_regime_change_criteria(max_days: int = 5, 
                                min_confidence: float = 0.7,
                                name: str = "Recent Regime Change") -> ScreeningCriteria:
    """
    Create criteria for detecting recent regime changes.
    
    Args:
        max_days: Maximum days since regime change
        min_confidence: Minimum confidence in regime detection
        name: Name for the criteria
        
    Returns:
        ScreeningCriteria for regime changes
    """
    return ScreeningCriteria(
        name=name,
        description=f"Recent regime change (≤{max_days} days, ≥{min_confidence:.0%} confidence)",
        criteria_type='regime',
        max_days_in_regime=max_days,
        min_confidence=min_confidence,
        require_convergence=True
    )


def create_momentum_criteria(min_return: float = 0.1,
                           max_return: Optional[float] = None,
                           name: str = "Positive Momentum") -> ScreeningCriteria:
    """
    Create criteria for momentum-based filtering.
    
    Args:
        min_return: Minimum annualized return
        max_return: Maximum annualized return (optional)
        name: Name for the criteria
        
    Returns:
        ScreeningCriteria for momentum
    """
    return ScreeningCriteria(
        name=name,
        description=f"Momentum filter (return ≥{min_return:.1%})",
        criteria_type='momentum',
        min_return_20d=min_return,
        max_return_20d=max_return
    )


def create_volatility_criteria(min_vol: Optional[float] = None,
                             max_vol: Optional[float] = None,
                             name: str = "Volatility Filter") -> ScreeningCriteria:
    """
    Create criteria for volatility-based filtering.
    
    Args:
        min_vol: Minimum volatility (annualized)
        max_vol: Maximum volatility (annualized)
        name: Name for the criteria
        
    Returns:
        ScreeningCriteria for volatility
    """
    desc_parts = []
    if min_vol is not None:
        desc_parts.append(f"vol ≥{min_vol:.1%}")
    if max_vol is not None:
        desc_parts.append(f"vol ≤{max_vol:.1%}")
    
    description = f"Volatility filter ({', '.join(desc_parts)})"
    
    return ScreeningCriteria(
        name=name,
        description=description,
        criteria_type='volatility',
        min_volatility=min_vol,
        max_volatility=max_vol
    )


def create_bull_regime_criteria(min_confidence: float = 0.8,
                               max_days: int = 10) -> ScreeningCriteria:
    """Create criteria for detecting recent bull regime entries."""
    def is_bull_regime(analysis: Dict) -> bool:
        regime_stats = analysis['regime_analysis']['regime_statistics']['regime_stats']
        current_regime_id = analysis['current_regime']['regime']
        regime_mean = regime_stats[current_regime_id]['mean_return']
        return regime_mean > 0.005  # > 0.5% daily return
    
    return ScreeningCriteria(
        name="Bull Regime Entry",
        description=f"Recent entry into bull regime (≤{max_days} days, ≥{min_confidence:.0%} confidence)",
        criteria_type='custom',
        max_days_in_regime=max_days,
        min_confidence=min_confidence,
        custom_function=is_bull_regime
    )


def create_bear_regime_criteria(min_confidence: float = 0.8,
                               max_days: int = 10) -> ScreeningCriteria:
    """Create criteria for detecting recent bear regime entries."""
    def is_bear_regime(analysis: Dict) -> bool:
        regime_stats = analysis['regime_analysis']['regime_statistics']['regime_stats']
        current_regime_id = analysis['current_regime']['regime']
        regime_mean = regime_stats[current_regime_id]['mean_return']
        return regime_mean < -0.005  # < -0.5% daily return
    
    return ScreeningCriteria(
        name="Bear Regime Entry",
        description=f"Recent entry into bear regime (≤{max_days} days, ≥{min_confidence:.0%} confidence)",
        criteria_type='custom',
        max_days_in_regime=max_days,
        min_confidence=min_confidence,
        custom_function=is_bear_regime
    )


def create_high_confidence_criteria(min_confidence: float = 0.9) -> ScreeningCriteria:
    """Create criteria for high-confidence regime detection."""
    return ScreeningCriteria(
        name="High Confidence",
        description=f"High confidence regime detection (≥{min_confidence:.0%})",
        criteria_type='regime',
        min_confidence=min_confidence,
        require_convergence=True
    )


def create_quality_criteria(min_data_points: int = 200,
                          require_convergence: bool = True) -> ScreeningCriteria:
    """Create criteria for model quality filtering."""
    return ScreeningCriteria(
        name="Quality Filter",
        description=f"Quality filtering (≥{min_data_points} data points, convergence required)",
        criteria_type='quality',
        min_data_points=min_data_points,
        require_convergence=require_convergence
    )


def create_combined_criteria(*criteria_list: ScreeningCriteria, 
                           use_and: bool = True,
                           name: str = "Combined Criteria") -> ScreeningCriteria:
    """
    Combine multiple criteria into a single screening rule.
    
    Args:
        *criteria_list: Multiple ScreeningCriteria to combine
        use_and: Whether to use AND (True) or OR (False) logic
        name: Name for the combined criteria
        
    Returns:
        Combined ScreeningCriteria
    """
    def combined_function(analysis: Dict) -> bool:
        results = []
        for criteria in criteria_list:
            passes, _ = apply_screening_criteria(analysis, criteria)
            results.append(passes)
        
        if use_and:
            return all(results)
        else:
            return any(results)
    
    descriptions = [c.description for c in criteria_list]
    combined_description = f" {'AND' if use_and else 'OR'} ".join(descriptions)
    
    return ScreeningCriteria(
        name=name,
        description=combined_description,
        criteria_type='custom',
        custom_function=combined_function
    )