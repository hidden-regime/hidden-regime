"""
HMM vs Technical Indicators Comparison Framework

Provides comprehensive comparison between HMM regime detection and traditional
technical indicators. Enables analysis of when each approach provides superior
signals and how they can be combined for improved performance.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..models import HiddenMarkovModel, HMMConfig
from .calculator import IndicatorCalculator
from .signals import IndicatorSignalGenerator, generate_composite_signal


def compare_hmm_vs_indicators(
    data: pd.DataFrame,
    hmm_config: Optional[HMMConfig] = None,
    indicator_config: Optional[Dict] = None,
    comparison_period: int = 252
) -> Dict:
    """
    Comprehensive comparison between HMM regime detection and technical indicators.
    
    Args:
        data: Stock price data with OHLCV columns
        hmm_config: HMM configuration (uses defaults if None)
        indicator_config: Technical indicator configuration
        comparison_period: Number of days for comparison analysis
        
    Returns:
        Dictionary with comprehensive comparison results
    """
    if len(data) < comparison_period:
        warnings.warn(f"Data length {len(data)} is less than comparison period {comparison_period}")
        comparison_period = len(data)
    
    # Prepare data
    analysis_data = data.tail(comparison_period).copy()
    returns = np.log(analysis_data['close'] / analysis_data['close'].shift(1)).dropna()
    
    # Configure HMM
    if hmm_config is None:
        hmm_config = HMMConfig(
            n_states=3,
            max_iterations=100,
            tolerance=1e-4,
            random_seed=42
        )
    
    # Train HMM and get regime analysis
    hmm = HiddenMarkovModel(config=hmm_config)
    hmm.fit(returns, verbose=False)
    
    states = hmm.predict(returns)
    dates = analysis_data.index[1:]  # Match returns length
    regime_analysis = hmm.analyze_regimes(returns, dates)
    
    # Calculate technical indicators
    calculator = IndicatorCalculator()
    indicators = calculator.calculate_all_indicators(analysis_data)
    
    # Generate indicator signals
    signals = generate_composite_signal(analysis_data, indicators, indicator_config)
    
    # Align data for comparison
    common_index = returns.index.intersection(signals.index)
    aligned_returns = returns.reindex(common_index)
    aligned_states = pd.Series(states, index=returns.index).reindex(common_index)
    aligned_signals = signals.reindex(common_index)
    
    # Performance comparison
    performance_results = _compare_performance(
        aligned_returns, aligned_states, aligned_signals, regime_analysis
    )
    
    # Timing comparison
    timing_results = _compare_timing(
        aligned_returns, aligned_states, aligned_signals
    )
    
    # Signal correlation analysis
    correlation_results = _analyze_signal_correlations(
        aligned_states, aligned_signals, regime_analysis
    )
    
    # Regime validation with indicators
    validation_results = _validate_regimes_with_indicators(
        aligned_returns, aligned_states, indicators.reindex(common_index), regime_analysis
    )
    
    return {
        'analysis_period': comparison_period,
        'data_points': len(common_index),
        'hmm_config': hmm_config,
        'regime_analysis': regime_analysis,
        'performance_comparison': performance_results,
        'timing_comparison': timing_results,
        'correlation_analysis': correlation_results,
        'regime_validation': validation_results,
        'aligned_data': {
            'returns': aligned_returns,
            'states': aligned_states,
            'signals': aligned_signals,
            'indicators': indicators.reindex(common_index)
        }
    }


def _compare_performance(
    returns: pd.Series,
    states: pd.Series,
    signals: pd.DataFrame,
    regime_analysis: Dict
) -> Dict:
    """Compare trading performance between HMM and indicator signals."""
    performance = {}
    
    # HMM-based strategy performance
    regime_stats = regime_analysis['regime_statistics']['regime_stats']
    
    # Create HMM trading signals based on regime characteristics
    hmm_signals = pd.Series(0, index=states.index)
    for state in range(len(regime_stats)):
        state_mask = states == state
        regime_mean = regime_stats[state]['mean_return']
        
        if regime_mean > 0.001:  # Bullish regime
            hmm_signals[state_mask] = 1
        elif regime_mean < -0.001:  # Bearish regime
            hmm_signals[state_mask] = -1
        # Neutral regimes stay at 0
    
    # Calculate returns for each strategy
    hmm_returns = returns * hmm_signals.shift(1)  # Signal from previous day
    
    strategies = {
        'hmm': hmm_returns,
        'buy_hold': returns
    }
    
    # Add indicator-based strategies
    if 'composite_signal' in signals.columns:
        composite_returns = returns * signals['composite_signal'].shift(1)
        strategies['composite_indicators'] = composite_returns
    
    if 'rsi_signal' in signals.columns:
        rsi_returns = returns * signals['rsi_signal'].shift(1)
        strategies['rsi'] = rsi_returns
    
    if 'macd_signal' in signals.columns:
        macd_returns = returns * signals['macd_signal'].shift(1)
        strategies['macd'] = macd_returns
    
    # Calculate performance metrics for each strategy
    for strategy_name, strategy_returns in strategies.items():
        strategy_returns = strategy_returns.dropna()
        if len(strategy_returns) > 0:
            performance[strategy_name] = {
                'total_return': strategy_returns.sum(),
                'annualized_return': strategy_returns.mean() * 252,
                'volatility': strategy_returns.std() * np.sqrt(252),
                'sharpe_ratio': (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0,
                'max_drawdown': _calculate_max_drawdown(strategy_returns.cumsum()),
                'win_rate': (strategy_returns > 0).mean(),
                'avg_win': strategy_returns[strategy_returns > 0].mean() if (strategy_returns > 0).any() else 0,
                'avg_loss': strategy_returns[strategy_returns < 0].mean() if (strategy_returns < 0).any() else 0,
                'profit_factor': abs(strategy_returns[strategy_returns > 0].sum() / strategy_returns[strategy_returns < 0].sum()) if (strategy_returns < 0).any() else np.inf
            }
    
    return performance


def _compare_timing(
    returns: pd.Series,
    states: pd.Series,
    signals: pd.DataFrame
) -> Dict:
    """Compare timing accuracy between HMM and indicator signals."""
    timing_results = {}
    
    # Identify significant market moves
    volatility = returns.rolling(window=20).std()
    significant_moves = abs(returns) > volatility * 2  # 2-sigma moves
    
    up_moves = returns > volatility * 2
    down_moves = returns < -volatility * 2
    
    # HMM regime transitions
    regime_changes = states != states.shift(1)
    
    # Analyze timing relative to significant moves
    timing_results['hmm_timing'] = {
        'regime_changes_before_up_moves': _count_signals_before_moves(regime_changes, up_moves, window=5),
        'regime_changes_before_down_moves': _count_signals_before_moves(regime_changes, down_moves, window=5),
        'avg_days_before_up_moves': _avg_days_before_moves(regime_changes, up_moves, window=10),
        'avg_days_before_down_moves': _avg_days_before_moves(regime_changes, down_moves, window=10)
    }
    
    # Indicator signal timing
    for signal_name in signals.columns:
        if signal_name.endswith('_signal'):
            signal_series = signals[signal_name]
            signal_changes = signal_series != signal_series.shift(1)
            
            timing_results[signal_name] = {
                'signal_changes_before_up_moves': _count_signals_before_moves(signal_changes, up_moves, window=5),
                'signal_changes_before_down_moves': _count_signals_before_moves(signal_changes, down_moves, window=5),
                'avg_days_before_up_moves': _avg_days_before_moves(signal_changes, up_moves, window=10),
                'avg_days_before_down_moves': _avg_days_before_moves(signal_changes, down_moves, window=10)
            }
    
    return timing_results


def _analyze_signal_correlations(
    states: pd.Series,
    signals: pd.DataFrame,
    regime_analysis: Dict
) -> Dict:
    """Analyze correlations between HMM regimes and indicator signals."""
    correlation_results = {}
    
    # Convert regimes to signals for comparison
    regime_stats = regime_analysis['regime_statistics']['regime_stats']
    regime_signals = pd.Series(0, index=states.index)
    
    for state in range(len(regime_stats)):
        state_mask = states == state
        regime_mean = regime_stats[state]['mean_return']
        
        if regime_mean > 0.001:
            regime_signals[state_mask] = 1
        elif regime_mean < -0.001:
            regime_signals[state_mask] = -1
    
    # Calculate correlations with each indicator signal
    correlations = {}
    for col in signals.columns:
        if col.endswith('_signal'):
            correlation = regime_signals.corr(signals[col])
            correlations[col] = correlation if not pd.isna(correlation) else 0.0
    
    correlation_results['signal_correlations'] = correlations
    
    # Analyze regime-indicator agreement
    agreement_analysis = {}
    for col in signals.columns:
        if col.endswith('_signal'):
            # Calculate agreement (same sign)
            signal_series = signals[col]
            same_direction = (
                ((regime_signals > 0) & (signal_series > 0)) |
                ((regime_signals < 0) & (signal_series < 0)) |
                ((regime_signals == 0) & (signal_series == 0))
            )
            agreement_rate = same_direction.mean()
            
            # Calculate conflicting signals
            conflicting = (
                ((regime_signals > 0) & (signal_series < 0)) |
                ((regime_signals < 0) & (signal_series > 0))
            )
            conflict_rate = conflicting.mean()
            
            agreement_analysis[col] = {
                'agreement_rate': agreement_rate,
                'conflict_rate': conflict_rate,
                'neutral_rate': 1 - agreement_rate - conflict_rate
            }
    
    correlation_results['agreement_analysis'] = agreement_analysis
    
    return correlation_results


def _validate_regimes_with_indicators(
    returns: pd.Series,
    states: pd.Series,
    indicators: pd.DataFrame,
    regime_analysis: Dict
) -> Dict:
    """Validate HMM regimes using traditional technical indicators."""
    validation_results = {}
    regime_stats = regime_analysis['regime_statistics']['regime_stats']
    
    for state in range(len(regime_stats)):
        state_mask = states == state
        state_returns = returns[state_mask]
        state_indicators = indicators[state_mask]
        
        if len(state_returns) < 5:  # Skip regimes with too few observations
            continue
            
        regime_validation = {
            'regime_id': state,
            'frequency': state_mask.mean(),
            'avg_duration': regime_stats[state]['avg_duration'],
            'mean_return': regime_stats[state]['mean_return'],
            'volatility': regime_stats[state]['volatility'],
            'indicator_confirmation': {}
        }
        
        # RSI validation
        if 'rsi' in state_indicators.columns:
            avg_rsi = state_indicators['rsi'].mean()
            if regime_stats[state]['mean_return'] > 0.001:  # Bullish regime
                rsi_confirmation = avg_rsi > 50  # RSI should be above 50
            elif regime_stats[state]['mean_return'] < -0.001:  # Bearish regime
                rsi_confirmation = avg_rsi < 50  # RSI should be below 50
            else:  # Neutral regime
                rsi_confirmation = 40 <= avg_rsi <= 60  # RSI should be neutral
            
            regime_validation['indicator_confirmation']['rsi'] = {
                'avg_value': avg_rsi,
                'confirms_regime': rsi_confirmation
            }
        
        # MACD validation
        if 'macd' in state_indicators.columns:
            avg_macd = state_indicators['macd'].mean()
            if regime_stats[state]['mean_return'] > 0.001:  # Bullish regime
                macd_confirmation = avg_macd > 0
            elif regime_stats[state]['mean_return'] < -0.001:  # Bearish regime
                macd_confirmation = avg_macd < 0
            else:  # Neutral regime
                macd_confirmation = abs(avg_macd) < 0.001
            
            regime_validation['indicator_confirmation']['macd'] = {
                'avg_value': avg_macd,
                'confirms_regime': macd_confirmation
            }
        
        # Bollinger Bands validation
        if 'bb_position' in state_indicators.columns:
            avg_bb_pos = state_indicators['bb_position'].mean()
            if regime_stats[state]['mean_return'] > 0.001:  # Bullish regime
                bb_confirmation = avg_bb_pos > 0.5  # Price above middle band
            elif regime_stats[state]['mean_return'] < -0.001:  # Bearish regime
                bb_confirmation = avg_bb_pos < 0.5  # Price below middle band
            else:  # Neutral regime
                bb_confirmation = 0.3 <= avg_bb_pos <= 0.7  # Price near middle
            
            regime_validation['indicator_confirmation']['bb_position'] = {
                'avg_value': avg_bb_pos,
                'confirms_regime': bb_confirmation
            }
        
        # Overall confirmation score
        confirmations = [conf['confirms_regime'] for conf in regime_validation['indicator_confirmation'].values()]
        regime_validation['overall_confirmation_rate'] = sum(confirmations) / len(confirmations) if confirmations else 0
        
        validation_results[f'regime_{state}'] = regime_validation
    
    return validation_results


def generate_indicator_comparison_report(
    comparison_results: Dict,
    save_path: Optional[str] = None,
    include_plots: bool = True
) -> str:
    """
    Generate comprehensive report comparing HMM and indicator performance.
    
    Args:
        comparison_results: Results from compare_hmm_vs_indicators
        save_path: Optional path to save the report
        include_plots: Whether to include visualizations
        
    Returns:
        Report content as markdown string
    """
    performance = comparison_results['performance_comparison']
    timing = comparison_results['timing_comparison']
    correlation = comparison_results['correlation_analysis']
    validation = comparison_results['regime_validation']
    
    report_lines = [
        "# HMM vs Technical Indicators Comparison Report",
        "",
        f"**Analysis Period**: {comparison_results['analysis_period']} days",
        f"**Data Points**: {comparison_results['data_points']}",
        "",
        "## Performance Comparison",
        "",
        "| Strategy | Total Return | Ann. Return | Volatility | Sharpe | Max DD | Win Rate |",
        "|----------|--------------|-------------|------------|--------|--------|----------|"
    ]
    
    for strategy, metrics in performance.items():
        report_lines.append(
            f"| {strategy.replace('_', ' ').title()} | "
            f"{metrics['total_return']:.3f} | "
            f"{metrics['annualized_return']:.3f} | "
            f"{metrics['volatility']:.3f} | "
            f"{metrics['sharpe_ratio']:.3f} | "
            f"{metrics['max_drawdown']:.3f} | "
            f"{metrics['win_rate']:.3f} |"
        )
    
    report_lines.extend([
        "",
        "## Signal Correlation Analysis",
        "",
        "### HMM-Indicator Correlations",
        ""
    ])
    
    correlations = correlation['signal_correlations']
    for signal, corr in correlations.items():
        signal_name = signal.replace('_signal', '').replace('_', ' ').title()
        report_lines.append(f"- **{signal_name}**: {corr:.3f}")
    
    report_lines.extend([
        "",
        "### Agreement Rates",
        ""
    ])
    
    agreements = correlation['agreement_analysis']
    for signal, agreement in agreements.items():
        signal_name = signal.replace('_signal', '').replace('_', ' ').title()
        report_lines.append(
            f"- **{signal_name}**: {agreement['agreement_rate']:.1%} agreement, "
            f"{agreement['conflict_rate']:.1%} conflict"
        )
    
    report_lines.extend([
        "",
        "## Regime Validation",
        ""
    ])
    
    for regime_key, regime_val in validation.items():
        regime_name = f"Regime {regime_val['regime_id']}"
        mean_return = regime_val['mean_return']
        
        if mean_return > 0.001:
            regime_type = "Bullish"
        elif mean_return < -0.001:
            regime_type = "Bearish"
        else:
            regime_type = "Neutral"
        
        report_lines.extend([
            f"### {regime_name} ({regime_type})",
            f"- **Frequency**: {regime_val['frequency']:.1%}",
            f"- **Mean Return**: {regime_val['mean_return']:.4f}",
            f"- **Confirmation Rate**: {regime_val['overall_confirmation_rate']:.1%}",
            ""
        ])
    
    report_lines.extend([
        "",
        "## Key Insights",
        "",
        _generate_insights(comparison_results),
        "",
        "---",
        "*Report generated by Hidden Regime Analysis Framework*"
    ])
    
    report_content = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_content)
    
    return report_content


def validate_regime_with_indicators(
    ticker: str,
    regime_analysis: Dict,
    period_days: int = 252
) -> Dict:
    """
    Validate specific regime analysis using technical indicators.
    
    Args:
        ticker: Stock symbol
        regime_analysis: HMM regime analysis results
        period_days: Period for validation analysis
        
    Returns:
        Validation results dictionary
    """
    from ..data import DataLoader
    
    # Load recent data
    loader = DataLoader()
    data = loader.load_stock_data(ticker, period=f"{period_days}d")
    
    # Calculate indicators
    calculator = IndicatorCalculator()
    indicators = calculator.calculate_all_indicators(data)
    
    # Extract regime information
    current_regime = regime_analysis['current_regime']['regime']
    regime_stats = regime_analysis['regime_statistics']['regime_stats'][current_regime]
    
    # Recent indicator values (last 20 days)
    recent_indicators = indicators.tail(20)
    
    validation_results = {
        'ticker': ticker,
        'current_regime': current_regime,
        'regime_characteristics': regime_stats,
        'indicator_validation': {},
        'overall_score': 0.0
    }
    
    confirmations = []
    
    # RSI validation
    if 'rsi' in recent_indicators.columns:
        avg_rsi = recent_indicators['rsi'].mean()
        expected_rsi_range = _get_expected_rsi_range(regime_stats['mean_return'])
        rsi_confirms = expected_rsi_range[0] <= avg_rsi <= expected_rsi_range[1]
        confirmations.append(rsi_confirms)
        
        validation_results['indicator_validation']['rsi'] = {
            'current_value': avg_rsi,
            'expected_range': expected_rsi_range,
            'confirms_regime': rsi_confirms
        }
    
    # Add other indicator validations...
    validation_results['overall_score'] = sum(confirmations) / len(confirmations) if confirmations else 0.0
    
    return validation_results


# Helper functions
def _calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Calculate maximum drawdown from cumulative returns."""
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def _count_signals_before_moves(signals: pd.Series, moves: pd.Series, window: int = 5) -> float:
    """Count how often signals occur before significant moves."""
    if moves.sum() == 0:
        return 0.0
    
    signal_before_move = 0
    for move_date in moves[moves].index:
        window_start = max(0, moves.index.get_loc(move_date) - window)
        window_end = moves.index.get_loc(move_date)
        window_signals = signals.iloc[window_start:window_end]
        
        if window_signals.any():
            signal_before_move += 1
    
    return signal_before_move / moves.sum()


def _avg_days_before_moves(signals: pd.Series, moves: pd.Series, window: int = 10) -> float:
    """Calculate average days between signals and subsequent moves."""
    days_before = []
    
    for move_date in moves[moves].index:
        move_idx = moves.index.get_loc(move_date)
        window_start = max(0, move_idx - window)
        
        # Find most recent signal before the move
        recent_signals = signals.iloc[window_start:move_idx]
        if recent_signals.any():
            last_signal_idx = recent_signals[::-1].idxmax()  # Most recent signal
            days_before.append(move_idx - signals.index.get_loc(last_signal_idx))
    
    return np.mean(days_before) if days_before else np.nan


def _get_expected_rsi_range(mean_return: float) -> Tuple[float, float]:
    """Get expected RSI range based on regime characteristics."""
    if mean_return > 0.001:  # Bullish
        return (50, 80)
    elif mean_return < -0.001:  # Bearish
        return (20, 50)
    else:  # Neutral
        return (40, 60)


def _generate_insights(comparison_results: Dict) -> str:
    """Generate key insights from comparison analysis."""
    performance = comparison_results['performance_comparison']
    correlation = comparison_results['correlation_analysis']
    
    insights = []
    
    # Performance insights
    if 'hmm' in performance and 'buy_hold' in performance:
        hmm_sharpe = performance['hmm']['sharpe_ratio']
        bh_sharpe = performance['buy_hold']['sharpe_ratio']
        
        if hmm_sharpe > bh_sharpe:
            insights.append(f"HMM strategy outperforms buy-and-hold with {hmm_sharpe:.2f} vs {bh_sharpe:.2f} Sharpe ratio.")
        else:
            insights.append(f"Buy-and-hold outperforms HMM strategy with {bh_sharpe:.2f} vs {hmm_sharpe:.2f} Sharpe ratio.")
    
    # Correlation insights
    correlations = correlation['signal_correlations']
    highest_corr = max(correlations.items(), key=lambda x: abs(x[1]))
    insights.append(f"Strongest HMM correlation with {highest_corr[0].replace('_signal', '')} indicator ({highest_corr[1]:.2f}).")
    
    # Agreement insights
    agreements = correlation['agreement_analysis']
    avg_agreement = np.mean([a['agreement_rate'] for a in agreements.values()])
    insights.append(f"Average agreement rate between HMM and indicators: {avg_agreement:.1%}.")
    
    return "\n".join(f"- {insight}" for insight in insights)