"""
Content Generation Functions

Automated generators for creating blog content, market reports, and analysis
from HMM regime detection results. Produces publication-ready content with
charts, statistics, and insights.
"""

import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data import DataLoader
from ..historical import load_historical_period, validate_historical_detection
from ..models import HiddenMarkovModel, HMMConfig
from ..visualization import plot_returns_with_regimes, setup_financial_plot_style
from .templates import get_template, format_analysis_for_blog


def generate_market_report(
    ticker: str,
    period_days: int = 252,
    end_date: Optional[str] = None,
    include_charts: bool = True,
    save_path: Optional[str] = None,
    hmm_config: Optional[HMMConfig] = None
) -> Dict:
    """
    Generate comprehensive market regime report for a single ticker.
    
    Creates a complete analysis including current regime, recent transitions,
    statistical analysis, and publication-ready charts.
    
    Args:
        ticker: Stock symbol to analyze
        period_days: Number of days to analyze (default: 252, ~1 year)
        end_date: End date for analysis (defaults to today)
        include_charts: Generate and save charts
        save_path: Directory to save report and charts (optional)
        hmm_config: HMM configuration (uses defaults if None)
        
    Returns:
        Dictionary with complete report data and file paths
        
    Example:
        >>> report = generate_market_report('AAPL', period_days=180, save_path='blog_content/')
        >>> print(report['summary'])
        >>> # Files saved: blog_content/AAPL_regime_report_2024-12-09.md
    """
    # Calculate date range
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_dt = end_dt - timedelta(days=period_days + 30)  # Extra days for data availability
    start_date = start_dt.strftime('%Y-%m-%d')
    
    print(f"Generating market report for {ticker}")
    print(f"Period: {start_date} to {end_date}")
    
    # Load and prepare data
    loader = DataLoader()
    data = loader.load_stock_data(ticker, start_date, end_date)
    
    if data.empty or len(data) < 50:
        raise ValueError(f"Insufficient data for {ticker} in specified period")
    
    # Trim to exact period requested
    data = data.tail(period_days) if len(data) > period_days else data
    returns = data['log_return'].dropna()
    dates = data['date'][1:]  # Match returns length
    
    # Configure and run HMM analysis
    if hmm_config is None:
        hmm_config = HMMConfig.for_market_data(conservative=True)
    
    hmm = HiddenMarkovModel(config=hmm_config)
    hmm.fit(returns, verbose=False)
    
    states = hmm.predict(returns)
    regime_analysis = hmm.analyze_regimes(returns, dates)
    
    # Get current regime information
    current_regime = states[-1]
    current_regime_info = hmm.update_with_observation(returns.iloc[-1])
    
    # Build comprehensive report
    report = {
        'metadata': {
            'ticker': ticker,
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_period': f"{start_date} to {end_date}",
            'data_points': len(returns),
            'hmm_config': hmm_config
        },
        'current_regime': {
            'state': current_regime,
            'name': regime_analysis['regime_interpretations'][current_regime],
            'confidence': current_regime_info['confidence'],
            'regime_probabilities': current_regime_info['regime_probabilities'],
            'days_in_regime': _count_consecutive_days(states, current_regime),
        },
        'regime_analysis': regime_analysis,
        'summary_statistics': _calculate_summary_stats(data, returns, states),
        'key_insights': _generate_key_insights(regime_analysis, current_regime_info, ticker),
        'file_paths': {}
    }
    
    # Generate charts if requested
    if include_charts:
        chart_paths = _generate_report_charts(
            data, returns, dates, states, regime_analysis, ticker, save_path
        )
        report['file_paths'].update(chart_paths)
    
    # Generate written report if save path provided
    if save_path:
        report_path = _generate_written_report(report, save_path)
        report['file_paths']['report'] = report_path
        
        print(f"✓ Report generated: {report_path}")
        if include_charts:
            print(f"✓ Charts saved to: {save_path}")
    
    return report


def generate_historical_analysis(
    event_name: str,
    ticker: str = "SPY",
    include_validation: bool = True,
    include_charts: bool = True,
    save_path: Optional[str] = None,
    hmm_config: Optional[HMMConfig] = None
) -> Dict:
    """
    Generate historical market event analysis with regime detection validation.
    
    Perfect for creating "How HMM Detected the 2008 Crisis" type content.
    
    Args:
        event_name: Historical event from MAJOR_MARKET_EVENTS
        ticker: Stock symbol to analyze  
        include_validation: Include validation metrics
        include_charts: Generate publication-ready charts
        save_path: Directory to save analysis and charts
        hmm_config: HMM configuration
        
    Returns:
        Dictionary with historical analysis and validation results
    """
    print(f"Generating historical analysis: {event_name}")
    print(f"Ticker: {ticker}")
    
    # Load historical data and run analysis
    data_dict, event_info = load_historical_period(event_name, [ticker])
    data = data_dict[ticker]
    returns = data['log_return'].dropna()
    dates = data['date'][1:]
    
    # Run HMM analysis
    if hmm_config is None:
        hmm_config = HMMConfig.for_market_data()
    
    hmm = HiddenMarkovModel(config=hmm_config)
    hmm.fit(returns, verbose=False)
    
    states = hmm.predict(returns)
    regime_analysis = hmm.analyze_regimes(returns, dates)
    
    # Run validation if requested
    validation_results = None
    if include_validation:
        validation_results = validate_historical_detection(
            event_name, ticker, hmm_config, verbose=False
        )
    
    # Build analysis report
    analysis = {
        'metadata': {
            'event_name': event_info['name'],
            'event_key': event_name,
            'ticker': ticker,
            'period': f"{event_info['start_date']} to {event_info['end_date']}",
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
        'event_info': event_info,
        'regime_analysis': regime_analysis,
        'historical_context': _create_historical_context(event_info, regime_analysis),
        'validation_results': validation_results,
        'narrative_insights': _generate_historical_narrative(event_info, regime_analysis, validation_results),
        'file_paths': {}
    }
    
    # Generate charts
    if include_charts:
        chart_paths = _generate_historical_charts(
            data, returns, dates, states, regime_analysis, event_info, ticker, save_path
        )
        analysis['file_paths'].update(chart_paths)
    
    # Generate written analysis
    if save_path:
        analysis_path = _generate_historical_report(analysis, save_path)
        analysis['file_paths']['analysis'] = analysis_path
        print(f"✓ Historical analysis saved: {analysis_path}")
    
    return analysis


def generate_comparative_report(
    ticker: str,
    period: str = "1Y",
    indicators: Optional[List[str]] = None,
    include_charts: bool = True,
    save_path: Optional[str] = None
) -> Dict:
    """
    Generate comparative analysis of HMM vs traditional technical indicators.
    
    Perfect for "HMM vs MACD: Performance Analysis" type content.
    
    Args:
        ticker: Stock symbol to analyze
        period: Analysis period ('6M', '1Y', '2Y', etc.)
        indicators: List of indicators to compare (['MACD', 'RSI', 'BB'])
        include_charts: Generate comparison charts
        save_path: Directory to save report
        
    Returns:
        Dictionary with comparative analysis results
    """
    if indicators is None:
        indicators = ['MACD', 'RSI', 'BB']
    
    print(f"Generating comparative report: HMM vs {indicators}")
    print(f"Ticker: {ticker}, Period: {period}")
    
    # Parse period and calculate dates
    period_days = _parse_period_to_days(period)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_dt = datetime.now() - timedelta(days=period_days + 30)
    start_date = start_dt.strftime('%Y-%m-%d')
    
    # Load data
    loader = DataLoader()
    data = loader.load_stock_data(ticker, start_date, end_date)
    returns = data['log_return'].dropna()
    dates = data['date'][1:]
    
    # Run HMM analysis
    hmm = HiddenMarkovModel(n_states=3)
    hmm.fit(returns, verbose=False)
    
    states = hmm.predict(returns)
    regime_analysis = hmm.analyze_regimes(returns, dates)
    
    # Note: Technical indicators integration will be implemented in next phase
    # For now, create placeholder structure
    comparison = {
        'metadata': {
            'ticker': ticker,
            'period': period,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'indicators_compared': indicators,
        },
        'hmm_analysis': regime_analysis,
        'indicator_analysis': {},  # Will be populated when indicators module is ready
        'performance_comparison': {},  # Will be populated with backtesting results
        'timing_analysis': {},  # Signal timing comparison
        'narrative_summary': _generate_comparison_narrative(regime_analysis, indicators),
        'file_paths': {}
    }
    
    # Generate basic charts (full indicator integration coming in Phase B)
    if include_charts:
        chart_paths = _generate_basic_comparison_charts(
            data, returns, dates, states, regime_analysis, ticker, save_path
        )
        comparison['file_paths'].update(chart_paths)
    
    if save_path:
        report_path = _generate_comparison_report(comparison, save_path)
        comparison['file_paths']['report'] = report_path
        print(f"✓ Comparative report saved: {report_path}")
    
    return comparison


def generate_regime_update(
    tickers: List[str],
    lookback_days: int = 30,
    save_path: Optional[str] = None
) -> Dict:
    """
    Generate weekly/monthly regime update for multiple stocks.
    
    Perfect for "This Week in Regimes" or "Market Regime Scanner" content.
    
    Args:
        tickers: List of stock symbols to analyze
        lookback_days: Days to look back for regime changes
        save_path: Directory to save update
        
    Returns:
        Dictionary with regime update for all tickers
    """
    print(f"Generating regime update for {len(tickers)} tickers")
    print(f"Lookback period: {lookback_days} days")
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_dt = datetime.now() - timedelta(days=lookback_days + 100)  # Extra data for HMM
    start_date = start_dt.strftime('%Y-%m-%d')
    
    loader = DataLoader()
    updates = []
    
    for ticker in tickers:
        try:
            # Load data and run analysis
            data = loader.load_stock_data(ticker, start_date, end_date)
            returns = data['log_return'].dropna()
            
            if len(returns) < 50:  # Skip if insufficient data
                continue
            
            # Run HMM
            hmm = HiddenMarkovModel(n_states=3)
            hmm.fit(returns, verbose=False)
            
            states = hmm.predict(returns)
            current_regime_info = hmm.update_with_observation(returns.iloc[-1])
            
            # Check for recent regime changes
            recent_states = states[-lookback_days:] if len(states) >= lookback_days else states
            regime_changes = _detect_recent_regime_changes(recent_states, lookback_days)
            
            ticker_update = {
                'ticker': ticker,
                'current_regime': {
                    'state': states[-1],
                    'confidence': current_regime_info['confidence'],
                    'regime_probabilities': current_regime_info['regime_probabilities'],
                },
                'recent_changes': regime_changes,
                'performance_summary': _calculate_recent_performance(data, lookback_days),
                'alert_level': _calculate_alert_level(regime_changes, current_regime_info)
            }
            
            updates.append(ticker_update)
            
        except Exception as e:
            print(f"⚠️  Skipping {ticker}: {str(e)}")
            continue
    
    # Compile overall update
    regime_update = {
        'metadata': {
            'update_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'lookback_days': lookback_days,
            'tickers_analyzed': len(updates),
            'tickers_requested': len(tickers),
        },
        'ticker_updates': updates,
        'market_summary': _create_market_summary(updates),
        'highlights': _identify_regime_highlights(updates),
        'file_paths': {}
    }
    
    if save_path:
        update_path = _generate_regime_update_report(regime_update, save_path)
        regime_update['file_paths']['update'] = update_path
        print(f"✓ Regime update saved: {update_path}")
    
    return regime_update


# Helper functions for content generation

def _count_consecutive_days(states: np.ndarray, target_state: int) -> int:
    """Count consecutive days in current regime."""
    count = 0
    for i in range(len(states) - 1, -1, -1):
        if states[i] == target_state:
            count += 1
        else:
            break
    return count


def _calculate_summary_stats(data: pd.DataFrame, returns: pd.Series, states: np.ndarray) -> Dict:
    """Calculate summary statistics for the report."""
    return {
        'total_return': (data['price'].iloc[-1] / data['price'].iloc[0] - 1),
        'annualized_return': returns.mean() * 252,
        'volatility': returns.std() * np.sqrt(252),
        'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252),
        'max_drawdown': _calculate_max_drawdown(data['price']),
        'regime_transitions': np.sum(states[1:] != states[:-1]),
        'current_price': data['price'].iloc[-1],
        'period_high': data['price'].max(),
        'period_low': data['price'].min(),
    }


def _calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown."""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown.min()


def _generate_key_insights(regime_analysis: Dict, current_regime_info: Dict, ticker: str) -> List[str]:
    """Generate key insights for the report."""
    insights = []
    
    # Current regime insight
    current_state = current_regime_info['most_likely_regime']
    confidence = current_regime_info['confidence']
    
    regime_names = {0: "Bear", 1: "Sideways", 2: "Bull"}
    regime_name = regime_names.get(current_state, f"State {current_state}")
    
    insights.append(f"{ticker} is currently in a {regime_name} regime with {confidence:.1%} confidence")
    
    # Regime statistics insights
    regime_stats = regime_analysis['regime_statistics']['regime_stats']
    dominant_regime = max(regime_stats.keys(), key=lambda k: regime_stats[k]['frequency'])
    dominant_freq = regime_stats[dominant_regime]['frequency']
    
    insights.append(f"Dominant regime during this period: {regime_names.get(dominant_regime, f'State {dominant_regime}')} ({dominant_freq:.1%} of time)")
    
    # Performance insights
    if current_state in regime_stats:
        current_stats = regime_stats[current_state]
        mean_return = current_stats['mean_return']
        
        if mean_return > 0.005:
            insights.append(f"Current regime shows strong positive momentum (avg daily return: {mean_return:.3f})")
        elif mean_return < -0.005:
            insights.append(f"Current regime shows negative momentum (avg daily return: {mean_return:.3f})")
        else:
            insights.append(f"Current regime shows sideways movement (avg daily return: {mean_return:.3f})")
    
    return insights


def _parse_period_to_days(period: str) -> int:
    """Parse period string to number of days."""
    period = period.upper()
    if period.endswith('D'):
        return int(period[:-1])
    elif period.endswith('W'):
        return int(period[:-1]) * 7
    elif period.endswith('M'):
        return int(period[:-1]) * 30
    elif period.endswith('Y'):
        return int(period[:-1]) * 365
    else:
        # Default to days if no suffix
        try:
            return int(period)
        except ValueError:
            return 252  # Default to 1 year


def _generate_report_charts(
    data: pd.DataFrame,
    returns: pd.Series,
    dates: pd.Series,
    states: np.ndarray,
    regime_analysis: Dict,
    ticker: str,
    save_path: Optional[str]
) -> Dict:
    """Generate charts for market report."""
    chart_paths = {}
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Setup financial plotting style
        setup_financial_plot_style()
        
        # Main regime chart
        fig = plot_returns_with_regimes(
            returns, dates, states,
            title=f"{ticker} Market Regime Analysis",
            regime_names={0: "Bear", 1: "Sideways", 2: "Bull"}
        )
        
        chart_path = os.path.join(save_path, f"{ticker}_regime_analysis.png")
        fig.savefig(chart_path, dpi=300, bbox_inches='tight')
        chart_paths['regime_chart'] = chart_path
        plt.close(fig)
        
        print(f"✓ Chart saved: {chart_path}")
    
    return chart_paths


def _generate_written_report(report: Dict, save_path: str) -> str:
    """Generate written report using templates."""
    os.makedirs(save_path, exist_ok=True)
    
    template = get_template('market_report')
    report_content = format_analysis_for_blog(report, template)
    
    ticker = report['metadata']['ticker']
    date_str = datetime.now().strftime('%Y-%m-%d')
    filename = f"{ticker}_regime_report_{date_str}.md"
    report_path = os.path.join(save_path, filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_path


def _create_historical_context(event_info: Dict, regime_analysis: Dict) -> Dict:
    """Create historical context for the event."""
    return {
        'event_summary': event_info['description'],
        'expected_characteristics': event_info.get('characteristics', {}),
        'detected_regimes': regime_analysis['regime_statistics']['regime_stats'],
        'timeline_summary': f"Analysis covers {event_info['start_date']} to {event_info['end_date']}",
    }


def _generate_historical_narrative(event_info: Dict, regime_analysis: Dict, validation_results: Optional[Dict]) -> List[str]:
    """Generate narrative insights for historical analysis."""
    narrative = []
    
    narrative.append(f"Analysis of {event_info['name']} reveals distinct regime characteristics")
    
    if validation_results and validation_results['validation_metrics']['validation_passed']:
        match_score = validation_results['validation_metrics']['regime_match_score']
        narrative.append(f"HMM successfully detected expected regime patterns (validation score: {match_score:.2f})")
    
    # Add regime-specific insights
    regime_stats = regime_analysis['regime_statistics']['regime_stats']
    dominant_regime = max(regime_stats.keys(), key=lambda k: regime_stats[k]['frequency'])
    
    narrative.append(f"Dominant regime during this period showed {regime_stats[dominant_regime]['mean_return']:.3f} average daily returns")
    
    return narrative


def _generate_comparison_narrative(regime_analysis: Dict, indicators: List[str]) -> List[str]:
    """Generate narrative for comparative analysis."""
    narrative = []
    
    narrative.append(f"Comparing HMM regime detection with traditional indicators: {', '.join(indicators)}")
    narrative.append("HMM provides probabilistic regime detection vs binary signals from traditional indicators")
    
    # Add regime summary
    n_regimes = len(regime_analysis['regime_statistics']['regime_stats'])
    narrative.append(f"HMM detected {n_regimes} distinct market regimes during this period")
    
    return narrative


def _detect_recent_regime_changes(states: np.ndarray, lookback_days: int) -> List[Dict]:
    """Detect recent regime changes."""
    changes = []
    
    for i in range(1, len(states)):
        if states[i] != states[i-1]:
            days_ago = len(states) - i
            if days_ago <= lookback_days:
                changes.append({
                    'days_ago': days_ago,
                    'from_regime': states[i-1],
                    'to_regime': states[i],
                    'change_type': _classify_regime_change(states[i-1], states[i])
                })
    
    return changes


def _classify_regime_change(from_regime: int, to_regime: int) -> str:
    """Classify the type of regime change."""
    regime_names = {0: "Bear", 1: "Sideways", 2: "Bull"}
    
    from_name = regime_names.get(from_regime, f"State {from_regime}")
    to_name = regime_names.get(to_regime, f"State {to_regime}")
    
    return f"{from_name} → {to_name}"


def _calculate_recent_performance(data: pd.DataFrame, lookback_days: int) -> Dict:
    """Calculate recent performance metrics."""
    recent_data = data.tail(lookback_days)
    
    if len(recent_data) < 2:
        return {'insufficient_data': True}
    
    start_price = recent_data['price'].iloc[0]
    end_price = recent_data['price'].iloc[-1]
    
    return {
        'period_return': (end_price / start_price - 1),
        'start_price': start_price,
        'end_price': end_price,
        'high': recent_data['price'].max(),
        'low': recent_data['price'].min(),
        'days': len(recent_data)
    }


def _calculate_alert_level(regime_changes: List[Dict], current_regime_info: Dict) -> str:
    """Calculate alert level based on recent activity."""
    if len(regime_changes) >= 2:
        return "HIGH"  # Multiple recent changes
    elif len(regime_changes) == 1 and regime_changes[0]['days_ago'] <= 5:
        return "MEDIUM"  # Recent single change
    elif current_regime_info['confidence'] < 0.6:
        return "MEDIUM"  # Low confidence in current regime
    else:
        return "LOW"  # Stable regime


def _create_market_summary(updates: List[Dict]) -> Dict:
    """Create overall market summary from individual ticker updates."""
    if not updates:
        return {'no_data': True}
    
    # Count regimes
    regime_counts = {0: 0, 1: 0, 2: 0}
    total_changes = 0
    high_alerts = 0
    
    for update in updates:
        current_regime = update['current_regime']['state']
        regime_counts[current_regime] += 1
        total_changes += len(update['recent_changes'])
        
        if update['alert_level'] == 'HIGH':
            high_alerts += 1
    
    total_tickers = len(updates)
    
    return {
        'total_tickers': total_tickers,
        'regime_distribution': {
            'bear': regime_counts[0] / total_tickers,
            'sideways': regime_counts[1] / total_tickers, 
            'bull': regime_counts[2] / total_tickers,
        },
        'total_regime_changes': total_changes,
        'high_alert_tickers': high_alerts,
        'market_stability': 'LOW' if total_changes > total_tickers else 'HIGH'
    }


def _identify_regime_highlights(updates: List[Dict]) -> List[Dict]:
    """Identify notable regime changes and situations."""
    highlights = []
    
    for update in updates:
        ticker = update['ticker']
        
        # Recent regime changes
        if update['recent_changes']:
            latest_change = update['recent_changes'][-1]
            highlights.append({
                'type': 'regime_change',
                'ticker': ticker,
                'description': f"{ticker} transitioned to {latest_change['change_type']} {latest_change['days_ago']} days ago"
            })
        
        # High confidence regimes
        confidence = update['current_regime']['confidence']
        if confidence > 0.9:
            regime_names = {0: "Bear", 1: "Sideways", 2: "Bull"}
            regime_name = regime_names.get(update['current_regime']['state'], "Unknown")
            highlights.append({
                'type': 'high_confidence',
                'ticker': ticker,
                'description': f"{ticker} shows strong {regime_name} regime conviction ({confidence:.1%} confidence)"
            })
    
    # Sort by importance and return top highlights
    return sorted(highlights, key=lambda x: x['ticker'])[:10]


# Placeholder functions for chart generation (will be enhanced in visualization phase)

def _generate_historical_charts(data, returns, dates, states, regime_analysis, event_info, ticker, save_path):
    """Generate charts for historical analysis."""
    return _generate_report_charts(data, returns, dates, states, regime_analysis, ticker, save_path)


def _generate_basic_comparison_charts(data, returns, dates, states, regime_analysis, ticker, save_path):
    """Generate basic comparison charts (full indicator charts in Phase B)."""
    return _generate_report_charts(data, returns, dates, states, regime_analysis, ticker, save_path)


def _generate_historical_report(analysis, save_path):
    """Generate historical analysis report."""
    os.makedirs(save_path, exist_ok=True)
    
    event_key = analysis['metadata']['event_key']
    ticker = analysis['metadata']['ticker']
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    filename = f"{event_key}_{ticker}_analysis_{date_str}.md"
    report_path = os.path.join(save_path, filename)
    
    # Use basic template for now (will enhance with proper templates)
    content = f"""# {analysis['event_info']['name']} - HMM Regime Analysis

**Ticker:** {ticker}
**Period:** {analysis['metadata']['period']}
**Analysis Date:** {analysis['metadata']['analysis_date']}

## Event Overview
{analysis['event_info']['description']}

## Key Findings
"""
    
    for insight in analysis['narrative_insights']:
        content += f"- {insight}\n"
    
    if analysis['validation_results']:
        validation = analysis['validation_results']['validation_metrics']
        content += f"\n## Validation Results\n"
        content += f"- Match Score: {validation['regime_match_score']:.3f}\n"
        content += f"- Validation: {'✅ PASSED' if validation['validation_passed'] else '❌ FAILED'}\n"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return report_path


def _generate_comparison_report(comparison, save_path):
    """Generate comparison report."""
    os.makedirs(save_path, exist_ok=True)
    
    ticker = comparison['metadata']['ticker']
    period = comparison['metadata']['period']
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    filename = f"{ticker}_comparison_{period}_{date_str}.md"
    report_path = os.path.join(save_path, filename)
    
    content = f"""# HMM vs Traditional Indicators - {ticker}

**Period:** {period}
**Analysis Date:** {comparison['metadata']['analysis_date']}
**Indicators Compared:** {', '.join(comparison['metadata']['indicators_compared'])}

## Summary
"""
    
    for insight in comparison['narrative_summary']:
        content += f"- {insight}\n"
    
    content += "\n*Note: Full indicator integration and backtesting results will be available in Phase B.*\n"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return report_path


def _generate_regime_update_report(regime_update, save_path):
    """Generate regime update report."""
    os.makedirs(save_path, exist_ok=True)
    
    date_str = datetime.now().strftime('%Y-%m-%d')
    filename = f"regime_update_{date_str}.md"
    report_path = os.path.join(save_path, filename)
    
    content = f"""# Market Regime Update - {date_str}

**Tickers Analyzed:** {regime_update['metadata']['tickers_analyzed']}
**Lookback Period:** {regime_update['metadata']['lookback_days']} days

## Market Summary
"""
    
    summary = regime_update['market_summary']
    if 'regime_distribution' in summary:
        dist = summary['regime_distribution']
        content += f"- Bull Regimes: {dist['bull']:.1%}\n"
        content += f"- Sideways Regimes: {dist['sideways']:.1%}\n"
        content += f"- Bear Regimes: {dist['bear']:.1%}\n"
        content += f"- Total Regime Changes: {summary['total_regime_changes']}\n"
        content += f"- Market Stability: {summary['market_stability']}\n"
    
    content += "\n## Highlights\n"
    for highlight in regime_update['highlights']:
        content += f"- {highlight['description']}\n"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return report_path