#!/usr/bin/env python3
"""
Market Report Generation Example
================================

Demonstrates comprehensive market analysis report generation for a single stock
using HMM regime detection. Creates publication-ready content suitable for blog 
articles and market commentary.

This example showcases:
- Complete market regime analysis for a single ticker
- Professional chart generation with regime backgrounds 
- Automated report generation in markdown format
- Export capabilities for blog publication

Run this script to generate a complete market analysis report for Apple (AAPL)
that can be directly used in blog content.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import our framework
from hidden_regime.data import DataLoader
from hidden_regime.analysis import RegimeAnalyzer
from hidden_regime.config import DataConfig

def create_regime_visualization(data: pd.DataFrame, analysis: Dict[str, Any], 
                              output_dir: str) -> str:
    """Create a comprehensive regime analysis visualization"""
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Market Regime Analysis: {analysis["symbol"]}', 
                 fontsize=16, fontweight='bold')
    
    # Get returns from data (already calculated)
    returns = data['log_return'].dropna()
    
    # 1. Price chart with regime backgrounds
    ax1 = axes[0, 0]
    states = analysis['states']
    
    # Create regime colors
    regime_colors = {0: 'red', 1: 'gray', 2: 'green'}
    regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
    
    # Plot price
    price_dates = data['date'].iloc[-len(states):]
    ax1.plot(price_dates, data['price'].iloc[-len(states):], 'k-', linewidth=1.5, alpha=0.8)
    
    # Add regime backgrounds
    current_regime = states[0]
    start_idx = 0
    
    for i in range(1, len(states)):
        if states[i] != current_regime or i == len(states) - 1:
            end_idx = i if states[i] != current_regime else i + 1
            color = regime_colors.get(current_regime, 'gray')
            ax1.axvspan(price_dates[start_idx], price_dates[end_idx-1], 
                       alpha=0.2, color=color, label=regime_names.get(current_regime, f'State {current_regime}') if start_idx == 0 else "")
            current_regime = states[i]
            start_idx = i
    
    ax1.set_title('Price with Regime Detection', fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Returns with regime coloring
    ax2 = axes[0, 1]
    return_dates = data['date'].iloc[-len(returns):]
    
    for state in range(3):
        state_mask = states == state
        if np.any(state_mask):
            state_returns = returns.iloc[-len(states):].iloc[state_mask]
            state_dates = return_dates.iloc[state_mask]
            ax2.scatter(state_dates, state_returns, c=regime_colors.get(state, 'gray'), 
                       alpha=0.6, s=20, label=regime_names.get(state, f'State {state}'))
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('Returns by Regime', fontweight='bold')
    ax2.set_ylabel('Daily Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Regime probabilities over time
    ax3 = axes[1, 0]
    probabilities = analysis['probabilities']
    prob_dates = data['date'].iloc[-len(probabilities):]
    
    for state in range(3):
        ax3.plot(prob_dates, probabilities[:, state], 
                label=regime_names.get(state, f'State {state}'),
                color=regime_colors.get(state, 'gray'), linewidth=2)
    
    ax3.set_title('Regime Probabilities Over Time', fontweight='bold')
    ax3.set_ylabel('Probability')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Regime statistics
    ax4 = axes[1, 1]
    regime_stats = analysis['regime_stats']
    
    states_list = []
    returns_list = []
    volatility_list = []
    
    for state_key, stats in regime_stats.items():
        state_num = int(state_key.split('_')[1])
        states_list.append(regime_names.get(state_num, f'State {state_num}'))
        returns_list.append(stats['annualized_return'])
        volatility_list.append(stats['annualized_volatility'])
    
    x = np.arange(len(states_list))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, [r*100 for r in returns_list], width, 
                   label='Annualized Return (%)', alpha=0.8)
    bars2 = ax4.bar(x + width/2, [v*100 for v in volatility_list], width,
                   label='Annualized Volatility (%)', alpha=0.8)
    
    # Color bars by regime
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        color = regime_colors.get(i, 'gray')
        bar1.set_color(color)
        bar2.set_color(color)
        bar2.set_alpha(0.6)
    
    ax4.set_title('Regime Characteristics', fontweight='bold')
    ax4.set_ylabel('Percentage (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(states_list)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the plot
    chart_file = os.path.join(output_dir, f'{analysis["symbol"]}_regime_analysis.png')
    plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return chart_file

def generate_market_report(analysis: Dict[str, Any], chart_file: str, 
                          output_dir: str) -> str:
    """Generate a comprehensive markdown report"""
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    symbol = analysis['symbol']
    
    # Extract key metrics
    current_regime = analysis['current_regime']
    confidence = analysis['confidence']
    days_in_regime = analysis['days_in_regime']
    regime_changes = analysis['regime_changes']
    
    # Calculate some summary statistics
    total_days = analysis['data_length']
    regime_stats = analysis['regime_stats']
    
    # Generate report content
    report = f"""# Market Regime Analysis Report: {symbol}
*Generated on {timestamp}*

## Executive Summary

Our Hidden Markov Model analysis of **{symbol}** identifies the current market regime and provides insights for trading and investment decisions. The analysis covers **{total_days}** trading days with **{regime_changes}** regime transitions detected.

### Current Market State

- **Current Regime**: {current_regime}
- **Confidence Level**: {confidence:.1%}
- **Days in Current Regime**: {days_in_regime}
- **Expected Daily Return**: {analysis['expected_return']:.4f} ({analysis['expected_return']*252:.1%} annualized)
- **Expected Daily Volatility**: {analysis['expected_volatility']:.4f} ({analysis['expected_volatility']*np.sqrt(252):.1%} annualized)

## Regime Analysis Details

### Detected Market Regimes

Our analysis identified three distinct market regimes with the following characteristics:

"""
    
    # Add regime details
    regime_names = {'state_0': 'Bear Market', 'state_1': 'Sideways Market', 'state_2': 'Bull Market'}
    for state_key, stats in regime_stats.items():
        regime_name = regime_names.get(state_key, state_key)
        
        report += f"""
#### {regime_name}
- **Frequency**: {stats['frequency']:.1%} of trading days
- **Average Daily Return**: {stats['mean_return']:.4f} ({stats['annualized_return']:.1%} annualized)
- **Daily Volatility**: {stats['std_return']:.4f} ({stats['annualized_volatility']:.1%} annualized)
- **Total Days**: {stats['count']} days
"""
    
    report += f"""

## Investment Implications

### Current Regime: {current_regime}

Based on the current **{current_regime}** regime with **{confidence:.1%}** confidence:

"""
    
    # Add regime-specific commentary
    if current_regime == 'Bull':
        report += """
**Bullish Market Conditions Detected**
- Consider **increasing equity exposure** during bull regimes
- **Growth strategies** tend to outperform in this environment
- **Momentum strategies** are favored with trending price action
- Monitor for potential regime change signals as bull runs mature
"""
    elif current_regime == 'Bear':
        report += """
**Bearish Market Conditions Detected**
- Consider **reducing equity exposure** or implementing hedging strategies
- **Defensive positioning** is recommended during bear regimes
- **Value strategies** may present opportunities in oversold conditions
- **Cash positions** can preserve capital during market stress
"""
    else:
        report += """
**Sideways Market Conditions Detected**
- **Range-bound trading strategies** are optimal for sideways regimes
- Consider **mean reversion** approaches rather than momentum
- **Sector rotation** strategies may outperform broad market exposure
- Prepare for potential breakout in either direction
"""
    
    report += f"""

### Risk Management Considerations

- **Regime Persistence**: Currently {days_in_regime} days into the {current_regime} regime
- **Transition Risk**: {regime_changes} regime changes detected in analysis period
- **Volatility Environment**: Current expected volatility of {analysis['expected_volatility']*np.sqrt(252):.1%} (annualized)

## Technical Analysis

![Market Regime Analysis]({os.path.basename(chart_file)})

The chart above shows:
1. **Price Action with Regime Backgrounds**: Visual representation of regime transitions
2. **Returns by Regime**: Distribution of daily returns across different market states
3. **Regime Probabilities**: Confidence levels over time for regime classification
4. **Regime Characteristics**: Comparative returns and volatility across regimes

## Methodology

This analysis uses a **Hidden Markov Model (HMM)** with three hidden states representing different market regimes. The model analyzes patterns in daily log returns to identify persistent market conditions that are not directly observable.

### Key Features:
- **Probabilistic Framework**: Provides confidence levels rather than binary classifications
- **Adaptive Learning**: Model parameters adjust to changing market conditions
- **Regime Persistence**: Captures the tendency for market conditions to persist over time
- **Real-time Updates**: Can incorporate new data for dynamic regime detection

### Model Specifications:
- **States**: 3 (Bear, Sideways, Bull)
- **Observation Model**: Gaussian emissions based on daily returns
- **Training Period**: {total_days} trading days
- **Regime Changes**: {regime_changes} transitions detected

## Disclaimers

This analysis is for educational and informational purposes only. It does not constitute investment advice or recommendations. Past performance does not guarantee future results. Market regimes can change rapidly, and the model's predictions are based on historical patterns that may not persist. Always conduct your own research and consult with qualified financial advisors before making investment decisions.

---
*Analysis performed using Hidden Regime framework with Hidden Markov Models*
"""
    
    # Save report
    report_file = os.path.join(output_dir, f'{symbol}_market_report.md')
    with open(report_file, 'w') as f:
        f.write(report)
    
    return report_file

def main():
    """Generate comprehensive market report for AAPL with blog-ready content."""
    
    print("🚀 Hidden Regime Market Report Generation Example")
    print("="*60)
    
    # Configuration
    TICKER = "AAPL"
    OUTPUT_DIR = './output/market_reports'
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📊 Generating market report for {TICKER}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    try:
        # Step 1: Load stock data
        print("\n1️⃣ Loading stock data...")
        data_config = DataConfig()
        analyzer = RegimeAnalyzer(data_config)
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        analysis = analyzer.analyze_stock(TICKER, start_date, end_date)
        
        if analysis is None:
            raise ValueError(f"Could not analyze {TICKER}")
        
        print(f"   ✅ Loaded {analysis['data_length']} days of data")
        
        # Step 2: Display current regime info
        print("\n2️⃣ Current regime analysis...")
        print(f"   ✅ Current regime: {analysis['current_regime']} (confidence: {analysis['confidence']:.1%})")
        print(f"   📈 Days in current regime: {analysis['days_in_regime']}")
        print(f"   📊 Total regime changes: {analysis['regime_changes']}")
        
        # Step 3: Load full data for visualization
        print("\n3️⃣ Loading full data for visualization...")
        data_loader = DataLoader(data_config)
        data = data_loader.load_stock_data(TICKER, start_date, end_date)
        
        if data is None:
            raise ValueError(f"Could not load data for {TICKER}")
            
        print(f"   ✅ Loaded full dataset with {len(data)} days")
        
        # Step 4: Create visualizations
        print("\n4️⃣ Creating publication-ready charts...")
        chart_file = create_regime_visualization(data, analysis, OUTPUT_DIR)
        print(f"   ✅ Created chart: {os.path.basename(chart_file)}")
        
        # Step 5: Generate report
        print("\n5️⃣ Generating markdown report...")
        report_file = generate_market_report(analysis, chart_file, OUTPUT_DIR)
        print(f"   ✅ Generated report: {os.path.basename(report_file)}")
        
        # Step 6: Summary
        print("\n✅ Market report generation completed successfully!")
        print(f"📄 Report file: {report_file}")
        print(f"📊 Chart file: {chart_file}")
        print(f"📁 Output directory: {OUTPUT_DIR}")
        
        # Display key findings
        print(f"\n📈 KEY FINDINGS:")
        print(f"   Current Regime: {analysis['current_regime']} ({analysis['confidence']:.1%} confidence)")
        print(f"   Days in Regime: {analysis['days_in_regime']}")
        print(f"   Expected Return: {analysis['expected_return']*252:.1%} (annualized)")
        print(f"   Expected Volatility: {analysis['expected_volatility']*np.sqrt(252):.1%} (annualized)")
        
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
        print("💥 Example failed - check error messages above")
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Example completed successfully!")
        print("🔗 You can now use the generated markdown report in your blog or convert to Jupyter notebook")
    else:
        print("\n❌ Example failed")
        exit(1)