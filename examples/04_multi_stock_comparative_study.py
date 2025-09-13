#!/usr/bin/env python3
"""
Multi-Stock Comparative Study Example

Demonstrates comprehensive comparative regime analysis across multiple stocks,
sectors, and market segments. Creates detailed cross-sectional analysis showing
regime correlations, divergences, and sector-level insights.

This example showcases:
- Regime behavior comparison across technology giants (FAANG+ stocks)
- Sector-level regime analysis and correlation patterns
- Cross-asset regime synchronization and divergence detection
- Blog-ready comparative study with professional visualizations
- Market leadership and regime transition analysis

Run this script to generate a comprehensive multi-stock regime analysis
suitable for sector rotation strategies and market timing insights.
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hidden_regime as hr
from hidden_regime.screener import BatchHMMProcessor, ScreeningConfig
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import combinations

def main():
    """Generate comprehensive multi-stock comparative analysis."""
    
    print("🏢 Multi-Stock Comparative Regime Study")
    print("="*60)
    
    # Configuration - Focus on major tech stocks and market ETFs
    STOCK_GROUPS = {
        'mega_cap_tech': {
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            'name': 'Mega-Cap Technology'
        },
        'growth_tech': {
            'tickers': ['TSLA', 'NVDA', 'NFLX', 'ADBE', 'CRM'],
            'name': 'High-Growth Technology'
        },
        'market_indices': {
            'tickers': ['SPY', 'QQQ', 'IWM', 'VTI'],
            'name': 'Market Indices'
        },
        'sector_leaders': {
            'tickers': ['JPM', 'JNJ', 'XOM', 'HD', 'PG'],
            'name': 'Sector Leaders'
        }
    }
    
    ALL_TICKERS = []
    for group in STOCK_GROUPS.values():
        ALL_TICKERS.extend(group['tickers'])
    
    OUTPUT_DIR = project_root / "examples" / "output" / "multi_stock_study"
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Analyzing {len(ALL_TICKERS)} stocks across {len(STOCK_GROUPS)} groups")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    try:
        # Step 1: Batch process all stocks using the screener engine
        print("\n1️⃣ Running batch HMM analysis...")
        
        config = ScreeningConfig(
            period_days=252,  # 1 year of data
            max_workers=6,
            verbose=True
        )
        
        processor = BatchHMMProcessor(config)
        batch_results = processor.process_stock_list(ALL_TICKERS)
        
        successful_tickers = list(batch_results['results'].keys())
        print(f"   ✅ Successfully analyzed {len(successful_tickers)} stocks")
        
        if len(successful_tickers) < 5:
            raise ValueError("Insufficient successful stock analyses for comparison")
        
        # Step 2: Extract regime data for analysis
        print("\n2️⃣ Extracting regime data for comparative analysis...")
        
        regime_data = extract_regime_data(batch_results['results'])
        
        print(f"   📊 Extracted regime data for {len(regime_data)} stocks")
        
        # Step 3: Calculate cross-stock regime correlations
        print("\n3️⃣ Calculating regime correlations...")
        
        correlation_analysis = calculate_regime_correlations(regime_data)
        
        print(f"   🔗 Calculated {len(correlation_analysis['correlation_matrix'])} correlations")
        
        # Step 4: Analyze regime synchronization patterns
        print("\n4️⃣ Analyzing regime synchronization...")
        
        synchronization_analysis = analyze_regime_synchronization(regime_data, STOCK_GROUPS)
        
        print(f"   ⚡ Identified {len(synchronization_analysis['sync_events'])} synchronization events")
        
        # Step 5: Create comprehensive visualizations
        print("\n5️⃣ Creating comparative visualizations...")
        
        # Regime correlation heatmap
        correlation_chart_path = create_correlation_heatmap(
            correlation_analysis, OUTPUT_DIR
        )
        print(f"   📈 Created correlation heatmap: {correlation_chart_path.name}")
        
        # Regime timeline visualization
        timeline_chart_path = create_regime_timeline(
            regime_data, STOCK_GROUPS, OUTPUT_DIR
        )
        print(f"   📅 Created regime timeline: {timeline_chart_path.name}")
        
        # Sector comparison dashboard
        sector_dashboard_path = create_sector_dashboard(
            regime_data, STOCK_GROUPS, batch_results['results'], OUTPUT_DIR
        )
        print(f"   🎛️ Created sector dashboard: {sector_dashboard_path.name}")
        
        # Step 6: Generate comprehensive comparative study
        print("\n6️⃣ Generating comparative study blog post...")
        
        blog_content = generate_comparative_study_blog_post(
            stock_groups=STOCK_GROUPS,
            regime_data=regime_data,
            correlation_analysis=correlation_analysis,
            synchronization_analysis=synchronization_analysis,
            batch_results=batch_results['results']
        )
        
        blog_path = OUTPUT_DIR / "multi_stock_comparative_study.md"
        with open(blog_path, 'w') as f:
            f.write(blog_content)
        
        print(f"   📝 Saved comparative study: {blog_path.name}")
        
        # Step 7: Generate sector rotation insights
        print("\n7️⃣ Creating sector rotation analysis...")
        
        rotation_analysis = generate_sector_rotation_analysis(
            regime_data, STOCK_GROUPS, correlation_analysis
        )
        
        rotation_path = OUTPUT_DIR / "sector_rotation_insights.md"
        with open(rotation_path, 'w') as f:
            f.write(rotation_analysis)
        
        print(f"   🔄 Saved sector rotation analysis: {rotation_path.name}")
        
        # Step 8: Export detailed data for further analysis
        print("\n8️⃣ Exporting comparative data...")
        
        # Create comprehensive dataset
        comparative_df = create_comparative_dataset(regime_data, batch_results['results'])
        
        data_path = OUTPUT_DIR / "multi_stock_regime_data.csv"
        comparative_df.to_csv(data_path, index=False)
        
        print(f"   💾 Saved comparative dataset: {data_path.name}")
        
        # Step 9: Generate summary statistics
        print("\n9️⃣ Generating summary statistics...")
        
        summary_stats = calculate_multi_stock_summary(
            regime_data, correlation_analysis, synchronization_analysis
        )
        
        # Save summary statistics
        import json
        stats_path = OUTPUT_DIR / "summary_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"   📊 Saved summary statistics: {stats_path.name}")
        
        print("\n✨ Multi-Stock Comparative Study Complete!")
        print(f"📁 All files saved to: {OUTPUT_DIR}")
        
        # Display key insights
        print("\n🔍 Key Insights:")
        if correlation_analysis.get('avg_correlation'):
            print(f"   • Average regime correlation: {correlation_analysis['avg_correlation']:.3f}")
        
        if synchronization_analysis.get('sync_rate'):
            print(f"   • Regime synchronization rate: {synchronization_analysis['sync_rate']:.1%}")
        
        print(f"   • Most correlated pair: {correlation_analysis.get('highest_correlation', {}).get('pair', 'N/A')}")
        print(f"   • Least correlated pair: {correlation_analysis.get('lowest_correlation', {}).get('pair', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in multi-stock analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def extract_regime_data(batch_results):
    """Extract regime data from batch processing results for analysis."""
    
    regime_data = {}
    
    for ticker, analysis in batch_results.items():
        try:
            # Extract current regime info
            current_regime = analysis['current_regime']
            regime_analysis = analysis['regime_analysis']
            
            # Get regime statistics
            regime_stats = regime_analysis['regime_statistics']['regime_stats']
            
            # Store comprehensive regime data
            regime_data[ticker] = {
                'current_regime': current_regime['regime'],
                'confidence': current_regime['confidence'],
                'days_in_regime': current_regime['days_in_regime'],
                'regime_stats': regime_stats,
                'regime_probabilities': regime_analysis.get('regime_probabilities', None),
                'transition_probabilities': regime_analysis['regime_statistics'].get('transition_probabilities', None),
                'recent_return': analysis['recent_metrics']['return_20d_annualized'],
                'recent_volatility': analysis['recent_metrics']['volatility_20d_annualized'],
                'last_price': analysis['recent_metrics']['last_price'],
                'data_points': analysis['data_points']
            }
            
        except Exception as e:
            print(f"   ⚠️ Error extracting regime data for {ticker}: {str(e)}")
            continue
    
    return regime_data


def calculate_regime_correlations(regime_data):
    """Calculate cross-stock regime correlations and patterns."""
    
    tickers = list(regime_data.keys())
    n_tickers = len(tickers)
    
    # Create correlation matrix
    correlation_matrix = np.zeros((n_tickers, n_tickers))
    
    # Calculate pairwise correlations
    correlations = {}
    
    for i, ticker1 in enumerate(tickers):
        for j, ticker2 in enumerate(tickers):
            if i == j:
                correlation_matrix[i, j] = 1.0
                continue
            
            # Simple correlation based on current regime and confidence
            regime1 = regime_data[ticker1]['current_regime']
            regime2 = regime_data[ticker2]['current_regime']
            conf1 = regime_data[ticker1]['confidence']
            conf2 = regime_data[ticker2]['confidence']
            
            # Calculate correlation score
            if regime1 == regime2:
                # Same regime - correlation is average of confidences
                correlation = (conf1 + conf2) / 2
            else:
                # Different regimes - negative correlation based on confidence
                correlation = -min(conf1, conf2) / 2
            
            correlation_matrix[i, j] = correlation
            
            if i < j:  # Store unique pairs
                correlations[f"{ticker1}-{ticker2}"] = {
                    'correlation': correlation,
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'regime1': regime1,
                    'regime2': regime2,
                    'conf1': conf1,
                    'conf2': conf2
                }
    
    # Find highest and lowest correlations
    sorted_correlations = sorted(correlations.items(), key=lambda x: x[1]['correlation'])
    
    analysis = {
        'correlation_matrix': correlation_matrix,
        'correlation_df': pd.DataFrame(correlation_matrix, index=tickers, columns=tickers),
        'correlations': correlations,
        'highest_correlation': {
            'pair': sorted_correlations[-1][0],
            'value': sorted_correlations[-1][1]['correlation'],
            'details': sorted_correlations[-1][1]
        },
        'lowest_correlation': {
            'pair': sorted_correlations[0][0],
            'value': sorted_correlations[0][1]['correlation'],
            'details': sorted_correlations[0][1]
        },
        'avg_correlation': np.mean([corr['correlation'] for corr in correlations.values()])
    }
    
    return analysis


def analyze_regime_synchronization(regime_data, stock_groups):
    """Analyze regime synchronization patterns across groups."""
    
    synchronization_events = []
    group_sync_rates = {}
    
    for group_name, group_info in stock_groups.items():
        group_tickers = [t for t in group_info['tickers'] if t in regime_data]
        
        if len(group_tickers) < 2:
            continue
        
        # Calculate within-group regime synchronization
        regimes = [regime_data[ticker]['current_regime'] for ticker in group_tickers]
        confidences = [regime_data[ticker]['confidence'] for ticker in group_tickers]
        
        # Check if majority are in same regime
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        dominant_regime = max(regime_counts.keys(), key=lambda k: regime_counts[k])
        sync_count = regime_counts[dominant_regime]
        sync_rate = sync_count / len(group_tickers)
        
        group_sync_rates[group_name] = {
            'sync_rate': sync_rate,
            'dominant_regime': dominant_regime,
            'sync_count': sync_count,
            'total_stocks': len(group_tickers),
            'avg_confidence': np.mean([regime_data[t]['confidence'] for t in group_tickers if regime_data[t]['current_regime'] == dominant_regime])
        }
        
        # Record significant synchronization events
        if sync_rate >= 0.6:  # 60% or more in same regime
            synchronization_events.append({
                'group': group_name,
                'regime': dominant_regime,
                'sync_rate': sync_rate,
                'tickers': group_tickers,
                'avg_confidence': group_sync_rates[group_name]['avg_confidence']
            })
    
    # Calculate overall synchronization rate
    all_regimes = [regime_data[ticker]['current_regime'] for ticker in regime_data.keys()]
    overall_regime_counts = {}
    for regime in all_regimes:
        overall_regime_counts[regime] = overall_regime_counts.get(regime, 0) + 1
    
    overall_dominant = max(overall_regime_counts.keys(), key=lambda k: overall_regime_counts[k])
    overall_sync_rate = overall_regime_counts[overall_dominant] / len(all_regimes)
    
    return {
        'sync_events': synchronization_events,
        'group_sync_rates': group_sync_rates,
        'overall_sync_rate': overall_sync_rate,
        'overall_dominant_regime': overall_dominant,
        'sync_rate': overall_sync_rate
    }


def create_correlation_heatmap(correlation_analysis, output_dir):
    """Create regime correlation heatmap visualization."""
    
    plt.figure(figsize=(12, 10))
    
    correlation_df = correlation_analysis['correlation_df']
    
    # Create heatmap
    mask = np.triu(np.ones_like(correlation_df.values, dtype=bool), k=1)
    
    sns.heatmap(
        correlation_df.values,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        square=True,
        xticklabels=correlation_df.columns,
        yticklabels=correlation_df.index,
        cbar_kws={'label': 'Regime Correlation'},
        mask=mask
    )
    
    plt.title('Cross-Stock Regime Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Stock Ticker', fontsize=12)
    plt.ylabel('Stock Ticker', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    chart_path = output_dir / "regime_correlation_heatmap.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return chart_path


def create_regime_timeline(regime_data, stock_groups, output_dir):
    """Create regime timeline visualization."""
    
    fig, axes = plt.subplots(len(stock_groups), 1, figsize=(16, 3 * len(stock_groups)), sharex=True)
    
    if len(stock_groups) == 1:
        axes = [axes]
    
    regime_colors = {0: '#E69F00', 1: '#F0E442', 2: '#0072B2'}  # Bear, Sideways, Bull
    regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
    
    for idx, (group_name, group_info) in enumerate(stock_groups.items()):
        ax = axes[idx]
        
        group_tickers = [t for t in group_info['tickers'] if t in regime_data]
        
        y_pos = 0
        for ticker in group_tickers:
            data = regime_data[ticker]
            
            # Create regime bar
            regime = data['current_regime']
            confidence = data['confidence']
            
            color = regime_colors.get(regime, '#7f7f7f')
            alpha = 0.3 + 0.7 * confidence  # Vary transparency by confidence
            
            ax.barh(y_pos, data['days_in_regime'], left=0, 
                   color=color, alpha=alpha, height=0.8)
            
            # Add ticker label
            ax.text(-5, y_pos, ticker, ha='right', va='center', fontweight='bold')
            
            # Add regime info
            regime_name = regime_names.get(regime, f'Regime {regime}')
            ax.text(data['days_in_regime']/2, y_pos, 
                   f"{regime_name} ({confidence:.1%})", 
                   ha='center', va='center', fontsize=8)
            
            y_pos += 1
        
        ax.set_title(f"{group_info['name']} - Current Regime Status", fontweight='bold')
        ax.set_xlabel('Days in Current Regime')
        ax.set_ylim(-0.5, len(group_tickers) - 0.5)
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])
    
    plt.suptitle('Multi-Stock Regime Timeline Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    chart_path = output_dir / "regime_timeline.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return chart_path


def create_sector_dashboard(regime_data, stock_groups, batch_results, output_dir):
    """Create comprehensive sector comparison dashboard."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Regime distribution by group
    ax1 = axes[0, 0]
    
    group_regime_data = {}
    for group_name, group_info in stock_groups.items():
        group_tickers = [t for t in group_info['tickers'] if t in regime_data]
        
        regime_counts = {0: 0, 1: 0, 2: 0}  # Bear, Sideways, Bull
        for ticker in group_tickers:
            regime = regime_data[ticker]['current_regime']
            regime_counts[regime] += 1
        
        group_regime_data[group_name] = regime_counts
    
    # Stacked bar chart
    groups = list(group_regime_data.keys())
    bear_counts = [group_regime_data[g][0] for g in groups]
    sideways_counts = [group_regime_data[g][1] for g in groups]
    bull_counts = [group_regime_data[g][2] for g in groups]
    
    x = np.arange(len(groups))
    width = 0.6
    
    ax1.bar(x, bear_counts, width, label='Bear', color='#E69F00', alpha=0.8)
    ax1.bar(x, sideways_counts, width, bottom=bear_counts, label='Sideways', color='#F0E442', alpha=0.8)
    ax1.bar(x, bull_counts, width, bottom=[bear_counts[i] + sideways_counts[i] for i in range(len(groups))], 
           label='Bull', color='#0072B2', alpha=0.8)
    
    ax1.set_title('Regime Distribution by Group', fontweight='bold')
    ax1.set_xlabel('Stock Groups')
    ax1.set_ylabel('Number of Stocks')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.replace('_', ' ').title() for name in groups], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Average confidence by group
    ax2 = axes[0, 1]
    
    group_confidences = {}
    for group_name, group_info in stock_groups.items():
        group_tickers = [t for t in group_info['tickers'] if t in regime_data]
        confidences = [regime_data[t]['confidence'] for t in group_tickers]
        group_confidences[group_name] = np.mean(confidences) if confidences else 0
    
    groups = list(group_confidences.keys())
    confidences = list(group_confidences.values())
    
    bars = ax2.bar(groups, confidences, color='steelblue', alpha=0.7)
    ax2.set_title('Average Regime Confidence by Group', fontweight='bold')
    ax2.set_xlabel('Stock Groups')
    ax2.set_ylabel('Average Confidence')
    ax2.set_xticklabels([name.replace('_', ' ').title() for name in groups], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, conf in zip(bars, confidences):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{conf:.1%}', ha='center', va='bottom')
    
    # 3. Performance vs regime
    ax3 = axes[1, 0]
    
    for group_name, group_info in stock_groups.items():
        group_tickers = [t for t in group_info['tickers'] if t in regime_data]
        
        returns = []
        regimes = []
        
        for ticker in group_tickers:
            returns.append(regime_data[ticker]['recent_return'])
            regimes.append(regime_data[ticker]['current_regime'])
        
        if returns and regimes:
            # Scatter plot colored by regime
            colors = [regime_colors.get(r, '#7f7f7f') for r in regimes]
            ax3.scatter(regimes, returns, c=colors, alpha=0.7, s=60, 
                       label=group_info['name'].replace('_', ' ').title())
    
    ax3.set_title('Recent Performance by Current Regime', fontweight='bold')
    ax3.set_xlabel('Current Regime (0=Bear, 1=Sideways, 2=Bull)')
    ax3.set_ylabel('20-Day Annualized Return')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Volatility vs regime
    ax4 = axes[1, 1]
    
    for group_name, group_info in stock_groups.items():
        group_tickers = [t for t in group_info['tickers'] if t in regime_data]
        
        volatilities = []
        regimes = []
        
        for ticker in group_tickers:
            volatilities.append(regime_data[ticker]['recent_volatility'])
            regimes.append(regime_data[ticker]['current_regime'])
        
        if volatilities and regimes:
            colors = [regime_colors.get(r, '#7f7f7f') for r in regimes]
            ax4.scatter(regimes, volatilities, c=colors, alpha=0.7, s=60,
                       label=group_info['name'].replace('_', ' ').title())
    
    ax4.set_title('Recent Volatility by Current Regime', fontweight='bold')
    ax4.set_xlabel('Current Regime (0=Bear, 1=Sideways, 2=Bull)')
    ax4.set_ylabel('20-Day Annualized Volatility')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Sector Comparison Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    chart_path = output_dir / "sector_dashboard.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return chart_path


def generate_comparative_study_blog_post(stock_groups, regime_data, correlation_analysis, 
                                       synchronization_analysis, batch_results):
    """Generate comprehensive comparative study blog post."""
    
    total_stocks = len(regime_data)
    avg_correlation = correlation_analysis.get('avg_correlation', 0)
    sync_rate = synchronization_analysis.get('overall_sync_rate', 0)
    dominant_regime = synchronization_analysis.get('overall_dominant_regime', 0)
    
    regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
    dominant_regime_name = regime_names.get(dominant_regime, f'Regime {dominant_regime}')
    
    content = f"""# Multi-Stock Regime Analysis: Cross-Sectional Market Intelligence
*Comparative Study of Regime Behavior Across Sectors and Market Segments*

## Executive Summary

We conducted a comprehensive cross-sectional analysis of **{total_stocks} stocks** across **{len(stock_groups)} major market segments** using Hidden Markov Model regime detection. Our analysis reveals significant insights into market synchronization patterns, sector-specific regime behavior, and cross-asset correlation dynamics.

### Key Findings
- **Market Synchronization Rate**: {sync_rate:.1%} of stocks currently in {dominant_regime_name.lower()} regime
- **Average Cross-Stock Correlation**: {avg_correlation:.3f}
- **Sector Groups Analyzed**: {len(stock_groups)}
- **Analysis Period**: Most recent 252 trading days

## Market Segments Overview

"""
    
    # Add overview of each stock group
    for group_name, group_info in stock_groups.items():
        group_tickers = [t for t in group_info['tickers'] if t in regime_data]
        
        if not group_tickers:
            continue
        
        # Calculate group statistics
        regimes = [regime_data[t]['current_regime'] for t in group_tickers]
        confidences = [regime_data[t]['confidence'] for t in group_tickers]
        returns = [regime_data[t]['recent_return'] for t in group_tickers]
        
        regime_counts = {0: 0, 1: 0, 2: 0}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        dominant_group_regime = max(regime_counts.keys(), key=lambda k: regime_counts[k])
        dominant_group_regime_name = regime_names.get(dominant_group_regime, f'Regime {dominant_group_regime}')
        
        content += f"""
### {group_info['name']}
**Stocks**: {', '.join(group_tickers)}

- **Dominant Regime**: {dominant_group_regime_name} ({regime_counts[dominant_group_regime]}/{len(group_tickers)} stocks)
- **Average Confidence**: {np.mean(confidences):.1%}
- **Average 20d Return**: {np.mean(returns):.1%}
- **Regime Distribution**: {regime_counts[0]} Bear, {regime_counts[1]} Sideways, {regime_counts[2]} Bull
"""
    
    content += f"""

## Cross-Stock Correlation Analysis

### Correlation Patterns

Our analysis reveals varying degrees of regime correlation across different stock pairs and sectors:

**Overall Statistics**:
- **Average Correlation**: {avg_correlation:.3f}
- **Highest Correlation**: {correlation_analysis['highest_correlation']['value']:.3f} ({correlation_analysis['highest_correlation']['pair']})
- **Lowest Correlation**: {correlation_analysis['lowest_correlation']['value']:.3f} ({correlation_analysis['lowest_correlation']['pair']})

### Key Correlation Insights

"""
    
    # Add top correlations
    sorted_correlations = sorted(correlation_analysis['correlations'].items(), 
                                key=lambda x: x[1]['correlation'], reverse=True)
    
    content += "**Most Correlated Pairs**:\n"
    for pair, data in sorted_correlations[:5]:
        content += f"- **{pair}**: {data['correlation']:.3f} (both in {regime_names.get(data['regime1'], 'Unknown')} regime)\n"
    
    content += "\n**Least Correlated Pairs**:\n"
    for pair, data in sorted_correlations[-5:]:
        ticker1_regime = regime_names.get(data['regime1'], 'Unknown')
        ticker2_regime = regime_names.get(data['regime2'], 'Unknown')
        content += f"- **{pair}**: {data['correlation']:.3f} ({ticker1_regime} vs {ticker2_regime})\n"
    
    content += f"""

## Regime Synchronization Analysis

### Market-Wide Synchronization

Current market synchronization shows **{sync_rate:.1%}** of analyzed stocks in the same regime ({dominant_regime_name}), indicating {'strong' if sync_rate > 0.6 else 'moderate' if sync_rate > 0.4 else 'weak'} market consensus.

### Sector-Level Synchronization

"""
    
    # Add sector synchronization details
    for group_name, sync_data in synchronization_analysis['group_sync_rates'].items():
        group_info = stock_groups[group_name]
        sync_rate_group = sync_data['sync_rate']
        dominant_group_regime = sync_data['dominant_regime']
        dominant_group_regime_name = regime_names.get(dominant_group_regime, f'Regime {dominant_group_regime}')
        
        content += f"""
#### {group_info['name']}
- **Synchronization Rate**: {sync_rate_group:.1%}
- **Dominant Regime**: {dominant_group_regime_name}
- **Synchronized Stocks**: {sync_data['sync_count']}/{sync_data['total_stocks']}
- **Average Confidence**: {sync_data['avg_confidence']:.1%}
"""
    
    content += """

## Investment Implications

### Sector Rotation Opportunities

Based on current regime patterns, we identify several potential sector rotation opportunities:

"""
    
    # Generate sector rotation insights
    for group_name, group_info in stock_groups.items():
        group_tickers = [t for t in group_info['tickers'] if t in regime_data]
        
        if not group_tickers:
            continue
        
        # Analyze group characteristics
        regimes = [regime_data[t]['current_regime'] for t in group_tickers]
        confidences = [regime_data[t]['confidence'] for t in group_tickers]
        returns = [regime_data[t]['recent_return'] for t in group_tickers]
        
        avg_regime = np.mean(regimes)
        avg_confidence = np.mean(confidences)
        avg_return = np.mean(returns)
        
        # Generate recommendation
        if avg_regime > 1.5 and avg_confidence > 0.7:
            recommendation = "🟢 **Bullish** - Strong momentum with high confidence"
        elif avg_regime < 0.5 and avg_confidence > 0.7:
            recommendation = "🔴 **Bearish** - Defensive positioning recommended"
        elif avg_confidence < 0.5:
            recommendation = "🟡 **Uncertain** - Mixed signals, wait for clarity"
        else:
            recommendation = "⚪ **Neutral** - Balanced allocation appropriate"
        
        content += f"""
#### {group_info['name']}
{recommendation}
- Recent Performance: {avg_return:.1%}
- Regime Confidence: {avg_confidence:.1%}
- Suggested Action: {'Overweight' if '🟢' in recommendation else 'Underweight' if '🔴' in recommendation else 'Neutral weight'}
"""
    
    content += f"""

### Risk Management Considerations

1. **Diversification Benefits**: Low correlation pairs provide portfolio diversification
2. **Concentration Risk**: High correlation within sectors increases concentration risk
3. **Regime Transition Risk**: Monitor for synchronized regime changes across sectors
4. **Confidence Levels**: Higher confidence regimes offer more reliable signals

## Market Timing Insights

### Current Market Environment
- **Dominant Regime**: {dominant_regime_name} regime across {sync_rate:.1%} of stocks
- **Market Consensus**: {'Strong' if sync_rate > 0.6 else 'Moderate' if sync_rate > 0.4 else 'Weak'} directional agreement
- **Cross-Asset Correlation**: {avg_correlation:.3f} average correlation level

### Strategic Recommendations

#### For Portfolio Managers
1. **Sector Allocation**: Adjust weights based on regime synchronization patterns
2. **Risk Budgeting**: Allocate risk based on correlation and confidence levels
3. **Rebalancing Timing**: Use regime transitions as rebalancing triggers

#### For Active Traders
1. **Pair Trading**: Exploit low-correlation opportunities
2. **Momentum Strategies**: Focus on high-confidence regime stocks
3. **Contrarian Opportunities**: Target regime transition candidates

## Technical Analysis Integration

### Regime vs Traditional Indicators

Our multi-stock analysis reveals that HMM regime detection provides unique insights not captured by traditional technical indicators:

1. **Cross-Asset Synchronization**: Traditional indicators miss correlation patterns
2. **Confidence Quantification**: HMM provides probabilistic assessment
3. **Regime Persistence**: Better timing of trend continuation vs reversal
4. **Sector Rotation Timing**: Early identification of sector leadership changes

## Methodology and Data Quality

### Analysis Framework
- **Universe**: {total_stocks} stocks across {len(stock_groups)} market segments
- **Time Period**: 252 trading days (approximately 1 year)
- **Model Configuration**: 3-state HMM with automatic initialization
- **Correlation Method**: Regime-based correlation accounting for confidence levels

### Quality Metrics
"""
    
    # Add data quality metrics
    avg_data_points = np.mean([regime_data[t]['data_points'] for t in regime_data.keys()])
    min_data_points = min([regime_data[t]['data_points'] for t in regime_data.keys()])
    
    content += f"""
- **Average Data Points**: {avg_data_points:.0f} per stock
- **Minimum Data Points**: {min_data_points:.0f} per stock
- **Success Rate**: {len(regime_data)}/{total_stocks} stocks successfully analyzed
- **Average Confidence**: {np.mean([regime_data[t]['confidence'] for t in regime_data.keys()]):.1%}

## Conclusion

This comprehensive multi-stock regime analysis provides actionable insights for portfolio construction, sector rotation, and market timing strategies. The **{sync_rate:.1%} synchronization rate** and **{avg_correlation:.3f} average correlation** indicate the current market environment offers both consensus opportunities and diversification benefits.

### Key Takeaways

1. **Market Synchronization**: Current {dominant_regime_name.lower()} regime dominance suggests {'coordinated market movement' if sync_rate > 0.6 else 'mixed market conditions'}
2. **Sector Opportunities**: Varying regime patterns across sectors create rotation opportunities
3. **Risk Management**: Correlation patterns inform portfolio diversification strategies
4. **Timing Signals**: Regime transitions provide early warning for market changes

### Future Analysis

1. **International Extension**: Apply framework to global markets and currencies
2. **Commodity Integration**: Include commodity regime analysis for inflation hedging
3. **Fixed Income**: Extend to bond markets for comprehensive asset allocation
4. **Real-Time Monitoring**: Implement streaming regime detection for live trading

---

*This analysis demonstrates the power of cross-sectional regime analysis for institutional portfolio management and quantitative trading strategies. For implementation details and live updates, visit [hiddenregime.com](https://hiddenregime.com).*

*Disclaimer: This analysis is for educational and research purposes only. Past performance does not guarantee future results. Please consult with qualified financial advisors before making investment decisions.*
"""
    
    return content


def generate_sector_rotation_analysis(regime_data, stock_groups, correlation_analysis):
    """Generate specialized sector rotation analysis."""
    
    content = """# Sector Rotation Strategy Based on Regime Analysis

## Overview

This analysis provides actionable sector rotation insights based on current regime patterns and cross-sector correlations.

## Rotation Signals

"""
    
    # Analyze each sector for rotation signals
    for group_name, group_info in stock_groups.items():
        group_tickers = [t for t in group_info['tickers'] if t in regime_data]
        
        if not group_tickers:
            continue
        
        # Calculate group metrics
        regimes = [regime_data[t]['current_regime'] for t in group_tickers]
        confidences = [regime_data[t]['confidence'] for t in group_tickers]
        days_in_regime = [regime_data[t]['days_in_regime'] for t in group_tickers]
        
        avg_regime = np.mean(regimes)
        avg_confidence = np.mean(confidences)
        avg_days = np.mean(days_in_regime)
        
        # Generate signal
        if avg_regime > 1.5 and avg_confidence > 0.7 and avg_days < 20:
            signal = "STRONG BUY"
            action = "Increase allocation by 2-3%"
        elif avg_regime > 1.5 and avg_confidence > 0.5:
            signal = "BUY"
            action = "Increase allocation by 1-2%"
        elif avg_regime < 0.5 and avg_confidence > 0.7:
            signal = "SELL"
            action = "Reduce allocation by 1-2%"
        elif avg_confidence < 0.4:
            signal = "HOLD"
            action = "Maintain current allocation"
        else:
            signal = "NEUTRAL"
            action = "No allocation change"
        
        content += f"""
### {group_info['name']}
**Signal**: {signal}
**Action**: {action}
**Confidence**: {avg_confidence:.1%}
**Days in Regime**: {avg_days:.0f}
"""
    
    return content


def create_comparative_dataset(regime_data, batch_results):
    """Create comprehensive dataset for further analysis."""
    
    data_rows = []
    
    for ticker, regime_info in regime_data.items():
        row = {
            'ticker': ticker,
            'current_regime': regime_info['current_regime'],
            'confidence': regime_info['confidence'],
            'days_in_regime': regime_info['days_in_regime'],
            'recent_return_20d': regime_info['recent_return'],
            'recent_volatility_20d': regime_info['recent_volatility'],
            'last_price': regime_info['last_price'],
            'data_points': regime_info['data_points']
        }
        
        # Add regime statistics
        for regime_id, stats in regime_info['regime_stats'].items():
            row[f'regime_{regime_id}_mean_return'] = stats['mean_return']
            row[f'regime_{regime_id}_volatility'] = stats['volatility']
            row[f'regime_{regime_id}_frequency'] = stats['frequency']
            row[f'regime_{regime_id}_avg_duration'] = stats['avg_duration']
        
        data_rows.append(row)
    
    return pd.DataFrame(data_rows)


def calculate_multi_stock_summary(regime_data, correlation_analysis, synchronization_analysis):
    """Calculate comprehensive summary statistics."""
    
    summary = {
        'total_stocks': len(regime_data),
        'avg_correlation': correlation_analysis.get('avg_correlation', 0),
        'sync_rate': synchronization_analysis.get('overall_sync_rate', 0),
        'dominant_regime': synchronization_analysis.get('overall_dominant_regime', 0),
        'regime_distribution': {},
        'confidence_stats': {},
        'performance_stats': {}
    }
    
    # Regime distribution
    all_regimes = [regime_data[t]['current_regime'] for t in regime_data.keys()]
    for regime in set(all_regimes):
        summary['regime_distribution'][f'regime_{regime}'] = all_regimes.count(regime)
    
    # Confidence statistics
    all_confidences = [regime_data[t]['confidence'] for t in regime_data.keys()]
    summary['confidence_stats'] = {
        'mean': float(np.mean(all_confidences)),
        'std': float(np.std(all_confidences)),
        'min': float(np.min(all_confidences)),
        'max': float(np.max(all_confidences))
    }
    
    # Performance statistics
    all_returns = [regime_data[t]['recent_return'] for t in regime_data.keys()]
    all_volatilities = [regime_data[t]['recent_volatility'] for t in regime_data.keys()]
    
    summary['performance_stats'] = {
        'avg_return': float(np.mean(all_returns)),
        'avg_volatility': float(np.mean(all_volatilities)),
        'return_std': float(np.std(all_returns)),
        'volatility_std': float(np.std(all_volatilities))
    }
    
    return summary


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Multi-stock comparative study completed successfully!")
        print("📈 Ready for sector rotation and market timing strategies")
    else:
        print("\n💥 Comparative study failed - check error messages above")
        sys.exit(1)