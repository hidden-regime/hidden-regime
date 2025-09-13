#!/usr/bin/env python3
"""
Sector Momentum Screening Example

Demonstrates sector-based momentum screening using HMM regime detection across
different market sectors. Identifies momentum patterns within each sector and
provides insights for sector rotation strategies.

This example showcases:
- Multi-sector screening and comparison analysis
- Momentum pattern identification within sectors
- Sector rotation signal generation
- Cross-sector regime correlation analysis
- Professional sector analysis reports

Run this script to screen for sector momentum patterns and rotation opportunities
using regime-based analysis across major market sectors.
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
from hidden_regime.screener import (
    MarketScreener, ScreeningConfig,
    create_screening_report
)
from hidden_regime.screener.criteria import (
    create_momentum_criteria,
    create_regime_change_criteria,
    create_bull_regime_criteria,
    create_combined_criteria
)
from hidden_regime.screener.universes import (
    get_sector_universe,
    SECTOR_MAPPINGS,
    get_custom_universe
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """Run comprehensive sector momentum screening analysis."""
    
    print("🏭 Sector Momentum Screening Analysis")
    print("="*50)
    
    # Configuration
    OUTPUT_DIR = project_root / "examples" / "output" / "sector_momentum"
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define sectors to analyze
    SECTORS_TO_ANALYZE = [
        'Technology',
        'Financials', 
        'Healthcare',
        'Energy',
        'Consumer Discretionary',
        'Industrials'
    ]
    
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🎯 Analyzing {len(SECTORS_TO_ANALYZE)} sectors")
    
    try:
        # Step 1: Set up screening configuration
        print("\n1️⃣ Configuring sector screening...")
        
        screening_config = ScreeningConfig(
            period_days=200,  # ~8 months for momentum analysis
            max_workers=6,
            verbose=True
        )
        
        screener = MarketScreener(screening_config)
        
        # Step 2: Define screening criteria
        print("\n2️⃣ Setting up momentum screening criteria...")
        
        # High momentum criteria
        high_momentum_criteria = create_momentum_criteria(
            min_return=0.15,  # 15% minimum return
            name="High Momentum"
        )
        
        # Bull regime momentum
        bull_momentum_criteria = create_combined_criteria(
            create_bull_regime_criteria(min_confidence=0.7, max_days=20),
            create_momentum_criteria(min_return=0.05),
            use_and=True,
            name="Bull Regime Momentum"
        )
        
        # Recent momentum shift
        momentum_shift_criteria = create_combined_criteria(
            create_regime_change_criteria(max_days=14, min_confidence=0.6),
            create_momentum_criteria(min_return=0.0),  # Any positive momentum
            use_and=True,
            name="Momentum Shift"
        )
        
        print("   📋 Created 3 momentum screening criteria")
        
        # Step 3: Screen each sector
        print("\n3️⃣ Screening sectors for momentum patterns...")
        
        sector_results = {}
        sector_summaries = []
        
        for sector_name in SECTORS_TO_ANALYZE:
            print(f"\n   📊 Analyzing {sector_name} sector...")
            
            try:
                # Get sector stocks
                sector_stocks = get_sector_universe(sector_name, sample_size=15)  # Sample for demo
                
                if len(sector_stocks) < 3:
                    print(f"      ⚠️ Insufficient stocks in {sector_name}, skipping...")
                    continue
                
                print(f"      📈 Screening {len(sector_stocks)} stocks: {', '.join(sector_stocks[:5])}...")
                
                # Screen for different momentum patterns
                high_momentum_result = screener.screen_universe(
                    universe=sector_stocks,
                    criteria=high_momentum_criteria
                )
                
                bull_momentum_result = screener.screen_universe(
                    universe=sector_stocks,
                    criteria=bull_momentum_criteria
                )
                
                momentum_shift_result = screener.screen_universe(
                    universe=sector_stocks,
                    criteria=momentum_shift_criteria
                )
                
                # Store results
                sector_results[sector_name] = {
                    'stocks': sector_stocks,
                    'high_momentum': high_momentum_result,
                    'bull_momentum': bull_momentum_result,
                    'momentum_shift': momentum_shift_result
                }
                
                # Calculate sector summary metrics
                total_stocks = len(sector_stocks)
                high_momentum_count = high_momentum_result.passed_count
                bull_momentum_count = bull_momentum_result.passed_count
                momentum_shift_count = momentum_shift_result.passed_count
                
                sector_summary = {
                    'sector': sector_name,
                    'total_stocks': total_stocks,
                    'high_momentum_count': high_momentum_count,
                    'bull_momentum_count': bull_momentum_count,
                    'momentum_shift_count': momentum_shift_count,
                    'high_momentum_rate': high_momentum_count / total_stocks,
                    'bull_momentum_rate': bull_momentum_count / total_stocks,
                    'momentum_shift_rate': momentum_shift_count / total_stocks,
                    'overall_momentum_score': calculate_sector_momentum_score(
                        high_momentum_count, bull_momentum_count, momentum_shift_count, total_stocks
                    )
                }
                
                sector_summaries.append(sector_summary)
                
                print(f"      ✅ High momentum: {high_momentum_count}/{total_stocks}")
                print(f"      📈 Bull momentum: {bull_momentum_count}/{total_stocks}")
                print(f"      🔄 Momentum shifts: {momentum_shift_count}/{total_stocks}")
                print(f"      📊 Momentum score: {sector_summary['overall_momentum_score']:.2f}")
                
            except Exception as e:
                print(f"      ❌ Error analyzing {sector_name}: {str(e)}")
                continue
        
        # Step 4: Create sector comparison visualization
        print("\n4️⃣ Creating sector comparison visualizations...")
        
        if sector_summaries:
            comparison_chart_path = create_sector_comparison_chart(
                sector_summaries, OUTPUT_DIR
            )
            print(f"   📊 Created sector comparison chart: {comparison_chart_path.name}")
            
            momentum_heatmap_path = create_momentum_heatmap(
                sector_results, OUTPUT_DIR
            )
            print(f"   🔥 Created momentum heatmap: {momentum_heatmap_path.name}")
        
        # Step 5: Generate sector rotation analysis
        print("\n5️⃣ Generating sector rotation analysis...")
        
        rotation_analysis = generate_sector_rotation_analysis(
            sector_summaries, sector_results
        )
        
        rotation_path = OUTPUT_DIR / "sector_rotation_analysis.md"
        with open(rotation_path, 'w') as f:
            f.write(rotation_analysis)
        
        print(f"   🔄 Saved rotation analysis: {rotation_path.name}")
        
        # Step 6: Create comprehensive sector momentum report
        print("\n6️⃣ Generating comprehensive sector report...")
        
        sector_report = generate_comprehensive_sector_report(
            sector_summaries, sector_results, screening_config
        )
        
        report_path = OUTPUT_DIR / "sector_momentum_report.md"
        with open(report_path, 'w') as f:
            f.write(sector_report)
        
        print(f"   📝 Saved comprehensive report: {report_path.name}")
        
        # Step 7: Export detailed sector data
        print("\n7️⃣ Exporting sector screening data...")
        
        # Export sector summaries
        sector_df = pd.DataFrame(sector_summaries)
        sector_df = sector_df.round(3)
        
        summary_path = OUTPUT_DIR / "sector_momentum_summary.csv"
        sector_df.to_csv(summary_path, index=False)
        print(f"   💾 Saved sector summary: {summary_path.name}")
        
        # Export top opportunities by sector
        top_opportunities = compile_sector_opportunities(sector_results)
        
        opportunities_path = OUTPUT_DIR / "sector_opportunities.json"
        import json
        with open(opportunities_path, 'w') as f:
            json.dump(top_opportunities, f, indent=2, default=str)
        
        print(f"   🎯 Saved sector opportunities: {opportunities_path.name}")
        
        # Step 8: Generate sector rotation signals
        print("\n8️⃣ Generating rotation signals...")
        
        rotation_signals = generate_rotation_signals(sector_summaries)
        
        signals_path = OUTPUT_DIR / "rotation_signals.txt"
        with open(signals_path, 'w') as f:
            f.write(rotation_signals)
        
        print(f"   📡 Saved rotation signals: {signals_path.name}")
        
        print("\n✨ Sector Momentum Analysis Complete!")
        print(f"📁 All files saved to: {OUTPUT_DIR}")
        
        # Display key results
        if sector_summaries:
            print("\n📊 Sector Momentum Rankings:")
            ranked_sectors = sorted(sector_summaries, key=lambda x: x['overall_momentum_score'], reverse=True)
            
            for i, sector in enumerate(ranked_sectors, 1):
                momentum_status = get_momentum_status(sector['overall_momentum_score'])
                print(f"   {i}. {sector['sector']}: {sector['overall_momentum_score']:.2f} ({momentum_status})")
            
            print(f"\n🏆 Top Momentum Sector: {ranked_sectors[0]['sector']}")
            print(f"📉 Weakest Momentum: {ranked_sectors[-1]['sector']}")
            
            # Overall market momentum
            avg_momentum = np.mean([s['overall_momentum_score'] for s in sector_summaries])
            print(f"📈 Average Market Momentum: {avg_momentum:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in sector momentum analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def calculate_sector_momentum_score(high_momentum, bull_momentum, momentum_shift, total_stocks):
    """Calculate composite momentum score for sector."""
    
    if total_stocks == 0:
        return 0.0
    
    # Weight different momentum components
    high_momentum_weight = 0.4
    bull_momentum_weight = 0.4
    momentum_shift_weight = 0.2
    
    score = (
        (high_momentum / total_stocks) * high_momentum_weight +
        (bull_momentum / total_stocks) * bull_momentum_weight +
        (momentum_shift / total_stocks) * momentum_shift_weight
    )
    
    return score


def get_momentum_status(score):
    """Get momentum status description from score."""
    
    if score >= 0.6:
        return "Very Strong"
    elif score >= 0.4:
        return "Strong"
    elif score >= 0.25:
        return "Moderate"
    elif score >= 0.1:
        return "Weak"
    else:
        return "Very Weak"


def create_sector_comparison_chart(sector_summaries, output_dir):
    """Create sector comparison visualization."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data
    sectors = [s['sector'] for s in sector_summaries]
    momentum_scores = [s['overall_momentum_score'] for s in sector_summaries]
    high_momentum_rates = [s['high_momentum_rate'] for s in sector_summaries]
    bull_momentum_rates = [s['bull_momentum_rate'] for s in sector_summaries]
    momentum_shift_rates = [s['momentum_shift_rate'] for s in sector_summaries]
    
    # Colors for bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(sectors)))
    
    # 1. Overall momentum scores
    bars1 = ax1.bar(range(len(sectors)), momentum_scores, color=colors)
    ax1.set_title('Overall Sector Momentum Scores', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Momentum Score')
    ax1.set_xticks(range(len(sectors)))
    ax1.set_xticklabels([s.replace(' ', '\n') for s in sectors], rotation=0)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars1, momentum_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. High momentum rates
    bars2 = ax2.bar(range(len(sectors)), high_momentum_rates, color=colors, alpha=0.7)
    ax2.set_title('High Momentum Stock Rates by Sector', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Rate of High Momentum Stocks')
    ax2.set_xticks(range(len(sectors)))
    ax2.set_xticklabels([s.replace(' ', '\n') for s in sectors], rotation=0)
    ax2.grid(True, alpha=0.3)
    
    for bar, rate in zip(bars2, high_momentum_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
    
    # 3. Bull momentum rates
    bars3 = ax3.bar(range(len(sectors)), bull_momentum_rates, color=colors, alpha=0.7)
    ax3.set_title('Bull Regime Momentum Rates', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Rate of Bull Momentum Stocks')
    ax3.set_xticks(range(len(sectors)))
    ax3.set_xticklabels([s.replace(' ', '\n') for s in sectors], rotation=0)
    ax3.grid(True, alpha=0.3)
    
    for bar, rate in zip(bars3, bull_momentum_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
    
    # 4. Momentum shift rates
    bars4 = ax4.bar(range(len(sectors)), momentum_shift_rates, color=colors, alpha=0.7)
    ax4.set_title('Recent Momentum Shift Rates', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Rate of Momentum Shifts')
    ax4.set_xticks(range(len(sectors)))
    ax4.set_xticklabels([s.replace(' ', '\n') for s in sectors], rotation=0)
    ax4.grid(True, alpha=0.3)
    
    for bar, rate in zip(bars4, momentum_shift_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Sector Momentum Analysis Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    chart_path = output_dir / "sector_momentum_comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return chart_path


def create_momentum_heatmap(sector_results, output_dir):
    """Create momentum heatmap showing sector vs momentum type."""
    
    # Prepare heatmap data
    heatmap_data = []
    sectors = []
    momentum_types = ['High Momentum', 'Bull Momentum', 'Momentum Shift']
    
    for sector_name, results in sector_results.items():
        sectors.append(sector_name)
        
        total_stocks = len(results['stocks'])
        row = [
            results['high_momentum'].passed_count / total_stocks,
            results['bull_momentum'].passed_count / total_stocks,
            results['momentum_shift'].passed_count / total_stocks
        ]
        heatmap_data.append(row)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    
    heatmap_array = np.array(heatmap_data)
    
    sns.heatmap(
        heatmap_array,
        xticklabels=momentum_types,
        yticklabels=sectors,
        annot=True,
        fmt='.1%',
        cmap='YlOrRd',
        cbar_kws={'label': 'Rate of Stocks with Pattern'},
        square=False
    )
    
    plt.title('Sector Momentum Pattern Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Momentum Pattern Type', fontsize=12)
    plt.ylabel('Sector', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    heatmap_path = output_dir / "momentum_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return heatmap_path


def generate_sector_rotation_analysis(sector_summaries, sector_results):
    """Generate detailed sector rotation analysis."""
    
    # Sort sectors by momentum score
    ranked_sectors = sorted(sector_summaries, key=lambda x: x['overall_momentum_score'], reverse=True)
    
    content = f"""# Sector Rotation Analysis
*Momentum-Based Sector Allocation Insights*

## Executive Summary

Based on comprehensive momentum analysis across {len(sector_summaries)} major sectors, we identify clear sector rotation opportunities. The analysis reveals significant momentum dispersion across sectors, creating actionable allocation signals for portfolio managers.

### Momentum Rankings

"""
    
    # Add sector rankings
    content += """
| Rank | Sector | Momentum Score | Status | Recommendation |
|------|--------|----------------|---------|----------------|
"""
    
    for i, sector in enumerate(ranked_sectors, 1):
        status = get_momentum_status(sector['overall_momentum_score'])
        recommendation = get_rotation_recommendation(sector['overall_momentum_score'], i, len(ranked_sectors))
        
        content += f"| {i} | {sector['sector']} | {sector['overall_momentum_score']:.3f} | {status} | {recommendation} |\n"
    
    content += f"""

## Detailed Sector Analysis

"""
    
    # Add detailed analysis for each sector
    for sector in ranked_sectors:
        sector_name = sector['sector']
        
        content += f"""
### {sector_name} - {get_momentum_status(sector['overall_momentum_score'])} Momentum

**Key Metrics**:
- Momentum Score: {sector['overall_momentum_score']:.3f}
- High Momentum Stocks: {sector['high_momentum_count']}/{sector['total_stocks']} ({sector['high_momentum_rate']:.1%})
- Bull Regime Momentum: {sector['bull_momentum_count']}/{sector['total_stocks']} ({sector['bull_momentum_rate']:.1%})
- Recent Momentum Shifts: {sector['momentum_shift_count']}/{sector['total_stocks']} ({sector['momentum_shift_rate']:.1%})

"""
        
        # Add specific insights
        if sector['overall_momentum_score'] >= 0.4:
            content += f"""
**Investment Thesis**: Strong momentum across multiple metrics suggests this sector is in a favorable regime. Consider overweighting in growth-oriented portfolios.

**Key Drivers**:
- {sector['high_momentum_rate']:.1%} of stocks showing high momentum (>15% returns)
- {sector['bull_momentum_rate']:.1%} in confirmed bull regimes
- Recent momentum shifts indicate continued strength

**Recommended Action**: Increase allocation by 1-2% from neutral weight
"""
        elif sector['overall_momentum_score'] >= 0.2:
            content += f"""
**Investment Thesis**: Moderate momentum suggests selective opportunities within the sector. Focus on highest-quality names.

**Recommended Action**: Maintain neutral allocation, focus on stock selection
"""
        else:
            content += f"""
**Investment Thesis**: Weak momentum suggests defensive positioning may be appropriate. Consider underweighting or hedging exposure.

**Risk Factors**:
- Limited high-momentum opportunities
- Few stocks in confirmed bull regimes
- Sector may be entering challenging period

**Recommended Action**: Reduce allocation by 0.5-1% below neutral weight
"""
    
    content += """

## Portfolio Implementation Strategy

### Allocation Framework

**Strong Momentum Sectors (Score ≥ 0.4)**:
- Overweight by 1-2% above neutral
- Focus on momentum leaders within sector
- Use ETFs for broad exposure

**Moderate Momentum Sectors (Score 0.2-0.4)**:
- Maintain neutral allocation
- Emphasize stock selection over sector beta
- Monitor for momentum acceleration

**Weak Momentum Sectors (Score < 0.2)**:
- Underweight by 0.5-1% below neutral
- Consider defensive names or hedging
- Monitor for oversold opportunities

### Risk Management

1. **Maximum Overweight**: 3% above neutral for any sector
2. **Momentum Decay**: Reduce allocation if momentum score drops >30%
3. **Correlation Check**: Monitor for concentration in similar momentum profiles
4. **Rebalancing**: Monthly momentum reassessment and allocation adjustment

## Market Timing Implications

### Rotation Signals
"""
    
    # Generate rotation signals
    strongest_sector = ranked_sectors[0]
    weakest_sector = ranked_sectors[-1]
    
    content += f"""
- **Rotate Into**: {strongest_sector['sector']} (momentum score: {strongest_sector['overall_momentum_score']:.3f})
- **Rotate Out Of**: {weakest_sector['sector']} (momentum score: {weakest_sector['overall_momentum_score']:.3f})
- **Market Phase**: {'Growth/Momentum' if np.mean([s['overall_momentum_score'] for s in sector_summaries]) > 0.3 else 'Defensive/Value'}

### Tactical Considerations

1. **Economic Cycle**: Current momentum patterns suggest {'mid-cycle growth' if strongest_sector['sector'] in ['Technology', 'Consumer Discretionary'] else 'defensive positioning'}
2. **Interest Rate Environment**: {'Growth sectors' if strongest_sector['sector'] == 'Technology' else 'Value sectors'} currently favored
3. **Risk Appetite**: {'High' if np.mean([s['overall_momentum_score'] for s in sector_summaries]) > 0.4 else 'Moderate to Low'} based on momentum distribution

## Monitoring Framework

### Daily Monitoring
- Track top momentum stocks in each sector
- Monitor regime changes in sector leaders
- Watch for momentum shift signals

### Weekly Review
- Update sector momentum scores
- Assess rotation signal strength
- Review position sizing vs. targets

### Monthly Rebalancing
- Full sector momentum reassessment
- Portfolio allocation adjustments
- Performance attribution analysis

---

*This analysis provides systematic sector rotation insights based on quantitative momentum detection. For real-time updates and alerts, visit [hiddenregime.com](https://hiddenregime.com).*
"""
    
    return content


def get_rotation_recommendation(momentum_score, rank, total_sectors):
    """Get rotation recommendation based on momentum score and ranking."""
    
    if momentum_score >= 0.5:
        return "🟢 Strong Overweight (+2%)"
    elif momentum_score >= 0.4:
        return "🟢 Overweight (+1%)"
    elif momentum_score >= 0.25:
        return "🟡 Neutral Weight"
    elif momentum_score >= 0.15:
        return "🟠 Slight Underweight (-0.5%)"
    else:
        return "🔴 Underweight (-1%)"


def generate_comprehensive_sector_report(sector_summaries, sector_results, config):
    """Generate comprehensive sector momentum report."""
    
    total_stocks_analyzed = sum(s['total_stocks'] for s in sector_summaries)
    avg_momentum_score = np.mean([s['overall_momentum_score'] for s in sector_summaries])
    
    content = f"""# Comprehensive Sector Momentum Report
*Quantitative Analysis of Cross-Sector Momentum Patterns*

## Executive Summary

This comprehensive analysis evaluated momentum patterns across **{len(sector_summaries)} major market sectors**, analyzing **{total_stocks_analyzed} individual stocks** using Hidden Markov Model regime detection. The analysis reveals significant momentum dispersion across sectors with an average momentum score of **{avg_momentum_score:.3f}**.

### Key Findings
- **Sectors Analyzed**: {len(sector_summaries)}
- **Total Stocks**: {total_stocks_analyzed:,}
- **Average Momentum Score**: {avg_momentum_score:.3f}
- **Analysis Period**: {config.period_days} trading days

## Methodology

### Momentum Criteria
1. **High Momentum**: Stocks with >15% returns and strong trend characteristics
2. **Bull Regime Momentum**: Stocks in confirmed bullish regimes with positive momentum
3. **Momentum Shift**: Recent regime changes indicating momentum acceleration

### Scoring Framework
- **High Momentum Weight**: 40% of score
- **Bull Momentum Weight**: 40% of score  
- **Momentum Shift Weight**: 20% of score

### Universe Coverage
- Representative stocks from each major sector
- Quality filters applied for data reliability
- Minimum confidence thresholds for regime detection

## Sector Analysis Results

"""
    
    # Add detailed results for each sector
    for sector_summary in sorted(sector_summaries, key=lambda x: x['overall_momentum_score'], reverse=True):
        sector_name = sector_summary['sector']
        
        if sector_name in sector_results:
            sector_data = sector_results[sector_name]
            
            content += f"""
### {sector_name}

**Momentum Profile**: {get_momentum_status(sector_summary['overall_momentum_score'])}
- **Overall Score**: {sector_summary['overall_momentum_score']:.3f}
- **Stocks Analyzed**: {sector_summary['total_stocks']}

**Pattern Distribution**:
- High Momentum: {sector_summary['high_momentum_count']} stocks ({sector_summary['high_momentum_rate']:.1%})
- Bull Momentum: {sector_summary['bull_momentum_count']} stocks ({sector_summary['bull_momentum_rate']:.1%})
- Momentum Shifts: {sector_summary['momentum_shift_count']} stocks ({sector_summary['momentum_shift_rate']:.1%})

"""
            
            # Add top stocks if available
            if sector_data['high_momentum'].passed_count > 0:
                content += "**Top High Momentum Stocks**:\n"
                
                sorted_stocks = sorted(
                    sector_data['high_momentum'].passed_stocks.items(),
                    key=lambda x: x[1]['current_regime']['confidence'],
                    reverse=True
                )
                
                for ticker, analysis in sorted_stocks[:3]:  # Top 3
                    confidence = analysis['current_regime']['confidence']
                    recent_return = analysis['recent_metrics']['return_20d_annualized']
                    content += f"- **{ticker}**: {confidence:.1%} confidence, {recent_return:.1%} return\n"
                
                content += "\n"
    
    content += """

## Cross-Sector Momentum Analysis

### Momentum Distribution
"""
    
    # Add momentum distribution analysis
    high_momentum_sectors = [s for s in sector_summaries if s['overall_momentum_score'] >= 0.4]
    moderate_momentum_sectors = [s for s in sector_summaries if 0.2 <= s['overall_momentum_score'] < 0.4]
    low_momentum_sectors = [s for s in sector_summaries if s['overall_momentum_score'] < 0.2]
    
    content += f"""
- **High Momentum Sectors**: {len(high_momentum_sectors)} ({len(high_momentum_sectors)/len(sector_summaries):.1%})
- **Moderate Momentum Sectors**: {len(moderate_momentum_sectors)} ({len(moderate_momentum_sectors)/len(sector_summaries):.1%})
- **Low Momentum Sectors**: {len(low_momentum_sectors)} ({len(low_momentum_sectors)/len(sector_summaries):.1%})

### Market Regime Implications

**Current Market Character**: {'Growth-Oriented' if len(high_momentum_sectors) > len(low_momentum_sectors) else 'Value/Defensive-Oriented'}

**Sector Rotation Phase**: {'Active rotation opportunity' if max(s['overall_momentum_score'] for s in sector_summaries) - min(s['overall_momentum_score'] for s in sector_summaries) > 0.3 else 'Limited rotation opportunity'}

## Investment Strategy Applications

### Portfolio Construction
1. **Strategic Allocation**: Use momentum scores for sector weighting
2. **Tactical Overlay**: Focus on high-momentum sectors for growth strategies
3. **Risk Management**: Underweight low-momentum sectors in growth portfolios

### Trading Strategies
1. **Momentum Following**: Focus on sectors with accelerating momentum
2. **Regime Rotation**: Rotate based on sector momentum transitions
3. **Pairs Trading**: Long high-momentum vs. short low-momentum sectors

### Risk Considerations
- **Momentum Decay**: Monitor for momentum score deterioration
- **Sector Concentration**: Avoid overconcentration in similar momentum profiles
- **Regime Changes**: Prepare for momentum reversals in overextended sectors

## Future Monitoring

### Key Metrics to Track
1. **Momentum Score Changes**: Weekly momentum score updates
2. **Leadership Rotation**: Shifts in sector momentum rankings
3. **Breadth Analysis**: Number of stocks participating in sector momentum

### Rebalancing Triggers
- Momentum score changes >20% week-over-week
- New sectors entering high-momentum category
- Significant regime changes in sector leaders

---

*This analysis provides systematic sector momentum insights for institutional portfolio management and quantitative trading strategies. For real-time monitoring and alerts, visit [hiddenregime.com](https://hiddenregime.com).*

*Disclaimer: This analysis is for educational and research purposes only. Past performance does not guarantee future results. Please consult with qualified financial advisors before making investment decisions.*
"""
    
    return content


def compile_sector_opportunities(sector_results):
    """Compile top opportunities across all sectors."""
    
    all_opportunities = {}
    
    for sector_name, results in sector_results.items():
        sector_opportunities = {
            'high_momentum': [],
            'bull_momentum': [],
            'momentum_shift': []
        }
        
        # High momentum stocks
        for ticker, analysis in results['high_momentum'].passed_stocks.items():
            current_regime = analysis['current_regime']
            recent_metrics = analysis['recent_metrics']
            
            sector_opportunities['high_momentum'].append({
                'ticker': ticker,
                'confidence': current_regime['confidence'],
                'days_in_regime': current_regime['days_in_regime'],
                'recent_return': recent_metrics['return_20d_annualized'],
                'volatility': recent_metrics['volatility_20d_annualized']
            })
        
        # Bull momentum stocks
        for ticker, analysis in results['bull_momentum'].passed_stocks.items():
            current_regime = analysis['current_regime']
            recent_metrics = analysis['recent_metrics']
            
            sector_opportunities['bull_momentum'].append({
                'ticker': ticker,
                'confidence': current_regime['confidence'],
                'days_in_regime': current_regime['days_in_regime'],
                'recent_return': recent_metrics['return_20d_annualized'],
                'volatility': recent_metrics['volatility_20d_annualized']
            })
        
        # Momentum shift stocks
        for ticker, analysis in results['momentum_shift'].passed_stocks.items():
            current_regime = analysis['current_regime']
            recent_metrics = analysis['recent_metrics']
            
            sector_opportunities['momentum_shift'].append({
                'ticker': ticker,
                'confidence': current_regime['confidence'],
                'days_in_regime': current_regime['days_in_regime'],
                'recent_return': recent_metrics['return_20d_annualized'],
                'volatility': recent_metrics['volatility_20d_annualized']
            })
        
        # Sort each category by confidence
        for category in sector_opportunities:
            sector_opportunities[category].sort(key=lambda x: x['confidence'], reverse=True)
        
        all_opportunities[sector_name] = sector_opportunities
    
    return all_opportunities


def generate_rotation_signals(sector_summaries):
    """Generate sector rotation signals for immediate action."""
    
    ranked_sectors = sorted(sector_summaries, key=lambda x: x['overall_momentum_score'], reverse=True)
    
    signals = []
    signals.append("📡 SECTOR ROTATION SIGNALS")
    signals.append("=" * 40)
    signals.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    signals.append("")
    
    # Strong momentum signals
    strong_momentum = [s for s in ranked_sectors if s['overall_momentum_score'] >= 0.4]
    if strong_momentum:
        signals.append("🟢 OVERWEIGHT SECTORS (Strong Momentum)")
        signals.append("-" * 40)
        for sector in strong_momentum:
            signals.append(f"  📈 {sector['sector']}: Score {sector['overall_momentum_score']:.3f}")
            signals.append(f"     Action: Increase allocation by 1-2%")
        signals.append("")
    
    # Weak momentum signals
    weak_momentum = [s for s in ranked_sectors if s['overall_momentum_score'] < 0.2]
    if weak_momentum:
        signals.append("🔴 UNDERWEIGHT SECTORS (Weak Momentum)")
        signals.append("-" * 40)
        for sector in weak_momentum:
            signals.append(f"  📉 {sector['sector']}: Score {sector['overall_momentum_score']:.3f}")
            signals.append(f"     Action: Reduce allocation by 0.5-1%")
        signals.append("")
    
    # Market environment
    avg_momentum = np.mean([s['overall_momentum_score'] for s in sector_summaries])
    signals.append("🌍 MARKET ENVIRONMENT")
    signals.append("-" * 40)
    signals.append(f"Average Momentum Score: {avg_momentum:.3f}")
    
    if avg_momentum >= 0.35:
        signals.append("Environment: GROWTH/MOMENTUM FAVORED")
        signals.append("Strategy: Focus on high-momentum sectors")
    elif avg_momentum >= 0.25:
        signals.append("Environment: BALANCED CONDITIONS")
        signals.append("Strategy: Selective sector allocation")
    else:
        signals.append("Environment: DEFENSIVE CONDITIONS")
        signals.append("Strategy: Focus on quality and low volatility")
    
    signals.append("")
    signals.append("⚠️  Risk Management: Maximum 3% sector overweight")
    signals.append("📊 Monitor momentum scores weekly for changes")
    
    return "\n".join(signals)


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Sector momentum screening completed successfully!")
        print("🔄 Ready for sector rotation strategy implementation")
    else:
        print("\n💥 Sector screening failed - check error messages above")
        sys.exit(1)