#!/usr/bin/env python3
"""
Advanced Analysis Showcase - Phase 3 Enhancements Demo

Demonstrates the comprehensive analysis and visualization capabilities added in Phase 3:
- Enhanced FinancialAnalysis with technical indicator comparisons
- Comprehensive regime statistics and performance metrics
- Advanced report templates and visualizations
- Indicator performance comparison framework
- Comprehensive plotting and visualization suite

This example showcases the full pipeline with all Phase 3 enhancements.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hidden_regime.data import FinancialDataLoader
from hidden_regime.models import HiddenMarkovModel
from hidden_regime.observations import FinancialObservationGenerator
from hidden_regime.analysis import FinancialAnalysis, RegimePerformanceAnalyzer, IndicatorPerformanceComparator
from hidden_regime.reports import MarkdownReportGenerator
from hidden_regime.pipeline import Pipeline
from hidden_regime.config import (
    FinancialDataConfig,
    HMMConfig, 
    FinancialObservationConfig,
    FinancialAnalysisConfig,
    ReportConfig
)
from hidden_regime.visualization import (
    RegimePlotter,
    PerformancePlotter,
    ComparisonPlotter,
    create_regime_timeline_plot,
    create_multi_asset_regime_comparison
)

def create_sample_data():
    """Create sample data for demonstration if yfinance fails."""
    print("Creating sample data for demonstration...")
    
    # Generate synthetic price data with regime-like behavior
    np.random.seed(42)
    n_days = 500
    
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Regime states (0=Bear, 1=Sideways, 2=Bull)
    regime_states = []
    current_regime = 1  # Start sideways
    regime_duration = 0
    
    for i in range(n_days):
        if regime_duration == 0:
            # Determine regime duration
            if current_regime == 0:  # Bear
                regime_duration = np.random.poisson(15) + 5
            elif current_regime == 1:  # Sideways
                regime_duration = np.random.poisson(25) + 10
            else:  # Bull
                regime_duration = np.random.poisson(20) + 8
        
        regime_states.append(current_regime)
        regime_duration -= 1
        
        if regime_duration == 0:
            # Transition to new regime
            if current_regime == 0:  # Bear -> Sideways or Bull
                current_regime = np.random.choice([1, 2], p=[0.6, 0.4])
            elif current_regime == 1:  # Sideways -> Bear or Bull
                current_regime = np.random.choice([0, 2], p=[0.3, 0.7])
            else:  # Bull -> Bear or Sideways
                current_regime = np.random.choice([0, 1], p=[0.4, 0.6])
    
    # Generate price data based on regimes
    prices = [100.0]  # Starting price
    
    for i in range(1, n_days):
        regime = regime_states[i]
        
        if regime == 0:  # Bear
            daily_return = np.random.normal(-0.002, 0.025)
        elif regime == 1:  # Sideways
            daily_return = np.random.normal(0.0001, 0.015)
        else:  # Bull
            daily_return = np.random.normal(0.0015, 0.020)
        
        new_price = prices[-1] * (1 + daily_return)
        prices.append(max(new_price, 1.0))  # Prevent negative prices
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n_days)
    }, index=dates)
    
    # Ensure high >= close >= low and high >= open >= low
    data['high'] = data[['high', 'close', 'open']].max(axis=1)
    data['low'] = data[['low', 'close', 'open']].min(axis=1)
    
    return data

def main():
    """Main demonstration function."""
    print("🚀 Hidden Regime: Advanced Analysis Showcase (Phase 3)")
    print("=" * 60)
    
    # Configuration
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    
    try:
        # Load market data
        print(f"\n📊 Loading market data for {ticker}...")
        data_loader = FinancialDataLoader(FinancialDataConfig(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        ))
        
        raw_data = data_loader.update()
        
        if raw_data.empty:
            print("❌ No data retrieved, using sample data instead")
            raw_data = create_sample_data()
            ticker = "SAMPLE"
        else:
            print(f"✅ Loaded {len(raw_data)} days of {ticker} data")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        print("📝 Using sample data instead")
        raw_data = create_sample_data()
        ticker = "SAMPLE"
    
    # Create observation component
    print("\n🔍 Creating observations...")
    observation_config = FinancialObservationConfig(generators=['log_return'])
    observation_component = FinancialObservationGenerator(observation_config)
    
    # Generate observations
    observations = observation_component.update(raw_data)
    print(f"✅ Generated {len(observations)} observations")
    
    # Create and train HMM model
    print("\n🤖 Training HMM model...")
    model_config = HMMConfig(n_states=3, random_seed=42)
    hmm_model = HiddenMarkovModel(model_config)
    
    # Train model
    model_output = hmm_model.update(observations)
    print(f"✅ Model trained, generated {len(model_output)} predictions")
    print(f"📊 Current regime: {model_output['predicted_state'].iloc[-1]} "
          f"(confidence: {model_output['confidence'].iloc[-1]:.1%})")
    
    # Create enhanced analysis component
    print("\n🔬 Creating enhanced financial analysis...")
    analysis_config = FinancialAnalysisConfig(
        n_states=3,
        calculate_regime_statistics=True,
        include_duration_analysis=True,
        include_return_analysis=True,
        include_volatility_analysis=True,
        include_trading_signals=True,
        include_indicator_performance=True,
        indicator_comparisons=['rsi', 'macd', 'bollinger_bands', 'moving_average'],
        risk_adjustment=True
    )
    
    financial_analysis = FinancialAnalysis(analysis_config)
    
    # Run comprehensive analysis
    analysis_results = financial_analysis.update(model_output, raw_data=raw_data)
    print(f"✅ Analysis complete with {len(analysis_results.columns)} features")
    
    # Get comprehensive performance metrics
    print("\n📈 Generating comprehensive performance metrics...")
    performance_metrics = financial_analysis.get_comprehensive_performance_metrics()
    
    # Create indicator comparison analysis
    print("\n🔍 Running indicator performance comparison...")
    comparator = IndicatorPerformanceComparator()
    
    comparison_results = comparator.compare_regime_vs_indicators(
        analysis_results=analysis_results,
        raw_data=raw_data,
        indicators=['rsi', 'macd', 'bollinger_bands', 'moving_average']
    )
    
    print("✅ Indicator comparison complete")
    if 'indicator_analysis' in comparison_results:
        print(f"📊 Analyzed {len(comparison_results['indicator_analysis'])} indicators")
        
        # Show performance ratings
        for indicator, analysis in comparison_results['indicator_analysis'].items():
            rating = analysis['performance_rating']['rating']
            score = analysis['performance_rating']['composite_score']
            print(f"   • {indicator.upper()}: {rating} (score: {score:.3f})")
    
    # Generate advanced report
    print("\n📝 Generating comprehensive report...")
    report_config = ReportConfig(
        include_summary=True,
        include_regime_analysis=True,
        include_performance_metrics=True,
        include_risk_analysis=True,
        include_trading_signals=True,
        include_data_quality=True
    )
    
    report_generator = MarkdownReportGenerator(report_config)
    
    # Create comprehensive report
    full_report = report_generator.update(
        data=raw_data,
        observations=observations,
        model_output=model_output,
        analysis=analysis_results
    )
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"advanced_analysis_report_{ticker}_{timestamp}.md"
    
    with open(report_filename, 'w') as f:
        f.write(full_report)
    
    print(f"✅ Report saved as: {report_filename}")
    
    # Create advanced visualizations
    print("\n🎨 Creating advanced visualizations...")
    
    # 1. Comprehensive regime analysis dashboard
    print("   • Creating regime analysis dashboard...")
    regime_plotter = RegimePlotter(theme='financial', figsize=(20, 16))
    
    fig1 = regime_plotter.plot_comprehensive_regime_analysis(
        analysis_results=analysis_results,
        raw_data=raw_data,
        performance_metrics=performance_metrics
    )
    
    fig1.savefig(f'regime_dashboard_{ticker}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Performance analysis dashboard
    print("   • Creating performance dashboard...")
    performance_plotter = PerformancePlotter(theme='financial')
    
    fig2 = performance_plotter.plot_performance_dashboard(
        performance_metrics=performance_metrics
    )
    
    fig2.savefig(f'performance_dashboard_{ticker}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Indicator comparison dashboard
    print("   • Creating indicator comparison dashboard...")
    comparison_plotter = ComparisonPlotter(theme='financial')
    
    fig3 = comparison_plotter.plot_regime_vs_indicators_dashboard(
        comparison_results=comparison_results
    )
    
    fig3.savefig(f'indicator_comparison_{ticker}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Timeline visualization
    print("   • Creating timeline plot...")
    fig4 = create_regime_timeline_plot(
        analysis_results=analysis_results,
        raw_data=raw_data,
        figsize=(16, 10)
    )
    
    fig4.savefig(f'regime_timeline_{ticker}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # 5. Indicator comparison plot using the comparator
    if 'indicator_analysis' in comparison_results and comparison_results['indicator_analysis']:
        print("   • Creating indicator performance plot...")
        fig5 = comparator.plot_comparison_results(comparison_results)
        fig5.savefig(f'indicator_performance_{ticker}_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close(fig5)
    
    print("✅ All visualizations saved")
    
    # Create complete pipeline demonstration
    print("\n🔄 Demonstrating complete pipeline integration...")
    
    # Create pipeline with all enhanced components
    pipeline = Pipeline(
        data=data_loader,
        observation=observation_component,
        model=hmm_model,
        analysis=financial_analysis,
        report=report_generator
    )
    
    # Run pipeline update
    pipeline_result = pipeline.update()
    
    # Show pipeline summary
    summary_stats = pipeline.get_summary_stats()
    print(f"✅ Pipeline complete:")
    print(f"   • Updates: {summary_stats['update_count']}")
    print(f"   • Data shape: {summary_stats.get('data_shape', 'N/A')}")
    print(f"   • Components: {len([c for c in summary_stats['components'].values() if c])}")
    
    # Display key results
    print(f"\n📊 Key Results for {ticker}:")
    print("=" * 40)
    
    current_summary = financial_analysis.get_current_regime_summary()
    print(f"Current Regime: {current_summary.get('regime_type', 'Unknown')}")
    print(f"Confidence: {current_summary.get('confidence', 0):.1%}")
    print(f"Days in Regime: {current_summary.get('days_in_regime', 0)}")
    
    if 'position_signal' in current_summary:
        print(f"Position Signal: {current_summary['position_signal']:.3f}")
    
    # Performance summary
    if 'summary' in performance_metrics:
        perf_summary = performance_metrics['summary']
        print(f"\nPerformance Summary:")
        print(f"Quality Score: {perf_summary.get('quality_score', 0):.1%}")
        print(f"Stability Rating: {perf_summary.get('stability_rating', 'Unknown')}")
        print(f"Balance Score: {perf_summary.get('balance_score', 0):.1%}")
    
    # Top performing indicators
    if 'indicator_analysis' in comparison_results:
        print(f"\nTop Performing Indicators:")
        indicators_by_score = sorted(
            comparison_results['indicator_analysis'].items(),
            key=lambda x: x[1]['performance_rating']['composite_score'],
            reverse=True
        )
        
        for i, (indicator, analysis) in enumerate(indicators_by_score[:3]):
            score = analysis['performance_rating']['composite_score']
            rating = analysis['performance_rating']['rating']
            print(f"{i+1}. {indicator.upper()}: {score:.3f} ({rating})")
    
    print(f"\n🎉 Advanced Analysis Showcase Complete!")
    print(f"📁 Generated files:")
    print(f"   • Report: {report_filename}")
    print(f"   • Regime Dashboard: regime_dashboard_{ticker}_{timestamp}.png")
    print(f"   • Performance Dashboard: performance_dashboard_{ticker}_{timestamp}.png")
    print(f"   • Indicator Comparison: indicator_comparison_{ticker}_{timestamp}.png")
    print(f"   • Timeline Plot: regime_timeline_{ticker}_{timestamp}.png")
    if 'indicator_analysis' in comparison_results and comparison_results['indicator_analysis']:
        print(f"   • Indicator Performance: indicator_performance_{ticker}_{timestamp}.png")
    
    print(f"\n🔧 Phase 3 enhancements demonstrated:")
    print(f"   ✅ Enhanced FinancialAnalysis with technical indicators")
    print(f"   ✅ Comprehensive regime statistics and performance metrics")
    print(f"   ✅ Advanced report templates and visualizations")
    print(f"   ✅ Indicator performance comparison framework")
    print(f"   ✅ Comprehensive plotting and visualization suite")
    
    return analysis_results, performance_metrics, comparison_results

if __name__ == "__main__":
    results = main()
    print("\n" + "="*60)
    print("🚀 Phase 3 Analysis & Reports Enhancement: COMPLETE")
    print("="*60)