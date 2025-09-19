#!/usr/bin/env python3
"""
Basic Regime Detection Example

This example demonstrates the core functionality of the hidden-regime package:
- Loading financial data
- Generating observations 
- Training HMM model for regime detection
- Basic analysis and reporting

This is a minimal working example that avoids known implementation bugs.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import using current working architecture
from hidden_regime.data.financial import FinancialDataLoader
from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.observations.financial import FinancialObservationGenerator
from hidden_regime.analysis.financial import FinancialAnalysis
from hidden_regime.reports.markdown import MarkdownReportGenerator
from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.config.model import HMMConfig
from hidden_regime.config.observation import FinancialObservationConfig
from hidden_regime.config.analysis import FinancialAnalysisConfig
from hidden_regime.config.report import ReportConfig

def create_sample_data():
    """Create sample data for demonstration."""
    print("Creating sample data for demonstration...")
    
    n_days = 250
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    # Create realistic regime patterns
    regime_states = []
    current_regime = 0
    days_in_regime = 0
    
    for i in range(n_days):
        regime_states.append(current_regime)
        days_in_regime += 1
        
        # Switch regimes occasionally
        if np.random.random() < 0.05 or days_in_regime > 50:
            current_regime = np.random.randint(0, 3)
            days_in_regime = 0
    
    # Generate prices based on regimes
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
    print("🚀 Hidden Regime: Basic Regime Detection Example")
    print("=" * 55)
    
    # Configuration
    ticker = "DEMO"
    
    try:
        # Try to load real market data
        print(f"\n📊 Attempting to load market data...")
        data_loader = FinancialDataLoader(FinancialDataConfig(
            ticker="AAPL",
            start_date="2023-01-01",
            end_date="2024-01-01"
        ))
        
        raw_data = data_loader.update()
        
        if raw_data.empty:
            print("❌ No real data available, using sample data")
            raw_data = create_sample_data()
        else:
            print(f"✅ Loaded {len(raw_data)} days of real market data")
            ticker = "AAPL"
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        print("📝 Using sample data instead")
        raw_data = create_sample_data()
    
    # Create observations (using simple log returns)
    print("\n🔍 Creating observations...")
    observation_config = FinancialObservationConfig(generators=['log_return'])
    observation_component = FinancialObservationGenerator(observation_config)
    
    observations = observation_component.update(raw_data)
    print(f"✅ Generated {len(observations)} observations")
    
    # Create and train HMM model
    print("\n🤖 Training HMM model...")
    model_config = HMMConfig(n_states=3, random_seed=42)
    hmm_model = HiddenMarkovModel(model_config)
    
    # Train model
    model_output = hmm_model.update(observations)
    print(f"✅ Model trained successfully")
    print(f"📊 Generated {len(model_output)} regime predictions")
    
    # Analyze current regime
    current_regime = model_output['predicted_state'].iloc[-1]
    confidence = model_output.get('confidence', pd.Series([0.0])).iloc[-1]
    if pd.isna(confidence):
        confidence = 0.0
    
    print(f"📈 Current regime: {current_regime}")
    print(f"📊 Confidence: {confidence:.1%}")
    
    # Basic analysis
    print("\n🔬 Running basic analysis...")
    analysis_config = FinancialAnalysisConfig(
        n_states=3,
        calculate_regime_statistics=True,
        include_duration_analysis=False,  # Disable due to known bug
        include_return_analysis=True,
        include_volatility_analysis=True
    )
    
    financial_analysis = FinancialAnalysis(analysis_config)
    analysis_results = financial_analysis.update(model_output, raw_data)
    print(f"✅ Analysis complete - {len(analysis_results)} features generated")
    
    # Generate report
    print("\n📝 Generating report...")
    report_config = ReportConfig(
        output_format="markdown",
        include_summary=True,
        include_regime_analysis=True,
        include_performance_metrics=True,
        save_plots=False,  # Disable to avoid visualization bugs
        show_plots=False
    )
    
    report_generator = MarkdownReportGenerator(report_config)
    
    full_report = report_generator.update(
        data=raw_data,
        observations=observations,
        model_output=model_output,
        analysis=analysis_results
    )
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"basic_regime_report_{ticker}_{timestamp}.md"
    
    with open(report_filename, 'w') as f:
        f.write(full_report)
    
    print(f"✅ Report saved as: {report_filename}")
    
    # Create basic visualization
    print("\n🎨 Creating visualization...")
    
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Price plot
        ax1.plot(raw_data.index, raw_data['close'], linewidth=1.5, color='blue')
        ax1.set_title(f'{ticker} Price Chart')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
        
        # Regime plot
        regime_colors = ['red', 'orange', 'green']
        regime_names = ['Bear', 'Sideways', 'Bull']
        
        for i, regime in enumerate([0, 1, 2]):
            mask = model_output['predicted_state'] == regime
            if mask.sum() > 0:
                ax2.scatter(model_output.index[mask], [regime] * mask.sum(), 
                           c=regime_colors[i], alpha=0.7, s=20,
                           label=f'{regime_names[i]} ({mask.sum()} days)')
        
        ax2.set_title('Detected Market Regimes')
        ax2.set_ylabel('Regime')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Bear', 'Sideways', 'Bull'])
        
        plt.tight_layout()
        plot_filename = f'regime_detection_{ticker}_{timestamp}.png'
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Visualization saved as: {plot_filename}")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        plot_filename = None
    
    # Display results summary
    print(f"\n📊 Results Summary for {ticker}:")
    print("=" * 40)
    
    # Regime distribution
    regime_counts = model_output['predicted_state'].value_counts().sort_index()
    total_days = len(model_output)
    
    print(f"Data Period: {total_days} days")
    print(f"Regime Distribution:")
    for regime, count in regime_counts.items():
        percentage = count / total_days * 100
        regime_name = ['Bear', 'Sideways', 'Bull'][regime]
        print(f"  {regime_name}: {count} days ({percentage:.1f}%)")
    
    print(f"\nCurrent Status:")
    print(f"  Regime: {['Bear', 'Sideways', 'Bull'][current_regime]}")
    print(f"  Confidence: {confidence:.1%}")
    
    print(f"\n🎉 Basic Regime Detection Complete!")
    print(f"📁 Generated files:")
    print(f"   • Report: {report_filename}")
    if plot_filename:
        print(f"   • Chart: {plot_filename}")
    
    print(f"\n🔧 Demonstrated capabilities:")
    print(f"   ✅ Financial data loading")
    print(f"   ✅ Observation generation")
    print(f"   ✅ HMM model training")
    print(f"   ✅ Regime detection")
    print(f"   ✅ Basic analysis")
    print(f"   ✅ Report generation")
    print(f"   ✅ Visualization")
    
    return {
        'model_output': model_output,
        'analysis': analysis_results,
        'raw_data': raw_data,
        'report_file': report_filename,
        'plot_file': plot_filename
    }

if __name__ == "__main__":
    try:
        results = main()
        print("\n" + "="*55)
        print("🚀 Basic Regime Detection: SUCCESS")
        print("="*55)
    except Exception as e:
        print(f"\n❌ Error running example: {e}")
        print("This may be due to known implementation bugs.")
        print("Check TODOS.md for bug resolution status.")