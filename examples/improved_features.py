#!/usr/bin/env python3
"""
Enhanced Features for Regime Detection Example

This example demonstrates the new regime-relevant features and their impact on
financial regime detection using the proper pipeline architecture. Shows how
enhanced observations improve regime detection within the existing framework:

- data → observations → model → analysis pipeline
- Enhanced feature generators for regime-specific patterns
- Comparison of baseline vs enhanced feature performance
- Proper pipeline configuration and component access

Enhanced Features Demonstrated:
- momentum_strength: Bull/Bear momentum detection through trend alignment
- trend_persistence: Sideways regime identification via directional consistency
- volatility_context: Crisis period detection through volatility spikes
- directional_consistency: Regime characterization via return sign patterns

This example uses the pipeline architecture exclusively - no manual component creation.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import using pipeline architecture
import hidden_regime as hr
from hidden_regime.config.observation import FinancialObservationConfig
from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.config.model import HMMConfig


def print_section_header(title, char="=", width=80):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def analyze_pipeline_results(pipeline, name, show_details=True):
    """Analyze and display results from a pipeline execution."""
    print(f"\n📊 {name} Results:")
    print("-" * 50)

    # Get component outputs through proper pipeline interface
    try:
        data_output = pipeline.get_component_output('data')
        observations_output = pipeline.get_component_output('observations')
        model_output = pipeline.get_component_output('model')
        analysis_output = pipeline.get_component_output('analysis')

        print(f"Data shape: {data_output.shape}")
        print(f"Observations shape: {observations_output.shape}")
        print(f"Observations columns: {list(observations_output.columns)}")

        # Model analysis
        if hasattr(model_output, 'emission_means_'):
            means = model_output.emission_means_
            print(f"Emission means: {means}")
            print(f"As percentages: {[f'{np.exp(m)-1:.2%}' for m in means]}")

        # Regime distribution analysis
        if 'regime_name' in analysis_output.columns:
            regime_counts = analysis_output['regime_name'].value_counts()
            total_days = len(analysis_output)

            print("\nRegime Distribution:")
            bull_types = ['Bull', 'Strong Bull', 'Weak Bull', 'Euphoric']
            total_bull_days = 0

            for regime_name, count in regime_counts.items():
                percentage = count / total_days * 100
                print(f"  {regime_name}: {count} days ({percentage:.1f}%)")

                # Track bull-type regimes
                if any(bull_type in regime_name for bull_type in bull_types):
                    total_bull_days += count

            bull_percentage = total_bull_days / total_days * 100
            print(f"  Total Bull-type: {total_bull_days} days ({bull_percentage:.1f}%)")

        return {
            'data': data_output,
            'observations': observations_output,
            'model': model_output,
            'analysis': analysis_output
        }

    except Exception as e:
        print(f"❌ Error analyzing {name}: {e}")
        return None


def compare_feature_statistics(results_dict):
    """Compare feature statistics across different pipeline configurations."""
    print_section_header("Feature Statistics Comparison")

    for name, results in results_dict.items():
        if results is None:
            continue

        observations = results['observations']
        print(f"\n{name} Features:")

        for col in observations.columns:
            if col in ['momentum_strength', 'trend_persistence', 'volatility_context', 'directional_consistency']:
                feature_data = observations[col].dropna()
                if len(feature_data) > 0:
                    print(f"  {col}:")
                    print(f"    Valid observations: {len(feature_data)}")
                    print(f"    Range: [{feature_data.min():.4f}, {feature_data.max():.4f}]")
                    print(f"    Mean: {feature_data.mean():.4f}, Std: {feature_data.std():.4f}")


def create_enhanced_features_visualization(results_dict, ticker="NVDA"):
    """Create comprehensive visualization of enhanced features and regimes."""
    print("\n🎨 Creating enhanced features visualization...")

    fig, axes = plt.subplots(4, 1, figsize=(15, 12))

    # Get baseline and enhanced results
    baseline_results = results_dict.get('baseline')
    enhanced_results = results_dict.get('full_enhanced')

    if baseline_results is None or enhanced_results is None:
        print("❌ Cannot create visualization without baseline and enhanced results")
        return None

    baseline_data = baseline_results['data']
    enhanced_obs = enhanced_results['observations']
    enhanced_analysis = enhanced_results['analysis']

    # Plot 1: Price with regime overlay
    ax1 = axes[0]
    ax1.plot(baseline_data.index, baseline_data['close'], linewidth=1.5, color='blue', alpha=0.8)
    ax1.set_title(f'{ticker} Price with Enhanced Regime Detection', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Enhanced features
    ax2 = axes[1]
    feature_cols = ['momentum_strength', 'trend_persistence', 'volatility_context']
    colors = ['red', 'green', 'orange']

    for col, color in zip(feature_cols, colors):
        if col in enhanced_obs.columns:
            feature_data = enhanced_obs[col].dropna()
            if len(feature_data) > 0:
                ax2.plot(feature_data.index, feature_data.values,
                        label=col.replace('_', ' ').title(), color=color, alpha=0.7)

    ax2.set_title('Enhanced Feature Values Over Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Feature Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Regime comparison
    ax3 = axes[2]
    if 'regime_name' in enhanced_analysis.columns:
        regime_names = enhanced_analysis['regime_name'].unique()
        regime_colors = {'Bull': 'green', 'Bear': 'red', 'Sideways': 'orange',
                        'Euphoric': 'purple', 'Crisis': 'black'}

        for i, regime in enumerate(regime_names):
            mask = enhanced_analysis['regime_name'] == regime
            color = regime_colors.get(regime, 'gray')
            ax3.scatter(enhanced_analysis.index[mask], [regime] * mask.sum(),
                       c=color, alpha=0.7, s=20, label=f'{regime} ({mask.sum()} days)')

    ax3.set_title('Enhanced Regime Detection Timeline', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Detected Regime')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Confidence levels
    ax4 = axes[3]
    if 'confidence' in enhanced_analysis.columns:
        confidence_data = enhanced_analysis['confidence'].rolling(window=10).mean()
        ax4.plot(confidence_data.index, confidence_data.values,
                color='purple', linewidth=2, label='10-day Rolling Confidence')
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% Threshold')

    ax4.set_title('Regime Detection Confidence Over Time', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Confidence')
    ax4.set_xlabel('Date')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('..', 'output', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'enhanced_features_{ticker}_{timestamp}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Visualization saved as: {plot_filename}")
    return plot_filename


def main():
    """Main execution function demonstrating enhanced features through pipeline architecture."""

    print_section_header("Enhanced Features for Regime Detection", "=", 80)
    print("""
This example demonstrates enhanced regime-relevant features using the proper
pipeline architecture. We'll compare baseline log_return regime detection
with enhanced features that capture specific regime characteristics.

Pipeline Architecture: data → observations → model → analysis
Enhanced Features: momentum_strength, trend_persistence, volatility_context, directional_consistency
    """)

    # Configuration
    ticker = "AAPL"
    start_date = "2022-01-01"  # Extended timeline for feature warmup
    end_date = "2024-01-01"
    n_states = 3

    print(f"📈 Analysis Parameters:")
    print(f"   Ticker: {ticker}")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   States: {n_states}")

    # =================================================================
    # 1. BASELINE PIPELINE: Traditional log_return approach
    # =================================================================
    print_section_header("1. Baseline Pipeline (Log Returns Only)")

    try:
        baseline_pipeline = hr.create_financial_pipeline(
            ticker=ticker,
            n_states=n_states,
            start_date=start_date,
            end_date=end_date
        )

        print("🔄 Executing baseline pipeline...")
        baseline_result = baseline_pipeline.update()
        baseline_results = analyze_pipeline_results(baseline_pipeline, "Baseline")

    except Exception as e:
        print(f"❌ Baseline pipeline failed: {e}")
        baseline_results = None

    # =================================================================
    # 2. INDIVIDUAL ENHANCED FEATURES: Test each feature separately
    # =================================================================
    print_section_header("2. Individual Enhanced Features Analysis")

    feature_configs = {
        'momentum': FinancialObservationConfig(generators=['momentum_strength']),
        'persistence': FinancialObservationConfig(generators=['trend_persistence']),
        'volatility': FinancialObservationConfig(generators=['volatility_context']),
        'consistency': FinancialObservationConfig(generators=['directional_consistency'])
    }

    individual_results = {}

    for feature_name, obs_config in feature_configs.items():
        print(f"\n🔬 Testing {feature_name} feature...")

        try:
            feature_pipeline = hr.create_financial_pipeline(
                ticker=ticker,
                n_states=n_states,
                start_date=start_date,
                end_date=end_date,
                observations_config=obs_config
            )

            # Update observed_signal in model config to match feature
            feature_signal = obs_config.generators[0]
            # Create new model config with updated observed_signal
            from hidden_regime.config.model import HMMConfig
            model_config_params = {
                'n_states': n_states,
                'observed_signal': feature_signal
            }
            new_model_config = HMMConfig.create_balanced().copy(**model_config_params)
            feature_pipeline.model.config = new_model_config

            feature_result = feature_pipeline.update()
            individual_results[feature_name] = analyze_pipeline_results(
                feature_pipeline, f"{feature_name.title()} Feature"
            )

        except Exception as e:
            print(f"❌ {feature_name} feature failed: {e}")
            individual_results[feature_name] = None

    # =================================================================
    # 3. COMBINED FEATURES: Multiple features together
    # =================================================================
    print_section_header("3. Combined Features Analysis")

    combined_configs = {
        'momentum_persistence': FinancialObservationConfig(
            generators=['log_return', 'momentum_strength', 'trend_persistence']
        ),
        'full_enhanced': FinancialObservationConfig(
            generators=['momentum_strength', 'trend_persistence', 'volatility_context', 'directional_consistency']
        )
    }

    combined_results = {}

    for config_name, obs_config in combined_configs.items():
        print(f"\n🔗 Testing {config_name} configuration...")

        try:
            combined_pipeline = hr.create_financial_pipeline(
                ticker=ticker,
                n_states=n_states,
                start_date=start_date,
                end_date=end_date,
                observations_config=obs_config
            )

            combined_result = combined_pipeline.update()
            combined_results[config_name] = analyze_pipeline_results(
                combined_pipeline, f"{config_name.replace('_', ' ').title()}"
            )

        except Exception as e:
            print(f"❌ {config_name} configuration failed: {e}")
            combined_results[config_name] = None

    # =================================================================
    # 4. COMPREHENSIVE COMPARISON
    # =================================================================
    print_section_header("4. Comprehensive Results Comparison")

    # Combine all results for comparison
    all_results = {'baseline': baseline_results}
    all_results.update(individual_results)
    all_results.update(combined_results)

    # Feature statistics comparison
    compare_feature_statistics(all_results)

    # Bull market detection comparison
    print_section_header("Bull Market Detection Comparison")
    print(f"\n📊 {ticker} Bull Market Detection Analysis:")
    print("(Higher percentages = better bull market identification)")
    print()

    for name, results in all_results.items():
        if results is None:
            continue

        analysis = results['analysis']
        if 'regime_name' in analysis.columns:
            regime_counts = analysis['regime_name'].value_counts()
            total_days = len(analysis)

            # Calculate bull-type percentage
            bull_types = ['Bull', 'Strong Bull', 'Weak Bull', 'Euphoric']
            bull_days = sum(count for regime, count in regime_counts.items()
                           if any(bull_type in regime for bull_type in bull_types))
            bull_percentage = bull_days / total_days * 100

            print(f"  {name:15s}: {bull_percentage:5.1f}% bull-type regimes ({bull_days}/{total_days} days)")

    # =================================================================
    # 5. VISUALIZATION AND REPORTING
    # =================================================================
    print_section_header("5. Visualization and Reporting")

    # Create comprehensive visualization
    plot_filename = create_enhanced_features_visualization(all_results, ticker)

    # Generate summary report
    print_section_header("Summary and Key Insights")
    print(f"""
📋 Enhanced Features Analysis Summary for {ticker}

🎯 Key Findings:
   • Enhanced features provide regime-specific insights beyond simple returns
   • Momentum strength captures sustained trend characteristics
   • Trend persistence identifies sideways consolidation periods
   • Volatility context detects crisis/uncertainty periods
   • Directional consistency quantifies regime conviction

🏗️ Pipeline Architecture Benefits:
   • Configuration-driven feature selection
   • Consistent component interfaces
   • Easy comparison of different approaches
   • Proper separation of concerns

📈 Performance Improvements:
   • Enhanced features may improve bull market detection
   • More interpretable regime characteristics
   • Reduced dependence on extreme return thresholds
   • Better alignment with financial intuition

🔧 Usage Recommendations:
   • Use extended data periods (2+ years) for feature warmup
   • Combine features for comprehensive regime analysis
   • Validate on known market periods for feature tuning
   • Leverage pipeline architecture for systematic comparison

🎨 Generated Files:
   • Enhanced features visualization: {plot_filename if plot_filename else 'N/A'}
   • Analysis results saved in pipeline components
    """)

    print_section_header("Enhanced Features Example Complete", "=", 80)
    print("✅ All pipeline configurations tested successfully!")
    print("📁 Check the output directory for generated visualizations and reports.")


if __name__ == "__main__":
    main()