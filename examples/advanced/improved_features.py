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

import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend
import warnings
from datetime import datetime

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import using pipeline architecture
import hidden_regime as hr
from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.config.model import HMMConfig
from hidden_regime.config.observation import FinancialObservationConfig


def print_section_header(title, char="=", width=80):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def analyze_pipeline_results(pipeline, name, show_details=True):
    """Analyze and display results from a pipeline execution."""
    print(f"\n {name} Results:")
    print("-" * 50)

    # Get component outputs through proper pipeline interface
    try:
        data_output = pipeline.get_component_output("data")
        observations_output = pipeline.get_component_output("observations")
        model_output = pipeline.get_component_output("model")
        analysis_output = pipeline.get_component_output("analysis")

        print(f"Data shape: {data_output.shape}")
        print(f"Observations shape: {observations_output.shape}")
        print(f"Observations columns: {list(observations_output.columns)}")

        # Model analysis
        if hasattr(model_output, "emission_means_"):
            means = model_output.emission_means_
            print(f"Emission means: {means}")
            print(f"As percentages: {[f'{np.exp(m)-1:.2%}' for m in means]}")

        # Regime distribution analysis
        if "regime_name" in analysis_output.columns:
            regime_counts = analysis_output["regime_name"].value_counts()
            total_days = len(analysis_output)

            print("\nRegime Distribution:")
            bull_types = ["Bull", "Strong Bull", "Weak Bull", "Euphoric"]
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
            "data": data_output,
            "observations": observations_output,
            "model": model_output,
            "analysis": analysis_output,
        }

    except Exception as e:
        print(f" Error analyzing {name}: {e}")
        return None


def compare_feature_statistics(results_dict):
    """Compare feature statistics across different pipeline configurations."""
    print_section_header("Feature Statistics Comparison")

    for name, results in results_dict.items():
        if results is None:
            continue

        observations = results["observations"]
        print(f"\n{name} Features:")

        for col in observations.columns:
            if col in [
                "momentum_strength",
                "trend_persistence",
                "volatility_context",
                "directional_consistency",
            ]:
                feature_data = observations[col].dropna()
                if len(feature_data) > 0:
                    print(f"  {col}:")
                    print(f"    Valid observations: {len(feature_data)}")
                    print(
                        f"    Range: [{feature_data.min():.4f}, {feature_data.max():.4f}]"
                    )
                    print(
                        f"    Mean: {feature_data.mean():.4f}, Std: {feature_data.std():.4f}"
                    )


def create_enhanced_features_visualization(results_dict, ticker="NVDA"):
    """Create comprehensive visualization of enhanced features and regimes."""
    print("\n🎨 Creating enhanced features visualization...")

    fig, axes = plt.subplots(4, 1, figsize=(15, 12))

    # Get baseline and enhanced results
    baseline_results = results_dict.get("baseline")
    enhanced_results = results_dict.get("full_enhanced")

    if baseline_results is None or enhanced_results is None:
        print(" Cannot create visualization without baseline and enhanced results")
        return None

    baseline_data = baseline_results["data"]
    enhanced_obs = enhanced_results["observations"]
    enhanced_analysis = enhanced_results["analysis"]

    # Plot 1: Price with regime overlay
    ax1 = axes[0]
    ax1.plot(
        baseline_data.index,
        baseline_data["close"],
        linewidth=1.5,
        color="blue",
        alpha=0.8,
    )
    ax1.set_title(
        f"{ticker} Price with Enhanced Regime Detection", fontsize=14, fontweight="bold"
    )
    ax1.set_ylabel("Price ($)")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Enhanced features
    ax2 = axes[1]
    feature_cols = ["momentum_strength", "trend_persistence", "volatility_context"]
    colors = ["red", "green", "orange"]

    for col, color in zip(feature_cols, colors):
        if col in enhanced_obs.columns:
            feature_data = enhanced_obs[col].dropna()
            if len(feature_data) > 0:
                ax2.plot(
                    feature_data.index,
                    feature_data.values,
                    label=col.replace("_", " ").title(),
                    color=color,
                    alpha=0.7,
                )

    ax2.set_title("Enhanced Feature Values Over Time", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Feature Value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Regime comparison
    ax3 = axes[2]
    if "regime_name" in enhanced_analysis.columns:
        regime_names = enhanced_analysis["regime_name"].unique()
        regime_colors = {
            "Bull": "green",
            "Bear": "red",
            "Sideways": "orange",
            "Euphoric": "purple",
            "Crisis": "black",
        }

        for i, regime in enumerate(regime_names):
            mask = enhanced_analysis["regime_name"] == regime
            color = regime_colors.get(regime, "gray")
            ax3.scatter(
                enhanced_analysis.index[mask],
                [regime] * mask.sum(),
                c=color,
                alpha=0.7,
                s=20,
                label=f"{regime} ({mask.sum()} days)",
            )

    ax3.set_title("Enhanced Regime Detection Timeline", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Detected Regime")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Confidence levels
    ax4 = axes[3]
    if "confidence" in enhanced_analysis.columns:
        confidence_data = enhanced_analysis["confidence"].rolling(window=10).mean()
        ax4.plot(
            confidence_data.index,
            confidence_data.values,
            color="purple",
            linewidth=2,
            label="10-day Rolling Confidence",
        )
        ax4.axhline(
            y=0.5, color="red", linestyle="--", alpha=0.5, label="50% Threshold"
        )

    ax4.set_title(
        "Regime Detection Confidence Over Time", fontsize=12, fontweight="bold"
    )
    ax4.set_ylabel("Confidence")
    ax4.set_xlabel("Date")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("..", "output", "plots")
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(
        output_dir, f"enhanced_features_{ticker}_{timestamp}.png"
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f" Visualization saved as: {plot_filename}")
    return plot_filename


def create_individual_feature_plots(results_dict, ticker="AAPL"):
    """Create individual detailed plots for each enhanced feature."""
    print("\n🎨 Creating individual feature plots...")

    enhanced_features = [
        "momentum_strength",
        "trend_persistence",
        "volatility_context",
        "directional_consistency",
    ]
    feature_descriptions = {
        "momentum_strength": "Bull/Bear Momentum Detection",
        "trend_persistence": "Sideways Regime Identification",
        "volatility_context": "Crisis Period Detection",
        "directional_consistency": "Return Sign Pattern Analysis",
    }

    individual_plot_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use absolute path for output directory
    output_dir = os.path.abspath(os.path.join("..", "output", "plots"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Individual plots will be saved to: {output_dir}")

    # Get baseline data for reference
    baseline_results = results_dict.get("baseline")
    if baseline_results is None:
        print(" Cannot create individual plots without baseline results")
        return []

    baseline_data = baseline_results["data"]

    for feature_name in enhanced_features:
        # Check if we have results for this feature
        feature_results = None
        for key, results in results_dict.items():
            if results and "observations" in results:
                if feature_name in results["observations"].columns:
                    feature_results = results
                    break

        if feature_results is None:
            print(f"[WARNING]  No data found for {feature_name}, skipping...")
            continue

        print(f" Creating plot for {feature_name}...")

        # Create 3-panel plot for this feature
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        observations = feature_results["observations"]
        analysis = feature_results["analysis"]
        feature_data = observations[feature_name].dropna()

        if len(feature_data) == 0:
            print(f"[WARNING]  No valid data for {feature_name}, skipping...")
            continue

        # Panel 1: Price with regime overlay
        ax1 = axes[0]
        ax1.plot(
            baseline_data.index,
            baseline_data["close"],
            linewidth=1.5,
            color="blue",
            alpha=0.8,
        )

        # Add regime coloring if available
        if "regime_name" in analysis.columns:
            regime_colors = {
                "Bull": "green",
                "Bear": "red",
                "Sideways": "orange",
                "Euphoric": "purple",
                "Crisis": "black",
                "Strong Bull": "darkgreen",
                "Weak Bull": "lightgreen",
                "Strong Bear": "darkred",
                "Weak Bear": "lightcoral",
                "Strong Crisis": "darkred",
                "Weak Crisis": "pink",
            }

            for regime in analysis["regime_name"].unique():
                mask = analysis["regime_name"] == regime
                regime_periods = analysis.index[mask]
                if len(regime_periods) > 0:
                    color = regime_colors.get(regime, "gray")
                    for period in regime_periods:
                        if period in baseline_data.index:
                            ax1.axvline(x=period, color=color, alpha=0.3, linewidth=0.8)

        ax1.set_title(
            f"{ticker} Price - {feature_descriptions[feature_name]}",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_ylabel("Price ($)")
        ax1.grid(True, alpha=0.3)

        # Panel 2: Feature values over time with statistical bands
        ax2 = axes[1]
        ax2.plot(
            feature_data.index,
            feature_data.values,
            linewidth=2,
            color="red",
            alpha=0.8,
            label=feature_name,
        )

        # Add statistical bands
        mean_val = feature_data.mean()
        std_val = feature_data.std()
        ax2.axhline(
            y=mean_val,
            color="blue",
            linestyle="-",
            alpha=0.6,
            label=f"Mean ({mean_val:.3f})",
        )
        ax2.axhline(
            y=mean_val + std_val,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"±1 Std",
        )
        ax2.axhline(y=mean_val - std_val, color="gray", linestyle="--", alpha=0.5)
        ax2.axhline(
            y=mean_val + 2 * std_val,
            color="gray",
            linestyle=":",
            alpha=0.3,
            label=f"±2 Std",
        )
        ax2.axhline(y=mean_val - 2 * std_val, color="gray", linestyle=":", alpha=0.3)

        ax2.set_title(
            f'{feature_name.replace("_", " ").title()} Values Over Time',
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_ylabel("Feature Value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: Feature correlation with returns
        ax3 = axes[2]
        if "log_return" in observations.columns:
            log_returns = observations["log_return"].dropna()
            # Align feature and returns data
            common_index = feature_data.index.intersection(log_returns.index)
            if len(common_index) > 10:
                aligned_feature = feature_data.reindex(common_index)
                aligned_returns = log_returns.reindex(common_index)

                # Scatter plot with correlation
                ax3.scatter(aligned_feature, aligned_returns, alpha=0.6, s=20)

                # Calculate and display correlation
                correlation = aligned_feature.corr(aligned_returns)
                ax3.set_title(
                    f'{feature_name.replace("_", " ").title()} vs Log Returns (Correlation: {correlation:.3f})',
                    fontsize=12,
                    fontweight="bold",
                )
                ax3.set_xlabel("Feature Value")
                ax3.set_ylabel("Log Return")
                ax3.grid(True, alpha=0.3)

                # Add trend line if correlation is significant
                if abs(correlation) > 0.1:
                    z = np.polyfit(aligned_feature, aligned_returns, 1)
                    p = np.poly1d(z)
                    ax3.plot(
                        aligned_feature,
                        p(aligned_feature),
                        "r--",
                        alpha=0.8,
                        linewidth=2,
                    )
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "Insufficient data for correlation analysis",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                    fontsize=12,
                )
                ax3.set_title(
                    f'{feature_name.replace("_", " ").title()} Correlation Analysis',
                    fontsize=12,
                    fontweight="bold",
                )
        else:
            ax3.text(
                0.5,
                0.5,
                "Log returns not available for correlation",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=12,
            )
            ax3.set_title(
                f'{feature_name.replace("_", " ").title()} Analysis',
                fontsize=12,
                fontweight="bold",
            )

        plt.tight_layout()

        # Save individual plot
        plot_filename = os.path.join(
            output_dir, f"feature_{feature_name}_{ticker}_{timestamp}.png"
        )
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close()

        individual_plot_files.append(plot_filename)
        print(f" Individual plot saved: {plot_filename}")

    return individual_plot_files


def main():
    """Main execution function demonstrating enhanced features through pipeline architecture."""

    print_section_header("Enhanced Features for Regime Detection", "=", 80)
    print(
        """
This example demonstrates enhanced regime-relevant features using the proper
pipeline architecture. We'll compare baseline log_return regime detection
with enhanced features that capture specific regime characteristics.

Pipeline Architecture: data → observations → model → analysis
Enhanced Features: momentum_strength, trend_persistence, volatility_context, directional_consistency
    """
    )

    # Configuration
    ticker = "SPY"
    start_date = "2022-01-01"  # Extended timeline for feature warmup
    end_date = "2024-01-01"
    n_states = 3
    individual_results = {}

    print(f" Analysis Parameters:")
    print(f"   Ticker: {ticker}")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   States: {n_states}")

    # =================================================================
    # 1. BASELINE PIPELINE: Traditional log_return approach
    # =================================================================
    print_section_header("1. Baseline Pipeline (Log Returns Only)")

    try:
        baseline_pipeline = hr.create_financial_pipeline(
            ticker=ticker, n_states=n_states, start_date=start_date, end_date=end_date
        )

        print("🔄 Executing baseline pipeline...")
        baseline_result = baseline_pipeline.update()
        baseline_results = analyze_pipeline_results(baseline_pipeline, "Baseline")
        individual_results["baseline"] = analyze_pipeline_results(
            baseline_pipeline, "Baseline Feature"
        )

    except Exception as e:
        print(f" Baseline pipeline failed: {e}")
        baseline_results = None

    # =================================================================
    # 2. INDIVIDUAL ENHANCED FEATURES: Test each feature separately
    # =================================================================
    print_section_header("2. Individual Enhanced Features Analysis")

    feature_configs = {
        "momentum": FinancialObservationConfig(generators=["momentum_strength"]),
        "persistence": FinancialObservationConfig(generators=["trend_persistence"]),
        "volatility": FinancialObservationConfig(generators=["volatility_context"]),
        "consistency": FinancialObservationConfig(
            generators=["directional_consistency"]
        ),
    }

    for feature_name, obs_config in feature_configs.items():
        print(f"\n🔬 Testing {feature_name} feature...")

        try:
            feature_pipeline = hr.create_financial_pipeline(
                ticker=ticker,
                n_states=n_states,
                start_date=start_date,
                end_date=end_date,
                observations_config=obs_config,
            )

            # Update observed_signal in model config to match feature
            feature_signal = obs_config.generators[0]
            # Create new model config with updated observed_signal

            model_config_params = {
                "n_states": n_states,
                "observed_signal": feature_signal,
            }
            new_model_config = HMMConfig.create_balanced().copy(**model_config_params)
            feature_pipeline.model.config = new_model_config

            feature_result = feature_pipeline.update()
            individual_results[feature_name] = analyze_pipeline_results(
                feature_pipeline, f"{feature_name.title()} Feature"
            )

        except Exception as e:
            print(f" {feature_name} feature failed: {e}")
            individual_results[feature_name] = None

    # We'll generate individual plots after we have combined results
    individual_plot_files = []

    # =================================================================
    # 3. COMBINED FEATURES: Multiple features together
    # =================================================================
    print_section_header("3. Combined Features Analysis")

    combined_configs = {
        "momentum_persistence": FinancialObservationConfig(
            generators=["log_return", "momentum_strength", "trend_persistence"]
        ),
        "full_enhanced": FinancialObservationConfig(
            generators=[
                "momentum_strength",
                "trend_persistence",
                "volatility_context",
                "directional_consistency",
            ]
        ),
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
                observations_config=obs_config,
            )

            combined_result = combined_pipeline.update()
            combined_results[config_name] = analyze_pipeline_results(
                combined_pipeline, f"{config_name.replace('_', ' ').title()}"
            )

        except Exception as e:
            print(f" {config_name} configuration failed: {e}")
            combined_results[config_name] = None

    # =================================================================
    # 4. COMPREHENSIVE COMPARISON
    # =================================================================
    print_section_header("4. Comprehensive Results Comparison")

    # Combine all results for comparison
    all_results = {"baseline": baseline_results}
    all_results.update(individual_results)
    all_results.update(combined_results)

    # Generate individual feature plots for detailed analysis
    print_section_header("4.1. Individual Feature Visualization")
    individual_plot_files = create_individual_feature_plots(all_results, ticker)

    # Feature statistics comparison
    compare_feature_statistics(all_results)

    # Bull market detection comparison
    print_section_header("Bull Market Detection Comparison")
    print(f"\n {ticker} Bull Market Detection Analysis:")
    print("(Higher percentages = better bull market identification)")
    print()

    for name, results in all_results.items():
        if results is None:
            continue

        analysis = results["analysis"]
        if "regime_name" in analysis.columns:
            regime_counts = analysis["regime_name"].value_counts()
            total_days = len(analysis)

            # Calculate bull-type percentage
            bull_types = ["Bull", "Strong Bull", "Weak Bull", "Euphoric"]
            bull_days = sum(
                count
                for regime, count in regime_counts.items()
                if any(bull_type in regime for bull_type in bull_types)
            )
            bull_percentage = bull_days / total_days * 100

            print(
                f"  {name:15s}: {bull_percentage:5.1f}% bull-type regimes ({bull_days}/{total_days} days)"
            )

    # =================================================================
    # 5. VISUALIZATION AND REPORTING
    # =================================================================
    print_section_header("5. Visualization and Reporting")

    # Create comprehensive visualization
    plot_filename = create_enhanced_features_visualization(all_results, ticker)

    # Individual plots were already created in Section 2.1

    # Generate summary report
    print_section_header("Summary and Key Insights")
    print(
        f"""
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

 Performance Improvements:
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
   • Comprehensive visualization: {plot_filename if plot_filename else 'N/A'}
   • Individual feature plots: {len(individual_plot_files)} files generated
     {chr(10).join([f'     - {os.path.basename(f)}' for f in individual_plot_files]) if individual_plot_files else '     - None generated'}
   • Analysis results saved in pipeline components
    """
    )

    print_section_header("Enhanced Features Example Complete", "=", 80)
    print(" All pipeline configurations tested successfully!")
    print("📁 Check the output directory for generated visualizations and reports.")


if __name__ == "__main__":
    main()
