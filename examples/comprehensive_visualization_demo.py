#!/usr/bin/env python3
"""
Comprehensive Visualization Demo for Hidden Regime Package

This example demonstrates all visualization capabilities across the three main components:
1. DataLoader - Market data visualizations (price, returns, distributions, volume)
2. HiddenMarkovModel - Regime analysis visualizations (states, transitions, probabilities)
3. StateStandardizer - Regime characteristics and validation visualizations

Run this script to see professional-quality financial visualizations with consistent styling.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hidden_regime.data.loader import DataLoader
from hidden_regime.models.base_hmm import HiddenMarkovModel
from hidden_regime.models.state_standardizer import StateStandardizer
from hidden_regime.config.settings import DataConfig
from hidden_regime.models.config import HMMConfig


def main():
    """Run comprehensive visualization demonstration."""
    print("🎨 Hidden Regime - Comprehensive Visualization Demo")
    print("=" * 60)

    # Check if required packages are available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        print("✅ Matplotlib and seaborn available")
    except ImportError:
        print("❌ Visualization packages not available!")
        print("Install with: pip install matplotlib seaborn")
        return

    # Create output directory for plots
    output_dir = "visualization_outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Saving plots to: {output_dir}/")
    print()

    # =========================================================================
    # PART 1: DataLoader Visualizations
    # =========================================================================
    print("📊 PART 1: DataLoader Market Data Visualizations")
    print("-" * 50)

    try:
        # Initialize DataLoader
        config = DataConfig(cache_enabled=True, include_volume=True)
        loader = DataLoader(config=config)

        # Load sample data (SPY - S&P 500 ETF)
        print("Loading SPY data for the last 2 years...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years of data

        data = loader.load_stock_data(
            ticker="SPY", start_date=start_date, end_date=end_date
        )

        print(f"✅ Loaded {len(data)} observations")
        print(
            f"   Date range: {data['date'].min().date()} to {data['date'].max().date()}"
        )
        print()

        # Demonstration 1.1: Complete market data visualization
        print("1.1 Creating comprehensive market data visualization...")
        fig = loader.plot(data, plot_type="all", figsize=(16, 12))
        plt.savefig(
            f"{output_dir}/01_dataloader_comprehensive.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()
        print("✅ Saved: 01_dataloader_comprehensive.png")

        # Demonstration 1.2: Individual plot types
        plot_types = ["price", "returns", "distribution", "volume"]
        for i, plot_type in enumerate(plot_types, 2):
            print(f"1.{i} Creating {plot_type} visualization...")
            try:
                fig = loader.plot(data, plot_type=plot_type, figsize=(12, 6))
                plt.savefig(
                    f"{output_dir}/01_{i}_dataloader_{plot_type}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.show()
                print(f"✅ Saved: 01_{i}_dataloader_{plot_type}.png")
            except Exception as e:
                print(f"⚠️ {plot_type} plot skipped: {e}")

        print()

    except Exception as e:
        print(f"❌ DataLoader visualization failed: {e}")
        print("Continuing with synthetic data...")

        # Create synthetic data as fallback
        n_days = 500
        dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
        np.random.seed(42)

        # Generate synthetic price series with regime-like behavior
        returns = []
        current_regime = 0  # 0=bear, 1=sideways, 2=bull

        for i in range(n_days):
            # Occasional regime switches
            if np.random.random() < 0.02:  # 2% chance of regime switch
                current_regime = np.random.choice([0, 1, 2])

            if current_regime == 0:  # Bear market
                ret = np.random.normal(-0.002, 0.025)
            elif current_regime == 1:  # Sideways
                ret = np.random.normal(0.0001, 0.015)
            else:  # Bull market
                ret = np.random.normal(0.004, 0.020)

            returns.append(ret)

        # Convert to price series
        log_returns = np.array(returns)
        prices = 100 * np.exp(np.cumsum(log_returns))  # Start at $100
        volumes = np.random.lognormal(15, 0.5, n_days)  # Synthetic volume

        data = pd.DataFrame(
            {
                "date": dates,
                "price": prices,
                "log_return": log_returns,
                "volume": volumes,
            }
        )

        print("📈 Created synthetic market data")
        loader = DataLoader()  # Create loader for plotting

        print("1.1 Creating comprehensive synthetic data visualization...")
        fig = loader.plot(data, plot_type="all", figsize=(16, 12))
        plt.savefig(
            f"{output_dir}/01_synthetic_comprehensive.png", dpi=150, bbox_inches="tight"
        )
        plt.show()
        print("✅ Saved: 01_synthetic_comprehensive.png")

    # =========================================================================
    # PART 2: HiddenMarkovModel Visualizations
    # =========================================================================
    print("🔮 PART 2: Hidden Markov Model Regime Analysis Visualizations")
    print("-" * 60)

    # Use the data we have (either real or synthetic)
    returns = data["log_return"].values
    dates = data["date"].values

    # Train HMM model
    print("Training 3-state HMM model...")
    hmm_config = HMMConfig(
        n_states=3,
        max_iterations=100,
        initialization_method="kmeans",
        early_stopping=True,
        tolerance=1e-6,
    )

    hmm = HiddenMarkovModel(config=hmm_config)
    hmm.fit(returns, verbose=True)

    print(f"✅ Model trained with log-likelihood: {hmm.score(returns):.2f}")
    print()

    # Demonstration 2.1: Complete HMM analysis visualization
    print("2.1 Creating comprehensive HMM analysis visualization...")
    fig = hmm.plot(returns, dates=dates, plot_type="all", figsize=(18, 14))
    plt.savefig(f"{output_dir}/02_hmm_comprehensive.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: 02_hmm_comprehensive.png")

    # Demonstration 2.2: Individual HMM plot types
    hmm_plot_types = [
        "regimes",
        "probabilities",
        "transitions",
        "statistics",
        "convergence",
        "duration",
    ]
    for i, plot_type in enumerate(hmm_plot_types, 2):
        print(f"2.{i} Creating HMM {plot_type} visualization...")
        try:
            fig = hmm.plot(returns, dates=dates, plot_type=plot_type, figsize=(12, 8))
            plt.savefig(
                f"{output_dir}/02_{i}_hmm_{plot_type}.png", dpi=150, bbox_inches="tight"
            )
            plt.show()
            print(f"✅ Saved: 02_{i}_hmm_{plot_type}.png")
        except Exception as e:
            print(f"⚠️ HMM {plot_type} plot failed: {e}")

    print()

    # =========================================================================
    # PART 3: StateStandardizer Visualizations
    # =========================================================================
    print("🎯 PART 3: State Standardizer Regime Characteristics Visualizations")
    print("-" * 70)

    # Initialize StateStandardizer
    standardizer = StateStandardizer(regime_type="3_state")

    # Get model parameters for visualization
    emission_params = hmm.emission_params_
    print("Standardizing regime states...")
    state_mapping = standardizer.standardize_states(emission_params)

    print("📋 Detected regime mapping:")
    for state_idx, regime_name in state_mapping.items():
        mean_ret = emission_params[state_idx, 0]
        volatility = emission_params[state_idx, 1]
        print(
            f"   State {state_idx} → {regime_name}: μ={mean_ret:.4f}, σ={volatility:.4f}"
        )
    print()

    # Demonstration 3.1: Complete StateStandardizer analysis
    print("3.1 Creating comprehensive state standardization analysis...")
    try:
        fig = standardizer.plot(emission_params, plot_type="all", figsize=(16, 12))
        plt.savefig(
            f"{output_dir}/03_standardizer_comprehensive.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()
        print("✅ Saved: 03_standardizer_comprehensive.png")
    except Exception as e:
        print(f"⚠️ StateStandardizer comprehensive plot failed: {e}")

    # Demonstration 3.2: Individual StateStandardizer plot types
    standardizer_plot_types = [
        "characteristics",
        "validation",
        "comparison",
        "economic",
    ]
    for i, plot_type in enumerate(standardizer_plot_types, 2):
        print(f"3.{i} Creating StateStandardizer {plot_type} visualization...")
        try:
            fig = standardizer.plot(
                emission_params, plot_type=plot_type, figsize=(10, 8)
            )
            plt.savefig(
                f"{output_dir}/03_{i}_standardizer_{plot_type}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.show()
            print(f"✅ Saved: 03_{i}_standardizer_{plot_type}.png")
        except Exception as e:
            print(f"⚠️ StateStandardizer {plot_type} plot failed: {e}")

    print()

    # =========================================================================
    # PART 4: Multi-Configuration Comparison
    # =========================================================================
    print("🔄 PART 4: Multi-Configuration Regime Analysis Comparison")
    print("-" * 55)

    # Compare different regime configurations
    regime_configs = ["3_state", "4_state", "5_state"]

    for config_name in regime_configs:
        print(
            f"4.{regime_configs.index(config_name)+1} Training {config_name} model and visualizing..."
        )

        try:
            # Train model with different number of states
            n_states = int(config_name.split("_")[0])
            hmm_config_multi = HMMConfig(n_states=n_states, max_iterations=50)
            hmm_multi = HiddenMarkovModel(config=hmm_config_multi)
            hmm_multi.fit(returns, verbose=False)

            # Standardize with specific configuration
            standardizer_multi = StateStandardizer(regime_type=config_name)

            # Create regime comparison visualization
            fig = hmm_multi.plot(
                returns, dates=dates, plot_type="regimes", figsize=(14, 6)
            )
            plt.suptitle(
                f'{config_name.replace("_", "-").title()} Regime Classification',
                fontsize=14,
                fontweight="bold",
            )
            plt.savefig(
                f"{output_dir}/04_{regime_configs.index(config_name)+1}_{config_name}_regimes.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.show()

            # Create characteristics heatmap
            emission_multi = hmm_multi.emission_params_
            fig = standardizer_multi.plot(
                emission_multi, plot_type="characteristics", figsize=(10, 8)
            )
            plt.savefig(
                f"{output_dir}/04_{regime_configs.index(config_name)+1}_{config_name}_characteristics.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.show()

            print(f"✅ Saved: {config_name} visualizations")

        except Exception as e:
            print(f"⚠️ {config_name} analysis failed: {e}")

    print()

    # =========================================================================
    # PART 5: Style and Color Consistency Demonstration
    # =========================================================================
    print("🎨 PART 5: Style and Color Consistency Demonstration")
    print("-" * 52)

    # Demonstrate consistent regime coloring across all visualizations
    print("5.1 Creating regime color consistency comparison...")

    from hidden_regime.visualization.plotting import (
        get_regime_colors,
        setup_financial_plot_style,
    )

    # Setup consistent styling
    setup_financial_plot_style()

    # Get regime colors
    regime_names = ["Bear", "Sideways", "Bull", "Crisis", "Euphoric"]
    colors = get_regime_colors(regime_names)

    # Create color palette visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Color palette display
    y_pos = np.arange(len(regime_names))
    bars = ax1.barh(
        y_pos, [1] * len(regime_names), color=[colors[name] for name in regime_names]
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(regime_names, fontweight="bold")
    ax1.set_xlabel("Color Consistency")
    ax1.set_title("Regime Color Palette", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Style demonstration with sample data
    x = np.linspace(0, 10, 100)
    for i, (name, color) in enumerate(colors.items()):
        if name in regime_names[:3]:  # Show first 3 for clarity
            y = np.sin(x + i) + i * 0.5
            ax2.plot(x, y, color=color, linewidth=3, label=name, alpha=0.8)

    ax2.set_title("Consistent Styling Across Visualizations", fontweight="bold")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_style_consistency.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: 05_style_consistency.png")

    # =========================================================================
    # Summary and Conclusion
    # =========================================================================
    print()
    print("🎉 VISUALIZATION DEMO COMPLETE!")
    print("=" * 40)
    print(f"📁 All visualizations saved to: {output_dir}/")
    print()
    print("📊 Created visualizations:")
    print(
        "   • DataLoader: Market data analysis (price, returns, distributions, volume)"
    )
    print("   • HiddenMarkovModel: Regime detection and analysis")
    print("   • StateStandardizer: Regime characteristics and validation")
    print("   • Multi-configuration: Comparison across different regime setups")
    print("   • Style consistency: Unified color schemes and formatting")
    print()
    print("✨ Key Features Demonstrated:")
    print("   ✅ Date-aware time series plotting")
    print("   ✅ Consistent regime color coding")
    print("   ✅ Professional financial styling")
    print("   ✅ Comprehensive error handling")
    print("   ✅ Flexible plot types and configurations")
    print("   ✅ High-quality output for publications/presentations")
    print()
    print("🚀 Ready for production use in financial analysis workflows!")


if __name__ == "__main__":
    # Suppress some warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    main()
