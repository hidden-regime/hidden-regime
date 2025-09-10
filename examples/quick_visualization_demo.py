#!/usr/bin/env python3
"""
Quick Visualization Demo for Hidden Regime Package

A streamlined demonstration of the core visualization capabilities:
- DataLoader: Market data plots
- HMM: Regime detection plots
- StateStandardizer: Regime characteristics plots

Perfect for quick testing and getting started with visualizations.
"""

import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hidden_regime.config.settings import DataConfig
from hidden_regime.data.loader import DataLoader
from hidden_regime.models.base_hmm import HiddenMarkovModel
from hidden_regime.models.config import HMMConfig
from hidden_regime.models.state_standardizer import StateStandardizer


def main():
    """Run quick visualization demonstration."""
    print("🚀 Hidden Regime - Quick Visualization Demo")
    print("=" * 45)

    # Check visualization dependencies
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        print("✅ Visualization libraries available")
    except ImportError:
        print(
            "❌ Missing matplotlib/seaborn. Install with: pip install matplotlib seaborn"
        )
        return

    print()

    # =========================================================================
    # Step 1: Create/Load Sample Data
    # =========================================================================
    print("📈 Step 1: Preparing Sample Data")
    print("-" * 35)

    # Try loading real data first, fallback to synthetic
    try:
        loader = DataLoader(config=DataConfig(include_volume=True))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year

        print("Loading AAPL data...")
        data = loader.load_stock_data("AAPL", start_date, end_date)
        print(f"✅ Loaded {len(data)} days of AAPL data")

    except Exception as e:
        print(f"⚠️ Real data loading failed ({e}), using synthetic data...")

        # Generate synthetic data
        np.random.seed(42)
        n_days = 252  # 1 year of trading days
        dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

        # Create regime-switching returns
        regimes = np.random.choice(
            [0, 1, 2], n_days, p=[0.3, 0.4, 0.3]
        )  # bear, sideways, bull
        returns = []

        for regime in regimes:
            if regime == 0:  # Bear
                ret = np.random.normal(-0.003, 0.025)
            elif regime == 1:  # Sideways
                ret = np.random.normal(0.0005, 0.015)
            else:  # Bull
                ret = np.random.normal(0.006, 0.022)
            returns.append(ret)

        returns = np.array(returns)
        prices = 150 * np.exp(np.cumsum(returns))  # Start at $150

        data = pd.DataFrame(
            {
                "date": dates,
                "price": prices,
                "log_return": returns,
                "volume": np.random.lognormal(16, 0.3, n_days),
            }
        )

        loader = DataLoader()  # For plotting
        print(f"✅ Generated {len(data)} days of synthetic data")

    print()

    # =========================================================================
    # Step 2: DataLoader Visualizations
    # =========================================================================
    print("📊 Step 2: DataLoader Market Visualizations")
    print("-" * 42)

    print("Creating comprehensive market data visualization...")
    fig = loader.plot(data, plot_type="all", figsize=(15, 10))
    plt.suptitle("Market Data Analysis", fontsize=16, fontweight="bold", y=0.98)
    plt.show()
    print("✅ DataLoader visualization complete")
    print()

    # =========================================================================
    # Step 3: HMM Regime Analysis
    # =========================================================================
    print("🔮 Step 3: HMM Regime Detection Visualization")
    print("-" * 46)

    returns = data["log_return"].values
    dates = data["date"].values

    print("Training 3-state HMM...")
    config = HMMConfig(n_states=3, max_iterations=100, early_stopping=True)
    hmm = HiddenMarkovModel(config=config)
    hmm.fit(returns, verbose=False)

    log_likelihood = hmm.score(returns)
    print(f"✅ Model trained (log-likelihood: {log_likelihood:.2f})")

    print("Creating HMM regime analysis visualization...")
    fig = hmm.plot(returns, dates=dates, plot_type="all", figsize=(16, 12))
    plt.suptitle(
        "Hidden Markov Model - Regime Analysis", fontsize=16, fontweight="bold", y=0.96
    )
    plt.show()
    print("✅ HMM visualization complete")
    print()

    # =========================================================================
    # Step 4: State Standardization Analysis
    # =========================================================================
    print("🎯 Step 4: State Standardization Visualization")
    print("-" * 44)

    standardizer = StateStandardizer(regime_type="3_state")
    emission_params = hmm.emission_params_

    # Show regime mapping
    state_mapping = standardizer.standardize_states(emission_params)
    print("Detected regime characteristics:")
    for state_idx, regime_name in state_mapping.items():
        mean_ret = emission_params[state_idx, 0]
        vol = emission_params[state_idx, 1]
        ann_ret = mean_ret * 252 * 100
        ann_vol = vol * np.sqrt(252) * 100
        print(f"  {regime_name}: {ann_ret:+.1f}% return, {ann_vol:.1f}% volatility")

    print("Creating state standardization visualization...")
    fig = standardizer.plot(emission_params, plot_type="all", figsize=(14, 10))
    plt.suptitle(
        "Regime State Analysis & Validation", fontsize=16, fontweight="bold", y=0.95
    )
    plt.show()
    print("✅ StateStandardizer visualization complete")
    print()

    # =========================================================================
    # Step 5: Individual Plot Type Examples
    # =========================================================================
    print("🎨 Step 5: Individual Plot Type Examples")
    print("-" * 40)

    # DataLoader individual plots
    print("5.1 DataLoader - Returns Distribution Analysis")
    fig = loader.plot(data, plot_type="distribution", figsize=(10, 6))
    plt.show()

    # HMM individual plots
    print("5.2 HMM - Regime Time Series")
    fig = hmm.plot(returns, dates=dates, plot_type="regimes", figsize=(12, 6))
    plt.show()

    # StateStandardizer individual plots
    print("5.3 StateStandardizer - Regime Characteristics Heatmap")
    fig = standardizer.plot(
        emission_params, plot_type="characteristics", figsize=(10, 6)
    )
    plt.show()

    print("✅ Individual visualization examples complete")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("🎉 QUICK DEMO COMPLETE!")
    print("=" * 25)
    print()
    print("📋 What you just saw:")
    print("   • DataLoader: Price evolution, returns, distributions, volume")
    print("   • HiddenMarkovModel: Regime detection, transitions, probabilities")
    print("   • StateStandardizer: Regime characteristics and validation")
    print("   • Individual plot types: Focused analysis capabilities")
    print()
    print("✨ Key Features:")
    print("   ✅ Professional financial styling")
    print("   ✅ Consistent regime color coding")
    print("   ✅ Date-aware time series plots")
    print("   ✅ Comprehensive statistical overlays")
    print("   ✅ Flexible plot type selection")
    print()
    print("🚀 Ready to integrate into your financial analysis workflow!")
    print()
    print("💡 Next steps:")
    print(
        "   • Try with your own data: loader.load_stock_data('YOUR_TICKER', start, end)"
    )
    print(
        "   • Experiment with different regime configurations (3_state, 4_state, 5_state)"
    )
    print("   • Save plots: add save_path='your_filename.png' parameter")
    print("   • Customize: adjust figsize, plot_type parameters")


if __name__ == "__main__":
    # Clean output by suppressing warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    main()
