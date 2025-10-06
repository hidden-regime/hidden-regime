#!/usr/bin/env python3
"""
Regime Comparison Analysis Example

This example demonstrates how to compare regime detection results across
different assets and identify:

- Synchronous regime changes across markets
- Asset-specific regime patterns
- Correlation between different market segments
- Divergence and convergence patterns

Perfect for portfolio analysis and market timing strategies.
"""

import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hidden_regime.analysis.financial import FinancialAnalysis
from hidden_regime.config.analysis import FinancialAnalysisConfig
from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.config.model import HMMConfig
from hidden_regime.config.observation import FinancialObservationConfig

# Import using current working architecture
from hidden_regime.data.financial import FinancialDataLoader
from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.observations.financial import FinancialObservationGenerator


def create_correlated_sample_data(assets, n_days=300, correlation=0.7):
    """Create sample data with controlled correlation between assets."""
    print(f"Creating correlated sample data for {len(assets)} assets...")

    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")

    # Generate master regime sequence that affects all assets
    master_regime_states = []
    current_regime = 1  # Start in sideways
    days_in_regime = 0

    for i in range(n_days):
        master_regime_states.append(current_regime)
        days_in_regime += 1

        # Regime transitions
        transition_prob = 0.02 if days_in_regime < 20 else 0.05
        if days_in_regime > 40:
            transition_prob = 0.10

        if np.random.random() < transition_prob:
            current_regime = np.random.choice([0, 1, 2], p=[0.25, 0.5, 0.25])
            days_in_regime = 0

    # Create data for each asset
    asset_data = {}

    for asset in assets:
        # Individual regime states (correlated with master but with some independence)
        individual_regime_states = []

        for i, master_regime in enumerate(master_regime_states):
            if np.random.random() < correlation:
                # Follow master regime
                individual_regime_states.append(master_regime)
            else:
                # Independent regime choice
                individual_regime_states.append(np.random.choice([0, 1, 2]))

        # Generate prices based on individual regime states
        prices = [100.0 + np.random.uniform(-10, 10)]  # Random starting price

        for i in range(1, n_days):
            regime = individual_regime_states[i]

            # Add some asset-specific characteristics
            if "TECH" in asset or "GROWTH" in asset:
                # Tech stocks: higher volatility, higher growth potential
                vol_multiplier = 1.5
                growth_bias = 1.2
            elif "UTILS" in asset or "DEFENSIVE" in asset:
                # Utilities: lower volatility, stable
                vol_multiplier = 0.6
                growth_bias = 0.8
            else:
                # Broad market
                vol_multiplier = 1.0
                growth_bias = 1.0

            if regime == 0:  # Bear
                daily_return = np.random.normal(
                    -0.001 * growth_bias, 0.025 * vol_multiplier
                )
            elif regime == 1:  # Sideways
                daily_return = np.random.normal(
                    0.0001 * growth_bias, 0.015 * vol_multiplier
                )
            else:  # Bull
                daily_return = np.random.normal(
                    0.0015 * growth_bias, 0.020 * vol_multiplier
                )

            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 1.0))

        # Create OHLCV data
        asset_data[asset] = {
            "data": pd.DataFrame(
                {
                    "open": prices,
                    "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                    "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                    "close": prices,
                    "volume": np.random.randint(1000000, 5000000, n_days),
                },
                index=dates,
            ),
            "true_regimes": individual_regime_states,
        }

        # Ensure OHLC relationships
        data = asset_data[asset]["data"]
        data["high"] = data[["high", "close", "open"]].max(axis=1)
        data["low"] = data[["low", "close", "open"]].min(axis=1)

    return asset_data, master_regime_states


def analyze_asset_regime(asset_name, raw_data):
    """Analyze regime for a single asset."""
    # Create observations
    observation_config = FinancialObservationConfig(generators=["log_return"])
    observation_component = FinancialObservationGenerator(observation_config)
    observations = observation_component.update(raw_data)

    # Train HMM model
    model_config = HMMConfig(n_states=3, random_seed=42)
    hmm_model = HiddenMarkovModel(model_config)
    model_output = hmm_model.update(observations)

    # Basic analysis
    analysis_config = FinancialAnalysisConfig(
        n_states=3,
        calculate_regime_statistics=True,
        include_duration_analysis=False,  # Disable due to known bug
        include_return_analysis=True,
    )

    financial_analysis = FinancialAnalysis(analysis_config)
    analysis_results = financial_analysis.update(model_output, raw_data, model_component=hmm_model)

    return model_output, analysis_results


def calculate_regime_correlation(regime_data):
    """Calculate correlation between regime sequences."""
    # Create DataFrame with all regime sequences
    regime_df = pd.DataFrame(regime_data)

    # Calculate correlation matrix
    correlation_matrix = regime_df.corr()

    return correlation_matrix


def identify_regime_synchronization(regime_data, window=10):
    """Identify periods of regime synchronization."""
    assets = list(regime_data.keys())
    dates = regime_data[assets[0]].index

    synchronization_scores = []

    for i in range(len(dates)):
        # Get regime at this date for all assets
        regimes = [regime_data[asset].iloc[i] for asset in assets]

        # Calculate synchronization score (how many assets in same regime)
        regime_counts = pd.Series(regimes).value_counts()
        max_count = regime_counts.max()
        sync_score = max_count / len(assets)

        synchronization_scores.append(sync_score)

    return pd.Series(synchronization_scores, index=dates)


def main():
    """Main comparison analysis function."""
    print("🔄 Regime Comparison Analysis Example")
    print("=" * 50)

    # Define assets for comparison
    assets = {
        "BROAD_MARKET": "Broad Market Index",
        "TECH_GROWTH": "Technology Growth",
        "DEFENSIVE_UTILS": "Defensive Utilities",
        "INTL_EMERGING": "International Emerging",
    }

    print(f"\\n Analyzing {len(assets)} different market segments...")

    # Load/generate data for all assets
    try:
        # In a real scenario, you'd load different ETFs or indices
        # For demo, we'll create correlated sample data
        asset_data, master_regimes = create_correlated_sample_data(
            list(assets.keys()), n_days=250, correlation=0.6
        )
        print(" Sample data generated with 60% correlation")

    except Exception as e:
        print(f" Data generation failed: {e}")
        return

    # Analyze each asset
    print(f"\\n🔍 Running regime analysis for each asset...")
    regime_results = {}

    for asset_code, asset_name in assets.items():
        print(f"    Analyzing {asset_name}...")
        try:
            raw_data = asset_data[asset_code]["data"]
            model_output, analysis_results = analyze_asset_regime(asset_code, raw_data)

            regime_results[asset_code] = {
                "name": asset_name,
                "data": raw_data,
                "model_output": model_output,
                "analysis": analysis_results,
                "current_regime": model_output["predicted_state"].iloc[-1],
                "confidence": model_output.get("confidence", pd.Series([0.0])).iloc[-1],
            }
            print(f"    {asset_name} complete")

        except Exception as e:
            print(f"    {asset_name} failed: {e}")
            continue

    if len(regime_results) < 2:
        print(" Need at least 2 successful analyses for comparison")
        return

    # Extract regime sequences for comparison
    print(f"\\n🔬 Analyzing regime correlations...")

    regime_sequences = {}
    for asset_code, result in regime_results.items():
        regime_sequences[asset_code] = result["model_output"]["predicted_state"]

    # Calculate correlations
    correlation_matrix = calculate_regime_correlation(regime_sequences)

    # Calculate synchronization
    sync_scores = identify_regime_synchronization(regime_sequences)

    # Generate comparative report
    print(f"\\n📝 Generating comparison report...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    report_filename = os.path.join(
        output_dir, "reports", f"regime_comparison_analysis_{timestamp}.md"
    )

    with open(report_filename, "w") as f:
        f.write("# Regime Comparison Analysis Report\\n\\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")

        f.write("## Executive Summary\\n\\n")
        f.write(
            f"Comparative regime analysis across {len(regime_results)} market segments using "
        )
        f.write(
            "Hidden Markov Model detection. Analysis identifies synchronization patterns "
        )
        f.write("and divergences between different market areas.\\n\\n")

        f.write("## Current Market Regimes\\n\\n")
        regime_names = ["Bear Market", "Sideways Market", "Bull Market"]

        for asset_code, result in regime_results.items():
            current_regime = result["current_regime"]
            confidence = (
                result["confidence"] if not pd.isna(result["confidence"]) else 0.0
            )
            f.write(f"- **{result['name']}**: {regime_names[current_regime]} ")
            f.write(f"(Confidence: {confidence:.1%})\\n")

        f.write("\\n## Regime Correlation Matrix\\n\\n")
        f.write(
            "Correlation between regime sequences (1.0 = perfect correlation):\\n\\n"
        )

        # Format correlation matrix
        for i, asset1 in enumerate(correlation_matrix.index):
            f.write(f"**{assets[asset1]}**:\\n")
            for j, asset2 in enumerate(correlation_matrix.columns):
                if i != j:  # Skip self-correlation
                    corr = correlation_matrix.loc[asset1, asset2]
                    f.write(f"- vs {assets[asset2]}: {corr:.3f}\\n")
            f.write("\\n")

        f.write("## Synchronization Analysis\\n\\n")
        avg_sync = sync_scores.mean()
        max_sync = sync_scores.max()
        min_sync = sync_scores.min()

        f.write(f"- **Average Synchronization**: {avg_sync:.1%}\\n")
        f.write(f"- **Maximum Synchronization**: {max_sync:.1%}\\n")
        f.write(f"- **Minimum Synchronization**: {min_sync:.1%}\\n\\n")

        # Find most synchronized periods
        high_sync_periods = sync_scores[sync_scores > 0.8]
        if len(high_sync_periods) > 0:
            f.write(f"**High Synchronization Periods** (>80% agreement):\\n")
            f.write(
                f"- {len(high_sync_periods)} days out of {len(sync_scores)} total\\n"
            )
            f.write(
                f"- {len(high_sync_periods)/len(sync_scores):.1%} of analysis period\\n\\n"
            )

        f.write("## Regime Distribution by Asset\\n\\n")
        for asset_code, result in regime_results.items():
            f.write(f"### {result['name']}\\n\\n")
            regime_counts = (
                result["model_output"]["predicted_state"].value_counts().sort_index()
            )
            total_days = len(result["model_output"])

            for regime, count in regime_counts.items():
                percentage = count / total_days * 100
                f.write(
                    f"- **{regime_names[regime]}**: {count} days ({percentage:.1f}%)\\n"
                )
            f.write("\\n")

        f.write("## Interpretation\\n\\n")
        f.write("### Market Synchronization\\n")
        if avg_sync > 0.7:
            f.write(
                "Markets show **high synchronization**, suggesting strong macro factors "
            )
            f.write("driving regime changes across all segments.\\n\\n")
        elif avg_sync > 0.5:
            f.write(
                "Markets show **moderate synchronization**, with some common drivers "
            )
            f.write("but also segment-specific patterns.\\n\\n")
        else:
            f.write(
                "Markets show **low synchronization**, indicating strong sector-specific "
            )
            f.write("or asset-specific regime drivers.\\n\\n")

        f.write("### Diversification Implications\\n")
        min_corr = correlation_matrix.values[correlation_matrix.values != 1.0].min()
        max_corr = correlation_matrix.values[correlation_matrix.values != 1.0].max()

        if max_corr < 0.5:
            f.write(
                "Low regime correlations suggest **good diversification potential** "
            )
            f.write("across these market segments.\\n\\n")
        elif min_corr > 0.8:
            f.write(
                "High regime correlations suggest **limited diversification benefits** "
            )
            f.write("during regime transitions.\\n\\n")
        else:
            f.write(
                "Mixed regime correlations suggest **selective diversification opportunities** "
            )
            f.write("depending on specific asset combinations.\\n\\n")

    print(f" Report saved as: {report_filename}")

    # Create comparative visualization
    print(f"\\n🎨 Creating comparative visualization...")

    try:
        # Create multi-panel plot
        fig, axes = plt.subplots(
            len(regime_results) + 1, 1, figsize=(16, 4 * (len(regime_results) + 1))
        )

        if len(regime_results) == 1:
            axes = [axes]

        # Plot each asset's regime sequence
        regime_colors = ["#d32f2f", "#f57c00", "#388e3c"]  # Red, Orange, Green
        regime_names = ["Bear", "Sideways", "Bull"]

        for i, (asset_code, result) in enumerate(regime_results.items()):
            ax = axes[i]

            # Price chart background
            ax2 = ax.twinx()
            ax2.plot(
                result["data"].index,
                result["data"]["close"],
                color="gray",
                alpha=0.3,
                linewidth=1,
            )
            ax2.set_ylabel("Price", color="gray")

            # Regime sequence
            model_output = result["model_output"]
            for regime in [0, 1, 2]:
                mask = model_output["predicted_state"] == regime
                if mask.sum() > 0:
                    ax.scatter(
                        model_output.index[mask],
                        [regime] * mask.sum(),
                        c=regime_colors[regime],
                        alpha=0.8,
                        s=8,
                        label=f"{regime_names[regime]}",
                    )

            ax.set_title(f"{result['name']} - Regime Detection", fontweight="bold")
            ax.set_ylabel("Regime")
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(["Bear", "Sideways", "Bull"])
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        # Plot synchronization score
        ax_sync = axes[-1]
        ax_sync.plot(sync_scores.index, sync_scores.values, linewidth=2, color="purple")
        ax_sync.axhline(
            y=0.5, color="orange", linestyle="--", alpha=0.7, label="50% Sync"
        )
        ax_sync.axhline(y=0.8, color="red", linestyle="--", alpha=0.7, label="80% Sync")
        ax_sync.set_title("Market Synchronization Score", fontweight="bold")
        ax_sync.set_ylabel("Synchronization")
        ax_sync.set_xlabel("Date")
        ax_sync.grid(True, alpha=0.3)
        ax_sync.legend()

        plt.tight_layout()
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        plot_filename = os.path.join(
            output_dir, "plots", f"regime_comparison_{timestamp}.png"
        )
        fig.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f" Visualization saved as: {plot_filename}")

    except Exception as e:
        print(f" Visualization failed: {e}")
        plot_filename = None

    # Summary results
    print(f"\\n Comparison Summary:")
    print("=" * 30)

    print(f"Assets Analyzed: {len(regime_results)}")
    print(f"Average Regime Synchronization: {avg_sync:.1%}")

    print(f"\\nCurrent Regime Status:")
    for asset_code, result in regime_results.items():
        regime_name = regime_names[result["current_regime"]]
        confidence = result["confidence"] if not pd.isna(result["confidence"]) else 0.0
        print(f"  {result['name']}: {regime_name} ({confidence:.1%})")

    print(f"\\nStrongest Correlations:")
    # Find highest correlations
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    corr_pairs = correlation_matrix.where(mask).stack().sort_values(ascending=False)

    for (asset1, asset2), corr in corr_pairs.head(3).items():
        print(f"  {assets[asset1]} ↔ {assets[asset2]}: {corr:.3f}")

    print(f"\\n🎉 Regime Comparison Analysis Complete!")
    print(f"📁 Generated files:")
    print(f"   • Report: {report_filename}")
    if plot_filename:
        print(f"   • Visualization: {plot_filename}")

    return {
        "regime_results": regime_results,
        "correlation_matrix": correlation_matrix,
        "synchronization_scores": sync_scores,
        "report_file": report_filename,
        "plot_file": plot_filename,
    }


if __name__ == "__main__":
    try:
        results = main()
        print("\\n" + "=" * 50)
        print(" Regime Comparison Analysis: SUCCESS")
        print("=" * 50)
    except Exception as e:
        print(f"\\n Error running comparison: {e}")
        import traceback

        traceback.print_exc()
