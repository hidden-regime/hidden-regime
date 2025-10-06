#!/usr/bin/env python3
"""
Real Market Analysis Example

This example demonstrates regime detection on real market data with error handling
and fallback to sample data if market data is unavailable. Shows how to:

- Load real financial data with robust error handling
- Detect regime changes in actual market conditions
- Generate professional analysis reports
- Create publication-quality visualizations

This example handles data limitations gracefully and provides meaningful
results regardless of data availability.
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
from hidden_regime.config.report import ReportConfig

# Import using current working architecture
from hidden_regime.data.financial import FinancialDataLoader
from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.observations.financial import FinancialObservationGenerator
from hidden_regime.reports.markdown import MarkdownReportGenerator


def create_realistic_sample_data(ticker_name="SAMPLE", n_days=500):
    """Create realistic sample data that mimics real market behavior."""
    print(f"Creating realistic sample data for {ticker_name}...")

    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")

    # Create more realistic regime patterns
    regime_states = []
    current_regime = 1  # Start in sideways
    days_in_regime = 0

    for i in range(n_days):
        regime_states.append(current_regime)
        days_in_regime += 1

        # More realistic regime transitions
        transition_prob = 0.01  # Low base probability
        if days_in_regime > 10:  # After 10 days, increase transition probability
            transition_prob = 0.03
        if days_in_regime > 30:  # After 30 days, higher chance
            transition_prob = 0.08
        if days_in_regime > 60:  # Very long regimes eventually end
            transition_prob = 0.15

        if np.random.random() < transition_prob:
            # Choose next regime based on current regime (realistic transitions)
            if current_regime == 0:  # Bear -> usually Sideways or stay Bear
                current_regime = np.random.choice([0, 1], p=[0.3, 0.7])
            elif current_regime == 1:  # Sideways -> any direction
                current_regime = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
            else:  # Bull -> usually Sideways or stay Bull
                current_regime = np.random.choice([1, 2], p=[0.7, 0.3])

            days_in_regime = 0

    # Generate prices with regime-specific characteristics
    prices = [100.0]  # Starting price
    volumes = []

    for i in range(1, n_days):
        regime = regime_states[i]

        if regime == 0:  # Bear market
            daily_return = np.random.normal(-0.0008, 0.028)  # Negative drift, high vol
            volume_multiplier = np.random.uniform(1.2, 2.0)  # Higher volume in fear
        elif regime == 1:  # Sideways market
            daily_return = np.random.normal(0.0002, 0.012)  # Small drift, low vol
            volume_multiplier = np.random.uniform(0.8, 1.2)  # Normal volume
        else:  # Bull market
            daily_return = np.random.normal(
                0.0012, 0.018
            )  # Positive drift, moderate vol
            volume_multiplier = np.random.uniform(0.9, 1.4)  # Slightly higher volume

        new_price = prices[-1] * (1 + daily_return)
        prices.append(max(new_price, 1.0))  # Prevent negative prices

        # Volume based on price and regime
        base_volume = np.random.randint(800000, 1200000)
        volumes.append(int(base_volume * volume_multiplier))

    # Add volume for first day
    volumes.insert(0, np.random.randint(800000, 1200000))

    # Create OHLCV data with realistic intraday patterns
    data = pd.DataFrame(
        {"open": prices, "close": prices, "volume": volumes}, index=dates
    )

    # Generate realistic high/low based on volatility
    for i, regime in enumerate(regime_states):
        price = data["close"].iloc[i]

        if regime == 0:  # Bear - wider spreads
            spread = np.random.uniform(0.01, 0.04)
        elif regime == 1:  # Sideways - narrow spreads
            spread = np.random.uniform(0.005, 0.02)
        else:  # Bull - moderate spreads
            spread = np.random.uniform(0.008, 0.025)

        high_offset = np.random.uniform(0, spread)
        low_offset = np.random.uniform(0, spread)

        data.loc[data.index[i], "high"] = price * (1 + high_offset)
        data.loc[data.index[i], "low"] = price * (1 - low_offset)

        # Adjust open to be realistic
        if i > 0:
            prev_close = data["close"].iloc[i - 1]
            gap = np.random.normal(0, 0.005)  # Small overnight gaps
            data.loc[data.index[i], "open"] = prev_close * (1 + gap)

    # Ensure OHLC relationships are valid
    data["high"] = data[["high", "close", "open"]].max(axis=1)
    data["low"] = data[["low", "close", "open"]].min(axis=1)

    return data, regime_states


def analyze_stock(ticker, start_date, end_date):
    """Analyze a single stock with comprehensive error handling."""
    print(f"\\n Analyzing {ticker}...")

    try:
        # Try to load real data
        data_loader = FinancialDataLoader(
            FinancialDataConfig(ticker=ticker, start_date=start_date, end_date=end_date)
        )

        raw_data = data_loader.update()

        if raw_data.empty:
            print(f"    No real data available for {ticker}")
            raw_data, true_regimes = create_realistic_sample_data(ticker)
            print(f"   📝 Using realistic sample data ({len(raw_data)} days)")
        else:
            print(f"    Loaded {len(raw_data)} days of real market data")
            true_regimes = None

    except Exception as e:
        print(f"    Data loading failed: {e}")
        raw_data, true_regimes = create_realistic_sample_data(ticker)
        print(f"   📝 Using realistic sample data ({len(raw_data)} days)")

    # Create observations
    observation_config = FinancialObservationConfig(generators=["log_return"])
    observation_component = FinancialObservationGenerator(observation_config)
    observations = observation_component.update(raw_data)

    # Train HMM model
    model_config = HMMConfig(n_states=3, random_seed=42)
    hmm_model = HiddenMarkovModel(model_config)
    model_output = hmm_model.update(observations)

    # Perform analysis
    analysis_config = FinancialAnalysisConfig(
        n_states=3,
        calculate_regime_statistics=True,
        include_duration_analysis=False,  # Disable due to known bug
        include_return_analysis=True,
        include_volatility_analysis=True,
    )

    financial_analysis = FinancialAnalysis(analysis_config)
    analysis_results = financial_analysis.update(model_output, raw_data, model_component=hmm_model)

    # Calculate performance metrics
    regime_counts = model_output["predicted_state"].value_counts().sort_index()
    total_days = len(model_output)

    current_regime = model_output["predicted_state"].iloc[-1]
    confidence = model_output.get("confidence", pd.Series([0.0])).iloc[-1]
    if pd.isna(confidence):
        confidence = 0.0

    # Calculate regime statistics
    regime_stats = {}
    for regime in [0, 1, 2]:
        if regime in regime_counts:
            count = regime_counts[regime]
            percentage = count / total_days * 100
            regime_stats[regime] = {
                "days": count,
                "percentage": percentage,
                "name": ["Bear", "Sideways", "Bull"][regime],
            }

    # Calculate returns by regime
    regime_returns = {}
    for regime in regime_stats.keys():
        mask = model_output["predicted_state"] == regime
        if mask.sum() > 1:  # Need at least 2 days
            regime_data = raw_data[mask]
            returns = regime_data["close"].pct_change().dropna()
            if len(returns) > 0:
                regime_returns[regime] = {
                    "mean_return": returns.mean(),
                    "volatility": returns.std(),
                    "sharpe_ratio": (
                        returns.mean() / returns.std() if returns.std() > 0 else 0
                    ),
                }

    return {
        "ticker": ticker,
        "raw_data": raw_data,
        "model_output": model_output,
        "analysis_results": analysis_results,
        "current_regime": current_regime,
        "confidence": confidence,
        "regime_stats": regime_stats,
        "regime_returns": regime_returns,
        "total_days": total_days,
        "true_regimes": true_regimes,  # For sample data validation
    }


def main():
    """Main analysis function."""
    print(" Real Market Analysis Example")
    print("=" * 50)

    # Configuration
    tickers = ["AAPL", "MSFT", "SPY"]  # Start with major tickers
    start_date = "2023-01-01"
    end_date = "2024-01-01"

    results = {}

    # Analyze each ticker
    for ticker in tickers:
        try:
            result = analyze_stock(ticker, start_date, end_date)
            results[ticker] = result
            print(f"    {ticker} analysis complete")
        except Exception as e:
            print(f"    {ticker} analysis failed: {e}")
            continue

    if not results:
        print(" No successful analyses - exiting")
        return

    # Generate comprehensive report
    print(f"\\n📝 Generating analysis report...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    report_filename = os.path.join(
        output_dir, "reports", f"real_market_analysis_{timestamp}.md"
    )

    with open(report_filename, "w") as f:
        f.write("# Real Market Analysis Report\\n\\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        f.write(f"**Analysis Period**: {start_date} to {end_date}\\n\\n")

        f.write("## Executive Summary\\n\\n")
        f.write(
            f"Analyzed {len(results)} stocks using Hidden Markov Model regime detection.\\n\\n"
        )

        for ticker, result in results.items():
            f.write(f"### {ticker} Analysis\\n\\n")
            f.write(
                f"- **Current Regime**: {['Bear', 'Sideways', 'Bull'][result['current_regime']]}\\n"
            )
            f.write(f"- **Confidence**: {result['confidence']:.1%}\\n")
            f.write(f"- **Data Period**: {result['total_days']} days\\n\\n")

            f.write("#### Regime Distribution\\n\\n")
            for regime, stats in result["regime_stats"].items():
                f.write(
                    f"- **{stats['name']}**: {stats['days']} days ({stats['percentage']:.1f}%)\\n"
                )

            f.write("\\n")

            if result["regime_returns"]:
                f.write("#### Performance by Regime\\n\\n")
                for regime, perf in result["regime_returns"].items():
                    regime_name = ["Bear", "Sideways", "Bull"][regime]
                    f.write(f"- **{regime_name}**:\\n")
                    f.write(f"  - Mean Daily Return: {perf['mean_return']:.3%}\\n")
                    f.write(f"  - Daily Volatility: {perf['volatility']:.3%}\\n")
                    f.write(f"  - Sharpe Ratio: {perf['sharpe_ratio']:.2f}\\n")

                f.write("\\n")

        f.write("## Methodology\\n\\n")
        f.write(
            "This analysis uses a 3-state Hidden Markov Model to detect market regimes:\\n\\n"
        )
        f.write("- **Bear Market**: Declining prices with high volatility\\n")
        f.write("- **Sideways Market**: Range-bound prices with low volatility\\n")
        f.write("- **Bull Market**: Rising prices with moderate volatility\\n\\n")
        f.write(
            "The model analyzes log returns to identify regime patterns and transitions.\\n\\n"
        )

    print(f" Report saved as: {report_filename}")

    # Create visualizations
    print(f"\\n🎨 Creating visualizations...")

    for ticker, result in results.items():
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # Price chart with regime coloring
            raw_data = result["raw_data"]
            model_output = result["model_output"]

            ax1.plot(
                raw_data.index,
                raw_data["close"],
                linewidth=1.5,
                color="black",
                alpha=0.7,
            )

            # Color background by regime
            regime_colors = ["#ff6b6b", "#ffa726", "#66bb6a"]  # Red, Orange, Green

            for i, regime in enumerate([0, 1, 2]):
                mask = model_output["predicted_state"] == regime
                if mask.sum() > 0:
                    regime_dates = model_output.index[mask]
                    for date in regime_dates:
                        ax1.axvspan(
                            date,
                            date + pd.Timedelta(days=1),
                            color=regime_colors[i],
                            alpha=0.2,
                        )

            ax1.set_title(
                f"{ticker} Price Chart with Detected Regimes",
                fontsize=14,
                fontweight="bold",
            )
            ax1.set_ylabel("Price ($)", fontsize=12)
            ax1.grid(True, alpha=0.3)

            # Regime timeline
            regime_colors_solid = ["#d32f2f", "#f57c00", "#388e3c"]
            regime_names = ["Bear", "Sideways", "Bull"]

            for i, regime in enumerate([0, 1, 2]):
                mask = model_output["predicted_state"] == regime
                if mask.sum() > 0:
                    ax2.scatter(
                        model_output.index[mask],
                        [regime] * mask.sum(),
                        c=regime_colors_solid[i],
                        alpha=0.8,
                        s=15,
                        label=f"{regime_names[i]} ({mask.sum()} days)",
                    )

            ax2.set_title(
                "Market Regime Detection Timeline", fontsize=14, fontweight="bold"
            )
            ax2.set_ylabel("Market Regime", fontsize=12)
            ax2.set_xlabel("Date", fontsize=12)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax2.grid(True, alpha=0.3)
            ax2.set_yticks([0, 1, 2])
            ax2.set_yticklabels(["Bear", "Sideways", "Bull"])

            plt.tight_layout()
            os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
            plot_filename = os.path.join(
                output_dir, "plots", f"regime_analysis_{ticker}_{timestamp}.png"
            )
            fig.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"    {ticker} chart saved as: {plot_filename}")

        except Exception as e:
            print(f"    {ticker} visualization failed: {e}")

    # Summary results
    print(f"\\n Analysis Summary:")
    print("=" * 30)

    for ticker, result in results.items():
        current_regime_name = ["Bear", "Sideways", "Bull"][result["current_regime"]]
        print(f"{ticker}:")
        print(f"  Current Regime: {current_regime_name}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Analysis Period: {result['total_days']} days")

        if result["regime_returns"]:
            best_regime = max(
                result["regime_returns"].keys(),
                key=lambda x: result["regime_returns"][x]["mean_return"],
            )
            best_regime_name = ["Bear", "Sideways", "Bull"][best_regime]
            best_return = result["regime_returns"][best_regime]["mean_return"]
            print(
                f"  Best Performing Regime: {best_regime_name} ({best_return:.3%} daily)"
            )

        print()

    print(f"🎉 Real Market Analysis Complete!")
    print(f"📁 Generated files:")
    print(f"   • Report: {report_filename}")
    for ticker in results.keys():
        print(f"   • {ticker} Chart: regime_analysis_{ticker}_{timestamp}.png")

    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\\n" + "=" * 50)
        print(" Real Market Analysis: SUCCESS")
        print("=" * 50)
    except Exception as e:
        print(f"\\n Error running analysis: {e}")
        import traceback

        traceback.print_exc()
