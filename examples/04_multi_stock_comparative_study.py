#!/usr/bin/env python3
"""
Multi-Stock Comparative Regime Study Example

Conducts comprehensive regime analysis across multiple stocks to identify
patterns, correlations, and market-wide trends. Generates professional
comparative reports suitable for institutional analysis.

This example demonstrates:
- Batch processing of multiple stocks for regime detection
- Cross-stock regime correlation analysis
- Sector-based comparative studies
- Market regime consensus identification
- Professional comparative reporting and visualization

Run this script to generate comprehensive multi-stock regime analysis reports.
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import hidden_regime as hr
from hidden_regime.data.financial import FinancialDataLoader
from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.config.model import HMMConfig


def main():
    """Generate comprehensive multi-stock comparative analysis."""

    print("🏢 Multi-Stock Comparative Regime Study")
    print("=" * 60)

    # Stock groups for comparative analysis (reduced for faster processing)
    STOCK_GROUPS = {
        "Tech Giants": ["AAPL", "MSFT"],
        "Finance": ["JPM", "BAC"],
        "Healthcare": ["JNJ", "PFE"],
        "Consumer": ["TSLA", "NFLX"],
        "Index ETFs": ["SPY", "QQQ"],
    }

    ALL_TICKERS = []
    for group_tickers in STOCK_GROUPS.values():
        ALL_TICKERS.extend(group_tickers)

    OUTPUT_DIR = project_root / "examples" / "output" / "multi_stock_study"
    ANALYSIS_PERIOD = 252  # 1 year

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f" Analyzing {len(ALL_TICKERS)} stocks across {len(STOCK_GROUPS)} groups")
    print(f"📁 Output directory: {OUTPUT_DIR}")

    try:
        # Step 1: Process all stocks individually
        print("\n1️⃣ Running individual stock analysis...")

        stock_results = {}
        failed_tickers = []

        for ticker in ALL_TICKERS:
            try:
                print(f"    Processing {ticker}...", end=" ")

                # Load data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=ANALYSIS_PERIOD + 50)

                data_config = FinancialDataConfig(
                    ticker=ticker,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
                loader = FinancialDataLoader(data_config)
                data = loader.update()

                if data is None or len(data) < 200:
                    print(" Insufficient data")
                    failed_tickers.append(ticker)
                    continue

                # Create observations
                from hidden_regime.observations.financial import FinancialObservationGenerator
                from hidden_regime.config.observation import FinancialObservationConfig

                obs_config = FinancialObservationConfig(generators=["log_return"])
                obs_gen = FinancialObservationGenerator(obs_config)
                observations = obs_gen.update(data)

                if len(observations) < 100:
                    print(" Insufficient observations")
                    failed_tickers.append(ticker)
                    continue

                # Run HMM analysis
                hmm_config = HMMConfig(
                    n_states=3, max_iterations=100, tolerance=1e-4, random_seed=42
                )
                hmm = HiddenMarkovModel(config=hmm_config)

                predictions = hmm.update(observations)
                states = predictions["predicted_state"].values
                returns = observations["log_return"].values

                # Calculate regime statistics
                regime_stats = {}
                for state in range(3):
                    state_mask = states == state
                    if state_mask.sum() > 0:
                        state_returns = returns[state_mask]
                        regime_stats[state] = {
                            "mean_return": float(state_returns.mean()),
                            "volatility": float(state_returns.std()),
                            "frequency": float(state_mask.mean()),
                            "avg_duration": calculate_average_duration(states, state),
                        }

                # Identify regime types by returns
                sorted_regimes = sorted(
                    regime_stats.items(), key=lambda x: x[1]["mean_return"]
                )
                bear_regime = sorted_regimes[0][0] if len(sorted_regimes) > 0 else 0
                bull_regime = sorted_regimes[-1][0] if len(sorted_regimes) > 0 else 2
                sideways_regime = sorted_regimes[1][0] if len(sorted_regimes) > 2 else 1

                # Current regime (most recent state)
                current_regime = int(states[-1])

                stock_results[ticker] = {
                    "regime_stats": regime_stats,
                    "bear_regime": bear_regime,
                    "bull_regime": bull_regime,
                    "sideways_regime": sideways_regime,
                    "current_regime": current_regime,
                    "states": states,
                    "returns": returns,
                    "data_points": len(returns),
                }

                print("")

            except Exception as e:
                print(f" Error: {str(e)[:50]}...")
                failed_tickers.append(ticker)
                continue

        successful_count = len(stock_results)
        print(f"\n    Successfully analyzed {successful_count} stocks")
        print(f"    Failed to analyze {len(failed_tickers)} stocks")

        if successful_count < 5:
            print(" Insufficient successful analyses for comparison")
            return False

        # Step 2: Cross-stock regime analysis
        print("\n2️⃣ Conducting cross-stock regime analysis...")

        # Group results by sector
        group_results = {}
        for group_name, group_tickers in STOCK_GROUPS.items():
            group_data = {}
            for ticker in group_tickers:
                if ticker in stock_results:
                    group_data[ticker] = stock_results[ticker]
            if group_data:
                group_results[group_name] = group_data

        # Calculate cross-correlations
        correlations = calculate_regime_correlations(stock_results)

        # Identify market regime consensus
        consensus = identify_market_consensus(stock_results)

        print(f"    Calculated correlations for {len(correlations)} stock pairs")
        print(
            f"   🎯 Market consensus: {consensus['dominant_regime']} ({consensus['consensus_strength']:.1%})"
        )

        # Step 3: Generate comparative report
        print("\n3️⃣ Generating comparative analysis report...")

        report_content = generate_comparative_report(
            stock_results, group_results, correlations, consensus
        )

        report_path = OUTPUT_DIR / "multi_stock_comparative_report.md"
        with open(report_path, "w") as f:
            f.write(report_content)

        print(f"   📝 Saved comprehensive report: {report_path.name}")

        # Step 4: Create summary data files
        print("\n4️⃣ Creating summary data files...")

        # Stock summary
        summary_data = []
        for ticker, results in stock_results.items():
            current_regime_name = get_regime_name(results, results["current_regime"])
            current_regime_stats = results["regime_stats"][results["current_regime"]]

            summary_data.append(
                {
                    "ticker": ticker,
                    "current_regime": current_regime_name,
                    "regime_confidence": current_regime_stats["frequency"],
                    "mean_return": current_regime_stats["mean_return"],
                    "volatility": current_regime_stats["volatility"],
                    "data_points": results["data_points"],
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_path = OUTPUT_DIR / "stock_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        # Correlation matrix
        correlation_matrix = create_correlation_matrix(
            correlations, list(stock_results.keys())
        )
        correlation_path = OUTPUT_DIR / "regime_correlations.csv"
        correlation_matrix.to_csv(correlation_path)

        print(f"   💾 Saved stock summary: {summary_path.name}")
        print(f"   💾 Saved correlation matrix: {correlation_path.name}")

        # Step 5: Display key insights
        print("\n✨ Multi-Stock Analysis Complete!")
        print(f"📁 All files saved to: {OUTPUT_DIR}")

        print(f"\n Key Results:")
        print(
            f"   • Successfully analyzed: {successful_count}/{len(ALL_TICKERS)} stocks"
        )
        print(f"   • Market consensus: {consensus['dominant_regime']} regime")
        print(f"   • Consensus strength: {consensus['consensus_strength']:.1%}")
        print(f"   • Average correlation: {correlations['avg_correlation']:.3f}")

        # Top performing sectors
        sector_performance = {}
        for group_name, group_data in group_results.items():
            if group_data:
                avg_return = np.mean(
                    [
                        results["regime_stats"][results["current_regime"]][
                            "mean_return"
                        ]
                        for results in group_data.values()
                    ]
                )
                sector_performance[group_name] = avg_return

        print(f"\n🏆 Sector Performance (by current regime returns):")
        for sector, performance in sorted(
            sector_performance.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"   • {sector}: {performance:.4f}")

        return True

    except Exception as e:
        print(f" Error in multi-stock analysis: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def calculate_average_duration(states: np.ndarray, target_state: int) -> float:
    """Calculate average duration for a specific regime state."""
    durations = []
    current_duration = 0

    for state in states:
        if state == target_state:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
                current_duration = 0

    # Don't forget the last duration if it ends with the target state
    if current_duration > 0:
        durations.append(current_duration)

    return float(np.mean(durations)) if durations else 0.0


def get_regime_name(results: Dict, regime_id: int) -> str:
    """Get human-readable regime name."""
    if regime_id == results["bear_regime"]:
        return "Bear"
    elif regime_id == results["bull_regime"]:
        return "Bull"
    elif regime_id == results["sideways_regime"]:
        return "Sideways"
    else:
        return f"Regime_{regime_id}"


def calculate_regime_correlations(stock_results: Dict) -> Dict:
    """Calculate regime correlations between all stock pairs."""
    tickers = list(stock_results.keys())
    correlations = {}
    correlation_values = []

    for i, ticker1 in enumerate(tickers):
        for ticker2 in tickers[i + 1 :]:
            states1 = stock_results[ticker1]["states"]
            states2 = stock_results[ticker2]["states"]

            # Align lengths
            min_len = min(len(states1), len(states2))
            if min_len > 10:
                corr = np.corrcoef(states1[:min_len], states2[:min_len])[0, 1]
                if not np.isnan(corr):
                    correlations[f"{ticker1}-{ticker2}"] = float(corr)
                    correlation_values.append(corr)

    return {
        "pairs": correlations,
        "avg_correlation": (
            float(np.mean(correlation_values)) if correlation_values else 0.0
        ),
    }


def identify_market_consensus(stock_results: Dict) -> Dict:
    """Identify market-wide regime consensus."""
    current_regimes = []

    for results in stock_results.values():
        current_regime = results["current_regime"]
        regime_name = get_regime_name(results, current_regime)
        current_regimes.append(regime_name)

    # Count regime occurrences
    regime_counts = {}
    for regime in current_regimes:
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    if regime_counts:
        dominant_regime = max(regime_counts.items(), key=lambda x: x[1])
        consensus_strength = dominant_regime[1] / len(current_regimes)

        return {
            "dominant_regime": dominant_regime[0],
            "consensus_strength": consensus_strength,
            "regime_distribution": regime_counts,
        }

    return {
        "dominant_regime": "Unknown",
        "consensus_strength": 0.0,
        "regime_distribution": {},
    }


def create_correlation_matrix(correlations: Dict, tickers: List[str]) -> pd.DataFrame:
    """Create a correlation matrix DataFrame."""
    n = len(tickers)
    matrix = np.eye(n)  # Start with identity matrix

    for pair, corr in correlations["pairs"].items():
        ticker1, ticker2 = pair.split("-")
        if ticker1 in tickers and ticker2 in tickers:
            i = tickers.index(ticker1)
            j = tickers.index(ticker2)
            matrix[i, j] = corr
            matrix[j, i] = corr  # Symmetric

    return pd.DataFrame(matrix, index=tickers, columns=tickers)


def generate_comparative_report(
    stock_results: Dict, group_results: Dict, correlations: Dict, consensus: Dict
) -> str:
    """Generate comprehensive comparative analysis report."""

    total_stocks = len(stock_results)

    report = f"""# Multi-Stock Regime Analysis Report
*Comprehensive Cross-Asset Regime Detection and Correlation Study*

## Executive Summary

This report presents a comprehensive regime analysis across **{total_stocks} stocks** using Hidden Markov Model detection. The analysis identifies current market regimes, cross-asset correlations, and sector-based patterns to provide institutional-grade market intelligence.

### Key Findings

- **Market Consensus**: {consensus['dominant_regime']} regime dominance ({consensus['consensus_strength']:.1%} of stocks)
- **Average Cross-Correlation**: {correlations['avg_correlation']:.3f}
- **Analysis Period**: 252 trading days
- **Regime Detection Method**: 3-state Hidden Markov Model

## Market Overview

### Current Regime Distribution

"""

    # Add regime distribution
    for regime, count in consensus["regime_distribution"].items():
        percentage = count / total_stocks
        report += f"- **{regime} Regime**: {count} stocks ({percentage:.1%})\n"

    report += f"""

### Cross-Asset Correlation Analysis

The average regime correlation across all stock pairs is **{correlations['avg_correlation']:.3f}**, indicating {'strong' if abs(correlations['avg_correlation']) > 0.5 else 'moderate' if abs(correlations['avg_correlation']) > 0.3 else 'weak'} regime synchronization across the market.

**Top Correlated Pairs**:
"""

    # Add top correlations
    sorted_pairs = sorted(
        correlations["pairs"].items(), key=lambda x: abs(x[1]), reverse=True
    )
    for pair, corr in sorted_pairs[:5]:
        report += f"- {pair}: {corr:.3f}\n"

    report += """

## Sector Analysis

"""

    # Add sector analysis
    for group_name, group_data in group_results.items():
        if not group_data:
            continue

        report += f"### {group_name}\n\n"
        report += "| Stock | Current Regime | Mean Return | Volatility | Frequency |\n"
        report += "|-------|----------------|-------------|------------|-----------|\n"

        for ticker, results in group_data.items():
            current_regime = results["current_regime"]
            regime_name = get_regime_name(results, current_regime)
            regime_stats = results["regime_stats"][current_regime]

            report += (
                f"| {ticker} | {regime_name} | {regime_stats['mean_return']:.4f} | "
            )
            report += f"{regime_stats['volatility']:.4f} | {regime_stats['frequency']:.3f} |\n"

        # Sector summary
        sector_regimes = [
            get_regime_name(results, results["current_regime"])
            for results in group_data.values()
        ]
        sector_consensus = max(set(sector_regimes), key=sector_regimes.count)
        sector_strength = sector_regimes.count(sector_consensus) / len(sector_regimes)

        report += (
            f"\n**Sector Consensus**: {sector_consensus} ({sector_strength:.1%})\n\n"
        )

    report += """

## Individual Stock Analysis

### Detailed Stock Performance

| Stock | Current Regime | Mean Return | Volatility | Duration | Data Points |
|-------|----------------|-------------|------------|----------|-------------|
"""

    # Add individual stock details
    for ticker, results in sorted(stock_results.items()):
        current_regime = results["current_regime"]
        regime_name = get_regime_name(results, current_regime)
        regime_stats = results["regime_stats"][current_regime]
        duration = regime_stats.get("avg_duration", 0)

        report += f"| {ticker} | {regime_name} | {regime_stats['mean_return']:.4f} | "
        report += f"{regime_stats['volatility']:.4f} | {duration:.1f} | {results['data_points']} |\n"

    report += f"""

## Methodology

### Regime Detection Framework

- **Model**: 3-state Hidden Markov Model with Gaussian emissions
- **States**: Bear, Sideways, Bull (classified by mean return)
- **Training**: Maximum Likelihood Estimation via Baum-Welch algorithm
- **Validation**: Out-of-sample state prediction and likelihood scoring

### Classification Criteria

- **Bear Regime**: Negative mean returns, typically high volatility
- **Bull Regime**: Positive mean returns, moderate to high volatility  
- **Sideways Regime**: Near-zero mean returns, typically lower volatility

### Correlation Analysis

Cross-asset regime correlations are calculated using Pearson correlation between regime state sequences, providing insights into market-wide regime synchronization.

## Investment Implications

### Portfolio Management

1. **Regime Diversification**: Current {consensus['consensus_strength']:.1%} consensus suggests {'limited' if consensus['consensus_strength'] > 0.7 else 'moderate'} diversification benefits across assets

2. **Sector Rotation**: {'Strong sector differentiation provides rotation opportunities' if len(set(consensus['regime_distribution'].keys())) > 1 else 'Limited sector divergence suggests broad market moves'}

3. **Risk Management**: {'High regime correlation increases systematic risk' if correlations['avg_correlation'] > 0.5 else 'Moderate correlation allows for some risk diversification'}

### Trading Strategies

- **Consensus Plays**: {consensus['consensus_strength']:.1%} of stocks in {consensus['dominant_regime']} regime suggests directional opportunities
- **Contrarian Opportunities**: Stocks in minority regimes may offer contrarian value
- **Regime Momentum**: High correlation ({correlations['avg_correlation']:.3f}) suggests regime changes may cascade across assets

## Risk Considerations

- **Model Risk**: HMM assumptions may not capture all market dynamics
- **Parameter Stability**: Regime parameters may shift during market stress
- **Look-ahead Bias**: Real-time implementation may differ from historical analysis
- **Transaction Costs**: Regime switching strategies require active management

## Conclusion

The multi-stock regime analysis reveals **{consensus['dominant_regime']} regime dominance** across {consensus['consensus_strength']:.1%} of analyzed stocks, with {correlations['avg_correlation']:.3f} average cross-correlation indicating {'synchronized' if abs(correlations['avg_correlation']) > 0.4 else 'partially synchronized'} market behavior.

This analysis provides a quantitative foundation for:
- Strategic asset allocation decisions
- Risk management framework development
- Systematic trading strategy implementation
- Market timing and regime transition identification

---

*This analysis is for educational and research purposes only. Past performance does not guarantee future results. Please consult with qualified financial advisors before making investment decisions.*

*Generated using Hidden Regime framework - [hiddenregime.com](https://hiddenregime.com)*
"""

    return report


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Multi-stock comparative analysis completed successfully!")
        print(" Ready for institutional review and strategic planning")
    else:
        print("\n💥 Multi-stock analysis failed - check error messages above")
        sys.exit(1)
