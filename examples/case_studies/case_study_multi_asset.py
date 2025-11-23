#!/usr/bin/env python3
"""
Multi-Asset Case Study Example

Demonstrates running case studies across multiple assets for comparative analysis.
This example shows how to analyze regime behavior across different stocks,
sectors, or asset classes.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples.case_study import run_case_study_from_config
from hidden_regime.config.case_study import CaseStudyConfig


def run_multi_asset_case_study(
    tickers: list,
    start_date: str,
    end_date: str,
    base_output_dir: str = "./output/multi_asset_case_study",
) -> Dict[str, Any]:
    """
    Run case studies across multiple assets.

    Args:
        tickers: List of ticker symbols to analyze
        start_date: Analysis start date
        end_date: Analysis end date
        base_output_dir: Base directory for outputs

    Returns:
        Dictionary with results for each ticker
    """
    results = {}
    failed_tickers = []

    print(f" Multi-Asset Case Study")
    print(f"Analyzing {len(tickers)} assets: {', '.join(tickers)}")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 60)

    for i, ticker in enumerate(tickers):
        print(f"\n [{i+1}/{len(tickers)}] Analyzing {ticker}...")

        try:
            # Create configuration for this ticker
            config = CaseStudyConfig(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                n_training=120,  # 4 months training
                n_states=3,  # Keep simple for multi-asset
                frequency="business_days",
                # Optimize for speed across multiple assets
                include_technical_indicators=True,
                indicators_to_compare=["rsi", "macd"],  # Limited set for speed
                create_animations=False,  # Skip animations for speed
                save_individual_frames=False,
                generate_comprehensive_report=True,
                output_directory=f"{base_output_dir}/{ticker}",
            )

            # Run case study for this ticker
            ticker_results = run_case_study_from_config(config)
            results[ticker] = ticker_results

            # Print summary for this ticker
            if "final_comparison" in ticker_results:
                comparison = ticker_results["final_comparison"]
                if (
                    "comparison_summary" in comparison
                    and "strategy_ranking" in comparison["comparison_summary"]
                ):
                    best_strategy = comparison["comparison_summary"][
                        "strategy_ranking"
                    ][0]
                    print(
                        f"    Best strategy: {best_strategy[0]} (Sharpe: {best_strategy[1]:.3f})"
                    )
                else:
                    print(f"    Analysis complete (no comparison available)")
            else:
                print(f"    Analysis complete")

        except Exception as e:
            print(f"    Failed: {e}")
            failed_tickers.append(ticker)
            continue

    return {
        "individual_results": results,
        "failed_tickers": failed_tickers,
        "summary": generate_multi_asset_summary(results),
    }


def generate_multi_asset_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary across all analyzed assets."""
    if not results:
        return {"error": "No successful results to summarize"}

    summary = {
        "total_assets": len(results),
        "successful_analyses": len(results),
        "best_performers": {},
        "worst_performers": {},
        "average_metrics": {},
    }

    # Collect metrics across all assets
    all_metrics = {
        "sharpe_ratios": [],
        "total_returns": [],
        "max_drawdowns": [],
        "win_rates": [],
    }

    asset_performances = {}

    for ticker, result in results.items():
        if "final_comparison" not in result:
            continue

        comparison = result["final_comparison"]
        if "individual_results" not in comparison:
            continue

        # Get HMM strategy performance (if available)
        hmm_results = None
        for strategy_name, strategy_results in comparison["individual_results"].items():
            if "hmm" in strategy_name.lower():
                hmm_results = strategy_results
                break

        if hmm_results:
            asset_performances[ticker] = hmm_results
            all_metrics["sharpe_ratios"].append(hmm_results.get("sharpe_ratio", 0))
            all_metrics["total_returns"].append(hmm_results.get("total_return", 0))
            all_metrics["max_drawdowns"].append(hmm_results.get("max_drawdown", 0))
            all_metrics["win_rates"].append(hmm_results.get("win_rate", 0))

    # Calculate averages
    if all_metrics["sharpe_ratios"]:
        summary["average_metrics"] = {
            "avg_sharpe_ratio": sum(all_metrics["sharpe_ratios"])
            / len(all_metrics["sharpe_ratios"]),
            "avg_total_return": sum(all_metrics["total_returns"])
            / len(all_metrics["total_returns"]),
            "avg_max_drawdown": sum(all_metrics["max_drawdowns"])
            / len(all_metrics["max_drawdowns"]),
            "avg_win_rate": sum(all_metrics["win_rates"])
            / len(all_metrics["win_rates"]),
        }

        # Find best and worst performers
        if asset_performances:
            # Best Sharpe ratio
            best_sharpe_ticker = max(
                asset_performances.keys(),
                key=lambda t: asset_performances[t].get("sharpe_ratio", 0),
            )
            summary["best_performers"]["sharpe_ratio"] = {
                "ticker": best_sharpe_ticker,
                "value": asset_performances[best_sharpe_ticker].get("sharpe_ratio", 0),
            }

            # Best total return
            best_return_ticker = max(
                asset_performances.keys(),
                key=lambda t: asset_performances[t].get("total_return", 0),
            )
            summary["best_performers"]["total_return"] = {
                "ticker": best_return_ticker,
                "value": asset_performances[best_return_ticker].get("total_return", 0),
            }

            # Worst drawdown (smallest absolute value)
            best_drawdown_ticker = max(
                asset_performances.keys(),
                key=lambda t: asset_performances[t].get("max_drawdown", 0),
            )
            summary["best_performers"]["max_drawdown"] = {
                "ticker": best_drawdown_ticker,
                "value": asset_performances[best_drawdown_ticker].get(
                    "max_drawdown", 0
                ),
            }

    return summary


def print_multi_asset_summary(multi_asset_results: Dict[str, Any]):
    """Print comprehensive summary of multi-asset analysis."""
    print(f"\n Multi-Asset Analysis Summary")
    print("=" * 50)

    individual_results = multi_asset_results.get("individual_results", {})
    failed_tickers = multi_asset_results.get("failed_tickers", [])
    summary = multi_asset_results.get("summary", {})

    # Overall statistics
    print(f"Total Assets Analyzed: {len(individual_results)}")
    if failed_tickers:
        print(f"Failed Analyses: {len(failed_tickers)} ({', '.join(failed_tickers)})")

    # Average performance
    if "average_metrics" in summary:
        avg_metrics = summary["average_metrics"]
        print(f"\nAverage HMM Strategy Performance:")
        print(f"  Sharpe Ratio: {avg_metrics.get('avg_sharpe_ratio', 0):.3f}")
        print(f"  Total Return: {avg_metrics.get('avg_total_return', 0):.2%}")
        print(f"  Max Drawdown: {avg_metrics.get('avg_max_drawdown', 0):.2%}")
        print(f"  Win Rate: {avg_metrics.get('avg_win_rate', 0):.2%}")

    # Best performers
    if "best_performers" in summary:
        best = summary["best_performers"]
        print(f"\nBest Performers:")

        if "sharpe_ratio" in best:
            print(
                f"  Highest Sharpe: {best['sharpe_ratio']['ticker']} ({best['sharpe_ratio']['value']:.3f})"
            )

        if "total_return" in best:
            print(
                f"  Highest Return: {best['total_return']['ticker']} ({best['total_return']['value']:.2%})"
            )

        if "max_drawdown" in best:
            print(
                f"  Lowest Drawdown: {best['max_drawdown']['ticker']} ({best['max_drawdown']['value']:.2%})"
            )

    # Individual asset summary
    print(f"\nIndividual Asset Results:")
    for ticker, result in individual_results.items():
        print(f"  📁 {ticker}: {result['config'].output_directory}")


def main():
    """Run multi-asset case study example."""
    # Define asset groups for analysis
    asset_groups = {
        "tech_stocks": ["AAPL", "MSFT", "GOOGL"],
        "financial_stocks": ["JPM", "BAC", "WFC"],
        "etfs": ["SPY", "QQQ", "IWM"],
    }

    # Choose asset group (or define custom list)
    selected_group = "tech_stocks"  # Change this to test different groups
    tickers = asset_groups[selected_group]

    # Define analysis period
    start_date = "2024-03-01"
    end_date = "2024-08-01"

    try:
        # Run multi-asset case study
        results = run_multi_asset_case_study(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            base_output_dir=f"./output/multi_asset_{selected_group}",
        )

        # Print comprehensive summary
        print_multi_asset_summary(results)

        print(f"\n📁 All outputs saved to: ./output/multi_asset_{selected_group}")
        print(f"🎉 Multi-asset case study complete!")

    except Exception as e:
        print(f"\n Multi-asset case study failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
