#!/usr/bin/env python3
"""
Basic Case Study Example

Demonstrates a simple financial regime analysis using the financial-first architecture.
This example shows how to run a quick analysis on a single stock with the new
FinancialRegimeAnalysis unified entry point.
"""

import os
import sys
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hidden_regime.financial.analysis import FinancialRegimeAnalysis
from hidden_regime.financial.config import FinancialRegimeConfig


def main():
    """Run basic financial regime analysis example."""
    print(" Financial Regime Analysis - Basic Example")
    print("=" * 50)

    # Create a quick financial analysis configuration
    config = FinancialRegimeConfig.create_quick_analysis(
        ticker="AAPL",
        days_back=90,  # 3 months of analysis
        n_regimes=3,  # Simple 3-regime model
        initial_capital=50000.0,  # $50k for simulation
    )

    # Note: Can't modify frozen dataclass, but output_directory is auto-generated
    print(f"Auto-generated output directory: {config.output_directory}")

    print(f"Configuration created for {config.ticker}")
    print(f"Analysis period: {config.start_date} to {config.end_date}")
    print(f"Training days: {config.training_days}")
    print(f"Number of regimes: {config.n_regimes}")
    print(f"Initial capital: ${config.initial_capital:,.2f}")

    try:
        # Run the financial analysis
        print(f"\n Starting financial regime analysis...")
        start_time = datetime.now()

        # Create and run analysis
        analysis = FinancialRegimeAnalysis(config)
        results = analysis.run_complete_analysis()

        execution_time = (datetime.now() - start_time).total_seconds()

        # Display results summary
        print(f"\n Financial regime analysis completed!")
        print(f"Execution time: {execution_time:.1f} seconds")
        print(f"Output directory: {config.output_directory}")

        # Show regime analysis results
        if results.analysis_success:
            current_regime = results.current_regime_info
            print(f"\n🎯 Current Market Regime:")
            print(f"   Regime: {current_regime['regime_type']}")
            print(f"   Confidence: {current_regime['confidence']:.1%}")
            print(f"   Expected Return: {current_regime['expected_return']:.1%}")
            print(f"   Volatility: {current_regime['volatility']:.1%}")

        # Show simulation results if available
        if results.simulation_results is not None:
            sim = results.simulation_results
            print(f"\n💰 Trading Simulation Results:")
            print(f"   Total Return: {sim.total_return_pct:.2f}%")
            print(f"   Sharpe Ratio: {sim.sharpe_ratio:.3f}")
            print(f"   Max Drawdown: {sim.max_drawdown_pct:.2f}%")
            print(f"   Total Trades: {sim.total_trades}")
            print(f"   Best Strategy: {sim.best_strategy}")

        print(f"\n📁 Generated files in {config.output_directory}:")
        print(f"    Regime analysis charts")
        print(f"    Performance comparison plots")
        print(f"   💾 Trade journal and data exports")
        print(f"   📝 Comprehensive analysis report")

    except Exception as e:
        print(f"\n Financial analysis failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
