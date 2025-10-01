#!/usr/bin/env python3
"""
Comprehensive Financial Regime Analysis Example

Demonstrates a full-featured financial regime analysis with all advanced options enabled.
This example shows the complete capabilities of the financial-first architecture
including intelligent signal generation, comprehensive simulation, and detailed analysis.
"""

import os
import sys
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hidden_regime.financial.analysis import FinancialRegimeAnalysis
from hidden_regime.financial.config import FinancialRegimeConfig


def main():
    """Run comprehensive financial regime analysis example."""
    print("📊 Comprehensive Financial Regime Analysis")
    print("=" * 60)

    # Create a comprehensive financial analysis configuration
    config = FinancialRegimeConfig.create_comprehensive_analysis(
        ticker="SPY",
        start_date="2024-01-01",
        end_date="2024-06-01",
        n_regimes=4,  # 4-regime model for nuanced analysis
        initial_capital=100000.0,  # $100k for comprehensive simulation
    )

    # Note: Configuration is frozen - can't modify after creation
    # All customization must be done during creation or use defaults
    print(f"Auto-generated output directory: {config.output_directory}")

    print(f"Configuration created for {config.ticker}")
    print(f"Analysis period: {config.start_date} to {config.end_date}")
    print(f"Training days: {config.training_days}")
    print(f"Number of regimes: {config.n_regimes}")
    print(f"Initial capital: ${config.initial_capital:,.2f}")
    print(f"Signal strategies: {config.signal_strategies}")
    print(f"Technical indicators: {config.technical_indicators}")
    print(f"Animations enabled: {config.create_animations}")

    try:
        # Run the comprehensive financial analysis
        print(f"\n🚀 Starting comprehensive financial regime analysis...")
        print(
            "This may take several minutes due to animations and comprehensive analysis..."
        )

        start_time = datetime.now()

        # Create and run analysis
        analysis = FinancialRegimeAnalysis(config)
        results = analysis.run_complete_analysis()

        total_time = (datetime.now() - start_time).total_seconds()

        # Display comprehensive results
        print(f"\n✅ Comprehensive financial regime analysis completed!")
        print(f"Total execution time: {total_time:.1f} seconds")
        print(f"Output directory: {config.output_directory}")

        # Regime characterization results
        if results.analysis_success and results.regime_profiles:
            print(f"\n🎯 Regime Characterization Summary:")
            for regime_id, profile in results.regime_profiles.items():
                print(f"   Regime {regime_id} ({profile.regime_type.value}):")
                print(f"     Annual Return: {profile.annualized_return:.1%}")
                print(f"     Volatility: {profile.annualized_volatility:.1%}")
                print(f"     Win Rate: {profile.win_rate:.1%}")
                print(f"     Confidence: {profile.confidence_score:.2f}")

        # Current regime state
        if results.analysis_success:
            current_regime = results.current_regime_info
            print(f"\n📈 Current Market State:")
            print(f"   Regime Type: {current_regime['regime_type']}")
            print(f"   Confidence: {current_regime['confidence']:.1%}")
            print(f"   Expected Return: {current_regime['expected_return']:.1%}")
            print(f"   Volatility: {current_regime['volatility']:.1%}")

        # Simulation results
        if results.simulation_results is not None:
            sim = results.simulation_results
            print(f"\n💰 Trading Simulation Results:")
            print(f"   Total Return: {sim.total_return_pct:.2f}%")
            print(f"   Annualized Return: {sim.annualized_return:.2f}%")
            print(f"   Sharpe Ratio: {sim.sharpe_ratio:.3f}")
            print(f"   Sortino Ratio: {sim.sortino_ratio:.3f}")
            print(f"   Max Drawdown: {sim.max_drawdown_pct:.2f}%")
            print(f"   Total Trades: {sim.total_trades}")
            print(f"   Win Rate: {sim.win_rate:.1f}%")
            print(f"   Best Strategy: {sim.best_strategy}")

            # Strategy comparison
            if sim.strategy_results:
                print(f"\n📊 Strategy Performance Rankings:")
                sorted_strategies = sorted(
                    sim.strategy_results.items(),
                    key=lambda x: x[1].get("sharpe_ratio", 0),
                    reverse=True,
                )
                for i, (strategy, metrics) in enumerate(sorted_strategies[:5]):
                    sharpe = metrics.get("sharpe_ratio", 0)
                    returns = metrics.get("total_return_pct", 0)
                    print(
                        f"   {i+1}. {strategy.replace('_', ' ').title()}: "
                        f"{returns:.2f}% return, {sharpe:.3f} Sharpe"
                    )

        # Generated files summary
        print(f"\n📁 Generated files in {config.output_directory}:")
        print(f"   📊 Regime analysis and characterization charts")
        print(f"   📈 Performance comparison plots")
        print(f"   💼 Trading simulation results")
        if config.create_animations:
            print(f"   🎬 Regime evolution animations (GIF)")
        if config.save_individual_frames:
            print(f"   🖼️  Individual animation frames")
        print(f"   💾 Trade journals and data exports")
        print(f"   📝 Comprehensive analysis report")

        # Analysis insights
        if results.analysis_success:
            print(f"\n💡 Key Financial Insights:")
            print(f"   • Intelligent regime-based signal generation")
            print(
                f"   • Data-driven regime characterization (not naive state assumptions)"
            )
            print(
                f"   • Single-asset optimized position sizing (100% allocation capability)"
            )
            print(f"   • Zero transaction costs (retail-friendly)")
            print(f"   • Comprehensive risk-adjusted performance metrics")

    except Exception as e:
        print(f"\n❌ Comprehensive financial analysis failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
