#!/usr/bin/env python3
"""
Financial-First Case Study System

Demonstrates the new financial-first architecture for regime analysis.
This replaces the old pipeline-based approach with intelligent regime characterization,
data-driven signal generation, and optimized single-asset trading simulation.

Key improvements over the old system:
- Regime characterization based on actual financial metrics (not naive state assumptions)
- Intelligent signal generation using regime characteristics
- Single-asset optimized position sizing (100% allocation capability)
- Zero transaction cost defaults (retail-friendly)
- Unified configuration and analysis entry point
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hidden_regime.financial.config import FinancialRegimeConfig
from hidden_regime.financial.analysis import FinancialRegimeAnalysis


def run_financial_case_study_basic() -> Dict[str, Any]:
    """Run a basic financial regime analysis case study."""
    print("\n" + "="*60)
    print("📊 BASIC FINANCIAL REGIME ANALYSIS")
    print("="*60)

    # Quick analysis configuration
    config = FinancialRegimeConfig.create_quick_analysis(
        ticker="AAPL",
        days_back=90,
        n_regimes=3,
        initial_capital=50000.0
    )

    # Note: Config is frozen, output_directory is auto-generated

    print(f"📈 Analyzing {config.ticker} - {config.get_analysis_period_days()} days")
    print(f"💰 Capital: ${config.initial_capital:,.2f}")
    print(f"🎯 Regimes: {config.n_regimes}")

    try:
        start_time = datetime.now()

        analysis = FinancialRegimeAnalysis(config)
        results = analysis.run_complete_analysis()

        execution_time = (datetime.now() - start_time).total_seconds()

        print(f"\n✅ Analysis completed in {execution_time:.1f} seconds")

        # Display key results
        if results.analysis_success:
            current = results.current_regime_info
            print(f"🎯 Current: {current['regime_type']} ({current['confidence']:.1%} confidence)")

        if results.simulation_results is not None:
            sim = results.simulation_results
            print(f"💰 Return: {sim.total_return_pct:.2f}% | Sharpe: {sim.sharpe_ratio:.3f}")
            print(f"🏆 Best: {sim.best_strategy}")

        print(f"📁 Output: {config.output_directory}")
        return {"success": True, "results": results, "execution_time": execution_time}

    except Exception as e:
        print(f"❌ Failed: {e}")
        return {"success": False, "error": str(e)}


def run_financial_case_study_comprehensive() -> Dict[str, Any]:
    """Run a comprehensive financial regime analysis case study."""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE FINANCIAL REGIME ANALYSIS")
    print("="*60)

    # Comprehensive analysis configuration
    config = FinancialRegimeConfig.create_comprehensive_analysis(
        ticker="SPY",
        start_date="2024-01-01",
        end_date="2024-06-01",
        n_regimes=4,
        initial_capital=100000.0
    )

    # Note: Config is frozen, output_directory is auto-generated

    print(f"📈 Analyzing {config.ticker} ({config.start_date} to {config.end_date})")
    print(f"💰 Capital: ${config.initial_capital:,.2f}")
    print(f"🎯 Regimes: {config.n_regimes}")
    print(f"📊 Strategies: {len(config.signal_strategies)}")
    print(f"🔧 Indicators: {len(config.technical_indicators)}")

    try:
        start_time = datetime.now()

        analysis = FinancialRegimeAnalysis(config)
        results = analysis.run_complete_analysis()

        execution_time = (datetime.now() - start_time).total_seconds()

        print(f"\n✅ Analysis completed in {execution_time:.1f} seconds")

        # Display comprehensive results
        if results.analysis_success and results.regime_profiles:
            print(f"\n🎯 Regime Profiles:")
            for regime_id, profile in results.regime_profiles.items():
                print(f"   {regime_id}: {profile.regime_type.value.title()} "
                      f"({profile.annualized_return:.1%} return, "
                      f"{profile.annualized_volatility:.1%} vol)")

        if results.analysis_success:
            current = results.current_regime_info
            print(f"\n📈 Current State: {current['regime_type']} "
                  f"({current['confidence']:.1%} confidence)")

        if results.simulation_results is not None:
            sim = results.simulation_results
            print(f"\n💰 Performance Summary:")
            print(f"   Total Return: {sim.total_return_pct:.2f}%")
            print(f"   Sharpe Ratio: {sim.sharpe_ratio:.3f}")
            print(f"   Max Drawdown: {sim.max_drawdown_pct:.2f}%")
            print(f"   Best Strategy: {sim.best_strategy}")

        print(f"\n📁 Output: {config.output_directory}")
        return {"success": True, "results": results, "execution_time": execution_time}

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def run_financial_case_study_single_asset() -> Dict[str, Any]:
    """Run a single-asset optimized financial regime analysis case study."""
    print("\n" + "="*60)
    print("📊 SINGLE-ASSET OPTIMIZED ANALYSIS")
    print("="*60)

    # Single-asset optimized configuration
    config = FinancialRegimeConfig.create_single_asset_study(
        ticker="NVDA",
        start_date="2024-01-01",
        end_date="2024-09-01",
        initial_capital=100000.0,
        aggressive=False  # Conservative single-asset approach
    )

    # Note: Config is frozen, output_directory is auto-generated

    print(f"📈 Single-asset focus: {config.ticker}")
    print(f"💰 Dedicated capital: ${config.initial_capital:,.2f}")
    print(f"🎯 Max allocation: {config.max_position_pct:.0%}")
    print(f"💸 Transaction costs: {config.transaction_cost_type}")

    try:
        start_time = datetime.now()

        analysis = FinancialRegimeAnalysis(config)
        results = analysis.run_complete_analysis()

        execution_time = (datetime.now() - start_time).total_seconds()

        print(f"\n✅ Analysis completed in {execution_time:.1f} seconds")

        # Display single-asset optimized results
        if results.analysis_success:
            current = results.current_regime_info
            print(f"\n🎯 {config.ticker} Current Regime: {current['regime_type']}")
            print(f"   Confidence: {current['confidence']:.1%}")
            print(f"   Expected Return: {current['expected_return']:.1%}")

        if results.simulation_results is not None:
            sim = results.simulation_results
            print(f"\n💰 Single-Asset Performance:")
            print(f"   Capital Growth: ${sim.initial_capital:,.0f} → ${sim.final_value:,.0f}")
            print(f"   Total Return: {sim.total_return_pct:.2f}%")
            print(f"   Risk-Adjusted: {sim.sharpe_ratio:.3f} Sharpe")
            print(f"   Downside Protection: {sim.max_drawdown_pct:.2f}% max drawdown")

        print(f"\n📁 Output: {config.output_directory}")
        return {"success": True, "results": results, "execution_time": execution_time}

    except Exception as e:
        print(f"❌ Failed: {e}")
        return {"success": False, "error": str(e)}


def main():
    """Run all financial case study examples."""
    print("🚀 FINANCIAL-FIRST REGIME ANALYSIS SYSTEM")
    print("="*80)
    print("Demonstrates intelligent regime characterization and optimized trading")
    print("="*80)

    results = {}

    # Run all case studies
    case_studies = [
        ("basic", run_financial_case_study_basic),
        ("comprehensive", run_financial_case_study_comprehensive),
        ("single_asset", run_financial_case_study_single_asset)
    ]

    for name, runner in case_studies:
        try:
            result = runner()
            results[name] = result
        except Exception as e:
            print(f"\n❌ Case study '{name}' failed: {e}")
            results[name] = {"success": False, "error": str(e)}

    # Summary
    print("\n" + "="*80)
    print("📊 FINANCIAL CASE STUDY SUMMARY")
    print("="*80)

    success_count = sum(1 for r in results.values() if r.get("success", False))
    total_time = sum(r.get("execution_time", 0) for r in results.values() if r.get("success", False))

    print(f"✅ Completed: {success_count}/{len(case_studies)} case studies")
    print(f"⏱️  Total time: {total_time:.1f} seconds")

    for name, result in results.items():
        if result.get("success", False):
            print(f"   ✅ {name.title()}: {result.get('execution_time', 0):.1f}s")
        else:
            print(f"   ❌ {name.title()}: {result.get('error', 'Unknown error')}")

    print(f"\n💡 Key Advantages of Financial-First Architecture:")
    print(f"   • Intelligent regime characterization (not naive state assumptions)")
    print(f"   • Data-driven signal generation based on actual financial metrics")
    print(f"   • Single-asset optimized position sizing (100% allocation)")
    print(f"   • Zero transaction cost defaults (retail-friendly)")
    print(f"   • Unified configuration and analysis workflow")
    print(f"   • Colorblind-safe visualizations")

    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n🎉 Financial case study system completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in financial case study system: {e}")
        import traceback
        traceback.print_exc()