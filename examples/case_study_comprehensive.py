#!/usr/bin/env python3
"""
Comprehensive Case Study Example

Demonstrates a full-featured case study with all advanced options enabled.
This example shows the complete capabilities of the case study system
including animations, technical indicators, and detailed analysis.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hidden_regime.config.case_study import CaseStudyConfig
from examples.case_study import run_case_study_from_config


def main():
    """Run comprehensive case study example."""
    print("📊 Comprehensive Case Study Example")
    print("=" * 50)

    # Create a comprehensive case study configuration
    config = CaseStudyConfig.create_comprehensive_study(
        ticker="SPY",
        start_date="2024-01-01",
        end_date="2024-06-01",
        n_states=4  # 4-state model for more nuanced analysis
    )

    # Customize for comprehensive analysis
    config.output_directory = "./output/comprehensive_case_study_SPY"
    config.animation_fps = 2  # Slower for better visibility
    config.save_individual_frames = True  # Save frames for inspection

    print(f"Configuration created for {config.ticker}")
    print(f"Analysis period: {config.start_date} to {config.end_date}")
    print(f"Training days: {config.n_training}")
    print(f"Model states: {config.n_states}")
    print(f"Technical indicators: {config.indicators_to_compare}")
    print(f"Animations enabled: {config.create_animations}")

    try:
        # Run the comprehensive case study
        print(f"\n🚀 Starting comprehensive case study...")
        print("This may take several minutes due to animations and indicators...")

        start_time = datetime.now()
        results = run_case_study_from_config(config)
        total_time = (datetime.now() - start_time).total_seconds()

        # Display comprehensive results
        print(f"\n✅ Comprehensive case study completed!")
        print(f"Total execution time: {total_time:.1f} seconds")
        print(f"Output directory: {config.output_directory}")

        # Evolution results summary
        if 'evolution_results' in results:
            evolution_count = len(results['evolution_results'])
            print(f"Evolution periods analyzed: {evolution_count}")

        # Performance evolution summary
        if 'performance_evolution' in results:
            perf_count = len(results['performance_evolution'])
            if perf_count > 0:
                final_perf = results['performance_evolution'][-1]
                print(f"Final strategy performance:")
                print(f"  Total Return: {final_perf.get('total_return', 0):.2%}")
                print(f"  Sharpe Ratio: {final_perf.get('sharpe_ratio', 0):.3f}")
                print(f"  Max Drawdown: {final_perf.get('max_drawdown', 0):.2%}")

        # Strategy comparison results
        if 'final_comparison' in results and 'comparison_summary' in results['final_comparison']:
            summary = results['final_comparison']['comparison_summary']

            if 'strategy_ranking' in summary:
                print(f"\nStrategy Performance Rankings:")
                for i, (strategy, sharpe) in enumerate(summary['strategy_ranking'][:5]):
                    print(f"  {i+1}. {strategy.replace('_', ' ').title()}: Sharpe {sharpe:.3f}")

            # Best performers by metric
            if 'best_total_return' in summary:
                best_return = summary['best_total_return']
                print(f"\nBest Total Return: {best_return['strategy']} ({best_return['value']:.2%})")

            if 'best_sharpe_ratio' in summary:
                best_sharpe = summary['best_sharpe_ratio']
                print(f"Best Sharpe Ratio: {best_sharpe['strategy']} ({best_sharpe['value']:.3f})")

        # Generated files summary
        print(f"\n📁 Generated files in {config.output_directory}:")
        print(f"   📊 Static analysis chart")
        if config.create_animations:
            print(f"   🎬 Regime evolution animation (GIF)")
            print(f"   📈 Performance evolution animation (GIF)")
        if config.save_individual_frames:
            print(f"   🖼️  Individual animation frames")
        print(f"   📝 Comprehensive markdown report")
        print(f"   💾 Raw data files (JSON)")

    except Exception as e:
        print(f"\n❌ Comprehensive case study failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()