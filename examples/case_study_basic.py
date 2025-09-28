#!/usr/bin/env python3
"""
Basic Case Study Example

Demonstrates a simple case study using the case study system.
This example shows how to run a quick analysis on a single stock
with minimal configuration.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hidden_regime.config.case_study import CaseStudyConfig
from examples.case_study import run_case_study_from_config


def main():
    """Run basic case study example."""
    print("📊 Basic Case Study Example")
    print("=" * 40)

    # Create a simple case study configuration
    config = CaseStudyConfig(
        ticker="AAPL",
        start_date="2024-06-01",
        end_date="2024-09-01",
        n_training=60,  # 60 days of training
        n_states=3,     # Simple 3-state model
        frequency="business_days",

        # Simplified settings for basic example
        include_technical_indicators=True,  # Enable comprehensive technical analysis
        create_animations=True,   # Enable animations to see regime evolution
        save_individual_frames=False,
        generate_comprehensive_report=True,

        # Quick analysis
        animation_fps=3,
        color_scheme="colorblind_safe",  # Use colorblind-friendly colors
        output_directory="./output/basic_case_study"
    )

    print(f"Configuration created for {config.ticker}")
    print(f"Analysis period: {config.start_date} to {config.end_date}")
    print(f"Training days: {config.n_training}")
    print(f"Model states: {config.n_states}")

    try:
        # Run the case study
        print(f"\n🚀 Starting basic case study...")
        results = run_case_study_from_config(config)

        # Display results summary
        print(f"\n✅ Case study completed!")
        print(f"Execution time: {results.get('execution_time', 0):.1f} seconds")
        print(f"Output directory: {config.output_directory}")

        # Show final comparison if available
        if ('final_comparison' in results and
            results['final_comparison'] is not None and
            'comparison_summary' in results['final_comparison']):
            summary = results['final_comparison']['comparison_summary']
            if 'strategy_ranking' in summary:
                print(f"\nStrategy Performance:")
                for i, (strategy, sharpe) in enumerate(summary['strategy_ranking'][:3]):
                    print(f"  {i+1}. {strategy}: Sharpe {sharpe:.3f}")
        else:
            print(f"\nNote: No performance comparison available (likely due to data issues)")

        print(f"\n📁 Check the output directory for detailed results:")
        print(f"   {config.output_directory}")

    except Exception as e:
        print(f"\n❌ Case study failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()