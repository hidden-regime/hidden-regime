#!/usr/bin/env python3
"""
RECOMMENDED USAGE - Factory Pattern (VERSION 2.0.0+)

This example demonstrates the RECOMMENDED way to use hidden-regime v2.0.0+
using the factory pattern instead of direct component instantiation.

KEY PRINCIPLE: Use create_financial_pipeline() instead of creating components manually.

This approach:
- Ensures components are created correctly
- Handles all dependencies automatically
- Follows architectural best practices
- Avoids deprecation warnings

For backward compatibility examples (with warnings), see other files.
"""

import os
import sys
import warnings

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hidden_regime as hr


def main():
    """Demonstrate recommended factory pattern usage."""

    print("\n" + "=" * 80)
    print("RECOMMENDED USAGE: Factory Pattern (v2.0.0+)")
    print("=" * 80 + "\n")

    # ==========================================
    # RECOMMENDED: Use factory pattern
    # ==========================================
    print("✅ Creating pipeline using factory pattern...")
    print("   pipeline = hr.create_financial_pipeline('SPY', n_states=3)\n")

    # This is the RECOMMENDED way - no warnings!
    pipeline = hr.create_financial_pipeline(
        ticker="SPY",
        start_date="2020-01-01",
        end_date="2024-01-01",
        n_states=3,
        regime_interpretation="data_driven",
    )

    print("✅ Pipeline created successfully without warnings!\n")

    # ==========================================
    # Run the pipeline
    # ==========================================
    print("Running pipeline analysis...")

    try:
        # Update the pipeline (loads data, trains model, generates analysis)
        result = pipeline.update()

        print(f"\n✅ Analysis complete!")
        print(f"   - Data points: {len(result)}")
        print(f"   - Columns: {list(result.columns)}")

        # Access regime information
        if 'regime_label' in result.columns:
            unique_regimes = result['regime_label'].unique()
            print(f"   - Detected regimes: {list(unique_regimes)}")

        # Generate and save report
        if pipeline.report:
            report = pipeline.generate_report()
            with open('/tmp/recommended_usage_report.md', 'w') as f:
                f.write(report)
            print(f"\n✅ Report saved to /tmp/recommended_usage_report.md")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS:")
    print("=" * 80)
    print("1. ✅ ALWAYS use hr.create_financial_pipeline() for new code")
    print("2. ✅ This avoids deprecation warnings")
    print("3. ✅ Components are created with correct dependencies")
    print("4. ✅ Follows architectural best practices (Principle #1)")
    print("\n" + "=" * 80)
    print("For more details, see: REFACTORING_PLAN.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
