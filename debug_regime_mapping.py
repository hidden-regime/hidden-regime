#!/usr/bin/env python3
"""
Debug script to test regime mapping in visualization functions.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from hidden_regime.financial.analysis import FinancialRegimeAnalysis
from hidden_regime.financial.config import FinancialRegimeConfig


def main():
    """Debug regime mapping integration."""
    print("🔧 Debug: Testing regime mapping integration")
    print("=" * 60)

    # Create a small test configuration
    config = FinancialRegimeConfig.create_quick_analysis(ticker="AAPL", n_regimes=3)

    # Run analysis
    analysis = FinancialRegimeAnalysis(config)
    results = analysis.run_complete_analysis()

    if results.analysis_success:
        print("\n📊 Regime Profiles:")
        for state_id, profile in results.regime_profiles.items():
            print(
                f"   State {state_id}: {profile.regime_type.value} ({profile.annualized_return:.1%})"
            )

        print("\n🎨 Testing visualization functions:")
        from hidden_regime.visualization.plotting import (
            get_regime_colors,
            get_regime_names,
        )

        n_states = len(results.regime_profiles)

        # Test without regime profiles (old way)
        old_names = get_regime_names(n_states)
        old_colors = get_regime_colors(n_states)
        print(f"   Old names: {old_names}")

        # Test with regime profiles (new way)
        new_names = get_regime_names(n_states, regime_profiles=results.regime_profiles)
        new_colors = get_regime_colors(
            n_states, regime_profiles=results.regime_profiles
        )
        print(f"   New names: {new_names}")

        print(f"\n✅ Names match: {old_names != new_names}")
        print(f"✅ Colors differ: {old_colors != new_colors}")

    else:
        print("❌ Analysis failed")


if __name__ == "__main__":
    main()
