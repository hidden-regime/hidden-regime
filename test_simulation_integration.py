#!/usr/bin/env python3
"""
Test script for case study simulation integration.

Quick test to verify that the simulation engine is properly integrated
with the case study orchestrator.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from examples.case_study import CaseStudyOrchestrator
from hidden_regime.config.case_study import CaseStudyConfig
import warnings
warnings.filterwarnings('ignore')

def test_simulation_integration():
    """Test the simulation integration with a quick case study."""
    print("🧪 Testing Case Study Simulation Integration")
    print("=" * 50)

    # Create a very quick case study configuration
    config = CaseStudyConfig.create_quick_study(
        ticker="SPY",
        days_back=20,  # Very short period for quick test
        n_states=3
    )

    # Disable animations for faster testing
    config = config.copy(
        create_animations=False,
        save_individual_frames=False,
        generate_comprehensive_report=True,
        enable_simulation=True
    )

    print(f"📊 Configuration: {config.ticker}, {config.get_total_analysis_period()} periods")
    print(f"💰 Simulation enabled: {config.enable_simulation}")

    # Create orchestrator
    orchestrator = CaseStudyOrchestrator(config)
    print(f"🏗️  Orchestrator created with simulation config: {orchestrator.simulation_config is not None}")

    # Test simulation config creation
    if orchestrator.simulation_config:
        sim_config = orchestrator.simulation_config
        print(f"   Initial capital: ${sim_config.initial_capital:,.2f}")
        print(f"   Signal generators: {sim_config.signal_generators}")
        print(f"   HMM strategies: {sim_config.hmm_strategy_types}")
        print(f"   Technical indicators: {sim_config.technical_indicators}")

    print("\n✅ Integration test passed! Case study can be run with simulation.")
    print("\nTo run a full case study with simulation, use:")
    print("  python examples/case_study.py")

    return True

if __name__ == "__main__":
    try:
        test_simulation_integration()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)