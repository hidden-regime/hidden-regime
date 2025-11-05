"""
2008 Financial Crisis Case Study
Analyzes regime detection during the systemic financial crisis.

Training Period: 2005-2006 (pre-crisis bull market)
Analysis Period: 2007-2009 (crisis emergence, peak, recovery)

Key Questions:
1. Do HMMs detect early warning signs before Lehman collapse?
2. How do equity, financial, bond, and commodity regimes diverge?
3. Does safe-haven behavior appear in TLT and GLD?
4. What is the regime structure of the crisis peak?

This study uses MarketEventStudy API for clean, event-focused analysis
with professional visualizations of cross-asset regime dynamics.
"""

import hidden_regime as hr

# =============================================================================
# CONFIGURATION
# =============================================================================

# Date ranges
TRAINING_START = "2005-01-01"
TRAINING_END = "2006-12-31"
ANALYSIS_START = "2007-01-01"
ANALYSIS_END = "2009-12-31"

# Tickers to compare across asset classes
# SPY: Broad market equities
# XLF: Financial sector (crisis epicenter)
# TLT: 20+ year treasuries (safe haven)
# GLD: Gold (alternative safe haven)
TICKERS = ["SPY", "XLF", "TLT", "GLD"]

# HMM configuration
N_STATES = 3

# Key crisis events for snapshot generation
CRISIS_EVENTS = {
    "2007-10-09": "Market Peak (before credit crisis)",
    "2008-03-14": "Bear Stearns Collapse",
    "2008-09-15": "Lehman Brothers Bankruptcy",
    "2008-09-29": "TARP Passage",
    "2009-03-09": "Market Trough (S&P 500 at 676)",
    "2009-03-23": "Fed Announces QE (Recovery Begins)",
}

# Output directory
OUTPUT_DIR = "output/crisis_2008_study"

# Testing mode
TESTING_MODE = False  # Set to True for quick test
TESTING_DURATION = 60


# =============================================================================
# EXECUTION
# =============================================================================

def banner(info: str) -> None:
    """Print a simple banner"""
    print("\n" + "=" * 80)
    print(f"  {info}")
    print("=" * 80 + "\n")


def main():
    """Run 2008 financial crisis case study using MarketEventStudy API."""

    banner("2008 FINANCIAL CRISIS CASE STUDY")

    # Create market event study with proper event-study methodology
    study = hr.MarketEventStudy(
        ticker=TICKERS,
        training_start=TRAINING_START,
        training_end=TRAINING_END,
        analysis_start=ANALYSIS_START,
        analysis_end=ANALYSIS_END,
        n_states=N_STATES,
        key_events=CRISIS_EVENTS,
        output_dir=OUTPUT_DIR,
        generate_signals=True,  # Enable trading signal generation
        signal_strategy='regime_following',
    )

    # Run complete analysis with visualizations
    study.run(
        create_snapshots=True,
        create_animations=False,
        snapshot_window_days=90,
        animation_fps=5,
        testing_mode=TESTING_MODE,
        testing_limit_days=TESTING_DURATION,
    )

    # Print summary
    study.print_summary()

    # Export results
    study.export_results(format="csv")

    # Export trading signals for QuantConnect
    banner("TRADING SIGNALS EXPORT")
    study.export_signals_for_quantconnect()

    # Validate signal consistency
    study.print_signal_consistency_report()

    # Analyze regime paradigm shifts during crisis/recovery
    study.print_regime_paradigm_report()

    # Create paradigm shift visualization showing cross-asset dynamics
    banner("CRISIS REGIME PARADIGM SHIFT VISUALIZATION")
    study.create_paradigm_shift_visualization()

    # Create full-period timeline visualization
    banner("FULL-PERIOD TIMELINE VISUALIZATION")
    study.create_full_timeline_visualization()

    banner("2008 FINANCIAL CRISIS ANALYSIS COMPLETE")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
