"""
2000 Dot-Com Bubble Case Study
Analyzes regime detection during the tech bubble, crash, and recovery.

Training Period: 1998-1999 (pre-bubble euphoria)
Analysis Period: 2000-2002 (bubble peak, crash, bear market grind)

Key Questions:
1. Can HMM detect euphoric bull regime before peak?
2. How quickly does regime transition to bear market?
3. Do pure-play dot-coms (AMZN) differ from established tech (MSFT, INTC)?
4. What is the shape of recovery from bear market?

This study uses MarketEventStudy API for clean, event-focused analysis
with professional visualizations.
"""

import hidden_regime as hr

# =============================================================================
# CONFIGURATION
# =============================================================================

# Date ranges
TRAINING_START = "1998-01-01"
TRAINING_END = "1999-12-31"
ANALYSIS_START = "2000-01-01"
ANALYSIS_END = "2002-12-31"

# Tickers to compare
# MSFT, INTC, CSCO: Established tech with real earnings
# AMZN: Pure dot-com, speculative, no profits
# QQQ: Entire NASDAQ-100 tech sector
# SPY: Broad market for comparison
TICKERS = ["QQQ", "MSFT", "INTC", "CSCO", "AMZN", "SPY"]

# HMM configuration
N_STATES = 3

# Key dot-com bubble events for snapshot generation
BUBBLE_EVENTS = {
    "2000-03-10": "NASDAQ Peak (5,048)",
    "2000-06-10": "Initial Crash (-21%)",
    "2000-10-09": "Market Trough (-50%)",
    "2001-09-11": "9/11 Crisis",
    "2002-10-09": "Bear Market Bottom (-78% from peak)",
}

# Output directory
OUTPUT_DIR = "output/dotcom_study"

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
    """Run dot-com bubble case study using MarketEventStudy API."""

    banner("2000 DOT-COM BUBBLE CASE STUDY")

    # Create market event study with proper event-study methodology
    study = hr.MarketEventStudy(
        ticker=TICKERS,
        training_start=TRAINING_START,
        training_end=TRAINING_END,
        analysis_start=ANALYSIS_START,
        analysis_end=ANALYSIS_END,
        n_states=N_STATES,
        key_events=BUBBLE_EVENTS,
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

    # Analyze regime paradigm shifts during bubble/crash
    study.print_regime_paradigm_report()

    # Create paradigm shift visualization
    banner("REGIME PARADIGM SHIFT VISUALIZATION")
    study.create_paradigm_shift_visualization()

    # Create full-period timeline visualization
    banner("FULL-PERIOD TIMELINE VISUALIZATION")
    study.create_full_timeline_visualization()

    banner("DOT-COM BUBBLE ANALYSIS COMPLETE")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
