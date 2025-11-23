"""
Dot-Com Bubble Case Study (2000-2002)
Demonstrates regime detection during the tech sector collapse.

Training Period: 1998-1999 (bubble expansion)
Analysis Period: 2000-2002 (bubble burst → collapse → capitulation)

Key Questions:
1. How quickly does HMM detect the regime shift at NASDAQ peak (Mar 2000)?
2. Do pure-play internet stocks (AMZN) show different patterns than old tech (MSFT, INTC)?
3. Can we identify the "dead cat bounce" regimes vs true bear regimes?
4. Does the model stay stable through 9/11 shock?

This uses the MarketEventStudy API for clean, reusable analysis.
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
TICKERS = ["QQQ", "MSFT", "INTC", "CSCO", "AMZN", "SPY"]  # Tech heavy vs broad market

# HMM configuration
N_STATES = 3

# Key bubble events for snapshot generation
BUBBLE_EVENTS = {
    "2000-03-10": "NASDAQ Peak (5,048)",
    "2000-06-10": "Initial Crash (-21%)",
    "2000-10-09": "Market Trough (-50%)",
    "2001-09-11": "9/11 Crisis",
    "2002-10-09": "Bear Market Bottom (-78% from peak)",
}

# Output directory
OUTPUT_DIR = "output/dotcom_study"

# Testing mode - set to False for full 3-year analysis
TESTING_MODE = False  # Set to True to test with just first N days
TESTING_DURATION = 60  # Cap to N days


# =============================================================================
# EXECUTION
# =============================================================================

def banner(info: str) -> None:
    """Print a simple banner"""
    print("\n" + "=" * 80)
    print(f"  {info}")
    print("=" * 80 + "\n")


def main():
    """Run Dot-Com Bubble case study using MarketEventStudy API."""

    banner("2000 DOT-COM BUBBLE CASE STUDY")

    # Create market event study with signal generation enabled
    study = hr.MarketEventStudy(
        ticker=TICKERS,
        training_start=TRAINING_START,
        training_end=TRAINING_END,
        analysis_start=ANALYSIS_START,
        analysis_end=ANALYSIS_END,
        n_states=N_STATES,
        key_events=BUBBLE_EVENTS,
        output_dir=OUTPUT_DIR,
        generate_signals=True,  # NEW: Enable trading signal generation
        signal_strategy='regime_following',  # NEW: Regime-following strategy
    )

    # Run complete analysis
    study.run(
        create_snapshots=True,
        create_animations=False,  # Can enable for GIF animations
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

    # Validate signal consistency with training data
    study.print_signal_consistency_report()

    # Analyze regime paradigm shift
    study.print_regime_paradigm_report()

    # Create paradigm shift visualization
    banner("REGIME PARADIGM SHIFT VISUALIZATION")
    study.create_paradigm_shift_visualization()

    # Create full timeline visualization
    banner("FULL-PERIOD TIMELINE VISUALIZATION")
    study.create_full_timeline_visualization()

    banner("DOT-COM BUBBLE ANALYSIS COMPLETE")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
