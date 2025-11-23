"""
COVID-19 Crisis Case Study
Demonstrates regime detection during the fastest market crash in history.

Training Period: 2018-2019 (pre-COVID)
Analysis Period: 2020 (COVID crash and recovery)

Key Questions:
1. How quickly does HMM detect the crisis regime? (Feb 19-25?)
2. Does the model stay stable (not flipping daily)?
3. How do stocks differ in regime patterns?
4. Can we quantify recovery speed via regime metrics?

This refactored version uses the MarketEventStudy API for clean,
reusable analysis with minimal code duplication.
"""

import hidden_regime as hr

# =============================================================================
# CONFIGURATION
# =============================================================================

# Date ranges
TRAINING_START = "2018-01-01"
TRAINING_END = "2019-12-31"
ANALYSIS_START = "2020-01-01"
ANALYSIS_END = "2020-12-31"

# Tickers to compare
TICKERS = ["QQQ", "CCL", "WMT", "AMZN", "DIS", "INTC"] # Subjects of our study  
# TICKERS = ['INTC']  # Just for testing

# HMM configuration
N_STATES = 3

# Key COVID events for snapshot generation
COVID_EVENTS = {
    "2020-02-19": "Market Peak",
    "2020-03-11": "WHO Declares Pandemic",
    "2020-03-16": "Fed Emergency Rate Cut",
    "2020-03-23": "Market Bottom",
    "2020-03-27": "CARES Act Signed",
}

# Output directory
OUTPUT_DIR = "output/covid_study"

# Testing mode - set to False for full year analysis
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
    """Run COVID-19 case study using MarketEventStudy API."""

    banner("COVID-19 CRISIS CASE STUDY")

    # Create market event study with signal generation enabled
    study = hr.MarketEventStudy(
        ticker=TICKERS,
        training_start=TRAINING_START,
        training_end=TRAINING_END,
        analysis_start=ANALYSIS_START,
        analysis_end=ANALYSIS_END,
        n_states=N_STATES,
        key_events=COVID_EVENTS,
        output_dir=OUTPUT_DIR,
        generate_signals=True,  # NEW: Enable trading signal generation
        signal_strategy='regime_following',  # NEW: Regime-following strategy
    )

    # Run complete analysis
    study.run(
        create_snapshots=True,
        create_animations=True,
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
    banner("PARADIGM SHIFT VISUALIZATION")
    study.create_paradigm_shift_visualization()

    banner("COMPLETE")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
