"""
Market Event Case Study Template

This template shows how to use the MarketEventStudy API to analyze
any market event (crashes, bubbles, sector rotations, policy changes, etc.)

Simply:
1. Define your training and analysis periods
2. Specify key event dates you want to snapshot
3. Run the study
4. Export results

The framework handles all the heavy lifting:
- Data loading and splitting
- HMM training on pre-event data
- Day-by-day temporal analysis
- Snapshot PNG generation at key dates
- Animation creation
- Metric computation
- Result export
"""

import hidden_regime as hr

# =============================================================================
# CONFIGURATION - Customize these for your event
# =============================================================================

# Event details
STUDY_NAME = "My Market Event"
OUTPUT_DIR = "output/my_event_study"

# Ticker(s) to analyze
# Can be single ticker or list: ['QQQ', 'SPY', 'AAPL']
TICKER = "SPY"

# Date ranges
TRAINING_START = "2018-01-01"  # Start of pre-event training period
TRAINING_END = "2019-12-31"    # End of pre-event training period
ANALYSIS_START = "2020-01-01"  # Start of event period to analyze
ANALYSIS_END = "2020-12-31"    # End of event period (or None for "today")

# HMM configuration
N_STATES = 3  # Number of regime states (typically 3-4)

# Key events for snapshot generation
# Format: {'YYYY-MM-DD': 'Event Description'}
KEY_EVENTS = {
    "2020-02-19": "Event Start / Peak",
    "2020-03-23": "Event Bottom / Trough",
    "2020-06-30": "Mid-Event",
    "2020-12-31": "Event End",
}

# Optional: Additional pipeline kwargs
# Example: tolerance=1e-6, max_iterations=200, random_seed=42
PIPELINE_KWARGS = {}

# =============================================================================
# EXECUTION - Usually no changes needed below this line
# =============================================================================


def main():
    """Run the market event study."""
    print(f"\n{'=' * 80}")
    print(f"  {STUDY_NAME}")
    print(f"{'=' * 80}\n")

    # Create study
    study = hr.MarketEventStudy(
        ticker=TICKER,
        training_start=TRAINING_START,
        training_end=TRAINING_END,
        analysis_start=ANALYSIS_START,
        analysis_end=ANALYSIS_END,
        n_states=N_STATES,
        key_events=KEY_EVENTS,
        output_dir=OUTPUT_DIR,
        **PIPELINE_KWARGS,
    )

    # Run analysis
    # Set testing_mode=True to test with just first 30 days
    results = study.run(
        create_snapshots=True,
        create_animations=False,  # Set to True for GIF animations
        snapshot_window_days=90,
        testing_mode=False,  # Set to True for testing
        testing_limit_days=30,
    )

    # Print summary
    study.print_summary()

    # Export results
    study.export_results(format="csv")

    print(f"\n{'=' * 80}")
    print(f"  Analysis Complete!")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
