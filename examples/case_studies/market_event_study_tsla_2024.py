"""
Market Event Study: TSLA 2024

Tests visualization and plotting capabilities by analyzing TSLA throughout 2024
with 2022-2023 as training period.

This case study:
- Trains the HMM model on 2022-2023
- Analyzes TSLA through 2024 day-by-day (temporal analysis)
- Creates snapshots at key events
- Generates visualizations and metrics
- Tests the entire visualization pipeline

Key events marked:
- 2024-01-01: Year start
- 2024-03-15: Q1 end (investor expectations peak)
- 2024-06-15: Mid-year checkpoint
- 2024-10-01: Q4 starts (Cybertruck ramping)
- 2024-12-27: Year end
"""

import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from hidden_regime.analysis.market_event_study import MarketEventStudy


def main():
    """Run TSLA market event study with visualizations."""
    print("\n" + "=" * 80)
    print("TSLA Market Event Study: 2024 Analysis")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Define key events throughout 2024
    key_events = {
        "2024-01-01": "Year Start",
        "2024-01-30": "Earnings Surprise",
        "2024-03-15": "Q1 Close",
        "2024-04-01": "Cybertruck Ramp Begins",
        "2024-06-15": "Mid-Year Checkpoint",
        "2024-07-15": "Second Half Begins",
        "2024-09-15": "Pre-Q3 Earnings",
        "2024-10-01": "Q4 Starts",
        "2024-12-27": "Year End",
    }

    # Create market event study
    study = MarketEventStudy(
        ticker="TSLA",
        training_start="2022-01-01",
        training_end="2023-12-31",
        analysis_start="2024-01-01",
        analysis_end="2024-12-31",
        n_states=3,
        key_events=key_events,
        output_dir="/mnt/c/Workspace/hidden-regime/output/tsla_2024_market_study",
        generate_signals=True,  # Generate regime-following signals
        signal_strategy="regime_following",
    )

    print("Configuration:")
    print(f"  Ticker: TSLA")
    print(f"  Training: 2022-01-01 to 2023-12-31")
    print(f"  Analysis: 2024-01-01 to 2024-12-31")
    print(f"  Model States: 3")
    print(f"  Generate Signals: Yes (regime_following)")
    print(f"  Key Events: {len(key_events)}")
    print(f"  Output Directory: {study.output_dir}")
    print()

    # Run the study with visualizations
    print("Running Market Event Study...")
    print("-" * 80)

    results = study.run(
        create_snapshots=True,  # Generate PNG snapshots at key events
        create_animations=True,  # Skip animations for speed
        snapshot_window_days=30,  # Show 30-day window around each event
        testing_mode=False,  # Full analysis
    )

    # Print summary
    print()
    print("=" * 80)
    print("Analysis Complete")
    print("=" * 80)
    print()

    study.print_summary()

    print()
    print("Output Files:")
    print(f"  Directory: {study.output_dir}")
    print("  Contents:")
    print("    - Snapshots (PNG) for each key event")
    print("    - Analysis report (markdown)")
    print("    - Metrics summary")
    print()


if __name__ == "__main__":
    main()
