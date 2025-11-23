"""
Regime Quality Validation Example

Demonstrates how to assess and validate regime detection quality using built-in metrics.

Quality Metrics:
1. Log-Likelihood: Model fit to data (higher is better)
2. Regime Persistence: How stable regimes are (0-1 scale)
3. Regime Duration: Average days in each regime
4. Quality Warnings: Automatic detection of potential issues

This example shows:
- How to access quality metrics programmatically
- How to interpret the quality report
- How to identify and address quality issues
- How to compare quality across different configurations
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hidden_regime.config.model import HMMConfig
from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.config.observation import FinancialObservationConfig
from hidden_regime.factories import component_factory


def analyze_regime_quality(
    ticker: str,
    start_date: str,
    end_date: str,
    n_states: int = 3,
    initialization_method: str = 'kmeans'
):
    """
    Analyze regime detection quality for a given ticker and configuration.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date for analysis (YYYY-MM-DD)
        end_date: End date for analysis (YYYY-MM-DD)
        n_states: Number of HMM states
        initialization_method: 'kmeans', 'random', or 'custom'

    Returns:
        dict: Quality metrics and warnings
    """
    print(f"\n{'='*80}")
    print(f"Analyzing Regime Quality: {ticker}")
    print(f"Period: {start_date} to {end_date}")
    print(f"States: {n_states}, Initialization: {initialization_method}")
    print(f"{'='*80}\n")

    # Step 1: Load data
    print("Step 1: Loading data...")
    data_config = FinancialDataConfig(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )
    data_loader = component_factory.create_data_component(data_config)
    raw_data = data_loader.update()
    print(f"  Loaded {len(raw_data)} observations")

    # Step 2: Generate observations
    print("\nStep 2: Generating log return observations...")
    obs_config = FinancialObservationConfig(price_column='close')
    obs_gen = component_factory.create_observation_component(obs_config)
    observations = obs_gen.update(raw_data)
    print(f"  Generated {len(observations)} observations")

    # Step 3: Train HMM model
    print(f"\nStep 3: Training {n_states}-state HMM...")
    config = HMMConfig(
        n_states=n_states,
        observed_signal='log_return',
        max_iterations=100,
        tolerance=1e-6,
        initialization_method=initialization_method
    )

    hmm = component_factory.create_model_component(config)
    hmm.fit(observations)
    print("  Model training complete")

    # Step 4: Print quality report
    print(f"\n{'='*80}")
    print("QUALITY REPORT")
    print(f"{'='*80}\n")
    hmm.print_quality_report(observations)

    # Step 5: Get metrics programmatically
    print(f"\n{'='*80}")
    print("DETAILED QUALITY METRICS")
    print(f"{'='*80}\n")

    metrics = hmm.get_quality_metrics(observations)

    # Log-likelihood metrics
    print("📊 Log-Likelihood Metrics:")
    print(f"  Total Log-Likelihood: {metrics['log_likelihood']['total']:.2f}")
    print(f"  Per-Observation: {metrics['log_likelihood']['per_observation']:.4f}")

    # Regime persistence
    print("\n🔄 Regime Persistence:")
    print(f"  Average: {metrics['regime_persistence']['average']:.1%}")
    print(f"  By State:")
    for state, persistence in metrics['regime_persistence']['by_state'].items():
        print(f"    State {state}: {persistence:.1%}")

    # Regime durations
    print("\n⏱️  Regime Durations:")
    print(f"  Average: {metrics['regime_durations']['average']:.2f} days")
    print(f"  By State:")
    for state, duration in metrics['regime_durations']['by_state'].items():
        print(f"    State {state}: {duration:.2f} days")

    # Quality issues
    print("\n⚠️  Quality Assessment:")
    if metrics['quality_issues']:
        print(f"  Issues Found: {len(metrics['quality_issues'])}")
        for i, issue in enumerate(metrics['quality_issues'], 1):
            print(f"    {i}. {issue}")
    else:
        print("  ✅ No quality issues detected!")

    # Interpretation guidance
    print(f"\n{'='*80}")
    print("INTERPRETATION GUIDANCE")
    print(f"{'='*80}\n")

    print("📈 Log-Likelihood:")
    print("  - Higher is better (less negative)")
    print("  - Per-observation: > -2.0 is good, > -1.5 is excellent")
    print("  - Only log-likelihood is used for quality assessment")

    print("\n🔄 Regime Persistence:")
    print("  - Reported for information only (not a quality issue)")
    print("  - Higher values (>0.7) = more stable regimes")
    print("  - Lower values (<0.5) = frequent regime changes")
    print("  - Optimal value depends on market conditions")

    print("\n⏱️  Regime Duration:")
    print("  - Reported for information only (not a quality issue)")
    print("  - Longer durations = more stable regimes")
    print("  - Shorter durations = more responsive to market changes")
    print("  - Optimal value depends on trading timeframe")

    return metrics


def compare_configurations(ticker: str, start_date: str, end_date: str):
    """Compare regime quality across different HMM configurations."""

    print("\n" + "="*80)
    print("COMPARING CONFIGURATIONS")
    print("="*80 + "\n")

    configurations = [
        {'n_states': 2, 'init': 'kmeans'},
        {'n_states': 3, 'init': 'kmeans'},
        {'n_states': 4, 'init': 'kmeans'},
        {'n_states': 3, 'init': 'random'},
    ]

    results = []

    for config in configurations:
        metrics = analyze_regime_quality(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            n_states=config['n_states'],
            initialization_method=config['init']
        )

        results.append({
            'n_states': config['n_states'],
            'init_method': config['init'],
            'log_likelihood': metrics['log_likelihood']['per_observation'],
            'avg_persistence': metrics['regime_persistence']['average'],
            'avg_duration': metrics['regime_durations']['average'],
            'has_issues': len(metrics['quality_issues']) > 0
        })

    # Print comparison table
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80 + "\n")

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    print("\nRecommendation:")
    best_ll = df.loc[df['log_likelihood'].idxmax()]
    print(f"  Best log-likelihood: {best_ll['n_states']} states with {best_ll['init_method']} initialization")
    print(f"  Log-likelihood per observation: {best_ll['log_likelihood']:.4f}")


def main():
    """Run regime quality validation examples."""

    print("\n" + "="*80)
    print("REGIME QUALITY VALIDATION EXAMPLE")
    print("="*80)
    print("\nThis example demonstrates how to assess and validate regime detection quality.")
    print("We'll analyze SPY (S&P 500 ETF) from 2020-2024.\n")

    # Single configuration analysis
    metrics = analyze_regime_quality(
        ticker="SPY",
        start_date="2020-01-01",
        end_date="2024-01-01",
        n_states=3,
        initialization_method='kmeans'
    )

    # Compare multiple configurations
    print("\n" + "="*80)
    print("Now let's compare different configurations...")
    print("="*80)

    compare_configurations(
        ticker="SPY",
        start_date="2020-01-01",
        end_date="2024-01-01"
    )

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Quality validation helps you:
1. Assess model fit to data (log-likelihood)
2. Understand regime stability (persistence & duration)
3. Identify potential issues automatically
4. Compare different model configurations
5. Make informed decisions about n_states and initialization

Key Principle: Only log-likelihood indicates quality issues.
Persistence and duration are informational metrics that depend
on your specific use case and market conditions.
""")


if __name__ == '__main__':
    main()
