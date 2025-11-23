"""
Test the updated quality metrics that no longer judge persistence/duration.

Verify that:
1. Only log-likelihood is assessed
2. Persistence and duration are reported but not judged
3. No false warnings for low persistence on individual stocks
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.config.model import HMMConfig
from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.config.observation import FinancialObservationConfig
from hidden_regime.data.financial import FinancialDataLoader
from hidden_regime.observations.financial import FinancialObservationGenerator


def test_stock(symbol: str, start_date: str = "2023-01-01", end_date: str = "2024-01-01"):
    """Test quality metrics on a stock."""
    print(f"\n{'='*80}")
    print(f"Testing {symbol}")
    print(f"{'='*80}")

    # Load data
    data_loader = FinancialDataLoader(
        FinancialDataConfig(ticker=symbol, start_date=start_date, end_date=end_date)
    )
    raw_data = data_loader.update()

    # Generate observations
    obs_gen = FinancialObservationGenerator(
        FinancialObservationConfig(price_column='close')
    )
    observations = obs_gen.update(raw_data)

    # Create and fit model
    config = HMMConfig(
        n_states=3,
        observed_signal='log_return',
        max_iterations=100,
        tolerance=1e-6,
        initialization_method='kmeans'
    )

    hmm = HiddenMarkovModel(config)
    hmm.fit(observations)

    # Print quality report
    hmm.print_quality_report(observations)

    # Get metrics programmatically
    metrics = hmm.get_quality_metrics(observations)

    # Verify behavior
    print("\nVERIFICATION:")
    print(f"  Log-likelihood: {metrics['log_likelihood']['per_observation']:.4f}")
    print(f"  Avg Duration: {metrics['regime_durations']['average']:.2f} days")
    print(f"  Avg Persistence: {metrics['regime_persistence']['average']:.3f}")

    # Check that quality issues only relate to log-likelihood
    if metrics['quality_issues']:
        print(f"  Quality Issues: {metrics['quality_issues']}")
        for issue in metrics['quality_issues']:
            assert 'likelihood' in issue.lower(), f"Unexpected quality issue: {issue}"
            print(f"   Quality issue correctly relates to log-likelihood")
    else:
        print(f"   No quality issues (log-likelihood is good)")

    # Verify that low persistence/duration don't trigger issues
    if metrics['regime_persistence']['average'] < 0.6:
        print(f"   Low persistence ({metrics['regime_persistence']['average']:.1%}) correctly NOT flagged as issue")

    if metrics['regime_durations']['average'] < 2.0:
        print(f"   Short duration ({metrics['regime_durations']['average']:.2f} days) correctly NOT flagged as issue")


def main():
    """Test on multiple stocks with varying characteristics."""

    print("TESTING UPDATED QUALITY METRICS")
    print("="*80)
    print("\nObjective: Verify that only log-likelihood is assessed,")
    print("and persistence/duration are reported without judgment.\n")

    # Test on different asset types
    test_stock("SPY", "2022-01-01", "2024-01-01")     # Index - stable, high persistence
    test_stock("TSLA", "2022-01-01", "2024-01-01")    # Volatile - lower persistence
    test_stock("KO", "2022-01-01", "2024-01-01")      # Individual stock - moderate

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(" All tests passed!")
    print(" Only log-likelihood is assessed for quality")
    print(" Persistence and duration are reported as descriptive metrics")
    print(" No false warnings for low persistence/duration")


if __name__ == '__main__':
    main()
