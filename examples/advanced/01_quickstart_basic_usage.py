"""
Multivariate HMM Quickstart - Get Started in 5 Minutes

This is the simplest possible introduction to multivariate regime detection.
Perfect for: "I just want to see how this works"

Key Concepts:
1. Use hr.create_multivariate_pipeline() to create a complete pipeline
2. Multivariate HMMs combine returns + volatility to detect market regimes
3. Results include regime labels (Bull, Bear, Sideways) + confidence scores
4. Automatic feature standardization - no preprocessing needed!

This example loads real market data, trains a model, and shows results.
Total runtime: ~30 seconds with real data, ~5 seconds with cached data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hidden_regime as hr


def main():
    print("=" * 70)
    print("MULTIVARIATE HMM QUICKSTART")
    print("=" * 70)

    # Configuration
    ticker = 'SPY'
    n_states = 3
    start_date = '2023-01-01'
    end_date = '2024-01-01'

    print(f"\nConfiguration:")
    print(f"  Ticker: {ticker}")
    print(f"  States: {n_states}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Features: log_return + realized_vol (automatic)")

    # Create and run pipeline
    print("\n" + "-" * 70)
    print("Creating multivariate pipeline...")
    print("-" * 70)

    pipeline = hr.create_multivariate_pipeline(
        ticker=ticker,
        n_states=n_states,
        features=['log_return', 'realized_vol'],
        start_date=start_date,
        end_date=end_date
    )

    print("Training model...")
    report = pipeline.update()
    print("✓ Complete!\n")

    # Extract key metrics from interpreter output
    result = pipeline.component_outputs['interpreter']
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    latest = result.iloc[-1]
    print(f"\nLatest Regime (as of {result.index[-1].date()}):")
    print(f"  Regime: {latest['regime_label']}")
    print(f"  Confidence: {latest['confidence']:.1%}")
    print(f"  State: {latest['predicted_state']}")

    # Show regime breakdown
    print("\nRegime Distribution:")
    regime_counts = result['regime_label'].value_counts()
    for regime, count in regime_counts.items():
        pct = (count / len(result)) * 100
        print(f"  {regime:15s}: {count:4d} days ({pct:5.1f}%)")

    # Show multivariate metrics
    print("\nMultivariate Characteristics (latest):")
    print(f"  Eigenvalue ratio: {latest['multivariate_eigenvalue_ratio']:.2f}")
    print(f"    (1=isotropic, >10=extreme concentration)")
    print(f"  Variance concentration: {latest['multivariate_variance_concentration']}")
    print(f"  Feature correlation: {latest['multivariate_correlation_regime']}")

    # Statistics
    print("\nModel Statistics:")
    model = pipeline.model
    print(f"  Converged: {model.training_history_['converged']}")
    print(f"  Iterations: {model.training_history_['iterations']}")
    print(f"  Final log-likelihood: {model.training_history_['log_likelihoods'][-1]:.2f}")

    # Plot results
    print("\n" + "-" * 70)
    print("Creating visualization...")
    print("-" * 70)

    data = pipeline.component_outputs['data']

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Price with regime coloring
    ax = axes[0]
    for state in range(n_states):
        mask = result['predicted_state'] == state
        ax.scatter(
            result.index[mask], data['close'][mask],
            c=f'C{state}', alpha=0.6, s=20,
            label=f'State {state}'
        )
    ax.set_ylabel('Price ($)')
    ax.set_title(f'{ticker} - Market Regimes')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 2: Confidence scores
    ax = axes[1]
    ax.fill_between(result.index, result['confidence'], alpha=0.5, color='blue')
    ax.set_ylabel('Confidence')
    ax.set_ylim([0, 1])
    ax.set_title('Prediction Confidence')
    ax.grid(True, alpha=0.3)

    # Plot 3: Eigenvalue ratio (how concentrated variance is)
    ax = axes[2]
    ax.plot(result.index, result['multivariate_eigenvalue_ratio'], color='purple', linewidth=1)
    ax.axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='Isotropic (1.5)')
    ax.axhline(y=3, color='orange', linestyle='--', alpha=0.5, label='High (3)')
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Extreme (10)')
    ax.set_ylabel('Eigenvalue Ratio')
    ax.set_xlabel('Date')
    ax.set_title('Variance Concentration (multivariate metric)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quickstart_regime_detection.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot to quickstart_regime_detection.png")

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. Multivariate HMMs combine returns and volatility
   - Volatility clusters by regime (high in bear, low in bull)
   - This makes regime detection more stable than returns alone

2. Automatic feature standardization
   - Pipeline handles scale differences automatically
   - No preprocessing needed!

3. Eigenvalue ratio shows variance structure
   - Shows how "concentrated" risk is in a regime
   - Changes across regimes reveal market characteristics

4. Confidence scores indicate prediction reliability
   - Use for position sizing or signal filtering
   - Higher confidence = more reliable regime

Next Steps:
- See example 02 for real-world crisis detection (COVID 2020)
- See example 03 for feature selection guidance
- See notebook 03 (why_multivariate_wins.ipynb) for theory
- See docs/guides/MULTIVARIATE_HMM_GUIDE.md for comprehensive guide
    """)

    plt.show()


if __name__ == '__main__':
    main()
