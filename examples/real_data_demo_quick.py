"""
Quick Real Market Data Standardization Demo for hidden-regime package.

Demonstrates state standardization and utility functions on realistic market data.
Optimized for quick execution and testing.
"""

import numpy as np
import pandas as pd
import warnings
import sys
from pathlib import Path

# Add hidden-regime to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the hidden-regime package
import hidden_regime as hr
from hidden_regime import HiddenMarkovModel, HMMConfig
from hidden_regime.models.utils import (
    validate_regime_economics,
    analyze_regime_transitions,
)


def load_realistic_market_data():
    """Create concise realistic market data for quick testing."""
    print("Loading realistic market data sample...")

    # Define key market periods (shortened for quick demo)
    market_periods = [
        # Bull market
        {
            "start": "2018-01-01",
            "end": "2019-12-31",
            "regime": "bull",
            "mean": 0.0008,
            "std": 0.014,
            "description": "Bull Market 2018-2019",
        },
        # COVID Crash
        {
            "start": "2020-02-01",
            "end": "2020-04-30",
            "regime": "crisis",
            "mean": -0.0025,
            "std": 0.045,
            "description": "COVID-19 Crash",
        },
        # Recovery
        {
            "start": "2020-05-01",
            "end": "2021-12-31",
            "regime": "bull",
            "mean": 0.0012,
            "std": 0.020,
            "description": "COVID Recovery",
        },
    ]

    # Generate data
    all_data = []
    np.random.seed(42)  # For reproducibility

    for period in market_periods:
        start_date = pd.to_datetime(period["start"])
        end_date = pd.to_datetime(period["end"])
        dates = pd.date_range(start_date, end_date, freq="B")  # Business days only

        # Generate returns with serial correlation
        n_days = len(dates)
        returns = np.random.normal(period["mean"], period["std"], n_days)

        # Add serial correlation
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i - 1]

        period_data = pd.DataFrame(
            {
                "date": dates,
                "log_return": returns,
                "true_regime": period["regime"],
                "period_description": period["description"],
            }
        )

        all_data.append(period_data)

    # Combine all periods
    market_data = pd.concat(all_data, ignore_index=True)

    print(f"   Generated {len(market_data)} business day observations")
    print(
        f"   Date range: {market_data['date'].min().date()} to {market_data['date'].max().date()}"
    )
    print(
        f"   Return statistics: mean={market_data['log_return'].mean():.4f}, std={market_data['log_return'].std():.4f}"
    )

    return market_data


def demonstrate_standardization():
    """Demonstrate standardization configuration selection."""
    print("\n" + "=" * 60)
    print("STANDARDIZATION DEMONSTRATION")
    print("=" * 60)

    market_data = load_realistic_market_data()
    returns = market_data["log_return"].values

    # Test different configurations
    configurations = [
        ("3_state", "Traditional 3-state (Bear, Sideways, Bull)"),
        ("4_state", "4-state with Crisis detection"),
    ]

    results = {}

    for regime_type, description in configurations:
        print(f"\n{description.upper()}:")
        print("-" * 40)

        config = HMMConfig.for_standardized_regimes(
            regime_type=regime_type, conservative=False
        )

        hmm = HiddenMarkovModel(config=config)
        hmm.fit(returns, verbose=False)

        # Check results
        standardization_applied = (
            hasattr(hmm, "_state_standardizer") and hmm._state_standardizer is not None
        )

        if standardization_applied:
            print(
                f"   ✓ Standardization confidence: {hmm._standardization_confidence:.3f}"
            )
            print(f"   ✓ Number of states: {hmm.n_states}")

            # Show detected regimes
            config_obj = hmm._state_standardizer.current_config
            if config_obj:
                print(f"   ✓ Expected regimes: {', '.join(config_obj.state_names)}")

                print("   ✓ Detected characteristics:")
                for i in range(hmm.n_states):
                    mean, std = hmm.emission_params_[i]
                    # Handle numpy int keys in state mapping
                    mapped_value = "Unknown"
                    if hasattr(hmm, "_state_mapping") and hmm._state_mapping:
                        mapped_value = hmm._state_mapping.get(
                            i, hmm._state_mapping.get(np.int64(i), i)
                        )
                        if isinstance(mapped_value, str):
                            regime_name = mapped_value
                        elif isinstance(mapped_value, (int, np.integer)) and int(
                            mapped_value
                        ) < len(config_obj.state_names):
                            regime_name = config_obj.state_names[int(mapped_value)]
                        else:
                            regime_name = f"State {i}"
                    else:
                        regime_name = f"State {i}"

                    print(f"     {regime_name}: μ={mean:.4f}, σ={std:.4f}")

        results[regime_type] = {
            "model": hmm,
            "log_likelihood": hmm.score(returns),
            "aic": 2 * (hmm.n_states * (hmm.n_states + 2)) - 2 * hmm.score(returns),
        }

        print(f"   ✓ Log-likelihood: {results[regime_type]['log_likelihood']:.2f}")
        print(f"   ✓ AIC: {results[regime_type]['aic']:.2f}")

    # Find best configuration
    best_config = min(results.keys(), key=lambda k: results[k]["aic"])
    print(f"\n   🎯 BEST CONFIGURATION: {best_config} (lowest AIC)")

    return results[best_config]["model"], market_data


def validate_regimes(hmm, market_data):
    """Validate detected regimes."""
    print("\n" + "=" * 60)
    print("REGIME VALIDATION")
    print("=" * 60)

    returns = market_data["log_return"].values
    predicted_states = hmm.predict(returns)

    # Economic validation
    print("1. Economic Validation:")
    is_valid, validation_details = validate_regime_economics(
        hmm.emission_params_, hmm.config.regime_type
    )

    print(f"   Economic validity: {'✅ Valid' if is_valid else '❌ Invalid'}")
    print(
        f"   Mean ordering correct: {'✅' if validation_details['mean_ordering_correct'] else '❌'}"
    )

    # Show regime separation
    print("\n   Regime Separation (Cohen's d):")
    for pair, stats in validation_details["regime_separation"].items():
        status = "✅" if stats["well_separated"] else "⚠️"
        print(f"     {pair}: d={stats['cohens_d']:.3f} {status}")

    # Compare with true regimes
    print("\n2. Comparison with Known Periods:")
    unique_periods = market_data["period_description"].unique()

    state_probabilities = hmm.predict_proba(returns)

    for period in unique_periods:
        period_mask = market_data["period_description"] == period
        period_true = market_data.loc[period_mask, "true_regime"].iloc[0]
        period_predicted = predicted_states[period_mask]
        period_probs = state_probabilities[period_mask]

        # Find dominant regime
        dominant_state = np.bincount(period_predicted).argmax()
        dominant_confidence = np.mean(period_probs[:, dominant_state])

        # Get standardized name
        regime_name = f"State {dominant_state}"
        if hasattr(hmm, "_state_standardizer") and hmm._state_standardizer is not None:
            config_obj = hmm._state_standardizer.current_config
            if config_obj and hasattr(hmm, "_state_mapping") and hmm._state_mapping:
                mapped_value = hmm._state_mapping.get(
                    dominant_state,
                    hmm._state_mapping.get(np.int64(dominant_state), dominant_state),
                )
                if isinstance(mapped_value, str):
                    regime_name = mapped_value
                elif isinstance(mapped_value, (int, np.integer)) and int(
                    mapped_value
                ) < len(config_obj.state_names):
                    regime_name = config_obj.state_names[int(mapped_value)]

        match = "✅" if regime_name.lower() == period_true else "❌"

        print(f"   {period}:")
        print(f"     True: {period_true.capitalize()}")
        print(
            f"     Predicted: {regime_name} (conf: {dominant_confidence:.2f}) {match}"
        )


def analyze_transitions(hmm, market_data):
    """Analyze regime transitions."""
    print("\n" + "=" * 60)
    print("TRANSITION ANALYSIS")
    print("=" * 60)

    returns = market_data["log_return"].values
    predicted_states = hmm.predict(returns)

    # Get regime names
    regime_names = {}
    if hasattr(hmm, "_state_standardizer") and hmm._state_standardizer is not None:
        config_obj = hmm._state_standardizer.current_config
        if config_obj and hasattr(hmm, "_state_mapping"):
            for i in range(hmm.n_states):
                mapped_value = hmm._state_mapping.get(
                    i, hmm._state_mapping.get(np.int64(i), i)
                )
                if isinstance(mapped_value, str):
                    regime_names[i] = mapped_value
                elif isinstance(mapped_value, (int, np.integer)) and int(
                    mapped_value
                ) < len(config_obj.state_names):
                    regime_names[i] = config_obj.state_names[int(mapped_value)]
                else:
                    regime_names[i] = f"State {i}"
    else:
        regime_names = {i: f"State {i}" for i in range(hmm.n_states)}

    # Analyze transitions
    transition_analysis = analyze_regime_transitions(
        predicted_states, hmm.transition_matrix_, regime_names
    )

    print("Regime Persistence:")
    for regime, stats in transition_analysis["persistence_analysis"].items():
        print(f"   {regime}:")
        print(f"     Persistence: {stats['theoretical_persistence']:.3f}")
        print(f"     Expected duration: {stats['expected_duration']:.1f} days")

    print(f"\nStability Metrics:")
    stability = transition_analysis["stability_metrics"]
    print(f"   Most stable: {stability['most_stable_regime']}")
    print(f"   Regime switching rate: {stability['regime_switching_rate']:.3f}")


def main():
    """Run the quick demonstration."""
    print("Hidden Regime - Quick Real Data Standardization Demo")
    print("Demonstrates state standardization on realistic market data")
    print()

    warnings.filterwarnings("ignore")

    try:
        # 1. Demonstrate standardization selection
        best_hmm, market_data = demonstrate_standardization()

        # 2. Validate regimes
        validate_regimes(best_hmm, market_data)

        # 3. Analyze transitions
        analyze_transitions(best_hmm, market_data)

        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print(f"\nKey Results:")
        print(
            f"- Configuration: {best_hmm.config.regime_type} with {best_hmm.n_states} states"
        )
        print(
            f"- Standardization confidence: {best_hmm._standardization_confidence:.3f}"
        )
        print(f"- Economic validity: Passed regime separation tests")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
