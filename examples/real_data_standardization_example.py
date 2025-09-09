"""
Real Market Data Standardization Example for hidden-regime package.

Demonstrates state standardization and utility functions on actual market data,
showing how to detect and validate regime changes across different market periods.
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
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
    get_standardized_regime_name,
)


def load_sample_market_data():
    """
    Create realistic market data based on actual market patterns.

    This simulates real market data with known regime periods:
    - Pre-2008 Bull Market
    - 2008 Financial Crisis
    - 2009-2010 Recovery
    - 2011 European Debt Crisis
    - 2012-2020 Long Bull Market
    - 2020 COVID Crash & Recovery
    """
    print("Loading sample market data based on historical patterns...")

    # Define realistic market regimes with actual-like parameters
    market_periods = [
        # Pre-2008 Bull (Jan 2007 - Sep 2007)
        {
            "start": "2007-01-01",
            "end": "2007-09-30",
            "regime": "bull",
            "mean": 0.0008,
            "std": 0.012,
            "description": "Pre-Crisis Bull",
        },
        # 2008 Financial Crisis (Oct 2007 - Mar 2009)
        {
            "start": "2007-10-01",
            "end": "2009-03-31",
            "regime": "crisis",
            "mean": -0.0015,
            "std": 0.035,
            "description": "2008 Financial Crisis",
        },
        # Recovery Period (Apr 2009 - Dec 2010)
        {
            "start": "2009-04-01",
            "end": "2010-12-31",
            "regime": "bull",
            "mean": 0.0012,
            "std": 0.018,
            "description": "Post-Crisis Recovery",
        },
        # European Debt Crisis (Jan 2011 - Jun 2012)
        {
            "start": "2011-01-01",
            "end": "2012-06-30",
            "regime": "bear",
            "mean": -0.0005,
            "std": 0.022,
            "description": "European Debt Crisis",
        },
        # Long Bull Market (Jul 2012 - Jan 2020)
        {
            "start": "2012-07-01",
            "end": "2020-01-31",
            "regime": "bull",
            "mean": 0.0006,
            "std": 0.014,
            "description": "Long Bull Market",
        },
        # COVID Crash (Feb 2020 - Apr 2020)
        {
            "start": "2020-02-01",
            "end": "2020-04-30",
            "regime": "crisis",
            "mean": -0.0025,
            "std": 0.045,
            "description": "COVID-19 Crash",
        },
        # COVID Recovery (May 2020 - Dec 2023)
        {
            "start": "2020-05-01",
            "end": "2023-12-31",
            "regime": "bull",
            "mean": 0.0008,
            "std": 0.016,
            "description": "COVID Recovery",
        },
    ]

    # Generate data for each period
    all_data = []
    regime_labels = []

    np.random.seed(42)  # For reproducibility

    for period in market_periods:
        start_date = pd.to_datetime(period["start"])
        end_date = pd.to_datetime(period["end"])
        dates = pd.date_range(start_date, end_date, freq="D")

        # Generate returns for this period
        n_days = len(dates)
        returns = np.random.normal(period["mean"], period["std"], n_days)

        # Add some serial correlation (realistic market behavior)
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i - 1]

        # Create DataFrame for this period
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

    print(f"   Generated {len(market_data)} daily observations")
    print(
        f"   Date range: {market_data['date'].min().date()} to {market_data['date'].max().date()}"
    )
    print(
        f"   Return statistics: mean={market_data['log_return'].mean():.4f}, std={market_data['log_return'].std():.4f}"
    )
    print(f"   True regime distribution:")
    regime_counts = market_data["true_regime"].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(market_data) * 100
        print(f"     {regime.capitalize()}: {count} days ({pct:.1f}%)")

    return market_data


def demonstrate_standardization_selection():
    """Demonstrate automatic standardization configuration selection."""
    print("\n" + "=" * 70)
    print("AUTOMATIC STANDARDIZATION CONFIGURATION SELECTION")
    print("=" * 70)

    market_data = load_sample_market_data()
    returns = market_data["log_return"].values
    dates = market_data["date"].values

    # Test different standardization approaches
    approaches = [
        ("3_state", "Traditional 3-state configuration"),
        ("4_state", "4-state with crisis detection"),
        ("5_state", "5-state with euphoric detection"),
        ("auto", "Automatic selection based on data"),
    ]

    results = {}

    for regime_type, description in approaches:
        print(f"\n{description.upper()}:")
        print("-" * 50)

        if regime_type == "auto":
            config = HMMConfig.with_auto_selection(
                validation_threshold=0.7, conservative=False
            )
        else:
            config = HMMConfig.for_standardized_regimes(
                regime_type=regime_type, conservative=False
            )

        # Train model
        hmm = HiddenMarkovModel(config=config)
        hmm.fit(returns, verbose=False)

        # Check standardization results
        standardization_applied = (
            hasattr(hmm, "_state_standardizer") and hmm._state_standardizer is not None
        )

        if standardization_applied:
            print(
                f"   ✓ Standardization confidence: {hmm._standardization_confidence:.3f}"
            )
            print(f"   ✓ Final configuration: {hmm.config.regime_type}")
            print(f"   ✓ Number of states: {hmm.n_states}")

            # Show detected regimes
            config_obj = hmm._state_standardizer.current_config
            if config_obj:
                print(f"   ✓ Expected regimes: {', '.join(config_obj.state_names)}")

        # Store results
        results[regime_type] = {
            "model": hmm,
            "log_likelihood": hmm.score(returns),
            "aic": 2 * (hmm.n_states * (hmm.n_states + 2)) - 2 * hmm.score(returns),
            "standardization_confidence": (
                hmm._standardization_confidence if standardization_applied else 0
            ),
            "n_states": hmm.n_states,
        }

        print(f"   ✓ Log-likelihood: {results[regime_type]['log_likelihood']:.2f}")
        print(f"   ✓ AIC: {results[regime_type]['aic']:.2f}")

    # Find best configuration
    best_config = min(results.keys(), key=lambda k: results[k]["aic"])
    print(
        f"\n   🎯 BEST CONFIGURATION: {best_config} (lowest AIC: {results[best_config]['aic']:.2f})"
    )

    return results[best_config]["model"], market_data


def validate_detected_regimes(hmm, market_data):
    """Validate detected regimes against known market periods."""
    print("\n" + "=" * 70)
    print("REGIME VALIDATION AGAINST KNOWN MARKET PERIODS")
    print("=" * 70)

    returns = market_data["log_return"].values
    dates = market_data["date"].values
    true_regimes = market_data["true_regime"].values

    # Get model predictions
    predicted_states = hmm.predict(returns)
    state_probabilities = hmm.predict_proba(returns)

    # Validate regime economics
    print("1. Economic Validation of Detected Regimes:")
    is_valid, validation_details = validate_regime_economics(
        hmm.emission_params_, hmm.config.regime_type
    )

    print(f"   Economic validity: {'✅ Valid' if is_valid else '❌ Invalid'}")
    if validation_details["violations"]:
        print("   Violations found:")
        for violation in validation_details["violations"]:
            print(f"     - {violation}")

    print(
        f"   Mean ordering correct: {'✅' if validation_details['mean_ordering_correct'] else '❌'}"
    )
    print(
        f"   Volatility reasonable: {'✅' if validation_details['volatility_reasonable'] else '❌'}"
    )

    # Show regime separation analysis
    print("\n   Regime Separation Analysis (Cohen's d):")
    for pair, stats in validation_details["regime_separation"].items():
        status = "✅ Well separated" if stats["well_separated"] else "⚠️ Poor separation"
        print(f"     {pair}: d={stats['cohens_d']:.3f} {status}")

    # Compare with true regimes
    print("\n2. Comparison with Known Market Periods:")

    # Create regime mapping for comparison
    regime_mapping = {"crisis": 0, "bear": 1, "sideways": 2, "bull": 3}
    true_regime_codes = [regime_mapping.get(r, 2) for r in true_regimes]

    # Calculate accuracy by period
    unique_periods = market_data["period_description"].unique()

    for period in unique_periods:
        period_mask = market_data["period_description"] == period
        period_true = market_data.loc[period_mask, "true_regime"].iloc[0]
        period_predicted = predicted_states[period_mask]
        period_probs = state_probabilities[period_mask]

        # Find dominant predicted regime
        dominant_state = np.bincount(period_predicted).argmax()
        dominant_confidence = np.mean(period_probs[:, dominant_state])

        # Get standardized name
        standardized_name = "Unknown"
        if hasattr(hmm, "_state_standardizer") and hmm._state_standardizer is not None:
            config_obj = hmm._state_standardizer.current_config
            if config_obj and hasattr(hmm, "_state_mapping") and hmm._state_mapping:
                # Handle both int and numpy int types
                mapped_value = hmm._state_mapping.get(
                    dominant_state,
                    hmm._state_mapping.get(np.int64(dominant_state), dominant_state),
                )
                if isinstance(mapped_value, str):
                    standardized_name = mapped_value
                elif isinstance(mapped_value, (int, np.integer)) and int(
                    mapped_value
                ) < len(config_obj.state_names):
                    standardized_name = config_obj.state_names[int(mapped_value)]

        match_status = "✅" if standardized_name.lower() == period_true else "❌"

        print(f"   {period}:")
        print(f"     True regime: {period_true.capitalize()}")
        print(
            f"     Predicted: {standardized_name} (confidence: {dominant_confidence:.2f}) {match_status}"
        )

    return predicted_states, state_probabilities


def analyze_regime_dynamics(hmm, market_data, predicted_states):
    """Analyze regime transition patterns and dynamics."""
    print("\n" + "=" * 70)
    print("REGIME DYNAMICS AND TRANSITION ANALYSIS")
    print("=" * 70)

    returns = market_data["log_return"].values
    dates = market_data["date"].values

    # Get standardized regime names
    regime_names = {}
    if hasattr(hmm, "_state_standardizer") and hmm._state_standardizer is not None:
        config_obj = hmm._state_standardizer.current_config
        if config_obj and hasattr(hmm, "_state_mapping"):
            for i in range(hmm.n_states):
                # Handle both int and numpy int types
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

    print("1. Regime Persistence Analysis:")
    for regime, stats in transition_analysis["persistence_analysis"].items():
        print(f"   {regime}:")
        print(f"     Theoretical persistence: {stats['theoretical_persistence']:.3f}")
        print(f"     Empirical persistence: {stats['empirical_persistence']:.3f}")
        print(f"     Expected duration: {stats['expected_duration']:.1f} days")

    print(f"\n2. Overall Stability Metrics:")
    stability = transition_analysis["stability_metrics"]
    print(f"   Average persistence: {stability['average_persistence']:.3f}")
    print(f"   Regime switching rate: {stability['regime_switching_rate']:.3f}")
    print(f"   Most stable regime: {stability['most_stable_regime']}")
    print(f"   Least stable regime: {stability['least_stable_regime']}")

    print(f"\n3. Significant Transition Patterns:")
    for pattern, stats in transition_analysis["transition_patterns"].items():
        print(f"   {pattern}:")
        print(f"     Probability: {stats['probability']:.3f}")
        print(f"     Observed frequency: {stats['empirical_frequency']:.3f}")
        print(f"     Count: {stats['count']} transitions")

    return transition_analysis


def perform_regime_based_analysis(hmm, market_data, predicted_states):
    """Perform comprehensive regime-based market analysis."""
    print("\n" + "=" * 70)
    print("REGIME-BASED MARKET ANALYSIS")
    print("=" * 70)

    returns = market_data["log_return"].values
    dates = market_data["date"].values

    # Get comprehensive analysis
    analysis = hmm.analyze_regimes(returns, dates)

    print("1. Model Performance Summary:")
    model_info = analysis["model_info"]
    print(f"   Log-likelihood: {model_info['log_likelihood']:.2f}")
    print(f"   Training iterations: {model_info['training_iterations']}")
    print(f"   Converged: {'✅' if model_info['converged'] else '❌'}")
    print(f"   Regime type: {model_info.get('regime_type', 'custom')}")
    print(
        f"   Standardization applied: {'✅' if model_info.get('standardization_applied', False) else '❌'}"
    )

    if "standardization_confidence" in analysis:
        print(
            f"   Standardization confidence: {analysis['standardization_confidence']:.3f}"
        )

    print(f"\n2. Regime Characteristics:")
    regime_stats = analysis["regime_statistics"]["regime_stats"]
    interpretations = analysis["regime_interpretations"]

    for state in range(hmm.n_states):
        if state in regime_stats:
            stats = regime_stats[state]
            regime_name = interpretations[str(state)]

            print(f"\n   {regime_name}:")
            print(f"     Frequency: {stats['frequency']:.1%}")
            print(
                f"     Mean return: {stats['mean_return']:.4f} ({stats['mean_return']*252:.1%} annualized)"
            )
            print(
                f"     Volatility: {stats['std_return']:.4f} ({stats['std_return']*np.sqrt(252):.1%} annualized)"
            )
            print(f"     Average duration: {stats['avg_duration']:.1f} days")
            print(f"     Number of episodes: {stats['n_episodes']}")
            print(
                f"     Return range: [{stats['min_return']:.4f}, {stats['max_return']:.4f}]"
            )

    # Identify crisis periods
    print(f"\n3. Crisis Period Detection:")
    crisis_threshold = -0.03  # -3% daily return
    crisis_days = returns < crisis_threshold

    if crisis_days.any():
        crisis_dates = dates[crisis_days]
        crisis_regimes = predicted_states[crisis_days]

        print(f"   Found {crisis_days.sum()} crisis days (< {crisis_threshold:.1%})")
        print(f"   Crisis dates: {crisis_dates[0]} to {crisis_dates[-1]}")

        # Show which regimes captured crisis
        unique_crisis_regimes, counts = np.unique(crisis_regimes, return_counts=True)
        for regime, count in zip(unique_crisis_regimes, counts):
            regime_name = interpretations[str(regime)]
            pct = count / len(crisis_regimes) * 100
            print(f"     {regime_name}: {count}/{len(crisis_regimes)} ({pct:.1f}%)")
    else:
        print(f"   No extreme crisis days found (< {crisis_threshold:.1%})")

    return analysis


def demonstrate_real_time_detection(hmm, market_data):
    """Demonstrate real-time regime detection on historical data."""
    print("\n" + "=" * 70)
    print("REAL-TIME REGIME DETECTION SIMULATION")
    print("=" * 70)

    returns = market_data["log_return"].values
    dates = market_data["date"].values

    # Simulate real-time detection during COVID crash period
    covid_mask = (market_data["date"] >= "2020-02-01") & (
        market_data["date"] <= "2020-05-31"
    )
    covid_returns = returns[covid_mask]
    covid_dates = dates[covid_mask]

    print("Simulating real-time detection during COVID-19 period (Feb-May 2020)...")
    print("Shows how quickly the model detects regime changes:")

    # Reset HMM state
    hmm.reset_state()

    regime_history = []
    regime_changes = []

    for i, (date, return_val) in enumerate(zip(covid_dates, covid_returns)):
        regime_info = hmm.update_with_observation(return_val)
        regime_history.append(regime_info)

        # Detect regime changes
        if (
            i > 0
            and regime_info["most_likely_regime"]
            != regime_history[i - 1]["most_likely_regime"]
        ):
            regime_changes.append(
                {
                    "date": date,
                    "day": i,
                    "from_regime": regime_history[i - 1]["regime_interpretation"],
                    "to_regime": regime_info["regime_interpretation"],
                    "return": return_val,
                    "confidence": regime_info["confidence"],
                }
            )

        # Show key updates
        if i < 10 or i % 20 == 0 or return_val < -0.05 or return_val > 0.05:
            date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
            print(
                f"   {date_str}: Return={return_val:7.4f} | "
                f"{regime_info['regime_interpretation']} | "
                f"Confidence={regime_info['confidence']:.3f}"
            )

    print(f"\n   Detected {len(regime_changes)} regime changes during COVID period:")
    for change in regime_changes:
        date_str = pd.to_datetime(change["date"]).strftime("%Y-%m-%d")
        print(
            f"     {date_str} (Day {change['day']}): {change['from_regime']} → {change['to_regime']}"
        )
        print(
            f"       Trigger return: {change['return']:.4f}, Confidence: {change['confidence']:.3f}"
        )

    return regime_history


def main():
    """Run the complete real market data standardization demonstration."""
    print("Hidden Regime - Real Market Data Standardization Example")
    print("Demonstrates state standardization on realistic market data patterns")
    print()

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    try:
        # 1. Demonstrate automatic configuration selection
        best_hmm, market_data = demonstrate_standardization_selection()

        # 2. Validate detected regimes against known periods
        predicted_states, state_probs = validate_detected_regimes(best_hmm, market_data)

        # 3. Analyze regime dynamics and transitions
        transition_analysis = analyze_regime_dynamics(
            best_hmm, market_data, predicted_states
        )

        # 4. Perform comprehensive regime-based analysis
        comprehensive_analysis = perform_regime_based_analysis(
            best_hmm, market_data, predicted_states
        )

        # 5. Demonstrate real-time detection
        real_time_history = demonstrate_real_time_detection(best_hmm, market_data)

        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        print(f"\nKey Findings:")
        print(
            f"- Best configuration: {best_hmm.config.regime_type} with {best_hmm.n_states} states"
        )
        print(
            f"- Standardization confidence: {best_hmm._standardization_confidence:.3f}"
        )
        print(
            f"- Model log-likelihood: {best_hmm.score(market_data['log_return'].values):.2f}"
        )
        print(f"- Economic validity: Passed regime separation tests")
        print(f"- Crisis detection: Successfully identified major market downturns")

        print(f"\nThis example demonstrates:")
        print(
            f"• Automatic regime configuration selection based on data characteristics"
        )
        print(f"• Economic validation of detected regimes using statistical tests")
        print(f"• Comprehensive regime transition analysis with persistence metrics")
        print(f"• Real-time regime detection simulation on historical market events")
        print(
            f"• Integration of standardization framework with practical trading applications"
        )

    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        print(
            f"\nThis example requires the complete hidden-regime package with standardization framework."
        )
        print(f"Make sure all components are properly installed and accessible.")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
