"""
Simple State Standardization Demo for hidden-regime package.

Basic demonstration of the standardization framework without complex features.
"""

import sys
from pathlib import Path

import numpy as np

# Add hidden-regime to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the hidden-regime package
from hidden_regime import HiddenMarkovModel, HMMConfig
from hidden_regime.models.utils import validate_regime_economics


def create_test_data():
    """Create simple test data with clear regime structure."""
    print("Creating test data with clear regime structure...")

    np.random.seed(42)

    # Bear market: negative mean, high volatility
    bear_data = np.random.normal(-0.02, 0.03, 200)

    # Sideways market: near zero mean, low volatility
    sideways_data = np.random.normal(0.001, 0.015, 300)

    # Bull market: positive mean, moderate volatility
    bull_data = np.random.normal(0.015, 0.025, 200)

    # Combine data
    returns = np.concatenate([bear_data, sideways_data, bull_data])

    print(f"   Generated {len(returns)} observations")
    print(f"   Overall mean: {np.mean(returns):.4f}")
    print(f"   Overall std: {np.std(returns):.4f}")

    return returns


def test_3state_standardization():
    """Test 3-state standardization."""
    print("\n" + "=" * 50)
    print("3-STATE STANDARDIZATION TEST")
    print("=" * 50)

    returns = create_test_data()

    # Create standardized 3-state config
    config = HMMConfig.for_standardized_regimes("3_state", conservative=False)

    print(f"Configuration created:")
    print(f"  Regime type: {config.regime_type}")
    print(f"  Number of states: {config.n_states}")
    print(f"  Force ordering: {config.force_state_ordering}")

    # Create and train model
    print(f"\nTraining HMM...")
    hmm = HiddenMarkovModel(config=config)
    hmm.fit(returns, verbose=False, max_iterations=50)  # Limit iterations for speed

    print(f"  Converged: {hmm.training_history_['converged']}")
    print(f"  Iterations: {hmm.training_history_['iterations']}")
    print(f"  Log-likelihood: {hmm.training_history_['final_log_likelihood']:.2f}")

    # Check standardization
    if hasattr(hmm, "_state_standardizer") and hmm._state_standardizer:
        print(f"\n✓ Standardization applied successfully")
        print(f"  Confidence: {hmm._standardization_confidence:.3f}")

        if hasattr(hmm, "_state_mapping") and hmm._state_mapping:
            print(f"  State mapping: {dict(hmm._state_mapping)}")

        # Show regime parameters
        print(f"\nDetected Regime Parameters:")
        config_obj = hmm._state_standardizer.current_config
        for i in range(hmm.n_states):
            mean, std = hmm.emission_params_[i]

            # Get regime name
            regime_name = f"State {i}"
            if config_obj and hasattr(hmm, "_state_mapping"):
                mapped_value = hmm._state_mapping.get(
                    i, hmm._state_mapping.get(np.int64(i), i)
                )
                if isinstance(mapped_value, str):
                    regime_name = mapped_value
                elif isinstance(mapped_value, (int, np.integer)) and int(
                    mapped_value
                ) < len(config_obj.state_names):
                    regime_name = config_obj.state_names[int(mapped_value)]

            annualized_return = mean * 252
            annualized_vol = std * np.sqrt(252)
            print(
                f"  {regime_name}: μ={mean:.4f} ({annualized_return:.1%}/year), σ={std:.4f} ({annualized_vol:.1%}/year)"
            )

    else:
        print("⚠️ Standardization not applied")

    return hmm


def test_economic_validation(hmm):
    """Test economic validation of regimes."""
    print("\n" + "=" * 50)
    print("ECONOMIC VALIDATION TEST")
    print("=" * 50)

    # Validate regime economics
    is_valid, details = validate_regime_economics(
        hmm.emission_params_, hmm.config.regime_type
    )

    print(f"Economic validity: {'✅ VALID' if is_valid else '❌ INVALID'}")
    print(
        f"Mean ordering correct: {'✅' if details['mean_ordering_correct'] else '❌'}"
    )
    print(
        f"Volatility reasonable: {'✅' if details['volatility_reasonable'] else '❌'}"
    )

    if details["violations"]:
        print(f"Violations found:")
        for violation in details["violations"]:
            print(f"  - {violation}")

    print(f"\nRegime Separation Analysis:")
    for pair, stats in details["regime_separation"].items():
        status = "✅ Well separated" if stats["well_separated"] else "⚠️ Poor separation"
        print(f"  {pair}: Cohen's d = {stats['cohens_d']:.3f} {status}")


def test_regime_inference(hmm, returns):
    """Test regime inference capabilities."""
    print("\n" + "=" * 50)
    print("REGIME INFERENCE TEST")
    print("=" * 50)

    # Get predictions
    states = hmm.predict(returns)
    probs = hmm.predict_proba(returns)

    print(f"Regime Distribution:")
    unique_states, counts = np.unique(states, return_counts=True)
    for state, count in zip(unique_states, counts):
        pct = count / len(states) * 100
        print(f"  State {state}: {count} observations ({pct:.1f}%)")

    # Show some example predictions
    print(f"\nExample Predictions (first 10 observations):")
    for i in range(min(10, len(returns))):
        return_val = returns[i]
        state = states[i]
        confidence = probs[i, state]
        print(
            f"  Day {i+1}: return={return_val:7.4f} → State {state} (conf: {confidence:.3f})"
        )

    # Find extreme returns and their classifications
    print(f"\nExtreme Return Classifications:")

    # Most negative returns
    neg_idx = np.argpartition(returns, 5)[:5]  # 5 most negative
    print("  Most negative returns:")
    for idx in neg_idx:
        return_val = returns[idx]
        state = states[idx]
        confidence = probs[idx, state]
        print(
            f"    Day {idx+1}: {return_val:.4f} → State {state} (conf: {confidence:.3f})"
        )

    # Most positive returns
    pos_idx = np.argpartition(returns, -5)[-5:]  # 5 most positive
    print("  Most positive returns:")
    for idx in pos_idx:
        return_val = returns[idx]
        state = states[idx]
        confidence = probs[idx, state]
        print(
            f"    Day {idx+1}: {return_val:.4f} → State {state} (conf: {confidence:.3f})"
        )


def main():
    """Run the simple standardization demonstration."""
    print("Hidden Regime - Simple Standardization Demo")
    print("Basic demonstration of state standardization framework")

    try:
        # Test 3-state standardization
        hmm = test_3state_standardization()

        # Test economic validation
        test_economic_validation(hmm)

        # Test regime inference
        returns = create_test_data()
        test_regime_inference(hmm, returns)

        print("\n" + "=" * 50)
        print("DEMO COMPLETED SUCCESSFULLY! ✅")
        print("=" * 50)

        print(f"\nSummary:")
        print(f"- Successfully trained {hmm.n_states}-state HMM")
        print(f"- Standardization confidence: {hmm._standardization_confidence:.3f}")
        print(f"- Model converged in {hmm.training_history_['iterations']} iterations")
        print(f"- Economic validation passed")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
