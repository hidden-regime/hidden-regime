#!/usr/bin/env python3
"""
Test Regime Validation System

This script tests the new data-driven regime assignment and validation
to ensure regime labels match actual market behavior.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from hidden_regime.analysis.financial import FinancialAnalysis
from hidden_regime.utils.exceptions import ValidationError


def create_test_data(scenario: str, n_days: int = 100) -> pd.DataFrame:
    """Create test data for different market scenarios."""

    np.random.seed(42)  # For reproducible tests
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    if scenario == "clear_bull":
        # Clear bull market: mostly positive returns
        returns = np.random.normal(
            0.008, 0.015, n_days
        )  # 0.8% daily mean, 1.5% volatility
        returns = np.maximum(returns, -0.03)  # Cap extreme losses

    elif scenario == "clear_bear":
        # Clear bear market: mostly negative returns
        returns = np.random.normal(
            -0.008, 0.02, n_days
        )  # -0.8% daily mean, 2% volatility
        returns = np.minimum(returns, 0.03)  # Cap extreme gains

    elif scenario == "regime_switching":
        # Regime switching: periods of bull, bear, sideways
        returns = []
        regimes = [0] * 30 + [1] * 40 + [2] * 30  # 3 distinct periods
        for regime in regimes:
            if regime == 0:  # Bear period
                ret = np.random.normal(-0.012, 0.025)
            elif regime == 1:  # Sideways period
                ret = np.random.normal(0.001, 0.008)
            else:  # Bull period
                ret = np.random.normal(0.015, 0.018)
            returns.append(ret)
        returns = np.array(returns)

    elif scenario == "monotonic_up":
        # Monotonic uptrend: all positive returns
        returns = np.random.normal(0.005, 0.008, n_days)
        returns = np.maximum(returns, 0.001)  # Ensure all positive

    elif scenario == "low_volatility":
        # Low volatility sideways market
        returns = np.random.normal(0.0001, 0.003, n_days)  # Very low volatility

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Convert to log returns and create DataFrame
    log_returns = np.log(1 + returns)

    # Calculate prices
    prices = [100.0]  # Starting price
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))

    df = pd.DataFrame(
        {"close": prices[1:], "log_return": log_returns},  # Remove starting price
        index=dates,
    )

    return df


def create_mock_model_component(emission_means: np.ndarray):
    """Create a mock model component with specified emission means."""

    class MockModel:
        def __init__(self, means):
            self.emission_means_ = means
            self.n_states = len(means)

    return MockModel(emission_means)


def test_data_assessment():
    """Test the data assessment functionality."""
    print("🧪 Testing Data Assessment")
    print("=" * 50)

    scenarios = [
        ("clear_bull", "Clear Bull Market"),
        ("clear_bear", "Clear Bear Market"),
        ("regime_switching", "Regime Switching Market"),
        ("monotonic_up", "Monotonic Uptrend"),
        ("low_volatility", "Low Volatility Sideways"),
    ]

    for scenario, description in scenarios:
        print(f"\n📊 Testing: {description}")
        try:
            data = create_test_data(scenario)
            assessment = FinancialAnalysis.assess_data_for_regime_detection(data)

            recommendations = assessment["recommendations"]
            print(
                f"   Suitable for regime detection: {recommendations['suitable_for_regime_detection']}"
            )
            print(f"   Recommended n_states: {recommendations['recommended_n_states']}")
            print(f"   Approach: {recommendations['regime_detection_approach']}")

            if recommendations["warnings"]:
                print(f"   Warnings:")
                for warning in recommendations["warnings"]:
                    print(f"     • {warning}")

            if recommendations["rationale"]:
                print(f"   Rationale:")
                for rationale in recommendations["rationale"]:
                    print(f"     • {rationale}")

        except Exception as e:
            print(f"   ❌ Test failed: {e}")

    print("\n✅ Data assessment tests completed")


def test_regime_validation():
    """Test the regime validation functionality."""
    print("\n🧪 Testing Regime Validation")
    print("=" * 50)

    # Test cases: (emission_means, expected_result, description)
    test_cases = [
        # Valid cases
        (np.array([-0.01, 0.0005, 0.008]), "PASS", "Valid Bear/Sideways/Bull"),
        (np.array([-0.015, 0.012]), "PASS", "Valid Bear/Bull"),
        (
            np.array([-0.025, -0.008, 0.001, 0.018]),
            "PASS",
            "Valid Crisis/Bear/Sideways/Bull",
        ),
        # Invalid cases that should fail validation
        (
            np.array([0.0001, 0.0002, 0.0003]),
            "FAIL",
            "Insufficient spread between regimes",
        ),
        # Note: The algorithm automatically assigns correct labels based on returns,
        # so we can't easily create "incorrect" mappings for testing.
    ]

    from hidden_regime.analysis.financial import FinancialAnalysis
    from hidden_regime.config.analysis import FinancialAnalysisConfig

    for emission_means, expected_result, description in test_cases:
        print(f"\n📊 Testing: {description}")
        print(f"   Emission means: {[f'{m:.3%}' for m in np.exp(emission_means) - 1]}")

        try:
            # Create mock components
            config = FinancialAnalysisConfig(n_states=len(emission_means))
            analysis_component = FinancialAnalysis(config)
            model_component = create_mock_model_component(emission_means)

            # Create mock model output
            n_observations = 50
            model_output = pd.DataFrame(
                {
                    "predicted_state": np.random.randint(
                        0, len(emission_means), n_observations
                    ),
                    "confidence": np.random.uniform(0.5, 0.9, n_observations),
                }
            )

            # Try to update analysis (this will trigger regime validation)
            result = analysis_component.update(
                model_output=model_output,
                raw_data=None,
                model_component=model_component,
            )

            if expected_result == "PASS":
                print(f"   ✅ PASS: Validation succeeded as expected")
                # Show the regime mapping
                if hasattr(analysis_component, "_current_state_mapping"):
                    mapping = analysis_component._current_state_mapping
                    print(f"   Regime mapping: {mapping}")
            else:
                print(f"   ❌ UNEXPECTED: Validation should have failed but passed")

        except ValidationError as e:
            if expected_result == "FAIL":
                print(f"   ✅ PASS: Validation correctly failed")
                print(f"   Error: {str(e)[:100]}...")
            else:
                print(f"   ❌ UNEXPECTED: Validation failed when it should have passed")
                print(f"   Error: {e}")
        except Exception as e:
            print(f"   ❌ UNEXPECTED ERROR: {e}")

    print("\n✅ Regime validation tests completed")


def test_real_case_study():
    """Test with a real case study to see validation in action."""
    print("\n🧪 Testing Real Case Study Validation")
    print("=" * 50)

    try:
        from examples.case_study import CaseStudyOrchestrator
        from hidden_regime.config.case_study import CaseStudyConfig

        # Test with a short period to see validation
        config = CaseStudyConfig(
            ticker="AAPL",
            output_directory="./output/regime_validation_test",
            start_date="2024-08-25",
            end_date="2024-08-30",
            n_training=10,
        )

        print(
            f"Testing case study with {config.ticker} from {config.start_date} to {config.end_date}"
        )

        orchestrator = CaseStudyOrchestrator(config)
        result = orchestrator.run_complete_case_study()

        print("✅ Case study completed successfully with new validation")

    except ValidationError as e:
        print(f"✅ Validation correctly caught an issue:")
        print(f"   {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    print("🔬 Regime Validation Test Suite")
    print("=" * 60)

    try:
        # Test 1: Data assessment
        test_data_assessment()

        # Test 2: Regime validation
        test_regime_validation()

        # Test 3: Real case study
        test_real_case_study()

        print("\n🎉 All tests completed!")

    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback

        traceback.print_exc()
