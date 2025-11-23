"""
Quick validation script for RegimeStabilityMetrics.
Tests that the class works correctly before running full test suite.
"""

import sys
import pandas as pd
import numpy as np

# Add package to path
sys.path.insert(0, '/home/user/refactor-hidden-regime')

from hidden_regime.analysis.stability import RegimeStabilityMetrics


def test_basic_functionality():
    """Test basic RegimeStabilityMetrics functionality."""
    print("=" * 60)
    print("Testing RegimeStabilityMetrics Basic Functionality")
    print("=" * 60)
    print()

    # Create test data with stable regimes
    data = pd.DataFrame({
        'predicted_state': [0]*30 + [1]*35 + [2]*30,
        'confidence': [0.85 + np.random.randn() * 0.05 for _ in range(95)]
    })

    print("Test 1: Initialization")
    try:
        metrics = RegimeStabilityMetrics(data)
        print("✅ PASS: Initialized successfully")
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False

    print()
    print("Test 2: Get Metrics")
    try:
        result = metrics.get_metrics()
        print("✅ PASS: Metrics computed")
        print(f"  Mean duration: {result['mean_duration']:.1f}")
        print(f"  Persistence: {result['persistence']:.3f}")
        print(f"  Stability score: {result['stability_score']:.3f}")
        print(f"  Quality rating: {result['quality_rating']}")
        print(f"  Tradeable: {result['is_tradeable']}")
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("Test 3: Detailed Report")
    try:
        report = metrics.get_detailed_report()
        print("✅ PASS: Report generated")
        print()
        print(report)
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("Test 4: Validation for Trading")
    try:
        is_valid, failures = metrics.validate_for_trading()
        print(f"✅ PASS: Validation completed")
        print(f"  Valid: {is_valid}")
        if failures:
            print(f"  Failures: {failures}")
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_noisy_regimes():
    """Test with noisy, unstable regimes."""
    print()
    print("=" * 60)
    print("Testing with Noisy Regimes (Should be Poor Quality)")
    print("=" * 60)
    print()

    # Create noisy data
    data = pd.DataFrame({
        'predicted_state': [i % 3 for i in range(100)],  # Switches every observation
        'confidence': [0.5 + np.random.randn() * 0.2 for _ in range(100)]
    })

    try:
        metrics = RegimeStabilityMetrics(data)
        result = metrics.get_metrics()

        print(f"Mean duration: {result['mean_duration']:.1f}")
        print(f"Persistence: {result['persistence']:.3f}")
        print(f"Stability score: {result['stability_score']:.3f}")
        print(f"Quality rating: {result['quality_rating']}")
        print(f"Tradeable: {result['is_tradeable']}")

        # Verify it correctly identifies poor quality
        assert result['quality_rating'] == 'Poor', "Should be Poor quality"
        assert result['is_tradeable'] is False, "Should not be tradeable"
        print()
        print("✅ PASS: Correctly identified poor quality regimes")

    except AssertionError as e:
        print(f"❌ FAIL: {e}")
        return False
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_excellent_regimes():
    """Test with excellent quality regimes."""
    print()
    print("=" * 60)
    print("Testing with Excellent Regimes (Should be Excellent Quality)")
    print("=" * 60)
    print()

    # Create excellent regimes
    data = pd.DataFrame({
        'predicted_state': [0]*50 + [1]*55 + [2]*50,  # Long, stable regimes
        'confidence': [0.90 + np.random.randn() * 0.02 for _ in range(155)]
    })

    try:
        metrics = RegimeStabilityMetrics(data)
        result = metrics.get_metrics()

        print(f"Mean duration: {result['mean_duration']:.1f}")
        print(f"Persistence: {result['persistence']:.3f}")
        print(f"Stability score: {result['stability_score']:.3f}")
        print(f"Quality rating: {result['quality_rating']}")
        print(f"Tradeable: {result['is_tradeable']}")

        # Verify it correctly identifies excellent quality
        assert result['quality_rating'] in ['Excellent', 'Good'], "Should be Excellent or Good quality"
        assert result['is_tradeable'] is True, "Should be tradeable"
        print()
        print("✅ PASS: Correctly identified excellent quality regimes")

    except AssertionError as e:
        print(f"❌ FAIL: {e}")
        return False
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("REGIME STABILITY METRICS VALIDATION")
    print()

    success = True
    success = test_basic_functionality() and success
    success = test_noisy_regimes() and success
    success = test_excellent_regimes() and success

    print()
    print("=" * 60)
    if success:
        print("✅ ALL TESTS PASSED")
        print("RegimeStabilityMetrics is working correctly!")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the errors above")
    print("=" * 60)

    sys.exit(0 if success else 1)
