"""
Edge Case Handling - Robustness and Failure Recovery

This example demonstrates how to handle edge cases and failure modes that
occur with real-world data. Essential for production-grade regime detection.

Purpose: Show how to detect, handle, and recover from common edge cases

Real-World Scenarios Covered:
1. Small datasets (< 100 observations)
2. Singular covariance matrices (zero-variance features)
3. Extreme regime transitions (gaps larger than normal)
4. Insufficient data for multivariate model (fallback to univariate)
5. Data quality issues (gaps, outliers, NaN values)

This example uses synthetically-created problematic data to demonstrate
detection and recovery strategies.

Key Learning:
- Robustness requires explicit edge case handling
- Validation thresholds catch problems early
- Graceful degradation (fallback to simpler model) maintains functionality
- Monitoring enables proactive failure detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings

import hidden_regime as hr


class EdgeCaseValidator:
    """Validates data and configuration before model training."""

    @staticmethod
    def validate_data_quality(df, min_observations=100):
        """Check data quality and return diagnostic report."""
        issues = []
        warnings_list = []

        # Check minimum observations
        if len(df) < min_observations:
            issues.append(f"Insufficient data: {len(df)} observations < {min_observations} minimum")

        # Check for NaN values
        nan_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
        if nan_pct > 0:
            warnings_list.append(f"Data contains {nan_pct:.2f}% NaN values")

        # Check for zero variance
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].std() < 1e-10:
                issues.append(f"Zero variance in column '{col}'")

        # Check for extreme outliers (> 5 standard deviations)
        for col in df.select_dtypes(include=[np.number]).columns:
            outlier_pct = ((np.abs(df[col] - df[col].mean()) > 5 * df[col].std()).sum() / len(df)) * 100
            if outlier_pct > 5:
                warnings_list.append(f"Column '{col}' has {outlier_pct:.1f}% extreme outliers")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings_list,
            'n_observations': len(df),
            'nan_percentage': nan_pct
        }

    @staticmethod
    def validate_covariance_matrix(cov_matrix):
        """Check covariance matrix for numerical issues."""
        issues = []

        # Check condition number (ill-conditioning)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        condition_number = eigenvalues[-1] / (eigenvalues[0] + 1e-10)

        if condition_number > 1e6:
            issues.append(f"Ill-conditioned matrix (condition number: {condition_number:.2e})")

        # Check for negative eigenvalues
        if np.any(eigenvalues < -1e-10):
            issues.append("Negative eigenvalues detected (matrix not positive definite)")

        return {
            'condition_number': condition_number,
            'min_eigenvalue': eigenvalues[0],
            'max_eigenvalue': eigenvalues[-1],
            'issues': issues,
            'valid': len(issues) == 0
        }


def create_problematic_data(case):
    """Create synthetic data with specific edge case."""
    dates = pd.date_range('2023-01-01', periods=250, freq='D')

    if case == 'small_dataset':
        # Only 30 observations
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        returns = np.random.normal(0.0005, 0.015, len(dates))

    elif case == 'zero_variance':
        # One feature has zero variance
        returns = np.random.normal(0.0005, 0.015, len(dates))
        zero_feature = np.zeros(len(dates))
        return pd.DataFrame({
            'log_return': returns,
            'zero_feature': zero_feature
        }, index=dates)

    elif case == 'extreme_outliers':
        # Insert some extreme outliers
        returns = np.random.normal(0.0005, 0.015, len(dates))
        outlier_idx = np.random.choice(len(dates), 5, replace=False)
        returns[outlier_idx] = returns[outlier_idx] * 10

    elif case == 'high_correlation':
        # Two highly correlated features
        returns = np.random.normal(0.0005, 0.015, len(dates))
        almost_identical = returns + np.random.normal(0, 0.001, len(dates))
        return pd.DataFrame({
            'log_return': returns,
            'almost_identical': almost_identical
        }, index=dates)

    elif case == 'missing_data':
        # Insert NaN values
        returns = np.random.normal(0.0005, 0.015, len(dates))
        nan_idx = np.random.choice(len(dates), 20, replace=False)
        returns[nan_idx] = np.nan

    else:
        # Normal case
        returns = np.random.normal(0.0005, 0.015, len(dates))

    return pd.DataFrame({'log_return': returns}, index=dates)


def main():
    print("=" * 80)
    print("EDGE CASE HANDLING - ROBUSTNESS AND FAILURE RECOVERY")
    print("=" * 80)
    print("""
This example demonstrates detecting and handling edge cases that commonly
occur with real market data. Production systems must handle these gracefully.

Edge Cases Covered:
1. Small datasets (insufficient data for reliable HMM)
2. Zero variance features (can't estimate distribution)
3. Extreme outliers (violate Gaussian assumption)
4. High correlation (features are redundant)
5. Missing data (gaps or NaN values)

Strategy: Validate → Detect → Recover
- Validation detects issues before training
- Detection monitors during training
- Recovery uses fallback strategies
""")

    print("\n" + "=" * 80)
    print("EDGE CASE 1: SMALL DATASET (30 observations)")
    print("=" * 80)

    data_small = create_problematic_data('small_dataset')
    validator = EdgeCaseValidator()
    validation = validator.validate_data_quality(data_small, min_observations=100)

    print(f"\nValidation Result:")
    print(f"  Valid: {validation['valid']}")
    if validation['issues']:
        print(f"  Issues:")
        for issue in validation['issues']:
            print(f"    - {issue}")

    print(f"\nRecommended Action:")
    print(f"  Use univariate HMM (fewer parameters needed)")
    print(f"  Or: Collect more data (need minimum 100 observations)")
    print(f"  Or: Use transfer learning from similar asset")

    print("\n" + "=" * 80)
    print("EDGE CASE 2: ZERO VARIANCE FEATURE")
    print("=" * 80)

    data_zero_var = create_problematic_data('zero_variance')
    validation = validator.validate_data_quality(data_zero_var)

    print(f"\nValidation Result:")
    print(f"  Valid: {validation['valid']}")
    if validation['issues']:
        print(f"  Issues:")
        for issue in validation['issues']:
            print(f"    - {issue}")

    print(f"\nRecommended Action:")
    print(f"  Remove zero-variance feature (no information)")
    print(f"  Use only: log_return")

    print("\n" + "=" * 80)
    print("EDGE CASE 3: EXTREME OUTLIERS")
    print("=" * 80)

    data_outliers = create_problematic_data('extreme_outliers')
    validation = validator.validate_data_quality(data_outliers)

    print(f"\nValidation Result:")
    print(f"  Valid: {validation['valid']}")
    if validation['warnings']:
        print(f"  Warnings:")
        for warning in validation['warnings']:
            print(f"    - {warning}")

    print(f"\nRecommended Action:")
    print(f"  Investigate source of outliers (data errors or market events?)")
    print(f"  Option 1: Remove outliers (if data errors)")
    print(f"  Option 2: Keep outliers (if legitimate market events)")
    print(f"  Option 3: Use robust scaling (Winsorize extreme values)")

    # Show statistics
    returns = data_outliers['log_return'].dropna()
    outlier_mask = np.abs(returns - returns.mean()) > 5 * returns.std()
    print(f"\nOutlier Statistics:")
    print(f"  Outlier count: {outlier_mask.sum()}")
    print(f"  Normal return mean: {returns[~outlier_mask].mean():.6f}")
    print(f"  Normal return std: {returns[~outlier_mask].std():.6f}")
    print(f"  Outlier return mean: {returns[outlier_mask].mean():.6f}")

    print("\n" + "=" * 80)
    print("EDGE CASE 4: HIGHLY CORRELATED FEATURES")
    print("=" * 80)

    data_corr = create_problematic_data('high_correlation')
    validation = validator.validate_data_quality(data_corr)

    # Compute correlation
    correlation = data_corr['log_return'].corr(data_corr['almost_identical'])

    print(f"\nValidation Result:")
    print(f"  Valid: {validation['valid']}")

    print(f"\nFeature Correlation:")
    print(f"  Pearson correlation: {correlation:.4f}")

    print(f"\nInterpretation:")
    if correlation > 0.95:
        print(f"  REDUNDANT: Features contain same information")
    elif correlation > 0.7:
        print(f"  HIGHLY CORRELATED: Features mostly redundant")
    else:
        print(f"  INDEPENDENT: Features provide different information")

    print(f"\nRecommended Action:")
    if correlation > 0.9:
        print(f"  Remove 'almost_identical' feature (redundant)")
        print(f"  Use only: 'log_return'")
    else:
        print(f"  Both features acceptable for multivariate model")

    print("\n" + "=" * 80)
    print("EDGE CASE 5: MISSING DATA (NaN VALUES)")
    print("=" * 80)

    data_missing = create_problematic_data('missing_data')
    validation = validator.validate_data_quality(data_missing)

    nan_count = data_missing['log_return'].isna().sum()

    print(f"\nValidation Result:")
    print(f"  Valid: {validation['valid']}")
    if validation['warnings']:
        print(f"  Warnings:")
        for warning in validation['warnings']:
            print(f"    - {warning}")

    print(f"\nMissing Data Statistics:")
    print(f"  NaN count: {nan_count} out of {len(data_missing)}")
    print(f"  Missing percentage: {(nan_count / len(data_missing)) * 100:.1f}%")

    print(f"\nRecommended Action:")
    if nan_count / len(data_missing) < 0.05:
        print(f"  Use forward-fill: impute with previous value")
        print(f"  Data quality sufficient after imputation")
    elif nan_count / len(data_missing) < 0.20:
        print(f"  Use forward-fill with limit")
        print(f"  Or: Remove gaps > 3 days (data quality issue)")
    else:
        print(f"  Data has too many missing values")
        print(f"  Consider: different data source or date range")

    # Show gap analysis
    data_filled = data_missing.fillna(method='ffill')
    print(f"\nAfter Forward-Fill:")
    print(f"  Remaining NaN: {data_filled.isna().sum().sum()}")

    # Step 2: Practical validation workflow
    print("\n" + "=" * 80)
    print("VALIDATION WORKFLOW FOR REAL DATA")
    print("=" * 80)

    print("""
Use this checklist before training any multivariate HMM:

1. DATA SIZE
   ✓ At least 100 observations (prefer 500+)
   ✓ At least 1 year of data recommended

2. DATA QUALITY
   ✓ Missing data < 5%
   ✓ No zero-variance features
   ✓ Outliers investigated and documented

3. FEATURE CORRELATION
   ✓ Feature pairs correlation < 0.95
   ✓ At least two features with low correlation

4. COVARIANCE MATRIX
   ✓ Condition number < 1e4
   ✓ All eigenvalues positive
   ✓ Matrix is positive definite

5. FALLBACK STRATEGY
   ✓ If validation fails → downgrade to univariate
   ✓ If only 1-2 features valid → use univariate
   ✓ If insufficient data → use simpler HMM (2 states, fewer iterations)

6. MONITORING
   ✓ Track convergence metrics during training
   ✓ Monitor confidence scores after inference
   ✓ Alert if confidence drops below 0.5 (model uncertainty)
   ✓ Log covariance condition number for numerical stability
""")

    # Step 3: Failure mode demonstration
    print("\n" + "=" * 80)
    print("FAILURE MODE MATRIX")
    print("=" * 80)

    failure_modes = [
        {
            'issue': 'Too few observations',
            'symptom': 'Training does not converge',
            'detection': 'max_iterations reached without convergence',
            'recovery': 'Use univariate model or collect more data',
            'threshold': 'n_obs < 100'
        },
        {
            'issue': 'Zero variance',
            'symptom': 'Singular covariance matrix',
            'detection': 'Determinant = 0 or NaN eigenvalues',
            'recovery': 'Remove zero-variance features',
            'threshold': 'std < 1e-10'
        },
        {
            'issue': 'Extreme outliers',
            'symptom': 'Poor convergence, high log-likelihood variance',
            'detection': 'Values > 5 std from mean',
            'recovery': 'Winsorize or remove outliers',
            'threshold': '> 5% outliers'
        },
        {
            'issue': 'High correlation',
            'symptom': 'Unstable parameter estimates',
            'detection': 'Correlation > 0.95',
            'recovery': 'Remove redundant features',
            'threshold': 'corr > 0.95'
        },
        {
            'issue': 'Missing data',
            'symptom': 'Reduced effective sample size',
            'detection': 'NaN in calculations',
            'recovery': 'Forward-fill gaps < 3 days',
            'threshold': '> 5% missing'
        },
    ]

    table_data = [
        [fm['issue'], fm['symptom'], fm['detection'], fm['recovery']]
        for fm in failure_modes
    ]

    headers = ['Issue', 'Symptom', 'Detection', 'Recovery']
    print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))

    # Step 4: Production recommendations
    print("\n" + "=" * 80)
    print("PRODUCTION RECOMMENDATIONS")
    print("=" * 80)
    print("""
LEVEL 1: Minimal Safeguards
├─ Validate data size (n > 100)
├─ Check for NaN values
└─ Compute feature correlation

LEVEL 2: Moderate Safeguards (Recommended)
├─ All of Level 1, plus:
├─ Check zero-variance features
├─ Monitor convergence (log every 10 iterations)
├─ Track eigenvalue ratio (covariance concentration)
└─ Validate condition number < 1e4

LEVEL 3: Comprehensive Safeguards (Enterprise)
├─ All of Level 2, plus:
├─ Outlier detection (Tukey fences or 5-sigma rule)
├─ Covariance matrix regularization (add ridge term)
├─ Cross-validation on hold-out test period
├─ Monitoring dashboard with alerts
├─ Automatic model retraining on data drift
└─ Confidence score thresholding for trade execution

IMPLEMENTATION:
1. Create EdgeCaseValidator class (provided above)
2. Call validator.validate_data_quality() before training
3. Handle failures with explicit fallbacks:
   - If multivariate fails → try univariate
   - If univariate fails → return default regime (neutral)
4. Log all validation checks and results
5. Monitor model performance on holdout test data
""")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Integrate EdgeCaseValidator into your pipeline
2. Define your own validation thresholds based on use case
3. Implement monitoring dashboard for production systems
4. See notebook 06 for stress testing and detailed failure modes
5. See CLAUDE.md for architecture and configuration management
""")


if __name__ == '__main__':
    main()
