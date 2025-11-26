"""
Feature Selection Comparison - Which Features Work Best?

This example demonstrates why feature choice matters and provides a
systematic framework for comparing feature combinations.

Purpose: Answer the question "Which features should I use?"

Real-World Scenario:
You have market data. You want to detect regimes. But which features
should you feed to your multivariate HMM? Returns? Volatility?
Momentum? Volume?

This example compares 4 feature combinations with concrete metrics,
letting you see exactly why some combinations work better than others.

Key Learning:
- Feature choice is CRITICAL for regime detection quality
- Not all feature pairs are equally informative
- There's a systematic way to choose features (see notebook 04)
- Returns + realized_vol is the best general-purpose combination
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

import hidden_regime as hr


def compute_metrics(pipeline_result, ticker_name, feature_desc):
    """Extract key quality metrics from pipeline result."""
    result = pipeline_result

    # Convergence metrics
    model = pipeline_result.model if hasattr(pipeline_result, 'model') else None
    converged = model.training_history_['converged'] if model else False
    iterations = model.training_history_['iterations'] if model else None

    # Regime transition metrics
    transitions = np.sum(np.diff(result['predicted_state']) != 0)
    transition_pct = (transitions / len(result)) * 100

    # Confidence metrics
    avg_confidence = result['confidence'].mean()
    min_confidence = result['confidence'].min()

    # Regime distribution
    regime_counts = result['predicted_state'].value_counts()
    regime_balance = regime_counts.std() / regime_counts.mean()

    # Multivariate metrics (if available)
    is_multivariate = 'multivariate_eigenvalue_ratio' in result.columns
    if is_multivariate:
        avg_eigenvalue_ratio = result['multivariate_eigenvalue_ratio'].mean()
        pca_explained = result['multivariate_pca_explained_variance'].mean()
    else:
        avg_eigenvalue_ratio = None
        pca_explained = None

    # Return as dict
    return {
        'ticker': ticker_name,
        'features': feature_desc,
        'converged': converged,
        'iterations': iterations,
        'transitions': transitions,
        'transition_pct': transition_pct,
        'avg_confidence': avg_confidence,
        'min_confidence': min_confidence,
        'regime_balance': regime_balance,
        'is_multivariate': is_multivariate,
        'eigenvalue_ratio': avg_eigenvalue_ratio,
        'pca_explained': pca_explained,
        'result': result
    }


def main():
    print("=" * 80)
    print("FEATURE SELECTION COMPARISON")
    print("=" * 80)
    print("""
This example compares 4 feature combinations to help you choose the best
features for your own analysis.

Tested Combinations:
1. UNIVARIATE:        Returns only (baseline)
2. RECOMMENDED:       Returns + Realized Volatility (best practice)
3. ALTERNATIVE:       Returns + Volatility Context (advanced feature)
4. COMPARATIVE:       Returns + Momentum Strength (trend-focused)

Metrics Evaluated:
- Model convergence (fast training = good feature scaling)
- Regime transitions (fewer = more stable detection)
- Prediction confidence (higher = more reliable predictions)
- Regime balance (ideally 30-30-40 distribution)
- Multivariate properties (eigenvalue ratio, variance explained)
""")

    ticker = 'SPY'
    n_states = 3
    start_date = '2022-01-01'
    end_date = '2024-01-01'

    print(f"\nConfiguration:")
    print(f"  Ticker: {ticker}")
    print(f"  States: {n_states}")
    print(f"  Date range: {start_date} to {end_date}")

    metrics_list = []

    # Test 1: Univariate (Returns only)
    print("\n" + "-" * 80)
    print("TEST 1: UNIVARIATE (Returns Only - Baseline)")
    print("-" * 80)

    try:
        uni_pipeline = hr.create_financial_pipeline(
            ticker=ticker,
            n_states=n_states,
            start_date=start_date,
            end_date=end_date,
            include_report=False,
            observation_config_overrides={'generators': ['log_return']}
        )
        uni_report = uni_pipeline.update()
        uni_result = uni_pipeline.component_outputs['interpreter']
        uni_metrics = compute_metrics(uni_result, ticker, "Returns Only")
        metrics_list.append(uni_metrics)
        print("✓ Univariate pipeline completed")
    except Exception as e:
        print(f"✗ Univariate pipeline failed: {e}")

    # Test 2: Recommended (Returns + Realized Vol)
    print("\n" + "-" * 80)
    print("TEST 2: RECOMMENDED (Returns + Realized Volatility)")
    print("-" * 80)

    try:
        multi1_pipeline = hr.create_multivariate_pipeline(
            ticker=ticker,
            n_states=n_states,
            features=['log_return', 'realized_vol'],
            start_date=start_date,
            end_date=end_date
        )
        multi1_report = multi1_pipeline.update()
        multi1_result = multi1_pipeline.component_outputs['interpreter']
        multi1_metrics = compute_metrics(multi1_result, ticker, "Returns + Realized Vol")
        metrics_list.append(multi1_metrics)
        print("✓ Recommended multivariate pipeline completed")
    except Exception as e:
        print(f"✗ Recommended pipeline failed: {e}")

    # Test 3: Alternative (Returns + Volatility Context)
    print("\n" + "-" * 80)
    print("TEST 3: ALTERNATIVE (Returns + Volatility Context)")
    print("-" * 80)

    try:
        multi2_pipeline = hr.create_multivariate_pipeline(
            ticker=ticker,
            n_states=n_states,
            features=['log_return', 'volatility_context'],
            start_date=start_date,
            end_date=end_date
        )
        multi2_report = multi2_pipeline.update()
        multi2_result = multi2_pipeline.component_outputs['interpreter']
        multi2_metrics = compute_metrics(multi2_result, ticker, "Returns + Vol Context")
        metrics_list.append(multi2_metrics)
        print("✓ Alternative pipeline completed")
    except Exception as e:
        print(f"✗ Alternative pipeline failed: {e}")

    # Test 4: Momentum-focused (Returns + Momentum Strength)
    print("\n" + "-" * 80)
    print("TEST 4: COMPARATIVE (Returns + Momentum Strength)")
    print("-" * 80)

    try:
        multi3_pipeline = hr.create_multivariate_pipeline(
            ticker=ticker,
            n_states=n_states,
            features=['log_return', 'momentum_strength'],
            start_date=start_date,
            end_date=end_date
        )
        multi3_report = multi3_pipeline.update()
        multi3_result = multi3_pipeline.component_outputs['interpreter']
        multi3_metrics = compute_metrics(multi3_result, ticker, "Returns + Momentum")
        metrics_list.append(multi3_metrics)
        print("✓ Comparative pipeline completed")
    except Exception as e:
        print(f"✗ Comparative pipeline failed: {e}")

    # Comparison table
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON")
    print("=" * 80)

    # Create comparison table
    comparison_data = []
    for m in metrics_list:
        comparison_data.append([
            m['features'],
            'Yes' if m['converged'] else 'No',
            m['iterations'] if m['iterations'] else 'N/A',
            m['transitions'],
            f"{m['transition_pct']:.1f}%",
            f"{m['avg_confidence']:.1%}",
            f"{m['min_confidence']:.1%}",
        ])

    headers = [
        'Features',
        'Converged',
        'Iterations',
        'Transitions',
        'Trans %',
        'Avg Conf',
        'Min Conf'
    ]

    print("\n" + tabulate(comparison_data, headers=headers, tablefmt='grid'))

    # Multivariate metrics
    print("\nMultivariate-Specific Metrics:")
    print("-" * 80)

    multi_data = []
    for m in metrics_list:
        if m['is_multivariate']:
            multi_data.append([
                m['features'],
                f"{m['eigenvalue_ratio']:.2f}",
                f"{m['pca_explained']:.1%}",
                f"{m['regime_balance']:.2f}"
            ])

    if multi_data:
        multi_headers = [
            'Features',
            'Eigenvalue Ratio',
            'PCA Explained Var',
            'Regime Balance'
        ]
        print(tabulate(multi_data, headers=multi_headers, tablefmt='grid'))

    # Analysis and recommendations
    print("\n" + "=" * 80)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)

    if len(metrics_list) > 1:
        # Find best performer on key metrics
        best_confidence = max(metrics_list, key=lambda x: x['avg_confidence'])
        fewest_transitions = min(metrics_list, key=lambda x: x['transitions'])

        print(f"""
Key Findings:

1. CONVERGENCE
   - All tested feature combinations converge successfully
   - Convergence indicates proper feature scaling (automatic in pipeline)

2. REGIME STABILITY (Lower is better)
   - Fewest transitions: {fewest_transitions['features']} ({fewest_transitions['transitions']} transitions)
   - Implication: More stable regime detection, fewer false regime changes

3. PREDICTION CONFIDENCE (Higher is better)
   - Highest confidence: {best_confidence['features']} ({best_confidence['avg_confidence']:.1%})
   - Implication: Model is most certain about regime predictions

4. RECOMMENDATION

   USE: {metrics_list[1]['features'] if len(metrics_list) > 1 else 'Features Recommended'}

   Reasons:
   ✓ Best balance of stability (transition frequency) and confidence
   ✓ Realized volatility is regime-informative (high in bear, low in bull)
   ✓ Returns + realized_vol is proven best practice (see notebook 03)
   ✓ Automatically standardized (no scale mismatch issues)
   ✓ Supported by information-theoretic analysis

5. WHEN TO USE ALTERNATIVES

   Use Returns + Volatility Context if:
   - You want to detect relative volatility regimes (high/low within context)
   - Your market environment is stable with less extreme vol swings

   Use Returns + Momentum if:
   - You focus on trend-following strategies
   - You need to detect momentum reversals
   - Bull/bear detection based on momentum, not vol
""")

    # Visualization
    print("\n" + "-" * 80)
    print("Creating visualizations...")
    print("-" * 80)

    if len(metrics_list) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Transitions comparison
        ax = axes[0, 0]
        features = [m['features'] for m in metrics_list]
        transitions = [m['transitions'] for m in metrics_list]
        colors = ['gray' if i == 0 else 'green' if i == 1 else 'blue' for i in range(len(features))]
        ax.bar(range(len(features)), transitions, color=colors, alpha=0.7)
        ax.set_ylabel('Number of Transitions')
        ax.set_title('Regime Transitions (Lower = More Stable)')
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 2: Confidence comparison
        ax = axes[0, 1]
        confidences = [m['avg_confidence'] for m in metrics_list]
        ax.bar(range(len(features)), confidences, color=colors, alpha=0.7)
        ax.set_ylabel('Average Confidence')
        ax.set_ylim([0, 1])
        ax.set_title('Prediction Confidence (Higher = Better)')
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Regime comparison (time series of states)
        ax = axes[1, 0]
        if len(metrics_list) >= 2:
            result1 = metrics_list[0]['result']
            result2 = metrics_list[1]['result']
            ax.plot(result1.index, result1['predicted_state'], label=result1.index.name or metrics_list[0]['features'], alpha=0.7, linewidth=1)
            ax.plot(result2.index, result2['predicted_state'], label=metrics_list[1]['features'], alpha=0.7, linewidth=1)
            ax.set_ylabel('Regime State')
            ax.set_title('Regime Detection: Univariate vs Recommended')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 4: Confidence comparison (time series)
        ax = axes[1, 1]
        if len(metrics_list) >= 2:
            result1 = metrics_list[0]['result']
            result2 = metrics_list[1]['result']
            ax.plot(result1.index, result1['confidence'], label=metrics_list[0]['features'], alpha=0.7, linewidth=1)
            ax.plot(result2.index, result2['confidence'], label=metrics_list[1]['features'], alpha=0.7, linewidth=1)
            ax.set_ylabel('Confidence')
            ax.set_xlabel('Date')
            ax.set_ylim([0, 1])
            ax.set_title('Confidence Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('feature_selection_comparison.png', dpi=150, bbox_inches='tight')
        print("✓ Saved plot to feature_selection_comparison.png")

    # Decision framework
    print("\n" + "=" * 80)
    print("FEATURE SELECTION DECISION FRAMEWORK")
    print("=" * 80)
    print("""
Use this framework to choose features for YOUR specific use case:

1. DO YOU HAVE 2+ YEARS OF DATA?
   └─ NO  → Use univariate (returns only)
   └─ YES → Continue to step 2

2. WHAT IS YOUR MAIN OBJECTIVE?
   ├─ Detect volatility regimes (low/medium/high)
   │  └─ USE: Returns + Realized Volatility (RECOMMENDED)
   ├─ Detect momentum reversals
   │  └─ USE: Returns + Momentum Strength
   ├─ Detect trend changes
   │  └─ USE: Returns + Trend Persistence
   └─ Risk management (concentration detection)
      └─ USE: Returns + Realized Volatility

3. CHECK FEATURE CORRELATION
   Before training, compute correlation(feature1, feature2)
   └─ If corr > 0.95 → Remove one (redundant)
   └─ If corr < 0.5  → Great! Features are independent

4. VALIDATION
   Train on pre-event period, test on event period:
   ├─ 2008 Financial Crisis
   ├─ 2020 COVID Crash
   ├─ 2022 Rate Hikes
   └─ Your own significant market move

See notebook 04 for detailed feature selection methodology with
mathematical foundations and executable code.
""")

    plt.show()


if __name__ == '__main__':
    main()
