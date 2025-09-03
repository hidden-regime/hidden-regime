"""
Basic Hidden Markov Model Demo for hidden-regime package.

Demonstrates core HMM functionality for market regime detection including
training, inference, and real-time regime tracking.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta

# Import the hidden-regime package
import hidden_regime as hr
from hidden_regime import HiddenMarkovModel, HMMConfig


def generate_sample_regime_data(n_observations=200, random_seed=42):
    """Generate synthetic market data with regime switching."""
    np.random.seed(random_seed)
    
    # Define regime characteristics
    regimes = {
        0: {'mean': -0.02, 'std': 0.03, 'name': 'Bear'},      # Bear market
        1: {'mean': 0.005, 'std': 0.015, 'name': 'Sideways'}, # Sideways market  
        2: {'mean': 0.015, 'std': 0.025, 'name': 'Bull'}      # Bull market
    }
    
    # Generate regime sequence with persistence
    true_states = []
    current_regime = 1  # Start in sideways
    duration = 0
    
    for i in range(n_observations):
        # Stay in regime or switch
        duration += 1
        if duration > np.random.poisson(15):  # Average 15-day regime duration
            current_regime = np.random.choice([0, 1, 2])
            duration = 0
        
        true_states.append(current_regime)
    
    # Generate returns based on regime sequence
    returns = []
    for state in true_states:
        regime = regimes[state]
        return_val = np.random.normal(regime['mean'], regime['std'])
        returns.append(return_val)
    
    # Create dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_observations)]
    
    # Convert to arrays
    returns = np.array(returns)
    true_states = np.array(true_states)
    
    return dates, returns, true_states, regimes


def basic_hmm_training_demo():
    """Demonstrate basic HMM training and inference."""
    print("=" * 60)
    print("BASIC HMM TRAINING DEMO")
    print("=" * 60)
    
    # Generate sample data
    print("1. Generating synthetic regime-switching market data...")
    dates, returns, true_states, regimes = generate_sample_regime_data(200)
    
    print(f"   Generated {len(returns)} daily returns")
    print(f"   Data period: {dates[0].date()} to {dates[-1].date()}")
    print(f"   Return statistics: mean={np.mean(returns):.4f}, std={np.std(returns):.4f}")
    
    # Configure and train HMM
    print("\n2. Training 3-state Hidden Markov Model...")
    
    config = HMMConfig(
        n_states=3,
        max_iterations=100,
        tolerance=1e-6,
        initialization_method='kmeans',
        random_seed=42
    )
    
    hmm = HiddenMarkovModel(config=config)
    
    # Train with verbose output
    start_time = datetime.now()
    hmm.fit(returns, verbose=True)
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n   ✓ Training completed in {training_time:.2f} seconds")
    print(f"   ✓ Converged: {hmm.training_history_['converged']}")
    print(f"   ✓ Iterations: {hmm.training_history_['iterations']}")
    print(f"   ✓ Final log-likelihood: {hmm.training_history_['final_log_likelihood']:.2f}")
    
    # Display learned parameters
    print("\n3. Learned HMM Parameters:")
    print(f"   Initial probabilities: {hmm.initial_probs_}")
    print(f"\n   Transition matrix:")
    for i in range(3):
        print(f"     {hmm.transition_matrix_[i]}")
    
    print(f"\n   Emission parameters (mean, std):")
    for i in range(3):
        mean, std = hmm.emission_params_[i]
        print(f"     State {i}: mean={mean:.4f}, std={std:.4f}")
    
    # Perform inference
    print("\n4. Regime Inference:")
    
    # Most likely state sequence (Viterbi)
    predicted_states = hmm.predict(returns)
    
    # State probabilities (Forward-Backward)  
    state_probabilities = hmm.predict_proba(returns)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_states == true_states)
    print(f"   State prediction accuracy: {accuracy:.2%}")
    
    # Analyze regime characteristics
    for state in range(3):
        state_mask = predicted_states == state
        if state_mask.any():
            state_returns = returns[state_mask]
            mean_return = np.mean(state_returns)
            std_return = np.std(state_returns)
            frequency = np.mean(state_mask)
            
            print(f"   Detected State {state}: frequency={frequency:.1%}, "
                  f"mean={mean_return:.4f}, std={std_return:.4f}")
    
    return hmm, returns, predicted_states, state_probabilities, true_states


def real_time_regime_tracking_demo(hmm, test_returns):
    """Demonstrate real-time regime tracking."""
    print("\n" + "=" * 60)
    print("REAL-TIME REGIME TRACKING DEMO")
    print("=" * 60)
    
    print("Simulating real-time regime detection...")
    
    # Reset HMM state
    hmm.reset_state()
    
    # Process returns one by one
    regime_history = []
    
    for i, new_return in enumerate(test_returns[:20]):  # Show first 20 updates
        regime_info = hmm.update_with_observation(new_return)
        regime_history.append(regime_info)
        
        if i < 10 or i % 5 == 0:  # Show first 10, then every 5th
            print(f"   Day {i+1:2d}: Return={new_return:7.4f} | "
                  f"Regime={regime_info['most_likely_regime']} | "
                  f"Confidence={regime_info['confidence']:.3f} | "
                  f"Interpretation: {regime_info['regime_interpretation']}")
    
    # Show final regime probabilities
    final_probs = regime_history[-1]['regime_probabilities']
    print(f"\n   Final regime probabilities:")
    for i, prob in enumerate(final_probs):
        print(f"     State {i}: {prob:.3f}")
    
    return regime_history


def comprehensive_regime_analysis_demo(hmm, returns, dates):
    """Demonstrate comprehensive regime analysis."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE REGIME ANALYSIS DEMO")
    print("=" * 60)
    
    # Perform full regime analysis
    analysis = hmm.analyze_regimes(returns, dates)
    
    print("Model Information:")
    model_info = analysis['model_info']
    print(f"   Number of states: {model_info['n_states']}")
    print(f"   Observations: {model_info['n_observations']}")
    print(f"   Log-likelihood: {model_info['log_likelihood']:.2f}")
    print(f"   Training iterations: {model_info['training_iterations']}")
    
    print("\nRegime Interpretations:")
    for state, interpretation in analysis['regime_interpretations'].items():
        print(f"   State {state}: {interpretation}")
    
    print("\nRegime Statistics:")
    regime_stats = analysis['regime_statistics']['regime_stats']
    
    for state in range(3):
        if state in regime_stats:
            stats = regime_stats[state]
            print(f"\n   State {state} ({analysis['regime_interpretations'][str(state)]}):")
            print(f"     Frequency: {stats['frequency']:.1%}")
            print(f"     Mean return: {stats['mean_return']:.4f}")
            print(f"     Volatility: {stats['std_return']:.4f}")
            print(f"     Average duration: {stats['avg_duration']:.1f} days")
            print(f"     Number of episodes: {stats['n_episodes']}")
    
    return analysis


def model_persistence_demo(hmm, returns):
    """Demonstrate model saving and loading."""
    print("\n" + "=" * 60)
    print("MODEL PERSISTENCE DEMO")
    print("=" * 60)
    
    import tempfile
    from pathlib import Path
    
    # Save model
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        model_path = Path(f.name)
    
    print(f"Saving model to: {model_path}")
    hmm.save(model_path)
    print("   ✓ Model saved successfully")
    
    # Load model
    print(f"Loading model from: {model_path}")
    loaded_hmm = HiddenMarkovModel.load(model_path)
    print("   ✓ Model loaded successfully")
    
    # Verify loaded model
    original_likelihood = hmm.score(returns)
    loaded_likelihood = loaded_hmm.score(returns)
    
    print(f"   Original model likelihood: {original_likelihood:.4f}")
    print(f"   Loaded model likelihood: {loaded_likelihood:.4f}")
    print(f"   Difference: {abs(original_likelihood - loaded_likelihood):.8f}")
    
    # Clean up
    model_path.unlink()
    print("   ✓ Temporary file cleaned up")
    
    return loaded_hmm


def convenience_functions_demo():
    """Demonstrate convenience functions."""
    print("\n" + "=" * 60)
    print("CONVENIENCE FUNCTIONS DEMO")
    print("=" * 60)
    
    # Generate data
    _, returns, true_states, _ = generate_sample_regime_data(150, random_seed=123)
    
    # Use convenience function for regime detection
    print("1. Using hr.detect_regimes():")
    detected_states = hr.detect_regimes(returns, n_states=3, max_iterations=50, verbose=False)
    
    accuracy = np.mean(detected_states == true_states)
    print(f"   ✓ Detected regimes for {len(returns)} observations")
    print(f"   ✓ Accuracy vs true regimes: {accuracy:.2%}")
    
    # Use convenience function for comprehensive analysis
    print("\n2. Using hr.analyze_regime_transitions():")
    dates = pd.date_range('2023-01-01', periods=len(returns), freq='D')
    analysis = hr.analyze_regime_transitions(
        returns, 
        dates, 
        n_states=3, 
        max_iterations=50,
        verbose=False
    )
    
    print(f"   ✓ Complete regime analysis generated")
    print(f"   ✓ Model converged: {analysis['model_info']['converged']}")
    print(f"   ✓ Final log-likelihood: {analysis['model_info']['log_likelihood']:.2f}")
    
    # Show regime frequencies
    regime_stats = analysis['regime_statistics']['regime_stats']
    print("\n   Regime frequencies:")
    for state in range(3):
        if state in regime_stats:
            freq = regime_stats[state]['frequency']
            interpretation = analysis['regime_interpretations'][str(state)]
            print(f"     {interpretation}: {freq:.1%}")


def plotting_demo(hmm, returns, predicted_states, true_states, dates):
    """Create visualizations of regime detection results."""
    print("\n" + "=" * 60)
    print("VISUALIZATION DEMO")
    print("=" * 60)
    
    try:
        # Create plots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Hidden Markov Model Regime Detection Results', fontsize=16)
        
        # Plot 1: Returns with true regimes
        axes[0].scatter(range(len(returns)), returns, c=true_states, cmap='viridis', alpha=0.7, s=20)
        axes[0].set_title('True Regime States')
        axes[0].set_ylabel('Log Returns')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Returns with predicted regimes
        axes[1].scatter(range(len(returns)), returns, c=predicted_states, cmap='viridis', alpha=0.7, s=20)
        axes[1].set_title('Predicted Regime States')
        axes[1].set_ylabel('Log Returns')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Regime probabilities over time
        state_probs = hmm.predict_proba(returns)
        
        colors = ['red', 'yellow', 'green']
        regime_names = ['Bear', 'Sideways', 'Bull']
        
        for state in range(3):
            axes[2].plot(range(len(returns)), state_probs[:, state], 
                        color=colors[state], label=f'State {state} ({regime_names[state]})', alpha=0.8)
        
        axes[2].set_title('Regime Probabilities Over Time')
        axes[2].set_xlabel('Time (Days)')
        axes[2].set_ylabel('Probability')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Try to save plot
        try:
            plt.savefig('hmm_regime_detection_demo.png', dpi=150, bbox_inches='tight')
            print("   ✓ Plot saved as 'hmm_regime_detection_demo.png'")
        except Exception as e:
            print(f"   ⚠ Could not save plot: {e}")
        
        # Try to show plot
        try:
            plt.show()
            print("   ✓ Interactive plot displayed")
        except Exception as e:
            print(f"   ⚠ Could not display plot: {e}")
            
    except ImportError:
        print("   ⚠ Matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"   ✗ Plotting failed: {e}")


def main():
    """Run the complete HMM demonstration."""
    print("Hidden Regime - Basic HMM Demo")
    print("Demonstrates market regime detection using Hidden Markov Models")
    print()
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    try:
        # Basic training and inference demo
        hmm, returns, predicted_states, state_probs, true_states = basic_hmm_training_demo()
        dates, _, _, _ = generate_sample_regime_data(200, 42)
        
        # Real-time tracking demo
        real_time_regime_tracking_demo(hmm, returns[150:])  # Use last part of data
        
        # Comprehensive analysis demo
        comprehensive_regime_analysis_demo(hmm, returns, dates)
        
        # Model persistence demo  
        model_persistence_demo(hmm, returns)
        
        # Convenience functions demo
        convenience_functions_demo()
        
        # Plotting demo
        plotting_demo(hmm, returns, predicted_states, true_states, dates)
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("- Try with your own market data using hr.load_stock_data()")
        print("- Experiment with different HMMConfig parameters")
        print("- Explore regime-based trading strategies")
        print("- Check out portfolio_analysis_example.py for multi-asset analysis")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        print("\nThis demo uses synthetic data and should work without external dependencies.")
        print("If you see this error, there may be an issue with the HMM implementation.")
        raise


if __name__ == "__main__":
    main()