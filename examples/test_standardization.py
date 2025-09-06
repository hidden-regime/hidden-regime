"""
Quick test script to verify state standardization functionality.
"""

import numpy as np
import sys
from pathlib import Path

# Add hidden-regime to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the hidden-regime package
from hidden_regime import HiddenMarkovModel, HMMConfig

def test_standardization():
    """Test basic standardization functionality."""
    print("Testing State Standardization Framework")
    print("=" * 50)
    
    # Generate sample data with clear regimes
    np.random.seed(42)
    
    # Create regime switching data
    n_obs = 500
    returns = []
    true_regimes = []
    
    # Bear market period
    bear_returns = np.random.normal(-0.02, 0.03, 100)
    returns.extend(bear_returns)
    true_regimes.extend([0] * 100)
    
    # Sideways period
    sideways_returns = np.random.normal(0.001, 0.015, 200)
    returns.extend(sideways_returns)
    true_regimes.extend([1] * 200)
    
    # Bull market period
    bull_returns = np.random.normal(0.015, 0.025, 200)
    returns.extend(bull_returns)
    true_regimes.extend([2] * 200)
    
    returns = np.array(returns)
    
    print(f"Generated {len(returns)} observations")
    print(f"Mean return: {np.mean(returns):.4f}")
    print(f"Std return: {np.std(returns):.4f}")
    
    # Test 3-state standardized configuration
    print("\nTesting 3-state standardized configuration:")
    config = HMMConfig.for_standardized_regimes('3_state')
    hmm = HiddenMarkovModel(config=config)
    
    print("Fitting model...")
    hmm.fit(returns, verbose=False)
    
    print(f"Model converged: {hmm.training_history_['converged']}")
    print(f"Final log-likelihood: {hmm.training_history_['final_log_likelihood']:.2f}")
    
    # Check standardization
    if hasattr(hmm, '_state_standardizer') and hmm._state_standardizer is not None:
        print(f"✓ Standardization applied")
        print(f"✓ Confidence: {hmm._standardization_confidence:.3f}")
        
        if hasattr(hmm, '_state_mapping'):
            print(f"✓ State mapping: {hmm._state_mapping}")
            
        # Show regime characteristics
        print("\nDetected Regimes:")
        config_obj = hmm._state_standardizer.current_config
        if config_obj:
            for i in range(hmm.n_states):
                mean, std = hmm.emission_params_[i]
                mapped_state = hmm._state_mapping.get(i, i) if hasattr(hmm, '_state_mapping') else i
                if isinstance(mapped_state, int) and mapped_state < len(config_obj.state_names):
                    regime_name = config_obj.state_names[mapped_state]
                else:
                    regime_name = f"State {i}"
                print(f"  {regime_name}: μ={mean:.4f}, σ={std:.4f}")
    else:
        print("⚠ Standardization not applied")
    
    print("\n" + "=" * 50)
    print("Test completed successfully!")
    
    return hmm

if __name__ == "__main__":
    test_standardization()