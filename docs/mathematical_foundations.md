# Mathematical Foundations of Hidden Markov Models

*A comprehensive mathematical guide to HMM theory, algorithms, and implementation*

---

## Table of Contents

1. [Overview](#overview)
2. [Probability Theory Foundation](#probability-theory-foundation)
3. [Hidden Markov Model Definition](#hidden-markov-model-definition)
4. [The Three Fundamental Problems](#the-three-fundamental-problems)
5. [Forward Algorithm](#forward-algorithm)
6. [Backward Algorithm](#backward-algorithm)
7. [Viterbi Algorithm](#viterbi-algorithm)
8. [Baum-Welch Algorithm (EM)](#baum-welch-algorithm-em)
9. [Online Learning Mathematics](#online-learning-mathematics)
10. [Numerical Stability](#numerical-stability)
11. [Convergence Theory](#convergence-theory)
12. [Implementation Details](#implementation-details)
13. [Market Regime Applications](#market-regime-applications)
14. [References](#references)

---

## Overview

Hidden Markov Models (HMMs) provide a powerful statistical framework for modeling sequences where the underlying state is hidden but generates observable data. In financial markets, this translates to hidden market regimes (bull, bear, sideways) that generate observable returns.

### Why HMMs Work for Financial Markets

1. **Regime Persistence**: Markets exhibit persistent behavioral phases
2. **State-Dependent Dynamics**: Different regimes have distinct statistical properties
3. **Transition Dynamics**: Markets transition between regimes following probabilistic patterns
4. **Observable Manifestations**: Hidden regimes generate observable return patterns
5. **Uncertainty Quantification**: Probabilistic framework captures regime uncertainty

---

## Probability Theory Foundation

### Basic Definitions

Let's define the fundamental mathematical objects:

- **Time Index**: t ∈ {1, 2, ..., T}
- **Hidden States**: S_t ∈ {1, 2, ..., N} (market regimes)
- **Observations**: O_t ∈ ℝ (log returns)
- **State Sequence**: S = (S_1, S_2, ..., S_T)
- **Observation Sequence**: O = (O_1, O_2, ..., O_T)

### Markov Property

The **first-order Markov property** assumes:

```
P(S_t+1 | S_1, S_2, ..., S_t) = P(S_t+1 | S_t)
```

This means the future state depends only on the current state, not the entire history.

### Conditional Independence

Observations are **conditionally independent** given the hidden state:

```
P(O_t | S_1, ..., S_T, O_1, ..., O_t-1, O_t+1, ..., O_T) = P(O_t | S_t)
```

This means once we know the current regime, the observation is independent of all other information.

---

## Hidden Markov Model Definition

An HMM is characterized by the tuple λ = (π, A, B) where:

### Initial State Distribution (π)
```
π_i = P(S_1 = i),    i = 1, ..., N
```
with constraints:
```
π_i ≥ 0,    Σ_{i=1}^N π_i = 1
```

### Transition Matrix (A)
```
A_{ij} = P(S_t+1 = j | S_t = i),    i, j = 1, ..., N
```
with constraints:
```
A_{ij} ≥ 0,    Σ_{j=1}^N A_{ij} = 1    ∀i
```

### Emission Distribution (B)
For Gaussian emissions in financial markets:
```
B_i(o) = P(O_t = o | S_t = i) = N(o; μ_i, σ_i²)
```
where:
```
N(o; μ_i, σ_i²) = (1/√(2πσ_i²)) exp(-(o - μ_i)²/(2σ_i²))
```

### Joint Probability
The joint probability of observations and states is:
```
P(O, S | λ) = π_{S_1} ∏_{t=2}^T A_{S_{t-1},S_t} ∏_{t=1}^T B_{S_t}(O_t)
```

---

## The Three Fundamental Problems

### Problem 1: Evaluation
**Given**: Model λ = (π, A, B) and observation sequence O
**Find**: P(O | λ) - likelihood of observations

**Solution**: Forward algorithm

### Problem 2: Decoding  
**Given**: Model λ and observation sequence O
**Find**: Most likely state sequence S* = argmax P(S | O, λ)

**Solution**: Viterbi algorithm

### Problem 3: Learning
**Given**: Observation sequence O
**Find**: Model parameters λ* = argmax P(O | λ)

**Solution**: Baum-Welch algorithm (Expectation-Maximization)

---

## Forward Algorithm

The forward algorithm computes P(O | λ) efficiently using dynamic programming.

### Forward Variables
Define the forward variable:
```
α_t(i) = P(O_1, O_2, ..., O_t, S_t = i | λ)
```

This represents the probability of observing the first t observations and being in state i at time t.

### Initialization
```
α_1(i) = π_i B_i(O_1),    i = 1, ..., N
```

### Recursion
```
α_{t+1}(j) = [Σ_{i=1}^N α_t(i) A_{ij}] B_j(O_{t+1}),    j = 1, ..., N,  t = 1, ..., T-1
```

### Termination
```
P(O | λ) = Σ_{i=1}^N α_T(i)
```

### Computational Complexity
- **Time**: O(N²T)
- **Space**: O(NT)

### Implementation Pseudocode
```python
def forward_algorithm(observations, initial_probs, transition_matrix, emission_params):
    T = len(observations)
    N = len(initial_probs)
    
    # Initialize forward variables
    alpha = np.zeros((T, N))
    
    # Initialization step
    for i in range(N):
        alpha[0, i] = initial_probs[i] * gaussian_pdf(observations[0], emission_params[i])
    
    # Recursion step
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = 0
            for i in range(N):
                alpha[t, j] += alpha[t-1, i] * transition_matrix[i, j]
            alpha[t, j] *= gaussian_pdf(observations[t], emission_params[j])
    
    # Termination
    likelihood = np.sum(alpha[T-1, :])
    
    return alpha, likelihood
```

---

## Backward Algorithm

The backward algorithm computes conditional probabilities needed for parameter estimation.

### Backward Variables
Define the backward variable:
```
β_t(i) = P(O_{t+1}, O_{t+2}, ..., O_T | S_t = i, λ)
```

This represents the probability of observing the remaining observations from time t+1 to T, given that we're in state i at time t.

### Initialization
```
β_T(i) = 1,    i = 1, ..., N
```

### Recursion
```
β_t(i) = Σ_{j=1}^N A_{ij} B_j(O_{t+1}) β_{t+1}(j),    i = 1, ..., N,  t = T-1, ..., 1
```

### Implementation Pseudocode
```python
def backward_algorithm(observations, transition_matrix, emission_params):
    T = len(observations)
    N = transition_matrix.shape[0]
    
    # Initialize backward variables
    beta = np.zeros((T, N))
    
    # Initialization step (time T)
    beta[T-1, :] = 1.0
    
    # Recursion step (time T-1 down to 1)
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t, i] = 0
            for j in range(N):
                beta[t, i] += transition_matrix[i, j] * \
                             gaussian_pdf(observations[t+1], emission_params[j]) * \
                             beta[t+1, j]
    
    return beta
```

---

## Viterbi Algorithm

The Viterbi algorithm finds the most likely state sequence using dynamic programming.

### Viterbi Variables
Define the Viterbi variable:
```
δ_t(i) = max_{s_1,...,s_{t-1}} P(S_1 = s_1, ..., S_{t-1} = s_{t-1}, S_t = i, O_1, ..., O_t | λ)
```

### Path Tracking
Define the path variable:
```
ψ_t(i) = argmax_{s_1,...,s_{t-1}} P(S_1 = s_1, ..., S_{t-1} = s_{t-1}, S_t = i, O_1, ..., O_t | λ)
```

### Algorithm Steps

#### 1. Initialization
```
δ_1(i) = π_i B_i(O_1),    i = 1, ..., N
ψ_1(i) = 0
```

#### 2. Recursion
```
δ_t(j) = max_{i} [δ_{t-1}(i) A_{ij}] B_j(O_t),    j = 1, ..., N,  t = 2, ..., T
ψ_t(j) = argmax_{i} [δ_{t-1}(i) A_{ij}],    j = 1, ..., N,  t = 2, ..., T
```

#### 3. Termination
```
P* = max_{i} δ_T(i)
S_T* = argmax_{i} δ_T(i)
```

#### 4. Path Backtracking
```
S_t* = ψ_{t+1}(S_{t+1}*),    t = T-1, T-2, ..., 1
```

### Implementation Pseudocode
```python
def viterbi_algorithm(observations, initial_probs, transition_matrix, emission_params):
    T = len(observations)
    N = len(initial_probs)
    
    # Initialize Viterbi variables
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    
    # Initialization
    for i in range(N):
        delta[0, i] = initial_probs[i] * gaussian_pdf(observations[0], emission_params[i])
        psi[0, i] = 0
    
    # Recursion
    for t in range(1, T):
        for j in range(N):
            # Find the most likely previous state
            probabilities = delta[t-1, :] * transition_matrix[:, j]
            delta[t, j] = np.max(probabilities) * gaussian_pdf(observations[t], emission_params[j])
            psi[t, j] = np.argmax(probabilities)
    
    # Termination
    best_path_prob = np.max(delta[T-1, :])
    best_last_state = np.argmax(delta[T-1, :])
    
    # Path backtracking
    path = np.zeros(T, dtype=int)
    path[T-1] = best_last_state
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    
    return path, best_path_prob
```

---

## Baum-Welch Algorithm (EM)

The Baum-Welch algorithm is an instance of the Expectation-Maximization (EM) algorithm for learning HMM parameters.

### E-Step: Compute Posterior Probabilities

#### Gamma Variables (State Probabilities)
```
γ_t(i) = P(S_t = i | O, λ) = (α_t(i) β_t(i)) / P(O | λ)
```

This gives the probability of being in state i at time t given all observations.

#### Xi Variables (Transition Probabilities)  
```
ξ_t(i,j) = P(S_t = i, S_{t+1} = j | O, λ) = (α_t(i) A_{ij} B_j(O_{t+1}) β_{t+1}(j)) / P(O | λ)
```

This gives the probability of transitioning from state i to state j at time t.

### M-Step: Update Parameters

#### Initial State Probabilities
```
π̂_i = γ_1(i)
```

#### Transition Probabilities
```
Â_{ij} = (Σ_{t=1}^{T-1} ξ_t(i,j)) / (Σ_{t=1}^{T-1} γ_t(i))
```

#### Emission Parameters (Gaussian)
Mean:
```
μ̂_i = (Σ_{t=1}^T γ_t(i) O_t) / (Σ_{t=1}^T γ_t(i))
```

Variance:
```
σ̂_i² = (Σ_{t=1}^T γ_t(i) (O_t - μ̂_i)²) / (Σ_{t=1}^T γ_t(i))
```

### Complete Algorithm
```python
def baum_welch_algorithm(observations, initial_guess, max_iterations=100, tolerance=1e-6):
    """
    Baum-Welch algorithm for HMM parameter estimation
    """
    # Initialize parameters
    pi, A, emission_params = initial_guess
    N = len(pi)
    T = len(observations)
    
    prev_log_likelihood = -np.inf
    
    for iteration in range(max_iterations):
        # E-step: Forward-Backward algorithm
        alpha, forward_likelihood = forward_algorithm(observations, pi, A, emission_params)
        beta = backward_algorithm(observations, A, emission_params)
        
        # Compute gamma and xi
        gamma = np.zeros((T, N))
        xi = np.zeros((T-1, N, N))
        
        for t in range(T):
            for i in range(N):
                gamma[t, i] = alpha[t, i] * beta[t, i] / forward_likelihood
        
        for t in range(T-1):
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = alpha[t, i] * A[i, j] * \
                                 gaussian_pdf(observations[t+1], emission_params[j]) * \
                                 beta[t+1, j] / forward_likelihood
        
        # M-step: Parameter updates
        # Update initial probabilities
        pi = gamma[0, :]
        
        # Update transition probabilities
        for i in range(N):
            for j in range(N):
                A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])
        
        # Update emission parameters
        for i in range(N):
            # Mean
            emission_params[i]['mean'] = np.sum(gamma[:, i] * observations) / np.sum(gamma[:, i])
            # Variance
            diff = observations - emission_params[i]['mean']
            emission_params[i]['variance'] = np.sum(gamma[:, i] * diff**2) / np.sum(gamma[:, i])
        
        # Check convergence
        log_likelihood = np.log(forward_likelihood)
        if abs(log_likelihood - prev_log_likelihood) < tolerance:
            break
        prev_log_likelihood = log_likelihood
    
    return pi, A, emission_params, log_likelihood
```

---

## Online Learning Mathematics

Online HMM learning adapts parameters incrementally without retraining on the entire dataset.

### Exponential Forgetting Framework

#### Forgetting Factor
Let λ ∈ (0, 1) be the forgetting factor. Recent observations have weight λ⁰ = 1, previous observations have weight λ¹ = λ, and so on.

#### Effective Sample Size
The effective sample size after T observations is:
```
N_eff = (1 - λ^T) / (1 - λ) ≈ 1 / (1 - λ)    for large T
```

### Sufficient Statistics with Forgetting

#### State Occupation Statistics
```
γ̃_t^(n)(i) = λ γ̃_{t-1}^(n)(i) + γ_t(i)
```

#### Transition Statistics
```
ξ̃_t^(n)(i,j) = λ ξ̃_{t-1}^(n)(i,j) + ξ_t(i,j)
```

#### Observation Statistics
Mean numerator:
```
μ̃_t^(n)(i) = λ μ̃_{t-1}^(n)(i) + γ_t(i) O_t
```

Variance numerator:
```
σ̃_t^(n)(i) = λ σ̃_{t-1}^(n)(i) + γ_t(i) O_t²
```

### Online Parameter Updates

#### Transition Probabilities
```
A_{ij}^(n) = ξ̃_t^(n)(i,j) / Σ_k ξ̃_t^(n)(i,k)
```

#### Emission Parameters
Mean:
```
μ_i^(n) = μ̃_t^(n)(i) / γ̃_t^(n)(i)
```

Variance:
```
σ_i²^(n) = (σ̃_t^(n)(i) / γ̃_t^(n)(i)) - (μ_i^(n))²
```

### Parameter Smoothing

To prevent excessive parameter volatility:
```
θ_new = (1 - α) θ_old + α θ_updated
```
where α ∈ (0, 1) is the adaptation rate.

### Recursive State Probability Update

For online processing, we need to compute state probabilities recursively:

#### Prediction Step
```
π_{t|t-1}(j) = Σ_i π_{t-1|t-1}(i) A_{ij}
```

#### Update Step
```
π_{t|t}(j) = (π_{t|t-1}(j) B_j(O_t)) / (Σ_k π_{t|t-1}(k) B_k(O_t))
```

---

## Numerical Stability

### Log-Space Computations

All probability computations should be performed in log-space to prevent numerical underflow.

#### Log-Forward Algorithm
```
log α_t(i) = log π_i + log B_i(O_1)                           # Initialization
log α_{t+1}(j) = log B_j(O_{t+1}) + logsumexp_i(log α_t(i) + log A_{ij})  # Recursion
```

#### Log-Sum-Exp Function
The numerically stable log-sum-exp function:
```python
def logsumexp(x):
    """Numerically stable log-sum-exp"""
    x_max = np.max(x)
    if np.isinf(x_max):
        return x_max
    return x_max + np.log(np.sum(np.exp(x - x_max)))
```

### Regularization Techniques

#### Minimum Variance Constraint
Prevent degenerate variance estimates:
```
σ_i² = max(σ_i², ε_min)
```
where ε_min ≈ 1e-6.

#### Transition Matrix Regularization
Add small constant to prevent zero probabilities:
```
A_{ij} = (A_{ij} + ε) / (Σ_k (A_{ik} + ε))
```

#### Emission Parameter Bounds
Constrain parameters to reasonable ranges:
```
μ_i ∈ [-μ_max, μ_max]
σ_i ∈ [σ_min, σ_max]
```

---

## Convergence Theory

### EM Convergence Properties

#### Monotonic Increase
The Baum-Welch algorithm guarantees:
```
L(λ^(n+1)) ≥ L(λ^(n))
```
where L(λ) = log P(O | λ) is the log-likelihood.

#### Convergence to Local Optimum
The algorithm converges to a local optimum of the likelihood function.

### Practical Convergence Criteria

#### Relative Improvement
```
|L(λ^(n+1)) - L(λ^(n))| / |L(λ^(n))| < tolerance
```

#### Parameter Stability
```
||θ^(n+1) - θ^(n)|| / ||θ^(n)|| < tolerance
```

### Online Learning Convergence

#### Stochastic Approximation Theory
Under regularity conditions, online parameter updates converge to the true parameters if:
1. Learning rate satisfies Robbins-Monro conditions
2. Forgetting factor is chosen appropriately
3. Data is stationary or changes slowly

---

## Implementation Details

### Numerical Considerations

#### Scaling Factors
Use scaling factors to prevent numerical underflow:
```python
def scaled_forward_algorithm(observations, pi, A, B):
    T, N = len(observations), len(pi)
    alpha = np.zeros((T, N))
    c = np.zeros(T)  # Scaling factors
    
    # Initialize
    alpha[0] = pi * B[:, 0]
    c[0] = 1.0 / np.sum(alpha[0])
    alpha[0] *= c[0]
    
    # Recurse
    for t in range(1, T):
        alpha[t] = np.dot(alpha[t-1], A) * B[:, t]
        c[t] = 1.0 / np.sum(alpha[t])
        alpha[t] *= c[t]
    
    # Log-likelihood
    log_likelihood = -np.sum(np.log(c))
    
    return alpha, c, log_likelihood
```

#### Memory Management
For long sequences, use windowed processing:
```python
def windowed_hmm_training(observations, window_size=1000, overlap=100):
    """Train HMM on long sequences using sliding windows"""
    results = []
    for i in range(0, len(observations) - window_size + 1, window_size - overlap):
        window = observations[i:i + window_size]
        hmm_result = train_hmm(window)
        results.append(hmm_result)
    return combine_results(results)
```

### Initialization Strategies

#### K-Means Initialization
```python
def kmeans_initialization(observations, n_states):
    """Initialize HMM parameters using K-means clustering"""
    from sklearn.cluster import KMeans
    
    # Reshape for clustering
    X = observations.reshape(-1, 1)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_states, random_state=42)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_.flatten()
    
    # Initialize parameters
    pi = np.bincount(labels[:10]) / 10  # Initial distribution from first 10 points
    
    # Transition matrix from label sequence
    A = compute_transition_matrix(labels)
    
    # Emission parameters from clusters
    emission_params = []
    for i in range(n_states):
        cluster_obs = observations[labels == i]
        emission_params.append({
            'mean': centers[i],
            'variance': np.var(cluster_obs) if len(cluster_obs) > 1 else 0.01
        })
    
    return pi, A, emission_params
```

### Validation Techniques

#### Cross-Validation for HMMs
```python
def hmm_cross_validation(observations, n_states_range, n_folds=5):
    """Cross-validation for HMM model selection"""
    fold_size = len(observations) // n_folds
    results = {}
    
    for n_states in n_states_range:
        fold_scores = []
        
        for fold in range(n_folds):
            # Split data
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size
            
            train_data = np.concatenate([
                observations[:start_idx],
                observations[end_idx:]
            ])
            test_data = observations[start_idx:end_idx]
            
            # Train and evaluate
            hmm = train_hmm(train_data, n_states)
            score = hmm.score(test_data)
            fold_scores.append(score)
        
        results[n_states] = {
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores)
        }
    
    return results
```

---

## Market Regime Applications

### Financial Interpretation

#### Regime Characteristics
- **Bull Regime**: μ > 0, moderate σ, persistent transitions
- **Bear Regime**: μ < 0, high σ, shorter duration
- **Sideways Regime**: μ ≈ 0, low σ, stable periods

#### Regime-Specific Metrics
Expected duration in regime i:
```
E[T_i] = 1 / (1 - A_{ii})
```

Regime transition probabilities:
```
P(Bull → Bear) = A_{bull,bear}
P(Bear → Bull) = A_{bear,bull}
```

### Trading Applications

#### Position Sizing
```python
def regime_position_size(current_regime, confidence, base_size=1.0):
    """Calculate position size based on regime and confidence"""
    regime_multipliers = {
        'bull': 1.0,
        'sideways': 0.3,
        'bear': -0.5
    }
    
    position = base_size * regime_multipliers[current_regime] * confidence
    return np.clip(position, -1.0, 1.0)
```

#### Risk Management
```python
def regime_var(regime_params, confidence_level=0.05):
    """Calculate Value at Risk by regime"""
    regime_var = {}
    for regime, (mu, sigma) in regime_params.items():
        # Daily VaR at confidence level
        z_score = scipy.stats.norm.ppf(confidence_level)
        var = mu + sigma * z_score
        regime_var[regime] = var
    return regime_var
```

---

## References

1. **Rabiner, L. R. (1989)**. "A tutorial on hidden Markov models and selected applications in speech recognition." *Proceedings of the IEEE*, 77(2), 257-286.

2. **Baum, L. E., Petrie, T., Soules, G., & Weiss, N. (1970)**. "A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains." *The Annals of Mathematical Statistics*, 41(1), 164-171.

3. **Viterbi, A. (1967)**. "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm." *IEEE Transactions on Information Theory*, 13(2), 260-269.

4. **Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977)**. "Maximum likelihood from incomplete data via the EM algorithm." *Journal of the Royal Statistical Society*, 39(1), 1-38.

5. **Hamilton, J. D. (1989)**. "A new approach to the economic analysis of nonstationary time series and the business cycle." *Econometrica*, 57(2), 357-384.

6. **Kim, C. J., & Nelson, C. R. (1999)**. *State-space models with regime switching: classical and Gibbs-sampling approaches with applications*. MIT Press.

7. **Cappé, O., Moulines, E., & Rydén, T. (2005)**. *Inference in hidden Markov models*. Springer Science & Business Media.

8. **Murphy, K. P. (2012)**. *Machine learning: a probabilistic perspective*. MIT Press.

---

*This document provides the complete mathematical foundation for understanding and implementing Hidden Markov Models in the Hidden Regime framework. For practical implementation details, see the [API Reference](../hidden_regime/models/README.md) and [Online HMM Documentation](online_hmm.md).*