# HMM Model Retraining Strategy Roadmap

## Overview

This roadmap describes the implementation of a production-grade adaptive HMM retraining system that keeps the model flexible and responsive to market changes while maintaining interpretability and avoiding expensive re-trains.

**Key Characteristics**:
- Anchored regime interpretation (stable labels despite parameter changes)
- Hierarchical update schedule: Weekly (1% cost) → Monthly (5% cost) → Quarterly (100% cost)
- Statistical drift monitoring with SLRT + KL divergence + Hellinger distance
- Triggered retrains when market structure shifts detected
- Temporal isolation preserved for V&V compliant backtesting

---

## Strategy Components

### 1. Anchored Interpreter (PRIORITY 1)
Provides stable regime labels despite model parameter evolution.

**Problem Solved**:
- Standard interpreters map HMM states to regimes based on current parameters
- When parameters drift (via online updates), regime labels can flip unexpectedly
- Users lose confidence in the system if "Bull" suddenly becomes "Bear"

**Solution**:
- Maintain "regime anchors" - templates for each regime (mean/std distributions)
- Map HMM states to regimes by **minimum KL divergence to anchors**
- Slowly update anchors via exponential smoothing (α=0.01 → 69 days to half-adapt)
- Result: regime labels stay stable while model parameters evolve

**Files**:
- New: `hidden_regime/interpreter/anchored.py`
- Modified: `hidden_regime/config/model.py`, `hidden_regime/interpreter/financial.py`

---

### 2. Drift Detection System (PRIORITY 2)
Monitors when the market has fundamentally changed, triggering full retrains.

**Three Statistical Tests** (redundant for robustness):

1. **SLRT (Sequential Likelihood Ratio Test)**
   - Cumulative sum test for likelihood ratio
   - Threshold: 10.0 → α ≈ 0.005 (~1 false alarm per 200 days)
   - Fast, online, continuous monitoring

2. **KL Divergence of Emission Distributions**
   - Measures information-theoretic distance between old/new parameters
   - Hard threshold: 2.0 nats (critical drift)
   - Soft threshold: 1.0 nat (elevated, monitor)
   - Captures both mean and variance shifts

3. **Hellinger Distance**
   - Robust, symmetric version of KL divergence
   - Bounded [0, 1], insensitive to variance changes
   - Threshold: 0.3 (substantial drift)
   - Used for cross-validation of other metrics

**Decision Logic**:
- If critical drift detected: full retrain immediately
- If soft drift detected + time passed: full retrain
- Otherwise: follow scheduled updates

**Files**:
- New: `hidden_regime/monitoring/drift_detector.py`, `hidden_regime/monitoring/retraining_policy.py`

---

### 3. Hierarchical Update Methods (PRIORITY 3)
Three-tier update strategy for computational efficiency.

**Tier 1: Emission-Only Update (Weekly, ~1% cost)**
- Keep transition matrix fixed
- Update emission parameters (means, variances) only
- Use forward-backward with fixed A matrix
- Cost: O(N² × 30 observations) vs O(T² × N²) for full retrain
- Keeps regime volatilities current without expensive retraining

**Tier 2: Transition-Only Update (Monthly, ~5% cost)**
- Keep emission parameters fixed
- Viterbi decode state sequence from recent data (60 days)
- Update transition matrix from state transitions
- Add Dirichlet smoothing (α=0.1) to prevent degeneracy
- Cost: O(N² × 60 observations)
- Adapts to changing regime persistence (bull markets get longer, etc.)

**Tier 3: Full Retrain (Quarterly, 100% cost + Triggered)**
- Full Baum-Welch algorithm on recent window (252 trading days)
- Escapes local optima from incremental updates
- Optional: use current parameters as informative prior
- Scheduled: every 63 trading days
- Triggered: immediately if drift detected
- Cost: full retraining (expensive but necessary)

**Files**:
- Modified: `hidden_regime/models/hmm.py` (add three new methods)

---

### 4. Retraining Policy Engine (PRIORITY 3)
Orchestrates when and how often to apply each update type.

**Scheduling Parameters**:
- `emission_update_frequency_days`: 5 (weekly)
- `transition_update_frequency_days`: 21 (monthly)
- `full_retrain_frequency_days`: 63 (quarterly)
- `max_days_without_retrain`: 90 (hard ceiling)
- `min_days_between_retrains`: 14 (prevent thrashing)

**Trigger Priority** (executed in order):
1. Hard constraint: `max_days` exceeded → full retrain
2. Minimum constraint: `min_days` not met → no retrain
3. Critical drift: SLRT > threshold OR KL > 2.0 → immediate full retrain
4. Prediction error: exceeds 95th percentile → full retrain
5. Soft drift + time: KL > 1.0 AND days > half-cycle → full retrain
6. Scheduled: emission/transition/full retrain based on calendar

**Files**:
- New: `hidden_regime/monitoring/retraining_policy.py`
- Modified: `hidden_regime/models/hmm.py` (integrate with `update()`)

---

### 5. Configuration System Extensions
New `AdaptiveRetrainingConfig` dataclass with three profiles.

**Moderate (DEFAULT)**:
```python
emission_update_frequency_days = 5
transition_update_frequency_days = 21
full_retrain_frequency_days = 63
max_days_without_retrain = 90
min_days_between_retrains = 14
slrt_threshold = 10.0
kl_divergence_hard_threshold = 2.0
anchor_update_rate = 0.01
use_anchored_interpretation = True
```

**Conservative** (regulatory/compliance):
```python
full_retrain_frequency_days = 180
slrt_threshold = 15.0
kl_divergence_hard_threshold = 3.0
# Slower adaptation, more stable labels
```

**Aggressive** (high-frequency trading):
```python
full_retrain_frequency_days = 30
slrt_threshold = 5.0
kl_divergence_hard_threshold = 1.0
# Rapid adaptation, accepts label instability
```

**Files**:
- Modified: `hidden_regime/config/model.py`

---

## Implementation Tiers

### Tier 1: Anchored Interpretation Foundation
**Goal**: Stable regime labels as foundation for all other updates

Tasks:
- [ ] Create `hidden_regime/interpreter/anchored.py`
  - `AnchoredInterpreter` class
  - `interpret_states()` method (minimum KL divergence matching)
  - `_update_anchors()` method (exponential smoothing)
- [ ] Extend `hidden_regime/config/model.py`
  - Add anchor configuration to `AdaptiveRetrainingConfig`
  - Add `use_anchored_interpretation: bool = True` (default enabled)
  - Add anchor update rate (default 0.01)
- [ ] Integrate into `FinancialInterpreter`
  - Add toggle to use `AnchoredInterpreter` wrapper
  - Default: enabled for all new pipelines
- [ ] Comprehensive unit tests
  - KL divergence matching logic
  - Anchor update exponential smoothing
  - Confidence score computation
  - Label stability under parameter drift

**Success Criteria**:
- Regime labels remain stable even when parameters drift 10%+
- Anchors eventually adapt (but slowly) to true parameter changes
- 100% test coverage for anchored interpreter module

---

### Tier 2: Drift Detection & Monitoring
**Goal**: Detect when market structure changes, trigger full retrains

Tasks:
- [ ] Create `hidden_regime/monitoring/drift_detector.py`
  - `DriftDetector` class with SLRT implementation
  - `ParameterMonitor` class with three-metric assessment
  - Supporting methods for KL divergence and Hellinger distance
- [ ] Create `hidden_regime/monitoring/retraining_policy.py`
  - `RetrainingPolicy` class
  - `should_retrain()` method
  - `_determine_update_type()` method
  - Priority-based decision logic
- [ ] Add configuration parameters to `AdaptiveRetrainingConfig`
  - Drift thresholds (SLRT, KL, Hellinger)
  - Schedule parameters (frequencies, max/min days)
- [ ] Comprehensive unit tests
  - SLRT cumsum behavior and forgetting factor
  - KL divergence computation correctness
  - Hellinger distance properties (symmetric, bounded)
  - Retraining policy decision logic and priority order
  - Schedule adherence (triggers on correct days)

**Success Criteria**:
- SLRT false positive rate ≈ theoretical α (0.005 for threshold 10.0)
- All three drift metrics agree on major market shifts
- Policy triggers full retrain within max_days constraint
- Min_days constraint prevents unnecessary thrashing

---

### Tier 3: Hierarchical Update Methods
**Goal**: Implement three-tier update strategy with cost hierarchy

Tasks:
- [ ] Implement `HMM.update_emissions_only(new_data, n_recent=30)`
  - Forward-backward with fixed transition matrix
  - M-step on emissions only
  - Parameter update via standard EM equations
  - Verify: cost ≈ 1% of full retrain
- [ ] Implement `HMM.update_transitions_only(new_data, n_recent=60)`
  - Viterbi decode state sequence
  - Count state transitions
  - Apply Dirichlet smoothing (α=0.1)
  - Normalize to probability matrix
  - Verify: cost ≈ 5% of full retrain
- [ ] Implement `HMM.full_retrain_with_informed_prior()`
  - Full Baum-Welch on recent window
  - Optional: use current params as informative prior
  - Standard initialization and convergence logic
- [ ] Add scheduling variables to HMM
  - `days_since_emission_update`, `days_since_transition_update`, `days_since_full_retrain`
  - Initialize to 0 on construction/retrain
- [ ] Unit tests for each method
  - Emission update: transition matrix unchanged, likelihood ≥ baseline
  - Transition update: emissions unchanged, probability constraints satisfied
  - Full retrain: parameters converge, likelihood improvement
  - Sequence test: repeated updates eventually match full retrain result

**Success Criteria**:
- Cost hierarchy verified: emission < transition < full retrain
- All updates monotonically increase (or maintain) likelihood
- Repeated incremental updates converge to full retrain solution
- No numerical instability or parameter explosion

---

### Tier 4: Integration & Orchestration
**Goal**: Tie all components together into unified system

Tasks:
- [ ] Enhance `HMM.update()` method
  - Call `RetrainingPolicy` to determine update type
  - Route to emission/transition/full retrain methods
  - Call `DriftDetector` to assess drift
  - Trigger early retrain if needed
- [ ] Update `TemporalController`
  - Support incremental updates in `step_through_time()`
  - Option: `use_incremental_updates: bool = False`
  - Log which update type used at each timestep
  - Verify: no lookahead bias with incremental updates
- [ ] Integration tests
  - End-to-end workflow: weekly → monthly → quarterly
  - Regime labels consistent across update boundaries
  - Forward-walking backtest with incremental updates
  - Temporal isolation maintained with incremental updates
- [ ] Documentation & examples
  - Update README with new configuration options
  - Create example: `examples/advanced/adaptive_retraining_workflow.py`
  - Document Conservative/Aggressive configuration profiles
- [ ] Final validation
  - Smoke test with real ticker data (SPY, AAPL, QQQ)
  - Verify no regressions in existing tests
  - 100% test coverage for new modules

**Success Criteria**:
- All existing tests still pass
- New tests achieve >90% coverage of new code
- No lookahead bias in temporal backtests
- Configuration system supports all three (Conservative/Moderate/Aggressive) profiles
- Documentation is clear and examples are runnable

---

## Configuration Defaults (Moderate)

**Rationale**: Balance responsiveness to market changes with interpretation stability.

```python
# Schedule: Weekly cheap, Monthly medium, Quarterly expensive
emission_update_frequency_days = 5
transition_update_frequency_days = 21
full_retrain_frequency_days = 63
max_days_without_retrain = 90
min_days_between_retrains = 14

# Drift detection: Moderate thresholds (balanced detection)
slrt_threshold = 10.0                    # α ≈ 0.005
kl_divergence_hard_threshold = 2.0       # Critical drift
kl_divergence_soft_threshold = 1.0       # Elevated drift
hellinger_distance_threshold = 0.3       # Substantial drift
slrt_forgetting_factor = 0.99

# Data weighting: 60-day halflife
exponential_decay_halflife_days = 60     # ESS ≈ 87 observations
adaptive_halflife_enabled = False

# Interpretation: Slow anchor updates
use_anchored_interpretation = True
anchor_update_rate = 0.01                # 69 days to half-adapt
regime_anchors = {
    'BULLISH': {'mean': 0.0010, 'std': 0.008},
    'BEARISH': {'mean': -0.0008, 'std': 0.012},
    'SIDEWAYS': {'mean': 0.0001, 'std': 0.006},
    'CRISIS': {'mean': -0.0030, 'std': 0.025}
}
```

---

## Files to Create

```
hidden_regime/
  monitoring/
    __init__.py
    drift_detector.py           # DriftDetector, ParameterMonitor
    retraining_policy.py        # RetrainingPolicy

  interpreter/
    anchored.py                 # AnchoredInterpreter

tests/
  test_monitoring/
    __init__.py
    test_drift_detector.py
    test_parameter_monitor.py
    test_retraining_policy.py

  test_interpreter/
    test_anchored_interpreter.py
```

---

## Files to Modify

```
hidden_regime/
  config/
    model.py                    # Add AdaptiveRetrainingConfig

  models/
    hmm.py                      # Add update_emissions_only, update_transitions_only,
                                # full_retrain_with_informed_prior, _determine_update_type

  interpreter/
    financial.py                # Integrate AnchoredInterpreter

  pipeline/
    temporal.py                 # Support incremental updates in step_through_time

tests/
  test_models/
    test_hmm.py                 # Add tests for new update methods

  test_pipeline/
    test_temporal.py            # Add incremental update tests
```

---

## Key Insights & Rationale

### Why Anchored Interpretation First?
Stable regime labels are foundational. All other updates depend on users trusting that regime definitions remain meaningful. Without anchoring, online updates cause label flipping and destroy user confidence.

### Why Three Update Types?
- Pure online learning: parameter drift, label flipping, expensive computation
- Full retrains only: infrequent updates, sudden parameter jumps, disruption
- Hierarchy: cheap frequent (weekly/monthly) + expensive periodic (quarterly) ✓
- Cost ratio: 1% + 5% + 100% = computationally efficient, stable interpretation

### Why SLRT + KL + Hellinger?
- SLRT alone: catches likelihood drops but misses parameter drift
- KL alone: sensitive to variance changes, can be brittle
- Hellinger alone: robust but may miss smaller shifts
- Combined: redundancy provides robustness, triggers are precise not brittle

### Why 60-Day Halflife?
- ESS = 87 observations: statistical weight without being noisy
- 3-month memory: reasonable for market regimes (not too short, not too long)
- Information-theoretic: maximum entropy solution given constraints
- Configurable: users can adjust for their time horizon

### Why Dirichlet Smoothing?
- Prevents zeros in transition matrix (conceptually misleading)
- α=0.1: weak prior, data dominates (α>>1 would be too strong)
- Guarantees monotonic likelihood with EM updates
- Standard practice in statistical learning

---

## Testing Strategy

### Unit Tests (Isolation)
- Drift detector SLRT, KL, Hellinger math
- Anchored interpreter regime assignment and anchor updates
- Emission/transition update methods
- Retraining policy decision logic
- Configuration dataclass validation

### Integration Tests (Components Together)
- Emission + transition + full retrain sequence
- Regime labels stable across update boundaries
- Drift detector triggers full retrain correctly
- TemporalController with incremental updates
- Forward-walking backtest correctness

### Validation (Real Data)
- Smoke test with SPY, AAPL, QQQ daily data
- Verify cost hierarchy: emission < transition < full retrain
- Check no lookahead bias in temporal backtests
- Confirm regime labels stable despite parameter drift
- Validate configuration profiles (Conservative/Moderate/Aggressive)

---

## Success Metrics

1. **Stability**: Regime labels change < 5% per week despite model updates
2. **Responsiveness**: Model responds to market shifts within 14 days
3. **Efficiency**: Weekly updates cost <5% of quarterly full retrain
4. **Robustness**: Three drift metrics agree on major shifts (>95% correlation)
5. **Correctness**: Incremental updates converge to full retrain solution
6. **Coverage**: >90% test coverage for all new modules
7. **Compliance**: No lookahead bias in temporal backtests

---

## Timeline & Dependencies

This is a phased, modular implementation plan:

- **Phase 1** (Anchored Interpreter) is independent, can start immediately
- **Phase 2** (Drift Detection) depends only on Phase 1 for testing
- **Phase 3** (Update Methods) depends on nothing, can be parallel with Phase 2
- **Phase 4** (Integration) depends on all previous phases

No dependencies on external libraries beyond what's already in use (scipy, numpy, sklearn).

---

## References

- Baum-Welch algorithm: standard EM for HMMs
- SLRT: sequential hypothesis testing (Wald, 1945)
- KL divergence: information-theoretic distance (Kullback-Leibler)
- Hellinger distance: symmetric version of KL (Hellinger, 1909)
- Dirichlet smoothing: Bayesian regularization for multinomial distributions
- Anchored interpretation: novel approach in Hidden Regime for stable labels
