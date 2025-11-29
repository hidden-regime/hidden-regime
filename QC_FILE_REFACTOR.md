# QuantConnect LEAP Cloud Deployment Strategy for hidden-regime

## Executive Summary

**Objective:** Deploy hidden-regime CORE PIPELINE to QuantConnect LEAP cloud (Free tier, 32 KB per-file limit).

**Scope:** Core pipeline only
- HMM model (inference)
- Financial interpreter
- Signal generation
- QuantConnect integration
- EXCLUDE: case studies, visualization, analysis, monitoring

**Critical Constraint:** QuantConnect Free tier = 32 KB per-file limit
- **Blocker files that exceed limit:**
  - hmm.py (134 KB) → 4.2x limit
  - financial.py (51 KB) → 1.6x limit
  - models/utils.py (51 KB) → 1.6x limit

**Solution Approach: Inference Mode (Recommended)**
- Extract Viterbi inference from HMM (15 KB instead of 134 KB)
- Create slim interpreter with core logic (20 KB instead of 51 KB)
- Train HMM locally, export parameters as JSON
- Load pre-trained parameters in QuantConnect for inference
- All 11 templates work without modification

---

## Implementation Plan

### Phase 1: Code Refactoring (10 hours)

**1.1 Refactor hmm.py (134 KB → 15 KB)**
- Extract Viterbi inference algorithm (keep only prediction code)
- Remove Baum-Welch training code (not needed on QuantConnect)
- Remove helper methods for initialization (use pre-computed params)
- Keep: viterbi(), forward(), backward() helpers for inference only
- Remove: initialization(), train_baum_welch(), score() (training only)

**1.2 Refactor financial.py (51 KB → 20 KB)**
- Keep: core regime interpretation logic
- Remove: detailed financial analysis (win rates, drawdowns, etc.)
- Remove: indicator comparison and technical analysis
- Remove: visualization helpers
- Keep only: state-to-regime mapping, profile computation

**1.3 Handle models/utils.py (51 KB)**
- Don't deploy to QuantConnect (pre-compute parameters locally)
- Create export utility to save HMM params as JSON

### Phase 2: Parameter Export System (5 hours)

**2.1 Create parameter export utility**
- Export trained HMM as JSON: transitions, emissions, initial state
- Include regime interpretation metadata
- Version parameters for tracking

**2.2 Update training pipeline**
- Add export function to local training scripts
- Generate JSON files for each strategy

### Phase 3: QuantConnect Integration (3 hours)

**3.1 Update templates**
- Load pre-computed parameters from JSON instead of training
- Initialize HMM with loaded parameters
- All 11 templates work without modification

**3.2 Deployment package**
- Core pipeline files (HMM inference, interpreter)
- Parameter JSON files for each strategy
- All templates
- Total size: ~50-75 KB (well under 32 KB individual limits)

---

## File Modifications

**Files to refactor (in-place):**
1. `hidden_regime/models/hmm.py` - Remove training, keep inference
2. `hidden_regime/interpreter/financial.py` - Remove analysis, keep interpretation
3. `hidden_regime/models/utils.py` - Mark as local-only, create export utility

**Files to exclude from QC deployment:**
- Tests, examples, docs, visualization, analysis, monitoring, case studies
- Local-only utilities (utils.py, simulation, etc.)

**Files to create:**
1. Parameter export/import utility
2. Updated qc_deploy.sh script to include parameters

---

## Deployment Steps

### Step 1: Local Training (Weekly)

```bash
python examples/qc_parameter_export.py --ticker SPY --output params/
```
Produces: `params/basic_regime_switching.json`, `params/crisis_hedging.json`, etc.

### Step 2: Deploy to QuantConnect

```bash
./scripts/qc_deploy.sh basic_regime_switching
```

Outputs `qc_deploy/` with:
- main.py (template)
- hidden_regime/ (HMM inference + interpreter only)
- parameters.json (pre-trained HMM parameters)
- All files < 32 KB

### Step 3: Upload to QC Web IDE

1. Drag qc_deploy/ into QuantConnect terminal
2. Click "Run Backtest"
3. Logs show: "Loaded parameters", regime changes, allocations

---

## Implementation Phases

### Phase 1: Refactoring (10 hours)
Focus on reducing hmm.py and financial.py to fit 32 KB limit

### Phase 2: Parameter Export (5 hours)
Create utility to export trained HMM parameters as JSON

### Phase 3: QuantConnect Integration (3 hours)
Update templates and deployment script

**Total effort: 18 hours (~1 week)**

---

## Key Design Decisions

**Why Inference Mode?**
- Simpler implementation (only extract, don't redesign)
- Faster deployment (5-7 KB per file achievable)
- Flexible retraining (update parameters locally, upload JSON)
- All 11 templates work unchanged

**Why Refactor In-Place?**
- Single source of truth
- Local training code unchanged
- Clean separation between training/inference
- Easier maintenance long-term

**Why JSON Parameters?**
- Text format, easy to version
- Human-readable and debuggable
- Can be updated without code changes
- Supports rolling updates

---

## Critical Files to Modify

1. **hidden_regime/models/hmm.py** (134 KB → 15 KB)
   - Remove: Baum-Welch, initialization methods
   - Keep: Viterbi inference, forward/backward helpers

2. **hidden_regime/interpreter/financial.py** (51 KB → 20 KB)
   - Remove: Financial analysis (win rates, indicators, etc.)
   - Keep: Regime mapping, profile computation

3. **NEW: hidden_regime/quantconnect/parameter_loader.py**
   - Load HMM parameters from JSON
   - Initialize model from disk

4. **scripts/qc_deploy.sh** (update)
   - Include parameter JSON files
   - Verify all files < 32 KB before deployment

---

## Deployment Outcome

After implementation:
- ✅ All core pipeline files < 32 KB
- ✅ Can train locally, deploy parameters
- ✅ All 11 templates work on QuantConnect Free tier
- ✅ Weekly parameter updates via JSON
- ✅ Zero on-platform training overhead
