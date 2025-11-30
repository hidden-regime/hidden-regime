# QuantConnect Regime Allocation Migration Guide

**This guide helps you migrate from string-based regime allocations to type-safe `RegimeType` enum-based allocations.**

## What Changed?

Hidden Regime v1.2+ requires **breaking change**: All QuantConnect algorithms must use `RegimeTypeAllocations` with the `RegimeType` enum instead of string-based regime allocation dictionaries.

### Why This Change?

For 4+ state HMMs, the interpreter **discovers regime names dynamically** from data. For example:

- **3-state model**: Usually discovers "Bull", "Bear", "Sideways"
- **4-state model**: Might discover "Bull", "Flat", "Bear", "Crisis" (different names depending on data!)
- **After retraining**: Could discover "Uptrend", "Neutral", "Downtrend", "Crisis" (same semantics, different names)

**Problem with string-based allocations:**
```python
# This approach has a critical flaw:
regime_allocations = {
    "Bull": 1.0,
    "Bear": 0.0,
    "Sideways": 0.5,
    "Choppy": 0.0  # This regime name may never be discovered!
}
```

If the interpreter discovers "Uptrend" instead of "Bull", your "Bull" allocation is never used. If the model retrains and discovers different names, your allocations break.

**Solution with RegimeType enum:**
```python
# This approach is stable across all discoveries:
regime_allocations = RegimeTypeAllocations(
    bullish=1.0,      # Maps to BULLISH regime type (not name)
    bearish=0.0,
    sideways=0.5,
    crisis=0.0,
    mixed=0.25
)
```

The `RegimeType` enum (`BULLISH`, `BEARISH`, `SIDEWAYS`, `CRISIS`, `MIXED`) is **always** output by the interpreter, independent of discovered names. This ensures temporal stability.

---

## Migration Checklist

### Step 1: Import RegimeTypeAllocations

**Before:**
```python
from hidden_regime.quantconnect import HiddenRegimeAlgorithm
```

**After:**
```python
from hidden_regime.quantconnect import HiddenRegimeAlgorithm
from hidden_regime.quantconnect.config import RegimeTypeAllocations
```

### Step 2: Replace Allocation Dictionary

**Before:**
```python
self.initialize_regime_detection(
    ticker="SPY",
    n_states=3,
    regime_allocations={
        "Bull": 1.0,
        "Bear": 0.0,
        "Sideways": 0.5,
        "Crisis": 0.0,
    }
)
```

**After:**
```python
allocations = RegimeTypeAllocations(
    bullish=1.0,
    bearish=0.0,
    sideways=0.5,
    crisis=0.0,
    mixed=0.25  # New! Can now handle 5 regime types
)

self.initialize_regime_detection(
    ticker="SPY",
    n_states=3,
    regime_type_allocations=allocations
)
```

### Step 3: Update Parameter Names

If you were using QuantConnect parameters for allocations:

**Before:**
```python
bull_allocation = float(self.GetParameter("bull_allocation", 1.0))
bear_allocation = float(self.GetParameter("bear_allocation", 0.0))
sideways_allocation = float(self.GetParameter("sideways_allocation", 0.5))
```

**After:**
```python
regime_allocations = RegimeTypeAllocations(
    bullish=float(self.GetParameter("bullish_allocation", 1.0)),
    bearish=float(self.GetParameter("bearish_allocation", 0.0)),
    sideways=float(self.GetParameter("sideways_allocation", 0.5)),
    crisis=float(self.GetParameter("crisis_allocation", 0.0)),
    mixed=float(self.GetParameter("mixed_allocation", 0.25))
)
```

---

## Migration Examples

### Example 1: Simple 3-State Strategy

**Before (Old - Will Error):**
```python
class MyStrategy(HiddenRegimeAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # This will now RAISE ValueError
        self.initialize_regime_detection(
            ticker="SPY",
            n_states=3,
            regime_allocations={
                "Bull": 1.0,
                "Bear": 0.0,
                "Sideways": 0.5,
            }
        )
```

**After (New - Works):**
```python
from hidden_regime.quantconnect.config import RegimeTypeAllocations

class MyStrategy(HiddenRegimeAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        allocations = RegimeTypeAllocations(
            bullish=1.0,
            bearish=0.0,
            sideways=0.5,
            crisis=0.0,
            mixed=0.25
        )

        self.initialize_regime_detection(
            ticker="SPY",
            n_states=3,
            regime_type_allocations=allocations
        )
```

### Example 2: 4-State Strategy (Data-Driven Discovery)

For 4-state models, the interpreter discovers regime names dynamically. Using `RegimeTypeAllocations` ensures stability:

**Before (Fragile):**
```python
# This doesn't work reliably with 4-state models!
self.initialize_regime_detection(
    ticker="SPY",
    n_states=4,
    regime_allocations={
        "Bull": 1.0,
        "Bear": 0.0,
        "Sideways": 0.5,
        "Choppy": 0.1,  # May not be discovered!
    }
)
```

**After (Robust):**
```python
allocations = RegimeTypeAllocations(
    bullish=1.0,
    bearish=0.0,
    sideways=0.5,
    crisis=0.0,
    mixed=0.25  # Handles "unclear" regimes
)

self.initialize_regime_detection(
    ticker="SPY",
    n_states=4,
    regime_type_allocations=allocations
)
```

Even if the interpreter discovers "Uptrend", "Flat", "Downtrend", "Crisis" (different names), the allocations remain correct because they're mapped to `RegimeType`.

### Example 3: Using Factory Methods

You can use pre-configured allocation strategies:

```python
# Conservative strategy
allocations = RegimeTypeAllocations.create_conservative()

# Aggressive strategy
allocations = RegimeTypeAllocations.create_aggressive()

# Market-neutral strategy
allocations = RegimeTypeAllocations.create_market_neutral()

self.initialize_regime_detection(
    ticker="SPY",
    n_states=3,
    regime_type_allocations=allocations
)
```

### Example 4: QuantConnect Parameters

If you're deploying to QuantConnect cloud and want to parameterize allocations:

```python
allocations = RegimeTypeAllocations(
    bullish=float(self.GetParameter("bullish_allocation", 1.0)),
    bearish=float(self.GetParameter("bearish_allocation", 0.0)),
    sideways=float(self.GetParameter("sideways_allocation", 0.5)),
    crisis=float(self.GetParameter("crisis_allocation", 0.0)),
    mixed=float(self.GetParameter("mixed_allocation", 0.25))
)

self.initialize_regime_detection(
    ticker="SPY",
    n_states=3,
    regime_type_allocations=allocations
)
```

Then in QuantConnect UI, set parameters:
- `bullish_allocation: 1.0`
- `bearish_allocation: 0.0`
- `sideways_allocation: 0.5`
- `crisis_allocation: 0.0`
- `mixed_allocation: 0.25`

---

## RegimeType Enum Reference

All allocations must use these `RegimeType` values:

| RegimeType | Meaning | Typical Characteristics |
|-----------|---------|----------------------|
| `BULLISH` | Strong positive returns | High mean return, moderate volatility |
| `bearish` | Negative returns | Low/negative returns, often high volatility |
| `SIDEWAYS` | Low returns | Low volatility, mean near zero |
| `CRISIS` | Extreme volatility | Very negative returns, very high volatility |
| `MIXED` | Unclear regime | Ambiguous characteristics, defensive allocation |

---

## Troubleshooting

### Error: "String-based regime allocations are no longer supported"

**Problem:**
```
ValueError: String-based regime allocations are no longer supported.
Use RegimeTypeAllocations instead:
  from hidden_regime.quantconnect.config import RegimeTypeAllocations
  allocations = RegimeTypeAllocations(bullish=1.0, bearish=0.0, ...)
See migration guide: docs/guides/quantconnect_migration.md
```

**Solution:** Follow the migration steps above. Replace your string dict with `RegimeTypeAllocations`.

### Error: "regime_type not in metadata"

**Problem:** Your pipeline isn't outputting `regime_type` field.

**Solution:** Ensure you're using the current interpreter. The interpreter must output `regime_type` as `RegimeType` enum. This is handled automatically by the framework.

### Allocations don't match discovered regime names

**Problem:** You see "Uptrend" or "Flat" in regime names, but your allocations still work.

**This is correct behavior!** The allocations are mapped via `RegimeType`, not regime names. Even if names change, allocations remain stable.

---

## Architecture Background

### Why Enum-Based Is Better

1. **Type Safety:** Compiler/linter can catch mistakes
2. **Temporal Stability:** Names can change, enum values remain constant
3. **Clear Intent:** `bullish=0.8` is clearer than `"Bull": 0.8`
4. **Extensibility:** Easy to add new regime types in future

### Pipeline Architecture

```
Data → Observation → Model (states 0, 1, 2...)
  → Interpreter (BULLISH, BEARISH, SIDEWAYS, CRISIS, MIXED)
  → Signal Adapter (allocations via RegimeType enum)
  → Strategy (SetHoldings)
```

The `RegimeType` enum ensures the interpreter's output deterministically maps to allocations, regardless of discovered regime names.

---

## Questions?

If you encounter issues:

1. Check the `test_regime_allocations.py` tests for working examples
2. Review the `basic_regime_switching.py` template for a canonical implementation
3. Ensure `regime_type` is in your pipeline output
4. Verify `RegimeTypeAllocations` values are 0.0-2.0 (allowing leverage/shorting)

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Allocation type | `Dict[str, float]` | `RegimeTypeAllocations` (dataclass) |
| Keys | Regime names (strings) | `RegimeType` enum values |
| Stability | Breaks on name changes | Stable across retrains |
| 4-state support | Unreliable | Fully supported |
| Type checking | None | Full type hints |

Migration is straightforward: replace the dict with `RegimeTypeAllocations` object, update parameter names, and you're done!
