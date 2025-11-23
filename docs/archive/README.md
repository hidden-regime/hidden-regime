# Archived Documentation

This directory contains aspirational and historical documentation that is no longer actively maintained but kept for reference.

## Files in This Archive

### **phase3_architecture.md**
Design document for Phase 3 (deferred) advanced features:
- Online learning systems
- Bayesian uncertainty quantification
- Automatic model selection
- Fat-tailed emission models
- Multi-asset regime correlation

**Status:** Deferred to Phase 4. Only refer to this if implementing Phase 3 features.

### **state_standardization.md**
Documentation for the `StateStandardizer` utility (OLD APPROACH).

**Status:** OUTDATED - Replaced by the new Interpreter component (Priority #1).
- Old approach: StateStandardizer post-processing utility
- New approach: Interpreter component (cleaner architecture)

See `/private-hidden-regime/MIGRATION_GUIDE.md` for migration instructions.

### **online_hmm.md**
Technical specification for online/incremental HMM learning.

**Status:** Deferred to Phase 4D. Reference this when implementing online learning.

---

## When to Use These Documents

**Use if:**
- Implementing Phase 3 or Phase 4 features
- Understanding historical design decisions
- Reviewing deferred feature specifications

**Don't use if:**
- Building current hidden-regime applications (use main docs/)
- Migrating from old architecture (use MIGRATION_GUIDE.md instead)
- Learning how the system works now (use current docs/)

---

*For current documentation, see the parent [docs/](../) directory.*
