# AgentOS Improvement Issues

This file tracks all issues for mitigating AgentOS disadvantages. These can be converted to GitHub issues.

## Legend
- 🔴 **Critical** - High impact, blocks usability
- 🟡 **High** - Significant impact on user experience
- 🟢 **Medium** - Nice to have features
- ⚪ **Low** - Minor improvements

---

## 🔴 CRITICAL ISSUES

### Issue #1: Cold Start Problem
**Priority:** Critical | **Estimate:** 2-3 days | **Complexity:** Medium

**Problem:** No semantic slices initially, importance scores need history to stabilize. First few turns have poor "semantic cache hit rate".

**Mitigation Strategy:**
- [ ] Implement L3 bootstrap from domain knowledge files
  - Add `l3_bootstrap_paths` to `AgentOSConfig`
  - Support loading pre-computed slices from JSONL
- [ ] Implement session restore from previous L3 state
  - Add `l3_restore_path` to `AgentOSConfig`
  - Serialize/deserialize L3 storage
- [ ] Implement adaptive importance scoring during warmup
  - Create `AdaptiveImportanceScoring` class
  - Boost recency weight during first N turns

**Files to modify:**
- `src/agentos/agentos.py` - Add config parameters
- `src/agentos/memory/smmu.py` - Add bootstrap/restore logic
- `src/agentos/memory/scoring.py` - Add adaptive scorer (NEW)

**Success criteria:**
- L3 can be pre-seeded with domain knowledge
- Sessions can be restored from previous run
- Warmup period reduces from 5 turns to 2-3 turns

---

### Issue #2: Parameter Tuning Burden
**Priority:** Critical | **Estimate:** 3-4 days | **Complexity:** High

**Problem:** 20+ sensitive parameters (l1_max_tokens, density_threshold, weights, etc.). Wrong settings cause degraded performance.

**Mitigation Strategy:**
- [ ] Implement Auto-Tuner module
  - Create `src/agentos/auto_tuner.py`
  - `suggest_memory_config(model_size, available_memory)`
  - `suggest_sync_params(agent_count, avg_latency)`
  - `suggest_for_environment(model_name, available_memory, profile)`
- [ ] Implement configuration profiles
  - Create `src/agentos/profiles.py`
  - Define: "fast", "balanced", "thorough" profiles
  - Each profile pre-tuned for typical workloads

**Files to modify:**
- `src/agentos/auto_tuner.py` (NEW)
- `src/agentos/profiles.py` (NEW)
- `src/agentos/agentos.py` - Auto-detect environment if no config

**Success criteria:**
- User can deploy with: `deploy_agentos(profile="balanced")`
- Auto-tuner produces sensible defaults for detected hardware
- Pre-defined profiles work out-of-box for common scenarios

---

## 🟡 HIGH PRIORITY ISSUES

### Issue #3: Debugging Difficulty
**Priority:** High | **Estimate:** 3-5 days | **Complexity:** High

**Problem:** Distributed state across L1/L2/L3/page-tables/drift-tracking makes debugging hard.

**Mitigation Strategy:**
- [ ] Implement Semantic State Inspector CLI tool
  - Create `tools/inspect_state.py`
  - `inspect_all(system)` - Show full system state
  - `trace_slice(slice_id)` - Trace slice through memory hierarchy
  - `inspect_agent(agent_id)` - Show agent state
  - `inspect_drift()` - Show drift statistics
- [ ] Implement Observability Hooks
  - Create `src/agentos/observability/hooks.py`
  - Hook points: `on_slice_created`, `on_slice_promoted`, `on_sync_pulse`
  - Event logging for all state transitions

**Files to modify:**
- `tools/inspect_state.py` (NEW)
- `src/agentos/observability/__init__.py` (NEW)
- `src/agentos/observability/hooks.py` (NEW)
- `src/agentos/memory/smmu.py` - Add hooks to state transitions
- `src/agentos/sync/orchestrator.py` - Add hooks to sync events

**Success criteria:**
- Can trace any slice through memory hierarchy with single command
- All state transitions emit events for observability
- Inspector CLI works on running systems

---

### Issue #4: Cascading Failures
**Priority:** High | **Estimate:** 2-3 days | **Complexity:** Medium

**Problem:** Failure in one component (Slicer → S-MMU → Scheduler → CSP) propagates and brings down entire system.

**Mitigation Strategy:**
- [x] Implement Health Checks
  - Added `health_check()` method to Kernel and S-MMU
  - Returns `HealthStatus(healthy, details)`
  - Integrated at critical points (before collaboration starts)
  - Added `/health` CLI command for manual inspection
- [ ] Implement Circuit Breaker pattern
  - **DEFERRED** - Removed unused implementation
  - Will add when specific operations show repeated failure patterns
  - Can wrap: `kernel.process()`, `smmu.get_slice()`, agent operations
- [ ] Implement Bulkhead pattern
  - Resource limits per component (memory, CPU)
  - Prevent runaway resource consumption

**Files to modify:**
- `src/agentos/common/health.py` (NEW) ✓
- `src/agentos/kernel/reasoning_kernel.py` - Added health_check() ✓
- `src/agentos/memory/smmu.py` - Added health_check() ✓
- `src/agentos/agentos.py` - Added check_system_health() ✓
- `src/agentos/cli.py` - Added /health command ✓
- `src/agentos/scheduler/cognitive_scheduler.py` - TODO: Add health_check()
- `src/agentos/sync/orchestrator.py` - TODO: Add health_check()

**Success criteria:**
- [x] Health checks detect problems before failures
- [x] Unhealthy components block operations with clear error messages
- [ ] Component failures are isolated and don't cascade (DEFERRED)
- [ ] Failed components auto-recover after timeout (DEFERRED)

---

## 🟢 MEDIUM PRIORITY ISSUES

### Issue #5: Semantic Loss from Compression
**Priority:** Medium | **Estimate:** 2-3 days | **Complexity:** Medium

**Problem:** Semantic slices compress text, potentially losing nuance and detail when demoted to L3.

**Mitigation Strategy:**
- [ ] Implement Confidence-Based Retention
  - Add `full_original` field to `SemanticSlice`
  - Store full text for high-importance slices (score > 0.8)
- [ ] Implement Multi-Resolution Storage
  - Create `MultiResolutionSlice` class
  - Store: summary (50 chars), condensed (200 chars), full
  - Return appropriate resolution based on memory level

**Files to modify:**
- `src/agentos/memory/slicing/types.py` - Add full_original field
- `src/agentos/memory/slicing/slicer.py` - Implement multi-resolution
- `src/agentos/memory/smmu.py` - Use resolution based on tier

**Success criteria:**
- High-importance slices preserve full content
- L1 uses summaries, L2 uses condensed, L3 uses full
- Memory overhead is manageable (< 2x for full content)

---

### Issue #6: Synchronization Overhead
**Priority:** Medium | **Estimate:** 1-2 days | **Complexity:** Medium

**Problem:** CSP pulses add 50-200ms overhead per sync. Fixed intervals may be too frequent or too sparse.

**Mitigation Strategy:**
- [ ] Implement Adaptive Sync Intervals
  - Create `AdaptiveSyncScheduler` class
  - Adjust interval based on recent drift rate
  - High drift → more frequent, Low drift → less frequent
- [ ] Implement Incremental Sync
  - Only sync changed slices, not full state
  - Track slice versions to detect changes

**Files to modify:**
- `src/agentos/sync/adaptive_scheduler.py` (NEW)
- `src/agentos/sync/orchestrator.py` - Use adaptive scheduler
- `src/agentos/sync/dsm.py` - Add change tracking

**Success criteria:**
- Sync interval adapts to drift (0.5x to 2x base interval)
- Incremental sync reduces data transferred by ~70%
- No loss in sync accuracy

---

## ⚪ LOW PRIORITY ISSUES

### Issue #7: Memory Pressure from Metadata
**Priority:** Low | **Estimate:** 1 day | **Complexity:** Low

**Problem:** Each slice stores 500-1000 bytes of metadata, 50-100x overhead compared to raw text.

**Mitigation Strategy:**
- [ ] Implement L3 Compression
  - Use zlib compression before storing
  - Create `CompressedL3Storage` wrapper
- [ ] Implement LRU with Tombstones
  - Keep metadata tombstone after eviction
  - Allows reconstruction without full content

**Files to modify:**
- `src/agentos/memory/storage/l3_storage.py` - Add compression
- `src/agentos/memory/lru.py` (NEW) - LRU with tombstones

**Success criteria:**
- L3 storage compressed to ~30% of original size
- Tombstones preserve metadata after eviction
- Compression/decompression is fast (< 10ms per slice)

---

### Issue #8: Model Architecture Coupling
**Priority:** Low | **Estimate:** 2-3 days | **Complexity:** High

**Problem:** Requires `output_attentions=True`, incompatible with Flash Attention 2, can't use API-based models.

**Mitigation Strategy:**
- [ ] Implement Pluggable Attention Extractors
  - Create `AttentionExtractor` protocol
  - `TransformersAttentionExtractor` - For local models
  - `APIBasedExtractor` - For API models (proxy attention)
- [ ] Implement Hybrid Mode (fallback to traditional)
  - Create `HybridAgentOS` class
  - Fall back to traditional LLM if attention unavailable

**Files to modify:**
- `src/agentos/kernel/extractors.py` (NEW)
- `src/agentos/kernel/reasoning_kernel.py` - Use pluggable extractor
- `src/agentos/hybrid.py` (NEW)

**Success criteria:**
- Can use API-based models (with proxy attention)
- Falls back gracefully when attention unavailable
- Pluggable architecture allows custom extractors

---

### Issue #9: Operational Complexity
**Priority:** Low | **Estimate:** 2-3 days | **Complexity:** Medium

**Problem:** Need to manage L3 cleanup, page table integrity, drift monitoring, memory rebalancing manually.

**Mitigation Strategy:**
- [ ] Implement Self-Healing SMMU
  - Auto-detect orphaned slices, memory leaks, corrupted page tables
  - Auto-repair with logging
- [ ] Implement Maintenance Scheduler
  - Daily L3 cleanup (remove old slices)
  - Hourly page table compaction
  - 10-minute memory rebalancing
- [ ] Implement One-Click Deployment
  - Create `tools/deploy.py`
  - Auto-detect environment and configure

**Files to modify:**
- `src/agentos/maintenance.py` (NEW)
- `tools/deploy.py` (NEW)
- `src/agentos/memory/smmu.py` - Add self-healing methods

**Success criteria:**
- Maintenance runs automatically without user intervention
- Self-healing detects and repairs common issues
- Deploy command works with single function call

---

### Issue #10: System Complexity
**Priority:** Low | **Estimate:** Ongoing | **Complexity:** Varies

**Problem:** 5 interconnected subsystems are complex to understand and navigate.

**Mitigation Strategy:**
- [ ] Improve documentation with architecture diagrams
- [ ] Add inline code examples for common patterns
- [ ] Create "How it Works" interactive tutorial
- [ ] Simplify API surface where possible

**Files to modify:**
- `docs/architecture/` (NEW) - Architecture diagrams
- `docs/tutorials/` (NEW) - Interactive tutorials
- `examples/` - Add more examples

**Success criteria:**
- New users can understand system in < 30 minutes
- Common patterns have copy-paste examples
- Architecture is visually documented

---

## Summary

| Issue | Priority | Estimate | Key Files |
|-------|----------|----------|-----------|
| #1 Cold Start | 🔴 Critical | 2-3 days | `scoring.py`, `smmu.py` |
| #2 Parameter Tuning | 🔴 Critical | 3-4 days | `auto_tuner.py`, `profiles.py` |
| #3 Debugging | 🟡 High | 3-5 days | `inspect_state.py`, `hooks.py` |
| #4 Cascading Failures | 🟡 High | 2-3 days | `circuit_breaker.py`, `health.py` |
| #5 Semantic Loss | 🟢 Medium | 2-3 days | `types.py`, `slicer.py` |
| #6 Sync Overhead | 🟢 Medium | 1-2 days | `adaptive_scheduler.py` |
| #7 Memory Pressure | ⚪ Low | 1 day | `l3_storage.py`, `lru.py` |
| #8 Model Coupling | ⚪ Low | 2-3 days | `extractors.py`, `hybrid.py` |
| #9 Operations | ⚪ Low | 2-3 days | `maintenance.py`, `deploy.py` |
| #10 Complexity | ⚪ Low | Ongoing | Documentation |

**Total Estimate:** ~20-30 days of work

---

## Conversion to GitHub Issues

To convert these to GitHub issues:

```bash
# Using GitHub CLI
gh issue create --title "Cold Start Problem" --body "See Issue #1 in ISSUES.md" --label "critical,enhancement"
gh issue create --title "Parameter Tuning Burden" --body "See Issue #2 in ISSUES.md" --label "critical,enhancement"
# ... etc
```

Or create issues manually using the templates above.
