#!/usr/bin/env python3
"""
Phase 4 Demo: Multi-Agent Synchronization

Demonstrates:
- Cognitive Drift Tracking (Formula 3 from paper)
- CSP Orchestrator (Algorithm 3: Cognitive Sync Pulse)
- Global State Reconciliation
- Perception Alignment Protocol
- Distributed Shared Memory with version vectors
"""

from __future__ import annotations

import sys
import time as time_module
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from agentos.sync import (
    CSPOrchestrator,
    CSPOrchestratorConfig,
    CognitiveDriftTracker,
    DriftTrackerConfig,
    DistributedSharedMemory,
    PerceptionAlignmentConfig,
    PerceptionAlignmentProtocol,
    ReconciliationConfig,
    StateReconciler,
    StoreBackend,
)
from agentos.sync.types import (
    AgentState,
    SemanticSliceVersion,
    SyncTrigger,
)
from agentos.memory.slicing.types import SemanticSlice


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_metric(name: str, value: float, max_value: float = 2.0) -> None:
    """Print a metric with visual bar."""
    bar_length = 30
    filled = int((value / max_value) * bar_length)
    filled = min(max(0, filled), bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    status = "✓" if value < 1.0 else "⚠" if value < 1.5 else "✗"
    print(f"  {name}: {bar} {value:.3f} {status}")


def main():
    """Demonstrate Phase 4 functionality."""
    print_section("AgentOS Phase 4: Multi-Agent Synchronization Demo")

    # ========================================================================
    # 1. Cognitive Drift Tracker
    # ========================================================================
    print("\n1. Cognitive Drift Tracker (Formula 3)")
    print("-" * 70)
    print("Δψᵢ(t) = ∫₀ᵗ ‖∇Φᵢ(σ,τ) - ∇S_global(τ)‖ dτ")
    print()

    drift_config = DriftTrackerConfig(
        drift_threshold=1.0,
        check_interval_sec=0.5,
        use_ema=True,
        ema_alpha=0.2,
    )
    drift_tracker = CognitiveDriftTracker(drift_config)

    # Create simulated agents with semantic gradients
    agent_gradients = {
        "agent_001": np.array([0.5, 0.3, 0.8, 0.1], dtype=np.float32),
        "agent_002": np.array([0.6, 0.4, 0.7, 0.2], dtype=np.float32),
        "agent_003": np.array([0.4, 0.5, 0.6, 0.3], dtype=np.float32),
    }

    # Register agents
    for agent_id, gradient in agent_gradients.items():
        drift_tracker.register_agent(agent_id, gradient)
        print(f"✓ Registered {agent_id}")

    # Calculate global gradient (average)
    global_gradient = np.mean(list(agent_gradients.values()), axis=0)
    print(f"\nGlobal gradient: [{global_gradient[0]:.2f}, {global_gradient[1]:.2f}, ...]")
    print()

    # Simulate drift over time
    print("Simulating cognitive drift over time...")
    print()

    for step in range(5):
        print(f"Step {step + 1}:")

        for agent_id in agent_gradients.keys():
            # Add some random drift to each agent
            drift_amount = np.random.normal(0, 0.1, 4).astype(np.float32)
            agent_gradients[agent_id] += drift_amount

            # Update drift
            drift = drift_tracker.update_drift(
                agent_id, agent_gradients[agent_id], global_gradient
            )
            metrics = drift_tracker.get_metrics(agent_id)
            print_metric(f"  {agent_id}", drift)

        # Check for sync triggers
        triggers = drift_tracker.check_sync_triggers()
        if triggers:
            print(f"\n  ⚠ Sync trigger detected: {triggers[0].value}")
        print()

    # ========================================================================
    # 2. CSP Orchestrator (Cognitive Sync Pulse)
    # ========================================================================
    print_section("2. CSP Orchestrator (Algorithm 3)")
    print("Event-driven synchronization to maintain cognitive coherence")

    csp_config = CSPOrchestratorConfig(
        min_sync_interval_ms=500,
        max_sync_interval_ms=5000,
        min_agents_for_sync=2,
        sync_on_tool_completion=True,
    )

    orchestrator = CSPOrchestrator(config=csp_config)

    # Register agents with orchestrator
    print("\nRegistering agents with CSP...")
    agent_states = []
    for i, agent_id in enumerate(agent_gradients.keys()):
        state = AgentState(
            agent_id=agent_id,
            active_slices=[f"slice_{i}_a", f"slice_{i}_b"],
            semantic_gradients=agent_gradients[agent_id],
            metadata={"role": "reasoning"},
        )
        orchestrator.register_agent(state)
        agent_states.append(state)
        print(f"✓ {agent_id}: {len(state.active_slices)} active slices")

    print()
    print(f"Total agents registered: {len(orchestrator._agents)}")

    # Trigger a sync pulse
    print("\nTriggering DRIFT_THRESHOLD sync pulse...")
    pulse = orchestrator.trigger_sync(
        trigger=SyncTrigger.DRIFT_THRESHOLD,
        source_agent_id="agent_001",
    )

    print(f"✓ Sync pulse completed: {pulse.pulse_id}")
    print(f"  Trigger: {pulse.trigger.value}")
    print(f"  Agents synced: {pulse.agents_synced}")
    print(f"  Conflicts resolved: {pulse.conflicts_resolved}")
    print(f"  Duration: {pulse.duration_ms:.2f} ms")

    # Show drift after sync
    print("\nDrift after sync (reset to 0):")
    for agent_id, metrics in drift_tracker.get_all_metrics().items():
        print_metric(f"  {agent_id}", metrics.current_drift)

    # ========================================================================
    # 3. Global State Reconciliation
    # ========================================================================
    print_section("3. Global State Reconciliation")

    reconciler = StateReconciler(
        config=ReconciliationConfig(
            conflict_strategy="latest",
            similarity_threshold=0.9,
            use_voting=False,
        )
    )

    print("\nReconciler configuration:")
    print(f"  Conflict strategy: {reconciler.config.conflict_strategy}")
    print(f"  Similarity threshold: {reconciler.config.similarity_threshold}")
    print()

    # Simulate conflicting updates
    print("Simulating conflicting slice updates...")
    print()

    global_state = orchestrator.get_global_state()

    # Create slices with same ID from different agents
    conflicting_updates = []
    for agent in agent_states[:2]:
        version = SemanticSliceVersion(
            slice_id="shared_slice_01",
            agent_id=agent.agent_id,
            version=int(time_module.time() * 1000) % 1000000,
            content=f"Content from {agent.agent_id}",
            created_at=datetime.now(),
        )
        conflicting_updates.append((agent, version))

    # Reconcile
    resolutions = reconciler.reconcile(global_state, agent_states)

    print(f"Conflicts resolved: {len(resolutions)}")
    for resolution in resolutions:
        print(f"\n  Slice: {resolution.slice_id}")
        print(f"  Strategy: {resolution.resolution_strategy}")
        print(f"  Winner: {resolution.winning_version.agent_id if resolution.winning_version else 'None'}")
        print(f"  Coherence: {resolution.coherence_score:.3f}")

    # ========================================================================
    # 4. Perception Alignment Protocol
    # ========================================================================
    print_section("4. Perception Alignment Protocol")
    print("Advantageous Timing Matching - find optimal sync windows")

    pap = PerceptionAlignmentProtocol(
        config=PerceptionAlignmentConfig(
            min_confidence=0.7,
            min_window_duration_ms=100,
            max_window_duration_ms=2000,
            noise_threshold=0.3,
        )
    )

    print("\nProtocol configuration:")
    print(f"  Min confidence: {pap.config.min_confidence}")
    print(f"  Window duration: {pap.config.min_window_duration_ms} - {pap.config.max_window_duration_ms} ms")
    print()

    # Simulate confidence updates
    print("Simulating agent confidence over time...")
    print()

    now = time_module.time()
    agent_ids = list(agent_gradients.keys())

    for i in range(10):
        timestamp = now + i * 0.1
        for agent_id in agent_ids:
            # Simulate varying confidence
            confidence = 0.5 + 0.4 * np.sin(i * 0.5 + hash(agent_id) % 10)
            pap.update_confidence(agent_id, confidence, timestamp)

    # Find sync windows
    start_time = now
    end_time = now + 1.0
    windows = pap.find_sync_windows(start_time, end_time, agent_ids)

    print(f"Found {len(windows)} sync windows:")
    for i, window in enumerate(windows[:3]):
        print(f"\n  Window {i + 1}:")
        print(f"    Duration: {window.duration_ms:.1f} ms")
        print(f"    Confidence: {window.confidence_score:.3f}")
        print(f"    Valid: {window.is_valid}")

        # Calculate quality
        quality = pap.calculate_alignment_quality(window)
        print(f"    Quality: {quality:.3f}")

    # Get best window
    best = pap.get_best_sync_window(windows)
    if best:
        print(f"\n  Best window: {best.duration_ms:.1f} ms @ {best.confidence_score:.3f} confidence")

    # ========================================================================
    # 5. Distributed Shared Memory
    # ========================================================================
    print_section("5. Distributed Shared Memory")
    print("L2/L3 backed by distributed store with version vectors")

    dsm = DistributedSharedMemory(
        backend=StoreBackend.MEMORY,
        storage_path="./data/dsm_demo",
    )

    print("\nDistributed Memory Configuration:")
    print(f"  Backend: {dsm.backend.value}")
    print(f"  Storage path: {dsm.storage_path}")
    print()

    # Create and write slices
    print("Writing slices from multiple agents...")
    print()

    slices_to_write = []
    for i, agent_id in enumerate(agent_ids):
        for j in range(2):
            slice_data = SemanticSlice(
                id=f"{agent_id}_slice_{j}",
                start_pos=i * 10 + j,
                end_pos=i * 10 + j + 5,
                tokens=["token"],
                token_ids=[100 + i * 10 + j],
                content=f"Semantic content from {agent_id} #{j}",
                density_mean=0.5,
                density_std=0.1,
            )
            slices_to_write.append((agent_id, slice_data))

    # Write slices
    successful = 0
    conflicts = 0

    for agent_id, slice_data in slices_to_write:
        result = dsm.write_slice(slice_data, agent_id)
        if result:
            successful += 1
            print(f"  ✓ {slice_data.id} written by {agent_id}")
        else:
            conflicts += 1
            print(f"  ✗ Conflict on {slice_data.id}")

    print(f"\n  Successful: {successful}, Conflicts: {conflicts}")
    print()

    # Show version vectors
    print("Version vectors (conflict detection):")
    for agent_id in agent_ids:
        vv = dsm.get_version_vector(agent_id)
        print(f"  {agent_id}: {dict(list(vv.versions.items())[:3])}...")
    print()

    # Check for conflicts
    all_conflicts = dsm.get_conflicts()
    if all_conflicts:
        print(f"Detected {len(all_conflicts)} conflicts:")
        for slice_id, outdated in all_conflicts[:3]:
            print(f"  {slice_id}: {outdated}")
    else:
        print("No conflicts detected")
    print()

    # Read slices
    print("Reading slices from distributed memory...")
    print()

    for agent_id in agent_ids:
        for slice_id in [f"{agent_id}_slice_0", f"{agent_id}_slice_1"]:
            version = dsm.read_slice(slice_id, agent_id)
            if version:
                print(f"  ✓ {slice_id}: v{version.version} by {version.agent_id}")

    # ========================================================================
    # 6. Statistics Summary
    # ========================================================================
    print_section("6. Statistics Summary")

    drift_stats = drift_tracker.get_statistics()
    print("\nDrift Tracker:")
    print(f"  Total agents: {drift_stats['total_agents']}")
    print(f"  Average drift: {drift_stats['average_drift']:.3f}")
    print(f"  Max drift: {drift_stats['max_drift']:.3f}")
    print(f"  Critical agents: {drift_stats['critical_drift_count']}")

    csp_stats = orchestrator.get_statistics()
    print("\nCSP Orchestrator:")
    print(f"  Total agents: {csp_stats['total_agents']}")
    print(f"  Total syncs: {csp_stats['total_syncs']}")
    print(f"  Average drift: {csp_stats['average_drift']:.3f}")

    reconcile_stats = reconciler.get_statistics()
    print("\nState Reconciler:")
    print(f"  Total conflicts: {reconcile_stats['total_conflicts']}")
    print(f"  Resolved: {reconcile_stats['resolved_conflicts']}")
    print(f"  Pending updates: {reconcile_stats['pending_updates']}")

    pap_stats = pap.get_statistics()
    print("\nPerception Alignment:")
    print(f"  Total agents: {pap_stats['total_agents']}")
    print(f"  Windows detected: {pap_stats['total_windows_detected']}")

    dsm_stats = dsm.get_statistics()
    print("\nDistributed Memory:")
    print(f"  Total slices: {dsm_stats['total_slices']}")
    print(f"  Total agents: {dsm_stats['total_agents']}")
    print(f"  Total conflicts: {dsm_stats['total_conflicts']}")

    # ========================================================================
    # 7. Demo Complete
    # ========================================================================
    print_section("Demo Complete!")
    print()
    print("The Multi-Agent Synchronization system successfully:")
    print("  • Tracked cognitive drift across multiple agents")
    print("  • Orchestrated cognitive sync pulses (Algorithm 3)")
    print("  • Reconciled global state with conflict resolution")
    print("  • Found optimal sync windows via Perception Alignment")
    print("  • Maintained distributed shared memory with version vectors")
    print()


def short_id(id_str: str) -> str:
    """Get short ID for display."""
    return id_str.split("_")[-1][:8] if "_" in id_str else id_str[:8]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
