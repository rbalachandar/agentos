#!/usr/bin/env python3
"""
Phase 6 Demo: Full Integration - End-to-End AgentOS System

This demo shows ALL 5 PHASES working together:
- Phase 1: Reasoning Kernel & Semantic Slicing
- Phase 2: Cognitive Memory Hierarchy (S-MMU)
- Phase 3: Cognitive Scheduler & I/O Subsystem
- Phase 4: Multi-Agent Synchronization
- Phase 5: Evaluation & Metrics

The demo creates a unified AgentOS system with multiple agents
collaborating on a task.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

from agentos import (
    AgentOS,
    AgentOSConfig,
    create_agentos,
)
from agentos.scheduler import ThreadPriority


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():
    """Run the end-to-end AgentOS demonstration."""
    print_section("AgentOS Phase 6: Full Integration Demo")
    print()
    print("This demo shows all 5 phases working together:")
    print("  • Phase 1: Reasoning Kernel processes text → semantic slices")
    print("  • Phase 2: S-MMU manages slices across L1/L2/L3")
    print("  • Phase 3: Scheduler coordinates agent threads")
    print("  • Phase 4: CSP sync keeps agents aligned")
    print("  • Phase 5: Metrics track everything")

    # ========================================================================
    # 1. Create AgentOS System
    # ========================================================================
    print_section("1. Creating AgentOS System")

    config = AgentOSConfig(
        # Model configuration (Phase 1)
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="auto",
        l1_max_tokens=512,

        # Memory configuration (Phase 2)
        l1_max_tokens_cache=1000,  # Small to demonstrate paging
        l1_max_slices=5,
        l2_max_tokens=2000,
        l2_max_slices=20,
        l3_storage_path="./data/l3_integration",

        # Scheduler configuration (Phase 3)
        scheduler_time_slice_ms=100.0,
        scheduler_use_cognitive_fidelity=True,

        # Sync configuration (Phase 4)
        enable_sync=True,
        max_agents=10,
        drift_threshold=1.0,
        min_sync_interval_ms=500.0,

        # Metrics (Phase 5)
        enable_metrics=True,
    )

    # Create system using context manager (auto load/unload kernel)
    with create_agentos(config) as system:
        print(f"✓ AgentOS system created: {system.system_id}")
        print(f"  Model: {config.model_name}")
        print(f"  Max agents: {config.max_agents}")
        print(f"  Sync enabled: {config.enable_sync}")
        print(f"  Metrics enabled: {config.enable_metrics}")

        # ========================================================================
        # 2. Spawn Multiple Agents with Different Roles
        # ========================================================================
        print_section("2. Spawning Agents")

        # Check if generation is enabled
        use_generation = "--generate" in sys.argv or "-g" in sys.argv
        if use_generation:
            print("  LLM Generation: ENABLED (produces unique agent responses)")
        else:
            print("  LLM Generation: DISABLED - demo will show system components")
            print("  Note: Agent contributions and synthesis require --generate flag")
            print("  Enable with: python phase6_demo.py --generate")
        print()

        # Create specialized agents
        researcher = system.spawn_agent(
            name="Alice",
            role="researcher",
            priority=ThreadPriority.HIGH,
            metadata={"specialty": "information gathering"},
            use_generation=use_generation,
            max_new_tokens=768,  # More tokens for detailed research
        )
        print(f"  ✓ Spawned: {researcher.agent_id} ({researcher.config.role})")

        writer = system.spawn_agent(
            name="Bob",
            role="writer",
            priority=ThreadPriority.NORMAL,
            metadata={"specialty": "content creation"},
            use_generation=use_generation,
            max_new_tokens=512,
        )
        print(f"  ✓ Spawned: {writer.agent_id} ({writer.config.role})")

        analyst = system.spawn_agent(
            name="Charlie",
            role="analyst",
            priority=ThreadPriority.NORMAL,
            metadata={"specialty": "data analysis"},
            use_generation=use_generation,
            max_new_tokens=512,
        )
        print(f"  ✓ Spawned: {analyst.agent_id} ({analyst.config.role})")

        critic = system.spawn_agent(
            name="Diana",
            role="critic",
            priority=ThreadPriority.LOW,
            metadata={"specialty": "quality review"},
            use_generation=use_generation,
            max_new_tokens=768,  # More tokens for detailed critique
        )
        print(f"  ✓ Spawned: {critic.agent_id} ({critic.config.role})")

        print()
        print(f"Total agents: {len(system.list_agents())}")

        # ========================================================================
        # 3. Individual Agent Processing (Phase 1)
        # ========================================================================
        print_section("3. Individual Agent Processing")

        sample_text = """
        The human brain contains approximately 86 billion neurons organized
        into complex networks. These neurons communicate through electrochemical
        signals across synapses, enabling cognition and consciousness.

        In contrast, artificial neural networks are computational models inspired
        by biological neurons, using mathematical functions and weighted connections
        to process information through layers.

        The key difference lies in the fundamental architecture: biological neurons
        operate in a massively parallel, analog fashion, while artificial neurons
        typically process data in discrete, sequential layers.
        """.strip()

        print(f"\nInput text ({len(sample_text)} chars)...")
        print()

        # Each agent processes the text
        for agent in system.list_agents():
            print(f"{agent.config.name} ({agent.config.role}):")
            result = agent.process(sample_text)
            print(f"  Processed → {len(agent.memory.active_slices)} semantic slices")
            print(f"  Working context: {len(agent.memory.working_context)} chars")
            print()

        # ========================================================================
        # 4. Memory Management (Phase 2)
        # ========================================================================
        print_section("4. Memory Management (S-MMU)")

        memory_stats = system.smmu.get_memory_stats()

        print("L1 Cache (Active Attention Window):")
        print(f"  Utilization: {memory_stats['l1']['utilization']:.1%}")
        print(f"  Tokens: {memory_stats['l1']['used_tokens']}/{memory_stats['l1']['max_tokens']}")
        print(f"  Slices: {memory_stats['l1']['slice_count']}")

        print()
        print("L2 RAM (Deep Context):")
        print(f"  Utilization: {memory_stats['l2']['utilization']:.1%}")
        print(f"  Tokens: {memory_stats['l2']['used_tokens']}/{memory_stats['l2']['max_tokens']}")
        print(f"  Slices: {memory_stats['l2']['slice_count']}")

        print()
        print("L3 Storage (Knowledge Base):")
        print(f"  Slices: {memory_stats['l3']['slice_count']}")

        print()
        print("Page Table:")
        print(f"  L1 entries: {memory_stats['page_table']['l1']}")
        print(f"  L2 entries: {memory_stats['page_table']['l2']}")
        print(f"  L3 entries: {memory_stats['page_table']['l3']}")

        # ========================================================================
        # 5. Scheduler Coordination (Phase 3)
        # ========================================================================
        print_section("5. Scheduler Coordination")

        # Make scheduling decision
        decision = system.scheduler.schedule()

        print("Scheduling Decision:")
        print(f"  Selected thread: {decision.selected_thread_id[:16] if decision.selected_thread_id else 'None'}...")
        print(f"  Preempt: {decision.preempt}")
        print(f"  Reason: {decision.reason}")

        print()
        scheduler_stats = system.scheduler.get_statistics()
        print("Scheduler Statistics:")
        print(f"  Total threads: {scheduler_stats['total_threads']}")
        print(f"  Active threads: {scheduler_stats['active_threads']}")
        print(f"  Ready threads: {scheduler_stats['ready_threads']}")
        print(f"  Total schedules: {scheduler_stats['total_schedules']}")

        # ========================================================================
        # 6. Multi-Agent Synchronization (Phase 4)
        # ========================================================================
        print_section("6. Multi-Agent Synchronization")

        # Update agent drift
        print("Updating agent semantic gradients...")

        import numpy as np
        for agent in system.list_agents():
            # Simulate semantic gradient
            gradient = np.random.rand(128).astype(np.float32)
            agent._semantic_gradient = gradient

            # Update drift in orchestrator
            system.csp_orchestrator.update_agent_drift(
                agent_id=agent.agent_id,
                agent_gradient=agent._semantic_gradient,
            )

        print()
        drift_stats = system.csp_orchestrator.get_drift_statistics()
        print("Drift Statistics:")
        print(f"  Total agents: {drift_stats['total_agents']}")
        print(f"  Average drift: {drift_stats['average_drift']:.3f}")
        print(f"  Max drift: {drift_stats['max_drift']:.3f}")
        print(f"  Critical agents: {drift_stats['critical_drift_count']}")

        # Trigger sync pulse
        print()
        print("Triggering Cognitive Sync Pulse...")

        from agentos.sync.types import SyncTrigger
        pulse = system.csp_orchestrator.trigger_sync(
            trigger=SyncTrigger.DRIFT_THRESHOLD,
            source_agent_id=system.list_agents()[0].agent_id,
        )

        print(f"  Sync pulse: {pulse.pulse_id}")
        print(f"  Agents synced: {pulse.agents_synced}")
        print(f"  Duration: {pulse.duration_ms:.2f} ms")

        # ========================================================================
        # 7. Collaborative Task Execution
        # ========================================================================
        print_section("7. Collaborative Task Execution")

        task = "Analyze the differences between biological and artificial neural networks"

        print(f"Task: {task}")
        print()

        result = system.collaborate(
            task=task,
            timeout_seconds=60,
            sync_interval_ms=2000,
        )

        print("Collaboration Results:")
        print(f"  Task ID: {result.task_id}")
        print(f"  Duration: {result.duration_ms:.2f} ms")
        print(f"  Agents: {len(result.agents_participated)}")
        print(f"  Sync pulses: {result.total_sync_pulses}")
        print()
        print("  Agent Contributions:")
        for agent_id, contribution in result.agent_contributions.items():
            agent = system.get_agent(agent_id)
            print(f"    {agent.config.name} ({agent.config.role}):")
            # Show full contribution
            print(f"      {contribution}")
            print()

        print()
        print("  Final Synthesis:")
        print(f"    {result.final_result}")
        print()

        # ========================================================================
        # 8. Metrics Collection (Phase 5)
        # ========================================================================
        print_section("8. Metrics Collection")

        all_metrics = system.get_statistics()

        print("System Metrics:")
        print(f"  Uptime: {all_metrics['system']['uptime_seconds']:.1f} seconds")
        print(f"  Total agents: {all_metrics['agents']['total']}")
        print()
        print("  Agents by Role:")
        for role, count in all_metrics['agents']['by_role'].items():
            print(f"    {role}: {count}")

        print()
        print("Memory Utilization:")
        memory = all_metrics.get('memory', {})
        if memory:
            l1 = memory.get('l1', {})
            l2 = memory.get('l2', {})
            l3 = memory.get('l3', {})
            print(f"  L1 Cache: {l1.get('utilization', 0):.1%} ({l1.get('used_tokens', 0)}/{l1.get('max_tokens', 0)} tokens)")
            print(f"  L2 RAM: {l2.get('utilization', 0):.1%} ({l2.get('used_tokens', 0)}/{l2.get('max_tokens', 0)} tokens)")
            print(f"  L3 Storage: {l3.get('slice_count', 0)} slices")

        print()
        print("Cognitive Drift:")
        drift_stats = system.csp_orchestrator.get_drift_statistics()
        print(f"  Average drift: {drift_stats['average_drift']:.3f}")
        print(f"  Max drift: {drift_stats['max_drift']:.3f}")
        print(f"  Critical agents: {drift_stats['critical_drift_count']}")

        print()
        print("Sync Statistics:")
        csp_stats = system.csp_orchestrator.get_statistics()
        print(f"  Total sync pulses: {csp_stats['total_syncs']}")
        print(f"  Agents tracked: {csp_stats['total_agents']}")

        print()
        print("Scheduler Statistics:")
        sched_stats = system.scheduler.get_statistics()
        print(f"  Total threads: {sched_stats['total_threads']}")
        print(f"  Active threads: {sched_stats['active_threads']}")
        print(f"  Total schedules: {sched_stats['total_schedules']}")

        # ========================================================================
        # 9. System State
        # ========================================================================
        print_section("9. System State Snapshot")

        state = system.get_system_state()

        print(f"System ID: {state['system_id']}")
        print(f"Uptime: {state['uptime_seconds']:.1f} seconds")
        print(f"Kernel loaded: {state['kernel_loaded']}")
        print()

        print("Agent States:")
        for agent_id, agent_state in state['agent_states'].items():
            agent = system.get_agent(agent_id)
            print(f"  {agent.config.name} ({agent.config.role}):")
            print(f"    Active slices: {len(agent_state.active_slices)}")
            print(f"    Semantic gradient: {agent_state.semantic_gradients.shape if agent_state.semantic_gradients is not None else 'None'}")

        # ========================================================================
        # 10. Demo Complete
        # ========================================================================
        print_section("Demo Complete!")
        print()
        print("AgentOS successfully demonstrated:")
        print("  ✓ Phase 1: Reasoning Kernel processed text → semantic slices")
        print("  ✓ Phase 2: S-MMU managed memory across L1/L2/L3")
        print("  ✓ Phase 3: Scheduler coordinated multiple agent threads")
        print("  ✓ Phase 4: CSP sync kept agents aligned")
        print("  ✓ Phase 5: Metrics tracked system performance")
        print()
        print("All 5 phases integrated into a unified multi-agent system!")
        print()


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
