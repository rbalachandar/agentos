#!/usr/bin/env python3
"""
Quick Test Script for AgentOS

Tests the full system without requiring model loading.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentos import AgentOS, AgentOSConfig, create_agentos
from agentos.scheduler import ThreadPriority


def test_system_creation():
    """Test 1: System Creation"""
    print("=" * 60)
    print("TEST 1: System Creation")
    print("=" * 60)

    config = AgentOSConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_agents=5,
        enable_sync=True,
        enable_metrics=True,
    )

    system = AgentOS(config)

    print(f"✓ System created: {system.system_id}")
    print(f"  Config: {config.model_name}")
    print(f"  Max agents: {config.max_agents}")
    print()

    # Check components
    print("Components initialized:")
    print(f"  • S-MMU: {type(system.smmu).__name__}")
    print(f"  • Scheduler: {type(system.scheduler).__name__}")
    print(f"  • CSP: {type(system.csp_orchestrator).__name__}")
    print(f"  • DSM: {type(system.dsm).__name__}")
    print(f"  • Metrics: {type(system.metrics).__name__}")
    print()

    return system


def test_agent_spawning(system):
    """Test 2: Agent Spawning"""
    print("=" * 60)
    print("TEST 2: Agent Spawning")
    print("=" * 60)

    # Spawn agents with different roles
    roles = [
        ("Alice", "researcher", ThreadPriority.HIGH),
        ("Bob", "writer", ThreadPriority.NORMAL),
        ("Charlie", "analyst", ThreadPriority.NORMAL),
    ]

    for name, role, priority in roles:
        agent = system.spawn_agent(name=name, role=role, priority=priority)
        print(f"✓ Spawned: {agent.config.name} ({agent.config.role})")
        print(f"  ID: {agent.agent_id}")
        print(f"  Thread: {agent.thread_id}")
        print()

    print(f"Total agents: {len(system.list_agents())}")
    print()

    return system.list_agents()


def test_system_state(system):
    """Test 3: System State"""
    print("=" * 60)
    print("TEST 3: System State")
    print("=" * 60)

    state = system.get_system_state()

    print(f"System ID: {state['system_id']}")
    print(f"Uptime: {state['uptime_seconds']:.1f} seconds")
    print(f"Kernel loaded: {state['kernel_loaded']}")
    print()

    print("Scheduler Stats:")
    for key, value in state['scheduler_stats'].items():
        print(f"  {key}: {value}")
    print()

    print("CSP Stats:")
    csp_stats = system.csp_orchestrator.get_statistics()
    for key, value in csp_stats.items():
        print(f"  {key}: {value}")
    print()

    print("DSM Stats:")
    for key, value in state['dsm_stats'].items():
        print(f"  {key}: {value}")
    print()


def test_statistics(system):
    """Test 4: Statistics"""
    print("=" * 60)
    print("TEST 4: Statistics")
    print("=" * 60)

    stats = system.get_statistics()

    print("System Configuration:")
    print(f"  Model: {stats['system']['config']['model_name']}")
    print(f"  Max agents: {stats['system']['config']['max_agents']}")
    print(f"  Sync enabled: {stats['system']['config']['enable_sync']}")
    print()

    print("Agents:")
    print(f"  Total: {stats['agents']['total']}")
    for role, count in stats['agents']['by_role'].items():
        print(f"  {role}: {count}")
    print()

    print("Memory Utilization:")
    memory = stats['memory']
    print(f"  L1: {memory['l1']['utilization']:.1%}")
    print(f"  L2: {memory['l2']['utilization']:.1%}")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("AgentOS System Test Suite")
    print("=" * 60)
    print()

    try:
        # Test 1: System Creation
        system = test_system_creation()

        # Test 2: Agent Spawning
        agents = test_agent_spawning(system)

        # Test 3: System State
        test_system_state(system)

        # Test 4: Statistics
        test_statistics(system)

        # Summary
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print("✓ All tests passed!")
        print()
        print("To run the full demo:")
        print("  python examples/phase6_demo.py          # Fast mode (placeholders)")
        print("  python examples/phase6_demo.py --generate  # With LLM generation (slower)")
        print()
        print("To test individual phases:")
        print("  python examples/phase1_demo.py  # Semantic slicing")
        print("  python examples/phase2_demo.py  # Memory management")
        print("  python examples/phase3_demo.py  # Scheduler & I/O")
        print("  python examples/phase4_demo.py  # Multi-agent sync")
        print("  python examples/phase5_demo.py  # Metrics & viz")
        print()

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
