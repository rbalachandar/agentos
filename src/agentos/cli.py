#!/usr/bin/env python3
"""
AgentOS CLI - Interactive Multi-Agent System

A continuous running CLI application that demonstrates the AgentOS system
with multiple agents collaborating on tasks.
"""

import argparse
from agentos import AgentOS, AgentOSConfig, create_agentos
from agentos.scheduler import ThreadPriority


def print_banner():
    """Print the welcome banner."""
    print()
    print("=" * 60)
    print("  AgentOS CLI - Interactive Multi-Agent System")
    print("=" * 60)
    print()
    print("Commands:")
    print("  /help      - Show this help message")
    print("  /agents    - List all agents")
    print("  /stats     - Show system statistics")
    print("  /sync      - Trigger manual sync")
    print("  /memory    - Show memory utilization")
    print("  /quit or /exit - Quit the application")
    print()
    print("Just type your message and agents will respond!")
    print()


def print_agent_list(system):
    """Print list of all agents."""
    agents = system.list_agents()
    print()
    print(f"Active Agents ({len(agents)}):")
    print("-" * 40)
    for agent in agents:
        print(f"  • {agent.config.name} ({agent.config.role})")
        print(f"    ID: {agent.agent_id}")
        print(f"    Slices: {len(agent.memory.active_slices)}")
    print()


def print_stats(system):
    """Print system statistics."""
    stats = system.get_statistics()
    print()
    print("System Statistics:")
    print("-" * 40)
    print(f"  Uptime: {stats['system']['uptime_seconds']:.1f}s")
    print(f"  Agents: {stats['agents']['total']}")
    print()
    print("Memory:")
    mem = stats.get('memory', {})
    if mem:
        l1 = mem.get('l1', {})
        l2 = mem.get('l2', {})
        l3 = mem.get('l3', {})
        print(f"  L1 Cache: {l1.get('utilization', 0):.1%}")
        print(f"  L2 RAM: {l2.get('utilization', 0):.1%}")
        print(f"  L3 Storage: {l3.get('slice_count', 0)} slices")
    print()
    print("Cognitive Drift:")
    drift = system.csp_orchestrator.get_drift_statistics()
    print(f"  Average: {drift['average_drift']:.3f}")
    print(f"  Max: {drift['max_drift']:.3f}")
    print(f"  Sync pulses: {system.csp_orchestrator.get_statistics()['total_syncs']}")
    print()


def print_memory(system):
    """Print detailed memory utilization."""
    memory_stats = system.smmu.get_memory_stats()
    print()
    print("Memory Hierarchy:")
    print("-" * 40)
    print("L1 Cache (Active Attention Window):")
    l1 = memory_stats['l1']
    print(f"  Utilization: {l1['utilization']:.1%}")
    print(f"  Tokens: {l1['used_tokens']}/{l1['max_tokens']}")
    print(f"  Slices: {l1['slice_count']}")
    print()
    print("L2 RAM (Deep Context):")
    l2 = memory_stats['l2']
    print(f"  Utilization: {l2['utilization']:.1%}")
    print(f"  Tokens: {l2['used_tokens']}/{l2['max_tokens']}")
    print(f"  Slices: {l2['slice_count']}")
    print()
    print("L3 Storage (Knowledge Base):")
    l3 = memory_stats['l3']
    print(f"  Slices: {l3['slice_count']}")
    print(f"  Total size: {l3['total_size_bytes']} bytes")
    print()
    print("Page Table:")
    pt = memory_stats['page_table']
    print(f"  L1 entries: {pt['l1']}")
    print(f"  L2 entries: {pt['l2']}")
    print(f"  L3 entries: {pt['l3']}")
    print()


def trigger_sync(system):
    """Trigger a manual sync pulse."""
    print()
    print("Triggering Cognitive Sync Pulse...")
    from agentos.sync.types import SyncTrigger
    pulse = system.csp_orchestrator.trigger_sync(
        trigger=SyncTrigger.PERIODIC,
        source_agent_id="cli",
    )
    print(f"  Pulse ID: {pulse.pulse_id}")
    print(f"  Success: {pulse.success}")
    print(f"  Agents synced: {pulse.agents_synced}")
    if pulse.error:
        print(f"  Error: {pulse.error}")
    print()


def handle_collaboration(system, user_input, use_generation):
    """Handle user input through agent collaboration."""
    print()
    print(f"Processing: {user_input}")
    print()

    result = system.collaborate(
        task=user_input,
        timeout_seconds=120,
        sync_interval_ms=5000,
    )

    print("-" * 40)
    print("Agent Contributions:")
    print("-" * 40)
    for agent_id, contribution in result.agent_contributions.items():
        agent = system.get_agent(agent_id)
        print(f"{agent.config.name} ({agent.config.role}):")
        print(f"  {contribution}")
        print()

    print("-" * 40)
    print("Final Synthesis:")
    print("-" * 40)
    print(f"{result.final_result}")
    print()

    print(f"Duration: {result.duration_ms:.0f}ms | Sync pulses: {result.total_sync_pulses}")
    print()


def main():
    """Main CLI loop."""
    parser = argparse.ArgumentParser(
        description="AgentOS CLI - Interactive Multi-Agent System"
    )
    parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Enable LLM generation (slower but produces unique responses)"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model name to use"
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=4,
        help="Number of agents to spawn (default: 4)"
    )
    args = parser.parse_args()

    use_generation = args.generate

    # Initialize AgentOS
    config = AgentOSConfig(
        model_name=args.model,
        max_agents=args.agents,
        enable_sync=True,
        enable_metrics=True,
    )

    print("Initializing AgentOS...")
    system = AgentOS(config)

    # Spawn specialized agents
    roles = [
        ("Researcher", "researcher", ThreadPriority.HIGH),
        ("Critic", "critic", ThreadPriority.NORMAL),
    ]

    for name, role, priority in roles[:args.agents]:
        # Researcher gets more tokens for detailed analysis
        tokens = 768 if role == "researcher" else 512
        system.spawn_agent(
            name=name,
            role=role,
            priority=priority,
            use_generation=use_generation,
            max_new_tokens=tokens,
        )

    print(f"Ready! {len(system.list_agents())} agents loaded.")
    if use_generation:
        print("LLM Generation: ENABLED")
    else:
        print("LLM Generation: DISABLED - use --generate or -g to enable")
        print("Note: Agent collaboration requires LLM generation enabled")

    print_banner()

    # Main loop
    while True:
        try:
            user_input = input("You> ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["/quit", "/exit", "quit", "exit"]:
                print("Goodbye!")
                break

            elif user_input.lower() == "/help":
                print_banner()
                continue

            elif user_input.lower() == "/agents":
                print_agent_list(system)
                continue

            elif user_input.lower() == "/stats":
                print_stats(system)
                continue

            elif user_input.lower() == "/memory":
                print_memory(system)
                continue

            elif user_input.lower() == "/sync":
                trigger_sync(system)
                continue

            # Regular input - process through agents
            handle_collaboration(system, user_input, use_generation)

        except KeyboardInterrupt:
            print("\nInterrupted. Type /quit to exit or continue...")
            continue
        except EOFError:
            print("\nGoodbye!")
            break


def app():
    """Entry point for console script."""
    main()


if __name__ == "__main__":
    main()
