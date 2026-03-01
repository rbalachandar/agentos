#!/usr/bin/env python3
"""
Phase 3 Demo: Cognitive Scheduler & I/O Subsystem

Demonstrates:
- Multi-threaded reasoning with Cognitive Scheduler
- Tool execution via I/O Peripheral Registry
- Context switching via Reasoning Interrupt Cycle
- Interrupt Vector Table
"""

from __future__ import annotations

import sys
import time as time_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentos.scheduler import (
    CognitiveScheduler,
    Interrupt,
    InterruptType,
    SchedulerConfig,
    ThreadPriority,
    ThreadState,
)
from agentos.io import (
    PeripheralRegistry,
    PeripheralType,
    ReasoningInterruptCycle,
    register_builtins,
    STANDARD_VECTORS,
)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():
    """Demonstrate Phase 3 functionality."""
    print_section("AgentOS Phase 3: Cognitive Scheduler & I/O Demo")

    # Initialize scheduler
    print("\n1. Initializing Cognitive Scheduler...")
    scheduler = CognitiveScheduler(
        config=SchedulerConfig(
            time_slice_ms=100.0,
            use_cognitive_fidelity=True,
        )
    )
    print("✓ Scheduler created")

    # Initialize peripheral registry
    print("\n2. Initializing I/O Peripheral Registry...")
    peripherals = PeripheralRegistry()
    register_builtins(peripherals)

    # Register custom tools
    peripherals.register(
        name="text_analyzer",
        description="Analyze text for sentiment",
        peripheral_type=PeripheralType.CLASSIFIER,
        function=lambda args: f"Sentiment analysis: {args.get('text', '')[:50]}",
    )

    print(f"✓ Registered {peripherals.total_peripherals} peripherals:")
    for spec in peripherals.list_all():
        print(f"  - {spec.name} ({spec.peripheral_type.value})")

    # Initialize IVT
    print("\n3. Initializing Interrupt Vector Table...")
    from agentos.io import InterruptVectorTable

    ivt = InterruptVectorTable()
    print(f"✓ IVT created with {len(ivt.list_all())} vectors:")
    for vector in ivt.list_all():
        print(f"  - {vector.name}: 0x{vector.vector_address:02X} (priority {vector.priority})")

    # Initialize RIC
    print("\n4. Initializing Reasoning Interrupt Cycle...")
    ric = ReasoningInterruptCycle(
        scheduler=scheduler,
        peripherals=peripherals,
        ivt=ivt,
    )
    print("✓ RIC initialized")

    # Spawn multiple reasoning threads
    print_section("5. Spawning Reasoning Threads")

    thread1 = scheduler.spawn_thread(
        priority=ThreadPriority.HIGH,
        initial_slice_id="slice_001",
        metadata={"task": "analyze_data"},
    )
    print(f"✓ Spawned thread 1: {thread_id_1(thread1)} (HIGH priority)")

    thread2 = scheduler.spawn_thread(
        priority=ThreadPriority.NORMAL,
        initial_slice_id="slice_002",
        metadata={"task": "generate_report"},
    )
    print(f"✓ Spawned thread 2: {thread_id_2(thread2)} (NORMAL priority)")

    thread3 = scheduler.spawn_thread(
        priority=ThreadPriority.CRITICAL,
        initial_slice_id="slice_003",
        metadata={"task": "safety_check"},
    )
    print(f"✓ Spawned thread 3: {thread_id_3(thread3)} (CRITICAL priority)")

    # Get scheduler stats
    print_section("6. Scheduler Statistics")
    stats = scheduler.get_statistics()
    print(f"Total threads: {stats['total_threads']}")
    print(f"Active threads: {stats['active_threads']}")
    print(f"Ready threads: {stats['ready_threads']}")
    print(f"Blocked threads: {stats['blocked_threads']}")
    print()

    # Make scheduling decision
    print("7. Making Scheduling Decision...")
    decision = scheduler.schedule()
    print(f"Selected thread: {thread_id_short(decision.selected_thread_id)}")
    print(f"Preempt: {decision.preempt}")
    print(f"Reason: {decision.reason}")

    # Perform context switch
    if decision.selected_thread_id:
        print()
        print("8. Performing Context Switch...")
        switch = scheduler.context_switch(
            from_thread_id=thread1,
            to_thread_id=decision.selected_thread_id,
            reason="Initial scheduling",
        )
        print(f"✓ Switched: {thread_id_short(switch.from_thread_id)} → {thread_id_short(switch.to_thread_id)}")
        print(f"  Save time: {switch.save_time_ms:.3f} ms")
        print(f"  Restore time: {switch.restore_time_ms:.3f} ms")
        print(f"  Total time: {switch.total_time_ms:.3f} ms")

    # Simulate tool call from thread
    print_section("9. Tool Execution via Interrupt Cycle")

    print("Thread 2 needs to call calculator tool...")
    print("Triggering TOOL_CALL interrupt...")

    result = ric.trigger_tool_call(
        tool_id="calculator",
        arguments={"expression": "2 + 2 * 10"},
    )

    print(f"✓ Tool execution completed")
    print(f"  Success: {result.success}")
    print(f"  Result: {result.tool_result}")
    print(f"  Duration: {result.duration_ms:.3f} ms")
    if result.error:
        print(f"  Error: {result.error}")

    # Simulate multiple interrupts
    print_section("10. Multiple Interrupt Handling")

    # Time slice interrupt
    time_interrupt = Interrupt(
        interrupt_type=InterruptType.TIME_SLICE,
        source_thread_id=thread1,
        payload={"reason": "Quantum expired"},
    )
    result = ric.handle_interrupt(time_interrupt)
    print(f"TIME_SLICE: {result.success}, new thread: {thread_id_short(result.handler_thread_id)}")

    # Tool call interrupt
    tool_interrupt = Interrupt(
        interrupt_type=InterruptType.TOOL_CALL,
        source_thread_id=thread2,
        payload={
            "tool_id": "text_analyzer",
            "arguments": {"text": "This is a great day!"},
        },
    )
    result = ric.handle_interrupt(tool_interrupt)
    print(f"TOOL_CALL: {result.success}")
    print(f"  Result: {str(result.tool_result)[:60]}...")

    # Preempt interrupt
    preempt_interrupt = Interrupt(
        interrupt_type=InterruptType.PREEMPT,
        source_thread_id=thread3,
        payload={"reason": "Critical thread ready"},
    )
    result = ric.handle_interrupt(preempt_interrupt)
    print(f"PREEMPT: {result.success}, switched to: {thread_id_short(result.handler_thread_id)}")

    # Final statistics
    print_section("11. Final Statistics")

    stats = scheduler.get_statistics()
    print("Scheduler:")
    print(f"  Total schedules: {stats['total_schedules']}")
    print(f"  Total switches: {stats['total_switches']}")
    print(f"  Current thread: {thread_id_short(stats['current_thread_id'])}")
    print()

    ric_stats = ric.get_statistics()
    print("RIC (Interrupt Handler):")
    print(f"  Total interrupts: {ric_stats['total_interrupts']}")
    print(f"  Success rate: {ric_stats['success_rate']:.1%}")
    print(f"  Total time: {ric_stats['total_interrupt_time_ms']:.3f} ms")
    print(f"  By type: {ric_stats['interrupts_by_type']}")
    print()

    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    print("The Cognitive Scheduler successfully:")
    print("  • Managed multiple reasoning threads with different priorities")
    print("  • Executed context switches with semantic state preservation")
    print("  • Handled tool calls via the interrupt cycle")
    print("  • Routed interrupts through the IVT")
    print()


def thread_id_1(thread_id: str) -> str:
    """Get short thread ID."""
    return thread_id[:20] + "..." if len(thread_id) > 20 else thread_id


def thread_id_2(thread_id: str) -> str:
    """Get short thread ID."""
    return thread_id[:20] + "..." if len(thread_id) > 20 else thread_id


def thread_id_3(thread_id: str) -> str:
    """Get short thread ID."""
    return thread_id[:20] + "..." if len(thread_id) > 20 else thread_id


def thread_id_short(thread_id: str | None) -> str:
    """Get short thread ID."""
    if not thread_id:
        return "None"
    return thread_id.split("_")[-1][:8]


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
