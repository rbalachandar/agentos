"""
Cognitive Scheduler.

Based on AgentOS paper Section 3.3:

The Cognitive Scheduler manages multiple reasoning threads, optimizing
for "Cognitive Fidelity" rather than CPU utilization.

Key principle: High-stakes threads (safety-critical) get priority,
and context switching preserves semantic state.
"""

from __future__ import annotations

from datetime import datetime
import time
from dataclasses import dataclass
from typing import Any

from agentos.scheduler.rcb import RCBManager
from agentos.scheduler.types import (
    ContextSwitch,
    Interrupt,
    ReasoningControlBlock,
    SchedulingDecision,
    ThreadPriority,
    ThreadState,
)


@dataclass
class SchedulerConfig:
    """Configuration for the Cognitive Scheduler."""

    # Time slice (quantum) for each thread in milliseconds
    time_slice_ms: float = 100.0

    # Priority weights for scheduling decisions
    priority_weights: dict[ThreadPriority, float] = None

    # Whether to use cognitive fidelity in scheduling
    use_cognitive_fidelity: bool = True

    # Minimum cognitive fidelity threshold
    min_fidelity_threshold: float = 0.5

    def __post_init__(self):
        if self.priority_weights is None:
            self.priority_weights = {
                ThreadPriority.CRITICAL: 1000.0,
                ThreadPriority.HIGH: 100.0,
                ThreadPriority.NORMAL: 10.0,
                ThreadPriority.LOW: 1.0,
            }

    def validate(self) -> None:
        """Validate configuration."""
        if self.time_slice_ms <= 0:
            raise ValueError("time_slice_ms must be positive")
        if not (0.0 <= self.min_fidelity_threshold <= 1.0):
            raise ValueError("min_fidelity_threshold must be in [0, 1]")


class CognitiveScheduler:
    """Cognitive Scheduler for multi-threaded reasoning.

    Manages reasoning threads using semantic-aware scheduling.
    Optimizes for cognitive fidelity rather than raw throughput.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        """Initialize the cognitive scheduler.

        Args:
            config: Scheduler configuration. If None, uses defaults.
        """
        self.config = config or SchedulerConfig()
        self.config.validate()

        # RCB manager for thread state
        self.rcb_manager = RCBManager()

        # Context switch history
        self.context_switches: list[ContextSwitch] = []

        # Current running thread
        self.current_thread_id: str | None = None

        # Scheduler statistics
        self.total_schedules = 0
        self.total_switches = 0

    def spawn_thread(
        self,
        parent_id: str | None = None,
        priority: ThreadPriority = ThreadPriority.NORMAL,
        initial_slice_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Spawn a new reasoning thread.

        Args:
            parent_id: Parent thread ID (if forked)
            priority: Thread priority
            initial_slice_id: Initial semantic slice
            metadata: Additional metadata

        Returns:
            New thread ID
        """
        rcb = self.rcb_manager.create(
            parent_id=parent_id,
            priority=priority,
            initial_slice_id=initial_slice_id,
            metadata=metadata,
        )

        return rcb.thread_id

    def schedule(self, interrupt: Interrupt | None = None) -> SchedulingDecision:
        """Make a scheduling decision.

        Selects the next thread to run based on:
        1. Thread priority (critical > high > normal > low)
        2. Cognitive fidelity (prefer threads with high fidelity)
        3. Fairness (avoid starvation)

        Args:
            interrupt: Optional interrupt that triggered scheduling

        Returns:
            SchedulingDecision with selected thread
        """
        self.total_schedules += 1

        # Get current running thread
        current_rcb = None
        if self.current_thread_id:
            current_rcb = self.rcb_manager.get(self.current_thread_id)

        # Get all ready threads
        ready_threads = self.rcb_manager.get_ready_threads()

        if not ready_threads:
            # No threads ready to run
            if current_rcb and current_rcb.is_running:
                # Keep current thread running
                return SchedulingDecision(
                    selected_thread_id=self.current_thread_id,
                    preempt=False,
                    reason="No other threads ready",
                )
            else:
                # No thread to run
                return SchedulingDecision(
                    selected_thread_id=None,
                    preempt=False,
                    reason="No threads ready",
                )

        # Score each thread
        best_thread = None
        best_score = -float("inf")

        for thread in ready_threads:
            score = self._compute_thread_score(thread)

            # Bonus for being the current thread (cache affinity)
            if thread.thread_id == self.current_thread_id:
                score *= 1.1

            if score > best_score:
                best_score = score
                best_thread = thread

        # Decide whether to preempt
        preempt = False
        if best_thread and best_thread.thread_id != self.current_thread_id:
            # Preempt if new thread has significantly higher priority
            if current_rcb:
                priority_diff = (
                    self.config.priority_weights.get(best_thread.priority, 0)
                    - self.config.priority_weights.get(current_rcb.priority, 0)
                )
                if priority_diff > 10:
                    preempt = True
                elif current_rcb.time_slice_remaining <= 0:
                    preempt = True

        return SchedulingDecision(
            selected_thread_id=best_thread.thread_id if best_thread else None,
            preempt=preempt,
            reason=f"Score: {best_score:.2f}",
        )

    def _compute_thread_score(self, thread: ReasoningControlBlock) -> float:
        """Compute scheduling score for a thread.

        Higher score = more likely to be scheduled.

        Args:
            thread: Thread to score

        Returns:
            Scheduling score
        """
        # Base score from priority
        priority_weight = self.config.priority_weights.get(thread.priority, 1.0)

        # Cognitive fidelity bonus
        fidelity_bonus = 0.0
        if self.config.use_cognitive_fidelity:
            fidelity_bonus = thread.cognitive_fidelity * 50.0

        # Wait time penalty (threads waiting longer get boost)
        # Note: ReasoningControlBlock.last_run_at is a datetime.
        wait_time = (
            (datetime.now() - thread.last_run_at).total_seconds()
            if thread.last_run_at
            else 0.0
        )
        wait_bonus = min(wait_time / 10.0, 20.0)  # Cap at 20 bonus

        # Context coherence bonus
        coherence_bonus = thread.context_coherence * 10.0

        return priority_weight + fidelity_bonus + wait_bonus + coherence_bonus

    def context_switch(
        self, from_thread_id: str, to_thread_id: str, reason: str = ""
    ) -> ContextSwitch:
        """Perform a context switch between threads.

        Implements Algorithm 1 from the paper:
        1. Save semantic state of from_thread
        2. Restore semantic state of to_thread
        3. Update thread states

        Args:
            from_thread_id: Thread to switch from
            to_thread_id: Thread to switch to
            reason: Reason for context switch

        Returns:
            ContextSwitch record
        """
        start_time = time.time()

        # Save state of current thread
        from_rcb = self.rcb_manager.get(from_thread_id)
        if from_rcb:
            self.rcb_manager.update_state(from_thread_id, ThreadState.READY)
            save_time = (time.time() - start_time) * 1000
        else:
            save_time = 0.0

        # Restore state of new thread
        restore_start = time.time()
        to_rcb = self.rcb_manager.get(to_thread_id)
        if to_rcb:
            self.rcb_manager.update_state(to_thread_id, ThreadState.RUNNING)
            to_rcb.last_run_at = datetime.now()
            to_rcb.time_slice_remaining = self.config.time_slice_ms
            restore_time = (time.time() - restore_start) * 1000
        else:
            restore_time = 0.0

        # Update current thread
        self.current_thread_id = to_thread_id

        # Record context switch
        switch = ContextSwitch(
            from_thread_id=from_thread_id,
            to_thread_id=to_thread_id,
            reason=reason,
            save_time_ms=save_time,
            restore_time_ms=restore_time,
        )
        self.context_switches.append(switch)
        self.total_switches += 1

        return switch

    def yield_cpu(self, thread_id: str) -> bool:
        """Thread voluntarily yields CPU.

        Args:
            thread_id: Thread yielding

        Returns:
            True if successful
        """
        rcb = self.rcb_manager.get(thread_id)
        if not rcb or rcb.state != ThreadState.RUNNING:
            return False

        self.rcb_manager.update_state(thread_id, ThreadState.READY)
        if self.current_thread_id == thread_id:
            self.current_thread_id = None

        return True

    def block_thread(self, thread_id: str) -> bool:
        """Block a thread (waiting for I/O).

        Args:
            thread_id: Thread to block

        Returns:
            True if successful
        """
        rcb = self.rcb_manager.get(thread_id)
        if not rcb:
            return False

        self.rcb_manager.update_state(thread_id, ThreadState.BLOCKED)
        if self.current_thread_id == thread_id:
            self.current_thread_id = None

        return True

    def unblock_thread(self, thread_id: str) -> bool:
        """Unblock a thread (I/O completed).

        Args:
            thread_id: Thread to unblock

        Returns:
            True if successful
        """
        rcb = self.rcb_manager.get(thread_id)
        if not rcb or rcb.state != ThreadState.BLOCKED:
            return False

        self.rcb_manager.update_state(thread_id, ThreadState.READY)
        return True

    def terminate_thread(self, thread_id: str) -> bool:
        """Terminate a thread.

        Args:
            thread_id: Thread to terminate

        Returns:
            True if successful
        """
        rcb = self.rcb_manager.get(thread_id)
        if not rcb:
            return False

        self.rcb_manager.update_state(thread_id, ThreadState.TERMINATED)

        # Clean up if it was the current thread
        if self.current_thread_id == thread_id:
            self.current_thread_id = None

        # Remove from RCB manager (optional - could keep for stats)
        self.rcb_manager.remove(thread_id)

        return True

    def get_thread(self, thread_id: str) -> ReasoningControlBlock | None:
        """Get a thread's RCB.

        Args:
            thread_id: Thread identifier

        Returns:
            ReasoningControlBlock if found
        """
        return self.rcb_manager.get(thread_id)

    def get_statistics(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary with scheduler stats
        """
        return {
            "total_threads": self.rcb_manager.total_threads,
            "active_threads": self.rcb_manager.active_threads,
            "current_thread_id": self.current_thread_id,
            "total_schedules": self.total_schedules,
            "total_switches": self.total_switches,
            "ready_threads": len(self.rcb_manager.get_ready_threads()),
            "blocked_threads": len(self.rcb_manager.get_blocked_threads()),
        }
