"""
Reasoning Control Block (RCB) Manager.

Manages the lifecycle of reasoning threads, creating, tracking,
and cleaning up RCBs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
import uuid

from agentos.scheduler.types import (
    AttentionFocus,
    ReasoningControlBlock,
    ThreadPriority,
    ThreadState,
)


class RCBManager:
    """Manages Reasoning Control Blocks for all threads.

    The RCB manager is responsible for:
    - Creating new RCBs when threads are spawned
    - Tracking RCB state during thread lifecycle
    - Cleaning up RCBs when threads terminate
    """

    def __init__(self) -> None:
        """Initialize the RCB manager."""
        self._rcbs: dict[str, ReasoningControlBlock] = {}
        self._thread_counter = 0

    def create(
        self,
        parent_id: str | None = None,
        priority: ThreadPriority = ThreadPriority.NORMAL,
        initial_slice_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ReasoningControlBlock:
        """Create a new reasoning thread.

        Args:
            parent_id: Parent thread ID (if forked)
            priority: Thread priority level
            initial_slice_id: Initial semantic slice to focus on
            metadata: Additional metadata

        Returns:
            The created ReasoningControlBlock
        """
        self._thread_counter += 1
        thread_id = f"thread_{self._thread_counter}_{uuid.uuid4().hex[:8]}"

        # Create attention focus
        attention_focus = AttentionFocus(
            active_slice_id=initial_slice_id,
            context_slices=[initial_slice_id] if initial_slice_id else [],
        )

        # Create RCB
        rcb = ReasoningControlBlock(
            thread_id=thread_id,
            parent_id=parent_id,
            priority=priority,
            state=ThreadState.READY,
            attention_focus=attention_focus,
            metadata=metadata or {},
        )

        self._rcbs[thread_id] = rcb
        return rcb

    def get(self, thread_id: str) -> ReasoningControlBlock | None:
        """Get an RCB by thread ID.

        Args:
            thread_id: Thread identifier

        Returns:
            ReasoningControlBlock if found, None otherwise
        """
        return self._rcbs.get(thread_id)

    def update_state(self, thread_id: str, new_state: ThreadState) -> bool:
        """Update the state of a thread.

        Args:
            thread_id: Thread identifier
            new_state: New thread state

        Returns:
            True if updated, False if thread not found
        """
        rcb = self._rcbs.get(thread_id)
        if not rcb:
            return False

        # Update timestamps for state transitions
        if new_state == ThreadState.RUNNING and rcb.state != ThreadState.RUNNING:
            rcb.last_run_at = datetime.now()
        elif new_state == ThreadState.TERMINATED:
            # Could record termination time here
            pass

        rcb.state = new_state
        return True

    def update_runtime(self, thread_id: str, runtime_ms: float) -> bool:
        """Add runtime to a thread's total.

        Args:
            thread_id: Thread identifier
            runtime_ms: Runtime to add in milliseconds

        Returns:
            True if updated, False if thread not found
        """
        rcb = self._rcbs.get(thread_id)
        if not rcb:
            return False

        rcb.total_runtime_ms += runtime_ms
        rcb.time_slice_remaining -= runtime_ms
        return True

    def add_tool_call(self, thread_id: str, tool_call: Any) -> bool:
        """Add a tool call to a thread's stack.

        Args:
            thread_id: Thread identifier
            tool_call: ToolCall to add

        Returns:
            True if added, False if thread not found
        """
        rcb = self._rcbs.get(thread_id)
        if not rcb:
            return False

        rcb.tool_call_stack.append(tool_call)
        return True

    def pop_tool_call(self, thread_id: str) -> Any | None:
        """Pop the most recent tool call from a thread's stack.

        Args:
            thread_id: Thread identifier

        Returns:
            ToolCall if found, None otherwise
        """
        rcb = self._rcbs.get(thread_id)
        if not rcb or not rcb.tool_call_stack:
            return None

        return rcb.tool_call_stack.pop()

    def get_top_tool_call(self, thread_id: str) -> Any | None:
        """Get the most recent tool call without popping.

        Args:
            thread_id: Thread identifier

        Returns:
            ToolCall if found, None otherwise
        """
        rcb = self._rcbs.get(thread_id)
        if not rcb or not rcb.tool_call_stack:
            return None

        return rcb.tool_call_stack[-1]

    def remove(self, thread_id: str) -> ReasoningControlBlock | None:
        """Remove an RCB (when thread terminates).

        Args:
            thread_id: Thread identifier

        Returns:
            Removed RCB if found, None otherwise
        """
        return self._rcbs.pop(thread_id, None)

    def get_all_by_state(self, state: ThreadState) -> list[ReasoningControlBlock]:
        """Get all RCBs in a given state.

        Args:
            state: Thread state to filter by

        Returns:
            List of ReasoningControlBlock in the given state
        """
        return [rcb for rcb in self._rcbs.values() if rcb.state == state]

    def get_ready_threads(self) -> list[ReasoningControlBlock]:
        """Get all threads ready to run."""
        return self.get_all_by_state(ThreadState.READY)

    def get_blocked_threads(self) -> list[ReasoningControlBlock]:
        """Get all threads blocked on I/O."""
        return self.get_all_by_state(ThreadState.BLOCKED)

    def get_running_thread(self) -> ReasoningControlBlock | None:
        """Get the currently running thread (should be only one)."""
        running = self.get_all_by_state(ThreadState.RUNNING)
        return running[0] if running else None

    @property
    def total_threads(self) -> int:
        """Total number of threads."""
        return len(self._rcbs)

    @property
    def active_threads(self) -> int:
        """Number of active threads (not terminated)."""
        return len([r for r in self._rcbs.values() if r.state != ThreadState.TERMINATED])
