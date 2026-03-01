"""
Core types for the Cognitive Scheduler.

Based on AgentOS paper Section 3.3:
- Reasoning Control Block (RCB): Track state of each reasoning thread
- Thread states: READY, RUNNING, BLOCKED (waiting for I/O), TERMINATED
- Interrupt types and vectors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ThreadState(str, Enum):
    """State of a reasoning thread."""

    READY = "ready"  # Ready to run
    RUNNING = "running"  # Currently executing
    BLOCKED = "blocked"  # Waiting for I/O (tool call)
    TERMINATED = "terminated"  # Finished execution


class ThreadPriority(str, Enum):
    """Priority level for a reasoning thread.

    Based on cognitive stakes and urgency.
    """

    CRITICAL = "critical"  # Safety-critical, must complete
    HIGH = "high"  # Important but not safety-critical
    NORMAL = "normal"  # Default priority
    LOW = "low"  # Background tasks


class InterruptType(str, Enum):
    """Types of interrupts (from Table 2 of AgentOS paper)."""

    # Tool execution interrupts (0x00-0x0F)
    TOOL_CALL = "0x01"  # Tool invocation requested
    TOOL_RESULT = "0x02"  # Tool execution completed
    TOOL_ERROR = "0x03"  # Tool execution failed

    # Memory management interrupts (0x10-0x1F)
    PAGE_FAULT = "0x11"  # Required slice not in L1
    COMPACTION = "0x12"  # Memory compaction needed

    # Scheduling interrupts (0x20-0x2F)
    TIME_SLICE = "0x20"  # Quantum expired
    YIELD = "0x21"  # Thread voluntarily yielding
    PREEMPT = "0x22"  # Higher priority thread ready

    # System interrupts (0x30-0x3F)
    SYNC_PULSE = "0x30"  # Cognitive sync pulse (multi-agent)
    SHUTDOWN = "0x31"  # System shutdown request
    ERROR = "0x32"  # Critical error occurred


@dataclass
class AttentionFocus:
    """The current attention focus of a reasoning thread.

    Represents what the thread is currently "thinking about".
    """

    # Active semantic slice (what's in L1 cache)
    active_slice_id: str | None = None

    # Token position within the slice
    token_position: int = 0

    # Attention weights for current context
    attention_weights: NDArray[np.float32] | None = None

    # Context window (relevant slice IDs)
    context_slices: list[str] = field(default_factory=list)

    @property
    def context_size(self) -> int:
        """Number of slices in current context."""
        return len(self.context_slices)


@dataclass
class ToolCall:
    """A pending or active tool call.

    Represents a request to execute an external tool/function.
    """

    tool_name: str
    tool_id: str  # Peripheral ID
    arguments: dict[str, Any]

    # Call state
    call_id: str = ""  # Unique identifier for this call
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None

    @property
    def is_pending(self) -> bool:
        """Whether this call is waiting to be executed."""
        return self.started_at is None and self.completed_at is None

    @property
    def is_running(self) -> bool:
        """Whether this call is currently executing."""
        return self.started_at is not None and self.completed_at is None

    @property
    def is_complete(self) -> bool:
        """Whether this call has completed."""
        return self.completed_at is not None

    @property
    def duration_ms(self) -> float | None:
        """Execution duration in milliseconds."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds() * 1000.0
        return None


@dataclass
class ReasoningControlBlock:
    """Reasoning Control Block (RCB).

    Analogous to a Process Control Block (PCB) in operating systems,
    but for reasoning threads instead of processes.

    Tracks all state needed to pause, resume, and manage a reasoning thread.
    """

    # Thread identification
    thread_id: str
    parent_id: str | None = None  # Parent thread (if forked)

    # Thread state and priority
    state: ThreadState = ThreadState.READY
    priority: ThreadPriority = ThreadPriority.NORMAL

    # Attention focus
    attention_focus: AttentionFocus = field(default_factory=AttentionFocus)

    # Active tool calls (stack for nested calls)
    tool_call_stack: list[ToolCall] = field(default_factory=list)

    # Semantic stack depth (for nested reasoning)
    semantic_stack_depth: int = 0

    # Timing and scheduling
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    last_run_at: datetime | None = None
    total_runtime_ms: float = 0.0
    time_slice_remaining: float = 100.0  # ms

    # Cognitive metrics
    cognitive_fidelity: float = 1.0  # 0-1, how "focused" is the thread
    context_coherence: float = 1.0  # 0-1, how coherent is the context

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """Whether thread is ready to run."""
        return self.state == ThreadState.READY

    @property
    def is_running(self) -> bool:
        """Whether thread is currently running."""
        return self.state == ThreadState.RUNNING

    @property
    def is_blocked(self) -> bool:
        """Whether thread is blocked (waiting for I/O)."""
        return self.state == ThreadState.BLOCKED

    @property
    def is_terminated(self) -> bool:
        """Whether thread has finished."""
        return self.state == ThreadState.TERMINATED

    @property
    def has_pending_tool_call(self) -> bool:
        """Whether thread has a pending tool call."""
        return len(self.tool_call_stack) > 0 and self.tool_call_stack[-1].is_pending

    @property
    def is_waiting_for_tool(self) -> bool:
        """Whether thread is waiting for tool completion."""
        return len(self.tool_call_stack) > 0 and self.tool_call_stack[-1].is_running


@dataclass
class Interrupt:
    """An interrupt signal.

    Represents an event that requires scheduler attention.
    """

    interrupt_type: InterruptType
    source_thread_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Optional payload
    payload: dict[str, Any] = field(default_factory=dict)

    # Handler result
    handled: bool = False
    result: Any = None


@dataclass
class SchedulingDecision:
    """Result of a scheduling decision."""

    selected_thread_id: str | None = None
    preempt: bool = False
    reason: str = ""


@dataclass
class ContextSwitch:
    """A record of a context switch operation."""

    from_thread_id: str
    to_thread_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""
    save_time_ms: float = 0.0
    restore_time_ms: float = 0.0

    @property
    def total_time_ms(self) -> float:
        """Total time for context switch."""
        return self.save_time_ms + self.restore_time_ms
