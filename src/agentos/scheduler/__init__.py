"""AgentOS Scheduler Package."""

from __future__ import annotations

from agentos.scheduler.cognitive_scheduler import (
    CognitiveScheduler,
    SchedulerConfig,
)
from agentos.scheduler.rcb import RCBManager
from agentos.scheduler.types import (
    AttentionFocus,
    ContextSwitch,
    Interrupt,
    InterruptType,
    ReasoningControlBlock,
    SchedulingDecision,
    ThreadPriority,
    ThreadState,
    ToolCall,
)

__all__ = [
    # Types
    "ThreadState",
    "ThreadPriority",
    "InterruptType",
    "AttentionFocus",
    "ToolCall",
    "ReasoningControlBlock",
    "Interrupt",
    "SchedulingDecision",
    "ContextSwitch",
    # RCB Manager
    "RCBManager",
    # Scheduler
    "CognitiveScheduler",
    "SchedulerConfig",
]
