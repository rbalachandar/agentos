"""AgentOS I/O Subsystem Package."""

from __future__ import annotations

from agentos.io.interrupt_handler import (
    PerceptionAlignmentConfig,
    ReasoningInterruptCycle,
)
from agentos.io.interrupt_table import (
    InterruptVector,
    InterruptVectorTable,
    STANDARD_VECTORS,
)
from agentos.io.peripherals import (
    PeripheralRegistry,
    PeripheralSpec,
    PeripheralType,
    calculator_tool,
    register_builtins,
    web_search_tool,
)

__all__ = [
    # Types
    "PeripheralType",
    "PeripheralSpec",
    "InterruptVector",
    # Registry
    "PeripheralRegistry",
    "register_builtins",
    # Built-ins
    "web_search_tool",
    "calculator_tool",
    # IVT
    "InterruptVectorTable",
    "STANDARD_VECTORS",
    # RIC
    "ReasoningInterruptCycle",
    "PerceptionAlignmentConfig",
]
