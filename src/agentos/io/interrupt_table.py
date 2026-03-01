"""
Interrupt Vector Table (IVT).

Based on AgentOS paper Table 2:

The IVT maps interrupt types to their handler functions.
Each interrupt type has a unique vector address and priority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agentos.scheduler.types import InterruptType


@dataclass
class InterruptVector:
    """An entry in the Interrupt Vector Table."""

    interrupt_type: InterruptType
    vector_address: int  # Hex address (0x00-0xFF)
    priority: int  # 0-255, lower = higher priority
    name: str  # Human-readable name
    description: str  # What this interrupt does

    # Handler function
    handler: Callable[..., Any] | None = None


# Standard IVT from Table 2 of AgentOS paper
STANDARD_VECTORS: list[InterruptVector] = [
    # Tool execution interrupts (0x00-0x0F)
    InterruptVector(
        interrupt_type=InterruptType.TOOL_CALL,
        vector_address=0x01,
        priority=10,
        name="TOOL_CALL",
        description="Tool invocation requested by reasoning thread",
    ),
    InterruptVector(
        interrupt_type=InterruptType.TOOL_RESULT,
        vector_address=0x02,
        priority=5,
        name="TOOL_RESULT",
        description="Tool execution completed, result ready",
    ),
    InterruptVector(
        interrupt_type=InterruptType.TOOL_ERROR,
        vector_address=0x03,
        priority=3,  # High priority (error)
        name="TOOL_ERROR",
        description="Tool execution failed",
    ),

    # Memory management interrupts (0x10-0x1F)
    InterruptVector(
        interrupt_type=InterruptType.PAGE_FAULT,
        vector_address=0x11,
        priority=8,
        name="PAGE_FAULT",
        description="Required semantic slice not in L1 cache",
    ),
    InterruptVector(
        interrupt_type=InterruptType.COMPACTION,
        vector_address=0x12,
        priority=6,
        name="COMPACTION",
        description="Memory compaction needed",
    ),

    # Scheduling interrupts (0x20-0x2F)
    InterruptVector(
        interrupt_type=InterruptType.TIME_SLICE,
        vector_address=0x20,
        priority=7,
        name="TIME_SLICE",
        description="Thread time quantum expired",
    ),
    InterruptVector(
        interrupt_type=InterruptType.YIELD,
        vector_address=0x21,
        priority=7,
        name="YIELD",
        description="Thread voluntarily yielding CPU",
    ),
    InterruptVector(
        interrupt_type=InterruptType.PREEMPT,
        vector_address=0x22,
        priority=2,  # Very high priority
        name="PREEMPT",
        description="Higher priority thread ready to run",
    ),

    # System interrupts (0x30-0x3F)
    InterruptVector(
        interrupt_type=InterruptType.SYNC_PULSE,
        vector_address=0x30,
        priority=4,
        name="SYNC_PULSE",
        description="Cognitive sync pulse (multi-agent)",
    ),
    InterruptVector(
        interrupt_type=InterruptType.SHUTDOWN,
        vector_address=0x31,
        priority=1,  # Highest priority
        name="SHUTDOWN",
        description="System shutdown request",
    ),
    InterruptVector(
        interrupt_type=InterruptType.ERROR,
        vector_address=0x32,
        priority=0,  # Highest priority (critical)
        name="ERROR",
        description="Critical system error",
    ),
]


class InterruptVectorTable:
    """Interrupt Vector Table (IVT).

    Maps interrupt types to their handlers, following the standard
    x86-style IVT architecture but for cognitive interrupts.
    """

    def __init__(self) -> None:
        """Initialize the IVT with standard vectors."""
        # Map: vector_address -> InterruptVector
        self._vectors: dict[int, InterruptVector] = {}

        # Map: interrupt_type -> InterruptVector
        self._type_map: dict[InterruptType, InterruptVector] = {}

        # Initialize with standard vectors
        for vector in STANDARD_VECTORS:
            self._vectors[vector.vector_address] = vector
            self._type_map[vector.interrupt_type] = vector

    def register_handler(
        self, interrupt_type: InterruptType, handler: Callable[..., Any]
    ) -> bool:
        """Register a handler for an interrupt type.

        Args:
            interrupt_type: Type of interrupt
            handler: Handler function

        Returns:
            True if registered, False if interrupt type not found
        """
        vector = self._type_map.get(interrupt_type)
        if not vector:
            return False

        vector.handler = handler
        return True

    def get_vector(self, interrupt_type: InterruptType) -> InterruptVector | None:
        """Get the vector for an interrupt type.

        Args:
            interrupt_type: Type of interrupt

        Returns:
            InterruptVector if found, None otherwise
        """
        return self._type_map.get(interrupt_type)

    def get_by_address(self, address: int) -> InterruptVector | None:
        """Get a vector by its address.

        Args:
            address: Vector address

        Returns:
            InterruptVector if found, None otherwise
        """
        return self._vectors.get(address)

    def get_priority(self, interrupt_type: InterruptType) -> int:
        """Get the priority level for an interrupt type.

        Lower number = higher priority (0 = highest).

        Args:
            interrupt_type: Type of interrupt

        Returns:
            Priority level (0-255), or 255 if not found
        """
        vector = self._type_map.get(interrupt_type)
        return vector.priority if vector else 255

    def handle(self, interrupt_type: InterruptType, *args: Any, **kwargs: Any) -> Any:
        """Handle an interrupt by calling its handler.

        Args:
            interrupt_type: Type of interrupt
            *args: Arguments to pass to handler
            **kwargs: Keyword arguments to pass to handler

        Returns:
            Handler result, or None if no handler registered
        """
        vector = self._type_map.get(interrupt_type)
        if not vector or not vector.handler:
            return None

        return vector.handler(*args, **kwargs)

    def list_all(self) -> list[InterruptVector]:
        """List all interrupt vectors.

        Returns:
            List of all InterruptVector
        """
        return list(self._type_map.values())

    def get_info(self, interrupt_type: InterruptType) -> dict[str, Any] | None:
        """Get information about an interrupt type.

        Args:
            interrupt_type: Type of interrupt

        Returns:
            Dictionary with interrupt info, or None if not found
        """
        vector = self._type_map.get(interrupt_type)
        if not vector:
            return None

        return {
            "type": vector.interrupt_type,
            "address": f"0x{vector.vector_address:02X}",
            "priority": vector.priority,
            "name": vector.name,
            "description": vector.description,
            "has_handler": vector.handler is not None,
        }
