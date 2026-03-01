"""
I/O Peripheral Registry.

Based on AgentOS paper Section 3.3.1:

The I/O Peripheral Registry manages external tools and functions,
treating them as "devices" with interrupt vectors.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from agentos.scheduler.types import InterruptType, ToolCall


class PeripheralType(str, Enum):
    """Category of peripheral/tool."""

    # Information retrieval
    SEARCH = "search"  # Web search, database query
    RETRIEVAL = "retrieval"  # File read, RAG retrieval

    # Computation
    CALCULATOR = "calculator"  # Math computation
    CODE_EXECUTOR = "code_executor"  # Code execution

    # Communication
    MESSENGER = "messenger"  # Send messages, emails
    NOTIFIER = "notifier"  # Send notifications

    # Data manipulation
    FILE_WRITER = "file_writer"  # File write
    DATABASE = "database"  # Database operations

    # AI/ML tools
    EMBEDDING = "embedding"  # Get embeddings
    CLASSIFIER = "classifier"  # Classification

    # System
    SHELL = "shell"  # Shell commands
    CUSTOM = "custom"  # Custom user-defined tool


@dataclass
class PeripheralSpec:
    """Specification for an I/O peripheral (tool)."""

    tool_id: str  # Unique identifier
    name: str  # Human-readable name
    description: str  # What this tool does
    peripheral_type: PeripheralType  # Category

    # Interrupt vector for this tool type
    interrupt_vector: InterruptType = InterruptType.TOOL_CALL

    # Tool function
    function: Callable[[dict[str, Any]], Any] | None = None

    # Execution config
    is_async: bool = False  # Whether execution is async
    timeout_ms: float = 5000.0  # Timeout for execution

    # Metadata
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PeripheralRegistry:
    """Registry for I/O peripherals (external tools).

    Manages registration, lookup, and execution of tools.
    """

    def __init__(self) -> None:
        """Initialize the peripheral registry."""
        self._peripherals: dict[str, PeripheralSpec] = {}
        self._active_calls: dict[str, ToolCall] = {}

    def register(
        self,
        name: str,
        description: str,
        peripheral_type: PeripheralType,
        function: Callable[[dict[str, Any]], Any] | None = None,
        tool_id: str | None = None,
        is_async: bool = False,
        timeout_ms: float = 5000.0,
        metadata: dict[str, Any] | None = None,
    ) -> PeripheralSpec:
        """Register a new peripheral/tool.

        Args:
            name: Human-readable name
            description: What this tool does
            peripheral_type: Category of tool
            function: Tool function to execute
            tool_id: Unique ID (auto-generated if None)
            is_async: Whether execution is async
            timeout_ms: Execution timeout
            metadata: Additional metadata

        Returns:
            The registered PeripheralSpec
        """
        if tool_id is None:
            tool_id = f"tool_{peripheral_type.value}_{uuid.uuid4().hex[:8]}"

        # Check for duplicate
        if tool_id in self._peripherals:
            raise ValueError(f"Tool ID {tool_id} already registered")

        spec = PeripheralSpec(
            tool_id=tool_id,
            name=name,
            description=description,
            peripheral_type=peripheral_type,
            function=function,
            is_async=is_async,
            timeout_ms=timeout_ms,
            metadata=metadata or {},
        )

        self._peripherals[tool_id] = spec
        return spec

    def unregister(self, tool_id: str) -> bool:
        """Unregister a peripheral.

        Args:
            tool_id: Tool identifier

        Returns:
            True if unregistered, False if not found
        """
        return self._peripherals.pop(tool_id, None) is not None

    def get(self, tool_id: str) -> PeripheralSpec | None:
        """Get a peripheral by ID.

        Args:
            tool_id: Tool identifier

        Returns:
            PeripheralSpec if found, None otherwise
        """
        return self._peripherals.get(tool_id)

    def find_by_type(self, peripheral_type: PeripheralType) -> list[PeripheralSpec]:
        """Find all peripherals of a given type.

        Args:
            peripheral_type: Type to filter by

        Returns:
            List of matching PeripheralSpec
        """
        return [
            spec
            for spec in self._peripherals.values()
            if spec.peripheral_type == peripheral_type
        ]

    def find_by_name(self, name: str) -> PeripheralSpec | None:
        """Find a peripheral by name.

        Args:
            name: Name to search for

        Returns:
            PeripheralSpec if found, None otherwise
        """
        for spec in self._peripherals.values():
            if spec.name == name:
                return spec
        return None

    def execute(self, tool_call: ToolCall) -> ToolCall:
        """Execute a tool call.

        Args:
            tool_call: ToolCall to execute

        Returns:
            Updated ToolCall with result/error
        """
        from datetime import datetime

        spec = self._peripherals.get(tool_call.tool_id)
        if not spec:
            tool_call.error = f"Tool {tool_call.tool_id} not found"
            tool_call.completed_at = datetime.now()
            return tool_call

        # Set call ID if not set
        if not tool_call.call_id:
            tool_call.call_id = f"call_{uuid.uuid4().hex[:8]}"

        # Mark as started
        tool_call.started_at = datetime.now()
        self._active_calls[tool_call.call_id] = tool_call

        # Execute the function
        try:
            if spec.function:
                result = spec.function(tool_call.arguments)
                tool_call.result = result
            else:
                tool_call.error = f"Tool {tool_call.tool_id} has no function registered"

        except Exception as e:
            tool_call.error = str(e)

        # Mark as completed
        tool_call.completed_at = datetime.now()

        # Remove from active calls
        self._active_calls.pop(tool_call.call_id, None)

        return tool_call

    def create_call(
        self, tool_id: str, arguments: dict[str, Any]
    ) -> ToolCall | None:
        """Create a ToolCall for a peripheral.

        Args:
            tool_id: Tool to call
            arguments: Tool arguments

        Returns:
            ToolCall if tool exists, None otherwise
        """
        spec = self._peripherals.get(tool_id)
        if not spec:
            return None

        return ToolCall(
            tool_name=spec.name,
            tool_id=tool_id,
            arguments=arguments,
            call_id=f"call_{uuid.uuid4().hex[:8]}",
        )

    @property
    def total_peripherals(self) -> int:
        """Total number of registered peripherals."""
        return len(self._peripherals)

    @property
    def active_calls_count(self) -> int:
        """Number of currently active tool calls."""
        return len(self._active_calls)

    def list_all(self) -> list[PeripheralSpec]:
        """List all registered peripherals.

        Returns:
            List of all PeripheralSpec
        """
        return list(self._peripherals.values())


# Built-in peripheral implementations

def web_search_tool(args: dict[str, Any]) -> str:
    """Mock web search tool."""
    query = args.get("query", "")
    return f"Search results for: {query}"


def calculator_tool(args: dict[str, Any]) -> float:
    """Mock calculator tool."""
    expression = args.get("expression", "")
    try:
        # Safe evaluation (in production, use proper expression parser)
        return eval(str(expression), {"__builtins__": {}}, {})
    except:
        return 0.0


def register_builtins(registry: PeripheralRegistry) -> None:
    """Register built-in peripherals.

    Args:
        registry: Registry to register to
    """
    registry.register(
        name="web_search",
        description="Search the web for information",
        peripheral_type=PeripheralType.SEARCH,
        function=web_search_tool,
    )

    registry.register(
        name="calculator",
        description="Perform mathematical calculations",
        peripheral_type=PeripheralType.CALCULATOR,
        function=calculator_tool,
    )
