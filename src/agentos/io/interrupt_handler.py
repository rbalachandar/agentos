"""
Reasoning Interrupt Cycle (RIC).

Based on AgentOS paper Algorithm 1 (Section 3.3.2):

The RIC handles context switching when a reasoning thread needs to
invoke an external tool. It saves semantic state, blocks the thread,
executes the tool, and then restores state.

Key innovation: Perception Alignment - filter/recode tool output
to match the cognitive model's expectations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agentos.io.interrupt_table import InterruptVectorTable
from agentos.io.peripherals import PeripheralRegistry
from agentos.scheduler.cognitive_scheduler import CognitiveScheduler
from agentos.scheduler.types import (
    ContextSwitch,
    Interrupt,
    InterruptType,
    ThreadState,
    ToolCall,
)

if TYPE_CHECKING:
    from agentos.kernel.reasoning_kernel import ReasoningKernel

logger = logging.getLogger(__name__)


@dataclass
class PerceptionAlignmentConfig:
    """Configuration for perception alignment."""

    # Whether to filter tool output
    enable_filtering: bool = True

    # Maximum length of tool output (in tokens)
    max_output_tokens: int = 1000

    # Whether to recode tool output into semantic slice
    enable_recoding: bool = True

    # Threshold for semantic similarity
    similarity_threshold: float = 0.7


@dataclass
class InterruptResult:
    """Result of handling an interrupt."""

    success: bool
    interrupt_type: InterruptType
    handler_thread_id: str | None = None
    context_switch: ContextSwitch | None = None
    tool_result: Any = None
    error: str | None = None
    duration_ms: float = 0.0


class ReasoningInterruptCycle:
    """Reasoning Interrupt Cycle (RIC) handler.

    Implements Algorithm 1 from the paper:
    1. Receive interrupt (e.g., tool call)
    2. Save semantic state of current thread
    3. Block current thread
    4. Execute handler (tool execution)
    5. Apply Perception Alignment to output
    6. Unblock thread with result
    7. Schedule next thread
    """

    def __init__(
        self,
        scheduler: CognitiveScheduler,
        peripherals: PeripheralRegistry,
        ivt: InterruptVectorTable,
        perception_config: PerceptionAlignmentConfig | None = None,
        kernel: "ReasoningKernel | None" = None,
    ) -> None:
        """Initialize the RIC.

        Args:
            scheduler: Cognitive scheduler for thread management
            peripherals: Peripheral registry for tool execution
            ivt: Interrupt vector table
            perception_config: Perception alignment config
            kernel: Optional ReasoningKernel for state management during interrupts
        """
        self.scheduler = scheduler
        self.peripherals = peripherals
        self.ivt = ivt
        self.perception_config = perception_config or PerceptionAlignmentConfig()
        self.kernel = kernel

        # Interrupt history
        self.interrupt_history: list[InterruptResult] = []

    def set_kernel(self, kernel: "ReasoningKernel") -> None:
        """Set the ReasoningKernel reference.

        This allows the RIC to interrupt/resume the kernel during tool calls.

        Args:
            kernel: The ReasoningKernel instance
        """
        self.kernel = kernel
        logger.info("RIC: ReasoningKernel reference set")

    def handle_interrupt(self, interrupt: Interrupt) -> InterruptResult:
        """Handle an interrupt using the interrupt cycle.

        Args:
            interrupt: Interrupt to handle

        Returns:
            InterruptResult with outcome
        """
        start_time = time.time()

        result = InterruptResult(
            success=False,
            interrupt_type=interrupt.interrupt_type,
            handler_thread_id=None,
            error=None,
        )

        try:
            if interrupt.interrupt_type == InterruptType.TOOL_CALL:
                result = self._handle_tool_call(interrupt)
            elif interrupt.interrupt_type == InterruptType.TOOL_RESULT:
                result = self._handle_tool_result(interrupt)
            elif interrupt.interrupt_type == InterruptType.TIME_SLICE:
                result = self._handle_time_slice(interrupt)
            elif interrupt.interrupt_type == InterruptType.PREEMPT:
                result = self._handle_preempt(interrupt)
            elif interrupt.interrupt_type == InterruptType.YIELD:
                result = self._handle_yield(interrupt)
            else:
                # Try IVT handler
                handler_result = self.ivt.handle(interrupt.interrupt_type, interrupt)
                result.success = handler_result is not None

        except Exception as e:
            result.error = str(e)

        result.duration_ms = (time.time() - start_time) * 1000
        self.interrupt_history.append(result)

        return result

    def _handle_tool_call(self, interrupt: Interrupt) -> InterruptResult:
        """Handle a tool call interrupt.

        Algorithm 1 steps:
        1. Save semantic state
        2. Block current thread
        3. Execute tool
        4. Apply perception alignment
        5. Unblock thread with result

        Args:
            interrupt: Tool call interrupt

        Returns:
            InterruptResult
        """
        # Get current thread
        current_thread_id = self.scheduler.current_thread_id
        if not current_thread_id:
            return InterruptResult(
                success=False,
                interrupt_type=interrupt.interrupt_type,
                error="No current thread to handle tool call",
            )

        # Extract tool call info from payload
        tool_id = interrupt.payload.get("tool_id")
        arguments = interrupt.payload.get("arguments", {})

        if not tool_id:
            return InterruptResult(
                success=False,
                interrupt_type=interrupt.interrupt_type,
                error="No tool_id in interrupt payload",
            )

        # If tool_id is not found directly, try to find by name
        spec = self.peripherals.get(tool_id)
        if not spec:
            # Try finding by name
            spec = self.peripherals.find_by_name(tool_id)
            if spec:
                tool_id = spec.tool_id

        # Create tool call
        tool_call = self.peripherals.create_call(tool_id, arguments)
        if not tool_call:
            return InterruptResult(
                success=False,
                interrupt_type=interrupt.interrupt_type,
                error=f"Tool {tool_id} not found",
            )

        # Step 1: Save semantic state (via ReasoningKernel if available)
        saved_kernel_state = None
        if self.kernel is not None and self.kernel.kernel_state.value != "uninitialized":
            saved_kernel_state = self.kernel.interrupt(
                reason=f"tool_call:{tool_id}"
            )

        save_start = time.time()

        # Step 2: Block current thread
        self.scheduler.block_thread(current_thread_id)

        # Add tool call to thread's RCB
        self.scheduler.rcb_manager.add_tool_call(current_thread_id, tool_call)

        save_time = (time.time() - save_start) * 1000

        # Step 3: Execute tool
        exec_start = time.time()
        tool_call = self.peripherals.execute(tool_call)
        exec_time = (time.time() - exec_start) * 1000

        # Step 4: Apply Perception Alignment
        aligned_result = self._apply_perception_alignment(tool_call)

        # Step 5: Unblock thread with result
        self.scheduler.unblock_thread(current_thread_id)

        # Update tool call with aligned result
        tool_call.result = aligned_result

        # Resume kernel if it was interrupted
        if self.kernel is not None and saved_kernel_state is not None:
            self.kernel.resume(saved_kernel_state)

        # Create result
        return InterruptResult(
            success=tool_call.error is None,
            interrupt_type=interrupt.interrupt_type,
            handler_thread_id=current_thread_id,
            tool_result=aligned_result,
            error=tool_call.error,
            duration_ms=save_time + exec_time,
        )

    def _handle_tool_result(self, interrupt: Interrupt) -> InterruptResult:
        """Handle a tool result interrupt.

        Called when an async tool completes.

        Args:
            interrupt: Tool result interrupt

        Returns:
            InterruptResult
        """
        # Extract result info
        thread_id = interrupt.payload.get("thread_id")
        call_id = interrupt.payload.get("call_id")
        result = interrupt.payload.get("result")

        if not thread_id or not call_id:
            return InterruptResult(
                success=False,
                interrupt_type=interrupt.interrupt_type,
                error="Missing thread_id or call_id in payload",
            )

        # Get the thread's RCB and find the tool call
        rcb = self.scheduler.rcb_manager.get(thread_id)
        if not rcb:
            return InterruptResult(
                success=False,
                interrupt_type=interrupt.interrupt_type,
                error=f"Thread {thread_id} not found",
            )

        # Find and update the tool call
        for call in rcb.tool_call_stack:
            if call.call_id == call_id:
                call.result = result
                call.completed_at = datetime.now()
                break

        # Unblock the thread
        self.scheduler.unblock_thread(thread_id)

        return InterruptResult(
            success=True,
            interrupt_type=interrupt.interrupt_type,
            handler_thread_id=thread_id,
            tool_result=result,
        )

    def _handle_time_slice(self, interrupt: Interrupt) -> InterruptResult:
        """Handle a time slice expired interrupt.

        Args:
            interrupt: Time slice interrupt

        Returns:
            InterruptResult
        """
        # Make scheduling decision
        decision = self.scheduler.schedule(interrupt)

        # Context switch if needed
        if decision.preempt and self.current_thread_id and decision.selected_thread_id:
            switch = self.scheduler.context_switch(
                self.current_thread_id,
                decision.selected_thread_id,
                reason="Time slice expired",
            )

            return InterruptResult(
                success=True,
                interrupt_type=interrupt.interrupt_type,
                handler_thread_id=decision.selected_thread_id,
                context_switch=switch,
            )

        return InterruptResult(
            success=True,
            interrupt_type=interrupt.interrupt_type,
        )

    def _handle_preempt(self, interrupt: Interrupt) -> InterruptResult:
        """Handle a preemption interrupt.

        Args:
            interrupt: Preempt interrupt

        Returns:
            InterruptResult
        """
        decision = self.scheduler.schedule(interrupt)

        if decision.selected_thread_id and decision.selected_thread_id != self.current_thread_id:
            if self.current_thread_id:
                switch = self.scheduler.context_switch(
                    self.current_thread_id,
                    decision.selected_thread_id,
                    reason="Higher priority thread ready",
                )

                return InterruptResult(
                    success=True,
                    interrupt_type=interrupt.interrupt_type,
                    handler_thread_id=decision.selected_thread_id,
                    context_switch=switch,
                )

        return InterruptResult(
            success=True,
            interrupt_type=interrupt.interrupt_type,
        )

    def _handle_yield(self, interrupt: Interrupt) -> InterruptResult:
        """Handle a yield interrupt.

        Args:
            interrupt: Yield interrupt

        Returns:
            InterruptResult
        """
        thread_id = interrupt.payload.get("thread_id", self.current_thread_id)

        if thread_id:
            self.scheduler.yield_cpu(thread_id)

        return InterruptResult(
            success=True,
            interrupt_type=interrupt.interrupt_type,
            handler_thread_id=None,
        )

    def _apply_perception_alignment(self, tool_call: ToolCall) -> Any:
        """Apply perception alignment to tool output.

        Perception Alignment filters and recodes tool output to match
        the cognitive model's expectations.

        Args:
            tool_call: Completed tool call

        Returns:
            Aligned result
        """
        result = tool_call.result

        if not self.perception_config.enable_filtering:
            return result

        # Filter: truncate if too long
        if self.perception_config.max_output_tokens > 0:
            if isinstance(result, str):
                # Rough token estimation (4 chars per token)
                max_chars = self.perception_config.max_output_tokens * 4
                if len(result) > max_chars:
                    result = result[:max_chars] + "..."

        # Recoding: wrap in semantic structure
        if self.perception_config.enable_recoding:
            if isinstance(result, (str, int, float, bool)):
                result = {
                    "type": "tool_result",
                    "tool_name": tool_call.tool_name,
                    "content": result,
                    "timestamp": datetime.now().isoformat(),
                }

        return result

    def trigger_tool_call(
        self, tool_id: str, arguments: dict[str, Any]
    ) -> InterruptResult:
        """Trigger a tool call interrupt.

        Convenience method to invoke a tool from a reasoning thread.

        Args:
            tool_id: Tool to call
            arguments: Tool arguments

        Returns:
            InterruptResult
        """
        interrupt = Interrupt(
            interrupt_type=InterruptType.TOOL_CALL,
            source_thread_id=self.scheduler.current_thread_id or "",
            payload={"tool_id": tool_id, "arguments": arguments},
        )

        return self.handle_interrupt(interrupt)

    def get_statistics(self) -> dict[str, Any]:
        """Get RIC statistics.

        Returns:
            Dictionary with statistics
        """
        total_interrupts = len(self.interrupt_history)
        successful = sum(1 for r in self.interrupt_history if r.success)

        return {
            "total_interrupts": total_interrupts,
            "successful_interrupts": successful,
            "success_rate": successful / total_interrupts if total_interrupts > 0 else 1.0,
            "total_interrupt_time_ms": sum(r.duration_ms for r in self.interrupt_history),
            "interrupts_by_type": self._count_by_type(),
        }

    def _count_by_type(self) -> dict[str, int]:
        """Count interrupts by type."""
        counts: dict[str, int] = {}
        for result in self.interrupt_history:
            itype = str(result.interrupt_type)
            counts[itype] = counts.get(itype, 0) + 1
        return counts
