"""
Reasoning Kernel (RK).

The central processing unit of AgentOS, implementing the Contextual Transition
Function from the paper:

    𝓕: (Sₜ, 𝒞ₐddᵣ) → Sₜ₊₁

Where:
- Sₜ: Current cognitive state
- 𝒞ₐddᵣ: Addressable Context Space
- Sₜ₊₁: Next cognitive state

The RK performs context transformations through attention, synthesizing
information and simulating cognitive reasoning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from agentos.memory.slicing.slicer import SemanticSlicer, SemanticSlicerConfig
from agentos.memory.slicing.types import AttentionOutput, SemanticSlice, SlicingResult
from agentos.models.transformers_backend import TransformersBackend, BackendConfig

logger = logging.getLogger(__name__)


class KernelState(str, Enum):
    """Possible states of the Reasoning Kernel."""

    UNINITIALIZED = "uninitialized"
    IDLE = "idle"
    PROCESSING = "processing"
    INTERRUPTED = "interrupted"  # Reserved for future interrupt handling (Reasoning Interrupt Cycle)
    ERROR = "error"


@dataclass
class CognitiveState:
    """The cognitive state Sₜ of the Reasoning Kernel.

    Represents the current "mental state" including:
    - Active context slices in L1 (attention window)
    - Current focus of attention
    - Pending operations (reserved for future interrupt handling)
    """

    # The currently active context window (L1 cache)
    active_slices: list[SemanticSlice] = field(default_factory=list)

    # Current focus/concentration metric
    attention_focus: float = 1.0

    # Stack depth for nested reasoning
    semantic_stack_depth: int = 0

    # Pending operations (reserved for future Reasoning Interrupt Cycle)
    # Will track pending tool calls and interrupt state
    pending_operations: list[dict[str, Any]] = field(default_factory=list)

    # State metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def active_token_count(self) -> int:
        """Total tokens in active slices."""
        return sum(s.token_count for s in self.active_slices)

    @property
    def is_empty(self) -> bool:
        """Check if state has any active context."""
        return len(self.active_slices) == 0


@dataclass
class ProcessingResult:
    """Result from a Reasoning Kernel processing operation."""

    # Input that was processed
    input_text: str

    # Attention output from forward pass
    attention_output: AttentionOutput

    # Semantic slicing result
    slicing_result: SlicingResult

    # Resulting cognitive state
    resulting_state: CognitiveState

    # Processing metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningKernelConfig:
    """Configuration for the Reasoning Kernel."""

    # Backend configuration
    backend_config: BackendConfig = None

    # Semantic slicer configuration
    slicer_config: SemanticSlicerConfig = None

    # L1 cache limits (tokens)
    l1_max_tokens: int = 4096

    # Whether to automatically slice on processing
    auto_slice: bool = True

    def __post_init__(self):
        """Initialize default configs if not provided."""
        if self.backend_config is None:
            self.backend_config = BackendConfig()
        if self.slicer_config is None:
            self.slicer_config = SemanticSlicerConfig()


class ReasoningKernel:
    """The Reasoning Kernel - central processing unit of AgentOS.

    Implements the Contextual Transition Function:
        𝓕: (Sₜ, 𝒞ₐddᵣ) → Sₜ₊₁

    The RK:
    1. Takes current cognitive state Sₜ and addressable context 𝒞ₐddᵣ
    2. Performs a forward pass through the LLM
    3. Extracts attention weights and hidden states
    4. Computes semantic slices from attention patterns
    5. Updates cognitive state to Sₜ₊₁
    """

    def __init__(self, config: ReasoningKernelConfig | None = None) -> None:
        """Initialize the Reasoning Kernel.

        Args:
            config: Kernel configuration. If None, uses defaults.
        """
        self.config = config or ReasoningKernelConfig()

        self._backend = TransformersBackend(self.config.backend_config)
        self._slicer = SemanticSlicer(self.config.slicer_config)

        self._state: CognitiveState = CognitiveState()
        self._kernel_state: KernelState = KernelState.UNINITIALIZED

        logger.info("Reasoning Kernel initialized (unloaded)")

    @property
    def backend(self) -> TransformersBackend:
        """Get the LLM backend."""
        return self._backend

    @property
    def slicer(self) -> SemanticSlicer:
        """Get the semantic slicer."""
        return self._slicer

    @property
    def state(self) -> CognitiveState:
        """Get the current cognitive state."""
        return self._state

    @property
    def kernel_state(self) -> KernelState:
        """Get the kernel state."""
        return self._kernel_state

    def load(self) -> None:
        """Load the underlying model."""
        if self._kernel_state == KernelState.UNINITIALIZED:
            self._backend.load()
            self._kernel_state = KernelState.IDLE
            logger.info("Reasoning Kernel loaded and ready")
        elif self._kernel_state == KernelState.ERROR:
            self._backend.load()
            self._kernel_state = KernelState.IDLE
            logger.info("Reasoning Kernel reloaded after error")

    def unload(self) -> None:
        """Unload the model to free memory."""
        self._backend.unload()
        self._kernel_state = KernelState.UNINITIALIZED
        self._state = CognitiveState()
        logger.info("Reasoning Kernel unloaded")

    def process(
        self,
        input_text: str,
        max_length: int | None = None,
    ) -> ProcessingResult:
        """Process input through the Contextual Transition Function.

        Implements: 𝓕: (Sₜ, 𝒞ₐddᵣ) → Sₜ₊₁

        Args:
            input_text: Input text to process.
            max_length: Maximum sequence length.

        Returns:
            ProcessingResult with attention output, slices, and new state.
        """
        if self._kernel_state == KernelState.UNINITIALIZED:
            self.load()

        self._kernel_state = KernelState.PROCESSING

        try:
            # Step 1: Run forward pass through LLM
            logger.debug(f"Processing input: {input_text[:100]}...")
            attention_output = self._backend.forward(input_text, max_length=max_length)

            # Step 2: Perform semantic slicing if enabled
            if self.config.auto_slice:
                slicing_result = self._slicer.slice(
                    attention_output,
                    tokenizer=self._backend.tokenizer,
                )
            else:
                # Create empty slicing result
                from agentos.memory.slicing.types import DensityProfile
                slicing_result = SlicingResult(
                    slices=[],
                    density_profile=DensityProfile(
                        densities=attention_output.attention_weights.mean(axis=(0, 1)),
                        entropy=attention_output.attention_weights.mean(axis=(0, 1)),
                        gradients=attention_output.attention_weights.mean(axis=(0, 1)),
                    ),
                )

            # Step 3: Update cognitive state (Contextual Transition)
            new_state = self._transition_state(
                self._state,
                attention_output,
                slicing_result,
            )

            # Step 4: Store result
            result = ProcessingResult(
                input_text=input_text,
                attention_output=attention_output,
                slicing_result=slicing_result,
                resulting_state=new_state,
                metadata={
                    "num_slices": len(slicing_result.slices),
                    "total_tokens": slicing_result.total_tokens,
                    "input_length": len(input_text),
                },
            )

            self._state = new_state
            self._kernel_state = KernelState.IDLE

            logger.debug(
                f"Processing complete: {len(slicing_result.slices)} slices, "
                f"{slicing_result.total_tokens} tokens"
            )

            return result

        except Exception as e:
            self._kernel_state = KernelState.ERROR
            logger.error(f"Error during processing: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt.
            max_new_tokens: Maximum number of tokens to generate.
            system_prompt: Optional system prompt to prepend.
            **kwargs: Additional generation arguments.

        Returns:
            Generated text.
        """
        if self._kernel_state == KernelState.UNINITIALIZED:
            self.load()

        # Use chat template for proper formatting (Qwen, LLaMA, etc.)
        # Pass system_prompt and prompt separately for correct formatting
        return self._backend.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
            **kwargs,
        )

    def _transition_state(
        self,
        previous_state: CognitiveState,
        attention_output: AttentionOutput,
        slicing_result: SlicingResult,
    ) -> CognitiveState:
        """Perform the Contextual Transition: Sₜ → Sₜ₊₁.

        This is the core operation of the Reasoning Kernel.

        Implements: 𝓕: (Sₜ, 𝒞ₐddᵣ) → Sₜ₊₁

        Args:
            previous_state: Current cognitive state Sₜ.
            attention_output: Output from LLM forward pass.
            slicing_result: Semantic slices from attention analysis.

        Returns:
            New cognitive state Sₜ₊₁.
        """
        # Create new state based on previous
        new_state = CognitiveState(
            attention_focus=previous_state.attention_focus,
            semantic_stack_depth=previous_state.semantic_stack_depth,
            pending_operations=list(previous_state.pending_operations),
        )

        # Add new slices to L1 (subject to capacity limits)
        # For now, just add all slices (L1 management will be in S-MMU)
        new_state.active_slices = list(previous_state.active_slices)
        new_state.active_slices.extend(slicing_result.slices)

        # Compute aggregated attention once (used for multiple calculations)
        avg_attention = None
        if attention_output.attention_weights is not None:
            # Average attention across all layers and heads
            # Shape: (num_layers, num_heads, seq_len, seq_len) → (seq_len, seq_len)
            avg_attention = attention_output.attention_weights.mean(axis=(0, 1))

        # Update attention focus using attention patterns
        # Attention focus = mean attention entropy (lower = more focused)
        if avg_attention is not None:
            # Compute entropy per position: -sum(p * log(p))
            # Normalize each row to get probability distribution
            row_sums = avg_attention.sum(axis=1, keepdims=True) + 1e-9
            probs = avg_attention / row_sums
            # Avoid log(0)
            probs = np.clip(probs, 1e-9, 1.0)
            entropy = -(probs * np.log(probs)).sum(axis=1)
            # Mean entropy normalized by log(seq_len)
            max_entropy = np.log(len(entropy))
            normalized_entropy = entropy.mean() / max_entropy if max_entropy > 0 else 0
            # Focus = 1 - entropy (higher = more focused)
            new_state.attention_focus = 1.0 - normalized_entropy
        elif len(slicing_result.density_profile.densities) > 0:
            # Fallback to density profile if no attention weights
            mean_density = float(slicing_result.density_profile.densities.mean())
            new_state.attention_focus = mean_density

        # Update semantic stack depth based on attention patterns
        # Deeper reasoning = more attention to earlier tokens (long-range dependencies)
        if avg_attention is not None:
            seq_len = avg_attention.shape[0]
            if seq_len > 1:
                first_half_attention = avg_attention[:, :seq_len//2].sum()
                second_half_attention = avg_attention[:, seq_len//2:].sum()
                # If more attention to first half, indicates deeper reasoning
                if first_half_attention > second_half_attention:
                    new_state.semantic_stack_depth = min(
                        previous_state.semantic_stack_depth + 1,
                        10  # Max depth
                    )
                else:
                    new_state.semantic_stack_depth = max(
                        previous_state.semantic_stack_depth - 1,
                        0  # Min depth
                    )

        # Store hidden states in metadata for semantic gradient computation
        if attention_output.hidden_states is not None:
            # Use mean hidden state as semantic context vector
            semantic_vector = attention_output.hidden_states.mean(axis=0)
            new_state.metadata.update({
                "semantic_vector": semantic_vector.tolist(),
                "hidden_dim": semantic_vector.shape[0],
            })

        # Update metadata
        new_state.metadata.update({
            "last_update": "transition",
            "slice_count": len(new_state.active_slices),
            "token_count": new_state.active_token_count,
            "seq_len": len(attention_output.tokens),
        })

        return new_state

    def reset_state(self) -> None:
        """Reset cognitive state to initial condition."""
        self._state = CognitiveState()
        logger.debug("Cognitive state reset")

    def interrupt(self, reason: str = "external_interrupt") -> CognitiveState:
        """Interrupt current processing and save cognitive state.

        Called by the Reasoning Interrupt Cycle when an interrupt occurs.
        The current cognitive state is saved to pending_operations and can
        be restored later with resume().

        Args:
            reason: Reason for the interrupt

        Returns:
            The cognitive state that was saved
        """
        if self._kernel_state != KernelState.PROCESSING:
            logger.warning(f"Cannot interrupt kernel in state: {self._kernel_state}")
            return self._state

        logger.info(f"Interrupting kernel: {reason}")
        self._kernel_state = KernelState.INTERRUPTED

        # Save current state to pending operations
        saved_state = CognitiveState(
            active_slices=list(self._state.active_slices),
            attention_focus=self._state.attention_focus,
            semantic_stack_depth=self._state.semantic_stack_depth,
            pending_operations=[
                {
                    "type": "interrupt",
                    "reason": reason,
                    "saved_at": self._state.metadata.copy(),
                }
            ],
            metadata=self._state.metadata.copy(),
        )

        return saved_state

    def resume(self, from_state: CognitiveState | None = None) -> None:
        """Resume processing after an interrupt.

        Restores the cognitive state that was saved during interrupt().

        Args:
            from_state: Previously saved state to restore. If None, creates fresh state.
        """
        logger.info("Resuming kernel after interrupt")

        if from_state is not None:
            self._state = from_state
        else:
            self._state = CognitiveState()

        self._kernel_state = KernelState.IDLE

    def add_pending_operation(self, operation: dict[str, Any]) -> None:
        """Add a pending operation to the cognitive state.

        Used by the RIC to track tool calls and other interruptions.

        Args:
            operation: Operation dict with type and metadata
        """
        self._state.pending_operations.append(operation)
        logger.debug(f"Added pending operation: {operation.get('type', 'unknown')}")

    def pop_pending_operation(self) -> dict[str, Any] | None:
        """Pop and return the most recent pending operation.

        Returns:
            The operation dict, or None if no pending operations
        """
        if self._state.pending_operations:
            return self._state.pending_operations.pop()
        return None

    @property
    def has_pending_operations(self) -> bool:
        """Check if there are pending operations."""
        return len(self._state.pending_operations) > 0

    def get_context_summary(self) -> dict[str, Any]:
        """Get a summary of the current context/state.

        Returns:
            Dictionary with state information.
        """
        return {
            "kernel_state": self._kernel_state.value,
            "active_slices": len(self._state.active_slices),
            "active_tokens": self._state.active_token_count,
            "attention_focus": self._state.attention_focus,
            "stack_depth": self._state.semantic_stack_depth,
            "pending_ops": len(self._state.pending_operations),
            "device": str(self._backend.device),
            "model": self._backend.config.model_name,
        }

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        _ = exc_type, exc_val, exc_tb  # Unused but required by protocol
        self.unload()
        return False


def create_kernel(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "auto",
    l1_max_tokens: int = 4096,
    **kwargs,
) -> ReasoningKernel:
    """Convenience function to create a Reasoning Kernel.

    Args:
        model_name: Model name or path.
        device: Device to use ("auto", "cpu", "cuda", "mps").
        l1_max_tokens: Maximum tokens in L1 cache.
        **kwargs: Additional configuration options.

    Returns:
        Initialized ReasoningKernel.
    """
    backend_config = BackendConfig(model_name=model_name, device=device, **kwargs)
    kernel_config = ReasoningKernelConfig(
        backend_config=backend_config,
        l1_max_tokens=l1_max_tokens,
    )
    return ReasoningKernel(kernel_config)
