"""
Agent: Individual reasoning agent within the AgentOS system.

Each Agent has:
- Access to the Reasoning Kernel (Phase 1)
- Access to Memory Hierarchy via S-MMU (Phase 2)
- A thread in the Cognitive Scheduler (Phase 3)
- Participation in Multi-Agent Sync (Phase 4)
- Metrics collection (Phase 5)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from agentos.agentos import AgentOS
    from agentos.kernel import ReasoningKernel
    from agentos.memory import SMMU
    from agentos.scheduler import CognitiveScheduler
    from agentos.sync import CSPOrchestrator, DistributedSharedMemory
    from agentos.eval import MetricsCalculator
    from agentos.memory.slicing.types import SemanticSlice

from agentos.sync.types import AgentState as SyncAgentState, SemanticSliceVersion


logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for an Agent."""

    agent_id: str
    name: str
    role: str = "general"  # researcher, writer, analyst, etc.
    system_id: str = "default"

    # Capabilities
    can_use_tools: bool = True
    can_sync: bool = True

    # Generation settings
    use_generation: bool = False  # Set to True to use actual LLM generation
    max_new_tokens: int = 80

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMemory:
    """Memory state for an agent."""

    # Store actual semantic slice objects, not just IDs
    active_slices: list["SemanticSlice"] = field(default_factory=list)
    # Also keep a quick lookup dict for slice_id -> slice
    _slices_by_id: dict[str, "SemanticSlice"] = field(default_factory=dict)

    working_context: str = ""
    last_processed: datetime | None = None

    def add_slice(self, slice_obj: "SemanticSlice") -> None:
        """Add a semantic slice to memory.

        Args:
            slice_obj: The semantic slice to add
        """
        self.active_slices.append(slice_obj)
        self._slices_by_id[slice_obj.id] = slice_obj

    def get_slice(self, slice_id: str) -> "SemanticSlice | None":
        """Get a slice by ID.

        Args:
            slice_id: Slice identifier

        Returns:
            SemanticSlice if found, None otherwise
        """
        return self._slices_by_id.get(slice_id)

    def clear_slices(self) -> None:
        """Clear all slices from memory."""
        self.active_slices.clear()
        self._slices_by_id.clear()


class Agent:
    """
    Individual reasoning agent.

    Each agent is an autonomous entity that can:
    - Process text through the Reasoning Kernel
    - Manage its own memory via S-MMU
    - Participate in scheduled reasoning
    - Sync with other agents
    - Have its own state tracked
    """

    def __init__(
        self,
        config: AgentConfig,
        kernel: "ReasoningKernel",
        smmu: "SMMU",
        scheduler: "CognitiveScheduler",
        csp_orchestrator: "CSPorchestrator",
        dsm: "DistributedSharedMemory",
        metrics: "MetricsCalculator | None" = None,
    ) -> None:
        """Initialize an Agent.

        Args:
            config: Agent configuration
            kernel: Shared reasoning kernel
            smmu: Shared semantic memory management unit
            scheduler: Shared cognitive scheduler
            csp_orchestrator: Shared CSP orchestrator for sync
            dsm: Shared distributed memory
            metrics: Optional metrics calculator
        """
        self.config = config
        self._kernel = kernel
        self._smmu = smmu
        self._scheduler = scheduler
        self._csp_orchestrator = csp_orchestrator
        self._dsm = dsm
        self._metrics = metrics

        # Agent state
        self._thread_id: str | None = None
        self._semantic_gradient: NDArray[np.float32] | None = None
        self._current_confidence: float = 0.8

        # Memory
        self.memory = AgentMemory()

        # Creation time
        self.created_at = datetime.now()

    @property
    def agent_id(self) -> str:
        """Get agent ID."""
        return self.config.agent_id

    @property
    def thread_id(self) -> str | None:
        """Get scheduler thread ID."""
        return self._thread_id

    def set_thread_id(self, thread_id: str) -> None:
        """Set scheduler thread ID."""
        self._thread_id = thread_id

    def process(self, input_text: str) -> str:
        """Process input through the reasoning kernel.

        Args:
            input_text: Text to process

        Returns:
            Processed output
        """
        import time

        # Track processing time for metrics
        process_start = time.time()

        # Process through kernel
        result = self._kernel.process(input_text, max_length=512)

        processing_time_ms = (time.time() - process_start) * 1000

        # Update semantic gradient from attention output
        if result.attention_output.hidden_states is not None:
            # Use mean of last hidden state as gradient approximation
            # Shape: (seq_len, hidden_dim) → mean over seq_len → (hidden_dim,)
            self._semantic_gradient = result.attention_output.hidden_states.mean(axis=0)

            # Update CSP orchestrator with new gradient for drift tracking
            if self._csp_orchestrator:
                self._csp_orchestrator.update_agent_drift(
                    agent_id=self.agent_id,
                    agent_gradient=self._semantic_gradient,
                )

        # Process slices through S-MMU
        promoted_l1_slice_ids: list[str] = []
        if result.slicing_result and result.attention_output.hidden_states is not None:
            promoted_l1_slice_ids = self._smmu.process_slices(
                result.slicing_result,
                result.attention_output.hidden_states,
            )

        # Update active slices with actual semantic slice objects
        # Clear existing slices and add the new ones
        self.memory.clear_slices()

        if promoted_l1_slice_ids:
            # Add promoted L1 slices as actual slice objects
            # Find the actual slice objects from the slicing result
            promoted_slice_objects = [
                s for s in result.slicing_result.slices
                if s.id in promoted_l1_slice_ids
            ]
            for slice_obj in promoted_slice_objects:
                self.memory.add_slice(slice_obj)
        elif result.slicing_result:
            # If no slices promoted, track all slices for visibility
            for slice_obj in result.slicing_result.slices:
                self.memory.add_slice(slice_obj)

        # Update memory context (keep recent context)
        self.memory.working_context = input_text[-2000:]

        self.memory.last_processed = datetime.now()

        # Sync RK state to scheduler's RCB (for cognitive scheduling decisions)
        self._sync_kernel_state_to_rcb(result)

        # Track RK processing metrics (Phase 1 → Phase 5 connection)
        if self._metrics is not None:
            token_count = len(result.attention_output.tokens)
            slice_count = len(result.slicing_result.slices) if result.slicing_result else 0
            self._metrics.calculate_rk_performance(
                processing_time_ms=processing_time_ms,
                token_count=token_count,
                slice_count=slice_count,
                attention_focus=self._kernel.state.attention_focus,
                semantic_stack_depth=self._kernel.state.semantic_stack_depth,
                kernel_state=self._kernel.kernel_state.value,
            )

        # Track S-MMU memory utilization (Phase 2 → Phase 5 connection)
        if self._metrics is not None:
            memory_stats = self._smmu.get_memory_stats()
            l1_stats = memory_stats["l1"]
            l2_stats = memory_stats["l2"]
            l3_stats = memory_stats["l3"]

            self._metrics.calculate_utilization_efficiency(
                active_context_tokens=l1_stats["used_tokens"],
                max_context_tokens=l1_stats["max_tokens"],
                total_slices=l1_stats["slice_count"] + l2_stats["slice_count"],
                utilized_slices=l1_stats["slice_count"],  # Only L1 slices are "actively utilized"
                l1_tokens=l1_stats["used_tokens"],
                l2_tokens=l2_stats["used_tokens"],
                l3_tokens=l3_stats.get("total_size_bytes", 0) // 1000,  # Rough estimate
            )

        # Return result (in production, would generate actual response)
        # Return full input text as the processed result for visibility
        return f"[Processed by {self.config.name}]: {input_text}"

    def _sync_kernel_state_to_rcb(self, result) -> None:
        """Sync ReasoningKernel state to the scheduler's ReasoningControlBlock.

        This connects Phase 1 (RK) to Phase 3 (Scheduler) by using the RK's
        cognitive state to inform scheduling decisions.

        Args:
            result: ProcessingResult from the kernel
        """
        if self._thread_id is None:
            return

        # Get the RCB for this thread
        rcb = self._scheduler.rcb_manager.get(self._thread_id)
        if not rcb:
            return

        # Update cognitive_fidelity from RK's attention_focus
        # RK's attention_focus is computed from attention entropy (1.0 = focused, 0.0 = diffuse)
        # Map to RCB's cognitive_fidelity (same scale)
        rcb.cognitive_fidelity = self._kernel.state.attention_focus

        # Update semantic_stack_depth from RK's state
        # Both track nested reasoning depth
        rcb.semantic_stack_depth = self._kernel.state.semantic_stack_depth

        # Update context_coherence based on slice density
        # Higher average density = more coherent context
        if result.slicing_result and len(result.slicing_result.slices) > 0:
            densities = [s.density_mean for s in result.slicing_result.slices]
            # Density is already in [0, 1], use mean as coherence measure
            rcb.context_coherence = float(sum(densities) / len(densities))
        else:
            rcb.context_coherence = 1.0  # No slices = fully coherent (empty context)

        # Update attention_focus in RCB to track active slice
        if result.slicing_result and len(result.slicing_result.slices) > 0:
            # Most recent slice is the active one
            rcb.attention_focus.active_slice_id = result.slicing_result.slices[-1].id
            # Track slice IDs in context
            rcb.attention_focus.context_slices = [s.id for s in result.slicing_result.slices]

    def get_state(self) -> SyncAgentState:
        """Get current agent state for synchronization.

        Returns:
            AgentState with current information including actual slice content
        """
        state = SyncAgentState(
            agent_id=self.agent_id,
            semantic_gradients=self._semantic_gradient,
            metadata={
                "name": self.config.name,
                "role": self.config.role,
                "thread_id": self._thread_id,
                "last_processed": self.memory.last_processed.isoformat() if self.memory.last_processed else None,
            },
        )

        # Add actual semantic slice objects to the state
        for slice_obj in self.memory.active_slices:
            state.add_slice(slice_obj)

        return state

    def sync_with_global(self) -> bool:
        """Synchronize agent state with global state.

        Returns:
            True if sync was successful
        """
        if not self.config.can_sync:
            return False

        # Read global slices (sync up to 3 slices)
        for slice_obj in self.memory.active_slices[:3]:
            version = self._dsm.sync_slice(slice_obj.id, self.agent_id)
            if version:
                # Agent updates its understanding by reading the global version
                # The agent could update its local slice content with the global version
                pass

        return True

    def update_confidence(self, confidence: float) -> None:
        """Update agent's confidence level.

        Args:
            confidence: Confidence score (0-1)
        """
        self._current_confidence = max(0.0, min(1.0, confidence))

    def get_confidence(self) -> float:
        """Get agent's current confidence.

        Returns:
            Confidence score (0-1)
        """
        return self._current_confidence

    def get_semantic_gradient(self) -> NDArray[np.float32] | None:
        """Get agent's current semantic gradient.

        Returns:
            Semantic gradient vector or None
        """
        return self._semantic_gradient

    def write_to_dsm(self, slice_id: str, content: str) -> bool:
        """Write a slice to distributed shared memory.

        Args:
            slice_id: Slice identifier
            content: Slice content

        Returns:
            True if write was successful
        """
        from agentos.memory.slicing.types import SemanticSlice

        slice_data = SemanticSlice(
            id=slice_id,
            start_pos=0,
            end_pos=len(content),
            tokens=["content"],
            token_ids=[0],
            content=content,
            density_mean=0.5,
            density_std=0.1,
        )

        return self._dsm.write_slice(slice_data, self.agent_id)

    def read_from_dsm(self, slice_id: str) -> SemanticSliceVersion | None:
        """Read a slice from distributed shared memory.

        Args:
            slice_id: Slice identifier

        Returns:
            Slice version if found
        """
        return self._dsm.read_slice(slice_id, self.agent_id)

    def read_global_slices(self) -> dict[str, str]:
        """Read all semantic slices from global state.

        This allows agents to see what other agents have contributed.

        Returns:
            Dictionary mapping slice_id to slice content
        """
        from agentos.sync.types import SemanticSliceVersion

        global_state = self._csp_orchestrator.get_global_state()
        slices_content = {}

        for slice_id, slice_version in global_state.slices.items():
            slices_content[slice_id] = slice_version.content

        return slices_content

    def synthesize_from_global_state(self, task: str, agent_contributions: dict[str, str]) -> str:
        """Synthesize a final answer from global semantic state.

        Uses semantic synthesis based on AgentOS paper Section 3.5:
        - Clusters semantically related concepts from all agents
        - Weights contributions by cognitive fidelity (attention focus)
        - Detects and resolves conflicts between perspectives
        - Produces integrated synthesis

        Args:
            task: The original task/question
            agent_contributions: Individual agent contributions for reference

        Returns:
            Synthesized final answer
        """
        from agentos.synthesis import SemanticSynthesizer, create_synthesizer

        # Collect agent states for semantic metadata
        agent_states = {}
        for agent_id in agent_contributions.keys():
            # Get agent state from CSP orchestrator
            if hasattr(self._csp_orchestrator, '_agents'):
                agent_state = self._csp_orchestrator._agents.get(agent_id)
                if agent_state:
                    agent_states[agent_id] = agent_state

        # Create semantic synthesizer
        synthesizer = create_synthesizer(
            strategy="semantic_merge",
            similarity_threshold=0.7,
            use_confidence_weighting=True,
            use_llm_synthesis=self.config.use_generation,
        )

        # Create LLM generation function if generation is enabled
        llm_fn = None
        if self.config.use_generation:
            llm_fn = lambda prompt, max_new_tokens, system_prompt: self._kernel.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
            )

        # Perform semantic synthesis
        result = synthesizer.synthesize(
            task=task,
            agent_contributions=agent_contributions,
            agent_states=agent_states,
            llm_generate_fn=llm_fn,
        )

        # Log synthesis metrics
        logger.info(
            f"Semantic synthesis: coherence={result.coherence_score:.2f}, "
            f"coverage={result.coverage_score:.2f}, "
            f"clusters={len(result.concept_clusters)}, "
            f"conflicts={len(result.conflicts_resolved)}"
        )

        return result.final_synthesis

    def contribute_to_task(self, task: str) -> str:
        """Generate a contribution for a collaborative task.

        Args:
            task: Task description

        Returns:
            Contribution text
        """
        # Process the task (for semantic slicing and memory)
        self.process(task)

        # Generate contribution text
        contribution_text = self._generate_contribution_text(task)

        # Write contribution to distributed shared memory as a special slice
        # This allows other agents (especially synthesizer) to access it
        contribution_slice_id = f"contribution_{self.agent_id}_{int(datetime.now().timestamp())}"
        self.write_to_dsm(contribution_slice_id, contribution_text)

        return contribution_text

    def _generate_contribution_text(self, task: str) -> str:
        """Generate the actual contribution text.

        Args:
            task: Task description

        Returns:
            Contribution text
        """
        role_system_prompts = {
            "researcher": (
                "You are a researcher. Your role is to conduct thorough research and present "
                "comprehensive, well-structured information. Provide detailed analysis including: "
                "key concepts, technical details, examples, comparisons, and evidence-based conclusions. "
                "Break down complex topics into clear sections. Use specific examples and data points "
                "where applicable. Aim for depth and completeness in your analysis."
            ),
            "writer": (
                "You are a writer. Your role is to articulate ideas clearly and engagingly. "
                "Focus on clear expression and narrative flow."
            ),
            "analyst": (
                "You are an analyst. Your role is to examine data and provide insights. "
                "Focus on patterns, trends, and actionable conclusions."
            ),
            "critic": (
                "You are a critical reviewer. Your role is to evaluate content thoroughly and "
                "provide detailed feedback. Identify strengths, weaknesses, gaps in reasoning, "
                "areas for improvement, and potential counterarguments. Be specific in your critique "
                "and provide constructive suggestions for enhancement."
            ),
            "reviewer": (
                "You are a critical reviewer. Your role is to evaluate content thoroughly and "
                "provide detailed feedback. Identify strengths, weaknesses, gaps in reasoning, "
                "areas for improvement, and potential counterarguments. Be specific in your critique "
                "and provide constructive suggestions for enhancement."
            ),
            "synthesizer": (
                "You are a synthesizer. Your role is to integrate diverse perspectives "
                "into a coherent whole. Focus on connections and unified understanding."
            ),
            "general": (
                "You are a helpful assistant. Provide clear and thoughtful responses."
            ),
        }

        system_prompt = role_system_prompts.get(
            self.config.role,
            role_system_prompts["general"]
        )

        # Generation is required for contribution
        if not self.config.use_generation:
            raise RuntimeError(
                f"Agent '{self.config.name}' (role: {self.config.role}) cannot generate contribution "
                f"because use_generation=False. Enable use_generation or provide a pre-configured response."
            )

        # Generate contribution using LLM
        contribution = self._kernel.generate(
            prompt=task,
            max_new_tokens=self.config.max_new_tokens,
            system_prompt=system_prompt,
        )
        return contribution.strip()

    def __repr__(self) -> str:
        """String representation of agent."""
        return f"Agent(id={self.agent_id}, name={self.config.name}, role={self.config.role})"


def create_agent(
    name: str,
    role: str = "general",
    system_id: str = "default",
    **kwargs,
) -> AgentConfig:
    """Create an agent configuration.

    Convenience function for creating agent configs.

    Args:
        name: Agent name
        role: Agent role
        system_id: System identifier
        **kwargs: Additional config options

    Returns:
        AgentConfig
    """
    agent_id = f"{name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}"
    return AgentConfig(
        agent_id=agent_id,
        name=name,
        role=role,
        system_id=system_id,
        **kwargs,
    )
