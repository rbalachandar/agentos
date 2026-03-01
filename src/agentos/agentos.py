"""
AgentOS: Integrated Multi-Agent Cognitive Architecture.

This is the main entry point that orchestrates all phases:
- Phase 1: Reasoning Kernel & Semantic Slicing
- Phase 2: Cognitive Memory Hierarchy (S-MMU)
- Phase 3: Cognitive Scheduler & I/O Subsystem
- Phase 4: Multi-Agent Synchronization
- Phase 5: Evaluation & Metrics
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from agentos.agent import Agent, AgentConfig, AgentMemory
from agentos.eval import MetricsCalculator
from agentos.io import (
    InterruptVectorTable,
    PeripheralRegistry,
    ReasoningInterruptCycle,
    STANDARD_VECTORS,
    register_builtins,
)
from agentos.kernel.reasoning_kernel import ReasoningKernel, create_kernel
from agentos.memory import SMMU, SMMUConfig, L1CacheConfig, L2Config, L3Config
from agentos.scheduler import CognitiveScheduler, SchedulerConfig, ThreadPriority
from agentos.sync import (
    CSPOrchestrator,
    CSPOrchestratorConfig,
    DistributedSharedMemory,
    DriftTrackerConfig,
    PerceptionAlignmentConfig,
    StateReconciler,
    ReconciliationConfig,
    StoreBackend,
)


@dataclass
class AgentOSConfig:
    """Configuration for the AgentOS system."""

    # Model configuration
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "auto"
    l1_max_tokens: int = 512

    # Memory configuration
    l1_max_tokens_cache: int = 1000
    l1_max_slices: int = 10
    l2_max_tokens: int = 10000
    l2_max_slices: int = 100
    l3_storage_path: str = "./data/l3"

    # Cold start mitigation: L3 bootstrap and restore
    l3_bootstrap_paths: list[str] | None = None  # Paths to JSONL files for L3 bootstrap
    l3_restore_path: str | None = None  # Path to restore L3 state from previous session
    l3_save_path: str | None = "./data/l3_state"  # Path to save L3 state for restore
    enable_adaptive_scoring: bool = True  # Enable adaptive importance scoring during warmup
    warmup_turns: int = 5  # Number of turns for warmup period

    # Scheduler configuration
    scheduler_time_slice_ms: float = 100.0
    scheduler_use_cognitive_fidelity: bool = True

    # Sync configuration
    enable_sync: bool = True
    max_agents: int = 10
    drift_threshold: float = 1.0
    min_sync_interval_ms: float = 1000.0
    max_sync_interval_ms: float = 10000.0
    min_confidence: float = 0.7

    # Distributed memory
    dsm_backend: StoreBackend = StoreBackend.MEMORY
    dsm_storage_path: str = "./data/dsm"

    # Metrics
    enable_metrics: bool = True
    metrics_output_path: str = "./data/metrics"

    # Logging
    log_level: str = "INFO"
    log_path: str | None = None

    def validate(self) -> None:
        """Validate configuration."""
        if self.max_agents <= 0:
            raise ValueError("max_agents must be positive")
        if self.drift_threshold <= 0:
            raise ValueError("drift_threshold must be positive")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be in [0, 1]")


@dataclass
class CollaborationResult:
    """Result of a multi-agent collaboration."""

    task_id: str
    task_description: str
    start_time: datetime
    end_time: datetime
    duration_ms: float

    # Participants
    agents_participated: list[str]
    total_sync_pulses: int
    total_tool_calls: int

    # Outcomes
    final_result: str | None
    agent_contributions: dict[str, str]

    # Metrics
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)

    # Success
    success: bool = True
    error: str | None = None


class AgentOS:
    """
    Main AgentOS system orchestrator.

    Integrates all phases into a unified multi-agent cognitive architecture.
    """

    def __init__(self, config: AgentOSConfig | None = None) -> None:
        """Initialize the AgentOS system.

        Args:
            config: System configuration. If None, uses defaults.
        """
        self.config = config or AgentOSConfig()
        self.config.validate()

        # System ID
        self.system_id = f"agentos_{uuid.uuid4().hex[:8]}"
        self.start_time = datetime.now()

        # Phase 1: Reasoning Kernel (lazy loaded)
        self._kernel: ReasoningKernel | None = None
        self._kernel_loaded = False

        # Phase 2: Memory Hierarchy
        self.smmu = SMMU(
            SMMUConfig(
                l1_config=L1CacheConfig(
                    max_tokens=self.config.l1_max_tokens_cache,
                    max_slices=self.config.l1_max_slices,
                ),
                l2_config=L2Config(
                    max_tokens=self.config.l2_max_tokens,
                    max_slices=self.config.l2_max_slices,
                ),
                l3_config=L3Config(
                    storage_path=self.config.l3_storage_path,
                ),
            )
        )

        # Cold Start Mitigation: L3 Bootstrap
        if self.config.l3_bootstrap_paths:
            loaded = self.smmu.bootstrap_l3_from_files(self.config.l3_bootstrap_paths)
            if loaded > 0:
                self._log(f"Bootstrapped L3 with {loaded} slices from domain knowledge")

        # Cold Start Mitigation: Session Restore
        if self.config.l3_restore_path:
            restored = self.smmu.restore_l3_state(self.config.l3_restore_path)
            if restored > 0:
                self._log(f"Restored L3 state with {restored} slices from previous session")

        # Cold Start Mitigation: Adaptive Scoring
        if self.config.enable_adaptive_scoring:
            self.smmu.enable_adaptive_scoring(warmup_turns=self.config.warmup_turns)
            self._log(f"Enabled adaptive importance scoring (warmup: {self.config.warmup_turns} turns)")

        # Phase 3: Scheduler & I/O
        self.scheduler = CognitiveScheduler(
            config=SchedulerConfig(
                time_slice_ms=self.config.scheduler_time_slice_ms,
                use_cognitive_fidelity=self.config.scheduler_use_cognitive_fidelity,
            )
        )

        self.peripherals = PeripheralRegistry()
        register_builtins(self.peripherals)

        # IVT automatically initializes with STANDARD_VECTORS
        self.ivt = InterruptVectorTable()

        # RIC initialized with kernel=None (set when kernel is loaded)
        self.ric = ReasoningInterruptCycle(
            scheduler=self.scheduler,
            peripherals=self.peripherals,
            ivt=self.ivt,
            kernel=None,  # Will be set when kernel property is first accessed
        )

        # Phase 4: Multi-Agent Sync
        self.csp_orchestrator = CSPOrchestrator(
            config=CSPOrchestratorConfig(
                min_sync_interval_ms=self.config.min_sync_interval_ms,
                max_sync_interval_ms=self.config.max_sync_interval_ms,
                min_agents_for_sync=2,
                sync_on_tool_completion=True,
            ),
            drift_config=DriftTrackerConfig(
                drift_threshold=self.config.drift_threshold,
            ),
        )

        self.dsm = DistributedSharedMemory(
            backend=self.config.dsm_backend,
            storage_path=self.config.dsm_storage_path,
        )

        self.reconciler = StateReconciler(
            config=ReconciliationConfig(
                conflict_strategy="latest",
            ),
        )

        # Phase 5: Metrics
        self.metrics = MetricsCalculator() if self.config.enable_metrics else None

        # Agent registry
        self._agents: dict[str, Agent] = {}

        # Task history
        self._task_history: list[CollaborationResult] = []

    @property
    def kernel(self) -> ReasoningKernel:
        """Get or create the reasoning kernel (lazy loaded)."""
        if self._kernel is None:
            self._kernel = create_kernel(
                model_name=self.config.model_name,
                device=self.config.device,
                l1_max_tokens=self.config.l1_max_tokens,
            )
            # Register kernel with RIC for interrupt handling
            self.ric.set_kernel(self._kernel)
        return self._kernel

    def load_kernel(self) -> None:
        """Load the reasoning kernel model."""
        if not self._kernel_loaded:
            self.kernel.load()
            self._kernel_loaded = True

    def unload_kernel(self) -> None:
        """Unload the reasoning kernel model."""
        if self._kernel_loaded and self._kernel:
            self.kernel.unload()
            self._kernel_loaded = False

    def spawn_agent(
        self,
        name: str,
        role: str = "general",
        priority: ThreadPriority = ThreadPriority.NORMAL,
        initial_task: str | None = None,
        metadata: dict[str, Any] | None = None,
        use_generation: bool = False,
        max_new_tokens: int = 80,
    ) -> Agent:
        """Spawn a new agent.

        Args:
            name: Agent name/identifier
            role: Agent's role (researcher, writer, analyst, etc.)
            priority: Thread priority for this agent
            initial_task: Initial task description
            metadata: Additional metadata
            use_generation: Whether to use actual LLM generation for responses
            max_new_tokens: Maximum tokens to generate per response

        Returns:
            The spawned Agent
        """
        agent_id = f"{name}_{uuid.uuid4().hex[:6]}"

        agent_config = AgentConfig(
            agent_id=agent_id,
            name=name,
            role=role,
            system_id=self.system_id,
            use_generation=use_generation,
            max_new_tokens=max_new_tokens,
        )

        agent = Agent(
            config=agent_config,
            kernel=self.kernel,
            smmu=self.smmu,
            scheduler=self.scheduler,
            csp_orchestrator=self.csp_orchestrator,
            dsm=self.dsm,
            metrics=self.metrics,
        )

        # Register with CSP orchestrator
        agent_state = agent.get_state()
        self.csp_orchestrator.register_agent(agent_state)

        # Store agent
        self._agents[agent_id] = agent

        # Spawn scheduler thread
        thread_id = self.scheduler.spawn_thread(
            priority=priority,
            initial_slice_id=None,
            metadata={"agent_id": agent_id, "role": role},
        )
        agent.set_thread_id(thread_id)

        return agent

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get an agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent if found, None otherwise
        """
        return self._agents.get(agent_id)

    def list_agents(self) -> list[Agent]:
        """List all agents.

        Returns:
            List of all agents
        """
        return list(self._agents.values())

    def collaborate(
        self,
        task: str,
        agents: list[str] | None = None,
        timeout_seconds: float = 300.0,
        sync_interval_ms: float = 5000.0,
    ) -> CollaborationResult:
        """Execute a collaborative task with multiple agents.

        Args:
            task: Task description
            agents: List of agent IDs to use (None = all agents)
            timeout_seconds: Maximum time to spend
            sync_interval_ms: How often to sync agents

        Returns:
            CollaborationResult with outcomes
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()

        # Determine which agents to use
        if agents is None:
            agents = list(self._agents.keys())

        participating_agents = [self._agents[aid] for aid in agents if aid in self._agents]

        if not participating_agents:
            return CollaborationResult(
                task_id=task_id,
                task_description=task,
                start_time=start_time,
                end_time=datetime.now(),
                duration_ms=0,
                agents_participated=[],
                total_sync_pulses=0,
                total_tool_calls=0,
                final_result=None,
                agent_contributions={},
                success=False,
                error="No agents available",
            )

        # Record metrics start
        if self.metrics:
            _ = self.metrics.calculate_cognitive_latency(
                interrupt_time=start_time.timestamp(),
                stable_time=start_time.timestamp(),
            )

        # Simulate collaboration
        contributions: dict[str, str] = {}
        total_sync_pulses = 0
        total_tool_calls = 0
        last_sync = start_time.timestamp()

        # Phase 1: Each agent processes the task in parallel
        for agent in participating_agents:
            contribution = agent.contribute_to_task(task)
            contributions[agent.agent_id] = contribution

            # Check if sync needed
            now = time.time()
            time_since_sync = (now - last_sync) * 1000
            if time_since_sync >= sync_interval_ms and self.config.enable_sync:
                # Trigger sync pulse to merge agent slices to global state
                pulse = self.csp_orchestrator.trigger_sync(
                    trigger="tool_completion",  # Use a valid trigger
                    source_agent_id=agent.agent_id,
                )
                total_sync_pulses += 1
                last_sync = now

        # Phase 2: Final sync to ensure all contributions are in global state
        if self.config.enable_sync:
            final_pulse = self.csp_orchestrator.trigger_sync(
                trigger="drift_threshold",  # Final sync trigger
                source_agent_id=participating_agents[0].agent_id if participating_agents else None,
            )
            total_sync_pulses += 1

        # Phase 3: Synthesizer produces final answer from global state
        final_result = self._produce_final_synthesis(task, contributions, participating_agents)

        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Collect metrics snapshot
        metrics_snapshot = {}
        if self.metrics:
            metrics_snapshot = self.metrics.get_summary()

        result = CollaborationResult(
            task_id=task_id,
            task_description=task,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            agents_participated=agents,
            total_sync_pulses=total_sync_pulses,
            total_tool_calls=total_tool_calls,
            final_result=final_result,
            agent_contributions=contributions,
            metrics_snapshot=metrics_snapshot,
        )

        self._task_history.append(result)

        # Cold Start Mitigation: Advance turn for adaptive scoring
        self.smmu.advance_turn()

        # Log warmup progress if in warmup period
        if not self.smmu.is_warmed_up:
            warmup_stats = self.get_warmup_stats()
            if warmup_stats:
                self._log(
                    f"Warmup progress: {warmup_stats['warmup_progress']:.1%} "
                    f"(turn {warmup_stats['turn_count']}/{warmup_stats['warmup_turns']})"
                )

        return result

    def _produce_final_synthesis(
        self,
        task: str,
        contributions: dict[str, str],
        participating_agents: list,
    ) -> str:
        """Produce final synthesized answer from global state.

        Args:
            task: The original task
            contributions: Individual agent contributions
            participating_agents: List of agents that participated

        Returns:
            Final synthesized answer
        """
        # Try to find an existing synthesizer agent
        synthesizer = None
        for agent in participating_agents:
            if agent.config.role == "synthesizer":
                synthesizer = agent
                break

        # If no synthesizer exists, create a temporary one for synthesis
        if synthesizer is None:
            # Create a temporary synthesizer config
            from agentos.agent import AgentConfig

            # Inherit generation setting from system config or participating agents
            use_gen = any(
                a.config.use_generation for a in participating_agents
            ) if participating_agents else False

            temp_config = AgentConfig(
                agent_id=f"temp_synthesizer_{uuid.uuid4().hex[:6]}",
                name="Synthesizer",
                role="synthesizer",
                system_id=self.system_id,
                use_generation=use_gen,  # Inherit from participating agents
            )

            # Create temporary synthesizer agent
            synthesizer = Agent(
                config=temp_config,
                kernel=self.kernel,
                smmu=self.smmu,
                scheduler=self.scheduler,
                csp_orchestrator=self.csp_orchestrator,
                dsm=self.dsm,
                metrics=self.metrics,
            )

        # Produce synthesis from global state
        return synthesizer.synthesize_from_global_state(task, contributions)

    def get_system_state(self) -> dict[str, Any]:
        """Get current system state.

        Returns:
            Dictionary with system state
        """
        return {
            "system_id": self.system_id,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "kernel_loaded": self._kernel_loaded,
            "total_agents": len(self._agents),
            "agent_states": {
                aid: agent.get_state() for aid, agent in self._agents.items()
            },
            "scheduler_stats": self.scheduler.get_statistics(),
            "csp_stats": self.csp_orchestrator.get_statistics(),
            "dsm_stats": self.dsm.get_statistics(),
            "metrics_summary": self.metrics.get_summary() if self.metrics else {},
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive system statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "system": {
                "system_id": self.system_id,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "config": {
                    "model_name": self.config.model_name,
                    "max_agents": self.config.max_agents,
                    "drift_threshold": self.config.drift_threshold,
                    "enable_sync": self.config.enable_sync,
                },
            },
            "agents": {
                "total": len(self._agents),
                "by_role": {},
                "agent_ids": list(self._agents.keys()),
            },
            "memory": self.smmu.get_memory_stats(),
            "scheduler": self.scheduler.get_statistics(),
            "sync": self.csp_orchestrator.get_statistics(),
            "dsm": self.dsm.get_statistics(),
            "tasks": {
                "total_tasks": len(self._task_history),
                "task_ids": [t.task_id for t in self._task_history],
            },
        }

        # Count agents by role
        for agent in self._agents.values():
            role = agent.config.role
            stats["agents"]["by_role"][role] = stats["agents"]["by_role"].get(role, 0) + 1

        # Add metrics if enabled
        if self.metrics:
            stats["metrics"] = self.metrics.get_summary()

        return stats

    def shutdown(self) -> None:
        """Shutdown the AgentOS system gracefully."""
        # Unload kernel
        self.unload_kernel()

        # Persist DSM if using FILE backend
        if self.dsm.backend == StoreBackend.FILE:
            self.dsm.persist_to_disk()

        # Cold Start Mitigation: Save L3 state for restore
        if self.config.l3_save_path:
            if self.smmu.save_l3_state(self.config.l3_save_path):
                self._log(f"Saved L3 state to {self.config.l3_save_path}")

        # Persist metrics if enabled
        if self.metrics and self.config.metrics_output_path:
            # Could save metrics to file here
            pass

    def _log(self, message: str, level: str = "INFO") -> None:
        """Internal logging method.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        import logging
        logger = logging.getLogger(__name__)
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(f"[AgentOS] {message}")

    def save_session(self, save_path: str | None = None) -> bool:
        """Save current session state for future restore.

        Args:
            save_path: Path to save state. If None, uses config.l3_save_path

        Returns:
            True if successful, False otherwise
        """
        path = save_path or self.config.l3_save_path
        if not path:
            return False
        return self.smmu.save_l3_state(path)

    def is_warmed_up(self) -> bool:
        """Check if system has completed warmup period.

        Returns:
            True if warmed up, False if still in warmup
        """
        return self.smmu.is_warmed_up

    def get_warmup_stats(self) -> dict[str, object] | None:
        """Get warmup statistics from adaptive scorer.

        Returns:
            Warmup stats dict if adaptive scoring is enabled, None otherwise
        """
        if hasattr(self.smmu, "adaptive_scorer") and self.smmu.adaptive_scorer:
            return self.smmu.adaptive_scorer.get_warmup_stats()
        return None

    def __enter__(self) -> "AgentOS":
        """Context manager entry."""
        self.load_kernel()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()


def create_agentos(config: AgentOSConfig | None = None) -> AgentOS:
    """Create an AgentOS system.

    Convenience function for creating an AgentOS instance.

    Args:
        config: System configuration

    Returns:
        Configured AgentOS instance
    """
    return AgentOS(config)
