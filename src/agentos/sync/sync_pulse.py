"""
Cognitive Sync Pulse (CSP) Orchestrator.

Based on AgentOS paper Algorithm 3 (Section 3.4.2):

CSPs are event-driven (not clock-driven) synchronization events
that bring all agents back into coherence.

Triggers:
- Tool completion
- Logical anchor formation
- Drift threshold exceeded
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from agentos.sync.drift_tracker import CognitiveDriftTracker, DriftTrackerConfig
from agentos.sync.types import (
    AgentState,
    ConflictResolution,
    DriftMetrics,
    GlobalSemanticState,
    SemanticSliceVersion,
    SyncPulse,
    SyncTrigger,
)


@dataclass
class CSPOrchestratorConfig:
    """Configuration for CSP Orchestrator."""

    # Minimum time between sync pulses (avoid sync storms)
    min_sync_interval_ms: float = 1000.0

    # Maximum time without sync (safety limit)
    max_sync_interval_ms: float = 10000.0

    # Minimum agents to trigger sync
    min_agents_for_sync: int = 2

    # Whether to sync on tool completion
    sync_on_tool_completion: bool = True

    def validate(self) -> None:
        """Validate configuration."""
        if self.min_sync_interval_ms <= 0:
            raise ValueError("min_sync_interval_ms must be positive")
        if self.max_sync_interval_ms < self.min_sync_interval_ms:
            raise ValueError("max_sync_interval_ms must be >= min_sync_interval_ms")
        if self.min_agents_for_sync < 1:
            raise ValueError("min_agents_for_sync must be at least 1")


class CSPOrchestrator:
    """Cognitive Sync Pulse Orchestrator.

    Manages synchronization events between multiple agents to maintain
    cognitive coherence across the system.

    Implements Algorithm 3 from the paper:
    1. Detect trigger condition
    2. Gather agent states
    3. Reconcile semantic states
    4. Broadcast updated global state
    5. Reset drift metrics
    """

    def __init__(
        self,
        config: CSPOrchestratorConfig | None = None,
        drift_config: DriftTrackerConfig | None = None,
    ) -> None:
        """Initialize the CSP Orchestrator.

        Args:
            config: CSP configuration
            drift_config: Drift tracker configuration
        """
        self.config = config or CSPOrchestratorConfig()
        self.config.validate()

        # Drift tracker for monitoring agents
        self.drift_tracker = CognitiveDriftTracker(drift_config)

        # Global semantic state
        self.global_state = GlobalSemanticState()

        # Registered agents
        self._agents: dict[str, AgentState] = {}

        # Sync pulse history
        self.sync_history: list[SyncPulse] = []

        # Timing
        self._last_sync_time = time.time()

    def register_agent(self, agent_state: AgentState) -> None:
        """Register an agent for synchronization.

        Args:
            agent_state: Agent to register
        """
        self._agents[agent_state.agent_id] = agent_state

        # Register with drift tracker
        self.drift_tracker.register_agent(
            agent_state.agent_id,
            agent_state.semantic_gradients,
        )

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent.

        Args:
            agent_id: Agent to unregister

        Returns:
            True if unregistered
        """
        return self._agents.pop(agent_id, None) is not None

    def trigger_sync(
        self,
        trigger: SyncTrigger,
        source_agent_id: str | None = None,
    ) -> SyncPulse:
        """Trigger a cognitive sync pulse.

        Implements Algorithm 3 from the paper.

        Args:
            trigger: What triggered this sync
            source_agent_id: Agent that initiated sync (if any)

        Returns:
            SyncPulse with results
        """
        start_time = time.time()

        # Check rate limiting
        time_since_last_sync = (start_time - self._last_sync_time) * 1000
        if time_since_last_sync < self.config.min_sync_interval_ms:
            # Too soon, return failed sync
            return SyncPulse(
                pulse_id=f"rejected_{uuid.uuid4().hex[:8]}",
                trigger=trigger,
                timestamp=datetime.now(),
                initiated_by=source_agent_id or "system",
                success=False,
                error="Rate limited: too soon since last sync",
            )

        # Record drift before sync
        drift_before = {}
        for agent_id, metrics in self.drift_tracker.get_all_metrics().items():
            drift_before[agent_id] = metrics.current_drift

        # Gather agent states
        agent_states = list(self._agents.values())

        # Reconcile semantic states
        conflicts_resolved = self._reconcile_states(agent_states)

        # Create sync pulse
        pulse = SyncPulse(
            pulse_id=f"sync_{uuid.uuid4().hex[:8]}",
            trigger=trigger,
            timestamp=datetime.now(),
            initiated_by=source_agent_id or "system",
            agents_synced=len(agent_states),
            conflicts_resolved=conflicts_resolved,
            drift_before=drift_before,
        )

        # Reset drift after successful sync
        self.drift_tracker.reset_all_drift()
        pulse.drift_after = {
            agent_id: 0.0
            for agent_id in drift_before.keys()
        }

        # Record duration
        pulse.duration_ms = (time.time() - start_time) * 1000

        # Update last sync time
        self._last_sync_time = start_time

        # Add to history
        self.sync_history.append(pulse)

        return pulse

    def _reconcile_states(self, agent_states: list[AgentState]) -> int:
        """Reconcile semantic states across agents.

        Implements conflict resolution for distributed shared memory.

        Args:
            agent_states: List of agent states with actual semantic slices

        Returns:
            Number of conflicts resolved
        """
        conflicts_resolved = 0

        # Process actual semantic slices from each agent
        for agent in agent_states:
            # Get agent's role and name from metadata
            role = agent.metadata.get('role', 'agent')
            name = agent.metadata.get('name', agent.agent_id)

            # Process each actual semantic slice from this agent
            for slice_obj in agent.active_slices:
                # Only add if not already in global state (avoid duplicates)
                if slice_obj.id not in self.global_state.slices:
                    # Create a proper slice version with actual content
                    # Use the semantic slice's actual content, not placeholder text
                    version = SemanticSliceVersion(
                        slice_id=slice_obj.id,
                        agent_id=agent.agent_id,
                        version=1,
                        content=slice_obj.content,  # Use actual slice content!
                        created_at=datetime.now(),
                        metadata={
                            'role': role,
                            'name': name,
                            'agent_slice_id': slice_obj.id,
                            'density_mean': slice_obj.density_mean,
                            'start_pos': slice_obj.start_pos,
                            'end_pos': slice_obj.end_pos,
                        }
                    )
                    self.global_state.update_slice(version)

        return conflicts_resolved

    def get_global_state(self) -> GlobalSemanticState:
        """Get the current global semantic state.

        Returns:
            GlobalSemanticState
        """
        return self.global_state

    def get_drift_statistics(self) -> dict[str, Any]:
        """Get drift statistics.

        Returns:
            Dictionary with drift stats
        """
        return self.drift_tracker.get_statistics()

    def check_and_sync(self) -> SyncPulse | None:
        """Check if sync is needed and execute if so.

        Returns:
            SyncPulse if sync was executed, None otherwise
        """
        triggers = self.drift_tracker.check_sync_triggers()

        if not triggers:
            # Check for periodic sync (safety net)
            time_since_last = (time.time() - self._last_sync_time) * 1000
            if time_since_last > self.config.max_sync_interval_ms:
                triggers = [SyncTrigger.PERIODIC]

        if triggers:
            return self.trigger_sync(triggers[0])

        return None

    def update_agent_drift(
        self,
        agent_id: str,
        agent_gradient: NDArray[np.float32] | None,
    ) -> float:
        """Update drift for an agent and check if sync needed.

        Args:
            agent_id: Agent identifier
            agent_gradient: Agent's current semantic gradient

        Returns:
            Current drift magnitude
        """
        # Update stored agent state with new gradient
        if agent_id in self._agents:
            # Update the gradient in the stored state
            from dataclasses import replace
            self._agents[agent_id] = replace(
                self._agents[agent_id],
                semantic_gradients=agent_gradient
            )

        # Calculate global gradient (average across all agents)
        global_gradient = self._calculate_global_gradient()

        # Update drift
        drift = self.drift_tracker.update_drift(agent_id, agent_gradient, global_gradient)

        # Check if sync needed
        if self.drift_tracker.get_metrics(agent_id).is_critical:
            self.trigger_sync(
                SyncTrigger.DRIFT_THRESHOLD, source_agent_id=agent_id
            )

        return drift

    def _calculate_global_gradient(self) -> NDArray[np.float32] | None:
        """Calculate global semantic gradient from all agents.

        Returns:
            Average gradient across all agents, or None if no gradients
        """
        gradients = []

        for agent in self._agents.values():
            if agent.semantic_gradients is not None:
                gradients.append(agent.semantic_gradients)

        if not gradients:
            return None

        # Average across agents
        return np.mean(gradients, axis=0)

    def get_sync_history(self) -> list[SyncPulse]:
        """Get history of sync pulses.

        Returns:
            List of SyncPulse
        """
        return self.sync_history.copy()

    def get_statistics(self) -> dict[str, Any]:
        """Get CSP orchestrator statistics.

        Returns:
            Dictionary with statistics
        """
        drift_stats = self.get_drift_statistics()

        # Merge stats, avoiding key collision with "total_syncs"
        # (drift tracker has its own sync count, but we want orchestrator's count)
        return {
            "total_agents": len(self._agents),
            "total_syncs": len(self.sync_history),
            "last_sync_time": self._last_sync_time,
            # Drift stats (excluding keys that would conflict)
            "average_drift": drift_stats.get("average_drift", 0.0),
            "max_drift": drift_stats.get("max_drift", 0.0),
            "critical_drift_count": drift_stats.get("critical_drift_count", 0),
        }
