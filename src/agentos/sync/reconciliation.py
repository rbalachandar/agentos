"""
Global State Reconciliation.

Based on AgentOS paper Section 3.4.3:

Reconciliation aggregates semantic slices across agents and resolves
conflicts when multiple agents have different versions of the same slice.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cosine

from agentos.sync.types import (
    AgentState,
    ConflictResolution,
    GlobalSemanticState,
    SemanticSliceVersion,
    SyncPulse,
    SyncTrigger,
)


@dataclass
class ReconciliationConfig:
    """Configuration for state reconciliation."""

    # Strategy for resolving conflicts
    conflict_strategy: str = "latest"  # "latest", "merge", "highest_fidelity"

    # Semantic similarity threshold for considering slices as "same"
    similarity_threshold: float = 0.9

    # Whether to use voting for conflict resolution
    use_voting: bool = False

    def validate(self) -> None:
        """Validate configuration."""
        valid_strategies = ["latest", "merge", "highest_fidelity"]
        if self.conflict_strategy not in valid_strategies:
            raise ValueError(f"conflict_strategy must be one of {valid_strategies}")
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in [0, 1]")


@dataclass
class SliceUpdate:
    """An update to a semantic slice from an agent."""

    slice_id: str
    agent_id: str
    version: SemanticSliceVersion
    timestamp: datetime


class StateReconciler:
    """Reconciles semantic state across multiple agents.

    Handles:
    - Aggregating slices from multiple agents
    - Detecting conflicts (same slice, different versions)
    - Resolving conflicts using configured strategy
    - Updating global state
    """

    def __init__(self, config: ReconciliationConfig | None = None) -> None:
        """Initialize the state reconciler.

        Args:
            config: Configuration for reconciliation
        """
        self.config = config or ReconciliationConfig()
        self.config.validate()

        # Update history (for conflict resolution)
        self._pending_updates: dict[str, list[SliceUpdate]] = {}

        # Statistics
        self._total_conflicts = 0
        self._resolved_conflicts = 0

    def reconcile(
        self,
        global_state: GlobalSemanticState,
        agent_states: list[AgentState],
    ) -> list[ConflictResolution]:
        """Reconcile agent states into global state.

        Args:
            global_state: Current global semantic state
            agent_states: List of agent states to reconcile

        Returns:
            List of ConflictResolution for resolved conflicts
        """
        resolutions = []

        # Collect all slice updates from agents
        for agent in agent_states:
            for slice_id in agent.active_slices:
                # Create update (in production, would have actual slice data)
                version = SemanticSliceVersion(
                    slice_id=slice_id,
                    agent_id=agent.agent_id,
                    version=int(time.time() * 1000) % 1000000,  # Simple versioning
                    content=f"[content from {agent.agent_id}]",
                    created_at=datetime.now(),
                )

                self._add_update(global_state, slice_id, agent.agent_id, version)

        # Detect and resolve conflicts
        conflicts = self._detect_conflicts(global_state)

        for conflict in conflicts:
            resolution = self._resolve_conflict(global_state, conflict)
            resolutions.append(resolution)

            # Apply resolution to global state
            if resolution.winning_version:
                global_state.update_slice(resolution.winning_version)

        return resolutions

    def _add_update(
        self,
        global_state: GlobalSemanticState,
        slice_id: str,
        agent_id: str,
        version: SemanticSliceVersion,
    ) -> None:
        """Add a slice update to pending updates.

        Args:
            global_state: Global semantic state
            slice_id: Slice identifier
            agent_id: Agent providing update
            version: New version of slice
        """
        if slice_id not in self._pending_updates:
            self._pending_updates[slice_id] = []

        self._pending_updates[slice_id].append(
            SliceUpdate(
                slice_id=slice_id,
                agent_id=agent_id,
                version=version,
                timestamp=datetime.now(),
            )
        )

    def _detect_conflicts(
        self, global_state: GlobalSemanticState
    ) -> list[list[SliceUpdate]]:
        """Detect conflicting slice updates.

        Args:
            global_state: Global semantic state

        Returns:
            List of groups of conflicting updates
        """
        conflicts = []

        for slice_id, updates in self._pending_updates.items():
            if len(updates) > 1:
                # Multiple agents updated the same slice = conflict
                self._total_conflicts += 1
                conflicts.append(updates)

        return conflicts

    def _resolve_conflict(
        self,
        global_state: GlobalSemanticState,
        updates: list[SliceUpdate],
    ) -> ConflictResolution:
        """Resolve a conflict between multiple slice versions.

        Args:
            global_state: Global semantic state
            updates: Conflicting updates

        Returns:
            ConflictResolution with result
        """
        slice_id = updates[0].slice_id
        versions = [u.version for u in updates]

        winning_version: SemanticSliceVersion | None = None
        rejected_versions: list[SemanticSliceVersion] = []

        if self.config.conflict_strategy == "latest":
            # Pick the most recent (by timestamp)
            sorted_updates = sorted(updates, key=lambda u: u.timestamp, reverse=True)
            winning_version = sorted_updates[0].version
            rejected_versions = [u.version for u in sorted_updates[1:]]

        elif self.config.conflict_strategy == "highest_fidelity":
            # Pick the version with highest cognitive fidelity
            # Fidelity scores should be computed from slice metadata
            raise NotImplementedError(
                "highest_fidelity conflict resolution strategy requires implementation. "
                "Fidelity scores must be computed from slice metadata (attention focus, "
                "semantic coherence, etc.) to determine which version has highest quality."
            )

        elif self.config.conflict_strategy == "merge":
            # Merge strategy: combine content from all versions
            # Create a merged version
            merged_content = self._merge_versions(versions)

            winning_version = SemanticSliceVersion(
                slice_id=slice_id,
                agent_id="merged",
                version=max(v.version for v in versions) + 1,
                content=merged_content,
                created_at=datetime.now(),
            )
            rejected_versions = versions

        self._resolved_conflicts += 1

        # Calculate coherence score for the resolution
        coherence_score = self._calculate_resolution_coherence(
            winning_version, rejected_versions, global_state
        )

        return ConflictResolution(
            slice_id=slice_id,
            winning_version=winning_version,
            rejected_versions=rejected_versions,
            resolution_strategy=self.config.conflict_strategy,
            coherence_score=coherence_score,
        )

    def _calculate_resolution_coherence(
        self,
        winning_version: SemanticSliceVersion,
        rejected_versions: list[SemanticSliceVersion],
        global_state: GlobalSemanticState,
    ) -> float:
        """Calculate coherence score for a conflict resolution.

        Higher coherence means the resolution maintains better semantic alignment.

        Args:
            winning_version: The version that won
            rejected_versions: Versions that were rejected
            global_state: Current global semantic state

        Returns:
            Coherence score (0-1, higher is better)
        """
        all_versions = [winning_version] + rejected_versions

        # If we have embeddings, use semantic similarity
        embeddings = [v.embedding for v in all_versions if v.embedding is not None]

        if len(embeddings) >= 2:
            # Calculate pairwise cosine similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = 1.0 - cosine(embeddings[i], embeddings[j])
                    similarities.append(sim)

            # Coherence = average similarity
            return float(np.mean(similarities)) if similarities else 0.5

        # Fallback: coherence based on content length consistency
        # Versions with similar content lengths suggest coherent interpretations
        lengths = [len(v.content) for v in all_versions]
        if len(lengths) >= 2:
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)
            # Lower std relative to mean = higher coherence
            cv = std_length / mean_length if mean_length > 0 else 1.0
            return float(max(0.0, 1.0 - cv))

        # No information available
        return 0.5

    def _merge_versions(
        self, versions: list[SemanticSliceVersion]
    ) -> str:
        """Merge multiple slice versions into one.

        Args:
            versions: Versions to merge

        Returns:
            Merged content
        """
        # Simple merging: concatenate with separator
        contents = [v.content for v in versions]
        return " | ".join(contents)

    def calculate_coherence(
        self,
        global_state: GlobalSemanticState,
    ) -> float:
        """Calculate overall coherence of global state.

        Measures how well-aligned all agents are with the global state.

        Args:
            global_state: Global semantic state

        Returns:
            Coherence score (0-1, higher is better)
        """
        if not global_state.slices:
            return 1.0

        # Calculate coherence based on semantic similarity of slices
        # If slices have embeddings, use cosine similarity
        embeddings = [
            v.embedding for v in global_state.slices.values()
            if v.embedding is not None
        ]

        if len(embeddings) >= 2:
            # Calculate average pairwise similarity
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = 1.0 - cosine(embeddings[i], embeddings[j])
                    similarities.append(sim)

            # Coherence = average similarity
            return float(np.mean(similarities)) if similarities else 0.5

        # No embeddings available - use conflict-based heuristic
        # Fewer conflicts relative to slice count = higher coherence
        if len(global_state.slices) > 0:
            conflict_ratio = self._total_conflicts / len(global_state.slices)
            return float(max(0.0, 1.0 - conflict_ratio))

        return 1.0

    def get_statistics(self) -> dict[str, Any]:
        """Get reconciler statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_conflicts": self._total_conflicts,
            "resolved_conflicts": self._resolved_conflicts,
            "pending_updates": len(self._pending_updates),
            "conflict_strategy": self.config.conflict_strategy,
        }


def reconcile_sync_pulse(
    reconciler: StateReconciler,
    global_state: GlobalSemanticState,
    agent_states: list[AgentState],
) -> SyncPulse:
    """Reconcile states as part of a sync pulse.

    Convenience function that performs reconciliation and returns
    an updated SyncPulse.

    Args:
        reconciler: State reconciler
        global_state: Current global state
        agent_states: Agent states to reconcile

    Returns:
        Updated SyncPulse with reconciliation results
    """
    resolutions = reconciler.reconcile(global_state, agent_states)

    return SyncPulse(
        pulse_id=f"sync_reconciled_{uuid.uuid4().hex[:8]}",
        trigger=SyncTrigger.PERIODIC,
        timestamp=datetime.now(),
        initiated_by="reconciler",
        agents_synced=len(agent_states),
        conflicts_resolved=len(resolutions),
    )
