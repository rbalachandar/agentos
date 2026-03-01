"""
Core types for Multi-Agent Synchronization.

Based on AgentOS paper Section 3.4:
- Cognitive Drift: How much an agent's state diverges from global
- Cognitive Sync Pulse (CSP): Events that trigger re-synchronization
- Version Vectors: Track conflicting updates to shared memory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from agentos.memory.slicing.types import SemanticSlice


class SyncTrigger(str, Enum):
    """Events that trigger a sync pulse (Section 3.4.2)."""

    TOOL_COMPLETION = "tool_completion"  # Tool finished executing
    LOGICAL_ANCHOR = "logical_anchor"  # New stable semantic anchor formed
    DRIFT_THRESHOLD = "drift_threshold"  # Agent drifted too far
    PERIODIC = "periodic"  # Time-based sync (fallback)
    USER_REQUEST = "user_request"  # Explicit sync request


@dataclass
class DriftMetrics:
    """Cognitive drift metrics for an agent.

    Formula (3) from paper:
        Δψᵢ(t) = ∫₀ᵗ ‖∇Φᵢ(σ,τ) - ∇S_global(τ)‖ dτ

    Where:
    - ∇Φᵢ(σ,τ): Agent i's semantic gradient
    - ∇S_global(τ): Global semantic gradient
    - Δψᵢ(t): Cumulative drift over time
    """

    agent_id: str
    current_drift: float  # Current drift magnitude
    drift_rate: float  # How fast drift is accumulating
    gradient_norm: float  # ‖∇Φᵢ - ∇S_global‖

    # History
    drift_history: list[tuple[datetime, float]] = field(default_factory=list)

    # Threshold
    drift_threshold: float = 1.0  # Alert when drift exceeds this

    @property
    def is_critical(self) -> bool:
        """Whether drift has exceeded critical threshold."""
        return self.current_drift > self.drift_threshold


@dataclass
class SyncPulse:
    """A Cognitive Sync Pulse event.

    CSPs synchronize all agents' semantic states to maintain coherence.
    """

    pulse_id: str
    trigger: SyncTrigger
    timestamp: datetime
    initiated_by: str  # Agent ID or "system"

    # Pulse metrics
    duration_ms: float = 0.0
    agents_synced: int = 0
    conflicts_resolved: int = 0

    # Before/after drift snapshots
    drift_before: dict[str, float] = field(default_factory=dict)
    drift_after: dict[str, float] = field(default_factory=dict)

    # Result
    success: bool = True
    error: str | None = None


@dataclass
class SemanticSliceVersion:
    """A version of a semantic slice for conflict resolution.

    Used in distributed shared memory with version vectors.
    """

    slice_id: str
    agent_id: str  # Who created this version
    version: int  # Monotonically increasing

    content: str  # Slice content
    embedding: NDArray[np.float32] | None = None  # Semantic embedding

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    merged_at: datetime | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_ms(self) -> float:
        """Age of this version in milliseconds."""
        delta = datetime.now() - self.created_at
        return delta.total_seconds() * 1000.0


@dataclass
class ConflictResolution:
    """Result of resolving conflicting slice versions."""

    slice_id: str
    winning_version: SemanticSliceVersion
    rejected_versions: list[SemanticSliceVersion]

    resolution_strategy: str  # "latest", "merge", "highest_fidelity"
    coherence_score: float  # 0-1, how coherent is the result


@dataclass
class GlobalSemanticState:
    """The global semantic state shared across all agents.

    Represents the "ground truth" that agents try to stay synchronized with.
    """

    # Shared semantic memory
    slices: dict[str, SemanticSliceVersion] = field(default_factory=dict)  # slice_id -> latest version

    # Global semantic gradient (average across agents)
    global_gradient: NDArray[np.float32] | None = None

    # Version vector for conflict detection
    # Map: agent_id -> version number for each slice
    version_vectors: dict[str, dict[str, int]] = field(default_factory=dict)

    # Metadata
    last_update: datetime = field(default_factory=datetime.now)
    update_count: int = 0

    def get_slice(self, slice_id: str) -> SemanticSliceVersion | None:
        """Get the current version of a slice.

        Args:
            slice_id: Slice identifier

        Returns:
            SemanticSliceVersion if found
        """
        return self.slices.get(slice_id)

    def update_slice(self, version: SemanticSliceVersion) -> bool:
        """Update a slice with a new version.

        Args:
            version: New version of the slice

        Returns:
            True if updated, False if conflict detected
        """
        # Check version vector for conflicts
        current = self.slices.get(version.slice_id)

        if current:
            # Check if this is a newer version
            if version.version <= current.version:
                # Stale version, reject
                return False

        # Update slice
        self.slices[version.slice_id] = version
        self.last_update = datetime.now()
        self.update_count += 1

        return True

    def get_conflicts(self) -> list[tuple[str, list[SemanticSliceVersion]]]:
        """Find conflicting slice versions.

        Returns:
            List of (slice_id, list of conflicting versions)
        """
        # In production, would scan version vectors for divergent versions
        # For now, return empty (no conflicts)
        return []


@dataclass
class AgentState:
    """The current state of an agent.

    Used for drift tracking and synchronization.
    """

    agent_id: str
    semantic_gradients: NDArray[np.float32] | None = None

    # Active context (which slices are in L1)
    # Now stores actual slice objects with content, not just IDs
    active_slices: list[SemanticSlice] = field(default_factory=list)

    # Quick lookup for slices by ID
    _slices_by_id: dict[str, SemanticSlice] = field(default_factory=dict)

    # Current drift
    drift: float = 0.0

    # Last sync time
    last_sync: datetime | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_slice(self, slice_obj: SemanticSlice) -> None:
        """Add a semantic slice to this agent state.

        Args:
            slice_obj: The semantic slice to add
        """
        self.active_slices.append(slice_obj)
        self._slices_by_id[slice_obj.id] = slice_obj

    def get_slice(self, slice_id: str) -> SemanticSlice | None:
        """Get a slice by ID.

        Args:
            slice_id: Slice identifier

        Returns:
            SemanticSlice if found, None otherwise
        """
        return self._slices_by_id.get(slice_id)

    def get_slice_ids(self) -> list[str]:
        """Get list of slice IDs.

        Returns:
            List of slice IDs
        """
        return [s.id for s in self.active_slices]
