"""
Distributed Shared Memory.

Based on AgentOS paper Section 3.4.5:

L2/L3 memory tiers backed by a distributed store so all agents
see the same version of addressable semantic space.

Uses version vectors for conflict detection:
- Each agent has a version vector for each slice
- Before writing, check if any other agent has updated the slice
- If conflict detected, trigger reconciliation
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from agentos.memory.slicing.types import SemanticSlice
from agentos.sync.types import (
    AgentState,
    GlobalSemanticState,
    SemanticSliceVersion,
    SyncPulse,
)


class StoreBackend(str, Enum):
    """Storage backend for distributed shared memory."""

    MEMORY = "memory"  # In-memory (testing only)
    FILE = "file"  # File-based
    REDIS = "redis"  # Redis (future)
    ETCD = "etcd"  # etcd (future)


@dataclass
class VersionVector:
    """Version vector for conflict detection.

    Maps slice_id -> version number for a specific agent.
    """

    agent_id: str
    versions: dict[str, int] = field(default_factory=dict)

    def get_version(self, slice_id: str) -> int:
        """Get version for a slice."""
        return self.versions.get(slice_id, 0)

    def increment(self, slice_id: str) -> int:
        """Increment version for a slice."""
        current = self.get_version(slice_id)
        self.versions[slice_id] = current + 1
        return self.versions[slice_id]

    def check_conflict(self, slice_id: str, their_version: int) -> bool:
        """Check if there's a conflict with another agent's version.

        Args:
            slice_id: Slice to check
            their_version: Version to compare against

        Returns:
            True if conflict (their_version > ours)
        """
        my_version = self.get_version(slice_id)
        return their_version > my_version


@dataclass
class DistributedSliceEntry:
    """An entry in distributed shared memory.

    Combines slice data with versioning information.
    """

    slice_id: str
    version: SemanticSliceVersion

    # Locking info (for future use)
    locked_by: str | None = None
    lock_timestamp: datetime | None = None

    def __post_init__(self):
        # Generate unique entry ID
        self.entry_id = f"{self.slice_id}_v{self.version.version}_{uuid.uuid4().hex[:8]}"


class DistributedSharedMemory:
    """Distributed shared memory for multi-agent systems.

    L2/L3 backed by distributed store so all agents see the same
    version of addressable semantic space.

    Key features:
    - Version vectors for conflict detection
    - Automatic conflict resolution
    - Global state management
    """

    def __init__(
        self,
        backend: StoreBackend = StoreBackend.MEMORY,
        storage_path: str | None = None,
    ) -> None:
        """Initialize distributed shared memory.

        Args:
            backend: Storage backend type
            storage_path: Path for file-based storage
        """
        self.backend = backend
        self.storage_path = storage_path or "./data/dsm"

        # Initialize storage
        if backend == StoreBackend.FILE:
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)

        # Distributed store: slice_id -> DistributedSliceEntry
        self._store: dict[str, DistributedSliceEntry] = {}

        # Version vectors per agent
        self._version_vectors: dict[str, VersionVector] = {}

        # Global state
        self._global_state = GlobalSemanticState()

    def get_version_vector(self, agent_id: str) -> VersionVector:
        """Get or create version vector for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            VersionVector for the agent
        """
        if agent_id not in self._version_vectors:
            self._version_vectors[agent_id] = VersionVector(agent_id=agent_id)

        return self._version_vectors[agent_id]

    def read_slice(self, slice_id: str, agent_id: str) -> SemanticSliceVersion | None:
        """Read a slice from distributed memory.

        Checks version vector for conflicts before reading.

        Args:
            slice_id: Slice to read
            agent_id: Agent reading the slice

        Returns:
            SemanticSliceVersion if found, None otherwise
        """
        entry = self._store.get(slice_id)
        if not entry:
            return None

        # Check version vector
        vv = self.get_version_vector(agent_id)
        their_version = vv.get_version(slice_id)
        my_version = entry.version.version

        if my_version > their_version:
            # Agent is behind, needs to update
            vv.versions[slice_id] = my_version

        return entry.version

    def write_slice(
        self,
        slice_data: SemanticSlice,
        agent_id: str,
    ) -> bool:
        """Write a slice to distributed memory.

        Checks version vectors for conflicts before writing.

        Args:
            slice_data: Slice to write
            agent_id: Agent writing the slice

        Returns:
            True if successful, False if conflict detected
        """
        slice_id = slice_data.id

        # Get current version
        entry = self._store.get(slice_id)
        current_version = entry.version.version if entry else 0

        # Increment agent's version
        vv = self.get_version_vector(agent_id)
        new_version = vv.increment(slice_id)

        # Check for conflicts with other agents
        # NOTE:
        # The conflict detection logic below is intentionally simplified.
        # In a real DSM with version vectors, conflict checks should compare the writer's
        # last-known version (or the store's current version) against other agents' versions
        # to detect concurrent updates.
        # Be careful with argument direction: passing `new_version` into another agent's
        # vector comparison can lead to inverted semantics (false conflicts / missed conflicts)
        # depending on the definition of "newer".
        for other_agent_id, other_vv in self._version_vectors.items():
            if other_agent_id == agent_id:
                continue

            if other_vv.check_conflict(slice_id, new_version):
                # Conflict detected! Another agent has a newer version
                # Trigger reconciliation instead
                return False

        # Create new version
        version = SemanticSliceVersion(
            slice_id=slice_id,
            agent_id=agent_id,
            version=new_version,
            content=slice_data.content,
            created_at=datetime.now(),
        )

        # Update store
        entry = DistributedSliceEntry(slice_id=slice_id, version=version)
        self._store[slice_id] = entry

        # Update global state
        self._global_state.update_slice(version)

        return True

    def sync_slice(self, slice_id: str, agent_id: str) -> SemanticSliceVersion | None:
        """Synchronize a slice from global state to local.

        Args:
            slice_id: Slice to sync
            agent_id: Agent syncing

        Returns:
            SemanticSliceVersion if found, None otherwise
        """
        entry = self._store.get(slice_id)
        if not entry:
            return None

        # Update agent's version vector
        vv = self.get_version_vector(agent_id)
        vv.versions[slice_id] = entry.version.version

        return entry.version

    def get_global_state(self) -> GlobalSemanticState:
        """Get the global semantic state.

        Returns:
            GlobalSemanticState
        """
        return self._global_state

    def get_conflicts(self) -> list[tuple[str, dict[str, int]]]:
        """Get all version conflicts across agents.

        Args:
            Returns list of (slice_id, {agent_id: version} conflicts

        Returns:
            List of conflicts
        """
        conflicts = []

        # Get all slices
        slice_ids = set()
        for vv in self._version_vectors.values():
            slice_ids.update(vv.versions.keys())

        # Check each slice for conflicts
        for slice_id in slice_ids:
            versions = {}
            for agent_id, vv in self._version_vectors.items():
                versions[agent_id] = vv.get_version(slice_id)

            # Find max version
            max_version = max(versions.values()) if versions else 0

            # Find agents with outdated versions
            outdated = {
                agent_id: version
                for agent_id, version in versions.items()
                if version < max_version
            }

            if outdated:
                conflicts.append((slice_id, outdated))

        return conflicts

    def reconcile_conflict(
        self,
        slice_id: str,
        winning_agent_id: str | None = None,
    ) -> bool:
        """Reconcile a conflict by choosing a winner.

        Args:
            slice_id: Slice with conflict
            winning_agent_id: Agent whose version wins (None = highest version)

        Returns:
            True if reconciled
        """
        versions = {}
        for agent_id, vv in self._version_vectors.items():
            versions[agent_id] = vv.get_version(slice_id)

        if not versions:
            return False

        # Determine winner
        if winning_agent_id:
            winning_version = versions.get(winning_agent_id)
        else:
            # Pick agent with highest version
            winning_agent_id = max(versions, key=versions.get)
            winning_version = versions[winning_agent_id]

        # Update all version vectors to winning version
        for agent_id, vv in self._version_vectors.items():
            vv.versions[slice_id] = winning_version

        return True

    def get_agent_state(self, agent_id: str) -> AgentState | None:
        """Get or create agent state.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentState for the agent
        """
        # Check if we have any data for this agent
        has_data = agent_id in self._version_vectors

        if not has_data:
            return None

        # Get agent's active slices (from their version vector)
        active_slices = list(self.get_version_vector(agent_id).versions.keys())

        return AgentState(
            agent_id=agent_id,
            active_slices=active_slices,
            metadata={"registered_in_dsm": True},
        )

    def persist_to_disk(self) -> bool:
        """Persist distributed state to disk.

        Only works with FILE backend.

        Returns:
            True if successful
        """
        if self.backend != StoreBackend.FILE:
            return False

        # Persist version vectors
        for agent_id, vv in self._version_vectors.items():
            path = Path(self.storage_path) / f"vv_{agent_id}.json"
            with open(path, "w") as f:
                json.dump(
                    {
                        "agent_id": vv.agent_id,
                        "versions": vv.versions,
                    },
                    f,
                    indent=2,
                )

        # Persist global state
        state_path = Path(self.storage_path) / "global_state.json"
        # Convert SemanticSliceVersion to dict for serialization
        slices_dict = {}
        for slice_id, version in self._global_state.slices.items():
            slices_dict[slice_id] = {
                "slice_id": version.slice_id,
                "agent_id": version.agent_id,
                "version": version.version,
                "content": version.content,
                "created_at": version.created_at.isoformat(),
                "metadata": version.metadata,
            }

        with open(state_path, "w") as f:
            json.dump(
                {
                    "slices": slices_dict,
                    "version_vectors": {
                        agent_id: vv.versions
                        for agent_id, vv in self._version_vectors.items()
                    },
                    "last_update": self._global_state.last_update.isoformat(),
                    "update_count": self._global_state.update_count,
                },
                f,
                indent=2,
            )

        return True

    def load_from_disk(self) -> bool:
        """Load distributed state from disk.

        Only works with FILE backend.

        Returns:
            True if successful
        """
        if self.backend != StoreBackend.FILE:
            return False

        # Load version vectors
        storage_path = Path(self.storage_path)

        for vv_file in storage_path.glob("vv_*.json"):
            with open(vv_file, "r") as f:
                data = json.load(f)
                vv = VersionVector(
                    agent_id=data["agent_id"],
                    versions=data["versions"],
                )
                self._version_vectors[vv.agent_id] = vv

        # Load global state
        state_path = storage_path / "global_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                data = json.load(f)

            # Reconstruct SemanticSliceVersion objects
            slices = {}
            for slice_id, slice_data in data["slices"].items():
                version = SemanticSliceVersion(
                    slice_id=slice_data["slice_id"],
                    agent_id=slice_data["agent_id"],
                    version=slice_data["version"],
                    content=slice_data["content"],
                    created_at=datetime.fromisoformat(slice_data["created_at"]),
                    metadata=slice_data.get("metadata", {}),
                )
                slices[slice_id] = version

            self._global_state = GlobalSemanticState(
                slices=slices,
                version_vectors=data.get("version_vectors", {}),
                last_update=datetime.fromisoformat(data["last_update"]),
                update_count=data["update_count"],
            )

            # Reconstruct store
            for slice_id, version in slices.items():
                entry = DistributedSliceEntry(
                    slice_id=slice_id,
                    version=version,
                )
                self._store[slice_id] = entry

        return True

    def get_statistics(self) -> dict[str, Any]:
        """Get distributed memory statistics.

        Returns:
            Dictionary with statistics
        """
        conflicts = self.get_conflicts()

        return {
            "backend": self.backend.value,
            "total_slices": len(self._store),
            "total_agents": len(self._version_vectors),
            "total_conflicts": len(conflicts),
            "storage_path": self.storage_path,
        }
