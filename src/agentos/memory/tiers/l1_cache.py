"""
L1 Cache: Active Attention Window.

Based on AgentOS paper Section 3.2.1:

L1 is the fastest memory tier, directly accessible by the Reasoning Kernel.
It corresponds to the active KV-cache of the transformer model.

Key properties:
- Limited capacity (typically 4K-8K tokens)
- Fast access (sub-millisecond)
- Holds "semantic anchors" that should remain pinned
- Managed by S-MMU paging algorithm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from agentos.memory.slicing.types import SemanticSlice
from agentos.memory.types import MemoryStats, MemoryTier


@dataclass
class L1CacheConfig:
    """Configuration for L1 Cache."""

    # Capacity in tokens
    max_tokens: int = 4096

    # Maximum number of slices to store
    max_slices: int = 100

    # Percentage of capacity reserved for pinned slices
    pinned_reserve_percent: float = 0.2

    # Whether to use FIFO or semantic-aware eviction
    use_semantic_eviction: bool = True

    def validate(self) -> None:
        """Validate configuration."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.max_slices <= 0:
            raise ValueError("max_slices must be positive")
        if not (0.0 <= self.pinned_reserve_percent <= 1.0):
            raise ValueError("pinned_reserve_percent must be in [0, 1]")


@dataclass
class L1Entry:
    """An entry in the L1 cache.

    Combines semantic slice data with KV-cache state.
    """

    slice_id: str
    slice_data: SemanticSlice

    # KV-cache state (hidden states, attention keys/values)
    # In a real implementation, this would be the actual KV-cache
    kv_cache: dict[str, NDArray[np.float32]] | None = None

    # Position in the cache (for ordering)
    position: int = 0

    # Whether this slice is pinned (should not be evicted)
    is_pinned: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


class L1Cache:
    """L1 Cache - Active Attention Window.

    Manages the fast memory tier directly accessible by the Reasoning Kernel.
    """

    def __init__(self, config: L1CacheConfig | None = None) -> None:
        """Initialize the L1 cache.

        Args:
            config: Configuration for L1 cache. If None, uses defaults.
        """
        self.config = config or L1CacheConfig()
        self.config.validate()

        # Map: slice_id -> L1Entry
        self._entries: dict[str, L1Entry] = {}

        # Order of entries (for FIFO/eviction)
        self._order: list[str] = []

        # Position counter
        self._next_position = 0

    def add(self, slice_data: SemanticSlice, pin: bool = False) -> L1Entry:
        """Add a slice to L1 cache.

        Args:
            slice_data: Semantic slice to add
            pin: Whether to pin this slice (prevent eviction)

        Returns:
            The created L1Entry

        Raises:
            ValueError: If cache is full and cannot make space
        """
        slice_id = slice_data.id

        # Update if already exists
        if slice_id in self._entries:
            entry = self._entries[slice_id]
            entry.slice_data = slice_data
            if pin:
                entry.is_pinned = True
            return entry

        # Check capacity
        if not self._can_add(slice_data.token_count, pin):
            raise ValueError(
                f"Cannot add slice ({slice_data.token_count} tokens): "
                f"L1 cache full ({self.used_tokens}/{self.config.max_tokens} tokens used)"
            )

        # Create entry
        entry = L1Entry(
            slice_id=slice_id,
            slice_data=slice_data,
            position=self._next_position,
            is_pinned=pin,
        )
        self._next_position += 1

        self._entries[slice_id] = entry
        self._order.append(slice_id)

        return entry

    def get(self, slice_id: str) -> L1Entry | None:
        """Get a slice from L1 cache.

        Args:
            slice_id: Semantic hash of the slice

        Returns:
            L1Entry if found, None otherwise
        """
        return self._entries.get(slice_id)

    def remove(self, slice_id: str) -> L1Entry | None:
        """Remove a slice from L1 cache.

        Args:
            slice_id: Semantic hash of the slice

        Returns:
            Removed L1Entry if found, None otherwise
        """
        if slice_id not in self._entries:
            return None

        entry = self._entries.pop(slice_id)
        self._order.remove(slice_id)

        return entry

    def evict_lowest_importance(
        self, importance_scores: dict[str, float], count: int = 1
    ) -> list[L1Entry]:
        """Evict lowest-importance slices from cache.

        Args:
            importance_scores: Map of slice_id -> importance score
            count: Maximum number of slices to evict

        Returns:
            List of evicted L1Entry
        """
        # Filter out pinned slices
        candidates = [
            (sid, entry)
            for sid, entry in self._entries.items()
            if not entry.is_pinned
        ]

        # Sort by importance (lowest first)
        candidates = sorted(
            candidates, key=lambda x: importance_scores.get(x[0], 0.0)
        )

        # Evict the lowest importance
        evicted = []
        for slice_id, _ in candidates[:count]:
            entry = self.remove(slice_id)
            if entry:
                evicted.append(entry)

        return evicted

    def evict_oldest(self, count: int = 1) -> list[L1Entry]:
        """Evict oldest slices from cache (FIFO).

        Args:
            count: Maximum number of slices to evict

        Returns:
            List of evicted L1Entry
        """
        evicted = []

        for slice_id in self._order[:count]:
            entry = self._entries.get(slice_id)
            if entry and not entry.is_pinned:
                evicted.append(self.remove(slice_id))

        return evicted

    def _can_add(self, token_count: int, is_pinned: bool) -> bool:
        """Check if we can add a slice with given token count."""
        # Calculate available capacity
        pinned_reserve = int(self.config.max_tokens * self.config.pinned_reserve_percent)

        if is_pinned:
            # Pinned slices can use the reserved space
            available = self.config.max_tokens - self.used_tokens_pinned
        else:
            # Non-pinned slices can only use non-reserved space
            available = (
                self.config.max_tokens
                - pinned_reserve
                - (self.used_tokens - self.used_tokens_pinned)
            )

        return token_count <= available

    def get_slices(self) -> list[L1Entry]:
        """Get all slices in the cache."""
        return list(self._entries.values())

    def get_pinned_slices(self) -> list[L1Entry]:
        """Get all pinned slices in the cache."""
        return [e for e in self._entries.values() if e.is_pinned]

    def get_token_count(self, slice_id: str) -> int:
        """Get token count for a slice."""
        entry = self._entries.get(slice_id)
        return entry.slice_data.token_count if entry else 0

    @property
    def used_tokens(self) -> int:
        """Total tokens used in the cache."""
        return sum(e.slice_data.token_count for e in self._entries.values())

    @property
    def used_tokens_pinned(self) -> int:
        """Tokens used by pinned slices."""
        return sum(
            e.slice_data.token_count
            for e in self._entries.values()
            if e.is_pinned
        )

    @property
    def slice_count(self) -> int:
        """Number of slices in the cache."""
        return len(self._entries)

    @property
    def utilization(self) -> float:
        """Cache utilization ratio [0, 1]."""
        if self.config.max_tokens == 0:
            return 0.0
        return self.used_tokens / self.config.max_tokens

    @property
    def available_tokens(self) -> int:
        """Remaining token capacity (accounting for pinned reserve)."""
        pinned_reserve = int(self.config.max_tokens * self.config.pinned_reserve_percent)
        non_pinned_used = self.used_tokens - self.used_tokens_pinned
        available = self.config.max_tokens - pinned_reserve - non_pinned_used
        return max(0, available)

    def get_stats(self) -> MemoryStats:
        """Get statistics about the L1 cache."""
        return MemoryStats(
            tier=MemoryTier.L1,
            capacity_tokens=self.config.max_tokens,
            used_tokens=self.used_tokens,
            total_slices=self.slice_count,
        )

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._entries.clear()
        self._order.clear()
        self._next_position = 0
