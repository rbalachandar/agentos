"""
Semantic Page Table (SPT).

Based on AgentOS paper Section 3.2.2:

The SPT tracks the location of all semantic slices across the memory hierarchy.
It maps semantic hashes (slice IDs) to their current tier and metadata.

Similar to a CPU page table, but for semantic slices instead of memory pages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from agentos.memory.types import MemoryTier, PageTableEntry


@dataclass
class PageTableConfig:
    """Configuration for the Semantic Page Table."""

    # Maximum entries in each tier
    max_l1_entries: int = 100
    max_l2_entries: int = 10000
    max_l3_entries: int = 1000000  # Essentially unlimited

    def validate(self) -> None:
        """Validate configuration."""
        if self.max_l1_entries <= 0:
            raise ValueError("max_l1_entries must be positive")
        if self.max_l2_entries < self.max_l1_entries:
            raise ValueError("max_l2_entries must be >= max_l1_entries")
        if self.max_l3_entries < self.max_l2_entries:
            raise ValueError("max_l3_entries must be >= max_l2_entries")


class PageTable:
    """Semantic Page Table - tracks slices across memory tiers.

    The page table maps slice IDs (semantic hashes) to their current location
    and metadata (importance, access stats, etc.).

    This is the central index that the S-MMU uses to manage memory.
    """

    def __init__(self, config: PageTableConfig | None = None) -> None:
        """Initialize the page table.

        Args:
            config: Configuration for the page table. If None, uses defaults.
        """
        self.config = config or PageTableConfig()
        self.config.validate()

        # Map: slice_id -> PageTableEntry
        self._entries: dict[str, PageTableEntry] = {}

        # Tier-specific indices for fast lookup
        self._l1_index: dict[str, str] = {}  # position -> slice_id
        self._l2_index: dict[str, str] = {}  # collection -> list of slice_ids
        self._l3_index: dict[str, str] = {}  # path -> slice_id

    def register(
        self,
        slice_id: str,
        tier: MemoryTier,
        importance_score: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> PageTableEntry:
        """Register a new slice in the page table.

        Args:
            slice_id: Semantic hash of the slice
            tier: Initial tier for this slice
            importance_score: Initial importance score ℐ
            metadata: Additional metadata

        Returns:
            The created PageTableEntry
        """
        now = datetime.now()

        entry = PageTableEntry(
            slice_id=slice_id,
            tier=tier,
            importance_score=importance_score,
            created_at=now,
            last_accessed=now,
            access_count=0,
            metadata=metadata or {},
        )

        self._entries[slice_id] = entry
        return entry

    def get(self, slice_id: str) -> PageTableEntry | None:
        """Get a page table entry by slice ID.

        Args:
            slice_id: Semantic hash of the slice

        Returns:
            PageTableEntry if found, None otherwise
        """
        return self._entries.get(slice_id)

    def update_tier(self, slice_id: str, new_tier: MemoryTier) -> bool:
        """Update the tier of a slice.

        Called when a slice is paged between tiers.

        Args:
            slice_id: Semantic hash of the slice
            new_tier: New tier for the slice

        Returns:
            True if updated, False if slice not found
        """
        entry = self._entries.get(slice_id)
        if entry is None:
            return False

        entry.tier = new_tier
        return True

    def record_access(self, slice_id: str) -> bool:
        """Record an access to a slice.

        Updates last_accessed and increments access_count.

        Args:
            slice_id: Semantic hash of the slice

        Returns:
            True if updated, False if slice not found
        """
        entry = self._entries.get(slice_id)
        if entry is None:
            return False

        entry.last_accessed = datetime.now()
        entry.access_count += 1
        return True

    def update_importance(self, slice_id: str, importance: float) -> bool:
        """Update the importance score of a slice.

        Args:
            slice_id: Semantic hash of the slice
            importance: New importance score ℐ in [0, 1]

        Returns:
            True if updated, False if slice not found
        """
        entry = self._entries.get(slice_id)
        if entry is None:
            return False

        entry.importance_score = float(np.clip(importance, 0.0, 1.0))
        return True

    def set_l1_position(self, slice_id: str, position: int) -> bool:
        """Set the L1 position of a slice.

        Args:
            slice_id: Semantic hash of the slice
            position: Position in L1 cache

        Returns:
            True if updated, False if slice not found
        """
        entry = self._entries.get(slice_id)
        if entry is None:
            return False

        entry.tier = MemoryTier.L1
        entry.l1_position = position
        return True

    def set_l2_collection(self, slice_id: str, collection: str) -> bool:
        """Set the L2 collection of a slice.

        Args:
            slice_id: Semantic hash of the slice
            collection: Collection name in L2

        Returns:
            True if updated, False if slice not found
        """
        entry = self._entries.get(slice_id)
        if entry is None:
            return False

        entry.tier = MemoryTier.L2
        entry.l2_collection = collection
        return True

    def set_l3_path(self, slice_id: str, path: str) -> bool:
        """Set the L3 storage path of a slice.

        Args:
            slice_id: Semantic hash of the slice
            path: Storage path in L3

        Returns:
            True if updated, False if slice not found
        """
        entry = self._entries.get(slice_id)
        if entry is None:
            return False

        entry.tier = MemoryTier.L3
        entry.l3_path = path
        return True

    def get_slices_by_tier(self, tier: MemoryTier) -> list[PageTableEntry]:
        """Get all slices in a given tier.

        Args:
            tier: Memory tier to query

        Returns:
            List of PageTableEntry for slices in this tier
        """
        return [e for e in self._entries.values() if e.tier == tier]

    def get_l1_slices_sorted_by_importance(
        self, reverse: bool = True
    ) -> list[PageTableEntry]:
        """Get L1 slices sorted by importance.

        Args:
            reverse: If True, highest importance first (for eviction)

        Returns:
            List of PageTableEntry sorted by importance_score
        """
        l1_entries = self.get_slices_by_tier(MemoryTier.L1)
        return sorted(l1_entries, key=lambda e: e.importance_score, reverse=reverse)

    def get_eviction_candidates(self, count: int) -> list[PageTableEntry]:
        """Get candidates for eviction from L1.

        Returns the lowest-importance, non-pinned slices in L1.

        Args:
            count: Maximum number of candidates to return

        Returns:
            List of PageTableEntry sorted by importance (lowest first)
        """
        l1_entries = self.get_slices_by_tier(MemoryTier.L1)

        # Filter out pinned slices
        candidates = [e for e in l1_entries if not e.is_pinned]

        # Sort by importance (lowest first)
        candidates = sorted(candidates, key=lambda e: e.importance_score)

        return candidates[:count]

    def remove(self, slice_id: str) -> bool:
        """Remove a slice from the page table.

        Args:
            slice_id: Semantic hash of the slice

        Returns:
            True if removed, False if not found
        """
        if slice_id in self._entries:
            del self._entries[slice_id]
            return True
        return False

    @property
    def total_entries(self) -> int:
        """Total number of entries in the page table."""
        return len(self._entries)

    @property
    def l1_count(self) -> int:
        """Number of slices in L1."""
        return len(self.get_slices_by_tier(MemoryTier.L1))

    @property
    def l2_count(self) -> int:
        """Number of slices in L2."""
        return len(self.get_slices_by_tier(MemoryTier.L2))

    @property
    def l3_count(self) -> int:
        """Number of slices in L3."""
        return len(self.get_slices_by_tier(MemoryTier.L3))

    def get_stats(self) -> dict[str, int]:
        """Get statistics about the page table.

        Returns:
            Dictionary with counts per tier
        """
        return {
            "total": self.total_entries,
            "l1": self.l1_count,
            "l2": self.l2_count,
            "l3": self.l3_count,
        }
