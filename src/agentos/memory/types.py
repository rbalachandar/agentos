"""
Core types for the Cognitive Memory Hierarchy (CMH).

Based on AgentOS paper Section 3.2:
- L1: Active Attention Window (fast, limited)
- L2: Deep Context (vector DB, larger)
- L3: Knowledge Base (cold storage, unlimited)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class MemoryTier(str, Enum):
    """Memory tier in the Cognitive Memory Hierarchy."""

    L1 = "L1"  # Active Attention Window (KV-cache)
    L2 = "L2"  # Deep Context (Vector DB)
    L3 = "L3"  # Knowledge Base (Cold storage)


@dataclass
class PageTableEntry:
    """Entry in the Semantic Page Table.

    Tracks the location and metadata of a semantic slice.
    """

    slice_id: str  # Semantic hash
    tier: MemoryTier  # Current tier
    importance_score: float  # ℐ (0-1)

    # Access tracking
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0

    # Location info
    l1_position: int | None = None  # Position in L1 cache (if in L1)
    l2_collection: str | None = None  # Collection name in L2 (if in L2)
    l3_path: str | None = None  # Storage path in L3 (if in L3)

    # Metadata
    metadata: dict = field(default_factory=dict)

    @property
    def is_pinned(self) -> bool:
        """Whether this slice is pinned (should not be evicted)."""
        return self.metadata.get("pinned", False)

    @property
    def is_dirty(self) -> bool:
        """Whether this slice has been modified since last sync."""
        return self.metadata.get("dirty", False)


@dataclass
class RetrievalResult:
    """Result of a semantic retrieval operation."""

    slice_id: str
    content: str
    tokens: list[str]
    embedding: NDArray[np.float32] | None = None
    score: float = 0.0  # Similarity score for semantic search
    tier: MemoryTier = MemoryTier.L2
    metadata: dict = field(default_factory=dict)


@dataclass
class PagingResult:
    """Result of a paging operation."""

    pages_promoted: list[str]  # Slice IDs moved from lower to higher tier
    pages_demoted: list[str]  # Slice IDs moved from higher to lower tier
    pages_evicted: list[str]  # Slice IDs removed from cache

    # Statistics
    l1_utilization_before: float
    l1_utilization_after: float
    total_time_ms: float = 0.0

    @property
    def total_changes(self) -> int:
        return len(self.pages_promoted) + len(self.pages_demoted) + len(self.pages_evicted)


@dataclass
class MemoryStats:
    """Statistics for memory tier utilization."""

    tier: MemoryTier
    capacity_tokens: int  # Max tokens this tier can hold
    used_tokens: int  # Current tokens in this tier
    total_slices: int  # Number of slices stored

    # Performance metrics
    avg_retrieval_time_ms: float = 0.0
    cache_hit_rate: float = 0.0

    @property
    def utilization(self) -> float:
        """Utilization ratio (0-1)."""
        if self.capacity_tokens == 0:
            return 0.0
        return self.used_tokens / self.capacity_tokens

    @property
    def available_tokens(self) -> int:
        """Remaining token capacity."""
        return max(0, self.capacity_tokens - self.used_tokens)
