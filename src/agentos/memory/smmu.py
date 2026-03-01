"""
Semantic Memory Management Unit (S-MMU).

Based on AgentOS paper Algorithm 2 (Section 3.2.2):

The S-MMU manages the Cognitive Memory Hierarchy (CMH), handling paging
between L1 ↔ L2 ↔ L3 based on:
- Current task relevance
- Importance score ℐ
- Capacity constraints

Key responsibilities:
- Promote slices from L2/L3 to L1 when needed
- Demote slices from L1 to L2/L3 when full
- Evict low-importance slices when necessary
- Track memory utilization across tiers
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from agentos.memory.importance import ImportanceCalculator
from agentos.memory.page_table import PageTable
from agentos.memory.slicing.types import SemanticSlice, SlicingResult
from agentos.memory.tiers.l1_cache import L1Cache, L1CacheConfig
from agentos.memory.tiers.l2_ram import L2RAM, L2Config, compute_slice_embedding
from agentos.memory.tiers.l3_storage import L3Storage, L3Config
from agentos.memory.types import MemoryTier, PageTableEntry, PagingResult


@dataclass
class SMMUConfig:
    """Configuration for the S-MMU."""

    # Tier configurations
    l1_config: L1CacheConfig = field(default_factory=L1CacheConfig)
    l2_config: L2Config = field(default_factory=L2Config)
    l3_config: L3Config = field(default_factory=L3Config)

    # Paging thresholds
    l1_utilization_threshold: float = 0.8  # Trigger paging when L1 > 80% full
    l2_utilization_threshold: float = 0.9  # Trigger paging when L2 > 90% full

    # L1 promotion threshold
    l1_promotion_threshold: float = 0.2  # Importance > 0.2 to promote to L1

    # Importance calculator config (for internal use)
    importance_w_attention: float = 0.4
    importance_w_recency: float = 0.2
    importance_w_frequency: float = 0.2
    importance_w_user: float = 0.2

    def validate(self) -> None:
        """Validate configuration."""
        self.l1_config.validate()
        self.l2_config.validate()
        self.l3_config.validate()

        if not (0.0 <= self.l1_utilization_threshold <= 1.0):
            raise ValueError("l1_utilization_threshold must be in [0, 1]")
        if not (0.0 <= self.l2_utilization_threshold <= 1.0):
            raise ValueError("l2_utilization_threshold must be in [0, 1]")


class SMMU:
    """Semantic Memory Management Unit.

    Manages paging of semantic slices between memory tiers based on
    importance and relevance to the current task.
    """

    def __init__(self, config: SMMUConfig | None = None) -> None:
        """Initialize the S-MMU.

        Args:
            config: Configuration for S-MMU. If None, uses defaults.
        """
        self.config = config or SMMUConfig()
        self.config.validate()

        # Initialize tiers
        self.l1 = L1Cache(self.config.l1_config)
        self.l2 = L2RAM(self.config.l2_config)
        self.l3 = L3Storage(self.config.l3_config)

        # Initialize page table and importance calculator
        self.page_table = PageTable()
        from agentos.memory.importance import ImportanceConfig
        importance_config = ImportanceConfig(
            w_attention=self.config.importance_w_attention,
            w_recency=self.config.importance_w_recency,
            w_frequency=self.config.importance_w_frequency,
            w_user=self.config.importance_w_user,
        )
        self.importance = ImportanceCalculator(importance_config)

    def process_slices(
        self,
        slicing_result: SlicingResult,
        hidden_states: NDArray[np.float32],
    ) -> list[str]:
        """Process a new batch of semantic slices.

        Adds slices to the memory hierarchy, paging as needed.

        Args:
            slicing_result: Result from semantic slicing
            hidden_states: (seq_len, hidden_dim) hidden states

        Returns:
            List of slice IDs that were added to L1
        """
        l1_slice_ids = []

        for slice_data in slicing_result.slices:
            # Compute embedding
            embedding = compute_slice_embedding(
                slice_data, hidden_states, method="mean"
            )

            # Compute initial importance
            importance = self.importance.compute(
                slicing_result.density_profile.densities[
                    slice_data.start_pos : slice_data.end_pos
                ]
            )

            # Register in page table
            entry = self.page_table.register(
                slice_id=slice_data.id,
                tier=MemoryTier.L2,  # Start in L2
                importance_score=importance,
            )

            # Add to L2
            self.l2.add(slice_data, embedding)
            self.page_table.set_l2_collection(slice_data.id, self.config.l2_config.collection_name)

            # High-importance slices go to L1
            if importance > self.config.l1_promotion_threshold and self.l1.available_tokens >= slice_data.token_count:
                self._promote_to_l1(slice_data.id)
                l1_slice_ids.append(slice_data.id)

        # Trigger paging if L1 is too full
        if self.l1.utilization > self.config.l1_utilization_threshold:
            self._page_out_from_l1()

        return l1_slice_ids

    def get_slice(self, slice_id: str) -> SemanticSlice | None:
        """Get a slice from memory.

        Searches L1 first, then L2, then L3.
        Promotes the slice to L1 if found in lower tier.

        Args:
            slice_id: Semantic hash of the slice

        Returns:
            SemanticSlice if found, None otherwise
        """
        # Record access in page table
        self.page_table.record_access(slice_id)

        # Check L1
        l1_entry = self.l1.get(slice_id)
        if l1_entry:
            return l1_entry.slice_data

        # Check L2
        l2_entry = self.l2.get(slice_id)
        if l2_entry:
            # Promote to L1 if space available
            if self.l1.available_tokens >= l2_entry.slice_data.token_count:
                self._promote_to_l1(slice_id)
            return l2_entry.slice_data

        # Check L3
        l3_entry = self.l3.get(slice_id)
        if l3_entry:
            # Promote to L2, then L1 if space available
            self._promote_from_l3_to_l2(l3_entry)
            if self.l1.available_tokens >= l3_entry.slice_data.token_count:
                self._promote_to_l1(slice_id)
            return l3_entry.slice_data

        return None

    def semantic_search(self, query_embedding: NDArray[np.float32], top_k: int = 10) -> list:
        """Search for similar slices in L2.

        Args:
            query_embedding: Query vector
            top_k: Number of results

        Returns:
            List of RetrievalResult
        """
        return self.l2.semantic_search(query_embedding, top_k)

    def _promote_to_l1(self, slice_id: str) -> bool:
        """Promote a slice from L2/L3 to L1.

        Args:
            slice_id: Semantic hash of the slice

        Returns:
            True if promoted, False otherwise
        """
        # Get slice from L2
        l2_entry = self.l2.get(slice_id)
        if not l2_entry:
            return False

        # Make space in L1 if needed
        required_tokens = l2_entry.slice_data.token_count
        while self.l1.available_tokens < required_tokens:
            if not self._page_out_from_l1(count=1):
                break

        # Add to L1
        try:
            self.l1.add(l2_entry.slice_data)
            self.page_table.set_l1_position(slice_id, self.l1.slice_count - 1)
            return True
        except ValueError:
            return False

    def _promote_from_l3_to_l2(self, l3_entry) -> bool:
        """Promote a slice from L3 to L2.

        Args:
            l3_entry: L3 entry to promote

        Returns:
            True if promoted, False otherwise
        """
        # Compute embedding from slice data
        # Use a deterministic hash-based method since we don't have original hidden_states
        # This preserves some semantic properties better than random
        token_ids = l3_entry.slice_data.token_ids
        if token_ids:
            # Create a simple deterministic embedding based on token IDs
            # Hash token IDs to create a pseudo-embedding
            embedding_dim = self.config.l2_config.embedding_dim
            embedding = np.zeros(embedding_dim, dtype=np.float32)
            for i, tid in enumerate(token_ids[:embedding_dim]):
                # Use token ID to set embedding dimension (deterministic)
                idx = tid % embedding_dim
                embedding[idx] += 1.0 / len(token_ids)
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        else:
            # Fallback to zero embedding
            embedding = np.zeros(self.config.l2_config.embedding_dim, dtype=np.float32)

        self.l2.add(l3_entry.slice_data, embedding)
        self.page_table.set_l2_collection(
            l3_entry.slice_id, self.config.l2_config.collection_name
        )
        return True

    def _page_out_from_l1(self, count: int = 1) -> bool:
        """Page out low-importance slices from L1 to L2.

        Args:
            count: Number of slices to page out

        Returns:
            True if any slices were paged out
        """
        # Get importance scores
        l1_entries = self.l1.get_slices()
        importance_scores = {}
        for entry in l1_entries:
            pt_entry = self.page_table.get(entry.slice_id)
            if pt_entry:
                importance_scores[entry.slice_id] = pt_entry.importance_score

        # Evict lowest importance
        evicted = self.l1.evict_lowest_importance(importance_scores, count)

        for entry in evicted:
            # Move to L2 (already there, just update page table)
            self.page_table.update_tier(entry.slice_id, MemoryTier.L2)

        return len(evicted) > 0

    def _demote_to_l3(self, slice_id: str) -> bool:
        """Demote a slice from L2 to L3.

        Args:
            slice_id: Semantic hash of the slice

        Returns:
            True if demoted, False otherwise
        """
        l2_entry = self.l2.get(slice_id)
        if not l2_entry:
            return False

        # Add to L3
        self.l3.add(l2_entry.slice_data)
        l3_path = self.l3.get_slice_path(slice_id)
        if l3_path:
            self.page_table.set_l3_path(slice_id, str(l3_path))

        # Remove from L2
        self.l2.remove(slice_id)

        return True

    def compact_l2(self) -> PagingResult:
        """Compact L2 by paging low-importance slices to L3.

        Returns:
            PagingResult with statistics
        """
        # Get all L2 entries sorted by importance
        l2_entries = self.l2.get_all_entries()

        l2_importance_map = {}
        for entry in l2_entries:
            pt_entry = self.page_table.get(entry.slice_id)
            if pt_entry:
                l2_importance_map[entry.slice_id] = pt_entry.importance_score

        # Sort by importance (lowest first)
        sorted_entries = sorted(
            l2_entries, key=lambda e: l2_importance_map.get(e.slice_id, 0.0)
        )

        # Demote low-importance slices (bottom 20%)
        num_to_demote = max(1, len(sorted_entries) // 5)
        pages_demoted = []

        for entry in sorted_entries[:num_to_demote]:
            if self._demote_to_l3(entry.slice_id):
                pages_demoted.append(entry.slice_id)

        # Trigger L1 paging if needed - compute importance for L1 entries
        pages_evicted = []
        if self.l1.utilization > self.config.l1_utilization_threshold:
            l1_entries = self.l1.get_slices()
            l1_importance_map = {}
            for entry in l1_entries:
                pt_entry = self.page_table.get(entry.slice_id)
                if pt_entry:
                    l1_importance_map[entry.slice_id] = pt_entry.importance_score
            evicted = self.l1.evict_lowest_importance(l1_importance_map, count=5)
            pages_evicted = [e.slice_id for e in evicted]

        return PagingResult(
            pages_promoted=[],
            pages_demoted=pages_demoted,
            pages_evicted=pages_evicted,
            l1_utilization_before=0.0,
            l1_utilization_after=self.l1.utilization,
        )

    def get_memory_stats(self) -> dict[str, Any]:
        """Get statistics for all memory tiers.

        Returns:
            Dictionary with stats for L1, L2, L3
        """
        return {
            "l1": {
                "utilization": self.l1.utilization,
                "used_tokens": self.l1.used_tokens,
                "max_tokens": self.config.l1_config.max_tokens,
                "slice_count": self.l1.slice_count,
            },
            "l2": {
                "utilization": self.l2.utilization,
                "used_tokens": self.l2.used_tokens,
                "max_tokens": self.config.l2_config.max_tokens,
                "slice_count": self.l2.slice_count,
            },
            "l3": {
                "slice_count": self.l3.slice_count,
                "total_size_bytes": self.l3.total_size_bytes,
            },
            "page_table": self.page_table.get_stats(),
        }
