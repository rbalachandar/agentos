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

# Health check types (imported locally to avoid circular dependency)
HealthStatus = Any


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
            if (importance > self.config.l1_promotion_threshold and
                    self.l1.available_tokens >= slice_data.token_count):
                self._promote_to_l1(slice_data.id)
                l1_slice_ids.append(slice_data.id)

        # Trigger paging if L1 is too full
        if self.l1.utilization > self.config.l1_utilization_threshold:
            self._page_out_from_l1()

        # NOTE:
        # `SMMUConfig.l2_utilization_threshold` is currently not used to automatically
        # trigger L2 compaction/demotion to L3 during normal processing.
        # If `process_slices()` is called repeatedly, L2 can keep accumulating slices
        # unless `compact_l2()` is invoked by the caller or a higher-level maintenance loop.

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

    def bootstrap_l3_from_files(self, bootstrap_paths: list[str]) -> int:
        """Bootstrap L3 storage from domain knowledge files.

        Loads pre-computed semantic slices from JSONL files to provide
        initial knowledge and reduce cold start effects.

        Args:
            bootstrap_paths: List of paths to JSONL files containing slices

        Returns:
            Number of slices loaded

        Example JSONL format:
            {"id": "slice_001", "start_pos": 0, "end_pos": 10,
             "tokens": ["Hello", "world"], "token_ids": [123, 456],
             "content": "Hello world", "density_mean": 0.5, "density_std": 0.1,
             "importance_score": 0.7, "metadata": {"source": "bootstrap"}}
        """
        import json
        from pathlib import Path

        loaded_count = 0
        for path_str in bootstrap_paths:
            path = Path(path_str)
            if not path.exists():
                continue

            with open(path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        slice_dict = json.loads(line)
                        # Reconstruct SemanticSlice
                        slice_data = SemanticSlice(
                            id=slice_dict["id"],
                            start_pos=slice_dict["start_pos"],
                            end_pos=slice_dict["end_pos"],
                            tokens=slice_dict["tokens"],
                            token_ids=slice_dict["token_ids"],
                            content=slice_dict["content"],
                            density_mean=slice_dict["density_mean"],
                            density_std=slice_dict["density_std"],
                            importance_score=slice_dict.get("importance_score", 0.5),
                            metadata=slice_dict.get("metadata", {}),
                        )

                        # Add to L3
                        self.l3.add(slice_data)

                        # Register in page table
                        self.page_table.register(
                            slice_id=slice_data.id,
                            tier=MemoryTier.L3,
                            importance_score=slice_data.importance_score,
                        )

                        # Set L3 path in page table
                        l3_path = self.l3.get_slice_path(slice_data.id)
                        if l3_path:
                            self.page_table.set_l3_path(slice_data.id, str(l3_path))

                        loaded_count += 1
                    except (json.JSONDecodeError, KeyError) as e:
                        # Skip invalid entries
                        continue

        return loaded_count

    def save_l3_state(self, save_path: str) -> bool:
        """Save L3 state for session restore.

        Serializes L3 storage and page table state to enable
        restoring a previous session and reducing cold start.

        Args:
            save_path: Path to save the state

        Returns:
            True if successful, False otherwise
        """
        import json
        from pathlib import Path

        try:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save L3 index
            l3_index_path = save_dir / "l3_index.json"
            with open(l3_index_path, "w") as f:
                json.dump(self.l3._index, f, indent=2)

            # Save page table state
            page_table_path = save_dir / "page_table.json"
            with open(page_table_path, "w") as f:
                json.dump(self.page_table.to_dict(), f, indent=2)

            # Save metadata
            metadata_path = save_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump({
                    "l1_utilization": self.l1.utilization,
                    "l2_utilization": self.l2.utilization,
                    "l3_slice_count": self.l3.slice_count,
                }, f, indent=2)

            return True
        except Exception:
            return False

    def restore_l3_state(self, restore_path: str) -> int:
        """Restore L3 state from previous session.

        Loads L3 storage and page table state to continue from
        a previous session without cold start.

        Args:
            restore_path: Path to restore the state from

        Returns:
            Number of slices restored
        """
        import json
        from pathlib import Path

        restore_dir = Path(restore_path)
        if not restore_dir.exists():
            return 0

        restored_count = 0

        # Restore L3 index
        l3_index_path = restore_dir / "l3_index.json"
        if l3_index_path.exists():
            with open(l3_index_path, "r") as f:
                self.l3._index = json.load(f)

        # Restore page table state
        page_table_path = restore_dir / "page_table.json"
        if page_table_path.exists():
            with open(page_table_path, "r") as f:
                page_table_dict = json.load(f)
                self.page_table.from_dict(page_table_dict)

        # Count restored slices
        restored_count = len(self.l3._index)

        return restored_count

    def enable_adaptive_scoring(self, warmup_turns: int = 5) -> None:
        """Enable adaptive importance scoring for cold start mitigation.

        During warmup, recency weight is boosted and new slices receive
        bonus importance to build a useful knowledge base quickly.

        Args:
            warmup_turns: Number of turns for warmup period
        """
        from agentos.memory.scoring import AdaptiveImportanceScorer, AdaptiveScoringConfig

        config = AdaptiveScoringConfig(warmup_turns=warmup_turns)
        self.adaptive_scorer = AdaptiveImportanceScorer(config)

    def disable_adaptive_scoring(self) -> None:
        """Disable adaptive importance scoring."""
        self.adaptive_scorer = None

    def advance_turn(self) -> None:
        """Advance to the next turn (for adaptive scoring)."""
        if hasattr(self, "adaptive_scorer") and self.adaptive_scorer:
            self.adaptive_scorer.advance_turn()

    @property
    def is_warmed_up(self) -> bool:
        """Check if system has completed warmup period."""
        if hasattr(self, "adaptive_scorer") and self.adaptive_scorer:
            return self.adaptive_scorer.is_warmed_up
        return True  # No adaptive scorer = always "warmed up"

    def health_check(self) -> HealthStatus:
        """Check the health of the S-MMU.

        Returns:
            HealthStatus indicating S-MMU health
        """
        from agentos.common.health import HealthState, HealthStatus, healthy, degraded, unhealthy

        stats = self.get_memory_stats()

        # Check each tier for issues
        issues = []

        # L1 health
        l1_util = stats["l1"]["utilization"]
        if l1_util > 0.95:
            issues.append(("L1", f"L1 cache nearly full: {l1_util:.1%}"))
        elif l1_util > 0.9:
            issues.append(("L1", f"L1 cache highly utilized: {l1_util:.1%}"))

        # L2 health
        l2_util = stats["l2"]["utilization"]
        if l2_util > 0.95:
            issues.append(("L2", f"L2 nearly full: {l2_util:.1%}"))

        # L3 health
        l3_count = stats["l3"]["slice_count"]
        if l3_count > self.config.l3_config.max_slices * 0.9:
            issues.append(("L3", f"L3 approaching capacity: {l3_count} slices"))

        # Page table health
        pt_stats = stats["page_table"]
        total_entries = pt_stats["total"]
        if total_entries > 100000:  # Abnormally large
            issues.append(("PageTable", f"Too many entries: {total_entries}"))

        if issues:
            # Determine severity
            has_critical = any(
                tier == "L1" or tier == "PageTable" for tier, _ in issues
            )

            if has_critical:
                return unhealthy(
                    component="smmu",
                    message=f"S-MMU has critical issues: {len(issues)}",
                    details={"issues": issues},
                )
            else:
                return degraded(
                    component="smmu",
                    message=f"S-MMU is degraded: {len(issues)} issues",
                    details={"issues": issues},
                )

        return healthy(
            component="smmu",
            message="S-MMU is healthy",
            details=stats,
        )
