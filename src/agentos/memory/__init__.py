"""
AgentOS Memory Hierarchy.

Exports components for the Cognitive Memory Hierarchy (CMH):
- Semantic Page Table (SPT)
- S-MMU (Semantic Memory Management Unit)
- L1 Cache, L2 RAM, L3 Storage
- Importance Calculator
- Adaptive Importance Scoring (Cold Start mitigation)
"""

from agentos.memory.importance import (
    ImportanceCalculator,
    ImportanceConfig,
    compute_importance,
)
from agentos.memory.page_table import PageTable, PageTableConfig
from agentos.memory.scoring import (
    AdaptiveImportanceScorer,
    AdaptiveScoringConfig,
    create_adaptive_scorer,
)
from agentos.memory.smmu import SMMU, SMMUConfig
from agentos.memory.tiers.l1_cache import L1Cache, L1CacheConfig, L1Entry
from agentos.memory.tiers.l2_ram import L2RAM, L2Config, L2Entry
from agentos.memory.tiers.l3_storage import L3Config, L3Entry, L3Storage
from agentos.memory.types import (
    MemoryStats,
    MemoryTier,
    PageTableEntry,
    PagingResult,
    RetrievalResult,
)

__all__ = [
    # Types
    "MemoryTier",
    "PageTableEntry",
    "RetrievalResult",
    "PagingResult",
    "MemoryStats",
    "L1CacheConfig",
    "L2Config",
    "L3Config",
    # Importance
    "ImportanceCalculator",
    "ImportanceConfig",
    "compute_importance",
    # Adaptive Scoring (Cold Start)
    "AdaptiveImportanceScorer",
    "AdaptiveScoringConfig",
    "create_adaptive_scorer",
    # Page Table
    "PageTable",
    "PageTableConfig",
    # Tiers
    "L1Cache",
    "L1Entry",
    "L2RAM",
    "L2Entry",
    "L3Storage",
    "L3Entry",
    # S-MMU
    "SMMU",
    "SMMUConfig",
]
