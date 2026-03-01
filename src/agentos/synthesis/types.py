"""
Types for Semantic Synthesis.
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


class SynthesisStrategy(str, Enum):
    """Strategy for synthesizing agent contributions."""

    SEMANTIC_MERGE = "semantic_merge"  # Merge by semantic similarity
    WEIGHTED_AGGREGATION = "weighted_aggregation"  # Weight by confidence
    CONFLICT_AWARE = "conflict_aware"  # Resolve contradictions explicitly
    HIERARCHICAL = "hierarchical"  # Build hierarchical structure


@dataclass
class AgentContribution:
    """An agent's contribution with semantic metadata."""

    agent_id: str
    agent_name: str
    agent_role: str

    # The contribution text
    content: str

    # Semantic metadata
    semantic_gradient: NDArray[np.float32] | None = None  # Agent's semantic state
    confidence: float = 0.5  # Agent's confidence (0-1)
    attention_focus: float = 0.5  # Agent's attention focus (0-1)

    # Semantic slices from this agent
    slices: list[SemanticSlice] = field(default_factory=list)

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def token_count(self) -> int:
        """Approximate token count of contribution."""
        return len(self.content.split())

    @property
    def has_semantic_info(self) -> bool:
        """Whether this contribution has semantic metadata."""
        return (
            self.semantic_gradient is not None
            or self.attention_focus > 0
            or len(self.slices) > 0
        )


@dataclass
class ConceptCluster:
    """A cluster of semantically related concepts from multiple agents."""

    cluster_id: str
    concepts: list[AgentContribution] = field(default_factory=list)

    # Cluster semantics
    centroid_embedding: NDArray[np.float32] | None = None
    coherence_score: float = 0.0  # How coherent is this cluster (0-1)

    # Metadata
    key_themes: list[str] = field(default_factory=list)
    consensus_view: str | None = None  # What agents agree on
    diverging_views: list[str] = field(default_factory=list)  # Where they disagree


@dataclass
class ConflictResolution:
    """A resolved conflict between agent contributions."""

    conflict_type: str  # "contradiction", "incompleteness", "ambiguity"
    agents_involved: list[str] = field(default_factory=list)

    # Conflicting views
    conflicting_statements: list[tuple[str, str]] = field(default_factory=list)  # (agent, statement)

    # Resolution
    resolution_strategy: str = ""  # "merge", "select_highest_confidence", "synthesize_new"
    resolved_content: str = ""
    confidence: float = 0.0


@dataclass
class SynthesisResult:
    """Result of semantic synthesis."""

    # Final synthesized answer
    final_synthesis: str

    # How synthesis was constructed
    strategy_used: SynthesisStrategy = SynthesisStrategy.SEMANTIC_MERGE

    # Clusters of related concepts
    concept_clusters: list[ConceptCluster] = field(default_factory=list)

    # Conflicts resolved
    conflicts_resolved: list[ConflictResolution] = field(default_factory=list)

    # Agent participation
    agents_participated: list[str] = field(default_factory=list)

    # Quality metrics
    coherence_score: float = 0.0  # Overall coherence (0-1)
    coverage_score: float = 0.0  # How well all perspectives covered (0-1)

    # Metadata
    synthesis_duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisConfig:
    """Configuration for semantic synthesis."""

    # Strategy
    strategy: SynthesisStrategy = SynthesisStrategy.SEMANTIC_MERGE

    # Semantic similarity threshold
    similarity_threshold: float = 0.7  # Cosine similarity for clustering

    # Confidence weighting
    use_confidence_weighting: bool = True
    confidence_weight_factor: float = 2.0  # How much to weight by confidence

    # Conflict resolution
    detect_conflicts: bool = True
    conflict_resolution_strategy: str = "merge"  # "merge", "highest_confidence", "synthesize"

    # LLM synthesis settings
    use_llm_synthesis: bool = True  # Use LLM for final synthesis
    llm_synthesis_max_tokens: int = 1024  # Increased for complete synthesis

    # Clustering
    max_clusters: int = 10
    min_cluster_size: int = 1

    def validate(self) -> None:
        """Validate configuration."""
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in [0, 1]")
        if not (0.0 <= self.confidence_weight_factor <= 10.0):
            raise ValueError("confidence_weight_factor must be in [0, 10]")
