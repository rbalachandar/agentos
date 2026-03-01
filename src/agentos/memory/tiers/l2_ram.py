"""
L2 RAM: Deep Context.

Based on AgentOS paper Section 3.2.1:

L2 is the secondary memory tier using a vector database for semantic retrieval.
It stores compressed representations of semantic slices with their embeddings.

Key properties:
- Larger capacity than L1 (typically 100K+ tokens)
- Semantic search via vector similarity
- Fast retrieval (milliseconds)
- Managed by S-MMU paging algorithm

Uses ChromaDB for local vector storage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from agentos.memory.slicing.types import SemanticSlice
from agentos.memory.types import MemoryStats, MemoryTier, RetrievalResult


@dataclass
class L2Config:
    """Configuration for L2 RAM."""

    # Capacity in tokens
    max_tokens: int = 100000

    # Maximum number of slices to store
    max_slices: int = 10000

    # ChromaDB collection name
    collection_name: str = "agentos_l2"

    # Embedding dimension (for Qwen2.5-0.5B, hidden_dim=896)
    embedding_dim: int = 896

    # Path for ChromaDB persistence
    persist_directory: str = "./data/chroma_db"

    # Number of results to return from semantic search
    top_k: int = 10

    def validate(self) -> None:
        """Validate configuration."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.max_slices <= 0:
            raise ValueError("max_slices must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")


@dataclass
class L2Entry:
    """An entry in L2 storage.

    Represents a compressed semantic slice with embedding.
    """

    slice_id: str
    slice_data: SemanticSlice

    # Embedding vector for semantic search
    embedding: NDArray[np.float32]

    # Storage metadata
    created_at: str
    compressed: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


class L2RAM:
    """L2 RAM - Deep Context with vector database.

    Stores semantic slices with embeddings for fast semantic retrieval.
    """

    def __init__(self, config: L2Config | None = None) -> None:
        """Initialize L2 RAM.

        Args:
            config: Configuration for L2. If None, uses defaults.
        """
        self.config = config or L2Config()
        self.config.validate()

        # In-memory storage (in production, use ChromaDB)
        self._entries: dict[str, L2Entry] = {}

        # Simple in-memory vector index
        self._embeddings: dict[str, NDArray[np.float32]] = {}

    def add(self, slice_data: SemanticSlice, embedding: NDArray[np.float32]) -> L2Entry:
        """Add a slice to L2 storage.

        Args:
            slice_data: Semantic slice to store
            embedding: Embedding vector for semantic search

        Returns:
            The created L2Entry
        """
        from datetime import datetime

        entry = L2Entry(
            slice_id=slice_data.id,
            slice_data=slice_data,
            embedding=embedding.astype(np.float32),
            created_at=datetime.now().isoformat(),
        )

        self._entries[slice_data.id] = entry
        self._embeddings[slice_data.id] = embedding.astype(np.float32)

        return entry

    def get(self, slice_id: str) -> L2Entry | None:
        """Get a slice from L2 by ID.

        Args:
            slice_id: Semantic hash of the slice

        Returns:
            L2Entry if found, None otherwise
        """
        return self._entries.get(slice_id)

    def remove(self, slice_id: str) -> L2Entry | None:
        """Remove a slice from L2.

        Args:
            slice_id: Semantic hash of the slice

        Returns:
            Removed L2Entry if found, None otherwise
        """
        if slice_id not in self._entries:
            return None

        entry = self._entries.pop(slice_id)
        self._embeddings.pop(slice_id, None)

        return entry

    def semantic_search(
        self, query_embedding: NDArray[np.float32], top_k: int | None = None
    ) -> list[RetrievalResult]:
        """Search for similar slices by embedding.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return (uses config default if None)

        Returns:
            List of RetrievalResult sorted by similarity
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        k = top_k or self.config.top_k

        # Compute cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        similarities = []
        for slice_id, emb in self._embeddings.items():
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                continue

            # Cosine similarity
            similarity = float(np.dot(query_embedding, emb) / (query_norm * emb_norm))
            similarities.append((slice_id, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for slice_id, score in similarities[:k]:
            entry = self._entries.get(slice_id)
            if entry:
                results.append(
                    RetrievalResult(
                        slice_id=slice_id,
                        content=entry.slice_data.content,
                        tokens=entry.slice_data.tokens,
                        embedding=entry.embedding,
                        score=score,
                        tier=MemoryTier.L2,
                    )
                )

        return results

    def get_all_entries(self) -> list[L2Entry]:
        """Get all entries in L2."""
        return list(self._entries.values())

    @property
    def used_tokens(self) -> int:
        """Total tokens stored in L2."""
        return sum(e.slice_data.token_count for e in self._entries.values())

    @property
    def slice_count(self) -> int:
        """Number of slices in L2."""
        return len(self._entries)

    @property
    def utilization(self) -> float:
        """L2 utilization ratio [0, 1]."""
        if self.config.max_tokens == 0:
            return 0.0
        return self.used_tokens / self.config.max_tokens

    def get_stats(self) -> MemoryStats:
        """Get statistics about L2."""
        return MemoryStats(
            tier=MemoryTier.L2,
            capacity_tokens=self.config.max_tokens,
            used_tokens=self.used_tokens,
            total_slices=self.slice_count,
        )

    def clear(self) -> None:
        """Clear all entries from L2."""
        self._entries.clear()
        self._embeddings.clear()


def create_embedding_from_hidden_states(
    hidden_states: NDArray[np.float32], method: str = "mean"
) -> NDArray[np.float32]:
    """Create an embedding from hidden states.

    Args:
        hidden_states: (seq_len, hidden_dim) hidden states
        method: Aggregation method ("mean", "max", "last")

    Returns:
        (hidden_dim,) embedding vector
    """
    if method == "mean":
        return np.mean(hidden_states, axis=0).astype(np.float32)
    elif method == "max":
        return np.max(hidden_states, axis=0).astype(np.float32)
    elif method == "last":
        return hidden_states[-1].astype(np.float32)
    else:
        raise ValueError(f"Unknown embedding method: {method}")


def compute_slice_embedding(
    slice_data: SemanticSlice,
    hidden_states: NDArray[np.float32],  # Full sequence hidden states
    method: str = "mean",
) -> NDArray[np.float32]:
    """Compute embedding for a semantic slice.

    Args:
        slice_data: Semantic slice
        hidden_states: (seq_len, hidden_dim) hidden states for full sequence
        method: Aggregation method

    Returns:
        (hidden_dim,) embedding vector
    """
    # Extract hidden states for this slice
    start, end = slice_data.start_pos, slice_data.end_pos
    slice_hidden = hidden_states[start:end, :]

    return create_embedding_from_hidden_states(slice_hidden, method)
