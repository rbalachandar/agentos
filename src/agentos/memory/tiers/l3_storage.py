"""
L3 Storage: Knowledge Base.

Based on AgentOS paper Section 3.2.1:

L3 is the cold storage tier for long-term knowledge retention.
It represents the external RAG system and archival storage.

Key properties:
- Essentially unlimited capacity
- Slower access (requires explicit I/O)
- Stores compressed/archived slices
- Used for rarely-accessed but important information

This is a simple file-based implementation. In production, this would
connect to an external vector database (Pinecone, Weaviate, etc.) or
a distributed file system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from agentos.memory.slicing.types import SemanticSlice
from agentos.memory.types import MemoryStats, MemoryTier, RetrievalResult


@dataclass
class L3Config:
    """Configuration for L3 Storage."""

    # Storage directory
    storage_path: str = "./data/l3_storage"

    # Maximum number of slices (essentially unlimited)
    max_slices: int = 1000000

    # Compression options
    compress: bool = True

    # Index file name
    index_filename: str = "l3_index.json"

    def validate(self) -> None:
        """Validate configuration."""
        if self.max_slices <= 0:
            raise ValueError("max_slices must be positive")


@dataclass
class L3Entry:
    """An entry in L3 storage.

    Represents an archived semantic slice on disk.
    """

    slice_id: str
    slice_data: SemanticSlice

    # File path where slice is stored
    file_path: str

    # Storage metadata
    size_bytes: int
    compressed: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


class L3Storage:
    """L3 Storage - Knowledge Base / Cold Storage.

    Stores semantic slices on disk for long-term retention.
    """

    def __init__(self, config: L3Config | None = None) -> None:
        """Initialize L3 storage.

        Args:
            config: Configuration for L3. If None, uses defaults.
        """
        self.config = config or L3Config()
        self.config.validate()

        # Create storage directory
        self.storage_dir = Path(self.config.storage_path)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index (loaded from disk on init)
        self._index: dict[str, dict] = {}

        # Load existing index if available
        self._load_index()

    def _get_slice_path(self, slice_id: str) -> Path:
        """Get file path for a slice."""
        return self.storage_dir / f"{slice_id}.json"

    def _load_index(self) -> None:
        """Load index from disk."""
        index_path = self.storage_dir / self.config.index_filename
        if index_path.exists():
            with open(index_path, "r") as f:
                self._index = json.load(f)

    def _save_index(self) -> None:
        """Save index to disk."""
        index_path = self.storage_dir / self.config.index_filename
        with open(index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def add(self, slice_data: SemanticSlice) -> L3Entry:
        """Add a slice to L3 storage.

        Args:
            slice_data: Semantic slice to store

        Returns:
            The created L3Entry
        """
        slice_id = slice_data.id
        file_path = self._get_slice_path(slice_id)

        # Serialize slice data
        slice_dict = {
            "id": slice_data.id,
            "start_pos": slice_data.start_pos,
            "end_pos": slice_data.end_pos,
            "tokens": slice_data.tokens,
            "token_ids": slice_data.token_ids,
            "content": slice_data.content,
            "density_mean": slice_data.density_mean,
            "density_std": slice_data.density_std,
            "importance_score": slice_data.importance_score,
            "metadata": slice_data.metadata,
        }

        # Write to file
        with open(file_path, "w") as f:
            json.dump(slice_dict, f)

        # Update index
        self._index[slice_id] = {
            "file_path": str(file_path),
            "size_bytes": file_path.stat().st_size,
            "compressed": False,
        }

        self._save_index()

        return L3Entry(
            slice_id=slice_id,
            slice_data=slice_data,
            file_path=str(file_path),
            size_bytes=file_path.stat().st_size,
            compressed=False,
        )

    def get(self, slice_id: str) -> L3Entry | None:
        """Get a slice from L3 by ID.

        Loads the slice from disk.

        Args:
            slice_id: Semantic hash of the slice

        Returns:
            L3Entry if found, None otherwise
        """
        if slice_id not in self._index:
            return None

        entry_data = self._index[slice_id]
        file_path = Path(entry_data["file_path"])

        if not file_path.exists():
            # File missing, remove from index
            del self._index[slice_id]
            self._save_index()
            return None

        # Load slice from file
        with open(file_path, "r") as f:
            slice_dict = json.load(f)

        # Reconstruct SemanticSlice
        from agentos.memory.slicing.types import SemanticSlice

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

        return L3Entry(
            slice_id=slice_id,
            slice_data=slice_data,
            file_path=str(file_path),
            size_bytes=entry_data["size_bytes"],
            compressed=entry_data.get("compressed", False),
        )

    def remove(self, slice_id: str) -> bool:
        """Remove a slice from L3.

        Args:
            slice_id: Semantic hash of the slice

        Returns:
            True if removed, False if not found
        """
        if slice_id not in self._index:
            return False

        entry_data = self._index[slice_id]
        file_path = Path(entry_data["file_path"])

        # Delete file
        if file_path.exists():
            file_path.unlink()

        # Remove from index
        del self._index[slice_id]
        self._save_index()

        return True

    def list_all(self) -> list[str]:
        """List all slice IDs in L3."""
        return list(self._index.keys())

    @property
    def slice_count(self) -> int:
        """Number of slices in L3."""
        return len(self._index)

    @property
    def total_size_bytes(self) -> int:
        """Total size of all stored slices in bytes."""
        return sum(e.get("size_bytes", 0) for e in self._index.values())

    def get_stats(self) -> MemoryStats:
        """Get statistics about L3."""
        return MemoryStats(
            tier=MemoryTier.L3,
            capacity_tokens=self.config.max_slices * 1000,  # Rough estimate
            used_tokens=self.slice_count * 100,  # Rough estimate
            total_slices=self.slice_count,
        )

    def clear(self) -> None:
        """Clear all entries from L3 (dangerous!)."""
        for slice_id in list(self._index.keys()):
            self.remove(slice_id)

    def get_slice_path(self, slice_id: str) -> Path | None:
        """Get the file path for a slice ID.

        Args:
            slice_id: Slice identifier

        Returns:
            Path to slice file, or None if not found
        """
        if slice_id not in self._index:
            return None
        return Path(self._index[slice_id]["file_path"])
