"""
Data types for semantic slicing.

Based on AgentOS paper:
- Semantic Slice σ: Coherent cluster of tokens based on attention cohesion
- Contextual Information Density D(t): Information density at position t
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class AttentionOutput:
    """Output from model forward pass with attention weights."""

    tokens: list[str]  # Tokenized text
    token_ids: list[int]  # Token IDs
    decoded_text: str  # Full decoded text (properly formatted)
    hidden_states: NDArray[np.float32]  # (seq_len, hidden_dim)
    attention_weights: NDArray[np.float32]  # (num_layers, num_heads, seq_len, seq_len)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DensityProfile:
    """Contextual Information Density (CID) profile.

    D(t) = 1 - [-1/H Σᵢ₌₁ᴴ Σⱼ₌₁ᵗ αᵢ,ⱼ log(αᵢ,ⱼ)]

    Where:
    - αᵢ,ⱼ: attention weight from token i to token j
    - H: number of attention heads
    """

    densities: NDArray[np.float32]  # (seq_len,) - D(t) at each position
    entropy: NDArray[np.float32]  # (seq_len,) - normalized entropy at each position
    gradients: NDArray[np.float32]  # (seq_len,) - ∂D(t)/∂t

    def get_boundaries(self, threshold: float) -> list[int]:
        """Get positions where gradient exceeds threshold."""
        return np.where(np.abs(self.gradients) > threshold)[0].tolist()


@dataclass
class SemanticSlice:
    """A semantic slice σ - coherent cluster of tokens.

    Corresponds to a "cognitive page" in the S-MMU.
    """

    id: str  # Semantic hash (e.g., SHA-256 of content)
    start_pos: int  # Start token position
    end_pos: int  # End token position (exclusive)
    tokens: list[str]  # Tokens in this slice
    token_ids: list[int]  # Token IDs
    content: str  # Decoded text content

    # Density information
    density_mean: float  # Average D(t) over this slice
    density_std: float  # Std dev of D(t) over this slice

    # Importance score (for S-MMU eviction decisions)
    importance_score: float = 0.5

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        """Number of tokens in this slice."""
        return len(self.tokens)

    def __len__(self) -> int:
        return self.token_count


@dataclass
class SlicingResult:
    """Result of semantic slicing operation."""

    slices: list[SemanticSlice]
    density_profile: DensityProfile
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_slices(self) -> int:
        return len(self.slices)

    @property
    def total_tokens(self) -> int:
        return sum(s.token_count for s in self.slices)

    def get_slice_at_position(self, pos: int) -> Optional[SemanticSlice]:
        """Get the slice containing the given token position."""
        for slice_ in self.slices:
            if slice_.start_pos <= pos < slice_.end_pos:
                return slice_
        return None

    def get_slice_statistics(self) -> dict[str, float]:
        """Get statistics about slice sizes and densities."""
        if not self.slices:
            return {
                "count": 0,
                "mean_tokens": 0.0,
                "std_tokens": 0.0,
                "mean_density": 0.0,
                "std_density": 0.0,
            }

        token_counts = [s.token_count for s in self.slices]
        densities = [s.density_mean for s in self.slices]

        return {
            "count": len(self.slices),
            "mean_tokens": float(np.mean(token_counts)),
            "std_tokens": float(np.std(token_counts)),
            "mean_density": float(np.mean(densities)),
            "std_density": float(np.std(densities)),
        }
