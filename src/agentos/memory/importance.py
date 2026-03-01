"""
Semantic Importance Score (ℐ) Calculator.

Based on AgentOS paper Section 3.2.2:

The importance score determines which semantic slices should be
kept in fast memory (L1) vs. paged out to slower tiers.

ℐ(σ) = w₁·I_attention + w₂·I_recency + w₃·I_frequency + w₄·I_user

Where:
- I_attention: Importance from attention gradients (focus)
- I_recency: Recent access bonus
- I_frequency: Frequent access bonus
- I_user: User-provided importance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray

from agentos.memory.types import PageTableEntry


@dataclass
class ImportanceConfig:
    """Configuration for importance score calculation."""

    # Weights for different factors (should sum to 1.0)
    w_attention: float = 0.4
    w_recency: float = 0.2
    w_frequency: float = 0.2
    w_user: float = 0.2

    # Recency decay: how fast old accesses lose importance
    # Higher = faster decay
    recency_half_life_hours: float = 1.0

    # Frequency normalization
    frequency_max_count: int = 100  # Access count that gives max score

    def validate(self) -> None:
        """Validate configuration."""
        total_weight = (
            self.w_attention + self.w_recency + self.w_frequency + self.w_user
        )
        if not (0.9 <= total_weight <= 1.1):  # Allow small floating point errors
            raise ValueError(
                f"Weights must sum to ~1.0, got {total_weight}. "
                f"w_attention={self.w_attention}, w_recency={self.w_recency}, "
                f"w_frequency={self.w_frequency}, w_user={self.w_user}"
            )
        if self.recency_half_life_hours <= 0:
            raise ValueError("recency_half_life_hours must be positive")
        if self.frequency_max_count <= 0:
            raise ValueError("frequency_max_count must be positive")


class ImportanceCalculator:
    """Calculate semantic importance scores for slices.

    The importance score ℐ(σ) is used by the S-MMU to make eviction decisions.
    Slices with higher importance are kept in L1 (fast memory).
    """

    def __init__(self, config: ImportanceConfig | None = None) -> None:
        """Initialize the importance calculator.

        Args:
            config: Configuration for importance calculation. If None, uses defaults.
        """
        self.config = config or ImportanceConfig()
        self.config.validate()

    def compute(
        self,
        density_values: NDArray[np.float32],
        entry: PageTableEntry | None = None,
    ) -> float:
        """Compute the importance score for a slice.

        Args:
            density_values: Density values D(t) for tokens in this slice
            entry: Page table entry (optional, for recency/frequency tracking)

        Returns:
            Importance score ℐ in range [0, 1]
        """
        # Calculate individual factors
        i_attention = self._attention_importance(density_values)
        i_recency = self._recency_importance(entry) if entry else 0.5
        i_frequency = self._frequency_importance(entry) if entry else 0.5
        i_user = self._user_importance(entry) if entry else 0.5

        # Weighted combination
        importance = (
            self.config.w_attention * i_attention
            + self.config.w_recency * i_recency
            + self.config.w_frequency * i_frequency
            + self.config.w_user * i_user
        )

        # Clamp to [0, 1]
        return float(np.clip(importance, 0.0, 1.0))

    def _attention_importance(
        self, density_values: NDArray[np.float32]
    ) -> float:
        """Calculate importance from attention density.

        Higher density = more focused attention = more important.

        Formula: mean(D(t)) over the slice
        """
        if len(density_values) == 0:
            return 0.0

        mean_density = float(np.mean(density_values))
        return mean_density  # D(t) is already in [0, 1]

    def _recency_importance(self, entry: PageTableEntry) -> float:
        """Calculate importance from recency of access.

        More recent access = higher importance.

        Uses exponential decay with configurable half-life.
        """
        if entry is None or entry.last_accessed is None:
            # New slices get high default recency importance
            return 1.0

        now = datetime.now()
        time_delta = now - entry.last_accessed
        hours_ago = time_delta.total_seconds() / 3600.0

        # Exponential decay: I = 2^(-t / half_life)
        decay_factor = 2.0 ** (-hours_ago / self.config.recency_half_life_hours)

        return float(decay_factor)

    def _frequency_importance(self, entry: PageTableEntry) -> float:
        """Calculate importance from access frequency.

        More accesses = higher importance (with diminishing returns).

        Uses logarithmic scaling.
        """
        if entry is None or entry.access_count == 0:
            # New slices get moderate default frequency importance
            return 0.5

        # Logarithmic scaling: I = log(1 + count) / log(1 + max_count)
        count = min(entry.access_count, self.config.frequency_max_count)
        score = np.log1p(count) / np.log1p(self.config.frequency_max_count)

        return float(score)

    def _user_importance(self, entry: PageTableEntry) -> float:
        """Get user-provided importance.

        Users can pin slices or provide explicit importance scores.
        """
        if entry is None:
            return 0.0

        # If pinned, give maximum importance
        if entry.is_pinned:
            return 1.0

        # Check for user-provided score in metadata
        user_score = entry.metadata.get("user_importance", 0.5)
        return float(np.clip(user_score, 0.0, 1.0))

    def compute_batch(
        self,
        slices: list[tuple[NDArray[np.float32], PageTableEntry | None]],
    ) -> NDArray[np.float32]:
        """Compute importance scores for multiple slices.

        Args:
            slices: List of (density_values, page_table_entry) tuples

        Returns:
            Array of importance scores, one per slice
        """
        scores = np.zeros(len(slices), dtype=np.float32)

        for i, (density_values, entry) in enumerate(slices):
            scores[i] = self.compute(density_values, entry)

        return scores


def compute_importance(
    density_values: NDArray[np.float32],
    entry: PageTableEntry | None = None,
    w_attention: float = 0.4,
    w_recency: float = 0.2,
    w_frequency: float = 0.2,
    w_user: float = 0.2,
) -> float:
    """Convenience function to compute importance score.

    Args:
        density_values: Density values D(t) for tokens in this slice
        entry: Page table entry (optional)
        w_attention: Weight for attention-based importance
        w_recency: Weight for recency-based importance
        w_frequency: Weight for frequency-based importance
        w_user: Weight for user-provided importance

    Returns:
        Importance score ℐ in range [0, 1]
    """
    config = ImportanceConfig(
        w_attention=w_attention,
        w_recency=w_recency,
        w_frequency=w_frequency,
        w_user=w_user,
    )
    calculator = ImportanceCalculator(config)
    return calculator.compute(density_values, entry)
