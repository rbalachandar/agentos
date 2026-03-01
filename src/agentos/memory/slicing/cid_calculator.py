"""
Contextual Information Density (CID) Calculator.

Implements Formula (2) from AgentOS paper:

    D(t) = 1 - [-1/H Σᵢ₌₁ᴴ Σⱼ₌₁ᵗ αᵢ,ⱼ log(αᵢ,ⱼ)]

Where:
- D(t): Contextual Information Density at position t
- αᵢ,ⱼ: attention weight from token i to token j
- H: number of attention heads
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from agentos.memory.slicing.types import AttentionOutput, DensityProfile


@dataclass
class CIDCalculatorConfig:
    """Configuration for CID calculation."""

    # Aggregation strategy across layers
    layer_aggregation: str = "mean"  # "mean", "max", "min", "last"

    # Aggregation strategy across heads
    head_aggregation: str = "mean"  # "mean", "max", "min"

    # Whether to normalize entropy by sequence length
    normalize_entropy: bool = True

    # Small constant to avoid log(0)
    epsilon: float = 1e-10

    def validate(self) -> None:
        """Validate configuration."""
        valid_layer_agg = {"mean", "max", "min", "last"}
        valid_head_agg = {"mean", "max", "min"}

        if self.layer_aggregation not in valid_layer_agg:
            raise ValueError(
                f"Invalid layer_aggregation: {self.layer_aggregation}. "
                f"Must be one of {valid_layer_agg}"
            )
        if self.head_aggregation not in valid_head_agg:
            raise ValueError(
                f"Invalid head_aggregation: {self.head_aggregation}. "
                f"Must be one of {valid_head_agg}"
            )


class CIDCalculator:
    """Calculate Contextual Information Density from attention weights.

    The CID measures how "focused" or "dense" the information is at each
    position in the sequence. Low entropy (high density) indicates the
    model is attending to specific information, while high entropy (low
    density) indicates more diffuse attention.
    """

    def __init__(self, config: CIDCalculatorConfig | None = None) -> None:
        """Initialize the CID calculator.

        Args:
            config: Configuration for CID calculation. If None, uses defaults.
        """
        self.config = config or CIDCalculatorConfig()
        self.config.validate()

    def compute(self, attention_output: AttentionOutput) -> DensityProfile:
        """Compute CID profile from attention weights.

        Args:
            attention_output: Output from model with attention weights.

        Returns:
            DensityProfile with densities, entropy, and gradients.
        """
        attention = attention_output.attention_weights
        _, _, seq_len, _ = attention.shape

        # Step 1: Aggregate attention across layers and heads
        # Result: (seq_len, seq_len) matrix
        aggregated_attention = self._aggregate_attention(attention)

        # Step 2: Compute entropy at each position
        # H(Pₜ) = -Σⱼ₌₁ᵗ αₜ,ⱼ log(αₜ,ⱼ)
        entropies = self._compute_entropy(aggregated_attention)

        # Step 3: Normalize entropy by log(t) for sequence-length invariance
        if self.config.normalize_entropy:
            normalized_entropies = self._normalize_entropy(entropies)
        else:
            normalized_entropies = entropies

        # Step 4: Compute D(t) = 1 - H(Pₜ) / log(t)
        # High density = low entropy, so we invert
        densities = 1.0 - normalized_entropies

        # Clamp to [0, 1] to handle floating point edge cases
        densities = np.clip(densities, 0.0, 1.0)

        # Step 5: Compute gradient ∂D(t)/∂t
        gradients = np.gradient(densities)

        return DensityProfile(
            densities=densities.astype(np.float32),
            entropy=normalized_entropies.astype(np.float32),
            gradients=gradients.astype(np.float32),
        )

    def _aggregate_attention(
        self,
        attention: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Aggregate attention weights across layers and heads.

        Args:
            attention: (num_layers, num_heads, seq_len, seq_len)

        Returns:
            Aggregated attention: (seq_len, seq_len)
        """
        # Aggregate across both layers and heads in one pass
        if self.config.layer_aggregation == "mean" and self.config.head_aggregation == "mean":
            # Optimize: single pass aggregation
            return attention.mean(axis=(0, 1))

        # Different aggregation strategies for layers and heads
        if self.config.layer_aggregation == "mean":
            layer_agg = attention.mean(axis=0)  # (num_heads, seq_len, seq_len)
        elif self.config.layer_aggregation == "max":
            layer_agg = attention.max(axis=0)
        elif self.config.layer_aggregation == "min":
            layer_agg = attention.min(axis=0)
        elif self.config.layer_aggregation == "last":
            layer_agg = attention[-1]  # Last layer only
        else:
            raise ValueError(f"Unknown layer aggregation: {self.config.layer_aggregation}")

        # Then aggregate across heads
        if self.config.head_aggregation == "mean":
            head_agg = layer_agg.mean(axis=0)  # (seq_len, seq_len)
        elif self.config.head_aggregation == "max":
            head_agg = layer_agg.max(axis=0)
        elif self.config.head_aggregation == "min":
            head_agg = layer_agg.min(axis=0)
        else:
            raise ValueError(f"Unknown head aggregation: {self.config.head_aggregation}")

        return head_agg

    def _compute_entropy(
        self, attention: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Compute Shannon entropy at each position.

        H(Pₜ) = -Σⱼ₌₁ᵗ αₜ,ⱼ log(αₜ,ⱼ)

        Args:
            attention: (seq_len, seq_len) aggregated attention matrix

        Returns:
            Entropy at each position: (seq_len,)
        """
        seq_len = attention.shape[0]
        entropies = np.zeros(seq_len, dtype=np.float32)

        for t in range(seq_len):
            # Get attention distribution from position t to all previous positions
            # (including self-attention)
            attn_t = attention[t, : t + 1]  # (t + 1,)

            # Add small epsilon to avoid log(0)
            attn_t = attn_t + self.config.epsilon

            # Normalize to ensure it's a proper probability distribution
            attn_sum = attn_t.sum()
            if attn_sum > 0:
                attn_t = attn_t / attn_sum
                # Compute entropy
                entropy = -np.sum(attn_t * np.log(attn_t))
            else:
                # Edge case: all zeros, return max entropy
                entropy = np.log(len(attn_t))

            entropies[t] = entropy

        return entropies

    def _normalize_entropy(self, entropies: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalize entropy by log(sequence_length).

        This makes entropy comparable across different sequence lengths.

        Args:
            entropies: (seq_len,) raw entropy values

        Returns:
            Normalized entropy: (seq_len,)
        """
        seq_len = len(entropies)
        # Avoid log(1) = 0 at position 0
        normalizer = np.log(np.arange(1, seq_len + 1, dtype=np.float32))
        normalizer[0] = 1.0  # Avoid division by zero

        return entropies / normalizer


def compute_cid(
    attention_output: AttentionOutput,
    layer_aggregation: str = "mean",
    head_aggregation: str = "mean",
) -> DensityProfile:
    """Convenience function to compute CID.

    Args:
        attention_output: Output from model with attention weights.
        layer_aggregation: How to aggregate across layers ("mean", "max", "min", "last")
        head_aggregation: How to aggregate across heads ("mean", "max", "min")

    Returns:
        DensityProfile with densities, entropy, and gradients.
    """
    config = CIDCalculatorConfig(
        layer_aggregation=layer_aggregation,
        head_aggregation=head_aggregation,
    )
    calculator = CIDCalculator(config)
    return calculator.compute(attention_output)
