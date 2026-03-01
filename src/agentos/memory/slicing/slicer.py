"""
Semantic Slicer.

Aggregates tokens into coherent semantic slices based on attention patterns
and information density boundaries.

From the AgentOS paper:
"Unlike traditional LLM inference, which treats the context window as a
monolithic, sliding buffer of N tokens, AgentOS implements Dynamic Semantic
Slicing. This process aggregates tokens into coherent clusters {σ₁, σ₂, ..., σₖ}
based on their mutual information and attention cohesion."
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from agentos.memory.slicing.boundary_detector import BoundaryDetector, BoundaryDetectorConfig
from agentos.memory.slicing.cid_calculator import CIDCalculator, CIDCalculatorConfig
from agentos.memory.slicing.types import (
    AttentionOutput,
    DensityProfile,
    SemanticSlice,
    SlicingResult,
)


@dataclass
class SemanticSlicerConfig:
    """Configuration for semantic slicing."""

    # CID calculation configuration
    cid_config: CIDCalculatorConfig = None

    # Boundary detection configuration
    boundary_config: BoundaryDetectorConfig = None

    # Whether to compute importance scores for slices
    compute_importance: bool = True

    # Method for computing importance
    importance_method: str = "density"  # "density", "variance", "combined"

    def __post_init__(self):
        """Initialize default configs if not provided."""
        if self.cid_config is None:
            self.cid_config = CIDCalculatorConfig()
        if self.boundary_config is None:
            self.boundary_config = BoundaryDetectorConfig()

    def validate(self) -> None:
        """Validate configuration."""
        self.cid_config.validate()
        self.boundary_config.validate()

        valid_importance = {"density", "variance", "combined"}
        if self.importance_method not in valid_importance:
            raise ValueError(
                f"Invalid importance_method: {self.importance_method}. "
                f"Must be one of {valid_importance}"
            )


class SemanticSlicer:
    """Slice text into semantic units based on attention patterns.

    The slicer:
    1. Computes Contextual Information Density (CID) from attention
    2. Detects boundaries where density gradient exceeds threshold
    3. Creates semantic slices between boundaries
    4. Assigns semantic hash and importance scores
    """

    def __init__(self, config: SemanticSlicerConfig | None = None) -> None:
        """Initialize the semantic slicer.

        Args:
            config: Configuration for semantic slicing. If None, uses defaults.
        """
        self.config = config or SemanticSlicerConfig()
        self.config.validate()

        self.cid_calculator = CIDCalculator(self.config.cid_config)
        self.boundary_detector = BoundaryDetector(self.config.boundary_config)

    def slice(
        self, attention_output: AttentionOutput, tokenizer: Any = None
    ) -> SlicingResult:
        """Slice text into semantic units.

        Args:
            attention_output: Output from model with attention weights.
            tokenizer: Optional tokenizer for proper text decoding. If provided,
                      slice content will use tokenizer.decode() for accurate
                      text reconstruction.

        Returns:
            SlicingResult with semantic slices and density profile.
        """
        # Step 1: Compute CID profile
        density_profile = self.cid_calculator.compute(attention_output)

        # Step 2: Detect boundaries
        boundary_positions, threshold = self.boundary_detector.detect_boundaries(
            density_profile
        )

        # Step 3: Create semantic slices
        slices = self._create_slices(
            attention_output,
            density_profile,
            boundary_positions,
            tokenizer,
        )

        return SlicingResult(
            slices=slices,
            density_profile=density_profile,
            metadata={
                "boundary_threshold": threshold,
                "num_boundaries": len(boundary_positions),
                "boundary_positions": boundary_positions,
            },
        )

    def _create_slices(
        self,
        attention_output: AttentionOutput,
        density_profile: DensityProfile,
        boundary_positions: list[int],
        tokenizer: Any = None,
    ) -> list[SemanticSlice]:
        """Create semantic slices from boundaries.

        Args:
            attention_output: Output from model with attention weights.
            density_profile: Computed density profile.
            boundary_positions: List of boundary positions.
            tokenizer: Optional tokenizer for proper text decoding.

        Returns:
            List of semantic slices.
        """
        slices = []
        tokens = attention_output.tokens
        token_ids = attention_output.token_ids

        # Sort boundaries and ensure they're within range
        boundaries = sorted([b for b in boundary_positions if 0 <= b < len(tokens)])

        if not boundaries:
            # No boundaries found, create single slice with all tokens
            boundaries = [0, len(tokens)]

        # Ensure first boundary is at 0
        if boundaries[0] != 0:
            boundaries.insert(0, 0)

        # Ensure last boundary is at end
        if boundaries[-1] != len(tokens):
            boundaries.append(len(tokens))

        # Create slices between boundaries
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            if start >= end:  # Skip invalid ranges
                continue

            slice_tokens = tokens[start:end]
            slice_token_ids = token_ids[start:end]

            # Decode slice content properly if tokenizer is available
            if tokenizer is not None:
                slice_content = tokenizer.decode(slice_token_ids, skip_special_tokens=True)
            else:
                # Fallback: clean up special token markers and join
                # Handle different tokenizer types:
                # - SentencePiece (Qwen, LLaMA): "Ġ" indicates a space
                # - BPE (GPT-2): "Ċ" indicates space before word
                cleaned_tokens = []
                for t in slice_tokens:
                    # Replace common space indicators
                    t = t.replace("Ġ", " ")  # SentencePiece
                    t = t.replace("Ċ", " ")  # BPE space prefix
                    t = t.replace("▁", " ")  # Another space variant
                    cleaned_tokens.append(t)
                slice_content = "".join(cleaned_tokens).strip()

            # Compute density statistics for this slice
            slice_densities = density_profile.densities[start:end]
            density_mean = float(slice_densities.mean())
            density_std = float(slice_densities.std())

            # Compute importance score
            importance = self._compute_importance(
                density_mean,
                density_std,
                slice_densities,
            )

            # Create semantic hash (SHA-256 of content)
            slice_hash = hashlib.sha256(slice_content.encode()).hexdigest()[:16]

            slice_ = SemanticSlice(
                id=slice_hash,
                start_pos=start,
                end_pos=end,
                tokens=slice_tokens,
                token_ids=slice_token_ids,
                content=slice_content,
                density_mean=density_mean,
                density_std=density_std,
                importance_score=importance,
            )

            slices.append(slice_)

        return slices

    def _compute_importance(
        self,
        density_mean: float,
        density_std: float,
        densities: list,  # Array-like but avoid numpy type issues
    ) -> float:
        """Compute importance score for a slice.

        Higher importance = more likely to be kept in L1 cache.

        Args:
            density_mean: Mean density of the slice.
            density_std: Standard deviation of density.
            densities: Density values for the slice.

        Returns:
            Importance score between 0 and 1.
        """
        if not self.config.compute_importance:
            return 0.5

        if self.config.importance_method == "density":
            # Higher density = more important
            return min(1.0, max(0.0, density_mean))

        elif self.config.importance_method == "variance":
            # Lower variance = more coherent/important
            # Normalize by assuming max reasonable std is 0.5
            variance_importance = 1.0 - min(1.0, density_std / 0.5)
            return variance_importance

        elif self.config.importance_method == "combined":
            # Weighted combination of density and variance
            density_score = min(1.0, max(0.0, density_mean))
            variance_score = 1.0 - min(1.0, density_std / 0.5)
            return 0.6 * density_score + 0.4 * variance_score

        else:
            return 0.5


def slice_semantic(
    attention_output: AttentionOutput,
    threshold_percentile: float = 75.0,
    layer_aggregation: str = "mean",
    head_aggregation: str = "mean",
    tokenizer: Any = None,
) -> SlicingResult:
    """Convenience function to perform semantic slicing.

    Args:
        attention_output: Output from model with attention weights.
        threshold_percentile: Percentile for adaptive threshold.
        layer_aggregation: How to aggregate across layers.
        head_aggregation: How to aggregate across heads.
        tokenizer: Optional tokenizer for proper text decoding.

    Returns:
        SlicingResult with semantic slices and density profile.
    """
    cid_config = CIDCalculatorConfig(
        layer_aggregation=layer_aggregation,
        head_aggregation=head_aggregation,
    )
    boundary_config = BoundaryDetectorConfig(
        threshold_strategy="adaptive_percentile",
        adaptive_percentile=threshold_percentile,
    )
    config = SemanticSlicerConfig(
        cid_config=cid_config,
        boundary_config=boundary_config,
    )
    slicer = SemanticSlicer(config)
    return slicer.slice(attention_output, tokenizer=tokenizer)
