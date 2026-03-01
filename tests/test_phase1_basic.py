"""
Basic unit tests for Phase 1 components.

These tests verify the core logic without requiring a model download.
"""

import numpy as np
import pytest

from agentos.memory.slicing.types import (
    AttentionOutput,
    DensityProfile,
    SemanticSlice,
    SlicingResult,
)
from agentos.memory.slicing.cid_calculator import (
    CIDCalculator,
    CIDCalculatorConfig,
    compute_cid,
)
from agentos.memory.slicing.boundary_detector import (
    BoundaryDetector,
    BoundaryDetectorConfig,
    ThresholdStrategy,
    detect_boundaries,
)
from agentos.memory.slicing.slicer import (
    SemanticSlicer,
    SemanticSlicerConfig,
    slice_semantic,
)


@pytest.fixture
def mock_attention_output():
    """Create a mock attention output for testing."""
    seq_len = 20
    num_layers = 4
    num_heads = 8
    hidden_dim = 64

    # Create realistic-looking attention weights
    # Some positions have more focused attention than others
    np.random.seed(42)
    attention = np.random.rand(num_layers, num_heads, seq_len, seq_len).astype(np.float32)

    # Make it more realistic: attention should be stronger to recent tokens
    for layer in range(num_layers):
        for head in range(num_heads):
            for i in range(seq_len):
                # Exponential decay for attention to previous tokens
                for j in range(i + 1):
                    attention[layer, head, i, j] = np.exp(-(i - j) / 5.0)

                # Normalize
                attention[layer, head, i, : i + 1] = (
                    attention[layer, head, i, : i + 1]
                    / attention[layer, head, i, : i + 1].sum()
                )

    # Create some "semantic boundaries" by making attention more uniform
    # at positions 7 and 14
    for layer in range(num_layers):
        for head in range(num_heads):
            for i in [7, 14]:
                if i < seq_len:
                    attention[layer, head, i, : i + 1] = np.ones(i + 1) / (i + 1)

    tokens = [f"token_{i}" for i in range(seq_len)]
    token_ids = list(range(seq_len))
    hidden_states = np.random.rand(seq_len, hidden_dim).astype(np.float32)

    # Create decoded text for testing
    decoded_text = " ".join(tokens)

    return AttentionOutput(
        tokens=tokens,
        token_ids=token_ids,
        decoded_text=decoded_text,
        hidden_states=hidden_states,
        attention_weights=attention,
        metadata={"num_layers": num_layers, "num_heads": num_heads, "seq_len": seq_len},
    )


class TestCIDCalculator:
    """Tests for CID calculator."""

    def test_compute_cid(self, mock_attention_output):
        """Test that CID computation works."""
        calculator = CIDCalculator()
        profile = calculator.compute(mock_attention_output)

        assert profile is not None
        assert len(profile.densities) == len(mock_attention_output.tokens)
        assert len(profile.entropy) == len(mock_attention_output.tokens)
        assert len(profile.gradients) == len(mock_attention_output.tokens)

    def test_density_range(self, mock_attention_output):
        """Test that density values are in valid range."""
        profile = compute_cid(mock_attention_output)

        # Densities should be between 0 and 1 (allow small floating-point errors)
        assert np.all(profile.densities >= -1e-6)
        assert np.all(profile.densities <= 1.0 + 1e-6)

    def test_entropy_normalized(self, mock_attention_output):
        """Test that entropy is normalized."""
        profile = compute_cid(mock_attention_output)

        # Normalized entropy should typically be in reasonable range
        assert np.all(profile.entropy >= 0.0)
        # Note: entropy can be > 1, so we don't upper-bound it


class TestBoundaryDetector:
    """Tests for boundary detector."""

    def test_detect_boundaries(self, mock_attention_output):
        """Test that boundary detection works."""
        profile = compute_cid(mock_attention_output)
        detector = BoundaryDetector()
        boundaries, threshold = detector.detect_boundaries(profile)

        assert isinstance(boundaries, list)
        assert isinstance(threshold, float)
        assert len(boundaries) > 0
        assert boundaries[0] == 0  # First boundary should be at start

    def test_adaptive_threshold(self, mock_attention_output):
        """Test adaptive threshold strategy."""
        profile = compute_cid(mock_attention_output)

        config = BoundaryDetectorConfig(
            threshold_strategy=ThresholdStrategy.ADAPTIVE_PERCENTILE,
            adaptive_percentile=75.0,
        )
        detector = BoundaryDetector(config)
        boundaries, threshold = detector.detect_boundaries(profile)

        # Threshold should be computed from gradient distribution
        assert threshold > 0

    def test_fixed_threshold(self, mock_attention_output):
        """Test fixed threshold strategy."""
        profile = compute_cid(mock_attention_output)

        config = BoundaryDetectorConfig(
            threshold_strategy=ThresholdStrategy.FIXED,
            fixed_threshold=0.1,
        )
        detector = BoundaryDetector(config)
        boundaries, threshold = detector.detect_boundaries(profile)

        assert threshold == 0.1


class TestSemanticSlicer:
    """Tests for semantic slicer."""

    def test_slice(self, mock_attention_output):
        """Test that slicing produces valid results."""
        slicer = SemanticSlicer()
        result = slicer.slice(mock_attention_output)

        assert isinstance(result, SlicingResult)
        assert len(result.slices) > 0
        assert result.total_slices > 0
        assert result.total_tokens == len(mock_attention_output.tokens)

    def test_slices_valid(self, mock_attention_output):
        """Test that slices have valid properties."""
        result = slice_semantic(mock_attention_output)

        for slice_ in result.slices:
            assert slice_.start_pos >= 0
            assert slice_.end_pos > slice_.start_pos
            assert slice_.token_count > 0
            assert len(slice_.tokens) == slice_.token_count
            assert slice_.density_mean >= 0.0
            assert slice_.density_mean <= 1.0
            assert slice_.importance_score >= 0.0
            assert slice_.importance_score <= 1.0

    def test_slices_cover_sequence(self, mock_attention_output):
        """Test that slices cover the entire sequence."""
        result = slice_semantic(mock_attention_output)

        # Check that slices are in order and cover all tokens
        sorted_slices = sorted(result.slices, key=lambda s: s.start_pos)
        assert sorted_slices == result.slices

        # Check coverage
        prev_end = 0
        for slice_ in result.slices:
            assert slice_.start_pos == prev_end
            prev_end = slice_.end_pos

        assert prev_end == len(mock_attention_output.tokens)

    def test_get_slice_statistics(self, mock_attention_output):
        """Test that slice statistics are computed correctly."""
        result = slice_semantic(mock_attention_output)
        stats = result.get_slice_statistics()

        assert stats["count"] == len(result.slices)
        assert stats["mean_tokens"] > 0
        assert stats["std_tokens"] >= 0
        assert stats["mean_density"] >= 0
        assert stats["mean_density"] <= 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_cid_function(self, mock_attention_output):
        """Test the convenience compute_cid function."""
        profile = compute_cid(
            mock_attention_output,
            layer_aggregation="mean",
            head_aggregation="mean",
        )

        assert profile is not None
        assert len(profile.densities) > 0

    def test_detect_boundaries_function(self, mock_attention_output):
        """Test the convenience detect_boundaries function."""
        profile = compute_cid(mock_attention_output)
        boundaries, threshold = detect_boundaries(
            profile,
            threshold_strategy=ThresholdStrategy.ADAPTIVE_PERCENTILE,
            adaptive_percentile=75.0,
        )

        assert isinstance(boundaries, list)
        assert isinstance(threshold, float)

    def test_slice_semantic_function(self, mock_attention_output):
        """Test the convenience slice_semantic function."""
        result = slice_semantic(
            mock_attention_output,
            threshold_percentile=75.0,
            layer_aggregation="mean",
            head_aggregation="mean",
        )

        assert isinstance(result, SlicingResult)
        assert len(result.slices) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
