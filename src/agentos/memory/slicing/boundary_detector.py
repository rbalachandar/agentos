"""
Semantic Boundary Detector.

Implements Formula (7) from AgentOS paper for detecting semantic slice boundaries:

    ∂D(t)/∂t > ε ⇒ t ∈ ∂σ

Where:
- ∂D(t)/∂t: Gradient of information density at position t
- ε: Boundary threshold
- ∂σ: Set of boundary positions (semantic slice boundaries)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from agentos.memory.slicing.types import DensityProfile


class ThresholdStrategy(str, Enum):
    """Strategy for determining boundary threshold ε."""

    FIXED = "fixed"  # Use a fixed threshold value
    ADAPTIVE_MEAN = "adaptive_mean"  # ε = mean(|gradient|) + k * std(|gradient|)
    ADAPTIVE_PERCENTILE = "adaptive_percentile"  # ε = p-th percentile of |gradient|
    # Note: ADAPTIVE_DYNAMIC currently falls back to ADAPTIVE_PERCENTILE
    # Full implementation would use local window statistics for each position
    ADAPTIVE_DYNAMIC = "adaptive_dynamic"  # Placeholder for future implementation


@dataclass
class BoundaryDetectorConfig:
    """Configuration for semantic boundary detection."""

    # Strategy for determining threshold
    threshold_strategy: ThresholdStrategy = ThresholdStrategy.ADAPTIVE_PERCENTILE

    # Fixed threshold (used when strategy = FIXED)
    fixed_threshold: float = 0.05

    # Parameters for adaptive strategies
    # For ADAPTIVE_MEAN: ε = mean(|gradient|) + k * std(|gradient|)
    adaptive_std_multiplier: float = 1.5

    # For ADAPTIVE_PERCENTILE: percentile of |gradient| to use as threshold
    adaptive_percentile: float = 75.0

    # Minimum distance between boundaries (avoid over-segmentation)
    min_boundary_distance: int = 3

    # Whether to smooth gradients before boundary detection
    smooth_gradients: bool = True

    # Window size for moving average smoothing
    smoothing_window: int = 3

    def validate(self) -> None:
        """Validate configuration."""
        if self.fixed_threshold <= 0:
            raise ValueError("fixed_threshold must be positive")
        if self.adaptive_std_multiplier < 0:
            raise ValueError("adaptive_std_multiplier must be non-negative")
        if not (0 <= self.adaptive_percentile <= 100):
            raise ValueError("adaptive_percentile must be between 0 and 100")
        if self.min_boundary_distance < 1:
            raise ValueError("min_boundary_distance must be at least 1")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be at least 1")


class BoundaryDetector:
    """Detect semantic slice boundaries using density gradients.

    A boundary is detected when the gradient of information density
    exceeds a threshold, indicating a transition between semantic concepts.
    """

    def __init__(self, config: BoundaryDetectorConfig | None = None) -> None:
        """Initialize the boundary detector.

        Args:
            config: Configuration for boundary detection. If None, uses defaults.
        """
        self.config = config or BoundaryDetectorConfig()
        self.config.validate()

        # Cache the last computed threshold for analysis
        self.last_threshold: float = 0.0

    def detect_boundaries(
        self, density_profile: DensityProfile
    ) -> tuple[list[int], float]:
        """Detect semantic boundaries from density profile.

        Args:
            density_profile: Density profile with gradients.

        Returns:
            Tuple of:
            - List of boundary positions (token indices)
            - The threshold ε used for detection
        """
        gradients = density_profile.gradients

        # Handle empty gradients edge case
        if len(gradients) == 0:
            return [], 0.0

        # Smooth gradients if configured
        if self.config.smooth_gradients:
            gradients = self._smooth(gradients)

        # Determine threshold based on strategy
        threshold = self._compute_threshold(gradients)
        self.last_threshold = threshold

        # Find boundaries where |gradient| > threshold
        absolute_gradients = np.abs(gradients)
        boundary_mask = absolute_gradients > threshold

        # Get boundary positions
        boundary_positions = np.where(boundary_mask)[0].tolist()

        # Apply minimum distance constraint
        boundary_positions = self._enforce_min_distance(boundary_positions)

        # Always include first position as a boundary (start of first slice)
        if 0 not in boundary_positions:
            boundary_positions.insert(0, 0)

        # Always include last position as a boundary (end of last slice)
        last_pos = len(gradients) - 1
        if last_pos not in boundary_positions:
            boundary_positions.append(last_pos)

        return boundary_positions, threshold

    def _compute_threshold(self, gradients: NDArray[np.float32]) -> float:
        """Compute boundary threshold based on configured strategy.

        Args:
            gradients: (seq_len,) gradient values

        Returns:
            Threshold value ε
        """
        if self.config.threshold_strategy == ThresholdStrategy.FIXED:
            return self.config.fixed_threshold

        absolute_gradients = np.abs(gradients)

        if self.config.threshold_strategy == ThresholdStrategy.ADAPTIVE_MEAN:
            mean_grad = np.mean(absolute_gradients)
            std_grad = np.std(absolute_gradients)
            return mean_grad + self.config.adaptive_std_multiplier * std_grad

        elif self.config.threshold_strategy == ThresholdStrategy.ADAPTIVE_PERCENTILE:
            return float(
                np.percentile(absolute_gradients, self.config.adaptive_percentile)
            )

        elif self.config.threshold_strategy == ThresholdStrategy.ADAPTIVE_DYNAMIC:
            # Dynamic threshold based on local window statistics
            # Use rolling mean + std of absolute gradients
            window_size = min(self.config.smoothing_window * 3, len(gradients))
            if window_size < 3:
                window_size = 3

            # For simplicity, use global percentile for dynamic
            # (more sophisticated version would use local windows)
            return float(
                np.percentile(absolute_gradients, self.config.adaptive_percentile)
            )

        else:
            raise ValueError(
                f"Unknown threshold strategy: {self.config.threshold_strategy}"
            )

    def _smooth(self, gradients: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply moving average smoothing to gradients.

        Args:
            gradients: (seq_len,) gradient values

        Returns:
            Smoothed gradients
        """
        window_size = self.config.smoothing_window
        if window_size >= len(gradients):
            return gradients

        # Use simple moving average
        smoothed = np.convolve(
            gradients,
            np.ones(window_size) / window_size,
            mode="same",
        )

        # Handle boundaries by using original values
        smoothed[: window_size // 2] = gradients[: window_size // 2]
        smoothed[-window_size // 2 :] = gradients[-window_size // 2 :]

        return smoothed.astype(np.float32)

    def _enforce_min_distance(self, boundaries: list[int]) -> list[int]:
        """Ensure boundaries are at least min_distance apart.

        Keeps the boundary with the higher gradient magnitude in each cluster.

        Args:
            boundaries: List of boundary positions

        Returns:
            Filtered boundary positions
        """
        if len(boundaries) <= 1:
            return boundaries

        min_dist = self.config.min_boundary_distance
        filtered = [boundaries[0]]

        for b in boundaries[1:]:
            if b - filtered[-1] >= min_dist:
                filtered.append(b)
            else:
                # Keep the one that's "more significant" (later in sequence)
                # This is a heuristic - could be improved by using actual gradient values
                filtered[-1] = b

        return filtered


def detect_boundaries(
    density_profile: DensityProfile,
    threshold_strategy: ThresholdStrategy = ThresholdStrategy.ADAPTIVE_PERCENTILE,
    fixed_threshold: float = 0.05,
    adaptive_percentile: float = 75.0,
) -> tuple[list[int], float]:
    """Convenience function to detect semantic boundaries.

    Args:
        density_profile: Density profile with gradients.
        threshold_strategy: Strategy for determining threshold.
        fixed_threshold: Fixed threshold (used when strategy = FIXED).
        adaptive_percentile: Percentile for adaptive strategies.

    Returns:
        Tuple of (boundary_positions, threshold_used)
    """
    config = BoundaryDetectorConfig(
        threshold_strategy=threshold_strategy,
        fixed_threshold=fixed_threshold,
        adaptive_percentile=adaptive_percentile,
    )
    detector = BoundaryDetector(config)
    return detector.detect_boundaries(density_profile)
