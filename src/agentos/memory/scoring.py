"""
Adaptive Importance Scoring for Cold Start Mitigation.

During cold start (first few turns), importance scores are unreliable
because we lack historical frequency/access data.

This module implements adaptive scoring that:
- Boosts recency weight during warmup period
- Applies warmup bonus to new slices
- Gradually transitions to normal scoring
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentos.memory.slicing.types import SemanticSlice


@dataclass
class AdaptiveScoringConfig:
    """Configuration for adaptive importance scoring."""

    # Warmup configuration
    warmup_turns: int = 5  # Number of turns before full scoring
    warmup_recency_boost: float = 2.0  # Multiplier for recency during warmup

    # Importance score adjustments
    new_slice_bonus: float = 0.2  # Bonus for slices created during warmup
    min_importance: float = 0.3  # Minimum importance during warmup

    # Transition
    transition_smooth: bool = True  # Gradually reduce boost
    transition_decay: float = 0.8  # Decay factor per turn

    def validate(self) -> None:
        """Validate configuration."""
        if self.warmup_turns < 0:
            raise ValueError("warmup_turns must be non-negative")
        if self.warmup_recency_boost < 1.0:
            raise ValueError("warmup_recency_boost must be >= 1.0")
        if not (0.0 <= self.new_slice_bonus <= 1.0):
            raise ValueError("new_slice_bonus must be in [0, 1]")


class AdaptiveImportanceScorer:
    """Adaptive importance scorer that adjusts behavior during warmup.

    During the warmup period (first N turns):
    - Recency weight is boosted to prioritize new information
    - New slices receive a bonus importance score
    - Minimum importance is higher to retain more context

    This helps the system build a useful knowledge base quickly before
    transitioning to normal long-term importance scoring.
    """

    def __init__(self, config: AdaptiveScoringConfig | None = None) -> None:
        """Initialize the adaptive scorer.

        Args:
            config: Configuration for adaptive scoring. If None, uses defaults.
        """
        self.config = config or AdaptiveScoringConfig()
        self.config.validate()

        # Track warmup progress
        self._turn_count = 0
        self._is_warmed_up = False

    @property
    def is_warmed_up(self) -> bool:
        """Whether the system has completed warmup."""
        return self._is_warmed_up

    @property
    def turn_count(self) -> int:
        """Current turn count."""
        return self._turn_count

    @property
    def warmup_progress(self) -> float:
        """Warmup progress from 0.0 to 1.0."""
        if self._is_warmed_up:
            return 1.0
        progress = self._turn_count / self.config.warmup_turns
        return min(1.0, progress)

    def advance_turn(self) -> None:
        """Advance to the next turn.

        Call this after each processing turn to update warmup state.
        """
        self._turn_count += 1
        if self._turn_count >= self.config.warmup_turns:
            self._is_warmed_up = True

    def reset(self) -> None:
        """Reset warmup state.

        Useful for starting a fresh session while keeping the same scorer.
        """
        self._turn_count = 0
        self._is_warmed_up = False

    def get_recency_multiplier(self) -> float:
        """Get the recency weight multiplier for current turn.

        During warmup, recency is boosted to help build initial context.
        After warmup, returns 1.0 (no boost).

        Returns:
            Multiplier for recency weight (>= 1.0 during warmup, 1.0 after)
        """
        if self._is_warmed_up:
            return 1.0

        if self.config.transition_smooth:
            # Gradually decay from boost to 1.0
            boost = self.config.warmup_recency_boost
            decay = self.config.transition_decay ** self._turn_count
            return max(1.0, boost * decay)
        else:
            # Full boost during warmup, 1.0 after
            return self.config.warmup_recency_boost

    def adjust_importance(
        self,
        base_importance: float,
        slice_data: "SemanticSlice | None" = None,
    ) -> float:
        """Adjust importance score based on warmup state.

        During warmup:
        - Applies new slice bonus for recently created slices
        - Enforces minimum importance threshold

        Args:
            base_importance: The base importance score from normal calculation
            slice_data: Optional slice data for bonus calculations

        Returns:
            Adjusted importance score in [0, 1]
        """
        if self._is_warmed_up:
            return base_importance

        adjusted = base_importance

        # Apply new slice bonus during warmup
        if slice_data is not None:
            # Check if slice was created recently (during warmup)
            slice_age = self._turn_count - slice_data.metadata.get("created_turn", 0)
            if slice_age <= 2:  # Created in last 2 turns
                adjusted += self.config.new_slice_bonus * (1.0 - slice_age / 2)

        # Enforce minimum importance during warmup
        adjusted = max(adjusted, self.config.min_importance)

        return min(1.0, adjusted)

    def get_warmup_stats(self) -> dict[str, object]:
        """Get statistics about warmup state.

        Returns:
            Dictionary with warmup statistics
        """
        return {
            "turn_count": self._turn_count,
            "is_warmed_up": self._is_warmed_up,
            "warmup_progress": self.warmup_progress,
            "recency_multiplier": self.get_recency_multiplier(),
            "warmup_turns": self.config.warmup_turns,
        }


def create_adaptive_scorer(
    warmup_turns: int = 5,
    recency_boost: float = 2.0,
) -> AdaptiveImportanceScorer:
    """Convenience function to create an adaptive scorer.

    Args:
        warmup_turns: Number of turns for warmup period
        recency_boost: Recency weight multiplier during warmup

    Returns:
        Configured AdaptiveImportanceScorer
    """
    config = AdaptiveScoringConfig(
        warmup_turns=warmup_turns,
        warmup_recency_boost=recency_boost,
    )
    return AdaptiveImportanceScorer(config)
