"""
Perception Alignment Protocol.

Based on AgentOS paper Section 3.4.4:

"Advantageous Timing Matching" - find optimal windows for synchronization
when agents have high-confidence states.

Key insights:
- Sync during uncertainty amplifies errors
- Sync during high-confidence states preserves coherence
- Filter noise from probabilistic inference
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ConfidenceWindow:
    """A time window with confidence metrics."""

    start_time: float  # Unix timestamp
    end_time: float  # Unix timestamp
    confidence_score: float  # 0-1, average confidence

    # Agent-specific confidences
    agent_confidences: dict[str, float]  # agent_id -> confidence

    @property
    def duration_ms(self) -> float:
        """Window duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000.0

    @property
    def is_valid(self) -> bool:
        """Whether this window is valid for syncing."""
        return self.confidence_score >= 0.7


@dataclass
class PerceptionAlignmentConfig:
    """Configuration for perception alignment."""

    # Minimum confidence threshold for sync window
    min_confidence: float = 0.7

    # Minimum window duration (milliseconds)
    min_window_duration_ms: float = 100.0

    # Maximum window duration (milliseconds)
    max_window_duration_ms: float = 5000.0

    # Noise filtering threshold
    noise_threshold: float = 0.3

    def validate(self) -> None:
        """Validate configuration."""
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be in [0, 1]")
        if self.min_window_duration_ms < 0:
            raise ValueError("min_window_duration_ms must be non-negative")
        if self.max_window_duration_ms < self.min_window_duration_ms:
            raise ValueError("max_window_duration_ms must be >= min_window_duration_ms")


class PerceptionAlignmentProtocol:
    """Perception Alignment Protocol.

    Finds optimal sync windows when agents have high-confidence states,
    avoiding sync during uncertainty which would amplify errors.
    """

    def __init__(self, config: PerceptionAlignmentConfig | None = None) -> None:
        """Initialize the perception alignment protocol.

        Args:
            config: Configuration for perception alignment
        """
        self.config = config or PerceptionAlignmentConfig()
        self.config.validate()

        # Confidence history
        self._confidence_history: dict[str, list[tuple[float, float]]] = {}

        # Detected windows
        self._windows: list[ConfidenceWindow] = []

    def update_confidence(
        self,
        agent_id: str,
        confidence: float,
        timestamp: float | None = None,
    ) -> None:
        """Update confidence history for an agent.

        Args:
            agent_id: Agent identifier
            confidence: Confidence score (0-1)
            timestamp: Unix timestamp (default: now)
        """
        if timestamp is None:
            timestamp = time.time()

        if agent_id not in self._confidence_history:
            self._confidence_history[agent_id] = []

        self._confidence_history[agent_id].append((timestamp, confidence))

        # Keep history bounded (last 1000 entries)
        if len(self._confidence_history[agent_id]) > 1000:
            self._confidence_history[agent_id] = self._confidence_history[agent_id][-1000:]

    def get_current_confidence(self, agent_id: str) -> float:
        """Get current confidence for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Current confidence score (0-1), or 0 if no history
        """
        history = self._confidence_history.get(agent_id)
        if not history:
            return 0.0

        # Return most recent confidence
        return history[-1][1]

    def find_sync_windows(
        self,
        start_time: float,
        end_time: float,
        agent_ids: list[str],
    ) -> list[ConfidenceWindow]:
        """Find optimal sync windows in a time range.

        A sync window is optimal when:
        - All agents have high confidence (> min_confidence)
        - Window duration is within acceptable range
        - Confidence is relatively stable (not fluctuating)

        Args:
            start_time: Start of time range (Unix timestamp)
            end_time: End of time range (Unix timestamp)
            agent_ids: Agents to include

        Returns:
            List of ConfidenceWindow suitable for syncing
        """
        windows = []

        if not agent_ids:
            return windows

        # Scan through time range
        window_start = None
        window_confidences: dict[str, list[tuple[float, float]]] = {}

        # Get confidence data for each agent
        for agent_id in agent_ids:
            history = self._confidence_history.get(agent_id, [])
            window_confidences[agent_id] = [
                (t, c) for t, c in history
                if start_time <= t <= end_time
            ]

        # Find contiguous high-confidence periods
        for i, (timestamp, _) in enumerate(window_confidences[agent_ids[0]]):
            # Check if all agents have high confidence at this time
            all_high_confidence = True
            min_confidence = 1.0

            for agent_id in agent_ids:
                conf = self.get_confidence_at_time(agent_id, timestamp)
                if conf is None:
                    all_high_confidence = False
                    break
                min_confidence = min(min_confidence, conf)
                if conf < self.config.min_confidence:
                    all_high_confidence = False
                    break

            if all_high_confidence:
                if window_start is None:
                    window_start = timestamp

                # Extend window
                window_end = timestamp

            else:
                # Low confidence, end current window if any
                if window_start is not None:
                    duration_ms = (window_end - window_start) * 1000
                    if duration_ms >= self.config.min_window_duration_ms:
                        windows.append(
                            ConfidenceWindow(
                                start_time=window_start,
                                end_time=window_end,
                                confidence_score=min_confidence,
                                agent_confidences={
                                    agent_id: self.get_confidence_at_time(agent_id, (window_start + window_end) / 2)
                                    for agent_id in agent_ids
                                },
                            )
                        )

                window_start = None

        # Handle unclosed window
        if window_start is not None:
            duration_ms = (end_time - window_start) * 1000
            if duration_ms >= self.config.min_window_duration_ms:
                windows.append(
                    ConfidenceWindow(
                        start_time=window_start,
                        end_time=end_time,
                        confidence_score=min_confidence,
                        agent_confidences={
                            agent_id: self.get_confidence_at_time(agent_id, (window_start + end_time) / 2)
                            for agent_id in agent_ids
                        },
                    )
                )

        # Filter by max duration
        windows = [
            w for w in windows
            if w.duration_ms <= self.config.max_window_duration_ms
        ]

        return windows

    def get_confidence_at_time(
        self, agent_id: str, timestamp: float
    ) -> float | None:
        """Get confidence for an agent at a specific time.

        Args:
            agent_id: Agent identifier
            timestamp: Unix timestamp

        Returns:
            Confidence score, or None if no data
        """
        history = self._confidence_history.get(agent_id, [])

        # Find closest entry
        closest = None
        min_diff = float("inf")

        for t, c in history:
            diff = abs(t - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest = c

        return closest

    def filter_noise(
        self,
        agent_id: str,
        confidences: list[float],
    ) -> list[float]:
        """Filter noise from confidence scores using moving average.

        Args:
            agent_id: Agent identifier
            confidences: Raw confidence scores

        Returns:
            Filtered confidence scores
        """
        window_size = 5

        if len(confidences) < window_size:
            return confidences.copy()

        filtered = []
        for i in range(len(confidences)):
            # Simple moving average
            start = max(0, i - window_size // 2)
            end = min(len(confidences), i + window_size // 2 + 1)
            window = confidences[start:end]
            filtered.append(sum(window) / len(window))

        return filtered

    def calculate_alignment_quality(
        self,
        window: ConfidenceWindow,
    ) -> float:
        """Calculate the quality of a sync window.

        Higher quality when:
        - All agents have similar confidence (stability)
        - Average confidence is high
        - Duration is appropriate (not too short/long)

        Args:
            window: Confidence window to evaluate

        Returns:
            Quality score (0-1)
        """
        if not window.agent_confidences:
            return 0.0

        # Average confidence
        avg_confidence = window.confidence_score

        # Confidence stability (how similar are agents' confidences)
        confs = list(window.agent_confidences.values())
        if len(confs) < 2:
            stability = 1.0
        else:
            std_conf = np.std(confs)
            stability = 1.0 - min(std_conf, 1.0)

        # Duration appropriateness (prefer mid-range durations)
        duration_score = 1.0
        if window.duration_ms < self.config.min_window_duration_ms:
            duration_score = window.duration_ms / self.config.min_window_duration_ms
        elif window.duration_ms > self.config.max_window_duration_ms:
            duration_score = max(
                0.0,
                1.0 - (window.duration_ms - self.config.max_window_duration_ms)
                / (self.config.max_window_duration_ms * 0.5)
            )

        # Combined score
        return (avg_confidence * 0.5 + stability * 0.3 + duration_score * 0.2)

    def get_best_sync_window(
        self,
        windows: list[ConfidenceWindow],
    ) -> ConfidenceWindow | None:
        """Get the best sync window from a list.

        Args:
            windows: List of confidence windows

        Returns:
            Best window, or None if empty
        """
        if not windows:
            return None

        # Score each window
        best_window = None
        best_score = -1.0

        for window in windows:
            score = self.calculate_alignment_quality(window)
            if score > best_score:
                best_score = score
                best_window = window

        return best_window

    def get_statistics(self) -> dict[str, Any]:
        """Get perception alignment statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_agents": len(self._confidence_history),
            "total_windows_detected": len(self._windows),
            "config_min_confidence": self.config.min_confidence,
        }
