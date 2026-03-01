"""
Cognitive Drift Tracker.

Based on AgentOS paper Section 3.4.1, Formula (3):

    Δψᵢ(t) = ∫₀ᵗ ‖∇Φᵢ(σ,τ) - ∇S_global(τ)‖ dτ

Tracks how much each agent's internal state has diverged from the global
semantic state. When drift exceeds threshold, triggers sync pulse.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from agentos.sync.types import (
    AgentState,
    DriftMetrics,
    GlobalSemanticState,
    SyncTrigger,
)


@dataclass
class DriftTrackerConfig:
    """Configuration for drift tracking."""

    # Drift threshold (when to trigger sync)
    drift_threshold: float = 1.0

    # How often to check drift (seconds)
    check_interval_sec: float = 1.0

    # Drift rate threshold (how fast is drift accumulating)
    drift_rate_threshold: float = 0.1

    # Whether to use exponential moving average for smoothing
    use_ema: bool = True
    ema_alpha: float = 0.1

    def validate(self) -> None:
        """Validate configuration."""
        if self.drift_threshold <= 0:
            raise ValueError("drift_threshold must be positive")
        if self.check_interval_sec <= 0:
            raise ValueError("check_interval_sec must be positive")
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0, 1]")


class CognitiveDriftTracker:
    """Tracks cognitive drift for multiple agents.

    Each agent maintains its own internal semantic state. As they
    reason independently, their states diverge from the global state.
    The drift tracker monitors this and triggers syncs when needed.
    """

    def __init__(self, config: DriftTrackerConfig | None = None) -> None:
        """Initialize the drift tracker.

        Args:
            config: Configuration for drift tracking
        """
        self.config = config or DriftTrackerConfig()
        self.config.validate()

        # Track drift per agent
        self._agent_metrics: dict[str, DriftMetrics] = {}

        # Global state reference
        self._global_state: GlobalSemanticState | None = None

        # Tracking history
        self._last_check_time = time.time()
        self._sync_count = 0

    def register_agent(
        self,
        agent_id: str,
        initial_gradient: NDArray[np.float32] | None = None,
    ) -> DriftMetrics:
        """Register a new agent for drift tracking.

        Args:
            agent_id: Agent identifier
            initial_gradient: Initial semantic gradient

        Returns:
            DriftMetrics for the new agent
        """
        metrics = DriftMetrics(
            agent_id=agent_id,
            current_drift=0.0,
            drift_rate=0.0,
            gradient_norm=0.0,
            drift_threshold=self.config.drift_threshold,
        )

        self._agent_metrics[agent_id] = metrics
        return metrics

    def update_drift(
        self,
        agent_id: str,
        agent_gradient: NDArray[np.float32] | None,
        global_gradient: NDArray[np.float32] | None,
    ) -> float:
        """Update drift for an agent.

        Formula (3): ‖∇Φᵢ(σ,τ) - ∇S_global(τ)‖

        Args:
            agent_id: Agent identifier
            agent_gradient: Agent's current semantic gradient
            global_gradient: Global semantic gradient

        Returns:
            Current drift magnitude
        """
        metrics = self._agent_metrics.get(agent_id)
        if not metrics:
            # Auto-register agent
            metrics = self.register_agent(agent_id)

        # Calculate gradient norm
        if agent_gradient is not None and global_gradient is not None:
            # ‖∇Φᵢ - ∇S_global‖
            diff = agent_gradient - global_gradient
            gradient_norm = float(np.linalg.norm(diff))
        else:
            gradient_norm = 0.0

        # Update drift metrics
        if self.config.use_ema:
            # Exponential moving average
            old_drift = metrics.current_drift
            metrics.gradient_norm = (
                self.config.ema_alpha * gradient_norm
                + (1 - self.config.ema_alpha) * metrics.gradient_norm
            )
            metrics.current_drift = (
                self.config.ema_alpha * metrics.gradient_norm
                + (1 - self.config.ema_alpha) * old_drift
            )
        else:
            metrics.gradient_norm = gradient_norm
            metrics.current_drift += gradient_norm * self.config.check_interval_sec

        # Calculate drift rate
        if len(metrics.drift_history) > 0:
            prev_time, prev_drift = metrics.drift_history[-1]
            time_delta = (datetime.now() - prev_time).total_seconds()
            if time_delta > 0:
                metrics.drift_rate = (metrics.current_drift - prev_drift) / time_delta

        # Record history
        metrics.drift_history.append((datetime.now(), metrics.current_drift))

        return metrics.current_drift

    def get_metrics(self, agent_id: str) -> DriftMetrics | None:
        """Get drift metrics for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            DriftMetrics if found
        """
        return self._agent_metrics.get(agent_id)

    def get_all_metrics(self) -> dict[str, DriftMetrics]:
        """Get drift metrics for all agents.

        Returns:
            Map of agent_id -> DriftMetrics
        """
        return self._agent_metrics.copy()

    def check_sync_triggers(self) -> list[SyncTrigger]:
        """Check if any agents need syncing.

        Returns:
            List of sync triggers that should be fired
        """
        triggers = []

        # Check each agent for critical drift
        for agent_id, metrics in self._agent_metrics.items():
            if metrics.is_critical:
                triggers.append(SyncTrigger.DRIFT_THRESHOLD)

        return triggers

    def get_average_drift(self) -> float:
        """Get average drift across all agents.

        Returns:
            Average drift magnitude
        """
        if not self._agent_metrics:
            return 0.0

        total_drift = sum(m.current_drift for m in self._agent_metrics.values())
        return total_drift / len(self._agent_metrics)

    def get_max_drift(self) -> float:
        """Get maximum drift across all agents.

        Returns:
            Maximum drift magnitude
        """
        if not self._agent_metrics:
            return 0.0

        return max(m.current_drift for m in self._agent_metrics.values())

    def reset_drift(self, agent_id: str) -> bool:
        """Reset drift for an agent (after sync).

        Args:
            agent_id: Agent to reset

        Returns:
            True if successful
        """
        metrics = self._agent_metrics.get(agent_id)
        if not metrics:
            return False

        metrics.current_drift = 0.0
        metrics.drift_rate = 0.0
        metrics.drift_history.clear()

        return True

    def reset_all_drift(self) -> None:
        """Reset drift for all agents (after global sync)."""
        for metrics in self._agent_metrics.values():
            metrics.current_drift = 0.0
            metrics.drift_rate = 0.0
            metrics.drift_history.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get drift tracker statistics.

        Returns:
            Dictionary with statistics
        """
        drifts = [m.current_drift for m in self._agent_metrics.values()]

        return {
            "total_agents": len(self._agent_metrics),
            "average_drift": np.mean(drifts) if drifts else 0.0,
            "max_drift": np.max(drifts) if drifts else 0.0,
            "critical_drift_count": sum(1 for m in self._agent_metrics.values() if m.is_critical),
            "total_syncs": self._sync_count,
        }
