"""
Evaluation Metrics for AgentOS.

Based on AgentOS paper Section 5:

Metrics for evaluating cognitive architecture performance:
- Cognitive Latency (L꜀): Time from interrupt to stable state
- Contextual Utilization Efficiency (η): Formula (4) - context window utilization
- Sync Stability Index (Γ): Formula (11) - multi-agent coherence stability
- Spatial Decay Rate: Information loss over context distance
- Cognitive Collapse Point: Threshold where system degrades
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class CognitiveLatencyMetrics:
    """Metrics for cognitive latency (L꜀).

    L꜀ measures time from interrupt to stable state.
    """

    interrupt_time: float  # When interrupt occurred
    stable_time: float  # When system reached stable state
    latency_ms: float  # L꜀ in milliseconds

    # Breakdown of latency
    dispatch_time_ms: float  # Time to dispatch interrupt
    handling_time_ms: float  # Time to handle interrupt
    recovery_time_ms: float  # Time to recover stable state

    @property
    def total_latency_ms(self) -> float:
        """Total latency in milliseconds."""
        return self.latency_ms

    @property
    def dispatch_overhead_pct(self) -> float:
        """Dispatch time as percentage of total."""
        return (self.dispatch_time_ms / self.latency_ms * 100) if self.latency_ms > 0 else 0

    @property
    def handling_overhead_pct(self) -> float:
        """Handling time as percentage of total."""
        return (self.handling_time_ms / self.latency_ms * 100) if self.latency_ms > 0 else 0

    @property
    def recovery_overhead_pct(self) -> float:
        """Recovery time as percentage of total."""
        return (self.recovery_time_ms / self.latency_ms * 100) if self.latency_ms > 0 else 0


@dataclass
class ContextualUtilizationMetrics:
    """Metrics for Contextual Utilization Efficiency (η).

    Formula (4) from paper:
    η = |𝒞_active| / |𝒞_max|

    Measures how efficiently the system uses available context window.
    """

    active_context_tokens: int  # |𝒞_active| - tokens actively used
    max_context_tokens: int  # |𝒞_max| - maximum available tokens
    total_slices: int  # Number of semantic slices in context
    utilized_slices: int  # Number of slices actively contributing

    # Per-tier breakdown
    l1_tokens: int = 0
    l2_tokens: int = 0
    l3_tokens: int = 0

    @property
    def utilization_efficiency(self) -> float:
        """η = |𝒞_active| / |𝒞_max|"""
        if self.max_context_tokens == 0:
            return 0.0
        return self.active_context_tokens / self.max_context_tokens

    @property
    def utilization_pct(self) -> float:
        """Utilization as percentage."""
        return self.utilization_efficiency * 100

    @property
    def slice_efficiency(self) -> float:
        """What fraction of slices are actively used."""
        if self.total_slices == 0:
            return 0.0
        return self.utilized_slices / self.total_slices

    @property
    def headroom_tokens(self) -> int:
        """How many more tokens can be added."""
        return max(0, self.max_context_tokens - self.active_context_tokens)

    @property
    def tier_distribution(self) -> dict[str, float]:
        """Percentage of tokens in each tier."""
        total = self.l1_tokens + self.l2_tokens + self.l3_tokens
        if total == 0:
            return {"l1": 0.0, "l2": 0.0, "l3": 0.0}
        return {
            "l1": self.l1_tokens / total,
            "l2": self.l2_tokens / total,
            "l3": self.l3_tokens / total,
        }


@dataclass
class SyncStabilityMetrics:
    """Metrics for Sync Stability Index (Γ).

    Formula (11) from paper:
    Γ(t) = 1 - (1/N) Σᵢ₌₁ᴺ ‖∇Φᵢ(t) - ∇S_global(t)‖ / ‖∇S_global(t)‖

    Measures how stable multi-agent coherence is over time.
    """

    timestamp: float  # Unix timestamp
    agent_count: int  # N - number of agents

    # Drift measurements
    drift_before_sync: float  # Average drift before sync
    drift_after_sync: float  # Average drift after sync

    # Gradient norms
    global_gradient_norm: float  # ‖∇S_global(t)‖
    agent_drift_norms: list[float]  # Individual agent drifts

    # Sync pulse info
    sync_pulse_count: int  # Total sync pulses executed
    time_since_last_sync: float  # Time in seconds since last sync

    @property
    def stability_index(self) -> float:
        """Γ = 1 - average normalized drift"""
        if self.global_gradient_norm == 0 or self.agent_count == 0:
            return 1.0

        avg_drift = np.mean(self.agent_drift_norms)
        normalized_drift = avg_drift / self.global_gradient_norm
        return max(0.0, 1.0 - normalized_drift)

    @property
    def stability_pct(self) -> float:
        """Stability as percentage."""
        return self.stability_index * 100

    @property
    def drift_reduction_pct(self) -> float:
        """How much drift was reduced by sync."""
        if self.drift_before_sync == 0:
            return 100.0
        reduction = (self.drift_before_sync - self.drift_after_sync) / self.drift_before_sync
        return max(0.0, reduction * 100)

    @property
    def is_stable(self) -> bool:
        """Whether system is considered stable (Γ > 0.8)."""
        return self.stability_index >= 0.8

    @property
    def drift_rate_per_second(self) -> float:
        """Average drift accumulation rate."""
        if self.time_since_last_sync <= 0:
            return 0.0
        return self.drift_before_sync / self.time_since_last_sync


@dataclass
class SpatialDecayMetrics:
    """Metrics for Spatial Decay Rate.

    Measures how information degrades over context distance.
    """

    distances: list[int]  # Token distances from query
    similarities: list[float]  # Semantic similarities at each distance

    @property
    def decay_rate(self) -> float:
        """Exponential decay rate constant."""
        if len(self.distances) < 2:
            return 0.0

        # Fit exponential decay: similarity = exp(-k * distance)
        # log(similarity) = -k * distance
        # k = -log(similarity) / distance

        log_sims = []
        rates = []
        for dist, sim in zip(self.distances, self.similarities):
            if dist > 0 and sim > 0:
                log_sims.append(np.log(sim))
                rates.append(-log_sims[-1] / dist)

        return np.mean(rates) if rates else 0.0

    @property
    def half_life_distance(self) -> float:
        """Distance at which similarity drops to 50%."""
        if self.decay_rate == 0:
            return float("inf")
        return np.log(2) / self.decay_rate

    def retrieval_at_distance(self, distance: int) -> float:
        """Expected similarity at given distance."""
        return np.exp(-self.decay_rate * distance)


@dataclass
class CollapsePointMetrics:
    """Metrics for Cognitive Collapse Point.

    The threshold at which system performance degrades significantly.
    """

    agent_counts: list[int]  # Numbers of agents tested
    stability_indices: list[float]  # Corresponding stability indices
    cognitive_latencies: list[float]  # Corresponding latencies

    collapse_threshold: float = 0.5  # Stability threshold for collapse

    @property
    def collapse_point(self) -> int | None:
        """Number of agents at which collapse occurs."""
        for agents, stability in zip(self.agent_counts, self.stability_indices):
            if stability < self.collapse_threshold:
                return agents
        return None

    @property
    def max_stable_agents(self) -> int:
        """Maximum number of agents before collapse."""
        collapse = self.collapse_point
        if collapse is None:
            return self.agent_counts[-1] if self.agent_counts else 0
        return collapse - 1

    @property
    def degradation_rate(self) -> float:
        """Average stability loss per additional agent."""
        if len(self.agent_counts) < 2:
            return 0.0

        # Fit linear regression: stability = m * agents + b
        agents = np.array(self.agent_counts)
        stabilities = np.array(self.stability_indices)

        m = (len(agents) * np.sum(agents * stabilities) - np.sum(agents) * np.sum(stabilities))
        m /= (len(agents) * np.sum(agents ** 2) - (np.sum(agents)) ** 2)

        return abs(m)  # Return absolute degradation rate


@dataclass
class ReasoningKernelMetrics:
    """Metrics for ReasoningKernel (Phase 1) performance.

    Tracks how efficiently the RK processes text and produces semantic slices.
    """

    processing_time_ms: float  # Time to process input (forward pass + slicing)
    token_count: int  # Number of tokens processed
    slice_count: int  # Number of semantic slices produced

    # Per-token metrics
    tokens_per_second: float  # Processing throughput
    avg_tokens_per_slice: float  # Average slice size

    # State metrics
    attention_focus: float  # RK's attention focus (0-1, higher = more focused)
    semantic_stack_depth: int  # Current semantic stack depth
    kernel_state: str  # Current kernel state

    # Timestamp
    timestamp: float = 0.0  # When this metric was recorded

    @property
    def processing_efficiency(self) -> float:
        """Efficiency score combining speed and quality."""
        # Higher speed + higher attention focus = more efficient
        speed_score = min(1.0, self.tokens_per_second / 100.0)  # 100 tok/s = perfect
        quality_score = self.attention_focus
        return (speed_score + quality_score) / 2.0

    @property
    def slice_quality(self) -> float:
        """Quality score based on slice characteristics."""
        # Optimal slice size is around 20-50 tokens
        if self.slice_count == 0:
            return 0.0
        avg_size = self.token_count / self.slice_count
        if 20 <= avg_size <= 50:
            return 1.0
        elif avg_size < 20:
            return avg_size / 20.0
        else:
            return max(0.0, 1.0 - (avg_size - 50) / 50.0)


class MetricsCalculator:
    """Calculate all AgentOS evaluation metrics.

    This class provides methods to compute all metrics defined in
    the paper's evaluation section (Section 5).
    """

    def __init__(self) -> None:
        """Initialize the metrics calculator."""
        self._latency_history: list[CognitiveLatencyMetrics] = []
        self._utilization_history: list[ContextualUtilizationMetrics] = []
        self._stability_history: list[SyncStabilityMetrics] = []
        self._rk_history: list[ReasoningKernelMetrics] = []

    def calculate_cognitive_latency(
        self,
        interrupt_time: float,
        stable_time: float,
        dispatch_time_ms: float = 0,
        handling_time_ms: float = 0,
        recovery_time_ms: float | None = None,
    ) -> CognitiveLatencyMetrics:
        """Calculate cognitive latency (L꜀).

        Args:
            interrupt_time: When interrupt occurred (timestamp)
            stable_time: When system reached stable state (timestamp)
            dispatch_time_ms: Time to dispatch interrupt
            handling_time_ms: Time to handle interrupt
            recovery_time_ms: Time to recover (computed if None)

        Returns:
            CognitiveLatencyMetrics with all measurements
        """
        latency_ms = (stable_time - interrupt_time) * 1000

        if recovery_time_ms is None:
            recovery_time_ms = max(0, latency_ms - dispatch_time_ms - handling_time_ms)

        metrics = CognitiveLatencyMetrics(
            interrupt_time=interrupt_time,
            stable_time=stable_time,
            latency_ms=latency_ms,
            dispatch_time_ms=dispatch_time_ms,
            handling_time_ms=handling_time_ms,
            recovery_time_ms=recovery_time_ms,
        )

        self._latency_history.append(metrics)
        return metrics

    def calculate_utilization_efficiency(
        self,
        active_context_tokens: int,
        max_context_tokens: int,
        total_slices: int,
        utilized_slices: int,
        l1_tokens: int = 0,
        l2_tokens: int = 0,
        l3_tokens: int = 0,
    ) -> ContextualUtilizationMetrics:
        """Calculate contextual utilization efficiency (η).

        Formula (4): η = |𝒞_active| / |𝒞_max|

        Args:
            active_context_tokens: Tokens actively used
            max_context_tokens: Maximum available tokens
            total_slices: Total semantic slices in context
            utilized_slices: Slices actively contributing
            l1_tokens: Tokens in L1 cache
            l2_tokens: Tokens in L2 RAM
            l3_tokens: Tokens in L3 storage

        Returns:
            ContextualUtilizationMetrics
        """
        metrics = ContextualUtilizationMetrics(
            active_context_tokens=active_context_tokens,
            max_context_tokens=max_context_tokens,
            total_slices=total_slices,
            utilized_slices=utilized_slices,
            l1_tokens=l1_tokens,
            l2_tokens=l2_tokens,
            l3_tokens=l3_tokens,
        )

        self._utilization_history.append(metrics)
        return metrics

    def calculate_sync_stability(
        self,
        timestamp: float,
        agent_count: int,
        drift_before_sync: float,
        drift_after_sync: float,
        global_gradient_norm: float,
        agent_drift_norms: list[float],
        sync_pulse_count: int,
        time_since_last_sync: float,
    ) -> SyncStabilityMetrics:
        """Calculate sync stability index (Γ).

        Formula (11): Γ(t) = 1 - (1/N) Σᵢ₌₁ᴺ ‖∇Φᵢ(t) - ∇S_global(t)‖ / ‖∇S_global(t)‖

        Args:
            timestamp: Current time
            agent_count: Number of agents (N)
            drift_before_sync: Average drift before sync
            drift_after_sync: Average drift after sync
            global_gradient_norm: Norm of global gradient
            agent_drift_norms: Individual agent drifts
            sync_pulse_count: Total sync pulses executed
            time_since_last_sync: Seconds since last sync

        Returns:
            SyncStabilityMetrics
        """
        metrics = SyncStabilityMetrics(
            timestamp=timestamp,
            agent_count=agent_count,
            drift_before_sync=drift_before_sync,
            drift_after_sync=drift_after_sync,
            global_gradient_norm=global_gradient_norm,
            agent_drift_norms=agent_drift_norms,
            sync_pulse_count=sync_pulse_count,
            time_since_last_sync=time_since_last_sync,
        )

        self._stability_history.append(metrics)
        return metrics

    def calculate_spatial_decay(
        self,
        distances: list[int],
        similarities: list[float],
    ) -> SpatialDecayMetrics:
        """Calculate spatial decay rate.

        Args:
            distances: Token distances from query
            similarities: Semantic similarities at each distance

        Returns:
            SpatialDecayMetrics
        """
        return SpatialDecayMetrics(
            distances=distances,
            similarities=similarities,
        )

    def calculate_collapse_point(
        self,
        agent_counts: list[int],
        stability_indices: list[float],
        cognitive_latencies: list[float],
        collapse_threshold: float = 0.5,
    ) -> CollapsePointMetrics:
        """Calculate cognitive collapse point.

        Args:
            agent_counts: Numbers of agents tested
            stability_indices: Corresponding stability indices
            cognitive_latencies: Corresponding latencies
            collapse_threshold: Stability threshold for collapse

        Returns:
            CollapsePointMetrics
        """
        return CollapsePointMetrics(
            agent_counts=agent_counts,
            stability_indices=stability_indices,
            cognitive_latencies=cognitive_latencies,
            collapse_threshold=collapse_threshold,
        )

    def calculate_rk_performance(
        self,
        processing_time_ms: float,
        token_count: int,
        slice_count: int,
        attention_focus: float,
        semantic_stack_depth: int,
        kernel_state: str,
        timestamp: float | None = None,
    ) -> ReasoningKernelMetrics:
        """Calculate ReasoningKernel performance metrics.

        Tracks Phase 1 (ReasoningKernel & Semantic Slicing) performance.

        Args:
            processing_time_ms: Time to process input (forward + slicing)
            token_count: Number of tokens processed
            slice_count: Number of semantic slices produced
            attention_focus: RK's attention focus (0-1)
            semantic_stack_depth: Current semantic stack depth
            kernel_state: Current kernel state
            timestamp: When this metric was recorded

        Returns:
            ReasoningKernelMetrics
        """
        if timestamp is None:
            timestamp = time.time()

        # Calculate derived metrics
        tokens_per_second = (token_count * 1000.0 / processing_time_ms) if processing_time_ms > 0 else 0.0
        avg_tokens_per_slice = token_count / slice_count if slice_count > 0 else 0.0

        metrics = ReasoningKernelMetrics(
            processing_time_ms=processing_time_ms,
            token_count=token_count,
            slice_count=slice_count,
            tokens_per_second=tokens_per_second,
            avg_tokens_per_slice=avg_tokens_per_slice,
            attention_focus=attention_focus,
            semantic_stack_depth=semantic_stack_depth,
            kernel_state=kernel_state,
            timestamp=timestamp,
        )

        self._rk_history.append(metrics)
        return metrics

    def get_rk_statistics(self) -> dict[str, float]:
        """Get statistics on ReasoningKernel performance."""
        if not self._rk_history:
            return {
                "count": 0,
                "mean_processing_ms": 0.0,
                "mean_tokens_per_second": 0.0,
                "mean_attention_focus": 0.0,
                "mean_slice_count": 0.0,
            }

        return {
            "count": len(self._rk_history),
            "mean_processing_ms": float(np.mean([m.processing_time_ms for m in self._rk_history])),
            "min_processing_ms": float(np.min([m.processing_time_ms for m in self._rk_history])),
            "max_processing_ms": float(np.max([m.processing_time_ms for m in self._rk_history])),
            "mean_tokens_per_second": float(np.mean([m.tokens_per_second for m in self._rk_history])),
            "mean_attention_focus": float(np.mean([m.attention_focus for m in self._rk_history])),
            "mean_slice_count": float(np.mean([m.slice_count for m in self._rk_history])),
            "mean_stack_depth": float(np.mean([m.semantic_stack_depth for m in self._rk_history])),
            "mean_efficiency": float(np.mean([m.processing_efficiency for m in self._rk_history])),
        }

    def get_latency_statistics(self) -> dict[str, float]:
        """Get statistics on cognitive latency measurements."""
        if not self._latency_history:
            return {
                "count": 0,
                "mean_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "std_ms": 0.0,
            }

        latencies = [m.latency_ms for m in self._latency_history]

        return {
            "count": len(latencies),
            "mean_ms": float(np.mean(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "std_ms": float(np.std(latencies)),
        }

    def get_utilization_statistics(self) -> dict[str, float]:
        """Get statistics on utilization efficiency measurements."""
        if not self._utilization_history:
            return {
                "count": 0,
                "mean_efficiency": 0.0,
                "min_efficiency": 0.0,
                "max_efficiency": 0.0,
                "current_efficiency": 0.0,
            }

        efficiencies = [m.utilization_efficiency for m in self._utilization_history]

        return {
            "count": len(efficiencies),
            "mean_efficiency": float(np.mean(efficiencies)),
            "min_efficiency": float(np.min(efficiencies)),
            "max_efficiency": float(np.max(efficiencies)),
            "current_efficiency": efficiencies[-1],
        }

    def get_stability_statistics(self) -> dict[str, float]:
        """Get statistics on sync stability measurements."""
        if not self._stability_history:
            return {
                "count": 0,
                "mean_stability": 0.0,
                "min_stability": 0.0,
                "max_stability": 0.0,
                "current_stability": 0.0,
                "stable_count": 0,
            }

        stabilities = [m.stability_index for m in self._stability_history]
        stable_count = sum(1 for m in self._stability_history if m.is_stable)

        return {
            "count": len(stabilities),
            "mean_stability": float(np.mean(stabilities)),
            "min_stability": float(np.min(stabilities)),
            "max_stability": float(np.max(stabilities)),
            "current_stability": stabilities[-1],
            "stable_count": stable_count,
            "stable_pct": (stable_count / len(stabilities) * 100) if stabilities else 0.0,
        }

    def reset(self) -> None:
        """Reset all metric histories."""
        self._latency_history.clear()
        self._utilization_history.clear()
        self._stability_history.clear()

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all metrics."""
        return {
            "cognitive_latency": self.get_latency_statistics(),
            "utilization_efficiency": self.get_utilization_statistics(),
            "sync_stability": self.get_stability_statistics(),
            "rk_performance": self.get_rk_statistics(),
        }
