"""
Evaluation and Metrics Module.

This module provides metrics calculation and visualization tools
for evaluating AgentOS performance, as described in Section 5 of the paper.
"""

from __future__ import annotations

from agentos.eval.metrics import (
    CollapsePointMetrics,
    CognitiveLatencyMetrics,
    ContextualUtilizationMetrics,
    MetricsCalculator,
    ReasoningKernelMetrics,
    SpatialDecayMetrics,
    SyncStabilityMetrics,
)
from agentos.eval.viz import (
    AttentionHeatmap,
    CollapseAnalysis,
    DriftVisualization,
    MetricsDashboard,
    RadarChart,
    plot_attention_heatmap,
    plot_collapse_analysis,
    plot_drift_over_time,
)

__all__ = [
    # Metrics
    "MetricsCalculator",
    "CognitiveLatencyMetrics",
    "ContextualUtilizationMetrics",
    "SyncStabilityMetrics",
    "SpatialDecayMetrics",
    "CollapsePointMetrics",
    "ReasoningKernelMetrics",
    # Visualization
    "AttentionHeatmap",
    "DriftVisualization",
    "RadarChart",
    "CollapseAnalysis",
    "MetricsDashboard",
    # Convenience functions
    "plot_attention_heatmap",
    "plot_drift_over_time",
    "plot_collapse_analysis",
]
