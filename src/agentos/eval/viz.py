"""
Visualization Tools for AgentOS Evaluation.

Creates visualizations for:
- Attention heatmaps (Figure 3.2 from paper)
- Drift over time (Figure 4.1 from paper)
- Radar charts (Figure 5.1 from paper)
- Collapse point analysis (Figure 5.2 from paper)
- Metric dashboards
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


class AttentionHeatmap:
    """Create attention heatmap visualizations (Figure 3.2)."""

    def __init__(self, figsize: tuple[int, int] = (12, 8)):
        """Initialize heatmap creator.

        Args:
            figsize: Figure size (width, height)
        """
        self.figsize = figsize
        self.colormap = "viridis"

    def plot_attention_matrix(
        self,
        attention_weights: npt.NDArray[np.float32],
        tokens: list[str] | None = None,
        title: str = "Attention Matrix",
        save_path: str | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot attention matrix heatmap.

        Args:
            attention_weights: Attention matrix of shape (seq_len, seq_len)
            tokens: Token labels for axes
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot heatmap
        im = ax.imshow(attention_weights, cmap=self.colormap, aspect="auto")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention Weight", rotation=270, labelpad=20)

        # Set ticks
        seq_len = attention_weights.shape[0]
        if tokens:
            # Show subset of tokens if too many
            step = max(1, seq_len // 20)
            ax.set_xticks(range(0, seq_len, step))
            ax.set_yticks(range(0, seq_len, step))
            ax.set_xticklabels([tokens[i] for i in range(0, seq_len, step)], rotation=90, fontsize=8)
            ax.set_yticklabels([tokens[i] for i in range(0, seq_len, step)], fontsize=8)
        else:
            ax.set_xticks(range(0, seq_len, max(1, seq_len // 10)))
            ax.set_yticks(range(0, seq_len, max(1, seq_len // 10)))

        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def plot_multi_layer_attention(
        self,
        attention_weights: list[npt.NDArray[np.float32]],
        layer_names: list[str] | None = None,
        tokens: list[str] | None = None,
        save_path: str | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot attention matrices for multiple layers in a grid.

        Args:
            attention_weights: List of attention matrices, one per layer
            layer_names: Names for each layer
            tokens: Token labels
            save_path: Path to save figure
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        n_layers = len(attention_weights)
        n_cols = min(4, n_layers)
        n_rows = math.ceil(n_layers / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if n_layers == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]

        for i, (attn, ax) in enumerate(zip(attention_weights, axes.flat)):
            if i >= n_layers:
                ax.axis("off")
                continue

            im = ax.imshow(attn, cmap=self.colormap, aspect="auto")
            ax.set_title(layer_names[i] if layer_names else f"Layer {i + 1}", fontsize=10)

            # Set ticks (simplified)
            seq_len = attn.shape[0]
            step = max(1, seq_len // 10)
            ax.set_xticks(range(0, seq_len, step))
            ax.set_yticks(range(0, seq_len, step))

        # Remove extra subplots
        for i in range(n_layers, len(axes.flat)):
            axes.flat[i].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig


class DriftVisualization:
    """Create drift over time visualizations (Figure 4.1)."""

    def __init__(self, figsize: tuple[int, int] = (12, 6)):
        """Initialize drift visualizer.

        Args:
            figsize: Figure size (width, height)
        """
        self.figsize = figsize

    def plot_drift_over_time(
        self,
        timestamps: list[float],
        drift_values: dict[str, list[float]],  # agent_id -> list of drift values
        threshold: float = 1.0,
        sync_times: list[float] | None = None,
        title: str = "Cognitive Drift Over Time",
        save_path: str | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot drift over time for multiple agents (Figure 4.1).

        Args:
            timestamps: List of timestamps
            drift_values: Dict mapping agent_id to list of drift values
            threshold: Drift threshold line
            sync_times: Times when sync pulses occurred
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot drift for each agent
        for agent_id, drifts in drift_values.items():
            ax.plot(timestamps, drifts, label=agent_id, marker="o", markersize=3, linewidth=1.5)

        # Plot threshold line
        ax.axhline(y=threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold})")

        # Mark sync pulses
        if sync_times:
            for sync_time in sync_times:
                ax.axvline(x=sync_time, color="green", linestyle=":", linewidth=1.5, alpha=0.7)
            ax.text(sync_times[0], ax.get_ylim()[1] * 0.95, "Sync Pulse", color="green", fontsize=9)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Drift Magnitude")
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def plot_drift_reduction(
        self,
        drift_before: float,
        drift_after: float,
        agent_id: str = "System",
        save_path: str | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot before/after drift comparison.

        Args:
            drift_before: Drift before sync
            drift_after: Drift after sync
            agent_id: Agent identifier
            save_path: Path to save figure
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        categories = ["Before Sync", "After Sync"]
        values = [drift_before, drift_after]
        colors = ["#ff7f0e", "#2ca02c"]

        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor="black")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        # Add reduction percentage
        reduction = (drift_before - drift_after) / drift_before * 100 if drift_before > 0 else 0
        ax.set_title(f"Drift Reduction: {reduction:.1f}%", fontsize=14, fontweight="bold")
        ax.set_ylabel("Drift Magnitude")
        ax.set_ylim(0, max(drift_before, drift_after) * 1.2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig


class RadarChart:
    """Create radar chart visualizations (Figure 5.1)."""

    def __init__(self, figsize: tuple[int, int] = (10, 10)):
        """Initialize radar chart creator.

        Args:
            figsize: Figure size (width, height)
        """
        self.figsize = figsize

    def plot_metrics_radar(
        self,
        metrics: dict[str, list[float]],  # system_name -> list of metric values
        categories: list[str],
        title: str = "System Metrics Comparison",
        save_path: str | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot radar chart comparing systems (Figure 5.1).

        Args:
            metrics: Dict mapping system name to list of metric values
            categories: Category labels for each axis
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(projection="polar"))

        # Number of categories
        n_cats = len(categories)

        # Compute angle for each axis
        angles = [n / n_cats * 2 * math.pi for n in range(n_cats)]

        # Close the plot
        angles += angles[:1]

        # Plot each system
        colors = plt.cm.tab10(range(len(metrics)))
        for (system_name, values), color in zip(metrics.items(), colors):
            values_closed = values + values[:1]
            ax.plot(angles, values_closed, "o-", linewidth=2, label=system_name, color=color)
            ax.fill(angles, values_closed, alpha=0.15, color=color)

        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)

        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.7)

        # Add legend
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig


class CollapseAnalysis:
    """Create collapse point analysis visualizations (Figure 5.2)."""

    def __init__(self, figsize: tuple[int, int] = (12, 6)):
        """Initialize collapse analysis visualizer.

        Args:
            figsize: Figure size (width, height)
        """
        self.figsize = figsize

    def plot_collapse_point(
        self,
        agent_counts: list[int],
        stability_indices: list[float],
        collapse_threshold: float = 0.5,
        title: str = "Cognitive Collapse Point Analysis",
        save_path: str | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot collapse point analysis (Figure 5.2).

        Args:
            agent_counts: Numbers of agents tested
            stability_indices: Corresponding stability indices
            collapse_threshold: Stability threshold for collapse
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot stability curve
        ax.plot(agent_counts, stability_indices, "o-", linewidth=2.5, markersize=8, color="#1f77b4")

        # Plot collapse threshold line
        ax.axhline(y=collapse_threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({collapse_threshold})")

        # Find and mark collapse point
        collapse_point = None
        for agents, stability in zip(agent_counts, stability_indices):
            if stability < collapse_threshold:
                collapse_point = agents
                break

        if collapse_point:
            ax.axvline(x=collapse_point, color="orange", linestyle=":", linewidth=2, label=f"Collapse at {collapse_point} agents")
            ax.scatter([collapse_point], [stability_indices[agent_counts.index(collapse_point)]],
                      s=200, color="red", marker="x", linewidth=3, zorder=5)

        # Shade stable vs unstable regions
        if collapse_point:
            ax.axvspan(0, collapse_point, alpha=0.1, color="green", label="Stable Region")
            ax.axvspan(collapse_point, max(agent_counts), alpha=0.1, color="red", label="Unstable Region")

        ax.set_xlabel("Number of Agents", fontsize=12)
        ax.set_ylabel("Sync Stability Index (Γ)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

        # Add annotation for max stable agents
        if collapse_point:
            ax.annotate(
                f"Max stable: {collapse_point - 1} agents",
                xy=(collapse_point - 1, collapse_threshold + 0.1),
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3),
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def plot_latency_vs_agents(
        self,
        agent_counts: list[int],
        latencies: list[float],
        title: str = "Cognitive Latency vs Agent Count",
        save_path: str | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot latency scaling with agent count.

        Args:
            agent_counts: Numbers of agents tested
            latencies: Corresponding cognitive latencies
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(agent_counts, latencies, "o-", linewidth=2.5, markersize=8, color="#ff7f0e")

        # Fill area under curve
        ax.fill_between(agent_counts, 0, latencies, alpha=0.2, color="#ff7f0e")

        ax.set_xlabel("Number of Agents", fontsize=12)
        ax.set_ylabel("Cognitive Latency (ms)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add trend line annotation
        if len(agent_counts) > 1:
            z = np.polyfit(agent_counts, latencies, 2)
            p = np.poly1d(z)
            ax.plot(agent_counts, p(agent_counts), "--", color="gray", alpha=0.5, linewidth=1.5, label="Trend")
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig


class MetricsDashboard:
    """Create comprehensive dashboard showing all metrics."""

    def __init__(self, figsize: tuple[int, int] = (16, 10)):
        """Initialize dashboard creator.

        Args:
            figsize: Figure size (width, height)
        """
        self.figsize = figsize

    def plot_dashboard(
        self,
        latency_history: list[float],
        utilization_history: list[float],
        stability_history: list[float],
        timestamps: list[float],
        save_path: str | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot comprehensive metrics dashboard.

        Args:
            latency_history: Cognitive latency over time
            utilization_history: Utilization efficiency over time
            stability_history: Sync stability over time
            timestamps: Timestamps for all histories
            save_path: Path to save figure
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Cognitive Latency
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(timestamps, latency_history, "o-", color="#1f77b4", linewidth=2)
        ax1.set_title("Cognitive Latency (L꜀)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Latency (ms)")
        ax1.grid(True, alpha=0.3)
        ax1.text(0.02, 0.98, f"Mean: {np.mean(latency_history):.2f} ms",
                transform=ax1.transAxes, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        # 2. Utilization Efficiency
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(timestamps, [u * 100 for u in utilization_history], "o-", color="#2ca02c", linewidth=2)
        ax2.axhline(y=80, color="red", linestyle="--", alpha=0.5, label="Target (80%)")
        ax2.set_title("Contextual Utilization Efficiency (η)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Efficiency (%)")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="lower right")

        # 3. Sync Stability
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(timestamps, [s * 100 for s in stability_history], "o-", color="#ff7f0e", linewidth=2)
        ax3.axhline(y=80, color="green", linestyle="--", alpha=0.5, label="Stable Threshold (80%)")
        ax3.fill_between(timestamps, 0, [s * 100 for s in stability_history],
                        where=[s >= 0.8 for s in stability_history], alpha=0.2, color="green", label="Stable")
        ax3.fill_between(timestamps, 0, [s * 100 for s in stability_history],
                        where=[s < 0.8 for s in stability_history], alpha=0.2, color="red", label="Unstable")
        ax3.set_title("Sync Stability Index (Γ)", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Stability (%)")
        ax3.set_ylim(0, 105)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="best")

        fig.suptitle("AgentOS Metrics Dashboard", fontsize=16, fontweight="bold", y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig


# Convenience functions for common visualizations

def plot_attention_heatmap(
    attention: npt.NDArray[np.float32],
    tokens: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Quick function to create attention heatmap."""
    viz = AttentionHeatmap()
    return viz.plot_attention_matrix(attention, tokens=tokens, save_path=save_path, show=False)


def plot_drift_over_time(
    timestamps: list[float],
    drifts: dict[str, list[float]],
    threshold: float = 1.0,
    save_path: str | None = None,
) -> plt.Figure:
    """Quick function to create drift plot."""
    viz = DriftVisualization()
    return viz.plot_drift_over_time(timestamps, drifts, threshold=threshold, save_path=save_path, show=False)


def plot_collapse_analysis(
    agent_counts: list[int],
    stabilities: list[float],
    save_path: str | None = None,
) -> plt.Figure:
    """Quick function to create collapse analysis."""
    viz = CollapseAnalysis()
    return viz.plot_collapse_point(agent_counts, stabilities, save_path=save_path, show=False)
