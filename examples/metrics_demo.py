#!/usr/bin/env python3
"""
Phase 5 Demo: Evaluation & Metrics

Demonstrates:
- Cognitive Latency (L꜀) measurement
- Contextual Utilization Efficiency (η) calculation
- Sync Stability Index (Γ) computation
- Spatial Decay Rate analysis
- Cognitive Collapse Point detection
- Visualizations: heatmaps, drift charts, radar charts, collapse analysis

This demo uses synthetic data to demonstrate all evaluation metrics
without requiring expensive benchmark runs.
"""

from __future__ import annotations

import sys
import time as time_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for demo

import numpy as np

from agentos.eval import (
    AttentionHeatmap,
    CollapseAnalysis,
    DriftVisualization,
    MetricsCalculator,
    MetricsDashboard,
    RadarChart,
)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():
    """Demonstrate Phase 5 functionality."""
    print_section("AgentOS Phase 5: Evaluation & Metrics Demo")

    # ========================================================================
    # 1. Cognitive Latency Metrics
    # ========================================================================
    print("\n1. Cognitive Latency (L꜀)")
    print("-" * 70)
    print("Measures time from interrupt to stable state")

    calculator = MetricsCalculator()

    # Simulate interrupt handling with latency breakdown
    print("\nSimulating interrupt handling...")

    interrupt_time = time_module.time()
    time_module.sleep(0.01)  # Simulate dispatch
    dispatch_time = (time_module.time() - interrupt_time) * 1000

    time_module.sleep(0.02)  # Simulate handling
    handling_time = (time_module.time() - interrupt_time - 0.01) * 1000

    time_module.sleep(0.005)  # Simulate recovery
    stable_time = time_module.time()
    recovery_time = (stable_time - interrupt_time - 0.01 - 0.02) * 1000

    latency = calculator.calculate_cognitive_latency(
        interrupt_time=interrupt_time,
        stable_time=stable_time,
        dispatch_time_ms=dispatch_time,
        handling_time_ms=handling_time,
        recovery_time_ms=recovery_time,
    )

    print(f"\nCognitive Latency Breakdown:")
    print(f"  Dispatch:  {latency.dispatch_time_ms:.2f} ms ({latency.dispatch_overhead_pct:.1f}%)")
    print(f"  Handling:  {latency.handling_time_ms:.2f} ms ({latency.handling_overhead_pct:.1f}%)")
    print(f"  Recovery:  {latency.recovery_time_ms:.2f} ms ({latency.recovery_overhead_pct:.1f}%)")
    print(f"  ─────────────────────────────────")
    print(f"  Total L꜀:  {latency.latency_ms:.2f} ms")

    # ========================================================================
    # 2. Contextual Utilization Efficiency
    # ========================================================================
    print_section("2. Contextual Utilization Efficiency (η)")

    print("\nFormula (4): η = |𝒞_active| / |𝒞_max|")
    print("Measures how efficiently the system uses available context window")

    # Simulate memory tier usage
    utilization = calculator.calculate_utilization_efficiency(
        active_context_tokens=6500,
        max_context_tokens=8192,
        total_slices=25,
        utilized_slices=18,
        l1_tokens=2000,
        l2_tokens=3500,
        l3_tokens=1000,
    )

    print(f"\nContext Window Utilization:")
    print(f"  Active tokens:     {utilization.active_context_tokens:,} / {utilization.max_context_tokens:,}")
    print(f"  Efficiency (η):    {utilization.utilization_efficiency:.3f} ({utilization.utilization_pct:.1f}%)")
    print(f"  Headroom:          {utilization.headroom_tokens:,} tokens")
    print()
    print(f"  Slice Efficiency:  {utilization.slice_efficiency:.3f} ({utilization.utilized_slices}/{utilization.total_slices} slices)")
    print()
    print(f"  Tier Distribution:")
    print(f"    L1 (Fast):   {utilization.tier_distribution['l1']:.1%}")
    print(f"    L2 (Medium): {utilization.tier_distribution['l2']:.1%}")
    print(f"    L3 (Slow):   {utilization.tier_distribution['l3']:.1%}")

    # ========================================================================
    # 3. Sync Stability Index
    # ========================================================================
    print_section("3. Sync Stability Index (Γ)")

    print("\nFormula (11): Γ(t) = 1 - (1/N) Σᵢ₌₁ᴺ ‖∇Φᵢ(t) - ∇S_global(t)‖ / ‖∇S_global(t)‖")
    print("Measures multi-agent coherence stability")

    stability = calculator.calculate_sync_stability(
        timestamp=time_module.time(),
        agent_count=5,
        drift_before_sync=1.2,
        drift_after_sync=0.15,
        global_gradient_norm=1.0,
        agent_drift_norms=[0.1, 0.15, 0.08, 0.12, 0.09],
        sync_pulse_count=3,
        time_since_last_sync=5.0,
    )

    print(f"\nStability Metrics:")
    print(f"  Stability Index (Γ):  {stability.stability_index:.3f} ({stability.stability_pct:.1f}%)")
    print(f"  Status:               {'STABLE ✓' if stability.is_stable else 'UNSTABLE ✗'}")
    print()
    print(f"  Drift Before Sync:    {stability.drift_before_sync:.3f}")
    print(f"  Drift After Sync:     {stability.drift_after_sync:.3f}")
    print(f"  Reduction:            {stability.drift_reduction_pct:.1f}%")
    print()
    print(f"  Drift Rate:           {stability.drift_rate_per_second:.4f}/second")

    # ========================================================================
    # 4. Spatial Decay Rate
    # ========================================================================
    print_section("4. Spatial Decay Rate")

    print("\nMeasures how information degrades over context distance")

    # Simulate semantic similarity at different distances
    distances = [0, 10, 20, 50, 100, 200, 500, 1000]
    # Simulated exponential decay
    similarities = [np.exp(-0.003 * d) + np.random.normal(0, 0.05) for d in distances]
    similarities = [max(0, min(1, s)) for s in similarities]  # Clamp to [0, 1]

    spatial = calculator.calculate_spatial_decay(distances=distances, similarities=similarities)

    print(f"\nSpatial Decay Analysis:")
    print(f"  Decay Rate (k):       {spatial.decay_rate:.6f}")
    print(f"  Half-Life Distance:   {spatial.half_life_distance:.1f} tokens")
    print()
    print(f"  Similarity at Distance:")
    for dist in [100, 500, 1000, 2000]:
        sim = spatial.retrieval_at_distance(dist)
        print(f"    {dist:4d} tokens:  {sim:.3f}")

    # ========================================================================
    # 5. Cognitive Collapse Point
    # ========================================================================
    print_section("5. Cognitive Collapse Point")

    print("\nIdentifies threshold where system degrades")

    # Simulate system behavior at different scales
    agent_counts = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
    # Simulated degradation curve
    stabilities = [1.0 / (1 + 0.05 * (n - 1) ** 1.5) for n in agent_counts]
    latencies = [50 + n * 15 + n ** 2 * 2 for n in agent_counts]  # Simulated latencies

    collapse = calculator.calculate_collapse_point(
        agent_counts=agent_counts,
        stability_indices=stabilities,
        cognitive_latencies=latencies,
        collapse_threshold=0.5,
    )

    print(f"\nCollapse Analysis:")
    print(f"  Collapse Threshold:   {collapse.collapse_threshold}")
    print(f"  Collapse Point:       {collapse.collapse_point} agents" if collapse.collapse_point else "  Collapse Point:       None (stable at tested scales)")
    print(f"  Max Stable Agents:    {collapse.max_stable_agents}")
    print(f"  Degradation Rate:     {collapse.degradation_rate:.4f} per agent")

    # ========================================================================
    # 6. Visualizations
    # ========================================================================
    print_section("6. Generating Visualizations")

    output_dir = Path("./data/phase5_viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving visualizations to: {output_dir}")

    # 6.1 Drift Over Time
    print("\n  [1/4] Creating drift over time chart...")
    drift_viz = DriftVisualization()
    timestamps = list(range(0, 50, 5))
    drift_values = {
        "Agent_1": [0.1 + i * 0.05 + np.random.normal(0, 0.02) for i in range(len(timestamps))],
        "Agent_2": [0.15 + i * 0.04 + np.random.normal(0, 0.02) for i in range(len(timestamps))],
        "Agent_3": [0.08 + i * 0.06 + np.random.normal(0, 0.02) for i in range(len(timestamps))],
    }
    drift_viz.plot_drift_over_time(
        timestamps=timestamps,
        drift_values=drift_values,
        threshold=1.0,
        sync_times=[20, 40],
        save_path=str(output_dir / "drift_over_time.png"),
        show=False,
    )
    print(f"    ✓ Saved: drift_over_time.png")

    # 6.2 Radar Chart
    print("  [2/4] Creating radar chart comparison...")
    radar_viz = RadarChart()
    metrics_comparison = {
        "AgentOS": [0.85, 0.78, 0.92, 0.88, 0.75],  # η, Γ, L꜀ (inverted), precision, recall
        "Baseline A": [0.65, 0.55, 0.60, 0.70, 0.68],
        "Baseline B": [0.72, 0.62, 0.58, 0.75, 0.71],
    }
    categories = ["Utilization", "Stability", "Speed", "Precision", "Recall"]
    radar_viz.plot_metrics_radar(
        metrics=metrics_comparison,
        categories=categories,
        save_path=str(output_dir / "radar_comparison.png"),
        show=False,
    )
    print(f"    ✓ Saved: radar_comparison.png")

    # 6.3 Collapse Analysis
    print("  [3/4] Creating collapse point analysis...")
    collapse_viz = CollapseAnalysis()
    collapse_viz.plot_collapse_point(
        agent_counts=agent_counts,
        stability_indices=stabilities,
        collapse_threshold=0.5,
        save_path=str(output_dir / "collapse_analysis.png"),
        show=False,
    )
    print(f"    ✓ Saved: collapse_analysis.png")

    # 6.4 Metrics Dashboard
    print("  [4/4] Creating metrics dashboard...")
    dashboard_viz = MetricsDashboard()
    dashboard_viz.plot_dashboard(
        latency_history=[45.2, 52.1, 48.7, 55.3, 61.8, 58.4, 62.1, 59.7],
        utilization_history=[0.72, 0.78, 0.81, 0.75, 0.83, 0.79, 0.85, 0.82],
        stability_history=[0.92, 0.88, 0.95, 0.85, 0.78, 0.82, 0.75, 0.80],
        timestamps=list(range(8)),
        save_path=str(output_dir / "metrics_dashboard.png"),
        show=False,
    )
    print(f"    ✓ Saved: metrics_dashboard.png")

    # 6.5 Attention Heatmap (synthetic)
    print("  [5/5] Creating attention heatmap...")
    heatmap_viz = AttentionHeatmap()
    # Generate synthetic attention matrix
    seq_len = 50
    attention = np.random.rand(seq_len, seq_len)
    # Make it more realistic (local attention pattern)
    for i in range(seq_len):
        for j in range(seq_len):
            attention[i, j] *= np.exp(-0.1 * abs(i - j))
    attention = attention / attention.sum(axis=1, keepdims=True)  # Normalize

    tokens = [f"token_{i}" for i in range(seq_len)]
    heatmap_viz.plot_attention_matrix(
        attention_weights=attention,
        tokens=tokens,
        title="Synthetic Attention Matrix",
        save_path=str(output_dir / "attention_heatmap.png"),
        show=False,
    )
    print(f"    ✓ Saved: attention_heatmap.png")

    # ========================================================================
    # 7. Statistics Summary
    # ========================================================================
    print_section("7. Statistics Summary")

    # Add some more measurements for statistics
    for _ in range(5):
        calculator.calculate_cognitive_latency(
            interrupt_time=time_module.time(),
            stable_time=time_module.time() + 0.05,
            dispatch_time_ms=5,
            handling_time_ms=20,
        )
        calculator.calculate_utilization_efficiency(
            active_context_tokens=6000 + np.random.randint(-500, 500),
            max_context_tokens=8192,
            total_slices=20 + np.random.randint(-5, 5),
            utilized_slices=15 + np.random.randint(-3, 3),
        )
        calculator.calculate_sync_stability(
            timestamp=time_module.time(),
            agent_count=5,
            drift_before_sync=0.8 + np.random.random() * 0.5,
            drift_after_sync=0.1 + np.random.random() * 0.1,
            global_gradient_norm=1.0,
            agent_drift_norms=np.random.rand(5) * 0.2,
            sync_pulse_count=1,
            time_since_last_sync=5.0,
        )

    stats = calculator.get_summary()

    print("\nCognitive Latency Statistics:")
    print(f"  Measurements: {stats['cognitive_latency']['count']}")
    print(f"  Mean:        {stats['cognitive_latency']['mean_ms']:.2f} ms")
    print(f"  Min/Max:     {stats['cognitive_latency']['min_ms']:.2f} / {stats['cognitive_latency']['max_ms']:.2f} ms")
    print(f"  Std Dev:     {stats['cognitive_latency']['std_ms']:.2f} ms")
    print()

    print("Utilization Efficiency Statistics:")
    print(f"  Measurements: {stats['utilization_efficiency']['count']}")
    print(f"  Mean:        {stats['utilization_efficiency']['mean_efficiency']:.3f}")
    print(f"  Min/Max:     {stats['utilization_efficiency']['min_efficiency']:.3f} / {stats['utilization_efficiency']['max_efficiency']:.3f}")
    print(f"  Current:     {stats['utilization_efficiency']['current_efficiency']:.3f}")
    print()

    print("Sync Stability Statistics:")
    print(f"  Measurements: {stats['sync_stability']['count']}")
    print(f"  Mean:        {stats['sync_stability']['mean_stability']:.3f}")
    print(f"  Min/Max:     {stats['sync_stability']['min_stability']:.3f} / {stats['sync_stability']['max_stability']:.3f}")
    print(f"  Current:     {stats['sync_stability']['current_stability']:.3f}")
    print(f"  Stable:      {stats['sync_stability']['stable_count']}/{stats['sync_stability']['count']} ({stats['sync_stability']['stable_pct']:.1f}%)")

    # ========================================================================
    # 8. Demo Complete
    # ========================================================================
    print_section("Demo Complete!")
    print()
    print("The Evaluation & Metrics system successfully:")
    print("  • Measured cognitive latency with breakdown")
    print("  • Calculated contextual utilization efficiency")
    print("  • Computed sync stability index")
    print("  • Analyzed spatial decay rate")
    print("  • Identified cognitive collapse point")
    print("  • Generated visualization files:")
    print(f"    - {output_dir}/drift_over_time.png")
    print(f"    - {output_dir}/radar_comparison.png")
    print(f"    - {output_dir}/collapse_analysis.png")
    print(f"    - {output_dir}/metrics_dashboard.png")
    print(f"    - {output_dir}/attention_heatmap.png")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
