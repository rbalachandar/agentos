# Phase 5: Evaluation & Metrics - Implementation Summary

## Overview

Phase 5 implements the evaluation metrics and visualization tools described in **Section 5** of the AgentOS paper. This phase provides the means to measure, analyze, and visualize system performance without requiring expensive benchmark infrastructure.

## Components Implemented

### 1. Metrics Module (`src/agentos/eval/metrics.py`)

**Purpose**: Calculate all evaluation metrics from the paper

**Key Metrics Implemented**:

#### Cognitive Latency (L꜀)
Measures time from interrupt to stable state.

```python
@dataclass
class CognitiveLatencyMetrics:
    interrupt_time: float
    stable_time: float
    latency_ms: float
    dispatch_time_ms: float
    handling_time_ms: float
    recovery_time_ms: float

    @property
    def dispatch_overhead_pct(self) -> float: ...
    @property
    def handling_overhead_pct(self) -> float: ...
    @property
    def recovery_overhead_pct(self) -> float: ...
```

**Usage**:
```python
calculator = MetricsCalculator()
latency = calculator.calculate_cognitive_latency(
    interrupt_time=interrupt_time,
    stable_time=stable_time,
    dispatch_time_ms=5.0,
    handling_time_ms=20.0,
)
print(f"L꜀ = {latency.latency_ms:.2f} ms")
```

#### Contextual Utilization Efficiency (η)
Formula (4) from paper: `η = |𝒞_active| / |𝒞_max|`

Measures how efficiently the system uses available context window.

```python
@dataclass
class ContextualUtilizationMetrics:
    active_context_tokens: int  # |𝒞_active|
    max_context_tokens: int     # |𝒞_max|
    total_slices: int
    utilized_slices: int
    l1_tokens: int = 0
    l2_tokens: int = 0
    l3_tokens: int = 0

    @property
    def utilization_efficiency(self) -> float: ...
    @property
    def slice_efficiency(self) -> float: ...
    @property
    def tier_distribution(self) -> dict[str, float]: ...
```

**Usage**:
```python
utilization = calculator.calculate_utilization_efficiency(
    active_context_tokens=6500,
    max_context_tokens=8192,
    total_slices=25,
    utilized_slices=18,
    l1_tokens=2000,
    l2_tokens=3500,
    l3_tokens=1000,
)
print(f"η = {utilization.utilization_efficiency:.3f} ({utilization.utilization_pct:.1f}%)")
```

#### Sync Stability Index (Γ)
Formula (11) from paper:
```
Γ(t) = 1 - (1/N) Σᵢ₌₁ᴺ ‖∇Φᵢ(t) - ∇S_global(t)‖ / ‖∇S_global(t)‖
```

Measures multi-agent coherence stability.

```python
@dataclass
class SyncStabilityMetrics:
    timestamp: float
    agent_count: int
    drift_before_sync: float
    drift_after_sync: float
    global_gradient_norm: float
    agent_drift_norms: list[float]
    sync_pulse_count: int
    time_since_last_sync: float

    @property
    def stability_index(self) -> float: ...
    @property
    def drift_reduction_pct(self) -> float: ...
    @property
    def is_stable(self) -> bool: ...
```

**Usage**:
```python
stability = calculator.calculate_sync_stability(
    timestamp=time.time(),
    agent_count=5,
    drift_before_sync=1.2,
    drift_after_sync=0.15,
    global_gradient_norm=1.0,
    agent_drift_norms=[0.1, 0.15, 0.08, 0.12, 0.09],
    sync_pulse_count=3,
    time_since_last_sync=5.0,
)
print(f"Γ = {stability.stability_index:.3f} ({stability.stability_pct:.1f}%)")
```

#### Spatial Decay Rate
Measures how information degrades over context distance.

```python
@dataclass
class SpatialDecayMetrics:
    distances: list[int]
    similarities: list[float]

    @property
    def decay_rate(self) -> float: ...        # Exponential decay constant
    @property
    def half_life_distance(self) -> float: ...  # Distance for 50% similarity

    def retrieval_at_distance(self, distance: int) -> float: ...
```

#### Collapse Point Metrics
Identifies threshold where system degrades.

```python
@dataclass
class CollapsePointMetrics:
    agent_counts: list[int]
    stability_indices: list[float]
    cognitive_latencies: list[float]
    collapse_threshold: float = 0.5

    @property
    def collapse_point(self) -> int | None: ...
    @property
    def max_stable_agents(self) -> int: ...
    @property
    def degradation_rate(self) -> float: ...
```

### 2. Visualization Module (`src/agentos/eval/viz.py`)

**Purpose**: Create all paper figures and evaluation charts

**Key Visualizers**:

#### AttentionHeatmap
Creates attention matrix visualizations (Figure 3.2).

```python
viz = AttentionHeatmap(figsize=(12, 8))

# Single layer
fig = viz.plot_attention_matrix(
    attention_weights=attention,  # (seq_len, seq_len)
    tokens=tokens,
    title="Attention Matrix",
    save_path="attention.png",
)

# Multiple layers
fig = viz.plot_multi_layer_attention(
    attention_weights=[layer1_attn, layer2_attn, ...],
    layer_names=["Layer 1", "Layer 2", ...],
    tokens=tokens,
    save_path="multi_layer_attention.png",
)
```

#### DriftVisualization
Creates drift over time charts (Figure 4.1).

```python
viz = DriftVisualization(figsize=(12, 6))

# Multi-agent drift
fig = viz.plot_drift_over_time(
    timestamps=[0, 5, 10, 15, 20],
    drift_values={
        "Agent_1": [0.1, 0.2, 0.4, 0.7, 1.1],
        "Agent_2": [0.15, 0.25, 0.35, 0.6, 0.9],
    },
    threshold=1.0,
    sync_times=[10, 20],
    save_path="drift.png",
)

# Before/after comparison
fig = viz.plot_drift_reduction(
    drift_before=1.2,
    drift_after=0.15,
    save_path="drift_reduction.png",
)
```

#### RadarChart
Creates system comparison radar charts (Figure 5.1).

```python
viz = RadarChart(figsize=(10, 10))

fig = viz.plot_metrics_radar(
    metrics={
        "AgentOS": [0.85, 0.78, 0.92, 0.88, 0.75],
        "Baseline A": [0.65, 0.55, 0.60, 0.70, 0.68],
        "Baseline B": [0.72, 0.62, 0.58, 0.75, 0.71],
    },
    categories=["Utilization", "Stability", "Speed", "Precision", "Recall"],
    save_path="radar.png",
)
```

#### CollapseAnalysis
Creates collapse point analysis charts (Figure 5.2).

```python
viz = CollapseAnalysis(figsize=(12, 6))

# Stability vs agent count
fig = viz.plot_collapse_point(
    agent_counts=[1, 2, 3, 4, 5, 6, 7, 8, 10],
    stability_indices=[1.0, 0.95, 0.88, 0.75, 0.60, 0.45, 0.35, 0.25, 0.15],
    collapse_threshold=0.5,
    save_path="collapse.png",
)

# Latency scaling
fig = viz.plot_latency_vs_agents(
    agent_counts=[1, 2, 3, 4, 5, 6, 7, 8],
    latencies=[50, 65, 85, 110, 140, 175, 215, 260],
    save_path="latency_scaling.png",
)
```

#### MetricsDashboard
Creates comprehensive metrics dashboard.

```python
viz = MetricsDashboard(figsize=(16, 10))

fig = viz.plot_dashboard(
    latency_history=[45.2, 52.1, 48.7, 55.3],
    utilization_history=[0.72, 0.78, 0.81, 0.75],
    stability_history=[0.92, 0.88, 0.95, 0.85],
    timestamps=[0, 10, 20, 30],
    save_path="dashboard.png",
)
```

### 3. Demo Script (`examples/phase5_demo.py`)

**Purpose**: Demonstrate all metrics and visualizations with synthetic data

**What it does**:
1. Simulates interrupt handling and measures cognitive latency
2. Calculates contextual utilization efficiency with tier breakdown
3. Computes sync stability index from drift data
4. Analyzes spatial decay rate
5. Identifies cognitive collapse point
6. Generates 5 visualization files:
   - `drift_over_time.png` - Multi-agent drift chart
   - `radar_comparison.png` - System comparison radar
   - `collapse_analysis.png` - Collapse point chart
   - `metrics_dashboard.png` - Full metrics dashboard
   - `attention_heatmap.png` - Synthetic attention matrix
7. Displays statistics summary

**Run**:
```bash
python examples/phase5_demo.py
```

## Design Decisions

1. **Synthetic Data for Demo**: Since we don't have access to expensive baselines (MemGPT, AIOS), the demo uses synthetic data to demonstrate all functionality.

2. **Non-Interactive Backend**: Demo uses `matplotlib.use('Agg')` to save figures without displaying them, making it suitable for headless environments.

3. **Modular Design**: Each metric type has its own dataclass with computed properties for easy access to derived values.

4. **Statistics Tracking**: `MetricsCalculator` maintains histories of all measurements for statistical analysis (mean, min, max, std).

5. **Convenience Functions**: Quick plotting functions for common visualizations without needing to instantiate visualizer classes.

## Formulas Implemented

### Contextual Utilization Efficiency (η)
**Formula (4)**:
```
η = |𝒞_active| / |𝒞_max|
```

Where:
- `|𝒞_active|` = tokens actively in use
- `|𝒞_max|` = maximum context window size

### Sync Stability Index (Γ)
**Formula (11)**:
```
Γ(t) = 1 - (1/N) Σᵢ₌₁ᴺ ‖∇Φᵢ(t) - ∇S_global(t)‖ / ‖∇S_global(t)‖
```

Where:
- `N` = number of agents
- `∇Φᵢ(t)` = agent i's semantic gradient
- `∇S_global(t)` = global semantic gradient
- `Γ ∈ [0, 1]`, higher is more stable

### Spatial Decay Rate
**Exponential decay model**:
```
similarity(distance) = exp(-k * distance)
```

Where:
- `k` = decay rate constant
- Half-life distance = ln(2) / k

## File Structure

```
src/agentos/eval/
├── __init__.py           # Module exports
├── metrics.py            # All metric calculations
└── viz.py                # All visualization tools

examples/
└── phase5_demo.py        # Demonstration script
```

## Integration Points

Phase 5 integrates with all previous phases:
- **Phase 1**: Use attention matrices from Reasoning Kernel for heatmaps
- **Phase 2**: Track L1/L2/L3 utilization for efficiency metrics
- **Phase 3**: Measure interrupt latency from scheduler
- **Phase 4**: Use drift tracker data for stability calculations

## Performance Characteristics

- **Metric Calculation**: O(1) for individual metrics, O(n) for statistics
- **Heatmap Generation**: O(seq_len²) for attention matrices
- **Drift Visualization**: O(timestamps × agents)
- **Dashboard**: O(total metrics) for all charts

## Testing

Run the demo to verify all components:
```bash
python examples/phase5_demo.py
```

Expected output:
- Console output showing all calculated metrics
- 5 PNG files saved to `data/phase5_viz/`

## What Was NOT Implemented

The following components from the original plan were intentionally skipped due to resource constraints:

1. **Baseline Comparisons** (5.2):
   - Requires installing and running MemGPT
   - Requires installing and running AIOS
   - Could be added in future if access to these systems is available

2. **Full Benchmark Tasks** (5.3):
   - Long-context QA benchmarks
   - Multi-agent coordination benchmarks
   - Tool-use chain benchmarks
   - These require significant GPU resources and time

3. **Paper Figure Reproduction**:
   - Actual data from paper not available
   - Synthetic data used instead for demonstration

## Future Enhancements

- Add support for real benchmark data when available
- Implement statistical significance testing
- Add more sophisticated decay models (non-exponential)
- Support for multi-run aggregation and error bars
- Interactive dashboards using plotly
