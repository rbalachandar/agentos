# AgentOS Documentation

This directory contains explanations for each component of the AgentOS system.

## Overview

| Document | Description |
|----------|-------------|
| [Comparison](comparison.md) | AgentOS vs Traditional systems - unbiased analysis of advantages, disadvantages, and when to use each |

## Core Components

| Component | File | Description |
|-----------|------|-------------|
| Reasoning Kernel | [reasoning-kernel.md](reasoning-kernel.md) | Semantic slicing - How the system learns "reading comprehension" by breaking text into idea chunks |
| Memory Hierarchy | [memory-hierarchy.md](memory-hierarchy.md) | Cognitive memory management - L1/L2/L3 tiers with demand paging |
| Scheduler & I/O | [scheduler-io.md](scheduler-io.md) | Cognitive scheduling - Multi-tasking with interrupts and tools |
| Multi-Agent Sync | [multi-agent-sync.md](multi-agent-sync.md) | Synchronization - How multiple agents collaborate without conflicts |
| Evaluation Metrics | [evaluation-metrics.md](evaluation-metrics.md) | Performance measurement - Metrics for cognitive systems |
| Full Integration | [integration.md](integration.md) | Complete system - All components working together |

## Quick Reference

### Reasoning Kernel
- **Core Concept**: Breaking text into semantic "idea chunks" (σ₁, σ₂, σ₃...)
- **Mechanism**: Attention Matrix → Information Density → Semantic Slices
- **Innovation**: Treats context as addressable semantic space

### Memory Hierarchy
- **L1 Cache**: Fast, tiny active attention window (~500 tokens)
- **L2 RAM**: Slower, bigger deep context (~2000 tokens)
- **L3 Storage**: Slowest, unlimited knowledge base
- **S-MMU**: Moves slices based on importance scores

### Scheduler & I/O
- **RCB**: Reasoning Control Block tracks thread state
- **Cognitive Scheduler**: Decides which agent runs next
- **Interrupt Cycle**: Handles tool calls without losing state
- **I/O Registry**: Manages external tools and peripherals

### Multi-Agent Synchronization
- **Drift Tracking**: Measures when agents diverge semantically
- **Sync Pulses**: Events that bring agents back into alignment
- **Conflict Resolution**: Handles disagreements between agents
- **Perception Alignment**: Finds optimal timing for synchronization
- **Distributed Memory**: Version vectors for state consistency

### Evaluation & Metrics
- **Cognitive Latency (L꜀)**: Time to handle interruptions
- **Utilization Efficiency (η)**: How well memory is used
- **Sync Stability Index (Γ)**: How well agents stay aligned
- **Spatial Decay**: How information fades over distance
- **Collapse Point**: When the system stops scaling linearly
- **Visualizations**: Heatmaps, charts, dashboards

### Full Integration
- **AgentOS**: Main orchestrator class
- **Agent**: Individual reasoning agent with role
- **Collaboration**: Multi-agent task execution
- **Synthesis**: Semantic clustering for final output

## Reading Order

For a complete understanding, read in order:

1. **[Reasoning Kernel](reasoning-kernel.md)** - Foundation of semantic understanding
2. **[Memory Hierarchy](memory-hierarchy.md)** - Managing semantic chunks
3. **[Scheduler & I/O](scheduler-io.md)** - Multi-tasking with tools
4. **[Multi-Agent Sync](multi-agent-sync.md)** - Agent collaboration
5. **[Evaluation Metrics](evaluation-metrics.md)** - Measuring performance
6. **[Integration](integration.md)** - Complete working system

Each component builds on the previous one - like layers of an operating system!
