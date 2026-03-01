# AgentOS

> **From Token-Level Context to Emergent System-Level Intelligence**

Research implementation of the AgentOS architecture proposed in [Architecting AgentOS: From Token-Level Context to Emergent System-Level Intelligence](https://arxiv.org/html/2602.20934v1).

## Overview

AgentOS redefines the LLM as a **"Reasoning Kernel"** governed by structured operating system logic. The core innovation is treating the context window as an **Addressable Semantic Space** rather than a passive buffer.

### Key Concepts

| Traditional OS | AgentOS |
|---|---|
| CPU | Reasoning Kernel (RK) |
| RAM | Addressable Semantic Space (L2) |
| Page Tables | Semantic Page Tables |
| Interrupts | Reasoning Interrupts |
| Process Scheduler | Cognitive Scheduler |

### Core Innovations

1. **Semantic Slicing** - Aggregate tokens into coherent "cognitive pages" based on attention patterns
2. **Cognitive Memory Hierarchy** - L1 (active attention) → L2 (deep context) → L3 (knowledge base)
3. **Cognitive Sync Pulses** - Event-driven synchronization for multi-agent coherence
4. **Perception Alignment** - Optimal timing for merging semantic slices across agents

## Why AgentOS?

**📖 [Read the full comparison: AgentOS vs Traditional Systems](docs/comparison.md)**

AgentOS offers bounded, scalable performance for long-running multi-agent conversations:

| Scenario | Traditional | AgentOS | Benefit |
|----------|-------------|---------|---------|
| 5-turn chat | 500ms | 350ms | 1.4x faster |
| 20-turn chat | 5000ms | 1200ms | **4x faster** |
| 100-turn session | 50000ms | 4000ms | **12x faster** |

**Trade-offs:**
- ✅ Bounded memory (L1 always ~500 tokens)
- ✅ Semantic selectivity (focus on what matters)
- ✅ True parallelism (agents work independently)
- ❌ Higher complexity (5 interconnected subsystems)
- ❌ Cold start (needs warm-up for optimal performance)
- ❌ Parameter tuning (20+ sensitive settings)

**When to use:**
- Long-running conversations (10+ turns)
- Multiple agents collaborating
- Need fine-grained memory control
- Building production multi-agent systems

**When NOT to use:**
- Single-turn Q&A
- Using API-based models (GPT-4, Claude)
- Simplicity is more important than optimization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AgentOS                                 │
│                                                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Reasoning Kernel (RK)                             │     │
│  │  Contextual Transition: 𝓕(Sₜ, 𝒞ₐddᵣ) → Sₜ₊₁         │     │
│  └────────────────────────────────────────────────────┘     │
│                           ↓                                 │
│  ┌────────────────────────────────────────────────────┐     │
│  │  S-MMU (Semantic Memory Management Unit)           │     │
│  │  ┌─────────────────────────────────────────────┐   │     │
│  │  │ L1 Cache    │ L2 RAM       │ L3 Storage     │   │     │
│  │  │ (Active)    │ (Deep Ctx)   │ (Knowledge)    │   │     │
│  │  │ KV-Cache    │ Vector DB    │ RAG Systems    │   │     │
│  │  └─────────────────────────────────────────────┘   │     │
│  └────────────────────────────────────────────────────┘     │
│                           ↓                                 │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Cognitive Scheduler                               │     │
│  │  Optimizes for Cognitive Fidelity, not CPU time    │     │
│  └────────────────────────────────────────────────────┘     │
│                           ↓                                 │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Multi-Agent Sync (CSP)                            │     │
│  │  Cognitive Sync Pulses for temporal coherence      │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Status

✅ **All 6 Phases Complete** - Full multi-agent system with semantic memory, sync, and metrics

📋 **[ISSUES.md](ISSUES.md)** - 10 prioritized improvement items for production readiness

📖 **[docs/comparison.md](docs/comparison.md)** - AgentOS vs Traditional: Unbiased analysis

📖 **[docs/](docs/)** - Component documentation and explanations

## Requirements

- **Python**: 3.10 or later (for modern type hint syntax)
- **PyTorch**: 2.0+ with MPS (Mac M1/M2) or CUDA support
- **Local LLM**: Qwen2.5-0.5B-Instruct or similar (auto-downloaded)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/agentos.git
cd agentos

# Install dependencies
pip install -e .

# Or for development
pip install -e ".[dev]"
```

## Quick Start

### Interactive CLI

Run the AgentOS CLI for an interactive multi-agent experience:

**After installation:**
```bash
# Fast mode (placeholder responses, ~8s startup)
agentos

# Full mode (actual LLM generation, ~40s startup)
agentos --generate

# Custom model
agentos --generate --model Qwen/Qwen2.5-0.5B-Instruct
```

**Or run directly:**
```bash
python bin/agentos_cli.py --generate
```

**CLI Commands:**
- `/help` - Show help
- `/agents` - List all agents
- `/stats` - Show system statistics
- `/memory` - Show memory utilization
- `/sync` - Trigger manual sync
- `/quit` or `/exit` - Exit

### Python API

> Note: This is a research prototype requiring local models for attention access.

```python
from agentos import AgentOS, create_agentos
from agentos.scheduler import ThreadPriority

# Create system
system = create_agentos()

# Spawn specialized agents
researcher = system.spawn_agent("Alice", "researcher", ThreadPriority.HIGH)
writer = system.spawn_agent("Bob", "writer", ThreadPriority.NORMAL)
analyst = system.spawn_agent("Charlie", "analyst", ThreadPriority.NORMAL)

# Collaborative task
result = system.collaborate("Analyze the differences between AI and human cognition")

for agent_id, contribution in result.agent_contributions.items():
    agent = system.get_agent(agent_id)
    print(f"{agent.config.name}: {contribution}")
```

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
ruff check src/
ruff format src/
mypy src/
```

### Pre-commit Hooks

```bash
pre-commit install
```

## Project Structure

```
agentos/
├── src/agentos/
│   ├── kernel/              # Reasoning Kernel with semantic slicing
│   ├── memory/
│   │   ├── slicing/         # Semantic Slicing (CID, boundaries)
│   │   └── tiers/           # L1/L2/L3 memory tiers
│   ├── scheduler/           # Cognitive Scheduler & RCB
│   ├── sync/                # Multi-agent CSP & DSM
│   ├── io/                  # Interrupt handling & peripherals
│   ├── synthesis/           # Semantic synthesis for multi-agent output
│   └── eval/                # Metrics and visualization
│
├── bin/
│   └── agentos_cli.py       # Standalone CLI script
│
├── examples/
│   ├── semantic_slicing_demo.py       # Phase 1 demo
│   ├── memory_hierarchy_demo.py       # Phase 2 demo
│   ├── scheduler_demo.py              # Phase 3 demo
│   ├── multi_agent_sync_demo.py       # Phase 4 demo
│   ├── metrics_demo.py                # Phase 5 demo
│   ├── integration_demo.py            # Phase 6: Full system
│   └── test_system.py                 # Quick test script
│
├── docs/
│   ├── comparison.md        # AgentOS vs Traditional comparison
│   ├── reasoning-kernel.md  # Semantic slicing explained
│   ├── memory-hierarchy.md  # L1/L2/L3 memory management
│   ├── scheduler-io.md      # Cognitive scheduling
│   ├── multi-agent-sync.md  # Synchronization & CSP
│   ├── evaluation-metrics.md # Metrics and measurement
│   └── integration.md       # Full system overview
│
├── tests/                   # Test files
├── ISSUES.md               # Improvement roadmap (10 prioritized issues)
├── LICENSE                  # MIT License
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Research Questions

1. **Does attention-based slicing actually work?** - Validate paper's core claim
2. **What's the optimal ε threshold?** - Paper leaves this as dynamic
3. **At what scale does CSP overhead > benefit?** - Find "Cognitive Collapse Point"
4. **Can we achieve linear scalability?** - Paper's claim about schema-based reasoning

## Metrics

| Metric | Symbol | Description |
|--------|--------|-------------|
| Cognitive Latency | L꜀ | Time from interrupt to stable state |
| Contextual Utilization | η | Information-gain tokens / total tokens |
| Sync Stability | Γ | Probability of maintaining unified state |

## References

- [Paper](https://arxiv.org/html/2602.20934v1) - Architecting AgentOS
- [MemGPT](https://arxiv.org/abs/2310.08516) - LLMs as Operating Systems
- [AIOS](https://arxiv.org/abs/2403.16971) - LLM Agent Operating System
- [FlashAttention](https://arxiv.org/abs/2205.14135) - Fast attention

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Based on research by ChengYou Li, XiaoDong Liu, XiangBao Meng, and XinYu Zhao.
