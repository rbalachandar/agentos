"""
Semantic Synthesis Module.

Based on AgentOS paper Section 3.5: Multi-Agent Collaboration

Implements sophisticated synthesis that:
- Uses semantic similarity to find related concepts across agents
- Weights contributions by cognitive fidelity (attention focus)
- Resolves contradictions using the reconciliation system
- Produces integrated answer from diverse agent perspectives
"""

from agentos.synthesis.semantic_synthesizer import SemanticSynthesizer, create_synthesizer
from agentos.synthesis.types import SynthesisConfig, SynthesisResult

__all__ = [
    "SynthesisConfig",
    "SynthesisResult",
    "SemanticSynthesizer",
    "create_synthesizer",
]
