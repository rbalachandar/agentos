"""
Semantic Synthesizer - sophisticated multi-agent output synthesis.

Implements synthesis based on AgentOS paper Section 3.5.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cosine

from agentos.synthesis.types import (
    AgentContribution,
    ConceptCluster,
    ConflictResolution,
    SynthesisConfig,
    SynthesisResult,
    SynthesisStrategy,
)

logger = logging.getLogger(__name__)


class SemanticSynthesizer:
    """Synthesizes multi-agent outputs using semantic analysis.

    Implementation based on AgentOS paper:
    - Uses semantic similarity to cluster related concepts
    - Weights contributions by cognitive fidelity (attention focus)
    - Detects and resolves conflicts between agent perspectives
    - Produces integrated synthesis
    """

    def __init__(self, config: SynthesisConfig | None = None) -> None:
        """Initialize the semantic synthesizer.

        Args:
            config: Synthesis configuration
        """
        self.config = config or SynthesisConfig()
        self.config.validate()

    def synthesize(
        self,
        task: str,
        agent_contributions: dict[str, str],
        agent_states: dict[str, Any] | None = None,
        llm_generate_fn: callable | None = None,
    ) -> SynthesisResult:
        """Synthesize agent contributions into final answer.

        Args:
            task: The original task/question
            agent_contributions: Dictionary mapping agent_id to contribution text
            agent_states: Optional agent states with semantic metadata
            llm_generate_fn: Optional LLM generation function for final synthesis

        Returns:
            SynthesisResult with final integrated answer
        """
        start_time = time.time()

        # Step 1: Build AgentContribution objects with metadata
        contributions = self._build_contributions(
            agent_contributions, agent_states
        )

        # Step 2: Cluster semantically related concepts
        clusters = self._cluster_concepts(contributions)

        # Step 3: Detect and resolve conflicts
        conflicts = self._detect_conflicts(clusters) if self.config.detect_conflicts else []

        # Step 4: Generate final synthesis
        final_synthesis = self._generate_synthesis(
            task, contributions, clusters, conflicts, llm_generate_fn
        )

        # Step 5: Calculate quality metrics
        coherence = self._calculate_coherence(clusters)
        coverage = self._calculate_coverage(contributions, clusters)

        duration_ms = (time.time() - start_time) * 1000

        return SynthesisResult(
            final_synthesis=final_synthesis,
            strategy_used=self.config.strategy,
            concept_clusters=clusters,
            conflicts_resolved=conflicts,
            agents_participated=list(agent_contributions.keys()),
            coherence_score=coherence,
            coverage_score=coverage,
            synthesis_duration_ms=duration_ms,
            metadata={
                "task": task,
                "num_contributions": len(contributions),
                "num_clusters": len(clusters),
                "num_conflicts": len(conflicts),
            },
        )

    def _build_contributions(
        self,
        agent_contributions: dict[str, str],
        agent_states: dict[str, Any] | None = None,
    ) -> list[AgentContribution]:
        """Build AgentContribution objects with metadata.

        Args:
            agent_contributions: Raw contribution texts
            agent_states: Agent states with semantic metadata

        Returns:
            List of AgentContribution with metadata
        """
        contributions = []

        for agent_id, content in agent_contributions.items():
            # Get metadata from agent state if available
            semantic_gradient = None
            confidence = 0.5
            attention_focus = 0.5
            agent_name = agent_id
            agent_role = "agent"
            slices = []

            if agent_states and agent_id in agent_states:
                state = agent_states[agent_id]

                # Extract semantic gradient
                if hasattr(state, 'semantic_gradients') and state.semantic_gradients is not None:
                    semantic_gradient = state.semantic_gradients

                # Extract attention focus from RCB if available
                if hasattr(state, 'rcb') and state.rcb is not None:
                    attention_focus = state.rcb.cognitive_fidelity

                # Extract metadata
                if hasattr(state, 'metadata'):
                    agent_name = state.metadata.get('name', agent_name)
                    agent_role = state.metadata.get('role', agent_role)

                # Extract slices
                if hasattr(state, 'active_slices'):
                    slices = state.active_slices

            contribution = AgentContribution(
                agent_id=agent_id,
                agent_name=agent_name,
                agent_role=agent_role,
                content=content,
                semantic_gradient=semantic_gradient,
                confidence=attention_focus,  # Use attention focus as confidence proxy
                attention_focus=attention_focus,
                slices=slices,
            )
            contributions.append(contribution)

        return contributions

    def _cluster_concepts(
        self, contributions: list[AgentContribution]
    ) -> list[ConceptCluster]:
        """Cluster semantically related concepts from contributions.

        Uses semantic gradients if available, otherwise falls back to
        text-based similarity.

        Args:
            contributions: Agent contributions

        Returns:
            List of concept clusters
        """
        if not contributions:
            return []

        # Filter contributions that have semantic gradients
        contributions_with_embeddings = [
            c for c in contributions
            if c.semantic_gradient is not None
        ]

        if len(contributions_with_embeddings) < 2:
            raise RuntimeError(
                f"Semantic synthesis requires at least 2 contributions with semantic gradients. "
                f"Only {len(contributions_with_embeddings)} contributions have semantic information. "
                f"Ensure agents have semantic_gradient set from their reasoning kernel processing."
            )

        # Compute similarity matrix
        embeddings = np.array([c.semantic_gradient for c in contributions_with_embeddings])
        n = len(embeddings)

        # Simple clustering: group by similarity threshold
        clusters_dict: dict[str, list[AgentContribution]] = {}
        assigned = set()

        for i in range(n):
            if i in assigned:
                continue

            # Find similar contributions
            cluster_contributions = [contributions_with_embeddings[i]]
            cluster_id = f"cluster_{i}_{uuid.uuid4().hex[:6]}"

            for j in range(i + 1, n):
                if j in assigned:
                    continue

                # Compute cosine similarity
                similarity = 1.0 - cosine(embeddings[i], embeddings[j])

                if similarity >= self.config.similarity_threshold:
                    cluster_contributions.append(contributions_with_embeddings[j])
                    assigned.add(j)

            assigned.add(i)
            clusters_dict[cluster_id] = cluster_contributions

        # Create ConceptCluster objects
        clusters = []
        for cluster_id, cluster_contribs in clusters_dict.items():
            # Compute centroid
            cluster_embeddings = np.array([c.semantic_gradient for c in cluster_contribs])
            centroid = cluster_embeddings.mean(axis=0)

            # Compute coherence (average pairwise similarity)
            if len(cluster_contribs) >= 2:
                similarities = []
                for i in range(len(cluster_contribs)):
                    for j in range(i + 1, len(cluster_contribs)):
                        sim = 1.0 - cosine(
                            cluster_contribs[i].semantic_gradient,
                            cluster_contribs[j].semantic_gradient,
                        )
                        similarities.append(sim)
                coherence = float(np.mean(similarities)) if similarities else 0.5
            else:
                coherence = 1.0

            # Extract key themes from slices
            key_themes = self._extract_key_themes(cluster_contribs)

            # Find consensus and diverging views
            consensus, diverging = self._analyze_consensus(cluster_contribs)

            cluster = ConceptCluster(
                cluster_id=cluster_id,
                concepts=cluster_contribs,
                centroid_embedding=centroid,
                coherence_score=coherence,
                key_themes=key_themes,
                consensus_view=consensus,
                diverging_views=diverging,
            )
            clusters.append(cluster)

        return clusters

    def _extract_key_themes(
        self, contributions: list[AgentContribution]
    ) -> list[str]:
        """Extract key themes from semantic slices.

        Args:
            contributions: Agent contributions

        Returns:
            List of key themes
        """
        themes = []

        # Extract from slice contents
        for contrib in contributions:
            for slice_obj in contrib.slices:
                if slice_obj.content and len(slice_obj.content) > 10:
                    theme = slice_obj.content[:50].strip()
                    if theme and theme not in themes:
                        themes.append(theme)
                        if len(themes) >= 5:  # Limit themes
                            break

            if len(themes) >= 5:
                break

        return themes

    def _analyze_consensus(
        self, contributions: list[AgentContribution]
    ) -> tuple[str | None, list[str]]:
        """Analyze consensus and diverging views in a cluster.

        Args:
            contributions: Agent contributions in the cluster

        Returns:
            Tuple of (consensus_view, diverging_views)
        """
        if not contributions:
            return None, []

        # Simple heuristic: find common phrases/patterns
        # In production, would use more sophisticated NLP

        all_content = " ".join([c.content for c in contributions])

        # For now, use agent roles to identify potential divergence
        roles = set(c.agent_role for c in contributions)
        if len(roles) > 1:
            # Different roles = potential for diverse perspectives
            diverging = [f"Diverse perspectives from {', '.join(roles)}"]
        else:
            diverging = []

        # Consensus: approximate by taking first contribution's main point
        consensus = contributions[0].content[:100] if contributions else None

        return consensus, diverging

    def _detect_conflicts(
        self, clusters: list[ConceptCluster]
    ) -> list[ConflictResolution]:
        """Detect conflicts between agent contributions.

        Args:
            clusters: Concept clusters

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Check for conflicts within clusters
        for cluster in clusters:
            if len(cluster.concepts) < 2:
                continue

            # Low coherence suggests potential conflict
            if cluster.coherence_score < 0.5:
                # Extract conflicting statements
                statements = [
                    (c.agent_name, c.content[:100])
                    for c in cluster.concepts
                ]

                if len(statements) >= 2:
                    conflict = ConflictResolution(
                        conflict_type="semantic_divergence",
                        agents_involved=[c.agent_id for c in cluster.concepts],
                        conflicting_statements=statements,
                        resolution_strategy=self.config.conflict_resolution_strategy,
                        resolved_content="",  # Will be filled during synthesis
                        confidence=1.0 - cluster.coherence_score,
                    )
                    conflicts.append(conflict)

        return conflicts

    def _generate_synthesis(
        self,
        task: str,
        contributions: list[AgentContribution],
        clusters: list[ConceptCluster],
        conflicts: list[ConflictResolution],
        llm_generate_fn: callable | None = None,
    ) -> str:
        """Generate final synthesis from clustered concepts.

        Args:
            task: Original task
            contributions: Agent contributions
            clusters: Concept clusters
            conflicts: Detected conflicts
            llm_generate_fn: Optional LLM generation function

        Returns:
            Final synthesized answer
        """
        # Weight contributions by confidence if enabled
        if self.config.use_confidence_weighting:
            weighted_contributions = self._weight_by_confidence(contributions)
        else:
            weighted_contributions = contributions

        # Build synthesis structure
        synthesis_parts = []

        # 1. Introduction
        synthesis_parts.append(f"## Synthesis: {task}\n")

        # 2. Key themes from clusters
        if clusters:
            synthesis_parts.append("### Key Themes Identified\n")
            for i, cluster in enumerate(clusters[:5], 1):
                if cluster.key_themes:
                    synthesis_parts.append(f"{i}. {cluster.key_themes[0]}")
            synthesis_parts.append("")

        # 3. Agent perspectives (weighted by confidence)
        synthesis_parts.append("### Agent Perspectives\n")
        for contrib in sorted(
            weighted_contributions,
            key=lambda c: c.confidence,
            reverse=True,
        ):
            weight_indicator = "⭐" if contrib.confidence > 0.7 else ""
            synthesis_parts.append(
                f"**{contrib.agent_name} ({contrib.agent_role}){weight_indicator}**: "
                f"{contrib.content[:200]}{'...' if len(contrib.content) > 200 else ''}"
            )
        synthesis_parts.append("")

        # 4. Conflict resolution (if any conflicts)
        if conflicts:
            synthesis_parts.append("### Resolved Conflicts\n")
            for conflict in conflicts:
                synthesis_parts.append(
                    f"- **{conflict.conflict_type}**: Resolved using {conflict.resolution_strategy}"
                )
            synthesis_parts.append("")

        # 5. Use LLM for final integration if available
        if llm_generate_fn and self.config.use_llm_synthesis:
            # Create focused prompt for LLM
            llm_prompt = self._build_llm_synthesis_prompt(
                task, weighted_contributions, clusters, conflicts
            )

            try:
                llm_synthesis = llm_generate_fn(
                    prompt=llm_prompt,
                    max_new_tokens=self.config.llm_synthesis_max_tokens,
                    system_prompt=(
                        "You are a synthesizer. Integrate diverse agent perspectives "
                        "into a coherent, comprehensive answer. Highlight areas of "
                        "agreement and note any disagreements."
                    ),
                )

                # Add LLM synthesis
                synthesis_parts.append("### Integrated Answer\n")
                synthesis_parts.append(llm_synthesis)
            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}, using structured output")
                synthesis_parts.append(self._generate_structured_synthesis(
                    task, weighted_contributions, clusters
                ))
        else:
            # Use structured synthesis
            synthesis_parts.append("### Integrated Answer\n")
            synthesis_parts.append(self._generate_structured_synthesis(
                task, weighted_contributions, clusters
            ))

        return "\n".join(synthesis_parts)

    def _weight_by_confidence(
        self, contributions: list[AgentContribution]
    ) -> list[AgentContribution]:
        """Weight contributions by confidence/attention focus.

        Args:
            contributions: Agent contributions

        Returns:
            Contributions sorted by confidence weight
        """
        # Apply weight factor
        for contrib in contributions:
            # Higher confidence = higher weight
            # Use exponential to emphasize differences
            contrib.confidence = min(1.0, contrib.confidence ** self.config.confidence_weight_factor)

        return contributions

    def _build_llm_synthesis_prompt(
        self,
        task: str,
        contributions: list[AgentContribution],
        clusters: list[ConceptCluster],
        conflicts: list[ConflictResolution],
    ) -> str:
        """Build prompt for LLM-based synthesis.

        Args:
            task: Original task
            contributions: Weighted agent contributions
            clusters: Concept clusters
            conflicts: Detected conflicts

        Returns:
            LLM prompt
        """
        prompt_parts = [f"Task: {task}\n"]

        # Add high-confidence contributions first
        high_conf = [c for c in contributions if c.confidence > 0.7]
        if high_conf:
            prompt_parts.append("Key agent perspectives:")
            for c in high_conf[:3]:
                prompt_parts.append(f"- [{c.agent_role}]: {c.content[:150]}...")

        # Add cluster insights
        if clusters:
            prompt_parts.append("\nKey themes identified:")
            for cluster in clusters[:3]:
                if cluster.consensus_view:
                    prompt_parts.append(f"- {cluster.consensus_view[:100]}...")

        prompt_parts.append("\nPlease synthesize a comprehensive answer that integrates these perspectives.")

        return "\n".join(prompt_parts)

    def _generate_structured_synthesis(
        self,
        task: str,
        contributions: list[AgentContribution],
        clusters: list[ConceptCluster],
    ) -> str:
        """Generate structured synthesis without LLM.

        Args:
            task: Original task
            contributions: Agent contributions
            clusters: Concept clusters

        Returns:
            Structured synthesis text
        """
        parts = []

        # Start with highest confidence contribution
        if contributions:
            top_contrib = max(contributions, key=lambda c: c.confidence)
            parts.append(f"Based on {top_contrib.agent_name}'s analysis:")
            parts.append(top_contrib.content[:300])
            parts.append("")

        # Add insights from other perspectives
        other_contribs = [c for c in contributions if c != top_contrib]
        if other_contribs:
            parts.append("Additional perspectives:")
            for c in other_contribs[:2]:
                parts.append(f"- From {c.agent_name}: {c.content[:150]}...")

        parts.append("")
        parts.append(f"This synthesis integrates {len(contributions)} agent perspectives on: {task}")

        return "\n".join(parts)

    def _calculate_coherence(self, clusters: list[ConceptCluster]) -> float:
        """Calculate overall coherence of synthesis.

        Args:
            clusters: Concept clusters

        Returns:
            Coherence score (0-1)
        """
        if not clusters:
            return 0.0

        # Average coherence across clusters
        coherences = [c.coherence_score for c in clusters]
        return float(np.mean(coherences)) if coherences else 0.0

    def _calculate_coverage(
        self,
        contributions: list[AgentContribution],
        clusters: list[ConceptCluster],
    ) -> float:
        """Calculate how well synthesis covers all perspectives.

        Args:
            contributions: All agent contributions
            clusters: Concept clusters

        Returns:
            Coverage score (0-1)
        """
        if not contributions:
            return 0.0

        # Count how many contributions are represented in clusters
        agents_in_clusters = set()
        for cluster in clusters:
            agents_in_clusters.update(c.agent_id for c in cluster.concepts)

        coverage = len(agents_in_clusters) / len(contributions)
        return float(coverage)


def create_synthesizer(
    strategy: SynthesisStrategy = SynthesisStrategy.SEMANTIC_MERGE,
    similarity_threshold: float = 0.7,
    use_confidence_weighting: bool = True,
    **kwargs,
) -> SemanticSynthesizer:
    """Convenience function to create a semantic synthesizer.

    Args:
        strategy: Synthesis strategy to use
        similarity_threshold: Semantic similarity threshold for clustering
        use_confidence_weighting: Whether to weight by confidence
        **kwargs: Additional config options

    Returns:
        Configured SemanticSynthesizer
    """
    config = SynthesisConfig(
        strategy=strategy,
        similarity_threshold=similarity_threshold,
        use_confidence_weighting=use_confidence_weighting,
        **kwargs,
    )
    return SemanticSynthesizer(config)
