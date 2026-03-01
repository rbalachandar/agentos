"""
Multi-Agent Synchronization Module.

This module provides synchronization mechanisms for maintaining cognitive
coherence across multiple reasoning agents.

Based on AgentOS paper Section 3.4:
- Cognitive Drift Tracking (Formula 3)
- Cognitive Sync Pulse (CSP) Orchestrator (Algorithm 3)
- Global State Reconciliation
- Perception Alignment Protocol
- Distributed Shared Memory
"""

from __future__ import annotations

from agentos.sync.distributed_memory import (
    DistributedSharedMemory,
    DistributedSliceEntry,
    StoreBackend,
    VersionVector,
)
from agentos.sync.drift_tracker import (
    CognitiveDriftTracker,
    DriftTrackerConfig,
)
from agentos.sync.perception_alignment import (
    ConfidenceWindow,
    PerceptionAlignmentConfig,
    PerceptionAlignmentProtocol,
)
from agentos.sync.reconciliation import (
    ReconciliationConfig,
    StateReconciler,
    reconcile_sync_pulse,
)
from agentos.sync.sync_pulse import (
    CSPOrchestrator,
    CSPOrchestratorConfig,
)
from agentos.sync.types import (
    AgentState,
    ConflictResolution,
    DriftMetrics,
    GlobalSemanticState,
    SemanticSliceVersion,
    SyncPulse,
    SyncTrigger,
)

__all__ = [
    # Types
    "AgentState",
    "ConflictResolution",
    "DriftMetrics",
    "GlobalSemanticState",
    "SemanticSliceVersion",
    "SyncPulse",
    "SyncTrigger",
    # Drift Tracking
    "CognitiveDriftTracker",
    "DriftTrackerConfig",
    # CSP Orchestrator
    "CSPOrchestrator",
    "CSPOrchestratorConfig",
    # Reconciliation
    "StateReconciler",
    "ReconciliationConfig",
    "reconcile_sync_pulse",
    # Perception Alignment
    "PerceptionAlignmentProtocol",
    "PerceptionAlignmentConfig",
    "ConfidenceWindow",
    # Distributed Memory
    "DistributedSharedMemory",
    "DistributedSliceEntry",
    "VersionVector",
    "StoreBackend",
]
