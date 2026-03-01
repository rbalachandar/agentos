"""AgentOS: From Token-Level Context to Emergent System-Level Intelligence.

Research implementation of the AgentOS architecture.
"""

__version__ = "0.1.0"

# Core kernel
from agentos.kernel.reasoning_kernel import (
    CognitiveState,
    KernelState,
    ProcessingResult,
    ReasoningKernel,
    ReasoningKernelConfig,
    create_kernel,
)

# Memory slicing types
from agentos.memory.slicing.types import (
    AttentionOutput,
    DensityProfile,
    SemanticSlice,
    SlicingResult,
)

# Memory slicing components
from agentos.memory.slicing.cid_calculator import (
    CIDCalculator,
    CIDCalculatorConfig,
    compute_cid,
)
from agentos.memory.slicing.boundary_detector import (
    BoundaryDetector,
    BoundaryDetectorConfig,
    ThresholdStrategy,
    detect_boundaries,
)
from agentos.memory.slicing.slicer import (
    SemanticSlicer,
    SemanticSlicerConfig,
    slice_semantic,
)

# Model backends
from agentos.models.transformers_backend import (
    BackendConfig,
    DeviceType,
    TransformersBackend,
    create_backend,
)

# Integration (Phase 6)
from agentos.agentos import (
    AgentOS,
    AgentOSConfig,
    CollaborationResult,
    create_agentos,
)
from agentos.agent import (
    Agent,
    AgentConfig,
    AgentMemory,
    create_agent,
)

# CLI
from agentos.cli import app

__all__ = [
    # Version
    "__version__",
    # Integration (Phase 6)
    "AgentOS",
    "AgentOSConfig",
    "CollaborationResult",
    "create_agentos",
    "Agent",
    "AgentConfig",
    "AgentMemory",
    "create_agent",
    # CLI
    "app",
    # Kernel
    "ReasoningKernel",
    "ReasoningKernelConfig",
    "CognitiveState",
    "KernelState",
    "ProcessingResult",
    "create_kernel",
    # Types
    "AttentionOutput",
    "DensityProfile",
    "SemanticSlice",
    "SlicingResult",
    # CID
    "CIDCalculator",
    "CIDCalculatorConfig",
    "compute_cid",
    # Boundary detection
    "BoundaryDetector",
    "BoundaryDetectorConfig",
    "ThresholdStrategy",
    "detect_boundaries",
    # Slicing
    "SemanticSlicer",
    "SemanticSlicerConfig",
    "slice_semantic",
    # Backend
    "TransformersBackend",
    "BackendConfig",
    "DeviceType",
    "create_backend",
]
