"""
Auto-Tuner for AgentOS Configuration.

Automatically detects environment and suggests optimal configuration
parameters to reduce parameter tuning burden.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Any

import psutil


@dataclass
class SystemCapabilities:
    """Detected system capabilities."""

    # CPU
    cpu_count: int = 1
    cpu_freq_mhz: float = 0.0

    # Memory (bytes)
    total_memory: int = 0
    available_memory: int = 0

    # GPU
    has_cuda: bool = False
    has_mps: bool = False  # Apple Metal Performance Shaders
    has_gpu_memory: bool = False
    gpu_memory_mb: float = 0.0

    # Platform
    platform: str = ""
    python_version: str = ""

    @classmethod
    def detect(cls) -> "SystemCapabilities":
        """Detect system capabilities.

        Returns:
            SystemCapabilities with detected values
        """
        # CPU
        cpu_count = os.cpu_count() or 1
        cpu_freq = 0.0
        try:
            freq_info = psutil.cpu_freq()
            if freq_info:
                cpu_freq = freq_info.current or 0.0
        except (FileNotFoundError, AttributeError):
            # CPU frequency not available on some systems (e.g., macOS ARM)
            pass

        # Memory
        mem = psutil.virtual_memory()
        total_memory = mem.total
        available_memory = mem.available

        # GPU detection
        has_cuda = False
        has_mps = False
        try:
            import torch

            has_cuda = torch.cuda.is_available()
            has_mps = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
        except ImportError:
            pass

        # Platform
        sys_platform = platform.system()
        py_version = platform.python_version()

        return cls(
            cpu_count=cpu_count,
            cpu_freq_mhz=cpu_freq,
            total_memory=total_memory,
            available_memory=available_memory,
            has_cuda=has_cuda,
            has_mps=has_mps,
            platform=sys_platform,
            python_version=py_version,
        )

    def is_resource_constrained(self) -> bool:
        """Check if system is resource-constrained.

        Returns:
            True if less than 4GB RAM available
        """
        return self.available_memory < 4 * 1024**3

    def is_high_performance(self) -> bool:
        """Check if system is high-performance.

        Returns:
            True if 16GB+ RAM and 8+ CPU cores
        """
        return self.available_memory >= 16 * 1024**3 and self.cpu_count >= 8

    def recommended_profile(self) -> str:
        """Get recommended profile based on system capabilities.

        Returns:
            Profile name ("fast", "balanced", or "thorough")
        """
        if self.is_resource_constrained():
            return "fast"
        elif self.is_high_performance():
            return "thorough"
        else:
            return "balanced"


@dataclass
class TuningSuggestion:
    """Suggested configuration parameters."""

    profile: str = "balanced"
    parameters: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    warnings: list[str] = field(default_factory=list)

    def merge(self, base_config: dict[str, Any]) -> dict[str, Any]:
        """Merge suggestion with base configuration.

        Args:
            base_config: Base configuration dictionary

        Returns:
            Merged configuration
        """
        result = base_config.copy()
        result.update(self.parameters)
        return result


class AutoTuner:
    """Auto-tuner for AgentOS configuration.

    Detects system capabilities and suggests optimal parameters
    based on available resources and intended workload.
    """

    def __init__(self) -> None:
        """Initialize the auto-tuner."""
        self.capabilities = SystemCapabilities.detect()

    def suggest_memory_config(
        self,
        model_size: str = "0.5B",
        available_memory: int | None = None,
    ) -> TuningSuggestion:
        """Suggest memory configuration based on model size and available memory.

        Args:
            model_size: Model size (e.g., "0.5B", "7B", "13B")
            available_memory: Available memory in bytes (auto-detect if None)

        Returns:
            TuningSuggestion with memory parameters
        """
        mem = available_memory or self.capabilities.available_memory

        # Parse model size (e.g., "0.5B" -> 0.5 * 10^9)
        try:
            size_num = float(model_size.upper().replace("B", ""))
            size_bytes = int(size_num * 1e9)
        except ValueError:
            size_bytes = 500_000_000  # Default to 0.5B

        # Estimate model memory usage (rough approximation)
        model_memory_mb = size_bytes / (1024**2) * 2  # 2x for gradients
        available_mb = mem / (1024**2)

        # Allocate memory based on available resources
        if available_mb < 8192:  # < 8GB
            l1_tokens = 256
            l2_tokens = 5000
            profile = "fast"
        elif available_mb < 32768:  # 8-32GB
            l1_tokens = 512
            l2_tokens = 10000
            profile = "balanced"
        else:  # 32GB+
            l1_tokens = 1024
            l2_tokens = 50000
            profile = "thorough"

        # Warn if insufficient memory
        warnings = []
        if model_memory_mb > available_mb * 0.5:
            warnings.append(
                f"Model may require ~{model_memory_mb:.0f}MB but only "
                f"{available_mb:.0f}MB available. Consider using a smaller model."
            )

        return TuningSuggestion(
            profile=profile,
            parameters={
                "l1_max_tokens_cache": l1_tokens,
                "l1_max_slices": max(5, l1_tokens // 50),
                "l2_max_tokens": l2_tokens,
                "l2_max_slices": max(50, l2_tokens // 100),
            },
            reasoning=f"Based on {available_mb/1024:.0f}GB available memory and {model_size} model",
            warnings=warnings,
        )

    def suggest_sync_params(
        self,
        agent_count: int = 3,
        avg_latency_ms: float = 100.0,
    ) -> TuningSuggestion:
        """Suggest sync parameters based on agent count and latency.

        Args:
            agent_count: Number of agents in the system
            avg_latency_ms: Average agent response latency in milliseconds

        Returns:
            TuningSuggestion with sync parameters
        """
        # Calculate appropriate sync intervals
        if agent_count <= 2:
            min_sync = 2000.0
            max_sync = 15000.0
            drift_thresh = 1.5
        elif agent_count <= 5:
            min_sync = 1000.0
            max_sync = 10000.0
            drift_thresh = 1.0
        else:
            min_sync = 500.0
            max_sync = 5000.0
            drift_thresh = 0.8

        # Adjust for latency
        if avg_latency_ms > 500:
            min_sync *= 1.5
            max_sync *= 1.5

        return TuningSuggestion(
            profile="balanced",
            parameters={
                "min_sync_interval_ms": min_sync,
                "max_sync_interval_ms": max_sync,
                "drift_threshold": drift_thresh,
            },
            reasoning=f"Based on {agent_count} agents with {avg_latency_ms:.0f}ms avg latency",
        )

    def suggest_for_environment(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        profile: str | None = None,
        workload: str = "general",
    ) -> TuningSuggestion:
        """Suggest complete configuration for the environment.

        Args:
            model_name: Name/size of the model being used
            profile: Preferred profile (None = auto-detect)
            workload: Workload type ("general", "research", "production", "testing")

        Returns:
            TuningSuggestion with complete configuration
        """
        # Auto-detect profile if not specified
        if profile is None:
            profile = self.capabilities.recommended_profile()

        # Get base profile config
        from agentos.profiles import get_profile

        base_profile = get_profile(profile)
        base_config = base_profile.to_agentos_config()

        # Override for specific workloads
        if workload == "testing":
            base_config.update({
                "enable_metrics": False,
                "enable_sync": False,
            })
        elif workload == "production":
            base_config.update({
                "enable_metrics": True,
                "enable_sync": True,
            })
        elif workload == "research":
            # Use thorough profile for research
            if profile != "thorough":
                research_profile = get_profile("thorough")
                base_config.update(research_profile.to_agentos_config())

        return TuningSuggestion(
            profile=profile,
            parameters=base_config,
            reasoning=(
                f"Auto-detected {profile} profile for {workload} "
                f"workload on {self.capabilities.platform}"
            ),
        )


def suggest_config(
    profile: str | None = None,
    model_size: str = "0.5B",
    agent_count: int = 3,
    workload: str = "general",
) -> dict[str, Any]:
    """Convenience function to get suggested configuration.

    Args:
        profile: Profile name ("fast", "balanced", "thorough", or None for auto)
        model_size: Model size for memory allocation
        agent_count: Number of agents for sync parameters
        workload: Workload type

    Returns:
        Dictionary of suggested configuration parameters

    Example:
        config = suggest_config(profile="balanced")
    """
    tuner = AutoTuner()

    # Get environment suggestion as base
    env_suggestion = tuner.suggest_for_environment(
        model_name=model_size,
        profile=profile,
        workload=workload,
    )

    # Get memory suggestion
    memory_suggestion = tuner.suggest_memory_config(model_size)

    # Get sync suggestion
    sync_suggestion = tuner.suggest_sync_params(agent_count)

    # Merge all suggestions
    result = env_suggestion.parameters.copy()
    result.update(memory_suggestion.parameters)
    result.update(sync_suggestion.parameters)

    return result
