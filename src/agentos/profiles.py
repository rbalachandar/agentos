"""
Configuration Profiles for AgentOS.

Pre-tuned configurations for common use cases to reduce
parameter tuning burden.

Profiles:
- fast: Minimal resource usage, faster processing
- balanced: Balanced performance and quality (default)
- thorough: Maximum quality, higher resource usage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentos.memory import L1CacheConfig, L2Config, L3Config, SMMUConfig
from agentos.scheduler import SchedulerConfig
from agentos.sync import CSPOrchestratorConfig, DriftTrackerConfig


@dataclass
class ProfileConfig:
    """Base configuration for a profile."""

    name: str
    description: str

    # Memory configuration
    l1_max_tokens: int = 512
    l1_max_slices: int = 10
    l2_max_tokens: int = 10000
    l2_max_slices: int = 100

    # Scheduler configuration
    scheduler_time_slice_ms: float = 100.0
    scheduler_use_cognitive_fidelity: bool = True

    # Sync configuration
    min_sync_interval_ms: float = 1000.0
    max_sync_interval_ms: float = 10000.0
    drift_threshold: float = 1.0

    # Importance weights
    importance_w_attention: float = 0.4
    importance_w_recency: float = 0.2
    importance_w_frequency: float = 0.2
    importance_w_user: float = 0.2

    def to_agentos_config(self) -> dict[str, Any]:
        """Convert profile to AgentOSConfig kwargs.

        Returns:
            Dictionary of configuration parameters
        """
        return {
            "l1_max_tokens_cache": self.l1_max_tokens,
            "l1_max_slices": self.l1_max_slices,
            "l2_max_tokens": self.l2_max_tokens,
            "l2_max_slices": self.l2_max_slices,
            "scheduler_time_slice_ms": self.scheduler_time_slice_ms,
            "scheduler_use_cognitive_fidelity": self.scheduler_use_cognitive_fidelity,
            "min_sync_interval_ms": self.min_sync_interval_ms,
            "max_sync_interval_ms": self.max_sync_interval_ms,
            "drift_threshold": self.drift_threshold,
        }

    def to_smmu_config(self) -> SMMUConfig:
        """Convert profile to SMMUConfig.

        Returns:
            SMMUConfig instance
        """
        return SMMUConfig(
            l1_config=L1CacheConfig(
                max_tokens=self.l1_max_tokens,
                max_slices=self.l1_max_slices,
            ),
            l2_config=L2Config(
                max_tokens=self.l2_max_tokens,
                max_slices=self.l2_max_slices,
            ),
            l3_config=L3Config(),
            importance_w_attention=self.importance_w_attention,
            importance_w_recency=self.importance_w_recency,
            importance_w_frequency=self.importance_w_frequency,
            importance_w_user=self.importance_w_user,
        )


@dataclass
class FastProfile(ProfileConfig):
    """Fast profile - minimal resource usage.

    Suitable for:
    - Resource-constrained environments
    - Quick prototyping
    - Testing and development
    """

    name: str = "fast"
    description: str = "Minimal resource usage, faster processing"

    # Smaller memory footprint
    l1_max_tokens: int = 256
    l1_max_slices: int = 5
    l2_max_tokens: int = 5000
    l2_max_slices: int = 50

    # Faster scheduler (less accurate)
    scheduler_time_slice_ms: float = 50.0
    scheduler_use_cognitive_fidelity: bool = False

    # Less frequent sync
    min_sync_interval_ms: float = 2000.0
    max_sync_interval_ms: float = 15000.0


@dataclass
class BalancedProfile(ProfileConfig):
    """Balanced profile - default configuration.

    Suitable for:
    - General-purpose use
    - Production workloads
    - Most multi-agent scenarios
    """

    name: str = "balanced"
    description: str = "Balanced performance and quality (default)"

    # Default values from ProfileConfig


@dataclass
class ThoroughProfile(ProfileConfig):
    """Thorough profile - maximum quality.

    Suitable for:
    - Complex reasoning tasks
    - Long-running sessions
    - Research and analysis
    """

    name: str = "thorough"
    description: str = "Maximum quality, higher resource usage"

    # Larger memory footprint
    l1_max_tokens: int = 1024
    l1_max_slices: int = 20
    l2_max_tokens: int = 50000
    l2_max_slices: int = 500

    # Slower but more accurate scheduler
    scheduler_time_slice_ms: float = 200.0
    scheduler_use_cognitive_fidelity: bool = True

    # More frequent sync for better coherence
    min_sync_interval_ms: float = 500.0
    max_sync_interval_ms: float = 5000.0

    # Higher importance on attention patterns
    importance_w_attention: float = 0.5
    importance_w_recency: float = 0.15
    importance_w_frequency: float = 0.15
    importance_w_user: float = 0.2


# Profile registry
PROFILES: dict[str, ProfileConfig] = {
    "fast": FastProfile(),
    "balanced": BalancedProfile(),
    "thorough": ThoroughProfile(),
}


def get_profile(name: str) -> ProfileConfig:
    """Get a profile by name.

    Args:
        name: Profile name ("fast", "balanced", "thorough")

    Returns:
        ProfileConfig instance

    Raises:
        ValueError: If profile name is not found
    """
    name = name.lower()
    if name not in PROFILES:
        valid = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown profile: {name}. Valid profiles: {valid}")
    return PROFILES[name]


def list_profiles() -> list[str]:
    """List available profile names.

    Returns:
        List of profile names
    """
    return list(PROFILES.keys())


def apply_profile(
    profile_name: str,
    **overrides,
) -> dict[str, Any]:
    """Get configuration for a profile with optional overrides.

    Args:
        profile_name: Name of the profile to use
        **overrides: Configuration parameters to override

    Returns:
        Dictionary of configuration parameters

    Example:
        config = apply_profile("fast", l1_max_tokens=128)
    """
    profile = get_profile(profile_name)
    config = profile.to_agentos_config()
    config.update(overrides)
    return config
