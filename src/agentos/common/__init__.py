"""
Common utilities and patterns for AgentOS.

This module contains shared components used across the system:
- Health checking
"""

from agentos.common.health import (
    HealthState,
    HealthStatus,
    SystemHealth,
    degraded,
    healthy,
    unhealthy,
)

__all__ = [
    # Health
    "HealthState",
    "HealthStatus",
    "SystemHealth",
    "healthy",
    "degraded",
    "unhealthy",
]
