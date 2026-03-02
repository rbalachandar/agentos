"""
Common utilities and patterns for AgentOS.

This module contains shared components used across the system:
- Health checking
- Circuit breaker pattern
- Bulkhead pattern (resource isolation)
"""

from agentos.common.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerProtector,
    CircuitState,
    protect,
)
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
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerProtector",
    "CircuitState",
    "protect",
]
