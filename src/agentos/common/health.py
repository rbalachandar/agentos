"""
Health Check System for AgentOS Components.

Provides health status tracking and reporting for all major components
to detect problems before they become failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class HealthState(Enum):
    """Health states for components."""

    HEALTHY = "healthy"  # Component is functioning normally
    DEGRADED = "degraded"  # Component is functioning but with reduced capacity
    UNHEALTHY = "unhealthy"  # Component is failing or failed
    UNKNOWN = "unknown"  # Health status cannot be determined


@dataclass
class HealthStatus:
    """Health status of a component.

    Attributes:
        state: Current health state
        component: Name of the component
        message: Human-readable status message
        details: Additional status details
        timestamp: When this status was determined
        error: Exception that caused unhealthy state (if any)
    """

    state: HealthState
    component: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    error: Exception | None = None

    @property
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.state == HealthState.HEALTHY

    @property
    def is_degraded(self) -> bool:
        """Check if component is degraded."""
        return self.state == HealthState.DEGRADED

    @property
    def is_unhealthy(self) -> bool:
        """Check if component is unhealthy."""
        return self.state == HealthState.UNHEALTHY

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of health status
        """
        return {
            "state": self.state.value,
            "component": self.component,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "error": str(self.error) if self.error else None,
        }


def healthy(
    component: str,
    message: str = "Component is healthy",
    details: dict[str, Any] | None = None,
) -> HealthStatus:
    """Create a healthy status.

    Args:
        component: Component name
        message: Status message
        details: Additional details

    Returns:
        Healthy status
    """
    return HealthStatus(
        state=HealthState.HEALTHY,
        component=component,
        message=message,
        details=details or {},
    )


def degraded(
    component: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> HealthStatus:
    """Create a degraded status.

    Args:
        component: Component name
        message: Status message
        details: Additional details

    Returns:
        Degraded status
    """
    return HealthStatus(
        state=HealthState.DEGRADED,
        component=component,
        message=message,
        details=details or {},
    )


def unhealthy(
    component: str,
    message: str,
    error: Exception | None = None,
    details: dict[str, Any] | None = None,
) -> HealthStatus:
    """Create an unhealthy status.

    Args:
        component: Component name
        message: Status message
        error: Exception that caused the failure
        details: Additional details

    Returns:
        Unhealthy status
    """
    return HealthStatus(
        state=HealthState.UNHEALTHY,
        component=component,
        message=message,
        error=error,
        details=details or {},
    )


@dataclass
class SystemHealth:
    """Overall system health.

    Aggregates health status from all components.
    """

    components: dict[str, HealthStatus] = field(default_factory=dict)

    def register(self, status: HealthStatus) -> None:
        """Register a component's health status.

        Args:
            status: Health status to register
        """
        self.components[status.component] = status

    def get_component_health(self, component: str) -> HealthStatus | None:
        """Get health status of a specific component.

        Args:
            component: Component name

        Returns:
            Health status if found, None otherwise
        """
        return self.components.get(component)

    @property
    def overall_state(self) -> HealthState:
        """Get overall system health state.

        Returns:
            Worst health state across all components
        """
        if not self.components:
            return HealthState.UNKNOWN

        states = [status.state for status in self.components.values()]

        if any(s == HealthState.UNHEALTHY for s in states):
            return HealthState.UNHEALTHY
        elif any(s == HealthState.DEGRADED for s in states):
            return HealthState.DEGRADED
        elif all(s == HealthState.HEALTHY for s in states):
            return HealthState.HEALTHY
        else:
            return HealthState.UNKNOWN

    @property
    def is_healthy(self) -> bool:
        """Check if entire system is healthy."""
        return self.overall_state == HealthState.HEALTHY

    @property
    def unhealthy_components(self) -> list[str]:
        """Get list of unhealthy components.

        Returns:
            List of component names that are unhealthy
        """
        return [
            name for name, status in self.components.items()
            if status.state == HealthState.UNHEALTHY
        ]

    @property
    def degraded_components(self) -> list[str]:
        """Get list of degraded components.

        Returns:
            List of component names that are degraded
        """
        return [
            name for name, status in self.components.items()
            if status.state == HealthState.DEGRADED
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of system health
        """
        return {
            "overall_state": self.overall_state.value,
            "is_healthy": self.is_healthy,
            "unhealthy_components": self.unhealthy_components,
            "degraded_components": self.degraded_components,
            "components": {
                name: status.to_dict()
                for name, status in self.components.items()
            },
        }
