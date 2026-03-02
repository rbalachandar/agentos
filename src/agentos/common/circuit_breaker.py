"""
Circuit Breaker Pattern for AgentOS Components.

Prevents cascading failures by isolating failing components
and allowing them to recover after a timeout period.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

from agentos.common.health import HealthState, HealthStatus, unhealthy


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit is tripped, requests fail fast
    HALF_OPEN = "half_open"  # Testing if component has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    # Failure threshold
    failure_threshold: int = 5  # Failures before tripping
    success_threshold: int = 2  # Successes to close circuit

    # Timeouts
    open_timeout_seconds: float = 30.0  # How long to stay open
    half_open_timeout_seconds: float = 10.0  # How long to wait in half-open

    # Monitoring
    window_seconds: float = 60.0  # Time window for failure counting

    def validate(self) -> None:
        """Validate configuration."""
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.success_threshold <= 0:
            raise ValueError("success_threshold must be positive")
        if self.open_timeout_seconds <= 0:
            raise ValueError("open_timeout_seconds must be positive")


@dataclass
class CircuitBreakerState:
    """Current state of the circuit breaker."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: datetime | None = None
    last_state_change: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreaker:
    """Circuit breaker for preventing cascading failures.

    Tracks failures and trips to OPEN state when threshold is exceeded.
    Automatically transitions to HALF_OPEN after timeout, and back to
    CLOSED after sufficient successes.
    """

    def __init__(
        self,
        component_name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize the circuit breaker.

        Args:
            component_name: Name of the protected component
            config: Circuit breaker configuration
        """
        self.component_name = component_name
        self.config = config or CircuitBreakerConfig()
        self.config.validate()

        self._state = CircuitBreakerState()
        self._failure_times: list[datetime] = []

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state.state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state.state == CircuitState.HALF_OPEN

    def record_success(self) -> None:
        """Record a successful operation.

        Resets failure counters and may close the circuit.
        """
        self._state.consecutive_failures = 0
        self._state.consecutive_successes += 1

        if self._state.state == CircuitState.HALF_OPEN:
            if self._state.consecutive_successes >= self.config.success_threshold:
                self._close_circuit()

    def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed operation.

        May trip the circuit breaker if threshold is exceeded.

        Args:
            error: Exception that caused the failure
        """
        self._state.consecutive_failures += 1
        self._state.consecutive_successes = 0
        self._state.last_failure_time = datetime.now()

        # Track failure times for sliding window
        self._failure_times.append(datetime.now())
        self._cleanup_old_failures()

        # Check if we should trip the circuit
        if self._state.consecutive_failures >= self.config.failure_threshold:
            self._open_circuit()

    def _cleanup_old_failures(self) -> None:
        """Remove failures outside the monitoring window."""
        cutoff = datetime.now() - timedelta(seconds=self.config.window_seconds)
        self._failure_times = [
            t for t in self._failure_times if t > cutoff
        ]

    def _open_circuit(self) -> None:
        """Trip the circuit breaker to OPEN state."""
        self._state.state = CircuitState.OPEN
        self._state.last_state_change = datetime.now()

    def _close_circuit(self) -> None:
        """Close the circuit breaker back to CLOSED state."""
        self._state.state = CircuitState.CLOSED
        self._state.last_state_change = datetime.now()
        self._state.consecutive_failures = 0
        self._state.consecutive_successes = 0

    def _attempt_reset(self) -> None:
        """Attempt to reset from OPEN to HALF_OPEN state."""
        time_since_open = datetime.now() - self._state.last_state_change

        if time_since_open.total_seconds() >= self.config.open_timeout_seconds:
            self._state.state = CircuitState.HALF_OPEN
            self._state.last_state_change = datetime.now()
            self._state.consecutive_successes = 0

    def should_allow_request(self) -> bool:
        """Check if request should be allowed through circuit.

        Returns:
            True if request should proceed, False if it should fail fast
        """
        # If circuit is closed, allow requests
        if self._state.state == CircuitState.CLOSED:
            return True

        # If circuit is open, check if we can attempt reset
        if self._state.state == CircuitState.OPEN:
            self._attempt_reset()
            # Still open after reset attempt
            return self._state.state != CircuitState.OPEN

        # If half-open, allow limited requests for testing
        return True

    def get_health_status(self) -> HealthStatus:
        """Get health status based on circuit state.

        Returns:
            HealthStatus reflecting circuit state
        """
        if self._state.state == CircuitState.CLOSED:
            from agentos.common.health import healthy

            return healthy(
                component=self.component_name,
                message="Circuit is closed, operating normally",
                details={
                    "failure_count": len(self._failure_times),
                    "consecutive_failures": self._state.consecutive_failures,
                },
            )
        elif self._state.state == CircuitState.OPEN:
            return unhealthy(
                component=self.component_name,
                message=(
                    f"Circuit is open, blocking requests. "
                    f"Last failure: {self._state.last_failure_time}"
                ),
                details={
                    "consecutive_failures": self._state.consecutive_failures,
                    "time_since_open": (
                        datetime.now() - self._state.last_state_change
                    ).total_seconds(),
                },
            )
        else:  # HALF_OPEN
            from agentos.common.health import degraded

            return degraded(
                component=self.component_name,
                message="Circuit is half-open, testing recovery",
                details={
                    "consecutive_successes": self._state.consecutive_successes,
                    "success_threshold": self.config.success_threshold,
                },
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of circuit state
        """
        return {
            "component": self.component_name,
            "state": self._state.state.value,
            "failure_count": len(self._failure_times),
            "consecutive_failures": self._state.consecutive_failures,
            "consecutive_successes": self._state.consecutive_successes,
            "last_failure_time": (
                self._state.last_failure_time.isoformat()
                if self._state.last_failure_time
                else None
            ),
            "last_state_change": self._state.last_state_change.isoformat(),
        }


class CircuitBreakerProtector:
    """Context manager for protecting operations with circuit breaker."""

    def __init__(self, breaker: CircuitBreaker) -> None:
        """Initialize the protector.

        Args:
            breaker: Circuit breaker to use for protection
        """
        self.breaker = breaker

    def __enter__(self) -> "CircuitBreakerProtector":
        """Enter the protected context.

        Raises:
            RuntimeError: If circuit is open and blocking requests
        """
        if not self.breaker.should_allow_request():
            raise RuntimeError(
                f"Circuit breaker is {self.breaker.state.value} for "
                f"{self.breaker.component_name}. Blocking request."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the protected context.

        Records success or failure based on exception.
        """
        _ = exc_type, exc_val, exc_tb  # Unused but required by protocol

        if exc_type is None:
            self.breaker.record_success()
        else:
            self.breaker.record_failure(exc_val if isinstance(exc_val, Exception) else None)


def protect(component_name: str, config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
    """Create a circuit breaker for a component.

    Args:
        component_name: Name of the component to protect
        config: Circuit breaker configuration

    Returns:
        Configured CircuitBreaker instance

    Example:
        breaker = protect("kernel", CircuitBreakerConfig(failure_threshold=3))

        with CircuitBreakerProtector(breaker):
            result = kernel.process(text)
    """
    return CircuitBreaker(component_name, config)
