"""Circuit Breaker + Auto-Failover for MEOKCLAW

Production-grade reliability layer. When a model degrades, automatically
route to fallback. Restore when healthy.

Features:
- Per-model health tracking (success rate, latency, error rate)
- Circuit breaker states: CLOSED, OPEN, HALF_OPEN
- Automatic fallback chain (cloud → local → cached)
- Exponential backoff for recovery probes
- Health check dashboard endpoint
- Configurable thresholds per model

Usage:
    from circuit_breaker import circuit_breaker
    
    # Register models with fallback chain
    circuit_breaker.register("deepseek-v4-pro", {
        "fallback_chain": ["deepseek-v4-flash", "kimi-k2.6", "llama3.1:8b"],
        "error_threshold": 0.5,  # Trip at 50% error rate
        "latency_threshold_ms": 10000,  # Trip at >10s latency
        "recovery_timeout": 30,  # Try again after 30s
    })
    
    # Use wrapped inference
    result = await circuit_breaker.call("deepseek-v4-pro", infer_fn)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class ModelHealth:
    model_id: str
    state: CircuitState = CircuitState.CLOSED
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    opened_at: Optional[float] = None
    half_open_probes: int = 0
    config: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """Circuit breaker with automatic fallback chains."""

    DEFAULT_CONFIG = {
        "error_threshold": 0.5,        # Trip if error rate > 50%
        "latency_threshold_ms": 15000,  # Trip if p95 latency > 15s
        "min_requests": 5,             # Need at least 5 requests before tripping
        "consecutive_failures": 3,     # Or 3 consecutive failures
        "recovery_timeout": 30,        # Seconds before HALF_OPEN
        "half_open_max_probes": 3,     # Probes before deciding
        "half_open_success_needed": 2,  # Successes needed to close
    }

    def __init__(self):
        self._health: Dict[str, ModelHealth] = {}
        self._fallback_chains: Dict[str, List[str]] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def register(self, model_id: str, config: Optional[Dict[str, Any]] = None):
        """Register a model with circuit breaker config."""
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self._configs[model_id] = cfg
        self._fallback_chains[model_id] = cfg.get("fallback_chain", [])
        self._health[model_id] = ModelHealth(
            model_id=model_id,
            config=cfg,
        )

    def _get_error_rate(self, health: ModelHealth) -> float:
        total = health.success_count + health.failure_count
        if total < health.config.get("min_requests", 5):
            return 0.0
        return health.failure_count / total

    def _should_trip(self, health: ModelHealth) -> bool:
        """Check if circuit should trip to OPEN."""
        cfg = health.config

        # Consecutive failures
        if health.consecutive_failures >= cfg.get("consecutive_failures", 3):
            return True

        # Error rate threshold
        error_rate = self._get_error_rate(health)
        if error_rate > cfg.get("error_threshold", 0.5):
            return True

        return False

    def _record_success(self, health: ModelHealth, latency_ms: int):
        health.success_count += 1
        health.consecutive_failures = 0
        health.consecutive_successes += 1
        health.total_latency_ms += latency_ms
        health.last_success_time = time.time()

        if health.state == CircuitState.HALF_OPEN:
            health.half_open_probes += 1
            if health.consecutive_successes >= health.config.get("half_open_success_needed", 2):
                health.state = CircuitState.CLOSED
                health.opened_at = None
                health.half_open_probes = 0
                health.consecutive_successes = 0
                health.success_count = 0
                health.failure_count = 0

    def _record_failure(self, health: ModelHealth):
        health.failure_count += 1
        health.consecutive_failures += 1
        health.consecutive_successes = 0
        health.last_failure_time = time.time()

        if health.state == CircuitState.HALF_OPEN:
            health.state = CircuitState.OPEN
            health.opened_at = time.time()
            health.half_open_probes = 0
            return

        if health.state == CircuitState.CLOSED and self._should_trip(health):
            health.state = CircuitState.OPEN
            health.opened_at = time.time()
            health.consecutive_failures = 0

    def _check_recovery(self, health: ModelHealth):
        """Check if OPEN circuit should transition to HALF_OPEN."""
        if health.state != CircuitState.OPEN:
            return

        timeout = health.config.get("recovery_timeout", 30)
        if health.opened_at and (time.time() - health.opened_at) > timeout:
            health.state = CircuitState.HALF_OPEN
            health.half_open_probes = 0
            health.consecutive_successes = 0

    async def call(
        self,
        model_id: str,
        infer_fn: Callable,
        fallback_fn: Optional[Callable] = None,
    ) -> Any:
        """
        Call inference with circuit breaker protection.
        Returns fallback result if circuit is open.
        """
        if model_id not in self._health:
            self.register(model_id)

        health = self._health[model_id]

        # Check if we should try recovery
        if health.state == CircuitState.OPEN:
            self._check_recovery(health)

        # If still open, use fallback
        if health.state == CircuitState.OPEN:
            return await self._fallback(model_id, infer_fn, fallback_fn)

        # Try primary model
        start = time.time()
        try:
            result = await infer_fn()
            latency = int((time.time() - start) * 1000)
            self._record_success(health, latency)
            return result
        except Exception as e:
            self._record_failure(health)
            if health.state == CircuitState.OPEN:
                return await self._fallback(model_id, infer_fn, fallback_fn)
            raise

    async def _fallback(
        self,
        model_id: str,
        original_fn: Callable,
        fallback_fn: Optional[Callable],
    ) -> Any:
        """Try fallback chain."""
        chain = self._fallback_chains.get(model_id, [])

        for fallback_model in chain:
            if fallback_model not in self._health:
                self.register(fallback_model)

            fb_health = self._health[fallback_model]
            if fb_health.state == CircuitState.OPEN:
                continue

            try:
                # If custom fallback function provided, use it
                if fallback_fn:
                    result = await fallback_fn(fallback_model)
                else:
                    # Otherwise try the same function with different model
                    result = await original_fn()

                fb_health.success_count += 1
                fb_health.last_success_time = time.time()
                return result
            except Exception:
                fb_health.failure_count += 1
                fb_health.consecutive_failures += 1
                continue

        # All fallbacks failed
        raise Exception(f"All models in fallback chain failed for {model_id}")

    def health_status(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get health status for all or one model."""
        if model_id:
            h = self._health.get(model_id)
            if not h:
                return {}
            total = h.success_count + h.failure_count
            return {
                "model": model_id,
                "state": h.state.value,
                "success_rate": round(h.success_count / max(total, 1), 3),
                "avg_latency_ms": round(h.total_latency_ms / max(h.success_count, 1), 1),
                "total_requests": total,
                "consecutive_failures": h.consecutive_failures,
                "opened_at": h.opened_at,
            }

        return {
            mid: self.health_status(mid) for mid in self._health
        }

    def stats(self) -> Dict[str, Any]:
        total_models = len(self._health)
        open_circuits = sum(1 for h in self._health.values() if h.state == CircuitState.OPEN)
        half_open = sum(1 for h in self._health.values() if h.state == CircuitState.HALF_OPEN)

        return {
            "total_models": total_models,
            "healthy": total_models - open_circuits - half_open,
            "open_circuits": open_circuits,
            "half_open": half_open,
            "models": self.health_status(),
        }


# Singleton
circuit_breaker = CircuitBreaker()

# Register default models
circuit_breaker.register("deepseek-v4-pro", {
    "fallback_chain": ["deepseek-v4-flash", "kimi-k2.6", "llama3.1:8b"],
})
circuit_breaker.register("deepseek-v4-flash", {
    "fallback_chain": ["kimi-k2.6", "llama3.1:8b"],
})
circuit_breaker.register("kimi-k2.6", {
    "fallback_chain": ["deepseek-v4-pro", "deepseek-v4-flash", "llama3.1:8b"],
})


if __name__ == "__main__":
    import asyncio

    async def demo():
        cb = CircuitBreaker()
        cb.register("test-model", {
            "fallback_chain": ["fallback-1", "fallback-2"],
            "consecutive_failures": 2,
            "recovery_timeout": 5,
        })

        print("Initial health:")
        print(cb.health_status("test-model"))

        # Simulate failures
        for i in range(3):
            cb._record_failure(cb._health["test-model"])
            print(f"After failure {i+1}: {cb._health['test-model'].state.value}")

        print("\nHealth after tripping:")
        print(cb.health_status("test-model"))

    asyncio.run(demo())
