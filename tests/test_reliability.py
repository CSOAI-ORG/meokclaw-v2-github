"""Offline regression tests; no model service, key, network or live API required."""
import asyncio
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The legacy registry creates its data directory on import. Isolate that side
# effect from both the repository and the user's prompt store.
_import_dir = tempfile.TemporaryDirectory()
_previous_cwd = os.getcwd()
try:
    os.chdir(_import_dir.name)
    from prompt_registry import PromptRegistry
finally:
    os.chdir(_previous_cwd)
    _import_dir.cleanup()

from circuit_breaker import CircuitBreaker, CircuitState, FallbackUnavailableError
from semantic_cache import CacheContext, SemanticCache, cached_chat_completion
import semantic_cache
from structured_output import (
    StructuredOutputEngine, StructuredBackendNotConfigured,
    StructuredBackendProtocolError,
)
from pydantic import BaseModel


def _no_network(*args, **kwargs):
    raise AssertionError("Network access is forbidden in reliability tests")


_network_patches = []


def setUpModule():
    for target in ("socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex"):
        guard = patch(target, side_effect=_no_network)
        guard.start()
        _network_patches.append(guard)


def tearDownModule():
    for guard in reversed(_network_patches):
        guard.stop()


class PromptMetricsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        guard = patch.object(PromptRegistry, "DATA_DIR", Path(self.tmp.name))
        guard.start()
        self.addCleanup(guard.stop)
        self.registry = PromptRegistry()

    def test_failure_then_success_recovers_rate(self):
        self.registry.record_metrics("v", 1, 10, 2, success=False)
        self.registry.record_metrics("v", 3, 30, 4, success=True)
        metric = self.registry._metrics["v"]
        self.assertEqual(metric.calls, 2)
        self.assertEqual(metric.success_rate, 0.5)
        self.assertEqual(metric.avg_cost_usd, 2)
        self.assertEqual(metric.avg_latency_ms, 20)

    def test_mixed_results_count_every_call(self):
        outcomes = [True, False, True, False, False, True, True]
        for success in outcomes:
            self.registry.record_metrics("v", 0, 0, 0, success)
        self.assertAlmostEqual(self.registry._metrics["v"].success_rate, 4 / 7)

    def test_prompt_totals_weight_versions_by_calls(self):
        versions = [
            self.registry.register("prompt", "system", "template") for _ in range(3)
        ]
        for _ in range(3):
            self.registry.record_metrics(versions[0].id, 2, 100, 5)
        self.registry.record_metrics(versions[1].id, 10, 500, 5)
        result = self.registry.stats("prompt")
        self.assertEqual(result["total_calls"], 4)
        self.assertEqual(result["avg_cost_usd"], 4)
        self.assertEqual(result["avg_latency_ms"], 200)

    def test_no_calls_has_zero_means(self):
        self.registry.register("unused", "system", "template")
        self.assertEqual(self.registry.stats("unused")["avg_cost_usd"], 0)


class CircuitTests(unittest.IsolatedAsyncioTestCase):
    async def test_open_primary_without_dispatcher_is_never_retried(self):
        cb = CircuitBreaker()
        cb.register("primary", {"fallback_chain": ["fallback"]})
        h = cb._health["primary"]
        h.state, h.opened_at = CircuitState.OPEN, time.time()
        calls = []

        async def original():
            calls.append("primary")
            return "must not run"

        with self.assertRaisesRegex(FallbackUnavailableError, "requires fallback_fn"):
            await cb.call("primary", original)
        self.assertEqual(calls, [])
        self.assertNotIn("fallback", cb._health)

    async def test_trip_without_dispatcher_preserves_cause_and_accounting(self):
        cb = CircuitBreaker()
        cb.register("primary", {"consecutive_failures": 1, "fallback_chain": ["fallback"]})
        calls = []

        async def original():
            calls.append("primary")
            raise ValueError("primary failure")

        with self.assertRaises(FallbackUnavailableError) as caught:
            await cb.call("primary", original)
        self.assertIsInstance(caught.exception.__context__, ValueError)
        self.assertEqual(calls, ["primary"])
        self.assertEqual(cb.health_status("primary")["total_requests"], 1)
        self.assertNotIn("fallback", cb._health)

    async def test_dispatcher_receives_fallback_id_and_records_latency(self):
        cb = CircuitBreaker()
        cb.register("primary", {"fallback_chain": ["primary", "fallback", "fallback"]})
        cb._health["primary"].state = CircuitState.OPEN
        cb._health["primary"].opened_at = time.time()
        calls = []

        async def original():
            self.fail("Primary must not be called")

        async def dispatch(model):
            calls.append(model)
            return {"model": model}

        with patch("circuit_breaker.time.perf_counter", side_effect=[0, 0.01]):
            result = await cb.call("primary", original, dispatch)
        self.assertEqual(result["model"], "fallback")
        self.assertEqual(calls, ["fallback"])
        self.assertEqual(cb._health["fallback"].success_count, 1)
        self.assertEqual(cb._health["fallback"].total_latency_ms, 10)
        self.assertEqual(cb._health["primary"].success_count, 0)

    async def test_fallback_failures_trip_its_own_circuit_and_continue(self):
        cb = CircuitBreaker()
        cb.register("primary", {"fallback_chain": ["bad", "good"]})
        cb.register("bad", {"consecutive_failures": 1})
        cb._health["primary"].state = CircuitState.OPEN
        cb._health["primary"].opened_at = time.time()
        calls = []

        async def original():
            self.fail("Unexpected primary retry")

        async def dispatch(model):
            calls.append(model)
            if model == "bad":
                raise OSError("offline")
            return "ok"

        self.assertEqual(await cb.call("primary", original, dispatch), "ok")
        self.assertEqual(calls, ["bad", "good"])
        self.assertEqual(cb._health["bad"].state, CircuitState.OPEN)
        self.assertEqual(cb._health["bad"].failure_count, 1)
        self.assertEqual(cb._health["good"].success_count, 1)

    async def test_open_fallback_can_recover_on_expired_timeout(self):
        cb = CircuitBreaker()
        cb.register("primary", {"fallback_chain": ["recovering"]})
        cb.register("recovering", {"half_open_success_needed": 1})
        for model, age in (("primary", 0), ("recovering", 60)):
            cb._health[model].state = CircuitState.OPEN
            cb._health[model].opened_at = time.time() - age

        async def original():
            self.fail("Unexpected primary retry")

        async def dispatch(model):
            return model

        self.assertEqual(await cb.call("primary", original, dispatch), "recovering")
        self.assertEqual(cb._health["recovering"].state, CircuitState.CLOSED)

    async def test_closed_primary_works_without_dispatcher(self):
        cb = CircuitBreaker()

        async def original():
            return "primary"

        self.assertEqual(await cb.call("primary", original), "primary")
        self.assertEqual(cb._health["primary"].success_count, 1)


class Payload(BaseModel):
    count: int
    labels: list[str]


class FakeCompletion:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def __call__(self, *, model, messages):
        self.calls.append({"model": model, "messages": messages})
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return {
            "text": response, "model": "provider/actual-model",
            "cost_usd": 0.25, "latency_ms": 10,
        }


FUNCTIONS = [{
    "name": "get_weather",
    "description": "Lookup weather",
    "parameters": {
        "city": {"type": "string", "required": True},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
}]


class StructuredTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_and_system_prompt_reach_adapter_and_actual_model_returns(self):
        backend = FakeCompletion('{"count": 2, "labels": ["a"]}')
        result = await StructuredOutputEngine(backend).infer(
            "count labels", Payload, model="requested-model", system_prompt="Keep private scope",
        )
        self.assertTrue(result.parsed)
        self.assertEqual(result.model, "provider/actual-model")
        self.assertEqual(backend.calls[0]["model"], "requested-model")
        self.assertTrue(backend.calls[0]["messages"][0]["content"].startswith("Keep private scope\n"))
        self.assertIn('"items": {"type": "string"}', backend.calls[0]["messages"][0]["content"])
        self.assertEqual(result.retry_count, 0)

    async def test_missing_adapter_fails_before_provider_import(self):
        with self.assertRaises(StructuredBackendNotConfigured):
            await StructuredOutputEngine().infer("count", Payload)
        self.assertNotIn("dual_brain_orchestrator", sys.modules)

    async def test_invalid_output_retries_with_feedback_and_actual_totals(self):
        backend = FakeCompletion('{"count": "2", "labels": []}', '{"count": 2, "labels": []}')
        result = await StructuredOutputEngine(backend).infer("count", Payload)
        self.assertTrue(result.parsed)
        self.assertEqual(result.retry_count, 1)
        self.assertEqual(result.cost_usd, 0.5)
        self.assertEqual(result.latency_ms, 20)
        self.assertIn("failed validation", backend.calls[1]["messages"][1]["content"])

    async def test_duplicate_keys_nonfinite_and_malformed_json_are_not_repaired(self):
        for response in [
            '{"count": 1, "count": 2, "labels": []}',
            '{"count": NaN, "labels": []}',
            "{'count': 2, 'labels': []}",
            '{"count": 2, "labels": [],}',
            'prose {"count": 2, "labels": []}',
        ]:
            with self.subTest(response=response):
                backend = FakeCompletion(response)
                result = await StructuredOutputEngine(backend).infer("count", Payload, max_retries=1)
                self.assertFalse(result.parsed)
                self.assertEqual(result.retry_count, 0)
                self.assertEqual(len(backend.calls), 1)

    async def test_provider_failure_is_not_retried_as_json_failure(self):
        backend = FakeCompletion(OSError("offline"))
        with self.assertRaises(OSError):
            await StructuredOutputEngine(backend).infer("count", Payload)
        self.assertEqual(len(backend.calls), 1)

    async def test_adapter_must_report_model_and_finite_metrics(self):
        for value in [
            {"text": "{}", "cost_usd": 0, "latency_ms": 0},
            {"text": "{}", "model": "model", "cost_usd": float("nan"), "latency_ms": 0},
            {"text": "{}", "model": "model", "cost_usd": 0, "latency_ms": -1},
        ]:
            async def backend(**kwargs):
                return value
            with self.assertRaises(StructuredBackendProtocolError):
                await StructuredOutputEngine(backend).infer("count", Payload)

    async def test_legacy_function_schema_valid_call(self):
        backend = FakeCompletion('{"function":"get_weather","arguments":{"city":"London","units":"celsius"}}')
        result = await StructuredOutputEngine(backend).function_call("weather", FUNCTIONS, model="weather-model")
        self.assertEqual(result["arguments"]["city"], "London")
        self.assertEqual(backend.calls[0]["model"], "weather-model")
        self.assertIn("get_weather", backend.calls[0]["messages"][0]["content"])

    async def test_function_rejects_unknown_missing_extra_wrong_type_and_enum(self):
        invalid = [
            {"function": "delete_all", "arguments": {}},
            {"function": "get_weather", "arguments": {}},
            {"function": "get_weather", "arguments": {"city": "London", "admin": True}},
            {"function": "get_weather", "arguments": {"city": 123}},
            {"function": "get_weather", "arguments": {"city": "London", "units": "kelvin"}},
            {"function": "get_weather", "arguments": {"city": "London"}, "execute": True},
        ]
        for response in invalid:
            with self.subTest(response=response):
                backend = FakeCompletion(json.dumps(response))
                result = await StructuredOutputEngine(backend).function_call("weather", FUNCTIONS)
                self.assertIn("error", result)
                self.assertEqual(len(backend.calls), 3)

    async def test_standard_nested_schema_and_local_refs_are_enforced(self):
        functions = [{
            "name": "measure",
            "parameters": {
                "type": "object",
                "$defs": {"positive": {"type": "integer", "minimum": 1}},
                "properties": {
                    "values": {"type": "array", "items": {"$ref": "#/$defs/positive"}, "minItems": 1}
                },
                "required": ["values"],
                "additionalProperties": False,
            },
        }]
        backend = FakeCompletion(
            '{"function":"measure","arguments":{"values":[0]}}',
            '{"function":"measure","arguments":{"values":[2]}}',
        )
        result = await StructuredOutputEngine(backend).function_call("measure", functions)
        self.assertEqual(result["arguments"]["values"], [2])
        self.assertEqual(len(backend.calls), 2)

    async def test_external_schema_ref_and_duplicate_function_fail_before_generation(self):
        backend = FakeCompletion("{}")
        engine = StructuredOutputEngine(backend)
        with self.assertRaisesRegex(ValueError, "local"):
            await engine.function_call("x", [{
                "name": "remote",
                "parameters": {"type": "object", "properties": {"x": {"$ref": "https://example.invalid/schema"}}},
            }])
        with self.assertRaisesRegex(ValueError, "unique"):
            await engine.function_call("x", FUNCTIONS + FUNCTIONS)
        self.assertEqual(backend.calls, [])


def context():
    return CacheContext(
        tenant_id="tenant-a", principal_id="user-a", authorization_revision="auth-1",
        policy_revision="policy-1", model_id="model-a", model_revision="weights-1",
        prompt_revision="prompt-1", source_revision="source-1", request_revision="dialogue-1",
    )


class CacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 100.0
        self.cache = SemanticCache(clock=lambda: self.now)
        self.ctx = context()

    async def test_legacy_calls_fail_closed_without_changing_call_shape(self):
        await self.cache.set("question", "private answer", "model-a", 1)
        self.assertIsNone(await self.cache.get("question"))
        self.assertEqual(self.cache.stats()["entries"], 0)
        self.assertEqual(self.cache.stats()["bypasses"], 1)

    async def test_exact_scoped_hit_and_no_fabricated_savings(self):
        await self.cache.set("question", "answer", "model-a", 1, context=self.ctx)
        result = await self.cache.get("question", context=self.ctx)
        self.assertEqual(result[:2], ("answer", 1.0))
        self.assertEqual(result[2]["original_cost_usd"], 1)
        self.assertFalse(result[2]["savings_measured"])
        stats = self.cache.stats()
        self.assertEqual(stats["saved_cost_usd"], 0)
        self.assertEqual(stats["saved_latency_ms"], 0)
        self.assertEqual(stats["hits"], 1)

    async def test_every_namespace_and_revision_change_is_a_miss(self):
        await self.cache.set("question", "private answer", "model-a", 1, context=self.ctx)
        for field in self.ctx.__dataclass_fields__:
            with self.subTest(field=field):
                changed = replace(self.ctx, **{field: getattr(self.ctx, field) + "-changed"})
                self.assertIsNone(await self.cache.get("question", context=changed))
        self.assertIsNone(await self.cache.get("Question", context=self.ctx))
        self.assertIsNone(await self.cache.get("similar question", context=self.ctx))

    async def test_ttl_boundary_and_expiration_reclaim(self):
        await self.cache.set("q", "a", "model-a", 0, ttl_seconds=5, context=self.ctx)
        self.now = 105
        self.assertIsNone(await self.cache.get("q", context=self.ctx))
        self.assertEqual(self.cache.stats()["entries"], 0)
        self.assertEqual(self.cache.stats()["expirations"], 1)

    async def test_actual_model_mismatch_cannot_be_stored(self):
        await self.cache.set("q", "wrong model", "fallback-model", 0, context=self.ctx)
        self.assertIsNone(await self.cache.get("q", context=self.ctx))
        self.assertEqual(self.cache.stats()["bypassed_writes"], 1)

    async def test_bounded_cache_replacement_does_not_evict_another_entry(self):
        self.cache.MAX_ENTRIES = 2
        await self.cache.set("q1", "a1", "model-a", 0, context=self.ctx)
        self.now += 1
        await self.cache.set("q2", "a2", "model-a", 0, context=self.ctx)
        self.now += 1
        await self.cache.set("q2", "updated", "model-a", 0, context=self.ctx)
        self.assertEqual(self.cache.stats()["evictions"], 0)
        self.assertIsNotNone(await self.cache.get("q1", context=self.ctx))
        self.now += 1
        await self.cache.set("q3", "a3", "model-a", 0, context=self.ctx)
        self.assertIsNone(await self.cache.get("q1", context=self.ctx))
        self.assertEqual(self.cache.stats()["evictions"], 1)

    async def test_context_and_nonfinite_ttl_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.ctx, authorization_revision="")
        for ttl in [0, -1, float("nan"), float("inf")]:
            with self.subTest(ttl=ttl):
                with self.assertRaises(ValueError):
                    await self.cache.set("q", "a", "model-a", 0, ttl, context=self.ctx)

    async def test_wrapper_bypasses_legacy_and_checks_actual_model(self):
        calls = []

        async def generate():
            calls.append(1)
            return {"text": "answer", "model": "model-a", "cost_usd": 0.1}

        with patch.object(semantic_cache, "cache", self.cache):
            for _ in range(2):
                _, meta = await cached_chat_completion("q", generate, "model-a")
                self.assertFalse(meta["cached"])
            self.assertEqual(len(calls), 2)
            await cached_chat_completion("q", generate, "model-a", context=self.ctx)
            _, meta = await cached_chat_completion("q", generate, "model-a", context=self.ctx)
            self.assertTrue(meta["cached"])
            self.assertEqual(len(calls), 3)

    async def test_wrapper_does_not_cache_care_or_model_mismatch(self):
        with patch.object(semantic_cache, "cache", self.cache):
            for result in [
                {"text": "care", "model": "model-a", "care_flag": True},
                {"text": "fallback", "primary_model": "another-model"},
                {"text": "missing model"},
            ]:
                async def generate():
                    return result
                await cached_chat_completion("q", generate, "model-a", context=self.ctx)
                self.assertIsNone(await self.cache.get("q", context=self.ctx))


if __name__ == "__main__":
    unittest.main()
