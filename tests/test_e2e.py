#!/usr/bin/env python3
"""
E2E Test Suite — Full pipeline validation.
Tests router, orchestrator, API, and fallback chains.
"""
import asyncio
import sys
import time
sys.path.insert(0, "/Users/nicholas/clawd/sovereign-temple")

from dual_brain_router import CorpusCallosumRouter, Hemisphere
from dual_brain_orchestrator import DualBrainOrchestrator


class TestResult:
    def __init__(self, name: str, passed: bool, latency_ms: float = 0, error: str = "", details: dict = None):
        self.name = name
        self.passed = passed
        self.latency_ms = latency_ms
        self.error = error
        self.details = details or {}


async def run_tests():
    results: list[TestResult] = []
    orch = DualBrainOrchestrator()

    # ── Router Tests ──
    print("=" * 60)
    print("ROUTER TESTS")
    print("=" * 60)

    router = CorpusCallosumRouter()
    router_tests = [
        ("greeting", "Hello!", Hemisphere.LEFT, "llama3.1-vast"),
        ("greeting2", "How are you?", Hemisphere.LEFT, "llama3.1-vast"),
        ("coding", "Write a Python function to parse JSON", Hemisphere.LEFT, "deepseek-v4-flash"),
        ("debug", "Debug this TypeError in React", Hemisphere.LEFT, "deepseek-v4-flash"),
        ("creative", "Write a poem about clay soil", Hemisphere.RIGHT, "deepseek-v4-pro"),
        ("reasoning", "Why is thermal mass better than insulation", Hemisphere.RIGHT, "deepseek-v4-pro"),
        ("governance", "Design a council proposal for haulage", Hemisphere.BOTH, "deepseek-v4-pro"),
        ("care", "I want to hurt myself", Hemisphere.CARE, "care_membrane"),
        ("both_trigger", "What does everyone think about this", Hemisphere.BOTH, "deepseek-v4-pro"),
    ]

    for name, text, expected_hemi, expected_primary in router_tests:
        start = time.perf_counter()
        try:
            r = router.analyze(text)
            latency = (time.perf_counter() - start) * 1000
            passed = r.hemisphere == expected_hemi and expected_primary in r.primary_model
            results.append(TestResult(
                f"router.{name}", passed, latency,
                details={"hemisphere": r.hemisphere.value, "primary": r.primary_model}
            ))
            status = "✅" if passed else "❌"
            print(f"  {status} {name:20s} | {r.hemisphere.value:6s} | {r.primary_model:25s} | {latency:.2f}ms")
        except Exception as exc:
            results.append(TestResult(f"router.{name}", False, error=str(exc)))
            print(f"  ❌ {name:20s} | ERROR: {exc}")

    # ── Orchestrator Tests ──
    print("\n" + "=" * 60)
    print("ORCHESTRATOR TESTS")
    print("=" * 60)

    orch_tests = [
        ("greeting", "Hello!", "left", 30),
        ("coding", "Write a Python one-liner to flatten a list", "left", 60),
        ("reasoning", "Why does lime improve clay drainage?", "right", 60),
        ("governance", "Design a 3-person voting council", "both", 90),
    ]

    for name, text, expected_hemi, timeout_sec in orch_tests:
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(orch.think(text), timeout=timeout_sec)
            latency = (time.perf_counter() - start) * 1000
            passed = (
                result["hemisphere"] == expected_hemi
                and len(result["text"]) > 10
                and result["latency_ms"] > 0
            )
            results.append(TestResult(
                f"orch.{name}", passed, latency,
                details={"hemisphere": result["hemisphere"], "model": result["primary_model"], "cost": result["cost_usd"]}
            ))
            status = "✅" if passed else "❌"
            print(f"  {status} {name:20s} | {result['hemisphere']:6s} | {result['primary_model'][:30]:30s} | ${result['cost_usd']:.6f} | {latency:.0f}ms")
        except Exception as exc:
            results.append(TestResult(f"orch.{name}", False, error=str(exc)))
            print(f"  ❌ {name:20s} | ERROR: {exc}")

    await orch._client.close()

    # ── Fallback Tests ──
    print("\n" + "=" * 60)
    print("FALLBACK TESTS")
    print("=" * 60)

    # Test Ollama fallback directly
    from ollama_client import get_vast_ollama
    ollama = get_vast_ollama()
    try:
        start = time.perf_counter()
        r = await ollama.chat_completion("llama3.1:8b", [{"role": "user", "content": "Say hi"}], max_tokens=10)
        latency = (time.perf_counter() - start) * 1000
        passed = len(r.text) > 0
        results.append(TestResult("fallback.ollama_vast", passed, latency, details={"text": r.text[:30]}))
        status = "✅" if passed else "❌"
        print(f"  {status} ollama_vast          | {r.text[:30]:30s} | {latency:.0f}ms")
    except Exception as exc:
        results.append(TestResult("fallback.ollama_vast", False, error=str(exc)))
        print(f"  ❌ ollama_vast          | ERROR: {exc}")

    # ── API Tests ──
    print("\n" + "=" * 60)
    print("API TESTS")
    print("=" * 60)

    import httpx
    async with httpx.AsyncClient() as client:
        # Health
        try:
            r = await client.get("http://localhost:3201/health")
            passed = r.status_code == 200 and r.json()["status"] == "healthy"
            results.append(TestResult("api.health", passed))
            print(f"  {'✅' if passed else '❌'} health               | {r.status_code}")
        except Exception as exc:
            results.append(TestResult("api.health", False, error=str(exc)))
            print(f"  ❌ health               | ERROR: {exc}")

        # Chat
        try:
            r = await client.post(
                "http://localhost:3201/api/dual-brain",
                json={"message": "Say hello"},
                timeout=60,
            )
            data = r.json()
            passed = r.status_code == 200 and len(data.get("text", "")) > 0
            results.append(TestResult("api.chat", passed, details={"hemisphere": data.get("hemisphere")}))
            print(f"  {'✅' if passed else '❌'} chat                 | {data.get('hemisphere'):6s} | {data.get('primary_model', 'unknown')[:25]:25s}")
        except Exception as exc:
            results.append(TestResult("api.chat", False, error=str(exc)))
            print(f"  ❌ chat                 | ERROR: {exc}")

        # Router stats
        try:
            r = await client.get("http://localhost:3201/api/router-stats")
            passed = r.status_code == 200 and "total_decisions" in r.json()
            results.append(TestResult("api.router_stats", passed))
            print(f"  {'✅' if passed else '❌'} router_stats         | {r.status_code}")
        except Exception as exc:
            results.append(TestResult("api.router_stats", False, error=str(exc)))
            print(f"  ❌ router_stats         | ERROR: {exc}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print(f"  Total: {len(results)} | ✅ Passed: {passed} | ❌ Failed: {failed}")
    if failed > 0:
        print("\n  Failures:")
        for r in results:
            if not r.passed:
                print(f"    ❌ {r.name}: {r.error}")
    avg_latency = sum(r.latency_ms for r in results if r.passed) / max(1, passed)
    print(f"\n  Avg latency (passed): {avg_latency:.0f}ms")
    return passed, failed


if __name__ == "__main__":
    passed, failed = asyncio.run(run_tests())
    sys.exit(0 if failed == 0 else 1)
