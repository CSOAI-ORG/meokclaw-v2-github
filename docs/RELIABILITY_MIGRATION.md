# Reliability helpers: integration and migration

Reviewed baseline: `0409409be9e0540ab05f9a7794e11ec218ee9300` (2026-11-10).

These changes repair library behavior and add executable offline regression tests. They do not establish that the full API, Docker application or linked public demo is deployed or working.

## What is verified

| Helper | Repaired behavior | Required host integration |
| --- | --- | --- |
| Prompt registry | Success and failure both update the success rate; prompt-wide averages are weighted by calls. | Record actual outcome/cost/latency against the rendered version ID. Metrics remain process-local. |
| Circuit breaker | An open primary is never rerun under a fallback label; explicit dispatch updates the selected fallback's own health, latency and recovery. | Provide an authorized `fallback_fn(model_id)`. Without it, an open circuit raises `FallbackUnavailableError`. |
| Structured output | Requested model and full system/schema instructions reach an explicit adapter. Actual returned model is reported. Unknown functions and invalid arguments are rejected. | Supply an async adapter accepting keyword `model` and `messages`. The unconfigured singleton raises `StructuredBackendNotConfigured`. |
| Legacy cache | Exact response reuse requires a full host-owned context; old unscoped calls bypass. Every namespace/revision change misses. | Revalidate authorization and source freshness before constructing each context. Cache only pure, explicitly cacheable inference. |

## Structured output

`DualBrainOrchestrator.think(task_text, context=None)` accepts neither an explicit model nor a system prompt. The previous wrapper ignored both while labeling results as the requested model. It must not be used as an implicit adapter.

Construct `StructuredOutputEngine(completion_fn=host_completion)`. The host adapter must authorize the requested model and return a mapping containing:

- `text`: the actual response text.
- `model`: the provider's actual model identity, including any host-authorized fallback.
- `cost_usd` and `latency_ms`: finite nonnegative values actually supplied/measured by that adapter. Missing measurements are an adapter configuration error, not fabricated zero-cost success.

The engine forwards the requested model unchanged. It appends the complete Pydantic JSON Schema to the supplied system prompt. Provider errors propagate; only invalid response JSON/schema triggers correction attempts. The historical `max_retries` parameter still means maximum total attempts; `retry_count` now reports retries after the initial attempt.

JSON is parsed strictly: no duplicate keys, non-finite numbers (including NaN/Infinity and exponent overflow such as `1e999` at any depth), quote replacement, trailing-comma repair or surrounding prose. A single fenced JSON value is accepted. Pydantic validation runs in strict mode, so callers relying on string-to-number coercion must adapt their schema or producer.

`function_call` validates a proposal and never executes it. It accepts the original parameter-property map (with per-property `required: true`) or a JSON Schema 2020-12 object schema. Unknown function names, duplicate definitions, invalid enums, wrong types, missing required fields and extra top-level call fields are rejected. Parameter `additionalProperties` defaults to false; an explicitly supplied true/schema remains the caller's policy. Nested constraints and local references are validated. External schema references and nonlocal schema identifiers are disabled; validation cannot fetch remote schema resources.

## Cache migration

The legacy `SemanticCache` name and positional `get`, `set`, `cached_chat_completion` call shapes remain available. No context means cache miss/no write. No embeddings, similarity search or hash-vector fallback is used.

Use keyword `context=CacheContext(...)` with all of these host-controlled fields:

| Field | Must change when |
| --- | --- |
| `tenant_id`, `principal_id` | Requesting tenant or user changes. |
| `authorization_revision` | Current permissions, ownership or access membership changes. |
| `policy_revision` | Serving/caching policy changes. |
| `model_id`, `model_revision` | Actual model or immutable provider/weight revision changes. |
| `prompt_revision` | System/developer/template instructions change. |
| `source_revision` | Referenced data changes or is deleted. |
| `request_revision` | Conversation history, tool state, generation options or any other non-query input changes. |

These fields are namespace labels, not proof of permission. Never accept them directly from an untrusted caller. If current authorization, immutable model identity, source state or complete request state cannot be established, omit context and generate afresh. Do not use this cache to replay tool actions or other side effects.

Queries match exactly, including case and whitespace, within that context. Actual model identity must match `context.model_id` at write time; the wrapper requires returned `model` or `primary_model` and bypasses care/error responses. Entries expire at the TTL boundary, and capacity remains bounded. Replacing an existing key does not evict an unrelated entry. Cache state is memory-only and clears on restart.

The old `saved_cost_usd` and `saved_latency_ms` keys remain zero for compatibility with `savings_measured: false`. An entry's original generation price is not a measured counterfactual saving; hit rate is reported separately.

## What is not wired in this repository

At the reviewed baseline, `src/api.py` imports these helpers but its inference routes do not call `structured_engine.infer`, `circuit_breaker.call`, `cached_chat_completion`, or `prompt_registry.record_metrics`. Cache endpoints only expose statistics and clear entries; prompt endpoints register/list/report versions. Passing these module tests does not turn those capabilities on in a deployed conversation.

Full runtime packaging also requires a separate verified repair:

- `src/dual_brain_orchestrator.py` imports `reflection_engine`, which is absent from the tracked tree.
- `src/ollama_client.py` imports `aiohttp`, which the baseline full requirements file does not declare.
- `Dockerfile.api` runs `dual_brain_api.py`, which is absent; the tracked API is `src/api.py`.
- The Compose frontend build context `./meokclaw-v2` is absent from this repository.

No substitute reflection/approval implementation is fabricated here. The full app must be reconciled with its real host implementation before enabling these helpers in production. Circuit health remains in-process, not a distributed limiter. `latency_threshold_ms` and `half_open_max_probes` are reserved legacy settings, not enforced gates; recovery uses the configured fixed timeout.

## Tests and CI

```bash
python -m pip install -r requirements-reliability.txt
python -m unittest discover -s tests -p 'test_reliability.py' -v
```

The 30 focused tests cover the reproduced arithmetic/dispatch regressions, strict function proposals, provider metadata/error handling, tenant/principal/revision separation, TTL/capacity and wrapper bypass. Socket connection APIs are blocked during the suite. CI runs the suite on Python 3.11 and 3.12 and compiles the source. No API keys, model calls, GPU, service container or live endpoint is required.

`tests/test_e2e.py` is a separate legacy/live integration suite. Its historical README results are not evidence that it passes on this checkout; it is deliberately not included in offline CI.
