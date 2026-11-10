"""Fail-closed, scoped exact-response cache (legacy module/class names retained).

Unscoped legacy calls always bypass caching. Similarity alone cannot establish
authorization, model/prompt/source freshness or equivalent conversation state,
so semantic and hash-embedding reuse is disabled. The host must validate current
authorization and supply complete revisions before every lookup or write.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class CacheContext:
    """Host-owned namespace; these labels are not authorization credentials.

    request_revision covers all generation inputs besides query: conversation,
    tool state, sampling options and any other context affecting the response.
    model_revision must identify immutable weights/provider revision, not only
    a mutable model alias. source_revision must change on edits or deletion.
    """
    tenant_id: str
    principal_id: str
    authorization_revision: str
    policy_revision: str
    model_id: str
    model_revision: str
    prompt_revision: str
    source_revision: str
    request_revision: str

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"CacheContext.{field.name} must be non-empty")


@dataclass
class CacheEntry:
    response_text: str
    model: str
    cost_usd: float
    timestamp: float
    hit_count: int = 0
    ttl_seconds: float = 3600.0


class SemanticCache:
    """Compatibility name for a bounded exact cache requiring explicit context."""
    MAX_ENTRIES = 1000

    def __init__(self, ollama_url: str = "http://localhost:11434", *, clock=None):
        # ollama_url remains accepted for compatibility; no embedding service is used.
        self._clock = clock or time.monotonic
        self._cache: Dict[str, CacheEntry] = {}
        self.clear()

    def _key(self, query: str, context: Optional[CacheContext]) -> Optional[str]:
        if not isinstance(context, CacheContext):
            return None
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        payload = json.dumps(
            [asdict(context), query], ensure_ascii=False,
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _expire(self):
        now = self._clock()
        expired = [
            key for key, entry in self._cache.items()
            if now - entry.timestamp >= entry.ttl_seconds
        ]
        for key in expired:
            del self._cache[key]
        self._stats["expirations"] += len(expired)

    async def get(
        self, query: str, *, context: Optional[CacheContext] = None,
    ) -> Optional[Tuple[str, float, Dict[str, Any]]]:
        """Return the compatible tuple only for an exact query and full context."""
        key = self._key(query, context)
        if key is None:
            self._stats["bypasses"] += 1
            self._stats["misses"] += 1
            return None
        self._expire()
        entry = self._cache.get(key)
        if entry is None:
            self._stats["misses"] += 1
            return None
        entry.hit_count += 1
        self._stats["hits"] += 1
        return entry.response_text, 1.0, {
            "cached": True,
            "match": "exact_scoped",
            "similarity": 1.0,
            "original_model": entry.model,
            "original_cost_usd": entry.cost_usd,
            "saved_cost_usd": 0.0,
            "savings_measured": False,
            "cache_hit_count": entry.hit_count,
        }

    async def set(
        self,
        query: str,
        response_text: str,
        model: str,
        cost_usd: float,
        ttl_seconds: float = 3600.0,
        *,
        context: Optional[CacheContext] = None,
    ) -> None:
        """Store only with a complete scope and matching actual model identity."""
        key = self._key(query, context)
        if key is None or model != context.model_id:
            self._stats["bypassed_writes"] += 1
            return
        if (
            isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(ttl_seconds) or ttl_seconds <= 0
        ):
            raise ValueError("ttl_seconds must be positive and finite")
        if (
            isinstance(cost_usd, bool) or not isinstance(cost_usd, (int, float))
            or not math.isfinite(cost_usd) or cost_usd < 0
        ):
            raise ValueError("cost_usd must be nonnegative and finite")
        if not isinstance(response_text, str):
            raise TypeError("response_text must be a string")
        self._expire()
        if key not in self._cache and len(self._cache) >= self.MAX_ENTRIES:
            oldest = min(self._cache, key=lambda item: self._cache[item].timestamp)
            del self._cache[oldest]
            self._stats["evictions"] += 1
        self._cache[key] = CacheEntry(
            response_text, model, cost_usd, self._clock(),
            ttl_seconds=ttl_seconds,
        )

    def stats(self) -> Dict[str, Any]:
        self._expire()
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / max(total, 1)
        return {
            "mode": "exact_scoped",
            "legacy_unscoped": "bypass",
            "entries": len(self._cache),
            "hit_rate": round(hit_rate * 100, 2),
            **self._stats,
            # Compatibility keys, explicitly not measurements or estimates.
            "saved_cost_usd": 0.0,
            "saved_latency_ms": 0,
            "savings_measured": False,
            "efficiency": f"{hit_rate * 100:.1f}% hit rate; savings not measured",
        }

    def clear(self) -> None:
        self._cache.clear()
        self._stats = {
            "hits": 0, "misses": 0, "evictions": 0, "expirations": 0,
            "bypasses": 0, "bypassed_writes": 0,
        }


cache = SemanticCache()


async def cached_chat_completion(
    query: str,
    generate_fn,
    model: str = "unknown",
    ttl: float = 3600.0,
    *,
    context: Optional[CacheContext] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Compatibility wrapper. Without host context, always run generate_fn.

    The host must only enable caching for pure, cacheable inference after current
    authorization checks. Tool execution and side effects are never made safe by
    adding context labels. Actual model identity must be present and match scope.
    """
    scoped = context if isinstance(context, CacheContext) and model == context.model_id else None
    cached = await cache.get(query, context=scoped)
    if cached:
        text, _, meta = cached
        return text, {**meta, "source": "exact_scoped_cache"}
    result = await generate_fn()
    response_text = result.get("text", str(result)) if isinstance(result, dict) else str(result)
    if isinstance(result, dict):
        actual_model = result.get("model") or result.get("primary_model")
        if actual_model == model and not result.get("care_flag") and not result.get("error"):
            await cache.set(
                query, response_text, actual_model, result.get("cost_usd", 0.0),
                ttl, context=scoped,
            )
    return response_text, {"source": "inference", "cached": False}
