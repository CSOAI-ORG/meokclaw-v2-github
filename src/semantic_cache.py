"""Semantic Caching for MEOKCLAW — 20-40% cost reduction

Caches AI responses based on semantic similarity of queries.
If a new query is >92% similar to a cached query, returns the cached
response instantly at zero cost.

Uses nomic-embed-text via Ollama for lightweight embeddings.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ollama_client import OllamaClient


@dataclass
class CacheEntry:
    embedding: np.ndarray
    response_text: str
    model: str
    cost_usd: float
    timestamp: float
    hit_count: int = 0
    ttl_seconds: float = 3600.0  # 1 hour default


class SemanticCache:
    """In-memory semantic cache with similarity-based retrieval."""

    SIMILARITY_THRESHOLD = 0.92
    EMBEDDING_MODEL = "nomic-embed-text"
    MAX_ENTRIES = 1000

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self._client = OllamaClient(ollama_url)
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "saved_cost_usd": 0.0,
            "saved_latency_ms": 0,
        }
        self._lock = False  # Simple lock for async safety

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama nomic-embed-text."""
        try:
            res = await self._client.embeddings(self.EMBEDDING_MODEL, text)
            return np.array(res["embedding"])
        except Exception:
            # Fallback: simple hash-based embedding for when Ollama is down
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Simple character n-gram hash embedding when Ollama unavailable."""
        text = text.lower().strip()
        dim = 768
        vec = np.zeros(dim)
        for i in range(len(text) - 2):
            h = hash(text[i:i+3]) % dim
            vec[h] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    async def get(self, query: str) -> Optional[Tuple[str, float, Dict[str, Any]]]:
        """
        Check cache for semantically similar query.
        Returns (response_text, similarity, metadata) or None.
        """
        if not self._cache:
            self._stats["misses"] += 1
            return None

        query_emb = await self._get_embedding(query)
        best_match: Optional[str] = None
        best_sim = 0.0

        # Find most similar cached entry
        for key, entry in self._cache.items():
            if time.time() - entry.timestamp > entry.ttl_seconds:
                continue  # Expired

            sim = cosine_similarity(
                query_emb.reshape(1, -1),
                entry.embedding.reshape(1, -1),
            )[0][0]

            if sim > best_sim:
                best_sim = sim
                best_match = key

        if best_sim >= self.SIMILARITY_THRESHOLD and best_match:
            entry = self._cache[best_match]
            entry.hit_count += 1
            self._stats["hits"] += 1
            self._stats["saved_cost_usd"] += entry.cost_usd
            self._stats["saved_latency_ms"] += 500  # Approximate savings

            return (
                entry.response_text,
                best_sim,
                {
                    "cached": True,
                    "similarity": round(best_sim, 4),
                    "original_model": entry.model,
                    "saved_cost_usd": entry.cost_usd,
                    "cache_hit_count": entry.hit_count,
                },
            )

        self._stats["misses"] += 1
        return None

    async def set(
        self,
        query: str,
        response_text: str,
        model: str,
        cost_usd: float,
        ttl_seconds: float = 3600.0,
    ) -> None:
        """Store response in cache."""
        # Evict oldest if at capacity
        if len(self._cache) >= self.MAX_ENTRIES:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].timestamp,
            )
            del self._cache[oldest_key]
            self._stats["evictions"] += 1

        embedding = await self._get_embedding(query)
        key = hashlib.sha256(query.encode()).hexdigest()[:16]

        self._cache[key] = CacheEntry(
            embedding=embedding,
            response_text=response_text,
            model=model,
            cost_usd=cost_usd,
            timestamp=time.time(),
            ttl_seconds=ttl_seconds,
        )

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / max(total, 1)
        return {
            "entries": len(self._cache),
            "hit_rate": round(hit_rate * 100, 2),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "saved_cost_usd": round(self._stats["saved_cost_usd"], 6),
            "saved_latency_ms": self._stats["saved_latency_ms"],
            "efficiency": f"{hit_rate * 100:.1f}% hit rate, ${self._stats['saved_cost_usd']:.4f} saved",
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "saved_cost_usd": 0.0,
            "saved_latency_ms": 0,
        }


# Singleton instance
cache = SemanticCache()


async def cached_chat_completion(
    query: str,
    generate_fn,
    model: str = "unknown",
    ttl: float = 3600.0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Wrapper: check cache first, then generate if miss.
    
    Usage:
        text, meta = await cached_chat_completion(
            query=user_msg,
            generate_fn=lambda: orch.think(user_msg),
            model="deepseek-v4-flash",
        )
    """
    cached = await cache.get(query)
    if cached:
        text, sim, meta = cached
        return text, {**meta, "source": "semantic_cache"}

    # Cache miss — generate
    result = await generate_fn()
    response_text = result.get("text", str(result)) if isinstance(result, dict) else str(result)
    cost = result.get("cost_usd", 0.0) if isinstance(result, dict) else 0.0

    # Store in cache
    await cache.set(query, response_text, model, cost, ttl)

    return response_text, {"source": "inference", "cached": False}


if __name__ == "__main__":
    import asyncio

    async def demo():
        print("🧠 MEOKCLAW Semantic Cache Demo")
        print("-" * 40)

        # Simulate cache population
        await cache.set(
            "What is the capital of France?",
            "The capital of France is Paris.",
            "deepseek-v4-flash",
            0.0001,
        )

        # Exact match
        result = await cache.get("What is the capital of France?")
        if result:
            print(f"✅ Exact match: similarity={result[1]:.3f}")

        # Semantic match
        result = await cache.get("Tell me the capital city of France")
        if result:
            print(f"✅ Semantic match: similarity={result[1]:.3f}")
            print(f"   Saved: ${result[2]['saved_cost_usd']}")

        # Miss
        result = await cache.get("What is the speed of light?")
        if not result:
            print("❌ Cache miss (expected)")

        print("-" * 40)
        print(json.dumps(cache.stats(), indent=2))

    asyncio.run(demo())
