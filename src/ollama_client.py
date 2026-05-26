#!/usr/bin/env python3
"""
Ollama Async Client — For local and Vast.ai GPU inference.
"""
import time
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class OllamaResult:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float


class OllamaClient:
    """Async client for Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> OllamaResult:
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        start = time.perf_counter()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except aiohttp.ClientError as exc:
            # Retry once
            await asyncio.sleep(0.5)
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

        latency_ms = (time.perf_counter() - start) * 1000
        text = (data.get("message") or {}).get("content") or ""
        tokens_out = len(text.split())
        tokens_in = sum(len(m.get("content", "").split()) for m in messages)
        return OllamaResult(
            text=text,
            model=model_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )


# Singletons
_local_client: Optional[OllamaClient] = None
_vast_client: Optional[OllamaClient] = None


def get_local_ollama() -> OllamaClient:
    global _local_client
    if _local_client is None:
        _local_client = OllamaClient("http://localhost:11434")
    return _local_client


def get_vast_ollama() -> OllamaClient:
    global _vast_client
    if _vast_client is None:
        _vast_client = OllamaClient("http://localhost:11436")
    return _vast_client
