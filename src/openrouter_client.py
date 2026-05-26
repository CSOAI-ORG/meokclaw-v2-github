#!/usr/bin/env python3
"""
OpenRouter Client — Live inference for the Dual-Brain Architecture.
Supports Kimi K2.6, DeepSeek V4, Gemma 4, Nemotron, and all free tiers.
"""
import os
import json
import httpx
from typing import AsyncIterator, Dict, Any, Optional, List
from dataclasses import dataclass

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_KEY_2 = os.environ.get("OPENROUTER_API_KEY_2", "")
BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class InferenceResult:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    hemisphere: str
    reasoning: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


class OpenRouterClient:
    """Async client for OpenRouter API with key rotation and fallback."""

    def __init__(self):
        self.keys = [k for k in [OPENROUTER_API_KEY, OPENROUTER_API_KEY_2] if k and not k.startswith("REPLACE")]
        self.key_index = 0
        self._client = httpx.AsyncClient(timeout=60.0, headers={
            "Content-Type": "application/json",
            "HTTP-Referer": "https://meok.ai",
            "X-Title": "MEOKCLAW Dual Brain",
        })

    def _current_key(self) -> str:
        return self.keys[self.key_index % len(self.keys)]

    def _rotate_key(self):
        self.key_index += 1

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> InferenceResult:
        """Send a chat completion request to OpenRouter."""
        import time
        start = time.perf_counter()

        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        headers = {
            "Authorization": f"Bearer {self._current_key()}",
            "Content-Type": "application/json",
        }

        try:
            resp = await self._client.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                self._rotate_key()
                # Retry once with new key
                headers["Authorization"] = f"Bearer {self._current_key()}"
                resp = await self._client.post(
                    f"{BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            else:
                raise

        choices = data.get("choices") or [{}]
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning") or None

        usage = data.get("usage") or {}
        tokens_in = usage.get("prompt_tokens") or 0
        tokens_out = usage.get("completion_tokens") or 0

        # Cost calculation from API pricing
        cost = 0.0
        if "cost" in usage:
            cost = usage["cost"]
        elif "total_cost" in usage:
            cost = usage["total_cost"]
        else:
            cost = self._estimate_cost(model_id, tokens_in, tokens_out)

        latency = (time.perf_counter() - start) * 1000

        return InferenceResult(
            text=content,
            model=data.get("model", model_id),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            latency_ms=latency,
            hemisphere="unknown",
            reasoning=reasoning,
            tool_calls=message.get("tool_calls"),
        )

    def _estimate_cost(self, model_id: str, tokens_in: int, tokens_out: int) -> float:
        """Estimate cost when API doesn't return pricing."""
        rates = {
            "moonshotai/kimi-k2.6": (0.00000073, 0.00000349),
            "deepseek/deepseek-v4-pro": (0.000000435, 0.00000087),
            "deepseek/deepseek-v4-pro-20260423": (0.000000435, 0.00000087),
            "deepseek/deepseek-v4-flash": (0.0000001, 0.0000002),
            "deepseek/deepseek-v4-flash-20260423": (0.0000001, 0.0000002),
            "deepseek/deepseek-v4-flash:free": (0, 0),
            "google/gemma-4-31b-it:free": (0, 0),
            "google/gemma-4-26b-a4b-it:free": (0, 0),
            "nvidia/nemotron-3-super-120b-a12b:free": (0, 0),
        }
        in_rate, out_rate = rates.get(model_id, (0.000001, 0.000002))
        return (tokens_in * in_rate + tokens_out * out_rate) / 1000

    async def close(self):
        await self._client.aclose()


# Singleton
_client: Optional[OpenRouterClient] = None


def get_client() -> OpenRouterClient:
    global _client
    if _client is None:
        _client = OpenRouterClient()
    return _client
