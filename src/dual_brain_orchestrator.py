#!/usr/bin/env python3
"""
Dual-Brain Orchestrator — Full pipeline: Router → API → Response → Reflection.
Wires CorpusCallosumRouter to live OpenRouter inference.
"""
import asyncio
from typing import Dict, Any, List, Optional
from dual_brain_router import CorpusCallosumRouter, MODELS, Hemisphere, ReasoningDepth
from openrouter_client import get_client, InferenceResult
from ollama_client import get_local_ollama, get_vast_ollama, OllamaResult
from reflection_engine import ReflectionEngine


class DualBrainOrchestrator:
    """
    The Emperor. Routes tasks through the dual brain and returns fused responses.
    """

    def __init__(self):
        self.router = CorpusCallosumRouter()
        self.reflection = ReflectionEngine()
        self._client = get_client()
        self._local_ollama = get_local_ollama()
        self._vast_ollama = get_vast_ollama()
        self._history: List[Dict[str, str]] = []

    async def think(self, task_text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Full dual-brain inference pipeline."""
        analysis = self.router.analyze(task_text, context)

        # Crisis override
        if analysis.crisis_override:
            return {
                "text": "🚨 [CARE MODE] I'm connecting you with support resources.\n\nSamaritans: 116 123 (UK)\nCrisis Text Line: Text HOME to 741741",
                "hemisphere": "care",
                "primary_model": "care_membrane",
                "cost_usd": 0.0,
                "latency_ms": 50,
                "care_flag": True,
            }

        # Build messages
        messages = [{"role": "system", "content": self._system_prompt(analysis)}]
        messages.extend(self._history[-6:])  # Keep last 6 exchanges
        messages.append({"role": "user", "content": task_text})

        # Route to hemisphere
        if analysis.hemisphere == Hemisphere.BOTH:
            return await self._both_hemispheres(task_text, messages, analysis)
        else:
            return await self._single_hemisphere(task_text, messages, analysis)

    def _system_prompt(self, analysis) -> str:
        if analysis.hemisphere == Hemisphere.LEFT:
            return (
                "You are the LEFT BRAIN of MEOKCLAW — analytical, structured, precise. "
                "You excel at coding, logic, tool use, and sequential reasoning. "
                "You are Kimi K2.6, optimized for SWE-bench performance and long-context coding. "
                "Respond with clean, actionable output."
            )
        elif analysis.hemisphere == Hemisphere.RIGHT:
            return (
                "You are the RIGHT BRAIN of MEOKCLAW — creative, holistic, empathetic. "
                "You excel at reasoning, synthesis, multimodal understanding, and creative tasks. "
                "You are DeepSeek V4, with 1M context and Engram memory. "
                "Respond with insight, pattern recognition, and care."
            )
        else:
            return "You are MEOKCLAW, a sovereign AI assistant. Respond helpfully and accurately."

    async def _infer(
        self, model_cfg: Dict[str, Any], messages: List[Dict], temperature: float, max_tokens: int
    ) -> InferenceResult:
        """Route to OpenRouter or Ollama based on provider."""
        provider = model_cfg.get("provider", "openrouter")
        model_id = model_cfg["id"]

        if provider == "ollama":
            base_url = model_cfg.get("base_url", "http://localhost:11434")
            client = self._vast_ollama if "11436" in base_url else self._local_ollama
            oresult = await client.chat_completion(
                model_id=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return InferenceResult(
                text=oresult.text,
                model=oresult.model,
                tokens_in=oresult.tokens_in,
                tokens_out=oresult.tokens_out,
                cost_usd=0.0,
                latency_ms=oresult.latency_ms,
                hemisphere="unknown",
            )
        else:
            return await self._client.chat_completion(
                model_id=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    async def _single_hemisphere(
        self, task_text: str, messages: List[Dict], analysis
    ) -> Dict[str, Any]:
        """Execute via one hemisphere with OpenRouter → Ollama fallback."""
        primary_cfg = MODELS[analysis.primary_model]
        temperature = 0.7 if analysis.hemisphere == Hemisphere.LEFT else 0.8
        max_tokens = min(1024, primary_cfg.get("max_tokens", 4096))
        # Safety cap for cost control
        if primary_cfg.get("provider") == "openrouter" and "pro" in primary_cfg["id"]:
            max_tokens = min(2048, max_tokens)

        try:
            result = await self._infer(primary_cfg, messages, temperature, max_tokens)
        except Exception as exc:
            # Fallback chain: secondary → vast ollama → local ollama
            result = None
            if analysis.secondary_model:
                try:
                    secondary_cfg = MODELS[analysis.secondary_model]
                    result = await self._infer(secondary_cfg, messages, temperature, max_tokens)
                except Exception:
                    pass
            if result is None:
                # Ultimate fallback: Vast.ai llama3.1:8b
                try:
                    result = await self._infer(
                        {"id": "llama3.1:8b", "provider": "ollama", "base_url": "http://localhost:11436", "max_tokens": 32768},
                        messages, temperature, max_tokens
                    )
                except Exception:
                    raise exc  # Re-raise original if all fail
            if result is None:
                raise exc

        # Log to reflection engine
        self.reflection.reflect(
            task_type=analysis.hemisphere.value,
            task_input=task_text,
            task_summary=result.text[:200],
            model_used=result.model,
            latency_ms=result.latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            care_score=0.85,
            success=True,
        )

        # Update history
        self._history.append({"role": "user", "content": task_text})
        self._history.append({"role": "assistant", "content": result.text})

        return {
            "text": result.text,
            "hemisphere": analysis.hemisphere.value,
            "primary_model": result.model,
            "secondary_model": None,
            "reasoning_depth": analysis.reasoning_depth.value,
            "confidence": analysis.confidence,
            "cost_usd": result.cost_usd,
            "latency_ms": round(result.latency_ms, 1),
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "care_flag": False,
            "reasoning": result.reasoning,
        }

    async def _both_hemispheres(
        self, task_text: str, messages: List[Dict], analysis
    ) -> Dict[str, Any]:
        """Execute both hemispheres in parallel and fuse responses."""
        left_cfg = MODELS[analysis.primary_model]
        right_cfg = MODELS[analysis.secondary_model]

        # Parallel calls
        left_messages = messages.copy()
        left_messages[0]["content"] = (
            "You are the LEFT BRAIN of MEOKCLAW — analytical, structured, precise. "
            "Analyze this task from a logical, sequential, tool-oriented perspective. "
            "Provide your analysis in 3-5 bullet points."
        )
        right_messages = messages.copy()
        right_messages[0]["content"] = (
            "You are the RIGHT BRAIN of MEOKCLAW — creative, holistic, empathetic. "
            "Analyze this task from a creative, synthesizing, big-picture perspective. "
            "Provide your analysis in 3-5 bullet points."
        )

        left_future = self._infer(left_cfg, left_messages, 0.7, 1024)
        right_future = self._infer(right_cfg, right_messages, 0.8, 1024)

        left_result, right_result = await asyncio.gather(left_future, right_future, return_exceptions=True)

        if isinstance(left_result, Exception):
            left_result = InferenceResult(
                text="[Left brain offline — using fallback]",
                model=left_cfg["id"],
                tokens_in=0, tokens_out=0, cost_usd=0, latency_ms=0, hemisphere="left"
            )
        if isinstance(right_result, Exception):
            right_result = InferenceResult(
                text="[Right brain offline — using fallback]",
                model=right_cfg["id"],
                tokens_in=0, tokens_out=0, cost_usd=0, latency_ms=0, hemisphere="right"
            )

        # Fuse responses
        fused = self._fuse(left_result.text, right_result.text, left_result.model, right_result.model)

        total_cost = left_result.cost_usd + right_result.cost_usd
        total_latency = max(left_result.latency_ms, right_result.latency_ms)
        total_tokens_in = left_result.tokens_in + right_result.tokens_in
        total_tokens_out = left_result.tokens_out + right_result.tokens_out

        # Reflect
        self.reflection.reflect(
            task_type="both_hemispheres",
            task_input=task_text,
            task_summary=fused[:200],
            model_used=f"{left_result.model}+{right_result.model}",
            latency_ms=total_latency,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            care_score=0.9,
            success=True,
        )

        self._history.append({"role": "user", "content": task_text})
        self._history.append({"role": "assistant", "content": fused})

        return {
            "text": fused,
            "hemisphere": "both",
            "primary_model": left_result.model,
            "secondary_model": right_result.model,
            "reasoning_depth": "max",
            "confidence": analysis.confidence,
            "cost_usd": round(total_cost, 6),
            "latency_ms": round(total_latency, 1),
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "care_flag": False,
            "left_brain_text": left_result.text,
            "right_brain_text": right_result.text,
        }

    def _fuse(self, left_text: str, right_text: str, left_model: str = "", right_model: str = "") -> str:
        """Fuse left and right brain outputs into coherent response."""
        return (
            f"🧠 **Left Brain Analysis** ({left_model or 'analytical'}):\n{left_text}\n\n"
            f"🎨 **Right Brain Analysis** ({right_model or 'creative'}):\n{right_text}\n\n"
            f"---\n"
            f"*Fused via Corpus Callosum Router — both hemispheres consulted.*"
        )


async def demo():
    orch = DualBrainOrchestrator()
    tasks = [
        "Write a Python function to calculate fibonacci",
        "Why is passive drainage better than pumped systems in Lincolnshire?",
        "Design a council proposal for the new haulage contract",
    ]
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        result = await orch.think(task)
        print(f"HEMISPHERE: {result['hemisphere'].upper()}")
        print(f"MODEL: {result['primary_model']}")
        print(f"COST: ${result['cost_usd']:.6f} | LATENCY: {result['latency_ms']}ms")
        print(f"RESPONSE:\n{result['text'][:500]}...")
    await orch._client.close()


if __name__ == "__main__":
    asyncio.run(demo())
