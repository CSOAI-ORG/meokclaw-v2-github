#!/usr/bin/env python3
"""
Corpus Callosum Router — Dual-Brain Architecture for SOV3.
Left brain: Kimi K2.6 (structure, coding, tools)
Right brain: DeepSeek V4 (reasoning, creativity, synthesis)
Bridge: Hy3-inspired task classifier + BFT fusion

Every task analyzed in <10ms. Routed to optimal hemisphere.
Both hemispheres for ambiguous tasks + speculative consensus.
"""
import os
import json
import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class Hemisphere(Enum):
    LEFT = "left"      # Kimi K2.6: structure, coding, tools, sequencing
    RIGHT = "right"    # DeepSeek V4: reasoning, creativity, synthesis, spatial
    BOTH = "both"      # Parallel execution + BFT fusion
    CARE = "care"      # Crisis override → care membrane


class ReasoningDepth(Enum):
    NO_THINK = "no_think"      # <100ms, local llama3.2:3b
    LOW = "low"                # ~500ms, qwen2.5:7b
    MEDIUM = "medium"          # ~1.5s, gemma4:26b
    HIGH = "high"              # ~5s, kimi k2.6 / deepseek v4
    MAX = "max"                # ~15s, both hemispheres + council


@dataclass
class TaskAnalysis:
    task_text: str
    hemisphere: Hemisphere
    reasoning_depth: ReasoningDepth
    primary_model: str
    secondary_model: Optional[str]
    confidence: float
    estimated_cost_usd: float
    estimated_latency_ms: float
    care_flag: bool = False
    crisis_override: bool = False


# Model configs via OpenRouter
MODELS = {
    "kimi-k2.6": {
        "id": "moonshotai/kimi-k2.6",
        "provider": "openrouter",
        "hemisphere": Hemisphere.LEFT,
        "cost_in": 0.00000073,
        "cost_out": 0.00000349,
        "typical_latency_ms": 16000,
        "strengths": ["coding", "structure", "tools", "sequencing", "long_context"],
        "max_tokens": 200000,
    },
    "deepseek-v4-pro": {
        "id": "deepseek/deepseek-v4-pro",
        "provider": "openrouter",
        "hemisphere": Hemisphere.RIGHT,
        "cost_in": 0.000000435,
        "cost_out": 0.00000087,
        "typical_latency_ms": 8000,
        "strengths": ["reasoning", "creativity", "synthesis", "multimodal", "engram_memory"],
        "max_tokens": 64000,
    },
    "deepseek-v4-flash-free": {
        "id": "deepseek/deepseek-v4-flash:free",
        "provider": "openrouter",
        "hemisphere": Hemisphere.RIGHT,
        "cost_in": 0,
        "cost_out": 0,
        "typical_latency_ms": 800,
        "strengths": ["fast_reasoning", "creative", "chat"],
        "max_tokens": 64000,
    },
    "deepseek-v4-flash": {
        "id": "deepseek/deepseek-v4-flash",
        "provider": "openrouter",
        "hemisphere": Hemisphere.LEFT,
        "cost_in": 0.0000001,
        "cost_out": 0.0000002,
        "typical_latency_ms": 2500,
        "strengths": ["fast", "coding", "reasoning", "chat", "cheap"],
        "max_tokens": 64000,
    },
    "gemma4-31b-free": {
        "id": "google/gemma-4-31b-it:free",
        "provider": "openrouter",
        "hemisphere": Hemisphere.LEFT,
        "cost_in": 0,
        "cost_out": 0,
        "typical_latency_ms": 2000,
        "strengths": ["coding", "vision", "long_context"],
        "max_tokens": 128000,
    },
    "gemma4-26b-free": {
        "id": "google/gemma-4-26b-a4b-it:free",
        "provider": "openrouter",
        "hemisphere": Hemisphere.LEFT,
        "cost_in": 0,
        "cost_out": 0,
        "typical_latency_ms": 1500,
        "strengths": ["coding", "vision", "fast"],
        "max_tokens": 128000,
    },
    "nemotron-super-free": {
        "id": "nvidia/nemotron-3-super-120b-a12b:free",
        "provider": "openrouter",
        "hemisphere": Hemisphere.BOTH,
        "cost_in": 0,
        "cost_out": 0,
        "typical_latency_ms": 3000,
        "strengths": ["agentic", "orchestration", "tool_use"],
        "max_tokens": 128000,
    },
    "qwen3.5-local": {
        "id": "qwen3.5:9b",
        "provider": "ollama",
        "hemisphere": Hemisphere.RIGHT,
        "cost_in": 0,
        "cost_out": 0,
        "typical_latency_ms": 400,
        "strengths": ["creative", "emotional", "chat", "fast"],
        "max_tokens": 32768,
    },
    "gemma4-local": {
        "id": "gemma4:e4b",
        "provider": "ollama",
        "hemisphere": Hemisphere.LEFT,
        "cost_in": 0,
        "cost_out": 0,
        "typical_latency_ms": 2400,
        "strengths": ["fast", "coding", "vision", "local"],
        "max_tokens": 128000,
    },
    "llama3.1-vast": {
        "id": "llama3.1:8b",
        "provider": "ollama",
        "base_url": "http://localhost:11436",
        "hemisphere": Hemisphere.BOTH,
        "cost_in": 0,
        "cost_out": 0,
        "typical_latency_ms": 2500,
        "strengths": ["reliable", "coding", "reasoning", "chat", "gpu"],
        "max_tokens": 32768,
    },
}


class CorpusCallosumRouter:
    """
    The bridge between hemispheres.
    Analyzes tasks in <10ms and routes to optimal brain.
    """

    # Task-type classification patterns
    LEFT_TRIGGERS = [
        "code", "debug", "function", "api", "refactor", "implement",
        "typescript", "python", "json", "xml", "schema", "database",
        "deploy", "build", "test", "ci/cd", "docker", "kubernetes",
        "structure", "sequence", "plan", "schedule", "organize",
        "tool", "mcp", "execute", "command", "shell", "script",
        "flatten", "list", "array", "dict", "tuple", "string",
        "parse", "serialize", "deserialize", "convert", "transform",
        "oneliner", "one-liner", "one line", "single line",
    ]

    RIGHT_TRIGGERS = [
        "creative", "imagine", "story", "poem", "write", "design",
        "feel", "emotion", "empathy", "care", "support", "listen",
        "why", "analyze", "compare", "evaluate", "synthesize",
        "vision", "image", "look", "see", "screen", "camera",
        "strategy", "philosophy", "meaning", "purpose", "wisdom",
        "abuntu", "legacy", "drainage", "lime", "thermal", "aquaponics",
    ]

    CARE_TRIGGERS = [
        "kill", "suicide", "self-harm", "hurt", "crisis", "emergency",
        " Samaritans", "116 123", "mental health", "breakdown",
    ]

    BOTH_TRIGGERS = [
        "council", "proposal", "vote", "governance", "architecture",
        "system design", "multi-agent", "orchestration", "all models",
        "debate", "discuss", "everyone", "what do you all think",
    ]

    def __init__(self):
        self.decision_log: List[Dict] = []

    def analyze(self, task_text: str, context: Optional[Dict] = None) -> TaskAnalysis:
        """Analyze task and return routing decision. Target: <10ms."""
        start = time.perf_counter()
        text_lower = task_text.lower()

        # 1. Crisis override (highest priority)
        if any(t in text_lower for t in self.CARE_TRIGGERS):
            return TaskAnalysis(
                task_text=task_text,
                hemisphere=Hemisphere.CARE,
                reasoning_depth=ReasoningDepth.MAX,
                primary_model="care_membrane",
                secondary_model=None,
                confidence=1.0,
                estimated_cost_usd=0.0,
                estimated_latency_ms=500,
                care_flag=True,
                crisis_override=True,
            )

        # 2. Both-hemisphere triggers (complex governance)
        if any(t in text_lower for t in self.BOTH_TRIGGERS):
            result = TaskAnalysis(
                task_text=task_text,
                hemisphere=Hemisphere.BOTH,
                reasoning_depth=ReasoningDepth.MAX,
                primary_model="deepseek-v4-pro",
                secondary_model="deepseek-v4-flash",
                confidence=0.92,
                estimated_cost_usd=0.002,
                estimated_latency_ms=15000,
            )
            self._log_decision(result, start)
            return result

        # 3. Count trigger matches
        left_score = sum(1 for t in self.LEFT_TRIGGERS if t in text_lower)
        right_score = sum(1 for t in self.RIGHT_TRIGGERS if t in text_lower)

        # 4. Greeting fast path — only for genuine short greetings
        word_count = len(task_text.split())
        text_lower = task_text.lower().strip("!?. ")
        greeting_patterns = ["hello", "hi", "hey", "how are you", "good morning", "good afternoon", "good evening", "what's up", "sup"]
        is_greeting = any(text_lower.startswith(g) or text_lower == g for g in greeting_patterns)
        if word_count <= 5 and is_greeting:
            # Fast path: simple greeting → Vast.ai GPU (free, reliable)
            return TaskAnalysis(
                task_text=task_text,
                hemisphere=Hemisphere.LEFT,
                reasoning_depth=ReasoningDepth.NO_THINK,
                primary_model="llama3.1-vast",
                secondary_model=None,
                confidence=0.85,
                estimated_cost_usd=0.0,
                estimated_latency_ms=2500,
            )

        # 5. Determine hemisphere
        if left_score > right_score * 1.5:
            hemisphere = Hemisphere.LEFT
            primary = "deepseek-v4-flash"
            secondary = "deepseek-v4-pro"
            depth = ReasoningDepth.HIGH if left_score > 3 else ReasoningDepth.MEDIUM
        elif right_score > left_score * 1.5:
            hemisphere = Hemisphere.RIGHT
            primary = "deepseek-v4-pro"
            secondary = "deepseek-v4-flash"
            depth = ReasoningDepth.HIGH if right_score > 3 else ReasoningDepth.MEDIUM
        else:
            # Ambiguous → both hemispheres
            hemisphere = Hemisphere.BOTH
            primary = "deepseek-v4-flash"
            secondary = "deepseek-v4-pro"
            depth = ReasoningDepth.HIGH

        # 6. Cost/latency estimation
        primary_cfg = MODELS[primary]
        est_tokens_in = len(task_text) // 4
        est_tokens_out = 500 if depth == ReasoningDepth.HIGH else 200
        cost = (est_tokens_in * primary_cfg["cost_in"] + est_tokens_out * primary_cfg["cost_out"]) / 1000
        latency = primary_cfg["typical_latency_ms"]
        if hemisphere == Hemisphere.BOTH:
            cost *= 2.2
            latency *= 1.8

        result = TaskAnalysis(
            task_text=task_text,
            hemisphere=hemisphere,
            reasoning_depth=depth,
            primary_model=primary,
            secondary_model=secondary,
            confidence=min(0.95, 0.6 + max(left_score, right_score) * 0.1),
            estimated_cost_usd=cost,
            estimated_latency_ms=latency,
        )
        self._log_decision(result, start)
        return result

    def _log_decision(self, result: TaskAnalysis, start_time: float):
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.decision_log.append({
            "hemisphere": result.hemisphere.value,
            "primary": result.primary_model,
            "confidence": result.confidence,
            "latency_ms": round(elapsed_ms, 3),
            "timestamp": time.time(),
        })

    def stats(self) -> Dict[str, Any]:
        if not self.decision_log:
            return {"total": 0}
        total = len(self.decision_log)
        hemispheres = {}
        for d in self.decision_log:
            h = d["hemisphere"]
            hemispheres[h] = hemispheres.get(h, 0) + 1
        avg_latency = sum(d["latency_ms"] for d in self.decision_log) / total
        return {
            "total_decisions": total,
            "hemisphere_distribution": hemispheres,
            "avg_routing_latency_ms": round(avg_latency, 3),
            "max_routing_latency_ms": round(max(d["latency_ms"] for d in self.decision_log), 3),
        }


# Simple inference wrapper
async def dual_brain_infer(router: CorpusCallosumRouter, task_text: str, history: List[Dict] = None) -> Dict[str, Any]:
    """Full dual-brain inference pipeline."""
    analysis = router.analyze(task_text)
    
    if analysis.crisis_override:
        return {
            "response": "[Serious] I'm connecting you with support resources. Samaritans: 116 123.",
            "hemisphere": "care",
            "model": "care_membrane",
            "latency_ms": 100,
        }

    primary_cfg = MODELS[analysis.primary_model]
    
    # In production, this would call the actual API
    # For now, return the routing decision + simulated response
    return {
        "response": f"[{analysis.hemisphere.value.upper()} BRAIN via {analysis.primary_model}] Processing...",
        "hemisphere": analysis.hemisphere.value,
        "primary_model": primary_cfg["id"],
        "secondary_model": MODELS[analysis.secondary_model]["id"] if analysis.secondary_model else None,
        "reasoning_depth": analysis.reasoning_depth.value,
        "confidence": analysis.confidence,
        "estimated_cost_usd": analysis.estimated_cost_usd,
        "estimated_latency_ms": analysis.estimated_latency_ms,
        "care_flag": analysis.care_flag,
    }


if __name__ == "__main__":
    router = CorpusCallosumRouter()
    
    test_tasks = [
        "Hello, how are you?",                                    # FAST → no_think
        "Write a Python function to parse JSON",                  # LEFT → kimi
        "Why do we exist? What's the meaning of life?",           # RIGHT → deepseek
        "Design a council proposal for the new haulage contract", # BOTH → fusion
        "I want to hurt myself",                                  # CARE → crisis
        "Debug this TypeScript error: Type 'string' not assignable", # LEFT → kimi
        "Imagine a passive drainage system for Lincolnshire clay", # RIGHT → deepseek
    ]
    
    for task in test_tasks:
        result = router.analyze(task)
        print(f"\n🧠 Task: {task[:50]}...")
        print(f"   Hemisphere: {result.hemisphere.value.upper()}")
        print(f"   Model: {result.primary_model} + {result.secondary_model or 'none'}")
        print(f"   Depth: {result.reasoning_depth.value}")
        print(f"   Cost: ${result.estimated_cost_usd:.6f} | Latency: ~{result.estimated_latency_ms}ms")
        print(f"   Confidence: {result.confidence:.2f}")
    
    print("\n📊 Router Stats:", router.stats())
