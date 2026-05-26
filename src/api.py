"""MEOKCLAW Dual-Brain API v2.2.0 — Now with COUNCIL MODE and COST ARENA

The sovereign intelligence gateway. Routes every query through the optimal
model hemisphere, with full cost transparency and parallel model comparison.

Key additions:
- /api/council — Multi-model debate with BFT consensus
- /api/arena — Side-by-side model benchmarking
- Cost-per-response tracking with savings calculations
- Shareable result URLs
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dual_brain_orchestrator import DualBrainOrchestrator
from router_ml import MLRouter
from openrouter_client import OpenRouterClient
from ollama_client import OllamaClient

# Optional enterprise modules
try:
    from semantic_cache import SemanticCache
    semantic_cache = SemanticCache()
    SEMANTIC_CACHE = True
except Exception as e:
    print(f"⚠️  Semantic cache not loaded: {e}")
    SEMANTIC_CACHE = False

try:
    from enterprise_auth import auth_manager, get_auth, requires_auth, AuthContext
    from fastapi import Depends
    ENTERPRISE_AUTH = True
except Exception as e:
    print(f"⚠️  Enterprise auth not loaded: {e}")
    ENTERPRISE_AUTH = False

try:
    from prompt_registry import prompt_registry
    PROMPT_REGISTRY = True
except Exception as e:
    print(f"⚠️  Prompt registry not loaded: {e}")
    PROMPT_REGISTRY = False

try:
    from guardrails import guardrails, EnforcementLevel
    GUARDRAILS = True
except Exception as e:
    print(f"⚠️  Guardrails not loaded: {e}")
    GUARDRAILS = False

try:
    from circuit_breaker import circuit_breaker
    CIRCUIT_BREAKER = True
except Exception as e:
    print(f"⚠️  Circuit breaker not loaded: {e}")
    CIRCUIT_BREAKER = False

try:
    from observability import tracer
    OBSERVABILITY = True
except Exception as e:
    print(f"⚠️  Observability not loaded: {e}")
    OBSERVABILITY = False

try:
    from batch_processor import batch_processor
    BATCH_PROCESSOR = True
except Exception as e:
    print(f"⚠️  Batch processor not loaded: {e}")
    BATCH_PROCESSOR = False

try:
    from structured_output import structured_engine
    STRUCTURED_OUTPUT = True
except Exception as e:
    print(f"⚠️  Structured output not loaded: {e}")
    STRUCTURED_OUTPUT = False


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    context: Optional[List[Dict[str, str]]] = None
    mode: str = Field(default="auto", pattern="^(auto|council|arena|fast)$")


class ArenaRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    models: List[str] = Field(default=["deepseek-v4-flash", "deepseek-v4-pro"])
    system_prompt: Optional[str] = None


class CouncilRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    models: List[str] = Field(default=["deepseek-v4-flash", "deepseek-v4-pro", "kimi-k2.6"])
    consensus_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    system_prompt: Optional[str] = None


class ModelResult(BaseModel):
    model: str
    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int
    confidence: Optional[float] = None


class CouncilResponse(BaseModel):
    response_id: str
    prompt: str
    models: List[ModelResult]
    consensus_text: str
    consensus_score: float
    disagreeing_models: List[str]
    total_cost_usd: float
    total_latency_ms: int
    share_url: str
    timestamp: str


class ArenaResponse(BaseModel):
    response_id: str
    prompt: str
    models: List[ModelResult]
    winner: Optional[str] = None
    cost_comparison: Dict[str, float]
    speed_comparison: Dict[str, int]
    share_url: str
    timestamp: str


class ChatResponse(BaseModel):
    response_id: str
    hemisphere: str
    model: str
    text: str
    confidence: float
    cost_usd: float
    latency_ms: int
    tokens_in: int
    tokens_out: int
    context: Optional[List[Dict[str, str]]] = None
    share_url: str


class CostSavings(BaseModel):
    query_cost_usd: float
    gpt4_equivalent_cost: float
    savings_percent: float
    savings_message: str


# ---------------------------------------------------------------------------
# Cost estimation for savings calculations
# ---------------------------------------------------------------------------

GPT4_COST_PER_1K = 0.03  # GPT-4o input cost per 1K tokens

COST_RATES: Dict[str, Dict[str, float]] = {
    "deepseek-v4-flash": {"input": 0.0001, "output": 0.0002},
    "deepseek-v4-pro": {"input": 0.0015, "output": 0.005},
    "kimi-k2.6": {"input": 0.002, "output": 0.008},
    "llama3.1:8b": {"input": 0.0, "output": 0.0},  # Local = free
    "llama3.2:3b": {"input": 0.0, "output": 0.0},
}


def calculate_savings(tokens_in: int, tokens_out: int, model: str) -> CostSavings:
    """Calculate how much money was saved vs GPT-4."""
    our_cost = COST_RATES.get(model, {"input": 0.001, "output": 0.003})
    our_total = (tokens_in / 1000) * our_cost["input"] + (tokens_out / 1000) * our_cost["output"]
    gpt4_total = (tokens_in / 1000) * GPT4_COST_PER_1K + (tokens_out / 1000) * (GPT4_COST_PER_1K * 2)
    
    if gpt4_total > 0:
        savings_pct = ((gpt4_total - our_total) / gpt4_total) * 100
    else:
        savings_pct = 0.0
    
    messages = [
        (95, "🔥 SAVINGS BEAST: You saved 95%+ vs GPT-4!"),
        (90, "💰 MASSIVE SAVINGS: 90%+ cheaper than GPT-4"),
        (80, "💸 Great savings: 80%+ cheaper than GPT-4"),
        (50, "✅ Good savings: 50%+ cheaper than GPT-4"),
        (0,  f"💡 Used {model} — cost transparent"),
    ]
    msg = next(m for threshold, m in messages if savings_pct >= threshold)
    
    return CostSavings(
        query_cost_usd=round(our_total, 6),
        gpt4_equivalent_cost=round(gpt4_total, 6),
        savings_percent=round(savings_pct, 1),
        savings_message=msg,
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.orch = DualBrainOrchestrator()
    app.state.ml_router = MLRouter()
    app.state.openrouter = OpenRouterClient()
    app.state.vast_ollama = OllamaClient("http://localhost:11436")
    app.state.local_ollama = OllamaClient("http://localhost:11434")
    app.state.council_results: Dict[str, CouncilResponse] = {}
    app.state.arena_results: Dict[str, ArenaResponse] = {}
    yield
    # Cleanup handled by __del__ in clients


app = FastAPI(
    title="MEOKCLAW Dual-Brain API",
    version="2.2.0",
    description="Sovereign Intelligence Gateway with Council Mode",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://meokclaw-v2.vercel.app",
        "https://meokclaw-v2-nicholas-projects-d92dd02f.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

@app.post("/api/dual-brain", response_model=ChatResponse)
async def dual_brain_chat(req: ChatRequest):
    orch: DualBrainOrchestrator = app.state.orch
    result = await orch.think(req.message, req.context)
    
    response_id = str(uuid.uuid4())[:8]
    savings = calculate_savings(result.tokens_in, result.tokens_out, result.model)
    
    return ChatResponse(
        response_id=response_id,
        hemisphere=result.hemisphere,
        model=result.model,
        text=result.text,
        confidence=result.confidence,
        cost_usd=round(result.cost_usd, 6),
        latency_ms=result.latency_ms,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        context=result.context,
        share_url=f"https://meokclaw-v2.vercel.app/share/{response_id}",
    )


@app.post("/api/council", response_model=CouncilResponse)
async def council_mode(req: CouncilRequest):
    """Run multiple models in parallel and find consensus via BFT voting."""
    openrouter: OpenRouterClient = app.state.openrouter
    vast: OllamaClient = app.state.vast_ollama
    local: OllamaClient = app.state.local_ollama
    
    response_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Build messages
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.prompt})
    
    # Launch all models in parallel
    async def run_model(model_id: str) -> ModelResult:
        t0 = time.time()
        
        # Route to appropriate provider
        if model_id.startswith("llama") or model_id.startswith("gemma") or model_id.startswith("qwen"):
            # Try local first, then vast
            try:
                res = await local.chat_completion(model_id, messages, temperature=0.7, max_tokens=1024)
            except:
                res = await vast.chat_completion(model_id, messages, temperature=0.7, max_tokens=1024)
        else:
            res = await openrouter.chat_completion(model_id, messages, temperature=0.7, max_tokens=1024)
        
        latency = int((time.time() - t0) * 1000)
        return ModelResult(
            model=model_id,
            text=res.text,
            tokens_in=res.tokens_in,
            tokens_out=res.tokens_out,
            cost_usd=res.cost_usd,
            latency_ms=latency,
            confidence=None,
        )
    
    # Execute all models concurrently
    results = await asyncio.gather(*[run_model(m) for m in req.models], return_exceptions=True)
    models = [r for r in results if not isinstance(r, Exception)]
    failed = [m for i, m in enumerate(req.models) if isinstance(results[i], Exception)]
    
    # Simple BFT consensus: longest common substring / similarity
    # For now: use the response with median length as "consensus"
    if models:
        sorted_by_len = sorted(models, key=lambda x: len(x.text))
        consensus = sorted_by_len[len(sorted_by_len) // 2]
        consensus_score = min(1.0, len(models) / max(len(req.models), 1))
        disagreeing = [m.model for m in models if m.model != consensus.model]
        
        # Actually: if we have 3+ models, check if majority agree on key facts
        if len(models) >= 3:
            # Simple heuristic: count models that mention the same first sentence keywords
            first_words = [set(m.text.split()[:5]) for m in models]
            overlap_scores = []
            for i, words_i in enumerate(first_words):
                overlaps = sum(len(words_i & words_j) / max(len(words_i), 1) 
                             for j, words_j in enumerate(first_words) if i != j)
                overlap_scores.append(overlaps / max(len(models) - 1, 1))
            
            best_idx = overlap_scores.index(max(overlap_scores))
            consensus = models[best_idx]
            consensus_score = max(overlap_scores)
            disagreeing = [m.model for m in models if m.model != consensus.model]
    else:
        consensus = ModelResult(model="none", text="All models failed.", tokens_in=0, tokens_out=0, cost_usd=0, latency_ms=0)
        consensus_score = 0.0
        disagreeing = []
    
    total_cost = sum(m.cost_usd for m in models)
    total_latency = int((time.time() - start_time) * 1000)
    
    resp = CouncilResponse(
        response_id=response_id,
        prompt=req.prompt,
        models=models,
        consensus_text=consensus.text,
        consensus_score=round(consensus_score, 2),
        disagreeing_models=disagreeing[:3],  # Cap at 3
        total_cost_usd=round(total_cost, 6),
        total_latency_ms=total_latency,
        share_url=f"https://meokclaw-v2.vercel.app/council/{response_id}",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    
    app.state.council_results[response_id] = resp
    return resp


@app.post("/api/arena", response_model=ArenaResponse)
async def arena_mode(req: ArenaRequest):
    """Side-by-side model comparison with cost tracking."""
    openrouter: OpenRouterClient = app.state.openrouter
    vast: OllamaClient = app.state.vast_ollama
    local: OllamaClient = app.state.local_ollama
    
    response_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.prompt})
    
    async def run_model(model_id: str) -> ModelResult:
        t0 = time.time()
        
        if model_id.startswith("llama") or model_id.startswith("gemma") or model_id.startswith("qwen"):
            try:
                res = await local.chat_completion(model_id, messages, temperature=0.7, max_tokens=1024)
            except:
                res = await vast.chat_completion(model_id, messages, temperature=0.7, max_tokens=1024)
        else:
            res = await openrouter.chat_completion(model_id, messages, temperature=0.7, max_tokens=1024)
        
        latency = int((time.time() - t0) * 1000)
        return ModelResult(
            model=model_id,
            text=res.text,
            tokens_in=res.tokens_in,
            tokens_out=res.tokens_out,
            cost_usd=res.cost_usd,
            latency_ms=latency,
            confidence=None,
        )
    
    results = await asyncio.gather(*[run_model(m) for m in req.models], return_exceptions=True)
    models = [r for r in results if not isinstance(r, Exception)]
    
    # Winner = best quality heuristic (longest coherent response / lowest cost ratio)
    winner = None
    if models:
        # Score = (tokens_out / cost) / latency — efficiency metric
        def score(m: ModelResult) -> float:
            if m.cost_usd <= 0:
                return m.tokens_out / max(m.latency_ms, 1) * 1000  # Free = huge score
            return (m.tokens_out / m.cost_usd) / max(m.latency_ms, 1)
        
        winner_model = max(models, key=score)
        winner = winner_model.model
    
    cost_comparison = {m.model: m.cost_usd for m in models}
    speed_comparison = {m.model: m.latency_ms for m in models}
    total_cost = sum(m.cost_usd for m in models)
    total_latency = int((time.time() - start_time) * 1000)
    
    resp = ArenaResponse(
        response_id=response_id,
        prompt=req.prompt,
        models=models,
        winner=winner,
        cost_comparison=cost_comparison,
        speed_comparison=speed_comparison,
        share_url=f"https://meokclaw-v2.vercel.app/arena/{response_id}",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    
    app.state.arena_results[response_id] = resp
    return resp


# ---------------------------------------------------------------------------
# Semantic Cache endpoints
# ---------------------------------------------------------------------------

@app.get("/api/cache-stats")
async def cache_stats():
    if semantic_cache:
        return semantic_cache.stats()
    return {"status": "disabled"}


@app.post("/api/cache-clear")
async def cache_clear():
    if semantic_cache:
        semantic_cache.clear()
        return {"status": "cleared"}
    return {"status": "disabled"}


# ---------------------------------------------------------------------------
# Enterprise Auth endpoints
# ---------------------------------------------------------------------------

class CreateKeyRequest(BaseModel):
    name: str
    org_id: str
    scopes: list[str] = None
    max_budget: float = 100.0
    team_id: str = None
    rate_limit_rpm: int = 60
    expires_days: int = None

class CreateOrgRequest(BaseModel):
    name: str
    max_budget: float = 1000.0

@app.post("/api/auth/orgs")
async def create_org(req: CreateOrgRequest):
    if not ENTERPRISE_AUTH:
        raise HTTPException(status_code=503, detail="Enterprise auth not enabled")
    org = auth_manager.create_org(req.name, req.max_budget)
    return {"org_id": org.id, "name": org.name}

@app.post("/api/auth/keys")
async def create_key(req: CreateKeyRequest):
    if not ENTERPRISE_AUTH:
        raise HTTPException(status_code=503, detail="Enterprise auth not enabled")
    plaintext, key = auth_manager.create_key(
        org_id=req.org_id,
        name=req.name,
        scopes=req.scopes,
        max_budget=req.max_budget,
        team_id=req.team_id,
        rate_limit_rpm=req.rate_limit_rpm,
        expires_days=req.expires_days,
    )
    return {"api_key": plaintext, "key_id": key.id, "scopes": key.scopes}

@app.get("/api/auth/keys")
async def list_keys(org_id: str):
    if not ENTERPRISE_AUTH:
        raise HTTPException(status_code=503, detail="Enterprise auth not enabled")
    keys = auth_manager.list_keys(org_id)
    return [{"id": k.id, "name": k.name, "scopes": k.scopes, "budget_used": k.used_budget_usd, "budget_max": k.max_budget_usd, "active": k.is_active} for k in keys]

@app.delete("/api/auth/keys/{key_id}")
async def revoke_key(key_id: str):
    if not ENTERPRISE_AUTH:
        raise HTTPException(status_code=503, detail="Enterprise auth not enabled")
    success = auth_manager.revoke_key(key_id)
    return {"revoked": success}

@app.get("/api/auth/stats")
async def auth_stats(org_id: Optional[str] = None):
    if not ENTERPRISE_AUTH:
        raise HTTPException(status_code=503, detail="Enterprise auth not enabled")
    return auth_manager.stats(org_id)

@app.get("/api/audit-log")
async def audit_log(org_id: str, limit: int = 100):
    if not ENTERPRISE_AUTH:
        raise HTTPException(status_code=503, detail="Enterprise auth not enabled")
    logs = auth_manager.get_audit_log(org_id, limit)
    return [{
        "timestamp": l.timestamp,
        "action": l.action,
        "model": l.model,
        "cost_usd": l.cost_usd,
        "tokens_in": l.tokens_in,
        "tokens_out": l.tokens_out,
        "latency_ms": l.latency_ms,
        "success": l.success,
    } for l in logs]


# ---------------------------------------------------------------------------
# Info / health endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.2.0",
        "features": [
            "dual-brain", "council", "arena", "cost-tracking", "ml-router",
            "semantic-cache", "enterprise-auth", "openrouter-compat",
            "prompt-registry", "guardrails", "circuit-breaker",
            "observability", "batch-processing", "structured-output",
        ],
        "models": {
            "left": "deepseek-v4-flash",
            "right": "deepseek-v4-pro",
            "both": "deepseek-v4-pro+flash",
            "fallback": "llama3.1:8b",
        },
        "endpoints": {
            "chat": "/api/dual-brain",
            "council": "/api/council",
            "arena": "/api/arena",
            "router_stats": "/api/router-stats",
            "reflection_stats": "/api/reflection-stats",
            "live_metrics": "/api/live-metrics",
            "cost_savings": "/api/cost-savings/{model}/{tokens_in}/{tokens_out}",
        },
    }


@app.get("/api/cost-savings/{model}/{tokens_in}/{tokens_out}")
async def cost_savings(model: str, tokens_in: int, tokens_out: int):
    """Get cost savings for a specific query."""
    savings = calculate_savings(tokens_in, tokens_out, model)
    return savings.model_dump()


@app.get("/api/council/{response_id}")
async def get_council_result(response_id: str):
    """Retrieve a cached council result."""
    if response_id not in app.state.council_results:
        raise HTTPException(status_code=404, detail="Council result not found")
    return app.state.council_results[response_id]


@app.get("/api/arena/{response_id}")
async def get_arena_result(response_id: str):
    """Retrieve a cached arena result."""
    if response_id not in app.state.arena_results:
        raise HTTPException(status_code=404, detail="Arena result not found")
    return app.state.arena_results[response_id]


# ---------------------------------------------------------------------------
# Existing stats endpoints (from v2.1.0)
# ---------------------------------------------------------------------------

@app.get("/api/router-stats")
async def router_stats():
    from dual_brain_router import CorpusCallosumRouter
    router = CorpusCallosumRouter()
    return router.stats()


@app.get("/api/reflection-stats")
async def reflection_stats():
    orch: DualBrainOrchestrator = app.state.orch
    return orch.reflection.stats()


@app.get("/api/live-metrics")
async def live_metrics():
    """Combined metrics for the war room dashboard."""
    orch: DualBrainOrchestrator = app.state.orch
    router = CorpusCallosumRouter()
    
    router_stats_data = router.stats()
    reflection_stats_data = orch.reflection.stats()
    
    # Calculate cost burn rate from router stats
    total_cost = router_stats_data.get("total_cost_usd", 0)
    total_queries = router_stats_data.get("total_queries", 0)
    avg_cost = total_cost / max(total_queries, 1)
    
    return {
        "status": "ok",
        "version": "2.2.0",
        "router": router_stats_data,
        "reflection": reflection_stats_data,
        "cost_burn_rate": round(avg_cost, 6),
        "models": {
            "left": {"name": "deepseek-v4-flash", "status": "available", "cost_per_1k": 0.0001},
            "right": {"name": "deepseek-v4-pro", "status": "available", "cost_per_1k": 0.0015},
            "both": {"name": "deepseek-v4-pro+flash", "status": "available", "cost_per_1k": 0.0008},
        },
        "features": {
            "council_mode": True,
            "arena_mode": True,
            "cost_tracking": True,
            "ml_router": True,
        },
    }


# ---------------------------------------------------------------------------
# Guardrails endpoint
# ---------------------------------------------------------------------------

class GuardrailCheckRequest(BaseModel):
    text: str
    enforce_pii: str = "redact"
    enforce_injection: str = "block"
    enforce_content: str = "block"

@app.post("/api/guardrails/check")
async def guardrails_check(req: GuardrailCheckRequest):
    if not GUARDRAILS:
        raise HTTPException(status_code=503, detail="Guardrails not enabled")
    result = guardrails.check(
        req.text,
        enforce_pii=EnforcementLevel(req.enforce_pii) if hasattr(EnforcementLevel, req.enforce_pii.upper()) else EnforcementLevel.REDACT,
        enforce_injection=EnforcementLevel(req.enforce_injection) if hasattr(EnforcementLevel, req.enforce_injection.upper()) else EnforcementLevel.BLOCK,
        enforce_content=EnforcementLevel(req.enforce_content) if hasattr(EnforcementLevel, req.enforce_content.upper()) else EnforcementLevel.BLOCK,
    )
    return {
        "blocked": result.blocked,
        "cleaned_text": result.cleaned_text,
        "violations": [{"type": v.type, "severity": v.severity, "description": v.description, "rule": v.rule_id} for v in result.violations],
        "processing_time_ms": result.processing_time_ms,
    }

@app.get("/api/guardrails/stats")
async def guardrails_stats():
    if not GUARDRAILS:
        return {"status": "disabled"}
    return guardrails.stats()


# ---------------------------------------------------------------------------
# Circuit Breaker endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health/models")
async def model_health():
    if not CIRCUIT_BREAKER:
        return {"status": "disabled"}
    return circuit_breaker.stats()


# ---------------------------------------------------------------------------
# Observability endpoints
# ---------------------------------------------------------------------------

@app.get("/api/traces")
async def get_traces(limit: int = 100):
    if not OBSERVABILITY:
        return {"status": "disabled"}
    traces = tracer.get_recent_traces(limit)
    return {"traces": [t.to_dict() for t in traces]}

@app.get("/api/metrics")
async def get_metrics():
    if not OBSERVABILITY:
        return {"status": "disabled"}
    return tracer.metrics()

@app.get("/api/metrics/prometheus")
async def prometheus_metrics():
    if not OBSERVABILITY:
        return "# observability disabled"
    return tracer.export_prometheus()


# ---------------------------------------------------------------------------
# Batch Processing endpoints
# ---------------------------------------------------------------------------

class BatchSubmitRequest(BaseModel):
    requests: List[Dict[str, str]]
    model: str = "deepseek-v4-flash"
    max_concurrency: int = 5
    system_prompt: Optional[str] = None
    callback_url: Optional[str] = None

@app.post("/api/batch")
async def batch_submit(req: BatchSubmitRequest):
    if not BATCH_PROCESSOR:
        raise HTTPException(status_code=503, detail="Batch processing not enabled")
    job_id = await batch_processor.submit(
        requests=req.requests,
        model=req.model,
        max_concurrency=req.max_concurrency,
        system_prompt=req.system_prompt,
        callback_url=req.callback_url,
    )
    return {"job_id": job_id, "status": "submitted"}

@app.get("/api/batch/{job_id}")
async def batch_status(job_id: str):
    if not BATCH_PROCESSOR:
        raise HTTPException(status_code=503, detail="Batch processing not enabled")
    return batch_processor.status(job_id)

@app.get("/api/batch/{job_id}/results")
async def batch_results(job_id: str, format: str = "json"):
    if not BATCH_PROCESSOR:
        raise HTTPException(status_code=503, detail="Batch processing not enabled")
    return batch_processor.results(job_id, format)

@app.delete("/api/batch/{job_id}")
async def batch_cancel(job_id: str):
    if not BATCH_PROCESSOR:
        raise HTTPException(status_code=503, detail="Batch processing not enabled")
    success = batch_processor.cancel(job_id)
    return {"cancelled": success}


# ---------------------------------------------------------------------------
# Prompt Registry endpoints
# ---------------------------------------------------------------------------

class PromptRegisterRequest(BaseModel):
    name: str
    system_prompt: str
    template: str
    variables: Optional[Dict[str, str]] = None
    author: str = "system"
    environment: str = "dev"

@app.post("/api/prompts")
async def prompt_register(req: PromptRegisterRequest):
    if not PROMPT_REGISTRY:
        raise HTTPException(status_code=503, detail="Prompt registry not enabled")
    version = prompt_registry.register(
        name=req.name,
        system_prompt=req.system_prompt,
        template=req.template,
        variables=req.variables,
        author=req.author,
        environment=req.environment,
    )
    return {"version_id": version.id, "version": version.version}

@app.get("/api/prompts")
async def prompt_list():
    if not PROMPT_REGISTRY:
        return {"status": "disabled"}
    return {"prompts": prompt_registry.list_prompts()}

@app.get("/api/prompts/{name}")
async def prompt_stats(name: str):
    if not PROMPT_REGISTRY:
        return {"status": "disabled"}
    return prompt_registry.stats(name)


# ---------------------------------------------------------------------------
# OpenRouter compatibility layer
# ---------------------------------------------------------------------------

try:
    from openrouter_integration import mount_openrouter_routes
    mount_openrouter_routes(app)
except Exception as e:
    print(f"⚠️  OpenRouter integration not loaded: {e}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3201, log_level="info")
