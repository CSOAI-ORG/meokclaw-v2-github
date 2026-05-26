"""OpenRouter-compatible API layer for MEOKCLAW

Exposes /v1/chat/completions so MEOKCLAW can be used as a drop-in
replacement for OpenAI/OpenRouter in any application.

Usage:
    import openai
    client = openai.OpenAI(
        base_url="http://localhost:3201/v1",
        api_key="dummy"  # MEOKCLAW doesn't require auth locally
    )
    response = client.chat.completions.create(
        model="meokclaw-auto",  # or "meokclaw-council", "meokclaw-fast"
        messages=[{"role": "user", "content": "Hello"}]
    )
"""
from __future__ import annotations

import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from dual_brain_orchestrator import DualBrainOrchestrator

router = APIRouter(prefix="/v1")


# ---------------------------------------------------------------------------
# OpenAI-compatible request/response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant|tool)$")
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="meokclaw-auto")
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=8192)
    stream: Optional[bool] = False
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    # OpenRouter extensions
    models: Optional[List[str]] = None  # For council mode
    route: Optional[str] = None  # "auto", "left", "right", "both", "council", "fast"


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: Optional[float] = None
    savings_vs_gpt4: Optional[float] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage
    meokclaw_meta: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_id_to_meokclaw(model_id: str) -> tuple[str, Optional[List[str]]]:
    """Map OpenAI-style model ID to MEOKCLAW config."""
    if model_id == "meokclaw-council" or model_id.endswith("-council"):
        return "council", None
    if model_id == "meokclaw-fast" or model_id.endswith("-fast"):
        return "fast", None
    if model_id == "meokclaw-left" or model_id.endswith("-left"):
        return "left", None
    if model_id == "meokclaw-right" or model_id.endswith("-right"):
        return "right", None
    if model_id == "meokclaw-both" or model_id.endswith("-both"):
        return "both", None
    return "auto", None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    """OpenAI-compatible chat completions endpoint."""
    orch: DualBrainOrchestrator = request.app.state.orch
    
    mode, _ = _model_id_to_meokclaw(req.model)
    
    # Extract the user's message
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    user_message = user_messages[-1].content
    context = [{"role": m.role, "content": m.content} for m in req.messages[:-1]]
    
    # Route based on mode
    start_time = time.time()
    
    if mode == "council":
        # Use council mode
        models = req.models or ["deepseek-v4-flash", "deepseek-v4-pro", "kimi-k2.6"]
        from openrouter_client import OpenRouterClient
        from ollama_client import OllamaClient
        import asyncio
        
        openrouter = OpenRouterClient()
        vast = OllamaClient("http://localhost:11436")
        local = OllamaClient("http://localhost:11434")
        
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        
        async def run_one(model_id: str):
            if model_id.startswith("llama") or model_id.startswith("gemma"):
                try:
                    res = await local.chat_completion(model_id, messages, req.temperature, req.max_tokens)
                except:
                    res = await vast.chat_completion(model_id, messages, req.temperature, req.max_tokens)
            else:
                res = await openrouter.chat_completion(model_id, messages, req.temperature, req.max_tokens)
            return res
        
        results = await asyncio.gather(*[run_one(m) for m in models], return_exceptions=True)
        valid = [r for r in results if not isinstance(r, Exception)]
        
        if not valid:
            raise HTTPException(status_code=503, detail="All council models failed")
        
        # Use first valid result as "primary" for OpenAI compatibility
        primary = valid[0]
        total_cost = sum(r.cost_usd for r in valid)
        total_tokens = sum(r.tokens_out for r in valid)
        
        meta = {
            "mode": "council",
            "council_size": len(models),
            "successful_models": len(valid),
            "all_models": [{"model": m.model, "cost_usd": m.cost_usd} for m in valid],
        }
    else:
        # Standard dual-brain routing
        result = await orch.think(user_message, context if context else None)
        primary = result
        total_cost = result.get("cost_usd", 0)
        total_tokens = result.get("tokens_out", 0)
        
        meta = {
            "mode": mode,
            "hemisphere": result.get("hemisphere", "unknown"),
            "router_confidence": result.get("confidence", 0),
            "actual_model": result.get("model", result.get("primary_model", "unknown")),
        }
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    # Calculate savings
    from dual_brain_api import calculate_savings
    savings = calculate_savings(
        primary.get("tokens_in", primary.get("tokens_in", 0)),
        primary.get("tokens_out", primary.get("tokens_out", 0)),
        primary.get("model", primary.get("primary_model", "unknown")),
    )
    
    response = ChatCompletionResponse(
        id=f"meokclaw-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=primary.get("text", "[No response]")),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=primary.get("tokens_in", 0),
            completion_tokens=primary.get("tokens_out", 0),
            total_tokens=primary.get("tokens_in", 0) + primary.get("tokens_out", 0),
            cost_usd=round(total_cost, 6),
            savings_vs_gpt4=round(savings.savings_percent, 1),
        ),
        meokclaw_meta={
            **meta,
            "latency_ms": latency_ms,
            "savings_message": savings.savings_message,
        },
    )
    
    return response


@router.get("/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "meokclaw-auto",
                "object": "model",
                "created": 1700000000,
                "owned_by": "meokclaw",
                "description": "Auto-routing via Corpus Callosum",
            },
            {
                "id": "meokclaw-fast",
                "object": "model",
                "created": 1700000000,
                "owned_by": "meokclaw",
                "description": "Always use fastest/cheapest model",
            },
            {
                "id": "meokclaw-council",
                "object": "model",
                "created": 1700000000,
                "owned_by": "meokclaw",
                "description": "Multi-model BFT consensus",
            },
            {
                "id": "meokclaw-left",
                "object": "model",
                "created": 1700000000,
                "owned_by": "meokclaw",
                "description": "Left hemisphere (fast/cheap)",
            },
            {
                "id": "meokclaw-right",
                "object": "model",
                "created": 1700000000,
                "owned_by": "meokclaw",
                "description": "Right hemisphere (smart/expensive)",
            },
            {
                "id": "meokclaw-both",
                "object": "model",
                "created": 1700000000,
                "owned_by": "meokclaw",
                "description": "Both hemispheres with fusion",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Mount this router in the main FastAPI app
# ---------------------------------------------------------------------------

def mount_openrouter_routes(app):
    """Call this from dual_brain_api.py to add OpenRouter compatibility."""
    app.include_router(router)
