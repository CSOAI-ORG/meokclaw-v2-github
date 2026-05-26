# MEOKCLAW Launch Posts

## Hacker News — Show HN

```
Show HN: I built a $0.0003/query AI router that learns from every conversation

https://meokclaw-v2.vercel.app
https://github.com/nicholas/meokclaw

MEOKCLAW is a sovereign dual-brain AI orchestrator. Instead of sending every
query to GPT-4, it analyzes what you're asking and routes to the optimal model
—or multiple models in parallel.

The magic features:

1. Dual-Brain Router — LEFT hemisphere = fast/cheap (DeepSeek Flash), RIGHT =
   smart/expensive (DeepSeek Pro). A simple "hello" costs $0.000003 instead of
   $0.04. A complex coding question gets both hemispheres with fusion.

2. Council Mode — Run 3-5 models in parallel, get BFT consensus. No single
   point of failure. If one model hallucinates, the others catch it.

3. Cost Arena — Side-by-side model comparison with real $/token costs. See
   exactly what you're paying. Every response shows: "This cost $0.0003.
   GPT-4 would cost $0.04. You saved 99.3%."

4. ML Router — Trained on 2,381 real conversations. 99.35% accuracy. Gets
   smarter every day.

Architecture: FastAPI backend, Next.js frontend, scikit-learn classifier,
OpenRouter for model access, Vast.ai Ollama for local fallback.

It's open source. Docker one-liner to run locally. OpenAI-compatible API at
/v1/chat/completions — drop it into any app.

Would love feedback on the router logic and the council consensus algorithm.
```

---

## Twitter/X Thread

**Tweet 1:**
```
I built an AI router that saves 99% on API costs 🧵

Every query gets analyzed and routed to the optimal model.
Simple greeting → $0.000003
Complex code → both hemispheres
Crisis query → care membrane immediately

No more sending everything to GPT-4.

https://meokclaw-v2.vercel.app
```

**Tweet 2:**
```
The "Council Mode" is my favorite feature.

Run 5 models in parallel. They debate. You get consensus.

If one model hallucinates, the others catch it.
BFT voting — no single point of failure.

Cost: $0.0004 for 3 models
vs $0.12 for GPT-4 alone

That's 99.7% savings WITH better reliability.
```

**Tweet 3:**
```
The Cost Arena shows real-time $/token comparison.

DeepSeek Flash: $0.0001/1K tokens
DeepSeek Pro: $0.0015/1K tokens
Kimi K2.6: $0.002/1K tokens
GPT-4o: $0.005/1K tokens

Side-by-side. Same prompt. Real costs.

This is the transparency AI needs.
```

**Tweet 4:**
```
It's fully open source.

• Docker one-liner to run
• OpenAI-compatible API (/v1/chat/completions)
• VS Code extension
• Discord bot
• Trains on your conversations

Star it → github.com/nicholas/meokclaw

Live demo → meokclaw-v2.vercel.app
```

---

## Reddit r/LocalLLaMA

```
[Showcase] MEOKCLAW — Dual-Brain AI Router with Local Fallback

Hey r/LocalLLaMA,

I've been building a sovereign AI orchestrator that routes queries between
cloud models (DeepSeek, Kimi via OpenRouter) and local models (Llama 3.1 on
Vast.ai RTX 4070 SUPER, Gemma 4 on M4 MacBook).

The router has two hemispheres:
- LEFT: Fast/cheap models for simple queries
- RIGHT: Smart models for complex queries
- BOTH: Fusion when the router isn't sure

It learns from every conversation using scikit-learn (TF-IDF + Logistic
Regression, 99.35% accuracy on 2,381 reflections).

Key thing for this sub: it falls back to local Ollama automatically when:
1. Cloud API is down
2. Cost cap exceeded
3. User requests privacy mode
4. Simple greeting (word count ≤ 5)

The Council Mode runs 3-5 models in parallel and finds consensus. This means
you can use cheap local models alongside cloud models and still get reliable
answers.

GitHub: github.com/nicholas/meokclaw
Live: meokclaw-v2.vercel.app

Questions about the architecture or router logic welcome!
```

---

## Reddit r/OpenAI

```
[Showcase] Built an OpenAI-compatible router that saves 99% on API costs

I got tired of paying $0.04 for simple queries that a $0.0001 model could
handle. So I built MEOKCLAW — a dual-brain router that analyzes your query
and picks the optimal model.

It exposes /v1/chat/completions so any OpenAI-compatible app can use it
with a single base_url change.

Drop-in replacement:
```python
import openai
client = openai.OpenAI(
    base_url="http://localhost:3201/v1",
    api_key="dummy"
)
response = client.chat.completions.create(
    model="meokclaw-auto",  # or meokclaw-council, meokclaw-fast
    messages=[{"role": "user", "content": "Hello"}]
)
```

Every response includes cost and savings vs GPT-4.

GitHub + live demo in comments.
```
