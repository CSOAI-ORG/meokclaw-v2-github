#!/usr/bin/env python3
"""
MLRouter — sklearn-based dual-brain router for MEOKCLAW.
Loads a pre-trained TF-IDF + LogisticRegression model and returns
hemisphere + confidence. Falls back to keyword router on failure.
"""
import os
import sys
import warnings
from typing import Dict, Any, Optional

import joblib

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "router_sklearn.pkl")

# Keyword fallback triggers (mirrors dual_brain_router.py)
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
    "samaritans", "116 123", "mental health", "breakdown",
]

BOTH_TRIGGERS = [
    "council", "proposal", "vote", "governance", "architecture",
    "system design", "multi-agent", "orchestration", "all models",
    "debate", "discuss", "everyone", "what do you all think",
]


class MLRouter:
    """
    Loads a saved sklearn pipeline and predicts hemisphere + confidence.
    Falls back to keyword-based routing if the model is unavailable or fails.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or MODEL_PATH
        self.pipeline = None
        self.classes_: list = []
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"⚠️ Model not found at {self.model_path}. Using keyword fallback only.", file=sys.stderr)
            return
        try:
            self.pipeline = joblib.load(self.model_path)
            self.classes_ = list(self.pipeline.classes_)
            print(f"🔮 MLRouter loaded ({len(self.classes_)} classes: {', '.join(self.classes_)})")
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}. Using keyword fallback only.", file=sys.stderr)
            self.pipeline = None

    def _keyword_route(self, text: str) -> Dict[str, Any]:
        """Keyword-based fallback router (mirrors dual_brain_router logic)."""
        text_lower = text.lower()

        if any(t in text_lower for t in CARE_TRIGGERS):
            return {"hemisphere": "care", "confidence": 1.0, "source": "keyword_fallback"}

        if any(t in text_lower for t in BOTH_TRIGGERS):
            return {"hemisphere": "both", "confidence": 0.92, "source": "keyword_fallback"}

        left_score = sum(1 for t in LEFT_TRIGGERS if t in text_lower)
        right_score = sum(1 for t in RIGHT_TRIGGERS if t in text_lower)

        greeting_patterns = ["hello", "hi", "hey", "how are you", "good morning", "good afternoon", "good evening", "what's up", "sup"]
        word_count = len(text.split())
        is_greeting = any(text_lower.strip("!?,. ").startswith(g) or text_lower.strip("!?,. ") == g for g in greeting_patterns)
        if word_count <= 5 and is_greeting:
            return {"hemisphere": "left", "confidence": 0.85, "source": "keyword_fallback"}

        if left_score > right_score * 1.5:
            return {"hemisphere": "left", "confidence": min(0.95, 0.6 + left_score * 0.1), "source": "keyword_fallback"}
        elif right_score > left_score * 1.5:
            return {"hemisphere": "right", "confidence": min(0.95, 0.6 + right_score * 0.1), "source": "keyword_fallback"}
        else:
            return {"hemisphere": "both", "confidence": 0.75, "source": "keyword_fallback"}

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict hemisphere for the given text.
        Returns: {"hemisphere": str, "confidence": float, "source": str}
        """
        text_lower = text.lower().strip("!?. ")

        # 1. Crisis override — always keyword-first for safety
        if any(t in text_lower for t in CARE_TRIGGERS):
            return {"hemisphere": "care", "confidence": 1.0, "source": "care_override"}

        # 2. Greeting fast path — short genuine greetings → left (no-think)
        greeting_patterns = ["hello", "hi", "hey", "how are you", "good morning", "good afternoon", "good evening", "what's up", "sup"]
        word_count = len(text.split())
        is_greeting = any(text_lower.startswith(g) or text_lower == g for g in greeting_patterns)
        if word_count <= 5 and is_greeting:
            return {"hemisphere": "left", "confidence": 0.85, "source": "greeting_fastpath"}

        # 3. ML prediction
        if self.pipeline is not None:
            try:
                proba = self.pipeline.predict_proba([text])[0]
                pred_idx = proba.argmax()
                hemisphere = self.classes_[pred_idx]
                confidence = float(proba[pred_idx])
                result = {
                    "hemisphere": str(hemisphere),
                    "confidence": round(float(confidence), 4),
                    "source": "ml_model",
                    "all_probabilities": {str(cls): round(float(p), 4) for cls, p in zip(self.classes_, proba)},
                }
                # Fallback on low-confidence predictions (OOD inputs)
                if confidence < 0.5:
                    fallback = self._keyword_route(text)
                    if fallback["hemisphere"] != result["hemisphere"]:
                        fallback["ml_override"] = result
                        return fallback
                return result
            except Exception as e:
                print(f"⚠️ ML prediction failed: {e}. Falling back to keyword router.", file=sys.stderr)

        # 3. Keyword fallback
        return self._keyword_route(text)


if __name__ == "__main__":
    router = MLRouter()
    test_tasks = [
        "Write a Python function",
        "Why is drainage important",
        "Design a council",
        "Hello!",
        "I want to hurt myself",
        "Debug this TypeScript error",
        "Imagine a passive drainage system for Lincolnshire clay",
        "Write a poem about autumn",
        "Deploy the new API to Kubernetes",
        "What is the meaning of life?",
    ]

    print("\n🧠 MLRouter Test Results:")
    print("-" * 70)
    for task in test_tasks:
        result = router.predict(task)
        print(f"  Task: {task[:55]}")
        print(f"   → {result['hemisphere'].upper():6} | conf={result['confidence']:.2f} | src={result['source']}")
        if "all_probabilities" in result:
            print(f"      probs={result['all_probabilities']}")
        print()
