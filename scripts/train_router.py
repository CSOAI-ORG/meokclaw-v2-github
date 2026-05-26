#!/usr/bin/env python3
"""
Train a sklearn-based router classifier for the MEOKCLAW dual-brain system.
Extracts data from reflection_store.db, trains TF-IDF + LogisticRegression pipeline,
evaluates, saves model, and writes router_ml.py.
"""
import os
import sqlite3
import pickle
import warnings
from typing import Dict, Any, List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

DB_PATH = os.path.expanduser("~/clawd/sovereign-temple/data/reflection_store.db")
MODEL_PATH = os.path.expanduser("~/clawd/sovereign-temple/data/router_sklearn.pkl")
ROUTER_ML_PATH = os.path.expanduser("~/clawd/sovereign-temple/router_ml.py")

TASK_TYPE_TO_HEMISPHERE = {
    # Explicit labels
    "left": "left",
    "right": "right",
    "both_hemispheres": "both",
    # Tool / execution / structured → left
    "edit_file": "left",
    "open_terminal": "left",
    "web_search": "left",
    "browse_web": "left",
    "memory_query": "left",
    "session_log": "left",
    # Emotional / experiential / creative / conversational → right
    "emotion_observation": "right",
    "experience": "right",
    "episodic": "right",
    "session_summary": "right",
    "semantic": "right",
    "weather": "right",
    # Governance / complex → both
    "council_proposal": "both",
    # Ambiguous general response → both (model learns from text content)
    "response_generation": "both",
}


def load_data(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT task_summary, task_type FROM reflections WHERE success = 1")
    rows = cursor.fetchall()
    conn.close()

    texts, labels = [], []
    unmapped = {}
    for summary, task_type in rows:
        if not summary or not task_type:
            continue
        hemi = TASK_TYPE_TO_HEMISPHERE.get(task_type)
        if hemi is None:
            unmapped[task_type] = unmapped.get(task_type, 0) + 1
            continue
        texts.append(summary)
        labels.append(hemi)

    if unmapped:
        print("⚠️ Unmapped task types (skipped):")
        for tt, c in sorted(unmapped.items(), key=lambda x: -x[1]):
            print(f"   {tt}: {c}")

    return texts, labels


def train_and_evaluate(texts: List[str], labels: List[str]) -> Pipeline:
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    print(f"📊 Training on {len(X_train)} samples, testing on {len(X_test)}...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, digits=4))

    return pipeline


def save_model(pipeline: Pipeline, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"💾 Model saved to {path}")


ROUTER_ML_CODE = '''#!/usr/bin/env python3
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

    print("\\n🧠 MLRouter Test Results:")
    print("-" * 70)
    for task in test_tasks:
        result = router.predict(task)
        print(f"  Task: {task[:55]}")
        print(f"   → {result['hemisphere'].upper():6} | conf={result['confidence']:.2f} | src={result['source']}")
        if "all_probabilities" in result:
            print(f"      probs={result['all_probabilities']}")
        print()
'''


def write_router_ml(path: str):
    with open(path, "w") as f:
        f.write(ROUTER_ML_CODE)
    print(f"📝 router_ml.py written to {path}")


def main():
    print("🔍 Loading training data from reflection_store.db...")
    texts, labels = load_data(DB_PATH)
    print(f"   Loaded {len(texts)} labeled samples.")

    if len(texts) < 10:
        raise RuntimeError("Not enough training data.")

    # Class distribution
    from collections import Counter
    dist = Counter(labels)
    print("   Class distribution:", dict(dist))

    pipeline = train_and_evaluate(texts, labels)
    save_model(pipeline, MODEL_PATH)
    write_router_ml(ROUTER_ML_PATH)

    # Test on provided examples
    print("\n🧠 Testing on provided examples:")
    test_tasks = [
        "Write a Python function",
        "Why is drainage important",
        "Design a council",
        "Hello!",
        "I want to hurt myself",
    ]
    for task in test_tasks:
        proba = pipeline.predict_proba([task])[0]
        pred_idx = proba.argmax()
        pred = pipeline.classes_[pred_idx]
        conf = float(proba[pred_idx])
        probs = {str(cls): round(float(p), 4) for cls, p in zip(pipeline.classes_, proba)}
        print(f"   '{task}' → {pred.upper()} (conf={conf:.4f})  {probs}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
