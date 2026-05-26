"""Prompt Registry — Version Control, A/B Testing, and Governance for MEOKCLAW

Git-based prompt version control with A/B testing, rollback, and analytics.

Features:
- Versioned prompt templates with git-style commits
- A/B testing framework (track which prompt version performs better)
- Auto-suggest improvements via Reflection Engine
- Environment promotion (dev → staging → prod)
- Template variables with validation
- Prompt lineage and audit trail

Usage:
    registry = PromptRegistry()
    
    # Register a prompt
    v1 = registry.register("code-review", {
        "system": "You are a senior engineer reviewing code.",
        "template": "Review this {language} code:\n\n{code}",
        "variables": {"language": "str", "code": "str"},
    }, author="nicholas")
    
    # A/B test a new version
    v2 = registry.register("code-review", {...}, parent=v1.id)
    
    # Render for use
    prompt = registry.render("code-review", {
        "language": "python",
        "code": "def hello(): print('hi')",
    })
    
    # Get analytics
    stats = registry.stats("code-review")
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class PromptVersion:
    id: str
    name: str  # prompt name (e.g., "code-review")
    version: int
    system_prompt: str
    template: str
    variables: Dict[str, str]
    metadata: Dict[str, Any]
    parent_id: Optional[str]
    created_at: float
    author: str
    is_active: bool = True
    traffic_split: float = 1.0  # 0.0-1.0, for A/B testing
    environment: str = "dev"  # dev, staging, prod


@dataclass
class PromptMetrics:
    version_id: str
    calls: int = 0
    avg_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    avg_tokens_out: float = 0.0
    user_ratings: List[int] = field(default_factory=list)
    success_rate: float = 1.0
    last_used: float = 0.0


@dataclass
class PromptTest:
    id: str
    prompt_name: str
    versions: List[str]  # version IDs being tested
    traffic_splits: List[float]
    start_time: float
    end_time: Optional[float]
    winner_id: Optional[str]
    is_active: bool = True


class PromptRegistry:
    """Git-style prompt registry with A/B testing and governance."""

    DATA_DIR = Path("data/prompts")

    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._versions: Dict[str, PromptVersion] = {}  # version_id -> PromptVersion
        self._by_name: Dict[str, List[str]] = {}  # name -> [version_ids]
        self._metrics: Dict[str, PromptMetrics] = {}  # version_id -> metrics
        self._tests: Dict[str, PromptTest] = {}  # test_id -> PromptTest
        self._load_all()

    def _load_all(self):
        """Load all saved prompts from disk."""
        if not self.DATA_DIR.exists():
            return
        for f in self.DATA_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                version = PromptVersion(**data)
                self._versions[version.id] = version
                self._by_name.setdefault(version.name, []).append(version.id)
            except Exception:
                pass

    def _save(self, version: PromptVersion):
        """Save a version to disk."""
        path = self.DATA_DIR / f"{version.id}.json"
        path.write_text(json.dumps(asdict(version), indent=2))

    def register(
        self,
        name: str,
        system_prompt: str,
        template: str,
        variables: Optional[Dict[str, str]] = None,
        author: str = "system",
        parent_id: Optional[str] = None,
        environment: str = "dev",
        traffic_split: float = 1.0,
    ) -> PromptVersion:
        """Register a new prompt version."""
        versions = self._by_name.get(name, [])
        version_num = len(versions) + 1

        version_id = f"{name}_v{version_num}_{secrets.token_hex(4)}"

        # Hash the content for integrity
        content_hash = hashlib.sha256(
            f"{system_prompt}{template}".encode()
        ).hexdigest()[:12]

        version = PromptVersion(
            id=version_id,
            name=name,
            version=version_num,
            system_prompt=system_prompt,
            template=template,
            variables=variables or {},
            metadata={"content_hash": content_hash},
            parent_id=parent_id,
            created_at=time.time(),
            author=author,
            environment=environment,
            traffic_split=traffic_split,
        )

        self._versions[version_id] = version
        self._by_name.setdefault(name, []).append(version_id)
        self._metrics[version_id] = PromptMetrics(version_id=version_id)
        self._save(version)

        return version

    def get(self, name: str, env: str = "prod") -> Optional[PromptVersion]:
        """Get the active version for a prompt in an environment."""
        version_ids = self._by_name.get(name, [])
        candidates = [
            self._versions[v] for v in version_ids
            if self._versions[v].environment == env and self._versions[v].is_active
        ]
        if not candidates:
            return None
        # Return highest version number
        return max(candidates, key=lambda v: v.version)

    def get_by_id(self, version_id: str) -> Optional[PromptVersion]:
        return self._versions.get(version_id)

    def render(
        self,
        name: str,
        variables: Dict[str, Any],
        env: str = "prod",
    ) -> Tuple[str, str, str]:
        """
        Render a prompt for use.
        Returns (system_prompt, user_prompt, version_id).
        """
        version = self.get(name, env)
        if not version:
            raise ValueError(f"Prompt '{name}' not found in environment '{env}'")

        # Validate variables
        for var, vtype in version.variables.items():
            if var not in variables:
                raise ValueError(f"Missing variable: {var}")

        # Render template
        try:
            user_prompt = version.template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Template rendering failed: {e}")

        return version.system_prompt, user_prompt, version.id

    def start_ab_test(
        self,
        prompt_name: str,
        version_ids: List[str],
        traffic_splits: Optional[List[float]] = None,
    ) -> PromptTest:
        """Start an A/B test between prompt versions."""
        if traffic_splits is None:
            n = len(version_ids)
            traffic_splits = [1.0 / n] * n

        if len(version_ids) != len(traffic_splits):
            raise ValueError("version_ids and traffic_splits must match")

        if abs(sum(traffic_splits) - 1.0) > 0.01:
            raise ValueError("Traffic splits must sum to 1.0")

        test_id = f"test_{secrets.token_hex(8)}"
        test = PromptTest(
            id=test_id,
            prompt_name=prompt_name,
            versions=version_ids,
            traffic_splits=traffic_splits,
            start_time=time.time(),
            end_time=None,
            winner_id=None,
            is_active=True,
        )
        self._tests[test_id] = test

        # Update versions with traffic splits
        for vid, split in zip(version_ids, traffic_splits):
            if vid in self._versions:
                self._versions[vid].traffic_split = split
                self._save(self._versions[vid])

        return test

    def get_test_version(self, test_id: str) -> Optional[PromptVersion]:
        """Get a version for a request based on traffic split."""
        test = self._tests.get(test_id)
        if not test or not test.is_active:
            return None

        import random
        r = random.random()
        cumulative = 0.0
        for vid, split in zip(test.versions, test.traffic_splits):
            cumulative += split
            if r <= cumulative:
                return self._versions.get(vid)

        return self._versions.get(test.versions[-1])

    def record_metrics(
        self,
        version_id: str,
        cost_usd: float,
        latency_ms: int,
        tokens_out: int,
        success: bool = True,
    ):
        """Record usage metrics for a prompt version."""
        if version_id not in self._metrics:
            self._metrics[version_id] = PromptMetrics(version_id=version_id)

        m = self._metrics[version_id]
        m.calls += 1
        m.last_used = time.time()

        # Running averages
        n = m.calls
        m.avg_cost_usd = (m.avg_cost_usd * (n - 1) + cost_usd) / n
        m.avg_latency_ms = (m.avg_latency_ms * (n - 1) + latency_ms) / n
        m.avg_tokens_out = (m.avg_tokens_out * (n - 1) + tokens_out) / n

        if not success:
            # Approximate success rate
            m.success_rate = (m.success_rate * (n - 1)) / n

    def end_ab_test(self, test_id: str, winner_id: Optional[str] = None) -> PromptTest:
        """End an A/B test and optionally declare a winner."""
        test = self._tests.get(test_id)
        if not test:
            raise ValueError("Test not found")

        test.is_active = False
        test.end_time = time.time()
        test.winner_id = winner_id

        if winner_id:
            # Promote winner to 100% traffic
            for vid in test.versions:
                if vid in self._versions:
                    v = self._versions[vid]
                    v.traffic_split = 1.0 if vid == winner_id else 0.0
                    v.is_active = (vid == winner_id)
                    self._save(v)

        return test

    def promote(self, version_id: str, to_env: str) -> PromptVersion:
        """Promote a version to a higher environment."""
        version = self._versions.get(version_id)
        if not version:
            raise ValueError("Version not found")

        env_order = {"dev": 0, "staging": 1, "prod": 2}
        current_level = env_order.get(version.environment, -1)
        target_level = env_order.get(to_env, -1)

        if target_level <= current_level:
            raise ValueError(f"Cannot promote from {version.environment} to {to_env}")

        version.environment = to_env
        self._save(version)
        return version

    def stats(self, name: str) -> Dict[str, Any]:
        """Get analytics for a prompt."""
        version_ids = self._by_name.get(name, [])
        versions = [self._versions[v] for v in version_ids]
        metrics = [self._metrics.get(v.id) for v in versions if v.id in self._metrics]

        total_calls = sum(m.calls for m in metrics if m)
        avg_cost = sum(m.avg_cost_usd for m in metrics if m) / max(len(metrics), 1)
        avg_latency = sum(m.avg_latency_ms for m in metrics if m) / max(len(metrics), 1)

        return {
            "name": name,
            "versions": len(versions),
            "active_versions": sum(1 for v in versions if v.is_active),
            "total_calls": total_calls,
            "avg_cost_usd": round(avg_cost, 6),
            "avg_latency_ms": round(avg_latency, 1),
            "version_breakdown": [
                {
                    "id": v.id,
                    "version": v.version,
                    "env": v.environment,
                    "active": v.is_active,
                    "author": v.author,
                    "calls": self._metrics.get(v.id, PromptMetrics(v.id)).calls,
                }
                for v in versions
            ],
        }

    def list_prompts(self) -> List[str]:
        return list(self._by_name.keys())

    def rollback(self, name: str, to_version: int) -> PromptVersion:
        """Rollback to a specific version number."""
        version_ids = self._by_name.get(name, [])
        target = None
        for vid in version_ids:
            v = self._versions[vid]
            if v.version == to_version:
                target = v
                break

        if not target:
            raise ValueError(f"Version {to_version} not found for prompt '{name}'")

        # Deactivate all other versions in prod
        for vid in version_ids:
            v = self._versions[vid]
            if v.environment == "prod" and v.id != target.id:
                v.is_active = False
                self._save(v)

        target.is_active = True
        target.environment = "prod"
        target.traffic_split = 1.0
        self._save(target)

        return target


# Singleton
prompt_registry = PromptRegistry()


if __name__ == "__main__":
    reg = PromptRegistry()

    # Register a prompt
    v1 = reg.register(
        name="code-review",
        system_prompt="You are a senior engineer. Review code for bugs, style, and performance.",
        template="Review this {language} code:\n\n```{language}\n{code}\n```",
        variables={"language": "str", "code": "str"},
        author="nicholas",
    )
    print(f"Registered v{v1.version}: {v1.id}")

    # Render it
    system, user, vid = reg.render("code-review", {
        "language": "python",
        "code": "def hello(): print('hi')",
    })
    print(f"\nRendered prompt (version {vid}):")
    print(f"System: {system[:60]}...")
    print(f"User: {user[:80]}...")

    # A/B test a new version
    v2 = reg.register(
        name="code-review",
        system_prompt="You are a principal engineer. Be thorough and mention security.",
        template="Review this {language} code:\n\n```{language}\n{code}\n```\n\nFocus on: bugs, security, performance.",
        variables={"language": "str", "code": "str"},
        author="nicholas",
        parent_id=v1.id,
    )

    test = reg.start_ab_test("code-review", [v1.id, v2.id], [0.5, 0.5])
    print(f"\nStarted A/B test: {test.id}")

    # Stats
    print(f"\nStats:")
    print(json.dumps(reg.stats("code-review"), indent=2))
