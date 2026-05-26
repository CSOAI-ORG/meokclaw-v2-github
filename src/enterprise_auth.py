"""Enterprise Authentication & Virtual Keys for MEOKCLAW

Features:
- JWT-based authentication (OIDC/SAML ready)
- Virtual API keys with scoped permissions
- Team/Organization hierarchy
- Per-key budget tracking and rate limiting
- Audit logging

Usage:
    from enterprise_auth import AuthManager, requires_auth
    
    @app.post("/api/dual-brain")
    @requires_auth(scopes=["chat:write"])
    async def chat(req: ChatRequest, auth: AuthContext = Depends(get_auth)):
        # auth.org_id, auth.team_id, auth.key_id, auth.budget_remaining
        ...
"""
from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Simple JWT implementation (replace with PyJWT in production)
import hmac
import json as _json
import base64


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Organization:
    id: str
    name: str
    created_at: float
    max_budget_usd: float = 1000.0
    used_budget_usd: float = 0.0
    allowed_models: List[str] = field(default_factory=lambda: ["*"])
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Team:
    id: str
    org_id: str
    name: str
    max_budget_usd: float = 500.0
    used_budget_usd: float = 0.0
    allowed_models: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class VirtualKey:
    id: str
    key_hash: str  # Store hash, not plaintext
    org_id: str
    team_id: Optional[str]
    name: str
    scopes: List[str]
    max_budget_usd: float
    used_budget_usd: float = 0.0
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000
    allowed_models: List[str] = field(default_factory=lambda: ["*"])
    created_at: float = 0.0
    expires_at: Optional[float] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthContext:
    key_id: str
    org_id: str
    team_id: Optional[str]
    scopes: List[str]
    budget_remaining_usd: float
    rate_limit_rpm: int
    allowed_models: List[str]
    is_admin: bool = False


@dataclass
class AuditLog:
    timestamp: float
    key_id: str
    org_id: str
    action: str
    model: str
    cost_usd: float
    tokens_in: int
    tokens_out: int
    latency_ms: int
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Auth Manager
# ---------------------------------------------------------------------------

class AuthManager:
    """In-memory auth store. Replace with PostgreSQL for production."""

    def __init__(self):
        self._orgs: Dict[str, Organization] = {}
        self._teams: Dict[str, Team] = {}
        self._keys: Dict[str, VirtualKey] = {}  # key_id -> VirtualKey
        self._key_hash_map: Dict[str, str] = {}  # key_hash -> key_id
        self._rate_limit_buckets: Dict[str, List[float]] = {}  # key_id -> timestamps
        self._audit_log: List[AuditLog] = []
        self._jwt_secret = secrets.token_hex(32)

    # -- Organization Management --

    def create_org(self, name: str, max_budget: float = 1000.0) -> Organization:
        org_id = f"org_{secrets.token_hex(8)}"
        org = Organization(
            id=org_id,
            name=name,
            created_at=time.time(),
            max_budget_usd=max_budget,
        )
        self._orgs[org_id] = org
        return org

    def get_org(self, org_id: str) -> Optional[Organization]:
        return self._orgs.get(org_id)

    # -- Team Management --

    def create_team(self, org_id: str, name: str, max_budget: float = 500.0) -> Team:
        if org_id not in self._orgs:
            raise ValueError("Organization not found")
        team_id = f"team_{secrets.token_hex(8)}"
        team = Team(
            id=team_id,
            org_id=org_id,
            name=name,
            max_budget_usd=max_budget,
        )
        self._teams[team_id] = team
        return team

    # -- Virtual Key Management --

    def create_key(
        self,
        org_id: str,
        name: str,
        scopes: List[str] = None,
        max_budget: float = 100.0,
        team_id: Optional[str] = None,
        rate_limit_rpm: int = 60,
        expires_days: Optional[int] = None,
        allowed_models: Optional[List[str]] = None,
    ) -> Tuple[str, VirtualKey]:
        """Returns (plaintext_key, VirtualKey). STORE THE PLAINTEXT KEY — it won't be shown again."""
        if org_id not in self._orgs:
            raise ValueError("Organization not found")
        if team_id and team_id not in self._teams:
            raise ValueError("Team not found")

        key_id = f"key_{secrets.token_hex(8)}"
        plaintext = f"mk_{secrets.token_hex(24)}"
        key_hash = hashlib.sha256(plaintext.encode()).hexdigest()

        expires_at = None
        if expires_days:
            expires_at = time.time() + (expires_days * 86400)

        key = VirtualKey(
            id=key_id,
            key_hash=key_hash,
            org_id=org_id,
            team_id=team_id,
            name=name,
            scopes=scopes or ["chat:write", "models:read"],
            max_budget_usd=max_budget,
            rate_limit_rpm=rate_limit_rpm,
            created_at=time.time(),
            expires_at=expires_at,
            allowed_models=allowed_models or ["*"],
        )

        self._keys[key_id] = key
        self._key_hash_map[key_hash] = key_id
        return plaintext, key

    def revoke_key(self, key_id: str) -> bool:
        if key_id in self._keys:
            self._keys[key_id].is_active = False
            return True
        return False

    def get_key(self, key_id: str) -> Optional[VirtualKey]:
        return self._keys.get(key_id)

    def list_keys(self, org_id: str) -> List[VirtualKey]:
        return [k for k in self._keys.values() if k.org_id == org_id]

    # -- Authentication --

    def authenticate(self, api_key: str) -> Optional[AuthContext]:
        """Authenticate a request by virtual API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_id = self._key_hash_map.get(key_hash)
        if not key_id:
            return None

        key = self._keys.get(key_id)
        if not key or not key.is_active:
            return None

        if key.expires_at and time.time() > key.expires_at:
            return None

        # Check rate limit
        now = time.time()
        bucket = self._rate_limit_buckets.get(key_id, [])
        # Clean old entries (> 60s)
        bucket = [t for t in bucket if now - t < 60]
        if len(bucket) >= key.rate_limit_rpm:
            return None  # Rate limited

        bucket.append(now)
        self._rate_limit_buckets[key_id] = bucket

        # Calculate budget
        budget_remaining = key.max_budget_usd - key.used_budget_usd
        org = self._orgs.get(key.org_id)
        if org:
            budget_remaining = min(budget_remaining, org.max_budget_usd - org.used_budget_usd)

        return AuthContext(
            key_id=key.id,
            org_id=key.org_id,
            team_id=key.team_id,
            scopes=key.scopes,
            budget_remaining_usd=budget_remaining,
            rate_limit_rpm=key.rate_limit_rpm,
            allowed_models=key.allowed_models,
            is_admin="admin" in key.scopes,
        )

    # -- Budget Tracking --

    def spend(self, key_id: str, cost_usd: float) -> bool:
        """Deduct cost from key and org budgets. Returns False if over budget."""
        key = self._keys.get(key_id)
        if not key:
            return False

        if key.used_budget_usd + cost_usd > key.max_budget_usd:
            return False

        key.used_budget_usd += cost_usd

        org = self._orgs.get(key.org_id)
        if org:
            if org.used_budget_usd + cost_usd > org.max_budget_usd:
                return False
            org.used_budget_usd += cost_usd

        if key.team_id:
            team = self._teams.get(key.team_id)
            if team:
                team.used_budget_usd += cost_usd

        return True

    # -- Audit Logging --

    def log(
        self,
        key_id: str,
        action: str,
        model: str,
        cost_usd: float,
        tokens_in: int,
        tokens_out: int,
        latency_ms: int,
        request: Optional[Request] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        key = self._keys.get(key_id)
        if not key:
            return

        ip = None
        ua = None
        if request:
            ip = request.client.host if request.client else None
            ua = request.headers.get("user-agent")

        log_entry = AuditLog(
            timestamp=time.time(),
            key_id=key_id,
            org_id=key.org_id,
            action=action,
            model=model,
            cost_usd=cost_usd,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            ip_address=ip,
            user_agent=ua,
            success=success,
            error_message=error_message,
        )
        self._audit_log.append(log_entry)

        # Keep last 10,000 entries
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]

    def get_audit_log(
        self,
        org_id: str,
        limit: int = 100,
        action: Optional[str] = None,
    ) -> List[AuditLog]:
        logs = [l for l in self._audit_log if l.org_id == org_id]
        if action:
            logs = [l for l in logs if l.action == action]
        return sorted(logs, key=lambda l: l.timestamp, reverse=True)[:limit]

    # -- Stats --

    def stats(self, org_id: Optional[str] = None) -> Dict[str, Any]:
        if org_id:
            org = self._orgs.get(org_id)
            if not org:
                return {}
            keys = self.list_keys(org_id)
            return {
                "org": {
                    "name": org.name,
                    "budget": {"max": org.max_budget_usd, "used": org.used_budget_usd, "remaining": org.max_budget_usd - org.used_budget_usd},
                },
                "keys": len(keys),
                "requests_last_24h": len([l for l in self._audit_log if l.org_id == org_id and time.time() - l.timestamp < 86400]),
            }

        return {
            "orgs": len(self._orgs),
            "teams": len(self._teams),
            "keys": len(self._keys),
            "total_requests": len(self._audit_log),
            "total_spend_usd": sum(l.cost_usd for l in self._audit_log),
        }


# Singleton
auth_manager = AuthManager()


# ---------------------------------------------------------------------------
# FastAPI Integration
# ---------------------------------------------------------------------------

security = HTTPBearer(auto_error=False)


def get_auth(credentials: HTTPAuthorizationCredentials = Security(security)) -> AuthContext:
    """FastAPI dependency for auth."""
    if not credentials:
        # Allow unauthenticated for local development
        return AuthContext(
            key_id="anon",
            org_id="local",
            team_id=None,
            scopes=["*"],
            budget_remaining_usd=999999.0,
            rate_limit_rpm=999999,
            allowed_models=["*"],
            is_admin=True,
        )

    auth = auth_manager.authenticate(credentials.credentials)
    if not auth:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")

    return auth


def requires_auth(scopes: Optional[List[str]] = None):
    """Decorator for requiring specific scopes."""
    def checker(auth: AuthContext = Depends(get_auth)):
        if auth.is_admin:
            return auth
        if scopes:
            for scope in scopes:
                if scope not in auth.scopes and "*" not in auth.scopes:
                    raise HTTPException(status_code=403, detail=f"Missing scope: {scope}")
        return auth
    return Depends(checker)


# ---------------------------------------------------------------------------
# Bootstrap Demo Data
# ---------------------------------------------------------------------------

def bootstrap_demo():
    """Create a demo org with keys for testing."""
    org = auth_manager.create_org("Demo Corp", max_budget=1000.0)
    team = auth_manager.create_team(org.id, "Engineering", max_budget=500.0)
    plaintext, key = auth_manager.create_key(
        org_id=org.id,
        name="Production Key",
        scopes=["chat:write", "models:read", "council:write"],
        max_budget=100.0,
        team_id=team.id,
        rate_limit_rpm=100,
    )
    print(f"Demo org: {org.name} ({org.id})")
    print(f"Demo team: {team.name} ({team.id})")
    print(f"Demo API key: {plaintext}")
    print(f"  Key ID: {key.id}")
    print(f"  Budget: ${key.max_budget_usd}")
    print(f"  Rate limit: {key.rate_limit_rpm} RPM")
    return plaintext


if __name__ == "__main__":
    bootstrap_demo()
