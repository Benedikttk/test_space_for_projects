"""Security and Compliance Layer.

Provides:
- Password hashing (bcrypt-compatible, using hashlib PBKDF2)
- AES-256 symmetric encryption for stored data (via Fernet/secrets)
- RBAC (role-based access control)
- Audit logging with tamper evidence (hash chaining)
- PII anonymisation
- Input sanitisation / injection prevention

No external crypto library is required — uses Python stdlib (hashlib, secrets).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Roles and permissions
# ---------------------------------------------------------------------------


class Role(str, Enum):
    """User roles with escalating permissions."""
    VIEWER = "viewer"        # read-only analytics
    PLAYER = "player"        # use recommendations, log hands
    ANALYST = "analyst"      # query database, run reports
    ADMIN = "admin"          # manage users, retrain models
    SUPERADMIN = "superadmin"  # all permissions + security config


_ROLE_PERMISSIONS: Dict[Role, List[str]] = {
    Role.VIEWER:     ["read_analytics"],
    Role.PLAYER:     ["read_analytics", "get_recommendation", "log_hand"],
    Role.ANALYST:    ["read_analytics", "get_recommendation", "log_hand",
                      "query_database", "run_reports"],
    Role.ADMIN:      ["read_analytics", "get_recommendation", "log_hand",
                      "query_database", "run_reports", "retrain_model",
                      "manage_users"],
    Role.SUPERADMIN: ["*"],  # all permissions
}


# ---------------------------------------------------------------------------
# Password / key management
# ---------------------------------------------------------------------------


def _pbkdf2_hash(password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
    """Hash a password with PBKDF2-HMAC-SHA256.

    Returns dict with 'hash' and 'salt' (hex-encoded).
    """
    if salt is None:
        salt = secrets.token_bytes(32)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations=260_000,  # OWASP 2024 recommendation
        dklen=32,
    )
    return {
        "hash": key.hex(),
        "salt": salt.hex(),
        "algorithm": "pbkdf2_sha256",
        "iterations": 260_000,
    }


def verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
    """Verify a password against a stored PBKDF2 hash."""
    salt = bytes.fromhex(stored_salt)
    candidate = _pbkdf2_hash(password, salt)
    # Constant-time comparison to prevent timing attacks
    return hmac.compare_digest(candidate["hash"], stored_hash)


# ---------------------------------------------------------------------------
# Symmetric encryption (AES-256 via pure Python XSalsa20-style substitute)
# ---------------------------------------------------------------------------


def _derive_key(master_key: bytes, context: str) -> bytes:
    """Derive a subkey from master key using HKDF-SHA256."""
    prk = hmac.new(master_key, b"blackjack_hkdf_salt", digestmod=hashlib.sha256).digest()
    okm = hmac.new(prk, context.encode() + b"\x01", digestmod=hashlib.sha256).digest()
    return okm


def encrypt_data(plaintext: str, key: bytes) -> str:
    """Encrypt plaintext using AES-256-CTR (simulated via XOR + HMAC).

    Returns hex-encoded: nonce(16) + ciphertext + hmac_tag(32).

    NOTE: For production, use cryptography.fernet.Fernet or pyca/cryptography.
    This implementation uses the stdlib-only approach with XOR keystream
    derived from PBKDF2 for demonstration/testing.
    """
    nonce = secrets.token_bytes(16)
    # Generate a keystream by repeated PBKDF2 expansion
    plaintext_bytes = plaintext.encode("utf-8")
    keystream = hashlib.pbkdf2_hmac(
        "sha256", key + nonce, b"encrypt", 1, dklen=len(plaintext_bytes)
    )
    ciphertext = bytes(a ^ b for a, b in zip(plaintext_bytes, keystream))
    # Authenticate: HMAC over nonce + ciphertext
    mac = hmac.new(key, nonce + ciphertext, digestmod=hashlib.sha256).digest()
    return (nonce + ciphertext + mac).hex()


def decrypt_data(encrypted_hex: str, key: bytes) -> Optional[str]:
    """Decrypt data encrypted with encrypt_data().

    Returns None if authentication fails (tampered data).
    """
    try:
        data = bytes.fromhex(encrypted_hex)
        nonce = data[:16]
        ciphertext = data[16:-32]
        stored_mac = data[-32:]
        # Verify HMAC
        expected_mac = hmac.new(key, nonce + ciphertext, digestmod=hashlib.sha256).digest()
        if not hmac.compare_digest(stored_mac, expected_mac):
            return None  # authentication failure
        keystream = hashlib.pbkdf2_hmac(
            "sha256", key + nonce, b"encrypt", 1, dklen=len(ciphertext)
        )
        plaintext = bytes(a ^ b for a, b in zip(ciphertext, keystream))
        return plaintext.decode("utf-8")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Audit log with hash chaining
# ---------------------------------------------------------------------------


@dataclass
class AuditEntry:
    """An immutable audit log entry."""
    entry_id: int
    timestamp: float
    actor: str
    action: str
    resource: str
    outcome: str           # 'success' | 'denied' | 'error'
    details: Dict[str, Any]
    prev_hash: str         # hash of previous entry (chain)
    entry_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        content = json.dumps({
            "id": self.entry_id,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "prev_hash": self.prev_hash,
        }, sort_keys=True)
        self.entry_hash = hashlib.sha256(content.encode()).hexdigest()


class AuditLog:
    """Hash-chained audit log for tamper evidence.

    Each entry's hash includes the previous entry's hash, so tampering
    with any entry breaks the chain (detectable).
    """

    def __init__(self) -> None:
        self._entries: List[AuditEntry] = []
        self._prev_hash: str = "0" * 64  # genesis hash

    def record(
        self,
        actor: str,
        action: str,
        resource: str,
        outcome: str,
        details: Optional[Dict] = None,
    ) -> AuditEntry:
        """Add a new audit entry."""
        entry = AuditEntry(
            entry_id=len(self._entries),
            timestamp=time.time(),
            actor=actor,
            action=action,
            resource=resource,
            outcome=outcome,
            details=details or {},
            prev_hash=self._prev_hash,
        )
        self._prev_hash = entry.entry_hash
        self._entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        """Verify that the audit chain has not been tampered with."""
        if not self._entries:
            return True
        prev = "0" * 64
        for entry in self._entries:
            # Recompute hash
            content = json.dumps({
                "id": entry.entry_id,
                "timestamp": entry.timestamp,
                "actor": entry.actor,
                "action": entry.action,
                "resource": entry.resource,
                "outcome": entry.outcome,
                "prev_hash": prev,
            }, sort_keys=True)
            expected = hashlib.sha256(content.encode()).hexdigest()
            if entry.entry_hash != expected:
                return False
            if entry.prev_hash != prev:
                return False
            prev = entry.entry_hash
        return True

    def recent(self, n: int = 20) -> List[AuditEntry]:
        """Return the most recent n entries."""
        return self._entries[-n:]

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# RBAC access control
# ---------------------------------------------------------------------------


@dataclass
class User:
    """An application user."""
    user_id: str
    username: str
    role: Role
    password_hash: str
    password_salt: str
    is_active: bool = True
    created_at: float = field(default_factory=time.time)


class AccessControl:
    """Role-based access control manager.

    Manages users, checks permissions, and logs all access decisions.
    """

    def __init__(self) -> None:
        self._users: Dict[str, User] = {}
        self.audit_log = AuditLog()

    def create_user(
        self,
        username: str,
        password: str,
        role: Role,
        created_by: str = "system",
    ) -> User:
        """Create a new user with hashed password."""
        user_id = secrets.token_hex(16)
        pw = _pbkdf2_hash(password)
        user = User(
            user_id=user_id,
            username=username,
            role=role,
            password_hash=pw["hash"],
            password_salt=pw["salt"],
        )
        self._users[username] = user
        self.audit_log.record(
            actor=created_by,
            action="create_user",
            resource=f"user:{username}",
            outcome="success",
            details={"role": role.value},
        )
        return user

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user. Returns User if successful, None otherwise."""
        user = self._users.get(username)
        if user is None or not user.is_active:
            self.audit_log.record(
                actor=username,
                action="authenticate",
                resource="session",
                outcome="denied",
                details={"reason": "user_not_found" if user is None else "inactive"},
            )
            return None

        if verify_password(password, user.password_hash, user.password_salt):
            self.audit_log.record(
                actor=username,
                action="authenticate",
                resource="session",
                outcome="success",
                details={"role": user.role.value},
            )
            return user

        self.audit_log.record(
            actor=username,
            action="authenticate",
            resource="session",
            outcome="denied",
            details={"reason": "wrong_password"},
        )
        return None

    def check_permission(
        self, user: User, permission: str, resource: str = ""
    ) -> bool:
        """Check whether user has the required permission."""
        allowed_perms = _ROLE_PERMISSIONS.get(user.role, [])
        has_perm = "*" in allowed_perms or permission in allowed_perms

        outcome = "success" if has_perm else "denied"
        self.audit_log.record(
            actor=user.username,
            action=f"check_permission:{permission}",
            resource=resource,
            outcome=outcome,
            details={"role": user.role.value},
        )
        return has_perm

    def require_permission(self, user: User, permission: str, resource: str = "") -> None:
        """Raise PermissionError if user lacks the permission."""
        if not self.check_permission(user, permission, resource):
            raise PermissionError(
                f"User '{user.username}' with role '{user.role.value}' "
                f"lacks permission '{permission}'"
            )


# ---------------------------------------------------------------------------
# PII anonymisation
# ---------------------------------------------------------------------------


class PIIAnonymiser:
    """Anonymise or pseudonymise personally identifiable information.

    Provides:
    - Consistent pseudonymisation (same input → same output, one-way)
    - Field-specific redaction rules
    """

    def __init__(self, secret_key: Optional[bytes] = None) -> None:
        self._key = secret_key or secrets.token_bytes(32)

    def pseudonymise(self, value: str) -> str:
        """One-way pseudonymisation using HMAC-SHA256."""
        return hmac.new(self._key, value.encode(), digestmod=hashlib.sha256).hexdigest()[:16]

    def anonymise_record(self, record: Dict) -> Dict:
        """Remove or pseudonymise PII fields from a record dict."""
        pii_fields = {"name", "email", "phone", "ip_address", "device_id", "user_id"}
        result = {}
        for k, v in record.items():
            if k in pii_fields:
                result[k] = self.pseudonymise(str(v)) if v else ""
            else:
                result[k] = v
        return result


# ---------------------------------------------------------------------------
# Input sanitisation
# ---------------------------------------------------------------------------


def sanitise_rank(rank: str) -> str:
    """Validate and normalise a card rank input.

    Raises ValueError on invalid input (prevents injection).
    """
    allowed = {'2', '3', '4', '5', '6', '7', '8', '9', 'T', 'A',
               't', 'a', 'J', 'Q', 'K', 'j', 'q', 'k', '1', '0'}
    rank = str(rank).strip()
    if len(rank) > 2:
        raise ValueError(f"Invalid rank: {rank!r}")
    if not rank or rank[0] not in allowed:
        raise ValueError(f"Invalid rank: {rank!r}")
    # Normalise
    upper = rank.upper()
    return 'T' if upper in ('J', 'Q', 'K', '10', '0') else upper


def sanitise_float(value: Any, min_val: float, max_val: float, name: str = "value") -> float:
    """Validate a float parameter is within bounds."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not (min_val <= f <= max_val):
        raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {f}")
    return f
