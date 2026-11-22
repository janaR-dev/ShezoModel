"""Authentication and role-based access control.

Deliberately small: scrypt password hashing, opaque bearer tokens stored in the
database, 12-hour expiry. Every privileged read goes through
``AuthService.require_user`` plus an explicit patient-access check in the API.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from .models import Role, User, utcnow
from .storage import Repository, new_id

TOKEN_TTL_HOURS = 12
_SCRYPT = dict(n=2**14, r=8, p=1, dklen=32)


class AuthError(Exception):
    """Raised for invalid credentials or expired/invalid tokens."""


class RoleError(Exception):
    """Raised when a valid session lacks the role required by an action."""


def hash_password(password: str, salt: str) -> str:
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt.encode("utf-8"), **_SCRYPT)
    return digest.hex()


def create_user(
    repo: Repository,
    email: str,
    password: str,
    role: Role,
    display_name: str = "",
) -> User:
    """Create a user with a hashed password. Emails are unique and case-folded."""
    if not email or "@" not in email:
        raise ValueError("a valid email is required")
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    if repo.get_user_by_email(email):
        raise ValueError("an account with that email already exists")
    salt = secrets.token_hex(16)
    user = User(
        id=new_id("usr"),
        email=email.lower().strip(),
        display_name=display_name or email.split("@")[0],
        role=role,
        created_at=utcnow(),
        password_hash=hash_password(password, salt),
        salt=salt,
    )
    return repo.add_user(user)


class AuthService:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    def login(self, email: str, password: str) -> "tuple[str, User]":
        user = self.repo.get_user_by_email(email)
        if not user:
            # Constant-ish work to avoid a trivial user-enumeration oracle.
            hash_password(password, "shezo-dummy-salt")
            raise AuthError("invalid email or password")
        candidate = hash_password(password, user.salt)
        if not hmac.compare_digest(candidate, user.password_hash):
            raise AuthError("invalid email or password")
        token = secrets.token_urlsafe(32)
        expires = (datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS)).replace(
            microsecond=0
        ).isoformat()
        self.repo.save_token(token, user.id, expires)
        self.repo.audit(user.id, "auth.login", user.id)
        return token, user

    def logout(self, token: str) -> None:
        self.repo.delete_token(token)

    def require_user(self, token: Optional[str]) -> User:
        if not token:
            raise AuthError("missing bearer token")
        user = self.repo.token_user(token)
        if not user:
            raise AuthError("invalid or expired token")
        return user

    def require_role(self, token: Optional[str], *roles: Role) -> User:
        user = self.require_user(token)
        if roles and user.role not in roles:
            raise RoleError(f"this action requires role: {', '.join(r.value for r in roles)}")
        return user


def new_os_token() -> str:
    """Helper for seeding scripts that need a random non-password secret."""
    return os.urandom(16).hex()
