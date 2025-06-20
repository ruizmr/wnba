# pyright: reportMissingImports=false
"""JWT authentication helpers for Edge Serve.

The service expects an `Authorization: Bearer <token>` header on protected
routes.  Tokens are HS256 signed using the secret in `JWT_SECRET` env var.

Usage
-----
from fastapi import Depends
from serve.auth import verify_jwt

@app.post("/predict")
async def predict(req: Request, user=Depends(verify_jwt)): ...
"""
from __future__ import annotations

import os
from typing import Dict

import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

JWT_SECRET = os.getenv("JWT_SECRET", "edge-dev-secret-change-me")
JWT_ALGORITHM = "HS256"

auth_scheme = HTTPBearer(auto_error=False)


def verify_jwt(
    credentials: HTTPAuthorizationCredentials | None = Security(auth_scheme),
) -> Dict[str, str]:
    """Validate Authorization header and return decoded claims.

    Raises HTTPException(401/403) on missing or invalid token."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload  # type: ignore[return-value]
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Token expired") from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=403, detail="Invalid token") from exc