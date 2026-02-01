# ==============================
# 📄 api/routes/auth.py
# ==============================
"""
Authentication Endpoints (Mock Implementation)
"""

import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict

from fastapi import APIRouter, HTTPException, Depends, Header

from api.models.requests import LoginRequest, RegisterRequest
from api.models.responses import AuthResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Mock user store
_users: Dict[str, dict] = {}
_tokens: Dict[str, dict] = {}


def _hash_password(password: str) -> str:
    """Simple password hashing"""
    return hashlib.sha256(password.encode()).hexdigest()


def _generate_token() -> str:
    """Generate auth token"""
    return uuid.uuid4().hex + uuid.uuid4().hex


async def verify_token(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Verify authorization token"""
    if not authorization:
        return None
    
    token = authorization.replace("Bearer ", "")
    
    if token in _tokens:
        token_data = _tokens[token]
        if datetime.fromisoformat(token_data["expires"]) > datetime.now():
            return token_data
    
    return None


@router.post("/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    """
    📝 **Register New User**
    """
    if request.email in _users:
        raise HTTPException(400, "Email already registered")
    
    user_id = uuid.uuid4().hex[:12]
    
    _users[request.email] = {
        "id": user_id,
        "email": request.email,
        "name": request.name,
        "password_hash": _hash_password(request.password),
        "created_at": datetime.now().isoformat()
    }
    
    token = _generate_token()
    expires = datetime.now() + timedelta(days=7)
    
    _tokens[token] = {
        "user_id": user_id,
        "email": request.email,
        "expires": expires.isoformat()
    }
    
    return AuthResponse(
        status="registered",
        user_id=user_id,
        email=request.email,
        token=token,
        expires=expires.isoformat(),
        message="Registration successful"
    )


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """
    🔐 **User Login**
    """
    if request.email not in _users:
        raise HTTPException(401, "Invalid credentials")
    
    user = _users[request.email]
    
    if user["password_hash"] != _hash_password(request.password):
        raise HTTPException(401, "Invalid credentials")
    
    token = _generate_token()
    expires = datetime.now() + timedelta(days=7)
    
    _tokens[token] = {
        "user_id": user["id"],
        "email": request.email,
        "expires": expires.isoformat()
    }
    
    return AuthResponse(
        status="logged_in",
        user_id=user["id"],
        email=request.email,
        token=token,
        expires=expires.isoformat(),
        message="Login successful"
    )


@router.post("/logout")
async def logout(token_data: dict = Depends(verify_token)):
    """
    🚪 **User Logout**
    """
    if not token_data:
        raise HTTPException(401, "Not authenticated")
    
    # Remove token
    tokens_to_remove = [
        t for t, data in _tokens.items() 
        if data["user_id"] == token_data["user_id"]
    ]
    
    for t in tokens_to_remove:
        del _tokens[t]
    
    return {"status": "logged_out", "message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_current_user(token_data: dict = Depends(verify_token)):
    """
    👤 **Get Current User**
    """
    if not token_data:
        raise HTTPException(401, "Not authenticated")
    
    email = token_data["email"]
    
    if email not in _users:
        raise HTTPException(404, "User not found")
    
    user = _users[email]
    
    return UserResponse(
        user_id=user["id"],
        email=user["email"],
        name=user["name"],
        created_at=user["created_at"]
    )


@router.post("/refresh")
async def refresh_token(token_data: dict = Depends(verify_token)):
    """
    🔄 **Refresh Token**
    """
    if not token_data:
        raise HTTPException(401, "Not authenticated")
    
    new_token = _generate_token()
    expires = datetime.now() + timedelta(days=7)
    
    _tokens[new_token] = {
        "user_id": token_data["user_id"],
        "email": token_data["email"],
        "expires": expires.isoformat()
    }
    
    return {
        "token": new_token,
        "expires": expires.isoformat()
    }