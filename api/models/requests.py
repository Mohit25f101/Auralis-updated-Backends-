# ==============================
# 📄 api/models/requests.py
# ==============================
"""
Request Models for Auralis Ultimate API
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field, EmailStr


class FeedbackRequest(BaseModel):
    """Feedback submission for learning"""
    request_id: str = Field(..., description="ID of the analysis request to provide feedback on")
    correct_location: Optional[str] = Field(None, description="Correct location if misdetected")
    correct_situation: Optional[str] = Field(None, description="Correct situation if misdetected")
    correct_emotions: Optional[Dict[str, float]] = Field(None, description="Correct emotions if misdetected")
    notes: Optional[str] = Field(None, description="Additional notes")


class BatchFeedbackRequest(BaseModel):
    """Batch feedback submission"""
    feedbacks: List[FeedbackRequest] = Field(..., description="List of feedback items")


class LoginRequest(BaseModel):
    """User login request"""
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=6, description="User password")


class RegisterRequest(BaseModel):
    """User registration request"""
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=6, description="User password")
    name: str = Field(..., min_length=2, description="User name")
