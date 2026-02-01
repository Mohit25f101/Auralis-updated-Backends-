# ==============================
# 📄 api/routes/feedback.py
# ==============================
"""
Feedback and Learning Endpoints
"""

from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException

from api.models.requests import FeedbackRequest, BatchFeedbackRequest
from api.models.responses import FeedbackResponse, LearningStatsResponse
from services.learning_system import LearningSystem
from config import settings

router = APIRouter(prefix="/feedback", tags=["Feedback & Learning"])

# Global learning system reference
_learning: Optional[LearningSystem] = None
_history: dict = {}


def init_feedback_services(learning: LearningSystem, history: dict):
    """Initialize feedback services"""
    global _learning, _history
    _learning = learning
    _history = history


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    📝 **Submit Correction Feedback**
    
    Helps the system learn from corrections.
    Provide the correct location/situation to improve future analyses.
    """
    if not _learning:
        raise HTTPException(503, "Learning system not initialized")
    
    if request.request_id not in _history:
        raise HTTPException(404, f"Request ID '{request.request_id}' not found in history")
    
    history_entry = _history[request.request_id]
    
    text = history_entry.get('transcription', {}).get('text', '')
    sounds = list(history_entry.get('sounds', {}).get('sounds', {}).keys())
    
    learned_items = []
    
    if request.correct_location:
        if request.correct_location not in settings.LOCATIONS:
            raise HTTPException(400, f"Invalid location: {request.correct_location}")
        
        _learning.learn('location', request.correct_location, text, sounds)
        learned_items.append(f"location -> {request.correct_location}")
    
    if request.correct_situation:
        if request.correct_situation not in settings.SITUATIONS:
            raise HTTPException(400, f"Invalid situation: {request.correct_situation}")
        
        _learning.learn('situation', request.correct_situation, text, sounds)
        learned_items.append(f"situation -> {request.correct_situation}")
    
    if request.correct_emotions:
        for emotion, value in request.correct_emotions.items():
            _learning.learn('emotion', emotion, text, sounds)
            learned_items.append(f"emotion -> {emotion}")
    
    return FeedbackResponse(
        status="learned",
        request_id=request.request_id,
        corrections_applied=learned_items,
        total_corrections=_learning.corrections,
        timestamp=datetime.now().isoformat()
    )


@router.post("/batch", response_model=FeedbackResponse)
async def submit_batch_feedback(request: BatchFeedbackRequest):
    """
    📦 **Submit Batch Feedback**
    
    Submit multiple corrections at once.
    """
    if not _learning:
        raise HTTPException(503, "Learning system not initialized")
    
    total_learned = []
    errors = []
    
    for feedback in request.feedbacks:
        try:
            if feedback.request_id not in _history:
                errors.append(f"Request {feedback.request_id} not found")
                continue
            
            history_entry = _history[feedback.request_id]
            text = history_entry.get('transcription', {}).get('text', '')
            sounds = list(history_entry.get('sounds', {}).get('sounds', {}).keys())
            
            if feedback.correct_location:
                _learning.learn('location', feedback.correct_location, text, sounds)
                total_learned.append(f"{feedback.request_id}: location")
            
            if feedback.correct_situation:
                _learning.learn('situation', feedback.correct_situation, text, sounds)
                total_learned.append(f"{feedback.request_id}: situation")
                
        except Exception as e:
            errors.append(f"Error processing {feedback.request_id}: {str(e)}")
    
    return FeedbackResponse(
        status="batch_learned" if total_learned else "no_changes",
        request_id="batch",
        corrections_applied=total_learned,
        total_corrections=_learning.corrections,
        errors=errors if errors else None,
        timestamp=datetime.now().isoformat()
    )


@router.get("/stats", response_model=LearningStatsResponse)
async def get_learning_stats():
    """
    📊 **Learning Statistics**
    
    Get current learning system statistics.
    """
    if not _learning:
        raise HTTPException(503, "Learning system not initialized")
    
    stats = _learning.get_stats()
    
    return LearningStatsResponse(
        total_corrections=stats.get("total_corrections", 0),
        location_corrections=stats.get("location_corrections", 0),
        situation_corrections=stats.get("situation_corrections", 0),
        emotion_corrections=stats.get("emotion_corrections", 0),
        top_learned_patterns=stats.get("top_patterns", []),
        last_saved=stats.get("last_saved", "never"),
        file_size_kb=stats.get("file_size_kb", 0),
        timestamp=datetime.now().isoformat()
    )


@router.post("/save")
async def save_learning():
    """
    💾 **Force Save Learning Data**
    
    Manually trigger saving of learned patterns.
    """
    if not _learning:
        raise HTTPException(503, "Learning system not initialized")
    
    try:
        _learning.save()
        return {
            "status": "saved",
            "corrections": _learning.corrections,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to save: {str(e)}")


@router.delete("/reset")
async def reset_learning():
    """
    🗑️ **Reset Learning Data**
    
    ⚠️ Warning: This will delete all learned patterns!
    """
    if not _learning:
        raise HTTPException(503, "Learning system not initialized")
    
    try:
        _learning.reset()
        return {
            "status": "reset",
            "message": "All learning data has been cleared",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to reset: {str(e)}")


@router.get("/labels")
async def get_valid_labels():
    """
    🏷️ **Get Valid Labels**
    
    Returns all valid location and situation labels for feedback.
    """
    return {
        "locations": settings.LOCATIONS,
        "situations": settings.SITUATIONS,
        "emotions": [
            "neutral", "happy", "sad", "angry", "fearful",
            "surprised", "disgusted", "urgent", "excited", "hesitant"
        ]
    }