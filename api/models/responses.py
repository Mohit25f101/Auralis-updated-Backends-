# ==============================
# api/models/responses.py - COMPLETE FIXED VERSION
# ==============================
"""
Response Models for API Endpoints
All response schemas for Auralis Ultimate
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════
# EMOTION MODELS
# ══════════════════════════════════════════════════════════════

class MicroEmotions(BaseModel):
    """Micro-emotion detection results"""
    urgency: float = Field(default=0.0, ge=0.0, le=1.0)
    excitement: float = Field(default=0.0, ge=0.0, le=1.0)
    hesitation: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    stress: float = Field(default=0.0, ge=0.0, le=1.0)
    calmness: float = Field(default=0.0, ge=0.0, le=1.0)


class EmotionDetails(BaseModel):
    """Detailed emotion analysis"""
    primary_emotion: str = Field(default="neutral")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    micro_emotions: Optional[MicroEmotions] = None
    intensity: int = Field(default=5, ge=1, le=10)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.5, ge=0.0, le=1.0)
    asian_context_patterns: Optional[Dict[str, Any]] = None


# ══════════════════════════════════════════════════════════════
# CONFIDENCE MODELS
# ══════════════════════════════════════════════════════════════

class ConfidenceDetails(BaseModel):
    """Speaker confidence scoring details"""
    overall_score: int = Field(default=5, ge=1, le=10)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    certainty_indicators: List[str] = Field(default_factory=list)
    uncertainty_indicators: List[str] = Field(default_factory=list)
    recommendation: str = Field(default="")


# ══════════════════════════════════════════════════════════════
# IMPROVEMENT MODELS
# ══════════════════════════════════════════════════════════════

class ImprovementSuggestion(BaseModel):
    """Single improvement suggestion"""
    category: str = Field(default="general")
    priority: str = Field(default="medium")
    suggestion: str = Field(default="")
    rationale: str = Field(default="")
    example: Optional[str] = None


class ImprovementDetails(BaseModel):
    """Contextual improvement suggestions"""
    suggestions: List[ImprovementSuggestion] = Field(default_factory=list)
    overall_quality_score: int = Field(default=5, ge=1, le=10)
    strengths: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str] = Field(default_factory=list)
    contextual_tips: List[str] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# LANGUAGE MODELS
# ══════════════════════════════════════════════════════════════

class LanguageInfo(BaseModel):
    """Language detection information"""
    code: str = Field(default="en")
    name: str = Field(default="English")
    native_name: Optional[str] = None
    script: str = Field(default="Latin")
    family: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TranscriptionDetails(BaseModel):
    """Detailed transcription information"""
    text: str = Field(default="")
    original_text: Optional[str] = None
    detected_language: str = Field(default="en")
    detected_language_name: str = Field(default="English")
    is_translated: bool = Field(default=False)
    translation_note: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    word_count: int = Field(default=0)
    timestamps: Optional[List[Dict[str, Any]]] = None


# ══════════════════════════════════════════════════════════════
# SEMANTIC MODELS
# ══════════════════════════════════════════════════════════════

class SemanticAnalysis(BaseModel):
    """Semantic understanding results"""
    intent: str = Field(default="unknown")
    intent_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    content_type: str = Field(default="other")
    meaning_summary: str = Field(default="")
    detailed_meaning: str = Field(default="")
    what_they_mean: str = Field(default="")
    key_points: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    questions_asked: List[str] = Field(default_factory=list)
    has_questions: bool = Field(default=False)
    has_actions: bool = Field(default=False)
    target_audience: str = Field(default="general")
    urgency_level: str = Field(default="normal")
    sentiment: str = Field(default="neutral")
    listener_should: List[str] = Field(default_factory=list)


class EntityExtraction(BaseModel):
    """Extracted entities from speech"""
    train_numbers: List[str] = Field(default_factory=list)
    flight_numbers: List[str] = Field(default_factory=list)
    platform_numbers: List[str] = Field(default_factory=list)
    gate_numbers: List[str] = Field(default_factory=list)
    times: List[str] = Field(default_factory=list)
    durations: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    stations: List[str] = Field(default_factory=list)
    cities: List[str] = Field(default_factory=list)
    person_names: List[str] = Field(default_factory=list)
    phone_numbers: List[str] = Field(default_factory=list)
    money: List[str] = Field(default_factory=list)
    summary: str = Field(default="")
    has_transport_info: bool = Field(default=False)
    has_time_info: bool = Field(default=False)


# ══════════════════════════════════════════════════════════════
# LOCATION MODELS
# ══════════════════════════════════════════════════════════════

class LocationDetails(BaseModel):
    """Location detection details"""
    location: str = Field(default="Unknown")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    is_verified: bool = Field(default=False)
    detection_method: str = Field(default="combined")
    is_talking_about_location: bool = Field(default=False)
    warnings: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class AcousticAnalysis(BaseModel):
    """Acoustic feature analysis"""
    reverb_estimate: float = Field(default=0.5)
    noise_floor: float = Field(default=0.5)
    is_outdoor: bool = Field(default=False)
    crowd_density: str = Field(default="unknown")
    snr_estimate: float = Field(default=20.0)
    dynamic_range: float = Field(default=0.5)


# ══════════════════════════════════════════════════════════════
# MAIN ANALYSIS RESULT
# ══════════════════════════════════════════════════════════════

class AnalysisResult(BaseModel):
    """Complete analysis result"""
    
    # Core Analysis
    location: str = Field(default="Unknown")
    location_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    situation: str = Field(default="Unknown")
    situation_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    is_emergency: bool = Field(default=False)
    emergency_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Transcription
    transcribed_text: str = Field(default="")
    original_text: Optional[str] = None
    text_reliable: bool = Field(default=False)
    detected_language: str = Field(default="en")
    detected_language_name: str = Field(default="English")
    is_translated: bool = Field(default=False)
    translation_note: Optional[str] = None
    
    # Language Details
    language_info: Optional[LanguageInfo] = None
    
    # Sound Detection
    detected_sounds: List[str] = Field(default_factory=list)
    sound_confidences: Optional[Dict[str, float]] = None
    
    # Emotions
    emotions: Optional[EmotionDetails] = None
    
    # Confidence Scoring
    speaker_confidence: Optional[ConfidenceDetails] = None
    
    # Improvements
    improvement_suggestions: Optional[ImprovementDetails] = None
    
    # Semantic Analysis
    semantic_analysis: Optional[SemanticAnalysis] = None
    
    # Entity Extraction
    entities: Optional[EntityExtraction] = None
    
    # Evidence
    evidence: List[str] = Field(default_factory=list)
    summary: str = Field(default="")
    
    # Metadata
    processing_time_ms: float = Field(default=0.0)
    request_id: str = Field(default="")
    timestamp: str = Field(default="")
    audio_duration: float = Field(default=0.0)
    audio_sample_rate: int = Field(default=16000)
    
    # Regional Context
    asian_context_applied: bool = Field(default=False)
    regional_adaptations: Optional[Dict[str, Any]] = None
    languages_supported: int = Field(default=99)


class DetailedAnalysisResult(AnalysisResult):
    """Extended analysis result with additional details"""
    
    all_location_scores: Optional[Dict[str, float]] = None
    all_situation_scores: Optional[Dict[str, float]] = None
    all_sounds: Optional[Dict[str, float]] = None
    keyword_matches: Optional[Dict[str, List[str]]] = None
    sound_location_hints: Optional[List[tuple]] = None
    sound_situation_hints: Optional[List[tuple]] = None
    raw_transcription: Optional[str] = None
    acoustic_analysis: Optional[AcousticAnalysis] = None
    location_details: Optional[LocationDetails] = None


class QuickAnalysisResult(BaseModel):
    """Quick analysis result with essential fields only"""
    
    location: str = Field(default="Unknown")
    location_confidence: float = Field(default=0.5)
    situation: str = Field(default="Unknown")
    situation_confidence: float = Field(default=0.5)
    is_emergency: bool = Field(default=False)
    transcribed_text: str = Field(default="")
    original_text: Optional[str] = None
    detected_language: str = Field(default="en")
    detected_language_name: str = Field(default="English")
    is_translated: bool = Field(default=False)
    detected_sounds: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    request_id: str = Field(default="")


class EmotionAnalysisResult(BaseModel):
    """Emotion-focused analysis result"""
    
    primary_emotion: str = Field(default="neutral")
    emotion_confidence: float = Field(default=0.5)
    micro_emotions: Optional[Dict[str, float]] = None
    emotional_intensity: int = Field(default=5)
    speaker_confidence_score: int = Field(default=5)
    confidence_breakdown: Optional[Dict[str, float]] = None
    emotional_trajectory: Optional[List[Dict[str, Any]]] = None
    asian_context_patterns: Optional[Dict[str, Any]] = None
    transcribed_text: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    request_id: str = Field(default="")


class TranslationResult(BaseModel):
    """Translation-focused result"""
    
    original_text: str = Field(default="")
    english_text: str = Field(default="")
    detected_language: str = Field(default="unknown")
    detected_language_name: str = Field(default="Unknown")
    confidence: float = Field(default=0.5)
    word_count_original: int = Field(default=0)
    word_count_english: int = Field(default=0)
    timestamps: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: float = Field(default=0.0)
    request_id: str = Field(default="")


# ══════════════════════════════════════════════════════════════
# HEALTH & STATUS MODELS
# ══════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(default="unknown")
    version: str = Field(default="5.0.0")
    models: Dict[str, bool] = Field(default_factory=dict)
    languages_supported: int = Field(default=99)
    timestamp: str = Field(default="")
    uptime_seconds: int = Field(default=0)


class SystemStatusResponse(BaseModel):
    """System status response"""
    
    status: str = Field(default="unknown")
    version: str = Field(default="5.0.0")
    python_version: str = Field(default="")
    memory_used_mb: float = Field(default=0.0)
    memory_total_mb: float = Field(default=0.0)
    memory_percent: float = Field(default=0.0)
    cpu_percent: float = Field(default=0.0)
    services: Dict[str, bool] = Field(default_factory=dict)
    languages_supported: int = Field(default=99)
    uptime_seconds: int = Field(default=0)
    timestamp: str = Field(default="")


class ModelStatusResponse(BaseModel):
    """Model status response"""
    
    models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    languages_supported: int = Field(default=99)
    timestamp: str = Field(default="")


# ══════════════════════════════════════════════════════════════
# FEEDBACK & LEARNING MODELS
# ══════════════════════════════════════════════════════════════

class FeedbackResponse(BaseModel):
    """Feedback submission response"""
    
    status: str = Field(default="")
    request_id: str = Field(default="")
    corrections_applied: List[str] = Field(default_factory=list)
    total_corrections: int = Field(default=0)
    errors: Optional[List[str]] = None
    timestamp: str = Field(default="")


class LearningStatsResponse(BaseModel):
    """Learning statistics response"""
    
    total_corrections: int = Field(default=0)
    location_corrections: int = Field(default=0)
    situation_corrections: int = Field(default=0)
    emotion_corrections: int = Field(default=0)
    top_learned_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    last_saved: str = Field(default="")
    file_size_kb: float = Field(default=0.0)
    timestamp: str = Field(default="")


# ══════════════════════════════════════════════════════════════
# AUTH MODELS
# ══════════════════════════════════════════════════════════════

class AuthResponse(BaseModel):
    """Authentication response"""
    
    status: str = Field(default="")
    user_id: str = Field(default="")
    email: str = Field(default="")
    token: str = Field(default="")
    expires: str = Field(default="")
    message: str = Field(default="")


class UserResponse(BaseModel):
    """User information response"""
    
    user_id: str = Field(default="")
    email: str = Field(default="")
    name: str = Field(default="")
    created_at: str = Field(default="")


# ══════════════════════════════════════════════════════════════
# LANGUAGE SUPPORT MODELS
# ══════════════════════════════════════════════════════════════

class SupportedLanguagesResponse(BaseModel):
    """Response with supported languages list"""
    
    total_languages: int = Field(default=99)
    languages: Dict[str, str] = Field(default_factory=dict)
    language_families: Optional[Dict[str, List[str]]] = None
    scripts_supported: Optional[List[str]] = None