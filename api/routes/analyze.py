# ==============================
# 📄 api/routes/analyze.py
# ==============================
"""
Main Analysis Endpoint - Core Audio Processing
"""

from __future__ import annotations  # THIS MUST BE THE FIRST LINE!

import os
import time
import uuid
import tempfile
from datetime import datetime
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from services.analyzer import Analyzer

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from api.models.responses import (
    AnalysisResult,
    DetailedAnalysisResult,
    EmotionAnalysisResult,
    QuickAnalysisResult
)
from services.audio_loader import AudioLoader
from services.whisper_manager import WhisperManager
from services.yamnet_manager import YAMNetManager
from services.emotion_detector import EmotionDetector
from services.confidence_scorer import ConfidenceScorer
from services.context_synthesizer import ContextSynthesizer
from services.learning_system import LearningSystem
from config import settings
from core.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/analyze", tags=["Analysis"])

# Global instances (initialized in lifespan)
_loader: Optional[AudioLoader] = None
_whisper: Optional[WhisperManager] = None
_yamnet: Optional[YAMNetManager] = None
_language_detector = None
_ambient_analyzer = None
_location_detector = None
_emotion: Optional[EmotionDetector] = None
_confidence: Optional[ConfidenceScorer] = None
_synthesizer: Optional[ContextSynthesizer] = None
_analyzer: Optional[Analyzer] = None
_learning: Optional[LearningSystem] = None
_history: dict = {}


def init_services(
    loader: AudioLoader,
    whisper: WhisperManager,
    yamnet: YAMNetManager,
    language_detector,
    ambient_analyzer,
    location_detector,
    emotion: EmotionDetector,
    confidence: ConfidenceScorer,
    synthesizer: ContextSynthesizer,
    analyzer: Analyzer,
    learning: LearningSystem
):
    """Initialize all services from lifespan"""
    global _loader, _whisper, _yamnet, _language_detector, _ambient_analyzer
    global _location_detector, _emotion, _confidence, _synthesizer, _analyzer, _learning
    
    _loader = loader
    _whisper = whisper
    _yamnet = yamnet
    _language_detector = language_detector
    _ambient_analyzer = ambient_analyzer
    _location_detector = location_detector
    _emotion = emotion
    _confidence = confidence
    _synthesizer = synthesizer
    _analyzer = analyzer
    _learning = learning
    
    logger.info("Analyze route services initialized")


def get_loader() -> AudioLoader:
    if _loader is None:
        raise HTTPException(status_code=503, detail="Audio loader not initialized")
    return _loader


def get_whisper() -> WhisperManager:
    if _whisper is None:
        raise HTTPException(status_code=503, detail="Whisper not initialized")
    return _whisper


def get_yamnet() -> YAMNetManager:
    if _yamnet is None:
        raise HTTPException(status_code=503, detail="YAMNet not initialized")
    return _yamnet


def get_emotion() -> EmotionDetector:
    if _emotion is None:
        raise HTTPException(status_code=503, detail="Emotion detector not initialized")
    return _emotion


def get_confidence() -> ConfidenceScorer:
    if _confidence is None:
        raise HTTPException(status_code=503, detail="Confidence scorer not initialized")
    return _confidence


def get_synthesizer() -> ContextSynthesizer:
    if _synthesizer is None:
        raise HTTPException(status_code=503, detail="Context synthesizer not initialized")
    return _synthesizer


def get_analyzer() -> Analyzer:
    if _analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    return _analyzer


def get_learning() -> LearningSystem:
    if _learning is None:
        raise HTTPException(status_code=503, detail="Learning system not initialized")
    return _learning


@router.post("/", response_model=AnalysisResult)
async def analyze_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="Language code (e.g., 'en', 'hi')"),
    include_emotions: bool = Query(True, description="Include emotion analysis"),
    include_context: bool = Query(True, description="Include context synthesis")
):
    """
    Main audio analysis endpoint
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"[{request_id}] Starting analysis for: {file.filename}")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    supported = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm', '.aac'}
    if ext not in supported:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported format: {ext}. Supported: {supported}"
        )
    
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name
        
        loader = get_loader()
        audio, sr = loader.load(temp_path)
        duration = len(audio) / sr
        
        whisper = get_whisper()
        transcription = whisper.transcribe(audio, language=language)
        text = transcription.get("text", "").strip()
        detected_language = transcription.get("language", "unknown")
        
        yamnet = get_yamnet()
        sounds = yamnet.classify(audio)
        
        emotions = {}
        if include_emotions:
            emotion_detector = get_emotion()
            emotions = emotion_detector.analyze(audio, text, sounds)
        
        context = {}
        if include_context:
            synthesizer = get_synthesizer()
            context = synthesizer.synthesize(
                text=text,
                sounds=sounds,
                emotions=emotions,
                duration=duration
            )
        
        confidence_scorer = get_confidence()
        confidence = confidence_scorer.calculate(
            text=text,
            audio=audio,
            sounds=sounds
        )
        
        processing_time = time.time() - start_time
        
        result = AnalysisResult(
            request_id=request_id,
            success=True,
            transcription=text,
            language=detected_language,
            duration=round(duration, 2),
            sounds=sounds,
            emotions=emotions,
            context=context,
            confidence=confidence,
            processing_time=round(processing_time, 3),
            timestamp=datetime.now().isoformat()
        )
        
        _history[request_id] = {
            "result": result.model_dump(),
            "timestamp": datetime.now()
        }
        
        if len(_history) > 100:
            oldest = min(_history.keys(), key=lambda k: _history[k]["timestamp"])
            del _history[oldest]
        
        return result
        
    except Exception as e:
        logger.error(f"[{request_id}] Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@router.post("/quick", response_model=QuickAnalysisResult)
async def quick_analyze(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None)
):
    """Quick analysis - transcription only"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    temp_path = None
    try:
        ext = os.path.splitext(file.filename or ".wav")[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name
        
        loader = get_loader()
        audio, sr = loader.load(temp_path)
        
        whisper = get_whisper()
        result = whisper.transcribe(audio, language=language)
        
        return QuickAnalysisResult(
            request_id=request_id,
            success=True,
            transcription=result.get("text", "").strip(),
            language=result.get("language", "unknown"),
            duration=round(len(audio) / sr, 2),
            processing_time=round(time.time() - start_time, 3)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@router.post("/detailed", response_model=DetailedAnalysisResult)
async def detailed_analyze(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None)
):
    """Detailed analysis with full context synthesis"""
    analyzer = get_analyzer()
    
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    temp_path = None
    try:
        ext = os.path.splitext(file.filename or ".wav")[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name
        
        result = analyzer.analyze(temp_path)
        
        return DetailedAnalysisResult(
            request_id=request_id,
            success=True,
            **result,
            processing_time=round(time.time() - start_time, 3)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@router.post("/emotions", response_model=EmotionAnalysisResult)
async def analyze_emotions(
    file: UploadFile = File(...),
    text: Optional[str] = Query(None, description="Optional pre-transcribed text")
):
    """Emotion-focused analysis"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    temp_path = None
    try:
        ext = os.path.splitext(file.filename or ".wav")[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name
        
        loader = get_loader()
        audio, sr = loader.load(temp_path)
        
        if not text:
            whisper = get_whisper()
            result = whisper.transcribe(audio)
            text = result.get("text", "").strip()
        
        yamnet = get_yamnet()
        sounds = yamnet.classify(audio)
        
        emotion_detector = get_emotion()
        emotions = emotion_detector.analyze(audio, text, sounds)
        
        return EmotionAnalysisResult(
            request_id=request_id,
            success=True,
            text=text,
            emotions=emotions,
            processing_time=round(time.time() - start_time, 3)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@router.get("/history/{request_id}")
async def get_analysis_history(request_id: str):
    """Get previous analysis result by request ID"""
    if request_id not in _history:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return _history[request_id]["result"]


@router.get("/history")
async def list_analysis_history(limit: int = Query(10, ge=1, le=50)):
    """List recent analysis results"""
    sorted_history = sorted(
        _history.items(),
        key=lambda x: x[1]["timestamp"],
        reverse=True
    )[:limit]
    
    return [
        {
            "request_id": k,
            "timestamp": v["timestamp"].isoformat(),
            "transcription_preview": v["result"].get("transcription", "")[:100]
        }
        for k, v in sorted_history
    ]