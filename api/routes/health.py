# ==============================
# 📄 api/routes/health.py
# ==============================
"""
Health Check and Status Endpoints
"""

from datetime import datetime
from typing import Dict, Any
import psutil
import sys

from fastapi import APIRouter
from api.models.responses import HealthResponse, SystemStatusResponse, ModelStatusResponse

router = APIRouter(tags=["Health"])


# Service status tracking
_service_status: Dict[str, bool] = {
    "whisper": False,
    "yamnet": False,
    "emotion_detector": False,
    "confidence_scorer": False,
    "context_synthesizer": False
}

_start_time: datetime = datetime.now()


def update_service_status(service: str, status: bool):
    """Update service status"""
    global _service_status
    _service_status[service] = status


def get_service_status() -> Dict[str, bool]:
    """Get all service statuses"""
    return _service_status.copy()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    🏥 **Health Check Endpoint**
    
    Returns current health status of the API and all services.
    """
    all_healthy = all(_service_status.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="5.0.0",
        models=_service_status,
        timestamp=datetime.now().isoformat(),
        uptime_seconds=int((datetime.now() - _start_time).total_seconds())
    )


@router.get("/status", response_model=SystemStatusResponse)
async def system_status():
    """
    📊 **System Status Endpoint**
    
    Returns detailed system information including:
    - Memory usage
    - CPU usage
    - Model loading status
    - Service health
    """
    # Get system metrics
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    return SystemStatusResponse(
        status="operational" if all(_service_status.values()) else "partial",
        version="5.0.0",
        python_version=sys.version,
        
        # System metrics
        memory_used_mb=round(memory.used / (1024 * 1024), 1),
        memory_total_mb=round(memory.total / (1024 * 1024), 1),
        memory_percent=memory.percent,
        cpu_percent=cpu_percent,
        
        # Services
        services=_service_status,
        
        # Metadata
        uptime_seconds=int((datetime.now() - _start_time).total_seconds()),
        timestamp=datetime.now().isoformat()
    )


@router.get("/models", response_model=ModelStatusResponse)
async def model_status():
    """
    🤖 **Model Status Endpoint**
    
    Returns detailed information about loaded ML models.
    """
    return ModelStatusResponse(
        models={
            "whisper": {
                "loaded": _service_status.get("whisper", False),
                "name": "openai/whisper-small",
                "type": "speech-to-text",
                "languages": ["en", "hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa"]
            },
            "yamnet": {
                "loaded": _service_status.get("yamnet", False),
                "name": "google/yamnet",
                "type": "sound-classification",
                "classes": 521
            },
            "emotion_detector": {
                "loaded": _service_status.get("emotion_detector", False),
                "name": "auralis/emotion-v5",
                "type": "emotion-classification",
                "emotions": ["neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted"]
            },
            "confidence_scorer": {
                "loaded": _service_status.get("confidence_scorer", False),
                "name": "auralis/confidence-v5",
                "type": "confidence-scoring",
                "scale": "1-10"
            },
            "context_synthesizer": {
                "loaded": _service_status.get("context_synthesizer", False),
                "name": "auralis/synthesis-v5",
                "type": "improvement-generation"
            }
        },
        timestamp=datetime.now().isoformat()
    )


@router.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"ping": "pong", "timestamp": datetime.now().isoformat()}


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe
    """
    critical_services = ["whisper", "yamnet"]
    critical_ready = all(_service_status.get(s, False) for s in critical_services)
    
    if critical_ready:
        return {"ready": True}
    else:
        return {"ready": False, "missing": [s for s in critical_services if not _service_status.get(s, False)]}


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe
    """
    return {"alive": True, "timestamp": datetime.now().isoformat()}