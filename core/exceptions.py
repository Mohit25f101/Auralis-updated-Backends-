# ============================================
# app/core/exceptions.py - Custom Exceptions
# ============================================

from typing import Optional, Dict, Any


class AuralisException(Exception):
    """Base exception for Auralis Ultimate."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "AURALIS_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }


class AudioProcessingError(AuralisException):
    """Audio processing related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUDIO_PROCESSING_ERROR",
            details=details
        )


class ModelLoadError(AuralisException):
    """Model loading related errors."""
    
    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Failed to load model: {model_name}",
            error_code="MODEL_LOAD_ERROR",
            details={"model": model_name, **(details or {})}
        )


class TranscriptionError(AuralisException):
    """Transcription related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="TRANSCRIPTION_ERROR",
            details=details
        )


class ValidationError(AuralisException):
    """Input validation errors."""
    
    def __init__(self, field: str, message: str):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={"field": field}
        )


class AnalysisError(AuralisException):
    """Analysis related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="ANALYSIS_ERROR",
            details=details
        )