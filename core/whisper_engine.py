"""
============================================
AURALIS v5.0 - Whisper Engine
============================================
Speech recognition with hallucination detection.
"""

import re
from typing import Dict, Any, Optional, List

import numpy as np

from config import settings
from utils.logger import logger
from utils.constants import HALLUCINATION_PATTERNS
from utils.helpers import clean_text


class WhisperEngine:
    """
    Whisper-based speech recognition engine.
    
    Features:
    - Automatic language detection and English translation
    - Hallucination detection and filtering
    - Confidence scoring
    - Repetition detection
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize Whisper engine.
        
        Args:
            model_name: Whisper model name (default from settings)
        """
        self.model_name = model_name or settings.whisper_model
        self.pipe = None
        self.loaded = False
        
        # Configuration
        self.chunk_length = 30
        self.min_confidence = 0.5
    
    def load(self) -> bool:
        """
        Load Whisper model.
        
        Returns:
            True if loaded successfully
        """
        if self.loaded:
            return True
        
        try:
            from transformers import pipeline
            
            logger.info(f"Loading Whisper model: {self.model_name}")
            
            # Determine device
            device = -1  # CPU
            if settings.use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = 0
                        logger.info("Using GPU for Whisper")
                except:
                    pass
            
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                chunk_length_s=self.chunk_length,
                device=device,
            )
            
            self.loaded = True
            logger.info("Whisper loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            return False
    
    def transcribe(
        self,
        audio_path: str,
        audio_array: Optional[np.ndarray] = None,
        translate: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            audio_array: Optional audio array for additional analysis
            translate: Whether to translate to English
            
        Returns:
            Dictionary with transcription results
        """
        if not self.loaded:
            return self._error_result("Model not loaded")
        
        try:
            # Generate kwargs
            generate_kwargs = {
                "task": "translate" if translate else "transcribe",
                "language": "en" if translate else None,
            }
            
            # Run transcription
            result = self.pipe(
                audio_path,
                generate_kwargs={k: v for k, v in generate_kwargs.items() if v is not None}
            )
            
            raw_text = result.get("text", "").strip()
            
            # Process result
            return self._process_result(raw_text, audio_array)
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return self._error_result(str(e))
    
    def _process_result(
        self,
        raw_text: str,
        audio_array: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Process transcription result."""
        
        # Check for empty result
        if not raw_text or len(raw_text) < 3:
            return {
                "text": "",
                "raw_text": raw_text,
                "confidence": 0.2,
                "is_reliable": False,
                "issue": "empty"
            }
        
        # Check for hallucination
        if self._is_hallucination(raw_text):
            return {
                "text": "",
                "raw_text": raw_text,
                "confidence": 0.2,
                "is_reliable": False,
                "issue": "hallucination"
            }
        
        # Check for repetition
        if self._is_repetitive(raw_text):
            return {
                "text": "",
                "raw_text": raw_text,
                "confidence": 0.25,
                "is_reliable": False,
                "issue": "repetition"
            }
        
        # Check audio energy if provided
        if audio_array is not None:
            rms = np.sqrt(np.mean(audio_array ** 2))
            if rms < 0.005:
                return {
                    "text": "",
                    "raw_text": raw_text,
                    "confidence": 0.3,
                    "is_reliable": False,
                    "issue": "low_energy"
                }
        
        # Clean text
        clean = clean_text(raw_text)
        
        # Calculate confidence
        confidence = self._calculate_confidence(clean)
        
        return {
            "text": clean,
            "raw_text": raw_text,
            "confidence": confidence,
            "is_reliable": confidence >= self.min_confidence,
            "issue": None
        }
    
    def _is_hallucination(self, text: str) -> bool:
        """Check if text is a common hallucination."""
        text_lower = text.lower()
        
        for pattern in HALLUCINATION_PATTERNS:
            if pattern in text_lower:
                return True
        
        return False
    
    def _is_repetitive(self, text: str) -> bool:
        """Check if text has excessive repetition."""
        words = text.split()
        
        if len(words) <= 5:
            return False
        
        unique_words = set(words)
        ratio = len(unique_words) / len(words)
        
        return ratio < 0.35
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score based on text quality."""
        if not text:
            return 0.0
        
        words = text.split()
        word_count = len(words)
        
        # Base confidence
        if word_count >= 15:
            confidence = 0.92
        elif word_count >= 10:
            confidence = 0.88
        elif word_count >= 5:
            confidence = 0.82
        else:
            confidence = 0.75
        
        # Adjust for punctuation (indicates better recognition)
        if any(p in text for p in ".!?,"):
            confidence += 0.02
        
        # Adjust for capitalization
        if text[0].isupper():
            confidence += 0.01
        
        return min(confidence, 0.98)
    
    def _error_result(self, error: str) -> Dict[str, Any]:
        """Create error result."""
        return {
            "text": "",
            "raw_text": "",
            "confidence": 0.0,
            "is_reliable": False,
            "issue": "error",
            "error": error
        }
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.loaded