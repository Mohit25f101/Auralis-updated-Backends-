# ==============================
# 📄 services/analyzer.py
# ==============================
"""
Main Analyzer Service - Orchestrates All Analysis Components
"""

from __future__ import annotations

import os
import time
import tempfile
from typing import Dict, Any, Optional, List
from datetime import datetime

import numpy as np

from core.logger import get_logger

logger = get_logger()


class Analyzer:
    """
    Main analyzer that orchestrates all analysis components
    """
    
    def __init__(
        self,
        audio_loader=None,
        whisper_manager=None,
        yamnet_manager=None,
        language_detector=None,
        ambient_analyzer=None,
        location_detector=None,
        emotion_detector=None,
        confidence_scorer=None,
        context_synthesizer=None,
        entity_extractor=None,
        semantic_analyzer=None
    ):
        """
        Initialize analyzer with all components
        
        Args:
            audio_loader: Audio loading service
            whisper_manager: Speech-to-text service
            yamnet_manager: Sound classification service
            language_detector: Language detection service
            ambient_analyzer: Ambient sound analysis service
            location_detector: Location detection service
            emotion_detector: Emotion analysis service
            confidence_scorer: Confidence scoring service
            context_synthesizer: Context synthesis service
            entity_extractor: Entity extraction service
            semantic_analyzer: Semantic analysis service
        """
        self.audio_loader = audio_loader
        self.whisper = whisper_manager
        self.yamnet = yamnet_manager
        self.language_detector = language_detector
        self.ambient_analyzer = ambient_analyzer
        self.location_detector = location_detector
        self.emotion_detector = emotion_detector
        self.confidence_scorer = confidence_scorer
        self.context_synthesizer = context_synthesizer
        self.entity_extractor = entity_extractor
        self.semantic_analyzer = semantic_analyzer
        
        self.loaded = False
        self._analysis_count = 0
        
    def load(self):
        """Load all components"""
        if self.loaded:
            return
            
        logger.info("Loading Analyzer components...")
        
        # Load components that have load methods
        components = [
            ('audio_loader', self.audio_loader),
            ('whisper', self.whisper),
            ('yamnet', self.yamnet),
            ('emotion_detector', self.emotion_detector),
            ('confidence_scorer', self.confidence_scorer),
        ]
        
        for name, component in components:
            if component and hasattr(component, 'load'):
                try:
                    component.load()
                    logger.debug(f"Loaded {name}")
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")
        
        self.loaded = True
        logger.info("Analyzer loaded successfully")
    
    def analyze(
        self,
        audio_path: Optional[str] = None,
        audio_data: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        include_emotions: bool = True,
        include_entities: bool = True,
        include_semantics: bool = True,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive audio analysis
        
        Args:
            audio_path: Path to audio file
            audio_data: Raw audio data (alternative to path)
            sample_rate: Sample rate of audio data
            language: Language hint for transcription
            include_emotions: Whether to include emotion analysis
            include_entities: Whether to extract entities
            include_semantics: Whether to include semantic analysis
            include_context: Whether to synthesize context
            
        Returns:
            Comprehensive analysis results
        """
        if not self.loaded:
            self.load()
        
        start_time = time.time()
        self._analysis_count += 1
        
        result = {
            "analysis_id": f"analysis_{self._analysis_count}_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "transcription": "",
            "language": "unknown",
            "duration": 0.0,
            "sounds": {},
            "emotions": {},
            "entities": {},
            "semantics": {},
            "context": {},
            "confidence": {},
            "processing_time": 0.0
        }
        
        try:
            # Step 1: Load audio
            if audio_path:
                if self.audio_loader:
                    audio, sr = self.audio_loader.load(audio_path)
                else:
                    raise ValueError("Audio loader not configured")
            elif audio_data is not None:
                audio = audio_data
                sr = sample_rate
            else:
                raise ValueError("Either audio_path or audio_data must be provided")
            
            result["duration"] = len(audio) / sr
            
            # Step 2: Transcribe with Whisper
            if self.whisper:
                transcription = self.whisper.transcribe(audio, language=language)
                result["transcription"] = transcription.get("text", "").strip()
                result["language"] = transcription.get("language", "unknown")
            
            # Step 3: Classify sounds with YAMNet
            if self.yamnet:
                result["sounds"] = self.yamnet.classify(audio)
            
            # Step 4: Detect language (if detector available)
            if self.language_detector and result["transcription"]:
                try:
                    lang_result = self.language_detector.detect(result["transcription"])
                    if lang_result.get("language"):
                        result["language"] = lang_result["language"]
                except Exception as e:
                    logger.debug(f"Language detection failed: {e}")
            
            # Step 5: Analyze ambient sounds
            if self.ambient_analyzer:
                try:
                    ambient = self.ambient_analyzer.analyze(result["sounds"])
                    result["ambient_analysis"] = ambient
                except Exception as e:
                    logger.debug(f"Ambient analysis failed: {e}")
            
            # Step 6: Detect location
            if self.location_detector:
                try:
                    location = self.location_detector.detect(
                        result["transcription"],
                        result["sounds"]
                    )
                    result["location"] = location
                except Exception as e:
                    logger.debug(f"Location detection failed: {e}")
            
            # Step 7: Emotion analysis
            if include_emotions and self.emotion_detector:
                try:
                    result["emotions"] = self.emotion_detector.analyze(
                        audio,
                        result["transcription"],
                        result["sounds"]
                    )
                except Exception as e:
                    logger.debug(f"Emotion analysis failed: {e}")
            
            # Step 8: Entity extraction
            if include_entities and self.entity_extractor:
                try:
                    result["entities"] = self.entity_extractor.extract(
                        result["transcription"]
                    )
                except Exception as e:
                    logger.debug(f"Entity extraction failed: {e}")
            
            # Step 9: Semantic analysis
            if include_semantics and self.semantic_analyzer:
                try:
                    result["semantics"] = self.semantic_analyzer.analyze(
                        result["transcription"]
                    )
                except Exception as e:
                    logger.debug(f"Semantic analysis failed: {e}")
            
            # Step 10: Context synthesis
            if include_context and self.context_synthesizer:
                try:
                    result["context"] = self.context_synthesizer.synthesize(
                        text=result["transcription"],
                        sounds=result["sounds"],
                        emotions=result.get("emotions", {}),
                        duration=result["duration"]
                    )
                except Exception as e:
                    logger.debug(f"Context synthesis failed: {e}")
            
            # Step 11: Calculate confidence
            if self.confidence_scorer:
                try:
                    result["confidence"] = self.confidence_scorer.calculate(
                        text=result["transcription"],
                        audio=audio,
                        sounds=result["sounds"]
                    )
                except Exception as e:
                    logger.debug(f"Confidence scoring failed: {e}")
            
            result["success"] = True
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            result["error"] = str(e)
        
        result["processing_time"] = round(time.time() - start_time, 3)
        
        return result
    
    def analyze_text_only(self, text: str) -> Dict[str, Any]:
        """
        Analyze text without audio
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results
        """
        result = {
            "transcription": text,
            "entities": {},
            "semantics": {},
            "emotions": {}
        }
        
        if self.entity_extractor:
            result["entities"] = self.entity_extractor.extract(text)
        
        if self.semantic_analyzer:
            result["semantics"] = self.semantic_analyzer.analyze(text)
        
        # Text-only emotion analysis
        if self.emotion_detector:
            # Create dummy audio for emotion detector
            dummy_audio = np.zeros(16000, dtype=np.float32)
            result["emotions"] = self.emotion_detector.analyze(
                dummy_audio, text, {}
            )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            "loaded": self.loaded,
            "analysis_count": self._analysis_count,
            "components": {
                "audio_loader": self.audio_loader is not None,
                "whisper": self.whisper is not None,
                "yamnet": self.yamnet is not None,
                "emotion_detector": self.emotion_detector is not None,
                "entity_extractor": self.entity_extractor is not None,
                "semantic_analyzer": self.semantic_analyzer is not None,
                "context_synthesizer": self.context_synthesizer is not None
            }
        }