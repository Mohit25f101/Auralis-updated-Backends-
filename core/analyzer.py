# ==============================
# 📄 core/analyzer.py
# ==============================
# Main Analysis Orchestrator
# ==============================

import os
import re
import tempfile
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import numpy as np

from config import (
    LABELS, LOCATION_KEYWORDS, SITUATION_KEYWORDS,
    AUDIO, get_device
)


class Analyzer:
    """
    Main analysis orchestrator that combines:
    - Audio processing
    - Whisper transcription
    - YAMNet sound classification
    - Neural classification (when available)
    - Rule-based fallback
    - Learning system integration
    """
    
    def __init__(
        self,
        audio_processor=None,
        whisper_engine=None,
        yamnet_engine=None,
        neural_classifier=None,
        learning_system=None
    ):
        self.audio_processor = audio_processor
        self.whisper = whisper_engine
        self.yamnet = yamnet_engine
        self.neural = neural_classifier
        self.learning = learning_system
        
        # Analysis history for feedback
        self.history: Dict[str, Dict] = {}
        self.max_history = 1000
    
    def analyze(
        self,
        audio_path: str = None,
        audio_array: np.ndarray = None,
        use_neural: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze audio and return comprehensive results.
        
        Args:
            audio_path: Path to audio file
            audio_array: Pre-loaded audio array
            use_neural: Whether to use neural classifier
        
        Returns:
            Complete analysis result
        """
        import time
        start_time = time.time()
        request_id = self._generate_id()
        
        result = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "status": "processing",
        }
        
        try:
            # Step 1: Load audio
            if audio_array is None and audio_path:
                audio, duration = self.audio_processor.load(audio_path)
            elif audio_array is not None:
                audio = audio_array
                duration = len(audio) / AUDIO.sample_rate
            else:
                raise ValueError("No audio provided")
            
            # Truncate if too long
            max_samples = int(AUDIO.max_duration * AUDIO.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                duration = AUDIO.max_duration
            
            result["audio_duration"] = round(duration, 2)
            
            # Step 2: Get audio features
            audio_features = self.audio_processor.get_features(audio)
            
            # Step 3: Transcribe
            temp_wav = tempfile.mktemp(suffix=".wav")
            try:
                self.audio_processor.save_wav(audio, temp_wav)
                transcription = self.whisper.transcribe(temp_wav, audio)
            finally:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
            
            result["transcription"] = transcription
            result["transcribed_text"] = transcription.get("text", "")
            result["text_reliable"] = transcription.get("is_reliable", False)
            
            # Step 4: Analyze sounds
            sound_analysis = self.yamnet.analyze(audio)
            result["sounds"] = sound_analysis
            result["detected_sounds"] = list(sound_analysis.get("sounds", {}).keys())[:10]
            
            # Step 5: Scene classification
            if use_neural and self.neural and self.neural.is_loaded:
                # Use neural classifier
                scene_result = self._neural_classification(
                    transcription, sound_analysis, audio_features
                )
                result["analysis_mode"] = "neural"
            else:
                # Use rule-based
                scene_result = self._rule_based_classification(
                    transcription, sound_analysis, audio_features
                )
                result["analysis_mode"] = "rule_based"
            
            # Merge scene result
            result.update(scene_result)
            
            # Step 6: Build evidence and summary
            result["evidence"] = self._build_evidence(
                transcription, sound_analysis, scene_result
            )
            result["summary"] = self._build_summary(scene_result, transcription)
            
            # Calculate processing time
            result["processing_time_ms"] = round((time.time() - start_time) * 1000, 1)
            result["status"] = "success"
            
            # Store in history
            self._store_history(request_id, result, transcription, sound_analysis)
            
            return result
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["processing_time_ms"] = round((time.time() - start_time) * 1000, 1)
            return result
    
    def _neural_classification(
        self,
        transcription: Dict,
        sound_analysis: Dict,
        audio_features: Dict
    ) -> Dict[str, Any]:
        """Use neural classifier for prediction."""
        # Prepare features (simplified - in production, use proper embeddings)
        text = transcription.get("text", "")
        
        # Text features (placeholder - use actual embeddings)
        text_features = np.zeros(768, dtype=np.float32)
        
        # Audio features
        audio_feat = np.array([
            audio_features.get("rms_energy", 0),
            audio_features.get("zero_crossing_rate", 0),
            audio_features.get("spectral_centroid", 0),
        ], dtype=np.float32)
        audio_feat = np.pad(audio_feat, (0, 80 - len(audio_feat)))
        
        # Sound features (use YAMNet embeddings if available)
        embeddings = sound_analysis.get("embeddings")
        if embeddings is not None and len(embeddings) > 0:
            sound_features = np.mean(embeddings, axis=0)
        else:
            sound_features = np.zeros(1024, dtype=np.float32)
        
        # Get neural prediction
        neural_result = self.neural.predict(text_features, audio_feat, sound_features)
        
        # Combine with rule-based for better accuracy
        rule_result = self._rule_based_classification(
            transcription, sound_analysis, audio_features
        )
        
        # Weighted combination
        neural_weight = 0.6
        rule_weight = 0.4
        
        # If neural confidence is low, rely more on rules
        if neural_result.get("model_confidence", 0) < 0.5:
            neural_weight = 0.3
            rule_weight = 0.7
        
        # Combine confidences
        location_conf = (
            neural_result.get("location_confidence", 0) * neural_weight +
            rule_result.get("location_confidence", 0) * rule_weight
        )
        situation_conf = (
            neural_result.get("situation_confidence", 0) * neural_weight +
            rule_result.get("situation_confidence", 0) * rule_weight
        )
        
        # Choose best location/situation
        if neural_result.get("location_confidence", 0) > rule_result.get("location_confidence", 0):
            location = neural_result["location"]
        else:
            location = rule_result["location"]
        
        if neural_result.get("situation_confidence", 0) > rule_result.get("situation_confidence", 0):
            situation = neural_result["situation"]
        else:
            situation = rule_result["situation"]
        
        return {
            "location": location,
            "location_confidence": round(location_conf, 3),
            "situation": situation,
            "situation_confidence": round(situation_conf, 3),
            "is_emergency": neural_result.get("is_emergency", False) or rule_result.get("is_emergency", False),
            "emergency_probability": round(max(
                neural_result.get("emergency_probability", 0),
                rule_result.get("emergency_probability", 0)
            ), 3),
            "overall_confidence": round((location_conf + situation_conf) / 2, 3),
        }
    
    def _rule_based_classification(
        self,
        transcription: Dict,
        sound_analysis: Dict,
        audio_features: Dict
    ) -> Dict[str, Any]:
        """Rule-based classification using keywords and sound hints."""
        text = transcription.get("text", "").lower()
        raw_text = transcription.get("raw_text", text).lower()
        is_reliable = transcription.get("is_reliable", False)
        
        combined_text = f"{text} {raw_text}"
        
        location_hints = sound_analysis.get("location_hints", [])
        situation_hints = sound_analysis.get("situation_hints", [])
        detected_sounds = list(sound_analysis.get("sounds", {}).keys())
        
        # Weight based on text reliability
        text_weight = 0.7 if is_reliable and text else 0.25
        sound_weight = 1.0 - text_weight
        
        # Detect location
        location, loc_conf = self._detect_location(
            combined_text, location_hints, text_weight, sound_weight, detected_sounds
        )
        
        # Detect situation
        situation, sit_conf, is_emergency = self._detect_situation(
            combined_text, situation_hints, text_weight, sound_weight, detected_sounds
        )
        
        # Emergency probability
        emg_prob = 0.98 if is_emergency else self._calc_emergency_prob(
            combined_text, situation_hints
        )
        
        return {
            "location": location,
            "location_confidence": round(loc_conf, 3),
            "situation": situation,
            "situation_confidence": round(sit_conf, 3),
            "is_emergency": is_emergency,
            "emergency_probability": round(emg_prob, 3),
            "overall_confidence": round((loc_conf + sit_conf) / 2, 3),
        }
    
    def _detect_location(
        self,
        text: str,
        hints: List[Tuple[str, float]],
        text_weight: float,
        sound_weight: float,
        sounds: List[str]
    ) -> Tuple[str, float]:
        """Detect location using keywords and sound hints."""
        scores = {}
        
        for location, keywords in LOCATION_KEYWORDS.items():
            score = 0.0
            
            # Keyword scoring
            kw_score = sum(w for k, w in keywords.items() if k in text)
            if kw_score > 0:
                score += min(kw_score / 10, 1.5) * text_weight
            
            # Sound hints
            for hint_loc, hint_score in hints:
                if hint_loc == location:
                    score += hint_score * sound_weight * 2
            
            # Learning boost
            if self.learning:
                boost = self.learning.get_boost('location', location, text, sounds)
                score += boost
            
            if score > 0.1:
                scores[location] = score
        
        if not scores:
            if hints:
                best = hints[0]
                return best[0], 0.55 + best[1] * 0.3
            return "Unknown", 0.40
        
        best = max(scores, key=scores.get)
        conf = 0.70 + min(scores[best] / 2, 0.28)
        return best, conf
    
    def _detect_situation(
        self,
        text: str,
        hints: List[Tuple[str, float]],
        text_weight: float,
        sound_weight: float,
        sounds: List[str]
    ) -> Tuple[str, float, bool]:
        """Detect situation using keywords and sound hints."""
        candidates = []
        
        for situation, keywords in SITUATION_KEYWORDS.items():
            score = 0.0
            
            # Keyword scoring
            kw_score = sum(w for k, w in keywords.items() if k in text)
            if kw_score > 0:
                score += min(kw_score / 8, 1.5) * text_weight
            
            # Sound hints
            for hint_sit, hint_score in hints:
                if hint_sit == situation:
                    score += hint_score * sound_weight * 2
            
            # Learning boost
            if self.learning:
                boost = self.learning.get_boost('situation', situation, text, sounds)
                score += boost
            
            is_emergency = situation in ["Emergency", "Medical Emergency", "Accident", "Security Alert"]
            priority = 10 if is_emergency else (7 if situation == "Boarding/Departure" else 5)
            
            if score > 0.1:
                candidates.append({
                    "situation": situation,
                    "score": score,
                    "priority": priority,
                    "is_emergency": is_emergency
                })
        
        if not candidates:
            if hints:
                best = hints[0]
                return best[0], 0.60, False
            return "Normal/Quiet", 0.55, False
        
        candidates.sort(key=lambda x: (x["priority"], x["score"]), reverse=True)
        best = candidates[0]
        conf = 0.68 + min(best["score"] / 1.5, 0.28)
        return best["situation"], conf, best["is_emergency"]
    
    def _calc_emergency_prob(self, text: str, hints: List[Tuple[str, float]]) -> float:
        """Calculate emergency probability."""
        prob = 0.03
        
        emergency_words = {
            "help": 0.15, "emergency": 0.2, "fire": 0.2,
            "accident": 0.18, "danger": 0.15, "ambulance": 0.18
        }
        
        for word, boost in emergency_words.items():
            if word in text:
                prob += boost
        
        for sit, score in hints:
            if sit in ["Emergency", "Medical Emergency"]:
                prob += score * 0.4
        
        return min(prob, 0.95)
    
    def _build_evidence(
        self,
        transcription: Dict,
        sound_analysis: Dict,
        scene_result: Dict
    ) -> List[str]:
        """Build evidence list for the analysis."""
        evidence = []
        
        text = transcription.get("text", "")
        is_reliable = transcription.get("is_reliable", False)
        
        if is_reliable and text:
            preview = text[:80] + "..." if len(text) > 80 else text
            evidence.append(f'🗣️ Speech: "{preview}"')
        else:
            evidence.append("🗣️ Speech unclear - using sound analysis")
        
        sounds = sound_analysis.get("sounds", {})
        if sounds:
            sound_list = list(sounds.keys())[:5]
            evidence.append(f"🔊 Sounds: {', '.join(sound_list)}")
        
        location_hints = sound_analysis.get("location_hints", [])
        if location_hints:
            hints_str = ", ".join([f"{h[0]} ({h[1]*100:.0f}%)" for h in location_hints[:2]])
            evidence.append(f"💡 Sound hints: {hints_str}")
        
        evidence.append(f"📍 Location: {scene_result['location']} ({scene_result['location_confidence']*100:.0f}%)")
        evidence.append(f"🎯 Situation: {scene_result['situation']} ({scene_result['situation_confidence']*100:.0f}%)")
        
        if scene_result.get("is_emergency"):
            evidence.append(f"🚨 EMERGENCY: {scene_result['emergency_probability']*100:.0f}%")
        
        return evidence
    
    def _build_summary(self, scene_result: Dict, transcription: Dict) -> str:
        """Build summary string."""
        parts = [
            f"📍 **{scene_result['location']}** ({scene_result['location_confidence']*100:.0f}%)",
            f"🎯 **{scene_result['situation']}** ({scene_result['situation_confidence']*100:.0f}%)"
        ]
        
        if scene_result.get("is_emergency"):
            parts.append(f"🚨 **EMERGENCY** ({scene_result['emergency_probability']*100:.0f}%)")
        
        text = transcription.get("text", "")
        if text and len(text) > 10:
            snippet = text[:50] + "..." if len(text) > 50 else text
            parts.append(f'💬 "{snippet}"')
        
        return " | ".join(parts)
    
    def _generate_id(self) -> str:
        """Generate unique request ID."""
        import uuid
        return uuid.uuid4().hex[:8]
    
    def _store_history(
        self,
        request_id: str,
        result: Dict,
        transcription: Dict,
        sound_analysis: Dict
    ):
        """Store analysis in history for feedback."""
        self.history[request_id] = {
            "result": result,
            "transcription": transcription,
            "sounds": sound_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
        # Trim history
        if len(self.history) > self.max_history:
            oldest = list(self.history.keys())[0]
            del self.history[oldest]
    
    def get_history(self, request_id: str) -> Optional[Dict]:
        """Get analysis from history."""
        return self.history.get(request_id)