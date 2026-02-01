# ==============================
# 📄 services/emotion_detector.py
# ==============================
"""
Emotion Detection Service - Audio & Text Emotion Analysis
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from core.logger import get_logger

logger = get_logger()


class EmotionDetector:
    """
    Multi-modal emotion detection from audio and text
    """
    
    def __init__(self):
        self.loaded = False
        
        # Text-based emotion keywords (expanded)
        self.emotion_keywords = {
            "joy": [
                "happy", "glad", "excited", "wonderful", "great", "amazing",
                "fantastic", "excellent", "delighted", "pleased", "thrilled",
                "overjoyed", "ecstatic", "cheerful", "joyful", "elated"
            ],
            "sadness": [
                "sad", "sorry", "unfortunately", "regret", "disappoint",
                "unhappy", "depressed", "upset", "miserable", "heartbroken",
                "grief", "sorrow", "melancholy", "gloomy", "down"
            ],
            "anger": [
                "angry", "furious", "annoyed", "frustrated", "irritated",
                "outraged", "mad", "livid", "enraged", "hostile",
                "resentful", "bitter", "indignant", "irate"
            ],
            "fear": [
                "afraid", "scared", "worried", "anxious", "nervous",
                "terrified", "frightened", "panicked", "alarmed", "concerned",
                "uneasy", "apprehensive", "dread", "horror"
            ],
            "surprise": [
                "surprised", "amazed", "astonished", "shocked", "unexpected",
                "startled", "stunned", "bewildered", "astounded", "wow"
            ],
            "neutral": [
                "attention", "please", "note", "information", "announce",
                "inform", "notice", "update", "regarding"
            ],
            "urgency": [
                "urgent", "immediately", "emergency", "now", "hurry",
                "critical", "asap", "right away", "at once", "quickly",
                "fast", "rush", "priority", "important"
            ],
            "calm": [
                "calm", "peaceful", "relaxed", "gentle", "quiet",
                "serene", "tranquil", "soothing", "steady", "patient"
            ]
        }
        
        # Micro-emotion patterns
        self.micro_emotion_patterns = {
            "hesitation": [r"\b(um+|uh+|er+|hmm+)\b", r"\.{3,}", r"\.\.\.", r"well\s*,"],
            "emphasis": [r"[A-Z]{3,}", r"!{2,}", r"\*\*[^*]+\*\*"],
            "uncertainty": [r"\b(maybe|perhaps|might|possibly|probably)\b"],
            "politeness": [r"\b(please|thank|kindly|appreciate|grateful)\b"],
            "formality": [r"\b(sir|madam|dear|respected|honorable)\b"]
        }
        
        # Sound-emotion mappings
        self.sound_emotions = {
            "Music": {"calm": 0.5, "joy": 0.3},
            "Alarm": {"urgency": 0.8, "fear": 0.4},
            "Siren": {"urgency": 0.9, "fear": 0.5},
            "Crying": {"sadness": 0.8},
            "Laughter": {"joy": 0.9},
            "Applause": {"joy": 0.7, "surprise": 0.3},
            "Crowd": {"neutral": 0.5},
            "Speech": {"neutral": 0.3},
            "Silence": {"calm": 0.4},
            "Yelling": {"anger": 0.6, "urgency": 0.5},
            "Screaming": {"fear": 0.7, "urgency": 0.8}
        }
        
        # Asian context patterns
        self.asian_patterns = {
            "respect_markers": ["ji", "sahab", "sir", "madam", "respected"],
            "group_harmony": ["we all", "everyone", "together", "collective"],
            "indirect_speech": ["it seems", "perhaps", "it appears", "one might"]
        }
    
    def load(self):
        """Initialize the emotion detector"""
        self.loaded = True
        logger.info("Emotion Detector loaded")
    
    def analyze(
        self, 
        audio: np.ndarray, 
        text: str, 
        sounds: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Comprehensive emotion analysis
        
        Args:
            audio: Audio waveform
            text: Transcribed text
            sounds: Detected ambient sounds
            
        Returns:
            Emotion analysis results
        """
        if not self.loaded:
            self.load()
        
        text_lower = text.lower()
        
        # Analyze from multiple sources
        text_emotions = self._analyze_text(text_lower)
        audio_emotions = self._analyze_audio(audio)
        sound_emotions = self._analyze_sounds(sounds)
        micro_emotions = self._detect_micro_emotions(text)
        
        # Combine emotions with weights
        combined = self._combine_emotions(
            text_emotions, audio_emotions, sound_emotions
        )
        
        # Determine primary emotion
        primary = max(combined.items(), key=lambda x: x[1]) if combined else ("neutral", 0.5)
        
        # Asian context detection
        asian_context = self._detect_asian_context(text)
        
        # Context emotions (situational)
        context_emotions = self._get_context_emotions(text, sounds)
        
        # Calculate overall valence and arousal
        valence, arousal = self._calculate_valence_arousal(combined)
        
        # Get emotion intensity
        intensity = self._calculate_intensity(audio, text)
        
        return {
            "primary_emotion": primary[0],
            "primary_score": round(primary[1], 3),
            "all_emotions": [
                {"emotion": k, "score": round(v, 3)} 
                for k, v in sorted(combined.items(), key=lambda x: -x[1])
                if v > 0.1
            ],
            "micro_emotions": micro_emotions,  # ADDED THIS!
            "context_emotions": context_emotions,
            "valence": round(valence, 3),  # Positive/negative (-1 to 1)
            "arousal": round(arousal, 3),  # Activation level (0 to 1)
            "intensity": round(intensity, 3),
            "asian_context_detected": asian_context,
            "audio_features_used": {
                "energy_variance": float(np.var(np.abs(audio[:min(len(audio), 16000)]))),
                "pitch_variance": float(np.var(audio[:min(len(audio), 16000)])),
                "speech_rate": self._estimate_speech_rate(text, len(audio) / 16000)
            }
        }
    
    def _analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze emotions from text"""
        emotions = {}
        words = text.split()
        
        for emotion, keywords in self.emotion_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text:
                    score += 0.2
                    # Boost if word is capitalized (emphasis)
                    if keyword.upper() in text.upper() and keyword.upper() != keyword:
                        score += 0.1
            emotions[emotion] = min(score, 1.0)
        
        return emotions
    
    def _analyze_audio(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze emotions from audio features"""
        emotions = {}
        
        # Simple audio feature extraction
        # Energy analysis
        energy = np.mean(np.abs(audio))
        energy_var = np.var(np.abs(audio))
        
        # High energy + high variance = excitement/urgency
        if energy > 0.1 and energy_var > 0.01:
            emotions["urgency"] = min(energy * 3, 0.8)
            emotions["anger"] = min(energy * 2, 0.6)
        
        # Low energy = calm/sad
        if energy < 0.05:
            emotions["calm"] = 0.5
            emotions["sadness"] = 0.3
        
        # Medium energy = neutral
        if 0.05 <= energy <= 0.1:
            emotions["neutral"] = 0.5
        
        return emotions
    
    def _analyze_sounds(self, sounds: Dict[str, float]) -> Dict[str, float]:
        """Derive emotions from ambient sounds"""
        emotions = {}
        
        for sound, confidence in sounds.items():
            if sound in self.sound_emotions:
                for emotion, base_score in self.sound_emotions[sound].items():
                    score = base_score * confidence
                    emotions[emotion] = max(emotions.get(emotion, 0), score)
        
        return emotions
    
    def _detect_micro_emotions(self, text: str) -> List[Dict[str, Any]]:
        """Detect subtle emotional cues (micro-emotions)"""
        micro = []
        
        for emotion_type, patterns in self.micro_emotion_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    micro.append({
                        "type": emotion_type,
                        "count": len(matches),
                        "examples": matches[:3]  # First 3 examples
                    })
                    break  # One match per type is enough
        
        # Detect urgency indicators
        urgent_words = ["urgent", "immediately", "emergency", "now", "hurry", "asap"]
        urgent_count = sum(1 for w in urgent_words if w in text.lower())
        if urgent_count > 0:
            micro.append({
                "type": "urgency",
                "count": urgent_count,
                "examples": [w for w in urgent_words if w in text.lower()]
            })
        
        return micro
    
    def _combine_emotions(
        self,
        text_emotions: Dict[str, float],
        audio_emotions: Dict[str, float],
        sound_emotions: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine emotions from all sources with weights"""
        weights = {
            "text": 0.5,
            "audio": 0.3,
            "sounds": 0.2
        }
        
        all_emotions = set(text_emotions.keys()) | set(audio_emotions.keys()) | set(sound_emotions.keys())
        
        combined = {}
        for emotion in all_emotions:
            score = (
                text_emotions.get(emotion, 0) * weights["text"] +
                audio_emotions.get(emotion, 0) * weights["audio"] +
                sound_emotions.get(emotion, 0) * weights["sounds"]
            )
            if score > 0:
                combined[emotion] = score
        
        return combined
    
    def _detect_asian_context(self, text: str) -> bool:
        """Detect Asian cultural context markers"""
        text_lower = text.lower()
        
        for category, patterns in self.asian_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return True
        
        return False
    
    def _get_context_emotions(
        self, 
        text: str, 
        sounds: Dict[str, float]
    ) -> List[str]:
        """Get situational/contextual emotions"""
        context = []
        text_lower = text.lower()
        
        # Travel anxiety
        if any(w in text_lower for w in ["delay", "late", "cancel", "missed"]):
            context.append("anxiety")
        
        # Safety concern
        if any(w in text_lower for w in ["emergency", "danger", "warning", "caution"]):
            context.append("concern")
        
        # Anticipation
        if any(w in text_lower for w in ["arriving", "approaching", "soon", "shortly"]):
            context.append("anticipation")
        
        # Relief
        if any(w in text_lower for w in ["resumed", "restored", "resolved", "cleared"]):
            context.append("relief")
        
        # Frustration from sounds
        if sounds.get("Crowd", 0) > 0.5 and sounds.get("Noise", 0) > 0.3:
            context.append("overwhelm")
        
        return context
    
    def _calculate_valence_arousal(
        self, 
        emotions: Dict[str, float]
    ) -> Tuple[float, float]:
        """Calculate valence (positive/negative) and arousal (activation)"""
        valence_map = {
            "joy": 1.0, "calm": 0.5, "surprise": 0.3,
            "neutral": 0.0,
            "sadness": -0.7, "fear": -0.8, "anger": -0.6
        }
        
        arousal_map = {
            "joy": 0.7, "surprise": 0.8, "anger": 0.9, "fear": 0.8, "urgency": 0.9,
            "calm": 0.2, "sadness": 0.3, "neutral": 0.4
        }
        
        total_weight = sum(emotions.values()) or 1
        
        valence = sum(
            valence_map.get(e, 0) * s 
            for e, s in emotions.items()
        ) / total_weight
        
        arousal = sum(
            arousal_map.get(e, 0.5) * s 
            for e, s in emotions.items()
        ) / total_weight
        
        return max(-1, min(1, valence)), max(0, min(1, arousal))
    
    def _calculate_intensity(self, audio: np.ndarray, text: str) -> float:
        """Calculate overall emotional intensity"""
        # Audio intensity
        audio_intensity = min(np.mean(np.abs(audio)) * 10, 1.0)
        
        # Text intensity (exclamation, caps)
        exclamation_count = text.count("!")
        caps_ratio = sum(1 for c in text if c.isupper()) / (len(text) or 1)
        
        text_intensity = min(exclamation_count * 0.1 + caps_ratio, 1.0)
        
        return (audio_intensity * 0.6 + text_intensity * 0.4)
    
    def _estimate_speech_rate(self, text: str, duration: float) -> float:
        """Estimate speech rate (words per second)"""
        if duration <= 0:
            return 0.0
        
        words = len(text.split())
        return words / duration
    
    def get_emotion_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable emotion summary"""
        primary = analysis.get("primary_emotion", "neutral")
        score = analysis.get("primary_score", 0.5)
        
        intensity_words = {
            (0, 0.3): "slightly",
            (0.3, 0.6): "moderately",
            (0.6, 0.8): "quite",
            (0.8, 1.0): "very"
        }
        
        intensity = "moderately"
        for (low, high), word in intensity_words.items():
            if low <= score < high:
                intensity = word
                break
        
        return f"The speaker appears {intensity} {primary}"