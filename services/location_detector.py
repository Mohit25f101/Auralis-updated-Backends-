# ==============================
# 📄 services/location_detector.py
# ==============================
"""
Smart Location Detection Service
Distinguishes between "talking about" vs "actually being at" a location
Uses ambient sounds, acoustic features, and contextual analysis
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import re


class LocationDetector:
    """
    Smart Location Detection Service
    
    PREVENTS FALSE POSITIVES by:
    1. Prioritizing ambient/background sounds over speech content
    2. Analyzing acoustic properties (reverb, noise floor, etc.)
    3. Detecting "talking about" vs "being at" patterns
    4. Requiring multiple evidence sources for high confidence
    5. Using temporal consistency (sounds throughout audio, not just mentions)
    """
    
    # Ambient sound signatures for each location (PRIMARY EVIDENCE)
    LOCATION_AMBIENT_SOUNDS = {
        "Airport Terminal": {
            "strong": ["aircraft", "airplane", "jet engine", "aircraft engine", "helicopter"],
            "medium": ["public address system", "crowd", "luggage wheel", "escalator"],
            "weak": ["typing", "cash register"],
            "acoustic_profile": {"reverb": "high", "noise_floor": "medium", "crowd_murmur": True}
        },
        "Railway Station": {
            "strong": ["train", "train horn", "train whistle", "rail transport", "railroad car"],
            "medium": ["public address system", "crowd", "brake squeal"],
            "weak": ["door", "footsteps"],
            "acoustic_profile": {"reverb": "high", "noise_floor": "high", "metallic_sounds": True}
        },
        "Hospital": {
            "strong": ["heart monitor", "medical equipment", "ambulance siren"],
            "medium": ["pager", "intercom", "ventilator"],
            "weak": ["door", "footsteps", "quiet speech"],
            "acoustic_profile": {"reverb": "low", "noise_floor": "low", "clinical_quiet": True}
        },
        "Street/Road": {
            "strong": ["traffic", "car horn", "engine", "motorcycle", "truck"],
            "medium": ["siren", "bus", "bicycle", "footsteps on pavement"],
            "weak": ["bird", "wind"],
            "acoustic_profile": {"reverb": "none", "noise_floor": "variable", "outdoor": True}
        },
        "Shopping Mall": {
            "strong": ["escalator", "elevator music", "cash register"],
            "medium": ["crowd", "shopping cart", "store announcement"],
            "weak": ["footsteps", "door"],
            "acoustic_profile": {"reverb": "medium", "noise_floor": "medium", "music": True}
        },
        "Office Building": {
            "strong": ["typing", "keyboard", "printer", "phone ringing"],
            "medium": ["air conditioning", "office equipment", "quiet conversation"],
            "weak": ["door", "footsteps"],
            "acoustic_profile": {"reverb": "low", "noise_floor": "low", "hvac_hum": True}
        },
        "Restaurant/Cafe": {
            "strong": ["dishes", "cutlery", "coffee machine", "blender"],
            "medium": ["conversation", "background music", "cash register"],
            "weak": ["door", "chair moving"],
            "acoustic_profile": {"reverb": "low", "noise_floor": "medium", "clinking": True}
        },
        "Construction Site": {
            "strong": ["power tool", "drill", "hammer", "jackhammer", "sawing"],
            "medium": ["truck", "heavy machinery", "metal clanging"],
            "weak": ["shouting", "radio"],
            "acoustic_profile": {"reverb": "none", "noise_floor": "very_high", "intermittent_loud": True}
        },
        "Park/Outdoor": {
            "strong": ["bird", "bird vocalization", "water", "stream", "wind"],
            "medium": ["children playing", "dog bark", "rustling leaves"],
            "weak": ["distant traffic", "footsteps"],
            "acoustic_profile": {"reverb": "none", "noise_floor": "low", "natural_sounds": True}
        },
        "Stadium/Arena": {
            "strong": ["crowd cheering", "applause", "whistle", "stadium horn"],
            "medium": ["public address", "crowd", "chanting"],
            "weak": ["footsteps", "vendors"],
            "acoustic_profile": {"reverb": "very_high", "noise_floor": "high", "echo": True}
        },
        "Home/Residential": {
            "strong": ["television", "doorbell", "microwave", "washing machine"],
            "medium": ["quiet conversation", "cooking sounds", "refrigerator"],
            "weak": ["clock ticking", "air conditioning"],
            "acoustic_profile": {"reverb": "low", "noise_floor": "very_low", "intimate": True}
        },
        "Religious Place": {
            "strong": ["bell", "church bell", "singing bowl", "chanting", "organ"],
            "medium": ["prayer", "choir", "hymn"],
            "weak": ["quiet footsteps", "whisper"],
            "acoustic_profile": {"reverb": "very_high", "noise_floor": "very_low", "sacred_quiet": True}
        },
        "Metro/Subway": {
            "strong": ["subway", "train on tracks", "air brakes", "door chime"],
            "medium": ["crowd", "public address", "turnstile"],
            "weak": ["footsteps", "escalator"],
            "acoustic_profile": {"reverb": "high", "noise_floor": "high", "tunnel_echo": True}
        },
        "Gym/Sports Center": {
            "strong": ["weight dropping", "treadmill", "exercise equipment"],
            "medium": ["music", "grunting", "counting"],
            "weak": ["conversation", "locker door"],
            "acoustic_profile": {"reverb": "medium", "noise_floor": "medium", "rhythmic": True}
        }
    }
    
    # Keywords that indicate "TALKING ABOUT" (not actually being there)
    TALKING_ABOUT_INDICATORS = [
        "yesterday", "tomorrow", "last week", "next week", "last month",
        "i went to", "i was at", "i visited", "we went", "we visited",
        "going to", "will go", "planning to", "want to visit",
        "remember when", "that time at", "heard about", "news about",
        "they said", "he said", "she said", "apparently",
        "on tv", "on the news", "in the movie", "in the video",
        "my friend at", "my brother at", "someone at",
        "i think", "maybe", "probably", "might be",
        "imagine", "pretend", "like at", "similar to",
        "talking about", "discussing", "mentioned",
        "story about", "telling you about", "describing"
    ]
    
    # Keywords that indicate "BEING AT" (actually present)
    BEING_AT_INDICATORS = [
        "right now", "currently", "at the moment", "here",
        "i am at", "we are at", "standing at", "waiting at",
        "can you hear", "so loud here", "it's busy here",
        "just arrived", "just got here", "arriving now",
        "look at this", "see this", "check this out",
        "excuse me", "attention please", "ladies and gentlemen",
        "next stop", "now arriving", "now departing", "now boarding",
        "platform number", "gate number", "terminal",
        "passengers are requested", "kindly note"
    ]
    
    def __init__(self):
        self.loaded = False
    
    def load(self) -> bool:
        """Load the location detector"""
        self.loaded = True
        print("✅ Smart Location Detector loaded")
        return True
    
    def detect(
        self,
        sounds: Dict[str, float],
        text: str,
        audio_features: Dict[str, Any],
        duration: float
    ) -> Dict[str, Any]:
        """
        Detect location with high accuracy
        
        Args:
            sounds: Detected sounds with confidence scores
            text: Transcribed text
            audio_features: Acoustic features of the audio
            duration: Audio duration in seconds
            
        Returns:
            Dictionary with location detection results
        """
        if not self.loaded:
            return self._unknown_result("Detector not loaded")
        
        # Step 1: Analyze ambient sounds (PRIMARY - 55% weight)
        ambient_scores = self._analyze_ambient_sounds(sounds)
        
        # Step 2: Analyze acoustic properties (20% weight)
        acoustic_scores = self._analyze_acoustics(audio_features)
        
        # Step 3: Analyze speech content with context verification (25% weight)
        speech_scores, is_talking_about = self._analyze_speech_context(text)
        
        # Step 4: Combine scores with weights
        final_scores = self._combine_scores(
            ambient_scores,
            acoustic_scores,
            speech_scores,
            is_talking_about
        )
        
        # Step 5: Apply temporal consistency check
        if duration > 30:
            # For longer audio, require more consistent evidence
            final_scores = self._apply_temporal_consistency(final_scores, sounds, duration)
        
        # Step 6: Determine final location
        location, confidence, evidence = self._determine_location(final_scores)
        
        # Step 7: Generate warnings if needed
        warnings = self._generate_warnings(
            location, confidence, is_talking_about, 
            ambient_scores, speech_scores
        )
        
        return {
            "location": location,
            "confidence": round(confidence, 3),
            "is_verified": confidence >= 0.65,
            "evidence": evidence,
            "warnings": warnings,
            "is_talking_about_location": is_talking_about,
            "detection_method": self._get_detection_method(ambient_scores, speech_scores),
            "all_scores": {k: round(v, 3) for k, v in final_scores.items() if v > 0.1},
            "ambient_evidence": self._get_ambient_evidence(sounds),
            "speech_evidence": self._get_speech_evidence(text)
        }
    
    def _analyze_ambient_sounds(self, sounds: Dict[str, float]) -> Dict[str, float]:
        """Analyze ambient sounds for location detection (PRIMARY METHOD)"""
        scores = defaultdict(float)
        
        for sound, confidence in sounds.items():
            sound_lower = sound.lower()
            
            for location, signatures in self.LOCATION_AMBIENT_SOUNDS.items():
                # Strong indicators (high weight)
                for sig in signatures.get("strong", []):
                    if sig in sound_lower:
                        scores[location] += confidence * 1.5
                
                # Medium indicators
                for sig in signatures.get("medium", []):
                    if sig in sound_lower:
                        scores[location] += confidence * 0.8
                
                # Weak indicators
                for sig in signatures.get("weak", []):
                    if sig in sound_lower:
                        scores[location] += confidence * 0.3
        
        # Normalize scores
        max_score = max(scores.values()) if scores else 1
        if max_score > 0:
            scores = {k: min(v / max_score, 1.0) for k, v in scores.items()}
        
        return dict(scores)
    
    def _analyze_acoustics(self, audio_features: Dict[str, Any]) -> Dict[str, float]:
        """Analyze acoustic properties for location hints"""
        scores = defaultdict(float)
        
        if not audio_features:
            return dict(scores)
        
        reverb_level = audio_features.get("reverb_estimate", 0.5)
        noise_floor = audio_features.get("noise_floor", 0.5)
        is_outdoor = audio_features.get("is_outdoor", False)
        
        # High reverb locations
        if reverb_level > 0.7:
            scores["Airport Terminal"] += 0.3
            scores["Railway Station"] += 0.3
            scores["Stadium/Arena"] += 0.4
            scores["Religious Place"] += 0.4
            scores["Metro/Subway"] += 0.3
        
        # Low reverb locations
        if reverb_level < 0.3:
            scores["Home/Residential"] += 0.4
            scores["Office Building"] += 0.3
            scores["Hospital"] += 0.3
        
        # High noise floor
        if noise_floor > 0.6:
            scores["Construction Site"] += 0.4
            scores["Street/Road"] += 0.3
            scores["Railway Station"] += 0.2
        
        # Low noise floor
        if noise_floor < 0.2:
            scores["Home/Residential"] += 0.3
            scores["Office Building"] += 0.2
            scores["Hospital"] += 0.3
        
        # Outdoor detection
        if is_outdoor:
            scores["Street/Road"] += 0.3
            scores["Park/Outdoor"] += 0.4
            scores["Construction Site"] += 0.2
            # Reduce indoor location scores
            for indoor in ["Home/Residential", "Office Building", "Hospital"]:
                scores[indoor] *= 0.5
        
        return dict(scores)
    
    def _analyze_speech_context(self, text: str) -> Tuple[Dict[str, float], bool]:
        """
        Analyze speech content with context verification
        Detects if person is TALKING ABOUT vs BEING AT a location
        """
        if not text:
            return {}, False
        
        text_lower = text.lower()
        
        # Check for "talking about" indicators
        talking_about_count = sum(
            1 for indicator in self.TALKING_ABOUT_INDICATORS 
            if indicator in text_lower
        )
        
        # Check for "being at" indicators
        being_at_count = sum(
            1 for indicator in self.BEING_AT_INDICATORS 
            if indicator in text_lower
        )
        
        # Determine if likely talking about (not actually there)
        is_talking_about = talking_about_count > being_at_count and talking_about_count >= 2
        
        # If talking about, significantly reduce speech-based detection weight
        speech_weight_modifier = 0.2 if is_talking_about else 1.0
        
        scores = defaultdict(float)
        
        # Location keywords with context awareness
        location_keywords = {
            "Airport Terminal": {
                "keywords": ["airport", "terminal", "flight", "airline", "boarding", "gate", "runway"],
                "strong_context": ["now boarding", "gate number", "final call", "passengers"],
            },
            "Railway Station": {
                "keywords": ["railway", "train", "platform", "station", "coach", "bogey"],
                "strong_context": ["platform number", "now arriving", "train number", "passengers"],
            },
            "Hospital": {
                "keywords": ["hospital", "doctor", "nurse", "patient", "emergency", "ward"],
                "strong_context": ["dr.", "doctor", "nurse station", "visiting hours"],
            },
            "Shopping Mall": {
                "keywords": ["mall", "shopping", "store", "sale", "discount"],
                "strong_context": ["attention shoppers", "store closing", "special offer"],
            },
            "Office Building": {
                "keywords": ["office", "meeting", "conference", "presentation"],
                "strong_context": ["meeting room", "conference call", "agenda"],
            },
        }
        
        for location, data in location_keywords.items():
            # Regular keywords (lower weight if talking about)
            keyword_matches = sum(1 for kw in data["keywords"] if kw in text_lower)
            scores[location] += keyword_matches * 0.15 * speech_weight_modifier
            
            # Strong context (higher weight, less affected by talking about)
            context_matches = sum(1 for ctx in data.get("strong_context", []) if ctx in text_lower)
            scores[location] += context_matches * 0.3 * (0.5 if is_talking_about else 1.0)
        
        return dict(scores), is_talking_about
    
    def _combine_scores(
        self,
        ambient_scores: Dict[str, float],
        acoustic_scores: Dict[str, float],
        speech_scores: Dict[str, float],
        is_talking_about: bool
    ) -> Dict[str, float]:
        """Combine all scores with appropriate weights"""
        final_scores = defaultdict(float)
        
        # Weights (ambient sounds are most important)
        ambient_weight = 0.55
        acoustic_weight = 0.20
        speech_weight = 0.25 if not is_talking_about else 0.10  # Reduce speech weight if talking about
        
        # Combine
        all_locations = set(ambient_scores.keys()) | set(acoustic_scores.keys()) | set(speech_scores.keys())
        
        for location in all_locations:
            ambient = ambient_scores.get(location, 0)
            acoustic = acoustic_scores.get(location, 0)
            speech = speech_scores.get(location, 0)
            
            # Require ambient evidence for high confidence
            if ambient > 0.3:
                final_scores[location] = (
                    ambient * ambient_weight +
                    acoustic * acoustic_weight +
                    speech * speech_weight
                )
            elif acoustic > 0.4:
                # Can use acoustic if strong
                final_scores[location] = (
                    ambient * ambient_weight +
                    acoustic * acoustic_weight +
                    speech * speech_weight
                ) * 0.7  # Reduce confidence
            elif speech > 0.5 and not is_talking_about:
                # Speech only if very strong and not talking about
                final_scores[location] = speech * speech_weight * 0.5  # Low confidence
        
        return dict(final_scores)
    
    def _apply_temporal_consistency(
        self,
        scores: Dict[str, float],
        sounds: Dict[str, float],
        duration: float
    ) -> Dict[str, float]:
        """For longer audio, require temporal consistency"""
        # This would ideally use time-segmented sound analysis
        # For now, boost locations with multiple sound evidence
        
        adjusted_scores = {}
        
        for location, score in scores.items():
            signatures = self.LOCATION_AMBIENT_SOUNDS.get(location, {})
            all_sigs = (
                signatures.get("strong", []) + 
                signatures.get("medium", []) + 
                signatures.get("weak", [])
            )
            
            # Count how many different signatures are present
            sig_count = sum(
                1 for sig in all_sigs 
                for sound in sounds.keys() 
                if sig in sound.lower()
            )
            
            # Boost if multiple signatures (more consistent)
            if sig_count >= 3:
                adjusted_scores[location] = score * 1.2
            elif sig_count >= 2:
                adjusted_scores[location] = score * 1.1
            elif sig_count == 1:
                adjusted_scores[location] = score * 0.9
            else:
                adjusted_scores[location] = score * 0.7
        
        return adjusted_scores
    
    def _determine_location(
        self,
        scores: Dict[str, float]
    ) -> Tuple[str, float, List[str]]:
        """Determine final location from scores"""
        if not scores:
            return "Unknown", 0.40, ["No clear location indicators found"]
        
        # Get best location
        best_location = max(scores, key=scores.get)
        best_score = scores[best_location]
        
        # Require minimum confidence
        if best_score < 0.35:
            return "Unknown", 0.40, ["Low confidence in location detection"]
        
        # Calculate final confidence
        confidence = 0.50 + min(best_score * 0.5, 0.48)
        
        # Build evidence
        evidence = []
        if best_score > 0.5:
            evidence.append(f"Strong ambient sound match for {best_location}")
        else:
            evidence.append(f"Moderate ambient sound match for {best_location}")
        
        # Check for runner-up (ambiguity)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) > 1:
            runner_up = sorted_scores[1]
            if runner_up[1] > best_score * 0.8:
                confidence *= 0.9  # Reduce confidence due to ambiguity
                evidence.append(f"Also possible: {runner_up[0]}")
        
        return best_location, confidence, evidence
    
    def _generate_warnings(
        self,
        location: str,
        confidence: float,
        is_talking_about: bool,
        ambient_scores: Dict[str, float],
        speech_scores: Dict[str, float]
    ) -> List[str]:
        """Generate warnings about detection reliability"""
        warnings = []
        
        if is_talking_about:
            warnings.append("⚠️ Speech may be describing a location, not current location")
        
        if not ambient_scores and speech_scores:
            warnings.append("⚠️ Detection based on speech only - verify with ambient sounds")
        
        if confidence < 0.6:
            warnings.append("⚠️ Low confidence - location may be inaccurate")
        
        if location == "Unknown":
            warnings.append("ℹ️ Could not determine location - insufficient audio evidence")
        
        return warnings
    
    def _get_detection_method(
        self,
        ambient_scores: Dict[str, float],
        speech_scores: Dict[str, float]
    ) -> str:
        """Get the primary detection method used"""
        ambient_max = max(ambient_scores.values()) if ambient_scores else 0
        speech_max = max(speech_scores.values()) if speech_scores else 0
        
        if ambient_max > 0.3:
            return "ambient_sounds"
        elif speech_max > 0.3:
            return "speech_content"
        else:
            return "combined_weak"
    
    def _get_ambient_evidence(self, sounds: Dict[str, float]) -> List[str]:
        """Get top ambient sounds as evidence"""
        sorted_sounds = sorted(sounds.items(), key=lambda x: x[1], reverse=True)
        return [f"{sound} ({score:.0%})" for sound, score in sorted_sounds[:5]]
    
    def _get_speech_evidence(self, text: str) -> List[str]:
        """Extract key phrases from speech as evidence"""
        if not text:
            return []
        
        # Find key phrases
        key_phrases = []
        text_lower = text.lower()
        
        for indicator in self.BEING_AT_INDICATORS[:5]:
            if indicator in text_lower:
                # Find context around the indicator
                idx = text_lower.find(indicator)
                start = max(0, idx - 20)
                end = min(len(text), idx + len(indicator) + 30)
                phrase = text[start:end].strip()
                key_phrases.append(f"...{phrase}...")
        
        return key_phrases[:3]
    
    def _unknown_result(self, reason: str) -> Dict[str, Any]:
        """Return unknown result"""
        return {
            "location": "Unknown",
            "confidence": 0.0,
            "is_verified": False,
            "evidence": [reason],
            "warnings": [reason],
            "is_talking_about_location": False,
            "detection_method": "none",
            "all_scores": {},
            "ambient_evidence": [],
            "speech_evidence": []
        }