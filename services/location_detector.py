# ==============================
# 📄 services/location_detector_improved.py
# ==============================
"""
IMPROVED Smart Location Detection Service
- Better ambient sound matching
- Improved acoustic analysis
- Enhanced context understanding
- Learns from corrections
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import re


class LocationDetectorImproved:
    """
    IMPROVED Smart Location Detection Service
    
    Key improvements:
    1. More comprehensive ambient sound signatures
    2. Better acoustic fingerprinting
    3. Contextual speech analysis
    4. Learning from corrections
    5. Handles edge cases better
    """
    
    # EXPANDED ambient sound signatures
    LOCATION_AMBIENT_SOUNDS = {
        "Airport Terminal": {
            "strong": ["aircraft", "airplane", "jet engine", "aircraft engine", "helicopter", "aviation"],
            "medium": ["public address system", "crowd", "luggage", "escalator", "boarding", "departure"],
            "weak": ["typing", "cash register", "announcement"],
            "keywords": ["flight", "gate", "terminal", "boarding", "departure", "arrival", "baggage"],
            "acoustic": {"reverb": "high", "noise_floor": "medium", "spatial": "very_large"}
        },
        "Railway Station": {
            "strong": ["train", "train horn", "train whistle", "rail", "railroad", "locomotive", "railway"],
            "medium": ["public address", "crowd", "brake squeal", "platform", "announcement"],
            "weak": ["door", "footsteps"],
            "keywords": ["platform", "train", "railway", "departure", "arrival", "track"],
            "acoustic": {"reverb": "high", "noise_floor": "high", "spatial": "large"}
        },
        "Bus Terminal": {
            "strong": ["bus", "diesel engine", "air brakes", "bus horn"],
            "medium": ["public address", "crowd", "engine idle"],
            "weak": ["door", "footsteps"],
            "keywords": ["bus", "terminal", "route", "departure"],
            "acoustic": {"reverb": "medium", "noise_floor": "medium", "spatial": "medium"}
        },
        "Hospital": {
            "strong": ["beep", "heart monitor", "medical equipment", "ambulance", "siren", "pager"],
            "medium": ["intercom", "ventilator", "medical"],
            "weak": ["door", "quiet conversation", "footsteps"],
            "keywords": ["patient", "doctor", "nurse", "emergency", "ward", "room"],
            "acoustic": {"reverb": "low", "noise_floor": "low", "spatial": "corridor"}
        },
        "Shopping Mall": {
            "strong": ["escalator", "mall music", "cash register", "shopping"],
            "medium": ["crowd", "store", "announcement", "background music"],
            "weak": ["footsteps", "door", "conversation"],
            "keywords": ["shop", "store", "mall", "shopping", "buy", "sale"],
            "acoustic": {"reverb": "medium", "noise_floor": "medium", "spatial": "open"}
        },
        "Office Building": {
            "strong": ["typing", "keyboard", "computer keyboard", "mouse click", "printer", "office equipment"],
            "medium": ["air conditioning", "hvac", "phone", "conversation", "meeting", "presentation"],
            "weak": ["door", "footsteps", "chair"],
            "keywords": ["meeting", "office", "work", "conference", "presentation", "colleague", "boss", "desk"],
            "acoustic": {"reverb": "low", "noise_floor": "low", "spatial": "room"}
        },
        "School/University": {
            "strong": ["bell", "school bell", "children", "classroom"],
            "medium": ["teacher", "students", "chatter", "crowd"],
            "weak": ["footsteps", "door"],
            "keywords": ["class", "teacher", "student", "lecture", "professor", "exam", "study"],
            "acoustic": {"reverb": "medium", "noise_floor": "medium", "spatial": "classroom"}
        },
        "Restaurant/Cafe": {
            "strong": ["dishes", "cutlery", "silverware", "coffee machine", "blender", "kitchen"],
            "medium": ["conversation", "background music", "cash register", "cooking"],
            "weak": ["door", "chair"],
            "keywords": ["food", "eat", "drink", "order", "menu", "waiter", "table"],
            "acoustic": {"reverb": "low", "noise_floor": "medium", "spatial": "dining"}
        },
        "Street/Road": {
            "strong": ["traffic", "car", "car horn", "engine", "motorcycle", "truck", "vehicle"],
            "medium": ["siren", "bus", "bicycle", "honk"],
            "weak": ["bird", "wind", "footsteps"],
            "keywords": ["street", "road", "traffic", "crossing", "sidewalk"],
            "acoustic": {"reverb": "none", "noise_floor": "high", "spatial": "outdoor"}
        },
        "Home/Residential": {
            "strong": ["television", "tv", "doorbell", "microwave", "washing machine", "vacuum"],
            "medium": ["conversation", "cooking", "refrigerator", "home appliance"],
            "weak": ["clock", "air conditioning"],
            "keywords": ["home", "house", "living room", "bedroom", "kitchen", "family"],
            "acoustic": {"reverb": "low", "noise_floor": "very_low", "spatial": "intimate"}
        },
        "Park/Outdoor": {
            "strong": ["bird", "bird song", "water", "stream", "wind", "nature"],
            "medium": ["children", "playground", "dog", "rustling"],
            "weak": ["distant traffic"],
            "keywords": ["park", "outdoor", "nature", "trees", "grass", "garden"],
            "acoustic": {"reverb": "none", "noise_floor": "low", "spatial": "open_air"}
        },
        "Stadium/Arena": {
            "strong": ["crowd", "cheering", "applause", "whistle", "horn"],
            "medium": ["announcer", "music", "chanting"],
            "weak": ["footsteps"],
            "keywords": ["game", "match", "team", "score", "stadium", "arena", "sport"],
            "acoustic": {"reverb": "very_high", "noise_floor": "very_high", "spatial": "massive"}
        },
        "Metro/Subway": {
            "strong": ["subway", "metro", "train", "rail", "underground", "tube"],
            "medium": ["crowd", "announcement", "door chime", "turnstile"],
            "weak": ["footsteps", "escalator"],
            "keywords": ["metro", "subway", "underground", "line", "station"],
            "acoustic": {"reverb": "high", "noise_floor": "high", "spatial": "tunnel"}
        },
        "Construction Site": {
            "strong": ["drill", "hammer", "jackhammer", "saw", "power tool", "machinery"],
            "medium": ["truck", "equipment", "bang", "drilling"],
            "weak": ["shouting"],
            "keywords": ["construction", "building", "site", "work"],
            "acoustic": {"reverb": "none", "noise_floor": "very_high", "spatial": "outdoor"}
        },
        "Parking Area": {
            "strong": ["car engine", "car door", "parking", "vehicle"],
            "medium": ["beep", "alarm", "engine idle"],
            "weak": ["footsteps", "key"],
            "keywords": ["parking", "car", "vehicle"],
            "acoustic": {"reverb": "low", "noise_floor": "medium", "spatial": "semi_enclosed"}
        },
    }
    
    # Context indicators for being AT vs TALKING ABOUT
    TALKING_ABOUT_INDICATORS = [
        # Past tense
        "yesterday", "last week", "last month", "last year",
        "i went to", "i was at", "i visited", "we went", "we visited",
        "went there", "been there", "used to",
        
        # Future tense
        "tomorrow", "next week", "next month",
        "going to", "will go", "planning to", "want to visit",
        
        # Describing
        "remember when", "that time at", "heard about", "news about",
        "they said", "he said", "she said", "apparently",
        "on tv", "on the news", "in the movie", "video about",
        
        # Others
        "my friend at", "someone at",
        "think", "maybe", "probably", "might be",
        "imagine", "pretend", "like at", "similar to",
        "talking about", "discussing", "mentioned",
    ]
    
    BEING_AT_INDICATORS = [
        # Present
        "right now", "currently", "at the moment", "here", "now",
        "i am at", "we are at", "i'm at", "we're at",
        "standing", "waiting", "sitting",
        
        # Perceptual
        "can you hear", "listen to", "look at", "see this",
        "so loud", "noisy here", "quiet here",
        
        # Arrival/departure
        "just arrived", "just got here", "arriving",
        "about to leave", "leaving soon",
        
        # Direct address
        "excuse me", "attention", "ladies and gentlemen",
        
        # Location-specific
        "next stop", "now boarding", "platform", "gate",
        "table number", "room number",
    ]
    
    def __init__(self):
        self.loaded = False
        self.learning_boosts = defaultdict(float)  # Learned patterns
        
    def load(self) -> bool:
        """Load the location detector"""
        self.loaded = True
        print("✅ IMPROVED Location Detector loaded")
        return True
    
    def detect(
        self,
        sounds: Dict[str, float],
        text: str,
        audio_features: Optional[Dict[str, Any]] = None,
        duration: float = 0.0
    ) -> Dict[str, Any]:
        """
        Detect location with improved accuracy
        
        Args:
            sounds: Detected sounds with confidence scores
            text: Transcribed text
            audio_features: Acoustic features (optional)
            duration: Audio duration in seconds
            
        Returns:
            Dictionary with location detection results
        """
        if not self.loaded:
            return self._unknown_result("Detector not loaded")
        
        # Ensure we have something to work with
        if not sounds and not text:
            return self._unknown_result("No audio or text data provided")
        
        # Step 1: Analyze ambient sounds (PRIMARY - 50% weight)
        ambient_scores = self._analyze_ambient_sounds(sounds)
        
        # Step 2: Analyze speech content and context (30% weight)
        speech_scores, is_talking_about = self._analyze_speech_context(text)
        
        # Step 3: Analyze keywords in text (20% weight)
        keyword_scores = self._analyze_keywords(text)
        
        # Step 4: Apply learning boosts
        learned_scores = self._apply_learning(text, sounds)
        
        # Step 5: Combine all scores
        final_scores = self._combine_scores(
            ambient_scores,
            speech_scores,
            keyword_scores,
            learned_scores,
            is_talking_about
        )
        
        # Step 6: Temporal consistency check (if duration available)
        if duration > 5.0:
            final_scores = self._apply_temporal_consistency(final_scores, sounds, duration)
        
        # Step 7: Determine final location
        location, confidence, evidence = self._determine_location(final_scores)
        
        # Step 8: Generate warnings
        warnings = self._generate_warnings(
            location, confidence, is_talking_about,
            ambient_scores, speech_scores
        )
        
        # Build result
        return {
            "location": location,
            "confidence": confidence,
            "is_verified": confidence > 0.65,
            "evidence": evidence,
            "warnings": warnings,
            "is_talking_about_location": is_talking_about,
            "detection_method": self._get_detection_method(ambient_scores, speech_scores, keyword_scores),
            "all_scores": dict(sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:5]),
            "ambient_evidence": self._get_ambient_evidence(sounds),
            "speech_evidence": self._get_speech_evidence(text),
        }
    
    def _analyze_ambient_sounds(self, sounds: Dict[str, float]) -> Dict[str, float]:
        """Analyze ambient sounds to score locations - IMPROVED"""
        scores = defaultdict(float)
        
        if not sounds:
            return dict(scores)
        
        # Analyze each sound
        for sound, confidence in sounds.items():
            sound_lower = sound.lower()
            
            # Check against each location's signatures
            for location, signatures in self.LOCATION_AMBIENT_SOUNDS.items():
                # Strong signatures (0.8 weight)
                for sig in signatures.get("strong", []):
                    if sig.lower() in sound_lower:
                        scores[location] += confidence * 0.8
                
                # Medium signatures (0.5 weight)
                for sig in signatures.get("medium", []):
                    if sig.lower() in sound_lower:
                        scores[location] += confidence * 0.5
                
                # Weak signatures (0.2 weight)
                for sig in signatures.get("weak", []):
                    if sig.lower() in sound_lower:
                        scores[location] += confidence * 0.2
        
        # Normalize scores
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {loc: score / max_score for loc, score in scores.items()}
        
        return dict(scores)
    
    def _analyze_speech_context(self, text: str) -> Tuple[Dict[str, float], bool]:
        """Analyze speech content for location mentions - IMPROVED"""
        scores = defaultdict(float)
        is_talking_about = False
        
        if not text:
            return dict(scores), is_talking_about
        
        text_lower = text.lower()
        
        # Check for "talking about" indicators
        for indicator in self.TALKING_ABOUT_INDICATORS:
            if indicator in text_lower:
                is_talking_about = True
                break
        
        # Check for "being at" indicators (overrides talking about)
        for indicator in self.BEING_AT_INDICATORS:
            if indicator in text_lower:
                is_talking_about = False
                break
        
        # Look for location mentions in text
        for location in self.LOCATION_AMBIENT_SOUNDS.keys():
            location_lower = location.lower()
            location_words = location_lower.split('/')
            
            for word in location_words:
                if word in text_lower:
                    # Score based on context
                    if is_talking_about:
                        scores[location] += 0.3  # Lower score if just talking about
                    else:
                        scores[location] += 0.8  # Higher score if actually there
        
        return dict(scores), is_talking_about
    
    def _analyze_keywords(self, text: str) -> Dict[str, float]:
        """Analyze keywords in text for location hints - NEW"""
        scores = defaultdict(float)
        
        if not text:
            return dict(scores)
        
        text_lower = text.lower()
        
        # Check each location's keywords
        for location, signatures in self.LOCATION_AMBIENT_SOUNDS.items():
            keywords = signatures.get("keywords", [])
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    scores[location] += 0.4
        
        # Normalize
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {loc: score / max_score for loc, score in scores.items()}
        
        return dict(scores)
    
    def _apply_learning(self, text: str, sounds: Dict[str, float]) -> Dict[str, float]:
        """Apply learned patterns - NEW"""
        scores = defaultdict(float)
        
        if not self.learning_boosts:
            return dict(scores)
        
        # Create feature string
        words = set(re.findall(r'\b\w{4,}\b', text.lower()))
        sound_list = list(sounds.keys())[:10]
        
        # Check learned patterns
        for location, boost in self.learning_boosts.items():
            # Check if any learned words match
            for word in words:
                key = f"{location}::{word}"
                if key in self.learning_boosts:
                    scores[location] += self.learning_boosts[key]
            
            # Check if any learned sounds match
            for sound in sound_list:
                key = f"{location}::sound::{sound.lower()}"
                if key in self.learning_boosts:
                    scores[location] += self.learning_boosts[key]
        
        return dict(scores)
    
    def _combine_scores(
        self,
        ambient_scores: Dict[str, float],
        speech_scores: Dict[str, float],
        keyword_scores: Dict[str, float],
        learned_scores: Dict[str, float],
        is_talking_about: bool
    ) -> Dict[str, float]:
        """Combine all scores with proper weighting - IMPROVED"""
        
        final_scores = defaultdict(float)
        
        # Weights
        ambient_weight = 0.50
        speech_weight = 0.15 if is_talking_about else 0.30
        keyword_weight = 0.20
        learned_weight = 0.15
        
        # Get all locations
        all_locations = set()
        all_locations.update(ambient_scores.keys())
        all_locations.update(speech_scores.keys())
        all_locations.update(keyword_scores.keys())
        all_locations.update(learned_scores.keys())
        
        # Combine scores
        for location in all_locations:
            ambient = ambient_scores.get(location, 0)
            speech = speech_scores.get(location, 0)
            keyword = keyword_scores.get(location, 0)
            learned = learned_scores.get(location, 0)
            
            # Calculate weighted score
            score = (
                ambient * ambient_weight +
                speech * speech_weight +
                keyword * keyword_weight +
                learned * learned_weight
            )
            
            # Require at least some ambient OR keyword evidence
            if ambient > 0.2 or keyword > 0.3:
                final_scores[location] = score
            elif speech > 0.5 and not is_talking_about:
                final_scores[location] = score * 0.6  # Reduced confidence
        
        return dict(final_scores)
    
    def _apply_temporal_consistency(
        self,
        scores: Dict[str, float],
        sounds: Dict[str, float],
        duration: float
    ) -> Dict[str, float]:
        """Boost locations with consistent sound evidence throughout audio"""
        
        adjusted_scores = {}
        
        for location, score in scores.items():
            signatures = self.LOCATION_AMBIENT_SOUNDS.get(location, {})
            all_sigs = (
                signatures.get("strong", []) +
                signatures.get("medium", [])
            )
            
            # Count matching signatures
            sig_count = sum(
                1 for sig in all_sigs
                for sound in sounds.keys()
                if sig.lower() in sound.lower()
            )
            
            # Boost for multiple signatures
            if sig_count >= 4:
                adjusted_scores[location] = score * 1.3
            elif sig_count >= 3:
                adjusted_scores[location] = score * 1.2
            elif sig_count >= 2:
                adjusted_scores[location] = score * 1.1
            else:
                adjusted_scores[location] = score * 0.85
        
        return adjusted_scores
    
    def _determine_location(
        self,
        scores: Dict[str, float]
    ) -> Tuple[str, float, List[str]]:
        """Determine final location from scores - IMPROVED"""
        
        if not scores:
            return "Unknown", 0.40, ["No clear location indicators found"]
        
        # Get best location
        best_location = max(scores, key=scores.get)
        best_score = scores[best_location]
        
        # LOWERED minimum confidence threshold
        if best_score < 0.25:
            return "Unknown", 0.40, ["Low confidence in location detection"]
        
        # Calculate final confidence (boosted)
        confidence = 0.45 + min(best_score * 0.52, 0.53)
        
        # Build evidence
        evidence = []
        if best_score > 0.6:
            evidence.append(f"Strong match for {best_location}")
        elif best_score > 0.4:
            evidence.append(f"Good match for {best_location}")
        else:
            evidence.append(f"Moderate match for {best_location}")
        
        # Check for ambiguity
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) > 1:
            runner_up = sorted_scores[1]
            if runner_up[1] > best_score * 0.75:
                confidence *= 0.92  # Slight reduction for ambiguity
                evidence.append(f"Also possible: {runner_up[0]}")
        
        return best_location, min(confidence, 0.98), evidence
    
    def _generate_warnings(
        self,
        location: str,
        confidence: float,
        is_talking_about: bool,
        ambient_scores: Dict[str, float],
        speech_scores: Dict[str, float]
    ) -> List[str]:
        """Generate warnings - IMPROVED"""
        
        warnings = []
        
        if is_talking_about:
            warnings.append("⚠️ Speech may be describing a location, not current location")
        
        if not ambient_scores and speech_scores:
            warnings.append("ℹ️ Detection based primarily on speech content")
        
        if confidence < 0.55:
            warnings.append("ℹ️ Moderate confidence - consider context")
        
        if location == "Unknown":
            warnings.append("ℹ️ Could not determine specific location")
        
        return warnings
    
    def _get_detection_method(
        self,
        ambient_scores: Dict[str, float],
        speech_scores: Dict[str, float],
        keyword_scores: Dict[str, float]
    ) -> str:
        """Get primary detection method"""
        
        ambient_max = max(ambient_scores.values()) if ambient_scores else 0
        speech_max = max(speech_scores.values()) if speech_scores else 0
        keyword_max = max(keyword_scores.values()) if keyword_scores else 0
        
        if ambient_max > 0.4:
            return "ambient_sounds"
        elif keyword_max > 0.4:
            return "keywords"
        elif speech_max > 0.3:
            return "speech_content"
        else:
            return "combined_weak"
    
    def _get_ambient_evidence(self, sounds: Dict[str, float]) -> List[str]:
        """Get top ambient sounds as evidence"""
        sorted_sounds = sorted(sounds.items(), key=lambda x: x[1], reverse=True)
        return [f"{sound} ({score:.0%})" for sound, score in sorted_sounds[:7]]
    
    def _get_speech_evidence(self, text: str) -> List[str]:
        """Extract key phrases from speech"""
        if not text:
            return []
        
        key_phrases = []
        text_lower = text.lower()
        
        # Find being-at indicators
        for indicator in self.BEING_AT_INDICATORS[:10]:
            if indicator in text_lower:
                idx = text_lower.find(indicator)
                start = max(0, idx - 25)
                end = min(len(text), idx + len(indicator) + 35)
                phrase = text[start:end].strip()
                key_phrases.append(f"...{phrase}...")
                if len(key_phrases) >= 3:
                    break
        
        return key_phrases
    
    def _unknown_result(self, reason: str) -> Dict[str, Any]:
        """Return unknown result"""
        return {
            "location": "Unknown",
            "confidence": 0.40,
            "is_verified": False,
            "evidence": [reason],
            "warnings": [reason],
            "is_talking_about_location": False,
            "detection_method": "none",
            "all_scores": {},
            "ambient_evidence": [],
            "speech_evidence": []
        }
    
    def learn_from_correction(self, correct_location: str, text: str, sounds: Dict[str, float]):
        """Learn from user corrections - NEW"""
        
        # Extract words
        words = set(re.findall(r'\b\w{4,}\b', text.lower()))
        
        # Boost keywords
        for word in words:
            key = f"{correct_location}::{word}"
            self.learning_boosts[key] = min(self.learning_boosts[key] + 0.08, 0.4)
        
        # Boost sounds
        for sound in list(sounds.keys())[:10]:
            key = f"{correct_location}::sound::{sound.lower()}"
            self.learning_boosts[key] = min(self.learning_boosts[key] + 0.10, 0.5)
        
        print(f"📚 Learned: {correct_location} (+{len(words)} words, +{len(sounds)} sounds)")